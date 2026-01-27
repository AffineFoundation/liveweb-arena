"""
Test new cache architecture.

Tests the new cache module, interceptor, and plugin interfaces.
"""

import asyncio
import pytest
from pathlib import Path
from random import Random
from unittest.mock import AsyncMock, MagicMock, patch

from liveweb_arena.core.cache import (
    CacheManager,
    CachedPage,
    PageRequirement,
    normalize_url,
    url_to_cache_dir,
)
from liveweb_arena.core.interceptor import CacheInterceptor, InterceptorStats
from liveweb_arena.core.template import (
    QuestionTemplate,
    GeneratedQuestion,
    register_template,
    get_template,
    clear_templates,
)
from liveweb_arena.plugins import get_plugin, get_all_plugins
from liveweb_arena.plugins.base import BasePlugin


class TestNormalizeUrl:
    """Test URL normalization."""

    def test_lowercase_domain(self):
        assert normalize_url("https://WWW.EXAMPLE.COM/path") == "https://www.example.com/path"

    def test_remove_default_ports(self):
        assert normalize_url("https://example.com:443/path") == "https://example.com/path"
        assert normalize_url("http://example.com:80/path") == "http://example.com/path"

    def test_remove_tracking_params(self):
        url = "https://example.com/page?utm_source=test&id=123"
        normalized = normalize_url(url)
        assert "utm_source" not in normalized
        assert "id=123" in normalized

    def test_sort_query_params(self):
        url1 = "https://example.com/page?b=2&a=1"
        url2 = "https://example.com/page?a=1&b=2"
        assert normalize_url(url1) == normalize_url(url2)


class TestUrlToCacheDir:
    """Test URL to cache directory conversion."""

    def test_simple_path(self):
        cache_dir = Path("/cache")
        url = "https://www.example.com/path/to/page"
        result = url_to_cache_dir(cache_dir, url)
        assert result == Path("/cache/www.example.com/path/to/page")

    def test_with_query_params(self):
        cache_dir = Path("/cache")
        url = "https://stooq.com/q/?s=aapl.us"
        result = url_to_cache_dir(cache_dir, url)
        # Query params should be appended to the path
        assert "stooq.com" in str(result)
        assert "s=aapl.us" in str(result)

    def test_root_path(self):
        cache_dir = Path("/cache")
        url = "https://example.com/"
        result = url_to_cache_dir(cache_dir, url)
        assert "_root_" in str(result)


class TestPageRequirement:
    """Test PageRequirement factory methods."""

    def test_nav_requirement(self):
        req = PageRequirement.nav("https://example.com/")
        assert req.url == "https://example.com/"
        assert req.need_api is False

    def test_data_requirement(self):
        req = PageRequirement.data("https://example.com/coin/bitcoin")
        assert req.url == "https://example.com/coin/bitcoin"
        assert req.need_api is True


class TestCachedPage:
    """Test CachedPage dataclass."""

    def test_to_dict(self):
        page = CachedPage(
            url="https://example.com",
            html="<html>test</html>",
            api_data={"price": 100},
            fetched_at=1000.0,
        )
        d = page.to_dict()
        assert d["url"] == "https://example.com"
        assert d["html"] == "<html>test</html>"
        assert d["api_data"] == {"price": 100}
        assert d["fetched_at"] == 1000.0

    def test_from_dict(self):
        d = {
            "url": "https://example.com",
            "html": "<html>test</html>",
            "api_data": {"price": 100},
            "fetched_at": 1000.0,
        }
        page = CachedPage.from_dict(d)
        assert page.url == "https://example.com"
        assert page.api_data == {"price": 100}

    def test_is_expired(self):
        import time
        page = CachedPage(
            url="https://example.com",
            html="<html></html>",
            api_data=None,
            fetched_at=time.time() - 100,  # 100 seconds ago
        )
        assert page.is_expired(50)  # TTL of 50 seconds
        assert not page.is_expired(200)  # TTL of 200 seconds


class TestInterceptorStats:
    """Test InterceptorStats."""

    def test_to_dict(self):
        stats = InterceptorStats(hits=10, misses=2, blocked=5, passed=3, errors=1)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 2
        assert d["total"] == 20
        assert d["hit_rate"] == 10 / 12  # hits / (hits + misses)


class TestCacheInterceptor:
    """Test CacheInterceptor."""

    def test_should_block(self):
        cached_pages = {}
        interceptor = CacheInterceptor(cached_pages, set())

        # Should block tracking/analytics
        assert interceptor._should_block("https://google-analytics.com/collect")
        assert interceptor._should_block("https://www.googletagmanager.com/gtm.js")

        # Should not block normal URLs
        assert not interceptor._should_block("https://www.coingecko.com/en/coins/bitcoin")

    def test_is_domain_allowed(self):
        cached_pages = {}
        allowed = {"example.com", "test.org"}
        interceptor = CacheInterceptor(cached_pages, allowed)

        assert interceptor._is_domain_allowed("https://example.com/page")
        assert interceptor._is_domain_allowed("https://sub.example.com/page")
        assert interceptor._is_domain_allowed("https://test.org/page")
        assert not interceptor._is_domain_allowed("https://other.com/page")

    def test_find_cached_page(self):
        page = CachedPage(
            url="https://www.example.com/page",
            html="<html>test</html>",
            api_data=None,
            fetched_at=1000.0,
        )
        cached_pages = {"https://www.example.com/page": page}
        interceptor = CacheInterceptor(cached_pages, set())

        # Exact match
        found = interceptor._find_cached_page("https://www.example.com/page")
        assert found == page

        # www/non-www matching
        found = interceptor._find_cached_page("https://example.com/page")
        assert found == page


class TestTemplateRegistry:
    """Test template registration."""

    def setup_method(self):
        clear_templates()

    def test_register_template(self):
        @register_template("test/example")
        class TestTemplate(QuestionTemplate):
            plugin_name = "test"
            expected_steps = 3

            def generate(self, rng):
                return GeneratedQuestion(
                    intent="Test question?",
                    required_pages=[],
                    answer_extractor=lambda x: "answer",
                    expected_steps=self.expected_steps,
                )

        assert get_template("test/example") is TestTemplate

    def teardown_method(self):
        clear_templates()


class TestPluginInterface:
    """Test new plugin interface."""

    def test_coingecko_plugin_attributes(self):
        plugin_class = get_plugin("coingecko")
        assert plugin_class is not None

        # Check class attributes
        assert hasattr(plugin_class, "name")
        assert hasattr(plugin_class, "allowed_domains")
        assert plugin_class.name == "coingecko"
        assert "coingecko.com" in plugin_class.allowed_domains

    def test_coingecko_plugin_fetch_api_data(self):
        plugin_class = get_plugin("coingecko")
        plugin = plugin_class()

        # Check method exists
        assert hasattr(plugin, "fetch_api_data")
        assert asyncio.iscoroutinefunction(plugin.fetch_api_data)

    def test_coingecko_extract_coin_id(self):
        plugin_class = get_plugin("coingecko")
        plugin = plugin_class()

        assert plugin._extract_coin_id("https://www.coingecko.com/en/coins/bitcoin") == "bitcoin"
        assert plugin._extract_coin_id("https://www.coingecko.com/en/coins/ethereum") == "ethereum"
        assert plugin._extract_coin_id("https://www.coingecko.com/") == ""

    def test_stooq_plugin_extract_symbol(self):
        plugin_class = get_plugin("stooq")
        plugin = plugin_class()

        assert plugin._extract_symbol("https://stooq.com/q/?s=aapl.us") == "aapl.us"
        assert plugin._extract_symbol("https://stooq.com/q/?s=^spx") == "^spx"
        assert plugin._extract_symbol("https://stooq.com/") == ""

    def test_weather_plugin_extract_location(self):
        plugin_class = get_plugin("weather")
        plugin = plugin_class()

        assert plugin._extract_location("https://wttr.in/London") == "London"
        assert plugin._extract_location("https://wttr.in/New+York") == "New York"
        assert plugin._extract_location("https://v2.wttr.in/Tokyo") == "Tokyo"

    def test_taostats_plugin_extract_subnet_id(self):
        plugin_class = get_plugin("taostats")
        plugin = plugin_class()

        assert plugin._extract_subnet_id("https://taostats.io/subnets/27") == "27"
        assert plugin._extract_subnet_id("https://taostats.io/subnets/1") == "1"
        assert plugin._extract_subnet_id("https://taostats.io/") == ""

    def test_all_plugins_have_required_interface(self):
        plugins = get_all_plugins()
        assert len(plugins) > 0

        for name, plugin_class in plugins.items():
            # Check required class attributes
            assert hasattr(plugin_class, "name"), f"{name} missing 'name'"
            assert hasattr(plugin_class, "allowed_domains"), f"{name} missing 'allowed_domains'"

            # Check required methods
            plugin = plugin_class()
            assert hasattr(plugin, "fetch_api_data"), f"{name} missing 'fetch_api_data'"
            assert asyncio.iscoroutinefunction(plugin.fetch_api_data), f"{name}.fetch_api_data must be async"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
