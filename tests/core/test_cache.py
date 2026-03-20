"""Tests for cache module — CachedPage, normalize_url, url_to_cache_dir, CacheManager helpers."""

import json
import time
from pathlib import Path

import pytest

from liveweb_arena.core.cache import (
    CachedPage,
    CacheFatalError,
    CacheManager,
    PageRequirement,
    normalize_url,
    safe_path_component,
    url_to_cache_dir,
)


# ── CachedPage ──────────────────────────────────────────────────────


class TestCachedPage:
    def test_is_complete_with_api(self):
        page = CachedPage(url="u", html="h", api_data={"k": "v"}, fetched_at=1.0, need_api=True)
        assert page.is_complete()

    def test_is_incomplete_missing_api(self):
        page = CachedPage(url="u", html="h", api_data=None, fetched_at=1.0, need_api=True)
        assert not page.is_complete()

    def test_is_incomplete_empty_api(self):
        page = CachedPage(url="u", html="h", api_data={}, fetched_at=1.0, need_api=True)
        assert not page.is_complete()

    def test_is_complete_no_api_needed(self):
        page = CachedPage(url="u", html="h", api_data=None, fetched_at=1.0, need_api=False)
        assert page.is_complete()

    def test_is_expired(self):
        page = CachedPage(url="u", html="h", api_data=None, fetched_at=time.time() - 100, need_api=False)
        assert page.is_expired(ttl=50)
        assert not page.is_expired(ttl=200)

    def test_roundtrip_to_dict_from_dict(self):
        page = CachedPage(
            url="https://example.com",
            html="<h1>hi</h1>",
            api_data={"price": 42},
            fetched_at=1234567890.0,
            accessibility_tree="heading: hi",
            need_api=True,
        )
        d = page.to_dict()
        restored = CachedPage.from_dict(d)
        assert restored.url == page.url
        assert restored.html == page.html
        assert restored.api_data == page.api_data
        assert restored.fetched_at == page.fetched_at
        assert restored.accessibility_tree == page.accessibility_tree
        assert restored.need_api == page.need_api

    def test_from_dict_defaults(self):
        """Old cache format without need_api defaults to True."""
        d = {"url": "u", "html": "h", "api_data": None, "fetched_at": 1.0}
        page = CachedPage.from_dict(d)
        assert page.need_api is True
        assert page.accessibility_tree is None

    def test_to_dict_omits_none_a11y(self):
        page = CachedPage(url="u", html="h", api_data=None, fetched_at=1.0)
        d = page.to_dict()
        assert "accessibility_tree" not in d

    def test_to_dict_includes_a11y(self):
        page = CachedPage(url="u", html="h", api_data=None, fetched_at=1.0, accessibility_tree="tree")
        d = page.to_dict()
        assert d["accessibility_tree"] == "tree"

    def test_from_dict_enriches_sparse_a11y_from_html(self):
        d = {
            "url": "https://taostats.io/subnets/75",
            "html": "<html><body><nav>taostats</nav><main><h1>Hippius</h1><div>Statistics</div><div>Price Impact</div></main></body></html>",
            "api_data": {"netuid": 75},
            "fetched_at": 1.0,
            "accessibility_tree": "taostats",
            "need_api": True,
        }
        page = CachedPage.from_dict(d)
        assert "Hippius" in page.accessibility_tree
        assert "Statistics" in page.accessibility_tree


# ── PageRequirement ──────────────────────────────────────────────────


class TestPageRequirement:
    def test_nav(self):
        req = PageRequirement.nav("https://example.com")
        assert req.url == "https://example.com"
        assert req.need_api is False

    def test_data(self):
        req = PageRequirement.data("https://example.com")
        assert req.need_api is True


# ── normalize_url ────────────────────────────────────────────────────


class TestNormalizeUrl:
    def test_lowercase_domain(self):
        assert "example.com" in normalize_url("https://EXAMPLE.COM/Path")

    def test_preserves_path(self):
        result = normalize_url("https://example.com/en/coins/bitcoin")
        assert "/en/coins/bitcoin" in result

    def test_removes_default_port_80(self):
        result = normalize_url("http://example.com:80/page")
        assert ":80" not in result
        assert "example.com/page" in result

    def test_removes_default_port_443(self):
        result = normalize_url("https://example.com:443/page")
        assert ":443" not in result

    def test_keeps_non_default_port(self):
        result = normalize_url("http://localhost:8080/api")
        assert ":8080" in result

    def test_strips_tracking_params(self):
        result = normalize_url("https://example.com/page?utm_source=google&id=123")
        assert "utm_source" not in result
        assert "id=123" in result

    def test_sorts_query_params(self):
        result = normalize_url("https://example.com/page?z=1&a=2")
        assert result.endswith("?a=2&z=1")

    def test_decodes_percent_encoding(self):
        result = normalize_url("https://example.com/caf%C3%A9")
        assert "café" in result

    def test_empty_path_gets_slash(self):
        result = normalize_url("https://example.com")
        assert result.endswith("example.com/")

    def test_idempotent(self):
        url = "https://www.coingecko.com/en/coins/bitcoin?utm_source=x"
        assert normalize_url(url) == normalize_url(normalize_url(url))


# ── safe_path_component ──────────────────────────────────────────────


class TestSafePathComponent:
    def test_replaces_dangerous_chars(self):
        result = safe_path_component('file<name>:with"bad|chars')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result
        assert "|" not in result

    def test_replaces_spaces(self):
        assert "_" in safe_path_component("hello world")

    def test_truncates_long_strings(self):
        result = safe_path_component("a" * 300)
        assert len(result) == 200


# ── url_to_cache_dir ─────────────────────────────────────────────────


class TestUrlToCacheDir:
    def test_basic_path(self):
        result = url_to_cache_dir(Path("/cache"), "https://www.coingecko.com/en/coins/bitcoin")
        assert result == Path("/cache/www.coingecko.com/en/coins/bitcoin")

    def test_query_params(self):
        result = url_to_cache_dir(Path("/cache"), "https://stooq.com/q/?s=aapl.us")
        parts = str(result)
        assert "stooq.com" in parts
        assert "aapl.us" in parts

    def test_root_path(self):
        result = url_to_cache_dir(Path("/cache"), "https://example.com")
        assert "_root_" in str(result)

    def test_strips_default_port(self):
        result = url_to_cache_dir(Path("/cache"), "https://example.com:443/page")
        assert ":443" not in str(result)


# ── CacheFatalError ──────────────────────────────────────────────────


class TestCacheFatalError:
    def test_basic(self):
        err = CacheFatalError("timeout", url="https://x.com")
        assert str(err) == "timeout"
        assert err.url == "https://x.com"

    def test_no_url(self):
        err = CacheFatalError("generic failure")
        assert err.url is None

    def test_structured_fields(self):
        err = CacheFatalError(
            "timeout",
            url="https://x.com",
            status_code=504,
            evidence={"k": "v"},
            soft_fail_applied=True,
            stale_fallback_used=False,
            plugin_name="coingecko",
        )
        assert err.status_code == 504
        assert err.evidence == {"k": "v"}
        assert err.soft_fail_applied is True
        assert err.stale_fallback_used is False
        assert err.plugin_name == "coingecko"


# ── CacheManager._load_if_valid (via file I/O) ──────────────────────


class TestCacheManagerLoadIfValid:
    def test_returns_none_for_missing_file(self, tmp_path):
        mgr = CacheManager(cache_dir=tmp_path, ttl=3600)
        assert mgr._load_if_valid(tmp_path / "nonexistent.json", need_api=False) is None

    def test_loads_valid_cache(self, tmp_path):
        cache_file = tmp_path / "page.json"
        page = CachedPage(url="https://x.com", html="<h1>x</h1>", api_data={"k": 1}, fetched_at=time.time(), need_api=True)
        with open(cache_file, "w") as f:
            json.dump(page.to_dict(), f)
        mgr = CacheManager(cache_dir=tmp_path, ttl=3600)
        loaded = mgr._load_if_valid(cache_file, need_api=True)
        assert loaded is not None
        assert loaded.url == "https://x.com"

    def test_rejects_expired_cache(self, tmp_path):
        cache_file = tmp_path / "page.json"
        page = CachedPage(url="https://x.com", html="h", api_data={"k": 1}, fetched_at=1.0, need_api=True)
        with open(cache_file, "w") as f:
            json.dump(page.to_dict(), f)
        mgr = CacheManager(cache_dir=tmp_path, ttl=3600)
        assert mgr._load_if_valid(cache_file, need_api=True) is None
        assert not cache_file.exists()  # expired cache deleted

    def test_rejects_incomplete_cache(self, tmp_path):
        cache_file = tmp_path / "page.json"
        page = CachedPage(url="https://x.com", html="h", api_data=None, fetched_at=time.time(), need_api=True)
        with open(cache_file, "w") as f:
            json.dump(page.to_dict(), f)
        mgr = CacheManager(cache_dir=tmp_path, ttl=3600)
        assert mgr._load_if_valid(cache_file, need_api=True) is None

    def test_rejects_corrupted_cache(self, tmp_path):
        cache_file = tmp_path / "page.json"
        cache_file.write_text("not json at all")
        mgr = CacheManager(cache_dir=tmp_path, ttl=3600)
        assert mgr._load_if_valid(cache_file, need_api=False) is None
        assert not cache_file.exists()  # corrupted cache deleted

    def test_upgrade_nav_to_data_rejects(self, tmp_path):
        """Cache was saved as nav (no API) but now we need data (need_api=True)."""
        cache_file = tmp_path / "page.json"
        page = CachedPage(url="https://x.com", html="h", api_data=None, fetched_at=time.time(), need_api=False)
        with open(cache_file, "w") as f:
            json.dump(page.to_dict(), f)
        mgr = CacheManager(cache_dir=tmp_path, ttl=3600)
        # need_api=True but cache has no api_data → rejected
        assert mgr._load_if_valid(cache_file, need_api=True) is None

    def test_load_with_status_uses_shared_cache_fallback(self, tmp_path, monkeypatch):
        local_cache = tmp_path / "local"
        shared_cache = tmp_path / "shared"
        monkeypatch.setenv("LIVEWEB_SHARED_CACHE_DIR", str(shared_cache))
        monkeypatch.setenv("LIVEWEB_ENABLE_SHARED_CACHE", "1")
        mgr = CacheManager(cache_dir=local_cache, ttl=3600)

        url = "https://stooq.com/q/?s=jnj.us"
        normalized = normalize_url(url)
        shared_file = url_to_cache_dir(shared_cache, normalized) / "page.json"
        shared_file.parent.mkdir(parents=True, exist_ok=True)
        shared_page = CachedPage(
            url=url,
            html="<html><body><h1>JNJ</h1><p>Price 123</p></body></html>",
            api_data={"symbol": "jnj.us"},
            fetched_at=time.time(),
            need_api=True,
        )
        with open(shared_file, "w") as f:
            json.dump(shared_page.to_dict(), f)

        local_file = url_to_cache_dir(local_cache, normalized) / "page.json"
        status, cached = mgr._load_with_status(normalized, local_file, need_api=True)
        assert status == "valid"
        assert cached is not None
        assert cached.api_data == {"symbol": "jnj.us"}
        assert local_file.exists()
