from types import SimpleNamespace

import pytest

from liveweb_arena.core.browser import (
    BrowserSession,
    _looks_like_non_html_navigation_target,
    _normalize_stooq_url,
    _should_fallback_to_direct_navigation,
)


class _FakeElement:
    def __init__(self, href: str | None, *, click_error: Exception | None = None):
        self._href = href
        self._click_error = click_error
        self.evaluated = False

    async def get_attribute(self, name: str):
        if name == "href":
            return self._href
        return None

    async def click(self, force: bool = False, timeout: int | None = None):
        if self._click_error is not None:
            raise self._click_error
        return None

    async def evaluate(self, script: str):
        self.evaluated = True
        return None


class _FakeLocator:
    def __init__(self, href: str | None, *, click_error: Exception | None = None):
        self._href = href
        self._click_error = click_error
        self.evaluated = False
        self.first = self

    async def get_attribute(self, name: str):
        if name == "href":
            return self._href
        return None

    async def click(self, force: bool = False, timeout: int | None = None):
        if self._click_error is not None:
            raise self._click_error
        return None

    async def evaluate(self, script: str):
        self.evaluated = True
        return None


class _FakePage:
    def __init__(self, href: str | None, *, element: _FakeElement | None = None):
        self.url = "https://taostats.io/"
        self._href = href
        self._element = element or _FakeElement(href)

    async def query_selector(self, selector: str):
        return self._element


class _FallbackClickPage:
    def __init__(self, *, url: str, click_results: dict[str, Exception | None]):
        self.url = url
        self._click_results = click_results
        self.clicked: list[str] = []

    async def click(self, selector: str, timeout: int | None = None):
        self.clicked.append(selector)
        outcome = self._click_results.get(selector, RuntimeError("not found"))
        if outcome is not None:
            raise outcome
        return None

    async def query_selector(self, selector: str):
        return None


@pytest.mark.asyncio
async def test_browser_session_direct_nav_fallback_from_selector():
    session = BrowserSession.__new__(BrowserSession)
    session._page = _FakePage("/subnets")

    visited = {}

    async def _goto(url: str):
        visited["url"] = url

    session._goto_with_recovery = _goto

    ok = await session._direct_nav_fallback_from_selector("a[href='/subnets']")
    assert ok is True
    assert visited["url"] == "https://taostats.io/subnets"


@pytest.mark.asyncio
async def test_browser_session_direct_nav_fallback_from_locator():
    session = BrowserSession.__new__(BrowserSession)
    session._page = SimpleNamespace(url="https://taostats.io/")

    visited = {}

    async def _goto(url: str):
        visited["url"] = url

    session._goto_with_recovery = _goto

    ok = await session._direct_nav_fallback_from_locator(_FakeLocator("/subnets/23"))
    assert ok is True
    assert visited["url"] == "https://taostats.io/subnets/23"


@pytest.mark.asyncio
async def test_browser_session_force_click_selector_fallback_uses_js_click():
    session = BrowserSession.__new__(BrowserSession)
    element = _FakeElement(None, click_error=RuntimeError("pointer intercepted"))
    session._page = _FakePage(None, element=element)

    ok = await session._force_click_selector_fallback(".dropdown-toggle")
    assert ok is True
    assert element.evaluated is True


@pytest.mark.asyncio
async def test_browser_session_force_click_locator_fallback_uses_js_click():
    session = BrowserSession.__new__(BrowserSession)
    locator = _FakeLocator(None, click_error=RuntimeError("pointer intercepted"))

    ok = await session._force_click_locator_fallback(locator)
    assert ok is True
    assert locator.evaluated is True


@pytest.mark.asyncio
async def test_browser_session_taostats_role_fallback_can_navigate_home_to_subnets():
    session = BrowserSession.__new__(BrowserSession)
    session._page = SimpleNamespace(url="https://taostats.io")

    visited = {}

    async def _goto(url: str):
        visited["url"] = url

    session._goto_with_recovery = _goto

    ok = await session._taostats_list_role_fallback("link", "View All", 5000)
    assert ok is True
    assert visited["url"] == "https://taostats.io/subnets"


@pytest.mark.asyncio
async def test_browser_session_taostats_selector_fallback_tries_semantic_candidates():
    session = BrowserSession.__new__(BrowserSession)
    session._page = _FallbackClickPage(
        url="https://taostats.io/subnets",
        click_results={
            '[data-testid="rows-select"]': RuntimeError("timeout"),
            ".ant-select-selector": None,
        },
    )

    ok = await session._taostats_list_selector_fallback('[data-testid="rows-select"]', 5000)
    assert ok is True
    assert session._page.clicked[:2] == ['[data-testid="rows-select"]', '.ant-select-selector']


def test_should_fallback_to_direct_navigation_for_intercepted_click_timeout():
    exc = RuntimeError("Page.click: Timeout 5000ms exceeded... <html> intercepts pointer events")
    assert _should_fallback_to_direct_navigation(exc) is True


def test_normalize_stooq_list_endpoint_preserves_original_path():
    normalized = _normalize_stooq_url("https://stooq.com/q/l/?s=jnj.us&f=sd2t2ohlcv&h=&e=csv")
    assert normalized == "https://stooq.com/q/l/?s=jnj.us&f=sd2t2ohlcv&h=&e=csv"


def test_normalize_stooq_slug_alias_preserves_original_slug():
    normalized = _normalize_stooq_url("https://stooq.com/q/uk100/")
    assert normalized == "https://stooq.com/q/uk100/"


def test_non_html_navigation_heuristic_flags_downloadish_urls():
    assert _looks_like_non_html_navigation_target("https://stooq.com/q/l/?s=jnj.us&f=sd2t2ohlcv&h=&e=csv") is True
    assert _looks_like_non_html_navigation_target("https://example.com/export/report.csv") is True
    assert _looks_like_non_html_navigation_target("https://example.com/data?format=json") is True


def test_non_html_navigation_heuristic_skips_normal_pages():
    assert _looks_like_non_html_navigation_target("https://stooq.com/q/?s=jnj.us") is False
    assert _looks_like_non_html_navigation_target("https://taostats.io/subnets") is False
