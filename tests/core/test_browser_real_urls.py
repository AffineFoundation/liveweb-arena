import os

import pytest


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("LIVEWEB_REAL_BROWSER_SMOKE") != "1",
        reason="set LIVEWEB_REAL_BROWSER_SMOKE=1 to run real-browser smoke tests",
    ),
]


async def test_real_browser_taostats_subnets_click_fallback():
    from liveweb_arena.core.browser import BrowserAction, BrowserEngine

    engine = BrowserEngine(headless=True)
    session = await engine.new_session()
    try:
        obs = await session.goto("https://taostats.io")
        assert "taostats.io" in obs.url

        obs = await session.execute_action(
            BrowserAction(
                action_type="click",
                params={"selector": "a[href='/subnets']", "timeout_ms": 5000},
            )
        )
        assert "taostats.io" in obs.url
        assert "Subnets" in obs.title
        assert len(obs.accessibility_tree.strip()) > 100
    finally:
        await session.close()
        await engine.stop()


async def test_real_browser_coingecko_binance_coin_page_reachable():
    from liveweb_arena.core.browser import BrowserEngine

    engine = BrowserEngine(headless=True)
    session = await engine.new_session()
    try:
        obs = await session.goto("https://www.coingecko.com/en/coins/binance-coin")
        assert "coingecko.com" in obs.url
        assert "/en/coins/" in obs.url
        assert obs.url.endswith("/bnb") or obs.url.endswith("/binance-coin")
        assert "BNB" in obs.title or "Binance" in obs.title
        assert len(obs.accessibility_tree.strip()) > 100
    finally:
        await session.close()
        await engine.stop()


async def test_real_browser_stooq_quote_page_reachable():
    from liveweb_arena.core.browser import BrowserEngine

    engine = BrowserEngine(headless=True)
    session = await engine.new_session()
    try:
        obs = await session.goto("https://stooq.com/q/?s=jnj.us")
        assert "stooq.com" in obs.url
        assert "/q/?s=jnj.us" in obs.url.lower()
        assert "jnj" in obs.title.lower()
        assert len(obs.accessibility_tree.strip()) > 50
    finally:
        await session.close()
        await engine.stop()


async def test_real_browser_stooq_download_endpoint_returns_synthetic_observation():
    from liveweb_arena.core.browser import BrowserEngine

    engine = BrowserEngine(headless=True)
    session = await engine.new_session()
    try:
        obs = await session.goto("https://stooq.com/q/l/?s=jnj.us&f=sd2t2ohlcv&h=&e=csv")
        assert obs.url == "https://stooq.com/q/l/?s=jnj.us&f=sd2t2ohlcv&h=&e=csv"
        assert obs.title == "Download"
        assert "file download" in obs.accessibility_tree
        assert "did not call page.goto()" in obs.accessibility_tree
    finally:
        await session.close()
        await engine.stop()
