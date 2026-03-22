import pytest

from liveweb_arena.plugins.base_client import APIFetchError
from liveweb_arena.plugins.stooq import api_client
from liveweb_arena.plugins.stooq.stooq import StooqPlugin


@pytest.mark.anyio
async def test_stooq_plugin_propagates_failure_metadata(monkeypatch):
    plugin = StooqPlugin()

    async def _raise(*args, **kwargs):
        raise APIFetchError(
            "Stooq API returned no data for symbol=msft",
            source="stooq",
            metadata={
                "plugin": "stooq",
                "failure_stage": "api_fetch",
                "failure_type": "no_data",
                "symbol": "msft",
                "request_url": "https://stooq.com/q/d/l/?s=msft&i=d",
            },
        )

    monkeypatch.setattr(
        "liveweb_arena.plugins.stooq.stooq.fetch_single_asset_data",
        _raise,
    )

    with pytest.raises(APIFetchError) as exc_info:
        await plugin.fetch_api_data("https://stooq.com/q/?s=msft")

    metadata = exc_info.value.metadata
    assert metadata["plugin"] == "stooq"
    assert metadata["failure_stage"] == "api_fetch"
    assert metadata["failure_type"] == "no_data"
    assert metadata["symbol"] == "msft"
    assert metadata["request_url"] == "https://stooq.com/q/d/l/?s=msft&i=d"
    assert metadata["page_url"] == "https://stooq.com/q/?s=msft"


@pytest.mark.anyio
async def test_fetch_single_asset_data_uses_cached_symbol_when_rate_limited(monkeypatch):
    monkeypatch.setattr(api_client, "_load_symbol_cache", lambda symbol, allow_stale=False: {"symbol": symbol, "close": 1.23})
    token = api_client._rate_limited.set(True)
    try:
        data = await api_client.fetch_single_asset_data("jnj.us")
    finally:
        api_client._rate_limited.reset(token)
    assert data["symbol"] == "jnj.us"


def test_stooq_quote_warmup_swallows_sync_failures(monkeypatch):
    plugin = StooqPlugin()
    monkeypatch.setattr(StooqPlugin, "_quote_warmup_started", False)
    monkeypatch.setattr("liveweb_arena.plugins.stooq.stooq.asyncio.get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError()))
    def _raise_run(coro):
        coro.close()
        raise RuntimeError("warmup failed")
    monkeypatch.setattr("liveweb_arena.plugins.stooq.stooq.asyncio.run", _raise_run)

    plugin._initialize_quote_warmup()


def test_stooq_quote_warmup_is_single_shot(monkeypatch):
    plugin = StooqPlugin()
    monkeypatch.setattr("liveweb_arena.plugins.stooq.stooq.initialize_cache", lambda: None)

    started = []

    class _DummyThread:
        def __init__(self, *, target, name, daemon):
            self._target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            started.append((self.name, self.daemon))

    monkeypatch.setattr("liveweb_arena.plugins.stooq.stooq.threading.Thread", _DummyThread)
    monkeypatch.setattr("liveweb_arena.plugins.stooq.stooq.asyncio.get_running_loop", lambda: object())
    monkeypatch.setattr(api_client, "_load_symbol_cache", lambda symbol, allow_stale=False: None)
    monkeypatch.setattr(StooqPlugin, "_quote_warmup_started", False)

    plugin._initialize_quote_warmup()
    plugin._initialize_quote_warmup()

    assert started == [("stooq-quote-warmup", True)]
