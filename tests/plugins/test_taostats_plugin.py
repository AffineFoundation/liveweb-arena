import pytest

from liveweb_arena.plugins.taostats.taostats import TaostatsPlugin


class _FakeLocator:
    def __init__(
        self,
        *,
        visible: bool = False,
        click_exc: Exception | None = None,
        wait_exc: Exception | None = None,
        select_exc: Exception | None = None,
    ):
        self._visible = visible
        self._click_exc = click_exc
        self._wait_exc = wait_exc
        self._select_exc = select_exc

    @property
    def first(self):
        return self

    async def is_visible(self, timeout=None):
        return self._visible

    async def click(self, timeout=None):
        if self._click_exc is not None:
            raise self._click_exc

    async def wait_for(self, timeout=None):
        if self._wait_exc is not None:
            raise self._wait_exc

    async def select_option(self, label=None, value=None, timeout=None):
        if self._select_exc is not None:
            raise self._select_exc


class _FakePage:
    def __init__(self, *, snapshot, text_wait_exc=None, text_locators=None, role_locators=None, css_locators=None):
        self._snapshot = snapshot
        self._text_wait_exc = text_wait_exc
        self._text_locators = text_locators or {}
        self._role_locators = role_locators or {}
        self._css_locators = css_locators or {}

    async def wait_for_function(self, script, arg, timeout=None):
        if self._text_wait_exc is not None:
            raise self._text_wait_exc

    async def evaluate(self, script):
        return self._snapshot

    def get_by_text(self, text, exact=False):
        return self._text_locators.get(text, _FakeLocator())

    def get_by_role(self, role, name=None):
        return self._role_locators.get((role, name), _FakeLocator())

    def locator(self, selector):
        return self._css_locators.get(selector, _FakeLocator())

    async def wait_for_timeout(self, timeout_ms):
        return None

    async def wait_for_load_state(self, state, timeout=None):
        return None


@pytest.mark.asyncio
async def test_taostats_list_setup_succeeds_when_show_all_fails_but_body_ready():
    plugin = TaostatsPlugin()
    page = _FakePage(
        snapshot={
            "textLength": 240,
            "textSample": "Subnets Netuid 1 2 3",
            "hasMain": True,
            "hasTable": True,
        },
        text_locators={"ALL": _FakeLocator(visible=False)},
        role_locators={("button", "ALL"): _FakeLocator(visible=False)},
        css_locators={"select": _FakeLocator(select_exc=RuntimeError("no select"))},
    )

    result = await plugin.setup_page_for_cache(page, "https://taostats.io/subnets")

    assert result["page_kind"] == "taostats_list"
    assert result["page_body_ready"] is True
    assert result["list_setup_soft_failed"] is True
    assert result["expanded"] is False


@pytest.mark.asyncio
async def test_taostats_detail_setup_succeeds_when_price_impact_missing_but_body_ready():
    plugin = TaostatsPlugin()
    page = _FakePage(
        snapshot={
            "textLength": 320,
            "textSample": "Subnet 73 Netuid 73 Statistics Emission Tempo",
            "hasMain": True,
            "hasTable": False,
        },
        css_locators={
            "text=Subnet": _FakeLocator(wait_exc=RuntimeError("missing heading")),
            "text=Netuid": _FakeLocator(wait_exc=RuntimeError("missing netuid")),
            "text=Statistics": _FakeLocator(wait_exc=RuntimeError("missing statistics")),
            "text=Transactions": _FakeLocator(wait_exc=RuntimeError("missing transactions")),
            "text=Holders": _FakeLocator(wait_exc=RuntimeError("missing holders")),
            "text=Price Impact": _FakeLocator(wait_exc=RuntimeError("missing price impact")),
        },
    )

    result = await plugin.setup_page_for_cache(page, "https://taostats.io/subnets/73")

    assert result["page_kind"] == "taostats_detail"
    assert result["page_body_ready"] is True
    assert result["detail_setup_soft_failed"] is True
    assert result["wait_target"] == "text=Price Impact"


@pytest.mark.asyncio
async def test_taostats_detail_setup_fails_when_primary_body_never_ready():
    plugin = TaostatsPlugin()
    page = _FakePage(
        snapshot={
            "textLength": 24,
            "textSample": "loading",
            "hasMain": False,
            "hasTable": False,
        },
        text_wait_exc=RuntimeError("body not ready"),
    )

    with pytest.raises(RuntimeError, match="body did not become ready"):
        await plugin.setup_page_for_cache(page, "https://taostats.io/subnets/73")
