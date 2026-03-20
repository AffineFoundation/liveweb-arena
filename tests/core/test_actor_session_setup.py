import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import Actor


class _FakePlugin:
    allowed_domains = ["stooq.com"]

    def is_url_allowed(self, url: str) -> bool:
        return "stooq.com" in url


class _FakeSession:
    def __init__(self, *, fail_once: bool = False):
        self._fail_once = fail_once
        self.closed = False

    async def set_cache_interceptor(self, interceptor):
        if self._fail_once:
            self._fail_once = False
            raise RuntimeError("Target page, context or browser has been closed")
        return None

    async def close(self):
        self.closed = True


class _FakeBrowser:
    def __init__(self):
        self.calls = 0
        self.sessions = [_FakeSession(fail_once=True), _FakeSession(fail_once=False)]
        self.ensure_calls = 0

    async def ensure_healthy(self):
        self.ensure_calls += 1

    async def new_session(self):
        session = self.sessions[self.calls]
        self.calls += 1
        return session


@pytest.mark.anyio
async def test_setup_interceptor_rebuilds_session_after_transport_error(tmp_path):
    actor = Actor(api_key="local", cache_dir=tmp_path, use_cache=True, llm_router=None)
    actor.browser = _FakeBrowser()

    session, interceptor = await actor._setup_interceptor(
        session=await actor.browser.new_session(),
        cached_pages={},
        allowed_domains={"stooq.com"},
        blocked_patterns=[],
        plugins_used={"stooq": _FakePlugin()},
    )

    assert interceptor is not None
    assert actor.browser.calls == 2
    assert actor.browser.ensure_calls == 1
    assert session is actor.browser.sessions[1]
    assert actor.browser.sessions[0].closed is True
