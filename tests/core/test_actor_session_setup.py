import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import Actor, _maybe_promote_disallowed_domain_failure
from liveweb_arena.core.task_registry_loader import parse_task_id as parse_task_id_for_runtime


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


class _FakeInterceptor:
    def __init__(self, metadata):
        self._metadata = metadata

    def get_last_blocked_document_metadata(self):
        return dict(self._metadata)


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


def test_actor_build_protocol_defaults_to_eval_mode(tmp_path):
    actor = Actor(
        api_key="local",
        cache_dir=tmp_path,
        use_cache=True,
        llm_router=None,
        llm_prompt_profile="gpt54_strict_domains",
    )

    assert actor.runtime_profile == "strict_eval"

    eval_prompt = actor._build_protocol(runtime_profile="strict_eval").build_step_prompt(
        obs=type("Obs", (), {"url": "https://example.com", "title": "t", "accessibility_tree": "tree"})(),
        trajectory=[],
        current_step=1,
        max_steps=5,
    )
    collect_prompt = actor._build_protocol(runtime_profile="fast_collect").build_step_prompt(
        obs=type("Obs", (), {"url": "https://example.com", "title": "t", "accessibility_tree": "tree"})(),
        trajectory=[],
        current_step=1,
        max_steps=5,
    )

    assert "Never call stop with {\"answers\": {}}." not in eval_prompt
    assert "Never call stop with {\"answers\": {}}." in collect_prompt


def test_maybe_promote_disallowed_domain_failure_only_in_collect_mode():
    interceptor = _FakeInterceptor(
        {
            "blocked_url": "https://finance.yahoo.com/quote/V/",
            "blocked_domain": "finance.yahoo.com",
            "allowed_domains": ["coingecko.com"],
        }
    )
    trajectory = [
        type(
            "Step",
            (),
            {
                "observation": type(
                    "Obs",
                    (),
                    {
                        "url": "https://finance.yahoo.com/quote/V/",
                        "accessibility_tree": "Domain not allowed",
                    },
                )()
            },
        )()
    ]

    eval_failure, eval_audit = _maybe_promote_disallowed_domain_failure(
        interceptor=interceptor,
        trajectory=trajectory,
        allowed_domains={"coingecko.com"},
        failure_reason="browser_error",
        reachability_audit=None,
        mode="eval",
    )
    collect_failure, collect_audit = _maybe_promote_disallowed_domain_failure(
        interceptor=interceptor,
        trajectory=trajectory,
        allowed_domains={"coingecko.com"},
        failure_reason="browser_error",
        reachability_audit=None,
        mode="collect",
    )

    assert eval_failure == "browser_error"
    assert eval_audit is None
    assert collect_failure == "disallowed_domain"
    assert collect_audit is not None
    assert collect_audit.classification == "model_disallowed_domain"


def test_strict_eval_can_use_external_task_registry(monkeypatch, tmp_path):
    registry_root = tmp_path / "upstream_like"
    registry_dir = registry_root / "liveweb_arena" / "core"
    registry_dir.mkdir(parents=True)
    (registry_dir / "task_registry.py").write_text(
        "def parse_task_id(task_id):\n"
        "    return {\n"
        "        'task_id': task_id,\n"
        "        'combo_index': 0,\n"
        "        'template_ids': (999,),\n"
        "        'templates': [('coingecko', 'coingecko_price')],\n"
        "        'variation_seed': 7,\n"
        "        'num_tasks': 3,\n"
        "    }\n"
        "\n"
        "def max_task_id():\n"
        "    return 999999\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LIVEWEB_STRICT_TASK_REGISTRY_DIR", str(registry_root))

    parsed = parse_task_id_for_runtime(123, runtime_profile="strict_eval")
    assert parsed["templates"] == [("coingecko", "coingecko_price")]
    assert parsed["num_tasks"] == 3
