import pytest

from liveweb_arena.core.agent_loop import AgentLoop, BrowserFatalError
from liveweb_arena.core.agent_protocol import FunctionCallingProtocol
from liveweb_arena.core.models import BrowserObservation, CompositeTask
from liveweb_arena.utils.llm_client import LLMResponse, ToolCall


class _FakeSession:
    async def goto(self, url: str):
        return BrowserObservation(url=url, title="Blank", accessibility_tree="root")

    async def execute_action(self, action):
        return BrowserObservation(url="https://example.com", title="Done", accessibility_tree="root")

    def get_last_navigation_metadata(self):
        return None


class _FakeSessionWithBlockedSequence:
    def __init__(self, observations, metadatas):
        self._observations = list(observations)
        self._metadatas = list(metadatas)
        self._last_navigation_metadata = None

    async def goto(self, url: str):
        self._last_navigation_metadata = None
        return BrowserObservation(url=url, title="Blank", accessibility_tree="root")

    async def execute_action(self, action):
        self._last_navigation_metadata = self._metadatas.pop(0)
        return self._observations.pop(0)

    def get_last_navigation_metadata(self):
        return self._last_navigation_metadata


class _FakeSubTask:
    plugin_name = "coingecko"
    answer_tag = "a1"
    expected_steps = 1


class _FakeLLMClient:
    def __init__(self, initial_response, recovery_responses=None):
        self.initial_response = initial_response
        self.recovery_responses = list(recovery_responses or [])
        self.recovery_calls = 0
        self.recovery_messages = []

    async def chat_with_tools(self, **kwargs):
        return self.initial_response

    async def chat_with_tools_recovery(self, **kwargs):
        self.recovery_calls += 1
        self.recovery_messages.append(kwargs.get("messages"))
        return self.recovery_responses.pop(0)


class _QueuedLLMClient:
    def __init__(self, responses):
        self._responses = list(responses)

    async def chat_with_tools(self, **kwargs):
        return self._responses.pop(0)

    async def chat_with_tools_recovery(self, **kwargs):
        raise AssertionError("Recovery should not be called in this test")


def _task():
    return CompositeTask(
        subtasks=[_FakeSubTask()],
        combined_intent="Find the answer",
        plugin_hints={},
        seed=1,
    )


@pytest.mark.anyio
async def test_agent_loop_recovers_from_truncated_tool_json(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_FORMAT_RECOVERY", "1")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_MAX_RETRIES", "2")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_MAX_NEW_TOKENS", "64")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content="<tool_call>{\"name\":\"goto\",\"arguments\":{\"url\":\"https://example.com\"}"),
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\"}}"})]),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
    )

    trajectory, final_answer, usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer == {"answers": {"a1": "42"}}
    assert loop.is_parse_failed() is False
    stats = loop.get_format_recovery_stats()
    assert stats["format_recovery_attempts"] == 1
    assert stats["format_recovery_successes"] == 1
    assert llm_client.recovery_calls == 1
    assert trajectory[-1].action.action_type == "stop"
    messages = llm_client.recovery_messages[0]
    assistant_messages = [item for item in messages if item["role"] == "assistant"]
    assert assistant_messages == []
    assert messages[-1]["role"] == "user"
    assert "invalid tool-call formatting" in messages[-1]["content"].lower()


@pytest.mark.anyio
async def test_agent_loop_does_not_recover_terminal_natural_language(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_FORMAT_RECOVERY", "1")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content="I will inspect the page and then answer in plain English."),
        recovery_responses=[],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
    )

    trajectory, final_answer, usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer is None
    assert loop.is_parse_failed() is True
    stats = loop.get_format_recovery_stats()
    assert stats["format_recovery_attempts"] == 0
    assert llm_client.recovery_calls == 0
    assert trajectory[-1].action is None


@pytest.mark.anyio
async def test_agent_loop_marks_parse_failed_after_recovery_exhausted(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_FORMAT_RECOVERY", "1")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_MAX_RETRIES", "2")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content="<tool_call>{\"name\":\"goto\""),
        recovery_responses=[
            LLMResponse(content="<tool_call>{\"name\":\"goto\""),
            LLMResponse(content="<tool_call>{\"name\":\"goto\""),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
    )

    trajectory, final_answer, usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer is None
    assert loop.is_parse_failed() is True
    stats = loop.get_format_recovery_stats()
    assert stats["format_recovery_attempts"] == 1
    assert stats["format_recovery_successes"] == 0
    assert stats["format_recovery_exhausted"] == 1
    assert llm_client.recovery_calls == 2
    assert trajectory[-1].action is None


@pytest.mark.anyio
async def test_agent_loop_limits_empty_recovery_retries(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_FORMAT_RECOVERY", "1")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_MAX_RETRIES", "4")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_EMPTY_MAX_RETRIES", "2")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content=""),
        recovery_responses=[
            LLMResponse(content=""),
            LLMResponse(content=""),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
    )

    trajectory, final_answer, usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer is None
    assert loop.is_parse_failed() is True
    stats = loop.get_format_recovery_stats()
    assert stats["format_recovery_attempts"] == 1
    assert stats["format_recovery_exhausted"] == 1
    assert llm_client.recovery_calls == 2


@pytest.mark.anyio
async def test_agent_loop_skips_recovery_when_context_budget_exceeded(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_FORMAT_RECOVERY", "1")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_CONTEXT_LENGTH", "64")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_MAX_NEW_TOKENS", "32")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_TOKEN_MARGIN", "16")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content="<tool_call>{\"name\":\"goto\""),
        recovery_responses=[],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
    )
    long_obs = BrowserObservation(url="https://example.com", title="Long", accessibility_tree="x" * 2000)
    loop._trajectory = []
    messages = loop._build_recovery_messages(
        system_prompt="sys",
        user_prompt=f"obs {long_obs.accessibility_tree}",
        failure_class="recoverable_truncated_tool_json",
    )

    raw, action, usage, override = await loop._attempt_format_recovery(
        model="qwen",
        seed=1,
        raw_response="<tool_call>{\"name\":\"goto\"",
        messages=messages,
        failure_class="recoverable_truncated_tool_json",
    )

    assert action is None
    assert usage is None
    assert override == "recoverable_context_overflow"
    assert llm_client.recovery_calls == 0


@pytest.mark.anyio
async def test_agent_loop_explicit_disable_recovery_overrides_env(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_FORMAT_RECOVERY", "1")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content="<tool_call>{\"name\":\"goto\""),
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\"}}"})]),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
        enable_format_recovery=False,
    )

    trajectory, final_answer, usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer is None
    assert loop.is_parse_failed() is True
    assert llm_client.recovery_calls == 0


@pytest.mark.anyio
async def test_agent_loop_failfasts_after_consecutive_disallowed_domain_hits(monkeypatch):
    monkeypatch.setenv("LIVEWEB_FAILFAST_DISALLOWED_DOMAIN_CONSECUTIVE", "2")
    monkeypatch.setenv("LIVEWEB_FAILFAST_DISALLOWED_DOMAIN_TOTAL", "3")

    blocked_obs = BrowserObservation(
        url="https://finance.yahoo.com/quote/V/",
        title="Domain not allowed",
        accessibility_tree="Domain not allowed. Please return to an allowed site and continue the task.",
    )
    session = _FakeSessionWithBlockedSequence(
        observations=[blocked_obs, blocked_obs],
        metadatas=[
            {
                "classification_hint": "model_disallowed_domain",
                "url": "https://finance.yahoo.com/quote/V/",
                "evidence": {"blocked_url": "https://finance.yahoo.com/quote/V/"},
            },
            {
                "classification_hint": "model_disallowed_domain",
                "url": "https://finance.yahoo.com/quote/V/",
                "evidence": {"blocked_url": "https://finance.yahoo.com/quote/V/"},
            },
        ],
    )
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "goto", "arguments": "{\"url\":\"https://finance.yahoo.com/quote/V/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="call_2", function={"name": "goto", "arguments": "{\"url\":\"https://finance.yahoo.com/quote/V/\"}"})]),
        ]
    )
    loop = AgentLoop(
        session=session,
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=5,
    )

    with pytest.raises(BrowserFatalError) as exc_info:
        await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert "disallowed-domain" in str(exc_info.value)
    assert exc_info.value.url == "https://finance.yahoo.com/quote/V/"
    assert len(loop.get_trajectory()) == 2


@pytest.mark.anyio
async def test_agent_loop_resets_disallowed_domain_consecutive_counter_after_recovery(monkeypatch):
    monkeypatch.setenv("LIVEWEB_FAILFAST_DISALLOWED_DOMAIN_CONSECUTIVE", "2")
    monkeypatch.setenv("LIVEWEB_FAILFAST_DISALLOWED_DOMAIN_TOTAL", "3")

    blocked_obs = BrowserObservation(
        url="https://finance.yahoo.com/quote/V/",
        title="Domain not allowed",
        accessibility_tree="Domain not allowed. Please return to an allowed site and continue the task.",
    )
    good_obs = BrowserObservation(
        url="https://example.com",
        title="Example",
        accessibility_tree="root content with enough accessible text to avoid blank observation failfast",
    )
    session = _FakeSessionWithBlockedSequence(
        observations=[blocked_obs, good_obs, blocked_obs],
        metadatas=[
            {
                "classification_hint": "model_disallowed_domain",
                "url": "https://finance.yahoo.com/quote/V/",
                "evidence": {"blocked_url": "https://finance.yahoo.com/quote/V/"},
            },
            None,
            {
                "classification_hint": "model_disallowed_domain",
                "url": "https://finance.yahoo.com/quote/V/",
                "evidence": {"blocked_url": "https://finance.yahoo.com/quote/V/"},
            },
        ],
    )
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "goto", "arguments": "{\"url\":\"https://finance.yahoo.com/quote/V/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="call_2", function={"name": "goto", "arguments": "{\"url\":\"https://example.com\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="call_3", function={"name": "goto", "arguments": "{\"url\":\"https://finance.yahoo.com/quote/V/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="call_4", function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\"}}"})]),
        ]
    )
    loop = AgentLoop(
        session=session,
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=5,
    )

    trajectory, final_answer, _usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer == {"answers": {"a1": "42"}}
    assert len(trajectory) == 4
