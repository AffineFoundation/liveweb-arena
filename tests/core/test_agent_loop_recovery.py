import pytest

from liveweb_arena.core.agent_loop import AgentLoop
from liveweb_arena.core.agent_protocol import FunctionCallingProtocol
from liveweb_arena.core.models import BrowserObservation, CompositeTask
from liveweb_arena.utils.llm_client import LLMResponse, ToolCall


class _FakeSession:
    async def goto(self, url: str):
        return BrowserObservation(url=url, title="Blank", accessibility_tree="root")

    async def execute_action(self, action):
        return BrowserObservation(url="https://example.com", title="Done", accessibility_tree="root")


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
