import asyncio
from types import SimpleNamespace

import pytest

from liveweb_arena.utils.llm_client import LLMClient, LLMResponse


class _FakeResponseUsage:
    def model_dump(self):
        return {"prompt_tokens": 10, "completion_tokens": 2}


class _FakeToolCall:
    def __init__(self, name: str, arguments: str):
        self.id = "call_1"
        self.function = SimpleNamespace(name=name, arguments=arguments)


class _FakeChoice:
    def __init__(self, content: str = "", tool_calls=None):
        self.message = SimpleNamespace(content=content, tool_calls=tool_calls or [])


class _FakeChatCompletions:
    def __init__(self, recorder, *, sleep_s: float = 0.0):
        self._recorder = recorder
        self._sleep_s = sleep_s

    async def create(self, **kwargs):
        self._recorder.append(kwargs)
        if self._sleep_s:
            await asyncio.sleep(self._sleep_s)
        return SimpleNamespace(
            id=kwargs["extra_body"]["request_id"],
            choices=[_FakeChoice(tool_calls=[_FakeToolCall("goto", '{"url":"https://example.com"}')])],
            usage=_FakeResponseUsage(),
        )


class _FakeAsyncOpenAI:
    def __init__(self, recorder, *, sleep_s: float = 0.0, **kwargs):
        self._recorder = recorder
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(recorder, sleep_s=sleep_s))

    async def close(self):
        return None


class _FakeHTTPXResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code


class _FakeAsyncHTTPClient:
    def __init__(self, recorder, *args, **kwargs):
        self._recorder = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json):
        self._recorder.append((url, json))
        return _FakeHTTPXResponse()


class _FakeBuiltHTTPClient:
    def __init__(self, recorder, *args, **kwargs):
        recorder.append(kwargs)


@pytest.mark.anyio
async def test_chat_with_tools_sends_request_id_in_extra_body(monkeypatch):
    requests = []

    def _factory(**kwargs):
        return _FakeAsyncOpenAI(requests, **kwargs)

    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.AsyncOpenAI", _factory)

    client = LLMClient(base_url="http://127.0.0.1:31050/v1", api_key="local")
    response = await client.chat_with_tools(
        system="system",
        user="user",
        model="qwen",
        tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
        temperature=0.0,
        timeout_s=5,
    )

    assert isinstance(response, LLMResponse)
    assert requests
    request_payload = requests[0]
    assert "extra_body" in request_payload
    assert request_payload["extra_body"]["request_id"].startswith("liveweb-tools-")
    assert response.request_id == request_payload["extra_body"]["request_id"]


@pytest.mark.anyio
async def test_chat_with_tools_sends_reasoning_controls_via_extra_body(monkeypatch):
    requests = []

    def _factory(**kwargs):
        return _FakeAsyncOpenAI(requests, **kwargs)

    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.AsyncOpenAI", _factory)
    monkeypatch.setenv("LIVEWEB_ENABLE_THINKING", "0")
    monkeypatch.setenv("LIVEWEB_SEPARATE_REASONING", "1")

    client = LLMClient(base_url="http://127.0.0.1:31050/v1", api_key="local")
    await client.chat_with_tools(
        system="system",
        user="user",
        model="qwen",
        tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
        temperature=0.0,
        timeout_s=5,
    )

    payload = requests[0]
    assert payload["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["extra_body"]["separate_reasoning"] is True


@pytest.mark.anyio
async def test_chat_with_tools_recovery_uses_stochastic_small_request(monkeypatch):
    requests = []

    def _factory(**kwargs):
        return _FakeAsyncOpenAI(requests, **kwargs)

    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.AsyncOpenAI", _factory)
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_TEMPERATURE", "0.35")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_TOP_P", "0.95")

    client = LLMClient(base_url="http://127.0.0.1:31050/v1", api_key="local")
    await client.chat_with_tools_recovery(
        system="system",
        user="user prompt",
        assistant_prefix="<tool_call>{\"name\":\"goto\"",
        model="qwen",
        tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
        max_new_tokens=96,
    )

    payload = requests[0]
    assert payload["temperature"] == 0.35
    assert payload["top_p"] == 0.95
    assert payload["max_tokens"] == 96
    assert payload["max_completion_tokens"] == 96
    assert payload["messages"][2] == {"role": "assistant", "content": "<tool_call>{\"name\":\"goto\""}
    assert "Continue from the assistant's last output" in payload["messages"][3]["content"]


@pytest.mark.anyio
async def test_chat_with_tools_timeout_triggers_abort(monkeypatch):
    requests = []
    aborts = []

    def _openai_factory(**kwargs):
        return _FakeAsyncOpenAI(requests, sleep_s=0.2, **kwargs)

    def _httpx_factory(*args, **kwargs):
        return _FakeAsyncHTTPClient(aborts, *args, **kwargs)

    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.AsyncOpenAI", _openai_factory)
    monkeypatch.setattr("liveweb_arena.utils.llm_client.httpx.AsyncClient", _httpx_factory)

    client = LLMClient(base_url="http://127.0.0.1:31050/v1", api_key="local", strict_serial=True)

    with pytest.raises(Exception):
        await client.chat_with_tools(
            system="system",
            user="user",
            model="qwen",
            tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
            temperature=0.0,
            timeout_s=0.01,
        )

    assert requests
    assert aborts
    abort_url, abort_payload = aborts[0]
    assert abort_url == "http://127.0.0.1:31050/abort_request"
    assert abort_payload["rid"] == requests[0]["extra_body"]["request_id"]
    assert abort_payload["abort_all"] is False


@pytest.mark.parametrize(
    ("base_url", "should_bypass"),
    [
        ("http://localhost:31050/v1", True),
        ("http://127.0.0.1:31050/v1", True),
        ("http://10.0.0.8:31050/v1", True),
        ("http://192.168.1.10:31050/v1", True),
        ("http://172.16.0.5:31050/v1", True),
        ("http://169.254.0.20:31050/v1", True),
        ("http://[::1]:31050/v1", True),
        ("https://api.openai.com/v1", False),
        ("https://example.com/v1", False),
    ],
)
def test_should_bypass_proxy_for_local_and_private_addresses(base_url, should_bypass):
    assert LLMClient._should_bypass_proxy(base_url) is should_bypass


def test_build_httpx_client_disables_trust_env_for_local_routes(monkeypatch):
    built_clients = []

    def _httpx_factory(*args, **kwargs):
        return _FakeBuiltHTTPClient(built_clients, *args, **kwargs)

    monkeypatch.setattr("liveweb_arena.utils.llm_client.httpx.AsyncClient", _httpx_factory)

    client = LLMClient(base_url="http://127.0.0.1:31050/v1", api_key="local")
    client._build_httpx_client(
        base_url="http://127.0.0.1:31050/v1",
        timeout=object(),
    )
    client._build_httpx_client(
        base_url="https://api.openai.com/v1",
        timeout=object(),
    )

    assert built_clients[0]["trust_env"] is False
    assert built_clients[1]["trust_env"] is True
