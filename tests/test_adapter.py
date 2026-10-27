"""Adapter tests with a fake transport. The response bodies are made up in the OpenAI format."""

import json

import httpx
import pytest

from harness.adapters import OpenAICompatAdapter, RequestRejected
from harness.adapters.openai_compat import sanitize

BASE = "https://api.example.com/v1"
RESPONSES_BODY = {
    "id": "resp_123",
    "model": "example-model",
    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}],
    "usage": {
        "input_tokens": 1200,
        "input_tokens_details": {"cached_tokens": 1024},
        "output_tokens": 50,
        "output_tokens_details": {"reasoning_tokens": 30},
        "total_tokens": 1250,
    },
}


def make_adapter(handler, cls=OpenAICompatAdapter, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return cls(BASE, api_key="test-key", client=client, sleep=kwargs.pop("sleep", lambda s: None), **kwargs)


def test_responses_call_normalizes_usage_and_sends_auth():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RESPONSES_BODY, headers={"x-ratelimit-remaining-requests": "2999"})

    result = make_adapter(handler).call("responses", {"model": "example-model", "input": "hello"})

    assert seen["url"] == "https://api.example.com/v1/responses"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["input"] == "hello"
    assert result.usage.cached_input_tokens == 1024
    assert result.usage.reasoning_tokens == 30
    assert result.request_id == "resp_123"
    assert result.attempts == 1
    assert result.provider == "example"
    assert result.rate_limit == {"x-ratelimit-remaining-requests": "2999"}


@pytest.mark.parametrize("protocol,path", [("chat_completions", "/v1/chat/completions"), ("messages", "/v1/messages")])
def test_protocol_paths(protocol, path):
    usage = {"prompt_tokens": 10, "completion_tokens": 2} if protocol == "chat_completions" else {"input_tokens": 10, "output_tokens": 2}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == path
        return httpx.Response(200, json={"model": "example-model", "usage": usage})

    result = make_adapter(handler).call(protocol, {"model": "example-model", "messages": []})
    assert result.usage.input_tokens == 10


def test_retries_429_using_retry_after():
    calls, sleeps = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "1.5"}, json={"error": "rate limited"})
        return httpx.Response(200, json=RESPONSES_BODY)

    result = make_adapter(handler, sleep=sleeps.append).call("responses", {"model": "example-model", "input": "x"})
    assert result.attempts == 3
    assert sleeps == [1.5, 1.5]


def test_gives_up_after_max_attempts():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "busy"})

    with pytest.raises(httpx.HTTPStatusError):
        make_adapter(handler, max_attempts=2).call("responses", {"model": "example-model", "input": "x"})


def test_client_errors_are_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": "bad request"})

    with pytest.raises(httpx.HTTPStatusError):
        make_adapter(handler).call("responses", {"model": "example-model", "input": "x"})
    assert len(calls) == 1


def test_key_comes_from_env(monkeypatch):
    monkeypatch.delenv("HARNESS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="HARNESS_API_KEY"):
        OpenAICompatAdapter(BASE)
    monkeypatch.setenv("MY_KEY", "from-env")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json=RESPONSES_BODY)

    adapter = OpenAICompatAdapter(BASE, api_key_env="MY_KEY", client=httpx.Client(transport=httpx.MockTransport(handler)))
    adapter.call("responses", {"model": "example-model", "input": "x"})
    assert seen["auth"] == "Bearer from-env"


@pytest.mark.parametrize(
    "protocol,request_body",
    [
        ("responses", {"model": "m", "input": "x", "previous_response_id": "resp_1"}),
        ("responses", {"model": "m", "input": "x", "stream": True}),
        ("responses", {"input": "x"}),
    ],
)
def test_sanitize_rejects(protocol, request_body):
    with pytest.raises(RequestRejected):
        sanitize(protocol, request_body)


def test_sanitize_copies_the_request():
    original = {"model": "m", "input": [{"role": "user", "content": "q"}]}
    cleaned = sanitize("responses", original)
    cleaned["input"].append("changed")
    assert len(original["input"]) == 1


def test_provider_rules_go_in_check():
    class StrictProvider(OpenAICompatAdapter):
        def check(self, protocol, body):
            if body.get("tool_choice") not in (None, "auto"):
                raise RequestRejected("this provider only allows tool_choice=auto")

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=RESPONSES_BODY)

    adapter = make_adapter(handler, cls=StrictProvider)
    with pytest.raises(RequestRejected):
        adapter.call("responses", {"model": "m", "input": "x", "tool_choice": "required"})
    assert calls == []  # blocked before any network call
    adapter.call("responses", {"model": "m", "input": "x", "tool_choice": "auto"})
    assert calls == [1]
