"""Adapter for any API that speaks the OpenAI format (Responses or Chat Completions), plus the
Anthropic-style Messages format if the provider supports it.

Set the base URL and the name of the env var that holds the key. The adapter retries on rate limits and
server errors, and it refuses `previous_response_id`, because then providers usually only report usage
for the newest turn and the cost tracking breaks. Send the whole conversation instead.

If your provider has extra rules (fields it rejects, values it doesn't allow), subclass this and add
them in `check`.
"""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Callable
from typing import Any

import httpx

from harness.adapters.base import Adapter, CallResult, RequestRejected
from harness.usage import Protocol, normalize

_PATHS: dict[Protocol, str] = {
    "responses": "/responses",
    "chat_completions": "/chat/completions",
    "messages": "/messages",
}

_RETRY_STATUS = {429, 500, 502, 503, 504}


class OpenAICompatAdapter(Adapter):
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "HARNESS_API_KEY",
        provider: str = "example",
        client: httpx.Client | None = None,
        max_attempts: int = 4,
        timeout_s: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        messages_input_includes_cached: bool = False,
    ):
        api_key = api_key or os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"API key missing: pass api_key or set {api_key_env}")
        self.provider = provider
        self._client = client or httpx.Client(timeout=timeout_s)
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._messages_input_includes_cached = messages_input_includes_cached

    def check(self, protocol: Protocol, body: dict[str, Any]) -> None:
        """Provider-specific rules go here in a subclass. Raise `RequestRejected` to block a request."""

    def call(self, protocol: Protocol, request: dict[str, Any]) -> CallResult:
        body = sanitize(protocol, request)
        self.check(protocol, body)
        url = self._base_url + _PATHS[protocol]
        started = time.perf_counter()
        attempts, http_response = self._post_with_retry(url, body)
        latency = time.perf_counter() - started
        payload = http_response.json()
        kwargs = {"input_includes_cached": self._messages_input_includes_cached} if protocol == "messages" else {}
        return CallResult(
            provider=self.provider,
            protocol=protocol,
            model=payload.get("model", body["model"]),
            request=body,
            response=payload,
            usage=normalize(protocol, payload.get("usage"), **kwargs),
            latency_s=latency,
            attempts=attempts,
            request_id=http_response.headers.get("x-request-id") or payload.get("id"),
            rate_limit={k: v for k, v in http_response.headers.items() if k.lower().startswith("x-ratelimit-")},
        )

    def _post_with_retry(self, url: str, body: dict[str, Any]) -> tuple[int, httpx.Response]:
        for attempt in range(1, self._max_attempts + 1):
            response = self._client.post(url, json=body, headers=self._headers)
            if response.status_code < 400:
                return attempt, response
            if response.status_code not in _RETRY_STATUS or attempt == self._max_attempts:
                response.raise_for_status()
            self._sleep(_retry_delay(response, attempt))
        raise AssertionError("unreachable")


def sanitize(protocol: Protocol, request: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the request that's safe to send and track, or raise `RequestRejected`."""
    if protocol not in _PATHS:
        raise RequestRejected(f"unknown protocol: {protocol}")
    body = copy.deepcopy(request)
    if "model" not in body:
        raise RequestRejected("request has no model")
    if body.get("stream"):
        raise RequestRejected("streaming isn't supported yet (usage only shows up in the last event)")
    if protocol == "responses" and body.get("previous_response_id"):
        raise RequestRejected("previous_response_id makes usage cover only the newest turn. Send the whole conversation")
    return body


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return min(2.0 ** attempt, 30.0)
