"""Fake provider for testing the harness without spending anything.

`SimulatedAdapter` acts like an API with automatic prefix caching. A request's cached tokens are the
pieces at the start that match the previous request with the same `prompt_cache_key`, as long as that's
at least `min_cached_tokens`. Tokens are counted as characters divided by `chars_per_token`.

This is my model of prefix caching, not how any real provider does it, and it says nothing about whether tasks
succeed. Its numbers are only for testing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from harness.adapters.base import Adapter, CallResult
from harness.segments import first_divergence, segment
from harness.usage import Protocol, Usage


class SimulatedAdapter(Adapter):
    provider = "example"

    def __init__(
        self,
        respond: Callable[[Protocol, dict[str, Any]], dict[str, Any]],
        *,
        model: str = "example-model",
        chars_per_token: float = 4.0,
        min_cached_tokens: int = 1024,
        output_tokens: int = 150,
        reasoning_tokens: int = 100,
    ):
        self.respond = respond
        self.model = model
        self.chars_per_token = chars_per_token
        self.min_cached_tokens = min_cached_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens
        self._last: dict[str | None, list] = {}

    def call(self, protocol: Protocol, request: dict[str, Any]) -> CallResult:
        segs = segment(protocol, request)
        key = request.get("prompt_cache_key")
        prev = self._last.get(key)
        cached_chars = 0
        if prev is not None:
            div = first_divergence(prev, segs)
            shared = len(prev) if div is None else div
            cached_chars = sum(s.chars for s in segs[: min(shared, len(segs))])
        self._last[key] = segs

        total = int(sum(s.chars for s in segs) / self.chars_per_token)
        cached = min(int(cached_chars / self.chars_per_token), total)
        if cached < self.min_cached_tokens:
            cached = 0
        usage = Usage(total, cached, self.output_tokens, self.reasoning_tokens, protocol)
        body = self.respond(protocol, request)
        return CallResult(self.provider, protocol, self.model, request, body, usage, 0.0, 1)


def scripted_agent_turns(turns: int, output_chars: int) -> list[dict[str, Any]]:
    """A fake tool-heavy conversation in the Responses format, with one big tool output per turn."""
    items: list[dict[str, Any]] = [{"role": "user", "content": "Investigate the failing job and fix it."}]
    for i in range(turns):
        body = "\n".join(f"[{i}] log line {j}: " + "x" * 60 for j in range(output_chars // 80))
        items.append({"type": "function_call", "call_id": f"c{i}", "name": "read_log", "arguments": json.dumps({"part": i})})
        items.append({"type": "function_call_output", "call_id": f"c{i}", "output": body})
    return items
