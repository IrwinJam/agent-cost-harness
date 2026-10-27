"""Cache layout: keep the start of the prompt the same between calls.

Prefix caching only helps if the start of the prompt is exactly the same every call. This policy sorts
the tool list and its keys so an agent that rebuilds its tools in a different order doesn't break the
cache. It moves lines with timestamps, dates or IDs out of the system prompt and into a short message
at the end of the conversation. It also sets `prompt_cache_key` so a task's calls hit the same cache,
plus `prompt_cache_retention` on Responses.

It never touches the conversation history, because rewriting history is exactly what breaks the cache.
"""

from __future__ import annotations

import re
from typing import Any

from harness.policies import conv
from harness.policies.base import Policy, SessionContext
from harness.usage import Protocol

DEFAULT_VOLATILE = (
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}",  # ISO date-time
    r"\b\d{1,2}:\d{2}(:\d{2})?\s?([AaPp][Mm])?\b",  # clock time
    r"\b\d{4}-\d{2}-\d{2}\b",  # ISO date
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",  # UUID
)


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    return value


class CacheStableLayout(Policy):
    name = "cache_layout"

    def __init__(
        self,
        *,
        volatile_patterns: tuple[str, ...] = DEFAULT_VOLATILE,
        context_role: str = "user",
        context_prefix: str = "[Context] ",
        cache_retention: str | None = "24h",
        set_cache_key: bool = True,
    ):
        # The role for moved lines is a setting because not every provider allows a "developer" or
        # "system" message in the middle of a conversation. "user" works everywhere.
        self._volatile = re.compile("|".join(f"(?:{p})" for p in volatile_patterns)) if volatile_patterns else None
        self._context_role = context_role
        self._context_prefix = context_prefix
        self._cache_retention = cache_retention
        self._set_cache_key = set_cache_key

    def apply(self, protocol: Protocol, request: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        conv.check(protocol)
        if request.get("tools"):
            request["tools"] = sorted((_canonical(t) for t in request["tools"]), key=conv.tool_schema_name)

        moved = self._move_volatile(protocol, request)
        if moved:
            conv.items(protocol, request).append(
                {"role": self._context_role, "content": self._context_prefix + "\n".join(moved)}
            )

        if self._set_cache_key:
            request.setdefault("prompt_cache_key", ctx.task_id)
        if protocol == "responses" and self._cache_retention:
            request.setdefault("prompt_cache_retention", self._cache_retention)
        return request

    def _move_volatile(self, protocol: Protocol, request: dict[str, Any]) -> list[str]:
        if self._volatile is None:
            return []
        moved: list[str] = []
        if protocol == "responses" and isinstance(request.get("instructions"), str):
            request["instructions"], lines = self._split(request["instructions"])
            moved += lines
        for item in conv.items(protocol, request):
            if item.get("role") not in ("system", "developer"):
                break  # only the system messages at the start. Anything after is history
            if isinstance(item.get("content"), str):
                item["content"], lines = self._split(item["content"])
                moved += lines
        return moved

    def _split(self, text: str) -> tuple[str, list[str]]:
        keep, moved = [], []
        for line in text.split("\n"):
            (moved if self._volatile.search(line) else keep).append(line)
        return "\n".join(keep), moved
