"""Cache layout: keep the start of the prompt the same between calls.

Prefix caching only helps if the start of the prompt is exactly the same every call. This policy sorts
the tool list and its keys so an agent that rebuilds its tools in a different order doesn't break the
cache.

It also watches the system prompt for lines that actually change between calls, like a timestamp that
ticks. A line counts as changing when it shows up differing only in its numbers from the line it
replaced in the previous call. From then on that line gets moved out of the system prompt into a short
message at the end of the conversation, so it stops breaking the cache for everything after it. Lines
that never change are left alone, even if they have times or dates in them. (The first version guessed
from patterns, moved a fixed example line in a real benchmark's policy, and made the agent worse.)

Last, it sets `prompt_cache_key`. By default the key is a hash of the tools and system prompt, so every
conversation that starts the same way shares one cache, plus `prompt_cache_retention` on Responses.

It never touches the conversation history, because rewriting history is exactly what breaks the cache.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from harness.policies import conv
from harness.policies.base import Policy, SessionContext
from harness.usage import Protocol

# Runs of hex digits that include at least one number: times, dates, counters, UUID pieces.
_NUMBERS = re.compile(r"[0-9A-Fa-f]*\d[0-9A-Fa-f]*")


def template(line: str) -> str:
    """The line with its numbers blanked out, so `12:01` and `12:02` versions of a line match."""
    return _NUMBERS.sub("#", line)


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
        volatile_patterns: tuple[str, ...] = (),
        context_role: str = "user",
        context_prefix: str = "[Context] ",
        cache_retention: str | None = "24h",
        cache_key: Literal["prefix", "task"] | None = "prefix",
    ):
        # volatile_patterns: regexes for lines you KNOW change, moved from the very first call.
        # By default nothing is moved until it's actually seen changing.
        # The role for moved lines is a setting because not every provider allows a "developer" or
        # "system" message in the middle of a conversation. "user" works everywhere.
        if cache_key not in ("prefix", "task", None):
            raise ValueError("cache_key must be 'prefix', 'task' or None")
        self._patterns = re.compile("|".join(f"(?:{p})" for p in volatile_patterns)) if volatile_patterns else None
        self._context_role = context_role
        self._context_prefix = context_prefix
        self._cache_retention = cache_retention
        self._cache_key = cache_key

    def apply(self, protocol: Protocol, request: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        conv.check(protocol)
        state = ctx.state_for(self.name)
        if request.get("tools"):
            request["tools"] = sorted((_canonical(t) for t in request["tools"]), key=conv.tool_schema_name)

        lines = [line for text in self._system_texts(protocol, request) for line in text.split("\n")]
        self._learn(state, lines)
        moved = self._move(protocol, request, state["volatile"])
        if moved:
            conv.items(protocol, request).append(
                {"role": self._context_role, "content": self._context_prefix + "\n".join(moved)}
            )

        if self._cache_key == "task":
            request.setdefault("prompt_cache_key", ctx.task_id)
        elif self._cache_key == "prefix":
            request.setdefault("prompt_cache_key", prefix_key(request.get("tools"), self._system_texts(protocol, request)))
        if protocol == "responses" and self._cache_retention:
            request.setdefault("prompt_cache_retention", self._cache_retention)
        return request

    @staticmethod
    def _learn(state: dict[str, Any], lines: list[str]) -> None:
        """Mark a line's template as changing when a new line replaced an old one with the same template."""
        volatile: set[str] = state.setdefault("volatile", set())
        prev = state.get("prev_lines")
        if prev is not None:
            prev_set, cur_set = set(prev), set(lines)
            gone = {template(l) for l in prev if l not in cur_set}
            for line in lines:
                if line not in prev_set and template(line) in gone:
                    volatile.add(template(line))
        state["prev_lines"] = lines

    def _is_volatile(self, line: str, volatile: set[str]) -> bool:
        if template(line) in volatile:
            return True
        return bool(self._patterns and self._patterns.search(line))

    def _move(self, protocol: Protocol, request: dict[str, Any], volatile: set[str]) -> list[str]:
        if not volatile and self._patterns is None:
            return []
        moved: list[str] = []

        def split(text: str) -> str:
            keep = []
            for line in text.split("\n"):
                (moved if self._is_volatile(line, volatile) else keep).append(line)
            return "\n".join(keep)

        if protocol == "responses" and isinstance(request.get("instructions"), str):
            request["instructions"] = split(request["instructions"])
        for item in conv.items(protocol, request):
            if item.get("role") not in ("system", "developer"):
                break  # only the system messages at the start. Anything after is history
            if isinstance(item.get("content"), str):
                item["content"] = split(item["content"])
        return moved

    @staticmethod
    def _system_texts(protocol: Protocol, request: dict[str, Any]) -> list[str]:
        texts = []
        if protocol == "responses" and isinstance(request.get("instructions"), str):
            texts.append(request["instructions"])
        for item in conv.items(protocol, request):
            if item.get("role") not in ("system", "developer"):
                break
            if isinstance(item.get("content"), str):
                texts.append(item["content"])
        return texts


def prefix_key(tools: Any, system_texts: list[str]) -> str:
    """Same tools and system prompt, same key, so conversations that start alike share a cache."""
    digest = hashlib.sha256(json.dumps([tools or [], system_texts], sort_keys=True).encode()).hexdigest()
    return "pfx-" + digest[:16]
