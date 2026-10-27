"""Split a request into pieces and hash each one.

A piece is one thing in the prompt: the tool list, the system prompt, one message, one tool call or
one tool output. The hashes show which pieces get resent from call to call, and where two calls in a
row first stop matching, which is usually where the cache broke.

I assume the prompt is built as tools first, then the system prompt, then the conversation. That's
how OpenAI and Anthropic style APIs are usually described. Your provider may not document it, so the
cache break checks built on this still need to be confirmed with real cached token numbers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from harness.usage import Protocol


@dataclass(frozen=True)
class Segment:
    index: int
    origin: str  # tool_schemas | system | message:<role> | tool_call | tool_output | reasoning
    ref: str | None  # tool call id or tool name when known
    hash: str
    chars: int


def _digest(value: Any) -> tuple[str, int]:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], len(text)


class _Builder:
    def __init__(self) -> None:
        self.segments: list[Segment] = []

    def add(self, origin: str, value: Any, ref: str | None = None) -> None:
        if value is None or value == "" or value == []:
            return
        digest, chars = _digest(value)
        self.segments.append(Segment(len(self.segments), origin, ref, digest, chars))


def segment(protocol: Protocol, request: dict[str, Any]) -> list[Segment]:
    builder = _Builder()
    builder.add("tool_schemas", request.get("tools"))
    if protocol == "responses":
        _responses(builder, request)
    elif protocol == "chat_completions":
        _chat(builder, request)
    elif protocol == "messages":
        _messages(builder, request)
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    return builder.segments


def _responses(b: _Builder, request: dict[str, Any]) -> None:
    b.add("system", request.get("instructions"))
    items = request.get("input")
    if isinstance(items, str):
        b.add("message:user", items)
        return
    for item in items or []:
        kind = item.get("type", "message")
        if kind == "message":
            b.add(f"message:{item.get('role', 'user')}", item.get("content"))
        elif kind == "function_call":
            b.add("tool_call", {"name": item.get("name"), "arguments": item.get("arguments")}, item.get("call_id"))
        elif kind == "function_call_output":
            b.add("tool_output", item.get("output"), item.get("call_id"))
        elif kind == "reasoning":
            b.add("reasoning", item.get("encrypted_content") or item.get("summary"))
        else:
            b.add(f"other:{kind}", item)


def _chat(b: _Builder, request: dict[str, Any]) -> None:
    for message in request.get("messages") or []:
        role = message.get("role", "user")
        if role in ("system", "developer"):
            b.add("system", message.get("content"))
        elif role == "tool":
            b.add("tool_output", message.get("content"), message.get("tool_call_id"))
        else:
            b.add(f"message:{role}", message.get("content"))
            for call in message.get("tool_calls") or []:
                fn = call.get("function", {})
                b.add("tool_call", {"name": fn.get("name"), "arguments": fn.get("arguments")}, call.get("id"))


def _messages(b: _Builder, request: dict[str, Any]) -> None:
    b.add("system", request.get("system"))
    for message in request.get("messages") or []:
        role = message.get("role", "user")
        content = message.get("content")
        if isinstance(content, str):
            b.add(f"message:{role}", content)
            continue
        # Plain blocks in a row become one message piece. Order stays the same because that's
        # the order of the prompt.
        pending: list[Any] = []
        for block in content or []:
            kind = block.get("type")
            if kind not in ("tool_use", "tool_result", "thinking", "redacted_thinking"):
                pending.append(block)
                continue
            b.add(f"message:{role}", pending)
            pending = []
            if kind == "tool_use":
                b.add("tool_call", {"name": block.get("name"), "arguments": block.get("input")}, block.get("id"))
            elif kind == "tool_result":
                b.add("tool_output", block.get("content"), block.get("tool_use_id"))
            else:
                b.add("reasoning", block.get("thinking") or block.get("data"))
        b.add(f"message:{role}", pending)


def first_divergence(previous: list[Segment], current: list[Segment]) -> int | None:
    """Index of the first piece in `current` that doesn't match `previous`.

    Returns None when `current` only adds to the end of `previous`, which is the best case for caching.
    Hashes only cover content, so the tool call id gets compared too, since it's part of the real prompt.
    """
    for i, (a, b) in enumerate(zip(previous, current)):
        if (a.hash, a.origin, a.ref) != (b.hash, b.origin, b.ref):
            return i
    if len(current) < len(previous):
        return len(current)
    return None
