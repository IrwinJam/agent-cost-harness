"""Output handles: shorten big tool outputs and let the model fetch the rest.

The harness stores the full output. The model sees a short version, a handle, and a `fetch_output` tool
to read any part of the original. The agent never sees `fetch_output` calls because the session runs
them and asks the model again.

JSON never comes out broken. Long lists keep their first items, long strings get clipped, and a
`_truncated` note says what got dropped. Other text gets cut on full lines, keeping the start and end.

`fetch_output` is added on every call from the start, even before anything gets shortened, so the tool
list never changes mid-run and breaks the cache. DTOC (2026) found 3 to 3.5x lower cost per solved task
on some models and mixed results on others. Versions with no way to get the output back did worse.
"""

from __future__ import annotations

import json
from typing import Any

from harness.policies import conv
from harness.policies.base import HarnessTool, Policy, SessionContext
from harness.usage import Protocol

FETCH_TOOL = "fetch_output"
FETCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "handle": {"type": "string", "description": "Handle given in a shortened tool output."},
        "start": {"type": "integer", "description": "First character to return (0-based).", "minimum": 0},
        "length": {"type": "integer", "description": "Number of characters to return.", "minimum": 1},
    },
    "required": ["handle", "start", "length"],
}
FETCH_DESCRIPTION = (
    "Read part of a tool output that was shortened to save context. Use only when the shortened "
    "version lacks something you need."
)


class OutputHandles(Policy):
    name = "output_handles"

    def __init__(self, *, max_chars: int = 4000, head_chars: int = 2000, tail_chars: int = 1000, max_fetch_chars: int = 8000):
        if head_chars + tail_chars >= max_chars:
            raise ValueError("head_chars + tail_chars must be smaller than max_chars")
        self.max_chars = max_chars
        self.head_chars = head_chars
        self.tail_chars = tail_chars
        self.max_fetch_chars = max_fetch_chars

    def setup(self, ctx: SessionContext) -> None:
        state = ctx.state_for(self.name)
        state.setdefault("handles", {})  # call_id -> handle
        state.setdefault("store", {})  # handle -> full text
        ctx.harness_tools[FETCH_TOOL] = HarnessTool(FETCH_TOOL, FETCH_DESCRIPTION, FETCH_PARAMETERS, lambda args: self._fetch(state, args))

    def apply(self, protocol: Protocol, request: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        state = ctx.state_for(self.name)
        items = conv.items(protocol, request)
        for out in conv.tool_outputs(protocol, items):
            if len(out.text) <= self.max_chars:
                continue
            handle = state["handles"].setdefault(out.call_id, f"out_{out.call_id}")
            state["store"].setdefault(handle, out.text)
            conv.set_output(protocol, items, out.index, self.shorten(out.text, handle))

        tools = list(request.get("tools") or [])
        if not any(conv.tool_schema_name(t) == FETCH_TOOL for t in tools):
            tools.append(conv.tool_schema(protocol, FETCH_TOOL, FETCH_DESCRIPTION, FETCH_PARAMETERS))
        request["tools"] = tools
        return request

    def shorten(self, text: str, handle: str) -> str:
        note = f"[shortened: {len(text):,} characters in full; read more with {FETCH_TOOL}(handle={handle!r}, start, length)]"
        structured = _shorten_json(text, self.max_chars - len(note) - 1)
        if structured is not None:
            return note + "\n" + structured
        return note + "\n" + _head_tail(text, self.head_chars, self.tail_chars)

    def _fetch(self, state: dict[str, Any], args: dict[str, Any]) -> str:
        text = state["store"].get(args.get("handle"))
        if text is None:
            return f"error: unknown handle {args.get('handle')!r}"
        start = max(0, int(args.get("start", 0)))
        length = max(1, min(int(args.get("length", self.max_fetch_chars)), self.max_fetch_chars))
        chunk = text[start : start + length]
        end = start + len(chunk)
        return f"[characters {start}-{end} of {len(text)}]\n{chunk}"


def _head_tail(text: str, head: int, tail: int) -> str:
    """Keep about `head` characters from the start and `tail` from the end, cut on full lines."""
    head_end = text.rfind("\n", 0, head)
    head_end = head if head_end <= 0 else head_end
    tail_start = text.find("\n", len(text) - tail)
    tail_start = len(text) - tail if tail_start < 0 else tail_start + 1
    if tail_start <= head_end:
        return text
    omitted = text[head_end:tail_start]
    return f"{text[:head_end]}\n... [{len(omitted):,} characters, {omitted.count(chr(10)):,} lines omitted] ...\n{text[tail_start:]}"


def _shorten_json(text: str, budget: int) -> str | None:
    """A shorter version of the JSON that fits in `budget` characters, or None if it isn't a JSON object or list."""
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    for items, chars in ((20, 400), (10, 200), (5, 100), (3, 60), (1, 40), (1, 0)):
        shortened = json.dumps(_clip(value, items, chars), ensure_ascii=False)
        if len(shortened) <= budget:
            return shortened
    return None


def _clip(value: Any, max_items: int, max_chars: int) -> Any:
    if isinstance(value, str):
        if max_chars and len(value) > max_chars:
            return value[:max_chars] + f"... [{len(value) - max_chars:,} chars truncated]"
        return value if max_chars else f"[string of {len(value):,} chars]"
    if isinstance(value, list):
        kept = [_clip(v, max_items, max_chars) for v in value[:max_items]]
        if len(value) > max_items:
            kept.append({"_truncated": f"{len(value) - max_items} more items"})
        return kept
    if isinstance(value, dict):
        keys = list(value)
        out = {k: _clip(value[k], max_items, max_chars) for k in keys[: max_items * 2]}
        if len(keys) > max_items * 2:
            out["_truncated"] = f"{len(keys) - max_items * 2} more keys"
        return out
    return value
