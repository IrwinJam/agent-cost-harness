"""Masking: hide old tool outputs in batches so the prompt cache survives.

Swapping old tool outputs for a short placeholder about halved SWE-agent cost without hurting the solve
rate ("The Complexity Trap", NeurIPS 2025). Done every turn, though, it changes something early in the
history each time and the cache breaks for everything after it. Cached input is often 80 to 95%
cheaper than regular input, so that can end up costing more than it saves.

So this hides outputs in batches. It keeps the newest `keep_last` outputs as they are and only hides
more once `batch` more have gotten old. In between, the hidden part stays exactly the same and the
cache holds. When a batch happens the cache breaks once, and the profiler logs it with its cost.

The placeholders only depend on the hidden output, so running this again on the agent's full history
gives the exact same prompt.
"""

from __future__ import annotations

from typing import Any

from harness.policies import conv
from harness.policies.base import Policy, SessionContext
from harness.usage import Protocol


class BatchedMasking(Policy):
    name = "masking"

    def __init__(self, *, keep_last: int = 4, batch: int = 4, min_chars: int = 1000):
        if keep_last < 0 or batch < 1:
            raise ValueError("keep_last must be >= 0 and batch >= 1")
        self.keep_last = keep_last
        self.batch = batch
        self.min_chars = min_chars

    def apply(self, protocol: Protocol, request: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        items = conv.items(protocol, request)
        outputs = conv.tool_outputs(protocol, items)
        state = ctx.state_for(self.name)
        watermark = state.get("watermark", 0)

        eligible = max(0, len(outputs) - self.keep_last)
        if eligible - watermark >= self.batch:
            watermark = eligible
        # Don't go past what's in this request, in case the agent cut its own history.
        watermark = min(watermark, eligible)
        state["watermark"] = watermark

        # If output handles already stored the full output, point the placeholder at it.
        handle_state = ctx.state_for("output_handles")
        handles, store = handle_state.get("handles", {}), handle_state.get("store", {})
        names = conv.tool_names(protocol, items)
        for out in outputs[:watermark]:
            if len(out.text) < self.min_chars or out.text.startswith(PLACEHOLDER_PREFIX):
                continue
            handle = handles.get(out.call_id)
            full_length = len(store[handle]) if handle in store else len(out.text)
            conv.set_output(protocol, items, out.index, placeholder(out, names.get(out.call_id), handle, full_length))
        return request


PLACEHOLDER_PREFIX = "[masked tool output]"


def placeholder(out: conv.ToolOutput, tool_name: str | None, handle: str | None, full_length: int) -> str:
    text = f"{PLACEHOLDER_PREFIX} {tool_name or 'tool'} call {out.call_id} returned {full_length:,} characters, hidden to save context."
    if handle:
        text += f" Read it with fetch_output(handle={handle!r}) if you need it again."
    else:
        text += " Call the tool again if you need it."
    return text
