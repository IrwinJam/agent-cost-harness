"""Find out why the cache missed between calls in a row.

Each call gets compared to the one before it. If it changed something that was already sent, like the
tools, the system prompt, an old tool output or dropped history, nothing after that point can come
from the cache. The first changed piece is the cause.

Sometimes nothing changed and the provider still cached less than it could, for example because the
cache expired. That gets logged as a provider miss.

Token amounts are estimates. There's some slack for cache block sizes, since providers don't always
document those.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.profiler.attribution import source_key
from harness.profiler.load import CallView
from harness.segments import first_divergence

CAUSES = {
    "model_switch": "model changed; the cache is per model",
    "tool_schemas_changed": "tool definitions changed; everything after them is re-billed",
    "system_prompt_changed": "system prompt changed (often an injected timestamp or ID)",
    "history_rewritten": "earlier conversation content was edited or masked",
    "history_truncated": "earlier conversation content was dropped",
    "provider_miss": "prefix unchanged but not served from cache (expiry, minimum size, or routing)",
}


@dataclass(frozen=True)
class CacheEvent:
    seq: int
    cause: str
    at_segment: str | None  # the first piece that changed, if anything changed
    lost_tokens: float
    lost_cost: float

    @property
    def explanation(self) -> str:
        return CAUSES[self.cause]


def diagnose_cache(calls: list[CallView], *, min_tokens: float = 1024, min_fraction: float = 0.05) -> list[CacheEvent]:
    events: list[CacheEvent] = []
    for prev, cur in zip(calls, calls[1:]):
        if not cur.shares or not prev.shares:
            continue
        rate = (cur.price.input - cur.price.cached_input) / 1_000_000
        tolerance = lambda base: max(min_tokens, min_fraction * base)  # noqa: E731

        if cur.model != prev.model:
            lost = min(_tokens(cur, 0, len(prev.shares)), cur.input_tokens) - cur.cached_input_tokens
            if lost > tolerance(cur.input_tokens):
                events.append(CacheEvent(cur.seq, "model_switch", None, lost, lost * rate))
            continue

        div = first_divergence(prev.segments, cur.segments)
        reusable_end = len(prev.shares) if div is None else div
        reusable = _tokens(cur, 0, reusable_end)

        # If this request is just a shorter version of the last one, nothing is lost.
        if div is not None and div < len(cur.shares):
            broken = _tokens(cur, div, min(len(prev.shares), len(cur.shares)))
            seg = cur.shares[div].segment
            label = " ".join(x for x in source_key(seg.origin, seg.ref, seg.hash) if x)
            later_prev = {(s.hash, s.ref) for s in prev.segments[div + 1 :]}
            if seg.origin == "tool_schemas":
                cause = "tool_schemas_changed"
            elif seg.origin == "system":
                cause = "system_prompt_changed"
            elif (seg.hash, seg.ref) in later_prev:
                cause = "history_truncated"  # older items got dropped and later ones moved up
            else:
                cause = "history_rewritten"
            if broken > tolerance(broken + reusable):
                events.append(CacheEvent(cur.seq, cause, label, broken, broken * rate))

        missed = reusable - cur.cached_input_tokens
        if missed > tolerance(reusable):
            events.append(CacheEvent(cur.seq, "provider_miss", None, missed, missed * rate))
    return events


def _tokens(call: CallView, start: int, end: int) -> float:
    return sum(s.tokens for s in call.shares[start:end])
