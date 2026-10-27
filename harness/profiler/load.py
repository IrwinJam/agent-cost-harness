"""Load a traced run with token estimates for each piece of the prompt.

Providers only give token counts for the whole request, not for each piece. So each piece gets a share
based on its character count, scaled so the shares add up to the real total. The cached tokens go to
the pieces at the start, since caching works on the start of the prompt. These are estimates and the
reports say so.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from harness.pricing import Price
from harness.segments import Segment
from harness.trace import TraceStore


@dataclass(frozen=True)
class SegmentShare:
    segment: Segment
    tokens: float  # estimated input tokens
    cached_tokens: float  # how much of tokens came from the cache
    cost: float  # dollars, scaled so a call's pieces add up to its real input cost


@dataclass(frozen=True)
class CallView:
    call_id: int
    seq: int
    provider: str
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    input_cost: float
    output_cost: float
    price: Price
    shares: tuple[SegmentShare, ...]

    @property
    def segments(self) -> list[Segment]:
        return [s.segment for s in self.shares]

    @property
    def reasoning_cost(self) -> float:
        return self.output_cost * self.reasoning_tokens / self.output_tokens if self.output_tokens else 0.0

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


def load_run(store: TraceStore, run_id: str) -> list[CallView]:
    """Prices come from the trace itself, so a profile always matches what the run was actually billed at."""
    views = []
    for row in store.calls(run_id):
        price = Price(
            provider=row["provider"],
            model=row["model"],
            effective_from=dt.date.fromisoformat(row["price_effective_from"]),
            input=row["price_input"],
            cached_input=row["price_cached_input"],
            output=row["price_output"],
            verified=bool(row["price_verified"]),
        )
        input_cost = row["cost_uncached_input"] + row["cost_cached_input"]
        shares = _shares(store.segments(row["id"]), row["input_tokens"], row["cached_input_tokens"], price, input_cost)
        views.append(
            CallView(
                call_id=row["id"],
                seq=row["seq"],
                provider=row["provider"],
                model=row["model"],
                input_tokens=row["input_tokens"],
                cached_input_tokens=row["cached_input_tokens"],
                output_tokens=row["output_tokens"],
                reasoning_tokens=row["reasoning_tokens"],
                input_cost=input_cost,
                output_cost=row["cost_output"],
                price=price,
                shares=tuple(shares),
            )
        )
    return views


def _shares(
    segments: list[Segment], input_tokens: int, cached_tokens: int, price: Price, input_cost: float
) -> list[SegmentShare]:
    total_chars = sum(s.chars for s in segments)
    if not segments or total_chars == 0:
        return []
    raw = []
    remaining_cache = float(cached_tokens)
    for seg in segments:
        tokens = input_tokens * seg.chars / total_chars
        cached = min(tokens, remaining_cache)
        remaining_cache -= cached
        raw_cost = (cached * price.cached_input + (tokens - cached) * price.input) / 1_000_000
        raw.append((seg, tokens, cached, raw_cost))
    raw_total = sum(r[3] for r in raw)
    scale = input_cost / raw_total if raw_total else 0.0
    return [SegmentShare(seg, tokens, cached, cost * scale) for seg, tokens, cached, cost in raw]
