"""Figure out what a run's cost came from.

Each piece of content counts once no matter how many calls resend it. A tool output is tracked by its
call id. Messages and the system prompt are tracked by their content. Reasoning and the visible answer
get their own lines. So a 10k token tool output carried through 12 turns shows up as one expensive item
with `sends == 12`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.profiler.load import CallView

OUTPUT_REASONING = ("output:reasoning", None)
OUTPUT_VISIBLE = ("output:visible", None)


@dataclass
class Source:
    origin: str
    key: str | None
    version: str | None = None  # content hash, only shown when one id has more than one version, like after masking
    sends: int = 0
    tokens: float = 0.0
    cached_tokens: float = 0.0
    cost: float = 0.0
    first_seq: int | None = None
    last_seq: int | None = None
    _seqs: set[int] = field(default_factory=set, repr=False)

    @property
    def label(self) -> str:
        base = f"{self.origin} {self.key}" if self.key else self.origin
        return f"{base} @{self.version}" if self.version else base

    @property
    def tokens_per_send(self) -> float:
        return self.tokens / self.sends if self.sends else 0.0

    @property
    def resend_cost(self) -> float:
        """What every send after the first one cost. Always zero for output."""
        if self.sends <= 1 or self.origin.startswith("output:"):
            return 0.0
        return self.cost * (self.sends - 1) / self.sends

    def _add(self, seq: int, tokens: float, cached: float, cost: float) -> None:
        if seq not in self._seqs:
            self._seqs.add(seq)
            self.sends += 1
        self.tokens += tokens
        self.cached_tokens += cached
        self.cost += cost
        self.first_seq = seq if self.first_seq is None else min(self.first_seq, seq)
        self.last_seq = seq if self.last_seq is None else max(self.last_seq, seq)


def source_key(origin: str, ref: str | None, digest: str) -> tuple[str, str | None]:
    if origin == "tool_schemas":
        return origin, digest[:8]
    if ref:
        return origin, ref
    return origin, digest[:8]


def attribute(calls: list[CallView]) -> list[Source]:
    """Sources sorted most expensive first. They add up to the run's total cost."""
    # Keyed by content too, so an edited or masked tool output counts separately from the
    # original even though they share a call id.
    sources: dict[tuple[str, str | None, str | None], Source] = {}

    def get(origin: str, key: str | None, digest: str | None = None) -> Source:
        k = (origin, key, digest)
        if k not in sources:
            sources[k] = Source(origin=origin, key=key)
        return sources[k]

    for call in calls:
        for share in call.shares:
            seg = share.segment
            origin, key = source_key(seg.origin, seg.ref, seg.hash)
            get(origin, key, seg.hash)._add(call.seq, share.tokens, share.cached_tokens, share.cost)
        if not call.shares and call.input_tokens:
            get("input:unsegmented", None)._add(call.seq, call.input_tokens, call.cached_input_tokens, call.input_cost)
        if call.reasoning_tokens:
            get(*OUTPUT_REASONING)._add(call.seq, call.reasoning_tokens, 0, call.reasoning_cost)
        visible = call.output_tokens - call.reasoning_tokens
        if visible:
            get(*OUTPUT_VISIBLE)._add(call.seq, visible, 0, call.output_cost - call.reasoning_cost)

    versions: dict[tuple[str, str | None], int] = {}
    for origin, key, _ in sources:
        versions[(origin, key)] = versions.get((origin, key), 0) + 1
    for (origin, key, digest), source in sources.items():
        if versions[(origin, key)] > 1 and digest:
            source.version = digest[:6]

    return sorted(sources.values(), key=lambda s: s.cost, reverse=True)


def cost_by_origin(sources: list[Source]) -> dict[str, float]:
    """Add up sources by type, like tool_output or system."""
    totals: dict[str, float] = {}
    for s in sources:
        totals[s.origin] = totals.get(s.origin, 0.0) + s.cost
    return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))
