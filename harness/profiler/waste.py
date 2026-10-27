"""Things that look wasteful. Each one is a lead to check, not proof."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from harness.profiler.attribution import Source
from harness.profiler.load import CallView


@dataclass(frozen=True)
class Finding:
    kind: str
    detail: str
    cost: float  # dollars this pattern cost, which is the most fixing it could save
    seqs: tuple[int, ...]


def find_waste(
    calls: list[CallView],
    sources: list[Source],
    *,
    large_output_tokens: float = 2000,
    min_resends: int = 3,
    heavy_reasoning_tokens: int = 1000,
    small_visible_tokens: int = 50,
) -> list[Finding]:
    findings = _carried_tool_outputs(sources, large_output_tokens, min_resends)
    findings += _repeated_tool_calls(calls)
    findings += _heavy_reasoning_small_steps(calls, heavy_reasoning_tokens, small_visible_tokens)
    return sorted(findings, key=lambda f: f.cost, reverse=True)


def _carried_tool_outputs(sources: list[Source], large: float, min_resends: int) -> list[Finding]:
    out = []
    for s in sources:
        if s.origin != "tool_output" or s.tokens_per_send < large or s.sends - 1 < min_resends:
            continue
        out.append(
            Finding(
                kind="large_tool_output_carried",
                detail=(
                    f"{s.label}: ~{s.tokens_per_send:,.0f} tokens re-sent in {s.sends - 1} later calls "
                    f"(calls {s.first_seq}-{s.last_seq}); candidate for masking or an output handle"
                ),
                cost=s.resend_cost,
                seqs=tuple(range(s.first_seq or 0, (s.last_seq or 0) + 1)),
            )
        )
    return out


def _repeated_tool_calls(calls: list[CallView]) -> list[Finding]:
    # Same tool and arguments under different call ids. Without knowing the state we can't tell a
    # wasted repeat from a real recheck after something changed, so it says "possible".
    first_seen: dict[str, int] = {}
    ids_by_hash: dict[str, set[str]] = defaultdict(set)
    for call in calls:
        for share in call.shares:
            seg = share.segment
            if seg.origin == "tool_call" and seg.ref:
                ids_by_hash[seg.hash].add(seg.ref)
                first_seen.setdefault(seg.hash, call.seq)
    out = []
    for digest, ids in ids_by_hash.items():
        if len(ids) > 1:
            out.append(
                Finding(
                    kind="possible_duplicate_tool_call",
                    detail=f"identical tool call issued {len(ids)} times (ids {', '.join(sorted(ids))})",
                    cost=0.0,  # the cost is in the extra round trips, see the call list
                    seqs=(first_seen[digest],),
                )
            )
    return out


def _heavy_reasoning_small_steps(calls: list[CallView], heavy: int, small: int) -> list[Finding]:
    hits = [c for c in calls if c.reasoning_tokens >= heavy and c.output_tokens - c.reasoning_tokens <= small]
    if not hits:
        return []
    return [
        Finding(
            kind="heavy_reasoning_small_step",
            detail=(
                f"{len(hits)} call(s) spent >= {heavy:,} reasoning tokens to produce <= {small} visible tokens; "
                "candidates for a lower reasoning effort"
            ),
            cost=sum(c.reasoning_cost for c in hits),
            seqs=tuple(c.seq for c in hits),
        )
    ]
