"""Put the profile together and print it as Markdown or JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from harness.profiler.attribution import Source, attribute, cost_by_origin
from harness.profiler.cache_diag import CacheEvent, diagnose_cache
from harness.profiler.load import CallView, load_run
from harness.profiler.waste import Finding, find_waste
from harness.trace import TraceStore

ESTIMATE_NOTE = (
    "Per-segment token counts are estimated from character counts and scaled to the provider-reported "
    "totals; call totals and costs are exact (given the price table)."
)


@dataclass
class Profile:
    run_id: str
    label: str | None
    calls: list[CallView]
    sources: list[Source]
    cache_events: list[CacheEvent]
    findings: list[Finding]
    prices_verified: bool

    @property
    def total_cost(self) -> float:
        return sum(c.total_cost for c in self.calls)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def cached_input_tokens(self) -> int:
        return sum(c.cached_input_tokens for c in self.calls)

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_input_tokens / self.input_tokens if self.input_tokens else 0.0

    def to_dict(self, top: int = 20) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "calls": len(self.calls),
            "total_cost_usd": self.total_cost,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "reasoning_tokens": sum(c.reasoning_tokens for c in self.calls),
            "cache_hit_rate": self.cache_hit_rate,
            "prices_verified": self.prices_verified,
            "cost_by_origin": cost_by_origin(self.sources),
            "top_sources": [
                {
                    "label": s.label,
                    "origin": s.origin,
                    "sends": s.sends,
                    "tokens": round(s.tokens),
                    "cached_tokens": round(s.cached_tokens),
                    "cost_usd": s.cost,
                    "resend_cost_usd": s.resend_cost,
                    "first_seq": s.first_seq,
                    "last_seq": s.last_seq,
                }
                for s in self.sources[:top]
            ],
            "cache_events": [
                {
                    "seq": e.seq,
                    "cause": e.cause,
                    "at_segment": e.at_segment,
                    "lost_tokens": round(e.lost_tokens),
                    "lost_cost_usd": e.lost_cost,
                    "explanation": e.explanation,
                }
                for e in self.cache_events
            ],
            "findings": [
                {"kind": f.kind, "detail": f.detail, "cost_usd": f.cost, "seqs": list(f.seqs)} for f in self.findings
            ],
            "note": ESTIMATE_NOTE,
        }

    def to_json(self, top: int = 20) -> str:
        return json.dumps(self.to_dict(top), indent=2)

    def to_markdown(self, top: int = 10) -> str:
        total = self.total_cost or 1.0
        lines = [
            f"# Cost profile: {self.label or self.run_id}",
            "",
            f"- Calls: {len(self.calls)}",
            f"- Total cost: ${self.total_cost:.4f}" + ("" if self.prices_verified else " (prices not yet verified)"),
            f"- Input tokens: {self.input_tokens:,} ({self.cache_hit_rate:.0%} from cache)",
            f"- Output tokens: {sum(c.output_tokens for c in self.calls):,} "
            f"(reasoning {sum(c.reasoning_tokens for c in self.calls):,})",
            "",
            "## Where the money went",
            "",
            "| Source | Sends | ~Tokens | Cost | Share | Re-send cost |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for s in self.sources[:top]:
            lines.append(
                f"| {s.label} | {s.sends} | {s.tokens:,.0f} | ${s.cost:.4f} | {s.cost / total:.0%} | ${s.resend_cost:.4f} |"
            )
        lines += ["", "## Cache breaks", ""]
        if self.cache_events:
            lines += ["| Call | Cause | At | ~Tokens lost | Cost |", "|---:|---|---|---:|---:|"]
            for e in self.cache_events:
                lines.append(
                    f"| {e.seq} | {e.explanation} | {e.at_segment or '-'} | {e.lost_tokens:,.0f} | ${e.lost_cost:.4f} |"
                )
        else:
            lines.append("No cache breaks above the tolerance.")
        lines += ["", "## Possible waste", ""]
        if self.findings:
            lines += [f"- **{f.kind}** (${f.cost:.4f}): {f.detail}" for f in self.findings]
        else:
            lines.append("Nothing flagged.")
        lines += ["", f"_{ESTIMATE_NOTE}_", ""]
        return "\n".join(lines)


def build_profile(store: TraceStore, run_id: str) -> Profile:
    run = store.run(run_id)
    if run is None:
        raise KeyError(f"no run {run_id}")
    calls = load_run(store, run_id)
    sources = attribute(calls)
    return Profile(
        run_id=run_id,
        label=run["label"],
        calls=calls,
        sources=sources,
        cache_events=diagnose_cache(calls),
        findings=find_waste(calls, sources),
        prices_verified=all(c.price.verified for c in calls),
    )
