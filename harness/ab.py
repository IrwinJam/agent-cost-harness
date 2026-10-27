"""Live A/B test on your own traffic.

    python -m harness serve --base-url https://api.example.com/v1 --policies cache,reason --ab
    python -m harness ab --db runs/serve.sqlite

With `--ab`, `serve` puts each conversation in one of two groups by its key. `control` goes straight to
your provider with no policies. `policies` gets the ones you picked. The same key always lands in the
same group, so a conversation never switches halfway.

`python -m harness ab` then compares cost per conversation between the groups, with a 95% confidence
interval from resampling conversations in each group.

Two things to keep in mind. This only measures cost. It can't tell if the agent did its job, so check
that some other way (the `eval` command does it on a benchmark). And conversations differ a lot in
length, so you need a good number of them, at least 30 per group, before the interval means much.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from harness.trace import TraceStore

CONTROL, TREATMENT = "control", "policies"
MIN_PER_GROUP = 30


def assign(key: str, share: float = 0.5) -> str:
    """Pick a group from the conversation key. Same key, same group, every time."""
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return TREATMENT if bucket < share else CONTROL


@dataclass
class Group:
    name: str
    costs: list[float]
    calls: int
    input_tokens: int
    cached_input_tokens: int

    @property
    def n(self) -> int:
        return len(self.costs)

    @property
    def mean(self) -> float:
        return sum(self.costs) / len(self.costs) if self.costs else 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_input_tokens / self.input_tokens if self.input_tokens else 0.0


@dataclass
class Comparison:
    control: Group
    treatment: Group
    change: float | None  # treatment vs control cost per conversation, as a fraction. -0.2 means 20% cheaper
    ci: tuple[float, float] | None


def load(store: TraceStore) -> dict[str, Group]:
    groups = {name: Group(name, [], 0, 0, 0) for name in (CONTROL, TREATMENT)}
    for run in store.runs():
        name = json.loads(run["meta"] or "{}").get("ab")
        if name not in groups:
            continue
        s = store.summary(run["id"])
        if not s["calls"]:
            continue
        g = groups[name]
        g.costs.append(s["cost_usd"])
        g.calls += s["calls"]
        g.input_tokens += s["input_tokens"]
        g.cached_input_tokens += s["cached_input_tokens"]
    return groups


def compare(control: Group, treatment: Group, *, reps: int = 10_000, seed: int = 0) -> Comparison:
    if not control.n or not treatment.n or not control.mean:
        return Comparison(control, treatment, None, None)
    change = treatment.mean / control.mean - 1
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        c = sum(rng.choices(control.costs, k=control.n)) / control.n
        t = sum(rng.choices(treatment.costs, k=treatment.n)) / treatment.n
        if c:
            samples.append(t / c - 1)
    samples.sort()
    ci = (samples[int(0.025 * len(samples))], samples[min(len(samples) - 1, int(0.975 * len(samples)))])
    return Comparison(control, treatment, change, ci)


def report(cmp: Comparison) -> str:
    lines = [f"{'group':<10} {'convos':>7} {'calls':>7} {'$/convo':>10} {'cache hit':>10}"]
    for g in (cmp.control, cmp.treatment):
        lines.append(f"{g.name:<10} {g.n:>7} {g.calls:>7} {g.mean:>10.4f} {g.cache_hit_rate:>9.1%}")
    lines.append("")
    if cmp.change is None:
        lines.append("not enough data yet: each group needs at least one conversation with a call")
        return "\n".join(lines)
    lo, hi = cmp.ci
    lines.append(f"cost per conversation: {cmp.change:+.1%} (95% CI {lo:+.1%} to {hi:+.1%})")
    if min(cmp.control.n, cmp.treatment.n) < MIN_PER_GROUP:
        lines.append(f"only {min(cmp.control.n, cmp.treatment.n)} conversations in the smaller group. Wait for {MIN_PER_GROUP}+ before trusting this")
    elif hi < 0:
        lines.append("the policies are cheaper, and the interval doesn't cross zero")
    elif lo > 0:
        lines.append("the policies cost more, and the interval doesn't cross zero")
    else:
        lines.append("no clear difference yet")
    lines.append("this measures cost only. Check separately that the agent still does its job")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness ab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="runs/serve.sqlite", help="trace database that `serve --ab` wrote to")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not Path(args.db).exists():
        print(f"no trace database at {args.db}", file=sys.stderr)
        return 1
    with TraceStore(args.db) as store:
        groups = load(store)
    print(report(compare(groups[CONTROL], groups[TREATMENT], seed=args.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
