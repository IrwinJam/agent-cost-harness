"""Compare a setup against the baseline, task by task.

For each task, success and cost get averaged over its trials, then both setups are compared on the same
tasks. Confidence intervals come from resampling whole tasks, so the pairing holds.

Cost is the change in total cost, with a 95% confidence interval. Success is the difference in success
rate in points. It counts as not worse when the one-sided lower bound stays above minus the margin.
pass^k is the chance that every one of k trials of a task passes, averaged over tasks.

Trials that errored are left out. A task only counts if both setups have at least one clean trial on it,
and dropped tasks get reported.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import comb

from harness.eval.results import TrialRow


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    tasks: int
    trials: int
    success_rate: float
    mean_agent_cost: float  # per trial
    mean_sim_cost: float  # per trial
    cost_per_success: float | None
    pass_hat_k: dict[int, float]


@dataclass(frozen=True)
class Comparison:
    treatment: str
    baseline: str
    tasks: int
    dropped_tasks: list[str]
    cost_change: float  # e.g. -0.42 = 42% cheaper
    cost_ci: tuple[float, float]
    success_diff_pp: float
    success_lower_pp: float  # one-sided lower bound
    margin_pp: float
    alpha: float
    noninferior: bool
    cheaper: bool  # the whole cost interval is below zero
    worse_tasks: list[str]  # tasks where this setup did worse on average

    @property
    def claim_holds(self) -> bool:
        return self.noninferior and self.cheaper


def _by_task(rows: list[TrialRow], arm: str) -> dict[str, list[TrialRow]]:
    out: dict[str, list[TrialRow]] = {}
    for r in rows:
        if r.arm == arm and r.status == "ok":
            out.setdefault(r.task_id, []).append(r)
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pass_hat_k(per_task_successes: list[tuple[int, int]], k: int) -> float:
    """Average over tasks of C(c, k) / C(n, k). Tasks with fewer than k trials are skipped."""
    vals = [comb(c, k) / comb(n, k) for c, n in per_task_successes if n >= k]
    return _mean(vals)


def summarize_arm(rows: list[TrialRow], arm: str) -> ArmSummary:
    tasks = _by_task(rows, arm)
    trials = [r for rs in tasks.values() for r in rs]
    successes = sum(1 for r in trials if r.success)
    counts = [(sum(1 for r in rs if r.success), len(rs)) for rs in tasks.values()]
    max_k = min((n for _, n in counts), default=0)
    total_cost = sum(r.agent_cost_usd for r in trials)
    return ArmSummary(
        arm=arm,
        tasks=len(tasks),
        trials=len(trials),
        success_rate=successes / len(trials) if trials else 0.0,
        mean_agent_cost=_mean([r.agent_cost_usd for r in trials]),
        mean_sim_cost=_mean([r.sim_cost_usd for r in trials]),
        cost_per_success=total_cost / successes if successes else None,
        pass_hat_k={k: pass_hat_k(counts, k) for k in range(1, max_k + 1)},
    )


def compare(
    rows: list[TrialRow],
    treatment: str,
    baseline: str,
    *,
    margin_pp: float,
    alpha: float,
    reps: int,
    seed: int,
) -> Comparison:
    t, b = _by_task(rows, treatment), _by_task(rows, baseline)
    all_tasks = sorted(set(t) | set(b) | {r.task_id for r in rows if r.arm in (treatment, baseline)})
    tasks = [x for x in all_tasks if x in t and x in b]
    if not tasks:
        raise ValueError(f"no tasks with clean trials in both {treatment!r} and {baseline!r}")

    cost_t = [_mean([r.agent_cost_usd for r in t[x]]) for x in tasks]
    cost_b = [_mean([r.agent_cost_usd for r in b[x]]) for x in tasks]
    succ_t = [_mean([float(bool(r.success)) for r in t[x]]) for x in tasks]
    succ_b = [_mean([float(bool(r.success)) for r in b[x]]) for x in tasks]

    def stats(idx: list[int]) -> tuple[float, float]:
        cb = sum(cost_b[i] for i in idx)
        ratio = (sum(cost_t[i] for i in idx) / cb - 1.0) if cb else 0.0
        diff = 100.0 * _mean([succ_t[i] - succ_b[i] for i in idx])
        return ratio, diff

    n = len(tasks)
    point_cost, point_diff = stats(list(range(n)))
    rng = random.Random(seed)
    boot_cost, boot_diff = [], []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        c, d = stats(idx)
        boot_cost.append(c)
        boot_diff.append(d)
    boot_cost.sort()
    boot_diff.sort()
    lo_cost, hi_cost = _quantile(boot_cost, alpha / 2), _quantile(boot_cost, 1 - alpha / 2)
    lower_diff = _quantile(boot_diff, alpha)

    return Comparison(
        treatment=treatment,
        baseline=baseline,
        tasks=n,
        dropped_tasks=[x for x in all_tasks if x not in tasks],
        cost_change=point_cost,
        cost_ci=(lo_cost, hi_cost),
        success_diff_pp=point_diff,
        success_lower_pp=lower_diff,
        margin_pp=margin_pp,
        alpha=alpha,
        noninferior=lower_diff > -margin_pp,
        cheaper=hi_cost < 0,
        worse_tasks=[x for i, x in enumerate(tasks) if succ_t[i] < succ_b[i]],
    )


def _quantile(sorted_values: list[float], q: float) -> float:
    """Quantile of a sorted list, interpolated between points."""
    if not sorted_values:
        raise ValueError("empty")
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)
