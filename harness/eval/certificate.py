"""The final report: the main comparison written as one claim you can check.

It's built only from saved results and the saved test plan. It says whether the claim held, and if not,
which part failed. It also lists every setup's numbers, whether the plan file changed after the run,
whether prices were verified, tasks that got dropped or got worse, and where the cost went.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from typing import Any

from harness.eval.protocol import Protocol
from harness.eval.results import EvalResults
from harness.eval.stats import ArmSummary, Comparison, compare, summarize_arm
from harness.profiler import build_profile
from harness.profiler.attribution import cost_by_origin


def build_certificate(results: EvalResults, eval_id: str, protocol_on_disk: Protocol | None = None) -> dict[str, Any]:
    ev = results.eval(eval_id)
    protocol = Protocol.from_dict(ev["protocol"])
    rows = results.trials(eval_id)
    a = protocol.analysis

    comparisons = {
        arm: compare(rows, arm, protocol.baseline, margin_pp=float(a["noninferiority_margin_pp"]), alpha=float(a["alpha"]), reps=int(a["bootstrap_reps"]), seed=int(a["seed"]))
        for arm in protocol.arms
        if arm != protocol.baseline and any(r.arm == arm and r.status == "ok" for r in rows)
    }
    if protocol.primary not in comparisons:
        raise ValueError(f"no results for the primary arm {protocol.primary!r}")
    primary = comparisons[protocol.primary]
    arms = {arm: summarize_arm(rows, arm) for arm in protocol.arms}
    expected = len({r.task_id for r in rows}) * protocol.trials

    run_ids = [r.run_id for r in rows if r.arm == protocol.primary and r.status == "ok" and r.run_id]
    origins: dict[str, float] = {}
    cache_breaks: dict[str, int] = {}
    prices_verified = True
    for run_id in run_ids:
        profile = build_profile(results.store, run_id)
        prices_verified &= profile.prices_verified
        for origin, cost in cost_by_origin(profile.sources).items():
            origins[origin] = origins.get(origin, 0.0) + cost
        for event in profile.cache_events:
            cache_breaks[event.cause] = cache_breaks.get(event.cause, 0) + 1

    simulated = bool(ev["meta"].get("simulated"))
    return {
        "claim": claim_sentence(protocol, primary, simulated),
        "claim_holds": primary.claim_holds,
        "simulated": simulated,
        "eval_id": eval_id,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "name": protocol.name,
            "hash": ev["protocol_hash"],
            "unchanged_on_disk": None if protocol_on_disk is None else protocol_on_disk.hash == ev["protocol_hash"],
            "benchmark": protocol.raw["benchmark"],
            "agent": protocol.agent,
            "user_simulator": protocol.raw.get("user_simulator"),
            "trials": protocol.trials,
            "analysis": a,
        },
        "prices_verified": prices_verified,
        "completeness": {arm: f"{s.trials}/{expected} trials" for arm, s in arms.items()},
        "errors": sum(1 for r in rows if r.status == "error"),
        "arms": {arm: asdict(s) for arm, s in arms.items()},
        "primary": asdict(primary),
        "secondary": {arm: asdict(c) for arm, c in comparisons.items() if arm != protocol.primary},
        "primary_cost_by_origin_usd": dict(sorted(origins.items(), key=lambda kv: kv[1], reverse=True)),
        "primary_cache_breaks": cache_breaks,
    }


def claim_sentence(protocol: Protocol, c: Comparison, simulated: bool) -> str:
    prefix = "SIMULATED, NOT EVIDENCE. " if simulated else ""
    bench = protocol.raw["benchmark"]
    where = f"{bench['name']} {bench.get('domain', '')}".strip()
    cost = f"agent cost per task {c.cost_change:+.0%} (95% CI {c.cost_ci[0]:+.0%} to {c.cost_ci[1]:+.0%})"
    success = (
        f"success rate difference {c.success_diff_pp:+.1f} pp, one-sided {1 - c.alpha:.0%} lower bound "
        f"{c.success_lower_pp:+.1f} pp against a pre-registered margin of -{c.margin_pp:g} pp"
    )
    head = f"{prefix}On {where} ({c.tasks} tasks x {protocol.trials} trials), {c.treatment} vs {c.baseline}: {cost}; {success}."
    if c.claim_holds:
        return head + " Savings are significant and success is non-inferior."
    failed = []
    if not c.cheaper:
        failed.append("the cost CI includes zero or an increase")
    if not c.noninferior:
        failed.append("non-inferiority on success was NOT shown")
    return head + " Claim does not hold: " + "; ".join(failed) + "."


def to_markdown(cert: dict[str, Any]) -> str:
    p = cert["protocol"]
    lines = [f"# Savings certificate: {p['name']}", ""]
    if cert["simulated"]:
        lines += ["> **Simulated run. These numbers test the harness and are not evidence of real savings.**", ""]
    lines += [f"**{cert['claim']}**", ""]
    lines += [
        f"- Eval `{cert['eval_id']}`, protocol hash `{p['hash']}`"
        + ("" if p["unchanged_on_disk"] is None else (" (matches file on disk)" if p["unchanged_on_disk"] else " (**file on disk has changed since the run**)")),
        f"- Agent: `{p['agent'].get('model')}` via {p['agent'].get('protocol')}; user simulator: {json.dumps(p['user_simulator'])}",
        f"- Prices verified against provider pages: {'yes' if cert['prices_verified'] else '**no**'}",
        f"- Trials with infrastructure errors (excluded): {cert['errors']}",
        "",
        "## Arms",
        "",
        "| Arm | Trials | Success | Agent $/trial | Sim $/trial | $/success | pass^k |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for arm, s in cert["arms"].items():
        passk = ", ".join(f"k={k}: {v:.0%}" for k, v in s["pass_hat_k"].items())
        per_success = f"${s['cost_per_success']:.4f}" if s["cost_per_success"] is not None else "-"
        lines.append(
            f"| {arm} | {cert['completeness'][arm]} | {s['success_rate']:.0%} | ${s['mean_agent_cost']:.4f} | "
            f"${s['mean_sim_cost']:.4f} | {per_success} | {passk} |"
        )
    pr = cert["primary"]
    lines += ["", "## Primary comparison", ""]
    lines += [
        f"- Cost change: {pr['cost_change']:+.1%} (95% CI {pr['cost_ci'][0]:+.1%} to {pr['cost_ci'][1]:+.1%})",
        f"- Success difference: {pr['success_diff_pp']:+.1f} pp (one-sided lower bound {pr['success_lower_pp']:+.1f} pp, margin -{pr['margin_pp']:g} pp)",
        f"- Tasks where {pr['treatment']} did worse than {pr['baseline']}: {', '.join(pr['worse_tasks']) or 'none'}",
        f"- Tasks dropped (no clean trial in one arm): {', '.join(pr['dropped_tasks']) or 'none'}",
    ]
    if cert["secondary"]:
        lines += ["", "## Exploratory comparisons (not pre-registered as the claim)", ""]
        for arm, c in cert["secondary"].items():
            lines.append(f"- {arm}: cost {c['cost_change']:+.1%} (95% CI {c['cost_ci'][0]:+.1%} to {c['cost_ci'][1]:+.1%}), success {c['success_diff_pp']:+.1f} pp (lower bound {c['success_lower_pp']:+.1f})")
    lines += ["", "## Where the primary arm's cost went", ""]
    total = sum(cert["primary_cost_by_origin_usd"].values()) or 1.0
    for origin, cost in list(cert["primary_cost_by_origin_usd"].items())[:8]:
        lines.append(f"- {origin}: ${cost:.4f} ({cost / total:.0%})")
    if cert["primary_cache_breaks"]:
        lines.append("- Cache breaks: " + ", ".join(f"{k} x{v}" for k, v in cert["primary_cache_breaks"].items()))
    return "\n".join(lines) + "\n"
