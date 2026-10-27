"""Run the test plan. Every setup runs on every task, `trials` times.

Tasks go in order. For each task and trial the setups run in a random order that's the same every time,
so things that drift over time, like rate limits, don't keep favoring one setup.
"""

from __future__ import annotations

import datetime as dt
import random
from collections.abc import Callable

from harness.adapters.base import Adapter
from harness.budget import Budget, BudgetExceeded
from harness.eval.benchmark import Benchmark
from harness.eval.protocol import Protocol
from harness.eval.results import EvalResults, TrialRow
from harness.pricing import PriceTable
from harness.session import Session
from harness.trace import TraceStore


class EvalStopped(RuntimeError):
    """The run stopped early because the budget ran out. Results so far are saved. Resume with the same eval id."""


def run_eval(
    protocol: Protocol,
    benchmark: Benchmark,
    adapter: Adapter,
    store: TraceStore,
    *,
    eval_id: str | None = None,
    prices: PriceTable | None = None,
    on: dt.date | None = None,
    meta: dict | None = None,
    log: Callable[[str], None] = print,
) -> str:
    try:
        return _run_eval(protocol, benchmark, adapter, store, eval_id=eval_id, prices=prices, on=on, meta=meta, log=log)
    finally:
        benchmark.close()


def _run_eval(protocol, benchmark, adapter, store, *, eval_id, prices, on, meta, log) -> str:
    results = EvalResults(store)
    if eval_id is None:
        eval_id = results.start(protocol.hash, protocol.raw, {"benchmark": benchmark.name, **(meta or {})})
    else:
        stored = results.eval(eval_id)
        if stored["protocol_hash"] != protocol.hash:
            raise ValueError(f"protocol changed since eval {eval_id} started ({stored['protocol_hash']} -> {protocol.hash})")

    tasks = protocol.task_ids(benchmark.task_ids())
    done = results.done(eval_id)
    budget = Budget(protocol.budget_usd, spent_usd=results.spent(eval_id))
    prices = prices or PriceTable.load()
    rng = random.Random(protocol.analysis["seed"])
    total = len(tasks) * protocol.trials * len(protocol.arms)

    n = 0
    for task_id in tasks:
        for trial in range(protocol.trials):
            arms = protocol.arms
            rng.shuffle(arms)  # shuffle every time, even for finished trials, so a resumed run keeps the same order
            for arm in arms:
                n += 1
                if (arm, task_id, trial) in done:
                    continue
                run_id = store.start_run(f"{arm}/{task_id}/{trial}", {"eval_id": eval_id, "arm": arm, "task": task_id, "trial": trial})
                session = Session(
                    adapter,
                    task_id=f"{eval_id}:{arm}:{task_id}:{trial}",
                    policies=protocol.build_policies(arm),
                    store=store,
                    run_id=run_id,
                    prices=prices,
                    on=on,
                    budget=budget,
                    model=protocol.agent.get("model"),
                )
                try:
                    outcome = benchmark.run(task_id, trial, session)
                except BudgetExceeded as exc:
                    _record_error(results, store, eval_id, arm, task_id, trial, run_id, f"budget: {exc}")
                    raise EvalStopped(f"eval {eval_id} stopped at trial {n}/{total}: {exc}") from exc
                except Exception as exc:  # something broke. Save it, leave it out of the stats, retry it on resume
                    _record_error(results, store, eval_id, arm, task_id, trial, run_id, f"{type(exc).__name__}: {exc}")
                    log(f"[{n}/{total}] {arm} {task_id}#{trial}: ERROR {exc}")
                    continue
                budget.charge(outcome.sim_cost_usd)
                agent_cost = store.summary(run_id)["cost_usd"]
                results.record(
                    eval_id,
                    TrialRow(arm, task_id, trial, run_id, "ok", outcome.success, outcome.reward, agent_cost, outcome.sim_cost_usd, None),
                )
                log(
                    f"[{n}/{total}] {arm} {task_id}#{trial}: {'pass' if outcome.success else 'fail'} "
                    f"agent ${agent_cost:.4f}  (spent ${budget.spent_usd:.2f} of ${budget.limit_usd:.2f})"
                )
    return eval_id


def _record_error(results: EvalResults, store: TraceStore, eval_id, arm, task_id, trial, run_id, message) -> None:
    results.record(
        eval_id,
        TrialRow(arm, task_id, trial, run_id, "error", None, None, store.summary(run_id)["cost_usd"], 0.0, message),
    )
