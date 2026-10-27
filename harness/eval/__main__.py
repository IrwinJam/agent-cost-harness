"""Eval commands.

    python -m harness eval validate --protocol eval/protocol.yaml
    python -m harness eval estimate --protocol eval/protocol.yaml --tasks 40 --cost-per-trial 0.2
    python -m harness eval run --protocol eval/protocol.yaml --simulate [--db runs/sim.sqlite]
    python -m harness eval certify --db runs/sim.sqlite [--eval ID] [--protocol eval/protocol.yaml] [--out DIR]

From the command line `run` only does the simulated benchmark. To test a real agent, write your own
`Benchmark` and call `harness.eval.run_eval` from Python.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harness.eval.certificate import build_certificate, to_markdown
from harness.eval.protocol import Protocol
from harness.eval.results import EvalResults
from harness.eval.runner import EvalStopped, run_eval
from harness.trace import TraceStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check a protocol file and print its hash")
    p.add_argument("--protocol", default="eval/protocol.yaml")

    p = sub.add_parser("estimate", help="estimate spend before running")
    p.add_argument("--protocol", default="eval/protocol.yaml")
    p.add_argument("--tasks", type=int, required=True)
    p.add_argument("--cost-per-trial", type=float, required=True, help="agent + simulator USD per trial, from the pilot")

    p = sub.add_parser("run", help="run (or resume) an evaluation")
    p.add_argument("--protocol", default="eval/protocol.yaml")
    p.add_argument("--db", default=None, help="default: runs/sim.sqlite")
    p.add_argument("--resume", help="eval id to resume")
    p.add_argument("--simulate", action="store_true", help="simulated benchmark and API (no credit used)")
    p.add_argument("--sim-tasks", type=int, default=10)

    p = sub.add_parser("certify", help="write the savings certificate for an evaluation")
    p.add_argument("--db", required=True)
    p.add_argument("--eval", help="eval id (default: most recent)")
    p.add_argument("--protocol", help="protocol file to check against the stored hash")
    p.add_argument("--out", help="directory to write certificate .md and .json (default: print markdown)")

    args = parser.parse_args(argv)
    return {"validate": _validate, "estimate": _estimate, "run": _run, "certify": _certify}[args.command](args)


def _validate(args) -> int:
    protocol = Protocol.load(args.protocol)
    print(f"{protocol.name}: valid, hash {protocol.hash}; arms {protocol.arms}; primary {protocol.primary} vs {protocol.baseline}")
    return 0


def _estimate(args) -> int:
    protocol = Protocol.load(args.protocol)
    trials = args.tasks * protocol.trials * len(protocol.arms)
    cost = trials * args.cost_per_trial
    print(f"{args.tasks} tasks x {protocol.trials} trials x {len(protocol.arms)} arms = {trials} trials, ~${cost:.2f}")
    print(f"budget ${protocol.budget_usd:.2f}: " + ("fits" if cost <= protocol.budget_usd else f"OVER by ${cost - protocol.budget_usd:.2f}"))
    return 0 if cost <= protocol.budget_usd else 1


def _run(args) -> int:
    protocol = Protocol.load(args.protocol)
    if not args.simulate:
        print("only --simulate runs from the command line. For a real agent, write a Benchmark and call run_eval from Python", file=sys.stderr)
        return 2
    db = args.db or "runs/sim.sqlite"
    from harness.eval.simbench import SimulatedBenchmark
    from harness.sim import SimulatedAdapter

    benchmark = SimulatedBenchmark(tasks=args.sim_tasks)
    adapter = SimulatedAdapter(lambda protocol, request: {"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
    meta = {"simulated": True}
    with TraceStore(db) as store:
        try:
            eval_id = run_eval(protocol, benchmark, adapter, store, eval_id=args.resume, meta=meta)
        except EvalStopped as exc:
            print(exc, file=sys.stderr)
            return 2
    print(f"eval {eval_id} complete; certify with: python -m harness.eval certify --db {db} --eval {eval_id}")
    return 0


def _certify(args) -> int:
    if not Path(args.db).exists():
        print(f"no database at {args.db}", file=sys.stderr)
        return 1
    with TraceStore(args.db) as store:
        results = EvalResults(store)
        evals = results.evals()
        if not evals:
            print("no evaluations in this database", file=sys.stderr)
            return 1
        eval_id = args.eval or evals[-1]["id"]
        cert = build_certificate(results, eval_id, Protocol.load(args.protocol) if args.protocol else None)
    md = to_markdown(cert)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"certificate_{eval_id}.md").write_text(md, encoding="utf-8")
        (out / f"certificate_{eval_id}.json").write_text(json.dumps(cert, indent=2), encoding="utf-8")
        print(f"wrote {out / f'certificate_{eval_id}.md'} and .json")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
