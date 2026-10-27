"""Compare policy setups on a fake tool-heavy agent. No API calls, no credit used.

    python examples/simulate.py [--turns 20] [--output-chars 12000] [--clock]

This checks that the pieces work together. The cost numbers come from a model of prefix caching and say
nothing about whether tasks succeed, so they aren't real savings.
"""

from __future__ import annotations

import argparse

from harness.policies import BatchedMasking, CacheStableLayout, OutputHandles
from harness.pricing import PriceTable
from harness.profiler import build_profile
from harness.session import Session
from harness.sim import SimulatedAdapter, scripted_agent_turns
from harness.trace import TraceStore

SYSTEM = "You are a debugging agent.\nCurrent time: 2026-10-26T14:03:11Z\n" + "Follow the runbook carefully. " * 60
TOOLS = [{"type": "function", "name": "read_log", "parameters": {"type": "object", "properties": {"part": {"type": "integer"}}}}]

ARMS = {
    "baseline": lambda: [],
    "cache": lambda: [CacheStableLayout()],
    "cache+handles": lambda: [CacheStableLayout(), OutputHandles()],
    "cache+handles+mask": lambda: [CacheStableLayout(), OutputHandles(), BatchedMasking()],
    "naive-masking": lambda: [CacheStableLayout(), BatchedMasking(keep_last=4, batch=1)],
}


def reply(protocol, request):
    return {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "next"}]}]}


def run_arm(store: TraceStore, name: str, turns: int, output_chars: int, clock: bool) -> str:
    run = store.start_run(name, {"simulated": True})
    session = Session(SimulatedAdapter(reply), task_id=f"sim-{name}", policies=ARMS[name](), store=store, run_id=run, prices=PriceTable.load())
    history = scripted_agent_turns(turns, output_chars)
    # Send the conversation one tool turn at a time, like a real agent would.
    for turn, end in enumerate(range(3, len(history) + 1, 2)):
        system = SYSTEM.replace("14:03:11", f"14:{3 + turn:02d}:11") if clock else SYSTEM
        session.call("responses", {"model": "example-model", "instructions": system, "tools": TOOLS, "input": history[:end]})
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--output-chars", type=int, default=12_000)
    parser.add_argument("--clock", action="store_true", help="put a ticking timestamp in the system prompt")
    args = parser.parse_args()

    print(f"SIMULATION ONLY: {args.turns} tool turns, ~{args.output_chars:,}-char outputs, modelled prefix cache\n")
    print(f"{'arm':<20}{'cost $':>10}{'input tok':>12}{'cache hit':>11}{'breaks':>8}  top cache-break causes")
    with TraceStore(":memory:") as store:
        baseline_cost = None
        for name in ARMS:
            profile = build_profile(store, run_arm(store, name, args.turns, args.output_chars, args.clock))
            baseline_cost = baseline_cost or profile.total_cost
            causes = {}
            for e in profile.cache_events:
                causes[e.cause] = causes.get(e.cause, 0) + 1
            print(
                f"{name:<20}{profile.total_cost:>10.4f}{profile.input_tokens:>12,}{profile.cache_hit_rate:>10.0%}"
                f"{len(profile.cache_events):>8}  {', '.join(f'{k} x{v}' for k, v in causes.items()) or '-'}"
                f"   ({profile.total_cost / baseline_cost - 1:+.0%} vs baseline)"
            )


if __name__ == "__main__":
    main()
