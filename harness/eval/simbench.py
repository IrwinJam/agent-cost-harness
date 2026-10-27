"""A fake benchmark for dry runs of the whole eval with no API calls.

Each task replays a fake tool-heavy conversation through the agent session, so the policies, traces,
budget and profiler all really run. Whether a task passes is random but fixed per task and doesn't
depend on the setup, so the real effect on success is zero on purpose. None of this says anything about
real agents.
"""

from __future__ import annotations

import hashlib

from harness.eval.benchmark import Benchmark, Outcome
from harness.session import Session
from harness.sim import scripted_agent_turns

SYSTEM = "You are a support agent.\nCurrent time: 2026-10-26T14:{minute:02d}:00Z\n" + "Follow the policy carefully. " * 60
TOOLS = [{"type": "function", "function": {"name": "read_log", "parameters": {"type": "object", "properties": {"part": {"type": "integer"}}}}}]


def _unit(*parts: object) -> float:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class SimulatedBenchmark(Benchmark):
    name = "simulated"

    def __init__(self, tasks: int = 10, turns: int = 8, output_chars: int = 6000, sim_cost_usd: float = 0.001):
        self.n_tasks = tasks
        self.turns = turns
        self.output_chars = output_chars
        self.sim_cost_usd = sim_cost_usd

    def task_ids(self) -> list[str]:
        return [f"sim-{i:03d}" for i in range(self.n_tasks)]

    def run(self, task_id: str, trial: int, agent: Session) -> Outcome:
        history = _to_chat(scripted_agent_turns(self.turns, self.output_chars))
        for turn, end in enumerate(range(3, len(history) + 1, 2)):
            messages = [{"role": "system", "content": SYSTEM.format(minute=turn)}] + history[:end]
            agent.call("chat_completions", {"model": "example-model", "tools": TOOLS, "messages": messages})
        p_success = 0.3 + 0.6 * _unit("task", task_id)
        return Outcome(success=_unit("trial", task_id, trial) < p_success, sim_cost_usd=self.sim_cost_usd)


def _to_chat(items: list[dict]) -> list[dict]:
    out = []
    for item in items:
        if item.get("type") == "function_call":
            out.append({"role": "assistant", "content": None, "tool_calls": [{"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": item["arguments"]}}]})
        elif item.get("type") == "function_call_output":
            out.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
        else:
            out.append(item)
    return out
