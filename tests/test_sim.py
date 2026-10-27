"""Policy/cache interaction checks on the offline simulator. These guard harness logic, not real savings."""

import datetime as dt

import pytest

from harness.policies import BatchedMasking, CacheStableLayout
from harness.pricing import PriceTable
from harness.profiler import build_profile
from harness.session import Session
from harness.sim import SimulatedAdapter, scripted_agent_turns
from harness.trace import TraceStore

PRICES = PriceTable({"example": {"models": {"example-model": [{"effective_from": "2026-01-01", "input": 1.25, "cached_input": 0.15, "output": 4.25}]}}})
SYSTEM = "Rules. " * 800


def reply(protocol, request):
    return {"output": []}


def simulate(policies, *, clock=False, turns=20, output_chars=3000):
    with TraceStore(":memory:") as store:
        run = store.start_run()
        s = Session(SimulatedAdapter(reply), task_id="t", policies=policies, store=store, run_id=run, prices=PRICES, on=dt.date(2026, 10, 26))
        history = scripted_agent_turns(turns, output_chars)
        for turn, end in enumerate(range(3, len(history) + 1, 2)):
            system = SYSTEM + (f"\nTime: 2026-10-26T14:{turn:02d}:00Z" if clock else "")
            s.call("responses", {"model": "example-model", "instructions": system, "input": history[:end]})
        return build_profile(store, run)


def test_simulator_caches_append_only_prefixes():
    profile = simulate([])
    assert profile.cache_events == []
    assert profile.cache_hit_rate > 0.85


def test_ticking_clock_breaks_cache_and_p1_fixes_it():
    baseline = simulate([], clock=True)
    assert {e.cause for e in baseline.cache_events} == {"system_prompt_changed"}
    with_p1 = simulate([CacheStableLayout()], clock=True)
    assert with_p1.cache_events == []
    assert with_p1.total_cost < baseline.total_cost * 0.5


def test_batched_masking_beats_naive_masking():
    naive = simulate([BatchedMasking(keep_last=4, batch=1)])
    batched = simulate([BatchedMasking(keep_last=4, batch=4)])
    assert len(naive.cache_events) > 3 * len(batched.cache_events)
    assert batched.total_cost < naive.total_cost
    assert all(e.cause == "history_rewritten" for e in naive.cache_events + batched.cache_events)


@pytest.mark.parametrize("batch", [1, 4])
def test_masking_reduces_input_tokens(batch):
    assert simulate([BatchedMasking(keep_last=4, batch=batch)]).input_tokens < simulate([]).input_tokens
