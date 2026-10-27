import copy
import json

import httpx
import pytest

from harness.adapters import RequestRejected
from harness.budget import Budget, BudgetExceeded
from harness.eval import (
    Benchmark,
    EvalResults,
    EvalStopped,
    Outcome,
    Protocol,
    ProtocolError,
    TrialRow,
    build_certificate,
    compare,
    run_eval,
    summarize_arm,
    to_markdown,
)
from harness.eval.proxy import ProxyServer, opening_key
from harness.eval.simbench import SimulatedBenchmark
from harness.eval.stats import pass_hat_k
from harness.pricing import PriceTable
from harness.session import Session
from harness.sim import SimulatedAdapter
from harness.trace import TraceStore

PROTOCOL = {
    "name": "test",
    "benchmark": {"name": "fake", "tasks": "all"},
    "agent": {"model": "example-model", "protocol": "chat_completions"},
    "trials": 2,
    "baseline": "base",
    "primary": "treat",
    "arms": {"base": [], "treat": [{"policy": "cache_layout"}]},
    "analysis": {"noninferiority_margin_pp": 3.0, "alpha": 0.05, "bootstrap_reps": 2000, "seed": 7},
    "budget_usd": 10.0,
}
PRICES = PriceTable({"example": {"models": {"example-model": [{"effective_from": "2026-01-01", "input": 1.25, "cached_input": 0.15, "output": 4.25}]}}})


def chat_reply(protocol, request):
    return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


# ---------------- protocol ----------------


def test_bundled_protocol_is_valid():
    p = Protocol.load("eval/protocol.yaml")
    assert p.primary in p.arms and p.baseline in p.arms
    assert len(p.hash) == 16


@pytest.mark.parametrize(
    "change",
    [
        {"primary": "nope"},
        {"primary": "base"},
        {"trials": 0},
        {"arms": {"base": [], "treat": [{"policy": "magic"}]}},
        {"arms": {"base": [], "treat": [{"policy": "masking", "bogus": 1}]}},
        {"analysis": {"noninferiority_margin_pp": 3, "alpha": 0.7, "bootstrap_reps": 2000, "seed": 1}},
        {"budget_usd": 0},
    ],
)
def test_protocol_validation(change):
    with pytest.raises(ProtocolError):
        Protocol.from_dict({**PROTOCOL, **change})


def test_protocol_hash_tracks_content():
    a = Protocol.from_dict(PROTOCOL)
    b = Protocol.from_dict({**PROTOCOL, "trials": 3})
    assert a.hash != b.hash
    assert a.hash == Protocol.from_dict(copy.deepcopy(PROTOCOL)).hash


def test_explicit_task_list_must_exist():
    p = Protocol.from_dict({**PROTOCOL, "benchmark": {"name": "fake", "tasks": ["t1", "t9"]}})
    with pytest.raises(ProtocolError):
        p.task_ids(["t1", "t2"])


# ---------------- stats ----------------


def rows(arm, outcomes, cost):
    """outcomes: {task: [success per trial]}"""
    return [TrialRow(arm, task, i, None, "ok", s, None, cost, 0.0, None) for task, trials in outcomes.items() for i, s in enumerate(trials)]


def test_pass_hat_k():
    assert pass_hat_k([(2, 4)], 1) == pytest.approx(0.5)
    assert pass_hat_k([(2, 4)], 2) == pytest.approx(1 / 6)
    assert pass_hat_k([(4, 4), (0, 4)], 4) == pytest.approx(0.5)


def test_compare_detects_real_savings_and_noninferiority():
    tasks = {f"t{i}": [i % 3 != 0, True, i % 2 == 0] for i in range(40)}
    data = rows("base", tasks, 0.10) + rows("treat", tasks, 0.06)
    c = compare(data, "treat", "base", margin_pp=3, alpha=0.05, reps=2000, seed=1)
    assert c.cost_change == pytest.approx(-0.4)
    assert c.cheaper and c.noninferior and c.claim_holds
    assert c.success_diff_pp == 0.0


def test_compare_flags_quality_loss():
    base = {f"t{i}": [True, True, True] for i in range(40)}
    treat = {f"t{i}": [i % 4 != 0, True, True] for i in range(40)}  # loses ~8 pp
    c = compare(rows("base", base, 0.10) + rows("treat", treat, 0.05), "treat", "base", margin_pp=3, alpha=0.05, reps=2000, seed=1)
    assert c.cheaper
    assert c.success_diff_pp == pytest.approx(-100 / 12)
    assert not c.noninferior and not c.claim_holds
    assert len(c.worse_tasks) == 10


def test_compare_wide_ci_on_noisy_success_fails_noninferiority():
    # Same true rate, but few tasks and noisy outcomes: the CI is too wide to show non-inferiority.
    base = {f"t{i}": [i % 2 == 0] for i in range(6)}
    treat = {f"t{i}": [i % 3 == 0] for i in range(6)}
    c = compare(rows("base", base, 0.1) + rows("treat", treat, 0.05), "treat", "base", margin_pp=3, alpha=0.05, reps=2000, seed=1)
    assert not c.noninferior


def test_compare_drops_tasks_and_ignores_errors():
    data = rows("base", {"a": [True], "b": [True]}, 0.1) + rows("treat", {"a": [True]}, 0.05)
    data.append(TrialRow("treat", "b", 0, None, "error", None, None, 0.02, 0.0, "boom"))
    c = compare(data, "treat", "base", margin_pp=3, alpha=0.05, reps=1000, seed=1)
    assert c.tasks == 1 and c.dropped_tasks == ["b"]


def test_summarize_arm():
    s = summarize_arm(rows("base", {"a": [True, False], "b": [False, False]}, 0.1), "base")
    assert s.success_rate == 0.25 and s.cost_per_success == pytest.approx(0.4)
    assert s.pass_hat_k == {1: pytest.approx(0.25), 2: 0.0}


# ---------------- runner ----------------


class FakeBenchmark(Benchmark):
    name = "fake"

    def __init__(self, tasks=3, fail_on=None, calls=2):
        self.tasks, self.fail_on, self.calls, self.seen = tasks, fail_on, calls, []

    def task_ids(self):
        return [f"t{i}" for i in range(self.tasks)]

    def run(self, task_id, trial, agent: Session):
        self.seen.append((agent.ctx.task_id.split(":")[1], task_id, trial))
        if (task_id, trial) == self.fail_on:
            raise RuntimeError("env crashed")
        for i in range(self.calls):
            agent.call("chat_completions", {"model": "example-model", "messages": [{"role": "system", "content": "s" * 8000}, {"role": "user", "content": f"turn {i}"}]})
        return Outcome(success=task_id != "t0", sim_cost_usd=0.001)


def test_run_eval_records_every_trial_with_traces():
    bench = FakeBenchmark()
    with TraceStore(":memory:") as store:
        eval_id = run_eval(Protocol.from_dict(PROTOCOL), bench, SimulatedAdapter(chat_reply), store, prices=PRICES, log=lambda m: None)
        trials = EvalResults(store).trials(eval_id)
        assert len(trials) == 3 * 2 * 2
        assert all(t.status == "ok" and t.agent_cost_usd > 0 for t in trials)
        assert len(store.calls(trials[0].run_id)) == 2
        assert {(a, t, n) for a, t, n in bench.seen} == {(t.arm, t.task_id, t.trial) for t in trials}


def test_errors_are_recorded_and_retried_on_resume():
    with TraceStore(":memory:") as store:
        protocol = Protocol.from_dict(PROTOCOL)
        eval_id = run_eval(protocol, FakeBenchmark(fail_on=("t1", 0)), SimulatedAdapter(chat_reply), store, prices=PRICES, log=lambda m: None)
        results = EvalResults(store)
        errors = [t for t in results.trials(eval_id) if t.status == "error"]
        assert len(errors) == 2 and errors[0].error == "RuntimeError: env crashed"

        bench = FakeBenchmark()
        run_eval(protocol, bench, SimulatedAdapter(chat_reply), store, eval_id=eval_id, prices=PRICES, log=lambda m: None)
        assert sorted(bench.seen) == sorted((a, "t1", 0) for a in ("base", "treat"))
        assert all(t.status == "ok" for t in results.trials(eval_id))


def test_budget_stops_and_resume_refuses_changed_protocol():
    tight = {**PROTOCOL, "budget_usd": 0.02}
    with TraceStore(":memory:") as store:
        with pytest.raises(EvalStopped):
            run_eval(Protocol.from_dict(tight), FakeBenchmark(), SimulatedAdapter(chat_reply), store, prices=PRICES, log=lambda m: None)
        results = EvalResults(store)
        eval_id = results.evals()[0]["id"]
        assert results.spent(eval_id) >= 0.02
        assert any(t.error and t.error.startswith("budget") for t in results.trials(eval_id))
        with pytest.raises(ValueError, match="protocol changed"):
            run_eval(Protocol.from_dict(PROTOCOL), FakeBenchmark(), SimulatedAdapter(chat_reply), store, eval_id=eval_id, prices=PRICES)


def test_setups_take_turns_going_first():
    orders = []
    for _ in range(2):
        bench = FakeBenchmark()
        with TraceStore(":memory:") as store:
            run_eval(Protocol.from_dict(PROTOCOL), bench, SimulatedAdapter(chat_reply), store, prices=PRICES, log=lambda m: None)
        orders.append(bench.seen)
    assert orders[0] == orders[1]  # same order every run, so resuming is safe
    pairs = [orders[0][i : i + 2] for i in range(0, len(orders[0]), 2)]  # one pair per (task, trial)
    assert all(pair[0][1:] == pair[1][1:] for pair in pairs)  # both setups run back to back
    firsts = [pair[0][0] for pair in pairs]
    assert firsts == ["base", "treat"] * (len(firsts) // 2)  # strictly alternating, never lopsided


def test_budget_object():
    b = Budget(1.0)
    b.charge(0.6)
    b.check()
    b.charge(0.5)
    with pytest.raises(BudgetExceeded):
        b.check()


# ---------------- certificate ----------------


def test_certificate_from_simulated_run(tmp_path):
    protocol = {**PROTOCOL, "arms": {"base": [], "treat": [{"policy": "cache_layout"}, {"policy": "output_handles"}], "cache": [{"policy": "cache_layout"}]}}
    with TraceStore(tmp_path / "e.sqlite") as store:
        p = Protocol.from_dict(protocol)
        eval_id = run_eval(p, SimulatedBenchmark(tasks=6, turns=4), SimulatedAdapter(chat_reply), store, prices=PRICES, meta={"simulated": True}, log=lambda m: None)
        cert = build_certificate(EvalResults(store), eval_id, p)
        changed = build_certificate(EvalResults(store), eval_id, Protocol.from_dict({**protocol, "trials": 5}))
    assert cert["simulated"] and cert["claim"].startswith("SIMULATED, NOT EVIDENCE.")
    assert cert["protocol"]["unchanged_on_disk"] is True
    assert changed["protocol"]["unchanged_on_disk"] is False
    assert set(cert["secondary"]) == {"cache"}
    assert cert["primary_cost_by_origin_usd"]
    md = to_markdown(cert)
    assert "Simulated run" in md and "## Primary comparison" in md and "pass^k" in md
    json.dumps(cert)  # serialisable


# ---------------- proxy ----------------


class StrictSim(SimulatedAdapter):
    """Simulator that blocks requests the way a strict provider would."""

    def call(self, protocol, request):
        if request.get("tool_choice") not in (None, "auto"):
            raise RequestRejected("only tool_choice=auto is allowed")
        return super().call(protocol, request)


def test_proxy_routes_through_sessions():
    adapter = StrictSim(chat_reply)
    made = []

    def factory(key):
        made.append(key)
        return Session(adapter, task_id=key, prices=PRICES)

    body = {"model": "example-model", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]}
    with ProxyServer(factory) as proxy:
        r = httpx.post(proxy.url_for("task-a") + "/chat/completions", json=body)
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "ok"
        httpx.post(proxy.base_url + "/v1/chat/completions", json=body)
        httpx.post(proxy.base_url + "/v1/chat/completions", json={**body, "messages": body["messages"] + [{"role": "assistant", "content": "x"}]})
        assert made == ["task-a", opening_key("chat_completions", body)]  # same opening -> same session

        assert httpx.post(proxy.base_url + "/v1/chat/completions", json={**body, "stream": True}).status_code == 400
        assert httpx.post(proxy.base_url + "/v1/embeddings", json=body).status_code == 404
        rejected = httpx.post(proxy.url_for("task-b") + "/chat/completions", json={**body, "tool_choice": "required"})
    assert made[-1] == "task-b"
    assert rejected.status_code == 400
    assert rejected.json()["error"]["type"] == "harness_rejected"


def test_proxy_only_binds_localhost():
    with pytest.raises(ValueError):
        ProxyServer(lambda key: None, host="0.0.0.0")


# ---------------- trial details ----------------


class InfoBenchmark(FakeBenchmark):
    def run(self, task_id, trial, agent):
        out = super().run(task_id, trial, agent)
        return Outcome(success=out.success, sim_cost_usd=0.001, info={"termination_reason": "user_stop", "reward_breakdown": {"DB": 1.0}})


def test_trial_details_are_saved():
    with TraceStore(":memory:") as store:
        eval_id = run_eval(Protocol.from_dict(PROTOCOL), InfoBenchmark(), SimulatedAdapter(chat_reply), store, prices=PRICES, log=lambda m: None)
        trials = EvalResults(store).trials(eval_id)
        assert all(t.info["termination_reason"] == "user_stop" for t in trials)
        assert trials[0].info["reward_breakdown"] == {"DB": 1.0}


def test_old_databases_get_the_info_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    with TraceStore(path) as store:
        old = sqlite3.connect(path)
        old.executescript(
            """CREATE TABLE evals (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, protocol_hash TEXT NOT NULL,
                                   protocol_json TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}');
               CREATE TABLE eval_trials (eval_id TEXT NOT NULL, arm TEXT NOT NULL, task_id TEXT NOT NULL,
                   trial INTEGER NOT NULL, run_id TEXT, status TEXT NOT NULL, success INTEGER, reward REAL,
                   agent_cost_usd REAL NOT NULL, sim_cost_usd REAL NOT NULL, error TEXT, finished_at TEXT NOT NULL,
                   PRIMARY KEY (eval_id, arm, task_id, trial));
               INSERT INTO evals VALUES ('e1', '2026-10-27', 'h', '{}', '{}');
               INSERT INTO eval_trials VALUES ('e1', 'base', 't0', 0, NULL, 'ok', 1, 1.0, 0.1, 0.0, NULL, '2026-10-27');"""
        )
        old.close()
        results = EvalResults(store)  # adds the missing column
        (row,) = results.trials("e1")
        assert row.info == {} and row.success is True
