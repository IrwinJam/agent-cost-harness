import datetime as dt

import httpx
import pytest

from harness.__main__ import main as harness_main
from harness.budget import Budget, BudgetExceeded, SpendLimit, start_of_day
from harness.pricing import PriceTable
from harness.serve import build_server, parse_policies
from harness.session import Session
from harness.sim import SimulatedAdapter
from harness.trace import TraceStore

# 1M input tokens cost $1, so the simulator's calls cost real, easy to read amounts.
PRICES = PriceTable({"example": {"models": {"example-model": [{"effective_from": "2026-01-01", "input": 1.0, "cached_input": 0.1, "output": 4.0}]}}})
BODY = {"model": "example-model", "messages": [{"role": "user", "content": "x" * 4000}]}


def reply(protocol, request):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}


def session(store, key="conv", **kw):
    run = store.start_run(key)
    return Session(SimulatedAdapter(reply), task_id=key, store=store, run_id=run, prices=PRICES, **kw)


def one_call_cost(tmp_path):
    with TraceStore(tmp_path / "probe.sqlite") as store:
        session(store).call("chat_completions", BODY)
        return store.spent()


# ---------------- the store ----------------


def test_spent_by_run_and_since(tmp_path):
    with TraceStore(tmp_path / "t.sqlite") as store:
        a, b = session(store, "a"), session(store, "b")
        a.call("chat_completions", BODY)
        a.call("chat_completions", BODY)
        b.call("chat_completions", BODY)
        total = store.spent()
        assert total > 0
        assert store.spent(run_id=a.run_id) == pytest.approx(total * 2 / 3)
        assert store.spent(since=dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=1)) == 0
        assert store.spent(since=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1)) == pytest.approx(total)
        top = store.spend_by_run()
        assert [r["label"] for r in top] == ["a", "b"] and top[0]["calls"] == 2


def test_calls_from_yesterday_dont_count_today(tmp_path):
    with TraceStore(tmp_path / "t.sqlite") as store:
        s = session(store)
        s.call("chat_completions", BODY)
        yesterday = (start_of_day() - dt.timedelta(hours=1)).astimezone(dt.timezone.utc).isoformat()
        with store.lock:
            store.connection.execute("UPDATE calls SET created_at = ?", (yesterday,))
            store.connection.commit()
        assert store.spent(since=start_of_day()) == 0
        assert store.spent() > 0
        SpendLimit(daily_usd=1e-9).check(store)  # nothing spent today, so no error


# ---------------- limits ----------------


@pytest.mark.parametrize("field", ["total_usd", "daily_usd", "per_conversation_usd"])
def test_limits_must_be_positive(field):
    with pytest.raises(ValueError):
        SpendLimit(**{field: 0})


def test_each_limit_stops_calls(tmp_path):
    cost = one_call_cost(tmp_path)
    for limit, message in [
        (SpendLimit(total_usd=cost * 1.5), "total limit"),
        (SpendLimit(daily_usd=cost * 1.5), "daily limit"),
        (SpendLimit(per_conversation_usd=cost * 1.5), "conversation limit"),
    ]:
        with TraceStore(tmp_path / f"{message}.sqlite") as store:
            s = session(store, limit=limit)
            s.call("chat_completions", BODY)
            s.call("chat_completions", BODY)  # allowed: only one call's worth spent before it
            with pytest.raises(BudgetExceeded, match=message):
                s.call("chat_completions", BODY)
            assert store.summary(s.run_id)["calls"] == 2  # the blocked call never went out


def test_per_conversation_limit_only_blocks_that_conversation(tmp_path):
    cost = one_call_cost(tmp_path)
    with TraceStore(tmp_path / "t.sqlite") as store:
        limit = SpendLimit(per_conversation_usd=cost * 0.5)
        a, b = session(store, "a", limit=limit), session(store, "b", limit=limit)
        a.call("chat_completions", BODY)
        with pytest.raises(BudgetExceeded):
            a.call("chat_completions", BODY)
        b.call("chat_completions", BODY)  # a different conversation still works


def test_limit_needs_a_store():
    with pytest.raises(ValueError, match="trace store"):
        Session(SimulatedAdapter(reply), task_id="x", prices=PRICES, limit=SpendLimit(total_usd=1))
    Session(SimulatedAdapter(reply), task_id="x", prices=PRICES, limit=SpendLimit())  # nothing set is fine


def test_simple_budget_still_works():
    b = Budget(1.0)
    b.charge(0.6)
    b.check()
    assert b.remaining_usd == pytest.approx(0.4)
    b.charge(0.5)
    with pytest.raises(BudgetExceeded):
        b.check()


# ---------------- serve ----------------


def test_total_limit_survives_a_server_restart(tmp_path):
    cost = one_call_cost(tmp_path)
    db = str(tmp_path / "serve.sqlite")
    limit = SpendLimit(total_usd=cost * 1.5)

    first = build_server(SimulatedAdapter(reply), chain=[], db=db, port=0, prices=PRICES, limit=limit)
    try:
        url = first.proxy.base_url + "/s/a/v1/chat/completions"
        assert httpx.post(url, json=BODY).status_code == 200
        assert httpx.post(url, json=BODY).status_code == 200
    finally:
        first.stop()

    second = build_server(SimulatedAdapter(reply), chain=[], db=db, port=0, prices=PRICES, limit=limit)
    try:
        r = httpx.post(second.proxy.base_url + "/s/new-key/v1/chat/completions", json=BODY)
        assert r.status_code == 429
        assert "total limit" in r.json()["error"]["message"]
    finally:
        second.stop()


def test_per_conversation_limit_in_serve(tmp_path):
    cost = one_call_cost(tmp_path)
    s = build_server(SimulatedAdapter(reply), chain=parse_policies("cache", []), db=str(tmp_path / "s.sqlite"), port=0, prices=PRICES, limit=SpendLimit(per_conversation_usd=cost * 0.5))
    try:
        base = s.proxy.base_url
        assert httpx.post(base + "/s/a/v1/chat/completions", json=BODY).status_code == 200
        assert httpx.post(base + "/s/a/v1/chat/completions", json=BODY).status_code == 429
        assert httpx.post(base + "/s/b/v1/chat/completions", json=BODY).status_code == 200
    finally:
        s.stop()


# ---------------- spend command ----------------


def test_spend_command(tmp_path, capsys):
    db = tmp_path / "s.sqlite"
    with TraceStore(db) as store:
        session(store, "serve/ticket-1").call("chat_completions", BODY)
    assert harness_main(["spend", "--db", str(db), "--max-total", "10", "--max-per-day", "1e-9"]) == 0
    out = capsys.readouterr().out
    assert "total: $" in out and "left" in out
    assert "LIMIT REACHED" in out  # the tiny daily limit
    assert "serve/ticket-1" in out
    assert harness_main(["spend", "--db", str(tmp_path / "missing.sqlite")]) == 1
