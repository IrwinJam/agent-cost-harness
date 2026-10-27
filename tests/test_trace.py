import datetime as dt
import json

import pytest

from harness.adapters.base import CallResult
from harness.pricing import PriceTable
from harness.trace import TraceStore
from harness.usage import Usage

PRICES = PriceTable(
    {"example": {"models": {"example-model": [{"effective_from": "2026-07-01", "input": 1.0, "cached_input": 0.1, "output": 4.0}]}}}
)


def result(input_tokens, cached, output, items):
    usage = Usage(input_tokens, cached, output, 0, "responses")
    return CallResult(
        provider="example",
        protocol="responses",
        model="example-model",
        request={"model": "example-model", "instructions": "sys", "input": items},
        response={"id": "r", "usage": {}},
        usage=usage,
        latency_s=0.5,
        attempts=1,
    )


def record(store, run_id, r):
    return store.record(run_id, r, PRICES.cost("example", r.model, r.usage, dt.date(2026, 10, 26)))


def test_record_summary_and_segments():
    with TraceStore(":memory:") as store:
        run_id = store.start_run("baseline", {"task": "task-1"})
        first = [{"role": "user", "content": "q"}]
        c1 = record(store, run_id, result(1_000_000, 0, 0, first))
        c2 = record(store, run_id, result(1_000_000, 900_000, 100_000, first + [{"role": "assistant", "content": "a"}]))

        rows = store.calls(run_id)
        assert [r["seq"] for r in rows] == [0, 1]
        assert [s.origin for s in store.segments(c2)] == ["system", "message:user", "message:assistant"]
        assert store.segments(c1)[1].hash == store.segments(c2)[1].hash

        summary = store.summary(run_id)
        assert summary["calls"] == 2
        assert summary["cache_hit_rate"] == pytest.approx(0.45)
        # call 1: $1.00 uncached; call 2: $0.10 uncached + $0.09 cached + $0.40 output
        assert summary["cost_usd"] == pytest.approx(1.59)
        assert summary["all_prices_verified"] is False


def test_empty_run_summary():
    with TraceStore(":memory:") as store:
        run_id = store.start_run()
        summary = store.summary(run_id)
        assert summary["calls"] == 0 and summary["cost_usd"] == 0 and summary["cache_hit_rate"] == 0.0


def test_runs_are_isolated_and_export(tmp_path):
    with TraceStore(tmp_path / "t.sqlite") as store:
        a, b = store.start_run("a"), store.start_run("b")
        record(store, a, result(100, 0, 10, "x"))
        record(store, b, result(200, 0, 10, "y"))
        assert store.summary(a)["input_tokens"] == 100
        out = tmp_path / "a.jsonl"
        assert store.export_jsonl(a, out) == 1
        line = json.loads(out.read_text().splitlines()[0])
        assert line["request"]["input"] == "x"
        assert line["segments"][0]["origin"] == "system"


def test_unknown_run_is_rejected():
    import sqlite3

    with TraceStore(":memory:") as store:
        with pytest.raises(sqlite3.IntegrityError):
            record(store, "missing", result(1, 0, 1, "x"))
