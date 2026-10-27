import json

import httpx
import pytest

from harness.ab import CONTROL, TREATMENT, Group, assign, compare, load, main, report
from harness.pricing import PriceTable
from harness.serve import ServeError, build_server, main as serve_main, parse_policies
from harness.sim import SimulatedAdapter
from harness.trace import TraceStore

PRICES = PriceTable({"example": {"models": {"example-model": [{"effective_from": "2026-01-01", "input": 1.25, "cached_input": 0.15, "output": 4.25}]}}})
BODY = {"model": "example-model", "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]}


def reply(protocol, request):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}


def test_assign_is_stable_and_roughly_even():
    keys = [f"conv-{i}" for i in range(2000)]
    groups = [assign(k) for k in keys]
    assert groups == [assign(k) for k in keys]
    assert 0.45 < groups.count(TREATMENT) / len(groups) < 0.55


def test_assign_share():
    keys = [f"k{i}" for i in range(2000)]
    share = [assign(k, 0.2) for k in keys].count(TREATMENT) / len(keys)
    assert 0.15 < share < 0.25


def test_serve_ab_splits_conversations(tmp_path):
    seen = []

    def recording(protocol, request):
        seen.append((protocol, request))
        return reply(protocol, request)

    s = build_server(SimulatedAdapter(recording), chain=parse_policies("cache", []), db=str(tmp_path / "s.sqlite"), port=0, prices=PRICES, ab=0.5)
    try:
        keys = [f"conv-{i}" for i in range(20)]
        for key in keys:
            r = httpx.post(s.proxy.base_url + f"/s/{key}/v1/chat/completions", json=BODY)
            assert r.status_code == 200
        runs = {json.loads(r["meta"])["key"]: json.loads(r["meta"]) for r in s.store.runs()}
        for key, (_, request) in zip(keys, seen):
            group = assign(key)
            assert runs[key]["ab"] == group
            assert runs[key]["policies"] == ([] if group == CONTROL else ["cache_layout"])
            assert ("prompt_cache_key" in request) == (group == TREATMENT)
        assert {m["ab"] for m in runs.values()} == {CONTROL, TREATMENT}
    finally:
        s.stop()


def test_serve_without_ab_has_no_groups(tmp_path):
    s = build_server(SimulatedAdapter(reply), chain=parse_policies("cache", []), db=str(tmp_path / "s.sqlite"), port=0, prices=PRICES)
    try:
        httpx.post(s.proxy.base_url + "/s/x/v1/chat/completions", json=BODY)
        assert "ab" not in json.loads(s.store.runs()[0]["meta"])
    finally:
        s.stop()


@pytest.mark.parametrize("args", [["--simulate", "--ab", "1.5"], ["--simulate", "--policies", "", "--ab"]])
def test_serve_rejects_bad_ab(args, capsys):
    assert serve_main(args) == 2


def test_compare_finds_a_clear_drop():
    control = Group(CONTROL, [1.0, 1.2, 0.8, 1.1, 0.9] * 10, 0, 0, 0)
    treatment = Group(TREATMENT, [0.8, 0.96, 0.64, 0.88, 0.72] * 10, 0, 0, 0)
    cmp = compare(control, treatment, reps=2000)
    assert cmp.change == pytest.approx(-0.2)
    lo, hi = cmp.ci
    assert lo < -0.2 < hi < 0
    text = report(cmp)
    assert "-20.0%" in text and "cheaper" in text


def test_report_warns_on_small_groups():
    cmp = compare(Group(CONTROL, [1.0, 1.1], 0, 0, 0), Group(TREATMENT, [0.5, 0.6], 0, 0, 0), reps=200)
    assert "Wait for 30+" in report(cmp)


def test_report_with_no_data():
    cmp = compare(Group(CONTROL, [], 0, 0, 0), Group(TREATMENT, [1.0], 0, 0, 0))
    assert cmp.change is None and "not enough data" in report(cmp)


def test_load_and_cli(tmp_path, capsys):
    db = tmp_path / "s.sqlite"
    s = build_server(SimulatedAdapter(reply), chain=parse_policies("cache", []), db=str(db), port=0, prices=PRICES, ab=0.5)
    try:
        for i in range(10):
            httpx.post(s.proxy.base_url + f"/s/c{i}/v1/chat/completions", json=BODY)
    finally:
        s.stop()
    with TraceStore(db) as store:
        groups = load(store)
    assert groups[CONTROL].n + groups[TREATMENT].n == 10
    assert main(["--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "control" in out and "policies" in out


def test_cli_missing_db(tmp_path, capsys):
    assert main(["--db", str(tmp_path / "nope.sqlite")]) == 1
