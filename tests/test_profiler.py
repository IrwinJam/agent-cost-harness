"""Profiler tests on synthetic traces.

Each simulated request reports ``input_tokens`` equal to its total segment characters (1 token per
char), so the character-based estimates in the profiler are exact and amounts can be asserted.
"""

import datetime as dt
import json

import pytest

from harness.adapters.base import CallResult
from harness.pricing import PriceTable
from harness.profiler import attribute, build_profile, diagnose_cache, find_waste, load_run
from harness.profiler.__main__ import main
from harness.segments import segment
from harness.trace import TraceStore
from harness.usage import Usage

PRICES = PriceTable(
    {
        "example": {
            "models": {
                "example-model": [{"effective_from": "2026-07-01", "input": 1.0, "cached_input": 0.1, "output": 4.0}],
                "example-model-large": [{"effective_from": "2026-07-01", "input": 4.0, "cached_input": 0.2, "output": 20.0}],
            }
        }
    }
)
DAY = dt.date(2026, 10, 26)
TOOLS = [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}]
SYSTEM = "You are a careful agent. " * 100  # ~2.5k chars
BIG = "x" * 20_000


def request(items, *, tools=TOOLS, system=SYSTEM, model="example-model"):
    return {"model": model, "tools": tools, "instructions": system, "input": items}


def chars(req):
    return sum(s.chars for s in segment("responses", req))


def record(store, run_id, req, *, cached=None, output=100, reasoning=0, prefix_of=None):
    """Record a call. ``prefix_of`` = previous request whose full size is served from cache."""
    total = chars(req)
    if cached is None:
        cached = min(chars(prefix_of), total) if prefix_of is not None else 0
    usage = Usage(total, cached, output, reasoning, "responses")
    result = CallResult("example", "responses", req["model"], req, {"usage": {}}, usage, 0.1, 1)
    store.record(run_id, result, PRICES.cost("example", req["model"], usage, DAY))


def call(i, args='{"path": "a.py"}'):
    return {"type": "function_call", "call_id": f"c{i}", "name": "read_file", "arguments": args}


def output(i, text=BIG):
    return {"type": "function_call_output", "call_id": f"c{i}", "output": text}


USER = {"role": "user", "content": "Fix the bug in a.py"}


@pytest.fixture
def store():
    with TraceStore(":memory:") as s:
        yield s


def healthy_run(store):
    """Append-only agent loop with a fully warm cache: the big tool output is carried for 4 more calls."""
    run = store.start_run("healthy")
    items = [USER]
    prev = None
    for i in range(5):
        req = request(list(items))
        record(store, run, req, prefix_of=prev)
        prev = req
        items += [call(i, f'{{"path": "f{i}.py"}}'), output(i, BIG if i == 0 else "short")]
    return run


def test_attribution_sums_to_total_and_counts_resends(store):
    run = healthy_run(store)
    calls = load_run(store, run)
    sources = attribute(calls)
    assert sum(s.cost for s in sources) == pytest.approx(sum(c.total_cost for c in calls))

    big = next(s for s in sources if s.label == "tool_output c0")
    assert big.sends == 4  # first appears in call 1, carried through call 4
    assert big.tokens_per_send == pytest.approx(len(BIG))
    assert big.resend_cost == pytest.approx(big.cost * 3 / 4)
    # A warm cache makes re-sends cheap: only the first send is uncached.
    assert big.cached_tokens == pytest.approx(3 * len(BIG))


def test_masked_version_is_a_separate_source(store):
    run = healthy_run(store)
    record(store, run, request([USER, call(0, '{"path": "f0.py"}'), output(0, "[masked]")]))
    sources = attribute(load_run(store, run))
    versions = [s for s in sources if s.origin == "tool_output" and s.key == "c0"]
    assert len(versions) == 2
    original = max(versions, key=lambda s: s.tokens)
    assert original.sends == 4 and original.tokens_per_send == pytest.approx(len(BIG))
    assert all(s.label.startswith("tool_output c0 @") for s in versions)


def test_healthy_run_has_no_cache_events(store):
    run = healthy_run(store)
    assert diagnose_cache(load_run(store, run)) == []


def test_provider_miss_when_prefix_unchanged_but_not_cached(store):
    run = store.start_run()
    first = request([USER, call(0), output(0)])
    record(store, run, first)
    record(store, run, request(first["input"] + [{"role": "user", "content": "go on"}]), cached=0)
    (event,) = diagnose_cache(load_run(store, run))
    assert event.cause == "provider_miss"
    assert event.lost_tokens == pytest.approx(chars(first))
    assert event.lost_cost == pytest.approx(chars(first) * 0.9 / 1e6)


def test_masked_tool_output_is_a_history_rewrite(store):
    run = store.start_run()
    first = request([USER, call(0), output(0), call(1), output(1, "y" * 5000)])
    record(store, run, first)
    masked = [USER, call(0), output(0, "[masked]"), call(1), output(1, "y" * 5000)]
    second = request(masked)
    # The cache can only serve what precedes the masked output.
    record(store, run, second, cached=chars(request([USER, call(0)])))
    (event,) = diagnose_cache(load_run(store, run))
    assert event.cause == "history_rewritten"
    assert event.at_segment == "tool_output c0"
    assert event.lost_tokens == pytest.approx(len("[masked]") + chars(request([call(1), output(1, "y" * 5000)], tools=None, system=None)))


@pytest.mark.parametrize(
    "change,cause",
    [
        (dict(tools=TOOLS + [{"type": "function", "name": "write_file", "parameters": {}}]), "tool_schemas_changed"),
        (dict(system="Current time: 12:01. " + SYSTEM), "system_prompt_changed"),
    ],
)
def test_changes_at_the_top_of_the_prompt(store, change, cause):
    run = store.start_run()
    items = [USER, call(0), output(0)]
    record(store, run, request(items))
    record(store, run, request(items, **change), cached=0)
    events = diagnose_cache(load_run(store, run))
    assert [e.cause for e in events] == [cause]


def test_dropping_old_turns_is_truncation(store):
    run = store.start_run()
    second_call, second_output = call(1, '{"path": "b.py"}'), output(1, "z" * 20_000)
    record(store, run, request([USER, call(0), output(0), second_call, second_output]))
    # The unchanged head (tools, system, first user message) is still served from cache.
    record(store, run, request([USER, second_call, second_output]), cached=chars(request([USER])))
    (event,) = diagnose_cache(load_run(store, run))
    assert event.cause == "history_truncated"
    assert event.at_segment == "tool_call c1"


def test_shrinking_to_a_prefix_loses_nothing(store):
    run = store.start_run()
    full = request([USER, call(0), output(0)])
    record(store, run, full)
    shorter = request([USER])
    record(store, run, shorter, cached=chars(shorter))
    assert diagnose_cache(load_run(store, run)) == []


def test_model_switch(store):
    run = store.start_run()
    items = [USER, call(0), output(0)]
    record(store, run, request(items))
    record(store, run, request(items, model="example-model-large"), cached=0)
    (event,) = diagnose_cache(load_run(store, run))
    assert event.cause == "model_switch"
    assert event.lost_cost == pytest.approx(chars(request(items)) * 3.8 / 1e6)


def test_small_misses_are_tolerated(store):
    run = store.start_run()
    first = request([USER, call(0), output(0)])
    record(store, run, first)
    second = request(first["input"] + [{"role": "user", "content": "ok"}])
    record(store, run, second, cached=chars(first) - 500)  # below the 1024-token tolerance
    assert diagnose_cache(load_run(store, run)) == []


def test_waste_findings(store):
    run = healthy_run(store)
    # Re-issue an identical tool call under a new id, with heavy reasoning for a tiny answer.
    items = [USER, call(0, '{"path": "f0.py"}'), output(0), call(9, '{"path": "f0.py"}')]
    record(store, run, request(items), output=1300, reasoning=1280)
    calls = load_run(store, run)
    kinds = {f.kind: f for f in find_waste(calls, attribute(calls))}
    assert "large_tool_output_carried" in kinds
    assert "tool_output c0" in kinds["large_tool_output_carried"].detail
    assert "c0, c9" in kinds["possible_duplicate_tool_call"].detail
    assert kinds["heavy_reasoning_small_step"].seqs == (5,)
    assert kinds["heavy_reasoning_small_step"].cost == pytest.approx(1280 * 4.0 / 1e6)


def test_profile_renders(store):
    run = healthy_run(store)
    profile = build_profile(store, run)
    md = profile.to_markdown()
    assert "# Cost profile: healthy" in md
    assert "tool_output c0" in md
    assert "No cache breaks" in md
    data = json.loads(profile.to_json())
    assert data["calls"] == 5
    assert data["total_cost_usd"] == pytest.approx(profile.total_cost)
    assert sum(data["cost_by_origin"].values()) == pytest.approx(profile.total_cost)


def test_build_profile_unknown_run(store):
    with pytest.raises(KeyError):
        build_profile(store, "nope")


def test_cli(tmp_path, capsys):
    db = tmp_path / "t.sqlite"
    with TraceStore(db) as s:
        run = s.start_run("cli-run")
        record(s, run, request([USER]))
    assert main(["--db", str(db), "--list"]) == 0
    assert run in capsys.readouterr().out
    # Rates are read from the trace, so the CLI needs no price table.
    assert main(["--db", str(db), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["label"] == "cli-run"
    assert main(["--db", str(tmp_path / "missing.sqlite")]) == 1
