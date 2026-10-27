"""Session tests with a scripted adapter (no network)."""

import copy
import datetime as dt
import json

import pytest

from harness.adapters.base import Adapter, CallResult
from harness.policies import CacheStableLayout, OutputHandles
from harness.pricing import PriceTable
from harness.profiler import build_profile
from harness.segments import first_divergence, segment
from harness.session import Session
from harness.trace import TraceStore
from harness.usage import Usage

PRICES = PriceTable({"example": {"models": {"m": [{"effective_from": "2026-01-01", "input": 1.0, "cached_input": 0.1, "output": 4.0}]}}})
DAY = dt.date(2026, 10, 26)


class ScriptedAdapter(Adapter):
    """Returns queued responses and records the requests it received."""

    provider = "example"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def call(self, protocol, request):
        self.requests.append(copy.deepcopy(request))
        body = self.responses.pop(0)
        usage = Usage(100, 0, 10, 0, protocol)
        return CallResult("example", protocol, "m", request, body, usage, 0.01, 1)


def fcall(call_id, name, args):
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args)}


def text(t):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": t}]}


BIG = "\n".join(f"row {i}" for i in range(2000))


def agent_history():
    return [
        {"role": "user", "content": "summarize the file"},
        fcall("c0", "read_file", {"path": "a.txt"}),
        {"type": "function_call_output", "call_id": "c0", "output": BIG},
    ]


def session(adapter, **kw):
    return Session(adapter, task_id="t1", policies=[CacheStableLayout(), OutputHandles(max_chars=1000, head_chars=300, tail_chars=200)], prices=PRICES, on=DAY, **kw)


def test_harness_tool_loop_is_invisible_to_agent():
    adapter = ScriptedAdapter(
        [
            {"output": [fcall("h1", "fetch_output", {"handle": "out_c0", "start": 5000, "length": 30})]},
            {"output": [text("done")]},
        ]
    )
    s = session(adapter)
    response = s.call("responses", {"model": "m", "input": agent_history()})

    assert response["output"] == [text("done")]
    assert len(adapter.requests) == 2
    second = adapter.requests[1]["input"]
    assert second[-2] == fcall("h1", "fetch_output", {"handle": "out_c0", "start": 5000, "length": 30})
    assert second[-1]["output"].startswith("[characters 5000-5030 of")


def test_injections_keep_the_prefix_stable_on_the_next_turn():
    adapter = ScriptedAdapter(
        [
            {"output": [fcall("h1", "fetch_output", {"handle": "out_c0", "start": 0, "length": 10})]},
            {"output": [text("first answer")]},
            {"output": [text("second answer")]},
        ]
    )
    s = session(adapter)
    hist = agent_history()
    first = s.call("responses", {"model": "m", "input": hist})
    # The agent appends what it saw (no harness call) and its next user message.
    hist = hist + first["output"] + [{"role": "user", "content": "thanks, now the totals"}]
    s.call("responses", {"model": "m", "input": hist})

    seen_by_model = segment("responses", adapter.requests[1])
    next_request = segment("responses", adapter.requests[2])
    assert first_divergence(seen_by_model, next_request) is None  # pure append: cache-friendly


def test_injection_dropped_when_agent_rewrites_history():
    adapter = ScriptedAdapter(
        [
            {"output": [fcall("h1", "fetch_output", {"handle": "out_c0", "start": 0, "length": 10})]},
            {"output": [text("a")]},
            {"output": [text("b")]},
        ]
    )
    s = session(adapter)
    s.call("responses", {"model": "m", "input": agent_history()})
    s.call("responses", {"model": "m", "input": [{"role": "user", "content": "new conversation"}]})
    assert not any(item.get("call_id") == "h1" for item in adapter.requests[2]["input"])


def test_mixed_turn_returns_agent_calls_only():
    adapter = ScriptedAdapter(
        [
            {"output": [fcall("h1", "fetch_output", {"handle": "out_c0", "start": 0, "length": 5}), fcall("c1", "search", {"q": "x"})]},
            {"output": [text("ok")]},
        ]
    )
    s = session(adapter)
    hist = agent_history()
    response = s.call("responses", {"model": "m", "input": hist})
    assert [o["call_id"] for o in response["output"]] == ["c1"]

    hist = hist + response["output"] + [{"type": "function_call_output", "call_id": "c1", "output": "found"}]
    s.call("responses", {"model": "m", "input": hist})
    ids = [item.get("call_id") for item in adapter.requests[1]["input"]]
    # Harness call and its result sit before the agent's own call, each output right after its call.
    assert ids[-4:] == ["h1", "h1", "c1", "c1"]


def test_round_limit_strips_harness_calls():
    loop = {"output": [text("thinking"), fcall("h", "fetch_output", {"handle": "out_c0", "start": 0, "length": 1})]}
    adapter = ScriptedAdapter([copy.deepcopy(loop) for _ in range(3)])
    s = session(adapter, max_harness_rounds=2)
    response = s.call("responses", {"model": "m", "input": agent_history()})
    assert response["output"] == [text("thinking")]
    assert len(adapter.requests) == 3


def test_bad_arguments_and_unknown_handle_return_errors_to_model():
    adapter = ScriptedAdapter(
        [
            {"output": [{"type": "function_call", "call_id": "h1", "name": "fetch_output", "arguments": "{not json"}]},
            {"output": [text("ok")]},
        ]
    )
    s = session(adapter)
    s.call("responses", {"model": "m", "input": agent_history()})
    assert adapter.requests[1]["input"][-1]["output"].startswith("error: arguments are not valid JSON")


def test_chat_completions_loop():
    def chat(message):
        return {"choices": [{"message": message, "finish_reason": "stop"}]}

    fetch = {"id": "h1", "type": "function", "function": {"name": "fetch_output", "arguments": '{"handle": "out_t0", "start": 0, "length": 4}'}}
    adapter = ScriptedAdapter([chat({"role": "assistant", "content": None, "tool_calls": [fetch]}), chat({"role": "assistant", "content": "done"})])
    s = session(adapter)
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t0", "type": "function", "function": {"name": "cat", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t0", "content": BIG},
    ]
    response = s.call("chat_completions", {"model": "m", "messages": messages})
    assert response["choices"][0]["message"]["content"] == "done"
    second = adapter.requests[1]["messages"]
    assert second[-2]["tool_calls"][0]["id"] == "h1"
    assert second[-1] == {"role": "tool", "tool_call_id": "h1", "content": "[characters 0-4 of %d]\nrow " % len(BIG)}


def test_session_traces_every_call_including_harness_rounds():
    adapter = ScriptedAdapter(
        [
            {"output": [fcall("h1", "fetch_output", {"handle": "out_c0", "start": 0, "length": 10})]},
            {"output": [text("done")]},
        ]
    )
    with TraceStore(":memory:") as store:
        run = store.start_run("handles")
        s = session(adapter, store=store, run_id=run)
        s.call("responses", {"model": "m", "input": agent_history()})
        profile = build_profile(store, run)
        assert len(profile.calls) == 2
        assert profile.total_cost == pytest.approx(2 * (100 * 1.0 + 10 * 4.0) / 1e6)


def test_tracing_requires_run_id():
    with TraceStore(":memory:") as store, pytest.raises(ValueError):
        Session(ScriptedAdapter([]), task_id="t", store=store)
