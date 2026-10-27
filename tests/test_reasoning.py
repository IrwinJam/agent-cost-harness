"""Reasoning effort per call: low by default, full effort again when the reply changes something."""

import copy

import pytest

from harness.adapters.base import Adapter, CallResult
from harness.policies import CacheStableLayout, ReasoningEffort
from harness.policies.base import SessionContext
from harness.pricing import PriceTable
from harness.serve import parse_policies
from harness.session import Session
from harness.trace import TraceStore
from harness.usage import Usage

PRICES = PriceTable({"example": {"models": {"m": [{"effective_from": "2026-01-01", "input": 1.0, "cached_input": 0.1, "output": 4.0}]}}})


class Scripted(Adapter):
    provider = "example"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def call(self, protocol, request):
        self.requests.append(copy.deepcopy(request))
        return CallResult("example", protocol, "m", request, self.responses.pop(0), Usage(100, 0, 10, 0, protocol), 0.01, 1)


def chat_reply(text="ok", tools=()):
    message = {"role": "assistant", "content": text}
    if tools:
        message["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": "{}"}} for i, n in enumerate(tools)]
    return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}


def chat_request(effort="medium"):
    req = {"model": "m", "messages": [{"role": "system", "content": "be helpful"}, {"role": "user", "content": "refund order 7"}]}
    if effort:
        req["reasoning_effort"] = effort
    return req


def run(responses, request, **kw):
    adapter = Scripted(responses)
    s = Session(adapter, task_id="t", policies=[ReasoningEffort(**kw)], prices=PRICES)
    return s, adapter, s.call("chat_completions", request)


def test_read_calls_stay_low():
    s, adapter, out = run([chat_reply(tools=["get_order_details"])], chat_request())
    assert [r["reasoning_effort"] for r in adapter.requests] == ["low"]
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_order_details"


def test_plain_text_reply_stays_low():
    _, adapter, _ = run([chat_reply("what's your zip code?")], chat_request())
    assert len(adapter.requests) == 1


def test_write_call_is_asked_again_at_agent_effort():
    s, adapter, out = run([chat_reply(tools=["refund_order"]), chat_reply("done", tools=["refund_order"])], chat_request("medium"))
    assert [r["reasoning_effort"] for r in adapter.requests] == ["low", "medium"]
    assert out["choices"][0]["message"]["content"] == "done"  # the agent sees the full-effort reply
    assert s.ctx.state_for("reasoning")["escalations"] == 1


def test_retry_keeps_the_same_prompt():
    _, adapter, _ = run([chat_reply(tools=["cancel_order"]), chat_reply()], chat_request())
    first, second = adapter.requests
    assert first["messages"] == second["messages"]


def test_high_overrides_agent_effort():
    _, adapter, _ = run([chat_reply(tools=["cancel_order"]), chat_reply()], chat_request("medium"), high="high")
    assert adapter.requests[1]["reasoning_effort"] == "high"


def test_no_agent_effort_means_provider_default_on_retry():
    _, adapter, _ = run([chat_reply(tools=["cancel_order"]), chat_reply()], chat_request(None))
    assert adapter.requests[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in adapter.requests[1]


def test_write_tools_list_wins_over_prefixes():
    _, adapter, _ = run([chat_reply(tools=["get_refund"]), chat_reply()], chat_request(), write_tools="get_refund")
    assert len(adapter.requests) == 2
    _, adapter, _ = run([chat_reply(tools=["cancel_order"])], chat_request(), write_tools=["get_refund"])
    assert len(adapter.requests) == 1


def test_read_tools_and_harness_tools_are_reads():
    policy = ReasoningEffort(read_tools=["calc_total"])
    ctx = SessionContext(task_id="t")
    assert not policy.is_write("calc_total", ctx)
    assert not policy.is_write("Search_Docs", ctx)
    assert policy.is_write("exchange_items", ctx)
    ctx.harness_tools["fetch_output"] = object()
    assert not policy.is_write("fetch_output", ctx)


def test_responses_protocol():
    adapter = Scripted([{"output": [{"type": "function_call", "call_id": "c0", "name": "delete_file", "arguments": "{}"}]}, {"output": []}])
    s = Session(adapter, task_id="t", policies=[ReasoningEffort()], prices=PRICES)
    s.call("responses", {"model": "m", "input": [{"role": "user", "content": "clean up"}], "reasoning": {"effort": "high", "summary": "auto"}})
    assert [r["reasoning"] for r in adapter.requests] == [{"effort": "low", "summary": "auto"}, {"effort": "high", "summary": "auto"}]


def test_both_calls_are_traced(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite")
    run_id = store.start_run("t")
    adapter = Scripted([chat_reply(tools=["refund_order"]), chat_reply()])
    Session(adapter, task_id="t", policies=[ReasoningEffort()], prices=PRICES, store=store, run_id=run_id).call("chat_completions", chat_request())
    assert len(store.calls(run_id)) == 2  # the thrown-away call still cost money


def test_works_with_cache_layout():
    adapter = Scripted([chat_reply(tools=["refund_order"]), chat_reply()])
    s = Session(adapter, task_id="t", policies=[CacheStableLayout(), ReasoningEffort()], prices=PRICES)
    s.call("chat_completions", chat_request())
    assert adapter.requests[0]["prompt_cache_key"] == adapter.requests[1]["prompt_cache_key"]


def test_serve_alias_and_settings():
    chain = parse_policies("cache,reason", ["reason.write_tools=refund_order,cancel_order", "reason.high=high"])
    assert chain == [("cache_layout", {}), ("reasoning", {"write_tools": "refund_order,cancel_order", "high": "high"})]


@pytest.mark.parametrize("effort", ["low", "medium"])
def test_set_effort_is_deterministic(effort):
    a, b = chat_request(effort), chat_request(effort)
    p = ReasoningEffort()
    ctx = SessionContext(task_id="t")
    assert p.apply("chat_completions", a, ctx) == p.apply("chat_completions", b, ctx)


def test_no_escalate_keeps_low_for_the_whole_conversation():
    s, adapter, out = run([chat_reply(tools=["refund_order"])], chat_request("medium"), escalate=False)
    assert [r["reasoning_effort"] for r in adapter.requests] == ["low"]
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "refund_order"
    assert "escalations" not in s.ctx.state_for("reasoning")


def test_escalate_setting_from_serve():
    chain = parse_policies("reason", ["reason.escalate=false"])
    assert chain == [("reasoning", {"escalate": False})]
