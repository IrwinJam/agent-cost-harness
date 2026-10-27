import copy
import json

import pytest

from harness.policies import BatchedMasking, CacheStableLayout, OutputHandles, SessionContext, default_chain
from harness.policies.conv import UnsupportedProtocol
from harness.policies.masking import PLACEHOLDER_PREFIX
from harness.policies.output_handles import FETCH_TOOL, _head_tail
from harness.segments import first_divergence, segment

TOOLS = [
    {"type": "function", "name": "zeta", "parameters": {"type": "object", "properties": {"b": {}, "a": {}}}},
    {"type": "function", "name": "alpha", "parameters": {"type": "object"}},
]


def ctx(task="task-1", policies=()):
    c = SessionContext(task_id=task)
    for p in policies:
        p.setup(c)
    return c


def run(policy, request, c, protocol="responses"):
    return policy.apply(protocol, copy.deepcopy(request), c)


def tool_turn(i, text):
    return [
        {"type": "function_call", "call_id": f"c{i}", "name": "read_file", "arguments": f'{{"n": {i}}}'},
        {"type": "function_call_output", "call_id": f"c{i}", "output": text},
    ]


def history(n, text="x" * 3000):
    items = [{"role": "user", "content": "go"}]
    for i in range(n):
        items += tool_turn(i, text + str(i))
    return items


# ---------------- cache layout ----------------


def test_cache_layout_canonical_tools_and_cache_settings():
    layout = CacheStableLayout()
    out = run(layout, {"model": "m", "tools": TOOLS, "input": "hi"}, ctx())
    assert [t["name"] for t in out["tools"]] == ["alpha", "zeta"]
    assert list(out["tools"][1]["parameters"]["properties"]) == ["a", "b"]
    assert out["prompt_cache_key"] == "task-1"
    assert out["prompt_cache_retention"] == "24h"


def test_cache_layout_tool_order_no_longer_breaks_prefix():
    layout = CacheStableLayout()
    a = run(layout, {"model": "m", "tools": TOOLS, "input": "hi"}, ctx())
    b = run(layout, {"model": "m", "tools": list(reversed(TOOLS)), "input": "hi"}, ctx())
    assert first_divergence(segment("responses", a), segment("responses", b)) is None


def test_cache_layout_moves_volatile_lines_to_the_end():
    layout = CacheStableLayout()
    instructions = "You are a support agent.\nCurrent time: 2026-10-26T14:03:11Z\nBe concise."
    out = run(layout, {"model": "m", "instructions": instructions, "input": [{"role": "user", "content": "q"}]}, ctx())
    assert out["instructions"] == "You are a support agent.\nBe concise."
    assert out["input"][-1] == {"role": "user", "content": "[Context] Current time: 2026-10-26T14:03:11Z"}

    later = run(layout, {"model": "m", "instructions": instructions.replace("14:03:11", "14:09:40"), "input": [{"role": "user", "content": "q"}]}, ctx())
    # The system prompt is now identical across calls; only the trailing context differs.
    assert later["instructions"] == out["instructions"]


def test_cache_layout_chat_leading_system_only():
    layout = CacheStableLayout()
    req = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "Rules.\nToday is 2026-10-26."},
            {"role": "user", "content": "Meet at 10:30?"},
        ],
    }
    out = run(layout, req, ctx(), protocol="chat_completions")
    assert out["messages"][0]["content"] == "Rules."
    assert out["messages"][1]["content"] == "Meet at 10:30?"  # history untouched
    assert out["messages"][-1]["content"] == "[Context] Today is 2026-10-26."
    assert "prompt_cache_retention" not in out  # Responses-only setting


def test_cache_layout_respects_existing_cache_key_and_rejects_messages_protocol():
    layout = CacheStableLayout()
    assert run(layout, {"model": "m", "input": "x", "prompt_cache_key": "mine"}, ctx())["prompt_cache_key"] == "mine"
    with pytest.raises(UnsupportedProtocol):
        layout.apply("messages", {"model": "m", "messages": []}, ctx())


# ---------------- masking ----------------


def masked(out):
    return [i for i, item in enumerate(out["input"]) if str(item.get("output", "")).startswith(PLACEHOLDER_PREFIX)]


def test_masking_masks_in_batches_and_keeps_recent():
    masker, c = BatchedMasking(keep_last=2, batch=3), ctx()
    # 4 outputs: 2 eligible, below the batch of 3 -> nothing masked yet.
    assert masked(run(masker, {"model": "m", "input": history(4)}, c)) == []
    # 5 outputs: 3 eligible -> watermark jumps to 3.
    out5 = run(masker, {"model": "m", "input": history(5)}, c)
    assert len(masked(out5)) == 3
    # 6 and 7 outputs: still 3 masked (prefix stable between batches).
    out6 = run(masker, {"model": "m", "input": history(6)}, c)
    out7 = run(masker, {"model": "m", "input": history(7)}, c)
    assert len(masked(out6)) == len(masked(out7)) == 3
    assert first_divergence(segment("responses", out5), segment("responses", out6)) is None
    assert first_divergence(segment("responses", out6), segment("responses", out7)) is None
    # 8 outputs: 6 eligible -> next batch.
    assert len(masked(run(masker, {"model": "m", "input": history(8)}, c))) == 6


def test_masking_placeholder_is_deterministic_and_names_tool():
    masker = BatchedMasking(keep_last=0, batch=1)
    a = run(masker, {"model": "m", "input": history(1)}, ctx())
    b = run(masker, {"model": "m", "input": history(1)}, ctx())
    assert a == b
    assert "read_file call c0 returned 3,001 characters" in a["input"][2]["output"]


def test_masking_skips_small_outputs():
    masker = BatchedMasking(keep_last=0, batch=1, min_chars=1000)
    out = run(masker, {"model": "m", "input": history(2, text="short")}, ctx())
    assert masked(out) == []


def test_masking_chat_protocol():
    masker = BatchedMasking(keep_last=0, batch=1)
    req = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "function": {"name": "ls", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "y" * 2000},
        ],
    }
    out = run(masker, req, ctx(), protocol="chat_completions")
    assert out["messages"][2]["content"].startswith(PLACEHOLDER_PREFIX + " ls call t1")


def test_masking_validates_parameters():
    with pytest.raises(ValueError):
        BatchedMasking(batch=0)


# ---------------- output handles ----------------


def test_output_handles_shortens_text_on_line_boundaries_and_stores_full():
    handles = OutputHandles(max_chars=1000, head_chars=300, tail_chars=200)
    c = ctx(policies=[handles])
    text = "\n".join(f"line {i:04d}" for i in range(500))
    out = run(handles, {"model": "m", "input": tool_turn(0, text)}, c)
    short = out["input"][1]["output"]
    assert short.startswith("[shortened: 4,999 characters in full")
    body_lines = short.split("\n")[1:]
    assert body_lines[0] == "line 0000"  # head kept intact
    assert body_lines[-1] == "line 0499"  # tail kept intact
    assert all(line.startswith(("line ", "...")) for line in body_lines)  # no partial lines
    assert c.state_for("output_handles")["store"]["out_c0"] == text


def test_output_handles_json_stays_valid():
    handles = OutputHandles(max_chars=1500, head_chars=500, tail_chars=200)
    data = {"orders": [{"id": i, "note": "n" * 300} for i in range(100)], "status": "ok"}
    out = run(handles, {"model": "m", "input": tool_turn(0, json.dumps(data))}, ctx(policies=[handles]))
    note, body = out["input"][1]["output"].split("\n", 1)
    parsed = json.loads(body)
    assert parsed["status"] == "ok"
    assert parsed["orders"][-1]["_truncated"].endswith("more items")
    assert len(out["input"][1]["output"]) <= 1500


def test_output_handles_leaves_small_outputs_and_always_adds_fetch_tool():
    handles = OutputHandles()
    out = run(handles, {"model": "m", "tools": TOOLS, "input": tool_turn(0, "small")}, ctx(policies=[handles]))
    assert out["input"][1]["output"] == "small"
    assert [t["name"] for t in out["tools"]][-1] == FETCH_TOOL
    chat = run(handles, {"model": "m", "messages": []}, ctx(policies=[handles]), protocol="chat_completions")
    assert chat["tools"][0]["function"]["name"] == FETCH_TOOL


def test_output_handles_fetch_reads_ranges():
    handles = OutputHandles(max_chars=100, head_chars=40, tail_chars=20, max_fetch_chars=50)
    c = ctx(policies=[handles])
    text = "".join(str(i % 10) for i in range(300))
    run(handles, {"model": "m", "input": tool_turn(0, text)}, c)
    fetch = c.harness_tools[FETCH_TOOL].handler
    assert fetch({"handle": "out_c0", "start": 100, "length": 10}) == "[characters 100-110 of 300]\n0123456789"
    assert fetch({"handle": "out_c0", "start": 0, "length": 999}).startswith("[characters 0-50 of 300]")
    assert fetch({"handle": "nope", "start": 0, "length": 1}).startswith("error")


def test_head_tail_returns_short_text_unchanged():
    assert _head_tail("abc", 10, 10) == "abc"


def test_output_handles_rejects_bad_sizes():
    with pytest.raises(ValueError):
        OutputHandles(max_chars=100, head_chars=80, tail_chars=30)


# ---------------- chain ----------------


def test_default_chain_masked_placeholder_points_to_handle():
    chain = default_chain()
    chain[2].keep_last, chain[2].batch = 0, 1
    c = ctx(policies=chain)
    req = {"model": "m", "tools": TOOLS, "input": history(1, text="z" * 10_000)}
    for policy in chain:
        req = policy.apply("responses", req, c)
    placeholder = req["input"][2]["output"]
    assert placeholder.startswith(PLACEHOLDER_PREFIX)
    assert "10,001 characters" in placeholder  # the full length, not the shortened one
    assert "fetch_output(handle='out_c0')" in placeholder
