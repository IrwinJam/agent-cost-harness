from harness.segments import first_divergence, segment

TOOLS = [{"type": "function", "name": "get_order", "parameters": {"type": "object"}}]


def responses_request(items):
    return {"model": "m", "tools": TOOLS, "instructions": "You are a support agent.", "input": items}


TURN_1 = [{"role": "user", "content": "Where is order 42?"}]
TURN_2 = TURN_1 + [
    {"type": "function_call", "call_id": "c1", "name": "get_order", "arguments": '{"id": 42}'},
    {"type": "function_call_output", "call_id": "c1", "output": '{"status": "shipped"}'},
]


def test_responses_segments_in_prefix_order():
    segs = segment("responses", responses_request(TURN_2))
    assert [s.origin for s in segs] == ["tool_schemas", "system", "message:user", "tool_call", "tool_output"]
    assert segs[3].ref == "c1" and segs[4].ref == "c1"
    assert [s.index for s in segs] == list(range(len(segs)))


def test_append_only_history_has_no_divergence():
    a = segment("responses", responses_request(TURN_1))
    b = segment("responses", responses_request(TURN_2))
    assert first_divergence(a, b) is None


def test_rewritten_tool_output_is_detected():
    masked = [dict(item) for item in TURN_2]
    masked[2]["output"] = "[output masked]"
    a = segment("responses", responses_request(TURN_2))
    b = segment("responses", responses_request(masked + [{"role": "user", "content": "thanks"}]))
    assert first_divergence(a, b) == 4


def test_changed_tools_break_at_the_top():
    a = segment("responses", responses_request(TURN_2))
    changed = responses_request(TURN_2)
    changed["tools"] = TOOLS + [{"type": "function", "name": "refund", "parameters": {"type": "object"}}]
    assert first_divergence(a, segment("responses", changed)) == 0


def test_truncated_history_is_a_divergence():
    a = segment("responses", responses_request(TURN_2))
    b = segment("responses", responses_request(TURN_1))
    assert first_divergence(a, b) == len(b)


def test_same_content_different_call_id_diverges():
    def with_id(call_id):
        return responses_request(
            TURN_1 + [{"type": "function_call", "call_id": call_id, "name": "get_order", "arguments": "{}"}]
        )

    a, b = segment("responses", with_id("c1")), segment("responses", with_id("c2"))
    assert a[3].hash == b[3].hash  # content hash ignores the id (used for duplicate-call detection)
    assert first_divergence(a, b) == 3  # but the prompt bytes differ


def test_hash_ignores_key_order():
    a = segment("responses", {"model": "m", "tools": [{"name": "t", "parameters": {"a": 1, "b": 2}}], "input": "x"})
    b = segment("responses", {"model": "m", "tools": [{"parameters": {"b": 2, "a": 1}, "name": "t"}], "input": "x"})
    assert a[0].hash == b[0].hash


def test_chat_segments():
    segs = segment(
        "chat_completions",
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "t1", "function": {"name": "f", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "t1", "content": "result"},
            ],
        },
    )
    assert [s.origin for s in segs] == ["system", "message:user", "tool_call", "tool_output"]
    assert segs[3].ref == "t1"


def test_messages_segments():
    segs = segment(
        "messages",
        {
            "model": "m",
            "system": "sys",
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": "checking"},
                        {"type": "tool_use", "id": "u1", "name": "f", "input": {}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": "ok"}]},
            ],
        },
    )
    assert [s.origin for s in segs] == [
        "system", "message:user", "reasoning", "message:assistant", "tool_call", "tool_output",
    ]
