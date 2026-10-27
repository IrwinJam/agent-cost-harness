"""Read and change the parts of a request the policies care about, in either format.

Works with Responses (`instructions` plus `input`) and Chat Completions (`messages`). The Messages
format isn't supported by the policies yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.usage import Protocol

SUPPORTED: tuple[Protocol, ...] = ("responses", "chat_completions")


class UnsupportedProtocol(ValueError):
    pass


def check(protocol: Protocol) -> None:
    if protocol not in SUPPORTED:
        raise UnsupportedProtocol(f"policies support {SUPPORTED}, not {protocol!r}")


def items(protocol: Protocol, request: dict[str, Any]) -> list[dict[str, Any]]:
    """The conversation list. A plain string `input` becomes one user message. Changes to it stick."""
    check(protocol)
    key = "input" if protocol == "responses" else "messages"
    value = request.get(key)
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if value is None:
        value = []
    request[key] = value
    return value


@dataclass(frozen=True)
class ToolOutput:
    index: int  # position in items()
    call_id: str
    text: str


def tool_outputs(protocol: Protocol, conv: list[dict[str, Any]]) -> list[ToolOutput]:
    out = []
    for i, item in enumerate(conv):
        if protocol == "responses" and item.get("type") == "function_call_output":
            text, call_id = item.get("output"), item.get("call_id")
        elif protocol == "chat_completions" and item.get("role") == "tool":
            text, call_id = item.get("content"), item.get("tool_call_id")
        else:
            continue
        if isinstance(text, str) and call_id:
            out.append(ToolOutput(i, call_id, text))
    return out


def set_output(protocol: Protocol, conv: list[dict[str, Any]], index: int, text: str) -> None:
    conv[index] = {**conv[index], ("output" if protocol == "responses" else "content"): text}


def tool_names(protocol: Protocol, conv: list[dict[str, Any]]) -> dict[str, str]:
    """Map each tool call id to its tool name."""
    names = {}
    for item in conv:
        if protocol == "responses" and item.get("type") == "function_call":
            names[item.get("call_id")] = item.get("name")
        elif protocol == "chat_completions":
            for call in item.get("tool_calls") or []:
                names[call.get("id")] = (call.get("function") or {}).get("name")
    return names


def tool_schema(protocol: Protocol, name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    if protocol == "responses":
        return {"type": "function", "name": name, "description": description, "parameters": parameters}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def tool_schema_name(tool: dict[str, Any]) -> str:
    return tool.get("name") or (tool.get("function") or {}).get("name") or ""


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str


def response_tool_calls(protocol: Protocol, response: dict[str, Any]) -> list[ToolCall]:
    if protocol == "responses":
        return [
            ToolCall(o["call_id"], o["name"], o.get("arguments") or "{}")
            for o in response.get("output") or []
            if o.get("type") == "function_call"
        ]
    message = _chat_message(response)
    return [
        ToolCall(c["id"], c["function"]["name"], c["function"].get("arguments") or "{}")
        for c in message.get("tool_calls") or []
    ]


def response_items(protocol: Protocol, response: dict[str, Any]) -> list[dict[str, Any]]:
    """What a client adds to the conversation after this response."""
    if protocol == "responses":
        return [dict(o) for o in response.get("output") or []]
    message = dict(_chat_message(response))
    return [message] if message else []


def call_items(protocol: Protocol, calls: list[ToolCall]) -> list[dict[str, Any]]:
    """Conversation items for these tool calls, the way the model made them."""
    if protocol == "responses":
        return [{"type": "function_call", "call_id": c.call_id, "name": c.name, "arguments": c.arguments} for c in calls]
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": c.call_id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in calls
            ],
        }
    ]


def output_item(protocol: Protocol, call_id: str, text: str) -> dict[str, Any]:
    if protocol == "responses":
        return {"type": "function_call_output", "call_id": call_id, "output": text}
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def strip_tool_calls(protocol: Protocol, response: dict[str, Any], names: set[str]) -> dict[str, Any]:
    """Copy of the response without the calls to these tools."""
    if protocol == "responses":
        output = [o for o in response.get("output") or [] if not (o.get("type") == "function_call" and o.get("name") in names)]
        return {**response, "output": output}
    choices = []
    for choice in response.get("choices") or []:
        message = dict(choice.get("message") or {})
        calls = [c for c in message.get("tool_calls") or [] if c["function"]["name"] not in names]
        message["tool_calls"] = calls or None
        choices.append({**choice, "message": message})
    return {**response, "choices": choices}


def _chat_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    return (choices[0].get("message") or {}) if choices else {}
