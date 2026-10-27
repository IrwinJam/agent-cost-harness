"""One agent conversation going through the harness.

`Session.call` takes the request the agent would have sent and gives back the response the agent
should see. In between it runs the policies, sends the request, saves the call, and handles any calls
to harness tools like `fetch_output` by running them and asking the model again.

When the model calls a harness tool, that call and its result become part of the prompt, but the agent
never sees them, so the agent's next request won't have them. Leaving them out would change the start
of the prompt and break the cache, and the model would lose what it already read. So the session
remembers where they go and puts them back into every later request. If the agent rewrites its history
before that spot, they get dropped.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from harness.adapters.base import Adapter
from harness.budget import Budget, SpendLimit
from harness.policies import conv
from harness.policies.base import Policy, SessionContext
from harness.pricing import PriceTable
from harness.trace import TraceStore
from harness.usage import Protocol

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Injection:
    anchor: int  # goes before this agent item. Equal to the list length means at the end
    prefix_hash: str  # hash of the agent items before that spot, to tell if the agent rewrote them
    items: tuple[dict[str, Any], ...]


def _prefix_hash(items: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


class Session:
    def __init__(
        self,
        adapter: Adapter,
        *,
        task_id: str,
        policies: list[Policy] | None = None,
        store: TraceStore | None = None,
        run_id: str | None = None,
        prices: PriceTable | None = None,
        max_harness_rounds: int = 3,
        on: dt.date | None = None,
        budget: Budget | None = None,
        limit: SpendLimit | None = None,
        model: str | None = None,
    ):
        if store is not None and run_id is None:
            raise ValueError("run_id is required when tracing")
        if limit is not None and limit.is_set and store is None:
            raise ValueError("a SpendLimit needs a trace store, since that's where spending is read from")
        self.adapter = adapter
        self.ctx = SessionContext(task_id=task_id)
        self.policies = list(policies or [])
        self.store = store
        self.run_id = run_id
        self.prices = prices or PriceTable.load()
        self.max_harness_rounds = max_harness_rounds
        self.on = on
        self.budget = budget
        self.limit = limit
        self.model = model  # the agent's model, for benchmarks that have to tell another program what to call
        self._injections: list[_Injection] = []
        for policy in self.policies:
            policy.setup(self.ctx)

    def call(self, protocol: Protocol, agent_request: dict[str, Any]) -> dict[str, Any]:
        conv.check(protocol)
        agent_request = copy.deepcopy(agent_request)
        agent_items = list(conv.items(protocol, agent_request))
        harness_names = set(self.ctx.harness_tools)

        for _ in range(self.max_harness_rounds + 1):
            response = self._send(protocol, agent_request, agent_items)
            calls = conv.response_tool_calls(protocol, response)
            mine = [c for c in calls if c.name in harness_names]
            if not mine:
                return response

            results = [conv.output_item(protocol, c.call_id, self._run_tool(c)) for c in mine]
            others = [c for c in calls if c.name not in harness_names]
            if others:
                # The model called our tool and the agent's tools. Give the agent its calls back without ours,
                # and keep ours and their results in the prompt from now on.
                self._inject(agent_items, conv.call_items(protocol, mine) + results)
                return conv.strip_tool_calls(protocol, response, harness_names)
            # Only our tools got called. Keep the model's turn plus our results and ask again.
            self._inject(agent_items, conv.response_items(protocol, response) + results)

        log.warning("harness tool round limit (%d) reached; returning response without harness calls", self.max_harness_rounds)
        return conv.strip_tool_calls(protocol, response, harness_names)

    def _send(self, protocol: Protocol, agent_request: dict[str, Any], agent_items: list[dict[str, Any]]) -> dict[str, Any]:
        request = copy.deepcopy(agent_request)
        conv.items(protocol, request)[:] = self._with_injections(agent_items)
        for policy in self.policies:
            request = policy.apply(protocol, request, self.ctx)
        response = self._send_one(protocol, request)
        # A policy can throw a reply away and ask again once, like asking with more reasoning.
        for policy in self.policies:
            retry = policy.review(protocol, request, response, self.ctx)
            if retry is not None:
                return self._send_one(protocol, retry)
        return response

    def _send_one(self, protocol: Protocol, request: dict[str, Any]) -> dict[str, Any]:
        if self.budget is not None:
            self.budget.check()
        if self.limit is not None and self.limit.is_set:
            self.limit.check(self.store, self.run_id)
        result = self.adapter.call(protocol, request)
        if self.store is not None or self.budget is not None:
            cost = self.prices.cost(result.provider, result.model, result.usage, self.on)
            if self.store is not None:
                self.store.record(self.run_id, result, cost)
            if self.budget is not None:
                self.budget.charge(cost.total)
        return result.response

    def _with_injections(self, agent_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        valid = []
        for inj in self._injections:
            if inj.anchor <= len(agent_items) and _prefix_hash(agent_items[: inj.anchor]) == inj.prefix_hash:
                valid.append(inj)
            else:
                log.info("dropping harness injection at %d: agent history changed before it", inj.anchor)
        self._injections = valid
        out: list[dict[str, Any]] = []
        by_anchor: dict[int, list[dict[str, Any]]] = {}
        for inj in valid:
            by_anchor.setdefault(inj.anchor, []).extend(copy.deepcopy(list(inj.items)))
        for i, item in enumerate(agent_items):
            out.extend(by_anchor.get(i, []))
            out.append(copy.deepcopy(item))
        out.extend(by_anchor.get(len(agent_items), []))
        return out

    def _inject(self, agent_items: list[dict[str, Any]], items: list[dict[str, Any]]) -> None:
        anchor = len(agent_items)
        self._injections.append(_Injection(anchor, _prefix_hash(agent_items), tuple(copy.deepcopy(items))))

    def _run_tool(self, call: conv.ToolCall) -> str:
        tool = self.ctx.harness_tools[call.name]
        try:
            args = json.loads(call.arguments or "{}")
        except ValueError:
            return f"error: arguments are not valid JSON: {call.arguments!r}"
        try:
            return tool.handler(args)
        except Exception as exc:  # a broken harness tool shouldn't crash the agent's run
            log.exception("harness tool %s failed", call.name)
            return f"error: {type(exc).__name__}: {exc}"
