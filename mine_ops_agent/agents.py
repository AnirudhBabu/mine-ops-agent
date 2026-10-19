"""Multi-agent orchestration: a supervisor that delegates to two specialists.

* supervisor  routes the request and combines the answers. Its only tools are
  the two specialists (agents as tools).
* maintenance owns asset health, work orders and maintenance procedures.
* procurement owns stock, service-kit availability and requisitions.

Each specialist sees only its own tools, so a maintenance prompt can never
reach the requisition endpoint directly. Every tool call still passes through
the governed ``ToolRuntime``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLM, AssistantTurn
from .tools import ToolRuntime, to_message_content
from .tracing import tracer

MAX_STEPS = 8

SPECIALISTS = {
    "maintenance": {
        "tools": ["find_overdue_assets", "get_asset", "list_work_orders", "search_procedures",
                  "create_work_order"],
        "prompt": ("You are the maintenance specialist for an open-pit mine. Use tools for every fact about "
                   "assets and work orders; never estimate hours or status yourself. Cite SOP passages from "
                   "search_procedures as [file#section]. Write actions are only drafted: report the action id "
                   "and who must approve it. If a tool is denied, explain why and stop."),
    },
    "procurement": {
        "tools": ["search_inventory", "check_parts_for_service", "search_procedures",
                  "create_purchase_requisition"],
        "prompt": ("You are the procurement specialist for mine maintenance stores. Use check_parts_for_service "
                   "for kit availability and its SuggestedOrderQty for order sizes. Requisitions are only drafted "
                   "and follow POL-PRC-002 approval limits; report the action id, estimated cost and approver. "
                   "If a tool is denied, explain why and stop."),
    },
}

SUPERVISOR_PROMPT = ("You coordinate maintenance and procurement specialists. Delegate each part of the request "
                     "to the right specialist, pass asset ids from maintenance to procurement, then give one "
                     "concise answer. Decline requests unrelated to mine maintenance or procurement. Never claim "
                     "an action was executed when it is pending approval.")

DELEGATE_TOOLS = [
    {"type": "function", "function": {
        "name": f"ask_{name}_agent",
        "description": f"Delegate a task to the {name} specialist and get its answer.",
        "parameters": {"type": "object", "properties": {"task": {"type": "string"}},
                       "required": ["task"], "additionalProperties": False}}}
    for name in SPECIALISTS
]


@dataclass
class RunResult:
    answer: str
    steps: int
    transcript: list[dict] = field(default_factory=list)


def _loop(agent: str, system: str, task: str, llm: LLM, tools: list[dict], handle) -> RunResult:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    with tracer.start_as_current_span(f"agent.{agent}") as span:
        span.set_attribute("agent.name", agent)
        for step in range(1, MAX_STEPS + 1):
            with tracer.start_as_current_span("llm.call") as llm_span:
                llm_span.set_attribute("agent.name", agent)
                turn: AssistantTurn = llm.complete(agent, messages, tools)
                llm_span.set_attribute("llm.tool_calls", len(turn.tool_calls))
            messages.append(turn.to_message())
            if not turn.tool_calls:
                span.set_attribute("agent.steps", step)
                return RunResult(turn.content or "", step, messages)
            for call in turn.tool_calls:
                result = handle(call.name, call.arguments)
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                 "content": to_message_content(result)})
        span.set_attribute("agent.step_limit_hit", True)
        return RunResult("Stopped: step limit reached without a final answer.", MAX_STEPS, messages)


class Orchestrator:
    def __init__(self, llm: LLM, runtime: ToolRuntime) -> None:
        self.llm, self.runtime = llm, runtime

    def run_specialist(self, name: str, task: str) -> dict:
        spec = SPECIALISTS[name]
        result = _loop(name, spec["prompt"], task, self.llm, self.runtime.schemas(spec["tools"]),
                       lambda tool, args: self.runtime.call(tool, args, agent=name)
                       if tool in spec["tools"] else {"error": f"{tool} is not available to {name}"})
        asset_ids = []
        for m in result.transcript:
            if m["role"] == "tool" and m["name"] == "find_overdue_assets":
                asset_ids = [r["AssetId"] for r in json.loads(m["content"]).get("overdue", [])]
        return {"answer": result.answer, "asset_ids": asset_ids}

    def run(self, request: str) -> RunResult:
        def delegate(tool: str, args: dict) -> dict:
            name = tool.removeprefix("ask_").removesuffix("_agent")
            if name not in SPECIALISTS:
                return {"error": f"Unknown specialist {name}"}
            return self.run_specialist(name, args.get("task", request))

        return _loop("supervisor", SUPERVISOR_PROMPT, request, self.llm, DELEGATE_TOOLS, delegate)
