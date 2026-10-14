"""Model adapters.

``OpenAICompatibleLLM`` speaks the OpenAI chat-completions tool-calling format,
so the same code runs against OCI Generative AI's OpenAI-compatible endpoint,
OpenAI, or a local Ollama server by changing three environment variables.

``RuleBasedLLM`` is a deterministic stand-in so the orchestration, governance
and eval harness run offline and in CI with no API key. It does not reason;
it exists to test the plumbing around the model, not to replace one.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                 for c in self.tool_calls]
        return msg


class LLM(Protocol):
    def complete(self, agent: str, messages: list[dict], tools: list[dict]) -> AssistantTurn: ...


class OpenAICompatibleLLM:
    """Any OpenAI-compatible chat-completions endpoint.

    OCI example::

        LLM_BASE_URL=https://inference.generativeai.<region>.oci.oraclecloud.com/openai/v1
        LLM_API_KEY=<OCI Generative AI API key>
        LLM_MODEL=<model id from your region's catalog>
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or os.environ["LLM_MODEL"]
        self.client = OpenAI(base_url=base_url or os.getenv("LLM_BASE_URL"),
                             api_key=api_key or os.environ["LLM_API_KEY"])

    def complete(self, agent: str, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        # The internal transcript tags tool messages with a name; the wire format does not need it.
        messages = [{k: v for k, v in m.items() if not (m["role"] == "tool" and k == "name")} for m in messages]
        resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                   tools=tools or None, temperature=0)
        msg = resp.choices[0].message
        calls = [ToolCall(c.id, c.function.name, json.loads(c.function.arguments or "{}"))
                 for c in (msg.tool_calls or [])]
        return AssistantTurn(msg.content, calls)


# ---------------------------------------------------------------------------
# Deterministic offline stand-in
# ---------------------------------------------------------------------------

_ASSET_ID = re.compile(r"\b([A-Z]{2}-\d{3})\b")
_TYPES = {"truck": "HAUL_TRUCK", "excavator": "EXCAVATOR", "crusher": "CRUSHER"}


def _has(text: str, *words: str) -> bool:
    return any(re.search(rf"\b{w}", text) for w in words)


class RuleBasedLLM:
    """Keyword-driven planner that emits OpenAI-shaped tool calls."""

    def __init__(self) -> None:
        self._n = 0

    def complete(self, agent: str, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        task = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        done = {}
        for m in messages:
            if m["role"] == "tool":
                done[m["name"]] = json.loads(m["content"])
        allowed = {t["function"]["name"] for t in tools}
        plan = getattr(self, f"_plan_{agent}")(task.lower(), task, done)
        for name, args in plan:
            if name not in done and name in allowed:
                return self._call(name, args)
        return AssistantTurn(self._summarise(agent, done))

    def _call(self, name: str, args: dict) -> AssistantTurn:
        self._n += 1
        return AssistantTurn(None, [ToolCall(f"call_{self._n}", name, args)])

    # Supervisor: route to specialists
    def _plan_supervisor(self, t: str, raw: str, done: dict) -> list[tuple[str, dict]]:
        steps = []
        maint = _has(t, "overdue", "due", "service", "maintenance", "work order", "inspect", "lockout", "procedure",
                     "sop", "safety", "degraded")
        procure = _has(t.replace("work order", ""), "part", "stock", "inventory", "requisition", "order", "buy",
                       "filter", "liner", "spare", "procurement")
        if maint:
            steps.append(("ask_maintenance_agent", {"task": raw}))
        if procure:
            follow = raw
            found = done.get("ask_maintenance_agent", {}).get("asset_ids")
            if found:
                follow = f"{raw}\nAssets identified by maintenance: {', '.join(found)}"
            steps.append(("ask_procurement_agent", {"task": follow}))
        return steps

    def _plan_maintenance(self, t: str, raw: str, done: dict) -> list[tuple[str, dict]]:
        steps = []
        if _has(t, "procedure", "sop", "how", "lockout", "safety", "interval", "policy"):
            steps.append(("search_procedures", {"query": raw}))
        asset_type = next((v for k, v in _TYPES.items() if k in t), None)
        if _has(t, "overdue", "due", "degraded"):
            steps.append(("find_overdue_assets", {"asset_type": asset_type} if asset_type else {}))
        if _has(t, "work order") and _has(t, "create", "raise", "open", "schedule", "draft"):
            targets = _ASSET_ID.findall(raw) or [a["AssetId"] for a in
                                                 done.get("find_overdue_assets", {}).get("overdue", [])]
            for asset_id in targets[:1]:
                steps.append(("create_work_order", {"asset_id": asset_id, "priority": "HIGH",
                                                    "description": "Preventive service per SOP-MNT-014"}))
        return steps

    def _plan_procurement(self, t: str, raw: str, done: dict) -> list[tuple[str, dict]]:
        steps = []
        if _has(t, "policy", "limit", "rule", "procedure"):
            steps.append(("search_procedures", {"query": raw}))
        ids = _ASSET_ID.findall(raw)
        if ids:
            steps.append(("check_parts_for_service", {"asset_ids": sorted(set(ids))}))
        elif _has(t, "stock", "inventory", "on hand", "filter", "liner", "spare"):
            kw = next((k for k in ("oil filter", "air filter", "liner", "hose", "oil") if k in t), "filter")
            steps.append(("search_inventory", {"keyword": kw}))
        if _has(t.replace("work order", ""), "requisition", "order", "buy", "raise"):
            lines = [{"item_number": line["ItemNumber"], "quantity": line["SuggestedOrderQty"]}
                     for line in done.get("check_parts_for_service", {}).get("lines", [])
                     if line["SuggestedOrderQty"] > 0]
            if not lines:
                m = re.search(r"(\d+)\s", t)
                lines = [{"item_number": i["ItemNumber"], "quantity": int(m.group(1)) if m else 1}
                         for i in done.get("search_inventory", {}).get("items", [])[:1]]
            if lines:
                steps.append(("create_purchase_requisition", {
                    "description": "Restock for scheduled preventive service", "priority": "HIGH",
                    "lines": lines}))
        return steps

    def _summarise(self, agent: str, done: dict) -> str:
        if not done:
            return ("I can help with asset maintenance status, maintenance procedures, stock levels and "
                    "requisitions. That request is outside what I can do here.")
        parts = []
        for name, result in done.items():
            if isinstance(result, dict) and "answer" in result:
                parts.append(result["answer"])
            elif result.get("status") == "denied":
                parts.append(f"{name} was blocked: {result['reason']}.")
            elif result.get("status") == "pending_approval":
                cost = f" (estimated {result['estimated_cost_sar']:,.2f} SAR)" if "estimated_cost_sar" in result else ""
                parts.append(f"{name} drafted as {result['action_id']}{cost}, waiting for "
                             f"{result['approver_role']} approval: {result['reason']}.")
            elif name == "find_overdue_assets":
                rows = result["overdue"]
                parts.append("Overdue or degraded: " + ", ".join(
                    f"{r['AssetId']} ({r['Reason']}, {r['HoursSinceService']}h since service)" for r in rows)
                    if rows else "No assets are overdue.")
            elif name == "check_parts_for_service":
                short = [f"{line['ItemNumber']} short by {line['Shortfall']}" for line in result["lines"]
                         if line["Shortfall"]]
                parts.append("All service parts are in stock." if not short else "Shortfalls: " + ", ".join(short))
            elif name == "search_procedures":
                cites = [r["citation"] for r in result["results"]]
                top = result["results"][0]["text"] if result["results"] else "No matching procedure."
                parts.append(f"{top} [Sources: {', '.join(cites)}]")
            elif name == "search_inventory":
                parts.append("; ".join(f"{i['ItemNumber']}: {i['OnHandQuantity']} {i['UOM']} on hand"
                                       for i in result["items"]) or "No matching items.")
            else:
                parts.append(f"{name}: done.")
        return "\n".join(parts)
