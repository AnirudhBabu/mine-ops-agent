"""Governance layer: role-based tool access, approval gates and an audit trail.

Policy is enforced in code around every tool call, never by the prompt. A
model that is talked into "skipping approval" still cannot write anything,
because the write tools only run through ``ApprovalQueue.approve``.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

REQUISITION_LIMIT_SAR = 5000.0  # POL-PRC-002

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "technician": {"read"},
    "planner": {"read", "create_work_order", "create_requisition"},
    "supervisor": {"read", "create_work_order", "create_requisition", "approve_high_value"},
}


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"


@dataclass
class PolicyResult:
    decision: Decision
    reason: str
    approver_role: str | None = None


@dataclass
class User:
    name: str
    role: str

    def __post_init__(self) -> None:
        if self.role not in ROLE_PERMISSIONS:
            raise ValueError(f"Unknown role {self.role!r}")

    def can(self, permission: str) -> bool:
        return permission in ROLE_PERMISSIONS[self.role]


def evaluate(user: User, permission: str, is_write: bool, estimated_cost: float | None = None) -> PolicyResult:
    if not user.can(permission):
        return PolicyResult(Decision.DENY, f"Role '{user.role}' lacks permission '{permission}'")
    if not is_write:
        return PolicyResult(Decision.ALLOW, "Read-only tool")
    if estimated_cost is not None and estimated_cost > REQUISITION_LIMIT_SAR:
        return PolicyResult(Decision.NEEDS_APPROVAL,
                            f"Estimated {estimated_cost:,.2f} SAR exceeds the {REQUISITION_LIMIT_SAR:,.0f} SAR "
                            "limit (POL-PRC-002); supervisor approval required",
                            approver_role="supervisor")
    return PolicyResult(Decision.NEEDS_APPROVAL, "Write actions require human confirmation before execution",
                        approver_role=user.role)


@dataclass
class PendingAction:
    action_id: str
    tool: str
    args: dict[str, Any]
    requested_by: User
    reason: str
    approver_role: str
    execute: Callable[[dict[str, Any], str | None], Any] = field(repr=False)
    status: str = "pending"
    result: Any = None


class ApprovalQueue:
    """Holds write actions until a human with the right role approves them."""

    def __init__(self, audit: "AuditLog") -> None:
        self.audit = audit
        self.actions: dict[str, PendingAction] = {}

    def submit(self, action: PendingAction) -> PendingAction:
        self.actions[action.action_id] = action
        self.audit.record("approval.requested", action.requested_by, tool=action.tool, args=action.args,
                          action_id=action.action_id, approver_role=action.approver_role, reason=action.reason)
        return action

    def pending(self) -> list[PendingAction]:
        return [a for a in self.actions.values() if a.status == "pending"]

    def approve(self, action_id: str, approver: User) -> Any:
        action = self._get_pending(action_id)
        if action.approver_role == "supervisor" and not approver.can("approve_high_value"):
            self.audit.record("approval.rejected", approver, action_id=action_id,
                              reason="approver lacks approve_high_value")
            raise PermissionError(f"{approver.role} cannot approve {action_id}; supervisor required")
        if action.approver_role != "supervisor" and approver.name != action.requested_by.name \
                and not approver.can("approve_high_value"):
            raise PermissionError("Only the requester or a supervisor can confirm this action")
        action.result = action.execute(action.args, approver.name)
        action.status = "approved"
        self.audit.record("approval.granted", approver, action_id=action_id, tool=action.tool, result=action.result)
        return action.result

    def reject(self, action_id: str, approver: User) -> None:
        action = self._get_pending(action_id)
        action.status = "rejected"
        self.audit.record("approval.rejected", approver, action_id=action_id, tool=action.tool)

    def _get_pending(self, action_id: str) -> PendingAction:
        action = self.actions.get(action_id)
        if action is None or action.status != "pending":
            raise KeyError(f"No pending action {action_id}")
        return action


class AuditLog:
    """Append-only event log. Optionally mirrored to a JSONL file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []

    def record(self, event: str, user: User, **details: Any) -> None:
        entry = {"ts": time.time(), "event": event, "user": user.name, "role": user.role, **details}
        self.events.append(entry)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(entry, default=str) + "\n")


def new_action_id() -> str:
    return f"ACT-{uuid.uuid4().hex[:8].upper()}"
