import pytest

from mine_ops_agent.governance import Decision, User, evaluate
from mine_ops_agent.session import build_session

PLANNER = User("p", "planner")
SUPERVISOR = User("s", "supervisor")
TECH = User("t", "technician")
LINES = {"description": "x", "lines": [{"item_number": "FLT-OIL-240", "quantity": 2}]}
BIG = {"description": "x", "lines": [{"item_number": "LNR-GYR-MNT", "quantity": 1}]}


def test_policy_matrix():
    assert evaluate(TECH, "read", False).decision is Decision.ALLOW
    assert evaluate(TECH, "create_requisition", True).decision is Decision.DENY
    assert evaluate(PLANNER, "create_requisition", True, 170).decision is Decision.NEEDS_APPROVAL
    high = evaluate(PLANNER, "create_requisition", True, 48500)
    assert high.decision is Decision.NEEDS_APPROVAL and high.approver_role == "supervisor"


def test_write_does_not_touch_system_of_record_until_approved():
    s = build_session(PLANNER)
    store = s.http.app.state.store
    out = s.runtime.call("create_purchase_requisition", LINES, agent="test")
    assert out["status"] == "pending_approval"
    assert store.data["purchaseRequisitions"] == []
    result = s.approvals.approve(out["action_id"], PLANNER)
    assert result["Requisition"].startswith("PR-")
    assert len(store.data["purchaseRequisitions"]) == 1


def test_high_value_needs_supervisor():
    s = build_session(PLANNER)
    out = s.runtime.call("create_purchase_requisition", BIG, agent="test")
    with pytest.raises(PermissionError):
        s.approvals.approve(out["action_id"], PLANNER)
    assert s.approvals.approve(out["action_id"], SUPERVISOR)["ApprovedBy"] == "s"


def test_actions_cannot_be_approved_twice_or_after_reject():
    s = build_session(PLANNER)
    a = s.runtime.call("create_purchase_requisition", LINES, agent="test")["action_id"]
    s.approvals.approve(a, PLANNER)
    with pytest.raises(KeyError):
        s.approvals.approve(a, PLANNER)
    b = s.runtime.call("create_purchase_requisition", LINES, agent="test")["action_id"]
    s.approvals.reject(b, PLANNER)
    with pytest.raises(KeyError):
        s.approvals.approve(b, SUPERVISOR)


def test_another_planner_cannot_confirm_someone_elses_action():
    s = build_session(PLANNER)
    a = s.runtime.call("create_work_order", {"asset_id": "HT-101", "description": "x"}, agent="test")["action_id"]
    with pytest.raises(PermissionError):
        s.approvals.approve(a, User("other", "planner"))


def test_denied_call_is_audited():
    s = build_session(TECH)
    assert s.runtime.call("create_work_order", {"asset_id": "HT-101", "description": "x"}, "t")["status"] == "denied"
    assert any(e["event"] == "policy.decision" and e["decision"] == "deny" for e in s.audit.events)
