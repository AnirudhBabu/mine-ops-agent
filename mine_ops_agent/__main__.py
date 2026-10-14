"""Command line entry point.

    python -m mine_ops_agent ask "Which haul trucks are overdue?" --role technician
    python -m mine_ops_agent demo
"""

from __future__ import annotations

import argparse
import json

from .governance import User
from .session import build_session


def _show_pending(session) -> None:
    for a in session.approvals.pending():
        print(f"  pending {a.action_id}: {a.tool} -> needs {a.approver_role} ({a.reason})")


def cmd_ask(args: argparse.Namespace) -> None:
    session = build_session(User(args.user, args.role))
    print(session.ask(args.request))
    _show_pending(session)
    for a in list(session.approvals.pending()):
        if input(f"Approve {a.action_id} as {args.role}? [y/N] ").strip().lower() == "y":
            try:
                print(json.dumps(session.approvals.approve(a.action_id, session.user), indent=2))
            except PermissionError as e:
                print(f"  not approved: {e}")


def cmd_demo(_: argparse.Namespace) -> None:
    planner = User("fahad.planner", "planner")
    supervisor = User("noura.supervisor", "supervisor")
    session = build_session(planner)

    def step(title: str, request: str) -> None:
        print(f"\n=== {title}\n> {request}\n")
        print(session.ask(request))
        _show_pending(session)

    step("1. Grounded answer with citations", "What is the lockout procedure before servicing a haul truck?")
    step("2. Multi-agent: maintenance finds assets, procurement checks kits and drafts a requisition",
         "Which haul trucks are overdue for service, do we have the parts, and raise a requisition for anything short?")
    action = session.approvals.pending()[0]
    print(f"\nPlanner confirms {action.action_id}:")
    print(json.dumps(session.approvals.approve(action.action_id, planner), indent=2))

    step("3. High-value requisition escalates to a supervisor",
         "Check spares for crusher CR-301 and raise a requisition for anything short.")
    action = session.approvals.pending()[0]
    try:
        session.approvals.approve(action.action_id, planner)
    except PermissionError as e:
        print(f"\nPlanner tries to approve: blocked ({e})")
    result = session.approvals.approve(action.action_id, supervisor)
    print(f"Supervisor approves: {result['Requisition']} {result['DocumentStatus']}, "
          f"{result['TotalAmount']:,.2f} {result['Currency']}")

    tech = build_session(User("omar.tech", "technician"))
    print("\n=== 4. Role-based access: a technician cannot raise requisitions")
    print(tech.ask("Check parts for HT-101 and raise a requisition for anything short."))

    print(f"\nAudit trail: {len(session.audit.events) + len(tech.audit.events)} events recorded.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mine_ops_agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("request")
    ask.add_argument("--role", default="planner", choices=["technician", "planner", "supervisor"])
    ask.add_argument("--user", default="cli.user")
    ask.set_defaults(fn=cmd_ask)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
