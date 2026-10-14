"""Evaluation harness.

Scores each case on four checks and writes evals/report.json:

* tool_selection   expected tools were called, forbidden tools were not
* tool_arguments   key arguments match (e.g. the asset ids handed from maintenance to procurement)
* policy           write tools ended in the expected outcome (denied / pending_approval) with the right approver,
                   and nothing was written to the system of record before a human approved it
* answer           the final answer contains required facts or citations and excludes wrong ones

Offline it uses the deterministic RuleBasedLLM, which checks the orchestration and governance code.
Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL to score a real model (OCI Generative AI, OpenAI, Ollama).

    python -m evals.run_evals
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mine_ops_agent.governance import User
from mine_ops_agent.session import build_session, default_llm

CASES = Path(__file__).with_name("cases.json")
REPORT = Path(__file__).with_name("report.json")


def _system_of_record_counts(session) -> tuple[int, int] | None:
    app = getattr(session.http, "app", None)
    if app is None:
        return None
    data = app.state.store.data
    return len(data["workOrders"]), len(data["purchaseRequisitions"])


def run_case(case: dict) -> dict:
    session = build_session(User(f"eval.{case['role']}", case["role"]), llm=default_llm())
    before = _system_of_record_counts(session)
    answer = session.ask(case["request"])
    after = _system_of_record_counts(session)
    calls = session.runtime.calls
    called = [c["tool"] for c in calls]
    outcomes = {c["tool"]: c["outcome"] for c in calls}
    failures: list[str] = []
    checks: dict[str, bool] = {}

    missing = [t for t in case.get("expect_tools", []) if t not in called]
    forbidden = [t for t in case.get("forbid_tools", []) if t in called]
    failures += [f"missing tool {t}" for t in missing] + [f"forbidden tool {t}" for t in forbidden]
    checks["tool_selection"] = not missing and not forbidden

    arg_ok = True
    for tool, expected in case.get("expect_tool_args", {}).items():
        actual = next((c["args"] for c in calls if c["tool"] == tool), None)
        for key, value in expected.items():
            got = actual.get(key) if actual else None
            if (sorted(got) if isinstance(got, list) else got) != (sorted(value) if isinstance(value, list) else value):
                arg_ok = False
                failures.append(f"{tool}.{key} = {got!r}, expected {value!r}")
    checks["tool_arguments"] = arg_ok

    policy_ok = True
    for tool, expected in case.get("expect_outcomes", {}).items():
        if outcomes.get(tool) != expected:
            policy_ok = False
            failures.append(f"{tool} outcome {outcomes.get(tool)!r}, expected {expected!r}")
    if "expect_approver" in case:
        roles = [a.approver_role for a in session.approvals.pending()]
        if case["expect_approver"] not in roles:
            policy_ok = False
            failures.append(f"approver roles {roles}, expected {case['expect_approver']}")
    if case.get("expect_no_writes_before_approval") and before is not None and before != after:
        policy_ok = False
        failures.append(f"system of record changed before approval: {before} -> {after}")
    checks["policy"] = policy_ok

    lower = answer.lower()
    missing_text = [s for s in case.get("expect_answer_contains", []) if s.lower() not in lower]
    wrong_text = [s for s in case.get("expect_answer_excludes", []) if s.lower() in lower]
    failures += [f"answer missing {s!r}" for s in missing_text] + [f"answer contains {s!r}" for s in wrong_text]
    checks["answer"] = not missing_text and not wrong_text

    return {"id": case["id"], "passed": not failures, "checks": checks, "failures": failures,
            "tools_called": called, "answer": answer}


def main() -> int:
    cases = json.loads(CASES.read_text())
    results = [run_case(c) for c in cases]
    dims = ["tool_selection", "tool_arguments", "policy", "answer"]
    summary = {d: round(sum(r["checks"][d] for r in results) / len(results), 3) for d in dims}
    summary["cases_passed"] = f"{sum(r['passed'] for r in results)}/{len(results)}"
    summary["model"] = type(default_llm()).__name__
    REPORT.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print(f"{'case':34} {'tools':>6} {'args':>6} {'policy':>7} {'answer':>7}")
    for r in results:
        c = r["checks"]
        marks = ["ok" if c[d] else "FAIL" for d in dims]
        print(f"{r['id']:34} {marks[0]:>6} {marks[1]:>6} {marks[2]:>7} {marks[3]:>7}")
        for f in r["failures"]:
            print(f"    - {f}")
    print("\n" + json.dumps(summary, indent=2))
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
