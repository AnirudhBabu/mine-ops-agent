"""Wire the API client, tools, governance and agents into one session."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx2
from fastapi.testclient import TestClient

from .agents import Orchestrator
from .fusion_api import create_app
from .governance import ApprovalQueue, AuditLog, User
from .llm import LLM, OpenAICompatibleLLM, RuleBasedLLM
from .tools import FusionClient, HTTPClient, ToolRuntime, build_tools
from .tracing import setup_tracing


@dataclass
class Session:
    user: User
    orchestrator: Orchestrator
    runtime: ToolRuntime
    approvals: ApprovalQueue
    audit: AuditLog
    http: HTTPClient

    def ask(self, request: str) -> str:
        return self.orchestrator.run(request).answer


def default_llm() -> LLM:
    return OpenAICompatibleLLM() if os.getenv("LLM_MODEL") else RuleBasedLLM()


def build_session(user: User, llm: LLM | None = None, audit_path: Path | None = None) -> Session:
    """Use FUSION_BASE_URL for a running API server, otherwise an in-process mock."""
    setup_tracing()
    base_url = os.getenv("FUSION_BASE_URL")
    http: HTTPClient = httpx2.Client(base_url=base_url, timeout=30) if base_url else TestClient(create_app())
    audit = AuditLog(audit_path)
    approvals = ApprovalQueue(audit)
    runtime = ToolRuntime(build_tools(FusionClient(http)), user, approvals, audit)
    return Session(user, Orchestrator(llm or default_llm(), runtime), runtime, approvals, audit, http)
