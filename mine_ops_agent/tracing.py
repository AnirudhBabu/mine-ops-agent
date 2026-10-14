"""OpenTelemetry tracing for agent runs, LLM calls, policy decisions and tool calls."""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter

_configured = False


def setup_tracing(exporter: SpanExporter | None = None) -> None:
    """Install a tracer provider once.

    ``MINEOPS_TRACE=console`` prints spans to stdout. Point an OTLP exporter at
    Tempo or Jaeger in the same way by passing it in.
    """
    global _configured
    if _configured:
        if exporter is not None:
            trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "mine-ops-agent"}))
    if exporter is None and os.getenv("MINEOPS_TRACE") == "console":
        exporter = ConsoleSpanExporter()
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _configured = True


tracer = trace.get_tracer("mine_ops_agent")
