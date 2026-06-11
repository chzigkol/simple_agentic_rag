"""OpenTelemetry tracing utilities."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from agentic_rag.settings import Settings

_TRACER_NAME = "agentic_rag"
_TRACING_CONFIGURED = False
_EXPORTER_FILES: list[TextIO] = []


def configure_tracing(settings: Settings) -> None:
    """Configure OpenTelemetry tracing once for the process."""
    global _TRACING_CONFIGURED
    if _TRACING_CONFIGURED or not settings.tracing_enabled:
        return

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.tracing_service_name})
    )
    exporter = _build_exporter(settings)
    if settings.tracing_exporter.lower() in {"otlp", "otlp_proto_grpc"}:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACING_CONFIGURED = True


def get_tracer() -> Tracer:
    """Return the package tracer."""
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def start_span(name: str, **attributes: Any) -> Iterator[Span]:
    """Start a span and safely record exceptions."""
    with get_tracer().start_as_current_span(name) as span:
        _set_attributes(span, attributes)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def set_span_attributes(span: Span, **attributes: Any) -> None:
    """Set non-null span attributes."""
    _set_attributes(span, attributes)


def _set_attributes(span: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def _build_exporter(settings: Settings) -> SpanExporter:
    exporter = settings.tracing_exporter.lower()
    if exporter == "console":
        return ConsoleSpanExporter()
    if exporter == "file":
        trace_file = Path(settings.tracing_file_path)
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        handle = trace_file.open("a", encoding="utf-8")
        _EXPORTER_FILES.append(handle)
        return ConsoleSpanExporter(out=handle)
    if exporter in {"otlp", "otlp_proto_grpc"}:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=settings.otlp_endpoint)
    msg = f"Unsupported OTEL_TRACES_EXPORTER: {settings.tracing_exporter!r}"
    raise ValueError(msg)
