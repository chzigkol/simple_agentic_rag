from pathlib import Path

from opentelemetry import trace

from agentic_rag.settings import Settings
from agentic_rag.telemetry import configure_tracing, start_span


def test_tracing_disabled_keeps_default_tracer_provider() -> None:
    settings = Settings()

    configure_tracing(settings)

    assert trace.get_tracer_provider() is not None


def test_start_span_records_attributes() -> None:
    with start_span("test.span", **{"test.attribute": "value"}) as span:
        span.set_attribute("test.extra", 1)

    assert span is not None


def test_file_tracing_writes_span(tmp_path: Path) -> None:
    trace_file = tmp_path / "traces.jsonl"
    settings = Settings(
        OTEL_TRACING_ENABLED=True,
        OTEL_TRACES_EXPORTER="file",
        OTEL_TRACES_FILE=trace_file,
    )

    configure_tracing(settings)
    with start_span("test.file_span"):
        pass

    assert trace_file.exists()
