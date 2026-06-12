# Agentic RAG

Production-oriented Python package for routing medical questions across:

- medical Q&A retrieval
- medical device manual retrieval
- web search fallback

## Common Commands

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy
uv run agentic-rag-evaluate --dataset datasets/evaluation_dataset.csv
```

Notebook examples should live in `notebooks/` and import from the package.

## Tracing

OpenTelemetry tracing is disabled by default. Enable local console spans with:

```bash
OTEL_TRACING_ENABLED=true \
OTEL_TRACES_EXPORTER=console \
uv run agentic-rag-evaluate --dataset datasets/evaluation_dataset.csv --max-examples 1
```

Write traces to a local file:

```bash
OTEL_TRACING_ENABLED=true \
OTEL_TRACES_EXPORTER=file \
OTEL_TRACES_FILE=otel_traces/agentic_rag_traces.jsonl \
uv run agentic-rag-evaluate --dataset datasets/evaluation_dataset.csv --max-examples 1
```

To export to an OTLP collector:

```bash
OTEL_TRACING_ENABLED=true \
OTEL_TRACES_EXPORTER=otlp \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
uv run agentic-rag-evaluate --dataset datasets/evaluation_dataset.csv --max-examples 1
```
