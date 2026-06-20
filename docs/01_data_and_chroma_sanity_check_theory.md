# 01 - Data And Chroma Sanity Check Theory

Before evaluating an agentic RAG system, first verify that the data and retrieval layer are reliable. Metrics are only useful when the inputs, document store, and basic retrieval behavior are sane.

## Core Idea

Start with data, not metrics. A low score might mean the model is bad, but it might also mean the dataset is malformed, key fields are missing, Chroma was not populated, or retrieval is querying the wrong collection.

## What This Notebook Checks

- Source CSV row counts and column counts
- Required schema fields for QnA and device datasets
- Missing values in important columns
- Chroma collection existence and document counts
- Sample stored documents and metadata
- Basic top-k retrieval for each local source

## Why It Matters

Downstream evals depend on this foundation. Router accuracy, retrieval metrics, and answer quality can all be misleading if the underlying data or vector collections are broken.

## Retrieval Smoke Tests

Smoke tests are quick health checks, not final evals. They answer: can the retriever return any plausible documents from the expected collection?

They do not prove retrieval quality. They only catch obvious failures early.

## Main Lesson

Do the boring checks first: schemas, nulls, counts, collection setup, and a few manual record inspections. Only trust automated metrics after the data and retrieval substrate are aligned.

## Next

[02 - Router Evaluation Theory](02_router_evaluation_theory.md)
