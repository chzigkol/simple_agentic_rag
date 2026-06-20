# 06 - Full Agentic RAG Benchmark Theory

The full benchmark evaluates the complete offline path: route, retrieve, produce an answer, and measure latency.

## Core Idea

End-to-end evaluation shows total system behavior. It is useful for summaries, but it should not replace component evals.

## What This Notebook Measures

- Router accuracy
- Retrieved document IDs
- Retrieval hit and recall
- Context relevance
- Answer relevance proxy
- Route, retrieval, and total latency

## Why Stage Columns Matter

Aggregate metrics can hide the cause of a failure. Keeping stage-level columns makes each row traceable back to routing, retrieval, or answering.

## Main Lesson

Use the full benchmark to see overall behavior, but use earlier component notebooks to debug. End-to-end metrics are summaries, not explanations.

## Next

[07 - Experiments And Ablation Theory](07_experiments_and_ablation_theory.md)
