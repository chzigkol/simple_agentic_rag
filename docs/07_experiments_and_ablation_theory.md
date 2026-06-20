# 07 - Experiments And Ablation Theory

Ablation evaluation compares variants to decide what to improve next.

## Core Idea

Evals become most useful when they support engineering decisions. Change one thing, compare results, inspect failures, and decide the next iteration.

## What This Notebook Compares

- Different `top_k` retrieval values
- Expected-source retrieval versus predicted-source retrieval
- Base examples versus challenging examples
- Retrieval quality under different policies
- Failure cases from hard examples

## Why Ablations Matter

A larger `top_k` may improve recall but add more context, cost, and noise. Predicted-source retrieval shows real routed behavior, while expected-source retrieval shows the best-case retrieval control.

## Main Lesson

Do not optimize only for the biggest aggregate score. Compare slices, inspect failures, and choose product changes based on the failure modes that matter most.
