# 02 - Router Evaluation Theory

Router evaluation checks one simple question: given a user query, does the router choose the correct source?

## Core Idea

Evaluate the smallest useful behavior first. Before testing retrieval or answer generation, isolate the router so later failures are easier to explain.

## What The Router Decides

The router chooses one source:

- `Retrieve_QnA` for general medical questions
- `Retrieve_Device` for medical device/manual questions
- `Web_Search` for recent, external, or out-of-domain questions

## What This Notebook Measures

- Accuracy: how often the predicted source matches the expected source
- Precision, recall, and F1 per route label
- Confusion matrix: which routes are confused with each other
- Failure rows: the exact queries the router got wrong
- Failure modes: grouped patterns in the mistakes

Overall accuracy tells you how often the router is correct across all examples, but it can hide source-specific mistakes. Precision answers: when the router chooses a route, how trustworthy is that choice? Low precision means the route is being overused. Recall answers: of the examples that should use a route, how many did the router catch? Low recall means the route is being missed. F1 summarizes precision and recall when you need one per-route score.

## Why Challenging Examples Matter

Easy examples can make a router look solved. Challenging examples reveal ambiguity, mixed intent, recency-sensitive wording, and source-selection weaknesses.

A drop from base accuracy to challenging accuracy is useful signal, not just bad news.

## Main Lesson

Use task-specific metrics, not vague quality scores. Router evals should be categorical and inspectable: correct or incorrect, with failure tables that show what to improve next.

Optional LLM-router comparisons are useful later, but they add cost, latency, and model variability.

## Next

[03 - Retrieval Evaluation Theory](03_retrieval_evaluation_theory.md)
