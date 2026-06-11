# 05 - Relevance And Web Fallback Theory

Fallback evaluation checks whether the system should trust local retrieved context or switch to web search.

## Core Idea

After retrieval, the system needs a policy decision: is the local context good enough? If not, fallback can recover from bad or missing context.

## What This Notebook Measures

- Context relevance
- Fallback rate
- Final source after fallback
- Local context relevance rate
- Web final rate

## Relevance Proxies

For examples with gold document IDs, a retrieved gold document is used as a relevance proxy.

For examples without gold IDs, lexical overlap gives a lightweight relevance signal. This is useful, but it is not a perfect human judgment.

## Main Lesson

Fallback is a guardrail-like policy. Too little fallback preserves bad context. Too much fallback increases cost, latency, and unnecessary web usage.
