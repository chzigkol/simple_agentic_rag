# 06 - Answer Quality Theory

Answer-quality evaluation checks whether the final answer is useful and relevant to the question.

## Core Idea

Answer quality is more subjective than routing or retrieval. Start with cheap, inspectable proxies before trusting model-graded scores.

## What This Notebook Measures

- Whether local context exists
- A simple extracted answer from retrieved context
- Lexical relevance between question and answer
- Lexical overlap between expected answer and actual answer
- Optional DeepEval answer relevancy score

## LLM-As-Judge Caution

LLM judge scores can be useful, but they need calibration. Before treating them as truth, compare them against trusted human labels or carefully reviewed examples.

## Main Lesson

Do not collapse answer quality into a vague score. Define the specific behavior you care about, use lightweight proxies for iteration, and use LLM judges only when their reliability is understood.
