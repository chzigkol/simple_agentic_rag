# 03 - Retrieval Evaluation Theory

Retrieval evaluation checks whether the system can find the right evidence from the right collection.

## Core Idea

Evaluate retrieval separately from answer generation. If the answer is bad, first ask whether the right documents were retrieved.

## What This Notebook Measures

- `Precision@k`: how many retrieved documents are relevant
- `Recall@k`: how many expected documents were found
- `Hit@k`: whether at least one expected document appears in the top-k results
- `MRR`: how early the first correct document appears
- Top-k behavior across different `k` values

## Why Gold IDs Matter

Retrieval metrics need document-level gold labels. If a row has no gold document IDs, it should not be averaged into retrieval scores.

Those rows can still be useful for qualitative inspection by looking at the top retrieved documents.

## Main Lesson

Use retrieval-specific metrics, not generic quality scores. Retrieval evals answer a narrow question: when searching the correct source, did we find the right evidence?
