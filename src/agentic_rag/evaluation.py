"""Evaluation helpers for routing, retrieval, and answer relevancy."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from agentic_rag.constants import SourceType
from agentic_rag.pipeline import AgenticRAG
from agentic_rag.schemas import EvaluationExample, EvaluationResult
from agentic_rag.telemetry import set_span_attributes, start_span


@dataclass(frozen=True)
class RetrievalMetrics:
    """Precision/recall style retrieval metrics."""

    precision_at_k: float
    recall_at_k: float
    hit_at_k: float
    mrr: float


def parse_doc_ids(value: object) -> list[str]:
    """Parse pipe-delimited doc ids from CSV values."""
    if value is None:
        return []
    normalized = str(value).strip()
    if normalized == "" or normalized.lower() == "nan":
        return []
    ids: list[str] = []
    for part in normalized.split("|"):
        text = part.strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        if text:
            ids.append(text)
    return ids


def score_retrieval(
    retrieved_doc_ids: list[str], expected_doc_ids: list[str]
) -> RetrievalMetrics:
    """Compute retrieval metrics for one query."""
    if not expected_doc_ids:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0)
    expected = set(expected_doc_ids)
    true_positives = len(set(retrieved_doc_ids) & expected)
    precision = true_positives / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
    recall = true_positives / len(expected)
    hit = 1.0 if true_positives else 0.0
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            mrr = 1.0 / rank
            break
    return RetrievalMetrics(precision, recall, hit, mrr)


def load_examples(path: str) -> list[EvaluationExample]:
    """Load evaluation examples from CSV."""
    df = pd.read_csv(path)
    if "Query" in df.columns:
        df = df.rename(
            columns={
                "Query": "query",
                "Expected_Source_Type": "expected_source_type",
            }
        )
    examples: list[EvaluationExample] = []
    for _, row in df.iterrows():
        examples.append(
            EvaluationExample(
                query=str(row["query"]),
                expected_source_type=SourceType(str(row["expected_source_type"])),
                expected_collection=str(row.get("expected_collection") or ""),
                expected_doc_ids=parse_doc_ids(row.get("expected_doc_ids", "")),
                expected_answer=str(row.get("expected_answer") or ""),
                category=str(row.get("category") or ""),
            )
        )
    return examples


async def evaluate_examples(
    pipeline: AgenticRAG,
    examples: list[EvaluationExample],
) -> list[EvaluationResult]:
    """Evaluate examples concurrently where useful."""
    with start_span(
        "evaluation.evaluate_examples",
        **{"evaluation.example_count": len(examples)},
    ) as span:
        results = await asyncio.gather(
            *(evaluate_one(pipeline, example) for example in examples)
        )
        set_span_attributes(
            span,
            **{
                "evaluation.route_accuracy": _mean(
                    [1.0 if result.route_correct else 0.0 for result in results]
                )
            },
        )
        return results


async def evaluate_one(
    pipeline: AgenticRAG, example: EvaluationExample
) -> EvaluationResult:
    """Evaluate one example."""
    with start_span(
        "evaluation.evaluate_one",
        **{
            "query.length": len(example.query),
            "evaluation.expected_source": example.expected_source_type.value,
            "evaluation.has_expected_doc_ids": bool(example.expected_doc_ids),
        },
    ) as span:
        response = await pipeline.answer(example.query)
        retrieved_ids = [document.doc_id for document in response.context]
        retrieval = score_retrieval(retrieved_ids, example.expected_doc_ids)
        route_correct = example.expected_source_type == response.source
        set_span_attributes(
            span,
            **{
                "evaluation.predicted_source": response.source.value,
                "evaluation.route_correct": route_correct,
                "retrieval.precision_at_k": retrieval.precision_at_k,
                "retrieval.recall_at_k": retrieval.recall_at_k,
                "retrieval.hit_at_k": retrieval.hit_at_k,
                "retrieval.mrr": retrieval.mrr,
            },
        )
        return EvaluationResult(
            query=example.query,
            expected_source_type=example.expected_source_type,
            predicted_source_type=response.source,
            route_correct=route_correct,
            retrieval_source=response.source,
            expected_doc_ids=example.expected_doc_ids,
            retrieved_doc_ids=retrieved_ids,
            precision_at_k=retrieval.precision_at_k,
            recall_at_k=retrieval.recall_at_k,
            hit_at_k=retrieval.hit_at_k,
            mrr=retrieval.mrr,
            actual_answer=response.answer,
        )


def summarize_results(results: list[EvaluationResult]) -> dict[str, Any]:
    """Build summary metrics."""
    labels = [source.value for source in SourceType]
    expected = [result.expected_source_type.value for result in results]
    predicted = [result.predicted_source_type.value for result in results]
    retrieval_results = [result for result in results if result.expected_doc_ids]
    return {
        "num_examples": len(results),
        "router_accuracy": accuracy_score(expected, predicted) if results else None,
        "retrieval_precision_at_k": _mean(
            [result.precision_at_k for result in retrieval_results]
        ),
        "retrieval_recall_at_k": _mean(
            [result.recall_at_k for result in retrieval_results]
        ),
        "retrieval_hit_at_k": _mean([result.hit_at_k for result in retrieval_results]),
        "retrieval_mrr": _mean([result.mrr for result in retrieval_results]),
        "classification_report": classification_report(
            expected,
            predicted,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            expected,
            predicted,
            labels=labels,
        ).tolist(),
        "confusion_matrix_labels": labels,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
