"""Evaluation helpers for routing, retrieval, and answer relevancy."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
    dataset_path = Path(path)
    df = pd.read_csv(dataset_path)
    if "Query" in df.columns:
        df = df.rename(
            columns={
                "Query": "query",
                "Expected_Source_Type": "expected_source_type",
            }
        )
    resolver = _ExpectedDocIdResolver.from_dataset_path(dataset_path)
    examples: list[EvaluationExample] = []
    for _, row in df.iterrows():
        csv_doc_ids = parse_doc_ids(row.get("expected_doc_ids", ""))
        examples.append(
            EvaluationExample(
                query=str(row["query"]),
                expected_source_type=SourceType(str(row["expected_source_type"])),
                expected_collection=str(row.get("expected_collection") or ""),
                expected_doc_ids=resolver.resolve(row) or csv_doc_ids,
                expected_answer=str(row.get("expected_answer") or ""),
                category=str(row.get("category") or ""),
            )
        )
    return examples


class _ExpectedDocIdResolver:
    """Resolve eval CSV gold rows to the document IDs used by Chroma."""

    def __init__(
        self,
        *,
        qna_id_by_question_answer: dict[tuple[str, str], str],
        qna_ids_by_question: dict[str, list[str]],
        device_lookup_rows: list[dict[str, str]],
    ) -> None:
        self._qna_id_by_question_answer = qna_id_by_question_answer
        self._qna_ids_by_question = qna_ids_by_question
        self._device_lookup_rows = device_lookup_rows

    @classmethod
    def empty(cls) -> _ExpectedDocIdResolver:
        return cls(
            qna_id_by_question_answer={},
            qna_ids_by_question={},
            device_lookup_rows=[],
        )

    @classmethod
    def from_dataset_path(cls, path: Path) -> _ExpectedDocIdResolver:
        datasets_dir = cls._find_datasets_dir(path)
        if datasets_dir is None:
            return cls.empty()

        qna_path = datasets_dir / "medical_qna_dataset.csv"
        device_path = datasets_dir / "medical_device_manuals_dataset.csv"
        qna_df = _read_csv_or_empty(qna_path)
        device_df = _read_csv_or_empty(device_path)

        return cls(
            qna_id_by_question_answer=cls._build_qna_question_answer_lookup(qna_df),
            qna_ids_by_question=cls._build_qna_question_lookup(qna_df),
            device_lookup_rows=cls._build_device_lookup_rows(device_df),
        )

    @staticmethod
    def _find_datasets_dir(path: Path) -> Path | None:
        candidates = [path.parent, path.parent / "datasets"]
        for candidate in candidates:
            if (
                (candidate / "medical_qna_dataset.csv").exists()
                or (candidate / "medical_device_manuals_dataset.csv").exists()
            ):
                return candidate
        return None

    @staticmethod
    def _build_qna_question_answer_lookup(
        qna_df: pd.DataFrame,
    ) -> dict[tuple[str, str], str]:
        if qna_df.empty or not {"Question", "Answer"}.issubset(qna_df.columns):
            return {}
        return {
            (_normalize_text(row["Question"]), _normalize_text(row["Answer"])): (
                f"qna-{row_index}"
            )
            for row_index, row in qna_df.reset_index(drop=True).iterrows()
        }

    @staticmethod
    def _build_qna_question_lookup(qna_df: pd.DataFrame) -> dict[str, list[str]]:
        if qna_df.empty or "Question" not in qna_df.columns:
            return {}

        ids_by_question: dict[str, list[str]] = {}
        for row_index, row in qna_df.reset_index(drop=True).iterrows():
            ids_by_question.setdefault(_normalize_text(row["Question"]), []).append(
                f"qna-{row_index}"
            )
        return ids_by_question

    @staticmethod
    def _build_device_lookup_rows(device_df: pd.DataFrame) -> list[dict[str, str]]:
        device_answer_columns = [
            "Indications_for_Use",
            "Contraindications",
            "Patient_Population",
        ]
        lookup_rows = []
        for row_index, row in device_df.reset_index(drop=True).iterrows():
            for column in device_answer_columns:
                if column not in row or pd.isna(row[column]):
                    continue
                lookup_rows.append(
                    {
                        "doc_id": f"device-{row_index}",
                        "answer": _normalize_text(row[column]),
                        "device_name": _normalize_text(row.get("Device_Name", "")),
                        "model_number": _normalize_text(row.get("Model_Number", "")),
                    }
                )
        return lookup_rows

    def resolve(self, row: pd.Series) -> list[str]:
        source = _source_from_value(str(row["expected_source_type"]))
        query = _normalize_text(row["query"])
        expected_answer = _normalize_text(row.get("expected_answer", ""))

        if source == SourceType.RETRIEVE_QNA:
            exact_match = self._qna_id_by_question_answer.get((query, expected_answer))
            if exact_match:
                return [exact_match]
            return self._qna_ids_by_question.get(query, [])

        if source == SourceType.RETRIEVE_DEVICE:
            candidates = [
                item
                for item in self._device_lookup_rows
                if item["answer"] == expected_answer
            ]
            model_matches = [
                item
                for item in candidates
                if item["model_number"] and item["model_number"] in query
            ]
            if model_matches:
                return _unique_sorted_doc_ids(model_matches)
            named_matches = [
                item
                for item in candidates
                if item["device_name"] and item["device_name"] in query
            ]
            if named_matches:
                return _unique_sorted_doc_ids(named_matches)
            return _unique_sorted_doc_ids(candidates)

        return []


def _source_from_value(value: str) -> SourceType | None:
    try:
        source = SourceType(value)
    except ValueError:
        return None
    if source in {SourceType.RETRIEVE_QNA, SourceType.RETRIEVE_DEVICE}:
        return source
    return None


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _normalize_text(value: object) -> str:
    if value is None or bool(pd.isna(cast(Any, value))):
        return ""
    return " ".join(str(value).lower().split())


def _unique_sorted_doc_ids(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["doc_id"] for row in rows})


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
