from pathlib import Path

import chromadb
import pandas as pd

from agentic_rag.constants import SourceType
from agentic_rag.evaluation import evaluate_examples, summarize_results
from agentic_rag.ingestion import ensure_chroma_collections
from agentic_rag.llm import StaticTextGenerator
from agentic_rag.pipeline import AgenticRAG
from agentic_rag.retrievers import ChromaRetriever
from agentic_rag.router import QueryRouter
from agentic_rag.schemas import EvaluationExample


class FakeWebSearcher:
    async def search(self, query: str) -> str:
        return f"web context for {query}"


async def test_pipeline_runs_end_to_end_with_local_retrieval(
    tmp_path: Path,
) -> None:
    chroma_path = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(chroma_path))
    ensure_chroma_collections(
        client,
        pd.DataFrame(
            [
                {
                    "Question": "What are symptoms of dystonia?",
                    "Answer": "Muscle contractions and repetitive movements.",
                    "qtype": "symptoms",
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "Device_Name": "Dialysis Machine",
                    "Model_Number": "DM-200",
                    "Manufacturer": "Example Medical",
                    "Patient_Population": "Adult",
                    "Indications_for_Use": "Renal replacement therapy.",
                    "Contraindications": "Do not use during severe hypotension.",
                    "Sterilization_Method": "Steam",
                }
            ]
        ),
    )
    pipeline = AgenticRAG(
        router=QueryRouter(),
        retriever=ChromaRetriever(chroma_path=str(chroma_path), top_k=1),
        generator=StaticTextGenerator("grounded answer"),
        web_searcher=FakeWebSearcher(),
    )

    response = await pipeline.answer("What are symptoms of dystonia?")

    assert response.source == SourceType.RETRIEVE_QNA
    assert response.answer == "grounded answer"
    assert response.context[0].doc_id == "qna-0"


async def test_evaluation_summarizes_end_to_end_pipeline_results(
    tmp_path: Path,
) -> None:
    chroma_path = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(chroma_path))
    ensure_chroma_collections(
        client,
        pd.DataFrame(
            [
                {
                    "Question": "What are symptoms of dystonia?",
                    "Answer": "Muscle contractions and repetitive movements.",
                    "qtype": "symptoms",
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "Device_Name": "Dialysis Machine",
                    "Model_Number": "DM-200",
                    "Manufacturer": "Example Medical",
                    "Patient_Population": "Adult",
                    "Indications_for_Use": "Renal replacement therapy.",
                    "Contraindications": "Do not use during severe hypotension.",
                    "Sterilization_Method": "Steam",
                }
            ]
        ),
    )
    pipeline = AgenticRAG(
        router=QueryRouter(),
        retriever=ChromaRetriever(chroma_path=str(chroma_path), top_k=1),
        generator=StaticTextGenerator("grounded answer"),
    )
    examples = [
        EvaluationExample(
            query="What are symptoms of dystonia?",
            expected_source_type=SourceType.RETRIEVE_QNA,
            expected_doc_ids=["qna-0"],
        ),
        EvaluationExample(
            query="What are contraindications for Dialysis Machine model DM-200?",
            expected_source_type=SourceType.RETRIEVE_DEVICE,
            expected_doc_ids=["device-0"],
        ),
    ]

    results = await evaluate_examples(pipeline, examples)
    summary = summarize_results(results)

    assert [result.route_correct for result in results] == [True, True]
    assert [result.hit_at_k for result in results] == [1.0, 1.0]
    assert summary["router_accuracy"] == 1.0
    assert summary["retrieval_hit_at_k"] == 1.0
