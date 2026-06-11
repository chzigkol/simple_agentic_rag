from agentic_rag.constants import SourceType
from agentic_rag.llm import StaticTextGenerator
from agentic_rag.pipeline import AgenticRAG
from agentic_rag.router import QueryRouter
from agentic_rag.schemas import RetrievedDocument


class StubRetriever:
    async def retrieve(
        self, source: SourceType, query: str
    ) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                doc_id="1",
                text=f"context for {query}",
                metadata={"source": source.value},
            )
        ]


async def test_pipeline_answers_with_retrieved_context() -> None:
    pipeline = AgenticRAG(
        router=QueryRouter(),
        retriever=StubRetriever(),  # type: ignore[arg-type]
        generator=StaticTextGenerator("answer"),
    )

    response = await pipeline.answer("What are the symptoms of flu?")

    assert response.source == SourceType.RETRIEVE_QNA
    assert response.answer == "answer"
    assert response.context[0].doc_id == "1"
