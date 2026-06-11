"""End-to-end agentic RAG pipeline."""

from agentic_rag.constants import SourceType
from agentic_rag.llm import AsyncTextGenerator
from agentic_rag.retrievers import ChromaRetriever
from agentic_rag.router import QueryRouter
from agentic_rag.schemas import RAGResponse, RetrievedDocument
from agentic_rag.telemetry import set_span_attributes, start_span
from agentic_rag.web_search import AsyncWebSearcher


class AgenticRAG:
    """Route, retrieve, and answer user queries."""

    def __init__(
        self,
        *,
        router: QueryRouter,
        retriever: ChromaRetriever,
        generator: AsyncTextGenerator,
        web_searcher: AsyncWebSearcher | None = None,
    ) -> None:
        self.router = router
        self.retriever = retriever
        self.generator = generator
        self.web_searcher = web_searcher

    async def answer(self, query: str) -> RAGResponse:
        with start_span("agentic_rag.answer", **{"query.length": len(query)}) as span:
            source = await self.router.route(query)
            context = await self._get_context(source, query)
            prompt = self._build_prompt(query, context)
            answer = await self.generator.generate(prompt)
            set_span_attributes(
                span,
                **{
                    "rag.source": source.value,
                    "rag.context.document_count": len(context),
                    "prompt.length": len(prompt),
                    "answer.length": len(answer),
                },
            )
            return RAGResponse(
                query=query,
                source=source,
                answer=answer,
                context=context,
            )

    async def _get_context(
        self, source: SourceType, query: str
    ) -> list[RetrievedDocument]:
        with start_span(
            "agentic_rag.get_context",
            **{"rag.source": source.value, "query.length": len(query)},
        ) as span:
            if source == SourceType.WEB_SEARCH:
                if self.web_searcher is None:
                    set_span_attributes(span, **{"rag.context.document_count": 0})
                    return []
                web_context = await self.web_searcher.search(query)
                if not web_context:
                    set_span_attributes(span, **{"rag.context.document_count": 0})
                    return []
                set_span_attributes(span, **{"rag.context.document_count": 1})
                return [
                    RetrievedDocument(doc_id="web:0", text=web_context, metadata={})
                ]

            documents = await self.retriever.retrieve(source, query)
            set_span_attributes(
                span, **{"rag.context.document_count": len(documents)}
            )
            return documents

    @staticmethod
    def _build_prompt(query: str, context: list[RetrievedDocument]) -> str:
        context_text = "\n".join(document.text for document in context)
        return f"""
Answer the following question using the context below.
Context:
{context_text}
Question: {query}
Please limit your answer to 50 words.
"""
