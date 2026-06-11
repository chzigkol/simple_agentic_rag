"""Async retrieval clients."""

import asyncio
from typing import Any, cast

import chromadb

from agentic_rag.constants import COLLECTION_BY_SOURCE, SourceType
from agentic_rag.ingestion import embed_texts
from agentic_rag.schemas import RetrievedDocument
from agentic_rag.telemetry import set_span_attributes, start_span


class ChromaRetriever:
    """Async facade over Chroma's synchronous client."""

    def __init__(self, *, chroma_path: str, top_k: int) -> None:
        self._client = chromadb.PersistentClient(path=chroma_path)
        self._top_k = top_k

    async def retrieve(self, source: SourceType, query: str) -> list[RetrievedDocument]:
        collection_name = COLLECTION_BY_SOURCE.get(source)
        if collection_name is None:
            return []

        with start_span(
            "retriever.chroma.query",
            **{
                "retrieval.source": source.value,
                "chroma.collection": collection_name,
                "retrieval.top_k": self._top_k,
                "query.length": len(query),
            },
        ) as span:

            def _query() -> list[RetrievedDocument]:
                collection = self._client.get_collection(collection_name)
                result = collection.query(
                    query_embeddings=embed_texts([query]),
                    n_results=self._top_k,
                )
                ids = cast(list[list[str]], result.get("ids") or [[]])[0]
                docs = cast(list[list[str]], result.get("documents") or [[]])[0]
                metadatas = cast(
                    list[list[dict[str, Any]]], result.get("metadatas") or [[]]
                )[0]
                return [
                    RetrievedDocument(
                        doc_id=str(doc_id),
                        text=str(text),
                        metadata=dict(metadata or {}),
                    )
                    for doc_id, text, metadata in zip(ids, docs, metadatas, strict=True)
                ]

            documents = await asyncio.to_thread(_query)
            set_span_attributes(
                span,
                **{
                    "retrieval.document_count": len(documents),
                    "retrieval.top_doc_id": documents[0].doc_id if documents else None,
                },
            )
            return documents
