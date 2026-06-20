"""Async web search integration."""

import asyncio
from typing import Protocol

from agentic_rag.telemetry import set_span_attributes, start_span


class AsyncWebSearcher(Protocol):
    """Protocol for async web search."""

    async def search(self, query: str) -> str:
        """Return web context for a query."""


class TavilyWebSearcher:
    """Async facade over langchain-tavily."""

    def __init__(
        self, *, max_results: int = 1, tavily_api_key: str | None = None
    ) -> None:
        self.max_results = max_results
        self.tavily_api_key = tavily_api_key

    async def search(self, query: str) -> str:
        with start_span(
            "web_search.tavily",
            **{"query.length": len(query), "web_search.max_results": self.max_results},
        ) as span:

            def _search() -> str:
                from langchain_tavily import TavilySearch

                tavily = TavilySearch(
                    topic="general",
                    max_results=self.max_results,
                    tavily_api_key=self.tavily_api_key,
                )
                result = tavily.invoke({"query": query})
                results = result.get("results", [])
                if not results:
                    return ""
                return str(results[0].get("content", ""))

            content = await asyncio.to_thread(_search)
            set_span_attributes(span, **{"web_search.content.length": len(content)})
            return content
