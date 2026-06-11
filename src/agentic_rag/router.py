"""Query routing."""

import re

from agentic_rag.constants import SourceType
from agentic_rag.llm import AsyncTextGenerator
from agentic_rag.telemetry import set_span_attributes, start_span


class QueryRouter:
    """Route queries to the best retrieval source."""

    _recent_terms = (
        "latest",
        "newest",
        "news",
        "this year",
        "today",
        "next year",
        "current",
        "changed",
        "guidance",
        "who will win",
    )
    _device_terms = (
        "device",
        "manual",
        "model",
        "manufacturer",
        "contraindication",
        "sterilization",
        "operating temperature",
        "patient population",
        "power fluctuation",
        "troubleshoot",
    )

    def __init__(
        self,
        *,
        generator: AsyncTextGenerator | None = None,
        mode: str = "heuristic",
    ) -> None:
        self.generator = generator
        self.mode = mode

    async def route(self, query: str) -> SourceType:
        with start_span(
            "router.route",
            **{"router.mode": self.mode, "query.length": len(query)},
        ) as span:
            if self.mode == "heuristic" or self.generator is None:
                source = self._heuristic_route(query)
                set_span_attributes(span, **{"router.source": source.value})
                return source

            prompt = f"""
You are a routing agent. Based on the user query, decide where to look for information.

Options:
- Retrieve_QnA: general medical knowledge, symptoms, disease risk, prevention, or treatment.
- Retrieve_Device: medical devices, manuals, model numbers, manufacturers, indications, contraindications, or usage instructions.
- Web_Search: recent news, current external facts, future events, or out-of-domain information.

Query: "{query}"

Respond ONLY with one of: Retrieve_QnA, Retrieve_Device, Web_Search
"""
            source = self._parse_source(await self.generator.generate(prompt))
            set_span_attributes(span, **{"router.source": source.value})
            return source

    def _heuristic_route(self, query: str) -> SourceType:
        normalized = query.lower()
        if any(term in normalized for term in self._recent_terms):
            return SourceType.WEB_SEARCH
        if any(term in normalized for term in self._device_terms):
            return SourceType.RETRIEVE_DEVICE
        return SourceType.RETRIEVE_QNA

    @staticmethod
    def _parse_source(value: str) -> SourceType:
        text = value.strip()
        for source in SourceType:
            if source.value.lower() == text.lower():
                return source
        match = re.search(r"Retrieve_QnA|Retrieve_Device|Web_Search", text, re.I)
        if match:
            return QueryRouter._parse_source(match.group(0))
        msg = f"Unknown route decision: {value!r}"
        raise ValueError(msg)
