"""Async LLM client helpers."""

import asyncio
from typing import Protocol

from openai import AsyncOpenAI

from agentic_rag.telemetry import set_span_attributes, start_span


class AsyncTextGenerator(Protocol):
    """Protocol for async text generation."""

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""


class OpenAITextGenerator:
    """Small async wrapper around OpenAI chat completions."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout: float,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=1)
        self._model = model

    async def generate(self, prompt: str) -> str:
        with start_span(
            "llm.generate",
            **{
                "llm.provider": "openai",
                "llm.model": self._model,
                "prompt.length": len(prompt),
            },
        ) as span:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content or ""
            set_span_attributes(span, **{"answer.length": len(answer)})
            return answer


class StaticTextGenerator:
    """Deterministic generator for tests and offline smoke checks."""

    def __init__(self, response: str) -> None:
        self.response = response

    async def generate(self, prompt: str) -> str:
        with start_span(
            "llm.generate.static",
            **{"prompt.length": len(prompt), "answer.length": len(self.response)},
        ):
            await asyncio.sleep(0)
            return self.response
