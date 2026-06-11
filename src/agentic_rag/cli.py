"""Command-line entry points."""

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from agentic_rag.evaluation import load_examples, summarize_results
from agentic_rag.llm import OpenAITextGenerator
from agentic_rag.pipeline import AgenticRAG
from agentic_rag.retrievers import ChromaRetriever
from agentic_rag.router import QueryRouter
from agentic_rag.settings import Settings
from agentic_rag.telemetry import configure_tracing
from agentic_rag.web_search import TavilyWebSearcher


def build_pipeline(settings: Settings, *, router_mode: str) -> AgenticRAG:
    """Create the default production pipeline."""
    generator = OpenAITextGenerator(
        api_key=settings.openai_api_key,
        model=settings.generation_model,
        timeout=settings.openai_timeout,
    )
    router = QueryRouter(generator=generator, mode=router_mode)
    return AgenticRAG(
        router=router,
        retriever=ChromaRetriever(
            chroma_path=str(settings.chroma_path),
            top_k=settings.top_k,
        ),
        generator=generator,
        web_searcher=TavilyWebSearcher(),
    )


async def run_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation CLI."""
    settings = Settings()
    configure_tracing(settings)
    pipeline = build_pipeline(settings, router_mode=args.router)
    examples = load_examples(args.dataset)
    if args.max_examples:
        examples = examples[: args.max_examples]
    from agentic_rag.evaluation import evaluate_examples

    results = await evaluate_examples(pipeline, examples)
    rows = [
        result.model_dump(mode="json", exclude_none=False)
        for result in results
    ]
    await asyncio.to_thread(pd.DataFrame(rows).to_csv, args.output, index=False)
    summary = summarize_results(results)
    await asyncio.to_thread(
        Path(args.summary_output).write_text,
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG CLI")
    parser.add_argument("--dataset", default="evaluation_dataset.csv")
    parser.add_argument("--output", default="evaluation_results_package.csv")
    parser.add_argument("--summary-output", default="evaluation_summary_package.json")
    parser.add_argument("--router", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run_evaluate(args))


if __name__ == "__main__":
    main()
