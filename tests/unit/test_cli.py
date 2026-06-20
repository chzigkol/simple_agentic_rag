from agentic_rag.cli import build_pipeline
from agentic_rag.settings import Settings
from agentic_rag.web_search import TavilyWebSearcher


def test_build_pipeline_disables_web_search_without_tavily_key(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="test-openai-key",
        TAVILY_API_KEY=None,
        chroma_path=tmp_path / "chroma",
    )

    pipeline = build_pipeline(settings, router_mode="heuristic")

    assert pipeline.web_searcher is None


def test_build_pipeline_passes_tavily_key_to_web_searcher(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="test-openai-key",
        TAVILY_API_KEY="test-tavily-key",
        chroma_path=tmp_path / "chroma",
    )

    pipeline = build_pipeline(settings, router_mode="heuristic")

    assert isinstance(pipeline.web_searcher, TavilyWebSearcher)
    assert pipeline.web_searcher.tavily_api_key == "test-tavily-key"
