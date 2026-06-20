from agentic_rag.constants import SourceType
from agentic_rag.router import QueryRouter


async def test_router_detects_device_query() -> None:
    router = QueryRouter()

    source = await router.route("What are the contraindications for model X?")

    assert source == SourceType.RETRIEVE_DEVICE


async def test_router_detects_web_query() -> None:
    router = QueryRouter()

    source = await router.route("What is the latest vaccine development news?")

    assert source == SourceType.WEB_SEARCH


async def test_router_defaults_to_qna() -> None:
    router = QueryRouter()

    source = await router.route("What are the symptoms of flu?")

    assert source == SourceType.RETRIEVE_QNA
