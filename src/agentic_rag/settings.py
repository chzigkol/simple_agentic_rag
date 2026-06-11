"""Runtime settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated app configuration loaded from environment and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    chroma_path: Path = Path("./chroma_db")
    top_k: int = Field(default=3, ge=1, le=20)
    generation_model: str = "gpt-5-nano"
    evaluator_model: str = "gpt-5-nano"
    openai_timeout: float = Field(default=60.0, gt=0)
    deepeval_timeout: int = Field(default=120, ge=0)
    tracing_enabled: bool = Field(default=False, alias="OTEL_TRACING_ENABLED")
    tracing_service_name: str = Field(
        default="agentic-rag", alias="OTEL_SERVICE_NAME"
    )
    tracing_exporter: str = Field(default="console", alias="OTEL_TRACES_EXPORTER")
    tracing_file_path: Path = Field(
        default=Path("otel_traces/agentic_rag_traces.jsonl"),
        alias="OTEL_TRACES_FILE",
    )
    otlp_endpoint: str | None = Field(default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")
