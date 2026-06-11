"""Pydantic models used across the pipeline and evaluation."""

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.constants import SourceType


class RetrievedDocument(BaseModel):
    """A retrieved Chroma document."""

    model_config = ConfigDict(extra="allow")

    doc_id: str
    text: str
    metadata: dict[str, object] = Field(default_factory=dict)


class RAGRequest(BaseModel):
    """User query request."""

    query: str = Field(min_length=1)


class RAGResponse(BaseModel):
    """Pipeline response."""

    query: str
    source: SourceType
    answer: str
    context: list[RetrievedDocument] = Field(default_factory=list)
    is_relevant: bool | None = None


class EvaluationExample(BaseModel):
    """Single evaluation row."""

    query: str = Field(min_length=1)
    expected_source_type: SourceType
    expected_collection: str | None = None
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_answer: str | None = None
    category: str | None = None


class EvaluationResult(BaseModel):
    """Row-level evaluation result."""

    query: str
    expected_source_type: SourceType
    predicted_source_type: SourceType
    route_correct: bool
    retrieval_source: SourceType
    expected_doc_ids: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    precision_at_k: float
    recall_at_k: float
    hit_at_k: float
    mrr: float
    actual_answer: str = ""
    answer_relevancy: float | None = None
    answer_relevancy_reason: str = ""
    answer_relevancy_success: bool | None = None
