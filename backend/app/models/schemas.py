from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True
    filters: dict[str, Any] | None = None


class SourceDocument(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query: str
    debug: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    accepted_files: list[str]
    message: str


class HealthResponse(BaseModel):
    status: str
    service: str
