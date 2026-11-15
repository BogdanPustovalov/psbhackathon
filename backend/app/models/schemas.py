from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True
    history: list[ChatMessage] = Field(default_factory=list)
    filters: dict[str, Any] | None = None


class SourceDocument(BaseModel):
    id: str
    n: int | None = None
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query: str
    confidence: dict[str, Any] | None = None
    missing: list[str] = Field(default_factory=list)
    clarification: dict[str, Any] | None = None
    debug: dict[str, Any] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    doc_id: str
    doc_title: str
    doc_type: str
    edition: dict[str, Any] | None = None
    related: list[str] = Field(default_factory=list)


class DocumentDetail(DocumentInfo):
    text: str


class IngestResponse(BaseModel):
    accepted_files: list[str]
    message: str


class HealthResponse(BaseModel):
    status: str
    service: str
