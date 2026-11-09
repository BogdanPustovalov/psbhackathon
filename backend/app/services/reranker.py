from abc import ABC, abstractmethod

from app.models.schemas import SourceDocument


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, documents: list[SourceDocument]) -> list[SourceDocument]:
        """Reorder retrieved chunks by query-document relevance."""


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def rerank(self, query: str, documents: list[SourceDocument]) -> list[SourceDocument]:
        # TODO: load a CrossEncoder and replace scores with reranker scores.
        return documents
