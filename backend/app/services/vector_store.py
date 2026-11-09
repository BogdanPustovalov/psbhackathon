from abc import ABC, abstractmethod

from app.models.schemas import SourceDocument


class VectorStore(ABC):
    @abstractmethod
    async def search(
        self,
        vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SourceDocument]:
        """Return the most similar document chunks."""


class QdrantVectorStore(VectorStore):
    """Qdrant adapter boundary; keep storage details out of the API layer."""

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SourceDocument]:
        # TODO: initialize QdrantClient and map points to SourceDocument.
        return []
