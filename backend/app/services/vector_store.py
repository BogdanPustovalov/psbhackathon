import asyncio
from abc import ABC, abstractmethod
from typing import Any

import chromadb

from app.core.config import settings
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


class ChromaVectorStore(VectorStore):
    """Persistent Chroma adapter kept behind the vector-store abstraction.

    Chroma runs embedded in the backend, so the database state is stored in
    ``settings.chroma_persist_directory`` and does not require a separate
    service in Docker Compose.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.collection_name = collection_name or settings.chroma_collection
        self.client = (
            client
            if client is not None
            else chromadb.PersistentClient(path=self.persist_directory)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _metadata_for_chroma(metadata: dict[str, Any]) -> dict[str, Any]:
        # Chroma does not accept an empty metadata object. Keep a private
        # marker for chunks that genuinely have no metadata and hide it again
        # when results are mapped back to the API model.
        return metadata or {"_chroma_empty": True}

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SourceDocument]:
        if not vector or top_k < 1:
            return []

        def query_collection() -> dict[str, Any] | None:
            count = self.collection.count()
            if count == 0:
                return None

            query_kwargs: dict[str, Any] = {
                "query_embeddings": [vector],
                "n_results": min(top_k, count),
                "include": ["documents", "metadatas", "distances"],
            }
            if filters:
                query_kwargs["where"] = filters
            return self.collection.query(**query_kwargs)

        result = await asyncio.to_thread(query_collection)
        if not result:
            return []

        ids = result.get("ids", [[]])[0]
        texts = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            SourceDocument(
                id=document_id,
                text=text or "",
                # The collection uses cosine distance, so convert it to the
                # similarity-style score expected by the API model.
                score=1.0 - float(distance),
                metadata={
                    key: value
                    for key, value in (metadata or {}).items()
                    if key != "_chroma_empty"
                },
            )
            for document_id, text, metadata, distance in zip(
                ids,
                texts,
                metadatas,
                distances,
            )
        ]

    async def upsert(
        self,
        documents: list[SourceDocument],
        vectors: list[list[float]],
    ) -> None:
        """Store document chunks and their embeddings in the collection."""
        if len(documents) != len(vectors):
            raise ValueError("Document and vector counts must match")
        if not documents:
            return

        await asyncio.to_thread(
            self.collection.upsert,
            ids=[document.id for document in documents],
            embeddings=vectors,
            documents=[document.text for document in documents],
            metadatas=[
                self._metadata_for_chroma(document.metadata)
                for document in documents
            ],
        )

    async def delete(self, ids: list[str]) -> None:
        """Delete indexed chunks by their stable identifiers."""
        if ids:
            await asyncio.to_thread(self.collection.delete, ids=ids)
