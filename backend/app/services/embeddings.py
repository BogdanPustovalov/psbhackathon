import math
import os
from abc import ABC, abstractmethod

import httpx

from app.core.config import settings


BATCH_SIZE = 32
BGE_M3_DIMENSION = 1024


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert texts into vectors."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generate dense embeddings through the configured OpenAI-compatible API."""

    def __init__(self, model: str) -> None:
        self.model = os.getenv("EMB_MODEL") or model
        base_url = (
            os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or settings.llm_base_url
        )
        self.endpoint = f"{base_url.rstrip('/')}/embeddings"
        self.api_key = (
            os.getenv("EMBEDDER_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or settings.llm_api_key
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise ValueError("Укажите ключ API для эмбеддингов в LLM_API_KEY.")

        embeddings = []
        dimension = BGE_M3_DIMENSION if self.model.lower() == "baai/bge-m3" else None
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(headers=headers, timeout=90.0) as client:
            for start in range(0, len(texts), BATCH_SIZE):
                batch = texts[start : start + BATCH_SIZE]
                response = await client.post(
                    self.endpoint,
                    json={"model": self.model, "input": batch, "encoding_format": "float"},
                )
                response.raise_for_status()

                rows = response.json()["data"]
                indexed = [
                    (int(row.get("index", position)), row["embedding"])
                    for position, row in enumerate(rows)
                ]
                indexed.sort(key=lambda item: item[0])
                if [index for index, _ in indexed] != list(range(len(batch))):
                    raise ValueError("API вернуло неполный список эмбеддингов.")

                for _, vector in indexed:
                    if dimension is None:
                        dimension = len(vector)
                    norm = math.hypot(*vector)
                    if len(vector) != dimension or not math.isfinite(norm) or norm == 0:
                        raise ValueError("API вернуло некорректный эмбеддинг.")
                    embeddings.append([value / norm for value in vector])

        return embeddings
