from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert texts into vectors."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Placeholder for an OpenAI-compatible embeddings adapter."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # TODO: call the configured OpenAI-compatible /embeddings endpoint.
        raise NotImplementedError("Подключите HTTP-клиент провайдера эмбеддингов")
