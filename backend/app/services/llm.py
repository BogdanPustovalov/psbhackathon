from abc import ABC, abstractmethod

from app.models.schemas import SourceDocument


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, question: str, context: list[SourceDocument]) -> str:
        """Generate a grounded answer using retrieved context."""


class OpenAICompatibleLLM(LLMProvider):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    async def generate(self, question: str, context: list[SourceDocument]) -> str:
        # TODO: implement chat-completions call and structured citations.
        if not context:
            return "В базе знаний пока не найдено релевантных документов."
        return "LLM-провайдер ещё не подключён. Найдены релевантные фрагменты контекста."
