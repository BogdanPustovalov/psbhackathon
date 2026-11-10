from app.core.config import settings
from app.models.schemas import QueryRequest, QueryResponse
from app.services.embeddings import OpenAIEmbeddingProvider
from app.services.llm import OpenAICompatibleLLM
from app.services.reranker import CrossEncoderReranker
from app.services.vector_store import ChromaVectorStore


class RAGPipeline:
    """Orchestrates retrieve -> rerank -> generate."""

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddingProvider(settings.embedding_model)
        self.vector_store = ChromaVectorStore()
        self.reranker = CrossEncoderReranker(settings.reranker_model)
        self.llm = OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )

    async def answer(self, request: QueryRequest) -> QueryResponse:
        # The provider calls are intentionally isolated so they can be implemented
        # or replaced independently during the next development iteration.
        sources = []
        debug = {"retrieved": 0, "reranked": False}

        if request.use_reranker and settings.reranker_enabled and sources:
            sources = await self.reranker.rerank(request.question, sources)
            debug["reranked"] = True

        answer = await self.llm.generate(request.question, sources)
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.question,
            debug=debug,
        )
