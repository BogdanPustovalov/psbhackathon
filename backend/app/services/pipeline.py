"""FastAPI adapter for the RAG implementation and prepared artifacts."""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from pathlib import Path

from app.core.config import ROOT_DIR, settings
from app.models.schemas import QueryRequest, QueryResponse, SourceDocument
from app.services.artifacts import Corpus, load_corpus
from app.services.rag import Document, RerankerError
from app.services.rag.embedding import EmbeddingAPIError, EmbeddingClient
from app.services.rag.llm import LLMAPIError, OpenAICompatibleLLM
from app.services.rag.pipeline import RAGSystem
from app.services.rag.reranker import Reranker


class ExtractiveLLM:
    model = "extractive"

    def generate(self, question: str, sources: list[Document]) -> str:
        if not sources:
            return "В доступной базе знаний не найдено подходящих фрагментов."
        excerpts = [f"{source.text.strip()[:600]} [{n}]" for n, source in enumerate(sources[:3], 1)]
        return "Найдены следующие фрагменты документов:\n\n" + "\n\n".join(excerpts)


@lru_cache(maxsize=1)
def get_corpus() -> Corpus:
    directory = Path(settings.data_hack_output_dir)
    return load_corpus(directory if directory.is_absolute() else ROOT_DIR / directory)


class RAGPipeline:
    def __init__(self, corpus: Corpus | None = None) -> None:
        self.corpus = corpus or get_corpus()
        api_key = settings.llm_api_key or os.getenv("OPENAI_API_KEY", "")
        llm = (
            OpenAICompatibleLLM(
                base_url=settings.llm_base_url,
                api_key=api_key,
                model=settings.llm_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                timeout=45,
                retries=1,
            )
            if api_key and api_key != "replace-me"
            else ExtractiveLLM()
        )
        reranker = (
            Reranker(
                model_name=settings.reranker_model,
                cache_folder=settings.reranker_cache_dir or None,
                device=settings.reranker_device,
                batch_size=settings.reranker_batch_size,
                max_length=settings.reranker_max_length,
            )
            if settings.reranker_enabled
            else None
        )
        self.rag = RAGSystem(
            embedding_client=EmbeddingClient(
                model_name=self.corpus.model,
                model_path=settings.embedding_model_path or None,
                cache_folder=settings.embedding_cache_dir or None,
                device=settings.embedding_device,
            ),
            reranker=reranker,
            llm=llm,
            retrieve_k=settings.top_k,
            rrf_top_k=settings.rrf_top_k,
            top_k=settings.final_context_top_k,
            rerank_candidates=settings.reranker_candidates,
            reranker_output_k=settings.reranker_top_n,
            semantic_weight=settings.rrf_semantic_weight,
            lexical_weight=settings.rrf_bm25_weight,
        )
        self.rag.index = self.corpus.index
        self.rag.chunks = self.corpus.chunks
        self._embedding_unavailable = False

    def _answer_sync(self, request: QueryRequest) -> QueryResponse:
        question = request.question.strip()
        previous = next(
            (message.content for message in reversed(request.history) if message.role == "user"),
            None,
        )
        search_question = f"{previous}\n{question}" if previous and len(question) < 80 else question
        mode = "bm25" if self._embedding_unavailable else "hybrid"
        use_reranker = request.use_reranker and settings.reranker_enabled
        for _ in range(3):
            try:
                sources = self.rag.retrieve(
                    search_question,
                    search_mode=mode,
                    top_k=min(request.top_k or settings.final_context_top_k, settings.final_context_top_k),
                    use_reranker=use_reranker,
                )
                break
            except EmbeddingAPIError:
                self._embedding_unavailable = True
                mode = "bm25"
            except RerankerError:
                use_reranker = False
        else:
            raise RuntimeError("RAG retrieval failed after model fallbacks")
        llm_status = "configured" if not isinstance(self.rag.llm, ExtractiveLLM) else "extractive"
        if not sources:
            answer = ExtractiveLLM().generate(question, [])
        else:
            try:
                answer = self.rag.llm.generate(question, sources)
            except LLMAPIError:
                answer = ExtractiveLLM().generate(question, sources)
                llm_status = "unavailable"
        mapped = [
            SourceDocument(
                id=f"{source.metadata['doc_id']}:{source.metadata['chunk_index']}",
                n=number,
                text=source.text,
                score=float(source.score or 0),
                metadata=source.metadata,
            )
            for number, source in enumerate(sources, 1)
        ]
        return QueryResponse(
            query=question,
            answer=answer,
            sources=mapped,
            confidence={"zone": "uncertain" if sources else "no_answer", "best_score": mapped[0].score if mapped else None},
            debug={
                "retrieved": len(sources),
                "search_mode": mode,
                "indexed_chunks": len(self.corpus.chunks),
                "llm": llm_status,
                "reranked": use_reranker,
            },
        )

    async def answer(self, request: QueryRequest) -> QueryResponse:
        return await asyncio.to_thread(self._answer_sync, request)
