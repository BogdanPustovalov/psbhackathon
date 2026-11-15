from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass, field
from importlib import import_module
import io
import math
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
import zipfile
from xml.etree import ElementTree as ET

import httpx

try:
    from dotenv import load_dotenv
except ImportError:
    # Environment variables can still be set by the shell when python-dotenv
    # is not installed.
    pass
else:
    load_dotenv(Path(__file__).with_name(".env"))


DEFAULT_CHUNK_SIZE = 450
DEFAULT_CHUNK_OVERLAP = 75
EMBEDDING_BATCH_SIZE = 8
FAISS_TOP_K = 30
BM25_TOP_K = 30
RRF_TOP_K = 40
RERANKER_INPUT_K = 40
RERANKER_TOP_K = 40
FAISS_PROTECTED_TOP_K = 3
BM25_PROTECTED_TOP_K = 3
BM25_CONFIRM_FAISS_TOP_K = 15
BM25_CONFIRM_RRF_TOP_K = 10
BM25_CONFIRM_RERANKER_SCORE = 0.0
FINAL_CONTEXT_TOP_K = 20
MAX_CONTEXT_TOKENS = 8000
MIN_CONTEXT_SCORE = 0.15
MIN_CONTEXT_RELATIVE_SCORE = 0.2
SOURCE_DIVERSITY_PENALTY = 0.03
FAISS_WEIGHT = 0.4
BM25_WEIGHT = 0.4
RRF_WEIGHT = 0.1
RERANKER_WEIGHT = 0.1
BATCH_SIZE = EMBEDDING_BATCH_SIZE
BGE_M3_DIMENSION = 1024
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".docx"}


@dataclass(slots=True)
class Document:

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


class RerankerError(RuntimeError):
    """..."""


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = _env(name)
        if value:
            return value
    return default


def _document_text(document: Any) -> str:

    if isinstance(document, str):
        return document
    if isinstance(document, Mapping):
        for key in ("text", "page_content", "content"):
            value = document.get(key)
            if value is not None:
                return str(value)
    for attribute in ("text", "page_content", "content"):
        value = getattr(document, attribute, None)
        if value is not None:
            return str(value)
    raise TypeError(
        "Ошибка! Документ должен быть строкой или предоставлять text, page_content или content"
    )

def _with_score(document: Any, score: float) -> Any:

    score = float(score)
    if isinstance(document, Mapping):
        result = dict(document)
        result["score"] = score
        return result

    model_copy = getattr(document, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"score": score})

    copy_method = getattr(document, "copy", None)
    if callable(copy_method):
        try:
            return copy_method(update={"score": score})
        except TypeError:
            pass

    if hasattr(document, "score"):
        try:
            document.score = score
        except (AttributeError, TypeError):
            pass
    return document




if __package__:
    from .embedding import EmbeddingAPIError, EmbeddingClient, EmbeddingProvider, OpenAIEmbeddingProvider
    from .llm import LLMAPIError, OpenAICompatibleLLM
    from .reranker import Reranker
    from .vector_store import FaissVectorIndex, build_faiss_index, embed_and_index
else:
    # Allow this file to be run directly by an IDE or debugger.
    package_parent = str(Path(__file__).resolve().parents[3])
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    from app.services.rag.embedding import EmbeddingAPIError, EmbeddingClient, EmbeddingProvider, OpenAIEmbeddingProvider
    from app.services.rag.llm import LLMAPIError, OpenAICompatibleLLM
    from app.services.rag.reranker import Reranker
    from app.services.rag.vector_store import FaissVectorIndex, build_faiss_index, embed_and_index

_PIPELINE_EXPORTS = {
    "RAGResult",
    "RAGSystem",
    "load_and_index",
    "load_and_rerank",
    "load_documents",
    "main",
    "rerank_documents",
    "run_platform_submission",
    "search_and_rerank",
    "split_documents",
    "split_into_chunks",
}


def __getattr__(name: str) -> Any:
    if name in _PIPELINE_EXPORTS:
        package_name = __package__ or "app.services.rag"
        pipeline = import_module(".pipeline", package_name)
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Document",
    "Reranker",
    "RerankerError",
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "EmbeddingClient",
    "EmbeddingAPIError",
    "OpenAICompatibleLLM",
    "LLMAPIError",
    "FaissVectorIndex",
    "RAGResult",
    "RAGSystem",
    "load_documents",
    "split_into_chunks",
    "split_documents",
    "build_faiss_index",
    "embed_and_index",
    "load_and_index",
    "search_and_rerank",
    "rerank_documents",
    "load_and_rerank",
    "run_platform_submission",
    "main",
]
