from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
import re
import unicodedata
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any
import warnings

from . import (
    BM25_TOP_K,
    BM25_WEIGHT,
    FAISS_WEIGHT,
    FAISS_TOP_K,
    RRF_TOP_K,
    Document,
)
from .embedding import EmbeddingClient


@dataclass(slots=True)
class HybridSearchTrace:
    """Ranked source lists and the weighted RRF pool sent to the reranker."""

    faiss_hits: list[Document]
    bm25_hits: list[Document]
    rrf_hits: list[Document]
    candidates: list[Document]

@lru_cache(maxsize=1)
def _morph_analyzer() -> Any | None:
    if os.getenv("BM25_LEMMATIZE", "1").casefold() in {"0", "false", "no"}:
        return None
    try:
        import pymorphy3
    except ImportError:
        warnings.warn(
            "pymorphy3 не установлен; BM25 использует исходные формы слов",
            stacklevel=2,
        )
        return None
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def _normal_form(token: str) -> str:
    analyzer = _morph_analyzer()
    if analyzer is None or not any("а" <= char <= "я" for char in token):
        return token
    return analyzer.parse(token)[0].normal_form


def _bm25_tokens(text: str) -> list[str]:
    normalized = (
        unicodedata.normalize("NFC", text)
        .casefold()
        .replace("ё", "е")
        .replace("_", " ")
    )
    words = re.findall(r"\w+", normalized, flags=re.UNICODE)
    return [_normal_form(word) for word in words]


def _searchable_text(document: Document) -> str:
    metadata = document.metadata
    filename = metadata.get("filename", "")
    document_id = metadata.get("document_id", "")
    fields = (
        filename,
        document_id if document_id != filename else "",
        metadata.get("section", ""),
        metadata.get("clause", ""),
        document.text,
    )
    return " ".join(str(field) for field in fields if field)


class _BM25Index:
    """..."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus_size = len(corpus)
        self.k1 = k1
        self.b = b
        self.document_lengths = [len(document) for document in corpus]
        self.average_length = (
            sum(self.document_lengths) / self.corpus_size if self.corpus_size else 0.0
        )
        self.term_frequencies: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        for document in corpus:
            frequencies: dict[str, int] = {}
            for token in document:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.term_frequencies.append(frequencies)
            for token in frequencies:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.idf = {
            token: math.log((self.corpus_size + 1.0) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores: list[float] = []
        for frequencies, document_length in zip(
            self.term_frequencies, self.document_lengths
        ):
            score = 0.0
            for token in query:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0
                    - self.b
                    + self.b * document_length / max(self.average_length, 1.0)
                )
                score += self.idf.get(token, 0.0) * (
                    (frequency * (self.k1 + 1.0)) / denominator + 1.0
                )
            scores.append(score)
        return scores


class FaissVectorIndex:
    """...
    """

    def __init__(self, dimension: int | None = None, metric: str = "cosine") -> None:
        if metric not in {"cosine", "l2"}:
            raise ValueError("metric must be 'cosine' or 'l2'")
        if dimension is not None and dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self.dimension = dimension
        self.metric = metric
        self.documents: list[Document] = []
        self._index: Any | None = None
        self._bm25: Any | None = None
        self._bm25_terms: list[set[str]] = []

    @property
    def is_empty(self) -> bool:
        return not self.documents

    @property
    def count(self) -> int:
        return len(self.documents)

    def _prepare_vectors(self, vectors: Iterable[Sequence[float]]) -> Any:
        import numpy as np

        vectors = np.asarray(list(vectors), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise ValueError("embeddings must be a non-empty 2D array")
        vector_dimension = vectors.shape[1]
        if self.dimension is None:
            self.dimension = vector_dimension
        if vector_dimension != self.dimension:
            raise ValueError(
                f"embedding dimension {vector_dimension} does not match index "
                f"dimension {self.dimension}"
            )

        if not np.isfinite(vectors).all():
            raise ValueError("embeddings must contain finite values")
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1)
            if np.any(norms == 0):
                raise ValueError("embeddings must not contain a zero vector")
            vectors = vectors / norms[:, None]
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def add(
        self,
        documents: Iterable[Document],
        embeddings: Iterable[Sequence[float]],
    ) -> None:
        """..."""

        documents = list(documents)
        embeddings = list(embeddings)
        if len(documents) != len(embeddings):
            raise ValueError("Ошибка! Количество документов и векторов встраивания должно совпадать")
        if not documents:
            return
        if self.documents and self._index is None:
            raise ValueError("Нельзя смешивать лексический индекс с векторным")

        try:
            corpus = [
                _bm25_tokens(_searchable_text(document))
                for document in self.documents + documents
            ]
            bm25 = _BM25Index(corpus) if any(corpus) else None
        except (TypeError, ValueError) as exc:
            raise ValueError("Ошибка! Невозможно построить индекс BM25") from exc

        vectors = self._prepare_vectors(embeddings)
        import faiss

        if self._index is None:
            self._index = (
                faiss.IndexFlatIP(self.dimension)
                if self.metric == "cosine"
                else faiss.IndexFlatL2(self.dimension)
            )
        self._index.add(vectors)
        self.documents.extend(documents)
        self._bm25 = bm25
        self._bm25_terms = [set(tokens) for tokens in corpus]

    def add_lexical(self, documents: Iterable[Document]) -> None:
        """..."""

        documents = list(documents)
        if not documents:
            return
        if self._index is not None:
            raise ValueError("Ошибка! Невозможно добавить документы только с лексическим содержанием в векторный индекс")
        self.documents.extend(documents)
        corpus = [_bm25_tokens(_searchable_text(document)) for document in self.documents]
        self._bm25 = _BM25Index(corpus) if any(corpus) else None
        self._bm25_terms = [set(tokens) for tokens in corpus]

    @classmethod
    def from_prebuilt(cls, documents: Iterable[Document], index_path: str) -> "FaissVectorIndex":
        """Attach a saved FAISS index to the matching chunk list and build BM25."""
        import faiss

        documents = list(documents)
        index = faiss.read_index(index_path)
        if index.ntotal != len(documents):
            raise ValueError("FAISS index and documents.json have different chunk counts")
        result = cls(dimension=index.d)
        result.add_lexical(documents)
        result._index = index
        return result

    def _semantic_hits(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("Ошибка! top_k должен быть больше нуля")
        if self._index is None or not self.documents:
            return []

        query = self._prepare_vectors([query_embedding])
        scores, indices = self._index.search(query, min(top_k, self.count))
        scored = [
            (int(index), float(score if self.metric == "cosine" else -score))
            for index, score in zip(indices[0], scores[0])
            if index >= 0
        ]
        scored.sort(key=lambda hit: (-hit[1], hit[0]))
        return scored

    def _bm25_hits(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("Ошибка! top_k должен быть больше нуля")
        if not isinstance(query, str):
            raise TypeError("Ошибка! query должен быть строкой")
        tokens = _bm25_tokens(query)
        if self._bm25 is None or not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        query_terms = set(tokens)
        ranked = sorted(
            (
                (index, float(score))
                for index, score in enumerate(scores)
                if score > 0 and self._bm25_terms[index] & query_terms
            ),
            key=lambda hit: (-hit[1], hit[0]),
        )
        return ranked[:top_k]

    def _result(self, index: int, score: float) -> Document:
        source = self.documents[index]
        return Document(
            text=source.text,
            metadata=dict(source.metadata),
            score=score,
        )

    def _chunk_id(self, index: int) -> str:
        """Identify a fragment across both indexes, including repeated ingestion."""

        document = self.documents[index]
        if document.metadata.get("chunk_id") is not None:
            return str(document.metadata["chunk_id"])
        source = (
            document.metadata.get("source")
            or document.metadata.get("document_id")
            or document.metadata.get("filename")
        )
        if source is None:
            return f"index:{index}"
        identity = (
            f"{source}\0{document.metadata.get('start_char')}\0"
            f"{document.metadata.get('end_char')}\0{document.text}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = FAISS_TOP_K,
    ) -> list[Document]:
        """..."""

        return [
            self._result(index, score)
            for index, score in self._semantic_hits(query_embedding, top_k)
        ]

    def semantic_search(
        self,
        query: str,
        embedding_client: EmbeddingClient | None = None,
        *,
        top_k: int = FAISS_TOP_K,
    ) -> list[Document]:
        """..."""

        if not isinstance(query, str):
            raise TypeError("Ошибка! query должен быть строкой")
        if top_k <= 0:
            raise ValueError("Ошибка! top_k должен быть больше нуля")
        if self.is_empty or self._index is None:
            return []
        client = embedding_client or EmbeddingClient()
        return self.search(client.embed([query])[0], top_k=top_k)

    def bm25_search(self, query: str, *, top_k: int = BM25_TOP_K) -> list[Document]:
        """..."""

        return [
            self._result(index, score)
            for index, score in self._bm25_hits(query, top_k)
        ]

    def hybrid_candidates(
        self,
        query: str,
        embedding_client: EmbeddingClient | None = None,
        *,
        top_k: int = RRF_TOP_K,
        faiss_k: int = FAISS_TOP_K,
        bm25_k: int = BM25_TOP_K,
        rrf_k: int = 60,
        semantic_weight: float = FAISS_WEIGHT,
        lexical_weight: float = BM25_WEIGHT,
    ) -> HybridSearchTrace:
        """Fuse FAISS and BM25 ranks using weighted RRF, then take Top-K."""

        if not isinstance(query, str):
            raise TypeError("Ошибка! query должен быть строкой")
        if min(top_k, faiss_k, bm25_k, rrf_k) <= 0:
            raise ValueError("top_k, faiss_k, bm25_k и rrf_k должны быть положительными")
        if (
            not math.isfinite(semantic_weight)
            or not math.isfinite(lexical_weight)
            or semantic_weight < 0
            or lexical_weight < 0
            or semantic_weight + lexical_weight == 0
        ):
            raise ValueError("Веса поисковиков должны быть неотрицательными и не оба нулевыми")
        if self.is_empty:
            return HybridSearchTrace([], [], [], [])

        semantic_hits: list[tuple[int, float]] = []
        if self._index is not None:
            client = embedding_client or EmbeddingClient()
            semantic_hits = self._semantic_hits(client.embed([query])[0], faiss_k)
        lexical_hits = self._bm25_hits(query, bm25_k)

        signals: dict[str, dict[str, Any]] = {}
        source_order: dict[str, list[str]] = {"faiss": [], "bm25": []}
        for name, hits in (("faiss", semantic_hits), ("bm25", lexical_hits)):
            for rank, (index, score) in enumerate(hits, start=1):
                chunk_id = self._chunk_id(index)
                if chunk_id not in signals:
                    signals[chunk_id] = {
                        "index": index,
                        "faiss_rank": None,
                        "faiss_score": None,
                        "bm25_rank": None,
                        "bm25_score": None,
                    }
                if signals[chunk_id][f"{name}_rank"] is not None:
                    continue
                signals[chunk_id][f"{name}_rank"] = rank
                signals[chunk_id][f"{name}_score"] = score
                source_order[name].append(chunk_id)

        active_weight = (semantic_weight if semantic_hits else 0.0) + (
            lexical_weight if lexical_hits else 0.0
        )
        max_rrf = active_weight / (rrf_k + 1)
        for signal in signals.values():
            score = 0.0
            if signal["faiss_rank"] is not None:
                score += semantic_weight / (rrf_k + signal["faiss_rank"])
            if signal["bm25_rank"] is not None:
                score += lexical_weight / (rrf_k + signal["bm25_rank"])
            signal["rrf_score"] = score
            signal["rrf_normalized"] = score / max_rrf if max_rrf else 0.0

        ranked_ids = sorted(
            signals,
            key=lambda chunk_id: (-signals[chunk_id]["rrf_score"], signals[chunk_id]["index"]),
        )
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            signals[chunk_id]["rrf_rank"] = rank

        def as_document(chunk_id: str, score: float) -> Document:
            signal = signals[chunk_id]
            document = self._result(signal["index"], score)
            document.metadata.update(
                chunk_id=chunk_id,
                faiss_rank=signal["faiss_rank"],
                faiss_score=signal["faiss_score"],
                bm25_rank=signal["bm25_rank"],
                bm25_score=signal["bm25_score"],
                rrf_rank=signal["rrf_rank"],
                rrf_score=signal["rrf_score"],
                rrf_normalized=signal["rrf_normalized"],
                rrf_k=rrf_k,
                rrf_faiss_weight=semantic_weight,
                rrf_bm25_weight=lexical_weight,
            )
            return document

        return HybridSearchTrace(
            faiss_hits=[
                as_document(chunk_id, signals[chunk_id]["faiss_score"])
                for chunk_id in source_order["faiss"]
            ],
            bm25_hits=[
                as_document(chunk_id, signals[chunk_id]["bm25_score"])
                for chunk_id in source_order["bm25"]
            ],
            rrf_hits=[
                as_document(chunk_id, signals[chunk_id]["rrf_score"])
                for chunk_id in ranked_ids[:top_k]
            ],
            candidates=[
                as_document(chunk_id, signals[chunk_id]["rrf_score"])
                for chunk_id in ranked_ids[:top_k]
            ],
        )

    def hybrid_search(
        self,
        query: str,
        embedding_client: EmbeddingClient | None = None,
        *,
        top_k: int = RRF_TOP_K,
        candidate_k: int = FAISS_TOP_K,
        rrf_k: int = 60,
        semantic_weight: float = FAISS_WEIGHT,
        lexical_weight: float = BM25_WEIGHT,
    ) -> list[Document]:
        """Return the weighted RRF pool for callers that only need documents."""

        return self.hybrid_candidates(
            query,
            embedding_client,
            top_k=top_k,
            faiss_k=candidate_k,
            bm25_k=candidate_k,
            rrf_k=rrf_k,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
        ).candidates


def build_faiss_index(
    documents: Iterable[Document],
    embeddings: Iterable[Sequence[float]],
    *,
    metric: str = "cosine",
) -> FaissVectorIndex:
    """..."""

    index = FaissVectorIndex(metric=metric)
    index.add(documents, embeddings)
    return index


def embed_and_index(
    documents: Iterable[Document],
    embedding_client: EmbeddingClient | None = None,
    *,
    metric: str = "cosine",
) -> FaissVectorIndex:
    """..."""

    documents = list(documents)
    client = embedding_client or EmbeddingClient()
    embeddings = client.embed([document.text for document in documents])
    return build_faiss_index(documents, embeddings, metric=metric)
