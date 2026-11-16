from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import sys
import uuid
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from . import (
    BM25_WEIGHT,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    FAISS_TOP_K,
    FAISS_WEIGHT,
    FINAL_CONTEXT_TOP_K,
    RERANKER_INPUT_K,
    RERANKER_TOP_K,
    RRF_TOP_K,
    SUPPORTED_EXTENSIONS,
    Document,
)
from .embedding import EmbeddingClient
from .llm import OpenAICompatibleLLM
from .relevance import document_temporal_metadata
from .reranker import Reranker
from .vector_store import FaissVectorIndex, HybridSearchTrace, embed_and_index


def _clean_text(text: str) -> str:
    """..."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def _read_plain_text(path: Path) -> str:
    """..."""

    return _decode_text_bytes(path.read_bytes())


def _decode_text_bytes(raw: bytes) -> str:
    """..."""

    for encoding in ("utf-8-sig", "cp1251", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_docx_bytes(raw: bytes) -> str:
    """..."""

    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = []
    for paragraph in root.iter(namespace + "p"):
        paragraphs.append(
            "".join(node.text or "" for node in paragraph.iter(namespace + "t"))
        )
    return "\n".join(paragraphs)


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return _read_plain_text(path)
    if suffix == ".docx":
        return _read_docx_bytes(path.read_bytes())
    raise ValueError(f"Неподдерживаемый формат документа: {path.suffix or '<none>'}")


def _load_zip_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/") or name.startswith("__MACOSX/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue
            try:
                raw = archive.read(name)
                text = (
                    _read_docx_bytes(raw)
                    if suffix == ".docx"
                    else _decode_text_bytes(raw)
                )
            except (UnicodeDecodeError, KeyError, ET.ParseError, zipfile.BadZipFile):
                print(f"Пропущен нечитаемый документ: {name}", file=sys.stderr)
                continue
            text = _clean_text(text)
            if text:
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source": f"{path}!{name}",
                            "filename": name,
                            "extension": suffix,
                        },
                    )
                )
    if not documents:
        raise ValueError("База знаний содержит нечитаемые документы")
    return documents


def load_documents(path: str | Path) -> list[Document]:
    """...
    """

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Путь к документу не существует: {root}")

    if root.is_file() and root.suffix.lower() == ".zip":
        return _load_zip_documents(root)

    if root.is_file():
        paths = [root]
    elif root.is_dir():
        paths = sorted(
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        raise ValueError(f"Путь к документу не является файлом или директорией: {root}")

    documents: list[Document] = []
    for file_path in paths:
        text = _clean_text(_read_file(file_path))
        if text:
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "extension": file_path.suffix.lower(),
                    },
                )
            )
    return documents


def _get_encoding(encoding_name: str):
    """Regex units (words and punctuation), not model tokenizer tokens."""
    return re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _token_windows(
    text: str, start: int, end: int, chunk_size: int, chunk_overlap: int
) -> list[tuple[int, int]]:
    tokens = list(_get_encoding("regex").finditer(text, start, end))
    windows: list[tuple[int, int]] = []
    position = 0
    while position < len(tokens):
        selected = tokens[position : position + chunk_size]
        windows.append((selected[0].start(), selected[-1].end()))
        if position + chunk_size >= len(tokens):
            break
        position += chunk_size - chunk_overlap
    return windows


def _structured_chunks(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[tuple[int, int, str | None, str | None]]:
    """Keep legal clauses intact; use overlapping regex windows for long clauses."""
    paragraphs = list(
        re.finditer(
            r"\S.*?(?=\n\s*\n|\n(?=(?:\d+(?:\.\d+)*[.)]\s+|статья\s+\d+|раздел\s+[IVX\d]+))|\Z)",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    if not paragraphs:
        return []

    units: list[tuple[int, int, str | None, str | None]] = []
    section: str | None = None
    current_start: int | None = None
    current_end = 0
    current_clause: str | None = None
    current_section: str | None = None

    def flush() -> None:
        nonlocal current_start
        if current_start is not None:
            units.append((current_start, current_end, current_section, current_clause))
            current_start = None

    for paragraph in paragraphs:
        start, end = paragraph.span()
        body = paragraph.group().strip()
        first_line = body.splitlines()[0].strip()
        article = bool(re.match(r"(?i)^статья\s+\d+\b", first_line))
        numbered = re.match(r"^(\d+(?:\.\d+)*[.)])\s+", first_line)
        short_heading = bool(
            re.match(r"^\d+\.\s+", first_line)
            and len(body) <= 110
            and "\n" not in body
            and not body.endswith((".", ";", ":"))
        )
        heading = article or short_heading or bool(
            re.match(r"(?i)^раздел\s+[IVX\d]+\b", first_line)
        )

        if heading or numbered:
            flush()
            if heading:
                section = first_line
                current_clause = None
            else:
                current_clause = numbered.group(1)
            current_section = section
            current_start = start
        elif current_start is None:
            current_start = start
            current_section = section
            current_clause = None
        current_end = end
    flush()

    chunks: list[tuple[int, int, str | None, str | None]] = []
    packed_start: int | None = None
    packed_end = 0
    packed_section: str | None = None
    packed_clauses: list[str] = []

    def flush_pack() -> None:
        nonlocal packed_start, packed_clauses
        if packed_start is not None:
            chunks.append(
                (packed_start, packed_end, packed_section, ", ".join(packed_clauses) or None)
            )
            packed_start = None
            packed_clauses = []

    for start, end, unit_section, clause in units:
        unit_tokens = len(_get_encoding("regex").findall(text[start:end]))
        if unit_tokens > chunk_size:
            flush_pack()
            for window_start, window_end in _token_windows(
                text, start, end, chunk_size, chunk_overlap
            ):
                chunks.append((window_start, window_end, unit_section, clause))
            continue
        if (
            packed_start is not None
            and (
                unit_section != packed_section
                or len(_get_encoding("regex").findall(text[packed_start:end])) > chunk_size
            )
        ):
            flush_pack()
        if packed_start is None:
            packed_start = start
            packed_section = unit_section
        packed_end = end
        if clause:
            packed_clauses.append(clause)
    flush_pack()
    return chunks


def split_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """Split legal clauses using regex word/punctuation units, not LLM tokens."""

    if not isinstance(text, str):
        raise TypeError("Текст должен быть строкой")
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть больше нуля")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap должен быть между 0 и chunk_size - 1")
    if not text.strip():
        return []

    return [
        text[start:end]
        for start, end, _, _ in _structured_chunks(text, chunk_size, chunk_overlap)
    ]


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    encoding_name: str = "cl100k_base",
) -> list[Document]:
    """Preserve clauses and attach source locations to each chunk."""

    chunks: list[Document] = []
    for document in documents:
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Некорректный размер чанка или перекрытие")
        spans = _structured_chunks(document.text, chunk_size, chunk_overlap)
        version_match = re.search(
            r"(?i)редакция\s+от\s+(\d{2}\.\d{2}\.\d{4})", document.text[:1500]
        )
        temporal = document_temporal_metadata(document.text)
        for chunk_index, (start, end, section, clause) in enumerate(spans):
            chunk_text = document.text[start:end]
            metadata = {
                **document.metadata,
                "chunk_index": chunk_index,
                "document_id": (
                    document.metadata.get("document_id")
                    or document.metadata.get("filename")
                    or document.metadata.get("source")
                ),
                "section": section,
                "clause": clause,
                "version": document.metadata.get("version") or (
                    version_match.group(1) if version_match else None
                ),
                **{
                    key: document.metadata.get(key) or value
                    for key, value in temporal.items()
                },
                "start_char": start,
                "end_char": end,
                "token_count": len(_get_encoding(encoding_name).findall(chunk_text)),
            }
            chunks.append(Document(text=chunk_text, metadata=metadata))
    return chunks


def rerank_documents(
    query: str,
    documents: Iterable[Document],
    *,
    reranker: Reranker | None = None,
    top_k: int | None = RERANKER_TOP_K,
) -> list[Document]:
    """...
    """

    documents_list = list(documents)
    client = reranker or Reranker()
    ranked = client.rerank(query, documents_list, top_k=top_k)
    return [document for document in ranked if isinstance(document, Document)]


def load_and_rerank(
    path: str | Path,
    query: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    top_k: int | None = RERANKER_TOP_K,
    reranker: Reranker | None = None,
) -> list[Document]:
    """..."""

    documents = split_documents(
        load_documents(path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return rerank_documents(
        query,
        documents,
        reranker=reranker,
        top_k=top_k,
    )


def load_and_index(
    path: str | Path,
    embedding_client: EmbeddingClient | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    metric: str = "cosine",
) -> FaissVectorIndex:
    """..."""

    documents = split_documents(
        load_documents(path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return embed_and_index(documents, embedding_client, metric=metric)


@dataclass(slots=True)
class HybridRanking:
    """The weighted RRF pool, reranker scores and final context."""

    reranker_input: list[Document]
    reranked: list[Document]
    final: list[Document]


def rank_hybrid_candidates(
    query: str,
    trace: HybridSearchTrace,
    *,
    reranker: Reranker | None,
    top_k: int = FINAL_CONTEXT_TOP_K,
    reranker_input_k: int = RERANKER_INPUT_K,
    reranker_output_k: int = RERANKER_TOP_K,
) -> HybridRanking:
    """Rerank the weighted RRF pool and pass its Top-N directly to the LLM."""

    if min(top_k, reranker_input_k, reranker_output_k) <= 0:
        raise ValueError("Context and reranker limits must be positive")

    reranker_input = trace.candidates[:reranker_input_k]
    if not reranker_input:
        return HybridRanking([], [], [])
    if reranker is None:
        final = [
            Document(
                doc.text,
                {**doc.metadata, "selection_reason": "rrf_fallback"},
                doc.score,
            )
            for doc in reranker_input[:top_k]
        ]
        return HybridRanking(reranker_input, [], final)
    if reranker_output_k < len(reranker_input):
        raise ValueError("Reranker must score every RRF candidate")

    copies = [
        Document(doc.text, dict(doc.metadata))
        for doc in reranker_input
    ]
    reranked = reranker.rerank(query, copies, top_k=reranker_output_k)
    expected = {str(doc.metadata["chunk_id"]) for doc in reranker_input}
    seen: set[str] = set()
    if len(expected) != len(reranker_input) or len(reranked) != len(reranker_input):
        raise ValueError("Reranker did not return all unique RRF candidates")
    for rank, doc in enumerate(reranked, start=1):
        if not isinstance(doc, Document):
            raise TypeError("Reranker returned an invalid document")
        chunk_id = str(doc.metadata.get("chunk_id"))
        if chunk_id not in expected or chunk_id in seen:
            raise ValueError("Reranker returned an unknown or duplicate chunk")
        if doc.score is None or not math.isfinite(float(doc.score)):
            raise ValueError("Reranker returned a non-finite or missing score")
        seen.add(chunk_id)
        doc.metadata["reranker_rank"] = rank
        doc.metadata["reranker_score"] = float(doc.score)
        if rank <= top_k:
            doc.metadata["selection_reason"] = "reranker_top_n"
    return HybridRanking(reranker_input, reranked, reranked[:top_k])

def search_and_rerank(
    query: str,
    index: FaissVectorIndex,
    embedding_client: EmbeddingClient | None = None,
    *,
    retrieve_k: int = FAISS_TOP_K,
    rrf_top_k: int = RRF_TOP_K,
    top_k: int = FINAL_CONTEXT_TOP_K,
    rerank_candidates: int = RERANKER_INPUT_K,
    reranker: Reranker | None = None,
    search_mode: str = "hybrid",
) -> list[Document]:
    """...
    """

    if retrieve_k <= 0:
        raise ValueError("retrieve_k должно быть больше нуля")
    if rrf_top_k <= 0:
        raise ValueError("rrf_top_k должно быть больше нуля")
    if top_k <= 0:
        raise ValueError("top_k должно быть больше нуля")
    if rerank_candidates <= 0:
        raise ValueError("rerank_candidates должно быть больше нуля")

    if search_mode == "hybrid":
        trace = index.hybrid_candidates(
            query,
            embedding_client,
            top_k=rrf_top_k,
            faiss_k=retrieve_k,
            bm25_k=retrieve_k,
        )
        return rank_hybrid_candidates(
            query,
            trace,
            reranker=reranker,
            top_k=top_k,
            reranker_input_k=rerank_candidates,
        ).final
    elif search_mode == "semantic":
        candidates = index.semantic_search(
            query, embedding_client, top_k=retrieve_k
        )
    elif search_mode == "bm25":
        candidates = index.bm25_search(query, top_k=retrieve_k)
    else:
        raise ValueError("search_mode должно быть 'hybrid', 'semantic', или 'bm25'")
    if reranker is None:
        return candidates[:top_k]
    return rerank_documents(
        query,
        candidates[:rerank_candidates],
        reranker=reranker,
        top_k=top_k,
    )


@dataclass(slots=True)
class RAGResult:
    """..."""

    query: str
    answer: str
    sources: list[Document]
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [
                {
                    "text": source.text,
                    "score": source.score,
                    "metadata": source.metadata,
                }
                for source in self.sources
            ],
            "debug": self.debug,
        }


class RAGSystem:
    """...
    """

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient | None = None,
        reranker: Reranker | None = None,
        llm: OpenAICompatibleLLM | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        metric: str = "cosine",
        search_mode: str = "hybrid",
        retrieve_k: int = FAISS_TOP_K,
        top_k: int = FINAL_CONTEXT_TOP_K,
        rerank_candidates: int | None = RERANKER_INPUT_K,
        reranker_output_k: int = RERANKER_TOP_K,
        rrf_top_k: int = RRF_TOP_K,
        semantic_weight: float = FAISS_WEIGHT,
        lexical_weight: float = BM25_WEIGHT,
    ) -> None:
        if search_mode not in {"hybrid", "semantic", "bm25"}:
            raise ValueError("search_mode должно быть 'hybrid', 'semantic', или 'bm25'")
        if retrieve_k <= 0 or top_k <= 0:
            raise ValueError("retrieve_k и top_k должны быть больше нуля")
        if rerank_candidates is not None and rerank_candidates <= 0:
            raise ValueError("rerank_candidates должен быть больше нуля")
        if reranker_output_k <= 0 or (
            rerank_candidates is not None and reranker_output_k < rerank_candidates
        ):
            raise ValueError("Reranker output must cover all reranker candidates")
        if rrf_top_k <= 0:
            raise ValueError("rrf_top_k должно быть больше нуля")
        if (
            not all(map(math.isfinite, (semantic_weight, lexical_weight)))
            or min(semantic_weight, lexical_weight) < 0
            or semantic_weight + lexical_weight == 0
        ):
            raise ValueError("Некорректные веса RRF")

        self.embedding_client = embedding_client or EmbeddingClient()
        self.reranker = reranker
        self.llm = llm or OpenAICompatibleLLM()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metric = metric
        self.search_mode = search_mode
        self.retrieve_k = retrieve_k
        self.top_k = top_k
        self.rerank_candidates = rerank_candidates
        self.reranker_output_k = reranker_output_k
        self.rrf_top_k = rrf_top_k
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.index = FaissVectorIndex(metric=metric)
        self.chunks: list[Document] = []

    @property
    def document_count(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        """..."""

        self.index = FaissVectorIndex(metric=self.metric)
        self.chunks = []

    def ingest(self, documents: Iterable[Document]) -> int:
        """..."""

        chunks = split_documents(
            list(documents),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            return 0

        import numpy as np

        digest = hashlib.sha256()
        digest.update(str(getattr(self.embedding_client, "model_name", "unknown")).encode("utf-8"))
        for chunk in chunks:
            encoded = chunk.text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        cache_dir = Path(
            os.getenv("RAG_EMBEDDING_CACHE_DIR", str(Path(__file__).with_name(".cache")))
        ).expanduser()
        cache_path = cache_dir / f"embeddings-v1-{digest.hexdigest()}.npy"
        embeddings = None
        if cache_path.is_file():
            try:
                cached = np.load(cache_path, allow_pickle=False)
                if (
                    cached.ndim == 2
                    and cached.shape[0] == len(chunks)
                    and cached.shape[1] > 0
                    and np.isfinite(cached).all()
                ):
                    embeddings = cached
                    print(f"Loaded {len(chunks)} cached embeddings", file=sys.stderr, flush=True)
            except (OSError, ValueError, TypeError) as exc:
                print(f"Embedding cache is unavailable: {exc}", file=sys.stderr)
        if embeddings is None:
            print(f"Embedding {len(chunks)} chunks...", file=sys.stderr, flush=True)
            embeddings = np.asarray(
                self.embedding_client.embed([chunk.text for chunk in chunks]),
                dtype=np.float32,
            )
            temporary: Path | None = None
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = cache_dir / f"{cache_path.stem}-{uuid.uuid4().hex}.npy"
                with temporary.open("xb") as handle:
                    np.save(handle, embeddings, allow_pickle=False)
                temporary.replace(cache_path)
            except OSError as exc:
                print(f"Could not save embedding cache: {exc}", file=sys.stderr)
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
        self.index.add(chunks, embeddings)
        self.chunks.extend(chunks)
        return len(chunks)

    def ingest_lexical(self, documents: Iterable[Document]) -> int:
        """..."""

        chunks = split_documents(
            list(documents),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            return 0
        self.index.add_lexical(chunks)
        self.chunks.extend(chunks)
        return len(chunks)

    def ingest_path(self, path: str | Path) -> int:
        """..."""

        return self.ingest(load_documents(path))

    def retrieve(
        self,
        question: str,
        *,
        search_mode: str | None = None,
        retrieve_k: int | None = None,
        top_k: int | None = None,
        rrf_top_k: int | None = None,
        use_reranker: bool = True,
    ) -> list[Document]:
        """..."""

        mode = search_mode or self.search_mode
        candidate_count = retrieve_k if retrieve_k is not None else self.retrieve_k
        result_count = top_k if top_k is not None else self.top_k
        fusion_count = rrf_top_k if rrf_top_k is not None else self.rrf_top_k
        if mode not in {"hybrid", "semantic", "bm25"}:
            raise ValueError("search_mode должно быть 'hybrid', 'semantic', или 'bm25'")
        if candidate_count <= 0 or result_count <= 0 or fusion_count <= 0:
            raise ValueError("retrieve_k, rrf_top_k и top_k должны быть больше нуля")
        if self.index.is_empty:
            return []

        if mode == "hybrid":
            trace = self.index.hybrid_candidates(
                question,
                self.embedding_client,
                top_k=fusion_count,
                faiss_k=candidate_count,
                bm25_k=candidate_count,
                semantic_weight=self.semantic_weight,
                lexical_weight=self.lexical_weight,
            )
            ranked = rank_hybrid_candidates(
                question,
                trace,
                reranker=self.reranker if use_reranker else None,
                top_k=result_count,
                reranker_input_k=self.rerank_candidates or fusion_count,
                reranker_output_k=self.reranker_output_k,
            )
            if use_reranker and self.reranker is not None:
                print(
                    f"Reranked {len(ranked.reranker_input)} candidates to "
                    f"{len(ranked.final)} context sources",
                    file=sys.stderr,
                    flush=True,
                )
            return ranked.final
        elif mode == "semantic":
            candidates = self.index.semantic_search(
                question,
                self.embedding_client,
                top_k=candidate_count,
            )
        else:
            candidates = self.index.bm25_search(
                question,
                top_k=candidate_count,
            )

        if use_reranker and self.reranker is not None and candidates:
            rerank_pool = candidates[:self.rerank_candidates]
            ranked = rerank_documents(
                question,
                rerank_pool,
                reranker=self.reranker,
                top_k=result_count,
            )
            print(
                f"Reranked {len(rerank_pool)} candidates to {len(ranked)} sources",
                file=sys.stderr,
                flush=True,
            )
            return ranked
        return candidates[:result_count]

    def answer(
        self,
        question: str,
        *,
        search_mode: str | None = None,
        retrieve_k: int | None = None,
        top_k: int | None = None,
        rrf_top_k: int | None = None,
        use_reranker: bool = True,
    ) -> RAGResult:
        """..."""

        sources = self.retrieve(
            question,
            search_mode=search_mode,
            retrieve_k=retrieve_k,
            top_k=top_k,
            rrf_top_k=rrf_top_k,
            use_reranker=use_reranker,
        )
        answer = self.llm.generate(question, sources)
        return RAGResult(
            query=question,
            answer=answer,
            sources=sources,
            debug={
                "indexed_chunks": self.document_count,
                "retrieved": len(sources),
                "search_mode": search_mode or self.search_mode,
                "rrf_pool": rrf_top_k if rrf_top_k is not None else self.rrf_top_k,
                "reranker_input_limit": self.rerank_candidates,
                "reranker_output_limit": self.reranker_output_k,
                "final_context_chunks": len(sources),
                "reranked": bool(use_reranker and self.reranker is not None),
            },
        )


def _platform_env(name: str, default: str = "") -> str:
    """..."""

    value = os.getenv(name, "").strip()
    if not value or value == name:
        return default
    return value


def _platform_first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = _platform_env(name)
        if value:
            return value
    return default


def _platform_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(_platform_env(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _platform_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(_platform_env(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _find_platform_file(data_dir: Path, *names: str) -> Path:
    """Find a platform input file without assuming the process working directory."""

    roots: list[Path] = [data_dir]
    module_dir = Path(__file__).resolve().parent
    roots.extend((module_dir / data_dir, Path.cwd() / data_dir, Path.cwd()))
    if data_dir.is_absolute():
        roots = [data_dir, module_dir / data_dir]

    checked: set[Path] = set()
    for root in roots:
        resolved_root = root.resolve()
        if resolved_root in checked:
            continue
        checked.add(resolved_root)
        for name in names:
            candidate = resolved_root / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"Could not find any of {', '.join(names)} under DATA_DIR={data_dir}"
    )


def _load_platform_questions(path: Path) -> list[tuple[str, str]]:
    """..."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [field.strip().casefold() for field in (reader.fieldnames or [])]
        if "id" not in fieldnames or "question" not in fieldnames:
            raise ValueError("questions.csv должны содержать столбцы id и question")

        id_field = reader.fieldnames[fieldnames.index("id")]
        question_field = reader.fieldnames[fieldnames.index("question")]
        questions: list[tuple[str, str]] = []
        for row in reader:
            identifier = str(row.get(id_field) or "").strip()
            question = str(row.get(question_field) or "")
            questions.append((identifier, question.strip()))
    if not questions or any(not identifier for identifier, _ in questions):
        raise ValueError("questions.csv должны содержать хотя бы один непустой id")
    return questions


def _single_line_answer(answer: Any) -> str:
    """..."""

    normalized = re.sub(r"\s+", " ", str(answer)).strip()
    return normalized or "Не удалось получить ответ."


def run_platform_submission() -> Path:
    """..."""

    data_dir = Path(_platform_env("DATA_DIR", "data"))
    questions_override = _platform_env("QUESTIONS_PATH")
    questions_path = (
        Path(questions_override).expanduser().resolve()
        if questions_override
        else _find_platform_file(data_dir, "questions.csv")
    )
    knowledge_base_override = _platform_env("KNOWLEDGE_BASE_PATH")
    knowledge_base_path = (
        Path(knowledge_base_override).expanduser().resolve()
        if knowledge_base_override
        else _find_platform_file(
            data_dir,
            "kb.zip",
            "knowledge-base.zip",
            "knowledge_base.zip",
        )
    )
    if not questions_path.is_file():
        raise FileNotFoundError(f"Файл вопросов не найден: {questions_path}")
    if not knowledge_base_path.is_file():
        raise FileNotFoundError(f"База знаний не найдена: {knowledge_base_path}")
    questions = _load_platform_questions(questions_path)
    documents = load_documents(knowledge_base_path)

    retrieve_k = _platform_int("TOP_K", FAISS_TOP_K, 1, 75)
    rrf_top_k = _platform_int("RRF_TOP_K", RRF_TOP_K, 1, 100)
    reranker_candidates = _platform_int(
        "RERANKER_CANDIDATES",
        RERANKER_INPUT_K,
        1,
        rrf_top_k,
    )
    reranker_top_n = _platform_int(
        "RERANKER_TOP_N", RERANKER_TOP_K, reranker_candidates, 100
    )
    final_context_top_k = _platform_int(
        "FINAL_CONTEXT_TOP_K", FINAL_CONTEXT_TOP_K, 1, rrf_top_k
    )
    use_reranker = _platform_env("USE_RERANKER", "1").casefold() in {"1", "true", "yes"}
    llm = OpenAICompatibleLLM(
        model=_platform_first_env(
            "LLM_MODEL",
            "GEN_MODEL",
            "OPENAI_MODEL",
            default="deepseek/deepseek-v4.1-flash",
        ),
        temperature=_platform_float("TEMPERATURE", 0.0, 0.0, 2.0),
        max_tokens=_platform_int("MAX_TOKENS", 1024, 128, 8192),
    )
    rag = RAGSystem(
        llm=llm,
        reranker=Reranker() if use_reranker else None,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        search_mode="hybrid",
        retrieve_k=retrieve_k,
        top_k=final_context_top_k,
        rerank_candidates=reranker_candidates,
        reranker_output_k=reranker_top_n,
        rrf_top_k=rrf_top_k,
        semantic_weight=_platform_float("RRF_SEMANTIC_WEIGHT", FAISS_WEIGHT, 0.0, 1.0),
        lexical_weight=_platform_float("RRF_BM25_WEIGHT", BM25_WEIGHT, 0.0, 1.0),
    )

    try:
        indexed = rag.ingest(documents)
        print(f"Индексировано {indexed} фрагментов с встраиванием", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - preserve the BM25 fallback
        # The README requires embedding failures to degrade to a simpler search.
        print(f"Встраивание недоступно; использование fallback BM25: {exc}", file=sys.stderr)
        rag.clear()
        indexed = rag.ingest_lexical(documents)
        print(f"Индексировано {indexed} фрагментов с BM25", file=sys.stderr)

    # Keep the diagnostic pass and answer generation on the same retrieval
    # result. This avoids embedding/reranking every question twice.
    from .diagnostic import _column_names as _diagnostic_column_names
    from .diagnostic import _documents_json, collect_diagnostic_result

    diagramm_faiss_top_k = retrieve_k
    diagramm_bm25_top_k = retrieve_k
    diagramm_rrf_top_k = rrf_top_k
    diagramm_reranker_input_k = reranker_candidates
    diagramm_reranker_top_k = reranker_top_n
    diagramm_final_context_top_k = final_context_top_k
    diagramm_columns = list(
        _diagnostic_column_names(
            diagramm_faiss_top_k,
            diagramm_bm25_top_k,
            diagramm_rrf_top_k,
            diagramm_reranker_input_k,
            diagramm_reranker_top_k,
            diagramm_final_context_top_k,
        )
    )

    output_path = Path(_platform_env("OUTPUT_PATH", "submission.csv")).expanduser().resolve()
    diagramm_override = _platform_env("DIAGRAMM_OUTPUT")
    diagramm_path = (
        Path(diagramm_override).expanduser().resolve()
        if diagramm_override
        else output_path.with_name("diagramm.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagramm_path.parent.mkdir(parents=True, exist_ok=True)

    def empty_diagramm_row(identifier: str, question: str, error: str) -> dict[str, str]:
        row = {
            "id": identifier,
            "question": question,
            "Final Context": "",
            "error": error,
        }
        row.update({column: "[]" for column in diagramm_columns[2:7]})
        return row

    def record_actual_context(
        row: dict[str, str], sources: list[Document], error: str
    ) -> dict[str, str]:
        row[diagramm_columns[7]] = _documents_json(sources)
        row["Final Context"] = OpenAICompatibleLLM._source_context(sources)
        row["error"] = "\n".join(filter(None, (row.get("error"), error)))
        return row

    def answer_one(
        position: int,
        item: tuple[str, str],
    ) -> tuple[int, str, dict[str, str]]:
        identifier, question = item
        print(
            f"Question {position + 1}/{len(questions)}: retrieving"
            + (" and reranking..." if use_reranker else "..."),
            file=sys.stderr,
            flush=True,
        )
        diagramm_row: dict[str, str] | None = None
        diagnostic_error = ""
        hybrid_error = ""
        try:
            diagnostic = collect_diagnostic_result(
                identifier,
                question,
                rag.index,
                rag.embedding_client,
                rag.reranker if use_reranker else None,
                faiss_top_k=diagramm_faiss_top_k,
                bm25_top_k=diagramm_bm25_top_k,
                rrf_top_k=diagramm_rrf_top_k,
                reranker_input_top_k=diagramm_reranker_input_k,
                reranker_top_k=diagramm_reranker_top_k,
                final_context_top_k=diagramm_final_context_top_k,
                rrf_semantic_weight=rag.semantic_weight,
                rrf_bm25_weight=rag.lexical_weight,
            )
            diagramm_row = diagnostic.row
            print(
                f"Question {position + 1}/{len(questions)}: calling LLM with "
                f"{len(diagnostic.final_hits)} chunks",
                file=sys.stderr,
                flush=True,
            )
            answer = rag.llm.generate(question, diagnostic.final_hits)
            return position, _single_line_answer(answer), diagramm_row
        except Exception as exc:  # noqa: BLE001 - continue through fallback stages
            diagnostic_error = str(exc)
            print(f"Question {position}: RAG request failed: {exc}", file=sys.stderr)
            if use_reranker:
                print(f"Question {position}: retrying without reranker", file=sys.stderr)

        try:
            sources = rag.retrieve(
                question,
                search_mode="hybrid",
                retrieve_k=retrieve_k,
                top_k=diagramm_final_context_top_k,
                rrf_top_k=rrf_top_k,
                use_reranker=False,
            )
            answer = rag.llm.generate(question, sources)
            row = diagramm_row or empty_diagramm_row(identifier, question, "")
            return position, _single_line_answer(answer), record_actual_context(
                row, sources, diagnostic_error
            )
        except Exception as exc:  # noqa: BLE001 - continue through fallback stages
            hybrid_error = str(exc)
            print(f"Question {position}: fallback hybrid request failed: {exc}", file=sys.stderr)

        try:
            sources = rag.retrieve(
                question,
                search_mode="bm25",
                retrieve_k=retrieve_k,
                top_k=diagramm_final_context_top_k,
                rrf_top_k=rrf_top_k,
                use_reranker=False,
            )
            answer = rag.llm.generate(question, sources)
            row = diagramm_row or empty_diagramm_row(
                identifier, question, diagnostic_error
            )
            return position, _single_line_answer(answer), record_actual_context(
                row, sources, hybrid_error
            )
        except Exception as exc:  # noqa: BLE001 - return the failed row to the caller
            print(f"Question {position}: fallback request failed: {exc}", file=sys.stderr)
            return (
                position,
                "Не удалось получить ответ.",
                diagramm_row
                or empty_diagramm_row(
                    identifier,
                    question,
                    diagnostic_error or hybrid_error or str(exc),
                ),
            )

    answers = ["Не удалось получить ответ."] * len(questions)
    diagramm_rows: list[dict[str, str] | None] = [None] * len(questions)
    max_workers = _platform_int("MAX_WORKERS", 1, 1, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(answer_one, position, item)
            for position, item in enumerate(questions)
        ]
        for future in as_completed(futures):
            position, answer, diagramm_row = future.result()
            answers[position] = answer
            diagramm_rows[position] = diagramm_row

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for (identifier, _), answer in zip(questions, answers):
            writer.writerow([identifier, answer])
    print(f"Записано {len(answers)} ответов в {output_path}", file=sys.stderr)

    with diagramm_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=diagramm_columns,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writeheader()
        for position, (identifier, question) in enumerate(questions):
            row = diagramm_rows[position] or empty_diagramm_row(
                identifier,
                question,
                "Диагностические данные не были собраны.",
            )
            writer.writerow({column: row.get(column, "") for column in diagramm_columns})
    print(f"Диагностика записана в {diagramm_path}", file=sys.stderr)
    return output_path


def main() -> None:
    run_platform_submission()


if __name__ == "__main__":
    main()
