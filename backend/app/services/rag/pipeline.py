from __future__ import annotations

import csv
from dataclasses import dataclass, field
import io
import math
import os
import re
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import zipfile
from xml.etree import ElementTree as ET

from . import (
    BM25_PROTECTED_TOP_K,
    BM25_CONFIRM_FAISS_TOP_K,
    BM25_CONFIRM_RERANKER_SCORE,
    BM25_CONFIRM_RRF_TOP_K,
    BM25_WEIGHT,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    FAISS_PROTECTED_TOP_K,
    FAISS_TOP_K,
    FAISS_WEIGHT,
    FINAL_CONTEXT_TOP_K,
    MAX_CONTEXT_TOKENS,
    MIN_CONTEXT_RELATIVE_SCORE,
    MIN_CONTEXT_SCORE,
    RERANKER_INPUT_K,
    RERANKER_TOP_K,
    RERANKER_WEIGHT,
    RRF_TOP_K,
    RRF_WEIGHT,
    SOURCE_DIVERSITY_PENALTY,
    SUPPORTED_EXTENSIONS,
    Document,
)
from .embedding import EmbeddingClient
from .llm import OpenAICompatibleLLM
from .relevance import (
    count_model_tokens,
    document_temporal_metadata,
    exact_legal_reference_match,
    question_reference_date,
    suspected_prompt_injection,
    temporal_status,
)
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
            flags=re.S | re.I,
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
    """The Qwen input/output and the final protected context selection."""

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
    faiss_weight: float = FAISS_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    rrf_weight: float = RRF_WEIGHT,
    reranker_weight: float = RERANKER_WEIGHT,
    bm25_confirm_faiss_k: int = BM25_CONFIRM_FAISS_TOP_K,
    bm25_confirm_rrf_k: int = BM25_CONFIRM_RRF_TOP_K,
    bm25_confirm_reranker_score: float = BM25_CONFIRM_RERANKER_SCORE,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    min_context_score: float = MIN_CONTEXT_SCORE,
    min_context_relative_score: float = MIN_CONTEXT_RELATIVE_SCORE,
    source_diversity_penalty: float = SOURCE_DIVERSITY_PENALTY,
    tokenizer_model: str = "gpt-4o",
    token_counter: Callable[[str], int] | None = None,
) -> HybridRanking:
    """Confirm BM25 protection and select a bounded, variable-size context."""

    if min(top_k, reranker_input_k, reranker_output_k) <= 0:
        raise ValueError("Context and reranker limits must be positive")
    weights = (faiss_weight, bm25_weight, rrf_weight, reranker_weight)
    if not all(math.isfinite(weight) and weight >= 0 for weight in weights) or not any(weights):
        raise ValueError("Final ranking weights must be finite and nonnegative")
    if min(bm25_confirm_faiss_k, bm25_confirm_rrf_k, max_context_tokens) <= 0:
        raise ValueError("Confirmation ranks and context token budget must be positive")
    if not math.isfinite(bm25_confirm_reranker_score):
        raise ValueError("BM25 reranker threshold must be finite")
    if (
        not math.isfinite(min_context_score) or min_context_score < 0
        or not 0 <= min_context_relative_score <= 1
        or not math.isfinite(source_diversity_penalty) or source_diversity_penalty < 0
    ):
        raise ValueError("Invalid adaptive context parameters")

    candidates = trace.candidates
    if not candidates:
        return HybridRanking([], [], [])
    pool_protected = {
        str(document.metadata["chunk_id"])
        for document in candidates if document.metadata.get("protected_by")
    }
    if len(pool_protected) > reranker_input_k:
        raise ValueError("Reranker pool is too small for protected source candidates")

    selected_input = set(pool_protected)
    for document in candidates:
        if len(selected_input) >= reranker_input_k:
            break
        selected_input.add(str(document.metadata["chunk_id"]))
    reranker_input = [
        document for document in candidates
        if str(document.metadata["chunk_id"]) in selected_input
    ]
    if reranker is not None and reranker_output_k < len(reranker_input):
        raise ValueError("Reranker must score every input candidate")

    ranked_by_qwen: list[Document] = []
    raw_scores: dict[str, float] = {}
    reranker_ranks: dict[str, int] = {}
    if reranker is not None and reranker_input:
        rerank_copies = [
            Document(text=doc.text, metadata=dict(doc.metadata))
            for doc in reranker_input
        ]
        output = reranker.rerank(query, rerank_copies, top_k=reranker_output_k)
        if len(output) != len(reranker_input) or any(
            not isinstance(doc, Document) for doc in output
        ):
            raise ValueError("Reranker did not return all input candidates")
        expected = {str(doc.metadata["chunk_id"]) for doc in reranker_input}
        for position, doc in enumerate(output, start=1):
            chunk_id = str(doc.metadata.get("chunk_id"))
            if chunk_id not in expected or chunk_id in reranker_ranks:
                raise ValueError("Reranker returned an unknown or duplicate chunk")
            if doc.score is None or not math.isfinite(float(doc.score)):
                raise ValueError("Reranker returned a non-finite or missing score")
            raw_scores[chunk_id] = float(doc.score)
            reranker_ranks[chunk_id] = position
            ranked_by_qwen.append(doc)

    # Qwen returns unbounded logits, so normalize only within this question.
    # A constant set of scores carries no ordering information (neutral 0.5).
    score_min = min(raw_scores.values()) if raw_scores else 0.0
    score_max = max(raw_scores.values()) if raw_scores else 0.0
    normalized_qwen = {
        chunk_id: (score - score_min) / (score_max - score_min)
        if score_max > score_min else 0.5
        for chunk_id, score in raw_scores.items()
    }

    scored: list[Document] = []
    by_id: dict[str, Document] = {}
    as_of = question_reference_date(query)
    for document in candidates:
        metadata = dict(document.metadata)
        chunk_id = str(metadata["chunk_id"])
        faiss_rank = metadata.get("faiss_rank")
        bm25_rank = metadata.get("bm25_rank")
        # Rank 1 has reciprocal score 1/2; dividing by that maximum yields
        # independent FAISS/BM25 signals in [0, 1] before 40/40 weighting.
        faiss_signal = 2 / (1 + faiss_rank) if faiss_rank is not None else 0.0
        bm25_signal = 2 / (1 + bm25_rank) if bm25_rank is not None else 0.0
        base_score = (faiss_signal + bm25_signal) / 2
        combined_score = (
            faiss_weight * faiss_signal
            + bm25_weight * bm25_signal
            + rrf_weight * float(metadata["rrf_normalized"])
            + reranker_weight * normalized_qwen.get(chunk_id, 0.0)
        )
        status = temporal_status(as_of, metadata)
        temporal_adjustment = (
            0.08 if status == "active" else
            -0.25 if status in {"future", "expired"} else 0.0
        )
        final_score = combined_score + temporal_adjustment
        exact_match = exact_legal_reference_match(query, document.text, metadata)
        injection_suspected = suspected_prompt_injection(document.text)
        bm25_confirmation: list[str] = []
        if "bm25" in metadata.get("protected_by", []):
            if faiss_rank is not None and faiss_rank <= bm25_confirm_faiss_k:
                bm25_confirmation.append("faiss_top")
            if metadata["rrf_rank"] <= bm25_confirm_rrf_k:
                bm25_confirmation.append("rrf_top")
            if raw_scores.get(chunk_id, float("-inf")) >= bm25_confirm_reranker_score:
                bm25_confirmation.append("reranker_score")
            if exact_match:
                bm25_confirmation.append("exact_legal_reference")
        metadata.update(
            faiss_normalized=faiss_signal,
            bm25_normalized=bm25_signal,
            base_score=base_score,
            reranker_rank=reranker_ranks.get(chunk_id),
            reranker_score=raw_scores.get(chunk_id),
            reranker_normalized=normalized_qwen.get(chunk_id),
            combined_score=combined_score,
            bm25_confirmation=bm25_confirmation,
            exact_legal_reference_match=exact_match,
            prompt_injection_suspected=injection_suspected,
            question_as_of=as_of.isoformat() if as_of else None,
            temporal_status=status,
            temporal_adjustment=temporal_adjustment,
            selected_by_protection=False,
            selection_reason=None,
            final_score=final_score,
        )
        scored_doc = Document(text=document.text, metadata=metadata, score=final_score)
        scored.append(scored_doc)
        by_id[chunk_id] = scored_doc

    scored.sort(
        key=lambda doc: (
            -float(doc.score),
            doc.metadata["rrf_rank"],
            str(doc.metadata["chunk_id"]),
        )
    )

    def source_key(document: Document) -> str:
        metadata = document.metadata
        return str(
            metadata.get("document_id") or metadata.get("filename")
            or metadata.get("source") or metadata["chunk_id"]
        )

    eligible: list[Document] = []
    faiss_protected: list[Document] = []
    bm25_confirmed: list[Document] = []
    seen_passages: set[tuple[str, str]] = set()
    for document in scored:
        metadata = document.metadata
        if metadata["prompt_injection_suspected"]:
            metadata["selection_reason"] = "prompt_injection_suspected"
            continue
        if "bm25" in metadata.get("protected_by", []) and not metadata["bm25_confirmation"]:
            metadata["selection_reason"] = "bm25_unconfirmed"
            continue
        passage = (source_key(document), re.sub(r"\s+", " ", document.text).strip().casefold())
        if passage in seen_passages:
            metadata["selection_reason"] = "duplicate_fragment"
            continue
        seen_passages.add(passage)
        eligible.append(document)
        if "faiss" in metadata.get("protected_by", []) and metadata["temporal_status"] not in {"future", "expired"}:
            faiss_protected.append(document)
        elif "bm25" in metadata.get("protected_by", []) and metadata["bm25_confirmation"] and metadata["temporal_status"] not in {"future", "expired"}:
            bm25_confirmed.append(document)

    if len(faiss_protected) + len(bm25_confirmed) > top_k:
        raise ValueError("Final context is too small for confirmed protected hits")
    best_score = max((float(doc.score) for doc in eligible), default=0.0)
    score_floor = max(min_context_score, min_context_relative_score * best_score)
    count_tokens = token_counter or (
        lambda text: count_model_tokens(text, tokenizer_model)
    )
    selected: list[Document] = []
    selected_ids: set[str] = set()
    source_counts: dict[str, int] = {}

    def add_to_context(document: Document, reason: str, protected: bool) -> bool:
        trial = sorted(
            selected + [document],
            key=lambda doc: (-float(doc.score), doc.metadata["rrf_rank"]),
        )
        if count_tokens(OpenAICompatibleLLM._source_context(trial)) > max_context_tokens:
            document.metadata["selection_reason"] = "token_budget_exceeded"
            return False
        selected.append(document)
        selected_ids.add(str(document.metadata["chunk_id"]))
        source = source_key(document)
        source_counts[source] = source_counts.get(source, 0) + 1
        document.metadata["selection_reason"] = reason
        document.metadata["selected_by_protection"] = protected
        return True

    for document in sorted(faiss_protected, key=lambda doc: doc.metadata["faiss_rank"]):
        if len(selected) >= top_k:
            break
        add_to_context(document, "faiss_protected", True)
    for document in sorted(bm25_confirmed, key=lambda doc: doc.metadata["bm25_rank"]):
        if len(selected) >= top_k:
            break
        add_to_context(document, "bm25_confirmed", True)

    protected_ids = {
        str(doc.metadata["chunk_id"]) for doc in faiss_protected + bm25_confirmed
    }
    remaining = [
        doc for doc in eligible
        if str(doc.metadata["chunk_id"]) not in selected_ids
        and str(doc.metadata["chunk_id"]) not in protected_ids
    ]
    while len(selected) < top_k and remaining:
        document = max(
            remaining,
            key=lambda doc: (
                float(doc.score) - source_diversity_penalty * source_counts.get(source_key(doc), 0),
                -doc.metadata["rrf_rank"],
            ),
        )
        remaining.remove(document)
        if float(document.score) < score_floor:
            document.metadata["selection_reason"] = "below_relevance_threshold"
            continue
        reason = "temporal_relevance" if document.metadata["temporal_status"] == "active" else "combined_score"
        add_to_context(document, reason, False)

    final = sorted(selected, key=lambda doc: (-float(doc.score), doc.metadata["rrf_rank"]))

    reranked = []
    for document in ranked_by_qwen:
        chunk_id = str(document.metadata["chunk_id"])
        metadata = dict(by_id[chunk_id].metadata)
        reranked.append(Document(document.text, metadata, raw_scores[chunk_id]))
    return HybridRanking(reranker_input, reranked, final)


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
        faiss_protected_k: int = FAISS_PROTECTED_TOP_K,
        bm25_protected_k: int = BM25_PROTECTED_TOP_K,
        faiss_weight: float = FAISS_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        rrf_weight: float = RRF_WEIGHT,
        reranker_weight: float = RERANKER_WEIGHT,
        bm25_confirm_faiss_k: int = BM25_CONFIRM_FAISS_TOP_K,
        bm25_confirm_rrf_k: int = BM25_CONFIRM_RRF_TOP_K,
        bm25_confirm_reranker_score: float = BM25_CONFIRM_RERANKER_SCORE,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
        min_context_score: float = MIN_CONTEXT_SCORE,
        min_context_relative_score: float = MIN_CONTEXT_RELATIVE_SCORE,
        source_diversity_penalty: float = SOURCE_DIVERSITY_PENALTY,
        token_counter: Callable[[str], int] | None = None,
        semantic_weight: float = 1.0,
        lexical_weight: float = 1.0,
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
        if min(faiss_protected_k, bm25_protected_k) < 0:
            raise ValueError("Protected Top-K cannot be negative")
        final_weights = (faiss_weight, bm25_weight, rrf_weight, reranker_weight)
        if not all(math.isfinite(weight) and weight >= 0 for weight in final_weights) or not any(final_weights):
            raise ValueError("Final ranking weights must be finite and nonnegative")
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
        self.faiss_protected_k = faiss_protected_k
        self.bm25_protected_k = bm25_protected_k
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
        self.rrf_weight = rrf_weight
        self.reranker_weight = reranker_weight
        self.bm25_confirm_faiss_k = bm25_confirm_faiss_k
        self.bm25_confirm_rrf_k = bm25_confirm_rrf_k
        self.bm25_confirm_reranker_score = bm25_confirm_reranker_score
        self.max_context_tokens = max_context_tokens
        self.min_context_score = min_context_score
        self.min_context_relative_score = min_context_relative_score
        self.source_diversity_penalty = source_diversity_penalty
        self.token_counter = token_counter
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

        embeddings = self.embedding_client.embed([chunk.text for chunk in chunks])
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
                faiss_protected_k=self.faiss_protected_k,
                bm25_protected_k=self.bm25_protected_k,
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
                faiss_weight=self.faiss_weight,
                bm25_weight=self.bm25_weight,
                rrf_weight=self.rrf_weight,
                reranker_weight=self.reranker_weight,
                bm25_confirm_faiss_k=self.bm25_confirm_faiss_k,
                bm25_confirm_rrf_k=self.bm25_confirm_rrf_k,
                bm25_confirm_reranker_score=self.bm25_confirm_reranker_score,
                max_context_tokens=self.max_context_tokens,
                min_context_score=self.min_context_score,
                min_context_relative_score=self.min_context_relative_score,
                source_diversity_penalty=self.source_diversity_penalty,
                tokenizer_model=getattr(self.llm, "model", "gpt-4o"),
                token_counter=self.token_counter,
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
    questions_path = _find_platform_file(data_dir, "questions.csv")
    knowledge_base_path = _find_platform_file(
        data_dir,
        "kb.zip",
        "knowledge-base.zip",
        "knowledge_base.zip",
    )
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
        faiss_protected_k=_platform_int("FAISS_PROTECTED_TOP_K", FAISS_PROTECTED_TOP_K, 0, retrieve_k),
        bm25_protected_k=_platform_int("BM25_PROTECTED_TOP_K", BM25_PROTECTED_TOP_K, 0, retrieve_k),
        faiss_weight=_platform_float("FAISS_WEIGHT", FAISS_WEIGHT, 0.0, 1.0),
        bm25_weight=_platform_float("BM25_WEIGHT", BM25_WEIGHT, 0.0, 1.0),
        rrf_weight=_platform_float("RRF_WEIGHT", RRF_WEIGHT, 0.0, 1.0),
        reranker_weight=_platform_float("RERANKER_WEIGHT", RERANKER_WEIGHT, 0.0, 1.0),
        bm25_confirm_faiss_k=_platform_int("BM25_CONFIRM_FAISS_TOP_K", BM25_CONFIRM_FAISS_TOP_K, 1, retrieve_k),
        bm25_confirm_rrf_k=_platform_int("BM25_CONFIRM_RRF_TOP_K", BM25_CONFIRM_RRF_TOP_K, 1, rrf_top_k),
        bm25_confirm_reranker_score=_platform_float("BM25_CONFIRM_RERANKER_SCORE", BM25_CONFIRM_RERANKER_SCORE, -20.0, 20.0),
        max_context_tokens=_platform_int("MAX_CONTEXT_TOKENS", MAX_CONTEXT_TOKENS, 256, 200_000),
        min_context_score=_platform_float("MIN_CONTEXT_SCORE", MIN_CONTEXT_SCORE, 0.0, 1.0),
        min_context_relative_score=_platform_float("MIN_CONTEXT_RELATIVE_SCORE", MIN_CONTEXT_RELATIVE_SCORE, 0.0, 1.0),
        source_diversity_penalty=_platform_float("SOURCE_DIVERSITY_PENALTY", SOURCE_DIVERSITY_PENALTY, 0.0, 1.0),
        semantic_weight=_platform_float("RRF_SEMANTIC_WEIGHT", 1.0, 0.0, 5.0),
        lexical_weight=_platform_float("RRF_BM25_WEIGHT", 1.0, 0.0, 5.0),
    )

    try:
        indexed = rag.ingest(documents)
        print(f"Индексировано {indexed} фрагментов с встраиванием", file=sys.stderr)
    except Exception as exc:
        # The README requires embedding failures to degrade to a simpler search.
        print(f"Встраивание недоступно; использование fallback BM25: {exc}", file=sys.stderr)
        rag.clear()
        indexed = rag.ingest_lexical(documents)
        print(f"Индексировано {indexed} фрагментов с BM25", file=sys.stderr)

    def answer_one(position: int, item: tuple[str, str]) -> tuple[int, str]:
        identifier, question = item
        del identifier  # The ID is written by the caller in the original order.
        print(
            f"Question {position + 1}/{len(questions)}: retrieving"
            + (" and reranking..." if use_reranker else "..."),
            file=sys.stderr,
            flush=True,
        )
        try:
            result = rag.answer(
                question,
                search_mode="hybrid",
                retrieve_k=retrieve_k,
                top_k=final_context_top_k,
                rrf_top_k=rrf_top_k,
                use_reranker=use_reranker,
            )
            return position, _single_line_answer(result.answer)
        except Exception as exc:
            print(f"Question {position}: RAG request failed: {exc}", file=sys.stderr)
            if use_reranker:
                print(f"Question {position}: retrying without reranker", file=sys.stderr)

        try:
            sources = rag.retrieve(
                question,
                search_mode="hybrid",
                retrieve_k=retrieve_k,
                top_k=final_context_top_k,
                rrf_top_k=rrf_top_k,
                use_reranker=False,
            )
            return position, _single_line_answer(rag.llm.generate(question, sources))
        except Exception as exc:
            print(f"Question {position}: fallback hybrid request failed: {exc}", file=sys.stderr)

        try:
            sources = rag.retrieve(
                question,
                search_mode="bm25",
                retrieve_k=retrieve_k,
                top_k=final_context_top_k,
                rrf_top_k=rrf_top_k,
                use_reranker=False,
            )
            return position, _single_line_answer(rag.llm.generate(question, sources))
        except Exception as exc:
            print(f"Question {position}: fallback request failed: {exc}", file=sys.stderr)
            return position, "Не удалось получить ответ."

    answers = ["Не удалось получить ответ."] * len(questions)
    max_workers = _platform_int("MAX_WORKERS", 1, 1, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(answer_one, position, item)
            for position, item in enumerate(questions)
        ]
        for future in as_completed(futures):
            position, answer = future.result()
            answers[position] = answer

    output_path = Path(_platform_env("OUTPUT_PATH", "submission.csv")).expanduser().resolve()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for (identifier, _), answer in zip(questions, answers):
            writer.writerow([identifier, answer])
    print(f"Записано {len(answers)} ответов в {output_path}", file=sys.stderr)
    return output_path


def main() -> None:
    run_platform_submission()


if __name__ == "__main__":
    main()
