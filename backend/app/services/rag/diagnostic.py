"""Export every retrieval stage of the RAG service to a CSV file.

The generated CSV contains one row per question.  Retrieval results are stored
as JSON arrays in the corresponding columns so that both the ranked text and
its metadata/score remain available for analysis.

Run from the repository root::

    python -m app.services.rag.diagnostic

The script does not call the LLM. ``Final Context`` is the exact context that
would be passed to it after conditional protection and adaptive selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from collections.abc import Callable
from typing import Any, Iterable, Sequence


if __package__ in {None, ""}:
    # Also support running this file directly from the repository root.
    package_parent = str(Path(__file__).resolve().parents[3])
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    from app.services.rag import (
        BM25_CONFIRM_FAISS_TOP_K,
        BM25_CONFIRM_RERANKER_SCORE,
        BM25_CONFIRM_RRF_TOP_K,
        BM25_PROTECTED_TOP_K,
        BM25_TOP_K,
        BM25_WEIGHT,
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
        Document,
    )
    from app.services.rag.embedding import EmbeddingClient
    from app.services.rag.llm import OpenAICompatibleLLM
    from app.services.rag.pipeline import (
        _find_platform_file,
        _load_platform_questions,
        _platform_float,
        _platform_int,
        load_documents,
        rank_hybrid_candidates,
        split_documents,
    )
    from app.services.rag.reranker import Reranker
    from app.services.rag.vector_store import FaissVectorIndex, embed_and_index
else:
    from . import (
        BM25_CONFIRM_FAISS_TOP_K,
        BM25_CONFIRM_RERANKER_SCORE,
        BM25_CONFIRM_RRF_TOP_K,
        BM25_PROTECTED_TOP_K,
        BM25_TOP_K,
        BM25_WEIGHT,
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
        Document,
    )
    from .embedding import EmbeddingClient
    from .llm import OpenAICompatibleLLM
    from .pipeline import (
        _find_platform_file,
        _load_platform_questions,
        _platform_float,
        _platform_int,
        load_documents,
        rank_hybrid_candidates,
        split_documents,
    )
    from .reranker import Reranker
    from .vector_store import FaissVectorIndex, embed_and_index


def _column_names(
    faiss_top_k: int,
    bm25_top_k: int,
    rrf_top_k: int,
    reranker_input_top_k: int,
    reranker_top_k: int,
    final_context_top_k: int,
) -> tuple[str, ...]:
    return (
        "id",
        "question",
        f"FAISS Top-{faiss_top_k}",
        f"BM25 Top-{bm25_top_k}",
        f"RRF Top-{rrf_top_k}",
        f"Qwen Reranker Input Top-{reranker_input_top_k}",
        f"Qwen Reranker Top-{reranker_top_k}",
        f"Final Top-{final_context_top_k}",
        "Final Context",
        "error",
    )


CSV_COLUMNS = _column_names(
    FAISS_TOP_K,
    BM25_TOP_K,
    RRF_TOP_K,
    RERANKER_INPUT_K,
    RERANKER_TOP_K,
    FINAL_CONTEXT_TOP_K,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _short_error(stage: str, error: Exception) -> str:
    return f"{stage}: {type(error).__name__}: {error}"


def _document_to_dict(document: Document, rank: int) -> dict[str, Any]:
    """Return a JSON-serializable representation of one ranked chunk."""

    return {
        "rank": rank,
        "score": document.score,
        "text": document.text,
        "metadata": document.metadata,
    }


def _documents_json(documents: Iterable[Document]) -> str:
    payload = [
        _document_to_dict(document, rank)
        for rank, document in enumerate(documents, start=1)
    ]
    return json.dumps(payload, ensure_ascii=False, default=str)


def collect_diagnostic_row(
    identifier: str,
    question: str,
    index: FaissVectorIndex,
    embedding_client: EmbeddingClient,
    reranker: Reranker | None,
    *,
    faiss_top_k: int = FAISS_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    rrf_top_k: int = RRF_TOP_K,
    reranker_input_top_k: int = RERANKER_INPUT_K,
    reranker_top_k: int = RERANKER_TOP_K,
    final_context_top_k: int = FINAL_CONTEXT_TOP_K,
    faiss_protected_top_k: int = FAISS_PROTECTED_TOP_K,
    bm25_protected_top_k: int = BM25_PROTECTED_TOP_K,
    faiss_weight: float = FAISS_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    rrf_weight: float = RRF_WEIGHT,
    reranker_weight: float = RERANKER_WEIGHT,
    rrf_semantic_weight: float = 1.0,
    rrf_bm25_weight: float = 1.0,
    bm25_confirm_faiss_k: int = BM25_CONFIRM_FAISS_TOP_K,
    bm25_confirm_rrf_k: int = BM25_CONFIRM_RRF_TOP_K,
    bm25_confirm_reranker_score: float = BM25_CONFIRM_RERANKER_SCORE,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    min_context_score: float = MIN_CONTEXT_SCORE,
    min_context_relative_score: float = MIN_CONTEXT_RELATIVE_SCORE,
    source_diversity_penalty: float = SOURCE_DIVERSITY_PENALTY,
    tokenizer_model: str = "gpt-4o",
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, str]:
    """Run all requested stages for one question and prepare one CSV row."""

    faiss_hits: list[Document] = []
    bm25_hits: list[Document] = []
    rrf_hits: list[Document] = []
    reranker_input: list[Document] = []
    reranked_hits: list[Document] = []
    final_hits: list[Document] = []
    errors: list[str] = []

    try:
        trace = index.hybrid_candidates(
            question,
            embedding_client,
            top_k=rrf_top_k,
            faiss_k=faiss_top_k,
            bm25_k=bm25_top_k,
            faiss_protected_k=faiss_protected_top_k,
            bm25_protected_k=bm25_protected_top_k,
            semantic_weight=rrf_semantic_weight,
            lexical_weight=rrf_bm25_weight,
        )
        faiss_hits = trace.faiss_hits
        bm25_hits = trace.bm25_hits
        rrf_hits = trace.rrf_hits
        reranker_input = trace.candidates[:reranker_input_top_k]
    except Exception as exc:
        errors.append(_short_error("Hybrid retrieval", exc))
    else:
        try:
            ranking = rank_hybrid_candidates(
                question,
                trace,
                reranker=reranker,
                top_k=final_context_top_k,
                reranker_input_k=reranker_input_top_k,
                reranker_output_k=reranker_top_k,
                faiss_weight=faiss_weight,
                bm25_weight=bm25_weight,
                rrf_weight=rrf_weight,
                reranker_weight=reranker_weight,
                bm25_confirm_faiss_k=bm25_confirm_faiss_k,
                bm25_confirm_rrf_k=bm25_confirm_rrf_k,
                bm25_confirm_reranker_score=bm25_confirm_reranker_score,
                max_context_tokens=max_context_tokens,
                min_context_score=min_context_score,
                min_context_relative_score=min_context_relative_score,
                source_diversity_penalty=source_diversity_penalty,
                tokenizer_model=tokenizer_model,
                token_counter=token_counter,
            )
        except Exception as exc:
            errors.append(_short_error("Qwen Reranker", exc))
            ranking = rank_hybrid_candidates(
                question,
                trace,
                reranker=None,
                top_k=final_context_top_k,
                reranker_input_k=reranker_input_top_k,
                reranker_output_k=reranker_top_k,
                faiss_weight=faiss_weight,
                bm25_weight=bm25_weight,
                rrf_weight=rrf_weight,
                reranker_weight=reranker_weight,
                bm25_confirm_faiss_k=bm25_confirm_faiss_k,
                bm25_confirm_rrf_k=bm25_confirm_rrf_k,
                bm25_confirm_reranker_score=bm25_confirm_reranker_score,
                max_context_tokens=max_context_tokens,
                min_context_score=min_context_score,
                min_context_relative_score=min_context_relative_score,
                source_diversity_penalty=source_diversity_penalty,
                tokenizer_model=tokenizer_model,
                token_counter=token_counter,
            )
        reranker_input = ranking.reranker_input
        reranked_hits = ranking.reranked
        final_hits = ranking.final

    final_context = OpenAICompatibleLLM._source_context(final_hits)
    columns = _column_names(
        faiss_top_k,
        bm25_top_k,
        rrf_top_k,
        reranker_input_top_k,
        reranker_top_k,
        final_context_top_k,
    )
    return {
        "id": identifier,
        "question": question,
        columns[2]: _documents_json(faiss_hits),
        columns[3]: _documents_json(bm25_hits),
        columns[4]: _documents_json(rrf_hits),
        columns[5]: _documents_json(reranker_input),
        columns[6]: _documents_json(reranked_hits),
        columns[7]: _documents_json(final_hits),
        "Final Context": final_context,
        "error": "\n".join(errors),
    }


def _resolve_input_paths(
    data_dir: Path,
    questions_path: Path | None,
    knowledge_base_path: Path | None,
) -> tuple[Path, Path]:
    if questions_path is None:
        questions_path = _find_platform_file(data_dir, "questions.csv")
    if knowledge_base_path is None:
        knowledge_base_path = _find_platform_file(
            data_dir,
            "kb.zip",
            "knowledge-base.zip",
            "knowledge_base.zip",
        )
    return questions_path.expanduser().resolve(), knowledge_base_path.expanduser().resolve()


def write_diagnostics(
    *,
    questions_path: Path,
    knowledge_base_path: Path,
    output_path: Path,
    faiss_top_k: int = FAISS_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    rrf_top_k: int = RRF_TOP_K,
    reranker_input_top_k: int = RERANKER_INPUT_K,
    reranker_top_k: int = RERANKER_TOP_K,
    final_context_top_k: int = FINAL_CONTEXT_TOP_K,
    faiss_protected_top_k: int = FAISS_PROTECTED_TOP_K,
    bm25_protected_top_k: int = BM25_PROTECTED_TOP_K,
    faiss_weight: float = FAISS_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    rrf_weight: float = RRF_WEIGHT,
    reranker_weight: float = RERANKER_WEIGHT,
    rrf_semantic_weight: float = 1.0,
    rrf_bm25_weight: float = 1.0,
    bm25_confirm_faiss_k: int = BM25_CONFIRM_FAISS_TOP_K,
    bm25_confirm_rrf_k: int = BM25_CONFIRM_RRF_TOP_K,
    bm25_confirm_reranker_score: float = BM25_CONFIRM_RERANKER_SCORE,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    min_context_score: float = MIN_CONTEXT_SCORE,
    min_context_relative_score: float = MIN_CONTEXT_RELATIVE_SCORE,
    source_diversity_penalty: float = SOURCE_DIVERSITY_PENALTY,
    tokenizer_model: str = "gpt-4o",
    token_counter: Callable[[str], int] | None = None,
) -> Path:
    """Build the index, run diagnostics, and write ``diagnostic.csv``."""

    questions = _load_platform_questions(questions_path)
    documents = load_documents(knowledge_base_path)
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("Knowledge base does not contain any text chunks")

    embedding_client = EmbeddingClient()
    index_error = ""
    try:
        index = embed_and_index(chunks, embedding_client)
    except Exception as exc:
        index_error = _short_error("Embedding fallback to BM25", exc)
        print(index_error, file=sys.stderr)
        index = FaissVectorIndex()
        index.add_lexical(chunks)
    use_reranker = os.getenv("USE_RERANKER", "1").casefold() in {"1", "true", "yes"}
    reranker = Reranker() if use_reranker else None

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = _column_names(
        faiss_top_k,
        bm25_top_k,
        rrf_top_k,
        reranker_input_top_k,
        reranker_top_k,
        final_context_top_k,
    )
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writeheader()
        for position, (identifier, question) in enumerate(questions, start=1):
            print(
                f"Question {position}/{len(questions)}: collecting diagnostics",
                file=sys.stderr,
                flush=True,
            )
            row = collect_diagnostic_row(
                identifier,
                question,
                index,
                embedding_client,
                reranker,
                faiss_top_k=faiss_top_k,
                bm25_top_k=bm25_top_k,
                rrf_top_k=rrf_top_k,
                reranker_input_top_k=reranker_input_top_k,
                reranker_top_k=reranker_top_k,
                final_context_top_k=final_context_top_k,
                faiss_protected_top_k=faiss_protected_top_k,
                bm25_protected_top_k=bm25_protected_top_k,
                faiss_weight=faiss_weight,
                bm25_weight=bm25_weight,
                rrf_weight=rrf_weight,
                reranker_weight=reranker_weight,
                rrf_semantic_weight=rrf_semantic_weight,
                rrf_bm25_weight=rrf_bm25_weight,
                bm25_confirm_faiss_k=bm25_confirm_faiss_k,
                bm25_confirm_rrf_k=bm25_confirm_rrf_k,
                bm25_confirm_reranker_score=bm25_confirm_reranker_score,
                max_context_tokens=max_context_tokens,
                min_context_score=min_context_score,
                min_context_relative_score=min_context_relative_score,
                source_diversity_penalty=source_diversity_penalty,
                tokenizer_model=tokenizer_model,
                token_counter=token_counter,
            )
            if index_error:
                row["error"] = "\n".join(filter(None, (index_error, row["error"])))
            writer.writerow(row)

    print(f"Diagnostics written to {output_path}", file=sys.stderr)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write FAISS, BM25, RRF and Qwen reranker diagnostics to CSV"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("DATA_DIR", "data")),
        help="directory containing questions.csv and kb.zip (default: data)",
    )
    parser.add_argument("--questions", type=Path, help="explicit questions.csv path")
    parser.add_argument("--knowledge-base", type=Path, help="explicit kb.zip path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("DIAGNOSTIC_OUTPUT", "diagnostic.csv")),
        help="output CSV path (default: diagnostic.csv)",
    )
    parser.add_argument(
        "--faiss-top-k", type=_positive_int,
        default=_platform_int("TOP_K", FAISS_TOP_K, 1, 75),
    )
    parser.add_argument(
        "--bm25-top-k", type=_positive_int,
        default=_platform_int("TOP_K", BM25_TOP_K, 1, 75),
    )
    parser.add_argument(
        "--rrf-top-k", type=_positive_int,
        default=_platform_int("RRF_TOP_K", RRF_TOP_K, 1, 100),
    )
    parser.add_argument(
        "--reranker-input-top-k",
        type=_positive_int,
        default=_platform_int("RERANKER_CANDIDATES", RERANKER_INPUT_K, 1, 100),
    )
    parser.add_argument(
        "--reranker-top-k",
        type=_positive_int,
        default=_platform_int("RERANKER_TOP_N", RERANKER_TOP_K, 1, 100),
    )
    parser.add_argument(
        "--final-context-top-k",
        type=_positive_int,
        default=_platform_int("FINAL_CONTEXT_TOP_K", FINAL_CONTEXT_TOP_K, 1, 100),
    )
    parser.add_argument(
        "--faiss-protected-top-k", type=int,
        default=_platform_int("FAISS_PROTECTED_TOP_K", FAISS_PROTECTED_TOP_K, 0, 75),
    )
    parser.add_argument(
        "--bm25-protected-top-k", type=int,
        default=_platform_int("BM25_PROTECTED_TOP_K", BM25_PROTECTED_TOP_K, 0, 75),
    )
    parser.add_argument(
        "--faiss-weight", type=float,
        default=_platform_float("FAISS_WEIGHT", FAISS_WEIGHT, 0.0, 1.0),
    )
    parser.add_argument(
        "--bm25-weight", type=float,
        default=_platform_float("BM25_WEIGHT", BM25_WEIGHT, 0.0, 1.0),
    )
    parser.add_argument(
        "--rrf-weight", type=float,
        default=_platform_float("RRF_WEIGHT", RRF_WEIGHT, 0.0, 1.0),
    )
    parser.add_argument(
        "--reranker-weight", type=float,
        default=_platform_float("RERANKER_WEIGHT", RERANKER_WEIGHT, 0.0, 1.0),
    )
    parser.add_argument(
        "--rrf-semantic-weight", type=float,
        default=_platform_float("RRF_SEMANTIC_WEIGHT", 1.0, 0.0, 5.0),
    )
    parser.add_argument(
        "--rrf-bm25-weight", type=float,
        default=_platform_float("RRF_BM25_WEIGHT", 1.0, 0.0, 5.0),
    )
    parser.add_argument(
        "--bm25-confirm-faiss-k", type=_positive_int,
        default=_platform_int("BM25_CONFIRM_FAISS_TOP_K", BM25_CONFIRM_FAISS_TOP_K, 1, 75),
    )
    parser.add_argument(
        "--bm25-confirm-rrf-k", type=_positive_int,
        default=_platform_int("BM25_CONFIRM_RRF_TOP_K", BM25_CONFIRM_RRF_TOP_K, 1, 100),
    )
    parser.add_argument(
        "--bm25-confirm-reranker-score", type=float,
        default=_platform_float("BM25_CONFIRM_RERANKER_SCORE", BM25_CONFIRM_RERANKER_SCORE, -20.0, 20.0),
    )
    parser.add_argument(
        "--max-context-tokens", type=_positive_int,
        default=_platform_int("MAX_CONTEXT_TOKENS", MAX_CONTEXT_TOKENS, 256, 200_000),
    )
    parser.add_argument(
        "--min-context-score", type=float,
        default=_platform_float("MIN_CONTEXT_SCORE", MIN_CONTEXT_SCORE, 0.0, 1.0),
    )
    parser.add_argument(
        "--min-context-relative-score", type=float,
        default=_platform_float("MIN_CONTEXT_RELATIVE_SCORE", MIN_CONTEXT_RELATIVE_SCORE, 0.0, 1.0),
    )
    parser.add_argument(
        "--source-diversity-penalty", type=float,
        default=_platform_float("SOURCE_DIVERSITY_PENALTY", SOURCE_DIVERSITY_PENALTY, 0.0, 1.0),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    questions_path, knowledge_base_path = _resolve_input_paths(
        args.data_dir,
        args.questions,
        args.knowledge_base,
    )
    return write_diagnostics(
        questions_path=questions_path,
        knowledge_base_path=knowledge_base_path,
        output_path=args.output,
        faiss_top_k=args.faiss_top_k,
        bm25_top_k=args.bm25_top_k,
        rrf_top_k=args.rrf_top_k,
        reranker_input_top_k=args.reranker_input_top_k,
        reranker_top_k=args.reranker_top_k,
        final_context_top_k=args.final_context_top_k,
        faiss_protected_top_k=args.faiss_protected_top_k,
        bm25_protected_top_k=args.bm25_protected_top_k,
        faiss_weight=args.faiss_weight,
        bm25_weight=args.bm25_weight,
        rrf_weight=args.rrf_weight,
        reranker_weight=args.reranker_weight,
        rrf_semantic_weight=args.rrf_semantic_weight,
        rrf_bm25_weight=args.rrf_bm25_weight,
        bm25_confirm_faiss_k=args.bm25_confirm_faiss_k,
        bm25_confirm_rrf_k=args.bm25_confirm_rrf_k,
        bm25_confirm_reranker_score=args.bm25_confirm_reranker_score,
        max_context_tokens=args.max_context_tokens,
        min_context_score=args.min_context_score,
        min_context_relative_score=args.min_context_relative_score,
        source_diversity_penalty=args.source_diversity_penalty,
        tokenizer_model=os.getenv("LLM_MODEL", "gpt-4o"),
    )


if __name__ == "__main__":
    main()
