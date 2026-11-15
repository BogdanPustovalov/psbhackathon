"""Read the prepared corpus without re-embedding or using the source files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.models.schemas import DocumentDetail, DocumentInfo
from app.services.rag import Document
from app.services.rag.vector_store import FaissVectorIndex


@dataclass(slots=True)
class Corpus:
    chunks: list[Document]
    documents: dict[str, DocumentDetail]
    index: FaissVectorIndex
    model: str

    def list_documents(self) -> list[DocumentInfo]:
        return [DocumentInfo.model_validate(doc.model_dump(exclude={"text"})) for doc in self.documents.values()]


def _document_type(filename: str) -> str:
    if filename.startswith("законы__"):
        return "law"
    if "Договор" in filename or "соглашение" in filename.lower():
        return "contract"
    return "internal"


def _title(filename: str) -> str:
    return Path(filename.split("__", 1)[-1]).stem.replace("_", " ")


def _edition(title: str) -> dict[str, str | None] | None:
    version = re.search(r"(?i)ред(?:акция)?\s+(20\d{2})", title)
    if version:
        label = f"редакция {version.group(1)}"
    elif "действующая" in title.lower():
        label = "действующая редакция"
    else:
        return None
    return {
        "label": label,
        "valid_from": None,
        "valid_to": None,
        "status": "expired" if "утратил силу" in title.lower() else "active",
    }


def _reconstruct(parts: list[tuple[int, int, str]]) -> tuple[str, list[int]]:
    text = ""
    offsets: list[int] = []
    for _, _, fragment in parts:
        if text and text[-1].isalnum() and fragment and fragment[0].isalnum():
            text += " "
        offsets.append(len(text))
        text += fragment
    return text, offsets


def load_corpus(directory: Path) -> Corpus:
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    raw_chunks = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
    vectors = np.load(directory / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if manifest.get("chunk_overlap") != 0:
        raise ValueError("Document reconstruction requires chunk_overlap=0")
    if len(raw_chunks) != manifest["document_count"] or vectors.shape[0] != len(raw_chunks):
        raise ValueError("Artifact chunk counts do not match")

    grouped: dict[str, list[tuple[int, int, str]]] = {}
    chunks: list[Document] = []
    for position, item in enumerate(raw_chunks):
        metadata = dict(item["metadata"])
        filename = str(metadata["filename"])
        chunk_index = int(metadata["chunk_index"])
        text = str(item["text"])
        grouped.setdefault(filename, []).append((chunk_index, position, text))
        chunks.append(Document(text=text, metadata=metadata))

    documents: dict[str, DocumentDetail] = {}
    for filename, parts in grouped.items():
        parts.sort()
        if [part[0] for part in parts] != list(range(len(parts))):
            raise ValueError(f"Non-contiguous chunks for {filename}")
        title = _title(filename)
        edition = _edition(title)
        full_text, offsets = _reconstruct(parts)
        documents[filename] = DocumentDetail(
            doc_id=filename,
            doc_title=title,
            doc_type=_document_type(filename),
            edition=edition,
            related=[],
            text=full_text,
        )
        for (_, position, text), offset in zip(parts, offsets):
            chunks[position].metadata.update(
                doc_id=filename,
                doc_title=title,
                doc_type=_document_type(filename),
                edition=edition,
                related=[],
                char_start=offset,
                char_end=offset + len(text),
            )

    index = FaissVectorIndex.from_prebuilt(chunks, str(directory / "index.faiss"))
    if vectors.ndim != 2 or index.dimension != vectors.shape[1]:
        raise ValueError("FAISS index and embeddings.npy have different dimensions")
    return Corpus(chunks=chunks, documents=documents, index=index, model=manifest["model"])
