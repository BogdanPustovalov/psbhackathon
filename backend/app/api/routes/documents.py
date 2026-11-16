from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import DocumentDetail, DocumentInfo, IngestResponse
from app.services.pipeline import get_corpus

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return get_corpus().list_documents()


@router.get("/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str) -> DocumentDetail:
    document = get_corpus().documents.get(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return document


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: Annotated[list[UploadFile], File()]) -> IngestResponse:
    names = [file.filename or "unnamed" for file in files]
    # TODO: передать файлы в асинхронную очередь ingestion worker.
    return IngestResponse(
        accepted_files=names,
        message="Файлы приняты. Подключите ingestion worker для чанкинга и индексации.",
    )
