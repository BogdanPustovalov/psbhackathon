from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from app.models.schemas import IngestResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: Annotated[list[UploadFile], File()]) -> IngestResponse:
    names = [file.filename or "unnamed" for file in files]
    # TODO: передать файлы в асинхронную очередь ingestion worker.
    return IngestResponse(
        accepted_files=names,
        message="Файлы приняты. Подключите ingestion worker для чанкинга и индексации.",
    )
