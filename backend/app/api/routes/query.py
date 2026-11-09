from fastapi import APIRouter, Depends

from app.models.schemas import QueryRequest, QueryResponse
from app.services.pipeline import RAGPipeline

router = APIRouter(prefix="/query", tags=["query"])


def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@router.post("", response_model=QueryResponse)
async def query_rag(
    payload: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    return await pipeline.answer(payload)
