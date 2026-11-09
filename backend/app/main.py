from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health, query
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Модульный API для retrieval augmented generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(query.router, prefix=settings.api_prefix)
app.include_router(documents.router, prefix=settings.api_prefix)
