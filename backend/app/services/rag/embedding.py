from __future__ import annotations

from abc import ABC, abstractmethod
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Any

import httpx

from . import BATCH_SIZE, BGE_M3_DIMENSION, EMBEDDING_BATCH_SIZE, _env, _first_env

class EmbeddingAPIError(RuntimeError):
    """Получается когда API встраивания не могут вернуть действительные векторы."""


class EmbeddingProvider(ABC):

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Создайте один вектор встраивания для каждого входного текста.

        Векторы должны быть нормализованы и иметь одинаковую размерность.
        """
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 90.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("Ошибка! timeout должен быть больше нуля")

        self.model = model or _first_env(
            "EMB_MODEL", "EMBEDDING_MODEL", default="text-embedding-3-small"
        )
        configured_base_url = base_url or _first_env(
            "LLM_BASE_URL",
            "BASE_URL",
            "OPENAI_BASE_URL",
            "EMBEDDING_BASE_URL",
            default="https://api.openai.com/v1",
        )
        self.endpoint = configured_base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key or _env(
            "EMBEDDER_API_KEY",
            _env("EMBEDDING_API_KEY", _env("LLM_API_KEY", _env("OPENAI_API_KEY"))),
        )
        self.timeout = timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("All texts must be strings")
        if not self.api_key:
            raise EmbeddingAPIError(
                "Set EMBEDDER_API_KEY, EMBEDDING_API_KEY, or LLM_API_KEY"
            )

        embeddings: list[list[float]] = []
        dimension = BGE_M3_DIMENSION if self.model.casefold() == "baai/bge-m3" else None
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(headers=headers, timeout=self.timeout) as client:
            for start in range(0, len(texts), BATCH_SIZE):
                batch = texts[start : start + BATCH_SIZE]
                response = await client.post(
                    self.endpoint,
                    json={
                        "model": self.model,
                        "input": batch,
                        "encoding_format": "float",
                    },
                )
                response.raise_for_status()

                try:
                    rows = response.json()["data"]
                except (KeyError, TypeError, ValueError) as exc:
                    raise EmbeddingAPIError(
                        "Ответ Embeddings API не содержит данных"
                    ) from exc

                indexed = [
                    (int(row.get("index", position)), row["embedding"])
                    for position, row in enumerate(rows)
                ]
                indexed.sort(key=lambda item: item[0])
                if [index for index, _ in indexed] != list(range(len(batch))):
                    raise EmbeddingAPIError(
                        "Ответ Embeddings API не содержит полного или действительного пакета"
                    )

                for _, vector in indexed:
                    if dimension is None:
                        dimension = len(vector)
                    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
                    if (
                        len(vector) != dimension
                        or not math.isfinite(norm)
                        or norm == 0
                    ):
                        raise EmbeddingAPIError(
                            "Ответ Embeddings API содержит недопустимый вектор"
                        )
                    embeddings.append(
                        [float(value) / norm for value in vector]
                    )
        return embeddings


class EmbeddingClient:
    """Offline BGE-M3 encoder, loaded from a local path or Hugging Face cache."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        model_path: str | None = None,
        device: str | None = None,
        cache_folder: str | None = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size должен быть больше нуля")
        self.model_name = (
            model_path
            or model_name
            or _first_env("EMBEDDING_MODEL_PATH", "LOCAL_EMBEDDING_MODEL", default="BAAI/bge-m3")
        )
        self.device = device or _env("EMBEDDING_DEVICE") or None
        self.cache_folder = cache_folder or _env("EMBEDDING_CACHE_DIR") or None
        self.batch_size = batch_size
        self._model: Any | None = None
        self._lock = RLock()

    def load(self) -> "EmbeddingClient":
        with self._lock:
            if self._model is not None:
                return self
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingAPIError(
                    "Для локальных эмбеддингов установите sentence-transformers"
                ) from exc

            model_path = Path(self.model_name).expanduser()
            if model_path.is_absolute() and not model_path.is_dir():
                raise EmbeddingAPIError(f"Локальная папка модели не найдена: {model_path}")
            try:
                model_source = self.model_name
                if self.model_name.casefold() == "baai/bge-m3":
                    from huggingface_hub import snapshot_download

                    model_source = snapshot_download(
                        repo_id="BAAI/bge-m3",
                        cache_dir=self.cache_folder,
                        local_files_only=True,
                    )
                print(
                    f"Loading embedding model on {self.device or 'auto'}...",
                    file=sys.stderr,
                    flush=True,
                )
                self._model = SentenceTransformer(
                    model_source,
                    device=self.device,
                    cache_folder=self.cache_folder,
                    local_files_only=True,
                )
            except Exception as exc:
                raise EmbeddingAPIError(
                    f"Не удалось открыть локальную модель {self.model_name!r}. "
                    "Укажите EMBEDDING_MODEL_PATH или поместите BAAI/bge-m3 в кэш Hugging Face."
                ) from exc
        return self

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Тексты должны быть строками")

        self.load()
        try:
            with self._lock:
                vectors = self._model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=len(texts) > self.batch_size,
                )
            if vectors.ndim != 2 or vectors.shape != (len(texts), BGE_M3_DIMENSION):
                raise EmbeddingAPIError("BGE-M3 вернула матрицу неправильного размера")
            if not all(math.isfinite(float(value)) for row in vectors for value in row):
                raise EmbeddingAPIError("BGE-M3 вернула нечисловой вектор")
            return vectors.tolist()
        except EmbeddingAPIError:
            raise
        except Exception as exc:
            raise EmbeddingAPIError(f"Ошибка локального встраивания BGE-M3: {exc}") from exc




