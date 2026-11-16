from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from threading import RLock
from typing import Any

from . import RERANKER_TOP_K, RerankerError, _document_text, _with_score

DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"


class Reranker:
    """Local Qwen3 reranker with lazy model loading.

    The model is downloaded by Hugging Face on first use and then reused from
    its local cache. Set ``RERANKER_DEVICE=cpu`` to force CPU inference or a
    CUDA device such as ``cuda:0`` when available.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        instruction: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        device: str | None = None,
        cache_folder: str | None = None,
    ) -> None:
        if batch_size is None:
            batch_size = int(os.getenv("RERANKER_BATCH_SIZE", "2"))
        if max_length is None:
            max_length = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")

        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL", DEFAULT_RERANKER_MODEL
        )
        self.instruction = instruction or os.getenv(
            "RERANKER_INSTRUCTION",
            "Given a web search query, retrieve relevant passages that answer the query",
        )
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or os.getenv("RERANKER_DEVICE") or None
        self.cache_folder = cache_folder or os.getenv("RERANKER_CACHE_DIR") or None
        self._model: Any | None = None
        self._model_lock = RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> Reranker:
        """Load Qwen locally only when it is first needed."""
        with self._model_lock:
            if self._model is not None:
                return self

            try:
                import torch
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerError(
                    "Local reranking requires sentence-transformers. "
                    "Install it with: pip install sentence-transformers"
                ) from exc

            selected_device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            if selected_device.startswith("cuda") and not torch.cuda.is_available():
                raise RerankerError(
                    "CUDA was requested but is unavailable in this PyTorch install "
                    f"(torch={torch.__version__}, CUDA build={torch.version.cuda}). "
                    "Install the GPU build with: "
                    "python -m pip install -r the matching PyTorch CUDA wheel"
                )

            kwargs: dict[str, Any] = {
                "prompts": {"retrieval": self.instruction},
                "default_prompt_name": "retrieval",
                "max_length": self.max_length,
            }
            if selected_device.startswith("cuda"):
                kwargs["model_kwargs"] = {"dtype": torch.float16}
            elif selected_device == "cpu":
                kwargs["model_kwargs"] = {"dtype": torch.float32}
            kwargs["device"] = selected_device
            if self.cache_folder:
                kwargs["cache_folder"] = self.cache_folder

            try:
                print(
                    f"Loading local reranker {self.model_name} on {selected_device}",
                    file=sys.stderr,
                    flush=True,
                )
                self._model = CrossEncoder(self.model_name, **kwargs)
            except Exception as exc:
                raise RerankerError(
                    f"Could not load local reranker {self.model_name!r}: {exc}"
                ) from exc
        return self

    def score_pairs(self, query: str, documents: Sequence[Any]) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not documents:
            return []

        texts = [_document_text(document) for document in documents]
        model = self.load()._model
        try:
            with self._model_lock:
                scores = model.predict(
                    [(query, text) for text in texts],
                    batch_size=self.batch_size,
                    show_progress_bar=True,
                )
            return [float(score) for score in scores]
        except Exception as exc:
            raise RerankerError(f"Local reranking failed: {exc}") from exc

    def score(self, query: str, document: Any) -> float:
        return self.score_pairs(query, [document])[0]

    def rerank(
        self,
        query: str,
        documents: Sequence[Any],
        *,
        top_k: int | None = RERANKER_TOP_K,
    ) -> list[Any]:
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        ranked = sorted(
            zip(documents, self.score_pairs(query, documents)),
            key=lambda item: item[1],
            reverse=True,
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return [_with_score(document, score) for document, score in ranked]
