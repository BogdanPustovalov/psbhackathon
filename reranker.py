"""Реранкер на базе Qwen3-Reranker.

Два режима (backend):
- "local" — модель с Hugging Face, считается на своём компьютере (Mac: GPU через MPS).
            Нужны torch и transformers — их НЕТ на платформе хакатона, поэтому
            они импортируются только внутри этого режима.
- "api"   — POST {LLM_BASE_URL}/rerank на платформе хакатона (Qwen3 Reranker 8B).

Оценка релевантности Qwen3-Reranker = вероятность, что модель ответит "yes"
на вопрос «подходит ли документ к запросу»:
    score = P(yes) / (P(yes) + P(no)) = sigmoid(logit_yes - logit_no)
"""

from __future__ import annotations

import math
import os
import sys
import time

# ---------------------------------------------------------------- константы

DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-Reranker-0.6B"   # для отладки; финальные метрики — на 8B
DEFAULT_API_MODEL = "Qwen/Qwen3-Reranker-8B"       # если платформа не передала RERANKER_MODEL

# Стандартная инструкция из карточки модели.
INSTRUCTION_DEFAULT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
# Инструкция под юридическую задачу кейса (на английском — так советует карточка модели).
INSTRUCTION_LEGAL = (
    "Given a question from a bank lawyer, judge whether the document (a provision of "
    "Russian law or of an internal company document) directly answers the question or "
    "is necessary to answer it. Pay attention to the exact counterparty name, contract "
    "number and document edition."
)

# Официальный шаблон Qwen3-Reranker (карточка модели на Hugging Face).
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Ограничение log-odds, чтобы не получить бесконечность при p = 0 или 1.
_EPS = 1e-6


def prob_to_logodds(p: float) -> float:
    """Вероятность -> log-odds: ln(p / (1 - p)). Удобнее для порогов, чем p около 0 и 1."""
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def logodds_to_prob(x: float) -> float:
    """log-odds -> вероятность (сигмоида)."""
    return 1.0 / (1.0 + math.exp(-x))


class RerankerError(RuntimeError):
    """Реранкер не смог посчитать оценки (например, API недоступен после повторов)."""


class Reranker:
    """Оценивает релевантность документов запросу и сортирует их.

    Методы принимают тексты, а возвращают индексы и оценки — метаданные фрагментов
    (документ, пункт, редакция) остаются у вызывающего кода.

    Пример:
        reranker = Reranker()                       # локально, Qwen3-Reranker-0.6B
        reranker.rerank(query, docs, top_n=5)       # [(индекс, оценка), ...]
    """

    def __init__(
        self,
        model_name: str | None = None,
        backend: str = "local",
        instruction: str = INSTRUCTION_LEGAL,
        batch_size: int = 8,
        max_length: int = 8192,
        device: str | None = None,
        api_use_template: bool = False,
        api_timeout: float = 60.0,
        api_max_docs_per_request: int = 64,
        api_retries: int = 3,
        debug: bool = False,
    ):
        if backend not in ("local", "api"):
            raise ValueError(f"Неизвестный backend: {backend!r}. Ожидается 'local' или 'api'.")

        if model_name is None:
            if backend == "local":
                model_name = DEFAULT_LOCAL_MODEL
            else:
                # На платформе имя модели приходит в переменной окружения (Технический регламент).
                model_name = os.getenv("RERANKER_MODEL") or DEFAULT_API_MODEL

        self.model_name = model_name
        self.backend = backend
        self.instruction = instruction
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.debug = debug

        # Настройки режима API.
        self.api_use_template = api_use_template      # оборачивать ли тексты в шаблон Qwen3 самим
        self.api_timeout = api_timeout
        self.api_max_docs_per_request = api_max_docs_per_request
        self.api_retries = api_retries

        # Лениво загружаемые объекты локального режима.
        self.model = None
        self.tokenizer = None
        self._torch = None
        self._prefix_ids: list[int] = []
        self._suffix_ids: list[int] = []
        self._yes_id: int | None = None
        self._no_id: int | None = None

    def __repr__(self) -> str:
        return (
            f"Reranker(model={self.model_name!r}, backend={self.backend!r}, "
            f"device={self.device!r}, batch_size={self.batch_size})"
        )

    # ============================================================ общий интерфейс

    def score(
        self, query: str, documents: list[str], as_logodds: bool = False
    ) -> list[float]:
        """Оценка релевантности каждого документа — в ИСХОДНОМ порядке документов.

        По умолчанию — вероятность от 0 до 1. as_logodds=True — log-odds
        (ln(p/(1-p))): удобнее для порогов и сравнения близких оценок.
        """
        if not documents:
            return []
        if self.backend == "local":
            logodds = self._score_local(query, documents)
        else:
            logodds = self._score_api(query, documents)
        return logodds if as_logodds else [logodds_to_prob(x) for x in logodds]

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        as_logodds: bool = False,
    ) -> list[tuple[int, float]]:
        """Пары (индекс документа, оценка), от самого релевантного. top_n=None — все."""
        scores = self.score(query, documents, as_logodds=as_logodds)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return ranked if top_n is None else ranked[:top_n]

    def format_pair(self, query: str, document: str) -> str:
        """Текст пары в формате Qwen3-Reranker (без системного префикса и суффикса)."""
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"

    # ============================================================ режим "local"

    def _pick_device(self) -> str:
        torch = self._torch
        if torch.backends.mps.is_available():
            return "mps"          # GPU чипов Apple M1–M5
        if torch.cuda.is_available():
            return "cuda"         # видеокарта NVIDIA
        return "cpu"

    def _load(self) -> None:
        """Загружает модель и токенизатор один раз, при первом вызове."""
        if self.model is not None:
            return
        # Ленивый импорт: на платформе этих библиотек нет, и режим "api" их не требует.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if self.device is None:
            self.device = self._pick_device()

        # Половинная точность на GPU: вдвое меньше памяти (8B: ~16 ГБ вместо ~32 ГБ).
        dtype = {
            "mps": torch.float16,
            "cuda": torch.bfloat16,
        }.get(self.device, torch.float32)

        started = time.perf_counter()
        # padding_side="left": последний токен каждой строки батча — её настоящий конец,
        # и именно на нём модель «отвечает» yes/no.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        except TypeError:  # transformers < 4.56 знает этот параметр как torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype)
        self.model.to(self.device).eval()

        self._yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self._no_id = self.tokenizer.convert_tokens_to_ids("no")
        self._prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self._suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)

        if self.debug:
            print(
                f"[reranker] {self.model_name} загружена на {self.device} ({dtype}) "
                f"за {time.perf_counter() - started:.1f} с",
                file=sys.stderr,
            )

    def _score_local(self, query: str, documents: list[str]) -> list[float]:
        self._load()
        torch = self._torch
        pairs = [self.format_pair(query, d) for d in documents]

        # Батчи из текстов похожей длины → меньше «пустых» токенов-заполнителей.
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i]))
        result = [0.0] * len(pairs)
        room = self.max_length - len(self._prefix_ids) - len(self._suffix_ids)

        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tokenizer(
                [pairs[i] for i in idx],
                padding=False,
                truncation="longest_first",
                max_length=room,
                return_attention_mask=False,
            )
            enc["input_ids"] = [self._prefix_ids + ids + self._suffix_ids for ids in enc["input_ids"]]
            batch = self.tokenizer.pad(enc, padding=True, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                try:
                    # logits_to_keep=1: считать логиты только для последней позиции
                    # (иначе [batch × длина × 151 тыс. слов] — гигабайты памяти).
                    logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
                except TypeError:  # старые версии transformers
                    logits = self.model(**batch).logits[:, -1, :]
            # log-odds = logit("yes") − logit("no"); считаем во float32 ради точности.
            logodds = (logits[:, self._yes_id] - logits[:, self._no_id]).float().tolist()
            for i, value in zip(idx, logodds):
                result[i] = value
        return result

    # ============================================================ режим "api"

    def _api_settings(self) -> tuple[str, str]:
        base = (
            os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
        ).rstrip("/")
        key = (
            os.getenv("RERANKER_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        )
        if not base or not key:
            raise RerankerError("Нет LLM_BASE_URL / LLM_API_KEY в окружении — режим 'api' недоступен.")
        return base, key

    def _api_texts(self, query: str, documents: list[str]) -> tuple[str, list[str]]:
        """Тексты для /rerank: как есть или обёрнутые в шаблон Qwen3 (как в примерах vLLM)."""
        if not self.api_use_template:
            return query, documents
        q = f"{PREFIX}<Instruct>: {self.instruction}\n<Query>: {query}\n"
        docs = [f"<Document>: {d}{SUFFIX}" for d in documents]
        return q, docs

    def _post_with_retries(self, client, url: str, key: str, payload: dict) -> dict:
        import httpx

        last_error: Exception | None = None
        for attempt in range(self.api_retries):
            try:
                response = client.post(
                    url, headers={"Authorization": f"Bearer {key}"}, json=payload
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise RerankerError(f"HTTP {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.TransportError, RerankerError) as exc:
                last_error = exc
                time.sleep(2**attempt)  # 1, 2, 4 с — ограниченный повтор, как просит регламент
            except httpx.HTTPStatusError as exc:  # 4xx кроме 429 — повтор не поможет
                raise RerankerError(f"HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc
        raise RerankerError(f"/rerank не ответил после {self.api_retries} попыток: {last_error}")

    def _score_api(self, query: str, documents: list[str]) -> list[float]:
        import httpx  # есть в requirements платформы

        base, key = self._api_settings()
        q, docs = self._api_texts(query, documents)
        result: list[float | None] = [None] * len(docs)

        with httpx.Client(timeout=self.api_timeout) as client:
            for start in range(0, len(docs), self.api_max_docs_per_request):
                part = docs[start : start + self.api_max_docs_per_request]
                payload = {
                    "model": self.model_name,
                    "query": q,
                    "documents": part,
                    "top_n": len(part),  # нужны оценки ВСЕХ документов, не только лучших
                }
                data = self._post_with_retries(client, base + "/rerank", key, payload)
                if self.debug:
                    print(f"[reranker] сырой ответ /rerank: {str(data)[:1000]}", file=sys.stderr)

                items = data.get("results") or data.get("data") or []
                for item in items:
                    i = int(item["index"])
                    raw = item.get("relevance_score", item.get("score"))
                    if raw is None or not 0 <= i < len(part):
                        continue
                    raw = float(raw)
                    # Вероятность [0, 1] → log-odds; иначе считаем, что это уже логит.
                    result[start + i] = prob_to_logodds(raw) if 0.0 <= raw <= 1.0 else raw

        missing = sum(x is None for x in result)
        if missing:
            raise RerankerError(f"/rerank не вернул оценки для {missing} из {len(result)} документов")
        return result  # type: ignore[return-value]


# ---------------------------------------------------------------- быстрая проверка

if __name__ == "__main__":
    query = "Какая арендная плата по договору с ООО «Северный Склад»?"
    docs = [
        "Договор аренды № А-08/2024 · п. 4.3. Арендодатель обязан обеспечить "
        "беспрепятственный доступ Арендатора к Помещению круглосуточно.",
        "Договор аренды № А-08/2024 · п. 3.1. Арендная плата составляет 450 000 рублей в "
        "месяц, включая НДС.",
        "ТК РФ ст. 70 · Срок испытания не может превышать трех месяцев.",
    ]
    reranker = Reranker(debug=True)
    print(reranker)
    started = time.perf_counter()
    for index, score in reranker.rerank(query, docs):
        print(f"{score:.4f}  (log-odds {prob_to_logodds(score):+.2f})  #{index}: {docs[index][:70]}")
    print(f"Время: {time.perf_counter() - started:.1f} с (первый запуск включает загрузку модели)")
