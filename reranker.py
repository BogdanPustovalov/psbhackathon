from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field


DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_API_MODEL = "Qwen/Qwen3-Reranker-8B"

INSTRUCTION_DEFAULT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
INSTRUCTION_LEGAL = (
    "Given a question from a bank lawyer, judge whether the document (a provision of "
    "Russian law or of an internal company document) directly answers the question or "
    "is necessary to answer it. Pay attention to the exact counterparty name, contract "
    "number and document edition."
)

INSTRUCTION_STRICT = (
    "Given a question from a bank lawyer, answer yes only if the document contains the "
    "specific provision that answers the question. A provision from the right contract or "
    "law that is about a different topic is not relevant. The counterparty name and "
    "contract number must match the question."
)

THRESHOLD_NO_ANSWER = 1.8
THRESHOLD_CONFIDENT = 6.1
RELATIVE_MARGIN = 4.0

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_EPS = 1e-6


def prob_to_logodds(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def logodds_to_prob(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Selection:

    indices: list[int]
    zone: str
    best: float
    scores: list[float] = field(default_factory=list)


def select_from_scores(
    logodds: list[float],
    groups: list[str] | None = None,
    min_k: int = 1,
    max_k: int = 6,
    max_per_group: int = 3,
    threshold_no_answer: float = THRESHOLD_NO_ANSWER,
    threshold_confident: float = THRESHOLD_CONFIDENT,
    relative_margin: float = RELATIVE_MARGIN,
) -> Selection:
    if not logodds:
        return Selection([], "no_answer", float("-inf"), [])
    order = sorted(range(len(logodds)), key=lambda i: logodds[i], reverse=True)
    best = logodds[order[0]]
    if best < threshold_no_answer:
        return Selection([], "no_answer", best, list(logodds))
    zone = "confident" if best >= threshold_confident else "uncertain"

    chosen: list[int] = []
    per_group: dict[str, int] = {}
    for i in order:
        if len(chosen) >= max_k:
            break
        good = logodds[i] >= threshold_no_answer and logodds[i] >= best - relative_margin
        if not good and len(chosen) >= min_k:
            break
        g = groups[i] if groups else str(i)
        if per_group.get(g, 0) >= max_per_group:
            continue
        per_group[g] = per_group.get(g, 0) + 1
        chosen.append(i)
    return Selection(chosen, zone, best, list(logodds))


class RerankerError(RuntimeError):
    pass


class Reranker:

    def __init__(
        self,
        model_name: str | None = None,
        backend: str = "local",
        instruction: str = INSTRUCTION_STRICT,
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
                model_name = os.getenv("RERANKER_MODEL") or DEFAULT_API_MODEL

        self.model_name = model_name
        self.backend = backend
        self.instruction = instruction
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.debug = debug

        self.api_use_template = api_use_template
        self.api_timeout = api_timeout
        self.api_max_docs_per_request = api_max_docs_per_request
        self.api_retries = api_retries

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


    def score(
        self, query: str, documents: list[str], as_logodds: bool = False
    ) -> list[float]:
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
        scores = self.score(query, documents, as_logodds=as_logodds)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return ranked if top_n is None else ranked[:top_n]

    def select(
        self,
        query: str,
        documents: list[str],
        groups: list[str] | None = None,
        **kwargs,
    ) -> Selection:
        return select_from_scores(self.score(query, documents, as_logodds=True), groups, **kwargs)

    def format_pair(self, query: str, document: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"


    def _pick_device(self) -> str:
        torch = self._torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if self.device is None:
            self.device = self._pick_device()

        dtype = {
            "mps": torch.float16,
            "cuda": torch.bfloat16,
        }.get(self.device, torch.float32)

        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        except TypeError:
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
                    logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
                except TypeError:
                    logits = self.model(**batch).logits[:, -1, :]
            logodds = (logits[:, self._yes_id] - logits[:, self._no_id]).float().tolist()
            for i, value in zip(idx, logodds):
                result[i] = value
        return result


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
                time.sleep(2**attempt)
            except httpx.HTTPStatusError as exc:
                raise RerankerError(f"HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc
        raise RerankerError(f"/rerank не ответил после {self.api_retries} попыток: {last_error}")

    def _score_api(self, query: str, documents: list[str]) -> list[float]:
        import httpx

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
                    "top_n": len(part),
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
                    result[start + i] = prob_to_logodds(raw) if 0.0 <= raw <= 1.0 else raw

        missing = sum(x is None for x in result)
        if missing:
            raise RerankerError(f"/rerank не вернул оценки для {missing} из {len(result)} документов")
        return result


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
    selection = reranker.select(query, docs)
    print(f"Зона: {selection.zone}, лучший log-odds {selection.best:+.2f}, для LLM: {selection.indices}")
    print(f"Время: {time.perf_counter() - started:.1f} с (первый запуск включает загрузку модели)")
