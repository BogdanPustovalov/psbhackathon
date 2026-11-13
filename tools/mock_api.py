"""Local stand-in for the organizers' OpenAI-compatible proxy.

Serves the three endpoints solution.py calls, so the whole run can be tested
without a key and without spending the 5 daily submissions:

    POST /chat/completions   echo-style answer built from the prompt (not a real model)
    POST /embeddings         deterministic hashed bag-of-words vectors (dim 1024)
    POST /rerank             word-overlap scores, Cohere/Jina-style response
    GET  /stats              call counters, latency, largest batch sizes

It checks what the platform would reject (missing Bearer key, malformed body)
and can inject 429/5xx errors and latency to test retries and the 20-minute budget.

Usage:
    python tools/mock_api.py --port 8099 --fail-rate 0.1 --latency-ms 300
    LLM_BASE_URL=http://127.0.0.1:8099 LLM_API_KEY=test-key python solution.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 1024
WORD = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)


def tokens(text: str) -> list[str]:
    # Crude stemming: first 6 letters, so "договора" and "договору" match.
    return [w[:6] for w in WORD.findall(text.lower().replace("ё", "е")) if len(w) > 2]


def embed(text: str) -> list[float]:
    vector = [0.0] * DIM
    for token in tokens(text):
        digest = hashlib.md5(token.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % DIM
        vector[index] += 1.0 if digest[4] % 2 else -1.0
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def overlap(query: str, document: str) -> float:
    q = set(tokens(query))
    return len(q & set(tokens(document))) / max(len(q), 1)


def chat_answer(messages: list[dict]) -> str:
    # Not a model: returns the first sentences of the context so the pipeline
    # produces a plausible, citeable answer and the CSV can be checked.
    text = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") != "system")
    sentences = re.findall(r"[^.\n]*\d+\.\d+\.[^.\n]*\.", text)
    body = " ".join(sentences[:2]) if sentences else "В предоставленных документах нет ответа на этот вопрос."
    return "[MOCK] " + body[:600]


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: dict[str, int] = {}
        self.errors_injected = 0
        self.max_batch: dict[str, int] = {}
        self.prompt_chars_total = 0
        self.started = time.time()

    def record(self, path: str, batch: int = 0, chars: int = 0) -> None:
        with self.lock:
            self.calls[path] = self.calls.get(path, 0) + 1
            self.max_batch[path] = max(self.max_batch.get(path, 0), batch)
            self.prompt_chars_total += chars

    def as_dict(self) -> dict:
        with self.lock:
            return {
                "calls": dict(self.calls),
                "errors_injected": self.errors_injected,
                "max_batch": dict(self.max_batch),
                "prompt_chars_total": self.prompt_chars_total,
                "uptime_s": round(time.time() - self.started, 1),
            }


def make_handler(key: str, fail_rate: float, latency_ms: int, stats: Stats, rng: random.Random):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict | str) -> None:
            body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/").endswith("/stats"):
                self._send(200, stats.as_dict())
            elif self.path.rstrip("/").endswith("/models"):
                self._send(200, {"data": [{"id": "mock"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.headers.get("Authorization") != f"Bearer {key}":
                self._send(401, {"error": "token_not_found_in_db"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except (ValueError, json.JSONDecodeError):
                self._send(400, {"error": "invalid json"})
                return

            if latency_ms:
                time.sleep(latency_ms / 1000 * (0.5 + rng.random()))
            if fail_rate and rng.random() < fail_rate:
                with stats.lock:
                    stats.errors_injected += 1
                self._send(rng.choice([429, 500, 502, 503]), {"error": "injected failure"})
                return

            path = self.path.rstrip("/")
            if path.endswith("/chat/completions"):
                messages = body.get("messages") or []
                if not messages or not body.get("model"):
                    self._send(400, {"error": "model and messages are required"})
                    return
                chars = sum(len(str(m.get("content", ""))) for m in messages)
                stats.record("chat", chars=chars)
                self._send(200, {
                    "id": "mock",
                    "object": "chat.completion",
                    "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": chat_answer(messages)}}],
                    "usage": {"prompt_tokens": chars // 4, "completion_tokens": 50},
                })
            elif path.endswith("/embeddings"):
                inputs = body.get("input")
                if isinstance(inputs, str):
                    inputs = [inputs]
                if not isinstance(inputs, list) or not body.get("model"):
                    self._send(400, {"error": "model and input are required"})
                    return
                stats.record("embeddings", batch=len(inputs))
                self._send(200, {
                    "object": "list",
                    "model": body["model"],
                    "data": [{"object": "embedding", "index": i, "embedding": embed(str(t))}
                             for i, t in enumerate(inputs)],
                })
            elif path.endswith("/rerank"):
                documents = body.get("documents")
                query = body.get("query")
                if not isinstance(documents, list) or not query or not body.get("model"):
                    self._send(400, {"error": "model, query and documents are required"})
                    return
                stats.record("rerank", batch=len(documents))
                texts = [d if isinstance(d, str) else str(d.get("text", "")) for d in documents]
                results = sorted(
                    ({"index": i, "relevance_score": round(overlap(query, t), 4)} for i, t in enumerate(texts)),
                    key=lambda r: -r["relevance_score"],
                )
                top_n = body.get("top_n") or len(results)
                self._send(200, {"model": body["model"], "results": results[: int(top_n)]})
            else:
                self._send(404, {"error": f"unknown endpoint {self.path}"})

        def log_message(self, *args) -> None:
            pass

    return Handler


def serve(port: int = 0, key: str = "test-key", fail_rate: float = 0.0, latency_ms: int = 0,
          seed: int = 0) -> tuple[ThreadingHTTPServer, Stats]:
    """Start the mock in a background thread; returns the server (its port is server.server_port)."""
    stats = Stats()
    handler = make_handler(key, fail_rate, latency_ms, stats, random.Random(seed))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--key", default="test-key", help="expected Bearer key (default: test-key)")
    parser.add_argument("--fail-rate", type=float, default=0.0, help="share of requests answered with 429/5xx")
    parser.add_argument("--latency-ms", type=int, default=0, help="average added latency per request")
    args = parser.parse_args()
    server, _ = serve(args.port, args.key, args.fail_rate, args.latency_ms)
    print(f"Mock API on http://127.0.0.1:{server.server_port}  (key: {args.key}). Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
