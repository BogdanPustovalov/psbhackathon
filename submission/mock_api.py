"""Local mock of the platform proxy for offline testing of solution.py.

Implements POST /embeddings, /rerank and /chat/completions with deterministic fake models:
  - embeddings: hashed bag of word stems (256 dims);
  - rerank: word-overlap relevance squashed to [0, 1];
  - chat: echoes the beginning of the first fragment (no real generation).
Every chat request is appended to MOCK_LOG (JSONL) so prompts and contexts can be inspected.
MOCK_CHAT_DELAY slows down /chat/completions. MOCK_FAIL_RATE (0..1) randomly returns HTTP 429/503 to exercise retries.

Usage:  python mock_api.py --port 8765
Not part of the submission ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORD = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
LOCK = threading.Lock()
FAIL_RATE = float(os.getenv("MOCK_FAIL_RATE", "0"))
CHAT_DELAY = float(os.getenv("MOCK_CHAT_DELAY", "0"))  # seconds, to test the time budget
LOG_PATH = os.getenv("MOCK_LOG", "mock_requests.jsonl")


def stems(text: str) -> list[str]:
    return [w[:5] for w in WORD.findall(text.lower().replace("ё", "е")) if len(w) > 2]


def embed(text: str, dims: int = 256) -> list[float]:
    vec = [0.0] * dims
    for s in stems(text):
        h = int(hashlib.md5(s.encode()).hexdigest(), 16)
        vec[h % dims] += 1.0 if (h >> 8) % 2 else -1.0
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def relevance(query: str, doc: str) -> float:
    q, d = set(stems(query)), set(stems(doc))
    if not q:
        return 0.0
    overlap = len(q & d) / len(q)
    return 1.0 / (1.0 + math.exp(-(overlap * 12 - 4)))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence default logging
        pass

    def reply(self, code: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if not self.headers.get("Authorization", "").startswith("Bearer "):
            return self.reply(401, {"error": "no key"})
        if FAIL_RATE and random.random() < FAIL_RATE:
            return self.reply(random.choice([429, 503]), {"error": "simulated"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        path = self.path.rstrip("/")
        if path.endswith("/embeddings"):
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return self.reply(200, {"data": [{"index": i, "embedding": embed(t)} for i, t in enumerate(texts)]})
        if path.endswith("/rerank"):
            scores = [relevance(body["query"], d) for d in body["documents"]]
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: body.get("top_n", len(scores))]
            return self.reply(200, {"results": [{"index": i, "relevance_score": scores[i]} for i in order]})
        if path.endswith("/chat/completions"):
            time.sleep(CHAT_DELAY)
            user = body["messages"][-1]["content"]
            with LOCK, open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(body, ensure_ascii=False) + "\n")
            m = re.search(r"\[1\] ([^\n]+)\n([^\n]+)", user)
            text = f"MOCK-ответ. Источник: {m.group(1)[:120]}. {m.group(2)[:200]}" if m else "MOCK-ответ: фрагментов нет."
            if "Черновик ответа" in user:
                text = "MOCK-проверено. " + user.rsplit("Черновик ответа:\n", 1)[-1][:400]
            return self.reply(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})
        return self.reply(404, {"error": "unknown path"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
