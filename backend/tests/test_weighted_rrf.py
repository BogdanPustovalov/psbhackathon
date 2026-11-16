"""Checks that the submission and diagram use weighted RRF then reranker Top-N."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.rag import Document, pipeline
from app.services.rag.diagnostic import collect_diagnostic_result
from app.services.rag.llm import OpenAICompatibleLLM
from app.services.rag.pipeline import rank_hybrid_candidates
from app.services.rag.vector_store import FaissVectorIndex


class FakeEmbedding:
    model_name = "fake-embedding-v1"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


class FakeReranker:
    def rerank(
        self, query: str, documents: list[Document], *, top_k: int
    ) -> list[Document]:
        assert top_k >= len(documents)
        scores = {"0": 10.0, "1": 5.0, "2": 1.0}
        for doc in documents:
            doc.score = scores[doc.metadata["chunk_id"]]
        return sorted(documents, key=lambda doc: doc.score, reverse=True)


class FakeLLM:
    def __init__(self, **kwargs: object) -> None:
        pass

    def generate(self, question: str, sources: list[Document]) -> str:
        return ",".join(doc.metadata["chunk_id"] for doc in sources)


class WeightedRRFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = FaissVectorIndex()
        self.index.documents = [
            Document(f"Чанк {i}", {"chunk_id": str(i), "filename": "doc.md"})
            for i in range(3)
        ]
        self.index._index = object()
        self.embedding = FakeEmbedding()

    def test_weighted_rrf_formula_and_reranker_final_order(self) -> None:
        with patch.object(
            self.index, "_semantic_hits", return_value=[(0, 0.9), (1, 0.8), (2, 0.7)]
        ), patch.object(
            self.index, "_bm25_hits", return_value=[(1, 12.0), (2, 11.0), (0, 10.0)]
        ):
            trace = self.index.hybrid_candidates(
                "вопрос", self.embedding, top_k=3, faiss_k=3, bm25_k=3
            )
        self.assertEqual([doc.metadata["chunk_id"] for doc in trace.rrf_hits], ["1", "0", "2"])
        self.assertTrue(math.isclose(trace.rrf_hits[0].score, 0.4 / 62 + 0.6 / 61))
        self.assertEqual(trace.rrf_hits[0].metadata["rrf_k"], 60)

        ranking = rank_hybrid_candidates(
            "вопрос", trace, reranker=FakeReranker(),
            top_k=2, reranker_input_k=3, reranker_output_k=3,
        )
        self.assertEqual([doc.metadata["chunk_id"] for doc in ranking.reranker_input], ["1", "0", "2"])
        self.assertEqual([doc.score for doc in ranking.reranked], [10.0, 5.0, 1.0])
        self.assertEqual([doc.metadata["chunk_id"] for doc in ranking.final], ["0", "1"])
        self.assertEqual(ranking.final, ranking.reranked[:2])

    def test_diagram_context_uses_reranker_top_n(self) -> None:
        with patch.object(
            self.index, "_semantic_hits", return_value=[(0, 0.9), (1, 0.8), (2, 0.7)]
        ), patch.object(
            self.index, "_bm25_hits", return_value=[(1, 12.0), (2, 11.0), (0, 10.0)]
        ):
            result = collect_diagnostic_result(
                "q1", "вопрос", self.index, self.embedding, FakeReranker(),
                faiss_top_k=3, bm25_top_k=3, rrf_top_k=3,
                reranker_input_top_k=3, reranker_top_k=3, final_context_top_k=2,
            )
        self.assertEqual(len(json.loads(result.row["Qwen Reranker Top-3"])), 3)
        self.assertEqual(
            [item["metadata"]["chunk_id"] for item in json.loads(result.row["Final Top-2"])],
            ["0", "1"],
        )
        self.assertEqual(result.row["Final Context"], OpenAICompatibleLLM._source_context(result.final_hits))

    def test_submission_and_diagram_use_same_ranked_chunks(self) -> None:
        def fake_ingest(rag: pipeline.RAGSystem, documents: list[Document]) -> int:
            rag.index.documents = list(self.index.documents)
            rag.index._index = object()
            return len(documents)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            questions = root / "questions.csv"
            with questions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows([["id", "question"], ["q2", "Вопрос 2"], ["q1", "Вопрос 1"]])
            kb = root / "kb.zip"
            kb.touch()
            output = root / "submission.csv"
            diagram = root / "diagramm.csv"
            settings = {
                "QUESTIONS_PATH": str(questions), "KNOWLEDGE_BASE_PATH": str(kb),
                "OUTPUT_PATH": str(output), "DIAGRAMM_OUTPUT": str(diagram),
                "USE_RERANKER": "1", "MAX_WORKERS": "1", "TOP_K": "3",
                "RRF_TOP_K": "3", "RERANKER_CANDIDATES": "3",
                "RERANKER_TOP_N": "3", "FINAL_CONTEXT_TOP_K": "2",
                "RRF_SEMANTIC_WEIGHT": "0.4", "RRF_BM25_WEIGHT": "0.6",
            }
            with patch.dict(os.environ, settings), patch.object(
                pipeline, "OpenAICompatibleLLM", FakeLLM
            ), patch.object(pipeline, "EmbeddingClient", FakeEmbedding), patch.object(
                pipeline, "Reranker", FakeReranker
            ), patch.object(
                pipeline, "load_documents", return_value=list(self.index.documents)
            ), patch.object(pipeline.RAGSystem, "ingest", fake_ingest), patch.object(
                FaissVectorIndex, "_semantic_hits",
                return_value=[(0, 0.9), (1, 0.8), (2, 0.7)],
            ), patch.object(
                FaissVectorIndex, "_bm25_hits",
                return_value=[(1, 12.0), (2, 11.0), (0, 10.0)],
            ):
                pipeline.run_platform_submission()

            with output.open(encoding="utf-8", newline="") as handle:
                answers = list(csv.DictReader(handle))
            with diagram.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["id"] for row in answers], ["q2", "q1"])
            self.assertEqual([row["answer"] for row in answers], ["0,1", "0,1"])
            self.assertEqual([row["id"] for row in rows], ["q2", "q1"])
            for row in rows:
                self.assertEqual(
                    [doc["metadata"]["chunk_id"] for doc in json.loads(row["Final Top-2"])],
                    ["0", "1"],
                )
                self.assertEqual(len(json.loads(row["Qwen Reranker Top-3"])), 3)
                self.assertEqual(row["error"], "")

    def test_unchanged_knowledge_base_reuses_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"RAG_EMBEDDING_CACHE_DIR": temporary}
        ):
            first_embedding = FakeEmbedding()
            first = pipeline.RAGSystem(embedding_client=first_embedding, llm=FakeLLM())
            first.ingest([Document("Одинаковый документ")])
            self.assertEqual(first_embedding.calls, 1)

            second_embedding = FakeEmbedding()
            second = pipeline.RAGSystem(embedding_client=second_embedding, llm=FakeLLM())
            second.ingest([Document("Одинаковый документ")])
            self.assertEqual(second_embedding.calls, 0)
            self.assertEqual(second.document_count, first.document_count)

            second.ingest([Document("Изменённый документ")])
            self.assertEqual(second_embedding.calls, 1)


if __name__ == "__main__":
    unittest.main()
