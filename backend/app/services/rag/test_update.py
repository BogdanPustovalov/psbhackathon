from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import faiss

import solution_update
from app.services.rag import Document
from app.services.rag.pipeline import split_documents
from app.services.rag.vector_store import FaissVectorIndex


class RetrievalUpdateTests(unittest.TestCase):
    def test_clause_boundaries_and_source_offsets(self) -> None:
        text = (
            "ДОГОВОР ПОСТАВКИ\n\n"
            "6. Приёмка Товара\n\n"
            "6.1. Приёмка по количеству производится при передаче.\n\n"
            "6.2. При обнаружении недостатков покупатель направляет претензию с актом."
        )
        document = Document(text, {"filename": "Договор П-17-2024.md"})
        chunks = split_documents([document], chunk_size=18, chunk_overlap=3)

        self.assertTrue(any(chunk.metadata["clause"] == "6.2." for chunk in chunks))
        self.assertTrue(any(chunk.metadata["section"] == "6. Приёмка Товара" for chunk in chunks))
        for chunk in chunks:
            start, end = chunk.metadata["start_char"], chunk.metadata["end_char"]
            self.assertEqual(chunk.text, text[start:end])
            self.assertLessEqual(chunk.metadata["token_count"], 18)

        compact = "6. Приёмка\n6.1. Срок приёмки — пять дней.\n6.2. Претензия сопровождается актом."
        compact_chunks = split_documents([Document(compact)], chunk_size=12, chunk_overlap=2)
        self.assertTrue(any(chunk.metadata["clause"] == "6.2." for chunk in compact_chunks))

    def test_bm25_uses_lemma_and_filename(self) -> None:
        index = FaissVectorIndex()
        target = Document("Покупателем выявлены недостатки товара.", {"filename": "Договор П-17-2024.md"})
        other = Document("Поставка оборудования произведена.", {"filename": "Другой договор.md"})
        index.add_lexical([target, other])

        self.assertEqual(index.bm25_search("покупателя", top_k=1)[0].text, target.text)
        self.assertEqual(index.bm25_search("П-17/2024", top_k=1)[0].text, target.text)

    def test_real_faiss_and_standalone_parity(self) -> None:
        documents = [Document("первый", {"filename": "a.md"}), Document("второй", {"filename": "b.md"})]
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        index = FaissVectorIndex()
        index.add(documents, vectors)
        self.assertIsInstance(index._index, faiss.IndexFlatIP)
        self.assertEqual(index.search([0.9, 0.1], top_k=1)[0].text, "первый")

        standalone = solution_update.FaissVectorIndex()
        standalone.add([solution_update.Document(item.text, item.metadata) for item in documents], vectors)
        self.assertEqual(standalone.search([0.9, 0.1], top_k=1)[0].text, "первый")

    def test_standalone_platform_writes_answers_in_question_order(self) -> None:
        class FakeEmbeddingClient:
            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        class FakeLLM:
            def __init__(self, **kwargs: object) -> None:
                pass

            def generate(self, question: str, sources: list[solution_update.Document]) -> str:
                return f"Ответ для {question}: {sources[0].metadata['filename']}"

        class FakeReranker:
            calls = 0

            def rerank(
                self, query: str, documents: list[solution_update.Document], *, top_k: int
            ) -> list[solution_update.Document]:
                self.calls += 1
                return documents[:top_k]

        fake_reranker = FakeReranker()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            with (data / "questions.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "question"])
                writer.writerows([["q2", "Второй вопрос"], ["q1", "Первый вопрос"]])
            with zipfile.ZipFile(data / "kb.zip", "w") as archive:
                archive.writestr("Договор.md", "6. Условия\n\n6.1. Покупатель направляет претензию.")

            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ, {"DATA_DIR": str(data), "LLM_API_KEY": "test", "USE_RERANKER": "1", "OUTPUT_PATH": str(root / "reranked.csv")}):
                    with patch.object(solution_update, "EmbeddingClient", FakeEmbeddingClient):
                        with patch.object(solution_update, "OpenAICompatibleLLM", FakeLLM):
                            with patch.object(solution_update, "Reranker", return_value=fake_reranker):
                                output = solution_update.run_platform_submission()
                with output.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
            finally:
                os.chdir(old_cwd)

        self.assertEqual([row["id"] for row in rows], ["q2", "q1"])
        self.assertEqual(output.name, "reranked.csv")
        self.assertIn("Второй вопрос", rows[0]["answer"])
        self.assertIn("Первый вопрос", rows[1]["answer"])
        self.assertEqual(fake_reranker.calls, 2)

    def test_hybrid_lexical_fallback_does_not_call_embedding(self) -> None:
        class FailingEmbeddingClient:
            def embed(self, texts: list[str]) -> list[list[float]]:
                raise AssertionError("embedding called for lexical-only index")

        index = FaissVectorIndex()
        index.add_lexical([Document("Претензия сопровождается актом.")])
        hits = index.hybrid_search("акт", FailingEmbeddingClient())
        self.assertEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
