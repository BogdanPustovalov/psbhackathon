from fastapi.testclient import TestClient

from app.api.routes.query import get_pipeline
from app.main import app
from app.services.pipeline import ExtractiveLLM, RAGPipeline, get_corpus


client = TestClient(app)


def test_prepared_corpus_and_document_routes() -> None:
    corpus = get_corpus()
    assert len(corpus.chunks) == 182
    assert corpus.index.count == len(corpus.chunks)

    listing = client.get("/api/v1/documents")
    assert listing.status_code == 200
    documents = listing.json()
    assert documents
    detail = client.get(f"/api/v1/documents/{documents[0]['doc_id']}")
    assert detail.status_code == 200
    assert detail.json()["text"]

    for chunk in corpus.chunks:
        document = corpus.documents[chunk.metadata["doc_id"]]
        assert document.text[chunk.metadata["char_start"] : chunk.metadata["char_end"]] == chunk.text


def test_query_returns_real_indexed_sources() -> None:
    pipeline = RAGPipeline(get_corpus())
    pipeline.rag.llm = ExtractiveLLM()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        response = client.post(
            "/api/v1/query",
            json={"question": "Арендная плата по договору А-08/2024", "top_k": 3},
        )
    finally:
        app.dependency_overrides.pop(get_pipeline, None)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["query"] == "Арендная плата по договору А-08/2024"
    assert body["sources"]
    assert all(source["metadata"]["doc_id"] in get_corpus().documents for source in body["sources"])
    assert body["debug"]["indexed_chunks"] == 182
