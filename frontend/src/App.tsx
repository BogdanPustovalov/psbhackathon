import { FormEvent, useState } from "react";
import { askQuestion, QueryResponse } from "./api";

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError("");
    try {
      setResult(await askQuestion(question.trim()));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Неизвестная ошибка");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">RAG PLATFORM</p>
        <h1>Спросите свою базу знаний</h1>
        <p className="subtitle">
          Семантический поиск, reranking и ответ LLM с прозрачными источниками.
        </p>

        <form className="query-form" onSubmit={handleSubmit}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Например: какие правила описаны в загруженных документах?"
            rows={4}
          />
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? "Ищу…" : "Задать вопрос"}
          </button>
        </form>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className="result-card">
            <h2>Ответ</h2>
            <p>{result.answer}</p>
            <h3>Источники</h3>
            {result.sources.length === 0 ? (
              <p className="muted">Источники пока не подключены.</p>
            ) : (
              <ul>
                {result.sources.map((source) => (
                  <li key={source.id}>
                    <span>{source.text}</span>
                    <small>score: {source.score.toFixed(3)}</small>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </section>
    </main>
  );
}
