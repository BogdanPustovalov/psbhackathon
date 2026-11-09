const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";

export type Source = {
  id: string;
  text: string;
  score: number;
  metadata: Record<string, unknown>;
};

export type QueryResponse = {
  answer: string;
  sources: Source[];
  query: string;
};

export async function askQuestion(question: string): Promise<QueryResponse> {
  const response = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: 5, use_reranker: true }),
  });

  if (!response.ok) {
    throw new Error("Не удалось получить ответ от API");
  }

  return response.json() as Promise<QueryResponse>;
}
