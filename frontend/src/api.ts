// Types follow docs/api.md. New fields are optional: the backend fills them gradually.

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";
const USE_MOCK = import.meta.env.VITE_USE_MOCK === "1";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type DocType = "contract" | "law" | "internal";

export type Edition = {
  label: string;
  valid_from: string | null;
  valid_to: string | null;
  status: "active" | "amended" | "expired";
};

export type SourceMetadata = {
  doc_id?: string;
  doc_title?: string;
  doc_type?: DocType;
  section?: string;
  edition?: Edition;
  char_start?: number;
  char_end?: number;
  [key: string]: unknown;
};

export type Source = {
  id: string;
  n?: number;
  text: string;
  score: number;
  metadata: SourceMetadata;
};

export type Zone = "no_answer" | "uncertain" | "confident";

export type Confidence = {
  zone: Zone;
  best_score: number | null;
};

export type Clarification = {
  question: string;
  options: string[];
};

export type QueryResponse = {
  query: string;
  answer: string;
  sources: Source[];
  confidence?: Confidence | null;
  missing?: string[];
  clarification?: Clarification | null;
  debug?: Record<string, unknown>;
};

export type DocumentInfo = {
  doc_id: string;
  doc_title: string;
  doc_type: DocType;
  edition: Edition | null;
  related: string[];
};

export type DocumentDetail = DocumentInfo & {
  text: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    throw new Error(`Ошибка API: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function askQuestion(
  question: string,
  history: ChatMessage[] = [],
): Promise<QueryResponse> {
  if (USE_MOCK) {
    const mock = await import("./mock");
    return mock.askQuestion(question, history);
  }
  return request<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history, top_k: 5, use_reranker: true }),
  });
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  if (USE_MOCK) {
    const mock = await import("./mock");
    return mock.listDocuments();
  }
  return request<DocumentInfo[]>("/documents");
}

export async function getDocument(docId: string): Promise<DocumentDetail> {
  if (USE_MOCK) {
    const mock = await import("./mock");
    return mock.getDocument(docId);
  }
  return request<DocumentDetail>(`/documents/${encodeURIComponent(docId)}`);
}
