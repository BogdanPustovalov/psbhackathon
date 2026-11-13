// Mock API. Enabled with VITE_USE_MOCK=1 (npm run dev:mock).
// Implements the same functions as the real API in ../api.ts, using the
// format from docs/api.md, so the interface can be built before the backend.

import type { ChatMessage, DocumentDetail, DocumentInfo, QueryResponse, Source } from "../api";
import { findDocument, MOCK_DOCUMENTS } from "./documents";
import { FALLBACK, SCENARIOS, type Scenario, type SourceSpec } from "./scenarios";

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function pickScenario(question: string, history: ChatMessage[]): Scenario {
  const current = question.toLowerCase();
  const byQuestion = SCENARIOS.find((s) => s.match(current));
  if (byQuestion) return byQuestion;

  // Follow-up questions ("А на сколько?", a clicked clarification option)
  // only make sense together with the previous user messages.
  const previous = history
    .filter((m) => m.role === "user")
    .slice(-2)
    .map((m) => m.content.toLowerCase());
  const withContext = [...previous, current].join(" ");
  return SCENARIOS.find((s) => s.match(withContext)) ?? FALLBACK;
}

function buildSource(spec: SourceSpec, n: number, scenario: string): Source {
  const doc = findDocument(spec.doc_id);
  if (!doc) throw new Error(`Mock: unknown document ${spec.doc_id}`);

  const start = doc.text.indexOf(spec.quote);
  if (start < 0) {
    console.warn(`Mock: quote not found in ${spec.doc_id} (scenario ${scenario})`);
  }

  return {
    id: `${spec.doc_id}#${n}`,
    n,
    text: spec.quote,
    score: spec.score,
    metadata: {
      doc_id: doc.doc_id,
      doc_title: doc.doc_title,
      doc_type: doc.doc_type,
      section: spec.section,
      edition: doc.edition ?? undefined,
      ...(start >= 0 ? { char_start: start, char_end: start + spec.quote.length } : {}),
    },
  };
}

export async function askQuestion(question: string, history: ChatMessage[]): Promise<QueryResponse> {
  const started = performance.now();
  await delay(600 + Math.random() * 900);

  const scenario = pickScenario(question, history);
  const sources = scenario.sources.map((spec, i) => buildSource(spec, i + 1, scenario.name));
  const best = sources.length ? Math.max(...sources.map((s) => s.score)) : 0.8;

  return {
    query: question,
    answer: scenario.answer,
    sources,
    confidence: { zone: scenario.zone, best_score: best },
    missing: scenario.missing ?? [],
    clarification: scenario.clarification ?? null,
    debug: {
      mock: true,
      scenario: scenario.name,
      history_length: history.length,
      timings_ms: { total: Math.round(performance.now() - started) },
    },
  };
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  await delay(200);
  return MOCK_DOCUMENTS.map(({ text: _text, ...info }) => info);
}

export async function getDocument(docId: string): Promise<DocumentDetail> {
  await delay(200);
  const doc = findDocument(docId);
  if (!doc) throw new Error("Ошибка API: 404");
  return doc;
}
