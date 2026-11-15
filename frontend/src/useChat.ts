// Dialogue state: the list of messages, sending a question with history,
// retry after an error, a new dialogue. The dialogue is kept in localStorage
// so it survives a page reload.

import { useCallback, useEffect, useState } from "react";
import { askQuestion, USE_MOCK, type ChatMessage, type QueryResponse } from "./api";

export type UserEntry = {
  id: string;
  role: "user";
  content: string;
};

export type AssistantEntry = {
  id: string;
  role: "assistant";
  content: string;
  response?: QueryResponse;
  error?: string;
  // How long the answer took, measured in the browser.
  elapsedMs?: number;
};

export type ChatEntry = UserEntry | AssistantEntry;

const STORAGE_KEY = USE_MOCK ? "legal-assistant:chat:mock:v1" : "legal-assistant:chat:v1";
const HISTORY_LIMIT = 10;

const newId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

function loadEntries(): ChatEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? (parsed as ChatEntry[]) : [];
  } catch {
    return [];
  }
}

function saveEntries(entries: ChatEntry[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Storage may be unavailable (private mode): the chat still works without it.
  }
}

// History for the API: answered turns only, without failed requests.
function toHistory(entries: ChatEntry[]): ChatMessage[] {
  const history: ChatMessage[] = [];
  for (const [index, entry] of entries.entries()) {
    const answer = entries[index + 1];
    if (entry.role === "user" && answer?.role === "assistant" && !answer.error) {
      history.push({ role: "user", content: entry.content }, { role: "assistant", content: answer.content });
    }
  }
  return history.slice(-HISTORY_LIMIT);
}

function errorText(error: unknown): string {
  if (error instanceof TypeError) return "Сервер недоступен. Проверьте, что бэкенд запущен.";
  if (error instanceof Error) return error.message;
  return "Неизвестная ошибка";
}

export function useChat() {
  const [entries, setEntries] = useState<ChatEntry[]>(loadEntries);
  const [pending, setPending] = useState(false);

  useEffect(() => saveEntries(entries), [entries]);

  const ask = useCallback(async (question: string, previous: ChatEntry[], retryId?: string) => {
    setPending(true);
    const started = performance.now();
    const id = retryId ?? newId();
    function saveAnswer(answer: AssistantEntry) {
      setEntries((current) =>
        retryId ? current.map((entry) => (entry.id === retryId ? answer : entry)) : [...current, answer],
      );
    }
    try {
      const response = await askQuestion(question, toHistory(previous));
      saveAnswer({
        id,
        role: "assistant",
        content: response.answer,
        response,
        elapsedMs: Math.round(performance.now() - started),
      });
    } catch (error) {
      saveAnswer({ id, role: "assistant", content: "", error: errorText(error) });
    } finally {
      setPending(false);
    }
  }, []);

  const send = useCallback(
    (text: string) => {
      const question = text.trim();
      if (!question || pending) return;
      const previous = entries;
      setEntries([...previous, { id: newId(), role: "user", content: question }]);
      void ask(question, previous);
    },
    [entries, pending, ask],
  );

  // Retry the question that preceded a failed answer.
  const retry = useCallback(
    (failedId: string) => {
      if (pending) return;
      const index = entries.findIndex((entry) => entry.id === failedId && entry.role === "assistant" && entry.error);
      const question = entries[index - 1];
      if (question?.role !== "user") return;
      void ask(question.content, entries.slice(0, index - 1), failedId);
    },
    [entries, pending, ask],
  );

  const reset = useCallback(() => {
    if (!pending) setEntries([]);
  }, [pending]);

  return { entries, pending, send, retry, reset };
}
