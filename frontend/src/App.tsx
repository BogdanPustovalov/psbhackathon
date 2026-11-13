import { useEffect, useRef } from "react";
import { Composer } from "./components/Composer";
import { Message } from "./components/Message";
import { EXAMPLE_QUESTIONS } from "./examples";
import { useChat } from "./useChat";

const IS_MOCK = import.meta.env.VITE_USE_MOCK === "1";

export default function App() {
  const { entries, pending, send, retry, reset } = useChat();
  const endRef = useRef<HTMLDivElement>(null);

  // Keep the latest message in view.
  useEffect(() => {
    if (entries.length === 0) {
      endRef.current?.closest(".chat")?.scrollTo({ top: 0 });
      return;
    }
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [entries.length, pending]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-title">
          <h1>Юридический ассистент</h1>
          <p className="muted">Отвечает только по документам базы знаний и ссылается на них</p>
        </div>
        <div className="header-actions">
          {IS_MOCK && <span className="badge">Демо-данные</span>}
          <button type="button" className="ghost-button" onClick={reset} disabled={pending || entries.length === 0}>
            Новый диалог
          </button>
        </div>
      </header>

      <main className="chat" aria-live="polite">
        <div className="chat-inner">
          {entries.length === 0 ? (
            <EmptyState onPick={send} />
          ) : (
            entries.map((entry, index) => (
              <Message
                key={entry.id}
                entry={entry}
                disabled={pending}
                isLast={index === entries.length - 1}
                onRetry={retry}
                onSend={send}
              />
            ))
          )}

          {pending && (
            <div className="message message-assistant">
              <div className="bubble bubble-pending">
                Ищу в документах<span className="dots" aria-hidden="true" />
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>
      </main>

      <footer className="composer-area">
        <Composer disabled={pending} onSend={send} />
        <p className="hint muted">Enter — отправить, Shift+Enter — новая строка</p>
      </footer>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="empty-state">
      <h2>Чем помочь?</h2>
      <p className="muted">
        Спросите о договорах, внутренних регламентах или нормах закона. Ассистент ответит со ссылками на
        конкретные пункты и честно скажет, если в базе нет нужных документов.
      </p>
      <div className="examples">
        {EXAMPLE_QUESTIONS.map((question) => (
          <button key={question} type="button" className="example" onClick={() => onPick(question)}>
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}
