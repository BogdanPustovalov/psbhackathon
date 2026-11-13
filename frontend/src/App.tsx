import { useCallback, useEffect, useRef, useState } from "react";
import type { Source } from "./api";
import { Composer } from "./components/Composer";
import { DocumentPanel, type PanelView } from "./components/DocumentPanel";
import { Message } from "./components/Message";
import { EXAMPLE_QUESTIONS } from "./examples";
import { useChat } from "./useChat";

const IS_MOCK = import.meta.env.VITE_USE_MOCK === "1";

type ActiveSource = { entryId: string; n: number };

export default function App() {
  const { entries, pending, send, retry, reset } = useChat();
  const [panel, setPanel] = useState<PanelView | null>(null);
  const [active, setActive] = useState<ActiveSource | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  // Keep the latest message in view.
  useEffect(() => {
    if (entries.length === 0) {
      endRef.current?.closest(".chat")?.scrollTo({ top: 0 });
      return;
    }
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [entries.length, pending]);

  const openSource = useCallback((entryId: string, source: Source) => {
    const { doc_id, section, char_start, char_end } = source.metadata;
    if (!doc_id) return;
    const n = source.n ?? 0;
    setActive({ entryId, n });
    setPanel({
      kind: "doc",
      docId: doc_id,
      highlight: { text: source.text, start: char_start, end: char_end },
      caption: [`[${n}]`, section].filter(Boolean).join(" "),
    });
  }, []);

  const openDocument = useCallback((docId: string) => {
    setActive(null);
    setPanel({ kind: "doc", docId });
  }, []);

  const showList = useCallback(() => {
    setActive(null);
    setPanel({ kind: "list" });
  }, []);

  const closePanel = useCallback(() => {
    setActive(null);
    setPanel(null);
  }, []);

  function newDialog() {
    closePanel();
    reset();
  }

  return (
    <div className={panel ? "layout layout-with-panel" : "layout"}>
      <div className="app">
        <header className="app-header">
          <div className="app-title">
            <h1>Юридический ассистент</h1>
          </div>
          <div className="header-actions">
            {IS_MOCK && <span className="badge">Демо-данные</span>}
            <button type="button" className="ghost-button" onClick={panel?.kind === "list" ? closePanel : showList}>
              Документы
            </button>
            <button type="button" className="ghost-button" onClick={newDialog} disabled={pending || entries.length === 0}>
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
                  activeN={active?.entryId === entry.id ? active.n : null}
                  onRetry={retry}
                  onSend={send}
                  onOpenSource={openSource}
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

      {panel && (
        <DocumentPanel view={panel} onOpenDocument={openDocument} onShowList={showList} onClose={closePanel} />
      )}
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
