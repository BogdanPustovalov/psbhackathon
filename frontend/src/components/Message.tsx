import type { Source } from "../api";
import type { AssistantEntry, ChatEntry } from "../useChat";

type Props = {
  entry: ChatEntry;
  disabled: boolean;
  isLast: boolean;
  onRetry: (id: string) => void;
  onSend: (text: string) => void;
};

export function Message({ entry, disabled, isLast, onRetry, onSend }: Props) {
  if (entry.role === "user") {
    return (
      <div className="message message-user">
        <div className="bubble">{entry.content}</div>
      </div>
    );
  }
  return <AssistantMessage entry={entry} disabled={disabled} isLast={isLast} onRetry={onRetry} onSend={onSend} />;
}

function AssistantMessage({ entry, disabled, isLast, onRetry, onSend }: Props & { entry: AssistantEntry }) {
  if (entry.error) {
    return (
      <div className="message message-assistant">
        <div className="bubble bubble-error" role="alert">
          <p>Не удалось получить ответ: {entry.error}</p>
          <button type="button" className="link-button" disabled={disabled} onClick={() => onRetry(entry.id)}>
            Повторить
          </button>
        </div>
      </div>
    );
  }

  const sources = entry.response?.sources ?? [];
  const clarification = entry.response?.clarification;

  return (
    <div className="message message-assistant">
      <div className="bubble">
        <p className="answer-text">{entry.content}</p>

        {clarification && clarification.options.length > 0 && (
          <div className="clarification">
            <p className="clarification-question">{clarification.question}</p>
            <div className="clarification-options">
              {clarification.options.map((option) => (
                <button
                  key={option}
                  type="button"
                  className="chip"
                  disabled={disabled || !isLast}
                  onClick={() => onSend(option)}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
        )}

        {sources.length > 0 && (
          <div className="sources">
            <p className="sources-title">Источники</p>
            <ol className="sources-list">
              {sources.map((source, index) => (
                <SourceLine key={source.id} source={source} n={source.n ?? index + 1} />
              ))}
            </ol>
          </div>
        )}
      </div>
    </div>
  );
}

function SourceLine({ source, n }: { source: Source; n: number }) {
  const { doc_title, section, edition } = source.metadata;
  const expired = edition?.status === "expired";
  return (
    <li className="source-line">
      <span className="source-n">[{n}]</span>
      <span>
        {doc_title ?? "Фрагмент"}
        {section && <span className="muted">, {section}</span>}
        {edition && (
          <span className={expired ? "edition edition-expired" : "edition"}>
            {edition.label}
            {expired && " · утратила силу"}
          </span>
        )}
      </span>
    </li>
  );
}
