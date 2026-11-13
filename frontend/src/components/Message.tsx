import type { Source } from "../api";
import type { AssistantEntry, ChatEntry } from "../useChat";
import { AnswerText } from "./AnswerText";
import { MessageActions } from "./MessageActions";
import { SourceCard } from "./SourceCard";
import { ConfidenceBadge, EditionNotice, MissingBlock } from "./TrustSignals";

type Props = {
  entry: ChatEntry;
  disabled: boolean;
  isLast: boolean;
  activeN: number | null;
  onRetry: (id: string) => void;
  onSend: (text: string) => void;
  onOpenSource: (entryId: string, source: Source) => void;
};

export function Message(props: Props) {
  const { entry } = props;
  if (entry.role === "user") {
    return (
      <div className="message message-user">
        <div className="bubble">{entry.content}</div>
      </div>
    );
  }
  return <AssistantMessage {...props} entry={entry} />;
}

function AssistantMessage({
  entry,
  disabled,
  isLast,
  activeN,
  onRetry,
  onSend,
  onOpenSource,
}: Props & { entry: AssistantEntry }) {
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
  const open = (source: Source) => onOpenSource(entry.id, source);

  return (
    <div className="message message-assistant">
      <div className="bubble">
        {entry.response && <ConfidenceBadge response={entry.response} />}
        <AnswerText text={entry.content} sources={sources} activeN={activeN} onCite={open} />
        <MissingBlock items={entry.response?.missing ?? []} />
        <EditionNotice sources={sources} />

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
            <p className="sources-title">Источники · {sources.length}</p>
            <ol className="sources-list">
              {sources.map((source, index) => {
                const n = source.n ?? index + 1;
                return <SourceCard key={source.id} source={source} n={n} active={n === activeN} onOpen={open} />;
              })}
            </ol>
          </div>
        )}

        <MessageActions answer={entry.content} sources={sources} elapsedMs={entry.elapsedMs} />
      </div>
    </div>
  );
}
