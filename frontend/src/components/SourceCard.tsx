import type { Source } from "../api";
import { DOC_TYPE_LABEL, EDITION_STATUS_LABEL } from "../labels";
import { EditionBadge } from "./EditionBadge";

type Props = {
  source: Source;
  n: number;
  active: boolean;
  onOpen: (source: Source) => void;
};

export function SourceCard({ source, n, active, onOpen }: Props) {
  const { doc_title, doc_type, section, edition } = source.metadata;
  const canOpen = Boolean(source.metadata.doc_id);

  return (
    <li>
      <button
        type="button"
        className={active ? "source-card source-card-active" : "source-card"}
        onClick={() => onOpen(source)}
        disabled={!canOpen}
        title={canOpen ? "Открыть фрагмент в документе" : undefined}
      >
        <span className="source-card-head">
          <span className="source-n">{n}</span>
          <span className="source-title">
            {doc_title ?? "Фрагмент документа"}
            {section && <span className="source-section"> · {section}</span>}
          </span>
        </span>
        <span className="source-meta">
          {doc_type && <span>{DOC_TYPE_LABEL[doc_type]}</span>}
          {edition && <EditionBadge edition={edition} title={EDITION_STATUS_LABEL[edition.status]} />}
        </span>
        <span className="source-quote">{source.text}</span>
      </button>
    </li>
  );
}
