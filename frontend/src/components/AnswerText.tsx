import type { ReactNode } from "react";
import type { Source } from "../api";

type Props = {
  text: string;
  sources: Source[];
  activeN: number | null;
  onCite: (source: Source) => void;
};

const CITATION = /\[(\d+)\]/g;

// Answer text where every [n] that matches a source becomes a button
// opening that source. Unknown numbers stay plain text.
export function AnswerText({ text, sources, activeN, onCite }: Props) {
  const byNumber = new Map(sources.map((source, index) => [source.n ?? index + 1, source]));
  const parts: ReactNode[] = [];
  let last = 0;

  for (const match of text.matchAll(CITATION)) {
    const n = Number(match[1]);
    const source = byNumber.get(n);
    if (!source) continue;

    const start = match.index ?? 0;
    if (start > last) parts.push(text.slice(last, start));
    const title = [source.metadata.doc_title, source.metadata.section].filter(Boolean).join(", ");
    parts.push(
      <button
        key={start}
        type="button"
        className={n === activeN ? "cite cite-active" : "cite"}
        title={title || `Источник ${n}`}
        aria-label={`Открыть источник ${n}${title ? `: ${title}` : ""}`}
        onClick={() => onCite(source)}
      >
        {n}
      </button>,
    );
    last = start + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));

  return <p className="answer-text">{parts}</p>;
}
