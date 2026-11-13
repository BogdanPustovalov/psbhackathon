import type { ReactNode } from "react";
import type { Source } from "../api";

type Props = {
  text: string;
  sources: Source[];
  activeN: number | null;
  onCite: (source: Source) => void;
};

type Block =
  | { kind: "p"; lines: string[] }
  | { kind: "ol"; start: number; items: string[] }
  | { kind: "ul"; items: string[] };

const ORDERED = /^\s*(\d+)[.)]\s+(.*)$/;
const UNORDERED = /^\s*[-•*]\s+(.*)$/;
const INLINE = /\*\*([^*]+)\*\*|\[(\d+)\]/g;

// Splits the answer into paragraphs and lists. The model is asked for plain
// text with simple Markdown: "1." / "-" lists and **bold**.
function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  for (const line of text.split("\n")) {
    const last = blocks[blocks.length - 1];
    const ordered = line.match(ORDERED);
    const unordered = line.match(UNORDERED);

    if (!line.trim()) {
      blocks.push({ kind: "p", lines: [] });
    } else if (ordered) {
      if (last?.kind === "ol") last.items.push(ordered[2]);
      else blocks.push({ kind: "ol", start: Number(ordered[1]), items: [ordered[2]] });
    } else if (unordered) {
      if (last?.kind === "ul") last.items.push(unordered[1]);
      else blocks.push({ kind: "ul", items: [unordered[1]] });
    } else if (last?.kind === "p") {
      last.lines.push(line);
    } else {
      blocks.push({ kind: "p", lines: [line] });
    }
  }
  return blocks.filter((block) => block.kind !== "p" || block.lines.length > 0);
}

// Answer text where every [n] that matches a source becomes a button opening
// that source. Unknown numbers stay plain text.
export function AnswerText({ text, sources, activeN, onCite }: Props) {
  const byNumber = new Map(sources.map((source, index) => [source.n ?? index + 1, source]));

  function inline(line: string, keyPrefix: string): ReactNode[] {
    const parts: ReactNode[] = [];
    let last = 0;
    for (const match of line.matchAll(INLINE)) {
      const start = match.index ?? 0;
      const key = `${keyPrefix}-${start}`;
      let node: ReactNode;

      if (match[1] !== undefined) {
        node = <strong key={key}>{match[1]}</strong>;
      } else {
        const n = Number(match[2]);
        const source = byNumber.get(n);
        if (!source) continue;
        const title = [source.metadata.doc_title, source.metadata.section].filter(Boolean).join(", ");
        node = (
          <button
            key={key}
            type="button"
            className={n === activeN ? "cite cite-active" : "cite"}
            title={title || `Источник ${n}`}
            aria-label={`Открыть источник ${n}${title ? `: ${title}` : ""}`}
            onClick={() => onCite(source)}
          >
            {n}
          </button>
        );
      }

      if (start > last) parts.push(line.slice(last, start));
      parts.push(node);
      last = start + match[0].length;
    }
    if (last < line.length) parts.push(line.slice(last));
    return parts;
  }

  return (
    <div className="answer-text">
      {parseBlocks(text).map((block, b) => {
        if (block.kind === "ol") {
          return (
            <ol key={b} start={block.start}>
              {block.items.map((item, i) => (
                <li key={i}>{inline(item, `${b}-${i}`)}</li>
              ))}
            </ol>
          );
        }
        if (block.kind === "ul") {
          return (
            <ul key={b}>
              {block.items.map((item, i) => (
                <li key={i}>{inline(item, `${b}-${i}`)}</li>
              ))}
            </ul>
          );
        }
        return (
          <p key={b}>
            {block.lines.map((line, i) => (
              <span key={i}>
                {i > 0 && <br />}
                {inline(line, `${b}-${i}`)}
              </span>
            ))}
          </p>
        );
      })}
    </div>
  );
}
