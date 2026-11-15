import { useEffect, useState } from "react";
import type { Source } from "../api";

type Props = {
  answer: string;
  sources: Source[];
  elapsedMs?: number;
};

// Answer with its list of sources, ready to paste into a letter or a chat.
function answerAsText(answer: string, sources: Source[]): string {
  if (sources.length === 0) return answer;
  const lines = sources.map((source, index) => {
    const { doc_title, section, edition } = source.metadata;
    const name = [doc_title ?? "Фрагмент", section, edition?.label].filter(Boolean).join(", ");
    return `[${source.n ?? index + 1}] ${name}`;
  });
  return `${answer}\n\nИсточники:\n${lines.join("\n")}`;
}

async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Fall back below.
  }
  // The clipboard API needs https or localhost; on a plain-http address use the old way.
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const ok = document.execCommand("copy");
  textarea.remove();
  return ok;
}

function formatSeconds(ms: number): string {
  return `${(ms / 1000).toLocaleString("ru-RU", { maximumFractionDigits: 1, minimumFractionDigits: 1 })} с`;
}

export function MessageActions({ answer, sources, elapsedMs }: Props) {
  const [status, setStatus] = useState<"idle" | "copied" | "failed">("idle");

  useEffect(() => {
    if (status === "idle") return;
    const timer = setTimeout(() => setStatus("idle"), 2000);
    return () => clearTimeout(timer);
  }, [status]);

  async function handleCopy() {
    setStatus((await copyText(answerAsText(answer, sources))) ? "copied" : "failed");
  }

  return (
    <div className="message-actions">
      <button type="button" className="link-button" onClick={handleCopy}>
        {status === "copied" ? "Скопировано ✓" : status === "failed" ? "Не удалось скопировать" : "Копировать ответ"}
      </button>
      {elapsedMs !== undefined && (
        <span className="muted" title="Время от отправки вопроса до ответа">
          Ответ за {formatSeconds(elapsedMs)}
        </span>
      )}
    </div>
  );
}
