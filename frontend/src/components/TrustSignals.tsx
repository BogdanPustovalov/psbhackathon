// Signals that help the user judge how far to trust an answer:
// confidence label, what is missing in the knowledge base, and warnings
// about non-current document editions.

import type { QueryResponse, Source } from "../api";

function plural(n: number, one: string, few: string, many: string): string {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
}

// "2025-02-28" -> "28.02.2025"
function formatDate(iso: string): string {
  const [year, month, day] = iso.split("-");
  return year && month && day ? `${day}.${month}.${year}` : iso;
}

// Unique documents among the sources, in order of first mention.
function uniqueDocs(sources: Source[]): Source[] {
  const seen = new Set<string>();
  return sources.filter((source) => {
    const id = source.metadata.doc_id;
    if (!id || seen.has(id)) return false;
    seen.add(id);
    return true;
  });
}

const docName = (source: Source) =>
  [source.metadata.doc_title, source.metadata.edition?.label].filter(Boolean).join(", ");

export function ConfidenceBadge({ response }: { response: QueryResponse }) {
  const zone = response.confidence?.zone;
  const count = response.sources.length;

  let kind: string;
  let text: string;
  let hint: string;

  if (response.clarification) {
    kind = "neutral";
    text = "Нужно уточнение";
    hint = "Вопрос можно понять по-разному. Выберите вариант, чтобы получить точный ответ.";
  } else if (zone === "confident" && count > 0) {
    kind = "ok";
    text = `Подтверждено: ${count} ${plural(count, "источник", "источника", "источников")}`;
    hint = "Ассистент нашёл в базе фрагменты, прямо отвечающие на вопрос.";
  } else if (zone === "uncertain") {
    kind = "warn";
    text = "Ответ неполный — проверьте источники";
    hint = "Найденные фрагменты отвечают на вопрос лишь частично.";
  } else if (zone === "no_answer" || count === 0) {
    kind = "none";
    text = "Ответа нет в базе знаний";
    hint = "Ассистент не нашёл подходящих документов и не отвечает по памяти.";
  } else {
    return null;
  }

  return (
    <p className={`trust-badge trust-${kind}`} title={hint}>
      <span className="trust-dot" aria-hidden="true" />
      {text}
    </p>
  );
}

export function MissingBlock({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div className="notice notice-missing">
      <p className="notice-title">Чего нет в базе знаний</p>
      <ul>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
      <p className="notice-hint">Ассистент не додумывает недостающее. Чтобы ответить точнее, нужны эти документы.</p>
    </div>
  );
}

// Warns when the answer relies on an expired edition or on a document
// amended by another one that is not among the sources.
export function EditionNotice({ sources }: { sources: Source[] }) {
  const docs = uniqueDocs(sources);
  const citedIds = new Set(docs.map((source) => source.metadata.doc_id));

  const expired = docs.filter((source) => source.metadata.edition?.status === "expired");
  const amended = docs.filter((source) => source.metadata.edition?.status === "amended");

  // An amended document is fine when the amendment itself is also cited.
  const amendedWithoutChange = amended.filter((source) => {
    const related = source.metadata.related;
    return !(Array.isArray(related) && related.some((id) => citedIds.has(id)));
  });

  if (expired.length === 0 && amendedWithoutChange.length === 0) return null;

  return (
    <div className="notice notice-edition">
      <p className="notice-title">Редакции документов</p>
      <ul>
        {expired.map((source) => {
          const until = source.metadata.edition?.valid_to;
          return (
            <li key={source.id}>
              {docName(source)} — <strong>утратила силу</strong>
              {until ? `, применяется к событиям до ${formatDate(until)} включительно.` : "."}
            </li>
          );
        })}
        {amendedWithoutChange.map((source) => (
          <li key={source.id}>
            {docName(source)} — <strong>изменён дополнительным соглашением</strong>. Проверьте, не изменён ли
            процитированный пункт.
          </li>
        ))}
      </ul>
    </div>
  );
}
