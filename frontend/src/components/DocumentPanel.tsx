import { useEffect, useRef, useState } from "react";
import { type DocType, type DocumentDetail, type DocumentInfo, getDocument, listDocuments } from "../api";
import { DOC_TYPE_LABEL } from "../labels";
import { EditionBadge } from "./EditionBadge";

export type Highlight = {
  text: string;
  start?: number;
  end?: number;
};

export type PanelView =
  | { kind: "list" }
  | { kind: "doc"; docId: string; highlight?: Highlight; caption?: string };

type Props = {
  view: PanelView;
  onOpenDocument: (docId: string) => void;
  onShowList: () => void;
  onClose: () => void;
};

// Documents do not change while the page is open: load each one once.
const documentCache = new Map<string, Promise<DocumentDetail>>();
let listCache: Promise<DocumentInfo[]> | null = null;

function loadDocument(docId: string): Promise<DocumentDetail> {
  let promise = documentCache.get(docId);
  if (!promise) {
    promise = getDocument(docId);
    documentCache.set(docId, promise);
    promise.catch(() => documentCache.delete(docId));
  }
  return promise;
}

function loadList(): Promise<DocumentInfo[]> {
  if (!listCache) {
    listCache = listDocuments();
    listCache.catch(() => (listCache = null));
  }
  return listCache;
}

// Position of the fragment in the document: the offsets from the API if they
// match the fragment text, otherwise a plain text search.
function locate(text: string, highlight?: Highlight): [number, number] | null {
  if (!highlight) return null;
  const { start, end } = highlight;
  if (start !== undefined && end !== undefined && text.slice(start, end) === highlight.text) {
    return [start, end];
  }
  const found = text.indexOf(highlight.text);
  return found >= 0 ? [found, found + highlight.text.length] : null;
}

export function DocumentPanel({ view, onOpenDocument, onShowList, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <aside className="doc-panel" aria-label="Просмотр документа">
      <div className="doc-panel-bar">
        {view.kind === "doc" ? (
          <button type="button" className="link-button" onClick={onShowList}>
            ← Все документы
          </button>
        ) : (
          <span className="doc-panel-heading">База знаний</span>
        )}
        <button ref={closeRef} type="button" className="icon-button" onClick={onClose} aria-label="Закрыть панель">
          ✕
        </button>
      </div>
      <div className="doc-panel-body">
        {view.kind === "list" ? (
          <DocumentList onOpen={onOpenDocument} />
        ) : (
          <DocumentView key={view.docId} view={view} onOpenDocument={onOpenDocument} />
        )}
      </div>
    </aside>
  );
}

function DocumentView({
  view,
  onOpenDocument,
}: {
  view: Extract<PanelView, { kind: "doc" }>;
  onOpenDocument: (docId: string) => void;
}) {
  const [doc, setDoc] = useState<DocumentDetail | null>(null);
  const [titles, setTitles] = useState<Map<string, string>>(new Map());
  const [error, setError] = useState("");
  const markRef = useRef<HTMLElement>(null);

  useEffect(() => {
    let alive = true;
    loadDocument(view.docId)
      .then((loaded) => alive && setDoc(loaded))
      .catch(() => alive && setError("Не удалось загрузить документ."));
    loadList()
      .then(
        (list) =>
          alive &&
          setTitles(
            new Map(
              // Editions of one document share a title: add the edition to tell them apart.
              list.map((item) => [item.doc_id, item.edition ? `${item.doc_title} (${item.edition.label})` : item.doc_title]),
            ),
          ),
      )
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [view.docId]);

  const range = doc ? locate(doc.text, view.highlight) : null;

  useEffect(() => {
    markRef.current?.scrollIntoView({ block: "center" });
  }, [doc, range?.[0], view.highlight]);

  if (error) return <p className="error-text">{error}</p>;
  if (!doc) return <p className="muted">Загружаю документ…</p>;

  return (
    <article className="doc-view">
      <p className="doc-type">{DOC_TYPE_LABEL[doc.doc_type]}</p>
      <h2 className="doc-title">{doc.doc_title}</h2>
      {doc.edition && (
        <p className="doc-edition">
          <EditionBadge edition={doc.edition} />
        </p>
      )}

      {doc.related.length > 0 && (
        <p className="doc-related">
          Связанные документы:{" "}
          {doc.related.map((id, index) => (
            <span key={id}>
              {index > 0 && ", "}
              <button type="button" className="link-button" onClick={() => onOpenDocument(id)}>
                {titles.get(id) ?? id}
              </button>
            </span>
          ))}
        </p>
      )}

      {view.highlight && (
        <p className={range ? "doc-caption" : "doc-caption doc-caption-warning"}>
          {range
            ? `Подсвечен фрагмент${view.caption ? ` ${view.caption}` : ""}, на который ссылается ответ.`
            : "Фрагмент из ответа не найден в тексте документа дословно."}
        </p>
      )}

      <div className="doc-text">
        {range ? (
          <>
            {doc.text.slice(0, range[0])}
            <mark ref={markRef}>{doc.text.slice(range[0], range[1])}</mark>
            {doc.text.slice(range[1])}
          </>
        ) : (
          doc.text
        )}
      </div>
    </article>
  );
}

const TYPE_ORDER: DocType[] = ["contract", "internal", "law"];
const GROUP_LABEL: Record<DocType, string> = {
  contract: "Договоры",
  internal: "Внутренние документы",
  law: "Законы",
};

function DocumentList({ onOpen }: { onOpen: (docId: string) => void }) {
  const [docs, setDocs] = useState<DocumentInfo[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    loadList()
      .then((list) => alive && setDocs(list))
      .catch(() => alive && setError("Не удалось загрузить список документов."));
    return () => {
      alive = false;
    };
  }, []);

  if (error) return <p className="error-text">{error}</p>;
  if (!docs) return <p className="muted">Загружаю список…</p>;
  if (docs.length === 0) return <p className="muted">В базе пока нет документов.</p>;

  return (
    <div className="doc-list">
      {TYPE_ORDER.map((type) => {
        const group = docs.filter((doc) => doc.doc_type === type);
        if (group.length === 0) return null;
        return (
          <section key={type}>
            <h3 className="doc-group">{GROUP_LABEL[type]}</h3>
            <ul>
              {group.map((doc) => (
                <li key={doc.doc_id}>
                  <button type="button" className="doc-list-item" onClick={() => onOpen(doc.doc_id)}>
                    <span>{doc.doc_title}</span>
                    {doc.edition && <EditionBadge edition={doc.edition} />}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
