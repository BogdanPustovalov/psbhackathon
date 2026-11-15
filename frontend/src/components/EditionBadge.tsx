import type { Edition } from "../api";
import { EDITION_STATUS_LABEL } from "../labels";

// Edition label; a non-current edition also says why.
export function EditionBadge({ edition, title }: { edition: Edition; title?: string }) {
  const note = edition.status === "active" ? "" : ` · ${EDITION_STATUS_LABEL[edition.status]}`;
  return (
    <span className={`edition edition-${edition.status}`} title={title}>
      {edition.label}
      {note}
    </span>
  );
}
