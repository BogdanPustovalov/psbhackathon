import type { DocType, Edition } from "./api";

export const DOC_TYPE_LABEL: Record<DocType, string> = {
  contract: "Договор",
  law: "Закон",
  internal: "Внутренний документ",
};

export const EDITION_STATUS_LABEL: Record<Edition["status"], string> = {
  active: "действует",
  amended: "изменён доп. соглашением",
  expired: "утратила силу",
};
