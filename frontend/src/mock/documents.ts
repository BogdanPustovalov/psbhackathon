// Knowledge-base documents for the mock API. Texts are copied from the case
// knowledge base (knowledge-base.zip) without changes.

import type { DocumentDetail } from "../api";
import a08 from "./kb/a-08-2024.txt?raw";
import p17 from "./kb/p-17-2024.txt?raw";
import dop1 from "./kb/dop-1-p-17-2024.txt?raw";
import pt05 from "./kb/pt-05-2023.txt?raw";
import reg2023 from "./kb/reglament-2023.txt?raw";
import reg2025 from "./kb/reglament-2025.txt?raw";
import fz152 from "./kb/152-fz-st9.txt?raw";

const normalize = (text: string) => text.replace(/\r\n/g, "\n");

export const MOCK_DOCUMENTS: DocumentDetail[] = [
  {
    doc_id: "a-08-2024",
    doc_title: "Договор аренды № А-08/2024 с ООО «Северный Склад»",
    doc_type: "contract",
    edition: { label: "от 01.08.2024", valid_from: "2024-08-01", valid_to: null, status: "active" },
    related: [],
    text: normalize(a08),
  },
  {
    doc_id: "p-17-2024",
    doc_title: "Договор поставки № П-17/2024 с ООО «Альфа-Трейд»",
    doc_type: "contract",
    edition: { label: "от 12.02.2024", valid_from: "2024-02-12", valid_to: null, status: "amended" },
    related: ["dop-1-p-17-2024"],
    text: normalize(p17),
  },
  {
    doc_id: "dop-1-p-17-2024",
    doc_title: "Дополнительное соглашение № 1 к договору поставки № П-17/2024",
    doc_type: "contract",
    edition: { label: "от 03.09.2024", valid_from: "2024-09-03", valid_to: null, status: "active" },
    related: ["p-17-2024"],
    text: normalize(dop1),
  },
  {
    doc_id: "pt-05-2023",
    doc_title: "Договор поставки № ПТ-05/2023 с ООО «Альфа-Трейдинг»",
    doc_type: "contract",
    edition: { label: "от 20.06.2023", valid_from: "2023-06-20", valid_to: null, status: "active" },
    related: [],
    text: normalize(pt05),
  },
  {
    doc_id: "reglament-2023",
    doc_title: "Регламент согласования договоров ООО «Меридиан»",
    doc_type: "internal",
    edition: {
      label: "ред. от 10.01.2023",
      valid_from: "2023-01-10",
      valid_to: "2025-02-28",
      status: "expired",
    },
    related: ["reglament-2025"],
    text: normalize(reg2023),
  },
  {
    doc_id: "reglament-2025",
    doc_title: "Регламент согласования договоров ООО «Меридиан»",
    doc_type: "internal",
    edition: { label: "ред. от 01.03.2025", valid_from: "2025-03-01", valid_to: null, status: "active" },
    related: ["reglament-2023"],
    text: normalize(reg2025),
  },
  {
    doc_id: "152-fz-st9",
    doc_title: "Федеральный закон № 152-ФЗ «О персональных данных», ст. 9",
    doc_type: "law",
    edition: { label: "действующая редакция", valid_from: null, valid_to: null, status: "active" },
    related: [],
    text: normalize(fz152),
  },
];

export function findDocument(docId: string): DocumentDetail | undefined {
  return MOCK_DOCUMENTS.find((doc) => doc.doc_id === docId);
}
