"""Оценка реранкера на размеченном наборе eval_set.json.

Запуск:
    python eval_reranker.py                                   # Qwen3-Reranker-0.6B, юридическая инструкция
    python eval_reranker.py --model Qwen/Qwen3-Reranker-8B
    python eval_reranker.py --instruction default             # стандартная инструкция из карточки
    python eval_reranker.py --doc-format section             # без названия документа в тексте фрагмента
    python eval_reranker.py --limit 3                         # первые 3 вопроса — проверить, что всё запускается

Метрики:
    nDCG@5   — качество порядка с учётом градаций 2/1/0 (1.0 = идеальный порядок);
    MRR      — 1 / место первого полностью релевантного фрагмента (метка 2);
    Hit@1/3  — доля вопросов, где фрагмент с меткой 2 попал в топ-1 / топ-3;
    s_max    — log-odds лучшего кандидата: по нему решаем, отказывать ли («в базе нет»).
Все оценки сохраняются в results_*.json — их можно прислать для разбора.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from reranker import INSTRUCTION_DEFAULT, INSTRUCTION_LEGAL, Reranker


def dcg(labels: list[int], k: int) -> float:
    return sum((2**rel - 1) / math.log2(pos + 2) for pos, rel in enumerate(labels[:k]))


def ndcg(ranked_labels: list[int], k: int = 5) -> float | None:
    ideal = dcg(sorted(ranked_labels, reverse=True), k)
    return None if ideal == 0 else dcg(ranked_labels, k) / ideal


def first_hit_rank(ranked_labels: list[int], target: int = 2) -> int | None:
    return next((pos + 1 for pos, rel in enumerate(ranked_labels) if rel >= target), None)


def prepare_doc(text: str, doc_format: str) -> str:
    """Как подать фрагмент реранкеру: целиком, без названия документа или только текст пункта.

    Эксперимент на «доминирование шапки»: когда все кандидаты из одного договора,
    одинаковая шапка с номером и контрагентом поднимает оценки ВСЕХ его пунктов.
    """
    if doc_format == "full":
        return text
    header, _, body = text.partition("\n")
    if doc_format == "body":
        return body or text
    # "section": убрать название документа, оставить «[раздел] п. X» и текст.
    if " · " in header:
        return header.split(" · ", 1)[1] + "\n" + body
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="eval_set.json")
    parser.add_argument("--model", default=None, help="по умолчанию Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--backend", default="local", choices=["local", "api"])
    parser.add_argument("--instruction", default="legal",
                        help="legal | default | любой свой текст на английском")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--doc-format", default="full", choices=["full", "section", "body"],
                        help="full — с шапкой документа; section — без названия документа; body — только текст")
    parser.add_argument("--limit", type=int, default=None, help="взять только первые N вопросов")
    parser.add_argument("--out", default=None, help="куда сохранить оценки (JSON)")
    args = parser.parse_args()

    instruction = {"legal": INSTRUCTION_LEGAL, "default": INSTRUCTION_DEFAULT}.get(
        args.instruction, args.instruction
    )
    questions = json.loads(Path(args.data).read_text(encoding="utf-8"))[: args.limit]
    reranker = Reranker(model_name=args.model, backend=args.backend,
                        instruction=instruction, batch_size=args.batch_size, debug=True)
    print(reranker)
    print(f"Инструкция: {instruction}")
    print(f"Формат фрагментов: {args.doc_format}\n")

    rows, saved = [], []
    started_all = time.perf_counter()
    for q in questions:
        texts = [c["text"] for c in q["candidates"]]
        inputs = [prepare_doc(t, args.doc_format) for t in texts]
        labels = [c["label"] for c in q["candidates"]]

        started = time.perf_counter()
        logodds = reranker.score(q["query"], inputs, as_logodds=True)
        seconds = time.perf_counter() - started

        order = sorted(range(len(texts)), key=lambda i: logodds[i], reverse=True)
        ranked_labels = [labels[i] for i in order]
        rank = first_hit_rank(ranked_labels)
        answerable = any(labels)
        rows.append({
            "id": q["id"], "type": q["type"], "answerable": answerable,
            "ndcg5": ndcg(ranked_labels), "ndcg5_input_order": ndcg(labels),
            "rr": (1 / rank) if rank else (0.0 if answerable else None),
            "hit1": rank == 1 if answerable else None,
            "hit3": (rank is not None and rank <= 3) if answerable else None,
            "s_max": logodds[order[0]],
            "s_gold": max((s for s, l in zip(logodds, labels) if l == 2), default=None),
            "n": len(texts), "seconds": seconds,
            "top1": texts[order[0]].split("\n")[0][:70],
        })
        saved.append({"id": q["id"], "query": q["query"],
                      "scores": [{"label": l, "logodds": round(s, 3), "text": t[:120]}
                                 for s, l, t in sorted(zip(logodds, labels, texts), reverse=True)]})
        r = rows[-1]
        ndcg_text = f"{r['ndcg5']:.2f}" if r["ndcg5"] is not None else "  — "
        print(f"{q['id']:4} {q['type'][:18]:18} nDCG@5={ndcg_text} место_ответа={rank or '—':>2} "
              f"s_max={r['s_max']:+6.2f}  {r['n']:2} канд. {seconds:5.1f} с | топ-1: {r['top1']}")

    total = time.perf_counter() - started_all
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    mean = lambda xs: statistics.mean(xs) if xs else float("nan")  # noqa: E731

    print("\n================ ИТОГ ================")
    print(f"Вопросов: {len(rows)} (с ответом {len(ans)}, без ответа {len(unans)})")
    print(f"nDCG@5:            {mean([r['ndcg5'] for r in ans]):.3f}"
          f"   (для сравнения, порядок без реранка: {mean([r['ndcg5_input_order'] for r in ans]):.3f})")
    print(f"MRR:               {mean([r['rr'] for r in ans]):.3f}")
    print(f"Hit@1 / Hit@3:     {mean([float(r['hit1']) for r in ans]):.2f} / {mean([float(r['hit3']) for r in ans]):.2f}")
    pairs = sum(r["n"] for r in rows)
    print(f"Время:             {total:.1f} с всего, {mean([r['seconds'] for r in rows]):.2f} с на вопрос, "
          f"{pairs / total:.1f} пар/с")

    # Разделимость «есть ответ / нет ответа» по s_max — основа порога отказа.
    if ans and unans:
        low_ans = min(r["s_max"] for r in ans)
        high_unans = max(r["s_max"] for r in unans)
        print(f"\ns_max у вопросов С ответом:  мин {low_ans:+.2f}, медиана {statistics.median(r['s_max'] for r in ans):+.2f}")
        print(f"s_max у вопросов БЕЗ ответа: макс {high_unans:+.2f}")
        if high_unans < low_ans:
            print(f"→ Разделимы: порог отказа можно искать между {high_unans:+.2f} и {low_ans:+.2f}")
        else:
            print("→ Пересекаются: одного порога мало, нужна проверка достаточности на стороне LLM")

    out = Path(args.out or f"results_{reranker.model_name.split('/')[-1]}_{args.instruction[:10]}_{args.doc_format}.json")
    out.write_text(json.dumps({"model": reranker.model_name, "instruction": instruction, "doc_format": args.doc_format,
                               "summary": rows, "scores": saved}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\nОценки сохранены в {out}")


if __name__ == "__main__":
    main()
