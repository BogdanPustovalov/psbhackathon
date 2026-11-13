"""Checks submission.csv against the platform rules before a real submit.

Errors make the platform reject the run; warnings cost points.

Usage:
    python tools/check_submission.py submission.csv --questions questions.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

MAX_CHARS = 10_000
RECOMMENDED_WORDS = 150
# Answers that mean the pipeline failed; the judge scores them as losses.
FAILURE_MARKERS = ("не удалось получить ответ", "[mock]", "traceback", "error:", "exception")
CITATION = re.compile(r"\b(п\.|пп\.|пункт|ст\.|стать|част|ч\.\s*\d|абз)", re.IGNORECASE)


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def read_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [(row["id"].strip(), row["question"]) for row in csv.DictReader(f)]


def all_fields_quoted(raw: str) -> bool:
    """True when every field in every record is wrapped in double quotes."""
    i, n = 0, len(raw)
    while i < n:
        if raw[i] != '"':
            return False
        i += 1
        while True:  # skip the quoted value, "" is an escaped quote
            j = raw.find('"', i)
            if j < 0:
                return False
            if raw[j + 1 : j + 2] == '"':
                i = j + 2
                continue
            i = j + 1
            break
        if i >= n:
            return True
        if raw[i] == ",":
            i += 1
        elif raw.startswith("\r\n", i):
            i += 2
        elif raw[i] == "\n":
            i += 1
        else:
            return False
    return True


def check(submission: Path, questions: Path | None = None, mock: bool = False) -> Report:
    report = Report()
    if not submission.is_file():
        report.errors.append(f"Файл не найден: {submission}")
        return report

    data = submission.read_bytes()
    try:
        raw = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        report.errors.append(f"Файл не в UTF-8: {exc}")
        return report
    if raw.startswith("﻿"):
        report.warnings.append("Файл начинается с BOM (UTF-8-sig); безопаснее писать encoding='utf-8'.")
        raw = raw[1:]

    rows = list(csv.reader(io.StringIO(raw)))
    if not rows:
        report.errors.append("Файл пустой.")
        return report
    if rows[0] != ["id", "answer"]:
        report.errors.append(f"Заголовок должен быть ровно id,answer, а сейчас: {rows[0]}")
    body = rows[1:]

    if not all_fields_quoted(raw):
        report.warnings.append("Не все поля в кавычках: используйте csv.QUOTE_ALL, как в стартере.")

    seen: set[str] = set()
    answers: dict[str, str] = {}
    for number, row in enumerate(body, start=2):
        if len(row) != 2:
            report.errors.append(f"Строка {number}: ожидалось 2 колонки, получено {len(row)}.")
            continue
        qid, answer = row[0].strip(), row[1]
        if qid in seen:
            report.errors.append(f"Повторяющийся id {qid}.")
        seen.add(qid)
        answers[qid] = answer
        if not answer.strip():
            report.errors.append(f"id {qid}: пустой ответ.")
        if len(answer) > MAX_CHARS:
            report.errors.append(f"id {qid}: {len(answer)} символов, максимум {MAX_CHARS}.")

    if questions:
        expected = [qid for qid, _ in read_questions(questions)]
        missing = [qid for qid in expected if qid not in answers]
        extra = [qid for qid in answers if qid not in set(expected)]
        if missing:
            report.errors.append(f"Нет ответов для id: {', '.join(missing[:20])}{' …' if len(missing) > 20 else ''}")
        if extra:
            report.errors.append(f"Лишние id, которых нет в вопросах: {', '.join(extra[:20])}")

    word_counts = []
    failures = long_answers = without_citation = 0
    for qid, answer in answers.items():
        words = len(answer.split())
        word_counts.append(words)
        lowered = answer.lower()
        markers = [m for m in FAILURE_MARKERS if not (mock and m == "[mock]")]
        if any(marker in lowered for marker in markers):
            failures += 1
            report.warnings.append(f"id {qid}: похоже на ответ-заглушку или ошибку: {answer[:80]!r}")
        if words > RECOMMENDED_WORDS:
            long_answers += 1
        if not CITATION.search(answer):
            without_citation += 1

    if long_answers:
        report.warnings.append(
            f"{long_answers} ответ(ов) длиннее {RECOMMENDED_WORDS} слов (рекомендация регламента)."
        )
    if answers:
        report.stats = {
            "answers": len(answers),
            "avg_words": round(sum(word_counts) / len(word_counts), 1),
            "max_words": max(word_counts),
            "failures": failures,
            "without_citation": without_citation,
        }
    return report


def print_report(report: Report) -> None:
    for error in report.errors:
        print(f"ОШИБКА: {error}")
    for warning in report.warnings:
        print(f"Внимание: {warning}")
    if report.stats:
        s = report.stats
        print(
            f"Ответов: {s['answers']}, слов в среднем {s['avg_words']}, максимум {s['max_words']}; "
            f"заглушек/ошибок: {s['failures']}; без ссылки на пункт/статью: {s['without_citation']}"
        )
    print("Формат в порядке." if report.ok else "Платформа отклонит этот файл.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--questions", type=Path, help="questions.csv, чтобы сверить набор id")
    args = parser.parse_args()
    report = check(args.submission, args.questions)
    print_report(report)
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
