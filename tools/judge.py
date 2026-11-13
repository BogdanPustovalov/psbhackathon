"""Local estimate of the platform score: LLM judge + quick offline checks.

The platform judge compares each team answer with the baseline answer, using the
golden answer as reference, in two rounds with swapped order; 1 / .75 / .5 /
.25 / 0 per round, a fabricated fact zeroes the question.
Score = 0.7 * MatchRate + 0.3 * WinRate (technical regulations, section 8).

Two modes:
  --offline   no LLM: recall of numbers and clause references from the golden
              answer, refusal agreement, numbers absent from the knowledge base
              (likely fabrications), length. Fast, free, rough.
  (default)   LLM judge through the same OpenAI-compatible API (env LLM_BASE_URL,
              LLM_API_KEY, JUDGE_MODEL or LLM_MODEL). Closer to the platform,
              but it is still our judge, not theirs.

Usage:
    python tools/judge.py --team runs/v1/submission.csv --baseline runs/baseline/submission.csv --offline
    python tools/judge.py --team runs/v1/submission.csv --baseline runs/baseline/submission.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from run_local import find_context  # noqa: E402

# ---------------------------------------------------------------- data


def read_csv(path: Path, value: str) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["id"].strip(): row[value] for row in csv.DictReader(f)}


def read_kb(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    if path.is_dir():
        return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in sorted(path.glob("*")) if p.is_file())
    import zipfile

    with zipfile.ZipFile(path) as archive:
        return "\n".join(archive.read(n).decode("utf-8", errors="ignore")
                         for n in archive.namelist() if not n.endswith("/"))


# ---------------------------------------------------------------- offline checks

CLAUSE = re.compile(r"(?<![\d.])\d{1,2}(?:\.\d{1,2})+(?![\d.]*\d)")
NUMBER = re.compile(r"(?<![\d.,])\d{1,3}(?:[  ]\d{3})+(?![\d])|(?<![\d.,])\d+(?:[.,]\d+)?(?![\d])")
REFUSAL = re.compile(
    r"(в (предоставленной )?базе( знаний)? (нет|отсутству)|не (содержит|найден)|нет (документ|данных|сведений)"
    r"|не могу (ответить|сказать)|недостаточно (данных|информации))",
    re.IGNORECASE,
)


def normalize_number(raw: str) -> str:
    return raw.replace(" ", "").replace(" ", "").replace(",", ".")


def clauses(text: str) -> set[str]:
    return {c.rstrip(".") for c in CLAUSE.findall(text)}


def numbers(text: str) -> set[str]:
    without_clauses = CLAUSE.sub(" ", text)
    found = {normalize_number(n) for n in NUMBER.findall(without_clauses)}
    return {n for n in found if n not in {"0", "1", "2"}}  # too common to mean anything


@dataclass
class Offline:
    qid: str
    number_recall: float | None
    clause_recall: float | None
    refusal_expected: bool
    refusal_given: bool
    unknown_numbers: list[str]
    words: int

    @property
    def proxy(self) -> float:
        """Rough 0..1 quality proxy; unknown numbers are treated as fabrications."""
        if self.unknown_numbers:
            return 0.0
        if self.refusal_expected != self.refusal_given:
            return 0.2 if self.refusal_expected else 0.3
        parts = [x for x in (self.number_recall, self.clause_recall) if x is not None]
        base = sum(parts) / len(parts) if parts else (1.0 if self.refusal_given else 0.6)
        return round(base * (0.9 if self.words > 150 else 1.0), 3)


def offline_check(qid: str, question: str, golden: str, answer: str, kb_numbers: set[str]) -> Offline:
    g_numbers, g_clauses = numbers(golden), clauses(golden)
    a_numbers, a_clauses = numbers(answer), clauses(answer)
    known = kb_numbers | numbers(question) | g_numbers
    return Offline(
        qid=qid,
        number_recall=round(len(g_numbers & a_numbers) / len(g_numbers), 3) if g_numbers else None,
        clause_recall=round(len(g_clauses & a_clauses) / len(g_clauses), 3) if g_clauses else None,
        refusal_expected=bool(REFUSAL.search(golden)) and not g_clauses,
        refusal_given=bool(REFUSAL.search(answer)) and not a_clauses,
        unknown_numbers=sorted(a_numbers - known),
        words=len(answer.split()),
    )


# ---------------------------------------------------------------- LLM judge

JUDGE_PROMPT = """Ты — строгий эксперт-юрист и судья качества ответов юридического ассистента.
Ассистент обязан отвечать только по базе знаний компании (договоры, регламенты, выдержки из законов).

Сравни два ответа (A и B) на вопрос, ориентируясь на эталонный ответ.
Критерии по важности:
1. Фактическая точность относительно эталона: суммы, сроки, пункты, стороны, редакции документов.
2. Выдуманные факты: утверждения, которых нет в эталоне и которые не следуют из него (особенно числа, сроки, номера пунктов, нормы права). Выдуманный факт — тяжёлая ошибка.
3. Полнота: есть ли все ключевые элементы эталона, указана ли ошибка в вопросе, сказано ли, чего не хватает в базе.
4. Ссылки на конкретные положения (пункт, статья, документ).
5. Ясность и краткость (до ~150 слов).

Вопрос:
<<<{question}>>>

Эталонный ответ:
<<<{golden}>>>

Ответ A:
<<<{a}>>>

Ответ B:
<<<{b}>>>

Верни только JSON без пояснений вокруг:
{{"verdict": "A_much_better" | "A_better" | "equal" | "B_better" | "B_much_better",
  "fabricated_A": true | false,
  "fabricated_B": true | false,
  "reason": "одно-два предложения по-русски"}}"""

# Score of answer B relative to answer A, as in the regulations.
B_SCORE = {"A_much_better": 0.0, "A_better": 0.25, "equal": 0.5, "B_better": 0.75, "B_much_better": 1.0}


@dataclass
class Judged:
    qid: str
    round1: float | None = None  # A = baseline, B = team
    round2: float | None = None  # A = team, B = baseline
    fabricated: bool = False
    reasons: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def score(self) -> float | None:
        if self.error or self.round1 is None or self.round2 is None:
            return None
        return 0.0 if self.fabricated else (self.round1 + self.round2) / 2


def call_judge(client, url: str, key: str, model: str, prompt: str) -> dict:
    import httpx

    for attempt in range(4):
        try:
            response = client.post(
                url, headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "temperature": 0, "max_tokens": 400,
                      "messages": [{"role": "user", "content": prompt}]},
            )
            if response.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retryable", request=response.request, response=response)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", content, re.DOTALL)
            data = json.loads(match.group(0) if match else content)
            if data.get("verdict") not in B_SCORE:
                raise ValueError(f"unexpected verdict {data.get('verdict')!r}")
            return data
        except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError) as exc:
            if attempt == 3:
                raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def judge_question(client, url, key, model, qid, question, golden, team, baseline) -> Judged:
    result = Judged(qid)
    try:
        r1 = call_judge(client, url, key, model,
                        JUDGE_PROMPT.format(question=question, golden=golden, a=baseline, b=team))
        r2 = call_judge(client, url, key, model,
                        JUDGE_PROMPT.format(question=question, golden=golden, a=team, b=baseline))
    except RuntimeError as exc:
        result.error = str(exc)
        return result
    result.round1 = B_SCORE[r1["verdict"]]
    result.round2 = 1.0 - B_SCORE[r2["verdict"]]
    result.fabricated = bool(r1.get("fabricated_B")) or bool(r2.get("fabricated_A"))
    result.reasons = [str(r1.get("reason", "")), str(r2.get("reason", ""))]
    return result


def platform_score(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"match_rate": 0.0, "win_rate": 0.5, "score": 0.35, "wins": 0, "losses": 0, "ties": 0}
    wins = sum(s > 0.5 for s in scores)
    losses = sum(s < 0.5 for s in scores)
    match_rate = sum(scores) / len(scores)
    win_rate = wins / (wins + losses) if wins + losses else 0.5
    return {"match_rate": round(match_rate, 3), "win_rate": round(win_rate, 3),
            "score": round(0.7 * match_rate + 0.3 * win_rate, 3),
            "wins": wins, "losses": losses, "ties": len(scores) - wins - losses}


# ---------------------------------------------------------------- main


def main() -> None:
    context = find_context()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", type=Path, required=True, help="submission.csv команды")
    parser.add_argument("--baseline", type=Path, help="submission.csv базового решения (стартера)")
    parser.add_argument("--questions", type=Path,
                        default=context / "case" / "demo_questions.csv" if context else None)
    parser.add_argument("--golden", type=Path, default=context / "case" / "demo_output.csv" if context else None)
    parser.add_argument("--kb", type=Path, default=context / "kb" if context else None)
    parser.add_argument("--offline", action="store_true", help="только быстрые проверки без LLM")
    parser.add_argument("--out", type=Path, help="куда сохранить отчёт (по умолчанию рядом с --team)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    for name in ("questions", "golden"):
        if not getattr(args, name) or not getattr(args, name).is_file():
            sys.exit(f"Не найден файл --{name}.")
    questions = read_csv(args.questions, "question")
    golden = read_csv(args.golden, "answer")
    team = read_csv(args.team, "answer")
    baseline = read_csv(args.baseline, "answer") if args.baseline else {}
    ids = [qid for qid in questions if qid in golden]
    out = args.out or args.team.parent / ("judge_offline.md" if args.offline else "judge.md")

    kb_numbers = numbers(read_kb(args.kb))
    offline = {qid: offline_check(qid, questions[qid], golden[qid], team.get(qid, ""), kb_numbers) for qid in ids}
    offline_base = ({qid: offline_check(qid, questions[qid], golden[qid], baseline.get(qid, ""), kb_numbers)
                     for qid in ids} if baseline else {})

    lines = [f"# Оценка: {args.team}", ""]
    if args.offline:
        lines += ["Режим без LLM: приблизительная оценка по числам, пунктам и отказам.", "",
                  "| id | числа из эталона | пункты из эталона | отказ ожидался/дан | чисел нет в базе | слов | прокси"
                  + (" | прокси baseline |" if baseline else " |"),
                  "|---|---|---|---|---|---|---" + ("|---|" if baseline else "|")]
        for qid in ids:
            o = offline[qid]
            fmt = lambda x: "—" if x is None else f"{x:.0%}"  # noqa: E731
            row = (f"| {qid} | {fmt(o.number_recall)} | {fmt(o.clause_recall)} | "
                   f"{'да' if o.refusal_expected else 'нет'}/{'да' if o.refusal_given else 'нет'} | "
                   f"{', '.join(o.unknown_numbers) or '—'} | {o.words} | {o.proxy:.2f}")
            row += f" | {offline_base[qid].proxy:.2f} |" if baseline else " |"
            lines.append(row)
        team_proxy = sum(offline[q].proxy for q in ids) / len(ids)
        lines += ["", f"Средний прокси команды: **{team_proxy:.3f}**"]
        if baseline:
            base_proxy = sum(offline_base[q].proxy for q in ids) / len(ids)
            pseudo = [min(1.0, max(0.0, 0.5 + (offline[q].proxy - offline_base[q].proxy))) for q in ids]
            lines += [f"Средний прокси baseline: **{base_proxy:.3f}**",
                      f"Условный Score против baseline: **{platform_score(pseudo)['score']:.3f}** "
                      "(очень грубо: только для сравнения версий между собой)"]
        print(f"Средний прокси команды: {team_proxy:.3f}")
    else:
        if not baseline:
            sys.exit("Для LLM-судьи нужен --baseline (судья сравнивает с базовым решением).")
        base_url = (os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
        key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        model = os.environ.get("JUDGE_MODEL") or os.environ.get("LLM_MODEL") or "deepseek/deepseek-v4.1-flash"
        if not base_url or not key:
            sys.exit("Для LLM-судьи нужны LLM_BASE_URL и LLM_API_KEY. Без них запустите с --offline.")
        import httpx

        with httpx.Client(timeout=90.0) as client, ThreadPoolExecutor(args.workers) as pool:
            judged = list(pool.map(
                lambda qid: judge_question(client, base_url + "/chat/completions", key, model, qid,
                                           questions[qid], golden[qid], team.get(qid, ""), baseline.get(qid, "")),
                ids))
        scores = [j.score for j in judged if j.score is not None]
        summary = platform_score(scores)
        lines += [f"Судья: `{model}`. Оценено вопросов: {len(scores)} из {len(ids)}.", "",
                  f"**Score ≈ {summary['score']}** (MatchRate {summary['match_rate']}, WinRate {summary['win_rate']}; "
                  f"побед {summary['wins']}, ничьих {summary['ties']}, поражений {summary['losses']})", "",
                  "| id | раунд 1 | раунд 2 | выдумка | итог | чисел нет в базе | комментарий |",
                  "|---|---|---|---|---|---|---|"]
        for j in judged:
            o = offline[j.qid]
            if j.error:
                lines.append(f"| {j.qid} | — | — | — | ошибка | — | {j.error[:80]} |")
                continue
            lines.append(f"| {j.qid} | {j.round1} | {j.round2} | {'да' if j.fabricated else ''} | {j.score:.2f} | "
                         f"{', '.join(o.unknown_numbers) or '—'} | {j.reasons[0][:160]} |")
        print(f"Score ≈ {summary['score']}  MatchRate {summary['match_rate']}  WinRate {summary['win_rate']}")
        (out.with_suffix(".json")).write_text(
            json.dumps({"summary": summary, "questions": [asdict(j) for j in judged]}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Отчёт: {out}")


if __name__ == "__main__":
    main()
