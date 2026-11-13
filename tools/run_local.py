"""Runs solution.py the way the platform does and checks the result.

- copies only the root .py files and requirements.txt into a clean work dir
  (nested folders are not available on the platform either);
- puts questions.csv and knowledge-base.zip into DATA_DIR;
- runs `python solution.py` with the 20-minute limit and platform env vars;
- validates submission.csv and saves a readable report next to it.

With --mock the run uses tools/mock_api.py instead of the real API: no key
needed, checks format, timing, retries (--fail-rate) and API call counts.

Usage:
    python tools/run_local.py --mock                       # demo questions, mock API
    python tools/run_local.py --mock --fail-rate 0.2       # with injected 429/5xx
    LLM_BASE_URL=... LLM_API_KEY=... python tools/run_local.py --name v1
    python tools/run_local.py --solution path/to/starter/solution.py --name baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))

from check_submission import check, print_report  # noqa: E402

TIME_LIMIT_S = 20 * 60
DEFAULT_ENV = {
    # Values the platform passes; the real ones may differ.
    "LLM_MODEL": "deepseek/deepseek-v4.1-flash",
    "EMB_MODEL": "baai/bge-m3",
    "TEMPERATURE": "0",
    "MAX_TOKENS": "1024",
    "MAX_WORKERS": "4",
}


def find_context() -> Path | None:
    """context/ lives only in the main checkout (it is not committed)."""
    candidates = [REPO / "context"]
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.strip()
        candidates.append(Path(common).parent / "context")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return next((c for c in candidates if c.is_dir()), None)


def build_kb_zip(source: Path, target: Path) -> int:
    """Packs a folder of KB files into knowledge-base.zip.

    Files unpacked as "folder__name.txt" go back into "folder/name.txt",
    like in the original archive.
    """
    if source.is_file() and source.suffix == ".zip":
        shutil.copy(source, target)
        with zipfile.ZipFile(target) as archive:
            return len([n for n in archive.namelist() if not n.endswith("/")])
    files = sorted(p for p in source.iterdir() if p.is_file() and not p.name.startswith("."))
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name.replace("__", "/", 1))
    return len(files)


def stage_solution(solution: Path, work: Path) -> list[str]:
    """Copies what the platform takes: root .py files and requirements.txt."""
    root = solution.parent
    copied = []
    for path in sorted(root.glob("*.py")):
        shutil.copy(path, work / path.name)
        copied.append(path.name)
    if (root / "requirements.txt").is_file():
        shutil.copy(root / "requirements.txt", work / "requirements.txt")
        copied.append("requirements.txt")
    if solution.name != "solution.py":
        shutil.copy(solution, work / "solution.py")
    return copied


def write_report(questions: Path, submission: Path, out: Path) -> None:
    with questions.open(encoding="utf-8-sig", newline="") as f:
        qs = {row["id"]: row["question"] for row in csv.DictReader(f)}
    with submission.open(encoding="utf-8-sig", newline="") as f:
        answers = {row["id"]: row["answer"] for row in csv.DictReader(f)}
    lines = ["# Ответы прогона", ""]
    for qid, question in qs.items():
        answer = answers.get(qid, "— нет ответа —")
        lines += [f"## {qid}. {question}", "", answer, "", f"_слов: {len(answer.split())}_", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    context = find_context()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solution", type=Path, default=REPO / "solution.py")
    parser.add_argument("--questions", type=Path,
                        default=context / "case" / "demo_questions.csv" if context else None)
    parser.add_argument("--kb", type=Path, help="папка с .txt базы или knowledge-base.zip",
                        default=context / "kb" if context else None)
    parser.add_argument("--name", default=time.strftime("run-%Y%m%d-%H%M%S"), help="имя прогона")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--mock", action="store_true", help="использовать tools/mock_api.py вместо API")
    parser.add_argument("--fail-rate", type=float, default=0.0, help="доля сбоев 429/5xx в моке")
    parser.add_argument("--latency-ms", type=int, default=0, help="задержка мока на запрос")
    parser.add_argument("--timeout", type=int, default=TIME_LIMIT_S, help="лимит времени, с")
    args = parser.parse_args()

    if not args.solution.is_file():
        sys.exit(f"Нет файла {args.solution}. Укажите --solution.")
    if not args.questions or not args.questions.is_file():
        sys.exit("Не найден questions.csv. Укажите --questions.")
    if not args.kb or not args.kb.exists():
        sys.exit("Не найдена база знаний. Укажите --kb.")

    run_dir = args.runs_dir / args.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    work = run_dir / "work"
    data = run_dir / "data"
    work.mkdir(parents=True)
    data.mkdir()

    copied = stage_solution(args.solution.resolve(), work)
    shutil.copy(args.questions, data / "questions.csv")
    documents = build_kb_zip(args.kb, data / "knowledge-base.zip")
    print(f"Файлы решения: {', '.join(copied)}")
    print(f"Документов в базе: {documents}")

    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLM_", "OPENAI_", "EMB", "RERANK"))}
    env.update(DEFAULT_ENV)
    # Real API settings come from the user's environment; never printed.
    for name in ("LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_MODEL",
                 "EMB_MODEL", "RERANKER_MODEL", "RERANKER_TOP_N", "TOP_K", "CHUNKER"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    env["DATA_DIR"] = str(data)
    env["PYTHONPATH"] = str(work)

    mock_stats = None
    if args.mock:
        from mock_api import serve

        server, mock_stats = serve(0, "test-key", args.fail_rate, args.latency_ms)
        env["LLM_BASE_URL"] = f"http://127.0.0.1:{server.server_port}"
        env["LLM_API_KEY"] = "test-key"
        print(f"Мок API: {env['LLM_BASE_URL']} (сбоев: {args.fail_rate:.0%})")
    elif not (env.get("LLM_BASE_URL") or env.get("OPENAI_BASE_URL")):
        sys.exit("Нет LLM_BASE_URL. Задайте переменные окружения или запустите с --mock.")

    print(f"Запуск solution.py, лимит {args.timeout} с…", flush=True)
    started = time.time()
    log_path = run_dir / "solution.log"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run([sys.executable, "solution.py"], cwd=work, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            code: int | str = result.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    elapsed = time.time() - started

    print(f"Код завершения: {code}; время: {elapsed:.0f} с ({elapsed / 60:.1f} мин); лог: {log_path}")
    if code == "timeout":
        print("ОШИБКА: превышен лимит времени — на платформе это «Превышено время обработки».")
    log_tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
    if log_tail:
        print("Конец лога:\n  " + "\n  ".join(log_tail))

    submission = work / "submission.csv"
    report = check(submission, data / "questions.csv", mock=args.mock)
    print_report(report)

    summary = {"name": args.name, "exit_code": code, "elapsed_s": round(elapsed, 1),
               "ok": report.ok and code == 0, "errors": report.errors, "stats": report.stats}
    if mock_stats:
        summary["mock_api"] = mock_stats.as_dict()
        calls = summary["mock_api"]["calls"]
        print(f"Вызовы API: {calls}; внесённых сбоев: {summary['mock_api']['errors_injected']}")
    if submission.is_file():
        shutil.copy(submission, run_dir / "submission.csv")
        write_report(data / "questions.csv", submission, run_dir / "answers.md")
        print(f"Ответы для чтения: {run_dir / 'answers.md'}")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
