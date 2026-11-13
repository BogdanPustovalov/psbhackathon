"""Builds the platform ZIP and checks it against the submission rules.

Takes solution.py, the other root .py modules and requirements.txt from the
repository root (the platform ignores nested folders) and checks:
  - every import is the standard library, a root module, or a package listed
    in requirements.txt (no torch/transformers on the platform);
  - requirements.txt pins versions with ==;
  - no pip install / subprocess pip calls;
  - no secrets: key-like strings, or the value of LLM_API_KEY from your env;
  - the archive is under 128 MB.

Usage:
    python tools/build_zip.py                      # -> dist/submission.zip
    python tools/build_zip.py --exclude eval_reranker.py
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAX_BYTES = 128 * 1024 * 1024
FORBIDDEN = {"torch", "transformers", "sentence_transformers", "tensorflow", "jax"}
# pip name -> import name, where they differ.
IMPORT_NAMES = {
    "python-docx": "docx", "beautifulsoup4": "bs4", "pyyaml": "yaml", "scikit-learn": "sklearn",
    "faiss-cpu": "faiss", "rank-bm25": "rank_bm25", "python-dotenv": "dotenv",
    "python-multipart": "multipart", "langchain-openai": "langchain_openai",
    "langchain-text-splitters": "langchain_text_splitters", "pypdf": "pypdf",
}
SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._-]{20,}|api[_-]?key\s*=\s*['\"][^'\"]{12,})",
                    re.IGNORECASE)


def requirement_modules(path: Path) -> tuple[set[str], list[str]]:
    modules, problems = set(), []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name = re.split(r"[=<>!~\[ ;]", line, maxsplit=1)[0].strip()
        if "==" not in line:
            problems.append(f"requirements.txt: версия не зафиксирована: {line}")
        modules.add(IMPORT_NAMES.get(name.lower(), name.lower().replace("-", "_")))
    return modules, problems


def imports(path: Path) -> list[tuple[str, int, bool]]:
    """(top-level module, line, imported at module level)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    top_level = {id(node) for node in tree.body}
    # Imports guarded by try/except at module level count as optional.
    for node in tree.body:
        if isinstance(node, ast.Try):
            top_level.discard(id(node))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            found.append((name.split(".")[0], node.lineno, id(node) in top_level))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=REPO, help="папка с solution.py (по умолчанию корень репозитория)")
    parser.add_argument("--out", type=Path, default=REPO / "dist" / "submission.zip")
    parser.add_argument("--exclude", nargs="*", default=[], help="корневые .py, которые не нужно класть в архив")
    args = parser.parse_args()

    src = args.src.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not (src / "solution.py").is_file():
        sys.exit(f"Нет {src / 'solution.py'} — без него платформа не запустит решение.")
    requirements = src / "requirements.txt"
    if not requirements.is_file():
        sys.exit("Нет requirements.txt в корне.")

    py_files = sorted(p for p in src.glob("*.py") if p.name not in set(args.exclude))
    local_modules = {p.stem for p in py_files}
    allowed, problems = requirement_modules(requirements)
    errors += problems
    stdlib = set(sys.stdlib_module_names)
    secret_value = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

    for path in py_files:
        text = path.read_text(encoding="utf-8")
        for module, line, at_top in imports(path):
            where = f"{path.name}:{line}"
            if module in stdlib or module in local_modules or module == "__future__":
                continue
            if module in FORBIDDEN:
                (errors if at_top else warnings).append(
                    f"{where}: import {module} — на платформе его нет"
                    + ("" if at_top else " (импорт внутри функции: упадёт, только если этот код вызовется)"))
            elif module not in allowed:
                (errors if at_top else warnings).append(f"{where}: пакета {module} нет в requirements.txt")
        if re.search(r"pip(3)?\s+install|-m\s+pip|['\"]pip['\"]", text):
            errors.append(f"{path.name}: вызов pip install запрещён регламентом")
        for match in SECRET.finditer(text):
            errors.append(f"{path.name}: похоже на ключ или токен: {match.group(0)[:12]}…")
        if secret_value and len(secret_value) >= 8 and secret_value in text:
            errors.append(f"{path.name}: в файле записано значение LLM_API_KEY из вашего окружения")

    print("В архив попадут:", ", ".join([p.name for p in py_files] + ["requirements.txt"]))
    for warning in warnings:
        print(f"Внимание: {warning}")
    if errors:
        for error in errors:
            print(f"ОШИБКА: {error}")
        sys.exit("Архив не собран: исправьте ошибки выше.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in py_files:
            archive.write(path, path.name)  # flat: files at the archive root
        archive.write(requirements, "requirements.txt")
    size = args.out.stat().st_size
    if size > MAX_BYTES:
        sys.exit(f"ОШИБКА: архив {size / 1e6:.1f} МБ, лимит 128 МБ.")
    print(f"Готово: {args.out} ({size / 1024:.1f} КБ). Загружайте на платформу; лимит — 5 сабмитов в день.")


if __name__ == "__main__":
    main()
