"""Build the platform submission ZIP: solution.py + requirements.txt at the archive root.

Usage (from the repository root):
    python submission/build_zip.py            # -> submission/dist/submission.zip
    python submission/build_zip.py -o my.zip

Checks before packing: the files compile, requirements are pinned (==), and no file
looks like it contains a secret (API keys must come only from environment variables).
"""

from __future__ import annotations

import argparse
import py_compile
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["solution.py", "requirements.txt"]          # root .py modules + requirements only
SECRET_RE = re.compile(r"(sk-[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9._-]{20,}|api[_-]?key\s*=\s*['\"][^'\"]{8,})",
                       re.IGNORECASE)
MAX_BYTES = 128 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default=str(HERE / "dist" / "submission.zip"))
    args = parser.parse_args()

    for name in FILES:
        path = HERE / name
        if not path.is_file():
            print(f"missing {name}", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8")
        if SECRET_RE.search(text):
            print(f"{name}: looks like it contains a secret, refusing to pack", file=sys.stderr)
            return 1
        if name.endswith(".py"):
            py_compile.compile(str(path), doraise=True)
    for line in (HERE / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" not in line:
            print(f"requirements.txt: version is not pinned: {line}", file=sys.stderr)
            return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in FILES:
            zf.write(HERE / name, arcname=name)   # no folder inside the archive
    size = out.stat().st_size
    if size > MAX_BYTES:
        print(f"archive is too large: {size} bytes", file=sys.stderr)
        return 1
    with zipfile.ZipFile(out) as zf:
        print(f"{out} ({size} bytes):", ", ".join(zf.namelist()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
