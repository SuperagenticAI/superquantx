#!/usr/bin/env python
"""Run the Python code blocks in the docs against the installed SuperQuantX.

Blocks on one page run in order in a shared namespace, the way a reader
follows the page. Put ``<!-- docs-check: skip -->`` on the line before a
fence to skip that block (for hardware snippets that need credentials, or
code that only illustrates a pattern). Every page runs in its own process.

Pages that define pytest tests (``def test_...``) are also run through
pytest, so the test bodies are executed and not only defined.

Usage:
    python scripts/check_docs_examples.py [paths...]   # default: docs/ and README.md

Environment:
    DOCS_CHECK_TIMEOUT  seconds allowed per page (default 600)
    DOCS_CHECK_SLOW     report blocks slower than this many seconds (default 10)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^(?P<indent>[ \t]*)```python[^\n]*\n(?P<body>.*?)^(?P=indent)```", re.S | re.M)
SKIP = "<!-- docs-check: skip -->"
TIMEOUT = int(os.environ.get("DOCS_CHECK_TIMEOUT", "600"))
SLOW = float(os.environ.get("DOCS_CHECK_SLOW", "10"))

RUNNER = r'''
import json, os, sys, time, warnings
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")
blocks = json.loads(sys.stdin.read())
ns = {"__name__": "__main__"}
real_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")
for line, code in blocks:
    t0 = time.time()
    real_stdout.write(f"@@START {line}\n"); real_stdout.flush()
    try:
        exec(compile(code, f"block@{line}", "exec"), ns)
        err = None
    except BaseException as e:
        err = f"{type(e).__name__}: {e}"[:300]
    real_stdout.write("@@RES " + json.dumps([line, err, round(time.time() - t0, 1)]) + "\n")
    real_stdout.flush()
'''


def blocks_for(path: Path):
    text = path.read_text()
    out, skipped = [], 0
    for m in FENCE.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        before = text[: m.start()].rstrip().splitlines()
        if before and before[-1].strip() == SKIP:
            skipped += 1
            continue
        indent = m.group("indent")
        body = "\n".join(l[len(indent):] if l.startswith(indent) else l for l in m.group("body").splitlines())
        out.append((line, body))
    return out, skipped


def run_pytest(rel: Path, blocks) -> tuple[int, int]:
    """Run a page's blocks as one pytest module. Returns (tests run, failures)."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        print(f"NOTE {rel}: pytest not installed, test bodies not executed")
        return 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        mod = Path(tmp) / f"test_docs_{rel.stem.replace('-', '_')}.py"
        mod.write_text("\n\n".join(code for _, code in blocks))
        try:
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(mod)],
                               capture_output=True, text=True, timeout=TIMEOUT, cwd=ROOT)
        except subprocess.TimeoutExpired:
            print(f"FAIL {rel}  pytest timed out")
            return 0, 1
    fails = [l.split("::", 1)[1] for l in p.stdout.splitlines() if l.startswith(("FAILED", "ERROR")) and "::" in l]
    for name in fails:
        print(f"FAIL {rel}  test {name}")
    m = re.search(r"(\d+) passed", p.stdout)
    passed = int(m.group(1)) if m else 0
    if p.returncode not in (0, 5) and not fails:
        print(f"FAIL {rel}  pytest exited {p.returncode}")
        return passed, 1
    return passed + len(fails), len(fails)


def main(argv):
    targets = [Path(a).resolve() for a in argv] or [ROOT / "docs", ROOT / "README.md"]
    files = []
    for t in targets:
        files += sorted(t.rglob("*.md")) if t.is_dir() else [t]
    ran = failed = skipped = tests = 0
    for f in files:
        blocks, sk = blocks_for(f)
        skipped += sk
        if not blocks:
            continue
        rel = f.relative_to(ROOT)
        try:
            p = subprocess.run([sys.executable, "-c", RUNNER], input=json.dumps(blocks),
                               capture_output=True, text=True, timeout=TIMEOUT, cwd=ROOT)
            out, crashed = p.stdout, p.returncode != 0
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            crashed = True
        done = [json.loads(l[6:]) for l in out.splitlines() if l.startswith("@@RES ")]
        for line, err, secs in done:
            ran += 1
            if err:
                failed += 1
                print(f"FAIL {rel}:{line}  {err}")
            elif secs > SLOW:
                print(f"SLOW {rel}:{line}  {secs}s")
        if crashed and len(done) < len(blocks):
            starts = re.findall(r"@@START (\d+)", out)
            ran += 1
            failed += 1
            print(f"FAIL {rel}:{starts[-1] if starts else '?'}  timed out or crashed")
        if any("def test_" in code for _, code in blocks):
            f_tests, f_fail = run_pytest(rel, blocks)
            tests += f_tests
            failed += f_fail
    print(f"\n{tests} doc tests run under pytest")
    print(f"{ran} blocks run, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
