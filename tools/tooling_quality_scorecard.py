#!/usr/bin/env python3
"""Tooling-quality scorecard — the ``tools/`` analogue of ``code_quality_scorecard``.

``code_quality_scorecard.py`` grades the *Go module*'s code-debt. But the ~300
Python files in ``tools/`` — which ENFORCE the repo (the commit gates, every
scorecard, the dispatch loop) — had no measuring stick of their own. They are
load-bearing and were the least-watched code in the tree: a 5300-line
``fleet_control_pane.py`` with no ceiling, many modules shipping without a
``_test.py``, no lint baseline.

This is that number, for the Python tooling. It scores ``tools/*.py`` on a small
set of mechanical KPIs, folds them into a weighted score and an A–F grade, and —
crucially — counts the corpus's **py-debt**: the total number of concrete,
re-derivable *defects* (an egregious god-file, a god-function, a non-trivial
module with no ``_test.py``, a ``ruff`` lint diagnostic — an unused import,
undefined name, or other dead-code signal). py-debt is an integer you can drive
toward zero, so an improvement program can state an honest, checkable target
("cut py-debt 2×, from N to N/2") instead of a vibe.

The KPIs (each 0–100):

  lint          ``ruff check`` is clean — no unused-import / dead-code / error  [toolchain]
  architecture  no egregious god-file / god-function outliers                  [static]
  tests         every non-trivial ``tools/*.py`` has a sibling ``_test.py``     [static]
  format        ``ruff format --check`` divergence (SOFT — never gates)         [toolchain]
  docstrings    modules carry a module docstring (SOFT — never gates)           [static]

Three KPIs are HARD (they emit py-debt and gate ``ok``): ``lint``,
``architecture``, ``tests``. Two are deliberately SOFT — they score (a
lint-divergent or doc-poor tree grades lower) but emit no hard debt:

* ``format`` — the repo has NOT adopted ``ruff format`` as its house style, so
  near-every file "would reformat". Counting that as debt would be 300 units of
  pure style-divergence noise, gaming-prone, and not a real defect — so it is an
  advisory nudge only (run ``ruff format`` to adopt; it never gates).
* ``docstrings`` — module-docstring coverage. The cheap way to move it is
  docstring spam, not quality, so it scores but never gates — the same WARN/HARD
  split ``code_quality``'s ``godoc``/``hygiene`` KPIs draw.

The ``lint`` and ``format`` KPIs fail-open when ``ruff`` is absent: a box without
``ruff`` scores them as *skipped* (100, a soft "unmeasured" note), never as a
failure — so a box without the toolchain does not grade the same tree lower.

``ok`` is False iff any HARD defect exists. Soft signals (near-threshold files,
ruff-format divergence, undocumented modules) are advisory and never gate.

Read-only by construction: it reads ``tools/*.py`` and shells out to ``ruff``
(read-only verbs); it edits nothing. Run from the repo ROOT::

    python tools/tooling_quality_scorecard.py                 # human scorecard
    python tools/tooling_quality_scorecard.py --json          # machine payload
    python tools/tooling_quality_scorecard.py --markdown      # snapshot body
    python tools/tooling_quality_scorecard.py --compare B.json # prove the debt moved
    python tools/tooling_quality_scorecard.py --no-toolchain  # static-only (skip ruff)

The companion process mirrors the code-2x program: each HARD defect is one unit
of py-debt to retire; re-running proves the number moved. CI runs it advisory
first, then ratchets — the same path ``docs_scorecard`` and ``code_quality``
followed.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fleet-tooling-quality-scorecard/1"

# The corpus: first-party Python tooling lives under tools/.
TOOLS_REL = "tools"

# ---------------------------------------------------------------------------
# Thresholds. Generous on purpose: a *legitimately* large script (a big CLI
# dispatch table, a data-heavy generator) should not be punished — only an
# EGREGIOUS outlier is hard debt. The softer ceiling is an advisory nudge.
# The hard file ceiling is set so the two ~5000-line god-files the issue named
# (fleet_control_pane.py / _test.py) are flagged while ~1000–1800-line modules
# are only a soft nudge.
# ---------------------------------------------------------------------------
FILE_SOFT_MAX = 1200    # advisory: a Python file this long is worth a second look
FILE_HARD_MAX = 3000    # hard debt: an egregious god-file
FUNC_SOFT_MAX = 80      # advisory: a function this long is worth a second look
FUNC_HARD_MAX = 200     # hard debt: an egregious god-function
TEST_MIN_DEFS = 4       # a module with >= this many def's and no _test.py is debt
DOC_SAMPLE = 12         # how many undocumented modules to name in the soft list

# Per-KPI weights for the composite score. tests + architecture + lint weigh
# most: they are the load-bearing "is it tested / sound / clean" axes. format
# and docstrings weigh least (and are soft): real, but the smallest signal.
KPI_WEIGHTS: dict[str, float] = {
    "lint": 0.25,
    "architecture": 0.25,
    "tests": 0.30,
    "format": 0.10,
    "docstrings": 0.10,
}

# Files under tools/ that are not gradable first-party modules.
PY_EXCLUDE_NAMES = {"__init__.py", "conftest.py"}
# `.dos`/`.fak`/`.claude`/`.tmp` hold full repo CHECKOUTS / copies the agent machinery
# leaves behind (`.dos/_dos_park/_iso_build/`, `.fak/tmp/`, `.claude/worktrees/`,
# `.tmp/pin-check/` + `.tmp/prplan-check/`); walking them
# double-grades every copied .py as phantom debt (a `.dos` tree once added 537 phantom .py
# here). Exclude them, matching the code-slop/code-quality scorecards.
PY_EXCLUDE_DIRS = {"__pycache__", ".git", ".dos", ".fak", ".claude", ".tmp", "testdata", "node_modules"}


def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each takes already-gathered inputs (so tests need no
# disk or toolchain) and returns
#   {kpi, score (0-100 int), detail, defects: [str], soft: [str]}
# where every item in `defects` is one HARD unit of py-debt and every item in
# `soft` is a judgment-call nudge (never gates `ok`).
# ---------------------------------------------------------------------------

def kpi_lint(diagnostics: list[dict[str, Any]] | None) -> dict[str, Any]:
    """``ruff check`` diagnostics over tools/. Each is one HARD unit of py-debt —
    an unused import (F401), undefined name (F821), unused variable (F841), or
    other pyflakes/pycodestyle error: a concrete dead-code / correctness signal.
    `diagnostics` is the parsed `ruff check --output-format json` list, or None
    when ruff is absent (KPI skipped, fail-open — never a failure)."""
    if diagnostics is None:
        return {"kpi": "lint", "score": 100, "detail": "skipped (ruff unavailable / --no-toolchain)",
                "defects": [], "soft": ["ruff not run (ruff unavailable / --no-toolchain)"]}
    defects: list[str] = []
    for d in diagnostics:
        code = d.get("code") or "?"
        fname = (d.get("filename") or "").replace("\\", "/")
        # keep paths repo-relative for stable, comparable output
        if "/tools/" in fname:
            fname = "tools/" + fname.split("/tools/", 1)[1]
        loc = d.get("location") or {}
        row = loc.get("row", "?")
        msg = (d.get("message") or "").strip()[:100]
        defects.append(f"ruff {code} {fname}:{row}: {msg}")
    n = len(defects)
    return {"kpi": "lint", "score": _clamp(100 - 8 * n),
            "detail": ("ruff check clean" if n == 0 else f"{n} ruff diagnostic(s)"),
            "defects": sorted(defects), "soft": []}


def kpi_architecture(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Structural ceilings. A file > FILE_HARD_MAX lines or a function >
    FUNC_HARD_MAX lines is an egregious outlier (hard debt). The softer
    thresholds are advisory nudges — a large-but-reasonable file is not punished.

    `files` is a list of {path, n_lines, long_funcs: [(name, length)]} already
    scanned from disk, so this fold is pure and testable.
    """
    defects: list[str] = []
    soft: list[str] = []
    god_files = 0
    god_funcs = 0
    for f in files:
        if f["n_lines"] > FILE_HARD_MAX:
            god_files += 1
            defects.append(f"god-file {f['path']} ({f['n_lines']} lines > {FILE_HARD_MAX})")
        elif f["n_lines"] > FILE_SOFT_MAX:
            soft.append(f"large file {f['path']} ({f['n_lines']} lines)")
        for name, length in f.get("long_funcs", []):
            if length > FUNC_HARD_MAX:
                god_funcs += 1
                defects.append(f"god-function {f['path']}:{name} ({length} lines > {FUNC_HARD_MAX})")
            elif length > FUNC_SOFT_MAX:
                soft.append(f"long function {f['path']}:{name} ({length} lines)")
    n = god_files + god_funcs
    return {"kpi": "architecture", "score": _clamp(100 - 12 * n - min(20, len(soft))),
            "detail": (f"{god_files} god-file(s), {god_funcs} god-function(s)"
                       if n else f"no egregious outliers ({len(soft)} near-threshold)"),
            "defects": sorted(defects), "soft": sorted(soft)}


def kpi_tests(untested: list[str], n_modules: int) -> dict[str, Any]:
    """Each non-trivial module (>= TEST_MIN_DEFS def's) with no sibling
    ``_test.py`` is one unit of debt — an untested tool. `untested` is the list
    of such module paths; `n_modules` is the count of non-trivial modules."""
    defects = [f"non-trivial module has no _test.py: {p}" for p in sorted(untested)]
    n = len(untested)
    tested = max(0, n_modules - n)
    pct = round(100 * tested / max(1, n_modules), 1)
    return {"kpi": "tests", "score": _clamp(pct),
            "detail": f"{tested}/{n_modules} non-trivial modules have a _test.py ({pct}%)",
            "defects": defects, "soft": []}


def kpi_format(n_reformat: int | None, n_total: int) -> dict[str, Any]:
    """SOFT only. ``ruff format --check`` divergence. The repo has NOT adopted
    ruff's formatter as its house style, so near-every file "would reformat" —
    counting that as hard debt would be hundreds of units of pure style noise,
    gaming-prone, not defects. It scores (a divergent tree grades a touch lower)
    but never gates `ok`. None = ruff absent → skipped (fail-open)."""
    if n_reformat is None:
        return {"kpi": "format", "score": 100, "detail": "skipped (ruff unavailable / --no-toolchain)",
                "defects": [], "soft": ["ruff format not run (ruff unavailable / --no-toolchain)"]}
    clean = max(0, n_total - n_reformat)
    pct = round(100 * clean / max(1, n_total), 1)
    soft = ([f"{n_reformat}/{n_total} file(s) diverge from `ruff format` "
             f"(advisory — repo has not adopted ruff format)"] if n_reformat else [])
    return {"kpi": "format", "score": _clamp(pct),
            "detail": (f"all {n_total} file(s) ruff-format-clean" if not n_reformat
                       else f"{clean}/{n_total} ruff-format-clean ({pct}%)"),
            "defects": [], "soft": soft}


def kpi_docstrings(n_modules: int, n_documented: int, undocumented_sample: list[str]) -> dict[str, Any]:
    """SOFT only. Module-docstring coverage. It scores (an undocumented surface
    grades lower) but emits NO hard debt — docstring spam to move a number is
    gaming, so this never gates `ok`."""
    if n_modules == 0:
        return {"kpi": "docstrings", "score": 100, "detail": "no modules",
                "defects": [], "soft": []}
    pct = round(100 * n_documented / n_modules, 1)
    soft = [f"module has no docstring: {s}" for s in undocumented_sample]
    extra = n_modules - n_documented - len(undocumented_sample)
    if extra > 0:
        soft.append(f"... and {extra} more module(s) with no docstring")
    return {"kpi": "docstrings", "score": _clamp(pct),
            "detail": f"{n_documented}/{n_modules} modules documented ({pct}%)",
            "defects": [], "soft": soft}


# ---------------------------------------------------------------------------
# Per-corpus fold
# ---------------------------------------------------------------------------

def grade_letter(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def build_payload(*, workspace: str, kpis: list[dict[str, Any]],
                  error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error,
            "next_action": "fix the read (run from repo ROOT), then re-run",
            "workspace": workspace, "corpus": {}, "kpis": [],
        }
    by_name = {k["kpi"]: k for k in kpis}
    score = sum(KPI_WEIGHTS[name] * by_name[name]["score"]
                for name in KPI_WEIGHTS if name in by_name)
    score = round(score, 1)
    py_debt = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)
    grade = grade_letter(score)
    breakdown = sorted(
        ({"kpi": k["kpi"], "score": k["score"], "debt": len(k["defects"]),
          "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    corpus = {
        "score": score,
        "grade": grade,
        "py_debt": py_debt,
        "soft_signals": n_soft,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
    }

    if py_debt == 0:
        ok, verdict, finding = True, "OK", "tooling_clean"
        reason = (f"tooling clean: score {score}/100 (grade {grade}), zero py-debt "
                  f"across {len(kpis)} KPIs ({n_soft} advisory signal(s))")
        next_action = "no required edit; re-run after the next tools/ change"
    else:
        ok, verdict, finding = False, "ACTION", "py_debt"
        worst = breakdown[0]
        reason = (f"{py_debt} unit(s) of py-debt; score {score}/100 (grade {grade}); "
                  f"heaviest KPI: {worst['kpi']} ({worst['debt']} defect(s))")
        next_action = ("retire py-debt worst-first (see corpus.breakdown + per-KPI defects): "
                       "fix ruff diagnostics, split god-files/functions, add a _test.py to "
                       "untested modules; re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "kpis": kpis,
    }


# ---------------------------------------------------------------------------
# Disk + toolchain gathering (the impure shell around the pure KPIs)
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _excluded_py(rel: str) -> bool:
    # A Private-Use-Area (U+E000–U+F8FF) or control char in the path is a cp1252<->utf-8
    # mojibake phantom (a faulted absolute path landing under a garbage name), never source.
    if any(0xE000 <= ord(c) <= 0xF8FF or ord(c) < 0x20 for c in rel):
        return True
    parts = Path(rel).parts
    if set(parts) & PY_EXCLUDE_DIRS:
        return True
    return Path(rel).name in PY_EXCLUDE_NAMES


def list_py_files(tools_dir: Path) -> list[str]:
    """Every gradable .py under tools/ (tools-relative posix path), minus
    excluded names/dirs. We walk the tree (not git) so an uncommitted
    improvement is scored immediately."""
    out: list[str] = []
    for p in tools_dir.rglob("*.py"):
        rel = p.relative_to(tools_dir).as_posix()
        if _excluded_py(rel):
            continue
        out.append(rel)
    return sorted(out)


def is_test_file(rel: str) -> bool:
    """A test file is `<stem>_test.py` or `test_<stem>.py` (both conventions
    appear in this tree)."""
    name = Path(rel).name
    return name.endswith("_test.py") or name.startswith("test_")


def scan_py_file(text: str) -> dict[str, Any]:
    """Return {n_lines, n_defs, long_funcs:[(name,len)], module_doc:bool}.

    Uses the stdlib ``ast`` — Python parses itself, so function length is exact
    (``end_lineno - lineno + 1``), not a brace-depth proxy. A file that does not
    parse (a syntax error) yields n_defs=0 / no long_funcs but still counts its
    raw line length, so a broken file is never silently dropped — and ``ruff``
    will have flagged the syntax error as a lint defect regardless."""
    n_lines = len(text.splitlines())
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {"n_lines": n_lines, "n_defs": 0, "long_funcs": [], "module_doc": False}
    long_funcs: list[tuple[str, int]] = []
    n_defs = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n_defs += 1
            end = getattr(node, "end_lineno", None) or node.lineno
            length = end - node.lineno + 1
            if length > FUNC_SOFT_MAX:
                long_funcs.append((node.name, length))
    module_doc = ast.get_docstring(tree) is not None
    return {"n_lines": n_lines, "n_defs": n_defs,
            "long_funcs": long_funcs, "module_doc": module_doc}


def gather(root: Path, *, run_toolchain: bool) -> list[dict[str, Any]]:
    """Read disk + (optionally) shell ruff, then run every pure KPI."""
    tools_dir = root / TOOLS_REL
    rels = list_py_files(tools_dir)

    scanned: list[dict[str, Any]] = []
    module_stems: set[str] = set()       # stems that have a sibling test
    nontest_modules: list[tuple[str, int]] = []  # (rel, n_defs) for non-test modules
    n_modules = 0
    n_documented = 0
    undocumented: list[str] = []

    # first pass: which stems have a test file
    for rel in rels:
        if is_test_file(rel):
            name = Path(rel).name
            if name.endswith("_test.py"):
                module_stems.add(name[: -len("_test.py")])
            elif name.startswith("test_"):
                module_stems.add(name[len("test_"):-len(".py")])

    for rel in rels:
        text = _safe_read(tools_dir / rel)
        info = scan_py_file(text)
        scanned.append({"path": f"tools/{rel}", "n_lines": info["n_lines"],
                        "long_funcs": info["long_funcs"]})
        if is_test_file(rel):
            continue
        # a gradable module (not a test); track docstring + triviality
        n_modules += 1
        if info["module_doc"]:
            n_documented += 1
        elif len(undocumented) < DOC_SAMPLE:
            undocumented.append(f"tools/{rel}")
        nontest_modules.append((rel, info["n_defs"]))

    # non-trivial modules with no sibling test
    untested: list[str] = []
    n_nontrivial = 0
    for rel, n_defs in nontest_modules:
        if n_defs < TEST_MIN_DEFS:
            continue
        n_nontrivial += 1
        stem = Path(rel).name[:-len(".py")]
        if stem not in module_stems:
            untested.append(f"tools/{rel}")

    # --- toolchain shells (ruff) ---
    lint_diags: list[dict[str, Any]] | None = None
    n_reformat: int | None = None
    if run_toolchain:
        lint_diags = _ruff_check(tools_dir)
        n_reformat = _ruff_format_check(tools_dir)

    return [
        kpi_lint(lint_diags),
        kpi_architecture(scanned),
        kpi_tests(untested, n_nontrivial),
        kpi_format(n_reformat, len(rels)),
        kpi_docstrings(n_modules, n_documented, undocumented),
    ]


# Sentinel return code from _run when the tool binary is not installed at all
# (vs. it ran and exited non-zero). A missing toolchain must score as SKIPPED.
_NO_BINARY = -1


def _run(cmd: list[str], cwd: Path, timeout: int = 240) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError as exc:
        return _NO_BINARY, "", f"binary not found: {exc}"
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"


def _ruff_check(tools_dir: Path) -> list[dict[str, Any]] | None:
    """`ruff check --output-format json`. Returns the parsed diagnostics list, or
    None if ruff is not installed (KPI skipped, fail-open). ruff exits 1 when it
    finds diagnostics and 0 when clean — both are valid, parseable verdicts."""
    code, out, _err = _run(["ruff", "check", ".", "--output-format", "json"], tools_dir)
    if code == _NO_BINARY:
        return None
    try:
        data = json.loads(out)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, list) else None


def _ruff_format_check(tools_dir: Path) -> int | None:
    """`ruff format --check` — count of files that would be reformatted. Returns
    None if ruff is absent (skipped). ruff exits 1 when any file would change."""
    code, out, _err = _run(["ruff", "format", "--check", "."], tools_dir)
    if code == _NO_BINARY:
        return None
    # ruff prints one "Would reformat: <path>" line per divergent file.
    return sum(1 for ln in out.splitlines() if ln.startswith("Would reformat:"))


def collect(workspace: Path, *, run_toolchain: bool = True) -> dict[str, Any]:
    root = workspace.resolve()
    if not (root / TOOLS_REL).is_dir():
        return build_payload(workspace=str(root), kpis=[],
                             error=f"no {TOOLS_REL}/ dir at {root} — run from the repo ROOT")
    kpis = gather(root, run_toolchain=run_toolchain)
    return build_payload(workspace=str(root), kpis=kpis)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    lines = [
        f"tooling-quality-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· PY-DEBT {c.get('py_debt', 0)} · {c.get('soft_signals', 0)} advisory"),
        "",
        "per-KPI (worst first):",
        f"  {'score':>5} {'debt':>4}  kpi            detail",
    ]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['kpi']:<14} {b['detail']}")
    lines.append("")
    lines.append("py-debt work-list:")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        lines.append(f"  {k['kpi']} ({len(k['defects'])}):")
        for it in k["defects"][:12]:
            lines.append(f"      - {it}")
        if len(k["defects"]) > 12:
            lines.append(f"      ... and {len(k['defects']) - 12} more")
    if not any_defect:
        lines.append("  (none — zero py-debt)")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    out: list[str] = []
    out.append("---")
    out.append('title: "fak tooling-quality scorecard — the py-debt measuring stick"')
    out.append('description: "fak\'s deterministic tooling-quality scorecard for the Python '
               'tools/ tree: KPIs folded into a composite score and the headline py-debt metric, '
               're-derived from disk and ruff."')
    out.append("---")
    out.append("")
    out.append("# Tooling-quality scorecard")
    out.append("")
    if stamp:
        out.append(f"<!-- tooling-quality-scorecard: {stamp} · process: tools/tooling_quality_scorecard.py -->")
        out.append("")
    out.append("This is the measuring stick for the Python `tools/` tree — the `tools/` "
               "counterpart of the code-quality scorecard. Every number below is re-derived "
               "from disk and `ruff` by `tools/tooling_quality_scorecard.py` — no hand-entry. "
               "The headline metric is **py-debt**: the count of concrete, mechanical defects "
               "(an egregious god-file, a god-function, a non-trivial module with no `_test.py`, "
               "a `ruff` lint diagnostic). Driving py-debt toward zero is what makes "
               "\"better tooling\" provable.")
    out.append("")
    out.append("> Regenerate: `python tools/tooling_quality_scorecard.py --markdown --stamp DATE "
               "> docs/TOOLING-QUALITY-SCORECARD.md`")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **py-debt (total HARD defects)** | **{c.get('py_debt', 0)}** |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append(f"| Advisory (soft) signals | {c.get('soft_signals', 0)} |")
    out.append("")
    out.append("## Per-KPI")
    out.append("")
    out.append("`debt` = units of HARD py-debt in that KPI. `format` and `docstrings` are "
               "advisory (they score but emit no hard debt — ruff-format style is not adopted, "
               "and docstring spam is gaming, not quality).")
    out.append("")
    out.append("| KPI | Score | Debt | Detail |")
    out.append("|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    out.append("## py-debt work-list")
    out.append("")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        out.append(f"### `{k['kpi']}` — {len(k['defects'])} defect(s), score {k['score']}")
        for it in k["defects"]:
            out.append(f"- {it}")
        out.append("")
    if not any_defect:
        out.append("No py-debt: every HARD KPI is clean. 🎉")
        out.append("")
    return "\n".join(out)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("py_debt", 0), cur.get("py_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ratio = "∞ (zero)" if cd == 0 else f"{bd / cd:.1f}×"
    lines = [
        f"py-debt: {bd} -> {cd}   ({ratio} fewer defects)",
        f"score:   {bo}/100 -> {co}/100   ({'+' if co >= bo else ''}{round(co - bo, 1)})",
    ]
    bk = b.get("debt_by_kpi") or {}
    ck = cur.get("debt_by_kpi") or {}
    for kpi in sorted(set(bk) | set(ck)):
        lines.append(f"  {kpi:<14} {bk.get(kpi, 0)} -> {ck.get(kpi, 0)}")
    target = max(0, bd // 2)
    if cd <= target:
        lines.append(f"VERDICT: >=2x py-debt reduction achieved ({bd} -> {cd}).")
    else:
        lines.append(f"VERDICT: not yet 2x — need py-debt <= {target} (now {cd}).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tooling-quality scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true",
                    help="emit the TOOLING-QUALITY-SCORECARD.md body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the py-debt delta vs a prior baseline JSON")
    ap.add_argument("--no-toolchain", action="store_true",
                    help="skip ruff (static-only, fast)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, run_toolchain=not args.no_toolchain)

    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"error: cannot read baseline {args.compare}: {exc}", file=sys.stderr)
            return 2
        print(render_compare(baseline, payload))
    elif args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    else:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
