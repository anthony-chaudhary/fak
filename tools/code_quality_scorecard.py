#!/usr/bin/env python3
"""Code-quality scorecard — the measuring stick that makes "better code" provable.

The docs already have a measuring stick (``docs_scorecard.py``): a number that
turns "make the docs better" from a vibe into a falsifiable target. The *code*
had no such number. "Cleaner code", "better architecture", "more tested" were
unfalsifiable claims — there was nothing to move, so there was no honest way to
say a change made the kernel better.

This is that number, for the Go module. It scores the codebase on ten
mechanical KPIs, folds them into a weighted score and an A–F grade, and —
crucially — counts the corpus's **code-debt**: the total number of concrete,
re-derivable *defects* (an unformatted file, a `go vet` diagnostic, an
egregious god-function, a non-trivial package with zero tests, an untagged
honesty claim, an external dependency, an unwitnessed ship).
Code-debt is an integer you can drive toward zero, so an improvement program
can state an honest, checkable target ("cut code-debt 2×, from N to N/2")
instead of a vibe.

The ten KPIs (each 0–100). Nine are tree-deterministic (same working tree →
same score). The exception is ``ship_integrity``, which grades recent *history*,
not the tree, so it is HEAD-relative — its number moves as commits land even
when the tree is byte-identical (pin ``--range`` to a fixed anchor for a stable
read). The two ``[toolchain]``/``[DOS]`` KPIs also fail-open when their tool is
absent: a missing ``go``/``gofmt``/``dos`` scores the KPI as *skipped* (100, a
soft "unmeasured" note), never as a failure — so a box without the toolchain
does not grade the same tree lower than one with it.

  build         `go build ./...` exits 0 (the kernel compiles)            [toolchain]
  vet           `go vet ./...` is clean (no diagnostics)                  [toolchain]
  format        `gofmt -l` is empty (every file is canonically formatted) [toolchain]
  deps          go.mod has zero external requires and there is no go.sum  [static]
  honesty       every `- [` line in CLAIMS.md carries exactly one tag     [static]
  architecture  no egregious god-file / god-function outliers            [static]
  tests         every non-trivial package has a real Test/Benchmark/Fuzz  [static]
  godoc         exported symbols carry a doc comment (SOFT — never gates) [static]
  hygiene       TODO/FIXME/HACK/XXX marker density (SOFT — never gates)   [static]
  ship_integrity  `dos review` shows zero unwitnessed RESIDUAL commits    [DOS]

``ship_integrity`` is the DOS grounding: a commit's *subject* is a claim the
author wrote, but the *files it touched* are what git recorded. `dos review`
partitions a commit range into CLEARED (the diff witnessed the claim),
RESIDUAL (a claim the diff could NOT witness — the only place review attention
is load-bearing), and UNVERIFIABLE. A RESIDUAL commit is a unit of code-debt
the kernel itself flags: the ship said something the diff can't back. This KPI
reads that verdict from evidence the committing agent could not author.

``godoc`` and ``hygiene`` are deliberately SOFT — they score (a doc-poor or
marker-heavy tree grades lower) but emit no hard debt, because the cheap way to
move either number is *gaming*, not quality: doc-comment spam (``// X does X``)
for godoc, and DELETING a ``// TODO`` (hiding the gap rather than closing it) for
hygiene — and this repo keeps an honest ``[STUB]`` ledger where a "TODO:
implement" on stub plumbing is a feature of that honesty, not a defect. Voice,
prose, and honest deferral markers are judgment calls; the same WARN/HARD split
``docs_scorecard`` draws.

``ok`` is False iff any HARD defect exists. Soft signals (near-threshold files,
undocumented exports) are advisory and never gate.

Read-only by construction: it reads ``.go`` files, ``go.mod``, ``CLAIMS.md``,
and shells out to ``go`` / ``gofmt`` / ``dos`` (all read-only verbs); it edits
nothing. Run from the repo ROOT::

    python tools/code_quality_scorecard.py                 # human scorecard
    python tools/code_quality_scorecard.py --json          # machine payload
    python tools/code_quality_scorecard.py --markdown      # CODE-QUALITY-SCORECARD.md body
    python tools/code_quality_scorecard.py --no-toolchain  # static-only (skip go/gofmt)
    python tools/code_quality_scorecard.py --no-dos        # skip the dos ship-integrity probe

The companion process is the code-2x program (the ``/quality-score`` skill):
each HARD defect is one unit of code-debt to retire; re-running proves the
number moved.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fleet-code-quality-scorecard/1"

# Repo-root-relative inputs (best-effort; a missing one degrades a check, never errors).
GOMOD_REL = "go.mod"
GOSUM_REL = "go.sum"
CLAIMS_REL = "CLAIMS.md"

# ---------------------------------------------------------------------------
# Thresholds. Generous on purpose: a *legitimately* large file (a tensor
# dequant kernel, a CLI dispatch table) should not be punished — only an
# EGREGIOUS outlier is hard debt. The softer ceiling is an advisory nudge.
# ---------------------------------------------------------------------------
FILE_SOFT_MAX = 800     # advisory: a file this long is worth a second look
FILE_HARD_MAX = 1500    # hard debt: an egregious god-file
FUNC_SOFT_MAX = 80      # advisory: a function this long is worth a second look
FUNC_HARD_MAX = 200     # hard debt: an egregious god-function
TEST_MIN_FUNCS = 4      # a package with >= this many funcs and no _test.go is debt
HYGIENE_CAP_PER_FILE = 3  # a file's marker count is capped (one messy file != 50 debt)
GODOC_SAMPLE = 12       # how many undocumented exports to name in the soft list

# Per-KPI weights for the composite score. build + vet + tests + architecture
# weigh most: they are the load-bearing "does it work / is it sound" axes.
# hygiene and godoc weigh least: real, but the smallest signal of quality.
KPI_WEIGHTS: dict[str, float] = {
    "build": 0.15,
    "vet": 0.12,
    "format": 0.08,
    "deps": 0.08,
    "honesty": 0.10,
    "architecture": 0.12,
    "tests": 0.12,
    "hygiene": 0.05,
    "godoc": 0.08,
    "ship_integrity": 0.10,
}

# Directories whose .go is NOT first-party shipped kernel code. testdata holds
# fixtures (intentionally odd); vendored/generated trees are not ours to grade.
GO_EXCLUDE_DIRS = {".git", "node_modules", "testdata", "vendor", "__pycache__"}

_MARKER_RE = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b")
# A real Go test entry point — used to confirm a _test.go is not just a bare
# `package foo` marker. Example funcs are legitimate tests; all four count.
_TESTFUNC_RE = re.compile(r"^func\s+(Test|Benchmark|Fuzz|Example)\w*\s*\(", re.MULTILINE)
# A top-level declaration we expect a doc comment on. Exported = capitalised name.
_EXPORTED_FUNC_RE = re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Z]\w*)\s*[\(\[]")
_EXPORTED_TYPE_RE = re.compile(r"^type\s+([A-Z]\w*)\b")
_EXPORTED_VARCONST_RE = re.compile(r"^(?:var|const)\s+([A-Z]\w*)\b")
# A function header (for the brace-depth length scan). Either exported or not.
_FUNC_HEADER_RE = re.compile(r"^func\b")


def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


def _external_requires(gomod_text: str) -> list[str]:
    """Module paths in go.mod's require directive(s). Handles both the single
    `require x v1` form and the `require ( ... )` block. The stdlib needs no
    require, so any entry here is an external dependency."""
    out: list[str] = []
    in_block = False
    for raw in gomod_text.splitlines():
        line = raw.strip()
        if line.startswith("//"):
            continue
        if in_block:
            if line.startswith(")"):
                in_block = False
                continue
            mod = line.split()[0] if line.split() else ""
            if mod:
                out.append(mod)
            continue
        if line.startswith("require ("):
            in_block = True
            continue
        if line.startswith("require "):
            parts = line[len("require "):].split()
            if parts:
                out.append(parts[0])
    return out


def _is_local_path(target: str) -> bool:
    """A replace RHS that points at the filesystem (in-tree / vendored locally)
    rather than an external module: ./x, ../x, /abs, or a Windows drive path."""
    return (target.startswith(("./", "../", "/", ".\\", "..\\"))
            or (len(target) >= 2 and target[1] == ":"))


def _external_replaces(gomod_text: str) -> list[str]:
    """Module paths pulled in by a `replace ... => <module> <version>` whose
    target is an EXTERNAL module (a versioned path), not a local directory. A
    local replace (`=> ./foo`) keeps the dep in-tree and is not counted."""
    out: list[str] = []
    in_block = False
    for raw in gomod_text.splitlines():
        line = raw.strip()
        if line.startswith("//"):
            continue
        body = ""
        if in_block:
            if line.startswith(")"):
                in_block = False
                continue
            body = line
        elif line.startswith("replace ("):
            in_block = True
            continue
        elif line.startswith("replace "):
            body = line[len("replace "):]
        if not body or "=>" not in body:
            continue
        rhs = body.split("=>", 1)[1].strip()
        parts = rhs.split()
        if not parts or _is_local_path(parts[0]):
            continue
        # external module target: `modpath vX.Y.Z` (two tokens) is the real dep
        if len(parts) >= 2:
            out.append(parts[0])
    return out


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each takes already-gathered inputs (so tests need no
# disk or toolchain) and returns
#   {kpi, score (0-100 int), detail, defects: [str], soft: [str]}
# where every item in `defects` is one HARD unit of code-debt and every item in
# `soft` is a judgment-call nudge (never gates `ok`).
# ---------------------------------------------------------------------------

def kpi_build(build_ok: bool | None, build_err: str) -> dict[str, Any]:
    """A build failure is the worst defect a codebase can have: nothing else is
    trustworthy if it does not compile. Counts as a large fixed block of debt."""
    if build_ok is None:
        return {"kpi": "build", "score": 100, "detail": "skipped (--no-toolchain)",
                "defects": [], "soft": ["build not checked (--no-toolchain)"]}
    if build_ok:
        return {"kpi": "build", "score": 100, "detail": "go build ./... exit 0",
                "defects": [], "soft": []}
    first = (build_err.strip().splitlines() or ["unknown error"])[0]
    return {"kpi": "build", "score": 0,
            "detail": "go build ./... FAILED",
            "defects": [f"build failure: {first[:160]}"], "soft": []}


def kpi_vet(vet_ok: bool | None, vet_diags: list[str]) -> dict[str, Any]:
    if vet_ok is None:
        return {"kpi": "vet", "score": 100, "detail": "skipped (--no-toolchain)",
                "defects": [], "soft": ["vet not checked (--no-toolchain)"]}
    n = len(vet_diags)
    defects = [f"vet: {d.strip()[:160]}" for d in vet_diags]
    return {"kpi": "vet", "score": _clamp(100 - 15 * n),
            "detail": ("clean" if n == 0 else f"{n} diagnostic(s)"),
            "defects": defects, "soft": []}


def kpi_format(unformatted: list[str] | None) -> dict[str, Any]:
    if unformatted is None:
        return {"kpi": "format", "score": 100, "detail": "skipped (--no-toolchain)",
                "defects": [], "soft": ["gofmt not checked (--no-toolchain)"]}
    n = len(unformatted)
    defects = [f"unformatted (run gofmt -w): {f}" for f in sorted(unformatted)]
    return {"kpi": "format", "score": _clamp(100 - 12 * n),
            "detail": ("all files gofmt-clean" if n == 0 else f"{n} unformatted file(s)"),
            "defects": defects, "soft": []}


def kpi_deps(gomod_text: str, gosum_exists: bool) -> dict[str, Any]:
    """Zero-external-dependency is a load-bearing property of this kernel (no
    go.sum, stdlib only). Each external require is one unit of debt; a present
    go.sum is the canary that a dep slipped in."""
    defects: list[str] = []
    external = _external_requires(gomod_text)
    for mod in external:
        defects.append(f"external dependency added: {mod}")
    # A `replace X => Y vVER` whose target is a *versioned module* (not a ./ ../ /
    # local path) pulls in an external dep that has no `require` line — count it.
    for mod in _external_replaces(gomod_text):
        defects.append(f"external dependency via replace: {mod}")
    if gosum_exists:
        defects.append("go.sum exists (the zero-dep invariant broke)")
    n = len(defects)
    n_ext = len(external) + len(_external_replaces(gomod_text))
    return {"kpi": "deps", "score": _clamp(100 - 25 * n),
            "detail": ("stdlib-only, no go.sum" if n == 0
                       else f"{n_ext} external dep(s)"
                            f"{' + go.sum' if gosum_exists else ''}"),
            "defects": defects,
            "soft": (["deps counts go.mod require/replace only; copied-in source is "
                      "invisible to a static scan"] if n == 0 else [])}


def kpi_honesty(claims_text: str) -> dict[str, Any]:
    """Re-implements the repo's own claims-lint: every line beginning `- [` in
    CLAIMS.md must carry exactly one of [SHIPPED]/[SIMULATED]/[STUB]. An
    untagged or double-tagged claim is one unit of debt — an honesty leak."""
    if not claims_text:
        return {"kpi": "honesty", "score": 100, "detail": "no CLAIMS.md (skipped)",
                "defects": [], "soft": ["CLAIMS.md not found"]}
    tags = ("[SHIPPED]", "[SIMULATED]", "[STUB]")
    total = 0
    violations: list[str] = []
    in_fence = False
    for line in claims_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence  # a fenced block: its `- [` lines are examples, not ledger claims
            continue
        if in_fence or not line.startswith("- ["):
            continue
        total += 1
        n_tags = sum(1 for t in tags if t in line)
        if n_tags != 1:
            violations.append(line.strip()[:120])
    defects = [f"untagged/double-tagged claim: {v}" for v in violations]
    score = _clamp(100 - 20 * len(violations))
    return {"kpi": "honesty", "score": score,
            "detail": (f"{total} claims, all tagged" if not violations
                       else f"{len(violations)}/{total} claim(s) mis-tagged"),
            "defects": defects, "soft": []}


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
            "defects": defects, "soft": soft}


def kpi_tests(untested: list[str], n_packages: int) -> dict[str, Any]:
    """Each non-trivial package (>= TEST_MIN_FUNCS funcs) with no _test.go is one
    unit of debt — an untested seam. `untested` is the list of such packages."""
    defects = [f"non-trivial package has no _test.go: {p}" for p in sorted(untested)]
    n = len(untested)
    tested = max(0, n_packages - n)
    pct = round(100 * tested / max(1, n_packages), 1)
    return {"kpi": "tests", "score": _clamp(pct),
            "detail": (f"{tested}/{n_packages} non-trivial packages tested ({pct}%)"),
            "defects": defects, "soft": []}


def kpi_hygiene(markers: list[tuple[str, int]]) -> dict[str, Any]:
    """SOFT only. TODO/FIXME/HACK/XXX markers in shipped (non-test) code, capped
    per file. Advisory, never hard debt: this repo keeps an honest [STUB] ledger,
    so a `// TODO: implement` on stub plumbing is a *feature* of that honesty, not
    a defect — making it hard debt would reward DELETING the marker (hiding the
    gap) instead of closing it. Same WARN-not-HARD split as godoc. It still
    scores, so a marker-heavy tree grades lower, but it never gates `ok`."""
    total = sum(c for _, c in markers)
    soft = [f"{c} marker(s) (TODO/FIXME/HACK/XXX): {p}" for p, c in sorted(markers)]
    return {"kpi": "hygiene", "score": _clamp(100 - 6 * total),
            "detail": ("no stray markers" if total == 0 else f"{total} marker(s) in {len(markers)} file(s)"),
            "defects": [], "soft": soft}


def kpi_godoc(n_exported: int, n_documented: int, undocumented_sample: list[str]) -> dict[str, Any]:
    """SOFT only. Exported-symbol doc-comment coverage. It scores (a poorly
    documented surface grades lower) but emits NO hard debt — writing doc
    comments to move a number is gaming, so this never gates `ok`."""
    if n_exported == 0:
        return {"kpi": "godoc", "score": 100, "detail": "no exported symbols",
                "defects": [], "soft": []}
    pct = round(100 * n_documented / n_exported, 1)
    soft = [f"undocumented exported symbol: {s}" for s in undocumented_sample]
    if n_exported - n_documented > len(undocumented_sample):
        soft.append(f"... and {n_exported - n_documented - len(undocumented_sample)} more undocumented exports")
    return {"kpi": "godoc", "score": _clamp(pct),
            "detail": f"{n_documented}/{n_exported} exported symbols documented ({pct}%)",
            "defects": [], "soft": soft}


def kpi_ship_integrity(dos: dict[str, Any] | None) -> dict[str, Any]:
    """The DOS grounding. `dos` is the parsed `dos review --json` payload (or
    None if skipped/unavailable). Each RESIDUAL commit — a claim the diff could
    not witness — is one unit of debt the kernel itself flagged."""
    if dos is None:
        return {"kpi": "ship_integrity", "score": 100, "detail": "skipped (--no-dos)",
                "defects": [], "soft": ["dos review not run (--no-dos / dos unavailable)"]}
    if dos.get("error"):
        return {"kpi": "ship_integrity", "score": 100,
                "detail": f"UNMEASURED (dos review unavailable): {dos['error'][:60]}",
                "defects": [],
                "soft": [f"ship_integrity UNMEASURED — dos unavailable, scored 100 "
                         f"(fail-open, not a witnessed-clean review): {dos['error'][:100]}"]}
    residual = dos.get("residual", []) or []
    n = len(residual)
    rng = dos.get("rev_range", "?")
    defects = [f"unwitnessed ship (RESIDUAL) in {rng}: {r.get('sha','?')} {r.get('subject','')[:80]}"
               for r in residual]
    cleared = dos.get("cleared_rate")
    detail = (f"{dos.get('checkable','?')} checkable commit(s) in {rng}, "
              f"{n} residual, cleared_rate {cleared}")
    return {"kpi": "ship_integrity", "score": _clamp(100 - 25 * n),
            "detail": detail, "defects": defects, "soft": []}


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
    code_debt = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)
    grade = grade_letter(score)
    # per-KPI debt breakdown, worst (most defectful) first
    breakdown = sorted(
        ({"kpi": k["kpi"], "score": k["score"], "debt": len(k["defects"]),
          "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    corpus = {
        "score": score,
        "grade": grade,
        "code_debt": code_debt,
        "soft_signals": n_soft,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
    }

    if code_debt == 0:
        ok, verdict, finding = True, "OK", "code_clean"
        reason = (f"code clean: score {score}/100 (grade {grade}), zero code-debt "
                  f"across {len(kpis)} KPIs ({n_soft} advisory signal(s))")
        next_action = "no required edit; re-run after the next code change"
    else:
        ok, verdict, finding = False, "ACTION", "code_debt"
        worst = breakdown[0]
        reason = (f"{code_debt} unit(s) of code-debt; score {score}/100 (grade {grade}); "
                  f"heaviest KPI: {worst['kpi']} ({worst['debt']} defect(s))")
        next_action = ("retire code-debt worst-first (see corpus.breakdown + per-KPI defects): "
                       "gofmt -w the unformatted files, fix vet diagnostics, split god-functions, "
                       "add tests to untested packages, resolve markers; re-run to prove the drop")

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


def _excluded_go(rel: str) -> bool:
    parts = set(Path(rel).parts)
    return bool(parts & GO_EXCLUDE_DIRS)


def list_go_files(root: Path, *, tests: bool) -> list[str]:
    """Tracked-ish enumeration: every .go under root minus excluded dirs. We walk
    the tree (not git) so an uncommitted improvement is scored immediately."""
    out: list[str] = []
    for p in root.rglob("*.go"):
        rel = p.relative_to(root).as_posix()
        if _excluded_go(rel):
            continue
        is_test = rel.endswith("_test.go")
        if is_test == tests:
            out.append(rel)
    return sorted(out)


def _code_only(line: str, in_raw: bool, in_block: bool) -> tuple[str, bool, bool]:
    """Return (code, in_raw, in_block): `line` with the *contents* of string,
    rune, and backtick literals and of `//` / `/* */` comments blanked out, so a
    brace inside a literal or comment is never counted by the depth scan. Tracks
    the two spans that cross lines — a backtick raw string (``in_raw``) and a
    block comment (``in_block``).

    This is the security-relevant fix: the old scanner counted raw `{`/`}` after
    only stripping a trailing `//`, so a single ``s := "}"`` line could collapse a
    250-line god-function to length 3 (erasing architecture debt) or a stray `{`
    in a literal could forge one. Blanking literals first makes both un-gameable.
    """
    out: list[str] = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if in_block:
            if c == "*" and i + 1 < n and line[i + 1] == "/":
                in_block = False
                i += 2
                continue
            i += 1
            continue
        if in_raw:
            if c == "`":
                in_raw = False
            i += 1
            continue
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            break  # rest of line is a // comment
        if c == "/" and i + 1 < n and line[i + 1] == "*":
            in_block = True
            i += 2
            continue
        if c == "`":
            in_raw = True
            i += 1
            continue
        if c == '"' or c == "'":
            quote = c
            i += 1
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out), in_raw, in_block


def scan_go_file(text: str) -> dict[str, Any]:
    """Return {n_lines, n_funcs, long_funcs:[(name,len)], exported:[(kind,name,documented)]}.

    Function length is a brace-depth scan from a `func` header to the line where
    depth returns to zero — an AST-free proxy (Python stdlib has no Go parser),
    deliberately generous because the thresholds that gate are generous too.
    Braces inside string/rune/backtick literals and comments are blanked first
    (`_code_only`), and a header line + its state-at-start are tracked so a `func`
    that begins *inside* a multi-line raw string / block comment is not mistaken
    for a declaration.
    """
    lines = text.splitlines()
    n_lines = len(lines)
    long_funcs: list[tuple[str, int]] = []
    exported: list[tuple[str, str, bool]] = []
    n_funcs = 0

    # Pre-pass: code-only form of each line + the (in_raw, in_block) state BEFORE
    # the line. Brace counting and declaration detection both use these.
    code_lines: list[str] = []
    state_at_start: list[tuple[bool, bool]] = []
    in_raw = in_block = False
    for ln in lines:
        state_at_start.append((in_raw, in_block))
        code, in_raw, in_block = _code_only(ln, in_raw, in_block)
        code_lines.append(code)

    i = 0
    while i < len(lines):
        raw = lines[i]
        sr, sb = state_at_start[i]
        in_literal = sr or sb  # this line begins inside a raw string / block comment
        if not in_literal and _FUNC_HEADER_RE.match(raw):
            n_funcs += 1
            # scan to the matching closing brace using net brace depth per line.
            # `seen_open` only flips once depth has actually been positive, so a
            # BALANCED `interface{}` / `struct{}` on a multi-line signature line
            # (net 0) no longer trips an early break.
            depth = 0
            seen_open = False
            j = i
            while j < len(lines):
                depth += code_lines[j].count("{") - code_lines[j].count("}")
                if depth > 0:
                    seen_open = True
                if seen_open and depth <= 0:
                    break
                j += 1
            length = j - i + 1
            mfunc = _EXPORTED_FUNC_RE.match(raw)
            name = mfunc.group(1) if mfunc else _anon_func_name(raw)
            if length > FUNC_SOFT_MAX:
                long_funcs.append((name, length))
            if mfunc:
                exported.append(("func", mfunc.group(1), _is_documented(lines, i)))
            i = j + 1
            continue
        if not in_literal:
            mtype = _EXPORTED_TYPE_RE.match(raw)
            if mtype:
                exported.append(("type", mtype.group(1), _is_documented(lines, i)))
            else:
                mvar = _EXPORTED_VARCONST_RE.match(raw)
                if mvar:
                    exported.append((raw.split()[0], mvar.group(1), _is_documented(lines, i)))
        i += 1

    return {"n_lines": n_lines, "n_funcs": n_funcs,
            "long_funcs": long_funcs, "exported": exported}


def _anon_func_name(header: str) -> str:
    m = re.match(r"^func\s+(?:\([^)]*\)\s*)?(\w+)", header)
    return m.group(1) if m else "func"


def _is_documented(lines: list[str], idx: int) -> bool:
    """A declaration is documented iff the immediately preceding non-blank line is
    a // comment (Go convention puts the doc comment directly above)."""
    k = idx - 1
    while k >= 0 and not lines[k].strip():
        k -= 1
    return k >= 0 and lines[k].lstrip().startswith("//")


def package_of(rel: str) -> str:
    return Path(rel).parent.as_posix()


def gather(root: Path, *, run_toolchain: bool, run_dos: bool,
           dos_range: str) -> list[dict[str, Any]]:
    """Read disk + (optionally) shell the toolchain, then run every pure KPI."""
    # --- static reads ---
    gomod = _safe_read(root / GOMOD_REL)
    gosum_exists = (root / GOSUM_REL).exists()
    claims = _safe_read(root / CLAIMS_REL)

    src_files = list_go_files(root, tests=False)
    test_files = list_go_files(root, tests=True)
    # A package counts as TESTED only if one of its _test.go files contains a real
    # Test/Benchmark/Fuzz/Example function — NOT merely a `package foo` marker file.
    # (Presence-only crediting let an empty _test.go clear the hard tests-debt.)
    test_pkgs: set[str] = set()
    for rel in test_files:
        if _TESTFUNC_RE.search(_safe_read(root / rel)):
            test_pkgs.add(package_of(rel))

    scanned: list[dict[str, Any]] = []
    pkg_funccount: dict[str, int] = {}
    markers: list[tuple[str, int]] = []
    n_exported = n_documented = 0
    undocumented: list[str] = []
    for rel in src_files:
        text = _safe_read(root / rel)
        info = scan_go_file(text)
        scanned.append({"path": rel, "n_lines": info["n_lines"],
                        "long_funcs": info["long_funcs"]})
        pkg = package_of(rel)
        # count FUNCTION declarations once (literal-aware) for the triviality gate.
        # NOT exported-symbol count (which re-counted exported funcs and folded in
        # types/vars, halving the effective TEST_MIN_FUNCS bar).
        pkg_funccount[pkg] = pkg_funccount.get(pkg, 0) + info["n_funcs"]
        for kind, name, documented in info["exported"]:
            n_exported += 1
            if documented:
                n_documented += 1
            elif len(undocumented) < GODOC_SAMPLE:
                undocumented.append(f"{rel}:{name} ({kind})")
        n_marks = len(_MARKER_RE.findall(text))
        if n_marks:
            markers.append((rel, min(HYGIENE_CAP_PER_FILE, n_marks)))

    # non-trivial packages with no _test.go
    untested: list[str] = []
    all_pkgs = {package_of(r) for r in src_files}
    for pkg in sorted(all_pkgs):
        if pkg in test_pkgs:
            continue
        if pkg_funccount.get(pkg, 0) >= TEST_MIN_FUNCS:
            untested.append(pkg)

    # --- toolchain shells ---
    build_ok = vet_ok = None
    build_err = ""
    vet_diags: list[str] = []
    unformatted: list[str] | None = None
    if run_toolchain:
        build_ok, build_err = _go_build(root)
        vet_ok, vet_diags = _go_vet(root)
        unformatted = _gofmt_list(root)

    # --- dos ship-integrity ---
    dos_payload = _dos_review(root, dos_range) if run_dos else None

    return [
        kpi_build(build_ok, build_err),
        kpi_vet(vet_ok, vet_diags),
        kpi_format(unformatted),
        kpi_deps(gomod, gosum_exists),
        kpi_honesty(claims),
        kpi_architecture(scanned),
        kpi_tests(untested, len(all_pkgs)),
        kpi_hygiene(markers),
        kpi_godoc(n_exported, n_documented, undocumented),
        kpi_ship_integrity(dos_payload),
    ]


# Sentinel return code from _run when the tool binary is not installed at all
# (vs. it ran and exited non-zero). A missing toolchain must score as SKIPPED,
# never as a build failure — else a box without `go` grades the same tree far
# lower than one with it.
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


def _go_build(root: Path) -> tuple[bool | None, str]:
    code, _out, err = _run(["go", "build", "./..."], root)
    if code == _NO_BINARY:
        return None, err  # go not installed -> KPI skipped, not a failure
    return code == 0, err


def _go_vet(root: Path) -> tuple[bool | None, list[str]]:
    code, _out, err = _run(["go", "vet", "./..."], root)
    if code == _NO_BINARY:
        return None, []  # go not installed -> KPI skipped, not a failure
    # vet writes diagnostics to stderr; each non-blank, non-progress line is one.
    diags = [ln for ln in err.splitlines()
             if ln.strip() and not ln.startswith(("#", "go: ", "vet: "))
             and ":" in ln]
    if code == 0:
        return True, []
    return False, diags or [err.strip().splitlines()[0] if err.strip() else "vet failed"]


def _gofmt_list(root: Path) -> list[str] | None:
    code, out, _err = _run(["gofmt", "-l", "."], root)
    if code == _NO_BINARY:
        return None  # gofmt not installed -> KPI skipped, not "0 unformatted"
    if code != 0:
        return []
    files: list[str] = []
    for ln in out.splitlines():
        ln = ln.strip().replace("\\", "/")
        if ln.endswith(".go") and not _excluded_go(ln):
            files.append(ln)
    return files


def _dos_review(root: Path, rev_range: str) -> dict[str, Any]:
    code, out, err = _run(["dos", "review", rev_range, "--json"], root, timeout=60)
    if code not in (0, 1):  # 0 clean, 1 residual present — both are valid verdicts
        return {"error": (err or out or "dos review failed").strip()[:200]}
    try:
        return json.loads(out)
    except (json.JSONDecodeError, ValueError):
        return {"error": "dos review emitted non-JSON"}


def collect(workspace: Path, *, run_toolchain: bool = True, run_dos: bool = True,
            dos_range: str = "HEAD~20..HEAD") -> dict[str, Any]:
    root = workspace.resolve()
    if not (root / GOMOD_REL).exists():
        return build_payload(workspace=str(root), kpis=[],
                             error=f"no {GOMOD_REL} at {root} — run from the repo ROOT")
    kpis = gather(root, run_toolchain=run_toolchain, run_dos=run_dos, dos_range=dos_range)
    return build_payload(workspace=str(root), kpis=kpis)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    lines = [
        f"code-quality-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· CODE-DEBT {c.get('code_debt', 0)} · {c.get('soft_signals', 0)} advisory"),
        "",
        "per-KPI (worst first):",
        f"  {'score':>5} {'debt':>4}  kpi            detail",
    ]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['kpi']:<14} {b['detail']}")
    lines.append("")
    lines.append("code-debt work-list:")
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
        lines.append("  (none — zero code-debt)")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    out: list[str] = []
    # Jekyll front matter so the published snapshot carries a <title> + meta
    # description (jekyll-seo-tag reads these) — the SEO/AEO scorecard counts a
    # published page with no title/description as discoverability debt.
    out.append("---")
    out.append('title: "fak code-quality scorecard — the code-debt measuring stick"')
    out.append('description: "fak\'s deterministic code-quality scorecard: ten KPIs folded into a '
               'composite score and the headline code-debt metric, re-derived from disk and Go tooling."')
    out.append("---")
    out.append("")
    out.append("# Code-quality scorecard")
    out.append("")
    if stamp:
        out.append(f"<!-- code-quality-scorecard: {stamp} · process: tools/code_quality_scorecard.py -->")
        out.append("")
    out.append("This is the measuring stick for the code-2x program — the code-side "
               "counterpart of the docs scorecard. Every number below is re-derived "
               "from disk and the Go toolchain by `tools/code_quality_scorecard.py` — no "
               "hand-entry. The headline metric is **code-debt**: the count of concrete, "
               "mechanical defects (an unformatted file, a `go vet` diagnostic, an egregious "
               "god-function, a non-trivial package with zero tests, an untagged honesty "
               "claim, an external dependency, an unwitnessed ship). Driving "
               "code-debt toward zero is what makes \"better code\" provable.")
    out.append("")
    out.append("> Regenerate: `python tools/code_quality_scorecard.py --markdown --stamp DATE > docs/CODE-QUALITY-SCORECARD.md`")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Code-debt (total HARD defects)** | **{c.get('code_debt', 0)}** |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append(f"| Advisory (soft) signals | {c.get('soft_signals', 0)} |")
    out.append("")
    out.append("## Per-KPI")
    out.append("")
    out.append("Ten KPIs, each 0–100. `debt` = units of HARD code-debt in that KPI. "
               "`godoc` is advisory (it scores but emits no hard debt — doc-comment "
               "spam is gaming, not quality).")
    out.append("")
    out.append("| KPI | Score | Debt | Detail |")
    out.append("|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    out.append("## Code-debt work-list")
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
        out.append("No code-debt: every KPI is clean. 🎉")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Code-quality scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true",
                    help="emit the CODE-QUALITY-SCORECARD.md body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--no-toolchain", action="store_true",
                    help="skip go build/vet + gofmt (static-only, fast)")
    ap.add_argument("--no-dos", action="store_true",
                    help="skip the dos review ship-integrity probe")
    ap.add_argument("--range", default="HEAD~20..HEAD",
                    help="git range for the dos ship-integrity KPI")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, run_toolchain=not args.no_toolchain,
                      run_dos=not args.no_dos, dos_range=args.range)

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    else:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
