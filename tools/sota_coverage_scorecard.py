#!/usr/bin/env python3
"""SOTA-coverage scorecard  -  the measuring stick for the prior-art matrix.

The sibling scorecards each grade a surface a reviewer cares about:
``bench_dx_scorecard`` grades the benchmarking developer experience,
``repo_hygiene_scorecard`` grades the shape of the tree, ``code_quality_scorecard``
grades the Go module. None of them grade the one datum that keeps kernel work from
RE-INVENTING known art: the SOTA prior-art matrix in ``internal/sotamatrix``  -  the
single source of truth an agent is supposed to consult BEFORE hand-rolling a kernel
("what is the reference for this op, and should we borrow / bind / stay-minimal?").
A matrix that points at deleted code, carries no link to read, has no verification
oracle, or has a BLIND SPOT (a kernel file no row covers) silently stops doing its
one job  -  and nothing measured it.

This is that number. It cross-checks every matrix row against the tracked tree and
folds the gaps into one ``sota_debt`` integer + an A-F grade. Each defect is one you
fix by ADDING the missing thing (a real file path, a primary link, an oracle, a
covering FileGlobs entry), not by writing prose.

  COMPLETE      -  every row points at code that exists, with a link and an oracle
    fak_path_exists   each row's FakPath names a file that exists on disk (a missing
                      file = the matrix points at deleted code) (HARD, +1 each)
    has_primary_link  each row carries a non-empty http(s):// SOTA link to read
                      (HARD, +1 each)
    has_oracle        each row carries a non-empty verification oracle (the witness
                      that holds a fak implementation honest) (HARD, +1 each)

  HONEST        -  the matrix has no blind spot the tree can fall into
    tree_coverage     every kernel file in the tree is covered by SOME row's
                      FileGlobs (an uncovered kernel file = the "we built a kernel
                      without checking prior art" risk the matrix exists to kill)
                      (HARD, +1 each uncovered file)

  FRESH         -  the matrix has not rotted since it was promoted from a note
    freshness         the matrix's dated provenance note is within the staleness
                      window (SOFT, +1 if past window; needs --today to evaluate)

The headline metric is **sota_debt**: the count of concrete HARD defects above plus
the SOFT freshness flag. Driving it to zero means the prior-art map is complete (no
dangling row) and honest (no blind spot), so an agent who runs ``fak sota <file>``
before touching a kernel always gets a real, current answer.

Deterministic + read-only by construction: it reads the git-tracked tree (so two
clones of the same commit score identically) and edits nothing. Freshness uses an
explicit ``--today YYYY-MM-DD`` (never ``datetime.now()``) so the score is
reproducible. Run from the repo ROOT::

    python tools/sota_coverage_scorecard.py                    # human scorecard
    python tools/sota_coverage_scorecard.py --json             # machine payload
    python tools/sota_coverage_scorecard.py --today 2026-06-30 # evaluate freshness
    python tools/sota_coverage_scorecard.py --check            # exit nonzero on HARD debt
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-sota-coverage-scorecard/1"

# CREATE_NO_WINDOW keeps the synchronous `git` probes below from popping a visible
# console window when this scorecard runs windowless (a scheduled tick spawned via
# pythonw, no console of its own) on Windows; 0 on POSIX (creationflags must be 0
# there). Defined below the imports  -  never injected between import lines (E402).
_CREATE_NO_WINDOW = 0x08000000


def _no_window() -> int:
    """``creationflags`` that suppress a console window on Windows; ``0`` on POSIX."""
    return _CREATE_NO_WINDOW if sys.platform == "win32" else 0


MATRIX_GO = "internal/sotamatrix/sotamatrix.go"

# The kernel-file population the matrix is responsible for covering. These are the
# production compute kernels  -  the actual contractions an agent might re-derive
# from scratch (a quantized GEMM, a fused attention, a quant SIMD lane, a device
# kernel)  -  expressed as git-pathspec globs. Test files (*_test.go) are
# verification, not kernel code, so they are excluded (a row covers a kernel, not
# its test).
#
# Deliberately NOT the whole "internal/compute/*.go" package: that sweeps in
# admission / scheduling / platform glue (capacity, contextsizer, discard_admit,
# anchorwarm, disk_*, hostmem_*, loadpath*, prewarm_admit, resource_cap, the
# *_arch.go capability stubs, the compute.go interface) which are NOT kernels an
# agent learns from a SOTA reference. Counting those as "uncovered kernels" would
# be the scorecard lying about the matrix's blind spot. The set below is the
# genuine kernel surface  -  every *.cu device kernel, the named compute-kernel
# files, and the model-side quant/moe SIMD lanes.
KERNEL_PATHSPECS = [
    # Every CUDA/device kernel is a kernel by definition.
    "internal/compute/*.cu",
    # The named compute-kernel files (GEMM, attention, sparse, prefill, device dispatch).
    "internal/compute/cpuref.go",
    "internal/compute/cuda.go",
    "internal/compute/cuda_kernels.go",
    "internal/compute/dsa.go",
    "internal/compute/dsa_*.go",
    "internal/compute/metal.go",
    "internal/compute/prefill.go",
    "internal/compute/prefill_*.go",
    "internal/compute/graph_cuda.go",
    "internal/compute/tf32_cuda.go",
    "internal/compute/quant_q4k.go",
    # Metal GEMM package + Metal shader sources.
    "internal/metalgemm/*",
    "internal/model/*.metal",
    # Model-side kernels: MoE dispatch, AWQ/GPTQ/EXL2 quant, KV cache, the quant SIMD
    # lanes (amd64/arm64/noasm K-quant + Q4_K/Q6_K  -  the "from scratch" risk surface).
    "internal/model/moe*.go",
    "internal/model/awq*.go",
    "internal/model/gptq*.go",
    "internal/model/exl2*.go",
    "internal/model/kv*.go",
    "internal/model/paging*.go",
    "internal/model/quant_*.go",
]

# How many uncovered files to name in the human/JSON detail before truncating.
UNCOVERED_CAP = 12

# Default staleness window for the matrix's provenance note (days). The matrix was
# promoted from a dated RESEARCH note; past this window without a refresh, the SOTA
# landscape it captured may have moved.
FRESHNESS_WINDOW_DAYS = 90

# Debt-threshold grade bands. sota_debt is the headline integer, so the grade is
# debt-driven (mirrors the debt-first siblings): a complete, honest matrix is an A;
# any HARD defect drops it; a matrix with a wide blind spot fails.
GRADE_BANDS = [(0, "A"), (2, "B"), (5, "C"), (10, "D")]  # else "F"


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent.parent, *here.parents]:
        if (p / "go.mod").exists() and (p / "cmd").is_dir():
            return p
    return here.parent.parent


def _git_lines(args: list[str], root: Path) -> list[str]:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True,
                             text=True, timeout=30, creationflags=_no_window())
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _read(root: Path, rel: str) -> str | None:
    try:
        return (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def kernel_files(root: Path) -> list[str]:
    """The tracked kernel files the matrix must cover, derived from git (never a
    hand-maintained list, so the score can't be gamed by editing the scorecard).
    Excludes *_test.go (verification, not kernel)."""
    files = _git_lines(["ls-files", *KERNEL_PATHSPECS], root)
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        norm = f.replace("\\", "/")
        if norm.endswith("_test.go"):
            continue
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return sorted(out)


# ---------------------------------------------------------------------------
# Parse the Go matrix literal as TEXT (the same way bench_dx parses the catalog
# registry). No Go build: split the `var matrix = []Op{ ... }` body into per-row
# blocks at brace boundaries and pull each row's fields by regex.
# ---------------------------------------------------------------------------

def _matrix_body(src: str) -> str:
    start = src.find("var matrix = []Op{")
    if start < 0:
        return ""
    i = src.find("{", start)
    depth = 0
    for j in range(i, len(src)):
        c = src[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
    return ""


def _split_rows(body: str) -> list[str]:
    """Split the matrix body into per-Op blocks. Each Op is a brace-balanced
    `{ ... }` at depth 1 of the slice literal."""
    rows: list[str] = []
    depth = 0
    buf: list[str] = []
    for c in body:
        if c == "{":
            depth += 1
            if depth == 1:
                buf = []
                continue
        if c == "}":
            depth -= 1
            if depth == 0:
                rows.append("".join(buf))
                continue
        if depth >= 1:
            buf.append(c)
    return rows


def _field(block: str, name: str) -> str:
    """A simple `Name: "value"` string field (first occurrence)."""
    m = re.search(name + r':\s*"((?:[^"\\]|\\.)*)"', block)
    return m.group(1) if m else ""


def _globs(block: str) -> list[str]:
    m = re.search(r'FileGlobs:\s*\[\]string\{([^}]*)\}', block)
    if not m:
        return []
    return re.findall(r'"([^"]+)"', m.group(1))


def _first_fakpath_file(fakpath: str) -> str:
    """The first concrete path in a FakPath string. FakPath is prose like
    ``internal/compute/cpuref.go (Reference, ...) + internal/model/parallel.go``; the
    path is the leading token, terminated by whitespace, ``(``, ``:``, ``;`` or
    ``,``. It is usually a file (``internal/compute/cpuref.go``) but may be a
    directory (``internal/metalgemm/``)  -  both point at real code. A trailing
    ``:LINE`` (e.g. ``cuda.go:1101``) and a trailing ``/`` are stripped."""
    s = fakpath.strip()
    m = re.match(r'(internal/[A-Za-z0-9_./-]+)', s)
    if not m:
        return ""
    token = m.group(1)
    token = re.sub(r':\d+$', "", token)   # drop a `:LINE` suffix
    token = token.rstrip("/")             # a directory pointer (metalgemm/)
    return token


def parse_matrix(src: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in _split_rows(_matrix_body(src)):
        slug = _field(block, "Slug")
        if not slug:
            continue
        rows.append({
            "slug": slug,
            "fak_path": _field(block, "FakPath"),
            "fak_path_file": _first_fakpath_file(_field(block, "FakPath")),
            "primary_link": _field(block, "PrimaryLink"),
            "oracle": _field(block, "Oracle"),
            "file_globs": _globs(block),
        })
    return rows


# ---------------------------------------------------------------------------
# Glob matching: a kernel file is covered if it matches at least one row's
# FileGlobs (fnmatch over normalized forward-slash separators).
# ---------------------------------------------------------------------------

def _glob_match(path: str, glob: str) -> bool:
    return fnmatch.fnmatch(path.replace("\\", "/"), glob.replace("\\", "/"))


def covered_by_matrix(path: str, all_globs: list[str]) -> bool:
    return any(_glob_match(path, g) for g in all_globs)


# ---------------------------------------------------------------------------
# Freshness: pull the dated provenance note the package doc cites
# (RESEARCH-backend-sota-matrix-YYYY-MM-DD.md) and flag if older than the window.
# ---------------------------------------------------------------------------

_PROVENANCE_RE = re.compile(r'RESEARCH-[A-Za-z0-9-]*?(\d{4})-(\d{2})-(\d{2})')


def provenance_date(src: str) -> str:
    m = _PROVENANCE_RE.search(src)
    if not m:
        return ""
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def _days_between(older: str, newer: str) -> int | None:
    """Whole days from ``older`` to ``newer`` (both YYYY-MM-DD), or None if either
    is unparseable. Uses date.fromisoformat  -  deterministic, no clock read."""
    from datetime import date
    try:
        a = date.fromisoformat(older)
        b = date.fromisoformat(newer)
    except ValueError:
        return None
    return (b - a).days


# ---------------------------------------------------------------------------
# KPIs. Each returns (passed, debt, detail, items). debt is the count of concrete
# defects (0 when passed). items is the list of offenders (named in output).
# ---------------------------------------------------------------------------

def kpi_fak_path_exists(root: Path, rows: list[dict[str, Any]]) -> tuple[bool, int, str, list[str]]:
    missing: list[str] = []
    for r in rows:
        f = r["fak_path_file"]
        if not f:
            missing.append(f"{r['slug']}: FakPath has no parseable path")
            continue
        if not (root / f).exists():
            missing.append(f"{r['slug']}: {f}")
    if not missing:
        return True, 0, f"all {len(rows)} rows point at code that exists", []
    return False, len(missing), f"{len(missing)} row(s) point at a missing path", missing


def kpi_has_primary_link(rows: list[dict[str, Any]]) -> tuple[bool, int, str, list[str]]:
    bad: list[str] = []
    for r in rows:
        link = r["primary_link"].strip()
        if not re.match(r'https?://\S+', link):
            bad.append(f"{r['slug']}: PrimaryLink={link or '(empty)'}")
    if not bad:
        return True, 0, f"all {len(rows)} rows carry an http(s) SOTA link", []
    return False, len(bad), f"{len(bad)} row(s) have no http(s) PrimaryLink", bad


def kpi_has_oracle(rows: list[dict[str, Any]]) -> tuple[bool, int, str, list[str]]:
    bad: list[str] = []
    for r in rows:
        if not r["oracle"].strip():
            bad.append(f"{r['slug']}: Oracle is empty")
    if not bad:
        return True, 0, f"all {len(rows)} rows carry a verification oracle", []
    return False, len(bad), f"{len(bad)} row(s) have no Oracle", bad


def kpi_tree_coverage(root: Path, rows: list[dict[str, Any]]) -> tuple[bool, int, str, list[str]]:
    all_globs: list[str] = []
    for r in rows:
        all_globs.extend(r["file_globs"])
    files = kernel_files(root)
    uncovered = [f for f in files if not covered_by_matrix(f, all_globs)]
    if not files:
        return False, 1, "no kernel files found (cannot evaluate coverage)", []
    if not uncovered:
        return True, 0, f"all {len(files)} kernel files are covered by some row", []
    detail = (f"{len(uncovered)}/{len(files)} kernel files are uncovered by any row "
              f"(matrix blind spot)")
    return False, len(uncovered), detail, uncovered


def kpi_freshness(src: str, today: str) -> tuple[bool, int, str, list[str]]:
    pdate = provenance_date(src)
    if not pdate:
        return True, 0, "no dated provenance note found (freshness not applicable)", []
    if not today:
        return True, 0, f"provenance dated {pdate}; pass --today to evaluate staleness", []
    days = _days_between(pdate, today)
    if days is None:
        return True, 0, f"provenance {pdate} / today {today}: unparseable", []
    if days <= FRESHNESS_WINDOW_DAYS:
        return True, 0, f"matrix provenance {pdate} is {days}d old (<= {FRESHNESS_WINDOW_DAYS}d window)", []
    return (False, 1,
            f"matrix provenance {pdate} is {days}d old (> {FRESHNESS_WINDOW_DAYS}d window; re-check SOTA)",
            [f"provenance {pdate} is {days}d stale"])


KPI_GROUP = {
    "fak_path_exists": "complete",
    "has_primary_link": "complete",
    "has_oracle": "complete",
    "tree_coverage": "honest",
    "freshness": "fresh",
}
# HARD KPIs contribute to the --check gate; SOFT ones score but don't fail CI.
HARD_KPIS = {"fak_path_exists", "has_primary_link", "has_oracle", "tree_coverage"}


def gather(root: Path, today: str) -> list[dict[str, Any]]:
    src = _read(root, MATRIX_GO) or ""
    rows = parse_matrix(src)
    results = [
        ("fak_path_exists", *kpi_fak_path_exists(root, rows)),
        ("has_primary_link", *kpi_has_primary_link(rows)),
        ("has_oracle", *kpi_has_oracle(rows)),
        ("tree_coverage", *kpi_tree_coverage(root, rows)),
        ("freshness", *kpi_freshness(src, today)),
    ]
    kpis: list[dict[str, Any]] = []
    for name, passed, debt, detail, items in results:
        kpis.append({
            "name": name,
            "group": KPI_GROUP[name],
            "hard": name in HARD_KPIS,
            "passed": bool(passed),
            "debt": int(debt),
            "detail": detail,
            "items": items[:UNCOVERED_CAP],
            "items_total": len(items),
        })
    return kpis


def grade_letter(debt: int) -> str:
    for threshold, letter in GRADE_BANDS:
        if debt <= threshold:
            return letter
    return "F"


def build_payload(workspace: str, rows: int, kpis: list[dict[str, Any]], error: str = "") -> dict[str, Any]:
    debt = sum(k["debt"] for k in kpis)
    hard_debt = sum(k["debt"] for k in kpis if k["hard"])
    debt_by_group = {"complete": 0, "honest": 0, "fresh": 0}
    for k in kpis:
        debt_by_group[k["group"]] += k["debt"]
    return {
        "schema": SCHEMA,
        "workspace": workspace,
        "error": error,
        "ok": (not error) and hard_debt == 0,
        "corpus": {
            "matrix_rows": rows,
            "sota_debt": debt,
            "hard_debt": hard_debt,
            "grade": grade_letter(debt),
            "debt_by_group": debt_by_group,
        },
        "kpis": kpis,
    }


def collect(workspace: Path, today: str = "") -> dict[str, Any]:
    root = workspace.resolve()
    if not (root / "go.mod").exists():
        return build_payload(workspace=str(root), rows=0, kpis=[],
                             error=f"not the fak repo root at {root} (no go.mod)")
    src = _read(root, MATRIX_GO)
    if src is None:
        return build_payload(workspace=str(root), rows=0, kpis=[],
                             error=f"matrix source {MATRIX_GO} is missing")
    kpis = gather(root, today)
    rows = len(parse_matrix(src))
    return build_payload(workspace=str(root), rows=rows, kpis=kpis)


def render(payload: dict[str, Any]) -> str:
    if payload.get("error"):
        return f"error: {payload['error']}"
    c = payload["corpus"]
    lines = [
        "SOTA-coverage scorecard  -  the prior-art matrix, complete + honest",
        f"  grade {c['grade']}   sota-debt {c['sota_debt']} (hard {c['hard_debt']})   "
        f"rows {c['matrix_rows']}",
        "",
        "  by group:",
        f"    complete (rows point at real code, link, oracle) debt {c['debt_by_group']['complete']}",
        f"    honest   (no blind spot in the tree)             debt {c['debt_by_group']['honest']}",
        f"    fresh    (provenance within window)              debt {c['debt_by_group']['fresh']}",
        "",
        "  KPIs (X = a defect to retire by ADDING the missing thing):",
    ]
    for k in payload["kpis"]:
        mark = "ok" if k["passed"] else "X"
        tag = "" if k["hard"] else " (soft)"
        extra = "" if k["passed"] else f"  [+{k['debt']}]"
        lines.append(f"    {mark} {k['name']:<18}{tag} {k['detail']}{extra}")
        if not k["passed"] and k["items"]:
            for it in k["items"]:
                lines.append(f"        - {it}")
            if k["items_total"] > len(k["items"]):
                lines.append(f"        ... and {k['items_total'] - len(k['items'])} more")
    if c["hard_debt"] == 0:
        lines.append("\n  No HARD sota-debt: every matrix row points at real code with a "
                     "link + oracle, and no kernel file is a blind spot.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SOTA-coverage scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--today", default="", metavar="YYYY-MM-DD",
                    help="evaluate freshness against this date (deterministic; no clock read)")
    ap.add_argument("--check", action="store_true",
                    help="exit nonzero when there is any HARD sota-debt (CI gate)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, today=args.today)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))

    if args.check:
        return 0 if payload.get("ok") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
