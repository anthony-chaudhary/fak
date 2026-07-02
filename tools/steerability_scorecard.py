#!/usr/bin/env python3
"""Steerability scorecard — the measuring stick for a project that stays as
steerable as it grows.

The sibling scorecards each measure the ABSOLUTE quality of one surface and report
it as a *count*: ``repo_hygiene`` counts duplicate docs, ``code_quality`` counts
god-files and untested packages, ``stability`` counts missing regression gates.
Every one of those headline numbers mechanically WORSENS as the repo grows — a
3×-larger tree has ~3× the surfaces, so the raw defect count climbs even when the
discipline is unchanged. None of them answers the question this scorecard exists for:

    As the repo doubles in size, does the EFFORT to steer / change / navigate it
    stay roughly FLAT? And if it drifts off track, do we KNOW, and can we CORRECT?

That is steerability, and it is a property of *shape*, not *size*. A repo can carry
zero hygiene-debt and still become unsteerable — every module balloons, one package
becomes a chokepoint every change must route through, and a single edit ripples
across the tree.

This is that number. The DEFINING design constraint, different from every sibling:
**every KPI is GROWTH-INVARIANT — a ratio, a density, or a distribution percentile,
never a raw count.** A 2×-larger repo with the same modular discipline scores
*identically*. A count-based KPI (the kind every sibling uses) would be WRONG here:
it trends just from getting bigger. So the headline is a 0–100 **steerability
index** — the weighted composite of the invariant KPIs — not a debt pile.

The scorecard still emits a ``steerability_debt`` integer so it can join the unified
control pane (``scorecard_control_pane.py``), but a *defect* is emitted only when an
invariant rate crosses a FIXED threshold — the score is the rate, the debt is the
count of threshold crossings. And it stays ORTHOGONAL to ``code_quality``: god-files
and tests are already that scorecard's HARD debt, so re-emitting them here would
double-count the same monolith in the shared portfolio sum. Steerability therefore
SCORES god-files/functions on the invariant rate (SOFT, advisory) and leaves the
raw count to ``code_quality``. The only HARD (debt-emitting) KPIs are the two whose
cheapest fix is genuinely real work, not gaming.

Twelve KPIs in four groups (the four faces of "stays steerable as it grows"):

  MODULARITY    — does each unit stay small as the whole grows
    file_size_dist   p90 file length vs a fixed reference          (SOFT)
    func_size_dist   p90 function length (literal-aware scan)      (SOFT)
    god_file_rate    rate of files over the hard ceiling           (SOFT — code_quality owns the count)
    god_func_rate    rate of functions over the hard ceiling       (SOFT — code_quality owns the count)

  COUPLING      — does one change ripple everywhere (blast radius)
    fan_in_gini      Gini of the internal-import fan-in graph       (SOFT)
    hub_share        the single most-depended-on package's share    (SOFT)
    dispatch_god_file a cmd/* dispatch file over the hard ceiling   (HARD)

  NAVIGABILITY  — can you still find the lever to pull
    package_doc_frac fraction of packages with a package/command doc-comment (SOFT)

  CORRECTION    — can drift be SEEN and REVERTED at constant cost
    ratchet_present  the control-pane baseline + this scorecard are wired (HARD)
    worst_pkg_drift  worst package's growth vs the pinned baseline   (SOFT)
    churn_concentration  Gini of recent commit churn (HEAD-relative) (SOFT)

The HARD set — ``dispatch_god_file`` + ``ratchet_present`` — is **0 debt on a
disciplined tree**: there is no cmd dispatch god-file and the baseline is pinned. So
the live-tree floor is zero debt, and the scorecard's signal lives in the *index* and
the SOFT drift signals, exactly as intended. ``churn_concentration`` is the one
HEAD-relative KPI (like ``code_quality``'s ``ship_integrity``): it reads recent git
history, so its number moves as commits land even when the tree is byte-identical —
pin ``--range`` for a stable read. It scores but can never anchor the baseline.

The growth-invariant KPI shape is the lesson this scorecard contributes back to the
``scorecard`` doctrine: score on an invariant rate, emit debt only on a fixed-
threshold crossing, and DROP any signal that is secretly size-coupled (files-per-
package spread, mean LOC-per-package — both widen from growth alone).

Deterministic + read-only by construction: the tree KPIs read the working tree (two
clones of one commit score identically); the one history KPI is documented as the
HEAD-relative exception. It edits nothing but the generated snapshot under
``--markdown``. Run from the repo ROOT::

    python tools/steerability_scorecard.py                 # human scorecard
    python tools/steerability_scorecard.py --json          # machine payload
    python tools/steerability_scorecard.py --markdown      # the committed snapshot body
    python tools/steerability_scorecard.py --compare base.json   # prove the index moved
    python tools/steerability_scorecard.py --range HEAD~40..HEAD # pin the churn window

The companion process is the ``/steerability-score`` skill: drive the index UP and
the SOFT drift signals DOWN by adding real modularity (split a monolith along a seam,
break a coupling hub), never by gaming a detector.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
from pathlib import Path
from typing import Any
install_no_window_subprocess_defaults(subprocess)

SCHEMA = "fak-steerability-scorecard/1"
GENERATED_SNAPSHOT = "docs/STEERABILITY-SCORECARD.md"
BASELINE_REL = "tools/scorecard_baseline.json"

# ---------------------------------------------------------------------------
# Single-source reuse: the literal-aware Go scanner + the size ceilings live in
# code_quality_scorecard (tested there). Import them so this scorecard's
# size-distribution KPIs measure exactly what code_quality's architecture KPI
# does — no second, drifting implementation. Each import falls back to a small
# inline copy so the tool still stands alone if shipped without its sibling.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from code_quality_scorecard import (  # the tested, literal-aware scanner + ceilings
        FILE_HARD_MAX, FUNC_HARD_MAX, list_go_files, scan_go_file, _safe_read,
    )
except Exception:  # noqa: BLE001 — stand-alone fallback
    FILE_HARD_MAX = 1500
    FUNC_HARD_MAX = 200
    # `.dos`/`.fak`/`.claude`/`.tmp` hold full repo CHECKOUTS / copies the agent machinery
    # leaves behind (`.dos/_dos_park/_iso_build/`, `.fak/tmp/`, `.claude/worktrees/`,
    # `.tmp/pin-check/` + `.tmp/prplan-check/`); walking them
    # double-grades every copied .go as phantom debt — a `.dos` tree once added 2357 phantom
    # .go to this corpus. Exclude them, matching the code-slop/code-quality scorecards.
    GO_EXCLUDE_DIRS = {".git", ".dos", ".fak", ".claude", ".tmp", "node_modules", "testdata", "vendor", "__pycache__"}

    def _safe_read(path: Path) -> str:  # type: ignore[misc]
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def list_go_files(root: Path, *, tests: bool) -> list[str]:  # type: ignore[misc]
        out: list[str] = []
        for p in root.rglob("*.go"):
            rel = p.relative_to(root).as_posix()
            # A Private-Use-Area (U+E000–U+F8FF) or control char in the path is a
            # cp1252<->utf-8 mojibake phantom (e.g. a faulted absolute path landing as a
            # U+F05C-named dir holding a repo copy), never tracked source.
            if any(0xE000 <= ord(c) <= 0xF8FF or ord(c) < 0x20 for c in rel):
                continue
            if set(Path(rel).parts) & GO_EXCLUDE_DIRS:
                continue
            if rel.endswith("_test.go") == tests:
                out.append(rel)
        return sorted(out)

    def scan_go_file(text: str) -> dict[str, Any]:  # type: ignore[misc]
        # Minimal fallback: line count only; long_funcs left empty so func_size
        # degrades gracefully rather than fabricating a number.
        lines = text.splitlines()
        return {"n_lines": len(lines), "n_funcs": 0, "long_funcs": [], "exported": []}

# The module path prefix is the WHOLE internal-vs-external filter for the import
# graph — read it from go.mod so a fork that renames the module still works.
DEFAULT_MODULE = "github.com/anthony-chaudhary/fak"

# ---------------------------------------------------------------------------
# Calibration. Each threshold is a deliberate, growth-invariant reference with a
# stated reason. The score-from-rate references are generous; the HARD ceilings
# are the same egregious-outlier lines code_quality uses.
# ---------------------------------------------------------------------------
P90_FILE_REF = 520        # p90 file length at/under this is a clean modularity floor
P90_FILE_SLOPE = 10       # points lost per line of p90 over the reference
FUNC_LONG_RATE_REF = 0.05  # 5% of functions over the soft length line is the score floor
GINI_GREEN = 0.60         # fan-in Gini at/under this is a flat, steerable graph
GINI_RED = 0.90           # fan-in Gini at/over this is a hub-dominated graph (score floor)
HUB_SHARE_GREEN = 0.30    # a single package depended on by <= 30% of packages is fine
HUB_SHARE_RED = 0.70      # >= 70% is a chokepoint every change routes through
GOD_FILE_RATE_REF = 0.02  # 2% of files over the ceiling is the score floor anchor
GOD_FUNC_RATE_REF = 0.01  # 1% of functions over the ceiling is the score floor anchor
DRIFT_WARN_PCT = 50.0     # a package growing > 50% vs baseline is a SOFT drift nudge
CHURN_RANGE_DEFAULT = "HEAD~40..HEAD"   # the recent-commit window for churn (HEAD-relative)
SAMPLE_CAP = 25           # cap on listed soft signals

GROUPS = ("modularity", "coupling", "navigability", "correction")

KPI_WEIGHTS: dict[str, float] = {
    # modularity (0.34) — the core "each unit stays small" axis
    "file_size_dist": 0.10,
    "func_size_dist": 0.10,
    "god_file_rate": 0.07,
    "god_func_rate": 0.07,
    # coupling (0.34) — blast radius is the heart of (un)steerability
    "fan_in_gini": 0.14,
    "hub_share": 0.12,
    "dispatch_god_file": 0.08,
    # navigability (0.10)
    "package_doc_frac": 0.10,
    # correction (0.22) — can drift be seen + reverted
    "ratchet_present": 0.10,
    "worst_pkg_drift": 0.06,
    "churn_concentration": 0.06,
}
KPI_GROUP: dict[str, str] = {
    "file_size_dist": "modularity", "func_size_dist": "modularity",
    "god_file_rate": "modularity", "god_func_rate": "modularity",
    "fan_in_gini": "coupling", "hub_share": "coupling",
    "dispatch_god_file": "coupling",
    "package_doc_frac": "navigability",
    "ratchet_present": "correction", "worst_pkg_drift": "correction",
    "churn_concentration": "correction",
}

# An internal-package import line, in any of the forms present in the tree:
#   "github.com/anthony-chaudhary/fak/internal/abi"
#   fakmodel "github.com/anthony-chaudhary/fak/internal/model"   (alias)
#   _ "github.com/anthony-chaudhary/fak/internal/registrations"  (blank — excluded)
# We key on the quoted path (ignoring any leading alias/`_`/`.`), then handle the
# blank-import case separately by inspecting the leading token.
def _import_line_re(module: str) -> "re.Pattern[str]":
    esc = re.escape(module)
    return re.compile(
        r'^\s*(?P<lead>[\w.]+\s+)?"(?P<path>' + esc + r'/(?:internal|pkg|cmd)/[^"]+)"')

_PACKAGE_DOC_RE = re.compile(r"^//\s*Package\s+\w", re.MULTILINE)
_COMMAND_DOC_RE = re.compile(r"^//\s*Command\s+[\w.-]+", re.MULTILINE)
_PACKAGE_LINE_RE = re.compile(r"^\s*package\s+(\w+)\b", re.MULTILINE)


# ---------------------------------------------------------------------------
# Small pure helpers (the testable core).
# ---------------------------------------------------------------------------

def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


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


def percentile(values: list[int], q: float) -> float:
    """The q-percentile (0..1) of `values` by nearest-rank — deterministic, no
    interpolation surprises. Empty -> 0.0. A percentile is scale-free: doubling
    the sample at the same shape returns the same number (the growth-invariance
    backbone of this scorecard)."""
    if not values:
        return 0.0
    s = sorted(values)
    if q <= 0:
        return float(s[0])
    if q >= 1:
        return float(s[-1])
    # nearest-rank: rank = ceil(q*n), 1-indexed
    rank = max(1, min(len(s), int(-(-len(s) * q // 1))))
    return float(s[rank - 1])


def gini(values: list[float]) -> float:
    """Gini coefficient of a non-negative distribution (0 = perfectly flat, ->1 =
    all mass on one node). Scale-free in value (multiplying every weight by a
    constant leaves it unchanged), which is why it suits a coupling KPI. Note the
    small-N upward bias — this is reported advisory (SOFT), never gated on an
    absolute threshold. Empty / all-zero -> 0.0."""
    xs = sorted(float(v) for v in values if v >= 0)
    n = len(xs)
    if n == 0:
        return 0.0
    total = sum(xs)
    if total <= 0:
        return 0.0
    # G = (2 * sum(i*x_i) / (n * sum(x))) - (n + 1) / n   for i = 1..n (sorted asc)
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def parse_internal_imports(text: str, module: str) -> list[str]:
    """The non-blank internal/pkg/cmd packages a Go file imports, as leaf names.
    Scans only WITHIN import spans (a single `import "x"` line or the body of an
    `import ( ... )` block) so a path inside a comment or string is never matched.
    A blank import (`_ "..."`) is a compile-time dependency but NOT a steerability
    coupling edge (the importer uses none of the imported API), so it is excluded
    — counting it would inflate registry hubs. Alias and dot imports ARE edges
    (the importer uses the package) and are kept. Returns leaf package names
    (e.g. 'abi', 'model'), de-duplicated."""
    line_re = _import_line_re(module)
    out: set[str] = set()
    lines = text.splitlines()
    i, n = 0, len(lines)
    in_block = False
    while i < n:
        raw = lines[i]
        stripped = raw.strip()
        if not in_block:
            if stripped.startswith("import ("):
                in_block = True
                i += 1
                continue
            if stripped.startswith("import "):
                _collect_import(stripped[len("import "):].strip(), line_re, out, module)
            i += 1
            continue
        # inside an import ( ... ) block
        if stripped.startswith(")"):
            in_block = False
            i += 1
            continue
        _collect_import(stripped, line_re, out, module)
        i += 1
    return sorted(out)


def _collect_import(body: str, line_re: "re.Pattern[str]", out: set[str], module: str) -> None:
    """Add the leaf package from one import entry to `out`, unless it is a blank
    (`_`) import. `body` is the import entry with `import (` already stripped."""
    # a blank import: the leading token is exactly `_`
    lead = body.split('"', 1)[0].strip()
    if lead == "_":
        return
    m = line_re.match("    " + body)  # re-anchor with leading space the pattern allows
    if not m:
        m = line_re.match(body)
    if not m:
        return
    path = m.group("path")
    # path is module + /internal|pkg|cmd/<leaf>[/...] — the leaf is the first
    # segment after the kind directory.
    tail = path[len(module):].lstrip("/")          # internal/abi/sub -> internal/abi/sub
    parts = tail.split("/")
    if len(parts) >= 2:
        out.add(parts[1])


def package_of(rel: str) -> str:
    """The package directory of a repo-relative .go path (its parent dir)."""
    return rel.rsplit("/", 1)[0] if "/" in rel else ""


def package_leaf(pkg_dir: str) -> str:
    """The leaf name of a package directory (internal/abi -> abi)."""
    return pkg_dir.rsplit("/", 1)[-1] if "/" in pkg_dir else pkg_dir


def package_name(text: str) -> str:
    m = _PACKAGE_LINE_RE.search(text)
    return m.group(1) if m else ""


def has_orientation_doc(text: str) -> bool:
    """True when a file carries the orientation doc a reader sees first.

    Libraries use the Go package-doc convention (`// Package x ...`). Command
    directories are `package main`; for those, Go's useful public form is
    `// Command name ...`, so counting only `Package` silently under-reports
    documented command surfaces.
    """
    if _PACKAGE_DOC_RE.search(text):
        return True
    return package_name(text) == "main" and bool(_COMMAND_DOC_RE.search(text))


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each returns
#   {kpi, group, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of steerability-debt; soft = score-only judgment nudges.
# ---------------------------------------------------------------------------

def kpi_file_size_dist(n_lines_list: list[int]) -> dict[str, Any]:
    """SOFT. The p90 file length — does the typical file stay small as the tree
    grows? Scored from the percentile (scale-free), never a count, so 2× the files
    at the same shape scores identically. Emits no debt: the cheap move (a cosmetic
    split with no real seam) games a count but barely moves a p90 over hundreds of
    files, and the raw god-file COUNT is code_quality's job."""
    p90 = percentile(n_lines_list, 0.90)
    p50 = percentile(n_lines_list, 0.50)
    over = max(0.0, p90 - P90_FILE_REF)
    score = _clamp(100 - over / P90_FILE_SLOPE)
    return {"kpi": "file_size_dist", "group": "modularity", "score": score,
            "detail": f"file length p50={p50:.0f} p90={p90:.0f} (ref {P90_FILE_REF}) over {len(n_lines_list)} files",
            "defects": [], "soft": ([f"file-length p90 {p90:.0f} > ref {P90_FILE_REF} — "
                                     f"the typical file is drifting large"] if over > 0 else [])}


def kpi_func_size_dist(n_long: int, n_total: int) -> dict[str, Any]:
    """SOFT. The fraction of functions over the SOFT length line (a "long" function,
    > the code_quality soft ceiling) — a genuine growth-invariant RATE, honest about
    what the literal-aware scan can see (it reports lengths only past the soft line).
    A fabricated percentile over a long-tail-only sample would read p90≈0; this rate
    is the real "are functions staying small" signal. Scored from the rate; no debt
    (the hard count is code_quality's). `n_long` = funcs over the soft line,
    `n_total` = all functions scanned."""
    rate = n_long / max(1, n_total)
    # a healthy tree keeps long functions rare; the reference rate is the score floor.
    score = _clamp(100 - 100 * rate / FUNC_LONG_RATE_REF) if rate > 0 else 100
    return {"kpi": "func_size_dist", "group": "modularity", "score": score,
            "detail": (f"{n_long}/{n_total} functions over the soft length line (rate {rate:.2%})"
                       if n_total else "no functions scanned"),
            "defects": [], "soft": ([f"{rate:.1%} of functions are long (> soft line) — "
                                     f"function size is drifting up"] if rate > FUNC_LONG_RATE_REF else [])}


def kpi_god_file_rate(n_god: int, n_total: int) -> dict[str, Any]:
    """SOFT (orthogonal to code_quality, which owns the COUNT). The RATE of files
    over the hard ceiling — growth-invariant, so it stays flat if discipline holds
    as the tree grows. Scored from the rate; emits no debt here to avoid double-
    counting the same monolith in the portfolio sum."""
    rate = n_god / max(1, n_total)
    score = _clamp(100 - 100 * rate / GOD_FILE_RATE_REF) if rate > 0 else 100
    return {"kpi": "god_file_rate", "group": "modularity", "score": score,
            "detail": f"{n_god}/{n_total} files > {FILE_HARD_MAX} lines (rate {rate:.2%})",
            "defects": [],
            "soft": ([f"{n_god} god-file(s) (rate {rate:.2%}) — code_quality.architecture "
                      f"owns the count; split along seams (/modularize)"] if n_god else [])}


def kpi_god_func_rate(n_god: int, n_total: int) -> dict[str, Any]:
    """SOFT (orthogonal to code_quality). The RATE of functions over the ceiling."""
    rate = n_god / max(1, n_total)
    score = _clamp(100 - 100 * rate / GOD_FUNC_RATE_REF) if rate > 0 else 100
    return {"kpi": "god_func_rate", "group": "modularity", "score": score,
            "detail": f"{n_god}/{n_total} functions > {FUNC_HARD_MAX} lines (rate {rate:.2%})",
            "defects": [],
            "soft": ([f"{n_god} god-function(s) (rate {rate:.2%}) — code_quality owns the count"]
                     if n_god else [])}


def kpi_fan_in_gini(fan_in: dict[str, int]) -> dict[str, Any]:
    """SOFT. The Gini of the per-package internal-import fan-in distribution: a flat
    graph (everyone depends on a few, evenly) is steerable; a hub-dominated graph
    (one package every change must route through) is not. Gini is scale-free in
    value but small-N biased, so this scores and reports — it never gates on an
    absolute threshold."""
    g = gini([float(v) for v in fan_in.values()])
    span = max(1e-9, GINI_RED - GINI_GREEN)
    score = _clamp(100 - 100 * (g - GINI_GREEN) / span)
    top = sorted(fan_in.items(), key=lambda kv: -kv[1])[:5]
    top_s = ", ".join(f"{k}:{v}" for k, v in top)
    return {"kpi": "fan_in_gini", "group": "coupling", "score": score,
            "detail": f"fan-in Gini {g:.2f} over {len(fan_in)} packages (top {top_s})",
            "defects": [],
            "soft": ([f"fan-in Gini {g:.2f} > green {GINI_GREEN:.2f} — coupling concentrates "
                      f"on a few hubs ({top_s})"] if g > GINI_GREEN else [])}


def kpi_hub_share(fan_in: dict[str, int], n_packages: int) -> dict[str, Any]:
    """SOFT. The single most-depended-on package's share of all packages: the
    chokepoint risk. Reported as a fraction (scale-free in form). Note it is
    DENOMINATOR-coupled — as the repo grows, a stable hub's share falls "for free"
    — so it is advisory only, a snapshot of the current blast radius."""
    if not fan_in or n_packages <= 0:
        return {"kpi": "hub_share", "group": "coupling", "score": 100,
                "detail": "no internal coupling edges", "defects": [], "soft": []}
    hub, importers = max(fan_in.items(), key=lambda kv: kv[1])
    share = importers / n_packages
    span = max(1e-9, HUB_SHARE_RED - HUB_SHARE_GREEN)
    score = _clamp(100 - 100 * (share - HUB_SHARE_GREEN) / span)
    return {"kpi": "hub_share", "group": "coupling", "score": score,
            "detail": f"top hub '{hub}' imported by {importers}/{n_packages} packages ({share:.0%})",
            "defects": [],
            "soft": ([f"'{hub}' is a chokepoint: imported by {share:.0%} of packages — "
                      f"a change to it ripples wide"] if share > HUB_SHARE_GREEN else [])}


def kpi_dispatch_god_file(cmd_god_files: list[tuple[str, int]]) -> dict[str, Any]:
    """HARD. A `cmd/*` dispatch file over the hard ceiling is the steerability
    failure that count-orthogonality does NOT cover elsewhere: the CLI dispatch
    surface is where every verb is wired, so a monolithic dispatch file means every
    new lever fights the same god-file. Each is one unit of debt — fixed by SPLITTING
    (the /modularize move), which is real steerability work, not gaming. `cmd_god_files`
    is the list of (path, n_lines) for cmd/* files over the ceiling."""
    defects = [f"cmd dispatch god-file {p} ({n} lines > {FILE_HARD_MAX}) — split the verb "
               f"table so a new command doesn't fight a monolith" for p, n in sorted(cmd_god_files)]
    return {"kpi": "dispatch_god_file", "group": "coupling",
            "score": _clamp(100 - 20 * len(defects)),
            "detail": (f"{len(defects)} cmd dispatch god-file(s)" if defects
                       else "no cmd dispatch god-file"),
            "defects": defects, "soft": []}


def kpi_package_doc_frac(n_documented: int, n_packages: int) -> dict[str, Any]:
    """SOFT. Fraction of packages carrying an orientation doc-comment: `// Package
    x ...` for libraries or `// Command x ...` for `package main` command surfaces.
    That is the first thing a reader (or an agent) reads to orient in a package.
    Scored as a fraction (scale-free), but NEVER debt: the cheap move is
    `// Package x provides x.` spam, which games the metric without aiding
    navigation (the godoc lesson)."""
    frac = n_documented / max(1, n_packages)
    return {"kpi": "package_doc_frac", "group": "navigability",
            "score": _clamp(100 * frac),
            "detail": f"{n_documented}/{n_packages} packages carry an orientation doc-comment ({frac:.0%})",
            "defects": [],
            "soft": ([f"{n_packages - n_documented} package(s) without a doc-comment header — "
                      f"a reader/agent has no one-line orientation"] if n_documented < n_packages else [])}


def kpi_ratchet_present(baseline: dict[str, Any] | None, wired: bool) -> dict[str, Any]:
    """HARD. Can drift be SEEN and REVERTED at constant cost? That requires the
    control-pane ratchet to actually exist: a baseline that PARSES, carries a
    `metrics` map and an int `total_debt`, AND has this scorecard wired into the
    fold. A bare file-exists check would be gameable by `touch`; this is substantive.
    One defect if the ratchet is absent or malformed — fixed by committing a real
    baseline + wiring, which is the genuine correction affordance."""
    problems: list[str] = []
    if not isinstance(baseline, dict):
        problems.append("control-pane baseline missing or unparseable "
                        f"({BASELINE_REL}) — `python tools/scorecard_control_pane.py --pin`")
    else:
        if not isinstance(baseline.get("metrics"), dict):
            problems.append(f"baseline has no `metrics` map ({BASELINE_REL})")
        td = baseline.get("total_debt")
        if not isinstance(td, int) or isinstance(td, bool):
            problems.append(f"baseline `total_debt` is not an int ({BASELINE_REL})")
        if isinstance(baseline.get("metrics"), dict) and not wired:
            problems.append("steerability not wired into the control-pane SCORECARDS fold "
                            "(scorecard_control_pane.py)")
    return {"kpi": "ratchet_present", "group": "correction",
            "score": 100 if not problems else _clamp(100 - 50 * len(problems)),
            "detail": ("control-pane ratchet present + this scorecard wired" if not problems
                       else f"{len(problems)} ratchet gap(s)"),
            "defects": problems, "soft": []}


def kpi_worst_pkg_drift(pkg_loc: dict[str, int], baseline_pkg_loc: dict[str, int]
                        ) -> dict[str, Any]:
    """SOFT. The worst package's LOC growth vs the pinned baseline — the "is a
    package quietly ballooning" lens. Advisory only: a re-pin blesses the drift, so
    making this HARD would punish every legitimate refactor until someone re-pins.
    `baseline_pkg_loc` empty (unpinned) -> a clean, informational read."""
    if not baseline_pkg_loc:
        return {"kpi": "worst_pkg_drift", "group": "correction", "score": 100,
                "detail": "no package-LOC baseline pinned (informational)",
                "defects": [], "soft": []}
    drifts: list[tuple[str, float, int, int]] = []
    for pkg, loc in pkg_loc.items():
        base = baseline_pkg_loc.get(pkg)
        if base and base > 0:
            pct = 100.0 * (loc - base) / base
            if pct > DRIFT_WARN_PCT:
                drifts.append((pkg, pct, base, loc))
    drifts.sort(key=lambda d: -d[1])
    worst_pct = drifts[0][1] if drifts else 0.0
    score = _clamp(100 - max(0.0, worst_pct - DRIFT_WARN_PCT) / 5.0)
    soft = [f"package '{pkg}' grew {pct:.0f}% vs baseline ({base}->{loc} LOC)"
            for pkg, pct, base, loc in drifts[:SAMPLE_CAP]]
    return {"kpi": "worst_pkg_drift", "group": "correction", "score": score,
            "detail": (f"{len(drifts)} package(s) over +{DRIFT_WARN_PCT:.0f}% vs baseline "
                       f"(worst +{worst_pct:.0f}%)" if drifts else "no package over the drift line"),
            "defects": [], "soft": soft}


def kpi_churn_concentration(churn: dict[str, int], rng: str, available: bool) -> dict[str, Any]:
    """SOFT, HEAD-RELATIVE. The Gini of recent-commit churn (lines changed per file
    over `--range`): is a small file-set absorbing most change — a hot spot the
    fleet keeps re-touching? This reads git HISTORY, so its number moves as commits
    land even on a byte-identical tree (the `ship_integrity` precedent). It scores
    but can NEVER anchor the baseline. Unavailable git -> fail-open (scored 100,
    soft note), never a failure."""
    if not available:
        return {"kpi": "churn_concentration", "group": "correction", "score": 100,
                "detail": f"churn unavailable (no git / empty range {rng})",
                "defects": [], "soft": [f"churn UNMEASURED for {rng} (git unavailable)"]}
    g = gini([float(v) for v in churn.values()])
    span = max(1e-9, GINI_RED - GINI_GREEN)
    score = _clamp(100 - 100 * (g - GINI_GREEN) / span)
    top = sorted(churn.items(), key=lambda kv: -kv[1])[:5]
    top_s = ", ".join(f"{k}:{v}" for k, v in top)
    return {"kpi": "churn_concentration", "group": "correction", "score": score,
            "detail": f"churn Gini {g:.2f} over {len(churn)} files in {rng} (HEAD-relative)",
            "defects": [],
            "soft": ([f"recent change concentrates (churn Gini {g:.2f}) in {rng}: {top_s} — "
                      f"a hot spot worth a second look (HEAD-relative, advisory)"]
                     if g > GINI_GREEN and churn else [])}


# ---------------------------------------------------------------------------
# Fold: KPIs -> steerability index, grade, steerability-debt, control-pane payload.
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, kpis: list[dict[str, Any]],
                  error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error,
            "next_action": "fix the read (run from repo ROOT, with git), then re-run",
            "workspace": workspace, "corpus": {}, "kpis": [],
        }
    by_name = {k["kpi"]: k for k in kpis}
    index = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"]
                      for n in KPI_WEIGHTS if n in by_name), 1)
    steerability_debt = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)
    grade = grade_letter(index)
    debt_by_group = {g: 0 for g in GROUPS}
    score_by_group: dict[str, list[int]] = {g: [] for g in GROUPS}
    for k in kpis:
        debt_by_group[k["group"]] += len(k["defects"])
        score_by_group[k["group"]].append(k["score"])
    group_index = {g: (round(sum(v) / len(v), 1) if v else 100.0)
                   for g, v in score_by_group.items()}
    breakdown = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "debt": len(k["defects"]), "soft": len(k["soft"]), "detail": k["detail"],
          "index_gain_to_clean": round(KPI_WEIGHTS.get(k["kpi"], 0) * max(0, 100 - k["score"]), 1)}
         for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))
    top_moves = [
        {
            "kpi": b["kpi"],
            "group": b["group"],
            "score": b["score"],
            "index_gain_to_clean": b["index_gain_to_clean"],
            "detail": b["detail"],
            "why": move_reason(b["kpi"]),
        }
        for b in sorted(breakdown, key=lambda x: (-x["index_gain_to_clean"], x["score"]))
        if b["index_gain_to_clean"] > 0
    ][:5]

    corpus = {
        # the headline: a 0-100 growth-invariant steerability INDEX, not a debt pile
        "index": index, "score": index, "grade": grade,
        "steerability_debt": steerability_debt,
        "soft_signals": n_soft,
        "debt_by_group": debt_by_group,
        "index_by_group": group_index,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
        "top_moves": top_moves,
    }

    if steerability_debt == 0:
        ok, verdict, finding = True, "OK", "steerable"
        reason = (f"steerability index {index}/100 (grade {grade}), zero hard debt "
                  f"across {len(kpis)} KPIs ({n_soft} advisory drift signal(s))")
        next_action = ("hold the index; watch the SOFT drift signals (coupling hubs, p90 "
                       "sizes, package drift) and re-run after the next structural change")
    else:
        ok, verdict, finding = False, "ACTION", "steerability_debt"
        worst = breakdown[0]
        reason = (f"{steerability_debt} unit(s) of steerability-debt; index {index}/100 "
                  f"(grade {grade}); heaviest: {worst['kpi']} ({worst['debt']} defect(s))")
        next_action = ("retire steerability-debt worst-first (see corpus.breakdown): split a "
                       "cmd dispatch god-file along its verb seams, or commit/wire the "
                       "control-pane ratchet; re-run to prove the index rose")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "kpis": kpis,
    }


def move_reason(kpi: str) -> str:
    return {
        "func_size_dist": "split long routines at tested seams",
        "package_doc_frac": "add one useful orientation sentence where readers start",
        "churn_concentration": "spread repeated edits by extracting stable helpers",
        "god_file_rate": "split oversized files along ownership boundaries",
        "god_func_rate": "split hard-to-review functions before they become shared chokepoints",
        "file_size_dist": "keep typical files below the p90 size line",
        "fan_in_gini": "reduce the few packages every change routes through",
        "hub_share": "move shared contracts down or split broad hubs",
        "dispatch_god_file": "split command dispatch so new verbs avoid one monolith",
        "ratchet_present": "restore the ratchet so regressions are visible",
        "worst_pkg_drift": "check packages that grew far beyond the pinned baseline",
    }.get(kpi, "improve this KPI with a structural change")


# ---------------------------------------------------------------------------
# Disk + git gathering (the impure shell around the pure KPIs).
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _git_lines(args: list[str], root: Path) -> list[str]:
    try:
        p = subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    if p.returncode != 0:
        return []
    return [ln for ln in p.stdout.splitlines() if ln.strip()]


def module_path(root: Path) -> str:
    """The module path from go.mod (the whole internal-vs-external import filter)."""
    text = _safe_read(root / "go.mod")
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("module "):
            return s[len("module "):].strip()
    return DEFAULT_MODULE


def gather_churn(root: Path, rng: str) -> tuple[dict[str, int], bool]:
    """Lines-changed per file over `rng` (HEAD-relative). Date-free (only +/- counts
    and paths), so it is clone-stable at a fixed commit. Returns (churn, available)."""
    lines = _git_lines(["log", "--numstat", "--format=", rng], root)
    if not lines:
        return {}, False
    churn: dict[str, int] = {}
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) != 3:
            continue
        added, deleted, path = parts
        if not path.endswith(".go") or path.endswith("_test.go"):
            continue
        try:
            n = (0 if added == "-" else int(added)) + (0 if deleted == "-" else int(deleted))
        except ValueError:
            continue
        churn[path] = churn.get(path, 0) + n
    return churn, True


def control_pane_wired(root: Path) -> bool:
    """True iff this scorecard is registered in the control-pane SCORECARDS fold —
    a substring check on the script name in the fold's source (the structural wire)."""
    text = _safe_read(root / "tools" / "scorecard_control_pane.py")
    return "steerability_scorecard.py" in text


def load_baseline(root: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads((root / BASELINE_REL).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def gather(root: Path, *, churn_range: str) -> list[dict[str, Any]]:
    """Read the tree (+ recent git history for churn) and run every pure KPI."""
    module = module_path(root)
    go_files = list_go_files(root, tests=False)

    # --- modularity: per-file size + per-function lengths (literal-aware scan) ---
    n_lines_list: list[int] = []
    n_long_funcs = 0          # functions over the SOFT line (all scan_go_file reports)
    n_god_files = n_god_funcs = 0
    n_total_funcs = 0
    pkg_loc: dict[str, int] = {}
    cmd_god_files: list[tuple[str, int]] = []
    # --- coupling: build the package fan-in graph ---
    importers_of: dict[str, set[str]] = {}  # target leaf -> set of importer leaves
    package_dirs: set[str] = set()
    documented_pkgs: set[str] = set()
    pkgs_seen_for_doc: set[str] = set()

    for rel in go_files:
        text = _safe_read(root / rel)
        info = scan_go_file(text)
        n = info["n_lines"]
        n_lines_list.append(n)
        if n > FILE_HARD_MAX:
            n_god_files += 1
            if rel.startswith("cmd/"):
                cmd_god_files.append((rel, n))
        n_total_funcs += info.get("n_funcs", 0)
        for _name, length in info.get("long_funcs", []):
            n_long_funcs += 1
            if length > FUNC_HARD_MAX:
                n_god_funcs += 1

        pkg = package_of(rel)
        package_dirs.add(pkg)
        pkg_loc[pkg] = pkg_loc.get(pkg, 0) + n

        # package doc-comment: count a package as documented if ANY of its files
        # carries a `// Package x` header or (for package main) `// Command x`.
        leaf = package_leaf(pkg)
        pkgs_seen_for_doc.add(leaf)
        if has_orientation_doc(text):
            documented_pkgs.add(leaf)

        # fan-in edges: this file's package imports these internal leaves
        src_leaf = package_leaf(pkg)
        for tgt_leaf in parse_internal_imports(text, module):
            if tgt_leaf == src_leaf:
                continue  # a package importing itself is not a coupling edge
            importers_of.setdefault(tgt_leaf, set()).add(src_leaf)

    fan_in = {tgt: len(srcs) for tgt, srcs in importers_of.items()}
    n_packages = len(pkgs_seen_for_doc)

    # --- correction: ratchet + drift + churn ---
    baseline = load_baseline(root)
    wired = control_pane_wired(root)
    baseline_pkg_loc = (baseline or {}).get("steer_pkg_loc") if isinstance(baseline, dict) else None
    if not isinstance(baseline_pkg_loc, dict):
        baseline_pkg_loc = {}
    churn, churn_ok = gather_churn(root, churn_range)

    return [
        kpi_file_size_dist(n_lines_list),
        kpi_func_size_dist(n_long_funcs, n_total_funcs),
        kpi_god_file_rate(n_god_files, len(go_files)),
        kpi_god_func_rate(n_god_funcs, max(1, n_total_funcs)),
        kpi_fan_in_gini(fan_in),
        kpi_hub_share(fan_in, n_packages),
        kpi_dispatch_god_file(cmd_god_files),
        kpi_package_doc_frac(len(documented_pkgs), n_packages),
        kpi_ratchet_present(baseline, wired),
        kpi_worst_pkg_drift(pkg_loc, baseline_pkg_loc),
        kpi_churn_concentration(churn, churn_range, churn_ok),
    ]


def collect(workspace: Path, *, churn_range: str = CHURN_RANGE_DEFAULT) -> dict[str, Any]:
    root = workspace.resolve()
    if not (root / ".git").exists() and not _git_lines(["rev-parse", "--git-dir"], root):
        return build_payload(workspace=str(root), kpis=[],
                             error=f"not a git repo at {root} — run from the repo ROOT")
    kpis = gather(root, churn_range=churn_range)
    return build_payload(workspace=str(root), kpis=kpis)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    lines = [
        f"steerability-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"STEERABILITY INDEX {c.get('index', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· hard debt {c.get('steerability_debt', 0)} · {c.get('soft_signals', 0)} drift signal(s)"),
        ("index by group: " + "  ".join(
            f"{g}:{c.get('index_by_group', {}).get(g, 0)}" for g in GROUPS)),
        "",
        "per-KPI (worst first):",
        f"  {'score':>5} {'debt':>4} {'soft':>4}  {'group':<13} {'kpi':<18} detail",
    ]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4} {b['soft']:>4}  {b['group']:<13} "
                     f"{b['kpi']:<18} {b['detail']}")
    lines.append("")
    lines.append("hard-debt work-list:")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        lines.append(f"  {k['kpi']} ({len(k['defects'])}):")
        for it in k["defects"][:12]:
            lines.append(f"      - {it}")
    if not any_defect:
        lines.append("  (none — zero steerability-debt; the index + drift signals carry the story)")
    # surface the heaviest SOFT drift signals so the operator sees where it's heading
    soft_all = [(k["kpi"], s) for k in payload.get("kpis", []) for s in k["soft"]]
    if soft_all:
        lines.append("")
        lines.append(f"drift signals (advisory — {len(soft_all)}):")
        for kpi, s in soft_all[:14]:
            lines.append(f"      · [{kpi}] {s}")
        if len(soft_all) > 14:
            lines.append(f"      ... and {len(soft_all) - 14} more")
    top_moves = c.get("top_moves") or []
    if top_moves:
        lines.append("")
        lines.append("highest-index moves:")
        for m in top_moves[:5]:
            lines.append(f"      +{m['index_gain_to_clean']:.1f} pts  {m['kpi']} ({m['score']}/100): {m['why']}")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    out: list[str] = []
    out.append("---")
    out.append('title: "fak steerability scorecard — the growth-invariant steering index"')
    out.append('description: "fak\'s deterministic steerability scorecard: eleven '
               'growth-invariant KPIs across modularity, coupling, navigability, and '
               'correction, folded into a 0-100 steerability index that stays flat as '
               'the repo grows — re-derived from the working tree."')
    out.append("---")
    out.append("")
    out.append("# Steerability scorecard")
    out.append("")
    if stamp:
        out.append(f"<!-- steerability-scorecard: {stamp} · process: tools/steerability_scorecard.py -->")
        out.append("")
    out.append("This is the measuring stick for the question no other fak scorecard asks: as the "
               "repo doubles in size, does the **effort to steer, change, and navigate it stay "
               "roughly flat** — and if it drifts, do we know, and can we correct? Every number "
               "below is re-derived from the working tree by `tools/steerability_scorecard.py` — "
               "no hand-entry.")
    out.append("")
    out.append("The headline is a 0–100 **steerability index**, not a debt count, because "
               "steerability is a property of *shape*, not *size*. Every KPI is **growth-"
               "invariant** — a ratio, density, or distribution percentile — so a 2×-larger repo "
               "with the same modular discipline scores *identically*. (A raw defect count, the "
               "kind every sibling scorecard uses, would climb just from getting bigger.)")
    out.append("")
    out.append("> Regenerate: `python tools/steerability_scorecard.py --markdown --stamp DATE > docs/STEERABILITY-SCORECARD.md`")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Steerability index** | **{c.get('index', 0)}/100 (grade {c.get('grade', '?')})** |")
    out.append(f"| Hard steerability-debt | {c.get('steerability_debt', 0)} |")
    out.append(f"| Advisory drift signals | {c.get('soft_signals', 0)} |")
    gi = c.get("index_by_group", {})
    out.append(f"| Index by group | modularity:{gi.get('modularity',0)} · coupling:{gi.get('coupling',0)} "
               f"· navigability:{gi.get('navigability',0)} · correction:{gi.get('correction',0)} |")
    out.append("")
    out.append("## Per-KPI")
    out.append("")
    out.append("Eleven KPIs, each 0–100, in four groups. `debt` = HARD steerability-debt (only "
               "`dispatch_god_file` and `ratchet_present` can emit it — everything else is "
               "advisory, because its cheapest fix would be gaming a detector). `god_file_rate` / "
               "`god_func_rate` SCORE the size rate but leave the raw count to `code_quality` "
               "(no portfolio double-count). `package_doc_frac` counts `// Package ...` docs for "
               "libraries and `// Command ...` docs for command packages. `churn_concentration` "
               "is HEAD-relative.")
    out.append("")
    out.append("| Group | KPI | Score | Debt | Clean-gain | Detail |")
    out.append("|---|---|---:|:--:|---:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | +{b.get('index_gain_to_clean', 0):.1f} | {b['detail']} |")
    out.append("")
    any_defect = any(k["defects"] for k in payload.get("kpis", []))
    if any_defect:
        out.append("## Hard-debt work-list")
        out.append("")
        for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
            if not k["defects"]:
                continue
            out.append(f"### `{k['kpi']}` ({k['group']}) — {len(k['defects'])} defect(s), score {k['score']}")
            for it in k["defects"]:
                out.append(f"- {it}")
            out.append("")
    else:
        out.append("No hard steerability-debt: the index and the advisory drift signals below "
                   "carry the story. 🎉")
        out.append("")
    soft_all = [(k["kpi"], s) for k in payload.get("kpis", []) for s in k["soft"]]
    if soft_all:
        out.append("## Drift signals (advisory)")
        out.append("")
        for kpi, s in soft_all:
            out.append(f"- **`{kpi}`** — {s}")
        out.append("")
    top_moves = c.get("top_moves") or []
    if top_moves:
        out.append("## Highest-index moves")
        out.append("")
        out.append("| Gain if clean | KPI | Why this helps | Current detail |")
        out.append("|---:|---|---|---|")
        for m in top_moves:
            out.append(f"| +{m['index_gain_to_clean']:.1f} | `{m['kpi']}` | {m['why']} | {m['detail']} |")
        out.append("")
    return "\n".join(out)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bi, ci = b.get("index", 0), cur.get("index", 0)
    bd, cd = b.get("steerability_debt", 0), cur.get("steerability_debt", 0)
    lines = [
        f"steerability index: {bi}/100 -> {ci}/100   ({ci - bi:+.1f})",
        f"hard debt:          {bd} -> {cd}",
    ]
    for gp in GROUPS:
        gb = (b.get("index_by_group") or {}).get(gp, 0)
        gc = (cur.get("index_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<13} {gb} -> {gc}  ({gc - gb:+.1f})")
    if ci >= bi and cd <= bd:
        lines.append("VERDICT: steerability held or improved (index up/flat, debt down/flat).")
    else:
        lines.append("VERDICT: steerability REGRESSED — index fell or debt rose.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Steerability scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the snapshot markdown body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--range", dest="churn_range", default=CHURN_RANGE_DEFAULT,
                    help=f"git range for the HEAD-relative churn KPI (default {CHURN_RANGE_DEFAULT})")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the steerability-index delta vs a prior baseline JSON")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, churn_range=args.churn_range)

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
