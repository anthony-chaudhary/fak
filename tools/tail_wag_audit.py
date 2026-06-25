#!/usr/bin/env python3
"""Tail-wag audit — the deterministic backing tool for the manual ``/tail-wag`` skill.

The ``/tail-wag`` skill (``.claude/skills/tail-wag``) finds "the tail wagging the
dog": a *peripheral* concern disproportionately driving *core* design. Until now it
ran by hand, so it could not run in CI or fold into a scorecard — the stability
scorecard emits exactly that as a SOFT note (``drift_detectors_wired``: "the
/tail-wag inverted-priority finder is a manual skill with no deterministic backing
tool"). This is the backing tool: it screens the git-tracked Go import graph for
inverted-priority candidates from three machine signals, the same three the skill
scores by hand:

  LAYER INVERSION       — a lower-layer package is reaching UP into the core. In a
                          layered-DAG module (``internal/architest`` enforces that a
                          package imports only its own tier or below), a *literal*
                          upward import can't exist — it fails the build. What CAN
                          accrete is CONCENTRATION: a low-tier module that an unusual
                          share of higher tiers bend around, a de-facto contract that
                          was never designed as one. We measure the tier SPAN: how
                          many layers up the influence of a module reaches.
  BLAST-RADIUS ASYMMETRY — the "tail" is a narrow slice but the accommodation spreads
                          wide. We measure CORE FAN-IN: how many strictly-higher-tier
                          packages import the module (the "dogs" it wags).
  COUNTERFACTUAL CHEAPNESS — if the tail vanished, a lot could be deleted with little
                          loss. A *small* module carrying a *wide* fan-in is the
                          cheapest-to-remove / highest-leverage tail. We measure SIZE
                          (non-test source lines) — smaller is more tail-like — and
                          report DENSITY (core dependents per 100 lines).

Each signal is bucketed 1–5 (0 = absent), exactly like the skill's manual
inversion/blast/counterfactual scores, and the tail-wag index is their product
(1–125) — the skill's "highest product wins" rule, made deterministic. A finding
needs **all three present** (each >= 2); the winner has all three *loudly* (each
>= 4), which is the only thing that flips the verdict to REVIEW.

This is a SCREEN, not a verdict of guilt: a deterministic tool cannot tell a
legitimate central primitive (a tier-0 frozen contract, a stdlib-only utility used
everywhere by design) from an accreted tail — only the human/LLM ``/tail-wag`` skill
can adjudicate. So tier-0 (the designed shared spine, e.g. ``abi``) is excluded by
construction, and the output frames every row as a *candidate for review*. The skill
wraps this tool the way ``/quality-score`` wraps its scorecard: the tool ranks, the
skill judges.

Deterministic + read-only by construction: it reads the git-tracked tree (so two
clones of the same commit score identically), parses import blocks and the
``internal/architest`` tier table out of source, and edits nothing. Run from the repo
ROOT::

    python tools/tail_wag_audit.py                 # human report (ranked candidates)
    python tools/tail_wag_audit.py --json          # machine payload (control-pane shape)
    python tools/tail_wag_audit.py --markdown       # a snapshot body
    python tools/tail_wag_audit.py --compare base.json  # prove the worst index moved
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-tail-wag-audit/1"

# The Go module path whose internal/ packages form the graph we screen.
MODULE = "github.com/anthony-chaudhary/fak"
INTERNAL_PREFIX = f"{MODULE}/internal/"

# The authoritative tier table lives as a Go map literal in the architest test —
# the SAME source the no-upward-imports invariant reads, so this tool and the build
# agree on every package's layer by construction (no hand-maintained second copy).
TIER_SOURCE = "internal/architest/architest_test.go"
_TIER_BLOCK_RE = re.compile(r"var tier = map\[string\]int\{(.*?)\n\}", re.DOTALL)
_TIER_ENTRY_RE = re.compile(r'"([^"]+)"\s*:\s*(\d+)')
# Tier names, lowest (root contract) to highest (integrator), matching architest.
TIER_NAMES = ["root", "foundation", "mechanism", "composer", "integrator"]
# Tier 0 is the designed shared spine (the frozen ABI): being depended on widely is
# its job, never a tail-wag. Exclude it from candidacy by construction.
ROOT_TIER = 0

# An import line we care about: a quoted path under this module's internal/.
_IMPORT_RE = re.compile(r'"' + re.escape(INTERNAL_PREFIX) + r'([A-Za-z0-9_./]+)"')

# --- signal -> 1..5 bucket ladders (0 = signal absent) ----------------------
# Each ladder is a list of (threshold, score), highest threshold first; the first
# threshold the value meets wins. Round, principled cut points — not tuned to force
# any particular verdict on the current tree.
BLAST_LADDER = [(10, 5), (6, 4), (4, 3), (2, 2), (1, 1)]          # core fan-in
CHEAP_LADDER = [(5, 0)]  # placeholder; size uses an inverted ladder below
# size (non-test source lines): SMALLER is more tail-like, so the ladder is on the
# upper bound — a module <= the bound earns that score.
SIZE_LADDER = [(80, 5), (200, 4), (500, 3), (1200, 2)]            # else -> 1
# A finding requires every signal present at least moderately.
FINDING_FLOOR = 2
# The winner — the only thing that flips the verdict to REVIEW — needs all three
# signals LOUD: tier span >= 3, core fan-in >= 6, and size <= 200 lines.
LOUD = 4

MAX_FINDINGS_SHOWN = 8  # the ranked head the renderers print; corpus keeps the count.


# ---------------------------------------------------------------------------
# Small pure helpers (the testable core).
# ---------------------------------------------------------------------------

def tier_name(t: int) -> str:
    return TIER_NAMES[t] if 0 <= t < len(TIER_NAMES) else f"tier{t}"


def parse_tiers(source: str | None) -> dict[str, int]:
    """Extract ``{package: tier}`` from the architest tier map literal. A package
    key is the directory leaf (``internal/<leaf>``); the value is its declared layer."""
    if not source:
        return {}
    m = _TIER_BLOCK_RE.search(source)
    if not m:
        return {}
    return {name: int(t) for name, t in _TIER_ENTRY_RE.findall(m.group(1))}


def parse_internal_imports(go_source: str) -> set[str]:
    """The set of internal package keys this Go file imports (the segment(s) after
    ``internal/``). A nested package (``internal/foo/bar``) keys as ``foo/bar``."""
    return set(_IMPORT_RE.findall(go_source))


def nonblank_lines(go_source: str) -> int:
    return sum(1 for ln in go_source.splitlines() if ln.strip())


def _ladder(value: int, ladder: list[tuple[int, int]], *, default: int) -> int:
    for thresh, score in ladder:
        if value >= thresh:
            return score
    return default


def blast_score(core_fan_in: int) -> int:
    """1..5 from the count of strictly-higher-tier importers (0 if none)."""
    return _ladder(core_fan_in, BLAST_LADDER, default=0)


def inversion_score(tier_span: int) -> int:
    """1..5 from how many layers up the influence reaches (span 1->2 .. span 4->5),
    0 if no higher-tier importer exists."""
    if tier_span <= 0:
        return 0
    return min(5, tier_span + 1)


def cheapness_score(size_lines: int) -> int:
    """1..5 from size — SMALLER is more tail-like (a tiny module wagging big ones is
    the cheapest to delete and the highest leverage). A package with no source is 0."""
    if size_lines <= 0:
        return 0
    for upper, score in SIZE_LADDER:
        if size_lines <= upper:
            return score
    return 1


def score_candidate(pkg: str, tier: int, size_lines: int,
                    up_importers: list[tuple[str, int]]) -> dict[str, Any]:
    """Build the scored row for one candidate package. ``up_importers`` is the list
    of (importer_pkg, importer_tier) for strictly-higher-tier importers — the core
    modules this one wags."""
    core_fan_in = len(up_importers)
    tier_span = max((t - tier for _, t in up_importers), default=0)
    inv = inversion_score(tier_span)
    blast = blast_score(core_fan_in)
    cheap = cheapness_score(size_lines)
    product = inv * blast * cheap
    density = round(100.0 * core_fan_in / size_lines, 2) if size_lines else 0.0
    is_finding = min(inv, blast, cheap) >= FINDING_FLOOR
    is_loud = inv >= LOUD and blast >= LOUD and cheap >= LOUD
    dogs = sorted(f"{p}({tier_name(t)})" for p, t in up_importers)
    return {
        "pkg": pkg,
        "tier": tier,
        "tier_name": tier_name(tier),
        "size_lines": size_lines,
        "core_fan_in": core_fan_in,
        "tier_span": tier_span,
        "density_per_100loc": density,
        "inversion": inv,
        "blast": blast,
        "cheapness": cheap,
        "product": product,
        "is_finding": is_finding,
        "is_loud": is_loud,
        "dogs": dogs,
    }


def rank_key(row: dict[str, Any]) -> tuple:
    """Total order for ranking: product desc, then core fan-in desc, then size asc,
    then package name — fully deterministic (no ties left to dict order)."""
    return (-row["product"], -row["core_fan_in"], row["size_lines"], row["pkg"])


# ---------------------------------------------------------------------------
# Fold: scored rows -> control-pane payload.
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, rows: list[dict[str, Any]],
                  error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error,
            "next_action": "fix the read (run from repo ROOT, with git), then re-run",
            "workspace": workspace, "corpus": {}, "findings": [],
        }
    ranked = sorted(rows, key=rank_key)
    findings = [r for r in ranked if r["is_finding"]]
    loud = [r for r in findings if r["is_loud"]]
    head = findings[:MAX_FINDINGS_SHOWN]

    corpus = {
        "packages_scored": len(rows),
        "candidate_findings": len(findings),
        "loud_findings": len(loud),
        "worst_product": findings[0]["product"] if findings else 0,
        "findings": head,
    }

    if not findings:
        ok, verdict, finding = True, "OK", "balanced"
        reason = (f"no inverted-priority candidate: scored {len(rows)} non-root packages; "
                  "none carries all three signals (layer inversion + core fan-in + small "
                  "size) at once — no peripheral module is visibly wagging the core")
        next_action = ("hold the line; re-run after a low-tier package grows a wide "
                       "higher-tier fan-in, or wire this into the stability scorecard's "
                       "drift group")
    elif not loud:
        ok, verdict, finding = True, "OK", "advisory"
        w = findings[0]
        reason = (f"{len(findings)} inverted-priority candidate(s), none LOUD: worst is "
                  f"`{w['pkg']}` (tier {w['tier_name']}, {w['size_lines']} lines, "
                  f"{w['core_fan_in']} higher-tier importer(s), product {w['product']}/125) "
                  "— present but below the all-three-loud bar")
        next_action = ("advisory only — review corpus.findings with /tail-wag if a "
                       "candidate's fan-in keeps climbing")
    else:
        ok, verdict, finding = False, "REVIEW", "inverted_priority"
        w = loud[0]
        reason = (f"{len(loud)} LOUD inverted-priority candidate(s); worst: `{w['pkg']}` "
                  f"(tier {w['tier_name']}, {w['size_lines']} lines) wags {w['core_fan_in']} "
                  f"higher-tier package(s) up to {w['tier_span']} layer(s) above it "
                  f"(inversion {w['inversion']}/5 · blast {w['blast']}/5 · cheapness "
                  f"{w['cheapness']}/5 = product {w['product']}/125) — a small peripheral "
                  "module is disproportionately driving the core")
        next_action = ("run /tail-wag on the worst candidate (corpus.findings[0]) to "
                       "confirm the inversion and propose the rebalance — push the "
                       "concentrated knowledge below the line, or promote the module to a "
                       "declared contract")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "findings": head,
    }


# ---------------------------------------------------------------------------
# Disk + git gathering (the impure shell around the pure scoring).
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


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _pkg_of(rel: str) -> str | None:
    """The internal package key for a tracked path, or None if not under internal/.
    ``internal/gateway/x.go`` -> ``gateway``; ``internal/foo/bar/x.go`` -> ``foo/bar``."""
    parts = rel.split("/")
    if len(parts) < 3 or parts[0] != "internal":
        return None
    return "/".join(parts[1:-1])


def gather(root: Path) -> list[dict[str, Any]]:
    """Read the git-tracked tree, build the internal import graph, score every
    non-root package that some strictly-higher-tier package imports."""
    tracked = set(_git_lines(["ls-files"], root))
    tiers = parse_tiers(_safe_read(root / TIER_SOURCE))
    if not tiers:
        return []

    # production import edges (non-test .go only) + per-package source size.
    importers_of: dict[str, set[str]] = {}   # imported pkg -> {importer pkgs}
    size_of: dict[str, int] = {}
    for rel in sorted(tracked):
        if not rel.endswith(".go") or rel.endswith("_test.go"):
            continue
        pkg = _pkg_of(rel)
        if pkg is None:
            continue
        src = _safe_read(root / rel)
        size_of[pkg] = size_of.get(pkg, 0) + nonblank_lines(src)
        for dep in parse_internal_imports(src):
            if dep == pkg:
                continue
            importers_of.setdefault(dep, set()).add(pkg)

    rows: list[dict[str, Any]] = []
    for pkg, tier in tiers.items():
        if tier <= ROOT_TIER:
            continue  # tier-0 root is the designed spine, never a tail
        size = size_of.get(pkg, 0)
        if size <= 0:
            continue  # no production source (test-only / generated dir) — not a tail
        up = [(imp, tiers[imp]) for imp in importers_of.get(pkg, set())
              if imp in tiers and tiers[imp] > tier]
        if not up:
            continue  # nothing higher depends on it — cannot be wagging the core
        rows.append(score_candidate(pkg, tier, size, up))
    return rows


def collect(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    if not (root / ".git").exists() and not _git_lines(["rev-parse", "--git-dir"], root):
        return build_payload(workspace=str(root), rows=[],
                             error=f"not a git repo at {root} — run from the repo ROOT")
    tiers = parse_tiers(_safe_read(root / TIER_SOURCE))
    if not tiers:
        return build_payload(workspace=str(root), rows=[],
                             error=f"could not parse the tier table from {TIER_SOURCE} — "
                                   "is this the fak repo root?")
    return build_payload(workspace=str(root), rows=gather(root))


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _row_line(r: dict[str, Any]) -> str:
    return (f"  {r['product']:>4} {r['inversion']}x{r['blast']}x{r['cheapness']}  "
            f"{r['pkg']:<16} {r['tier_name']:<11} {r['size_lines']:>5}ln  "
            f"fan-in {r['core_fan_in']:>2} (span {r['tier_span']})  "
            f"{r['density_per_100loc']:>5}/100loc")


def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    lines = [
        f"tail-wag-audit: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"packages scored {c.get('packages_scored', 0)} · "
         f"candidates {c.get('candidate_findings', 0)} · "
         f"loud {c.get('loud_findings', 0)} · "
         f"worst product {c.get('worst_product', 0)}/125"),
        "",
        "ranked inverted-priority candidates (worst first):",
        ("  prod inv×bl×ch  package          tier         size   "
         "core fan-in        density"),
    ]
    findings = c.get("findings", [])
    if not findings:
        lines.append("  (none — no peripheral module is wagging the core)")
    for r in findings:
        lines.append(_row_line(r))
        if r.get("dogs"):
            shown = ", ".join(r["dogs"][:8])
            more = f" … (+{len(r['dogs']) - 8})" if len(r["dogs"]) > 8 else ""
            lines.append(f"        wags: {shown}{more}")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    out: list[str] = []
    out.append("---")
    out.append('title: "fak tail-wag audit — deterministic inverted-priority screen"')
    out.append('description: "fak\'s deterministic backing tool for the /tail-wag skill: '
               'ranks peripheral modules that disproportionately drive the core, scored from '
               'the git-tracked Go import graph by layer inversion, blast-radius asymmetry, '
               'and counterfactual cheapness."')
    out.append("---")
    out.append("")
    out.append("# Tail-wag audit — is a peripheral concern driving the core?")
    out.append("")
    if stamp:
        out.append(f"<!-- tail-wag-audit: {stamp} · process: tools/tail_wag_audit.py -->")
        out.append("")
    out.append("Deterministic screen for the [`/tail-wag`](.claude/skills/tail-wag) skill. "
               "Every row is re-derived from the git-tracked Go import graph by "
               "`tools/tail_wag_audit.py` — no hand-entry. Each candidate is scored on the "
               "three signals the skill scores by hand: **layer inversion** (tier span up), "
               "**blast-radius asymmetry** (higher-tier fan-in), and **counterfactual "
               "cheapness** (small size). This is a SCREEN — the skill adjudicates whether a "
               "candidate is a real inversion or a legitimate shared primitive.")
    out.append("")
    out.append(f"**Verdict:** {payload.get('verdict')} ({payload.get('finding')}) — "
               f"{payload.get('reason')}")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Packages scored | {c.get('packages_scored', 0)} |")
    out.append(f"| Inverted-priority candidates | {c.get('candidate_findings', 0)} |")
    out.append(f"| LOUD candidates (all three signals >= {LOUD}) | {c.get('loud_findings', 0)} |")
    out.append(f"| Worst tail-wag index | {c.get('worst_product', 0)}/125 |")
    out.append("")
    out.append("## Ranked candidates")
    out.append("")
    findings = c.get("findings", [])
    if not findings:
        out.append("No candidate carries all three signals at once. 🎉")
        out.append("")
    else:
        out.append("| Index | inv×blast×cheap | Package | Tier | Lines | Core fan-in | "
                   "Span | Wags (higher-tier importers) |")
        out.append("|---:|:--:|---|---|---:|---:|:--:|---|")
        for r in findings:
            dogs = ", ".join(f"`{d}`" for d in r["dogs"][:8])
            if len(r["dogs"]) > 8:
                dogs += f" (+{len(r['dogs']) - 8})"
            out.append(f"| {r['product']} | {r['inversion']}×{r['blast']}×{r['cheapness']} "
                       f"| `{r['pkg']}` | {r['tier_name']} | {r['size_lines']} | "
                       f"{r['core_fan_in']} | {r['tier_span']} | {dogs} |")
        out.append("")
    out.append(f"> Next: {payload.get('next_action')}")
    out.append("")
    return "\n".join(out)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bw, cw = b.get("worst_product", 0), cur.get("worst_product", 0)
    bl, cl = b.get("loud_findings", 0), cur.get("loud_findings", 0)
    bc, cc = b.get("candidate_findings", 0), cur.get("candidate_findings", 0)
    lines = [
        f"worst tail-wag index: {bw} -> {cw}/125   ({'+' if cw >= bw else ''}{cw - bw})",
        f"loud candidates:      {bl} -> {cl}",
        f"all candidates:       {bc} -> {cc}",
    ]
    if cw <= bw and cl <= bl:
        lines.append("VERDICT: the tail is no louder (worst index and loud count did not rise).")
    else:
        lines.append("VERDICT: a tail grew — the worst index or loud count rose; review the head.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tail-wag audit (deterministic, read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the snapshot markdown body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the tail-wag delta vs a prior baseline JSON")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace)

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
