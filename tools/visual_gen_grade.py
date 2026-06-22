#!/usr/bin/env python3
"""visual_gen_grade.py -- a deterministic, static quality grader for the fleet's
visual-generation pipeline (the explanatory-diagram deck under
fleet-bite-test/visuals/: Mermaid `.mmd` sources rendered to SVG/PNG).

"Visual generation" in this repo is diagram production, not an image model. This
grader turns the hand-authored quality judgment in that deck's `_meta.json`
(`faithful` / `syntaxOk` / `issuesFound`) and the failure modes documented in
`RENDERING-NOTE.md` into a *computed* 0-1 score per figure, so the score can feed
the RSI keep-bit (internal/shipgate) and land in the benchmark catalog DB.

It is static-first by design: it reads the `.mmd` source and the emitted `.svg`
text and never invokes a browser/renderer (the `--render` flag opts back in when
node is present). That makes it deterministic, dependency-free, CI-safe, and
reproducible across machines -- the same `-live=false` witness discipline the rest
of the repo uses.

Two FAMILIES of checks, scored separately so the consumer can tell WHICH layer is
at fault (the source is author-controlled; the render is pipeline-controlled):

  source checks (the `.mmd`):
    - syntax_ok            first line is the required mermaid init block; brackets
                           balanced; no node id named `end`; risky labels quoted
    - class_coverage       every styled node id is assigned a classDef class
    - killer_number        the figure's killerNumber / caption punch appears verbatim
    - meta_faithful        `_meta.json[id].faithful && .syntaxOk` (carry the curated
                           human bar so the grader never silently drops below it)

  render checks (the `.svg`, skipped with score-neutral when no `.svg` exists):
    - text_renderable      real <text>/<tspan> present (NOT all-<foreignObject>,
                           which renders blank outside a browser) -- RENDERING-NOTE #1
    - intrinsic_size       root <svg> carries an explicit pixel height, not just
                           width="100%" with a viewBox -- RENDERING-NOTE #2

Usage:
  python tools/visual_gen_grade.py                 # grade the default deck, human table
  python tools/visual_gen_grade.py --json          # machine-readable report
  python tools/visual_gen_grade.py --floor 0.8     # set the below-floor threshold
  python tools/visual_gen_grade.py --out report.json
  python tools/visual_gen_grade.py --render        # render .mmd->.svg first (needs node)
  python tools/visual_gen_grade.py --dir <path>    # grade a different visuals dir

Exit status: 0 on a successful grade (a LOW score is data, not an error -- the same
"ACTION is a PASS, only BROKEN fails" contract loop-audit uses); 2 only when the
grader itself cannot run (the visuals dir is missing or unreadable).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "fleet-bite-test" / "visuals"
SCHEMA = "fleet-visual-gen-grade/1"
DEFAULT_FLOOR = 0.80

# Each check contributes its weight to the figure score; weights sum to 1.0 when a
# figure has both an .mmd and an .svg. When the .svg is absent, the render-check
# weights are dropped and the remaining (source) weights are renormalized, so a
# source-only figure is graded purely on source quality rather than penalized for a
# render it never claimed.
WEIGHTS = {
    # source family
    "syntax_ok": 0.25,
    "class_coverage": 0.15,
    "killer_number": 0.10,
    "meta_faithful": 0.15,
    # render family
    "text_renderable": 0.20,
    "intrinsic_size": 0.15,
}
SOURCE_CHECKS = ("syntax_ok", "class_coverage", "killer_number", "meta_faithful")
RENDER_CHECKS = ("text_renderable", "intrinsic_size")

# The verbatim init block every deck figure must open with (RENDERING-NOTE / _meta
# style invariant). We check the stable prefix, not the whole themeVariables blob,
# so a legitimate palette tweak does not trip the gate.
INIT_PREFIX = "%%{init:"
# A node id literally named `end` breaks mermaid's flowchart parser (reserved word).
RESERVED_END = re.compile(r"(^|\s)end[\s\[({]")
# Characters that, inside a node label, must be wrapped in quotes or mermaid mis-parses.
RISKY_LABEL_CHARS = set("()[]{}/,:")


def load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _meta_index(visuals_dir: Path) -> dict[str, dict]:
    """Map figure id -> its _meta.json record (the curated human judgment)."""
    meta = load_json(visuals_dir / "_meta.json")
    if not isinstance(meta, list):
        return {}
    out: dict[str, dict] = {}
    for rec in meta:
        if isinstance(rec, dict) and rec.get("id"):
            out[str(rec["id"])] = rec
    return out


# --------------------------------------------------------------------------------
# source checks (.mmd)
# --------------------------------------------------------------------------------
# The deck mixes two mermaid grammars: node-and-edge diagrams (flowchart / graph /
# stateDiagram) and data charts (xychart-beta). They have DIFFERENT correctness
# rules -- a chart has no classDef, no node ids, and legitimately uses unquoted
# `[..]` axis-range / series arrays -- so the source checks dispatch on the type
# rather than applying flowchart rules to a bar chart.

FLOW_KINDS = ("flowchart", "graph", "statediagram", "sequencediagram", "classdiagram")
CHART_KINDS = ("xychart", "pie", "quadrantchart", "sankey")


def diagram_kind(mmd: str) -> str:
    """The mermaid diagram kind, lowercased -- the first non-empty, non-%%{init}%%
    directive line. 'flow' for node-and-edge diagrams, 'chart' for data charts,
    'other' otherwise."""
    for ln in mmd.splitlines():
        s = ln.strip()
        if not s or s.startswith("%%"):
            continue
        if any(s.lower().startswith(k) for k in FLOW_KINDS):
            return "flow"
        if any(s.lower().startswith(k) for k in CHART_KINDS):
            return "chart"
        return "other"
    return "other"


def _strip_quoted(line: str) -> str:
    """Drop every "..." span AND every |...| edge-label span, so a risky-char or
    node-id scan sees only the structural skeleton, never text inside a label."""
    no_quotes = re.sub(r'"[^"]*"', "", line)
    return re.sub(r"\|[^|]*\|", "", no_quotes)


def _subgraph_ids(mmd: str) -> set[str]:
    """Subgraph ids (`subgraph BIN[...]`) are containers, not nodes -- they take a
    title, never a class -- so they must be excluded from the node set."""
    ids: set[str] = set()
    for m in re.finditer(r"^\s*subgraph\s+([A-Za-z_]\w*)", mmd, re.MULTILINE):
        ids.add(m.group(1))
    return ids


# Flowchart shape-bracket label bodies, AFTER quoted spans are removed. The shape
# DELIMITERS themselves (`[ ] ( ) { }`) are structure, never risky; a label wrapped
# in "..." is always safe (mermaid takes it verbatim). So we first delete every
# "..." span and every |...| edge label, THEN look for a shape bracket whose
# remaining body still carries ()[]{}/,: -- that is the genuine unquoted-label
# hazard. A quoted label collapses to an empty body and never trips.
_LABEL_BODY = re.compile(
    r"(?:\[\[|\(\(|\(\[|\{\{|\[|\(|\{)([^\[\]\(\)\{\}]*)(?:\]\]|\)\)|\]\)|\}\}|\]|\)|\})"
)


def _has_unquoted_risky_label(line: str) -> bool:
    scan = re.sub(r'"[^"]*"', "", line)          # drop quoted label spans
    scan = re.sub(r"\|[^|]*\|", "", scan)         # drop |edge-label| spans
    for m in _LABEL_BODY.finditer(scan):
        body = m.group(1).strip()
        if body and any(c in body for c in RISKY_LABEL_CHARS):
            return True
    return False


def check_syntax(mmd: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    nonempty = [ln for ln in mmd.splitlines() if ln.strip()]
    if not nonempty:
        return False, ["empty .mmd"]
    if not nonempty[0].lstrip().startswith(INIT_PREFIX):
        reasons.append("first non-empty line is not the %%{init:...}%% block")
    # balanced brackets across the whole source (applies to every kind)
    for open_c, close_c in (("(", ")"), ("[", "]"), ("{", "}")):
        if mmd.count(open_c) != mmd.count(close_c):
            reasons.append(f"unbalanced {open_c}{close_c}: "
                           f"{mmd.count(open_c)} vs {mmd.count(close_c)}")
    kind = diagram_kind(mmd)
    # The reserved-`end`-node-id and unquoted-label rules are FLOWCHART rules; a
    # data chart's `line [..]` / `x-axis "..." [..]` is not a node label.
    if kind == "flow":
        # A bare `end` is a legal subgraph terminator. The hazard is `end` used as a
        # NODE id -- i.e. followed by a shape bracket (`end[...]`, `end(...)`) or
        # wired with an edge (`end --> x`). Only those trip mermaid's parser.
        for ln in nonempty:
            body = _strip_quoted(ln).strip()
            if re.match(r"^end\s*(\[|\(|\{)", body) or re.match(r"^end\s*--", body) \
                    or re.search(r"(--?>|---)\s*end\s*(\[|\(|\{|$)", body):
                reasons.append("a node uses the reserved id `end`")
                break
        for ln in nonempty:
            if ln.lstrip().startswith("%%") or ln.lstrip().startswith("subgraph"):
                continue
            if _has_unquoted_risky_label(ln):
                reasons.append("a flowchart label contains ()[]{}/,: but is not quoted")
                break
    return (len(reasons) == 0), reasons


def _flow_node_ids(mmd: str) -> set[str]:
    """The flow node ids actually declared with a shape, with quoted labels and edge
    pipes removed first so a word INSIDE a label is never mistaken for a node id.
    Subgraph ids are excluded (they are containers, not nodes)."""
    subs = _subgraph_ids(mmd)
    declared: set[str] = set()
    for ln in mmd.splitlines():
        s = ln.strip()
        if not s or s.startswith("%%") or s.startswith("classDef") or s.startswith("class "):
            continue
        skeleton = _strip_quoted(ln)
        # An id is a token immediately followed by a shape opener; after stripping
        # labels, only real declarations keep their shape bracket adjacent.
        for m in re.finditer(r"(?:^|[\s>.-])([A-Za-z_]\w*)\s*(\[\[|\(\[|\{\{|\[|\(\(|\(|\{)", skeleton):
            nid = m.group(1)
            # Mermaid keywords that take a bracketed argument but are NOT styled
            # nodes: subgraph titles, the `title` directive, accessibility blocks.
            if nid in ("subgraph", "classDef", "class", "flowchart", "graph", "end",
                       "direction", "title", "accTitle", "accDescr", "style", "linkStyle"):
                continue
            if nid in subs:
                continue
            declared.add(nid)
    return declared


def check_class_coverage(mmd: str) -> tuple[bool, list[str]]:
    """Every flow node id that carries a shape should be assigned a classDef class,
    so the figure renders in the deck's shared palette. Not applicable to data
    charts (no classDef) -- those pass."""
    if diagram_kind(mmd) != "flow":
        return True, []
    classed: set[str] = set()
    # `class a,b,c name;` statements
    for m in re.finditer(r"^\s*class\s+([^;]+?)\s+\w+\s*;?\s*$", mmd, re.MULTILINE):
        for cid in m.group(1).split(","):
            classed.add(cid.strip())
    # inline `id[...]:::class` assignment. Quoted label bodies can themselves carry
    # brackets ("adjudicate (in-process)"), so strip quoted spans FIRST; then a node
    # collapses to `id[]:::class` / `id([]):::class` and the id-before-shape-before-:::
    # pattern matches anywhere on the line (incl. `x --> y[]:::k`).
    for ln in mmd.splitlines():
        if ":::" not in ln:
            continue
        skel = _strip_quoted(ln)
        for m in re.finditer(
            r"(?:^|[\s>.-])([A-Za-z_]\w*)\s*(?:\[\[|\(\[|\{\{|\[|\(\(|\(|\{)[^\]\)\}]*"
            r"(?:\]\]|\)\)|\]\)|\}\}|\]|\)|\})\s*:::",
            skel,
        ):
            classed.add(m.group(1))
    declared = _flow_node_ids(mmd)
    if not declared:
        return True, []
    missing = sorted(declared - classed)
    if missing:
        return False, [f"{len(missing)} node(s) unclassed: {', '.join(missing[:6])}"]
    return True, []


def check_killer_number(mmd: str, meta: dict | None) -> tuple[bool, list[str]]:
    """The figure's killer number / punch must actually appear in the diagram so the
    rendered image carries it. Lifted from the curated `_meta.json`.

    If the figure has no killerNumber declared, the check is not applicable and
    passes (a legend or FSM figure has no single punch number)."""
    if not meta:
        return True, []
    killer = str(meta.get("killerNumber") or "").strip()
    if not killer:
        return True, []
    # Compare on a punctuation/whitespace-insensitive squeeze so "~1000x" matches
    # "~1000x" regardless of surrounding spaces or a <br/> split.
    def squeeze(s: str) -> str:
        return re.sub(r"\s+", "", s).lower()
    if squeeze(killer) in squeeze(mmd):
        return True, []
    # Fall back to the most distinctive token (the longest alnum run) of the punch.
    tokens = [t for t in re.split(r"[^0-9A-Za-z.%x×]+", killer) if len(t) >= 3]
    if tokens and any(squeeze(t) in squeeze(mmd) for t in tokens):
        return True, []
    return False, [f"killer number '{killer}' not found in the .mmd"]


def check_meta_faithful(meta: dict | None) -> tuple[bool, list[str]]:
    """Carry the curated human judgment so the grader never scores a figure ABOVE
    its hand-reviewed bar. A figure with no _meta record is graded on its own
    static merits (neutral pass) rather than penalized for missing curation."""
    if not meta:
        return True, []
    faithful = bool(meta.get("faithful", True))
    syntax_ok = bool(meta.get("syntaxOk", True))
    if faithful and syntax_ok:
        return True, []
    bad = []
    if not faithful:
        bad.append("_meta marks faithful=false")
    if not syntax_ok:
        bad.append("_meta marks syntaxOk=false")
    return False, bad


# --------------------------------------------------------------------------------
# render checks (.svg)
# --------------------------------------------------------------------------------

def check_text_renderable(svg: str) -> tuple[bool, list[str]]:
    """The #1 whitespace bug (RENDERING-NOTE): a default mermaid SVG emits every
    label as a <foreignObject> HTML span, which only a browser renders -- so the
    image is BLANK in IDE previews, Slack/email thumbnails, PDFs. A correctly
    rendered (htmlLabels:false) SVG carries real <text>/<tspan>."""
    n_text = len(re.findall(r"<text\b", svg)) + len(re.findall(r"<tspan\b", svg))
    n_fo = len(re.findall(r"<foreignObject\b", svg))
    if n_text > 0:
        return True, []
    if n_fo > 0:
        return False, [f"all {n_fo} labels are <foreignObject> (blank outside a browser); "
                       "re-render with htmlLabels:false"]
    return False, ["no <text>/<tspan> and no <foreignObject> -- empty render?"]


def check_intrinsic_size(svg: str) -> tuple[bool, list[str]]:
    """RENDERING-NOTE #2: a root <svg width="100%"> with only a viewBox (no pixel
    height) collapses to zero/percentage height in some viewers. A robust artifact
    pins an explicit pixel width AND height."""
    m = re.search(r"<svg\b[^>]*>", svg)
    if not m:
        return False, ["no root <svg> element"]
    root = m.group(0)
    has_pct_width = 'width="100%"' in root
    has_px_height = bool(re.search(r'height="\d', root))
    if has_pct_width and not has_px_height:
        return False, ['root <svg> has width="100%" and no pixel height (collapses in '
                       "some viewers); pin width/height from the viewBox"]
    if not has_px_height and "viewBox" in root and 'width="100%"' in root:
        return False, ["root <svg> has no intrinsic height"]
    return True, []


# --------------------------------------------------------------------------------
# per-figure grade
# --------------------------------------------------------------------------------

def grade_figure(base: str, visuals_dir: Path, meta_idx: dict[str, dict]) -> dict:
    mmd_path = visuals_dir / f"{base}.mmd"
    svg_path = visuals_dir / f"{base}.svg"
    meta = meta_idx.get(base)
    mmd = mmd_path.read_text(encoding="utf-8") if mmd_path.exists() else ""
    has_svg = svg_path.exists()
    svg = svg_path.read_text(encoding="utf-8", errors="replace") if has_svg else ""

    checks: dict[str, bool] = {}
    reasons: list[str] = []

    def record(name: str, result: tuple[bool, list[str]]) -> None:
        ok, why = result
        checks[name] = ok
        if not ok:
            reasons.extend(f"{name}: {w}" for w in why)

    record("syntax_ok", check_syntax(mmd))
    record("class_coverage", check_class_coverage(mmd))
    record("killer_number", check_killer_number(mmd, meta))
    record("meta_faithful", check_meta_faithful(meta))
    if has_svg:
        record("text_renderable", check_text_renderable(svg))
        record("intrinsic_size", check_intrinsic_size(svg))

    active = list(SOURCE_CHECKS) + (list(RENDER_CHECKS) if has_svg else [])
    total_w = sum(WEIGHTS[c] for c in active)
    score = sum(WEIGHTS[c] for c in active if checks.get(c)) / total_w if total_w else 0.0

    return {
        "id": base,
        "score": round(score, 4),
        "has_svg": has_svg,
        "has_meta": meta is not None,
        "checks": checks,
        "fail_reasons": reasons,
    }


def discover_figures(visuals_dir: Path) -> list[str]:
    """Every base name that has a .mmd source, sorted deterministically."""
    return sorted(p.stem for p in visuals_dir.glob("*.mmd"))


def maybe_render(visuals_dir: Path) -> str | None:
    """Opt-in: invoke render.sh to (re)produce SVG/PNG before grading. Best-effort;
    returns an error string if it could not run (the grade proceeds on whatever
    artifacts already exist)."""
    render = visuals_dir / "render.sh"
    if not render.exists():
        return "render.sh not found"
    try:
        proc = subprocess.run(
            ["bash", str(render)], cwd=str(visuals_dir),
            capture_output=True, text=True, timeout=900,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return f"render failed to start: {e}"
    if proc.returncode != 0:
        return f"render exited {proc.returncode}: {proc.stderr.strip()[:200]}"
    return None


def grade_deck(visuals_dir: Path, floor: float, render: bool = False) -> dict:
    figures_meta = _meta_index(visuals_dir)
    render_note = None
    if render:
        render_note = maybe_render(visuals_dir)
    bases = discover_figures(visuals_dir)
    figures = [grade_figure(b, visuals_dir, figures_meta) for b in bases]
    n = len(figures)
    mean = round(sum(f["score"] for f in figures) / n, 4) if n else 0.0
    below = [f["id"] for f in figures if f["score"] < floor]
    report = {
        "schema": SCHEMA,
        "visuals_dir": str(visuals_dir.relative_to(ROOT)) if _under(visuals_dir, ROOT)
        else str(visuals_dir),
        "aggregate": {
            "mean_score": mean,
            "n": n,
            "floor": floor,
            "n_below_floor": len(below),
            "below_floor": below,
        },
        "figures": figures,
    }
    if render_note is not None:
        report["render_note"] = render_note
    return report


def _under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def print_table(report: dict) -> None:
    agg = report["aggregate"]
    print(f"visual-gen grade  mean={agg['mean_score']:.3f}  n={agg['n']}  "
          f"floor={agg['floor']:.2f}  below_floor={agg['n_below_floor']}")
    print("-" * 72)
    for f in sorted(report["figures"], key=lambda r: r["score"]):
        flag = " " if f["score"] >= agg["floor"] else "*"
        failed = ",".join(k for k, v in f["checks"].items() if not v) or "-"
        print(f"{flag} {f['score']:.3f}  {f['id']:<32}  fail:{failed}")
    if agg["n_below_floor"]:
        print("-" * 72)
        print(f"{agg['n_below_floor']} figure(s) below the {agg['floor']:.2f} floor "
              "(ACTION: re-render or fix the source).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Static quality grader for the visuals deck.")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help="visuals directory (default: fleet-bite-test/visuals)")
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR,
                    help=f"below-floor threshold (default {DEFAULT_FLOOR})")
    ap.add_argument("--render", action="store_true",
                    help="render .mmd->.svg via render.sh before grading (needs node)")
    ap.add_argument("--json", action="store_true", help="emit the JSON report")
    ap.add_argument("--out", type=Path, help="also write the JSON report to this path")
    args = ap.parse_args(argv)

    visuals_dir = args.dir
    if not visuals_dir.is_dir():
        print(f"[ERROR] visuals dir not found: {visuals_dir}", file=sys.stderr)
        return 2

    report = grade_deck(visuals_dir, args.floor, render=args.render)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
