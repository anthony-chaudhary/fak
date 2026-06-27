#!/usr/bin/env python3
"""intent_literal_scorecard  -  the intent-vs-literal / metric-honesty stick.

The lesson this generalizes (from the PM-review "coverage 100% reads as market
coverage" trap): every number a surface reports carries a LITERAL definition (what
it actually measures) but also INVITES an intent (what a reader assumes its name
means). When the two diverge  -  worst of all when the denominator is SELF-REFERENTIAL
("how much we do of X" where X is OUR own positioned set, not the external market)  -
a reader walks away believing a thing the metric never claimed: that "coverage 100%"
means we cover the field, that "grade A" means we win, that a "closure_rate" is
throughput, that a "hit_ratio" is a cache hit.

That is a distinct axis from the two sibling honesty sticks, and this scorecard is
careful not to duplicate them:
  - concept_disambiguation grades whether two similar NAMES blur (cache vs vCache).
  - conflation grades a number's PROVENANCE (who owns it: WITNESSED vs OBSERVED).
  - intent_literal grades whether one metric's NAME invites an intent its DEFINITION
    does not deliver  -  and, if it does diverge, whether the surface DISCLOSES the gap
    where a reader sees it.

A divergence is not a defect if it is disclosed: the gateway HELP text says a vDSO
ratio is "over kernel submissions", the in-kernel gauge says "last-request gauge, not
a cumulative counter", the scorecards carry a "Read this right" fence, dispatch-status
prints the closure_rate formula inline. The disciplined tree already keeps that line,
so the honest debt is ZERO  -  and this scorecard's durable job is to LOCK it: add a
divergent metric without a disclosure fence, or delete an existing fence, and the debt
rises and the ratchet reds.

It is a DATA-DIR scorecard cross-checked against the tree, so it cannot be gamed by
editing a JSON file: each row names a metric token that MUST appear in its surface, and
any divergent-denominator row MUST name a disclosure phrase that LITERALLY appears in
that same surface. The facts are in the Go source and the docs; the rows only position
the reader-intent judgment on top.

KPIs (pure over the rows + the surface texts):
  - well_formed (HARD): a row is shaped like a positioned metric; ids unique; the
    surface, kind, denominator, and verdict are in the closed vocabularies.
  - grounded (HARD): the metric's grounding token actually appears in its surface file
    (you cannot position a metric that does not exist).
  - intent_disclosed (HARD): a row whose denominator is DIVERGENT (self-referential /
    subset / snapshot / observed) must name a disclosure phrase that appears in the
    surface; an undisclosed divergence is one debt  -  a reader will misread the metric.
  - verdict_consistency (HARD): the declared verdict (clear / disclosed / misleading)
    must match the verdict re-derived from evidence; an overclaim is one debt.
  - coverage_surfaced (SOFT): metric names in the watched Go surfaces that carry an
    intent-laden token (ratio/rate/hits/total/...) but are not yet positioned by a row
    are advisory  -  the forward backlog, retired over time, never HARD debt.

Usage:
    python tools/intent_literal_scorecard.py                  # terminal work-list
    python tools/intent_literal_scorecard.py --json           # control-pane payload
    python tools/intent_literal_scorecard.py --compare b.json # prove the debt moved
    python tools/intent_literal_scorecard.py --markdown-dir docs/intent-literal-scorecard
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DATA_DIR_REL = "tools/intent_literal_scorecard.data"
GENERATED_DOC_REL = "docs/intent-literal-scorecard"
SCHEMA = "fak-intent-literal-scorecard/1"

# Closed vocabularies (DOCTRINE, not data-defined). A row declaring a value outside
# these sets is MALFORMED  -  the well_formed KPI refuses it.
KINDS = {"gauge", "counter", "ratio", "coverage", "headline", "scalar"}
DENOMINATORS = {
    "external-universe",  # the denominator IS the external/market universe (honest by shape)
    "self-referential",   # "how much WE do of X" where X is our own positioned set
    "subset",             # measured over a sub-population, not the whole the name implies
    "snapshot",           # a point-in-time / last-value, not the aggregate the name implies
    "observed",           # relayed from an external party, not a fak-authored count
    "absolute",           # an absolute count whose name matches its meaning
}
# Denominators whose NAME tends to invite a wider/other intent than the DEFINITION
# delivers  -  these MUST be disclosed at the surface or they are debt.
DIVERGENT = {"self-referential", "subset", "snapshot", "observed"}
VERDICTS = {"clear", "disclosed", "misleading"}

REQUIRED_FIELDS = [
    "id", "canonical", "surface", "kind", "invited_intent", "literal",
    "denominator", "grounding", "disclosure", "verdict", "gaps",
]

# Intent-laden tokens in a metric NAME that invite a reader to expect a cache-hit /
# quality / coverage reading the definition may not deliver. Used ONLY by the SOFT
# coverage lens to surface watched-surface metrics not yet positioned  -  never to create
# HARD debt. Deliberately EXCLUDES "_total" / "per_second": an absolute count or a true
# rate whose name matches its meaning is honest, not a candidate for over-reading.
INTENT_NAME_TOKENS = [
    "hit_ratio", "hit_rate", "_ratio", "_rate", "_hits", "hits_", "coverage", "_score",
]

# How the SOFT lens finds metric names in a Go reporting surface: the first string
# argument of writeHelpType/writeCounter/writeGauge is the metric name.
_METRIC_DECL = re.compile(r'write(?:HelpType|Counter|Gauge)\(\s*&?\w+,\s*"([A-Za-z0-9_]+)"')

KPI_WEIGHTS = {
    "well_formed": 0.20,
    "grounded": 0.20,
    "intent_disclosed": 0.40,   # the core axis
    "verdict_consistency": 0.20,
}


# ---- io / loading (the impure shell, kept thin) -----------------------------------------

def repo_root(start: Path | None = None) -> Path:
    return (start or Path(__file__)).resolve().parent.parent


def _read(root: Path, rel: str) -> str:
    try:
        return (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def load_data(data_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge _meta.json + every rows-*.json into (meta, rows), tagging each row with the
    rows-file it came from so a defect can be traced back to its source file."""
    meta: dict[str, Any] = {}
    meta_path = data_dir / "_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    rows: list[dict[str, Any]] = []
    for rows_path in sorted(data_dir.glob("rows-*.json")):
        try:
            doc = json.loads(rows_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for row in doc.get("rows", []):
            if isinstance(row, dict):
                row = dict(row)
                row.setdefault("_source_file", rows_path.name)
                rows.append(row)
    return meta, rows


# ---- pure helpers -----------------------------------------------------------------------

def _present(text: str, needle: str) -> bool:
    """A disclosure / grounding phrase is present if it appears in the surface text
    (case-insensitive, whitespace-collapsed) so prose wrapping does not hide it."""
    if not needle:
        return False
    hay = " ".join(text.split()).lower()
    return " ".join(needle.split()).lower() in hay


def _kpi(name: str, score: float, detail: str,
         defects: list[str] | None = None, soft: list[str] | None = None) -> dict[str, Any]:
    return {"kpi": name, "group": "honesty", "score": round(float(score), 1),
            "detail": detail, "defects": defects or [], "soft": soft or []}


def expected_verdict(row: dict[str, Any], disclosed: bool) -> str:
    """Re-derive the verdict from evidence so a row cannot self-report a kinder one.

    non-divergent denominator  -> clear     (the name matches the meaning)
    divergent + disclosed       -> disclosed (the gap is named where a reader sees it)
    divergent + not disclosed   -> misleading
    """
    if row.get("denominator") not in DIVERGENT:
        return "clear"
    return "disclosed" if disclosed else "misleading"


def _surface_index(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in meta.get("surfaces", []) if isinstance(s, dict) and s.get("id")}


# ---- KPIs (pure over rows + surface texts) ----------------------------------------------

def kpi_well_formed(rows: list[dict[str, Any]], surfaces: dict[str, dict[str, Any]]) -> dict[str, Any]:
    defects: list[str] = []
    seen: set[str] = set()
    for row in rows:
        rid = str(row.get("id") or "?")
        for f in REQUIRED_FIELDS:
            if f not in row:
                defects.append(f"{rid}: missing required field '{f}'")
        if rid in seen:
            defects.append(f"{rid}: duplicate id")
        seen.add(rid)
        if row.get("surface") not in surfaces:
            defects.append(f"{rid}: surface '{row.get('surface')}' is not declared in _meta.json")
        if row.get("kind") not in KINDS:
            defects.append(f"{rid}: kind '{row.get('kind')}' not in {sorted(KINDS)}")
        if row.get("denominator") not in DENOMINATORS:
            defects.append(f"{rid}: denominator '{row.get('denominator')}' not in {sorted(DENOMINATORS)}")
        if row.get("verdict") not in VERDICTS:
            defects.append(f"{rid}: verdict '{row.get('verdict')}' not in {sorted(VERDICTS)}")
        if not isinstance(row.get("gaps"), list):
            defects.append(f"{rid}: gaps must be a list")
    n = max(len(rows), 1)
    score = 100.0 if not defects else max(0.0, 100.0 * (1 - len(defects) / (n * 3)))
    return _kpi("well_formed", score,
                f"{len(rows) - len({d.split(':')[0] for d in defects})}/{len(rows)} rows well-formed"
                if defects else f"all {len(rows)} rows well-formed", defects)


def kpi_grounded(rows: list[dict[str, Any]], texts: dict[str, str]) -> dict[str, Any]:
    defects: list[str] = []
    for row in rows:
        rid = str(row.get("id") or "?")
        text = texts.get(row.get("surface", ""), "")
        if not _present(text, str(row.get("grounding") or "")):
            defects.append(f"{rid}: grounding token '{row.get('grounding')}' not found in surface "
                           f"'{row.get('surface')}' (a positioned metric must exist in the tree)")
    grounded = len(rows) - len(defects)
    score = 100.0 if not defects else max(0.0, 100.0 * grounded / max(len(rows), 1))
    return _kpi("grounded", score, f"{grounded}/{len(rows)} metric tokens present in their surface", defects)


def kpi_intent_disclosed(rows: list[dict[str, Any]], texts: dict[str, str]) -> dict[str, Any]:
    """The core axis: a divergent-denominator metric must disclose its true meaning where a
    reader sees it, or a reader will take the name at face value."""
    defects: list[str] = []
    divergent = 0
    for row in rows:
        if row.get("denominator") not in DIVERGENT:
            continue
        divergent += 1
        rid = str(row.get("id") or "?")
        text = texts.get(row.get("surface", ""), "")
        disc = str(row.get("disclosure") or "")
        if not disc:
            defects.append(f"{rid}: denominator '{row.get('denominator')}' diverges from the name "
                           f"\"{row.get('canonical')}\" but the row names no disclosure phrase")
        elif not _present(text, disc):
            defects.append(f"{rid}: disclosure \"{_clip(disc)}\" is not present in surface "
                           f"'{row.get('surface')}'  -  the divergence is undisclosed where a reader sees it")
    ok = divergent - len(defects)
    score = 100.0 if not defects else max(0.0, 100.0 * ok / max(divergent, 1))
    return _kpi("intent_disclosed", score,
                f"{ok}/{divergent} divergent metrics disclose their true meaning at the surface", defects)


def kpi_verdict_consistency(rows: list[dict[str, Any]], texts: dict[str, str]) -> dict[str, Any]:
    defects: list[str] = []
    for row in rows:
        rid = str(row.get("id") or "?")
        text = texts.get(row.get("surface", ""), "")
        disclosed = _present(text, str(row.get("disclosure") or ""))
        want = expected_verdict(row, disclosed)
        got = row.get("verdict")
        if got != want:
            defects.append(f"{rid}: declares verdict '{got}' but evidence says '{want}' "
                           f"(denominator '{row.get('denominator')}', "
                           f"disclosure {'present' if disclosed else 'absent'})")
    ok = len(rows) - len(defects)
    score = 100.0 if not defects else max(0.0, 100.0 * ok / max(len(rows), 1))
    return _kpi("verdict_consistency", score, f"{ok}/{len(rows)} verdicts match the re-derived evidence", defects)


def kpi_coverage_surfaced(rows: list[dict[str, Any]], meta: dict[str, Any],
                          texts: dict[str, str]) -> dict[str, Any]:
    """SOFT: intent-laden metric names in the watched Go surfaces not yet positioned by a
    row. Advisory backlog (the forward gradient), never HARD debt  -  so the scorecard cannot
    be gamed shut, but a clean tree is not punished for metrics whose names are honest."""
    positioned = {str(r.get("grounding") or "").lower() for r in rows}
    positioned |= {str(r.get("canonical") or "").lower() for r in rows}
    soft: list[str] = []
    for sid, s in _surface_index(meta).items():
        if not str(s.get("path", "")).endswith(".go"):
            continue
        text = texts.get(sid, "")
        for name in sorted(set(_METRIC_DECL.findall(text))):
            low = name.lower()
            if not any(tok in low for tok in INTENT_NAME_TOKENS):
                continue
            if low in positioned or any(low in p or p in low for p in positioned if len(p) >= 6):
                continue
            soft.append(f"{sid}: intent-laden metric '{name}' carries a name a reader may "
                        f"over-read; not yet positioned by a row")
    score = 100.0 if not soft else max(60.0, 100.0 - 2.0 * len(soft))
    return _kpi("coverage_surfaced", score,
                "every intent-laden watched metric is positioned" if not soft
                else f"{len(soft)} intent-laden watched metric(s) not yet positioned", soft=soft)


def _clip(s: str, n: int = 70) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "..."


# ---- fold + render ----------------------------------------------------------------------

def grade_letter(score: float) -> str:
    if score >= 95: return "A"
    if score >= 85: return "B"
    if score >= 75: return "C"
    if score >= 60: return "D"
    return "F"


def run(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    meta, rows = load_data(root / DATA_DIR_REL)
    surfaces = _surface_index(meta)
    texts = {sid: _read(root, str(s.get("path", ""))) for sid, s in surfaces.items()}

    kpis = [
        kpi_well_formed(rows, surfaces),
        kpi_grounded(rows, texts),
        kpi_intent_disclosed(rows, texts),
        kpi_verdict_consistency(rows, texts),
        kpi_coverage_surfaced(rows, meta, texts),
    ]
    hard = [k for k in kpis if k["kpi"] in KPI_WEIGHTS]
    composite = sum(KPI_WEIGHTS[k["kpi"]] * k["score"] for k in hard)
    debt = sum(len(k["defects"]) for k in kpis)
    soft_signals = sum(len(k["soft"]) for k in kpis)
    grade = grade_letter(composite)

    standing = {v: sum(1 for r in rows if r.get("verdict") == v) for v in sorted(VERDICTS)}
    verdict = "OK" if debt == 0 else "ACTION"
    if debt == 0:
        finding = ("every reported metric whose name invites a wider intent than it measures "
                   "discloses the gap at its surface")
        next_action = ("hold the line; position the next intent-laden metric (see the "
                       "coverage_surfaced advisory) and re-pin")
    else:
        worst = max(hard, key=lambda k: len(k["defects"]))
        finding = (f"{debt} intent-literal defect(s): a metric's name invites an intent its "
                   f"definition does not deliver, and the surface does not disclose the gap")
        next_action = (f"fix {worst['kpi']}: add the disclosure phrase to the surface or correct "
                       f"the row's verdict, then re-run")

    return {
        "schema": SCHEMA,
        "ok": debt == 0,
        "verdict": verdict,
        "finding": finding,
        "reason": "; ".join(d for k in kpis for d in k["defects"]) or "clean",
        "next_action": next_action,
        "workspace": str(root),
        "corpus": {
            "score": round(composite, 1),
            "grade": grade,
            "intent_literal_debt": debt,
            "soft_signals": soft_signals,
            "rows": len(rows),
            "surfaces": len(surfaces),
            "divergent": sum(1 for r in rows if r.get("denominator") in DIVERGENT),
            "standing": standing,
            "as_of": meta.get("meta", {}).get("as_of", ""),
            "fak_version": meta.get("meta", {}).get("fak_version", ""),
        },
        "kpis": kpis,
        "_data": {"rows": rows, "surfaces": list(surfaces.values())},
    }


def render(payload: dict[str, Any]) -> str:
    c = payload["corpus"]
    out = [f"intent-literal scorecard: {c['grade']} (score {c['score']}, "
           f"intent_literal_debt {c['intent_literal_debt']})",
           f"  {payload['finding']}",
           f"  {c['rows']} rows over {c['surfaces']} surface(s); {c['divergent']} divergent; "
           f"standing {c['standing']}", ""]
    for k in payload["kpis"]:
        mark = "ok " if not k["defects"] and not k["soft"] else "!! " if k["defects"] else "~  "
        out.append(f"  {mark}{k['kpi']:<20} {k['score']:5.1f}  {k['detail']}")
        for d in k["defects"]:
            out.append(f"       HARD  {d}")
        for s in k["soft"][:12]:
            out.append(f"       soft  {s}")
    out += ["", f"  next: {payload['next_action']}"]
    return "\n".join(out)


def _front_matter(title: str, desc: str) -> list[str]:
    return ["---", f'title: "{title}"', f'description: "{desc}"', "---", ""]


def render_doc(payload: dict[str, Any], stamp: str = "") -> str:
    c = payload["corpus"]
    rows = payload["_data"]["rows"]
    lines = _front_matter(
        "fak intent-literal scorecard - does a metric's name invite an intent its definition delivers",
        "Inward metric-honesty scorecard: each reported metric positioned on whether its NAME "
        "invites a wider intent than its LITERAL definition delivers (the self-referential-denominator "
        "trap: 'how much we do of X' where X is our own set, not the market) - and whether the surface "
        "discloses the gap. One driven number: intent-literal-debt.")
    lines += [
        "# Intent-literal / metric-honesty scorecard",
        "",
        "_Auto-generated by `tools/intent_literal_scorecard.py`. Do not hand-edit; re-run the tool._",
        "",
        f"**Grade {c['grade']}** - score {c['score']} - intent_literal_debt **{c['intent_literal_debt']}** - "
        f"{c['rows']} positioned metric(s) over {c['surfaces']} surface(s) - "
        f"{c['divergent']} with a divergent denominator - {c['soft_signals']} coverage advisory.",
        "",
        "The law: a metric whose NAME invites a wider intent than its LITERAL definition delivers  -  "
        "worst of all a SELF-REFERENTIAL denominator (\"how much we do of X\" where X is our own "
        "positioned set, not the external market)  -  must DISCLOSE that gap where a reader sees it. "
        "A disclosed divergence is honest; an undisclosed one is debt. This is distinct from "
        "concept-disambiguation (do two NAMES blur) and conflation (a number's PROVENANCE).",
        "",
        f"Standing: {', '.join(f'{k} {v}' for k, v in c['standing'].items())}.",
        "",
        "| metric | surface | invited intent | literal meaning | denominator | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r.get("surface", ""), r.get("id", ""))):
        lines.append(
            f"| `{r.get('canonical')}` | {r.get('surface')} | {_clip(str(r.get('invited_intent')), 80)} "
            f"| {_clip(str(r.get('literal')), 90)} | {r.get('denominator')} | {r.get('verdict')} |")
    work = [d for k in payload["kpis"] for d in k["defects"]]
    if work:
        lines += ["", "## Work list (HARD - undisclosed divergences / overclaims)", ""]
        lines += [f"- {d}" for d in work]
    soft = [s for k in payload["kpis"] for s in k["soft"]]
    if soft:
        lines += ["", "## Coverage advisory (SOFT - intent-laden metrics not yet positioned)", ""]
        lines += [f"- {s}" for s in soft]
    if stamp:
        lines += ["", f"<!-- {stamp} -->"]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="intent-vs-literal / metric-honesty scorecard")
    ap.add_argument("--json", action="store_true", help="emit the control-pane payload")
    ap.add_argument("--compare", metavar="BASELINE", help="print the debt delta vs a prior --json")
    ap.add_argument("--markdown", action="store_true", help="emit the committed snapshot body")
    ap.add_argument("--markdown-dir", metavar="DIR", help="regenerate the committed doc folder")
    ap.add_argument("--workspace", default="", help="repo root (default: inferred)")
    ap.add_argument("--stamp", default="", help="optional traceability stamp embedded in the doc")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = run(root)

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0 if payload["ok"] else 1
    if args.markdown:
        print(render_doc(payload, args.stamp))
        return 0 if payload["ok"] else 1
    if args.markdown_dir:
        out_dir = Path(args.markdown_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "README.md").write_text(render_doc(payload, args.stamp), encoding="utf-8")
        print(f"wrote {out_dir / 'README.md'} (intent_literal_debt {payload['corpus']['intent_literal_debt']})")
        return 0 if payload["ok"] else 1
    if args.compare:
        try:
            prior = json.loads(Path(args.compare).read_text(encoding="utf-8"))
            pd = prior.get("corpus", {}).get("intent_literal_debt")
        except (OSError, json.JSONDecodeError):
            pd = None
        cur = payload["corpus"]["intent_literal_debt"]
        print(render(payload))
        if pd is not None:
            delta = cur - pd
            verdict = "improved" if delta < 0 else "regressed" if delta > 0 else "flat"
            print(f"\n  compare: intent_literal_debt {pd} -> {cur} ({verdict} by {abs(delta)})")
        return 0 if payload["ok"] else 1
    print(render(payload))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
