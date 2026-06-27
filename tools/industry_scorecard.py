#!/usr/bin/env python3
"""Industry scorecard — the measuring stick for fak's place in the LLM-serving field.

The sibling scorecards all point *inward*: ``repo_hygiene_scorecard`` grades the
tree's shape, ``code_quality_scorecard`` grades the Go module, ``doc_appeal_scorecard``
grades one doc's prose. None of them point *outward* — at the thing a buyer, a
reviewer, or a skeptic actually asks: **where does fak stand against the industry?**

The hard lesson this redesign encodes: an outward scorecard must start from the
INDUSTRY, not from "what we happened to measure." A stick that only grades the
comparisons fak already ran is a highlight reel — it silently omits the dimensions
the field actually competes on (continuous batching, FP8/FP4 quant, speculative
decoding, disaggregated prefill, multi-LoRA, tokens-per-watt, …). So the source of
truth is now an **industry-first taxonomy**: a researched catalog of the dimensions
that matter in LLM inference-serving + agent infrastructure, each with the current
SOTA bar and a dated source. fak is then positioned honestly on EVERY dimension —
and for most of them the honest answer is a named gap, not a win.

That turns the user's real complaint ("we're missing 90% of what matters") into two
numbers an agent can drive:

  COVERAGE      — of the industry dimensions that matter, how many has the scorecard
                  even CONSIDERED + positioned fak against? (drive toward 100%)
    coverage_debt    in-scope taxonomy dimensions with no fak position row

  PARITY-DEBT   — of the comparisons that DO exist, how many are dishonest,
                  incomplete, or unsourced? (keep at 0)
    well_formed       a row is shaped like a comparison; its category + dim are valid
    competitor_named  every row names a concrete competitor, never a blank baseline
    axis_coverage     the contract's must-show regimes (incl. the single-stream
                      LOSSES) are present, not buried
    baseline_sota     no win against a NAIVE strawman — gain is vs the tuned / SOTA
    verdict_consistency  the verdict MATCHES the ratio (oracle→parity, ceiling≠lead)
    apples_disclosed  a cross-device / cross-precision row says what differs
    fak_traced        a SHIPPED claim traces to a commit / artifact / authority doc
    competitor_sourced  the competitor number cites where it came from

Two freshness lenses keep it current as the field — and fak — move:
    freshness          fak ``measured_on`` staleness (re-measure when a node is free)
    industry_freshness taxonomy ``source_date`` / competitor ``last_reviewed`` drift —
                       the SOTA bar a competitor published 8 months ago wants a re-check

The composite **score** blends honesty (of the rows that exist) with coverage (of the
industry that matters), so an incomplete map costs you grade even when every row you
DO have is honest. The companion ``/industry-score`` skill runs this, fills the
worst coverage gap or retires the worst parity-debt, re-runs to prove the drop, and
keeps the front-door competitive story complete + honest as new claims land and the
field moves. It folds into ``scorecard_control_pane`` via ``corpus.parity_debt``.

Deterministic + read-only over the data (two clones at one commit score identically);
the only disk writes are the generated doc folder under ``--markdown-dir``. The source
of truth is a DIRECTORY of small JSON files (modular, one concern per file) so the
taxonomy, the competitor registry, and each category's rows evolve independently::

    tools/industry_scorecard.data/
      _meta.json          meta + the declared category vocabulary (id → group)
      _taxonomy.json      the industry-first dimension catalog (what matters + SOTA bar)
      _competitors.json   the SOTA-system registry (versions + last_reviewed)
      rows-*.json         fak's position rows, grouped by category (one file per group)

Run from the repo ROOT::

    python tools/industry_scorecard.py                 # human scorecard
    python tools/industry_scorecard.py --json          # machine payload (the loop / control-pane)
    python tools/industry_scorecard.py --gaps          # the coverage backlog: dims fak hasn't positioned/measured
    python tools/industry_scorecard.py --stale         # the industry-drift backlog: SOTA bars due a re-check
    python tools/industry_scorecard.py --markdown-dir docs/industry-scorecard   # regenerate the doc folder
    python tools/industry_scorecard.py --compare base.json   # prove the debt moved
    python tools/industry_scorecard.py --verify-sources      # re-check fak numbers vs committed artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA = "fak-industry-scorecard/2"
DATA_DIR_REL = "tools/industry_scorecard.data"
LEGACY_DATA_REL = "tools/industry_scorecard.data.json"
GENERATED_DOC_DIR = "docs/industry-scorecard"

# ---------------------------------------------------------------------------
# Closed vocabularies. `category` is NOT hard-coded here — it is DATA-defined
# (declared in _meta.json) so the taxonomy can grow as the field does; a row /
# dimension whose category is not declared is MALFORMED, never silently skipped.
# The honesty vocabularies below ARE the doctrine and stay fixed.
# ---------------------------------------------------------------------------

# competitor_class — what KIND of baseline a row compares against. `naive` is in
# the vocabulary ONLY so the scorecard can REFUSE it: the industry question is the
# gain over the best already-shipped alternative, never a strawman.
CLASSES = {
    "sota": "the best already-shipped system on this axis (vLLM / SGLang / TensorRT-LLM / llama.cpp)",
    "tuned": "a tuned single-tenant baseline (warm per-agent KV) — the honest floor",
    "theoretical-ceiling": "fak's own exact upper bound (a ratio it approaches, never beats)",
    "reference-oracle": "a correctness oracle (HF transformers / cuBLAS) — match = parity",
    "naive": "a strawman (re-prefill every turn) — REFUSED by this scorecard",
}
WEAK_CLASSES = {"naive"}
VERDICTS = {"lead", "parity", "trails", "no-claim"}
STATUSES = {"shipped", "in-flight", "projected", "stub"}
# A no-claim / floating row is excused from a hard trace only when it SAYS it is
# not yet shipped — which keeps the absence honest.
UNSHIPPED = {"in-flight", "projected", "stub"}
# A verdict that asserts a measured win/loss needs a number behind it; an oracle
# match and an honest no-claim gap do not.
MEASURED_VERDICTS = {"lead", "trails"}

# Verdict thresholds on the favorability ratio (ratio > 1 ⇒ fak ahead).
LEAD_MIN = 1.05
PARITY_LO = 0.95
CEILING_PARITY = 0.90

GROUPS = ("structure", "completeness", "honesty", "traceability")
KPI_GROUP: dict[str, str] = {
    "well_formed": "structure",
    "competitor_named": "completeness",
    "axis_coverage": "completeness",
    "baseline_sota": "honesty",
    "verdict_consistency": "honesty",
    "apples_disclosed": "honesty",
    "fak_traced": "traceability",
    "competitor_sourced": "traceability",
    "freshness": "traceability",
}
KPI_WEIGHTS: dict[str, float] = {
    "well_formed": 0.10,
    "competitor_named": 0.10,
    "axis_coverage": 0.13,
    "baseline_sota": 0.15,
    "verdict_consistency": 0.20,
    "apples_disclosed": 0.10,
    "fak_traced": 0.10,
    "competitor_sourced": 0.07,
    "freshness": 0.05,
}
# The composite blends the honesty of the rows that EXIST with how much of the
# industry the scorecard has even considered. An incomplete map costs grade.
HONESTY_WEIGHT = 0.60
COVERAGE_WEIGHT = 0.40


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


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _nonempty(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def favorability(fak: Any, comp: Any, higher_is_better: bool) -> float | None:
    """Ratio of fak's standing to the competitor's; > 1 means fak is ahead.

    For a higher-is-better metric (tok/s, hit-rate) that is fak/comp; for a
    lower-is-better metric (minutes, latency) it inverts to comp/fak, so the
    "> 1 ⇒ ahead" reading holds either way. None when not computable."""
    f, c = _num(fak), _num(comp)
    if f is None or c is None:
        return None
    if higher_is_better:
        return None if c == 0 else f / c
    return None if f == 0 else c / f


def _has_numeric_basis(row: dict[str, Any]) -> bool:
    """A row carries a measurable comparison: a value pair, or a value + band."""
    rng = row.get("competitor_range")
    has_pair = _num(row.get("fak_value")) is not None and _num(row.get("competitor_value")) is not None
    has_band = (_num(row.get("fak_value")) is not None
                and isinstance(rng, list) and len(rng) == 2)
    return has_pair or has_band


def expected_verdict(row: dict[str, Any]) -> tuple[str | None, str]:
    """The verdict the row's evidence IMPLIES, with a one-line rationale.

    Returns (verdict_or_None, why). A None verdict means the numbers cannot
    mechanically decide it (treated as a soft 'unverifiable', not a hard miss) —
    but the class-driven cases (oracle, ceiling, no-claim) always decide."""
    cls = row.get("competitor_class")
    status = row.get("status")
    hib = bool(row.get("higher_is_better"))
    fak = row.get("fak_value")
    rng = row.get("competitor_range")

    # A correctness oracle: matching it IS parity by construction.
    if cls == "reference-oracle":
        return "parity", "matching a correctness oracle is parity by construction"

    # A declared gap: honest only when fak genuinely ships nothing on the axis.
    if row.get("verdict") == "no-claim":
        if status in UNSHIPPED:
            return "no-claim", f"an unbuilt axis (status={status}) is an honest gap, not a loss"
        return None, f"a no-claim verdict needs an unbuilt status ({sorted(UNSHIPPED)}), got {status}"

    # A band (e.g. SGLang's published 50–99% hit-rate): inside ⇒ parity.
    lo = hi = None
    if isinstance(rng, list) and len(rng) == 2:
        lo, hi = min(rng), max(rng)
    f = _num(fak)
    if lo is not None and f is not None:
        if lo <= f <= hi:
            return "parity", f"fak {f} sits inside the published [{lo}, {hi}] band"
        beyond = (f > hi) if hib else (f < lo)
        return ("lead" if beyond else "trails"), "fak sits outside the band"

    r = favorability(fak, row.get("competitor_value"), hib)
    if r is None:
        return None, "no comparable numbers to derive a verdict from"
    if cls == "theoretical-ceiling":
        return ("parity" if r >= CEILING_PARITY else "trails"), \
            f"{r:.0%} of an own-ceiling ({'≥' if r >= CEILING_PARITY else '<'}{CEILING_PARITY:.0%})"
    if r >= LEAD_MIN:
        return "lead", f"fak ahead by {r:.2f}×"
    if r >= PARITY_LO:
        return "parity", f"within ±5% ({r:.2f}×)"
    return "trails", f"fak behind ({r:.2f}×)"


def parse_date(s: Any) -> date | None:
    """Parse YYYY-MM-DD or YYYY-MM (industry source dates are often month-only)."""
    if not isinstance(s, str):
        return None
    parts = s.strip().split("-")
    try:
        if len(parts) == 2:
            y, m = (int(x) for x in parts)
            return date(y, m, 1)
        y, m, d = (int(x) for x in parts)
        return date(y, m, d)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Per-KPI pure checks (parity-debt = honesty of the rows that EXIST). Each returns
#   {kpi, group, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of parity-debt; soft = score-only judgment nudges.
# ---------------------------------------------------------------------------

def kpi_well_formed(rows: list[dict[str, Any]], categories: set[str],
                    dim_ids: set[str]) -> dict[str, Any]:
    """A row must be shaped like a comparison: required fields present, every
    enum inside its closed vocabulary, its category declared, its dim_id resolving
    to a taxonomy dimension, and numbers present when it asserts a measured win or
    loss. A malformed row can't be honestly graded, so it is hard debt."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id") or f"row[{i}]"
        if not _nonempty(r.get("id")):
            defects.append(f"{rid}: missing id")
        if categories and r.get("category") not in categories:
            defects.append(f"{rid}: category {r.get('category')!r} not declared in _meta.json categories")
        if dim_ids and r.get("dim_id") not in dim_ids:
            defects.append(f"{rid}: dim_id {r.get('dim_id')!r} does not resolve to a taxonomy dimension")
        if r.get("competitor_class") not in CLASSES:
            defects.append(f"{rid}: competitor_class {r.get('competitor_class')!r} not in {sorted(CLASSES)}")
        if r.get("verdict") not in VERDICTS:
            defects.append(f"{rid}: verdict {r.get('verdict')!r} not in {sorted(VERDICTS)}")
        if r.get("status") not in STATUSES:
            defects.append(f"{rid}: status {r.get('status')!r} not in {sorted(STATUSES)}")
        if not isinstance(r.get("higher_is_better"), bool):
            defects.append(f"{rid}: higher_is_better must be a bool")
        if not _nonempty(r.get("unit")):
            defects.append(f"{rid}: missing unit")
        if parse_date(r.get("measured_on")) is None:
            defects.append(f"{rid}: missing/invalid measured_on (YYYY-MM-DD)")
        # A measured win/loss must carry a numeric basis — UNLESS it is an oracle
        # match (qualitative by construction) or an explicitly `qualitative` capability
        # comparison (a real lead/loss with no clean ratio, e.g. hardware breadth),
        # which must still disclose itself in a comparison_note (honesty, below).
        if (r.get("verdict") in MEASURED_VERDICTS
                and r.get("competitor_class") != "reference-oracle"
                and not r.get("qualitative")
                and not _has_numeric_basis(r)):
            defects.append(f"{rid}: verdict {r.get('verdict')!r} asserts a measured result "
                           f"but carries no fak_value + competitor_value (or band) and is not "
                           f"flagged qualitative")
        # A qualitative capability lead/loss must say what it compares (no ratio to check).
        if r.get("qualitative") and not _nonempty(r.get("comparison_note")):
            defects.append(f"{rid}: qualitative=true but no comparison_note — a capability "
                           f"lead/loss with no ratio must disclose what differs")
    return {"kpi": "well_formed", "group": "structure",
            "score": _clamp(100 - 12 * len(defects)),
            "detail": (f"{len(defects)} malformed field(s)" if defects
                       else f"all {len(rows)} rows well-formed"),
            "defects": defects, "soft": []}


def kpi_competitor_named(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every row must name a concrete competitor. A blank 'baseline' is the
    oldest benchmark dodge — you can't grade a comparison against nothing."""
    defects = [f"{r.get('id', i)}: no named competitor (state the SOTA / next-best alternative)"
               for i, r in enumerate(rows) if not _nonempty(r.get("competitor"))]
    return {"kpi": "competitor_named", "group": "completeness",
            "score": _clamp(100 - 15 * len(defects)),
            "detail": (f"{len(defects)} row(s) with no named competitor" if defects
                       else "every row names a concrete competitor"),
            "defects": defects, "soft": []}


def kpi_axis_coverage(rows: list[dict[str, Any]],
                      required: list[dict[str, Any]]) -> dict[str, Any]:
    """The contract's must-show regimes are present — and where it says a regime
    MUST carry a loss (single-stream), that loss is shown, not buried. This is the
    'show where you lose' rule as a mechanical check. (Industry-wide COVERAGE of
    the taxonomy is a separate, larger metric — see coverage_report.)"""
    defects: list[str] = []
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cat.setdefault(r.get("category"), []).append(r)
    for req in required:
        cat = req.get("category") or req.get("regime")
        got = by_cat.get(cat, [])
        need = int(req.get("min_rows", 1))
        if len(got) < need:
            defects.append(f"category '{cat}' has {len(got)} row(s), needs ≥{need} "
                           f"({req.get('why', '')})")
        must_v = req.get("must_include_verdict")
        if must_v and not any(g.get("verdict") == must_v for g in got):
            defects.append(f"category '{cat}' must include a '{must_v}' row "
                           f"(show the loss, don't bury it) — none found")
    return {"kpi": "axis_coverage", "group": "completeness",
            "score": _clamp(100 - 20 * len(defects)),
            "detail": (f"{len(defects)} contract-coverage gap(s)" if defects
                       else f"all {len(required)} contracted regimes covered"),
            "defects": defects, "soft": []}


def kpi_baseline_sota(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """No comparison may be against a NAIVE baseline. A 60× win over re-prefill is
    a different (weaker) claim than a 4× win over a tuned cache; conflating them is
    the inflation this refuses."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        if r.get("competitor_class") in WEAK_CLASSES:
            defects.append(f"{r.get('id', i)}: compares vs a NAIVE baseline "
                           f"({r.get('competitor')!r}) — restate the gain vs the tuned/SOTA "
                           f"alternative, or move this to a naive-context note")
    return {"kpi": "baseline_sota", "group": "honesty",
            "score": _clamp(100 - 20 * len(defects)),
            "detail": (f"{len(defects)} naive-baseline row(s)" if defects
                       else "every comparison is vs a tuned / SOTA / next-best baseline"),
            "defects": defects, "soft": []}


def kpi_verdict_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The stated verdict must match what the numbers imply. A row that claims
    'lead' while fak trails, an oracle marked anything but 'parity', a ceiling
    marked 'lead', or a 'no-claim' on a shipped axis — each is one overclaim, the
    single most important thing this scorecard catches."""
    defects: list[str] = []
    soft: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        declared = r.get("verdict")
        exp, why = expected_verdict(r)
        if exp is None:
            soft.append(f"{rid}: verdict '{declared}' not mechanically checkable ({why})")
            continue
        if declared != exp:
            defects.append(f"{rid}: claims '{declared}' but evidence implies '{exp}' — {why}")
    return {"kpi": "verdict_consistency", "group": "honesty",
            "score": _clamp(100 - 25 * len(defects) - min(10, 2 * len(soft))),
            "detail": (f"{len(defects)} verdict mismatch(es)" if defects
                       else f"every verdict matches its evidence ({len(soft)} unverifiable)"),
            "defects": defects, "soft": soft}


def kpi_apples_disclosed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A comparison that is NOT apples-to-apples (different device / precision /
    workload) must say so in a note. A hidden non-comparability is how a benchmark
    lies while every number is 'true'."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        if r.get("apples_to_apples") is False and not _nonempty(r.get("comparison_note")):
            defects.append(f"{r.get('id', i)}: apples_to_apples=false with no comparison_note — "
                           f"state what differs (device / precision / workload)")
    return {"kpi": "apples_disclosed", "group": "honesty",
            "score": _clamp(100 - 15 * len(defects)),
            "detail": (f"{len(defects)} undisclosed non-comparable row(s)" if defects
                       else "every non-comparable row discloses what differs"),
            "defects": defects, "soft": []}


def kpi_fak_traced(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A SHIPPED competitive claim must trace to evidence — a commit, a committed
    artifact, or an authority doc. An unshipped row is excused only because it SAYS
    so (status), which keeps the absence honest."""
    defects: list[str] = []
    soft: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        traced = any(_nonempty(r.get(k)) for k in ("fak_commit", "fak_artifact", "fak_doc"))
        if r.get("status") == "shipped" and not traced:
            defects.append(f"{rid}: shipped but untraced — add a fak_commit, fak_artifact, or fak_doc")
        elif r.get("status") in UNSHIPPED and not traced:
            soft.append(f"{rid}: unshipped ({r.get('status')}) and untraced — fine for now, trace when it ships")
    return {"kpi": "fak_traced", "group": "traceability",
            "score": _clamp(100 - 15 * len(defects) - min(10, len(soft))),
            "detail": (f"{len(defects)} shipped-but-untraced row(s)" if defects
                       else "every shipped claim traces to evidence"),
            "defects": defects, "soft": soft}


def kpi_competitor_sourced(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The competitor's number must cite where it came from — a paper, a vendor
    doc, an on-box bench. A figure asserted from memory is exactly the unverifiable
    claim the rest of the repo refuses. A pure no-claim gap with no competitor
    number is exempt (nothing to source)."""
    defects = []
    for i, r in enumerate(rows):
        needs_source = (_num(r.get("competitor_value")) is not None
                        or isinstance(r.get("competitor_range"), list)
                        or r.get("verdict") in MEASURED_VERDICTS)
        if needs_source and not _nonempty(r.get("competitor_source")):
            defects.append(f"{r.get('id', i)}: competitor number has no source citation "
                           f"(competitor_source) — cite a paper / vendor doc / on-box bench")
    return {"kpi": "competitor_sourced", "group": "traceability",
            "score": _clamp(100 - 12 * len(defects)),
            "detail": (f"{len(defects)} unsourced competitor number(s)" if defects
                       else "every competitor number is sourced"),
            "defects": defects, "soft": []}


def kpi_freshness(rows: list[dict[str, Any]], as_of: date | None,
                  window_days: int) -> dict[str, Any]:
    """Advisory: a fak measurement should carry a recent date. Staleness is a SOFT
    nudge (re-measure when a bench node is free), never hard debt."""
    soft: list[str] = []
    if as_of is None:
        return {"kpi": "freshness", "group": "traceability", "score": 100,
                "detail": "no as_of in data meta — freshness not evaluated",
                "defects": [], "soft": ["data meta has no as_of date"]}
    for i, r in enumerate(rows):
        d = parse_date(r.get("measured_on"))
        if d is None:
            continue  # missing date is caught (hard) by well_formed
        age = (as_of - d).days
        if age > window_days:
            soft.append(f"{r.get('id', i)}: fak measured {age}d ago (> {window_days}d) — re-confirm when a bench node is free")
    return {"kpi": "freshness", "group": "traceability",
            "score": _clamp(100 - min(40, 6 * len(soft))),
            "detail": (f"{len(soft)} stale fak measurement(s) (> {window_days}d)" if soft
                       else f"every fak measurement within {window_days}d of {as_of}"),
            "defects": [], "soft": soft}


# ---------------------------------------------------------------------------
# Coverage + industry-freshness (the OUTWARD, industry-first lenses).
# ---------------------------------------------------------------------------

def coverage_report(dims: list[dict[str, Any]], rows: list[dict[str, Any]],
                    cat_group: dict[str, str]) -> dict[str, Any]:
    """How much of the industry the scorecard has even CONSIDERED. A dimension is
    'covered' when at least one fak position row links to it (a measured verdict OR
    an honest no-claim gap both count — both mean we looked). An in-scope dimension
    with no row at all is coverage_debt: a part of the field the scorecard silently
    omits, which is exactly the failure mode this redesign exists to kill."""
    covered_ids = {r.get("dim_id") for r in rows if _nonempty(r.get("dim_id"))}
    in_scope = [d for d in dims if d.get("in_scope", True)]
    by_group: dict[str, dict[str, int]] = {}
    uncovered: list[str] = []
    for d in in_scope:
        grp = cat_group.get(d.get("category"), "other")
        g = by_group.setdefault(grp, {"total": 0, "covered": 0})
        g["total"] += 1
        if d.get("id") in covered_ids:
            g["covered"] += 1
        else:
            uncovered.append(d.get("id"))
    n_in = len(in_scope)
    n_cov = sum(1 for d in in_scope if d.get("id") in covered_ids)
    pct = round(100.0 * n_cov / n_in, 1) if n_in else 100.0
    return {
        "dimensions_total": len(dims),
        "in_scope": n_in,
        "covered": n_cov,
        "coverage_pct": pct,
        "coverage_debt": n_in - n_cov,
        "uncovered": uncovered,
        "by_group": by_group,
    }


def measured_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Of the positioned dimensions, how many fak actually has a NUMBER on
    (lead/parity/trails, shipped) vs how many are honest gaps (no-claim/unbuilt).
    A gap is honest, not a defect — but the split must be visible so the score is
    never read as competitive dominance."""
    measured = sum(1 for r in rows if r.get("verdict") in ("lead", "parity", "trails")
                   and r.get("status") == "shipped")
    gap = len(rows) - measured
    pct = round(100.0 * measured / len(rows), 1) if rows else 0.0
    return {"measured": measured, "gap": gap, "measured_pct": pct, "rows": len(rows)}


def industry_freshness(dims: list[dict[str, Any]], systems: list[dict[str, Any]],
                       as_of: date | None, window_days: int) -> dict[str, Any]:
    """Advisory: the industry moves. A SOTA bar sourced long ago, or a competitor
    not reviewed within the window, wants a re-check on the web. SOFT, never debt —
    a number does not become false the day it crosses the window, it wants a look.
    This is the 'keep it current as the industry changes' lens."""
    stale_dims: list[str] = []
    stale_systems: list[str] = []
    if as_of is None:
        return {"stale_dimensions": [], "stale_systems": [], "window_days": window_days}
    for d in dims:
        # Prefer last_reviewed (tracks when the bar was re-confirmed);
        # fall back to source_date for backward compatibility.
        rd = parse_date(d.get("last_reviewed") or d.get("source_date"))
        if rd is not None and (as_of - rd).days > window_days:
            # Show both dates when they differ: original source_date vs last_reviewed
            src = d.get("source_date", "?")
            last = d.get("last_reviewed", src)
            if src != last:
                stale_dims.append(f"{d.get('id')}: SOTA bar sourced {src}, "
                                  f"last reviewed {(as_of - rd).days}d ago ({last}) — "
                                  f"re-check the current best")
            else:
                stale_dims.append(f"{d.get('id')}: SOTA bar sourced {(as_of - rd).days}d ago "
                                  f"({src}) — re-check the current best")
    for s in systems:
        rd = parse_date(s.get("last_reviewed"))
        if rd is not None and (as_of - rd).days > window_days:
            stale_systems.append(f"{s.get('name')}: last reviewed {(as_of - rd).days}d ago "
                                 f"({s.get('last_reviewed')}) — confirm latest version + numbers")
    return {"stale_dimensions": stale_dims, "stale_systems": stale_systems,
            "window_days": window_days}


# ---------------------------------------------------------------------------
# Fold: KPIs + coverage -> composite score, grade, parity-debt, payload.
# ---------------------------------------------------------------------------

def positions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The competitive standing summary: verdict counts overall + per category."""
    counts = {v: 0 for v in VERDICTS}
    by_cat: dict[str, dict[str, int]] = {}
    for r in rows:
        v = r.get("verdict")
        if v in counts:
            counts[v] += 1
        cat = r.get("category", "?")
        by_cat.setdefault(cat, {x: 0 for x in VERDICTS})
        if v in counts:
            by_cat[cat][v] += 1
    return {"overall": counts, "by_category": by_cat}


def _fmt_value(v: Any, unit: Any) -> str:
    n = _num(v)
    if n is None:
        return "—"
    s = f"{n:g}"
    return f"{s} {unit}".strip() if _nonempty(unit) else s


def _fmt_competitor(r: dict[str, Any]) -> str:
    rng = r.get("competitor_range")
    unit = r.get("unit")
    if isinstance(rng, list) and len(rng) == 2:
        lo, hi = rng
        return f"{lo}–{hi} {unit}".strip() if _nonempty(unit) else f"{lo}–{hi}"
    return _fmt_value(r.get("competitor_value"), unit)


def leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A flat, render-ready view of each comparison (data-derived, not hand-typed)."""
    out: list[dict[str, Any]] = []
    for r in rows:
        f, c = _num(r.get("fak_value")), _num(r.get("competitor_value"))
        rat = favorability(f, c, bool(r.get("higher_is_better")))
        out.append({
            "id": r.get("id"),
            "dim_id": r.get("dim_id"),
            "axis": r.get("axis", r.get("id")),
            "category": r.get("category"),
            "competitor": r.get("competitor"),
            "competitor_class": r.get("competitor_class"),
            "verdict": r.get("verdict"),
            "status": r.get("status"),
            "fak": _fmt_value(r.get("fak_value"), r.get("unit")),
            "competitor_value": _fmt_competitor(r),
            "ratio": (f"{rat:.2f}×" if rat is not None else "—"),
        })
    return out


def run_kpis(rows: list[dict[str, Any]], meta: dict[str, Any],
             required: list[dict[str, Any]], categories: set[str],
             dim_ids: set[str]) -> list[dict[str, Any]]:
    as_of = parse_date(meta.get("as_of"))
    window = int(meta.get("fresh_window_days", 120))
    return [
        kpi_well_formed(rows, categories, dim_ids),
        kpi_competitor_named(rows),
        kpi_axis_coverage(rows, required),
        kpi_baseline_sota(rows),
        kpi_verdict_consistency(rows),
        kpi_apples_disclosed(rows),
        kpi_fak_traced(rows),
        kpi_competitor_sourced(rows),
        kpi_freshness(rows, as_of, window),
    ]


def build_payload(*, workspace: str, data: dict[str, Any] | None,
                  error: str | None = None) -> dict[str, Any]:
    if error or not isinstance(data, dict):
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error or "no data",
            "next_action": f"fix the read (run from repo ROOT; check {DATA_DIR_REL}/), then re-run",
            "workspace": workspace, "corpus": {}, "kpis": [],
        }
    meta = data.get("meta") or {}
    rows = [r for r in (data.get("rows") or []) if isinstance(r, dict)]
    required = [r for r in (data.get("required_axes") or []) if isinstance(r, dict)]
    cat_defs = [c for c in (data.get("categories") or []) if isinstance(c, dict)]
    dims = [d for d in (data.get("taxonomy") or []) if isinstance(d, dict)]
    systems = [s for s in (data.get("competitors") or []) if isinstance(s, dict)]

    categories = {c.get("id") for c in cat_defs if _nonempty(c.get("id"))}
    cat_group = {c.get("id"): c.get("group", "other") for c in cat_defs}
    dim_ids = {d.get("id") for d in dims if _nonempty(d.get("id"))}

    kpis = run_kpis(rows, meta, required, categories, dim_ids)
    by_name = {k["kpi"]: k for k in kpis}
    honesty_score = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"]
                              for n in KPI_WEIGHTS if n in by_name), 1)
    parity_debt = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)

    cov = coverage_report(dims, rows, cat_group)
    meas = measured_report(rows)
    as_of = parse_date(meta.get("as_of"))
    ind_window = int(meta.get("industry_review_window_days", 180))
    ind_fresh = industry_freshness(dims, systems, as_of, ind_window)
    n_ind_soft = len(ind_fresh["stale_dimensions"]) + len(ind_fresh["stale_systems"])

    # Composite blends honesty (of existing rows) with coverage (of the industry).
    cov_pct = cov["coverage_pct"] if cov["in_scope"] else 100.0
    score = round(HONESTY_WEIGHT * honesty_score + COVERAGE_WEIGHT * cov_pct, 1)
    grade = grade_letter(score)

    debt_by_group = {g: 0 for g in GROUPS}
    for k in kpis:
        debt_by_group[k["group"]] += len(k["defects"])
    breakdown = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    pos = positions(rows)
    corpus = {
        "score": score, "grade": grade,
        "honesty_score": honesty_score,
        "parity_debt": parity_debt,
        "coverage_debt": cov["coverage_debt"],
        "coverage": cov,
        "measured": meas,
        "soft_signals": n_soft,
        "industry_soft_signals": n_ind_soft,
        "industry_freshness": ind_fresh,
        "rows": len(rows),
        "dimensions": len(dims),
        "competitors": len(systems),
        "as_of": meta.get("as_of", ""),
        "fak_version": meta.get("fak_version", ""),
        "positions": pos,
        "debt_by_group": debt_by_group,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
        "leaderboard": leaderboard(rows),
    }

    o = pos["overall"]
    standing = (f"{o['lead']} lead · {o['parity']} parity · {o['trails']} trails "
                f"· {o['no-claim']} honest gap")
    cov_line = (f"coverage {cov['coverage_pct']}% ({cov['covered']}/{cov['in_scope']} "
                f"industry dimensions positioned)")
    if parity_debt == 0 and cov["coverage_debt"] == 0:
        ok, verdict, finding = True, "OK", "scorecard_complete_and_honest"
        reason = (f"competitive map complete + honest: score {score}/100 (grade {grade}); "
                  f"{cov_line}; zero parity-debt across {len(kpis)} KPIs over {len(rows)} rows "
                  f"({standing}; {meas['measured']} measured, {meas['gap']} honest gaps; "
                  f"{n_soft}+{n_ind_soft} advisory)")
        next_action = ("hold the line; when the field moves add the new industry dimension to "
                       f"_taxonomy.json (coverage drops → position it), and when a benchmark "
                       f"lands turn a no-claim into a measured row; re-run to keep both debts at 0")
    elif cov["coverage_debt"] > 0 and parity_debt == 0:
        ok, verdict, finding = False, "ACTION", "coverage_debt"
        reason = (f"{cov['coverage_debt']} industry dimension(s) not yet positioned; "
                  f"{cov_line}; score {score}/100 (grade {grade}); rows are honest "
                  f"(parity-debt 0); standing {standing}")
        next_action = ("close coverage worst-group-first (see corpus.coverage.by_group + "
                       "--gaps): add an honest fak position row for each uncovered taxonomy "
                       "dimension (a no-claim gap is a valid position — most will be); re-run")
    else:
        ok, verdict, finding = False, "ACTION", "parity_debt"
        worst = breakdown[0]
        reason = (f"{parity_debt} unit(s) of parity-debt + {cov['coverage_debt']} coverage "
                  f"gap(s); score {score}/100 (grade {grade}); heaviest KPI: {worst['kpi']} "
                  f"({worst['debt']} defect(s)); {cov_line}; standing {standing}")
        next_action = ("retire parity-debt worst-first (corpus.breakdown + per-KPI defects): "
                       "name every competitor, align each verdict to its ratio, drop naive "
                       "baselines, disclose non-comparable rows, trace + source every number; "
                       "then close coverage gaps (--gaps); re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "kpis": kpis,
        "_data": {"dims": dims, "systems": systems, "categories": cat_defs, "rows": rows},
    }


# ---------------------------------------------------------------------------
# --verify-sources: re-check a row's fak number against its committed artifact.
# ---------------------------------------------------------------------------

def _dig(payload: Any, path: str) -> Any:
    """Walk a dotted/bracketed field path: 'cells[0].net_value_add_vs_tuned'."""
    cur = payload
    tok = ""
    i = 0
    parts: list[str | int] = []
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if tok:
                parts.append(tok); tok = ""
        elif ch == "[":
            if tok:
                parts.append(tok); tok = ""
            j = path.index("]", i)
            parts.append(int(path[i + 1:j]))
            i = j
        else:
            tok += ch
        i += 1
    if tok:
        parts.append(tok)
    for p in parts:
        try:
            cur = cur[p]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


def verify_sources(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each row carrying a `verify {field, expect, tol}` block over a present
    artifact, confirm the committed number still matches. Returns per-row results."""
    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        v = r.get("verify")
        art = r.get("fak_artifact")
        if not isinstance(v, dict) or not _nonempty(art):
            continue
        rid = r.get("id")
        path = root / art
        if not path.exists():
            out.append({"id": rid, "status": "skipped", "detail": f"artifact absent: {art}"})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            out.append({"id": rid, "status": "error", "detail": f"unreadable {art}: {exc}"})
            continue
        got = _dig(payload, v.get("field", ""))
        exp, tol = _num(v.get("expect")), _num(v.get("tol")) or 1e-6
        gn = _num(got)
        if gn is None:
            out.append({"id": rid, "status": "error",
                        "detail": f"field {v.get('field')} not numeric in {art}: {got!r}"})
        elif exp is not None and abs(gn - exp) <= tol:
            out.append({"id": rid, "status": "ok", "detail": f"{v.get('field')}={gn:g} matches"})
        else:
            out.append({"id": rid, "status": "MISMATCH",
                        "detail": f"{v.get('field')}={gn:g} != expect {exp:g} (tol {tol:g})"})
    return out


# ---------------------------------------------------------------------------
# Disk shell — read the modular data DIRECTORY (or a legacy single file).
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _read_json(path: Path) -> tuple[Any, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except (OSError, ValueError) as exc:
        return None, f"cannot parse {path.name}: {exc}"


def load_data_dir(d: Path) -> tuple[dict[str, Any] | None, str]:
    """Merge the modular data directory into one in-memory document.

    _meta.json → meta + categories + required_axes; _taxonomy.json → taxonomy;
    _competitors.json → competitors; every other rows-*.json (or *.json without a
    leading underscore) contributes its `rows`. Modular by construction: each
    concern is its own file, and a category's rows live in their own file."""
    meta_doc, err = _read_json(d / "_meta.json")
    if err:
        return None, err
    if not isinstance(meta_doc, dict):
        return None, "_meta.json is not a JSON object"
    out: dict[str, Any] = {
        "meta": meta_doc.get("meta") or {},
        "categories": meta_doc.get("categories") or [],
        "required_axes": meta_doc.get("required_axes") or [],
        "taxonomy": [],
        "competitors": [],
        "rows": [],
    }
    tax_path = d / "_taxonomy.json"
    if tax_path.exists():
        doc, err = _read_json(tax_path)
        if err:
            return None, err
        out["taxonomy"] = (doc or {}).get("dimensions") or []
    comp_path = d / "_competitors.json"
    if comp_path.exists():
        doc, err = _read_json(comp_path)
        if err:
            return None, err
        out["competitors"] = (doc or {}).get("systems") or []
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("_"):
            continue
        doc, err = _read_json(f)
        if err:
            return None, err
        rws = (doc or {}).get("rows") or []
        for r in rws:
            if isinstance(r, dict):
                r.setdefault("_source_file", f.name)
        out["rows"].extend(rws)
    return out, ""


def load_data(path: Path) -> tuple[dict[str, Any] | None, str]:
    """Load from the modular directory if `path` is one; else a legacy single file
    (old single-JSON format: {meta, required_axes, rows}, no taxonomy)."""
    if path.is_dir():
        return load_data_dir(path)
    if not path.exists():
        return None, f"missing data: {path}"
    doc, err = _read_json(path)
    if err:
        return None, err
    if not isinstance(doc, dict):
        return None, f"data file is not a JSON object: {path}"
    doc.setdefault("taxonomy", [])
    doc.setdefault("competitors", [])
    doc.setdefault("categories", [])
    return doc, ""


def default_data_path(root: Path) -> Path:
    """Prefer the modular directory; fall back to the legacy single file."""
    d = root / DATA_DIR_REL
    if d.is_dir():
        return d
    return root / LEGACY_DATA_REL


def collect(workspace: Path, *, data_path: Path | None = None) -> dict[str, Any]:
    root = workspace.resolve()
    path = data_path or default_data_path(root)
    data, err = load_data(path)
    return build_payload(workspace=str(root), data=data, error=err or None)


# ---------------------------------------------------------------------------
# Renderers — terminal, the modular doc folder, compare, verify, gaps, stale.
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    pos = (c.get("positions") or {}).get("overall", {})
    cov = c.get("coverage") or {}
    meas = c.get("measured") or {}
    lines = [
        f"industry-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· PARITY-DEBT {c.get('parity_debt', 0)} · COVERAGE-DEBT {c.get('coverage_debt', 0)} "
         f"· {c.get('soft_signals', 0)}+{c.get('industry_soft_signals', 0)} advisory"),
        (f"coverage: {cov.get('coverage_pct', 0)}% "
         f"({cov.get('covered', 0)}/{cov.get('in_scope', 0)} industry dimensions positioned) "
         f"· {meas.get('measured', 0)} measured, {meas.get('gap', 0)} honest gaps "
         f"· {c.get('dimensions', 0)} dims / {c.get('competitors', 0)} competitors tracked"),
        (f"standing: {pos.get('lead', 0)} lead · {pos.get('parity', 0)} parity "
         f"· {pos.get('trails', 0)} trails · {pos.get('no-claim', 0)} honest gap"),
        ("debt by group: " + "  ".join(
            f"{g}:{(c.get('debt_by_group') or {}).get(g, 0)}" for g in GROUPS)),
    ]
    by_group = cov.get("by_group") or {}
    if by_group:
        lines.append("coverage by group: " + "  ".join(
            f"{g}:{v.get('covered', 0)}/{v.get('total', 0)}" for g, v in sorted(by_group.items())))
    lines += [
        "",
        "competitive standing (data-derived):",
        f"  {'verdict':<8} {'category':<18} {'fak':<20} {'vs competitor':<20} {'ratio':<7} axis",
    ]
    order = {"lead": 0, "parity": 1, "trails": 2, "no-claim": 3}
    for row in sorted(c.get("leaderboard", []),
                      key=lambda x: (order.get(x["verdict"], 9), x.get("category") or "")):
        mark = {"lead": "▲", "parity": "≈", "trails": "▼", "no-claim": "○"}.get(row["verdict"], " ")
        lines.append(f"  {mark} {row['verdict']:<6} {str(row.get('category')):<18} "
                     f"{row['fak']:<20} {row['competitor_value']:<20} {row['ratio']:<7} {row['axis']}")
    lines += ["", "per-KPI (worst first):",
              f"  {'score':>5} {'debt':>4}  {'group':<13} {'kpi':<20} detail"]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['group']:<13} "
                     f"{b['kpi']:<20} {b['detail']}")
    lines.append("")
    if cov.get("uncovered"):
        lines.append(f"coverage gaps ({len(cov['uncovered'])} industry dimensions with no fak position):")
        for did in cov["uncovered"][:15]:
            lines.append(f"      - {did}")
        if len(cov["uncovered"]) > 15:
            lines.append(f"      ... and {len(cov['uncovered']) - 15} more (see --gaps)")
        lines.append("")
    lines.append("parity-debt work-list:")
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
        lines.append("  (none — zero parity-debt; every claimed comparison is honest)")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_gaps(payload: dict[str, Any]) -> str:
    """The two backlogs a maintainer drives: dimensions with NO position (coverage
    debt) and positioned-but-only-as-honest-gap dimensions fak could measure."""
    c = payload.get("corpus") or {}
    dims = {d.get("id"): d for d in (payload.get("_data") or {}).get("dims", [])}
    rows = (payload.get("_data") or {}).get("rows", [])
    by_dim: dict[str, dict[str, Any]] = {}
    for r in rows:
        if _nonempty(r.get("dim_id")):
            by_dim[r["dim_id"]] = r
    cov = c.get("coverage") or {}
    lines = ["industry coverage backlog (drive coverage toward 100%):", ""]
    lines.append(f"UNPOSITIONED — {len(cov.get('uncovered', []))} industry dimension(s) with no fak row at all:")
    if not cov.get("uncovered"):
        lines.append("  (none — every in-scope industry dimension is positioned)")
    for did in cov.get("uncovered", []):
        d = dims.get(did, {})
        lines.append(f"  - [{d.get('category', '?')}] {did}: {d.get('dimension', '')}")
        if _nonempty(d.get("sota_bar")):
            lines.append(f"        SOTA: {d['sota_bar']}")
    lines.append("")
    no_claim_dims = [(did, r) for did, r in by_dim.items() if r.get("verdict") == "no-claim"]
    lines.append(f"HONEST GAPS — {len(no_claim_dims)} positioned dimension(s) fak has no number on "
                 f"(a real gap to measure, or out-of-scope by design):")
    for did, r in sorted(no_claim_dims, key=lambda kv: kv[1].get("category") or ""):
        d = dims.get(did, {})
        lines.append(f"  - [{r.get('category', '?')}] {did}: {d.get('dimension', r.get('axis', ''))}")
    return "\n".join(lines)


def render_stale(payload: dict[str, Any]) -> str:
    """The industry-drift backlog: SOTA bars + competitor versions due a web re-check.
    This is the 'keep current as the industry changes' half of the update process."""
    c = payload.get("corpus") or {}
    ind = c.get("industry_freshness") or {}
    win = ind.get("window_days", "?")
    lines = [f"industry-drift backlog (re-check on the web; window {win}d):", ""]
    sd = ind.get("stale_dimensions", [])
    ss = ind.get("stale_systems", [])
    lines.append(f"STALE SOTA BARS — {len(sd)} dimension source(s) past the window:")
    for it in sd:
        lines.append(f"  - {it}")
    if not sd:
        lines.append("  (none — every SOTA bar sourced within the window)")
    lines.append("")
    lines.append(f"STALE COMPETITORS — {len(ss)} system registration(s) past the window:")
    for it in ss:
        lines.append(f"  - {it}")
    if not ss:
        lines.append("  (none — every competitor reviewed within the window)")
    lines.append("")
    fk = c.get("soft_signals", 0)
    lines.append(f"(fak-side: {fk} stale fak measurement advisory — see the main scorecard `freshness` KPI)")
    return "\n".join(lines)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("parity_debt", 0), cur.get("parity_debt", 0)
    bcov, ccov = b.get("coverage_debt", 0), cur.get("coverage_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    bcp = (b.get("coverage") or {}).get("coverage_pct", 0)
    ccp = (cur.get("coverage") or {}).get("coverage_pct", 0)
    pd_ratio = "∞ (zero)" if cd == 0 else f"{bd / cd:.1f}×"
    lines = [
        f"parity-debt:   {bd} -> {cd}   ({pd_ratio} fewer honesty defects)",
        f"coverage-debt: {bcov} -> {ccov}   (industry dimensions still unpositioned)",
        f"coverage:      {bcp}% -> {ccp}%",
        f"score:         {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<13} {gb} -> {gc}")
    target = max(0, bd // 2)
    if cd <= target and ccov <= max(0, bcov // 2):
        lines.append(f"VERDICT: >=2x reduction achieved (parity {bd}->{cd}, coverage {bcov}->{ccov}).")
    else:
        lines.append(f"VERDICT: not yet 2x — need parity-debt <= {target} (now {cd}) "
                     f"and coverage-debt <= {max(0, bcov // 2)} (now {ccov}).")
    return "\n".join(lines)


def render_verify(results: list[dict[str, Any]]) -> str:
    if not results:
        return "no rows carry a verify block (field/expect/tol over a committed artifact)"
    lines = ["source verification (fak numbers vs committed artifacts):"]
    bad = 0
    for r in results:
        if r["status"] in ("MISMATCH", "error"):
            bad += 1
        lines.append(f"  [{r['status']:>8}] {r['id']}: {r['detail']}")
    lines.append("")
    lines.append("all present sources match" if bad == 0 else f"{bad} source(s) failed — fix the data row or re-measure")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The modular doc FOLDER (one concern per page), generated from the data.
# render_doc_folder returns {relpath: content}; main() writes them under --markdown-dir.
# ---------------------------------------------------------------------------

_MARK = {"lead": "▲", "parity": "≈", "trails": "▼", "no-claim": "○"}
_VORDER = {"lead": 0, "parity": 1, "trails": 2, "no-claim": 3}


def _group_of(cat: str, cat_group: dict[str, str]) -> str:
    return cat_group.get(cat, "other")


def _front_matter(title: str, desc: str) -> list[str]:
    return ["---", f'title: "{title}"', f'description: "{desc}"', "---", ""]


def _standing_line(pos: dict[str, Any]) -> str:
    o = pos.get("overall", {})
    return (f"{o.get('lead', 0)} lead · {o.get('parity', 0)} parity · "
            f"{o.get('trails', 0)} trails · {o.get('no-claim', 0)} honest gap")


def render_index_chart(payload: dict[str, Any]) -> str:
    """An at-a-glance ASCII chart of fak's industry standing — what a reader sees first.

    Three views over the same numbers the rest of the page derives: field coverage,
    the standing split on the positioned axes (lead/parity/trails/no-claim, shown not
    hidden), and per-group coverage. Pure text + deterministic — two clones at one
    commit chart identically; no number here is hand-typed."""
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    pos = (c.get("positions") or {}).get("overall") or {}

    def bar(n: int, maxn: int, width: int = 24) -> str:
        n = max(0, int(n))
        maxn = max(1, int(maxn))
        fill = round(width * n / maxn)
        return "█" * fill + "·" * (width - fill)

    lines: list[str] = [
        (f"industry standing chart — {c.get('dimensions', 0)} dimensions · "
         f"{c.get('competitors', 0)} competitors · "
         f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) · "
         f"parity-debt {c.get('parity_debt', 0)}"),
        "",
        "coverage of the field (positioned / in-scope dimensions):",
        (f"  positioned  [{bar(cov.get('covered', 0), cov.get('in_scope', 0) or 1, 32)}]"
         f"  {cov.get('covered', 0)}/{cov.get('in_scope', 0)}  ({cov.get('coverage_pct', 0)}%)"),
        "",
        "standing on the positioned axes (shown, not hidden):",
    ]
    order = [("lead", "▲ lead"), ("parity", "≈ parity"),
             ("trails", "▼ trails"), ("no-claim", "○ no-claim")]
    maxn = max((int(pos.get(k, 0)) for k, _ in order), default=0)
    for key, label in order:
        n = int(pos.get(key, 0))
        lines.append(f"  {label:<11} {bar(n, maxn)} {n}")
    lines.append("")

    by_group = cov.get("by_group") or {}
    if by_group:
        lines.append("coverage by group:")
        gmax = max((int(v.get("total", 0)) for v in by_group.values()), default=0)
        for g in sorted(by_group):
            v = by_group[g]
            cvd, tot = int(v.get("covered", 0)), int(v.get("total", 0))
            lines.append(f"  {g:<14} {bar(cvd, gmax)} {cvd}/{tot}")
    return "\n".join(lines)

def render_index_page(payload: dict[str, Any], data: dict[str, Any],
                      *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    meas = c.get("measured") or {}
    cat_defs = data.get("categories") or []
    cat_group = {x.get("id"): x.get("group", "other") for x in cat_defs}
    groups = sorted({x.get("group", "other") for x in cat_defs})
    out = _front_matter(
        "fak industry scorecard — where fak stands in the LLM-serving field",
        "Industry-first competitive scorecard: a researched taxonomy of the dimensions that "
        "matter in LLM inference-serving + agent infrastructure (vLLM, SGLang, TensorRT-LLM, "
        "llama.cpp, …), the current SOTA bar on each, and fak's honest position — most as named "
        "gaps. Two driven numbers: coverage (of the field) and parity-debt (honesty of the rows).")
    out.append("# Industry scorecard — fak vs the LLM-serving field")
    out.append("")
    if stamp:
        out.append(f"<!-- industry-scorecard: {stamp} · process: tools/industry_scorecard.py · "
                   f"data: tools/industry_scorecard.data/ -->")
        out.append("")
    out.append("This is the **outward** measuring stick — the counterpart of the inward scorecards "
               "(hygiene, code, docs). It does not start from what fak happened to measure; it starts "
               "from the **industry**. The source of truth is a researched taxonomy of the dimensions a "
               "serious operator, buyer, or analyst uses to evaluate an LLM-serving system "
               "(`tools/industry_scorecard.data/_taxonomy.json`), each with the current SOTA bar and a "
               "dated source. fak is then positioned honestly on every dimension — and for most of them "
               "the honest answer is a **named gap**, not a win. Everything below is re-derived from the "
               "data by `tools/industry_scorecard.py`; no number is hand-typed.")
    out.append("")
    out.append("Two numbers are driven:")
    out.append("")
    out.append("- **coverage** — of the industry dimensions that matter, how many the scorecard has "
               "considered and positioned fak against (toward 100%).")
    out.append("- **parity-debt** — of the comparisons that exist, how many are dishonest, incomplete, "
               "or unsourced (kept at 0).")
    out.append("")
    out.append("> Regenerate: `python tools/industry_scorecard.py --markdown-dir docs/industry-scorecard`. "
               "Update process: [UPDATE-PROCESS.md](UPDATE-PROCESS.md). Full dimension catalog: "
               "[taxonomy.md](taxonomy.md).")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Coverage** | **{cov.get('coverage_pct', 0)}%** "
               f"({cov.get('covered', 0)}/{cov.get('in_scope', 0)} industry dimensions positioned) |")
    out.append(f"| **Parity-debt (honesty defects)** | **{c.get('parity_debt', 0)}** |")
    out.append(f"| Coverage-debt (unpositioned dimensions) | {c.get('coverage_debt', 0)} |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) — "
               f"honesty {c.get('honesty_score', 0)} × {int(HONESTY_WEIGHT*100)}% + "
               f"coverage {cov.get('coverage_pct', 0)}% × {int(COVERAGE_WEIGHT*100)}% |")
    out.append(f"| Standing | {_standing_line(c.get('positions') or {})} |")
    out.append(f"| Measured vs gap | {meas.get('measured', 0)} measured · {meas.get('gap', 0)} honest gaps |")
    out.append(f"| Tracked | {c.get('dimensions', 0)} dimensions · {c.get('competitors', 0)} competitors "
               f"· {c.get('rows', 0)} positions |")
    out.append(f"| As of | {c.get('as_of', '?')} (fak {c.get('fak_version', '?')}) |")
    out.append(f"| Advisory signals | {c.get('soft_signals', 0)} fak-freshness · "
               f"{c.get('industry_soft_signals', 0)} industry-drift |")
    out.append("")
    out.append("> **Read this right.** The score grades how *complete and honest fak's competitive map "
               "is* — not how much fak wins. fak is a focused reuse + trust kernel, so most dimensions "
               "are honest `no-claim` gaps (out-of-scope or not-yet-measured), shown plainly below.")
    out.append("")
    out.append("## Standing at a glance")
    out.append("")
    out.append("```text")
    out.append(render_index_chart(payload))
    out.append("```")
    out.append("")
    out.append("## Coverage by group")
    out.append("")
    out.append("| Group | Positioned / in-scope | Pages |")
    out.append("|---|---|---|")
    by_group = cov.get("by_group") or {}
    for g in groups:
        v = by_group.get(g, {"covered": 0, "total": 0})
        out.append(f"| {g} | {v.get('covered', 0)}/{v.get('total', 0)} | [{g}.md]({g}.md) |")
    out.append("")
    out.append("## Standing across the field (data-derived)")
    out.append("")
    out.append("▲ lead · ≈ parity · ▼ trails (shown, not hidden) · ○ honest gap (no claim yet).")
    out.append("")
    out.append("| | Verdict | Category | Axis | fak | vs competitor | Ratio | Competitor |")
    out.append("|---|---|---|---|---|---|---|---|")
    for row in sorted(c.get("leaderboard", []),
                      key=lambda x: (_VORDER.get(x["verdict"], 9), x.get("category") or "")):
        mark = _MARK.get(row["verdict"], " ")
        out.append(f"| {mark} | {row['verdict']} | {row.get('category')} | {row['axis']} | "
                   f"{row['fak']} | {row['competitor_value']} | {row['ratio']} | {row.get('competitor') or '—'} |")
    out.append("")
    out.append("## Per-KPI (parity-debt = honesty of the rows that exist)")
    out.append("")
    out.append("| Group | KPI | Score | Debt | Detail |")
    out.append("|---|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    if cov.get("uncovered"):
        out.append("## Coverage gaps (industry dimensions not yet positioned)")
        out.append("")
        for did in cov["uncovered"]:
            out.append(f"- `{did}`")
        out.append("")
    return "\n".join(out)


def render_group_page(group: str, payload: dict[str, Any], data: dict[str, Any]) -> str:
    cat_defs = data.get("categories") or []
    cat_group = {x.get("id"): x.get("group", "other") for x in cat_defs}
    cat_name = {x.get("id"): x.get("name", x.get("id")) for x in cat_defs}
    cats = [x.get("id") for x in cat_defs if x.get("group", "other") == group]
    dims = [d for d in (data.get("taxonomy") or []) if _group_of(d.get("category"), cat_group) == group]
    rows_by_dim: dict[str, dict[str, Any]] = {}
    for r in data.get("rows") or []:
        if _nonempty(r.get("dim_id")):
            rows_by_dim[r["dim_id"]] = r
    out = _front_matter(
        f"fak industry scorecard — {group}",
        f"The {group} dimensions that matter in LLM serving, the current SOTA bar on each, and "
        f"fak's honest position. Generated from tools/industry_scorecard.data/.")
    out.append(f"# {group} — the dimensions that matter, and where fak stands")
    out.append("")
    out.append(f"[← back to the scorecard index](README.md) · part of the industry-first scorecard. "
               f"Each dimension is a thing the field competes on; the fak column is honest — mostly "
               f"`no-claim` gaps for a focused reuse kernel.")
    out.append("")
    for cat in cats:
        cdims = [d for d in dims if d.get("category") == cat]
        if not cdims:
            continue
        out.append(f"## {cat_name.get(cat, cat)} (`{cat}`)")
        out.append("")
        for d in cdims:
            did = d.get("id")
            r = rows_by_dim.get(did)
            verdict = (r or {}).get("verdict", "—")
            mark = _MARK.get(verdict, "○")
            out.append(f"### {mark} {d.get('dimension', did)} — fak: **{verdict}**")
            out.append("")
            if _nonempty(d.get("why_it_matters")):
                out.append(f"*Why it matters:* {d['why_it_matters']}")
                out.append("")
            sota_sys = ", ".join(d.get("sota_systems") or []) or "—"
            out.append(f"- **SOTA bar:** {d.get('sota_bar', '—')}")
            out.append(f"- **Leading systems:** {sota_sys}")
            src = d.get("source_url", "")
            sdate = d.get("source_date", "")
            if _nonempty(src):
                out.append(f"- **Source:** [{src}]({src}) ({sdate})")
            elif _nonempty(sdate):
                out.append(f"- **Source date:** {sdate}")
            if r:
                fakv = _fmt_value(r.get("fak_value"), r.get("unit"))
                note = r.get("comparison_note", "")
                out.append(f"- **fak:** {verdict} — {fakv if fakv != '—' else 'no number'}"
                           + (f" ({r.get('status')})" if _nonempty(r.get('status')) else ""))
                if _nonempty(note):
                    out.append(f"- **fak note:** {note}")
                trace = " · ".join(t for t in (r.get("fak_commit"), r.get("fak_artifact"), r.get("fak_doc"))
                                   if _nonempty(t))
                if _nonempty(trace):
                    out.append(f"- **Trace:** {trace}")
            else:
                out.append("- **fak:** *(not yet positioned — coverage gap)*")
            out.append("")
    return "\n".join(out)


def render_taxonomy_page(data: dict[str, Any]) -> str:
    dims = data.get("taxonomy") or []
    cat_defs = data.get("categories") or []
    cat_group = {x.get("id"): x.get("group", "other") for x in cat_defs}
    out = _front_matter(
        "fak industry scorecard — the dimension catalog",
        "The full industry-first taxonomy: every dimension that matters in LLM serving + agent "
        "infrastructure, its current SOTA bar, and a dated source. The source of truth the scorecard "
        "positions fak against.")
    out.append("# The industry dimension catalog")
    out.append("")
    out.append("Every dimension the field competes on, researched industry-first (not derived from "
               "what fak measured). [← back to the index](README.md).")
    out.append("")
    out.append("Dimensions are grouped by the part of the stack they belong to. Each group is a section "
               "below; follow the group link for the full per-dimension analysis with fak's position.")
    out.append("")
    # One H2 section per group (sorted), each with its own sub-table. Section
    # headings keep the long catalog crawlable and quotable; the data is unchanged
    # (the former single table sorted by the same key, just split on group).
    groups = sorted({_group_of(d.get("category"), cat_group) for d in dims})
    for group in groups:
        gdims = sorted(
            (d for d in dims if _group_of(d.get("category"), cat_group) == group),
            key=lambda x: (x.get("category") or "", x.get("id") or ""))
        if not gdims:
            continue
        out.append(f"## {group}")
        out.append("")
        out.append(f"See the full [{group} analysis]({group}.md) for fak's honest position on each dimension.")
        out.append("")
        out.append("| Category | Dimension | SOTA bar | Leading systems | Source | Date |")
        out.append("|---|---|---|---|---|---|")
        for d in gdims:
            sys = ", ".join(d.get("sota_systems") or [])
            src = d.get("source_url", "")
            src_md = f"[link]({src})" if _nonempty(src) else "—"
            bar = (d.get("sota_bar", "") or "").replace("|", "\\|")
            out.append(f"| {d.get('category')} | {d.get('dimension')} | {bar} | {sys} | "
                       f"{src_md} | {d.get('source_date', '')} |")
        out.append("")
    return "\n".join(out)


def render_update_process_page(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    out = _front_matter(
        "fak industry scorecard — the update process",
        "How the industry scorecard stays current on both cadences: as the industry changes (new "
        "dimensions, moved SOTA bars) and as fak changes (a gap becomes a measured row).")
    out.append("# Keeping the scorecard current — the two cadences")
    out.append("")
    out.append("An outward scorecard rots in two directions. This is how each is caught and closed.")
    out.append("")
    out.append("## 1. As the industry changes")
    out.append("")
    out.append("New techniques appear and published SOTA bars move. Two mechanisms catch it:")
    out.append("")
    out.append("- **New dimension → coverage drops.** When the field starts competing on something new "
               "(a new quant format, a new decoding trick), add it to "
               "`tools/industry_scorecard.data/_taxonomy.json`. It is immediately uncovered, so "
               "`coverage` falls and the dimension shows up in `--gaps` until fak is positioned on it.")
    out.append("- **Stale SOTA bar → industry-drift backlog.** Every dimension carries a `source_date` "
               "and every competitor a `last_reviewed`. `python tools/industry_scorecard.py --stale` "
               "lists the bars past the `industry_review_window_days` window — re-check them on the web "
               "and update the number + date. (Advisory, never parity-debt: a number does not become "
               "false the day it crosses the window, it wants a look.)")
    out.append("")
    out.append("## 2. As fak changes")
    out.append("")
    out.append("- **A benchmark lands → a gap becomes a measured row.** When a new fak number ships "
               "(traced in `BENCHMARK-AUTHORITY.md`), turn the relevant `no-claim` position into a "
               "`lead`/`parity`/`trails` row, citing the commit/artifact. `--gaps` lists the honest "
               "gaps that are candidates to measure.")
    out.append("- **A fak number ages → re-confirm.** `measured_on` drives the `freshness` KPI; old "
               "measurements are flagged (advisory) to re-confirm when a bench node is free.")
    out.append("- **A number changes → never hand-edit the doc.** Edit the data file, regenerate.")
    out.append("")
    out.append("## The commands")
    out.append("")
    out.append("```bash")
    out.append("python tools/industry_scorecard.py               # the scorecard + both work-lists")
    out.append("python tools/industry_scorecard.py --gaps        # coverage backlog (what to position/measure)")
    out.append("python tools/industry_scorecard.py --stale       # industry-drift backlog (SOTA bars to re-check)")
    out.append("python tools/industry_scorecard.py --verify-sources   # fak numbers still match their artifacts")
    out.append("python tools/industry_scorecard.py --markdown-dir docs/industry-scorecard  # regenerate this folder")
    out.append("```")
    out.append("")
    out.append("## The one rule that overrides everything")
    out.append("")
    out.append("**Never invent a number or a win.** A fak figure must already exist in "
               "`BENCHMARK-AUTHORITY.md` / a committed artifact; a competitor figure must cite a real "
               "source. If a fix would require manufacturing evidence, the honest row is a `no-claim` "
               "gap — not a guess. The `/industry-score` skill enforces this.")
    out.append("")
    return "\n".join(out)


def render_doc_folder(payload: dict[str, Any], data: dict[str, Any],
                      *, stamp: str | None = None) -> dict[str, str]:
    """The whole modular doc folder as {relpath: content}. main() writes it."""
    cat_defs = data.get("categories") or []
    groups = sorted({x.get("group", "other") for x in cat_defs})
    files = {
        "README.md": render_index_page(payload, data, stamp=stamp),
        "taxonomy.md": render_taxonomy_page(data),
        "UPDATE-PROCESS.md": render_update_process_page(payload),
    }
    for g in groups:
        files[f"{g}.md"] = render_group_page(g, payload, data)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Industry (competitive) scorecard — industry-first, modular.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--data", default="", help=f"data dir or file (default: {DATA_DIR_REL}/)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--gaps", action="store_true", help="print the coverage backlog (unpositioned + honest-gap dims)")
    ap.add_argument("--stale", action="store_true", help="print the industry-drift backlog (SOTA bars due a re-check)")
    ap.add_argument("--markdown-dir", default="", metavar="DIR",
                    help="regenerate the modular doc folder under DIR")
    ap.add_argument("--stamp", default="", help="date stamp for the generated index header")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the debt delta vs a prior baseline JSON")
    ap.add_argument("--verify-sources", action="store_true",
                    help="re-check fak numbers against their committed artifacts (best-effort)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    data_path = Path(args.data).resolve() if args.data else default_data_path(root)
    payload = collect(root, data_path=data_path)

    if args.verify_sources:
        rows = (payload.get("_data") or {}).get("rows", [])
        results = verify_sources(root, rows)
        print(render_verify(results))
        return 0 if all(r["status"] not in ("MISMATCH", "error") for r in results) else 1

    if args.markdown_dir:
        data, err = load_data(data_path)
        if err or not isinstance(data, dict):
            print(f"error: cannot load data for doc generation: {err}", file=sys.stderr)
            return 2
        out_dir = Path(args.markdown_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        files = render_doc_folder(payload, data, stamp=args.stamp or None)
        for rel, content in files.items():
            (out_dir / rel).write_text(content + "\n", encoding="utf-8")
        print(f"wrote {len(files)} page(s) to {out_dir}:")
        for rel in sorted(files):
            print(f"  {rel}")
        return 0 if payload.get("ok") else 1

    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"error: cannot read baseline {args.compare}: {exc}", file=sys.stderr)
            return 2
        print(render_compare(baseline, payload))
    elif args.gaps:
        print(render_gaps(payload))
    elif args.stale:
        print(render_stale(payload))
    elif args.json:
        printable = {k: v for k, v in payload.items() if k != "_data"}
        print(json.dumps(printable, indent=2))
    else:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
