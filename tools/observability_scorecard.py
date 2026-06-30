#!/usr/bin/env python3
"""Observability scorecard — the measuring stick that makes "more observable" provable.

The sibling scorecards each watch one surface: ``code_quality_scorecard`` grades
the Go module, ``docs_scorecard`` the curated core docs, ``repo_hygiene_scorecard``
the shape of the whole tree, ``seo_aeo_scorecard`` discoverability. None of them
watch the surface that decides whether you can *see what the gate is doing in
production, and trust a claim about it*: the **observability** plane — the metrics
the binary emits, the dashboards and alerts that read them, the docs that tell an
operator which metric to query, the trace-id that ties a request across logs, and
the proofs/ship-audit that let a claim be verified rather than asserted.

"Make fak more observable" was a vibe. This is the number that makes it a
falsifiable target. It scores the observability plane on eight mechanical KPIs in
three groups, folds them into a weighted score and an A–F grade, and — the lever
that turns "10× more observable" into a checkable target — counts
**observability-debt**: the total of concrete, re-derivable defects you fix by
*making the live system more visible and more verifiable*.

  CORRELATION   — every dashboard / alert / doc points at a metric that EXISTS
    dashboard_integrity  a Grafana panel querying a fak_* family the binary emits
    alert_integrity      an alert rule firing on a fak_* family the binary emits
    doc_metric_drift     a doc telling an operator to query a fak_* family that exists

  INSTRUMENTATION — the live gate is visible from its own source
    trace_correlation    the X-Trace-Id surface is intact (honored, minted, logged)
    log_privacy          the structured access log carries NO request payload (HARD)
    metric_doc_coverage  emitted metrics are surfaced somewhere a reader finds (SOFT)

  VERIFIABILITY — a claim can be checked, not just asserted (rigor / DOS)
    proof_witness        a PROVEN theorem carries a runnable WITNESS; every theorem a VERDICT
    ship_integrity       `dos review` shows zero unwitnessed RESIDUAL commits     [DOS]

The headline metric is **observability-debt**: the count of HARD defects — a
phantom dashboard panel, an alert on a metric that does not exist, a doc that
sends an operator to a metric the binary never emits, a broken trace surface, a
log line that leaks a payload, a PROVEN proof with no witness command, an
unwitnessed ship the kernel flagged. Driving observability-debt toward zero is
what lets you run the gate in production and *trust what you see*.

It does NOT drift from the source of truth. The **emitted-metric set** is derived
live from the Go source (every ``"fak_..."`` family literal in ``internal/**``,
test files excluded) — the binary itself is the oracle, so a dashboard/alert/doc
reference is "backed" iff the binary actually emits that family (Prometheus
``_bucket``/``_sum``/``_count`` histogram suffixes and a counter cited without its
``_total`` suffix are normalized, so a *correct* abbreviated reference is never
flagged). The ``fleet_*`` namespace is emitted by a SEPARATE Python exporter
(``tools/fleet_bottleneck.py``), not the Go binary, so it is explicitly OUT OF
SCOPE here — this scorecard is the measuring stick for the **fak gateway's** own
observability, and silently grading another emitter's families against the Go
source would be a false positive.

``ship_integrity`` is HEAD-relative (it grades recent history, not the tree) and,
like ``code_quality_scorecard``'s, fails OPEN: a missing ``dos`` scores the KPI as
*skipped* (100, a soft "unmeasured" note), never as a failure — so a box without
the toolchain does not grade the same tree lower than one with it.
``metric_doc_coverage`` is deliberately SOFT (it scores but emits no hard debt):
the cheap way to move it is documenting every internal counter, which is noise,
not visibility — the same WARN/HARD split the sibling scorecards draw.

Deterministic + read-only by construction (except the HEAD-relative DOS probe):
it reads the git-tracked tree and shells out to ``dos review`` (a read-only verb);
it edits nothing. Run from the repo ROOT::

    python tools/observability_scorecard.py                 # human scorecard
    python tools/observability_scorecard.py --json          # machine payload
    python tools/observability_scorecard.py --markdown      # the committed snapshot body
    python tools/observability_scorecard.py --no-dos        # skip the dos ship-integrity probe
    python tools/observability_scorecard.py --compare base.json   # prove the debt moved (the 10x gate)

The companion process is the observability-10x program: each HARD defect is one
unit of observability-debt to retire by *fixing the reference, the surface, or the
witness* — never by deleting the signal; re-running with ``--compare`` proves the
number dropped.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
install_no_window_subprocess_defaults(subprocess)
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-observability-scorecard/1"

# ---------------------------------------------------------------------------
# Calibration. Each is a deliberate threshold with a stated reason.
# ---------------------------------------------------------------------------
# Prometheus exposition suffixes a histogram/summary base family sprouts at scrape
# time; a reference to `<family>_bucket` is backed by the base family `<family>`.
PROM_SUFFIXES = ("_bucket", "_sum", "_count")
# Structured-log field keys that would leak the payload a gate is meant to protect —
# request bodies, tool arguments, or result content. The access log is auditable
# WITHOUT any of these (the privacy invariant); one in a log event map is a leak.
FORBIDDEN_LOG_FIELDS = {
    "args", "arguments", "tool_args", "tool_arguments", "input", "inputs",
    "body", "request_body", "req_body", "response_body", "content", "result",
    "result_content", "payload", "prompt", "messages", "params", "data",
}
# A log event window: from a `"event":` field literal forward to the emit call.
LOG_EMIT_TOKENS = ("s.logf(", ".logf(", "logf(")
COVERAGE_SAMPLE = 30        # cap on listed soft coverage signals
DRIFT_SAMPLE = 40           # cap on listed drift defects per KPI render

# The fak gateway binary is the source of truth for the `fak_` namespace only.
# `fleet_` is a separate Python exporter (tools/fleet_bottleneck.py) with its own
# source; grading its families against the Go source would be a false positive.
METRIC_PREFIX = "fak_"

# Go source roots that define what the binary emits (production code only).
SOURCE_DIRS = ("internal/",)

# Operator-facing metric-documentation surface: the docs whose job is to tell an
# operator WHICH metric to scrape / query. A `fak_*` token here that the binary
# does not emit misdirects an operator, so it is drift. Scoped on purpose: the
# `fak_*` prefix is ALSO used for non-metric identifiers (benchmark-harness names
# like `fak_node_bench`, result columns like `fak_completion_tokens`, proof ids
# like `fak_laptop_test`) that live in benchmark / research / proof docs — grading
# those against the metric source of truth would be a false positive. The product
# + integration docs (and the dashboards' own README) are where metrics are
# documented; the generated snapshot is excluded (it lists the defect strings, so
# scoring it would oscillate).
GENERATED_SNAPSHOT = "docs/OBSERVABILITY-SCORECARD.md"
DOC_SURFACE_PREFIXES = ("docs/fak/", "docs/integrations/")
DOC_SURFACE_EXTRA = ("tools/grafana/README.md",)

# Dashboards + alerts: the Grafana surfaces that read the metrics.
DASHBOARD_GLOB_DIR = "tools/grafana/dashboards"
ALERTS_REL = "tools/grafana/prometheus-alerts.yml"

# Proofs corpus: the verifiability surface.
PROOFS_DIR = "docs/proofs"

GROUPS = ("correlation", "instrumentation", "verifiability")

KPI_WEIGHTS: dict[str, float] = {
    # correlation — points at a metric that exists
    "dashboard_integrity": 0.16,
    "alert_integrity": 0.14,
    "doc_metric_drift": 0.16,
    # instrumentation — the live gate is visible from its source
    "trace_correlation": 0.12,
    "log_privacy": 0.14,
    "metric_doc_coverage": 0.06,
    # verifiability — a claim can be checked (rigor / DOS)
    "proof_witness": 0.12,
    "ship_integrity": 0.10,
}
KPI_GROUP: dict[str, str] = {
    "dashboard_integrity": "correlation",
    "alert_integrity": "correlation",
    "doc_metric_drift": "correlation",
    "trace_correlation": "instrumentation",
    "log_privacy": "instrumentation",
    "metric_doc_coverage": "instrumentation",
    "proof_witness": "verifiability",
    "ship_integrity": "verifiability",
}

# A metric family is DECLARED two ways in the Go source: as the first string arg
# of a HELP/TYPE writer helper (`help("fak_x", ...)`, `writeCounter(b, "fak_x", ...)`,
# `writeHelpType(&b, "fak_x", ...)`), or written as a raw exposition literal where
# the name is immediately followed by `{` (labels) or a space (value). Both are
# emitted-metric signals; an MCP tool name (`case "fak_syscall":`) or a struct
# field (`"fak_native": v`) matches NEITHER, so the source of truth stays clean.
# The optional first arg is the exposition buffer the helper writes into — a Go
# identifier passed by value (`b,`) or by pointer (`&b,`); both forms are emitted.
_METRIC_DECL_RE = re.compile(
    r'(?:help|write[A-Za-z]*)\(\s*(?:&?[A-Za-z_]\w*\s*,\s*)?"(' + METRIC_PREFIX + r'[a-z0-9_]+)"')
_METRIC_EXPO_RE = re.compile(r'"(' + METRIC_PREFIX + r'[a-z0-9_]+)(?:\{| )')
_FAK_FAMILY_TOKEN_RE = re.compile(r'\b(' + METRIC_PREFIX + r'[a-z0-9_]+)\b')
# Every quoted "fak_..." literal in Go source, metric or not. Subtracting the
# emitted-metric set leaves the NON-metric identifiers that share the prefix —
# MCP tool names (`fak_syscall`, `fak_context_change`), route ids — which a doc
# legitimately references and which must NOT be flagged as a phantom metric.
_FAK_ANY_LITERAL_RE = re.compile(r'"(' + METRIC_PREFIX + r'[a-z0-9_]+)"')
_THEOREM_RE = re.compile(r"^#{2,}\s*THEOREM\b", re.IGNORECASE)
_WITNESS_RE = re.compile(r"\*\*\s*WITNESS", re.IGNORECASE)
_VERDICT_RE = re.compile(r"\*\*\s*VERDICT", re.IGNORECASE)
_TRACE_HEADER = "X-Trace-Id"


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


def normalize_family(name: str) -> str:
    """Strip a single Prometheus exposition suffix so a `<base>_bucket`/`_sum`/
    `_count` reference maps to the base family the source registers."""
    for suf in PROM_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def is_backed(ref: str, source: set[str]) -> bool:
    """Is a referenced fak_* family backed by an emitted family?

    Backed iff the reference, its histogram-suffix-stripped base, OR the reference
    plus a counter `_total` suffix is an emitted family. The `_total` forgiveness
    lets prose cite `fak_kernel_submits` for the emitted `fak_kernel_submits_total`
    without being flagged — a correct abbreviation is not drift."""
    if ref in source:
        return True
    base = normalize_family(ref)
    if base in source:
        return True
    if (ref + "_total") in source:
        return True
    if (base + "_total") in source:
        return True
    return False


def is_metric_shaped(token: str) -> bool:
    """A token that looks like a real metric family: the `fak_` prefix plus at
    least TWO non-empty segments (so `fak_gateway_up`, `fak_verdict_total` qualify
    but a bare-prefix mention `fak_gateway`, a trailing-underscore wildcard stem
    `fak_kernel_`, or a one-word id `fak_traced` do not). This is what separates a
    real metric reference from a prose fragment or a `fak_*` wildcard."""
    if not token.startswith(METRIC_PREFIX):
        return False
    parts = token[len(METRIC_PREFIX):].split("_")
    return len(parts) >= 2 and all(p for p in parts)


def extract_family_literals(go_text: str) -> set[str]:
    """The emitted-metric source of truth: every fak_* family DECLARED in Go source
    via a HELP/TYPE writer helper or written as a raw exposition literal. Excludes
    MCP tool names and struct-field literals that merely share the `fak_` prefix."""
    return set(_METRIC_DECL_RE.findall(go_text)) | set(_METRIC_EXPO_RE.findall(go_text))


def extract_family_tokens(text: str) -> set[str]:
    """Every metric-shaped fak_* family token referenced in a dashboard expr /
    alert expr / doc. The shape filter drops prose fragments and wildcard stems so
    only genuine metric references are checked against the source of truth."""
    return {t for t in _FAK_FAMILY_TOKEN_RE.findall(text) if is_metric_shaped(t)}


def dashboard_expr_text(json_text: str) -> str:
    """Concatenate every PromQL ``expr`` string in a Grafana dashboard JSON, so a
    reference is read from a panel QUERY — not a panel title, a legend, or a
    template-variable regex (which legitimately carries a ``fak_*`` wildcard stem).
    Falls back to the raw text if the JSON does not parse (the shape filter still
    guards against wildcard-stem false positives)."""
    try:
        obj = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return json_text
    exprs: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "expr" and isinstance(v, str):
                    exprs.append(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(obj)
    return "\n".join(exprs)


def alert_expr_text(yaml_text: str) -> str:
    """The text after every ``expr:`` key in a Prometheus alert-rules YAML, so a
    reference is read from a rule EXPRESSION — not a group ``- name:`` (which is a
    rule-group label that often mirrors a metric prefix) or a prose ``description:``
    (where a metric name is mentioned, not queried)."""
    out: list[str] = []
    for ln in yaml_text.split("\n"):
        m = re.match(r"\s*expr:\s*(.*)$", ln)
        if m:
            out.append(m.group(1))
    return "\n".join(out)


def log_event_field_keys(go_text: str) -> list[str]:
    """Field keys logged in a structured-log event map. From each `"event":`
    literal, scan forward to the emit call (`s.logf(`), collecting field keys from
    both the map literal (`"key":`) and conditional adds (`ev["key"] =`). Returns
    every key seen across all event windows (with duplicates → caller dedups)."""
    lines = go_text.split("\n")
    keys: list[str] = []
    n = len(lines)
    i = 0
    field_re = re.compile(r'"([a-z_]+)"\s*:')
    add_re = re.compile(r'ev\["([a-z_]+)"\]')
    while i < n:
        if '"event"' in lines[i] and ":" in lines[i]:
            j = i
            # bound the window so a malformed block can't run away
            while j < n and j < i + 60:
                ln = lines[j]
                for m in field_re.finditer(ln):
                    keys.append(m.group(1))
                for m in add_re.finditer(ln):
                    keys.append(m.group(1))
                if any(tok in ln for tok in LOG_EMIT_TOKENS):
                    break
                j += 1
            i = j + 1
            continue
        i += 1
    return keys


def split_theorems(md_text: str) -> list[str]:
    """Split a proof doc into per-theorem sections (each from a `## THEOREM`
    heading to the next, or EOF). Text before the first THEOREM is dropped."""
    lines = md_text.split("\n")
    sections: list[list[str]] = []
    cur: list[str] | None = None
    for ln in lines:
        if _THEOREM_RE.match(ln):
            if cur is not None:
                sections.append(cur)
            cur = [ln]
        elif cur is not None:
            cur.append(ln)
    if cur is not None:
        sections.append(cur)
    return ["\n".join(s) for s in sections]


def theorem_defects(rel: str, md_text: str) -> list[str]:
    """A proof-witness defect per theorem section that is either (A) unadjudicated
    — no VERDICT line — or (B) a PROVEN claim with no runnable WITNESS line. An
    honestly-OPEN / REFUTED / SCOPED-OUT theorem with no witness is NOT debt."""
    defects: list[str] = []
    for idx, sec in enumerate(split_theorems(md_text), start=1):
        head = sec.split("\n", 1)[0].lstrip("# ").strip()[:70]
        has_verdict = bool(_VERDICT_RE.search(sec))
        has_witness = bool(_WITNESS_RE.search(sec))
        if not has_verdict:
            defects.append(f"unadjudicated proof theorem in {rel} (#{idx} “{head}”): "
                           f"no VERDICT line — adjudicate it (PROVEN/OPEN/REFUTED)")
            continue
        verdict_line = next((ln for ln in sec.split("\n") if _VERDICT_RE.search(ln)), "")
        is_proven = "PROVEN" in verdict_line.upper() and "DISPROVEN" not in verdict_line.upper()
        if is_proven and not has_witness:
            defects.append(f"PROVEN theorem without a witness in {rel} (#{idx} “{head}”): "
                           f"add the **WITNESS.** command that re-derives it")
    return defects


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each returns
#   {kpi, group, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of observability-debt; soft = score-only judgment nudges.
# ---------------------------------------------------------------------------

def kpi_reference_integrity(kpi: str, refs_by_file: dict[str, set[str]],
                            source: set[str], noun: str) -> dict[str, Any]:
    """Shared core for dashboard / alert / doc reference integrity: every fak_*
    family a surface references must be a family the binary emits. ``refs_by_file``
    maps each surface file to the set of fak_* tokens it references."""
    defects: list[str] = []
    n_refs = 0
    n_files = 0
    for f in sorted(refs_by_file):
        refs = refs_by_file[f]
        if not refs:
            continue
        n_files += 1
        for ref in sorted(refs):
            n_refs += 1
            if not is_backed(ref, source):
                defects.append(f"phantom metric in {f}: {ref} — the binary emits no "
                               f"such family; fix the reference to an emitted one")
    score = _clamp(100 - 12 * len(defects))
    if defects:
        detail = f"{len(defects)} phantom {noun} reference(s) across {n_files} file(s)"
    else:
        detail = f"every fak_* reference in {n_files} {noun}(s) is emitted ({n_refs} ref(s))"
    return {"kpi": kpi, "group": KPI_GROUP[kpi], "score": score,
            "detail": detail, "defects": defects, "soft": []}


def kpi_trace_correlation(header_present: bool, traced_log_events: list[str],
                          all_log_events: list[str], response_header_set: bool) -> dict[str, Any]:
    """The trace surface that ties one request across metrics, logs, and the verdict:
    the X-Trace-Id header constant must exist, be set on the response, and every
    structured-log event must carry a trace_id. Each missing leg is one defect."""
    defects: list[str] = []
    if not header_present:
        defects.append(f"no {_TRACE_HEADER} trace header constant in the gateway source — "
                       f"a request cannot be correlated across surfaces")
    if not response_header_set:
        defects.append(f"the gateway never sets the {_TRACE_HEADER} response header — "
                       f"a caller cannot follow its own request id back")
    untraced = [e for e in all_log_events if e not in set(traced_log_events)]
    for e in sorted(untraced):
        defects.append(f"structured-log event “{e}” carries no trace_id — it cannot be "
                       f"joined to the metrics/verdict for the same request")
    score = _clamp(100 - 20 * len(defects))
    detail = ("trace surface intact: X-Trace-Id honored/minted, response header set, "
              f"all {len(all_log_events)} log event(s) carry trace_id"
              if not defects else f"{len(defects)} trace-correlation gap(s)")
    return {"kpi": "trace_correlation", "group": "instrumentation", "score": score,
            "detail": detail, "defects": defects, "soft": []}


def kpi_log_privacy(leaks: list[str], n_events: int) -> dict[str, Any]:
    """The privacy invariant: the structured access log is fully auditable WITHOUT
    logging the request body, tool arguments, or result content. A forbidden field
    in a log event map is a leak — one unit of debt, and a security regression."""
    defects = [f"log event leaks a payload field: {leak} — a gate can be audited "
               f"without logging the payload it protects; drop the field"
               for leak in sorted(leaks)]
    score = _clamp(100 - 25 * len(defects))
    detail = (f"{n_events} structured-log event(s) carry no payload field"
              if not defects else f"{len(defects)} payload-leaking log field(s)")
    return {"kpi": "log_privacy", "group": "instrumentation", "score": score,
            "detail": detail, "defects": defects, "soft": []}


def kpi_metric_doc_coverage(emitted: set[str], surfaced: set[str]) -> dict[str, Any]:
    """SOFT: emitted fak_* families surfaced in NO doc, dashboard, or alert — a
    signal the binary emits that no operator is told to look at. Scored on a rate
    (growth-invariant), never hard debt: documenting every internal counter is
    noise, not visibility."""
    unsurfaced = sorted(f for f in emitted if f not in surfaced)
    soft = [f"emitted but unsurfaced metric: {f} (no doc/dashboard/alert references it)"
            for f in unsurfaced[:COVERAGE_SAMPLE]]
    if len(unsurfaced) > COVERAGE_SAMPLE:
        soft.append(f"... and {len(unsurfaced) - COVERAGE_SAMPLE} more unsurfaced metric(s)")
    rate = len(unsurfaced) / max(1, len(emitted))
    return {"kpi": "metric_doc_coverage", "group": "instrumentation",
            "score": _clamp(100 - min(55, round(70 * rate))),
            "detail": (f"{len(emitted) - len(unsurfaced)}/{len(emitted)} emitted families "
                       f"surfaced in a doc/dashboard/alert ({100 - round(100 * rate)}%)"),
            "defects": [], "soft": soft}


def kpi_proof_witness(defects_by_doc: dict[str, list[str]], n_theorems: int,
                      n_docs: int) -> dict[str, Any]:
    """Every proof THEOREM must be adjudicated (a VERDICT) and every PROVEN one must
    carry a runnable WITNESS — otherwise a correctness claim is asserted, not
    verifiable. Each such theorem is one unit of debt."""
    defects: list[str] = []
    for f in sorted(defects_by_doc):
        defects.extend(defects_by_doc[f])
    bad = len(defects)
    score = _clamp(100 - 8 * bad)
    detail = (f"{n_theorems} theorem(s) across {n_docs} proof(s); all adjudicated and "
              f"every PROVEN one witnessed" if not bad
              else f"{bad} theorem(s) unadjudicated or PROVEN-without-witness "
                   f"across {n_theorems} theorem(s)")
    return {"kpi": "proof_witness", "group": "verifiability", "score": score,
            "detail": detail, "defects": defects, "soft": []}


def kpi_ship_integrity(dos: dict[str, Any] | None) -> dict[str, Any]:
    """The DOS grounding (reused from code_quality_scorecard): each RESIDUAL commit
    — a claim the diff could not witness — is one unit of debt the kernel itself
    flagged. Fails OPEN when dos is absent (scored 100, a soft unmeasured note)."""
    if dos is None:
        return {"kpi": "ship_integrity", "group": "verifiability", "score": 100,
                "detail": "skipped (--no-dos)", "defects": [],
                "soft": ["dos review not run (--no-dos / dos unavailable)"]}
    if dos.get("error"):
        return {"kpi": "ship_integrity", "group": "verifiability", "score": 100,
                "detail": f"UNMEASURED (dos review unavailable): {str(dos['error'])[:60]}",
                "defects": [],
                "soft": [f"ship_integrity UNMEASURED — dos unavailable, scored 100 "
                         f"(fail-open, not a witnessed-clean review): {str(dos['error'])[:100]}"]}
    residual = dos.get("residual", []) or []
    n = len(residual)
    rng = dos.get("rev_range", "?")
    defects = [f"unwitnessed ship (RESIDUAL) in {rng}: {r.get('sha', '?')} "
               f"{str(r.get('subject', ''))[:80]}" for r in residual]
    cleared = dos.get("cleared_rate")
    detail = (f"{dos.get('checkable', '?')} checkable commit(s) in {rng}, "
              f"{n} residual, cleared_rate {cleared}")
    return {"kpi": "ship_integrity", "group": "verifiability",
            "score": _clamp(100 - 25 * n), "detail": detail, "defects": defects, "soft": []}


# ---------------------------------------------------------------------------
# Fold: KPIs -> composite score, grade, observability-debt, control-pane payload.
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, kpis: list[dict[str, Any]],
                  emitted_count: int = 0, error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error,
            "next_action": "fix the read (run from repo ROOT, with git), then re-run",
            "workspace": workspace, "corpus": {}, "kpis": [],
        }
    by_name = {k["kpi"]: k for k in kpis}
    score = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"]
                      for n in KPI_WEIGHTS if n in by_name), 1)
    observability_debt = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)
    grade = grade_letter(score)
    debt_by_group = {g: 0 for g in GROUPS}
    for k in kpis:
        debt_by_group[k["group"]] += len(k["defects"])
    breakdown = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    corpus = {
        "score": score, "grade": grade,
        "observability_debt": observability_debt,
        "soft_signals": n_soft,
        "emitted_families": emitted_count,
        "debt_by_group": debt_by_group,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
    }

    if observability_debt == 0:
        ok, verdict, finding = True, "OK", "observable"
        reason = (f"observability clean: score {score}/100 (grade {grade}), zero "
                  f"observability-debt across {len(kpis)} KPIs ({n_soft} advisory signal(s))")
        next_action = "no required edit; re-run after the next metric / dashboard / proof change"
    else:
        ok, verdict, finding = False, "ACTION", "observability_debt"
        worst = breakdown[0]
        reason = (f"{observability_debt} unit(s) of observability-debt; score {score}/100 "
                  f"(grade {grade}); heaviest: {worst['kpi']} ({worst['debt']} defect(s))")
        next_action = ("retire observability-debt worst-first (see corpus.breakdown + "
                       "per-KPI defects): fix phantom dashboard/alert/doc metric references, "
                       "close trace/log gaps, witness PROVEN proofs, clear RESIDUAL ships; "
                       "re-run with --compare to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "kpis": kpis,
    }


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


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _dos_review(root: Path, rev_range: str) -> dict[str, Any]:
    try:
        p = subprocess.run(["dos", "review", rev_range, "--json"], cwd=str(root),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)[:200]}
    if p.returncode not in (0, 1):  # 0 clean, 1 residual present — both valid verdicts
        return {"error": (p.stderr or p.stdout or "dos review failed").strip()[:200]}
    try:
        return json.loads(p.stdout)
    except (json.JSONDecodeError, ValueError):
        return {"error": "dos review emitted non-JSON"}


def _is_metric_doc(rel: str) -> bool:
    if not rel.endswith(".md") or rel == GENERATED_SNAPSHOT:
        return False
    return rel.startswith(DOC_SURFACE_PREFIXES) or rel in DOC_SURFACE_EXTRA


def gather(root: Path, *, run_dos: bool = True,
           dos_range: str = "HEAD~20..HEAD") -> tuple[list[dict[str, Any]], int]:
    """Read the git-tracked tree and run every pure KPI. Returns (kpis, emitted_count)."""
    tracked = _git_lines(["ls-files"], root)

    # --- the source of truth: every fak_* family the Go binary emits ---
    go_src = [f for f in tracked
              if f.endswith(".go") and not f.endswith("_test.go")
              and f.startswith(SOURCE_DIRS)]
    emitted: set[str] = set()
    all_fak_literals: set[str] = set()
    gateway_go_text_parts: list[str] = []
    for f in go_src:
        text = _safe_read(root / f)
        emitted |= extract_family_literals(text)
        all_fak_literals |= set(_FAK_ANY_LITERAL_RE.findall(text))
        if f.startswith("internal/gateway/"):
            gateway_go_text_parts.append(text)
    gateway_text = "\n".join(gateway_go_text_parts)
    # NON-metric fak_ identifiers (MCP tool names, route ids) a doc may reference
    # without it being a phantom metric — excluded from doc drift.
    nonmetric_fak = all_fak_literals - emitted

    # --- correlation: dashboard / alert / doc reference integrity ---
    dash_dir = root / DASHBOARD_GLOB_DIR
    dash_refs: dict[str, set[str]] = {}
    if dash_dir.exists():
        for p in sorted(dash_dir.glob("*.json")):
            rel = p.relative_to(root).as_posix()
            dash_refs[rel] = extract_family_tokens(dashboard_expr_text(_safe_read(p)))
    alert_refs: dict[str, set[str]] = {}
    if (root / ALERTS_REL).exists():
        alert_refs[ALERTS_REL] = extract_family_tokens(
            alert_expr_text(_safe_read(root / ALERTS_REL)))
    doc_refs: dict[str, set[str]] = {}
    for f in tracked:
        if not _is_metric_doc(f):
            continue
        text = _safe_read(root / f)
        toks = extract_family_tokens(text) - nonmetric_fak
        if toks:
            doc_refs[f] = toks

    # --- instrumentation: trace surface + log privacy from gateway source ---
    header_present = (_TRACE_HEADER in gateway_text)
    response_header_set = bool(re.search(
        r"(w\.Header\(\)\.Set\(\s*traceHeader|Header\(\)\.Set\(\s*\"" + re.escape(_TRACE_HEADER) + r"\")",
        gateway_text))
    # log events + which carry trace_id: scan each event window
    all_events: list[str] = []
    traced_events: list[str] = []
    leaks: list[str] = []
    lines = gateway_text.split("\n")
    event_name_re = re.compile(r'"event"\s*:\s*"([a-z_]+)"')
    for i, ln in enumerate(lines):
        m = event_name_re.search(ln)
        if not m:
            continue
        ev_name = m.group(1)
        all_events.append(ev_name)
        window = "\n".join(lines[i:i + 60])
        win_keys = set(log_event_field_keys(window)) | set(re.findall(r'"([a-z_]+)"\s*:', window))
        if "trace_id" in window:
            traced_events.append(ev_name)
        for bad in sorted(FORBIDDEN_LOG_FIELDS & win_keys):
            leaks.append(f"{ev_name}.{bad}")

    # --- instrumentation: metric/doc coverage ---
    surfaced: set[str] = set()
    for refs in list(dash_refs.values()) + list(alert_refs.values()) + list(doc_refs.values()):
        for ref in refs:
            # map a reference back to an emitted family (so a surfaced `_bucket`
            # reference counts its base family as surfaced)
            for cand in (ref, normalize_family(ref), ref + "_total", normalize_family(ref) + "_total"):
                if cand in emitted:
                    surfaced.add(cand)

    # --- verifiability: proof witness coverage ---
    proofs_dir = root / PROOFS_DIR
    proof_defects_by_doc: dict[str, list[str]] = {}
    n_theorems = 0
    n_proof_docs = 0
    if proofs_dir.exists():
        for p in sorted(proofs_dir.glob("*.md")):
            rel = p.relative_to(root).as_posix()
            text = _safe_read(p)
            secs = split_theorems(text)
            if not secs:
                continue
            n_proof_docs += 1
            n_theorems += len(secs)
            d = theorem_defects(rel, text)
            if d:
                proof_defects_by_doc[rel] = d

    # --- verifiability: DOS ship integrity ---
    dos = _dos_review(root, dos_range) if run_dos else None

    kpis = [
        kpi_reference_integrity("dashboard_integrity", dash_refs, emitted, "dashboard"),
        kpi_reference_integrity("alert_integrity", alert_refs, emitted, "alert"),
        kpi_reference_integrity("doc_metric_drift", doc_refs, emitted, "doc"),
        kpi_trace_correlation(header_present, traced_events, all_events, response_header_set),
        kpi_log_privacy(leaks, len(all_events)),
        kpi_metric_doc_coverage(emitted, surfaced),
        kpi_proof_witness(proof_defects_by_doc, n_theorems, n_proof_docs),
        kpi_ship_integrity(dos),
    ]
    return kpis, len(emitted)


def collect(workspace: Path, *, run_dos: bool = True,
            dos_range: str = "HEAD~20..HEAD") -> dict[str, Any]:
    root = workspace.resolve()
    if not (root / ".git").exists() and not _git_lines(["rev-parse", "--git-dir"], root):
        return build_payload(workspace=str(root), kpis=[],
                             error=f"not a git repo at {root} — run from the repo ROOT")
    if not (root / "go.mod").exists():
        return build_payload(workspace=str(root), kpis=[],
                             error=f"no go.mod at {root} — the Go module is the repo ROOT")
    kpis, emitted_count = gather(root, run_dos=run_dos, dos_range=dos_range)
    return build_payload(workspace=str(root), kpis=kpis, emitted_count=emitted_count)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    lines = [
        f"observability-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· OBSERVABILITY-DEBT {c.get('observability_debt', 0)} "
         f"· {c.get('soft_signals', 0)} advisory "
         f"· {c.get('emitted_families', 0)} emitted fak_* families"),
        ("debt by group: " + "  ".join(
            f"{g}:{c.get('debt_by_group', {}).get(g, 0)}" for g in GROUPS)),
        "",
        "per-KPI (worst first):",
        f"  {'score':>5} {'debt':>4}  {'group':<15} {'kpi':<20} detail",
    ]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['group']:<15} "
                     f"{b['kpi']:<20} {b['detail']}")
    lines.append("")
    lines.append("observability-debt work-list:")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        lines.append(f"  {k['kpi']} ({len(k['defects'])}):")
        for it in k["defects"][:DRIFT_SAMPLE]:
            lines.append(f"      - {it}")
        if len(k["defects"]) > DRIFT_SAMPLE:
            lines.append(f"      ... and {len(k['defects']) - DRIFT_SAMPLE} more")
    if not any_defect:
        lines.append("  (none — zero observability-debt)")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    out: list[str] = []
    out.append("---")
    out.append('title: "fak observability scorecard — the observability-debt measuring stick"')
    out.append('description: "fak\'s deterministic observability scorecard: eight KPIs across '
               'correlation, instrumentation, and verifiability, folded into a composite score '
               'and the headline observability-debt metric, re-derived from the git-tracked tree '
               'with the Go binary as the metric source of truth."')
    out.append("---")
    out.append("")
    out.append("# Observability scorecard")
    out.append("")
    if stamp:
        out.append(f"<!-- observability-scorecard: {stamp} · process: tools/observability_scorecard.py -->")
        out.append("")
    out.append("This is the measuring stick for the observability-10x program — the counterpart "
               "of the code, docs, and repo-hygiene scorecards aimed at the **observability "
               "plane**: the metrics the gateway emits, the dashboards and alerts that read them, "
               "the docs that tell an operator which metric to query, the trace-id that ties a "
               "request across logs, and the proofs / ship-audit that let a claim be verified "
               "rather than asserted. Every number below is re-derived from the git-tracked tree "
               "by `tools/observability_scorecard.py` — no hand-entry. The **Go binary is the "
               "metric source of truth**: a dashboard, alert, or doc reference is a defect iff "
               "the binary emits no such `fak_*` family. The headline metric is "
               "**observability-debt**: the count of concrete, mechanical defects you fix by "
               "*making the live system more visible and more verifiable* — a phantom panel, an "
               "alert on a metric that does not exist, a doc that misdirects an operator, a "
               "broken trace surface, a log line that leaks a payload, a PROVEN proof with no "
               "witness, an unwitnessed ship.")
    out.append("")
    out.append("> Regenerate: `python tools/observability_scorecard.py --markdown --stamp DATE > docs/OBSERVABILITY-SCORECARD.md`")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Observability-debt (total HARD defects)** | **{c.get('observability_debt', 0)}** |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append(f"| Emitted `fak_*` metric families (source of truth) | {c.get('emitted_families', 0)} |")
    out.append(f"| Advisory (soft) signals | {c.get('soft_signals', 0)} |")
    g = c.get("debt_by_group", {})
    out.append(f"| Debt by group | correlation:{g.get('correlation', 0)} · "
               f"instrumentation:{g.get('instrumentation', 0)} · "
               f"verifiability:{g.get('verifiability', 0)} |")
    out.append("")
    out.append("## Per-KPI")
    out.append("")
    out.append("Eight KPIs, each 0–100, in three groups. `debt` = units of HARD "
               "observability-debt. `metric_doc_coverage` is advisory (it scores but emits no "
               "hard debt — documenting every internal counter is noise, not visibility); "
               "`ship_integrity` is HEAD-relative and fails open when `dos` is absent.")
    out.append("")
    out.append("| Group | KPI | Score | Debt | Detail |")
    out.append("|---|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    out.append("## Observability-debt work-list")
    out.append("")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        out.append(f"### `{k['kpi']}` ({k['group']}) — {len(k['defects'])} defect(s), score {k['score']}")
        for it in k["defects"]:
            out.append(f"- {it}")
        out.append("")
    if not any_defect:
        out.append("No observability-debt: every dashboard, alert, and doc points at a metric the "
                   "binary emits; the trace surface is intact; the log leaks no payload; every "
                   "PROVEN proof is witnessed; no ship is unwitnessed. 🎉")
        out.append("")
    return "\n".join(out)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("observability_debt", 0), cur.get("observability_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ratio = "∞ (zero)" if cd == 0 else f"{bd / cd:.1f}×"
    lines = [
        f"observability-debt: {bd} -> {cd}   ({ratio} fewer defects)",
        f"score:              {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<15} {gb} -> {gc}")
    # the 10x gate (matches seo_aeo_scorecard): need observability-debt <= base/10
    need = max(1, bd // 10)
    if cd <= need:
        lines.append(f"VERDICT: >=10x observability-debt reduction achieved ({bd} -> {cd}).")
    else:
        lines.append(f"VERDICT: not yet 10x — need observability-debt <= {need} (now {cd}).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Observability scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the snapshot markdown body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--no-dos", action="store_true", help="skip the dos ship-integrity probe")
    ap.add_argument("--dos-range", default="HEAD~20..HEAD", help="rev-range for dos review")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the observability-debt delta vs a prior baseline JSON (the 10x gate)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, run_dos=not args.no_dos, dos_range=args.dos_range)

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
