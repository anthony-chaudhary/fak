#!/usr/bin/env python3
"""Unified scorecard debt control-pane — fold every *-debt into one tracked trend.

Native port: this fold is ported to Go in ``internal/scorecardpane`` and exposed as
``fak scorecard control-pane [--json|--check|--pin]`` (issue #1449) — typed contracts,
one process startup, byte-compatible ``--json`` shapes. This script remains as a
compatibility shim until the Python baseline can shrink.

The repo has deterministic scorecards, each emitting a debt integer plus a
control-pane payload (``schema/ok/verdict/finding/reason/next_action``): docs,
README freshness, code, doc-appeal, seo, demo-quality, demo-robustness, repo-hygiene, the one
OUTWARD-facing stick — industry-parity (fak vs SOTA) — agent-readiness (can an
agent discover, adopt, and build on fak), product (can a PERSON pick up each fak
concept and use it today — durable / real / useful-today), persona (are the
top-10 personas who land on fak — free-tier dev through researcher — each served),
and stability (can we tell when a regression / tail-wag / confusion landed, and
revert to a stable version). They run independently and advisory. Nothing folds
{doc_debt, readme_debt, code_debt, appeal_debt, seo_debt, demo_debt, robustness_debt,
hygiene_debt, parity_debt, friction_debt, product_debt, persona_debt,
stability_debt} into one number, pins a per-metric baseline, and shows the trend
commit-over-commit.

This is that fold — the RSI checking layer for the whole scorecard family. It
runs each scorecard, extracts the debt integer + grade, sums one portfolio
``total_debt``, and compares against a pinned per-metric baseline so the answer
to "is the repo getting better or worse" is one query.

  python tools/scorecard_control_pane.py            # human snapshot + trend
  python tools/scorecard_control_pane.py --json      # machine payload
  python tools/scorecard_control_pane.py --pin       # pin today's debt as the baseline
  python tools/scorecard_control_pane.py --check     # CI ratchet gate (fail only on regression)

The baseline lives in a tracked file (``tools/scorecard_baseline.json``) so the
trend is commit-over-commit and shared: re-pin after a debt drop to ratchet it
down. Pure-stdlib Python, repo root like the other honesty gates.

``--check`` is the RSI ratchet the repo-3x epic (#506) names: it turns the one
folded number into an enforceable gate. Unlike the default exit code (green only
at ZERO debt), ``--check`` is GREEN while the portfolio holds at-or-below its
pinned baseline and RED only when debt *regresses* above it (or a scorecard
fails to report). That is the honest CI contract — debt may stay or fall, never
silently rise — without demanding the whole family be at zero first. Issue #509.
The README freshness scorecard is deliberately wired here, not as a bespoke
``--min-score`` CI line: its baseline pins ``readme_debt`` at zero, so a front-page
score-affordance regression reds through the existing green ratchet (#779/#893).

The portfolio ratchet has one blind spot: it folds every metric into one sum, so
a single metric's regression can hide under another metric's improvement (seo
rose 6->8 while the portfolio fell 44->40 — the ratchet stayed green, and a blind
``--pin`` would have blessed the seo rise as the new floor). The per-metric
EARLY-WARNING lens (#712) closes it: any metric whose debt rose vs its pinned
value is reported as an advisory WARN even when the portfolio total is green —
the trend carries an ``early_warning`` list, ``--check`` appends it to the
RATCHET OK line WITHOUT tripping the gate (the portfolio ratchet semantics are
unchanged), and the human snapshot prints it. So a hidden per-metric regression
surfaces BEFORE a re-pin locks it in.

There is a second, deeper blind spot the per-metric lens alone doesn't close:
the raw ``total_debt`` sums HETEROGENEOUS units. One ``code`` defect is a
god-file (bounded, ~tens); one ``slop``/``disambiguation`` defect is a single
occurrence over the whole tree (unbounded, hundreds). So the portfolio sum is
~91% two occurrence-counters (slop 535 + disambiguation 550 of 1187), and a real
regression in any of the other 23 metrics sits below their noise floor — the
universal ranking has stopped DISCRIMINATING across the family it folds. The
grade-weighted lens closes it: every scorecard already grades A-F on identical
thresholds (a scale-invariant signal the fold collected but never used), so this
folds those grades into one ``grade_debt`` where each metric contributes by
SEVERITY (A=0/B=1/C=2/D=4/F=8), not by raw unit count — a ``slop`` A->B
regression now weighs exactly as much as a ``stability`` A->B. ``grade_debt``
runs ALONGSIDE the raw ratchet (``total_debt`` and its gate are unchanged); its
own per-commit delta is a second advisory axis a raw-count improvement can no
longer mask.

The "stays Excellent" half (#1423, epic #1414): once a surface reaches grade A,
an advisory grade axis is not enough — a silent A->B regression must RED the
build, not just warn. So ``--check`` now GATES on the grade axis, not only the
raw sum. The baseline carries per-metric ``grade_weights`` (each scorecard's
letter-grade severity), and ``--check`` reds when any metric's grade slips below
its pin (A->B) OR the portfolio ``grade_debt`` rises — even when ``total_debt``
held flat. This mirrors the milestone climb ratchet (#1442): a distinct gate that
fires on a same-debt rung swap. It is HARD by default and env-overridable to the
old advisory behavior via ``FAK_SCORECARD_GRADE_RATCHET=0`` (for a deliberate
one-off pin on a known-dirty tree). Re-pin with ``--pin`` after a real drop to
ratchet both axes down together.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
from pathlib import Path
from typing import Any
install_no_window_subprocess_defaults(subprocess)

SCHEMA = "fak-scorecard-control-pane/1"
BASELINE_SCHEMA = "fak-scorecard-control-pane.baseline/1"
BASELINE_REL = "tools/scorecard_baseline.json"

# The grade-carrying ratchet knob (#1423 / #1414's "stays Excellent" half). When
# set, ``--check`` reds the gate on a SEVERITY regression (grade_debt rose vs the
# pinned baseline) EVEN IF the raw-unit total held — the A->B slip a flat
# total_debt would otherwise hide. Mirrors the milestone climb ratchet (#1442):
# a distinct gate that fires on a same-debt rung swap. Default ON; set
# FAK_SCORECARD_GRADE_RATCHET=0 to demote it back to advisory (the old behavior).
GRADE_RATCHET_ENV = "FAK_SCORECARD_GRADE_RATCHET"


def grade_ratchet_hard() -> bool:
    """True when the grade-debt ratchet should RED the gate (the default).

    Off only when the operator explicitly sets ``FAK_SCORECARD_GRADE_RATCHET`` to
    a falsey value (0/false/no/off) — then a severity regression is advisory-only,
    the pre-#1423 behavior, for a deliberate one-off pin on a known-dirty tree.
    """
    raw = os.environ.get(GRADE_RATCHET_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}

# --- grade-weighted portfolio lens (#712-follow-on) ------------------------
# The raw ``total_debt`` fold sums heterogeneous units: one ``code`` defect is a
# god-file (bounded, ~tens); one ``slop``/``disambiguation`` defect is a single
# occurrence over the WHOLE tree (unbounded, hundreds). So the portfolio sum is
# ~91% two occurrence-counters (slop 535 + disambiguation 550 of 1187), and a
# real regression in any of the other 23 metrics sits below their noise floor —
# the universal ranking has stopped discriminating across the family it folds.
#
# Every scorecard already emits the SAME scale-invariant signal: a letter grade
# on identical thresholds (score >=90 A / 80 B / 70 C / 60 D / else F). This lens
# folds those grades into one ``grade_debt`` where each metric contributes by
# SEVERITY, not by raw unit count — a ``slop`` A->B regression now counts exactly
# as much as a ``stability`` A->B regression. It runs ALONGSIDE the raw ratchet
# (``total_debt`` is unchanged); ``grade_debt`` is the cross-metric-comparable
# number, and its own per-commit delta is a second early-warning axis that a
# raw-count improvement can no longer mask.
GRADE_DEBT: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 4, "F": 8}

# The 4 scorecards that grade per-item but emit NO corpus-level letter grade
# (docs/seo/demo/robustness). Each DOES emit a corpus-level aggregate SCORE; this
# maps the metric key -> that score field so the lens can derive the TRUE grade
# from the score (scale-invariant, on the same 90/80/70/60 ladder the scorecards'
# own grade_letter uses) instead of from raw debt magnitude. Without this, an
# A-grade surface carrying occurrence-count debt is mis-ranked F — the very
# units-not-severity error the grade lens exists to kill (verified live: seo
# 92.5, demo 96.0, robustness 92.4, docs 96.9 — all A, all debt-derived to F/B).
# Keyed on the SCORECARDS `key` (note: the docs metric's key is "doc", not "docs").
SCORE_KEYS: dict[str, str] = {
    "doc": "mean_score",
    "seo": "overall_score",
    "demo": "mean_score",
    "robustness": "mean_score",
    "learning": "mean_score",
}


def grade_from_score(score: float) -> str:
    """A-F on the family's shared 90/80/70/60 ladder — the SAME thresholds every
    scorecard's own ``grade_letter`` uses, so a score-derived grade reproduces
    exactly the letter the scorecard would have emitted had it surfaced one."""
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def derive_grade(debt: int) -> str:
    """Last-resort grade for a scorecard that emits neither a letter NOR a score.

    A family member that reports only ``*_debt`` (readme-freshness) would be
    invisible to the grade-weighted lens without a fallback, re-opening the blind
    spot it closes. This maps debt onto the A-F ladder by magnitude. It is
    SCALE-VARIANT (debt units aren't comparable across metrics), so it is the
    lowest-precedence source — used only when no letter and no score exist.
    """
    if debt <= 0:
        return "A"
    if debt <= 2:
        return "B"
    if debt <= 5:
        return "C"
    if debt <= 10:
        return "D"
    return "F"


def display_grade(metric: dict[str, Any]) -> str:
    """The single source of truth for a metric's effective letter grade.

    Three-tier precedence: the scorecard's own EMITTED letter (scale-invariant) >
    a SCORE-derived letter on the shared ladder (scale-invariant) > a DEBT-derived
    letter by magnitude (scale-variant, last resort). Both the severity weight and
    the rendered breakdown read this, so the number and the displayed letter can
    never diverge.
    """
    grade = metric.get("grade")
    if isinstance(grade, str) and grade.upper() in GRADE_DEBT:
        return grade.upper()
    score = metric.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return grade_from_score(float(score))
    debt = metric.get("debt")
    return derive_grade(int(debt)) if isinstance(debt, int) and not isinstance(debt, bool) else "F"


def grade_weight(metric: dict[str, Any]) -> int:
    """The severity weight one measured metric contributes to ``grade_debt``,
    via the shared three-tier :func:`display_grade` precedence."""
    return GRADE_DEBT[display_grade(metric)]

# The scorecard family, in the canonical order the issue lists them. Each entry
# binds the scorecard's script to the debt integer it emits; the runner folds
# every debt key into one portfolio number.
SCORECARDS: list[dict[str, str]] = [
    {"key": "doc", "debt": "doc_debt", "script": "docs_scorecard.py", "label": "docs"},
    {"key": "readme", "debt": "readme_debt", "script": "readme_freshness_audit.py", "label": "readme-freshness"},
    # The control pane already runs inside the CI chain after build/vet/gofmt.
    # Keep this fold static so the live smoke stays inside its 120s/card budget
    # instead of duplicating the toolchain passes a second time.
    {"key": "code", "debt": "code_debt", "script": "code_quality_scorecard.py", "args": "--no-toolchain", "label": "code"},
    {"key": "appeal", "debt": "appeal_debt", "script": "doc_appeal_scorecard.py", "label": "doc-appeal"},
    {"key": "seo", "debt": "seo_debt", "script": "seo_aeo_scorecard.py", "label": "seo"},
    {"key": "demo", "debt": "demo_debt", "script": "demo_quality_scorecard.py", "label": "demo-quality"},
    {"key": "robustness", "debt": "robustness_debt", "script": "demo_robustness_scorecard.py", "label": "demo-robustness"},
    {"key": "hygiene", "debt": "hygiene_debt", "script": "repo_hygiene_scorecard.py", "label": "repo-hygiene"},
    {"key": "parity", "debt": "parity_debt", "script": "industry_scorecard.py", "label": "industry-parity"},
    {"key": "sota", "debt": "sota_debt", "script": "sota_coverage_scorecard.py", "label": "sota-coverage"},
    {"key": "agent", "debt": "friction_debt", "script": "agent_readiness_scorecard.py", "label": "agent-readiness"},
    {"key": "product", "debt": "product_debt", "script": "product_scorecard.py", "label": "product"},
    {"key": "persona", "debt": "persona_debt", "script": "persona_readiness_scorecard.py", "label": "persona"},
    {"key": "stability", "debt": "stability_debt", "script": "stability_scorecard.py", "label": "stability"},
    {"key": "slop", "debt": "slop_debt", "script": "code_slop_scorecard.py", "label": "code-slop"},
    {"key": "steer", "debt": "steerability_debt", "script": "steerability_scorecard.py", "label": "steerability"},
    {"key": "conflation", "debt": "conflation_debt", "script": "", "cmd": "go run ./cmd/fak conflation-scorecard --json", "label": "conflation"},
    # UI/UX-quality of the terminal surface (the `fak console` panes, `fak info`
    # overlay, `fak guard --split`): rune-safe truncation, cell-aware column pads,
    # empty-state branches, info-legend + console-help coverage. Go-native, no GPU,
    # deterministic, graded against the render source (the source IS the oracle).
    {"key": "ui_quality", "debt": "ui_quality_debt", "script": "", "cmd": "go run ./cmd/fak ui-quality-scorecard --json", "label": "ui-quality"},
    {"key": "disambiguation", "debt": "disambiguation_debt", "script": "concept_disambiguation_scorecard.py", "label": "concept-disambiguation"},
    {"key": "intent_literal", "debt": "intent_literal_debt", "script": "intent_literal_scorecard.py", "label": "intent-literal"},
    {"key": "tokendefaults", "debt": "token_defaults_debt", "script": "", "cmd": "go run ./cmd/fak token-defaults-scorecard --json", "label": "token-defaults"},
    {"key": "guard_rsi", "debt": "guard_rsi_debt", "script": "", "cmd": "go run ./cmd/fak guard-rsi-scorecard --json", "label": "guard-rsi"},
    {"key": "dogfood", "debt": "dogfood_debt", "script": "", "cmd": "go run ./cmd/fak dogfood-score --json", "label": "dogfood-loop"},
    {"key": "conceptusage", "debt": "conceptusage_debt", "script": "", "cmd": "go run ./cmd/fak concept-usage-score --json", "label": "concept-usage"},
    {"key": "maturity", "debt": "maturity_debt", "script": "", "cmd": "go run ./cmd/fak maturity --json", "label": "maturity"},
    {"key": "growth", "debt": "growth_debt", "script": "", "cmd": "go run ./cmd/fak coverage-matrix --json", "label": "growth-debt"},
    {"key": "support_maturity", "debt": "support_maturity_debt", "script": "", "cmd": "go run ./cmd/fak support-maturity-scorecard --json", "label": "support-maturity"},
    # The milestone scorecard (#1444, epic #1436): folds the milestone report's own
    # two dimensions (the maturity CLIMB distance-to-MATURED + the epic ROADMAP open
    # children) into one milestone_debt with a worst-first retire worklist, so the RSI
    # loop retires milestones like every other surface. Composes -- not duplicates --
    # support_maturity: that card fences each cell to its regime ceiling, this one
    # scores raw distance-to-matured across the grid as the headline climb + roadmap.
    {"key": "milestone", "debt": "milestone_debt", "script": "", "cmd": "go run ./cmd/fak milestone-scorecard --json", "label": "milestone"},
    # The milestone CLIMB ratchet (#1442, epic #1436): a DISTINCT gate from
    # milestone_debt. The two witnessed climb KPIs (matured_cells + milestone_progress)
    # are pinned in docs/milestones/baseline.json; climb_ratchet_debt is 1 when EITHER
    # regresses below the pin (else 0), so a same-debt rung swap that lowers the matured
    # count -- invisible to milestone_debt -- still reds the control pane here. Re-pin on
    # a real climb improvement with `fak milestone-scorecard --pin`.
    {"key": "milestone_climb", "debt": "climb_ratchet_debt", "script": "", "cmd": "go run ./cmd/fak milestone-scorecard --ratchet --json", "label": "milestone-climb"},
    # The agentic-coding loop-index (#1152, dev-ex epic #1148 spine): folds the six
    # loop stages (orient->plan->act->verify->ship->learn) into one loopindex_debt.
    # Registered here so a stage UN-WIRING (a regressed default) reds the ratchet —
    # the spine's "a regression reds the gate" DoD. Go-native, no GPU, deterministic.
    {"key": "loopindex", "debt": "loopindex_debt", "script": "", "cmd": "go run ./cmd/fak loop-index-scorecard --json", "label": "loop-index"},
    # Operator-heaviness (#1504): folds the operator-facing surface pressure into
    # the control pane through heaviness_debt while heaviness_pressure stays the
    # unbounded drift number operators watch over time.
    {"key": "heaviness", "debt": "heaviness_debt", "script": "", "cmd": "go run ./cmd/fak operator heaviness --json", "label": "operator-heaviness"},
    {"key": "propagation", "debt": "propagation_debt", "script": "", "cmd": "go run ./cmd/fak propagation-scorecard --json", "label": "propagation"},
    {"key": "claim_repro", "debt": "claim_repro_debt", "script": "claim_repro_scorecard.py", "label": "claim-repro"},
    {"key": "release", "debt": "release_debt", "script": "release_readiness_scorecard.py", "label": "release-readiness"},
    # Folded #1270: these emit a control-pane-compatible payload (corpus.*_debt +
    # mostly corpus.grade) and several explicitly say "folds into the control
    # pane" in their docstrings, but were never registered — an unfolded surface
    # can regress freely and never trip the ratchet. cuda_dev/bench_dx grade the
    # tree STATICALLY (no GPU needed) and degrade gracefully to grade A on a
    # GPU-less box. persona_fit is orthogonal to the wired persona_readiness
    # (matrix-integrity debt vs entry-path-gate debt — no persona double-count).
    {"key": "observability", "debt": "observability_debt", "script": "observability_scorecard.py", "label": "observability"},
    {"key": "learning", "debt": "learning_debt", "script": "learning_scorecard.py", "label": "learning"},
    {"key": "rsi_maturity", "debt": "rsi_debt", "script": "rsi_maturity_scorecard.py", "label": "rsi-maturity"},
    {"key": "tooling_quality", "debt": "py_debt", "script": "tooling_quality_scorecard.py", "label": "tooling-quality"},
    {"key": "bench_dx", "debt": "bench_dx_debt", "script": "bench_dx_scorecard.py", "label": "bench-dx"},
    {"key": "cuda_dev", "debt": "process_debt", "script": "cuda_dev_scorecard.py", "label": "cuda-dev"},
    {"key": "persona_fit", "debt": "persona_fit_debt", "script": "persona_fit_scorecard.py", "label": "persona-fit"},
]

# The scorecards folded via `go run ./cmd/fak …` (no python script). When one of
# THESE errors, the cause is almost always a working tree that does not COMPILE —
# the `go run` build step failed on uncommitted WIP — NOT a bug in the card. This
# is the B0 #1416 distinction the control pane must make legible: a build-break
# masquerading as "the scorecard is broken" sends the reader to debug the wrong
# thing, and an errored card silently drops out of the ratchet's fold.
GO_BACKED_KEYS: frozenset[str] = frozenset(
    c["key"] for c in SCORECARDS if "go run" in (c.get("cmd") or "")
)


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _git_line(args: list[str], root: Path) -> str:
    try:
        p = subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    if p.returncode != 0:
        return ""
    return p.stdout.strip()


def head_commit(root: Path) -> str:
    return _git_line(["rev-parse", "--short", "HEAD"], root) or "unknown"


# --- pure extraction / folding (the tested surface) ------------------------

def find_int(payload: Any, key: str) -> int | None:
    """First int value stored under ``key`` anywhere in the payload.

    The debt integer lives under ``corpus.<debt>`` for most scorecards and
    ``doc.<debt>`` for doc-appeal; a tolerant search keeps the fold from caring
    which nesting a given scorecard chose.
    """
    if isinstance(payload, dict):
        for nest in ("corpus", "doc"):
            sub = payload.get(nest)
            if isinstance(sub, dict) and isinstance(sub.get(key), bool) is False \
                    and isinstance(sub.get(key), int):
                return int(sub[key])
        val = payload.get(key)
        if isinstance(val, int) and not isinstance(val, bool):
            return int(val)
        for v in payload.values():
            got = find_int(v, key)
            if got is not None:
                return got
    elif isinstance(payload, list):
        for v in payload:
            got = find_int(v, key)
            if got is not None:
                return got
    return None


def find_grade(payload: Any) -> str | None:
    """The portfolio grade a scorecard reports at corpus/doc level, if any."""
    if isinstance(payload, dict):
        for nest in ("corpus", "doc"):
            sub = payload.get(nest)
            if isinstance(sub, dict) and isinstance(sub.get("grade"), str):
                return str(sub["grade"])
        if isinstance(payload.get("grade"), str):
            return str(payload["grade"])
    return None


def find_score(payload: Any, key: str) -> float | None:
    """The corpus/doc-level aggregate score stored under ``key``, if any.

    Scoped to the corpus/doc/top level ONLY (unlike :func:`find_int`'s deep walk)
    — per-item entries carry their own ``mean_score``-like fields, and a deep
    search could pick a page's score over the corpus aggregate. Used to derive the
    TRUE grade for the scorecards that emit a score but no corpus letter.
    """
    if not key or not isinstance(payload, dict):
        return None
    for nest in ("corpus", "doc"):
        sub = payload.get(nest)
        if isinstance(sub, dict) and isinstance(sub.get(key), (int, float)) \
                and not isinstance(sub.get(key), bool):
            return float(sub[key])
    val = payload.get(key)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def metric_from_payload(card: dict[str, str], payload: dict[str, Any] | None,
                        error: str = "") -> dict[str, Any]:
    debt_key = card["debt"]
    if error or not isinstance(payload, dict):
        return {
            "key": card["key"],
            "label": card["label"],
            "debt_key": debt_key,
            "debt": None,
            "grade": None,
            "score": None,
            "ok": False,
            "verdict": "ERROR",
            "error": error or "no payload",
        }
    debt = find_int(payload, debt_key)
    # Carry the corpus-level score for the scoreless-but-scored scorecards so the
    # severity lens can derive their TRUE grade instead of debt-magnitude.
    score_key = SCORE_KEYS.get(card["key"])
    return {
        "key": card["key"],
        "label": card["label"],
        "debt_key": debt_key,
        "debt": debt,
        "grade": find_grade(payload),
        "score": find_score(payload, score_key) if score_key else None,
        "ok": bool(payload.get("ok")),
        "verdict": str(payload.get("verdict") or ""),
        "error": "" if debt is not None else f"missing {debt_key} in payload",
    }


def build_break_hint(errored: list[dict[str, Any]]) -> str:
    """Guidance that distinguishes a working-tree BUILD BREAK from a real card bug.

    The B0 #1416 regression note, operationalized in the tool (not just the docs):
    the Go-backed cards shell ``go run ./cmd/fak …``, so uncommitted WIP that does
    not compile makes EVERY one of them error at once — a build break, not a card
    bug. Returns "" when no errored card is Go-backed (a python card erroring is a
    genuine card/measurement fault and needs no build-vs-bug triage)."""
    go_errs = [m["label"] for m in errored if m.get("key") in GO_BACKED_KEYS]
    if not go_errs:
        return ""
    return (
        f" — note: {len(go_errs)} Go-backed card(s) errored ({', '.join(sorted(go_errs))}); "
        "these shell `go run ./cmd/fak …`, so the usual cause is a working tree that "
        "does NOT compile, not a card bug. Triage with `go build ./...`: if it FAILS, "
        "commit or stash your WIP (or measure a pristine HEAD checkout that keeps .git, "
        "e.g. `git worktree add --detach <dir> HEAD`) and re-run; if `go build ./...` "
        "PASSES yet a card still errors, it is a real card bug — debug that card's "
        "--json. (clean-read recipe: .claude/skills/scorecard/SKILL.md)"
    )


def _compact_error(error: str, *, limit: int = 180) -> str:
    """One-line error preview for reason text.

    The full per-card error remains in ``metrics[].error``. The control-pane
    reason needs a compact root-cause preview so repeated Go build failures do
    not drown the operator in the same stderr copied once per card.
    """
    compact = " ".join(str(error or "unknown error").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def error_groups(errored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group errored cards by identical error text.

    This is the #2043 observability affordance: when one dirty-tree compile
    failure breaks N Go-backed cards, the control pane should say "one blocker
    hit N cards" instead of making the reader infer that from repeated rows.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for metric in errored:
        raw = str(metric.get("error") or "unknown error")
        group = grouped.setdefault(raw, {
            "error": raw,
            "summary": _compact_error(raw),
            "labels": [],
            "keys": [],
            "go_backed": False,
        })
        group["labels"].append(str(metric.get("label") or metric.get("key") or "unknown"))
        group["keys"].append(str(metric.get("key") or ""))
        group["go_backed"] = bool(group["go_backed"] or metric.get("key") in GO_BACKED_KEYS)
    out = []
    for group in grouped.values():
        labels = sorted(group["labels"])
        keys = sorted(k for k in group["keys"] if k)
        out.append({
            "count": len(labels),
            "labels": labels,
            "keys": keys,
            "go_backed": bool(group["go_backed"]),
            "summary": group["summary"],
            "error": group["error"],
        })
    return sorted(out, key=lambda g: (-int(g["count"]), str(g["summary"])))


def _error_group_reason(groups: list[dict[str, Any]]) -> str:
    if not groups:
        return ""
    previews = [
        f"{g['count']} card(s): {g['summary']}"
        for g in groups[:3]
    ]
    suffix = "" if len(groups) <= 3 else f"; and {len(groups) - 3} more"
    return "; unique blocker(s): " + "; ".join(previews) + suffix


def fold(metrics: list[dict[str, Any]], baseline: dict[str, Any] | None,
         *, workspace: str, commit: str) -> dict[str, Any]:
    """Fold per-scorecard metrics into one control-pane payload + trend."""
    measured = [m for m in metrics if isinstance(m.get("debt"), int)]
    errors = [m for m in metrics if not isinstance(m.get("debt"), int)]
    total_debt = sum(int(m["debt"]) for m in measured)
    # The scale-invariant companion to total_debt: every metric weighted by the
    # severity of its OWN grade, so the cross-family number isn't 91% two
    # occurrence-counters. Stamped onto each metric so the renderer/baseline see it.
    for m in measured:
        # eff_grade is the single three-tier truth (emitted > score-derived >
        # debt-derived); grade_weight is its severity. Stamp both so the renderer
        # and baseline never re-derive a letter that disagrees with the weight.
        m["eff_grade"] = display_grade(m)
        m["grade_weight"] = GRADE_DEBT[m["eff_grade"]]
    grade_debt = sum(int(m["grade_weight"]) for m in measured)

    trend = compute_trend(metrics, baseline, total_debt, grade_debt)

    by_debt = sorted(measured, key=lambda m: int(m["debt"]), reverse=True)
    breakdown = ", ".join(f"{m['label']} {m['debt']}" for m in by_debt) or "none"
    by_grade = sorted(measured, key=lambda m: int(m["grade_weight"]), reverse=True)
    grade_breakdown = ", ".join(
        f"{m['label']} {m['eff_grade']}({m['grade_weight']})"
        for m in by_grade if int(m["grade_weight"]) > 0) or "all A"

    regressed = trend["direction"] == "regressed"
    early_warning = trend.get("early_warning") or []
    ew_note = ""
    if early_warning and not regressed:
        # The hidden case the early-warning lens exists for (#712): a metric rose
        # but the portfolio held, so the ratchet stays green. Surface it advisory —
        # don't flip the verdict (the portfolio ratchet semantics are unchanged).
        ew_note = ("; EARLY-WARNING (advisory): "
                   + ", ".join(f"{e['label']} {e['from']}->{e['to']} (+{e['delta']})"
                               for e in early_warning)
                   + " rose vs baseline under a green portfolio — a hidden per-metric "
                     "regression; review before --pin re-floors it")
    grouped_errors = error_groups(errors)

    if errors:
        ok, verdict, finding = False, "ACTION", "scorecard_unmeasured"
        reason = (f"{len(errors)} scorecard(s) failed to report a debt integer "
                  f"({', '.join(m['label'] for m in errors)}); portfolio debt "
                  f"{total_debt} across {len(measured)} measured"
                  + _error_group_reason(grouped_errors))
        next_action = ("repair the failing scorecard(s) so the fold is complete; "
                       "start with the unique blocker(s) in error_groups; "
                       "re-run python tools/scorecard_control_pane.py"
                       + build_break_hint(errors))
    elif regressed:
        ok, verdict, finding = False, "ACTION", "scorecard_regressed"
        reason = (f"portfolio debt rose {trend['total_delta']:+d} to {total_debt} "
                  f"vs baseline @{trend['baseline_commit']} ({breakdown}); "
                  f"worsened: {', '.join(trend['worsened']) or 'see deltas'}")
        next_action = ("retire the regressed metric(s) worst-first with the owning "
                       "scorecard's skill, then re-pin: "
                       "python tools/scorecard_control_pane.py --pin")
    elif total_debt > 0:
        ok, verdict, finding = False, "ACTION", "scorecard_debt"
        reason = (f"portfolio debt {total_debt} across {len(measured)} scorecards "
                  f"({breakdown}); trend {trend['summary']}")
        next_action = ("retire debt worst-first (heaviest: "
                       f"{by_debt[0]['label']} {by_debt[0]['debt']}) with that "
                       "scorecard's skill; re-run to prove the portfolio drop")
    else:
        ok, verdict, finding = True, "OK", "all_clear"
        reason = (f"zero portfolio debt across {len(measured)} scorecards; "
                  f"trend {trend['summary']}")
        next_action = "hold the line; re-pin the baseline to lock the clean state"

    reason += ew_note
    if ew_note:
        # Point the operator at the offending metric(s) regardless of the verdict
        # ladder branch — the early-warning is the actionable signal here.
        next_action = ("review the per-metric early-warning ("
                       + ", ".join(e["label"] for e in early_warning)
                       + ") with that scorecard's skill BEFORE `--pin`, so a hidden "
                       "regression isn't blessed as the new floor; then: " + next_action)

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": workspace,
        "commit": commit,
        "total_debt": total_debt,
        "grade_debt": grade_debt,
        "grade_breakdown": grade_breakdown,
        "measured": len(measured),
        "errored": len(errors),
        "error_groups": grouped_errors,
        "early_warning": early_warning,
        "metrics": metrics,
        "trend": trend,
    }


def _base_int(baseline: dict[str, Any], key: str) -> int | None:
    val = baseline.get(key)
    return int(val) if isinstance(val, int) and not isinstance(val, bool) else None


def compute_trend(metrics: list[dict[str, Any]], baseline: dict[str, Any] | None,
                  total_debt: int, grade_debt: int = 0) -> dict[str, Any]:
    """Per-metric + portfolio delta vs a pinned baseline.

    Tracks two portfolio axes against the pin: ``total_debt`` (the raw-unit sum,
    the ratchet's gate) and ``grade_debt`` (the scale-invariant severity sum). A
    severity regression that a raw-count improvement would mask shows up as a
    positive ``grade_delta`` even when ``total_delta`` is flat-or-down.
    """
    base_metrics = {}
    base_grade_weights = {}
    base_commit = ""
    base_total = None
    base_grade = None
    if isinstance(baseline, dict):
        base_metrics = baseline.get("metrics") or {}
        base_grade_weights = baseline.get("grade_weights") or {}
        base_commit = str(baseline.get("commit") or "")
        base_total = _base_int(baseline, "total_debt")
        base_grade = _base_int(baseline, "grade_debt")

    if not base_metrics or base_total is None:
        return {
            "direction": "unpinned",
            "summary": "unpinned (no baseline; run --pin)",
            "total_delta": 0,
            "grade_delta": 0,
            "baseline_commit": base_commit,
            "baseline_total": base_total,
            "baseline_grade": base_grade,
            "grade_debt": grade_debt,
            "deltas": {},
            "worsened": [],
            "improved": [],
            "early_warning": [],
            "grade_regressed": [],
        }

    deltas: dict[str, int] = {}
    worsened: list[str] = []
    improved: list[str] = []
    # The per-metric GRADE-regression lens (#1423): every metric whose letter grade
    # dropped vs its pinned grade (A->B etc), independent of its raw debt. This is
    # the culprit list the grade ratchet reds on — a severity slip a flat raw total
    # would hide. Needs the baseline to carry per-metric grade_weights (added by
    # baseline_doc); when the pin predates that field this stays empty (advisory).
    grade_regressed: list[dict[str, Any]] = []
    # The per-metric early-warning lens (#712): EVERY metric whose debt rose vs its
    # pinned value, independent of where the portfolio total landed. The portfolio
    # ratchet only trips when the SUM regresses, so a single metric's rise can hide
    # under another's improvement (seo 6->8 while the portfolio fell 44->40). This
    # list surfaces that first downward move WITHIN a healthy envelope — before a
    # blind --pin blesses it as the new floor.
    early_warning: list[dict[str, Any]] = []
    for m in metrics:
        if not isinstance(m.get("debt"), int):
            continue
        prior = base_metrics.get(m["key"])
        if not isinstance(prior, int) or isinstance(prior, bool):
            continue
        delta = int(m["debt"]) - int(prior)
        deltas[m["key"]] = delta
        if delta > 0:
            worsened.append(m["label"])
            early_warning.append({"key": m["key"], "label": m["label"],
                                  "delta": delta, "from": int(prior), "to": int(m["debt"])})
        elif delta < 0:
            improved.append(m["label"])
        # GRADE regression (severity), tracked alongside the raw-debt delta. A
        # metric can hold its raw count yet drop a letter; the grade ratchet reds
        # on THAT. eff_grade/grade_weight are stamped onto m by fold() before
        # compute_trend runs.
        prior_w = base_grade_weights.get(m["key"])
        cur_w = m.get("grade_weight")
        if isinstance(prior_w, int) and not isinstance(prior_w, bool) \
                and isinstance(cur_w, int) and cur_w > prior_w:
            grade_regressed.append({
                "key": m["key"], "label": m["label"],
                "from_weight": int(prior_w), "to_weight": int(cur_w),
                "to_grade": m.get("eff_grade", "?"),
            })

    total_delta = total_debt - base_total
    grade_delta = grade_debt - base_grade if base_grade is not None else 0
    if total_delta > 0:
        direction = "regressed"
    elif total_delta < 0:
        direction = "improved"
    else:
        direction = "flat"
    summary = (f"{direction} {total_delta:+d} vs @{base_commit or 'baseline'} "
               f"(was {base_total}, now {total_debt})")
    if base_grade is not None and grade_delta != 0:
        summary += f"; grade-debt {base_grade}->{grade_debt} ({grade_delta:+d})"
    return {
        "direction": direction,
        "summary": summary,
        "total_delta": total_delta,
        "grade_delta": grade_delta,
        "baseline_commit": base_commit,
        "baseline_total": base_total,
        "baseline_grade": base_grade,
        "grade_debt": grade_debt,
        "deltas": deltas,
        "worsened": worsened,
        "improved": improved,
        "early_warning": early_warning,
        "grade_regressed": grade_regressed,
    }


def baseline_doc(payload: dict[str, Any]) -> dict[str, Any]:
    """The baseline file body to pin from a folded control-pane payload."""
    metrics = {
        m["key"]: int(m["debt"])
        for m in payload.get("metrics", [])
        if isinstance(m.get("debt"), int)
    }
    # Per-metric grade SEVERITY weight (A=0..F=8), pinned alongside the raw count so
    # the grade ratchet (#1423) can name exactly which scorecard slipped a letter —
    # an A->B regression that leaves the raw debt flat. fold() stamps grade_weight
    # onto each measured metric before this runs.
    grade_weights = {
        m["key"]: int(m["grade_weight"])
        for m in payload.get("metrics", [])
        if isinstance(m.get("debt"), int) and isinstance(m.get("grade_weight"), int)
    }
    return {
        "schema": BASELINE_SCHEMA,
        "commit": payload.get("commit", ""),
        "total_debt": payload.get("total_debt", 0),
        "grade_debt": payload.get("grade_debt", 0),
        "metrics": metrics,
        "grade_weights": grade_weights,
        "_doc": ("Pinned per-metric scorecard-debt baseline for the unified "
                 "control pane. total_debt is the raw-unit ratchet gate; grade_debt "
                 "is the scale-invariant severity companion, and grade_weights pins "
                 "each metric's letter-grade severity so an A->B slip reds the gate "
                 "even at flat raw debt (#1423). Re-pin after a debt "
                 "drop to ratchet the trend down: "
                 "python tools/scorecard_control_pane.py --pin"),
    }


# --- live runner -----------------------------------------------------------

def run_scorecard(root: Path, card: dict[str, str] | str, *, python: str, timeout: int) -> tuple[dict[str, Any] | None, str]:
    if isinstance(card, dict) and card.get("cmd"):
        argv = shlex.split(card["cmd"])
    else:
        script = card["script"] if isinstance(card, dict) else card
        script_path = root / "tools" / script
        if not script_path.exists():
            return None, f"missing scorecard: tools/{script}"
        argv = [python, str(script_path), "--json", *shlex.split(card.get("args", "") if isinstance(card, dict) else "")]
    try:
        proc = subprocess.run(
            argv,
            cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    try:
        return json.loads(proc.stdout), ""
    except ValueError:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
        return None, f"non-JSON output (exit {proc.returncode}): {tail[0][:160]}"


def collect(root: Path, *, python: str = "", timeout: int = 120) -> list[dict[str, Any]]:
    python = python or sys.executable
    metrics: list[dict[str, Any]] = []
    for card in SCORECARDS:
        payload, error = run_scorecard(root, card, python=python, timeout=timeout)
        metrics.append(metric_from_payload(card, payload, error))
    return metrics


def load_baseline(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def render(payload: dict[str, Any]) -> str:
    lines = [
        f"scorecard control pane — {payload['verdict']} ({payload['finding']})",
        f"  portfolio debt: {payload['total_debt']} (raw units)  "
        f"grade-debt: {payload.get('grade_debt', 0)} (severity, scale-invariant)  "
        f"({payload['measured']} measured, {payload['errored']} errored)  @{payload['commit']}",
        f"  grade severity: {payload.get('grade_breakdown', 'n/a')}",
        f"  trend: {payload['trend']['summary']}",
        "",
    ]
    for m in payload["metrics"]:
        debt = m["debt"] if m["debt"] is not None else f"ERR ({m['error']})"
        grade = f" [{m['grade']}]" if m.get("grade") else ""
        lines.append(f"  {m['label']:<16} {m['debt_key']:<16} {debt}{grade}")
    early_warning = payload.get("early_warning") or []
    if early_warning:
        lines.append("")
        for e in early_warning:
            lines.append(f"  WARN early-warning: {e['label']} rose {e['from']}->{e['to']} "
                         f"(+{e['delta']}) vs baseline — hidden under a green portfolio")
    grade_regressed = (payload.get("trend") or {}).get("grade_regressed") or []
    if grade_regressed:
        lines.append("")
        for g in grade_regressed:
            lines.append(f"  GRADE REGRESSION: {g['label']} slipped to {g['to_grade']} "
                         f"vs pinned grade — reds the grade ratchet (#1423)")
    lines.extend(["", f"  → {payload['next_action']}"])
    return "\n".join(lines)


def check_gate(payload: dict[str, Any]) -> tuple[int, str]:
    """The CI ratchet decision over a folded payload (pure: exit code + message).

    The default exit code is green only at ZERO portfolio debt — too strict to
    gate a repo that still carries real debt. This is the ratchet contract the
    repo-3x epic (#506) wants instead: debt may hold or fall, never rise.

      0  flat / improved   — the ratchet held (green even with nonzero debt)
      1  regressed         — debt rose above the pinned baseline (raw OR grade),
                             or a scorecard went unmeasured
      2  unpinned          — no baseline to ratchet against; run --pin first

    The GRADE ratchet (#1423, the "stays Excellent" half of #1414) is HARD by
    default: a per-metric letter-grade slip (A->B) or a portfolio grade_debt rise
    reds the gate EVEN WHEN the raw-unit total held — the regression a flat
    total_debt would otherwise hide. Mirror of the milestone climb ratchet (#1442).
    Set FAK_SCORECARD_GRADE_RATCHET=0 to demote it to advisory (the pre-#1423 read).
    """
    if int(payload.get("errored", 0)) > 0:
        errored = [m for m in payload.get("metrics", [])
                   if not isinstance(m.get("debt"), int)]
        return 1, (f"RATCHET FAIL: {payload['errored']} scorecard(s) unmeasured — "
                   f"{payload['reason']}" + build_break_hint(errored))
    trend = payload.get("trend") or {}
    direction = trend.get("direction")
    if direction == "unpinned":
        return 2, ("RATCHET UNPINNED: no baseline to ratchet against; run "
                   "`python tools/scorecard_control_pane.py --pin` to set one")
    if direction == "regressed":
        return 1, f"RATCHET FAIL: {trend['summary']}; worsened: {', '.join(trend['worsened']) or 'see deltas'}"

    # The grade ratchet (#1423): a SEVERITY regression the raw-unit sum can mask —
    # a metric dropping a letter (A->B) at flat raw debt, or the portfolio
    # grade_debt rising. HARD by default (reds the gate), env-overridable to
    # advisory. Named per-metric so the failure says exactly which surface slipped.
    grade_delta = int(trend.get("grade_delta") or 0) if isinstance(trend, dict) else 0
    grade_regressed = (trend.get("grade_regressed") or []) if isinstance(trend, dict) else []
    if (grade_regressed or grade_delta > 0) and grade_ratchet_hard():
        if grade_regressed:
            who = ", ".join(f"{g['label']} {_weight_letter(g['from_weight'])}->{g['to_grade']}"
                            for g in grade_regressed)
        else:
            who = payload.get("grade_breakdown", "see grade severity")
        return 1, (f"GRADE-RATCHET FAIL: grade-debt rose {grade_delta:+d} to "
                   f"{payload.get('grade_debt')} vs baseline @{trend.get('baseline_commit')} "
                   f"— a scorecard slipped a letter the raw-unit total held flat: {who}. "
                   f"Retire it with the owning scorecard's skill, then re-pin "
                   f"(`--pin`); or set {GRADE_RATCHET_ENV}=0 to demote this gate to "
                   f"advisory for a deliberate one-off pin.")

    msg = f"RATCHET OK: {trend['summary']} (debt {payload['total_debt']} held at-or-below baseline)"
    # The early-warning lens (#712): the portfolio ratchet held (exit 0), but a
    # per-metric rise is hiding under it — surface it ADVISORY without tripping the
    # gate, so it's seen before a re-pin re-floors it as the new baseline.
    early_warning = (trend.get("early_warning") or []) if isinstance(trend, dict) else []
    if early_warning:
        msg += ("; EARLY-WARNING (advisory, gate still green): "
                + ", ".join(f"{e['label']} +{e['delta']}" for e in early_warning)
                + " rose vs baseline — a hidden per-metric regression; review before --pin")
    # When the grade ratchet is DEMOTED to advisory (FAK_SCORECARD_GRADE_RATCHET=0),
    # a severity regression that survived the hard gate above still surfaces here as
    # a non-blocking warning — the pre-#1423 behavior, preserved under the knob.
    if grade_delta > 0 and not grade_ratchet_hard():
        msg += (f"; GRADE-DEBT WARN (advisory — ratchet demoted via "
                f"{GRADE_RATCHET_ENV}=0): severity rose {grade_delta:+d} to "
                f"{payload.get('grade_debt')} vs baseline "
                f"({payload.get('grade_breakdown')}) — review before --pin")
    return 0, msg


def _weight_letter(weight: int) -> str:
    """The letter grade a severity weight came from (inverse of GRADE_DEBT), for
    rendering a from->to grade slip in the ratchet failure message."""
    for letter, w in GRADE_DEBT.items():
        if w == weight:
            return letter
    return "?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Unified scorecard debt control-pane (read-only unless --pin).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--pin", action="store_true",
                    help=f"pin the current debt as the baseline ({BASELINE_REL})")
    ap.add_argument("--check", action="store_true",
                    help="CI ratchet gate: exit non-zero only if debt regressed above baseline (#506)")
    ap.add_argument("--baseline", default="", help=f"baseline JSON path (default: {BASELINE_REL})")
    ap.add_argument("--timeout", type=int, default=120, help="per-scorecard timeout seconds")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    baseline_path = Path(args.baseline).resolve() if args.baseline else (root / BASELINE_REL)

    metrics = collect(root, timeout=args.timeout)
    baseline = load_baseline(baseline_path)
    payload = fold(metrics, baseline, workspace=str(root), commit=head_commit(root))

    if args.pin:
        doc = baseline_doc(payload)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        if not args.json:
            print(f"pinned baseline @{doc['commit']} total_debt={doc['total_debt']} -> {baseline_path}")

    if args.check:
        code, message = check_gate(payload)
        if args.json:
            # Under --check the tool's contract IS the ratchet, not the raw fold:
            # ok/verdict reflect "did the portfolio hold at-or-below baseline?"
            # (green even with residual debt), not "is debt zero?". This is what a
            # loop runner reads to fold the pane — keep gate_exit/gate_message for
            # the literal exit code. #509.
            gated = {
                **payload,
                "ok": code == 0,
                "verdict": "OK" if code == 0 else "ACTION",
                "gate_exit": code,
                "gate_message": message,
            }
            print(json.dumps(gated, indent=2))
        else:
            print(message)
        return code

    if args.json:
        print(json.dumps(payload, indent=2))
    elif not args.pin:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
