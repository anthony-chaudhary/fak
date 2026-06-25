#!/usr/bin/env python3
"""Stability scorecard — the measuring stick for "can we tell when we broke something,
and get back to a known-good state?".

The sibling scorecards each grade a surface a reviewer cares about: ``code_quality``
grades the Go module, ``agent_readiness`` grades how easily an agent adopts fak,
``rsi_maturity`` grades the self-improver's *structure*. None of them answer the
question a team living on a rapidly-iterating trunk actually loses sleep over: as we
add items fast, **how do we KNOW a regression / tail-wag / confusion happened, and how
do we REVERT to a stable version?** That used to be a vibe ("we have lots of tests").
This is the number.

It scores the git-tracked tree on four groups — the four ways a system stays trustworthy
while it changes fast — folds them into a weighted score and an A-F grade, and counts
**stability-debt**: the total of concrete, re-derivable HARD defects that leave fak unable
to catch a regression or roll one back. Each is a defect you fix by *adding the real
sentinel / invariant / revert affordance*, never by relaxing a check.

  SENTINEL    — when a regression lands, SOMETHING fails (we find out)
    regression_gates_wired       the roster of HARD CI regression gates is wired (build/
                                 vet/test, gofmt, -race, claims-lint, the portfolio
                                 ratchet, the main-KPI track gate, dos-review, the
                                 no-blackhole tool-test runner, index-sync, leak-scan)
    ratchet_baselines_committed  every ratchet has a COMMITTED baseline to compare against
                                 (a ratchet with no pinned floor can't detect a regression)
    honesty_ledger_clean         CLAIMS.md exists and every claim carries exactly one tag —
                                 a drifted ledger is a silent over-claim regression

  INVARIANT   — the core assumptions are ENCODED as executable tests (not prose)
    invariant_tests_present      the load-bearing invariants have a named test (ABI freeze,
                                 tier/import DAG, interpreter-free + exec-free request path)
    frozen_pins_present          the golden / baseline pins those tests compare against exist
                                 on disk (the ABI golden, the RSI main-KPI baseline, …)
    fail_closed_witnessed        an executable test proves the system fails CLOSED on a bad
                                 input (malformed call denied, missing field denied)
    determinism_witness          the critical math/codecs carry a determinism / metamorphic
                                 witness (the proofs_witness_test.go convention) so a KEEP
                                 can't be a one-box fluke

  REVERT      — we can get back to a known-good state (revert / rollback / stable version)
    keep_revert_ladder           the committed mechanism that reverts a non-improving change
                                 to the known-good baseline (internal/shipgate's keep/revert
                                 ladder), plus a witness test
    version_pin                  a single VERSION marker + a resolver (internal/appversion)
                                 so "what version is this" has one answer
    release_tagging_gated        release tagging is CI-gated (release-cadence waits for green
                                 before it tags a stable version)
    rollback_runbook             a DOCUMENTED, linked operator runbook for reverting to a
                                 stable version — the downgrade path, the state-restore seam,
                                 the baseline re-pin, the stable-version pin

  DRIFT       — a small thing silently distorting a big thing gets caught (tail-wag / confusion)
    drift_detectors_wired        the roster of silent-drift detectors exists (readme freshness,
                                 index sync, commit-stamp coverage, claims salience, the
                                 portfolio trend)
    confusion_escalation_signal  SOFT: is there an explicit "I could not decide → escalate"
                                 (INDETERMINATE) disposition, or is confusion handled only by
                                 fail-open/fail-closed? (the verification-ladder frontier)

The headline metric is **stability-debt**: the count of concrete HARD defects above.
Driving it to zero means a trunk that, however fast it changes, fails loudly on a
regression and has a written path back to a stable version. The companion process — the
``/stability-score`` skill — runs this, retires the worst-first defect by ADDING the
real affordance, and re-runs to prove the drop. It folds into the unified
``scorecard_control_pane`` alongside the other inward sticks.

Deterministic + read-only by construction: it reads the git-tracked tree (so two clones
of the same commit score identically) and edits nothing. Run from the repo ROOT::

    python tools/stability_scorecard.py                 # human scorecard
    python tools/stability_scorecard.py --json          # machine payload (control-pane shape)
    python tools/stability_scorecard.py --markdown      # the committed snapshot body
    python tools/stability_scorecard.py --compare base.json   # prove the stability-debt moved
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-stability-scorecard/1"
GENERATED_SNAPSHOT = "docs/STABILITY-SCORECARD.md"

# ---------------------------------------------------------------------------
# The contract a stable, fast-moving trunk expects. Each constant is a named,
# load-bearing affordance — never a hand-picked file list where a rule would do.
# A fork that drops one of these scores lower, by construction. Every token /
# path / symbol below was confirmed present in the committed tree at authoring
# time, so a nonzero defect means a real removal, not a typo.
# ---------------------------------------------------------------------------

# SENTINEL — the HARD CI regression gates. A gate is "wired" if ANY of its tokens
# appears in the concatenated CI text (ci.yml + every workflow + Makefile + ci.ps1).
# These are the things that turn RED when a regression lands on the trunk.
REQUIRED_REGRESSION_GATES: list[tuple[str, list[str]]] = [
    ("build + vet + test", ["go test ./...", "go build ./..."]),
    ("gofmt format ratchet", ["gofmt -l"]),
    ("race detector", ["go test -race", "-race -count"]),
    ("claims honesty ratchet", ["claims-lint", "claims_salience_register"]),
    ("portfolio debt ratchet", ["scorecard_control_pane.py --check", "scorecard_control_pane"]),
    ("main-KPI regression gate", ["-mode track", "rsiloop -mode"]),
    ("commit-witness (dos-review)", ["dos review", "dos-review"]),
    ("no-blackhole tool tests", ["gated_tool_tests"]),
    ("index/llms drift", ["check_index_sync", "index-sync"]),
    ("public-leak scan", ["scrub_public_copy", "leak-scan"]),
]

# SENTINEL — every ratchet needs a COMMITTED baseline to ratchet against. A ratchet
# whose baseline file is missing silently can't detect a regression. (path, what).
RATCHET_BASELINES: list[tuple[str, str]] = [
    ("tools/scorecard_baseline.json", "portfolio scorecard-debt baseline"),
    ("internal/rsiloop/testdata/main-kpi-baseline.jsonl", "RSI main-KPI regression floor"),
    ("internal/abi/testdata/abi_v0.1.golden", "frozen ABI wire-contract golden"),
]

# INVARIANT — the load-bearing assumptions, each as a named test symbol that must
# appear in a tracked *_test.go. If the assumption breaks, the test goes red.
# (All confirmed in the COMMITTED tree — internal/architest + internal/abi.)
REQUIRED_INVARIANT_TESTS: list[tuple[str, str]] = [
    ("ABI wire-contract freeze", "TestABIGoldenFreeze"),
    ("closed refusal vocabulary", "TestClosedReasonVocabulary"),
    ("verdict restrictiveness lattice", "TestFoldRankOrdering"),
    ("every package declares a tier", "TestEveryPackageDeclaresTier"),
    ("no upward (layer-violating) imports", "TestNoUpwardImports"),
    ("request path is interpreter-free", "TestRequestPathInterpreterFree"),
    ("hot path has no os/exec", "TestHotPathHasNoExec"),
]

# INVARIANT — fail-closed witnesses: a bad input must be REFUSED, not waved through.
# (Committed tests in internal/preflight + cmd/fak — the deny-by-structure floor.)
REQUIRED_FAIL_CLOSED: list[tuple[str, str]] = [
    ("malformed input denied (preflight rung 0)", "TestRung0MalformedJSONDenied"),
    ("missing required field denied (preflight rung 1)", "TestRung1MissingRequiredFieldDenied"),
    ("default policy denies danger (guard)", "TestGuardDefaultPolicyDeniesDangerAllowsBenign"),
]

# INVARIANT — determinism witnesses follow the proofs_witness_test.go convention.
# A healthy kernel carries many; zero means the determinism floor was deleted.
DETERMINISM_WITNESS_GLOB = "proofs_witness_test.go"
DETERMINISM_WITNESS_FLOOR = 5  # below this many is a SOFT nudge, zero is a HARD defect.

# REVERT — the committed keep/revert ladder (internal/shipgate): the mechanism that
# reverts a non-improving change back to the known-good baseline. (file, [tokens]).
KEEP_REVERT_LADDER_FILE = "internal/shipgate/shipgate.go"
KEEP_REVERT_LADDER_TOKENS = ["func Evaluate", "REVERT"]
# REVERT — the portable state/session capture+restore seam. Present on the trunk
# (internal/snapshot + internal/sessionimage, fronted by `fak snapshot`); its ABSENCE
# from a tree (e.g. a fork that dropped it) is a SOFT note, never HARD debt.
STATE_SEAM_DIRS = ["internal/snapshot/", "internal/sessionimage/"]

# REVERT — the single version marker + its resolver.
VERSION_FILE = "VERSION"
APPVERSION_PKG = "internal/appversion/appversion.go"

# REVERT — release tagging is CI-gated (waits for green before it tags a stable version).
RELEASE_CADENCE_WF = ".github/workflows/release-cadence.yml"

# REVERT — the operator rollback runbook. Present if ANY candidate path exists; its
# CONTENT must cover the four real revert mechanisms (each token list is a REAL
# identifier — a git command, a tool path, a package, an env var — so an empty file
# token-stuffed with English phrases cannot pass); it must carry runnable commands
# (>= MIN_RUNBOOK_FENCES fenced blocks); and it must be discoverable (linked from an
# entry point). This is the gap a presence check is blind to between "we have
# snapshots" and "an operator can actually roll back".
ROLLBACK_RUNBOOK_CANDIDATES = [
    "docs/ROLLBACK.md", "docs/rollback.md", "docs/stability-rollback.md",
    "docs/runbooks/rollback.md", "ROLLBACK.md",
]
# Each (label, [tokens]); covered if ANY token appears. Tokens are REAL anchors — a
# git command, a `fak` verb / package, a tool path, an env var — not English phrases,
# so a section can't be satisfied by prose-stuffing the way a soft synonym would allow.
ROLLBACK_RUNBOOK_SECTIONS: list[tuple[str, list[str]]] = [
    ("version downgrade path", ["git checkout v", "git revert", "checkout the tag"]),
    ("state / snapshot restore", ["fak snapshot", "restore-fleet", "internal/snapshot",
                                  "sessionimage", "restorefleet"]),
    ("baseline re-pin / ratchet revert", ["scorecard_control_pane", "scorecard_baseline",
                                          "main-kpi-baseline"]),
    ("stable-version pin (no auto-upgrade)", ["fak_app_version"]),
]
ROLLBACK_RUNBOOK_LINKERS = ["AGENTS.md", "CONTRIBUTING.md", "README.md"]
MIN_RUNBOOK_FENCES = 2  # a runbook with no runnable commands is documentation theater.

# DRIFT — the silent-drift detectors. HARD if the tool file is MISSING; a present
# tool that isn't referenced in CI is a SOFT nudge (some run skill-driven, by design).
REQUIRED_DRIFT_DETECTORS: list[tuple[str, str]] = [
    ("readme freshness", "tools/readme_freshness_audit.py"),
    ("index / llms sync", "tools/check_index_sync.py"),
    ("commit-stamp coverage", "tools/commit_stamp_doctor.py"),
    ("claims salience (no-loss)", "tools/claims_salience_register.py"),
    ("portfolio debt trend", "tools/scorecard_control_pane.py"),
]
# DRIFT — a deterministic tail-wag finder backing the manual /tail-wag skill (frontier).
TAIL_WAG_TOOL_GLOB = "tail_wag"
# DRIFT — the "I can't decide → escalate" disposition. Scoped to the adjudication core.
CONFUSION_SIGNAL_FILES = ["internal/abi", "internal/adjudicator"]
CONFUSION_SIGNAL_TOKENS = ["indeterminate", "verdictindeterminate"]
VERIFICATION_LADDER_DOC = "docs/notes/verification-ladder-doctrine.md"

GROUPS = ("sentinel", "invariant", "revert", "drift")
KPI_GROUP: dict[str, str] = {
    "regression_gates_wired": "sentinel",
    "ratchet_baselines_committed": "sentinel",
    "honesty_ledger_clean": "sentinel",
    "invariant_tests_present": "invariant",
    "frozen_pins_present": "invariant",
    "fail_closed_witnessed": "invariant",
    "determinism_witness": "invariant",
    "keep_revert_ladder": "revert",
    "version_pin": "revert",
    "release_tagging_gated": "revert",
    "rollback_runbook": "revert",
    "drift_detectors_wired": "drift",
    "confusion_escalation_signal": "drift",
}
# Thirteen KPIs across the four ways a fast-moving trunk stays trustworthy. The sum is
# exactly 1.0 (a regression test asserts both the sum and that the weight set == the
# KPI set). confusion_escalation_signal carries weight but emits no HARD debt (it is a
# frontier judgment, not a work-list item).
KPI_WEIGHTS: dict[str, float] = {
    # sentinel (0.30) — do we find out?
    "regression_gates_wired": 0.14,
    "ratchet_baselines_committed": 0.08,
    "honesty_ledger_clean": 0.08,
    # invariant (0.30) — are the assumptions encoded?
    "invariant_tests_present": 0.12,
    "frozen_pins_present": 0.06,
    "fail_closed_witnessed": 0.06,
    "determinism_witness": 0.06,
    # revert (0.28) — can we get back?
    "keep_revert_ladder": 0.07,
    "version_pin": 0.05,
    "release_tagging_gated": 0.04,
    "rollback_runbook": 0.12,
    # drift (0.12) — does a small thing wagging a big thing get caught?
    "drift_detectors_wired": 0.08,
    "confusion_escalation_signal": 0.04,
}

CLAIMS_FILE = "CLAIMS.md"
CLAIM_TAGS = ("[SHIPPED]", "[SIMULATED]", "[STUB]")
CLAIM_LINE = re.compile(r"^\s*- \[")
_FENCE_RE = re.compile(r"^(```|~~~)")


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


def _has(text: str | None, *tokens: str) -> bool:
    """True if the text (case-insensitive) contains any of the tokens."""
    if not text:
        return False
    low = text.lower()
    return any(t.lower() in low for t in tokens)


def _count_fences(text: str | None) -> int:
    """Number of fenced code blocks in a doc — the count of runnable-command blocks.
    A rollback runbook with no fenced commands is documentation theater."""
    if not text:
        return 0
    opens = sum(1 for line in text.split("\n") if _FENCE_RE.match(line.strip()))
    return opens // 2


def untagged_claims(claims_text: str | None) -> list[str]:
    """Claim lines (`- [ …`) that do NOT carry exactly one status tag — the
    claims-lint rule, as the measure of a non-drifting honesty ledger."""
    if not claims_text:
        return []
    bad: list[str] = []
    for i, line in enumerate(claims_text.splitlines(), 1):
        if not CLAIM_LINE.match(line):
            continue
        n = sum(line.count(tag) for tag in CLAIM_TAGS)
        if n != 1:
            bad.append(f"CLAIMS.md:{i}: {n} status tag(s) (need exactly 1): {line.strip()[:80]}")
    return bad


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each returns
#   {kpi, group, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of stability-debt; soft = score-only judgment nudges.
# ---------------------------------------------------------------------------

def kpi_regression_gates_wired(missing: list[str]) -> dict[str, Any]:
    """The HARD CI regression gates that turn RED when a regression lands. Each
    gate not found anywhere in the CI text is one unit — a class of regression
    that could reach the trunk with nothing failing."""
    defects = [f"no CI regression gate found for: {label} — wire it HARD so a "
               "regression in this surface fails the build" for label in missing]
    covered = len(REQUIRED_REGRESSION_GATES) - len(missing)
    return {"kpi": "regression_gates_wired", "group": "sentinel",
            "score": _clamp(100 * covered / max(1, len(REQUIRED_REGRESSION_GATES))),
            "detail": f"{covered}/{len(REQUIRED_REGRESSION_GATES)} HARD regression gates wired in CI",
            "defects": defects, "soft": []}


def kpi_ratchet_baselines_committed(missing: list[str]) -> dict[str, Any]:
    """Every ratchet needs a committed floor to compare against. A missing baseline
    file means that ratchet silently can't detect a regression. One unit each."""
    defects = [f"missing committed baseline {path} — the ratchet has no floor to detect a "
               "regression against" for path in missing]
    covered = len(RATCHET_BASELINES) - len(missing)
    return {"kpi": "ratchet_baselines_committed", "group": "sentinel",
            "score": _clamp(100 * covered / max(1, len(RATCHET_BASELINES))),
            "detail": f"{covered}/{len(RATCHET_BASELINES)} ratchet baselines committed",
            "defects": defects, "soft": []}


def kpi_honesty_ledger_clean(present: bool, untagged: list[str]) -> dict[str, Any]:
    """CLAIMS.md with every claim tagged is the ledger a silent over-claim can't drift
    past. A missing ledger is hard; each untagged claim is one unit (capped)."""
    defects: list[str] = []
    if not present:
        defects.append(f"missing {CLAIMS_FILE} — the honesty ledger that catches a silent over-claim")
    else:
        defects.extend(untagged[:8])
    soft = ([f"... and {len(untagged) - 8} more untagged claim line(s)"]
            if present and len(untagged) > 8 else [])
    return {"kpi": "honesty_ledger_clean", "group": "sentinel",
            "score": _clamp((0 if not present else 100) - 12 * len([d for d in defects if present])),
            "detail": (f"{CLAIMS_FILE} present, {len(untagged)} untagged claim(s)" if present
                       else f"no {CLAIMS_FILE}"),
            "defects": defects, "soft": soft}


def kpi_invariant_tests_present(missing: list[str]) -> dict[str, Any]:
    """The load-bearing assumptions, each as a named executable test. A missing one
    means that assumption can silently break with nothing turning red. One unit each."""
    defects = [f"no executable test for the invariant: {label} ({sym}) — encode it so a "
               "violation fails the build" for label, sym in
               ((lbl, s) for lbl, s in REQUIRED_INVARIANT_TESTS if s in missing)]
    covered = len(REQUIRED_INVARIANT_TESTS) - len(missing)
    return {"kpi": "invariant_tests_present", "group": "invariant",
            "score": _clamp(100 * covered / max(1, len(REQUIRED_INVARIANT_TESTS))),
            "detail": f"{covered}/{len(REQUIRED_INVARIANT_TESTS)} core invariants have a named test",
            "defects": defects, "soft": []}


def kpi_frozen_pins_present(missing: list[str]) -> dict[str, Any]:
    """The golden / baseline pins the invariant + ratchet tests compare against. A
    golden test whose golden file is gone silently can't freeze anything."""
    defects = [f"missing frozen pin {path} ({what}) — the test that reads it can't catch a drift"
               for path, what in ((p, w) for p, w in RATCHET_BASELINES if p in missing)]
    covered = len(RATCHET_BASELINES) - len(missing)
    return {"kpi": "frozen_pins_present", "group": "invariant",
            "score": _clamp(100 * covered / max(1, len(RATCHET_BASELINES))),
            "detail": f"{covered}/{len(RATCHET_BASELINES)} frozen pins on disk",
            "defects": defects, "soft": []}


def kpi_fail_closed_witnessed(missing: list[str]) -> dict[str, Any]:
    """A bad input must be REFUSED, not waved through. Each missing fail-closed
    witness is a class of bad state that could deserialize/admit silently."""
    defects = [f"no fail-closed test for: {label} ({sym}) — prove a bad input is REFUSED"
               for label, sym in ((lbl, s) for lbl, s in REQUIRED_FAIL_CLOSED if s in missing)]
    covered = len(REQUIRED_FAIL_CLOSED) - len(missing)
    return {"kpi": "fail_closed_witnessed", "group": "invariant",
            "score": _clamp(100 * covered / max(1, len(REQUIRED_FAIL_CLOSED))),
            "detail": f"{covered}/{len(REQUIRED_FAIL_CLOSED)} fail-closed behaviours witnessed",
            "defects": defects, "soft": []}


def kpi_determinism_witness(count: int) -> dict[str, Any]:
    """The critical math/codecs carry a determinism / metamorphic witness (the
    proofs_witness_test.go convention) so a 'pass' can't be a one-box fluke. Zero
    witnesses is a HARD defect (the floor was deleted); a thin count is a SOFT nudge."""
    defects: list[str] = []
    soft: list[str] = []
    if count == 0:
        defects.append("no determinism / metamorphic witness anywhere "
                       f"(no {DETERMINISM_WITNESS_GLOB}) — a 'pass' could be a one-box fluke")
        score = 0
    elif count < DETERMINISM_WITNESS_FLOOR:
        soft.append(f"only {count} determinism witness file(s) (<{DETERMINISM_WITNESS_FLOOR}) — "
                    "widen the proofs_witness coverage of critical paths")
        score = _clamp(60 + 8 * count)
    else:
        score = 100
    return {"kpi": "determinism_witness", "group": "invariant", "score": score,
            "detail": f"{count} proofs_witness determinism witness file(s)",
            "defects": defects, "soft": soft}


def kpi_keep_revert_ladder(ladder_ok: bool, ladder_tested: bool,
                           state_seam_present: bool) -> dict[str, Any]:
    """The committed mechanism that reverts a non-improving change back to the
    known-good baseline (internal/shipgate's keep/revert ladder). A missing ladder or
    a ladder with no committed witness test is one unit each. The portable
    state/session capture+restore seam being absent from a tree (e.g. a fork that
    dropped internal/snapshot) is a SOFT note — state rollback then rides git-tag +
    this ladder."""
    defects: list[str] = []
    if not ladder_ok:
        defects.append(f"no committed keep/revert ladder ({KEEP_REVERT_LADDER_FILE}: "
                       "Evaluate + REVERT) — the mechanism that reverts a non-improving "
                       "change to the known-good baseline")
    if not ladder_tested:
        defects.append("the keep/revert ladder has no committed witness test "
                       "(internal/shipgate/*_test.go)")
    soft: list[str] = []
    if not defects and not state_seam_present:
        soft.append("no portable state/session capture+restore seam on this tree ("
                    + " · ".join(d.rstrip("/") for d in STATE_SEAM_DIRS)
                    + ") — state rollback rides git-tag + the keep/revert ladder")
    n = len(defects)
    return {"kpi": "keep_revert_ladder", "group": "revert",
            "score": _clamp(100 - 50 * n),
            "detail": ("committed keep/revert ladder exists + tested" if n == 0
                       else f"{n} gap(s) in the committed revert path"),
            "defects": defects, "soft": soft}


def kpi_version_pin(version_ok: bool, resolver_ok: bool) -> dict[str, Any]:
    """A single VERSION marker plus a resolver means 'what version is this' has one
    answer — the anchor a 'roll back to vX' instruction points at."""
    defects: list[str] = []
    if not version_ok:
        defects.append(f"no {VERSION_FILE} marker at the repo root — no single anchor for "
                       "'what version is this / roll back to'")
    if not resolver_ok:
        defects.append(f"no version resolver ({APPVERSION_PKG}) — the binary can't report its version")
    return {"kpi": "version_pin", "group": "revert",
            "score": _clamp(100 - 50 * len(defects)),
            "detail": ("VERSION marker + resolver present" if not defects
                       else f"{len(defects)} missing version affordance(s)"),
            "defects": defects, "soft": []}


def kpi_release_tagging_gated(present: bool) -> dict[str, Any]:
    """Release tagging that waits for CI green before it tags is what makes a tagged
    version a STABLE one you can roll back to with confidence."""
    defects: list[str] = []
    if not present:
        defects.append(f"no CI-gated release tagging ({RELEASE_CADENCE_WF}) — a stable, "
                       "roll-back-able tag isn't produced by a green-gated process")
    return {"kpi": "release_tagging_gated", "group": "revert",
            "score": 100 if present else 30,
            "detail": ("release tagging is CI-gated" if present
                       else "no CI-gated tagging of stable versions"),
            "defects": defects, "soft": []}


def kpi_rollback_runbook(present: bool, where: str, missing_sections: list[str],
                         linked: bool, has_commands: bool) -> dict[str, Any]:
    """A DOCUMENTED, linked operator runbook for reverting to a stable version — the
    gap between 'the snapshot/version mechanisms exist' and 'an operator can actually
    roll back'. Each absent piece (the doc, runnable commands, each uncovered
    mechanism, the link) is one unit. Each mechanism check is anchored to a REAL
    identifier (a git command, a `fak` verb, a tool path, an env var), so an empty
    file can't pass by prose-stuffing — the worst-first revert gap on a fast trunk."""
    defects: list[str] = []
    if not present:
        defects.append("no operator rollback runbook on disk (e.g. docs/ROLLBACK.md) — "
                       "the revert mechanisms exist but no doc tells an operator how to use them")
    elif not has_commands:
        defects.append(f"rollback runbook has no runnable commands (needs >= {MIN_RUNBOOK_FENCES} "
                       "fenced code blocks) — a runbook without commands is documentation theater")
    for label in missing_sections:
        defects.append(f"rollback runbook does not cover: {label}")
    if present and not linked:
        defects.append("rollback runbook exists but is linked from no entry point "
                       f"({'/'.join(ROLLBACK_RUNBOOK_LINKERS)}) — an operator can't find it")
    n = len(defects)
    return {"kpi": "rollback_runbook", "group": "revert",
            "score": _clamp(100 - 16 * n),
            "detail": (f"rollback runbook present ({where}), covers every mechanism" if n == 0
                       else f"{n} gap(s) in the operator rollback runbook"),
            "defects": defects, "soft": []}


def kpi_drift_detectors_wired(missing: list[str], no_tail_wag_tool: bool,
                              no_early_warning: bool) -> dict[str, Any]:
    """The silent-drift detectors — a small thing distorting a big thing gets caught
    before a reader/operator sees it. A missing detector tool is one unit. The
    tail-wag finder having no deterministic backing and the absence of an early-warning
    (pre-regression) trend signal are SOFT frontier nudges."""
    defects = [f"missing drift detector {path} ({label})"
               for label, path in ((lbl, p) for lbl, p in REQUIRED_DRIFT_DETECTORS if p in missing)]
    soft: list[str] = []
    if no_tail_wag_tool:
        soft.append("the /tail-wag inverted-priority finder is a manual skill with no "
                    "deterministic backing tool (tools/*tail_wag*.py) — it can't run in a gate")
    if no_early_warning:
        soft.append("the portfolio trend only flags a regression ABOVE the pinned baseline; "
                    "no early-warning on a first downward move WITHIN a healthy envelope")
    covered = len(REQUIRED_DRIFT_DETECTORS) - len(missing)
    return {"kpi": "drift_detectors_wired", "group": "drift",
            "score": _clamp(100 * covered / max(1, len(REQUIRED_DRIFT_DETECTORS))),
            "detail": f"{covered}/{len(REQUIRED_DRIFT_DETECTORS)} silent-drift detectors present",
            "defects": defects, "soft": soft}


def kpi_confusion_escalation_signal(has_signal: bool, doctrine_present: bool) -> dict[str, Any]:
    """SOFT: confusion is when a cheap rung can't conclusively decide. Is there an
    explicit 'I could not decide → escalate' (INDETERMINATE) disposition, or is
    confusion only ever fail-open / fail-closed? This emits no HARD debt — wiring a
    new verdict is a frontier feature, not a checklist item — but it scores the gap
    honestly and points at the doctrine that tracks it."""
    soft: list[str] = []
    if has_signal:
        score = 100
        detail = "an INDETERMINATE / escalate disposition exists in the adjudication core"
    else:
        score = 50
        detail = "no explicit INDETERMINATE / escalate disposition (confusion is fail-open/closed only)"
        note = ("no 'I could not decide -> escalate' (INDETERMINATE) verdict in "
                f"{', '.join(CONFUSION_SIGNAL_FILES)}")
        if doctrine_present:
            note += f" -- tracked as the frontier in {VERIFICATION_LADDER_DOC}"
        soft.append(note)
    return {"kpi": "confusion_escalation_signal", "group": "drift", "score": score,
            "detail": detail, "defects": [], "soft": soft}


# ---------------------------------------------------------------------------
# Fold: KPIs -> composite score, grade, stability-debt, control-pane payload.
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
    score = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"]
                      for n in KPI_WEIGHTS if n in by_name), 1)
    stability_debt = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)
    grade = grade_letter(score)

    debt_by_group = {g: 0 for g in GROUPS}
    for k in kpis:
        debt_by_group[k["group"]] += len(k["defects"])
    score_by_group = {g: 0.0 for g in GROUPS}
    wsum_by_group = {g: 0.0 for g in GROUPS}
    for k in kpis:
        w = KPI_WEIGHTS.get(k["kpi"], 0.0)
        score_by_group[k["group"]] += w * k["score"]
        wsum_by_group[k["group"]] += w
    group_scores = {g: (round(score_by_group[g] / wsum_by_group[g], 1)
                        if wsum_by_group[g] else 0.0) for g in GROUPS}

    breakdown = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    corpus = {
        "score": score, "grade": grade, "stability_debt": stability_debt,
        "soft_signals": n_soft,
        "group_scores": group_scores,
        "debt_by_group": debt_by_group,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
    }

    gs = group_scores
    standing = (f"sentinel {gs['sentinel']:.0f} · invariant {gs['invariant']:.0f} "
                f"· revert {gs['revert']:.0f} · drift {gs['drift']:.0f}")
    if stability_debt == 0:
        ok, verdict, finding = True, "OK", "stable"
        reason = (f"stable: score {score}/100 (grade {grade}), zero stability-debt across "
                  f"{len(kpis)} KPIs ({standing}; {n_soft} advisory). A regression fails "
                  "loudly and there is a written path back to a stable version")
        next_action = ("hold the line; re-run after a change to a gate, an invariant test, "
                       "a baseline, or the rollback runbook")
    else:
        ok, verdict, finding = False, "ACTION", "stability_debt"
        worst = breakdown[0]
        reason = (f"{stability_debt} unit(s) of stability-debt; score {score}/100 (grade {grade}); "
                  f"heaviest: {worst['kpi']} ({worst['debt']} defect(s)); standing {standing}")
        next_action = ("retire stability-debt worst-first (see corpus.breakdown + per-KPI "
                       "defects): wire the missing regression gate, encode the missing "
                       "invariant, commit the missing baseline, or write/link the rollback "
                       "runbook; re-run to prove the drop")

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
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _ci_text(root: Path) -> str:
    """The concatenated CI surface: every .github/workflows/*.yml + Makefile +
    scripts/ci.ps1. A regression gate counts as 'wired' if its token appears here."""
    parts: list[str] = []
    wf = root / ".github" / "workflows"
    if wf.is_dir():
        for p in sorted(wf.glob("*.yml")) + sorted(wf.glob("*.yaml")):
            parts.append(_safe_read(p))
    parts.append(_safe_read(root / "Makefile"))
    parts.append(_safe_read(root / "scripts" / "ci.ps1"))
    return "\n".join(parts)


def _tests_text(root: Path, tracked: set[str]) -> str:
    """The concatenated text of every tracked *_test.go — searched for the named
    invariant / fail-closed test symbols. Read once, substring-searched many times."""
    parts: list[str] = []
    for rel in sorted(tracked):
        if rel.endswith("_test.go"):
            parts.append(_safe_read(root / rel))
    return "\n".join(parts)


def gather(root: Path) -> list[dict[str, Any]]:
    """Read the git-tracked tree and run every pure KPI."""
    tracked = set(_git_lines(["ls-files"], root))

    def present(rel: str) -> bool:
        return rel in tracked or (root / rel).exists()

    ci_text = _ci_text(root)
    tests_text = _tests_text(root, tracked)

    # --- sentinel ---
    gates_missing = [label for label, toks in REQUIRED_REGRESSION_GATES
                     if not _has(ci_text, *toks)]
    baselines_missing = [path for path, _ in RATCHET_BASELINES if not present(path)]
    claims_present = present(CLAIMS_FILE)
    claims_text = _safe_read(root / CLAIMS_FILE) if claims_present else None
    untagged = untagged_claims(claims_text)

    # --- invariant ---
    invariants_missing = [sym for _, sym in REQUIRED_INVARIANT_TESTS if sym not in tests_text]
    fail_closed_missing = [sym for _, sym in REQUIRED_FAIL_CLOSED if sym not in tests_text]
    determinism_count = sum(1 for rel in tracked if rel.endswith(DETERMINISM_WITNESS_GLOB))

    # --- revert ---
    ladder_text = _safe_read(root / KEEP_REVERT_LADDER_FILE) \
        if KEEP_REVERT_LADDER_FILE in tracked else ""
    ladder_ok = all(tok in ladder_text for tok in KEEP_REVERT_LADDER_TOKENS)
    ladder_tested = any(rel.startswith("internal/shipgate/") and rel.endswith("_test.go")
                        for rel in tracked)
    state_seam_present = any(rel.startswith(d) for d in STATE_SEAM_DIRS for rel in tracked)
    version_ok = present(VERSION_FILE)
    resolver_ok = present(APPVERSION_PKG)
    release_gated = present(RELEASE_CADENCE_WF)

    runbook_where = next((c for c in ROLLBACK_RUNBOOK_CANDIDATES if present(c)), "")
    runbook_text = _safe_read(root / runbook_where) if runbook_where else ""
    missing_sections = [label for label, toks in ROLLBACK_RUNBOOK_SECTIONS
                        if not _has(runbook_text, *toks)]
    runbook_linked = any(_has(_safe_read(root / linker), Path(runbook_where).name)
                         for linker in ROLLBACK_RUNBOOK_LINKERS) if runbook_where else False
    runbook_has_commands = _count_fences(runbook_text) >= MIN_RUNBOOK_FENCES

    # --- drift ---
    detectors_missing = [path for _, path in REQUIRED_DRIFT_DETECTORS if not present(path)]
    no_tail_wag_tool = not any(TAIL_WAG_TOOL_GLOB in rel for rel in tracked
                               if rel.startswith("tools/") and rel.endswith(".py"))
    confusion_text = "\n".join(
        _safe_read(root / rel) for rel in tracked
        if rel.endswith(".go") and not rel.endswith("_test.go")
        and any(rel.startswith(d) for d in CONFUSION_SIGNAL_FILES))
    has_confusion_signal = _has(confusion_text, *CONFUSION_SIGNAL_TOKENS)
    doctrine_present = present(VERIFICATION_LADDER_DOC)

    return [
        kpi_regression_gates_wired(gates_missing),
        kpi_ratchet_baselines_committed(baselines_missing),
        kpi_honesty_ledger_clean(claims_present, untagged),
        kpi_invariant_tests_present(invariants_missing),
        kpi_frozen_pins_present(baselines_missing),
        kpi_fail_closed_witnessed(fail_closed_missing),
        kpi_determinism_witness(determinism_count),
        kpi_keep_revert_ladder(ladder_ok, ladder_tested, state_seam_present),
        kpi_version_pin(version_ok, resolver_ok),
        kpi_release_tagging_gated(release_gated),
        kpi_rollback_runbook(bool(runbook_where), runbook_where, missing_sections,
                             runbook_linked, runbook_has_commands),
        kpi_drift_detectors_wired(detectors_missing, no_tail_wag_tool, no_early_warning=True),
        kpi_confusion_escalation_signal(has_confusion_signal, doctrine_present),
    ]


def collect(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    if not (root / ".git").exists() and not _git_lines(["rev-parse", "--git-dir"], root):
        return build_payload(workspace=str(root), kpis=[],
                             error=f"not a git repo at {root} — run from the repo ROOT")
    return build_payload(workspace=str(root), kpis=gather(root))


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    gs = c.get("group_scores") or {}
    lines = [
        f"stability-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· STABILITY-DEBT {c.get('stability_debt', 0)} · {c.get('soft_signals', 0)} advisory"),
        (f"trustworthiness:  sentinel {gs.get('sentinel', 0):.0f}  ·  "
         f"invariant {gs.get('invariant', 0):.0f}  ·  revert {gs.get('revert', 0):.0f}  ·  "
         f"drift {gs.get('drift', 0):.0f}"),
        ("debt by group: " + "  ".join(
            f"{g}:{(c.get('debt_by_group') or {}).get(g, 0)}" for g in GROUPS)),
        "",
        "per-KPI (worst first):",
        f"  {'score':>5} {'debt':>4}  {'group':<10} {'kpi':<28} detail",
    ]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['group']:<10} "
                     f"{b['kpi']:<28} {b['detail']}")
    lines.append("")
    lines.append("stability-debt work-list:")
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
        lines.append("  (none — zero stability-debt)")
    soft_lines = [s for k in payload.get("kpis", []) for s in k["soft"]]
    if soft_lines:
        lines.append("")
        lines.append("advisory (soft) signals — frontier, not debt:")
        for s in soft_lines:
            lines.append(f"      - {s}")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    gs = c.get("group_scores") or {}
    out: list[str] = []
    out.append("---")
    out.append('title: "fak stability scorecard — the stability-debt measuring stick"')
    out.append('description: "fak\'s deterministic stability scorecard: KPIs across the four '
               'ways a fast-moving trunk stays trustworthy — sentinel (we find out a '
               'regression landed), invariant (the assumptions are encoded as tests), revert '
               '(we can roll back to a stable version), and drift (a small thing wagging a big '
               'thing gets caught) — folded into a composite score and the headline '
               'stability-debt metric, re-derived from the git-tracked tree."')
    out.append("---")
    out.append("")
    out.append("# Stability scorecard — can we tell when we broke something, and roll back")
    out.append("")
    if stamp:
        out.append(f"<!-- stability-scorecard: {stamp} · process: tools/stability_scorecard.py -->")
        out.append("")
    out.append("This is the measuring stick for fak's **stability under fast iteration** — the "
               "question a team living on a rapidly-changing trunk loses sleep over: as we add "
               "items fast, how do we **know** a regression, tail-wag, or confusion landed, and "
               "how do we **revert** to a stable version? Every number below is re-derived from "
               "the git-tracked tree by `tools/stability_scorecard.py` — no hand-entry. The "
               "headline metric is **stability-debt**: the count of concrete, mechanical defects "
               "that leave fak unable to catch a regression or roll one back — a missing CI gate, "
               "an unencoded invariant, an uncommitted baseline, a missing rollback runbook. "
               "Driving stability-debt to zero is what lets fak change fast *and* stay trustworthy.")
    out.append("")
    out.append("> Regenerate: `python tools/stability_scorecard.py --markdown --stamp DATE > "
               f"{GENERATED_SNAPSHOT}`")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Stability-debt (total HARD defects)** | **{c.get('stability_debt', 0)}** |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append(f"| Trustworthiness | sentinel {gs.get('sentinel', 0):.0f} · "
               f"invariant {gs.get('invariant', 0):.0f} · revert {gs.get('revert', 0):.0f} · "
               f"drift {gs.get('drift', 0):.0f} |")
    out.append(f"| Advisory (soft) signals | {c.get('soft_signals', 0)} |")
    g = c.get("debt_by_group", {})
    out.append(f"| Debt by group | sentinel:{g.get('sentinel',0)} · invariant:{g.get('invariant',0)} "
               f"· revert:{g.get('revert',0)} · drift:{g.get('drift',0)} |")
    out.append("")
    out.append("> The composite tops out below 100 even at zero debt: "
               "`confusion_escalation_signal` is a SOFT frontier signal (no INDETERMINATE "
               "verdict is wired in the adjudication core yet), scored 50 at weight 0.04, so it "
               "subtracts ~2 from the composite without adding HARD debt. Zero stability-debt is "
               "the headline; the ~2 is the honest frontier the soft signals track.")
    out.append("")
    out.append("## The four ways a fast-moving trunk stays trustworthy")
    out.append("")
    out.append(f"{len(payload.get('kpis', []))} KPIs, each 0–100, grouped by the property they "
               "defend. `debt` = units of HARD stability-debt. `confusion_escalation_signal` is "
               "advisory (it scores but emits no hard debt — wiring a new verdict is a frontier "
               "feature, not a checklist item).")
    out.append("")
    out.append("| Group | KPI | Score | Debt | Detail |")
    out.append("|---|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    out.append("## Stability-debt work-list")
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
        out.append("No stability-debt: a regression fails loudly and there is a written path "
                   "back to a stable version. 🎉")
        out.append("")
    soft_lines = [s for k in payload.get("kpis", []) for s in k["soft"]]
    if soft_lines:
        out.append("## Advisory (soft) signals — the frontier, not debt")
        out.append("")
        for s in soft_lines:
            out.append(f"- {s}")
        out.append("")
    return "\n".join(out)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("stability_debt", 0), cur.get("stability_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ratio = "∞ (zero)" if cd == 0 else f"{bd / cd:.1f}×"
    lines = [
        f"stability-debt: {bd} -> {cd}   ({ratio} fewer defects)",
        f"score:          {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<10} {gb} -> {gc}")
    target = max(0, bd // 2)
    if cd <= target:
        lines.append(f"VERDICT: >=2x stability-debt reduction achieved ({bd} -> {cd}).")
    else:
        lines.append(f"VERDICT: not yet 2x — need stability-debt <= {target} (now {cd}).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stability scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the snapshot markdown body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the stability-debt delta vs a prior baseline JSON")
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
