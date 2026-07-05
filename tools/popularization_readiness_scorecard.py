#!/usr/bin/env python3
"""Popularization-readiness scorecard — does fak's front door actually convert?

The sibling scorecards each grade fak through ONE lens: ``agent_readiness`` asks
whether an AI *agent* can discover/adopt/build on fak, ``persona_readiness`` asks
whether each of the top-10 humans who land on fak is served, ``industry`` grades fak
vs the field. None asks the *popularization* question — the one the concept-
popularization epic (``docs/notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md``,
dimension K, "adoption measurement") exists to answer:

  **When a first-time visitor lands on fak's front door — the README, the five-
  concepts framing, the install one-liner, the social-proof surfaces, the first
  runnable command — does that door actually CONVERT them, stage by stage, or does
  it LEAK them? Does the one affordance a visitor reaches for at each step of the
  funnel actually exist in the tree?**

That is this scorecard. It models the front door as an ordered CONVERSION FUNNEL a
first-time visitor walks — land → orient → trust → install → act — where each
**stage** is a row that names WHO the visitor is at that moment, the JOB that must
happen for them to advance, and a list of concrete **affordances**: the files,
docs, fenced commands, and honesty-ledger entries the front door needs to keep them
moving. Every affordance is a deterministic check against the REAL tree, so the
score cannot be gamed by editing data alone: to stop a stage leaking you ADD the
real affordance (a headline number on the README, a named concept explainer, a
one-line install that resolves, a first command with an observable result), exactly
the way ``persona_readiness`` is retired by adding the real persona affordance.

The affordance **check kinds** (the fixed doctrine, shared with the persona sibling):

  path_exists       a named file / dir exists in the tracked tree
  any_path_exists   any of several candidate paths exists (a doc that may move)
  doc_mentions      a doc exists and contains the token(s) a visitor greps for
  fenced_command    a copy-pasteable command lives in a fenced block of a doc
  command_resolves  a `go run ./cmd/<dir> <verb>` resolves (dir exists, verb documented)
  claim_section     a CLAIMS.md `##` concept section exists (optionally with a tag)

Each stage folds its met/unmet HARD affordances into a conversion fraction and a
**verdict** the evidence implies (the same overclaim catch ``persona`` uses):

  converts             every hard affordance this stage needs is present
  mostly-converts      >= 75% present (a small leak)
  partially-converts   >= 40% present
  leaks                the stage is mostly missing — visitors drop here

Two numbers are driven, mirroring ``persona`` / ``industry``:

  POPULARIZATION-DEBT  unmet HARD affordances + malformed rows + verdict overclaims +
                       coverage gaps (a required funnel stage with no row). Drive it
                       toward 0 — zero means every stage of the front door converts,
                       and the number becomes a regression sentinel (a removed
                       headline number, a dead concept explainer, a broken install
                       one-liner all raise it). Folds into
                       ``scorecard_control_pane`` via ``corpus.popularization_debt``.
  COVERAGE             of the declared funnel stages, how many are positioned at all.

Deterministic + read-only over the data (two clones at one commit score
identically); the only disk writes are the generated doc folder under
``--markdown-dir``. The source of truth is a DIRECTORY of small JSON files so the
funnel and each stage's affordances evolve independently::

    tools/popularization_readiness_scorecard.data/
      _meta.json        meta + the declared funnel phases + required_stages roster
      rows-*.json       stage rows, grouped by phase (land / orient / …)

Run from the repo ROOT::

    python tools/popularization_readiness_scorecard.py                 # human scorecard
    python tools/popularization_readiness_scorecard.py --chart         # at-a-glance ASCII chart
    python tools/popularization_readiness_scorecard.py --json          # machine payload (control-pane / loop)
    python tools/popularization_readiness_scorecard.py --critical      # the worst-leaking stages backlog
    python tools/popularization_readiness_scorecard.py --gaps          # the coverage backlog (unpositioned stages)
    python tools/popularization_readiness_scorecard.py --compare base.json    # prove popularization-debt dropped
    python tools/popularization_readiness_scorecard.py --markdown-dir docs/popularization-scorecard   # regenerate the doc folder
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

SCHEMA = "fak-popularization-readiness-scorecard/1"
DATA_DIR_REL = "tools/popularization_readiness_scorecard.data"
GENERATED_DOC_DIR = "docs/popularization-scorecard"
CLAIMS_REL = "CLAIMS.md"
CLI_REF_REL = "docs/cli-reference.md"

# ---------------------------------------------------------------------------
# Closed vocabularies. `phase` is DATA-defined (declared in _meta.json) so the
# funnel can grow; the vocabularies below ARE the doctrine and fixed.
# ---------------------------------------------------------------------------

# The affordance check kinds — each a pure predicate over real-tree facts.
CHECK_KINDS = {
    "path_exists": "a named file / dir exists in the tree",
    "any_path_exists": "any of several candidate paths exists",
    "doc_mentions": "a doc exists and contains the token(s) a visitor greps for",
    "fenced_command": "a copy-pasteable command lives in a fenced block of a doc",
    "command_resolves": "a `go run ./cmd/<dir> <verb>` resolves (dir + documented verb)",
    "claim_section": "a CLAIMS.md `##` concept section exists (optionally tagged)",
}

SEVERITIES = {"hard", "soft"}
# How much reading the visitor will invest at this stage (a landing visitor
# glances; a would-be adopter reads). The spectrum the funnel spans.
EFFORTS = {"glance", "skim", "read"}

# The conversion verdict ladder, best -> worst. The rank doubles as the "distance
# from converts" used to order the worst-leaking backlog.
VERDICTS = ["converts", "mostly-converts", "partially-converts", "leaks"]
VERDICT_RANK = {v: i for i, v in enumerate(VERDICTS)}

# The met-fraction thresholds the verdict ladder folds to (the evidence-implied
# verdict each stage is cross-checked against — the overclaim catch).
CONVERTS_AT = 1.0
MOSTLY_AT = 0.75
PARTIAL_AT = 0.40

GROUPS = ("well-formed", "reality", "honesty")
KPI_GROUP: dict[str, str] = {
    "stages_well_formed": "well-formed",
    "affordances_present": "reality",
    "verdict_honest": "honesty",
}
KPI_WEIGHTS: dict[str, float] = {
    "stages_well_formed": 0.15,
    "affordances_present": 0.55,
    "verdict_honest": 0.30,
}
KPI_PENALTY: dict[str, int] = {
    "stages_well_formed": 10,
    "verdict_honest": 25,
}
# The composite blends how well the stages that EXIST convert with how much of the
# declared funnel is even positioned. An incomplete funnel costs grade.
HONESTY_WEIGHT = 0.65
COVERAGE_WEIGHT = 0.35

REQUIRED_FIELDS = (
    "id", "stage", "visitor", "job", "phase", "effort", "entry_doc",
    "affordances", "verdict", "gaps",
)
AFFORDANCE_FIELDS = ("id", "kind", "need", "severity")


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


def _nonempty(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def norm_section(s: Any) -> str:
    """Normalize a `## header` / section to a comparable key: drop leading hashes,
    cut at the first parenthesis / em-dash / colon, lowercase."""
    s = (s or "")
    if not isinstance(s, str):
        return ""
    s = s.strip()
    while s.startswith("#"):
        s = s[1:]
    s = s.strip()
    for sep in ("(", "—", ":"):
        idx = s.find(sep)
        if idx != -1:
            s = s[:idx]
    return " ".join(s.lower().split())


def section_match(want: str, catalog_norm: str) -> bool:
    """Tolerant match between a wanted section and a catalog section: equal, or one
    normalized form contains the other (guarded against trivial overlaps)."""
    a, b = norm_section(want), catalog_norm
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 6 and len(b) >= 6 and (a in b or b in a):
        return True
    return False


def parse_command(cmd: Any) -> tuple[str | None, str | None]:
    """Pull (cmd_dir, verb) out of a `go run ./cmd/<dir> <verb> ...` command.

    cmd_dir is the directory under cmd/; verb is the next non-flag token (the fak
    subcommand). (None, None) when no `./cmd/<dir>` token is present."""
    if not _nonempty(cmd):
        return None, None
    m = re.search(r"\./cmd/([\w-]+)", cmd)
    if not m:
        return None, None
    cmd_dir = m.group(1)
    rest = cmd[m.end():].strip()
    verb = None
    for tok in rest.split():
        if tok.startswith("-"):
            break
        verb = tok
        break
    return cmd_dir, verb


_FENCE_RE = re.compile(r"^(```|~~~)")


def fenced_blocks(text: str) -> list[str]:
    """The contents of every fenced code block — where a copy-pasteable command
    lives. A token found only in prose does not count as a runnable command."""
    blocks: list[str] = []
    cur: list[str] = []
    in_fence = False
    for raw in (text or "").split("\n"):
        if _FENCE_RE.match(raw.strip()):
            if in_fence:
                blocks.append("\n".join(cur))
                cur = []
            in_fence = not in_fence
            continue
        if in_fence:
            cur.append(raw)
    return blocks


# ---------------------------------------------------------------------------
# Affordance evaluation — the pure predicate over tree facts. ``tree`` carries the
# callables the checks need; every kind returns (met: bool, detail: str).
# ---------------------------------------------------------------------------

def eval_affordance(aff: dict[str, Any], tree: dict[str, Any]) -> tuple[bool, str]:
    kind = aff.get("kind")
    exists: Callable[[str], bool] = tree.get("exists") or (lambda p: False)
    doc_text: Callable[[str], str] = tree.get("doc_text") or (lambda d: "")

    if kind == "path_exists":
        t = aff.get("target", "")
        if not _nonempty(t):
            return False, "path_exists with no target"
        ok = exists(t)
        return ok, (f"{t} exists" if ok else f"{t} missing")

    if kind == "any_path_exists":
        targets = [t for t in (aff.get("targets") or []) if _nonempty(t)]
        hit = next((t for t in targets if exists(t)), None)
        return (hit is not None), (f"found {hit}" if hit else f"none exist: {targets}")

    if kind == "doc_mentions":
        doc = aff.get("doc", "")
        tokens = [t for t in (aff.get("tokens") or []) if _nonempty(t)]
        match = aff.get("match", "any")
        text = doc_text(doc)
        if not text:
            return False, f"{doc} missing/empty"
        low = text.lower()
        present = [t for t in tokens if t.lower() in low]
        # An empty token list never satisfies — require at least one hit (and for
        # match=all, every token), so a token-less check can't pass vacuously.
        ok = bool(present) and (len(present) == len(tokens) if match == "all" else True)
        return ok, (f"{doc} mentions {present}" if ok
                    else f"{doc} lacks {tokens} (match={match})")

    if kind == "fenced_command":
        docs = [d for d in (aff.get("docs") or []) if _nonempty(d)]
        tokens = [t for t in (aff.get("tokens") or []) if _nonempty(t)]
        for d in docs:
            for block in fenced_blocks(doc_text(d)):
                low = block.lower()
                if any(t.lower() in low for t in tokens):
                    return True, f"fenced command in {d}"
        return False, f"no fenced block with {tokens} in {docs}"

    if kind == "command_resolves":
        cmd = aff.get("command", "")
        cmd_dir, verb = parse_command(cmd)
        if cmd_dir is None:
            return False, f"{cmd!r} has no ./cmd/<dir>"
        if cmd_dir not in (tree.get("cmd_dirs") or set()):
            return False, f"./cmd/{cmd_dir} does not exist"
        if cmd_dir == "fak" and verb and verb.lower() not in (tree.get("doc_verbs") or set()):
            return False, f"fak verb {verb!r} not documented in cli-reference"
        return True, f"resolves: ./cmd/{cmd_dir} {verb or ''}".rstrip()

    if kind == "claim_section":
        section = aff.get("section", "")
        tags = {t for t in (aff.get("tags") or []) if _nonempty(t)}
        section_tags = tree.get("section_tags") or {}
        for norm, tagset in section_tags.items():
            if section_match(section, norm):
                if not tags or (tagset & tags):
                    return True, f"CLAIMS section {section!r} present"
                return False, (f"CLAIMS section {section!r} lacks any of {sorted(tags)} "
                               f"(has {sorted(tagset)})")
        return False, f"no CLAIMS section matching {section!r}"

    return False, f"unknown check kind {kind!r}"


# ---------------------------------------------------------------------------
# Per-stage scoring — fold a row's affordances into met/unmet + a verdict.
# ---------------------------------------------------------------------------

def expected_verdict(frac: float, n_hard: int) -> str:
    """The conversion verdict the evidence implies, from the hard-affordance met
    fraction. A stage with no hard affordances is vacuously converting."""
    if n_hard == 0 or frac >= CONVERTS_AT:
        return "converts"
    if frac >= MOSTLY_AT:
        return "mostly-converts"
    if frac >= PARTIAL_AT:
        return "partially-converts"
    return "leaks"


def score_stage(row: dict[str, Any], tree: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for aff in (row.get("affordances") or []):
        if not isinstance(aff, dict):
            continue
        met, detail = eval_affordance(aff, tree)
        results.append({
            "id": aff.get("id"), "kind": aff.get("kind"),
            "severity": aff.get("severity"), "need": aff.get("need"),
            "met": met, "detail": detail,
        })
    hard = [r for r in results if r["severity"] == "hard"]
    soft = [r for r in results if r["severity"] == "soft"]
    hard_met = sum(1 for r in hard if r["met"])
    frac = (hard_met / len(hard)) if hard else 1.0
    return {
        "results": results, "hard": hard, "soft": soft,
        "hard_met": hard_met, "hard_total": len(hard),
        "frac": round(frac, 3), "expected_verdict": expected_verdict(frac, len(hard)),
    }


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each returns
#   {kpi, group, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of popularization-debt; soft = score-only judgment nudges.
# Every defect string is prefixed `<stage id>: ` so per-row debt is recoverable.
# ---------------------------------------------------------------------------

def _kpi(name: str, score: int, defects: list[str], detail: str,
         *, soft: list[str] | None = None) -> dict[str, Any]:
    return {"kpi": name, "group": KPI_GROUP[name], "score": _clamp(score),
            "detail": detail, "defects": defects, "soft": soft or []}


def _affordance_well_formed(aff: dict[str, Any]) -> list[str]:
    """Field-level defects for one affordance (returned without an id prefix; the
    caller prepends `<stage>/<aff>:`)."""
    out: list[str] = []
    for f in AFFORDANCE_FIELDS:
        if f not in aff:
            out.append(f"missing field '{f}'")
    if aff.get("kind") not in CHECK_KINDS:
        out.append(f"kind {aff.get('kind')!r} not in {sorted(CHECK_KINDS)}")
    if aff.get("severity") not in SEVERITIES:
        out.append(f"severity {aff.get('severity')!r} not in {sorted(SEVERITIES)}")
    kind = aff.get("kind")
    if kind == "path_exists" and not _nonempty(aff.get("target")):
        out.append("path_exists needs a 'target' string")
    if kind == "any_path_exists" and not (isinstance(aff.get("targets"), list) and aff["targets"]):
        out.append("any_path_exists needs a non-empty 'targets' list")
    if kind == "doc_mentions":
        if not _nonempty(aff.get("doc")):
            out.append("doc_mentions needs a 'doc' string")
        if not (isinstance(aff.get("tokens"), list) and aff["tokens"]):
            out.append("doc_mentions needs a non-empty 'tokens' list")
    if kind == "fenced_command":
        if not (isinstance(aff.get("docs"), list) and aff["docs"]):
            out.append("fenced_command needs a non-empty 'docs' list")
        if not (isinstance(aff.get("tokens"), list) and aff["tokens"]):
            out.append("fenced_command needs a non-empty 'tokens' list")
    if kind == "command_resolves" and not _nonempty(aff.get("command")):
        out.append("command_resolves needs a 'command' string")
    if kind == "claim_section" and not _nonempty(aff.get("section")):
        out.append("claim_section needs a 'section' string")
    return out


def kpi_stages_well_formed(rows: list[dict[str, Any]], phases: set[str]) -> dict[str, Any]:
    """A row must be shaped like a funnel stage: required fields present, phase
    declared, effort/verdict in their vocabularies, ids unique, and every affordance
    well-formed for its kind. A malformed row can't be honestly scored, so it is
    hard debt."""
    defects: list[str] = []
    seen: set[str] = set()
    for i, r in enumerate(rows):
        rid = r.get("id") if _nonempty(r.get("id")) else f"row[{i}]"
        for f in REQUIRED_FIELDS:
            if f not in r:
                defects.append(f"{rid}: missing field '{f}'")
        if not _nonempty(r.get("id")):
            defects.append(f"{rid}: missing id")
        elif r["id"] in seen:
            defects.append(f"{rid}: duplicate id")
        else:
            seen.add(r["id"])
        if phases and r.get("phase") not in phases:
            defects.append(f"{rid}: phase {r.get('phase')!r} not declared in _meta.json")
        if r.get("effort") not in EFFORTS:
            defects.append(f"{rid}: effort {r.get('effort')!r} not in {sorted(EFFORTS)}")
        if r.get("verdict") not in VERDICT_RANK:
            defects.append(f"{rid}: verdict {r.get('verdict')!r} not in {VERDICTS}")
        if not isinstance(r.get("gaps"), list):
            defects.append(f"{rid}: gaps must be a list")
        affs = r.get("affordances")
        if not (isinstance(affs, list) and affs):
            defects.append(f"{rid}: affordances must be a non-empty list")
            continue
        aff_ids: set[str] = set()
        for j, aff in enumerate(affs):
            if not isinstance(aff, dict):
                defects.append(f"{rid}: affordance[{j}] is not an object")
                continue
            aid = aff.get("id") if _nonempty(aff.get("id")) else f"aff[{j}]"
            if _nonempty(aff.get("id")):
                if aff["id"] in aff_ids:
                    defects.append(f"{rid}/{aid}: duplicate affordance id")
                aff_ids.add(aff["id"])
            for problem in _affordance_well_formed(aff):
                defects.append(f"{rid}/{aid}: {problem}")
    n = len(rows)
    return _kpi("stages_well_formed",
                100 - KPI_PENALTY["stages_well_formed"] * len(defects),
                defects,
                f"all {n} funnel stages well-formed" if not defects
                else f"{len(defects)} malformed field(s)")


def kpi_affordances_present(rows: list[dict[str, Any]],
                            scored: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The core check: every HARD affordance a stage needs must be PRESENT in the
    tree. Each unmet hard affordance is one unit of popularization-debt — a real
    leak you retire by ADDING the affordance (a headline number, a concept explainer,
    a resolving install one-liner), never by editing the data. Unmet SOFT affordances
    are score-only nudges."""
    defects: list[str] = []
    soft: list[str] = []
    hard_total = hard_met = 0
    for r in rows:
        rid = r.get("id", "?")
        s = scored.get(rid) or {}
        for a in s.get("hard", []):
            hard_total += 1
            if a["met"]:
                hard_met += 1
            else:
                defects.append(f"{rid}: missing affordance '{a['id']}' — {a['need']} "
                               f"({a['detail']})")
        for a in s.get("soft", []):
            if not a["met"]:
                soft.append(f"{rid}: soft affordance '{a['id']}' absent — {a['need']} "
                            f"({a['detail']})")
    rate = (100 * hard_met / hard_total) if hard_total else 100
    return _kpi("affordances_present", rate, defects,
                f"{hard_met}/{hard_total} hard affordances present "
                f"({rate:.0f}%)" if hard_total else "no hard affordances declared",
                soft=soft)


def kpi_verdict_honest(rows: list[dict[str, Any]],
                       scored: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The stated verdict must match what the affordance evidence implies. Declaring
    a stage 'converts' when its hard affordances are missing is an overclaim (hard
    debt) — the single most important thing this scorecard catches. Declaring a stage
    WORSE than the evidence (underclaim) is conservative — a soft nudge."""
    defects: list[str] = []
    soft: list[str] = []
    for r in rows:
        rid = r.get("id", "?")
        declared = r.get("verdict")
        s = scored.get(rid) or {}
        exp = s.get("expected_verdict")
        if declared not in VERDICT_RANK or exp not in VERDICT_RANK:
            continue
        dr, er = VERDICT_RANK[declared], VERDICT_RANK[exp]
        if dr < er:  # declared BETTER than the evidence -> overclaim
            defects.append(f"{rid}: claims '{declared}' but only "
                           f"{s.get('hard_met')}/{s.get('hard_total')} hard affordances "
                           f"present implies '{exp}' — overclaim")
        elif dr > er:  # declared worse than the evidence -> underclaim
            soft.append(f"{rid}: claims '{declared}' but evidence implies '{exp}' — "
                        f"underclaim (raise the verdict)")
    return _kpi("verdict_honest",
                100 - KPI_PENALTY["verdict_honest"] * len(defects),
                defects,
                "every verdict matches its affordance evidence" if not defects
                else f"{len(defects)} verdict overclaim(s)",
                soft=soft)


# ---------------------------------------------------------------------------
# Coverage (how much of the declared funnel is positioned at all).
# ---------------------------------------------------------------------------

def coverage_report(required: list[dict[str, str]],
                    rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A required stage is 'covered' when a row carries its id. An uncovered one is
    coverage_debt: a funnel stage fak commits to but the scorecard omits."""
    have = {r.get("id") for r in rows if _nonempty(r.get("id"))}
    uncovered = [p for p in required if p.get("id") not in have]
    total = len(required)
    covered = total - len(uncovered)
    pct = round(100.0 * covered / total, 1) if total else 100.0
    return {
        "required_total": total,
        "covered": covered,
        "coverage_pct": pct,
        "coverage_debt": len(uncovered),
        "uncovered": uncovered,
    }


# ---------------------------------------------------------------------------
# Fold: KPIs + coverage -> composite score, grade, popularization-debt, payload.
# ---------------------------------------------------------------------------

def standing(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {v: 0 for v in VERDICTS}
    for r in rows:
        v = r.get("verdict")
        if v in counts:
            counts[v] += 1
    return counts


def per_row_debt(rows: list[dict[str, Any]], kpis: list[dict[str, Any]]) -> dict[str, int]:
    # Key by the SAME label kpi_stages_well_formed uses (`row[i]` for a missing OR
    # empty id), so a malformed empty-id row's defects still attribute to a bucket.
    out: dict[str, int] = {(r["id"] if _nonempty(r.get("id")) else f"row[{i}]"): 0
                           for i, r in enumerate(rows)}
    for k in kpis:
        for d in k["defects"]:
            rid = d.split(":", 1)[0].split("/", 1)[0]
            if rid in out:
                out[rid] += 1
    return out


def leaderboard(rows: list[dict[str, Any]],
                scored: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        rid = r.get("id")
        s = scored.get(rid) or {}
        missing = [a["id"] for a in s.get("hard", []) if not a["met"]]
        out.append({
            "id": rid,
            "stage": r.get("stage"),
            "phase": r.get("phase"),
            "effort": r.get("effort"),
            "verdict": r.get("verdict"),
            "expected_verdict": s.get("expected_verdict"),
            "hard_met": s.get("hard_met", 0),
            "hard_total": s.get("hard_total", 0),
            "frac": s.get("frac", 1.0),
            "job": r.get("job"),
            "entry_doc": r.get("entry_doc"),
            "missing": missing,
        })
    return out


def critical_backlog(rows: list[dict[str, Any]], row_debt: dict[str, int],
                     scored: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The worst-leaking stages, worst-first by (defect count, distance-from-converts)."""
    out = []
    for r in rows:
        rid = r.get("id")
        s = scored.get(rid) or {}
        out.append({
            "id": rid,
            "stage": r.get("stage"),
            "phase": r.get("phase"),
            "verdict": r.get("verdict"),
            "debt": row_debt.get(rid, 0),
            "distance": VERDICT_RANK.get(r.get("verdict"), 9),
            "missing": [a["id"] for a in s.get("hard", []) if not a["met"]],
            "gaps": r.get("gaps") or [],
        })
    out.sort(key=lambda x: (-x["debt"], -x["distance"], x["id"] or ""))
    return out


def run_kpis(rows: list[dict[str, Any]], phases: set[str],
             scored: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        kpi_stages_well_formed(rows, phases),
        kpi_affordances_present(rows, scored),
        kpi_verdict_honest(rows, scored),
    ]


def build_payload(*, workspace: str, data: dict[str, Any] | None, tree: dict[str, Any],
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
    phase_defs = [t for t in (data.get("phases") or []) if isinstance(t, dict)]
    phases = {t.get("id") for t in phase_defs if _nonempty(t.get("id"))}
    required = [p for p in (data.get("required_stages") or []) if isinstance(p, dict)]

    scored = {r.get("id", f"row[{i}]"): score_stage(r, tree)
              for i, r in enumerate(rows)}

    kpis = run_kpis(rows, phases, scored)
    by_name = {k["kpi"]: k for k in kpis}
    honesty_score = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"]
                              for n in KPI_WEIGHTS if n in by_name), 1)
    honesty_defects = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)

    cov = coverage_report(required, rows)
    popularization_debt = honesty_defects + cov["coverage_debt"]

    cov_pct = cov["coverage_pct"] if cov["required_total"] else 100.0
    score = round(HONESTY_WEIGHT * honesty_score + COVERAGE_WEIGHT * cov_pct, 1)
    grade = grade_letter(score)

    debt_by_group = {g: 0 for g in GROUPS}
    for k in kpis:
        debt_by_group[k["group"]] += len(k["defects"])
    breakdown = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    row_debt = per_row_debt(rows, kpis)
    pos = standing(rows)
    lb = leaderboard(rows, scored)
    crit = critical_backlog(rows, row_debt, scored)
    n_converts = pos.get("converts", 0)

    hard_total = sum(s["hard_total"] for s in scored.values())
    hard_met = sum(s["hard_met"] for s in scored.values())

    corpus = {
        "score": score, "grade": grade,
        "honesty_score": honesty_score,
        "popularization_debt": popularization_debt,
        "honesty_defects": honesty_defects,
        "coverage_debt": cov["coverage_debt"],
        "coverage": cov,
        "soft_signals": n_soft,
        "rows": len(rows),
        "converting_stages": n_converts,
        "hard_affordances": hard_total,
        "hard_affordances_met": hard_met,
        "as_of": meta.get("as_of", ""),
        "fak_version": meta.get("fak_version", ""),
        "standing": pos,
        "debt_by_group": debt_by_group,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
        "leaderboard": lb,
        "critical": crit,
    }

    standing_line = (f"{pos['converts']} convert · {pos['mostly-converts']} mostly · "
                     f"{pos['partially-converts']} partial · {pos['leaks']} leak")
    cov_line = (f"coverage {cov['coverage_pct']}% ({cov['covered']}/{cov['required_total']} "
                f"funnel stages positioned)")
    if popularization_debt == 0:
        ok, verdict, finding = True, "OK", "front_door_converts"
        reason = (f"the front door converts end to end: score {score}/100 (grade {grade}); "
                  f"{cov_line}; zero popularization-debt across {len(kpis)} KPIs over "
                  f"{len(rows)} stages ({hard_met}/{hard_total} hard affordances present; "
                  f"{standing_line}; {n_soft} advisory)")
        next_action = ("hold the line; when a front-door affordance regresses (a removed "
                       "headline number, a dead concept explainer, a broken install "
                       "one-liner) popularization-debt rises — this is the regression "
                       "sentinel; re-run to keep debt at 0")
    elif honesty_defects == 0 and cov["coverage_debt"] > 0:
        ok, verdict, finding = False, "ACTION", "coverage_debt"
        reason = (f"{cov['coverage_debt']} funnel stage(s) not yet positioned; {cov_line}; "
                  f"score {score}/100 (grade {grade}); positioned rows are honest "
                  f"(0 honesty-debt); standing {standing_line}")
        next_action = ("close coverage (see --gaps): add an honest stage row for each "
                       "unpositioned funnel stage; re-run")
    else:
        ok, verdict, finding = False, "ACTION", "popularization_debt"
        worst = breakdown[0]
        reason = (f"{honesty_defects} affordance/honesty defect(s) + {cov['coverage_debt']} "
                  f"coverage gap(s) = popularization-debt {popularization_debt}; score "
                  f"{score}/100 (grade {grade}); heaviest KPI: {worst['kpi']} ({worst['debt']}); "
                  f"{cov_line}; standing {standing_line}")
        next_action = ("retire popularization-debt worst-first (--critical + per-KPI defects): "
                       "ADD the missing affordance for the worst-leaking stage (the real file / "
                       "doc / command / number), align each verdict to its evidence, then close "
                       "coverage (--gaps); re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "kpis": kpis,
        "_data": {"rows": rows, "phases": phase_defs, "required": required},
    }


# ---------------------------------------------------------------------------
# Disk shell — read the modular data DIRECTORY + the tree-facts the checks verify.
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
    """Merge the modular data directory into one document: _meta.json contributes
    meta + phases + required_stages; every other rows-*.json contributes its `rows`."""
    meta_doc, err = _read_json(d / "_meta.json")
    if err:
        return None, err
    if not isinstance(meta_doc, dict):
        return None, "_meta.json is not a JSON object"
    out: dict[str, Any] = {
        "meta": meta_doc.get("meta") or {},
        "phases": meta_doc.get("phases") or [],
        "required_stages": meta_doc.get("required_stages") or [],
        "rows": [],
    }
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("_"):
            continue
        doc, err = _read_json(f)
        if err:
            return None, err
        for r in (doc or {}).get("rows") or []:
            if isinstance(r, dict):
                r.setdefault("_source_file", f.name)
                out["rows"].append(r)
    return out, ""


def load_data(path: Path) -> tuple[dict[str, Any] | None, str]:
    if path.is_dir():
        return load_data_dir(path)
    return None, f"missing data directory: {path}"


def parse_claims_section_tags(text: str) -> dict[str, set[str]]:
    """{norm_section -> SET of every `- [TAG]` tag found under that section}."""
    tags: dict[str, set[str]] = {}
    cur: str | None = None
    tag_re = re.compile(r"^\s*-\s*\[(SHIPPED|SIMULATED|STUB)\]")
    for line in text.splitlines():
        if line.startswith("## "):
            cur = norm_section(line[3:].strip())
        elif cur is not None:
            m = tag_re.match(line)
            if m:
                tags.setdefault(cur, set()).add(m.group(1))
    return tags


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9-]+", text.lower()))


def load_tree(root: Path) -> dict[str, Any]:
    """The real-tree facts the affordance checks cross-check against. Docs are read
    lazily + cached so a `doc_mentions` / `fenced_command` over any path works."""
    claims_text = ""
    cp = root / CLAIMS_REL
    if cp.exists():
        try:
            claims_text = cp.read_text(encoding="utf-8")
        except OSError:
            claims_text = ""
    section_tags = parse_claims_section_tags(claims_text)

    cmd_dirs: set[str] = set()
    cmd_root = root / "cmd"
    if cmd_root.is_dir():
        cmd_dirs = {p.name for p in cmd_root.iterdir() if p.is_dir()}

    doc_verbs: set[str] = set()
    cr = root / CLI_REF_REL
    if cr.exists():
        try:
            doc_verbs = _word_set(cr.read_text(encoding="utf-8"))
        except OSError:
            doc_verbs = set()

    _cache: dict[str, str] = {}

    def doc_text(rel: str) -> str:
        if rel not in _cache:
            p = root / rel
            try:
                _cache[rel] = p.read_text(encoding="utf-8") if p.is_file() else ""
            except OSError:
                _cache[rel] = ""
        return _cache[rel]

    return {
        "exists": lambda p: bool(p) and (root / p).exists(),
        "doc_text": doc_text,
        "cmd_dirs": cmd_dirs,
        "doc_verbs": doc_verbs,
        "section_tags": section_tags,
    }


def collect(workspace: Path, *, data_path: Path | None = None) -> dict[str, Any]:
    root = workspace.resolve()
    path = data_path or (root / DATA_DIR_REL)
    data, err = load_data(path)
    tree = load_tree(root)
    return build_payload(workspace=str(root), data=data, tree=tree, error=err or None)


# ---------------------------------------------------------------------------
# Renderers — terminal, critical backlog, coverage gaps, compare, doc folder.
# ---------------------------------------------------------------------------

_MARK = {"converts": "★", "mostly-converts": "●", "partially-converts": "◐", "leaks": "○"}


def _bar(n: int, scale: int, width: int = 28, *, fill: str = "█", empty: str = "·") -> str:
    if scale <= 0:
        return empty * width
    cells = int(round(width * max(0, n) / scale))
    cells = max(0, min(width, cells))
    if n > 0 and cells == 0:
        cells = 1
    return fill * cells + empty * (width - cells)


def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    pos = c.get("standing") or {}
    lines = [
        f"popularization-readiness-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· POPULARIZATION-DEBT {c.get('popularization_debt', 0)} "
         f"(affordance/honesty {c.get('honesty_defects', 0)} + coverage {c.get('coverage_debt', 0)}) "
         f"· {c.get('soft_signals', 0)} advisory"),
        (f"coverage: {cov.get('coverage_pct', 0)}% "
         f"({cov.get('covered', 0)}/{cov.get('required_total', 0)} funnel stages positioned) "
         f"· {c.get('hard_affordances_met', 0)}/{c.get('hard_affordances', 0)} hard affordances present"),
        (f"standing: {pos.get('converts', 0)} convert · {pos.get('mostly-converts', 0)} mostly-convert · "
         f"{pos.get('partially-converts', 0)} partially-convert · {pos.get('leaks', 0)} leak"),
        ("debt by group: " + "  ".join(
            f"{g}:{(c.get('debt_by_group') or {}).get(g, 0)}" for g in GROUPS)),
        "",
        "funnel stages (best-converting first):",
        f"  {'verdict':<18} {'phase':<9} {'effort':<7} {'met':<7} stage",
    ]
    for row in sorted(c.get("leaderboard", []),
                      key=lambda x: (VERDICT_RANK.get(x["verdict"], 9), x.get("phase") or "")):
        mark = _MARK.get(row["verdict"], " ")
        met = f"{row.get('hard_met', 0)}/{row.get('hard_total', 0)}"
        flag = "" if row["verdict"] == row["expected_verdict"] else f"  ⚠ expected {row['expected_verdict']}"
        lines.append(f"  {mark} {row['verdict']:<16} {str(row.get('phase')):<9} "
                     f"{str(row.get('effort')):<7} {met:<7} {row.get('stage')}{flag}")
    lines += ["", "per-KPI (worst first):",
              f"  {'score':>5} {'debt':>4}  {'group':<11} {'kpi':<22} detail"]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['group']:<11} "
                     f"{b['kpi']:<22} {b['detail']}")
    lines.append("")
    if cov.get("uncovered"):
        lines.append(f"coverage gaps ({len(cov['uncovered'])} funnel stages with no row):")
        for p in cov["uncovered"]:
            lines.append(f"      - {p.get('id')} ({p.get('name', '?')})")
        lines.append("")
    lines.append("popularization-debt work-list:")
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
        lines.append("  (none — zero popularization-debt; the front door converts end to end)")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_critical(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    lines = ["popularization critical backlog (fix worst-leaking first):", ""]
    crit = c.get("critical", [])
    shown = False
    for it in crit:
        if it["debt"] == 0 and it["distance"] == 0:
            continue
        shown = True
        miss = (", ".join(it["missing"][:5]) or "—")
        lines.append(f"  [{it['debt']} debt · {it['verdict']}] {it['id']} "
                     f"({it['phase']}) — missing: {miss}")
        for g in (it.get("gaps") or [])[:3]:
            lines.append(f"      · {g}")
    if not shown:
        lines.append("  (none — every funnel stage converts; no critical backlog)")
    lines.append("")
    lines.append("(converting stages with 0 debt are omitted; a stage with a missing "
                 "affordance is shown because adding it is the work.)")
    return "\n".join(lines)


def render_gaps(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    lines = ["popularization coverage backlog (position every funnel stage):", ""]
    unc = cov.get("uncovered", [])
    lines.append(f"UNPOSITIONED — {len(unc)} required stage(s) with no row:")
    if not unc:
        lines.append("  (none — every required funnel stage is positioned)")
    for p in unc:
        lines.append(f"  - {p.get('id')} ({p.get('name', '?')}, phase {p.get('phase', '?')})")
    return "\n".join(lines)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("popularization_debt", 0), cur.get("popularization_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ratio = "∞ (zero)" if cd == 0 else f"{bd / cd:.1f}×"
    lines = [
        f"popularization-debt: {bd} -> {cd}   ({ratio} fewer defects+gaps)",
        f"  affordance/honesty: {b.get('honesty_defects', 0)} -> {cur.get('honesty_defects', 0)}",
        f"  coverage:           {b.get('coverage_debt', 0)} -> {cur.get('coverage_debt', 0)}",
        f"score:        {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
        f"converting:   {b.get('converting_stages', 0)} -> {cur.get('converting_stages', 0)} stages converting",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<11} {gb} -> {gc}")
    target3 = max(0, (bd + 2) // 3)
    target2 = max(0, bd // 2)
    if cd <= target3:
        lines.append(f"VERDICT: ≥3× reduction achieved (popularization-debt {bd}->{cd}, target ≤{target3}).")
    elif cd <= target2:
        lines.append(f"VERDICT: ≥2× (not yet 3×) — popularization-debt {bd}->{cd}; 3× needs ≤{target3}.")
    else:
        lines.append(f"VERDICT: not yet 2× — need popularization-debt ≤{target2} (now {cd}); 3× target ≤{target3}.")
    return "\n".join(lines)


def render_chart(payload: dict[str, Any]) -> str:
    """An at-a-glance ASCII chart of how well fak's front door converts — what a
    person sees first. Deterministic + pure text: two clones at one commit chart
    identically."""
    c = payload.get("corpus") or {}
    pos = c.get("standing") or {}
    cov = c.get("coverage") or {}
    lb = c.get("leaderboard") or []

    lines: list[str] = [
        (f"popularization-readiness chart — {c.get('rows', 0)} funnel stages · "
         f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) · "
         f"popularization-debt {c.get('popularization_debt', 0)}"),
        "",
        "verdict ladder (count of stages, best -> worst):",
    ]
    maxn = max((pos.get(v, 0) for v in VERDICTS), default=0)
    for v in VERDICTS:
        n = pos.get(v, 0)
        lines.append(f"  {_MARK.get(v, ' ')} {v:<19} {_bar(n, maxn)} {n}")
    lines.append("")

    # the funnel, in order (the conversion path a visitor walks).
    lines.append("the funnel (in order — where a visitor lands, and where they leak):")
    for r in sorted(lb, key=lambda x: (x.get("phase") or "")):
        ht = r.get("hard_total", 0)
        hm = r.get("hard_met", 0)
        lines.append(f"  {_MARK.get(r['verdict'], ' ')} {str(r.get('phase')):<9} "
                     f"{_bar(hm, ht, width=20)} {hm}/{ht}  {r.get('stage')}")
    lines.append("")

    pct = cov.get("coverage_pct", 0.0)
    gauge = _bar(int(round(pct)), 100, width=32)
    lines.append(f"coverage  [{gauge}] {pct}%  "
                 f"({cov.get('covered', 0)}/{cov.get('required_total', 0)} funnel stages positioned)")
    lines.append("")
    lines.append("legend: " + "   ".join(f"{_MARK[v]} {v}" for v in VERDICTS))
    return "\n".join(lines)


def _front_matter(title: str, desc: str) -> list[str]:
    return ["---", f'title: "{title}"', f'description: "{desc}"', "---", ""]


def render_doc_index(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    out = _front_matter(
        "fak popularization-readiness scorecard — does the front door convert?",
        "Inward popularization scorecard (concept-popularization epic, dimension K): fak's "
        "front door modeled as a conversion funnel (land → orient → trust → install → act), "
        "each stage positioned on whether the affordances a first-time visitor reaches for "
        "actually exist in the tree, with one honest conversion verdict per stage. Two driven "
        "numbers: coverage (of the funnel) and popularization-debt.")
    out.append("# Popularization-readiness scorecard — does the front door convert?")
    out.append("")
    if stamp:
        out.append(f"<!-- popularization-readiness-scorecard: {stamp} · process: tools/popularization_readiness_scorecard.py · "
                   f"data: tools/popularization_readiness_scorecard.data/ -->")
        out.append("")
    out.append("The sibling scorecards grade fak through one lens each: "
               "[`agent-readiness`](../AGENT-READINESS-SCORECARD.md) asks whether an AI *agent* "
               "can adopt fak, [`persona-readiness`](../persona-scorecard/README.md) whether each "
               "of the top-10 humans who land on fak is served. This one asks the "
               "**popularization** question the concept-popularization epic exists to answer "
               "(dimension K, adoption measurement): when a first-time visitor lands on fak's "
               "front door — the README, the five-concepts framing, the install one-liner, the "
               "social-proof surfaces, the first runnable command — **does that door actually "
               "convert them, stage by stage, or does it leak them?** The door is modeled as an "
               "ordered conversion funnel (land → orient → trust → install → act); every number "
               "below is re-derived from `tools/popularization_readiness_scorecard.data/` by "
               "`tools/popularization_readiness_scorecard.py` and cross-checked against the real "
               "tree, so no verdict is hand-typed: to stop a stage leaking you ADD the real "
               "affordance.")
    out.append("")
    out.append("> Regenerate: `python tools/popularization_readiness_scorecard.py --markdown-dir docs/popularization-scorecard`.")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Coverage** | **{cov.get('coverage_pct', 0)}%** "
               f"({cov.get('covered', 0)}/{cov.get('required_total', 0)} funnel stages positioned) |")
    out.append(f"| **Popularization-debt** | **{c.get('popularization_debt', 0)}** "
               f"(affordance/honesty {c.get('honesty_defects', 0)} + coverage {c.get('coverage_debt', 0)}) |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append(f"| Stages converting | {c.get('converting_stages', 0)} of {c.get('rows', 0)} |")
    out.append(f"| Hard affordances present | {c.get('hard_affordances_met', 0)} / {c.get('hard_affordances', 0)} |")
    out.append(f"| As of | {c.get('as_of', '?')} (fak {c.get('fak_version', '?')}) |")
    out.append("")
    out.append("> **Read this right.** The score grades how *complete and honest the funnel map "
               "is* — every declared stage positioned, every conversion verdict matching the "
               "affordances actually on disk. A missing affordance is a real leak you ADD; an "
               "*overclaimed* verdict is the defect this catches.")
    out.append("")
    out.append("## The funnel at a glance")
    out.append("")
    out.append("> Regenerate this chart in the terminal with `python tools/popularization_readiness_scorecard.py --chart`.")
    out.append("")
    out.append("```text")
    out.append(render_chart(payload))
    out.append("```")
    out.append("")
    out.append("## The conversion ladder")
    out.append("")
    out.append("| Verdict | Means |")
    out.append("|---|---|")
    out.append("| ★ converts | every hard affordance this stage needs is present — the visitor advances |")
    out.append("| ● mostly-converts | ≥ 75% of the hard affordances present — a small leak |")
    out.append("| ◐ partially-converts | ≥ 40% present — the stage is half-built |")
    out.append("| ○ leaks | < 40% present — visitors drop here |")
    out.append("")
    out.append("## The funnel stages (best-converting first)")
    out.append("")
    out.append("| | Verdict | Phase | Effort | Affordances | Stage — the job that must happen to advance |")
    out.append("|---|---|---|---|---|---|")
    for row in sorted(c.get("leaderboard", []),
                      key=lambda x: (VERDICT_RANK.get(x["verdict"], 9), x.get("phase") or "")):
        mark = _MARK.get(row["verdict"], " ")
        met = f"{row.get('hard_met', 0)}/{row.get('hard_total', 0)}"
        out.append(f"| {mark} | {row['verdict']} | {row.get('phase')} | {row.get('effort')} | "
                   f"{met} | **{row.get('stage')}** — {row.get('job')} |")
    out.append("")
    out.append("## Per-KPI (popularization-debt = affordance/honesty of the stages that exist)")
    out.append("")
    out.append("| Group | KPI | Score | Debt | Detail |")
    out.append("|---|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    if cov.get("uncovered"):
        out.append("## Coverage gaps (funnel stages not yet positioned)")
        out.append("")
        for p in cov["uncovered"]:
            out.append(f"- **{p.get('id')}** ({p.get('name', '?')}, phase {p.get('phase', '?')})")
        out.append("")
    crit = [x for x in c.get("critical", []) if x["debt"] > 0 or x["distance"] > 0]
    if crit:
        out.append("## Worst-leaking stages (fix worst-first)")
        out.append("")
        for it in crit[:12]:
            miss = ", ".join(it["missing"][:5]) or "—"
            out.append(f"- **{it['id']}** ({it['verdict']}, {it['debt']} debt) — missing: {miss}")
        out.append("")
    return "\n".join(out)


def render_doc_folder(payload: dict[str, Any], *, stamp: str | None = None) -> dict[str, str]:
    return {"README.md": render_doc_index(payload, stamp=stamp)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Popularization-readiness scorecard (read-only unless --markdown-dir).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--chart", action="store_true", help="an at-a-glance ASCII chart of the funnel")
    ap.add_argument("--critical", action="store_true", help="the worst-leaking stages backlog")
    ap.add_argument("--gaps", action="store_true", help="the coverage backlog (unpositioned stages)")
    ap.add_argument("--compare", default="", help="baseline JSON to prove popularization-debt dropped")
    ap.add_argument("--markdown-dir", default="", help=f"regenerate the doc folder (e.g. {GENERATED_DOC_DIR})")
    ap.add_argument("--data", default="", help=f"data directory (default: {DATA_DIR_REL})")
    ap.add_argument("--stamp", default="", help="optional stamp embedded in the generated doc")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    data_path = Path(args.data).resolve() if args.data else None
    payload = collect(root, data_path=data_path)

    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"cannot read baseline {args.compare}: {exc}", file=sys.stderr)
            return 2
        print(render_compare(baseline, payload))
        return 0 if payload.get("ok") else 1

    if args.markdown_dir:
        out_dir = Path(args.markdown_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        for rel, content in render_doc_folder(payload, stamp=args.stamp or None).items():
            (out_dir / rel).write_text(content + "\n", encoding="utf-8")
        if not args.json:
            print(f"wrote popularization-readiness doc folder -> {out_dir}")

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.chart:
        print(render_chart(payload))
    elif args.critical:
        print(render_critical(payload))
    elif args.gaps:
        print(render_gaps(payload))
    elif not args.markdown_dir:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
