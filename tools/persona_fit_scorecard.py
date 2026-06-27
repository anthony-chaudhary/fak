#!/usr/bin/env python3
"""Persona-fit scorecard — across each feature space, how much would each persona like fak?

The sibling ``persona_readiness`` scorecard asks ONE question of the top-10 personas
who land on fak: is each one's *entry path* served — does the first affordance that
person reaches for exist in the tree? It is a yes/no readiness gate per persona.

This scorecard asks the orthogonal question — the one a go-to-market person draws on
a 2x2 — **a matrix, not a gate**:

  Of the KEY FEATURE SPACES fak ships (the security floor, performance/reuse, agent
  memory, the in-kernel model, agent tooling, platform/adoption), how much would each
  persona LIKE each one? How does an *engineer* feel about the model internals vs how a
  *product manager* (the evaluator/decision-maker) feels about the benchmarks vs how a
  *researcher* feels about the reproducibility? The output is a persona x feature-space
  FIT MATRIX with one 0-100 cell per (persona, feature), each row's "what this persona
  loves most", and each column's "who this feature wins."

The fit of a feature space for a persona is NOT a hand-typed opinion. It is computed,
and grounded so it cannot be gamed by editing data:

  * Each persona carries a WEIGHT vector over a fixed set of VALUE DIMENSIONS — what
    that person actually values across any feature (does it run today? is it proven? is
    it measured/observable? operable? trustworthy? extensible? documented? benchmarked
    vs the field? efficient — cheap on tokens/compute?). A free-tier dev weights
    ``runs_today`` + ``efficient`` heavily; a security engineer weights ``trustworthy``;
    the decision-maker (≈ a PM) weights ``benchmarked`` + ``measured`` + ``efficient``.
  * Each feature space carries a DELIVERY vector over the SAME dimensions — and every
    unit of delivery names GROUNDING CHECKS against the real tree (a path that exists, a
    command that resolves, a CLAIMS.md section that is tagged). Delivery DEPTH is graded
    by how many DISTINCT checks resolve (capped), and each check must be TOPICALLY
    RELEVANT to the dimension it witnesses (a ``benchmarked`` witness names a benchmark, a
    ``measured`` witness names a metric/observability surface) — so a feature cannot inflate
    a dimension by pointing a resolving-but-unrelated check at it.
  * The cell fit = the weighted match (persona weights . feature delivery), 0-100.

So the matrix is a function of two things that are each tied to reality: the persona's
declared priorities and the feature's tree-witnessed delivery. To raise a cell you make
the feature actually deliver the dimension that persona values (ship the offline
command, write the witness, expose the metric) — never by editing the score.

Two numbers are driven, mirroring the persona-readiness / product scorecards:

  PERSONA-FIT-DEBT  the INTEGRITY of the matrix (NOT a low cell — a low fit is honest):
                    malformed rows + off-topic delivery evidence (a grounding check not
                    relevant to its dimension) + ungrounded delivery evidence (a check
                    that does not resolve in the tree) + dishonest self-reports (a
                    persona/feature whose declared "loves most" / "wins" disagrees with the
                    computed matrix) + coverage gaps (a roster persona with no weights, a
                    feature space with no delivery row). Drive it to 0
                    — then the matrix is complete and every cell is honestly grounded, and
                    the number becomes a regression sentinel. Folds into
                    ``scorecard_control_pane`` via ``corpus.persona_fit_debt``.
  COVERAGE          of the (personas x feature-spaces) grid, how many cells are positioned
                    (a persona has weights AND the feature has a delivery row).

Deterministic + read-only over the data (two clones at one commit score identically);
the only disk writes are the generated doc folder under ``--markdown-dir``. The source
of truth is a DIRECTORY of small JSON files so the persona roster, the value-dimension
weights, and each feature space's delivery evolve independently::

    tools/persona_fit_scorecard.data/
      _meta.json        meta + value-dimension doctrine + the persona roster w/ weights
                        + the declared feature-space columns
      rows-*.json       feature-space delivery rows (grounded per dimension)

Run from the repo ROOT::

    python tools/persona_fit_scorecard.py                  # the fit matrix + debt
    python tools/persona_fit_scorecard.py --matrix         # just the persona x feature grid
    python tools/persona_fit_scorecard.py --persona ml-researcher   # one persona's column read
    python tools/persona_fit_scorecard.py --feature security        # one feature's persona read
    python tools/persona_fit_scorecard.py --chart          # at-a-glance ASCII chart
    python tools/persona_fit_scorecard.py --json           # machine payload (control-pane / loop)
    python tools/persona_fit_scorecard.py --compare base.json       # prove persona-fit-debt dropped
    python tools/persona_fit_scorecard.py --markdown-dir docs/persona-fit-scorecard
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

SCHEMA = "fak-persona-fit-scorecard/1"
DATA_DIR_REL = "tools/persona_fit_scorecard.data"
GENERATED_DOC_DIR = "docs/persona-fit-scorecard"
CLAIMS_REL = "CLAIMS.md"
CLI_REF_REL = "docs/cli-reference.md"

# ---------------------------------------------------------------------------
# Doctrine: the fixed VALUE DIMENSIONS. Both a persona's weights and a feature
# space's delivery are vectors over THIS closed set, so the dot product is
# well-defined. A row naming a dimension outside this set is malformed.
# Order is the canonical display order (left -> right in the matrix legend).
# ---------------------------------------------------------------------------
VALUE_DIMENSIONS: list[dict[str, str]] = [
    {"id": "runs_today", "label": "runs today", "asks": "can I run it now, ideally offline on a laptop?"},
    {"id": "proven", "label": "proven", "asks": "is it backed by a witness / results / test that exists?"},
    {"id": "measured", "label": "measured", "asks": "does it expose metrics / observability I can watch?"},
    {"id": "operable", "label": "operable", "asks": "is there a deploy / config / ops surface to run it?"},
    {"id": "trustworthy", "label": "trustworthy", "asks": "does it have a security / refusal / determinism floor?"},
    {"id": "extensible", "label": "extensible", "asks": "can I build on it (frozen ABI, seams, a leaf)?"},
    {"id": "documented", "label": "documented", "asks": "is there an entry doc where a person learns it?"},
    {"id": "benchmarked", "label": "benchmarked", "asks": "is it measured against the field / SOTA?"},
    {"id": "efficient", "label": "efficient", "asks": "is it cheap to run — does it save tokens / compute / cache?"},
]
DIM_IDS = [d["id"] for d in VALUE_DIMENSIONS]
DIM_LABEL = {d["id"]: d["label"] for d in VALUE_DIMENSIONS}
DIM_SET = set(DIM_IDS)

# Dimension RELEVANCE — the ungameability backstop. A grounding check does not just have
# to RESOLVE in the tree (evidence_grounded); the tree fact it points at must be TOPICALLY
# RELEVANT to the dimension it claims to witness. Otherwise a feature's delivery (and thus
# the fit of every persona who values that dimension) could be inflated by pointing a
# resolving-but-unrelated check at it — e.g. grounding `benchmarked` with a security
# package. Each dimension declares the check KINDS it accepts (empty = any) and a set of
# subject TOKENS, at least one of which must appear in the check's subject (the path /
# command / doc / section it names — NOT the doc body). A check that grounds a dimension it
# is not registered for is hard debt and earns no delivery depth.
DIM_RELEVANCE: dict[str, dict[str, Any]] = {
    # runs_today is about runnability, so it must be a runnable command, of any topic.
    "runs_today": {"kinds": {"command_resolves", "fenced_command"}, "tokens": []},
    # proven = a witness exists: an implementation package, a test, a results/proof doc.
    "proven": {"kinds": {"path_exists", "any_path_exists", "claim_section"},
               "tokens": ["internal/", "test", "results", "benchmarks", "proof", "witness",
                          "experiments/", "cmd/"]},
    "measured": {"tokens": ["metric", "observ", "trajectory", "prometheus", "telemetry",
                            "rungobs", "gauge", "counter"]},
    "operable": {"tokens": ["config", "serve", "deploy", "ops", "polic", "gpu", "runbook",
                            "getting-started", "install"]},
    "trustworthy": {"tokens": ["polic", "adjudicat", "guard", "quarantine", "ctxmmu", "normgate",
                               "ifc", "wirescreen", "kvmmu", "secur", "capabilit", "preflight",
                               "codelint", "gate", "context-mmu", "refus"]},
    "extensible": {"tokens": ["extend", "architecture", "abi", "seam", "xengine", "register",
                              "leaf", "plugin", "integrat"]},
    "documented": {"tokens": [".md", "docs/", "readme", "reference", "guide", "getting-started",
                              "claims"]},
    "benchmarked": {"tokens": ["bench", "ablate", "turntax", "fanout", "fanbench", "longctx",
                               "rsiloop", "a2ademo", "ctxplan", "vcache", "regime", "sweep", "tau2"]},
    "efficient": {"tokens": ["vdso", "vcache", "token", "ablate", "prefill", "kv-reuse", "kvreuse",
                             "reuse", "turntax", "tax", "cache", "radixkv", "kvmmu", "quant"]},
}


def check_subject(chk: dict[str, Any]) -> str:
    """What a check POINTS at — the path / command / doc / section / tokens it names,
    lowercased. This is the identity relevance is judged on (NOT the doc body, which a
    feature does not control)."""
    parts = [
        str(chk.get("target", "")),
        " ".join(str(t) for t in (chk.get("targets") or [])),
        str(chk.get("command", "")),
        str(chk.get("doc", "")),
        " ".join(str(d) for d in (chk.get("docs") or [])),
        str(chk.get("section", "")),
        " ".join(str(t) for t in (chk.get("tokens") or [])),
    ]
    return " ".join(p for p in parts if p).lower()


def check_relevant(chk: dict[str, Any], dim: str) -> bool:
    """Is this grounding check topically relevant to the dimension it claims to witness?"""
    rel = DIM_RELEVANCE.get(dim)
    if not rel:
        return False
    kinds = rel.get("kinds")
    if kinds and chk.get("kind") not in kinds:
        return False
    tokens = rel.get("tokens") or []
    if not tokens:
        return True
    subj = check_subject(chk)
    return any(tok in subj for tok in tokens)

# Delivery DEPTH: a feature's delivery on a dimension is graded by how many DISTINCT
# grounding checks resolve, capped here. One resolving witness = a touch (~0.33); three
# = full delivery (1.0). Depth comes ONLY from real tree evidence, so a feature that
# delivers a dimension deeply (a package + a command + a tagged claim) honestly outscores
# one that merely touches it — and the grid discriminates instead of saturating at 100.
DEPTH_CAP = 3

# The grounding check kinds — each a pure predicate over real-tree facts. Same
# doctrine as persona_readiness so a reader who knows one knows both.
CHECK_KINDS = {
    "path_exists": "a named file / dir exists in the tree",
    "any_path_exists": "any of several candidate paths exists",
    "doc_mentions": "a doc exists and contains the token(s)",
    "fenced_command": "a copy-pasteable command lives in a fenced block of a doc",
    "command_resolves": "a `go run ./cmd/<dir> <verb>` resolves (dir + documented verb)",
    "claim_section": "a CLAIMS.md `##` concept section exists (optionally tagged)",
}

GROUPS = ("well-formed", "reality", "honesty")
KPI_GROUP: dict[str, str] = {
    "rows_well_formed": "well-formed",
    "dimension_relevant": "reality",
    "evidence_grounded": "reality",
    "fit_honest": "honesty",
}
KPI_WEIGHTS: dict[str, float] = {
    "rows_well_formed": 0.12,
    "dimension_relevant": 0.25,
    "evidence_grounded": 0.38,
    "fit_honest": 0.25,
}
KPI_PENALTY: dict[str, int] = {
    "rows_well_formed": 10,
    "dimension_relevant": 15,
    "fit_honest": 20,
}
# The composite blends the honesty/grounding of the rows that EXIST with how much
# of the (persona x feature) grid is even positioned. An incomplete grid costs grade.
HONESTY_WEIGHT = 0.65
COVERAGE_WEIGHT = 0.35

PERSONA_FIELDS = ("id", "persona", "weights", "top_feature")
FEATURE_FIELDS = ("id", "feature", "group", "what", "delivery", "top_persona")
DELIVERY_FIELDS = ("dim", "checks")


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
    """Normalize a `## header` / section to a comparable key."""
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
    a, b = norm_section(want), catalog_norm
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 6 and len(b) >= 6 and (a in b or b in a):
        return True
    return False


def parse_command(cmd: Any) -> tuple[str | None, str | None]:
    """Pull (cmd_dir, verb) out of a `go run ./cmd/<dir> <verb> ...` command."""
    if not _nonempty(cmd):
        return None, None
    m = re.search(r"\./cmd/([\w-]+)", cmd)
    if not m:
        return None, None
    cmd_dir = m.group(1)
    rest = cmd[m.end():].strip()
    for tok in rest.split():
        if tok.startswith("-"):
            break
        return cmd_dir, tok
    return cmd_dir, None


_FENCE_RE = re.compile(r"^(```|~~~)")


def fenced_blocks(text: str) -> list[str]:
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


def check_signature(chk: dict[str, Any]) -> str:
    """A stable identity for a grounding check, so two checks that witness the SAME tree
    fact (the same path / command / doc-tokens / section) count once toward depth."""
    kind = chk.get("kind")
    if kind == "path_exists":
        return f"path_exists:{chk.get('target', '')}"
    if kind == "any_path_exists":
        return "any_path_exists:" + ",".join(sorted(str(t) for t in (chk.get("targets") or [])))
    if kind == "doc_mentions":
        return (f"doc_mentions:{chk.get('doc', '')}:"
                + ",".join(sorted(str(t).lower() for t in (chk.get("tokens") or []))))
    if kind == "fenced_command":
        return ("fenced_command:" + ",".join(sorted(str(d) for d in (chk.get("docs") or [])))
                + ":" + ",".join(sorted(str(t).lower() for t in (chk.get("tokens") or []))))
    if kind == "command_resolves":
        return f"command_resolves:{chk.get('command', '')}"
    if kind == "claim_section":
        return (f"claim_section:{norm_section(chk.get('section', ''))}:"
                + ",".join(sorted(str(t) for t in (chk.get("tags") or []))))
    return f"{kind}:{json.dumps(chk, sort_keys=True)}"


def normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    """Project a persona's raw nonneg weights onto the dimension vocabulary and
    normalize to sum 1. Unknown keys are dropped; missing dims are 0. A zero/empty
    vector normalizes to all-zero (the honesty KPI flags it separately)."""
    vec = {d: 0.0 for d in DIM_IDS}
    if isinstance(weights, dict):
        for k, v in weights.items():
            if k in vec and isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                vec[k] = float(v)
    total = sum(vec.values())
    if total <= 0:
        return vec
    return {d: vec[d] / total for d in DIM_IDS}


# ---------------------------------------------------------------------------
# Grounding evaluation — the pure predicate over tree facts. Returns (met, detail).
# ---------------------------------------------------------------------------

def eval_check(chk: dict[str, Any], tree: dict[str, Any]) -> tuple[bool, str]:
    kind = chk.get("kind")
    exists: Callable[[str], bool] = tree.get("exists") or (lambda p: False)
    doc_text: Callable[[str], str] = tree.get("doc_text") or (lambda d: "")

    if kind == "path_exists":
        t = chk.get("target", "")
        if not _nonempty(t):
            return False, "path_exists with no target"
        ok = exists(t)
        return ok, (f"{t} exists" if ok else f"{t} missing")

    if kind == "any_path_exists":
        targets = [t for t in (chk.get("targets") or []) if _nonempty(t)]
        hit = next((t for t in targets if exists(t)), None)
        return (hit is not None), (f"found {hit}" if hit else f"none exist: {targets}")

    if kind == "doc_mentions":
        doc = chk.get("doc", "")
        tokens = [t for t in (chk.get("tokens") or []) if _nonempty(t)]
        match = chk.get("match", "any")
        text = doc_text(doc)
        if not text:
            return False, f"{doc} missing/empty"
        low = text.lower()
        present = [t for t in tokens if t.lower() in low]
        ok = bool(present) and (len(present) == len(tokens) if match == "all" else True)
        return ok, (f"{doc} mentions {present}" if ok else f"{doc} lacks {tokens} (match={match})")

    if kind == "fenced_command":
        docs = [d for d in (chk.get("docs") or []) if _nonempty(d)]
        tokens = [t for t in (chk.get("tokens") or []) if _nonempty(t)]
        for d in docs:
            for block in fenced_blocks(doc_text(d)):
                low = block.lower()
                if any(t.lower() in low for t in tokens):
                    return True, f"fenced command in {d}"
        return False, f"no fenced block with {tokens} in {docs}"

    if kind == "command_resolves":
        cmd = chk.get("command", "")
        cmd_dir, verb = parse_command(cmd)
        if cmd_dir is None:
            return False, f"{cmd!r} has no ./cmd/<dir>"
        if cmd_dir not in (tree.get("cmd_dirs") or set()):
            return False, f"./cmd/{cmd_dir} does not exist"
        if cmd_dir == "fak" and verb and verb.lower() not in (tree.get("doc_verbs") or set()):
            return False, f"fak verb {verb!r} not documented in cli-reference"
        return True, f"resolves: ./cmd/{cmd_dir} {verb or ''}".rstrip()

    if kind == "claim_section":
        section = chk.get("section", "")
        tags = {t for t in (chk.get("tags") or []) if _nonempty(t)}
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
# Feature delivery — fold a feature's per-dimension grounding checks into a
# delivery vector over the dimensions (each entry the fraction of checks met).
# ---------------------------------------------------------------------------

def feature_delivery(row: dict[str, Any], tree: dict[str, Any]) -> dict[str, Any]:
    vec = {d: 0.0 for d in DIM_IDS}
    detail: dict[str, dict[str, Any]] = {}
    ground_defects: list[str] = []
    relevance_defects: list[str] = []
    reuse_soft: list[str] = []
    seen_row: dict[str, str] = {}   # witness signature -> the dim that first claimed it
    fid = row.get("id", "?")
    for entry in (row.get("delivery") or []):
        if not isinstance(entry, dict):
            continue
        dim = entry.get("dim")
        if dim not in DIM_SET:
            continue
        checks = [c for c in (entry.get("checks") or []) if isinstance(c, dict)]
        met = 0
        results: list[dict[str, Any]] = []
        for c in checks:
            # Gate 1 — relevance: the tree fact this check names must be topically
            # relevant to the dimension, else it cannot witness it (no depth, hard debt).
            if not check_relevant(c, dim):
                relevance_defects.append(
                    f"{fid}/{dim}: grounding check {check_signature(c)} is not relevant to "
                    f"'{dim}' — a {dim} witness must name a {dim}-related fact")
                results.append({"ok": False, "relevant": False, "detail": "irrelevant to dimension"})
                continue
            # Gate 2 — resolution: a relevant check must resolve in the tree, else phantom.
            ok, d = eval_check(c, tree)
            results.append({"ok": ok, "relevant": True, "detail": d})
            if ok:
                met += 1
                sig = check_signature(c)
                if sig in seen_row:
                    reuse_soft.append(f"{fid}: witness {sig} grounds both '{seen_row[sig]}' and "
                                      f"'{dim}' — a dual-purpose fact counts toward both")
                else:
                    seen_row[sig] = dim
            else:
                ground_defects.append(f"{fid}/{dim}: grounding check did not resolve — {d}")
        # Delivery depth: graded by how many RELEVANT checks RESOLVE (capped). An irrelevant
        # or phantom check earns no depth; both are charged as hard debt above.
        frac = min(1.0, met / DEPTH_CAP) if DEPTH_CAP > 0 else 0.0
        vec[dim] = round(frac, 3)
        detail[dim] = {"met": met, "total": len(checks), "frac": round(frac, 3), "results": results}
    return {"vec": vec, "detail": detail, "ground_defects": ground_defects,
            "relevance_defects": relevance_defects, "reuse_soft": reuse_soft}


# ---------------------------------------------------------------------------
# The matrix — fit_{persona,feature} = normalized weights . delivery, 0-100.
# ---------------------------------------------------------------------------

def cell_fit(weights: dict[str, float], delivery: dict[str, float]) -> int:
    return _clamp(100.0 * sum(weights.get(d, 0.0) * delivery.get(d, 0.0) for d in DIM_IDS))


def argmax_id(scores: dict[str, int]) -> str | None:
    """The id with the max score; ties broken by id order for determinism. None when
    every score is zero (no honest 'best')."""
    if not scores:
        return None
    best = max(scores.values())
    if best <= 0:
        return None
    for k in sorted(scores):
        if scores[k] == best:
            return k
    return None


def build_matrix(personas: list[dict[str, Any]], features: list[dict[str, Any]],
                 deliveries: dict[str, dict[str, float]]) -> dict[str, Any]:
    """The full persona x feature fit grid + per-row / per-column reads."""
    weight_vecs = {p["id"]: normalize_weights(p.get("weights")) for p in personas if _nonempty(p.get("id"))}
    cells: dict[str, dict[str, int]] = {}
    for p in personas:
        pid = p.get("id")
        if not _nonempty(pid):
            continue
        w = weight_vecs.get(pid, {})
        cells[pid] = {}
        for f in features:
            fid = f.get("id")
            if not _nonempty(fid):
                continue
            cells[pid][fid] = cell_fit(w, deliveries.get(fid, {}))
    # per-persona read: how much this persona likes each feature + their favourite.
    persona_read: dict[str, dict[str, Any]] = {}
    for pid, row in cells.items():
        fav = argmax_id(row)
        persona_read[pid] = {
            "row": row,
            "mean": round(sum(row.values()) / len(row), 1) if row else 0,
            "top_feature": fav,
            "top_fit": row.get(fav, 0) if fav else 0,
        }
    # per-feature read: who this feature wins + how broadly it appeals.
    feature_read: dict[str, dict[str, Any]] = {}
    for f in features:
        fid = f.get("id")
        if not _nonempty(fid):
            continue
        col = {pid: cells[pid][fid] for pid in cells}
        champ = argmax_id(col)
        feature_read[fid] = {
            "col": col,
            "mean": round(sum(col.values()) / len(col), 1) if col else 0,
            "top_persona": champ,
            "top_fit": col.get(champ, 0) if champ else 0,
        }
    return {"cells": cells, "persona_read": persona_read, "feature_read": feature_read,
            "weight_vecs": weight_vecs}


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each returns
#   {kpi, group, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of persona-fit-debt; soft = score-only nudges.
# ---------------------------------------------------------------------------

def _kpi(name: str, score: int, defects: list[str], detail: str,
         *, soft: list[str] | None = None) -> dict[str, Any]:
    return {"kpi": name, "group": KPI_GROUP[name], "score": _clamp(score),
            "detail": detail, "defects": defects, "soft": soft or []}


def kpi_rows_well_formed(personas: list[dict[str, Any]], features: list[dict[str, Any]],
                         groups: set[str]) -> dict[str, Any]:
    """Personas and feature rows must be shaped like positions: required fields present,
    ids unique, weights nonneg over the dimension vocabulary, every delivery entry on a
    real dimension with a well-formed check. A malformed row can't be honestly scored."""
    defects: list[str] = []
    seen_p: set[str] = set()
    for i, p in enumerate(personas):
        pid = p.get("id") if _nonempty(p.get("id")) else f"persona[{i}]"
        for f in PERSONA_FIELDS:
            if f not in p:
                defects.append(f"{pid}: missing field '{f}'")
        if not _nonempty(p.get("id")):
            defects.append(f"{pid}: missing id")
        elif p["id"] in seen_p:
            defects.append(f"{pid}: duplicate persona id")
        else:
            seen_p.add(p["id"])
        w = p.get("weights")
        if not isinstance(w, dict) or not w:
            defects.append(f"{pid}: weights must be a non-empty object over {DIM_IDS}")
        else:
            for k, v in w.items():
                if k not in DIM_SET:
                    defects.append(f"{pid}: weight dimension '{k}' not in the value-dimension vocabulary")
                elif not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                    defects.append(f"{pid}: weight '{k}'={v!r} must be a nonneg number")
            if sum(vv for kk, vv in w.items()
                   if kk in DIM_SET and isinstance(vv, (int, float)) and not isinstance(vv, bool) and vv > 0) <= 0:
                defects.append(f"{pid}: weights sum to 0 — a persona must value at least one dimension")

    seen_f: set[str] = set()
    for i, f in enumerate(features):
        fid = f.get("id") if _nonempty(f.get("id")) else f"feature[{i}]"
        for fld in FEATURE_FIELDS:
            if fld not in f:
                defects.append(f"{fid}: missing field '{fld}'")
        if not _nonempty(f.get("id")):
            defects.append(f"{fid}: missing id")
        elif f["id"] in seen_f:
            defects.append(f"{fid}: duplicate feature id")
        else:
            seen_f.add(f["id"])
        if groups and f.get("group") not in groups:
            defects.append(f"{fid}: group {f.get('group')!r} not declared in _meta.json")
        delivery = f.get("delivery")
        if not isinstance(delivery, list) or not delivery:
            defects.append(f"{fid}: delivery must be a non-empty list of per-dimension entries")
            continue
        seen_dim: set[str] = set()
        for j, entry in enumerate(delivery):
            if not isinstance(entry, dict):
                defects.append(f"{fid}: delivery[{j}] is not an object")
                continue
            dim = entry.get("dim")
            if dim not in DIM_SET:
                defects.append(f"{fid}: delivery[{j}] dim {dim!r} not in the value-dimension vocabulary")
            elif dim in seen_dim:
                defects.append(f"{fid}: delivery dimension {dim!r} declared twice")
            else:
                seen_dim.add(dim)
            checks = entry.get("checks")
            if not isinstance(checks, list) or not checks:
                defects.append(f"{fid}/{dim}: checks must be a non-empty list")
                continue
            sigs: set[str] = set()
            for k, chk in enumerate(checks):
                if not isinstance(chk, dict):
                    defects.append(f"{fid}/{dim}: check[{k}] is not an object")
                    continue
                if chk.get("kind") not in CHECK_KINDS:
                    defects.append(f"{fid}/{dim}: check kind {chk.get('kind')!r} not in {sorted(CHECK_KINDS)}")
                # Depth is graded by DISTINCT resolving checks, so a delivery entry may not
                # pad its depth with duplicate witnesses (the same path/command twice).
                sig = check_signature(chk)
                if sig in sigs:
                    defects.append(f"{fid}/{dim}: duplicate grounding check {sig} — depth cannot be "
                                   f"padded with the same witness twice")
                sigs.add(sig)
    n = len(personas) + len(features)
    return _kpi("rows_well_formed", 100 - KPI_PENALTY["rows_well_formed"] * len(defects), defects,
                f"all {n} persona+feature rows well-formed" if not defects
                else f"{len(defects)} malformed field(s)")


def kpi_dimension_relevant(features: list[dict[str, Any]],
                           deliv: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The ungameability backstop: every grounding check must be TOPICALLY RELEVANT to the
    dimension it claims to witness (a `benchmarked` witness names a benchmark, a `measured`
    witness names a metric/observability surface, …). Without this a feature's delivery —
    and the fit of every persona who values that dimension — could be inflated by pointing a
    resolving-but-unrelated check at it (e.g. grounding `benchmarked` with a security
    package). Each irrelevant check is one unit of persona-fit-debt and earns no depth.
    Cross-dimension witness reuse (a genuinely dual-purpose fact counted under two
    dimensions) is surfaced as a SOFT advisory, not debt."""
    defects: list[str] = []
    soft: list[str] = []
    total = relevant = 0
    for f in features:
        fid = f.get("id", "?")
        d = deliv.get(fid) or {}
        for dim, dd in (d.get("detail") or {}).items():
            for r in dd.get("results", []):
                total += 1
                if r.get("relevant"):
                    relevant += 1
        defects.extend(d.get("relevance_defects") or [])
        soft.extend(d.get("reuse_soft") or [])
    rate = (100 * relevant / total) if total else 100
    return _kpi("dimension_relevant", rate, defects,
                f"{relevant}/{total} grounding checks are relevant to their dimension ({rate:.0f}%)"
                if total else "no grounding checks declared", soft=soft)


def kpi_evidence_grounded(features: list[dict[str, Any]],
                          deliv: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The core grounding lever: every RELEVANT grounding check a feature names must RESOLVE
    in the tree. A check that points at a path / command / CLAIMS section that does not exist
    is phantom evidence — one unit of persona-fit-debt. You cannot inflate a feature's
    delivery (and thus the cells of every persona who values that dimension) by naming
    evidence that is not there; you ADD the real thing. (Relevance is graded separately by
    dimension_relevant, so an irrelevant check is not double-charged here.)"""
    defects: list[str] = []
    relevant_total = met = 0
    for f in features:
        fid = f.get("id", "?")
        d = deliv.get(fid) or {}
        for dim, dd in (d.get("detail") or {}).items():
            for r in dd.get("results", []):
                if r.get("relevant"):
                    relevant_total += 1
            met += dd.get("met", 0)
        defects.extend((d.get("ground_defects") or []))
    rate = (100 * met / relevant_total) if relevant_total else 100
    return _kpi("evidence_grounded", rate, defects,
                f"{met}/{relevant_total} relevant grounding checks resolve in the tree ({rate:.0f}%)"
                if relevant_total else "no relevant grounding checks declared")


def kpi_fit_honest(personas: list[dict[str, Any]], features: list[dict[str, Any]],
                   matrix: dict[str, Any]) -> dict[str, Any]:
    """The matrix cannot lie about who loves what. Each persona declares the feature it
    likes MOST (``top_feature``) and each feature declares the persona it WINS
    (``top_persona``); both must match the computed argmax of the grid. A declared
    favourite that disagrees with the math is an overclaim — the single most important
    thing this scorecard catches, the same discipline product/persona-readiness use for
    their verdicts. (A genuine tie or all-zero column resolves to the deterministic
    lowest id / 'none'; declaring 'none' there is honest.)"""
    defects: list[str] = []
    soft: list[str] = []
    cells = matrix.get("cells") or {}
    pr = matrix.get("persona_read") or {}
    fr = matrix.get("feature_read") or {}

    def _max_set(scores: dict[str, int]) -> set[str]:
        """Every id tied for the max positive score — declaring ANY of them is honest
        (a genuine tie has no single favourite; the alphabetical argmax is just one of them)."""
        if not scores:
            return set()
        best = max(scores.values())
        return {k for k, v in scores.items() if v == best and best > 0}

    for p in personas:
        pid = p.get("id")
        if not _nonempty(pid) or pid not in cells:
            continue
        declared = p.get("top_feature")
        winners = _max_set(cells[pid])
        computed = pr.get(pid, {}).get("top_feature")
        if not _nonempty(declared):
            soft.append(f"{pid}: declares no top_feature; computed favourite is {computed!r}")
        elif declared not in winners:
            tie = f" (tie among {sorted(winners)})" if len(winners) > 1 else ""
            defects.append(f"{pid}: claims its favourite feature is '{declared}' but the matrix "
                           f"implies '{computed}'{tie} — overclaim")
    for f in features:
        fid = f.get("id")
        if not _nonempty(fid) or fid not in fr:
            continue
        declared = f.get("top_persona")
        col = {pid: cells[pid].get(fid, 0) for pid in cells}
        winners = _max_set(col)
        computed = fr[fid]["top_persona"]
        if not _nonempty(declared):
            soft.append(f"{fid}: declares no top_persona; computed champion is {computed!r}")
        elif declared not in winners:
            tie = f" (tie among {sorted(winners)})" if len(winners) > 1 else ""
            defects.append(f"{fid}: claims it wins persona '{declared}' but the matrix implies "
                           f"'{computed}'{tie} — overclaim")
    return _kpi("fit_honest", 100 - KPI_PENALTY["fit_honest"] * len(defects), defects,
                "every declared favourite matches the computed matrix" if not defects
                else f"{len(defects)} favourite overclaim(s)", soft=soft)


# ---------------------------------------------------------------------------
# Coverage (how much of the persona x feature grid is positioned at all).
# ---------------------------------------------------------------------------

def coverage_report(required_personas: list[dict[str, Any]], required_features: list[dict[str, Any]],
                    personas: list[dict[str, Any]], features: list[dict[str, Any]]) -> dict[str, Any]:
    have_p = {p.get("id") for p in personas
              if _nonempty(p.get("id")) and isinstance(p.get("weights"), dict) and p.get("weights")}
    have_f = {f.get("id") for f in features
              if _nonempty(f.get("id")) and isinstance(f.get("delivery"), list) and f.get("delivery")}
    unc_p = [p for p in required_personas if p.get("id") not in have_p]
    unc_f = [f for f in required_features if f.get("id") not in have_f]
    total_cells = len(required_personas) * len(required_features)
    covered_cells = len(have_p & {p.get("id") for p in required_personas}) * \
        len(have_f & {f.get("id") for f in required_features})
    pct = round(100.0 * covered_cells / total_cells, 1) if total_cells else 100.0
    return {
        "required_personas": len(required_personas),
        "required_features": len(required_features),
        "positioned_personas": len(have_p & {p.get("id") for p in required_personas}),
        "positioned_features": len(have_f & {f.get("id") for f in required_features}),
        "total_cells": total_cells,
        "covered_cells": covered_cells,
        "coverage_pct": pct,
        "coverage_debt": len(unc_p) + len(unc_f),
        "uncovered_personas": unc_p,
        "uncovered_features": unc_f,
    }


# ---------------------------------------------------------------------------
# Fold: KPIs + coverage -> composite score, grade, persona-fit-debt, payload.
# ---------------------------------------------------------------------------

def per_row_debt(personas: list[dict[str, Any]], features: list[dict[str, Any]],
                 kpis: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, p in enumerate(personas):
        out[p.get("id") if _nonempty(p.get("id")) else f"persona[{i}]"] = 0
    for i, f in enumerate(features):
        out[f.get("id") if _nonempty(f.get("id")) else f"feature[{i}]"] = 0
    for k in kpis:
        for d in k["defects"]:
            rid = d.split(":", 1)[0].split("/", 1)[0]
            if rid in out:
                out[rid] += 1
    return out


def run_kpis(personas: list[dict[str, Any]], features: list[dict[str, Any]], groups: set[str],
             deliv: dict[str, dict[str, Any]], matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        kpi_rows_well_formed(personas, features, groups),
        kpi_dimension_relevant(features, deliv),
        kpi_evidence_grounded(features, deliv),
        kpi_fit_honest(personas, features, matrix),
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
    personas = [p for p in (data.get("personas") or []) if isinstance(p, dict)]
    features = [f for f in (data.get("features") or []) if isinstance(f, dict)]
    group_defs = [g for g in (data.get("feature_groups") or []) if isinstance(g, dict)]
    groups = {g.get("id") for g in group_defs if _nonempty(g.get("id"))}
    required_personas = [p for p in (data.get("required_personas") or []) if isinstance(p, dict)]
    required_features = [f for f in (data.get("required_features") or []) if isinstance(f, dict)]

    deliv = {f.get("id", f"feature[{i}]"): feature_delivery(f, tree) for i, f in enumerate(features)}
    delivery_vecs = {fid: dd["vec"] for fid, dd in deliv.items()}
    matrix = build_matrix(personas, features, delivery_vecs)

    kpis = run_kpis(personas, features, groups, deliv, matrix)
    by_name = {k["kpi"]: k for k in kpis}
    honesty_score = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"]
                              for n in KPI_WEIGHTS if n in by_name), 1)
    honesty_defects = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)

    cov = coverage_report(required_personas, required_features, personas, features)
    persona_fit_debt = honesty_defects + cov["coverage_debt"]

    cov_pct = cov["coverage_pct"] if cov["total_cells"] else 100.0
    score = round(HONESTY_WEIGHT * honesty_score + COVERAGE_WEIGHT * cov_pct, 1)
    grade = grade_letter(score)

    debt_by_group = {g: 0 for g in GROUPS}
    for k in kpis:
        debt_by_group[k["group"]] += len(k["defects"])
    breakdown = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    row_debt = per_row_debt(personas, features, kpis)

    # All cells as a flat list (for the matrix renderer + doc).
    persona_meta = {p.get("id"): p for p in personas if _nonempty(p.get("id"))}
    feature_meta = {f.get("id"): f for f in features if _nonempty(f.get("id"))}

    def _importance(pid: str) -> float:
        v = (persona_meta.get(pid) or {}).get("importance")
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else 1.0

    grid = []
    for pid, read in matrix["persona_read"].items():
        grid.append({
            "persona": pid,
            "persona_name": (persona_meta.get(pid) or {}).get("persona", pid),
            "tier": (persona_meta.get(pid) or {}).get("tier", ""),
            "importance": _importance(pid),
            "mean": read["mean"],
            "top_feature": read["top_feature"],
            "top_fit": read["top_fit"],
            "row": read["row"],
        })
    grid.sort(key=lambda r: (-r["mean"], r["persona"]))
    imp_total = sum(_importance(pid) for pid in matrix["persona_read"]) or 1.0
    feature_summary = []
    for fid, read in matrix["feature_read"].items():
        # Importance-weighted appeal: a feature that wins the personas who matter most
        # (a mass-market free-tier dev over a niche role) outranks a flat-mean tie. Default
        # importance is 1, so this collapses to the flat mean unless a roster sets weights.
        wmean = round(sum(read["col"].get(pid, 0) * _importance(pid)
                          for pid in matrix["persona_read"]) / imp_total, 1)
        feature_summary.append({
            "feature": fid,
            "feature_name": (feature_meta.get(fid) or {}).get("feature", fid),
            "group": (feature_meta.get(fid) or {}).get("group", ""),
            "mean": read["mean"],
            "weighted_mean": wmean,
            "top_persona": read["top_persona"],
            "top_fit": read["top_fit"],
            "delivery": delivery_vecs.get(fid, {}),
        })
    feature_summary.sort(key=lambda r: (-r["mean"], r["feature"]))

    corpus = {
        "score": score, "grade": grade,
        "honesty_score": honesty_score,
        "persona_fit_debt": persona_fit_debt,
        "honesty_defects": honesty_defects,
        "coverage_debt": cov["coverage_debt"],
        "coverage": cov,
        "soft_signals": n_soft,
        "personas": len(personas),
        "features": len(features),
        "dimensions": len(DIM_IDS),
        "as_of": meta.get("as_of", ""),
        "fak_version": meta.get("fak_version", ""),
        "debt_by_group": debt_by_group,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
        "row_debt": row_debt,
        "grid": grid,
        "feature_summary": feature_summary,
        "dimension_order": DIM_IDS,
    }

    feat_line = (f"{cov['positioned_features']}/{cov['required_features']} feature spaces × "
                 f"{cov['positioned_personas']}/{cov['required_personas']} personas")
    if persona_fit_debt == 0:
        ok, verdict, finding = True, "OK", "matrix_complete_and_grounded"
        reason = (f"persona-fit matrix complete + grounded: score {score}/100 (grade {grade}); "
                  f"{feat_line} positioned; zero persona-fit-debt across {len(kpis)} KPIs "
                  f"(every grounding check resolves, every declared favourite matches the math; "
                  f"{n_soft} advisory)")
        next_action = ("hold the line; when a feature's grounding regresses (a removed witness, a "
                       "renamed command) its delivery drops and every persona who values that "
                       "dimension loses fit — persona-fit-debt rises; re-run to keep it at 0")
    elif honesty_defects == 0 and cov["coverage_debt"] > 0:
        ok, verdict, finding = False, "ACTION", "coverage_debt"
        reason = (f"{cov['coverage_debt']} grid gap(s): {feat_line} positioned; score {score}/100 "
                  f"(grade {grade}); positioned rows are honest (0 honesty-debt)")
        next_action = ("close coverage: add weights for each unpositioned persona and a grounded "
                       "delivery row for each unpositioned feature space; re-run")
    else:
        ok, verdict, finding = False, "ACTION", "persona_fit_debt"
        worst = breakdown[0]
        reason = (f"{honesty_defects} grounding/honesty defect(s) + {cov['coverage_debt']} coverage "
                  f"gap(s) = persona-fit-debt {persona_fit_debt}; score {score}/100 (grade {grade}); "
                  f"heaviest KPI: {worst['kpi']} ({worst['debt']}); {feat_line} positioned")
        next_action = ("retire persona-fit-debt worst-first: make each phantom grounding check "
                       "resolve (ADD the real witness/command/section), align every declared "
                       "favourite to the computed matrix, then close coverage; re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "kpis": kpis,
        "_data": {"personas": personas, "features": features, "groups": group_defs,
                  "dimensions": VALUE_DIMENSIONS, "delivery": delivery_vecs,
                  "matrix": matrix["cells"]},
    }


# ---------------------------------------------------------------------------
# Disk shell — read the modular data DIRECTORY + the tree facts the checks verify.
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
    """Merge the modular data directory: _meta.json contributes meta + value-dimension
    doctrine echo + feature_groups + the personas (with weights) + required rosters;
    every rows-*.json contributes its feature `rows` (the delivery rows)."""
    meta_doc, err = _read_json(d / "_meta.json")
    if err:
        return None, err
    if not isinstance(meta_doc, dict):
        return None, "_meta.json is not a JSON object"
    out: dict[str, Any] = {
        "meta": meta_doc.get("meta") or {},
        "feature_groups": meta_doc.get("feature_groups") or [],
        "personas": meta_doc.get("personas") or [],
        "required_personas": meta_doc.get("required_personas") or meta_doc.get("personas") or [],
        "required_features": meta_doc.get("required_features") or [],
        "features": [],
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
                out["features"].append(r)
    return out, ""


def load_data(path: Path) -> tuple[dict[str, Any] | None, str]:
    if path.is_dir():
        return load_data_dir(path)
    return None, f"missing data directory: {path}"


def parse_claims_section_tags(text: str) -> dict[str, set[str]]:
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
# Renderers — terminal, matrix, per-persona, per-feature, chart, compare, doc.
# ---------------------------------------------------------------------------

def _fit_mark(v: int) -> str:
    """A one-char heat mark for a fit cell — at-a-glance density."""
    if v >= 70:
        return "█"
    if v >= 50:
        return "▓"
    if v >= 30:
        return "▒"
    if v >= 15:
        return "░"
    return "·"


def _short(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def render_matrix(payload: dict[str, Any]) -> str:
    """The headline: the persona x feature fit grid (0-100 per cell)."""
    c = payload.get("corpus") or {}
    fs = c.get("feature_summary") or []
    grid = c.get("grid") or []
    feats = [f["feature"] for f in fs]
    if not feats or not grid:
        return "persona-fit matrix: (no data)"
    head = f"  {'persona':<20} " + " ".join(f"{_short(fid, 6):>6}" for fid in feats) + "   mean  loves"
    lines = ["persona × feature-space fit matrix (0-100; how much each persona likes each feature):",
             "", head, "  " + "-" * (len(head) - 2)]
    for r in grid:
        row = r["row"]
        cells = " ".join(f"{row.get(fid, 0):>6}" for fid in feats)
        loves = r.get("top_feature") or "—"
        lines.append(f"  {_short(r['persona'], 20):<20} {cells}   {r['mean']:>4}  {loves}")
    lines.append("  " + "-" * (len(head) - 2))
    means = " ".join(f"{next((f['mean'] for f in fs if f['feature'] == fid), 0):>6}" for fid in feats)
    lines.append(f"  {'feature mean':<20} {means}")
    wins = " ".join(f"{_short(next((f['top_persona'] or '—' for f in fs if f['feature'] == fid), '—'), 6):>6}"
                    for fid in feats)
    lines.append(f"  {'wins persona':<20} {wins}")
    return "\n".join(lines)


def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    lines = [
        f"persona-fit-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· PERSONA-FIT-DEBT {c.get('persona_fit_debt', 0)} "
         f"(grounding/honesty {c.get('honesty_defects', 0)} + coverage {c.get('coverage_debt', 0)}) "
         f"· {c.get('soft_signals', 0)} advisory"),
        (f"grid: {cov.get('positioned_personas', 0)}/{cov.get('required_personas', 0)} personas × "
         f"{cov.get('positioned_features', 0)}/{cov.get('required_features', 0)} feature spaces "
         f"({c.get('dimensions', 0)} value dimensions); coverage {cov.get('coverage_pct', 0)}%"),
        ("debt by group: " + "  ".join(
            f"{g}:{(c.get('debt_by_group') or {}).get(g, 0)}" for g in GROUPS)),
        "",
        render_matrix(payload),
        "",
        "feature spaces (broadest appeal first):",
    ]
    for f in c.get("feature_summary", []):
        lines.append(f"  {_fit_mark(int(f['mean']))} {_short(f['feature'], 14):<14} mean {f['mean']:>4}  "
                     f"wins {f.get('top_persona') or '—'}  ({f.get('feature_name')})")
    lines += ["", "personas (most-served by the feature set first):"]
    for r in c.get("grid", []):
        lines.append(f"  {_fit_mark(int(r['mean']))} {_short(r['persona'], 18):<18} mean {r['mean']:>4}  "
                     f"loves {r.get('top_feature') or '—'} ({r.get('top_fit', 0)})")
    lines += ["", "per-KPI (worst first):",
              f"  {'score':>5} {'debt':>4}  {'group':<11} {'kpi':<20} detail"]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['group']:<11} {b['kpi']:<20} {b['detail']}")
    lines.append("")
    lines.append("persona-fit-debt work-list:")
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
        lines.append("  (none — zero persona-fit-debt; the matrix is complete and every cell is grounded)")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_persona(payload: dict[str, Any], pid: str) -> str:
    """How ONE persona likes each feature space — the 'how an engineer might like it' read."""
    c = payload.get("corpus") or {}
    row = next((r for r in c.get("grid", []) if r["persona"] == pid), None)
    if not row:
        ids = ", ".join(r["persona"] for r in c.get("grid", []))
        return f"no persona '{pid}'. known: {ids}"
    p = next((x for x in payload["_data"]["personas"] if x.get("id") == pid), {})
    w = normalize_weights(p.get("weights"))
    fs = {f["feature"]: f for f in c.get("feature_summary", [])}
    lines = [f"how '{row['persona_name']}' ({pid}) likes each feature space "
             f"— overall {row['mean']}/100, loves {row.get('top_feature') or '—'}:", ""]
    lines.append("  what this persona values most (normalized weight):")
    for d in sorted(DIM_IDS, key=lambda d: -w.get(d, 0)):
        if w.get(d, 0) <= 0:
            continue
        lines.append(f"      {DIM_LABEL[d]:<13} {_bar(int(round(w[d] * 100)), 100, width=20)} {w[d] * 100:.0f}%")
    lines += ["", "  fit per feature space (best first):"]
    for fid, fit in sorted(row["row"].items(), key=lambda kv: (-kv[1], kv[0])):
        meta = fs.get(fid) or {}
        lines.append(f"      {_fit_mark(fit)} {_short(fid, 14):<14} {_bar(fit, 100, width=24)} {fit:>3}  "
                     f"({meta.get('feature_name', fid)})")
    return "\n".join(lines)


def render_feature(payload: dict[str, Any], fid: str) -> str:
    """Who likes ONE feature space, and how it delivers across the value dimensions."""
    c = payload.get("corpus") or {}
    fsum = next((f for f in c.get("feature_summary", []) if f["feature"] == fid), None)
    if not fsum:
        ids = ", ".join(f["feature"] for f in c.get("feature_summary", []))
        return f"no feature '{fid}'. known: {ids}"
    deliv = fsum.get("delivery") or {}
    lines = [f"feature space '{fsum['feature_name']}' ({fid}) "
             f"— broadest-appeal mean {fsum['mean']}/100, wins {fsum.get('top_persona') or '—'}:", ""]
    lines.append("  delivery across the value dimensions (tree-grounded fraction):")
    for d in DIM_IDS:
        v = deliv.get(d, 0.0)
        lines.append(f"      {DIM_LABEL[d]:<13} {_bar(int(round(v * 100)), 100, width=24)} {v * 100:.0f}%")
    lines += ["", "  fit per persona (best first):"]
    col = {r["persona"]: r["row"].get(fid, 0) for r in c.get("grid", [])}
    for pid, fit in sorted(col.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"      {_fit_mark(fit)} {_short(pid, 18):<18} {_bar(fit, 100, width=24)} {fit:>3}")
    return "\n".join(lines)


def _bar(n: int, scale: int, width: int = 28, *, fill: str = "█", empty: str = "·") -> str:
    if scale <= 0:
        return empty * width
    cells = int(round(width * max(0, n) / scale))
    cells = max(0, min(width, cells))
    if n > 0 and cells == 0:
        cells = 1
    return fill * cells + empty * (width - cells)


def render_chart(payload: dict[str, Any]) -> str:
    """An at-a-glance ASCII chart: the fit heat-grid + feature/persona appeal bars.
    Deterministic + pure text: two clones at one commit chart identically."""
    c = payload.get("corpus") or {}
    grid = c.get("grid") or []
    fs = c.get("feature_summary") or []
    feats = [f["feature"] for f in fs]
    lines = [
        (f"persona-fit chart — {c.get('personas', 0)} personas × {c.get('features', 0)} feature spaces · "
         f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) · "
         f"persona-fit-debt {c.get('persona_fit_debt', 0)}"),
        "",
        "fit heat-grid (each cell = one persona×feature; denser = better fit):",
        "  " + " ".join(_short(fid, 4).rjust(4) for fid in feats) + "   persona",
    ]
    for r in grid:
        marks = " ".join(f"{_fit_mark(r['row'].get(fid, 0))}".rjust(4) for fid in feats)
        lines.append(f"  {marks}   {_short(r['persona'], 22)}")
    lines += ["", "feature-space appeal (mean fit across all personas):"]
    maxf = max((f["mean"] for f in fs), default=0)
    for f in fs:
        lines.append(f"  {_short(f['feature'], 14):<14} {_bar(int(f['mean']), int(maxf) or 1, width=26)} "
                     f"{f['mean']:>4}  wins {f.get('top_persona') or '—'}")
    lines += ["", "persona satisfaction (mean fit across all feature spaces):"]
    maxp = max((r["mean"] for r in grid), default=0)
    for r in sorted(grid, key=lambda r: (-r["mean"], r["persona"])):
        lines.append(f"  {_short(r['persona'], 18):<18} {_bar(int(r['mean']), int(maxp) or 1, width=26)} "
                     f"{r['mean']:>4}  loves {r.get('top_feature') or '—'}")
    lines += ["", "legend: █ ≥70   ▓ ≥50   ▒ ≥30   ░ ≥15   · <15  (fit 0-100)"]
    return "\n".join(lines)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("persona_fit_debt", 0), cur.get("persona_fit_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ratio = "∞ (zero)" if cd == 0 else f"{bd / cd:.1f}×"
    lines = [
        f"persona-fit-debt: {bd} -> {cd}   ({ratio} fewer defects+gaps)",
        f"  grounding/honesty: {b.get('honesty_defects', 0)} -> {cur.get('honesty_defects', 0)}",
        f"  coverage:          {b.get('coverage_debt', 0)} -> {cur.get('coverage_debt', 0)}",
        f"score:        {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<11} {gb} -> {gc}")
    target3 = max(0, (bd + 2) // 3)
    target2 = max(0, bd // 2)
    if cd <= target3:
        lines.append(f"VERDICT: ≥3× reduction achieved (persona-fit-debt {bd}->{cd}, target ≤{target3}).")
    elif cd <= target2:
        lines.append(f"VERDICT: ≥2× (not yet 3×) — persona-fit-debt {bd}->{cd}; 3× needs ≤{target3}.")
    else:
        lines.append(f"VERDICT: not yet 2× — need persona-fit-debt ≤{target2} (now {cd}); 3× target ≤{target3}.")
    return "\n".join(lines)


def _front_matter(title: str, desc: str) -> list[str]:
    return ["---", f'title: "{title}"', f'description: "{desc}"', "---", ""]


def render_doc_index(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    fs = c.get("feature_summary") or []
    grid = c.get("grid") or []
    feats = [f["feature"] for f in fs]
    out = _front_matter(
        "fak persona-fit scorecard — across each feature space, how much would each persona like fak?",
        "Inward persona×feature-space scorecard: a fit matrix of fak's top-10 personas (free-tier dev "
        "→ infra engineer → researcher → decision-maker) against its key feature spaces (security, "
        "performance, memory, model, tooling, platform). Each cell is a tree-grounded fit score — the "
        "weighted match of what a persona values and what a feature delivers. Two driven numbers: "
        "coverage of the grid and persona-fit-debt (the integrity of the matrix).")
    out.append("# Persona-fit scorecard — how much would each persona like each feature space?")
    out.append("")
    if stamp:
        out.append(f"<!-- persona-fit-scorecard: {stamp} · process: tools/persona_fit_scorecard.py · "
                   f"data: tools/persona_fit_scorecard.data/ -->")
        out.append("")
    out.append("The sibling [`persona-readiness`](../persona-scorecard/README.md) scorecard asks whether "
               "each top persona's *entry path* is served. This one asks the matrix question a "
               "go-to-market person draws: **across the key feature spaces fak ships, how much would each "
               "persona LIKE each one?** How an *engineer* feels about the model internals, how a "
               "*decision-maker* (≈ a product manager) feels about the benchmarks, how a *researcher* "
               "feels about reproducibility. Every cell is computed, not typed: a persona's WEIGHTS over "
               "the value dimensions × a feature's tree-grounded DELIVERY. To raise a cell you make the "
               "feature actually deliver what that persona values — never by editing the score.")
    out.append("")
    out.append("> Regenerate: `python tools/persona_fit_scorecard.py --markdown-dir docs/persona-fit-scorecard`.")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Coverage** | **{cov.get('coverage_pct', 0)}%** "
               f"({cov.get('positioned_personas', 0)}/{cov.get('required_personas', 0)} personas × "
               f"{cov.get('positioned_features', 0)}/{cov.get('required_features', 0)} feature spaces) |")
    out.append(f"| **Persona-fit-debt** | **{c.get('persona_fit_debt', 0)}** "
               f"(grounding/honesty {c.get('honesty_defects', 0)} + coverage {c.get('coverage_debt', 0)}) |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append(f"| Value dimensions | {c.get('dimensions', 0)} |")
    out.append(f"| As of | {c.get('as_of', '?')} (fak {c.get('fak_version', '?')}) |")
    out.append("")
    out.append("> **Read this right.** A LOW cell is not a defect — it is honest (a free-tier dev simply "
               "does not value the model internals the way a researcher does). The score grades the "
               "*integrity of the matrix*: every grounding check relevant to its dimension AND resolving "
               "in the tree, every declared favourite matching the computed matrix, the whole grid "
               "positioned. An *off-topic* or *ungrounded* delivery claim, or an *overclaimed* favourite, "
               "is the defect this catches. (A sub-100 cell can mean a dimension is genuinely under-served "
               "for that persona OR simply under-declared — delivery depth caps at "
               f"{DEPTH_CAP} grounded checks per dimension, so a 33% delivery means one witness was named, "
               "not that two failed.)")
    out.append("")
    out.append("## The fit matrix")
    out.append("")
    out.append("Each cell is the 0-100 fit of a feature space for a persona (the weighted match of what "
               "the persona values and what the feature delivers).")
    out.append("")
    header = "| Persona | " + " | ".join(_short(fid, 10) for fid in feats) + " | mean | loves |"
    sep = "|---|" + "".join("---:|" for _ in feats) + "---:|---|"
    out.append(header)
    out.append(sep)
    for r in grid:
        cells = " | ".join(str(r["row"].get(fid, 0)) for fid in feats)
        out.append(f"| **{r['persona']}** | {cells} | {r['mean']} | {r.get('top_feature') or '—'} |")
    means = " | ".join(str(next((f["mean"] for f in fs if f["feature"] == fid), 0)) for fid in feats)
    out.append(f"| _feature mean_ | {means} | | |")
    out.append("")
    out.append("## Standing at a glance")
    out.append("")
    out.append("> Regenerate this chart with `python tools/persona_fit_scorecard.py --chart`.")
    out.append("")
    out.append("```text")
    out.append(render_chart(payload))
    out.append("```")
    out.append("")
    out.append("## The value dimensions")
    out.append("")
    out.append("Both a persona's weights and a feature's delivery are vectors over this fixed set.")
    out.append("")
    out.append("| Dimension | A lander asks |")
    out.append("|---|---|")
    for d in VALUE_DIMENSIONS:
        out.append(f"| `{d['id']}` | {d['asks']} |")
    out.append("")
    out.append("## Feature spaces — delivery + who they win")
    out.append("")
    out.append("| Feature space | Group | Mean appeal | Weighted appeal | Wins persona |")
    out.append("|---|---|---:|---:|---|")
    for f in fs:
        out.append(f"| **{f['feature_name']}** (`{f['feature']}`) | {f.get('group', '')} | "
                   f"{f['mean']} | {f.get('weighted_mean', f['mean'])} | {f.get('top_persona') or '—'} |")
    out.append("")
    out.append("> _Weighted appeal_ weights each persona by its market `importance` (a mass-market "
               "free-tier dev counts more than a niche role), so a broad win outranks a flat-mean tie.")
    out.append("")
    out.append("## Personas — who the feature set serves best")
    out.append("")
    out.append("| Persona | Mean fit | Loves most |")
    out.append("|---|---:|---|")
    for r in grid:
        out.append(f"| **{r['persona_name']}** (`{r['persona']}`) | {r['mean']} | "
                   f"{r.get('top_feature') or '—'} ({r.get('top_fit', 0)}) |")
    out.append("")
    out.append("## Per-KPI (persona-fit-debt = grounding/honesty of the rows that exist)")
    out.append("")
    out.append("| Group | KPI | Score | Debt | Detail |")
    out.append("|---|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    work = [d for k in payload.get("kpis", []) for d in k["defects"]]
    if work:
        out.append("## Persona-fit-debt work list")
        out.append("")
        for d in work[:30]:
            out.append(f"- {d}")
        out.append("")
    return "\n".join(out)


def render_doc_folder(payload: dict[str, Any], *, stamp: str | None = None) -> dict[str, str]:
    return {"README.md": render_doc_index(payload, stamp=stamp)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Persona-fit scorecard (read-only unless --markdown-dir).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--matrix", action="store_true", help="just the persona × feature fit grid")
    ap.add_argument("--persona", default="", help="how one persona likes each feature space (by id)")
    ap.add_argument("--feature", default="", help="who likes one feature space + its delivery (by id)")
    ap.add_argument("--chart", action="store_true", help="an at-a-glance ASCII chart")
    ap.add_argument("--compare", default="", help="baseline JSON to prove persona-fit-debt dropped")
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
            print(f"wrote persona-fit doc folder -> {out_dir}")

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.persona:
        print(render_persona(payload, args.persona))
    elif args.feature:
        print(render_feature(payload, args.feature))
    elif args.matrix:
        print(render_matrix(payload))
    elif args.chart:
        print(render_chart(payload))
    elif not args.markdown_dir:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
