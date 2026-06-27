#!/usr/bin/env python3
"""Concept-disambiguation scorecard - is every similar-sounding fak concept CRYSTAL-CLEAR?

The sibling scorecards grade the tree's internals (``code_quality`` the Go module,
``docs`` the corpus), its competitive standing (``industry_scorecard``), whether a
person can use each concept (``product_scorecard``), and whether each REPORTED NUMBER
labels its provenance (``conflation_scorecard``). None of them asks the question that
bites a reader hardest as the system grows:

  **Of the massive, growing set of similar-SOUNDING names fak uses, is each distinct
  concept crystal-clear - one canonical name, a written definition, and an explicit
  line drawn against the siblings it is confused with - or do `cache`, `vCache`,
  `KV cache`, `cachemeta`, and the provider prompt-cache all blur into one fog?**

That is this scorecard. The "various items" are fak's CONFUSABLE concepts, grouped
into ``families`` (a shared root word that is overloaded: ``cache``, ``attention``,
``guard``/``gate``, ...). Each catalog row is one DISTINCT concept positioned on the
clarity axes a confused reader actually feels:

  GROUNDED      - the name really appears in the tree (a symbol / path / doc / metric);
                  you cannot disambiguate a name nobody uses, and you cannot invent one.
  DEFINED       - there is one distinguishing sentence saying what it IS.
  DISAMBIGUATED - for a concept that shares a family with siblings, an explicit
                  ``distinct_from`` names the siblings + a one-line ``distinction``
                  draws the boundary (what it is NOT). This is the crystal-clarity test.
  ANCHORED      - the distinction is written down somewhere DISCOVERABLE (a glossary
                  anchor that exists), so a newcomer can find it - not tribal knowledge.

The clarity **verdict** ladder folds those into one honest label per concept:

  crystal       grounded + defined + distinction + distinct_from resolves + anchor exists
  defined       grounded + defined + distinction, but the line is not written in a doc
  drifting      grounded + defined, but NO line drawn against its siblings
  colliding     shares a canonical name with another concept (a true ambiguity)
  undocumented  appears in the tree, but the catalog gives no definition

Every check CROSS-CHECKS the row against the real tree: the grounding token must
appear in the production corpus, the glossary anchor must exist on disk, a
``distinct_from`` reference must resolve to a real catalog id. So the score CANNOT be
gamed by editing the data alone; to drop debt you fix the real thing (rename a true
collision, write a definition, draw + anchor the distinction).

Two numbers are driven, mirroring ``product_scorecard``:

  DISAMBIGUATION-DEBT  honesty/clarity defects of the rows that EXIST + coverage gaps
                       (confusable concepts DISCOVERED in the tree with no row).
                       Folds into ``scorecard_control_pane`` via ``corpus.disambiguation_debt``.
  COVERAGE             of the confusable concept-tokens discovered in the tree, how many
                       are positioned (named + disambiguated) in the catalog at all.

The coverage UNIVERSE is the ungameable part: for each watched family the scorecard
DISCOVERS the distinct compound tokens in the tree, but a token only counts as a real
concept when it has genuine PRESENCE - it spans >= ``min_files`` production files, OR
is a package/dir name, OR is a doc heading. A one-off local field never inflates the
universe; a concept used across the kernel always does. You cannot shrink the universe
by editing the data dir.

Deterministic + read-only over the data (two clones at one commit score identically);
the only disk writes are the generated doc folder under ``--markdown-dir``. The source
of truth is a DIRECTORY of small JSON files so the family vocabulary and each family's
rows evolve independently::

    tools/concept_disambiguation_scorecard.data/
      _meta.json        meta + the declared family vocabulary (roots, ignore, min_files)
      rows-*.json       fak's confusable-concept rows, grouped by family

Run from the repo ROOT::

    python tools/concept_disambiguation_scorecard.py                 # human scorecard
    python tools/concept_disambiguation_scorecard.py --chart         # at-a-glance ASCII chart
    python tools/concept_disambiguation_scorecard.py --json          # machine payload (control-pane)
    python tools/concept_disambiguation_scorecard.py --critical      # worst-first clarity backlog
    python tools/concept_disambiguation_scorecard.py --gaps          # coverage backlog (unpositioned tree tokens)
    python tools/concept_disambiguation_scorecard.py --compare base.json   # prove the debt dropped
    python tools/concept_disambiguation_scorecard.py --markdown-dir docs/concept-disambiguation-scorecard
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

SCHEMA = "fak-concept-disambiguation-scorecard/1"
DATA_DIR_REL = "tools/concept_disambiguation_scorecard.data"
GENERATED_DOC_DIR = "docs/concept-disambiguation-scorecard"
CLI_REF_REL = "docs/cli-reference.md"

# ---------------------------------------------------------------------------
# Closed vocabularies. `family` is DATA-defined (declared in _meta.json) so the
# confusable-concept map can grow; the vocabularies below ARE the doctrine and fixed.
# ---------------------------------------------------------------------------

# What KIND of named thing the concept is - so a reader knows what surface to look at.
KINDS = {
    "concept": "a conceptual entity (an idea/layer), not a single symbol",
    "subsystem": "an internal mechanism / package",
    "symbol": "a specific code identifier (type / func / field / const)",
    "config": "a knob / flag / env var / default constant",
    "metric": "a reported counter / gauge / summary value",
    "cli-verb": "a fak subcommand a person runs",
    "doc-term": "a term that lives mainly in prose / a heading",
}
# How the grounding token is verified against the tree.
GROUNDING_KINDS = {"symbol", "path", "claims", "doc", "metric", "verb"}

# The clarity verdict ladder, best -> worst. The rank doubles as the "distance from
# crystal clarity" used to order the worst-first backlog.
VERDICTS = ["crystal", "defined", "drifting", "colliding", "undocumented"]
VERDICT_RANK = {v: i for i, v in enumerate(VERDICTS)}

GROUPS = ("well-formed", "distinctness", "grounded", "honesty")
KPI_GROUP: dict[str, str] = {
    "well_formed": "well-formed",
    "canonical_unique": "distinctness",
    "defined": "distinctness",
    "disambiguated": "distinctness",
    "grounded": "grounded",
    "anchored": "grounded",
    "clarity_consistent": "honesty",
}
KPI_WEIGHTS: dict[str, float] = {
    "well_formed": 0.10,
    "canonical_unique": 0.20,
    "defined": 0.14,
    "disambiguated": 0.24,
    "grounded": 0.14,
    "anchored": 0.08,
    "clarity_consistent": 0.10,
}
KPI_PENALTY: dict[str, int] = {
    "well_formed": 12,
    "canonical_unique": 25,
    "defined": 16,
    "disambiguated": 20,
    "grounded": 16,
    "anchored": 12,
    "clarity_consistent": 18,
}
# The composite blends the clarity of the rows that EXIST with how much of the
# discovered confusable space is even positioned. An un-mapped concept space costs
# grade HARD: crystal clarity is a property of the WHOLE namespace, not a few rows.
CLARITY_WEIGHT = 0.35
COVERAGE_WEIGHT = 0.65

REQUIRED_FIELDS = (
    "id", "canonical", "family", "kind", "definition", "distinction",
    "distinct_from", "aliases", "grounding", "grounding_kind",
    "glossary_anchor", "verdict", "gaps",
)


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


def norm_token(s: Any) -> str:
    """Collapse a name to its comparable token: lowercase, keep only [a-z0-9].

    So 'kv_cache', 'KV cache', and 'kvCache' all normalize to 'kvcache' - exactly
    the spelling-variant collapse that lets a catalog row match its tree token and
    that surfaces a true canonical-name collision between two rows."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def token_match(a: str, b: str) -> bool:
    """Does normalized token `a` denote the same concept as `b`? Equal, or one
    contains the other with a length guard so trivial overlaps ('id' in 'guard')
    do not match."""
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 5 and len(b) >= 5 and (a in b or b in a):
        return True
    return False


def row_tokens(r: dict[str, Any]) -> set[str]:
    """Every normalized name a row answers to: its canonical, its aliases, and its
    grounding token. Used to decide whether a discovered tree token is covered."""
    toks = {norm_token(r.get("canonical", "")), norm_token(r.get("grounding", ""))}
    for a in r.get("aliases") or []:
        toks.add(norm_token(a))
    return {t for t in toks if t}


# ---------------------------------------------------------------------------
# Cross-row collision detection (shared by the KPI and the verdict).
# ---------------------------------------------------------------------------

def find_collisions(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Row ids whose CANONICAL name collides with another row's canonical name.

    Two distinct concepts sharing THE canonical name is the worst confusability there
    is - it is never resolvable by a distinct_from note, only by a rename. Returns
    {id -> [colliding-other-id, ...]}. (Alias/cross-family token overlap is handled
    softer, by the disambiguated KPI + the clarity verdict, to avoid double-charging.)"""
    by_canon: dict[str, list[str]] = {}
    for i, r in enumerate(rows):
        rid = r.get("id") or f"row[{i}]"
        c = norm_token(r.get("canonical", ""))
        if c:
            by_canon.setdefault(c, []).append(rid)
    out: dict[str, list[str]] = {}
    for _canon, ids in by_canon.items():
        if len(ids) > 1:
            for rid in ids:
                out[rid] = [o for o in ids if o != rid]
    return out


def cluster_sizes(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How many positioned rows each family has (a family with >= 2 members is one
    whose members MUST disambiguate against each other)."""
    sizes: dict[str, int] = {}
    for r in rows:
        fam = r.get("family")
        if _nonempty(fam):
            sizes[fam] = sizes.get(fam, 0) + 1
    return sizes


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each returns
#   {kpi, group, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of disambiguation-debt; soft = score-only judgment nudges.
# Every defect string is prefixed `<id>: ` so per-row debt is recoverable.
# ---------------------------------------------------------------------------

def _kpi(name: str, defects: list[str], ok_detail: str, *, soft: list[str] | None = None,
         bad_detail: str | None = None) -> dict[str, Any]:
    soft = soft or []
    pen = KPI_PENALTY[name]
    detail = (bad_detail or f"{len(defects)} defect(s)") if defects else ok_detail
    return {"kpi": name, "group": KPI_GROUP[name],
            "score": _clamp(100 - pen * len(defects) - min(10, 2 * len(soft))),
            "detail": detail, "defects": defects, "soft": soft}


def kpi_well_formed(rows: list[dict[str, Any]], families: set[str]) -> dict[str, Any]:
    """A row must be shaped like a concept position: required fields present, every
    enum inside its closed vocabulary, family declared, id unique. A malformed row
    cannot be honestly graded, so it is hard debt."""
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
        if families and r.get("family") not in families:
            defects.append(f"{rid}: family {r.get('family')!r} not declared in _meta.json")
        if r.get("kind") not in KINDS:
            defects.append(f"{rid}: kind {r.get('kind')!r} not in {sorted(KINDS)}")
        if r.get("grounding_kind") not in GROUNDING_KINDS:
            defects.append(f"{rid}: grounding_kind {r.get('grounding_kind')!r} not in {sorted(GROUNDING_KINDS)}")
        if r.get("verdict") not in VERDICT_RANK:
            defects.append(f"{rid}: verdict {r.get('verdict')!r} not in {VERDICTS}")
        for listf in ("distinct_from", "aliases", "gaps"):
            if not isinstance(r.get(listf), list):
                defects.append(f"{rid}: {listf} must be a list")
        if not _nonempty(r.get("canonical")):
            defects.append(f"{rid}: missing canonical (the one true name)")
        if not _nonempty(r.get("grounding")):
            defects.append(f"{rid}: missing grounding (a token that must appear in the tree)")
    return _kpi("well_formed", defects, f"all {len(rows)} rows well-formed",
                bad_detail=f"{len(defects)} malformed field(s)")


def kpi_canonical_unique(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """No two concepts may share THE canonical name. A real collision is an
    irreducible ambiguity - the fix is a rename, not a note."""
    coll = find_collisions(rows)
    defects = [f"{rid}: canonical name collides with {', '.join(others)} - rename one"
               for rid, others in sorted(coll.items())]
    return _kpi("canonical_unique", defects, "every concept has a unique canonical name",
                bad_detail=f"{len(defects)} canonical-name collision(s)")


def kpi_defined(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every concept must carry one distinguishing DEFINITION sentence - the minimum
    clarity. A positioned concept with no definition is the catalog admitting it is
    still fog (it shows up as an 'undocumented' verdict too)."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        if not _nonempty(r.get("definition")):
            defects.append(f"{rid}: no definition - one sentence on what it IS")
    return _kpi("defined", defects, "every concept has a definition",
                bad_detail=f"{len(defects)} undefined concept(s)")


def kpi_disambiguated(rows: list[dict[str, Any]], sizes: dict[str, int]) -> dict[str, Any]:
    """THE crystal-clarity test. A concept that shares a family with >= 1 sibling must
    (a) carry a non-empty `distinction` line drawing the boundary, and (b) name >= 1
    sibling in `distinct_from` that RESOLVES to a real catalog id. A confusable concept
    that never says what it is NOT is the core debt this scorecard exists to retire.

    A lone concept in its family (no sibling positioned) is excused - there is nothing
    yet to disambiguate against; raising the family's coverage is the work instead."""
    ids = {r.get("id") for r in rows if _nonempty(r.get("id"))}
    canon = {norm_token(r.get("canonical", "")): r.get("id") for r in rows}
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        fam = r.get("family")
        if sizes.get(fam, 0) < 2:
            continue  # no positioned sibling to draw a line against (yet).
        if not _nonempty(r.get("distinction")):
            defects.append(f"{rid}: in family '{fam}' with siblings but no distinction line "
                           f"(what is it NOT?)")
        refs = [x for x in (r.get("distinct_from") or []) if _nonempty(x)]
        resolved = [x for x in refs if (x in ids or norm_token(x) in canon)]
        if not refs:
            defects.append(f"{rid}: distinct_from is empty but family '{fam}' has siblings")
        elif not resolved:
            defects.append(f"{rid}: distinct_from {refs} resolves to no catalog id/canonical")
    return _kpi("disambiguated", defects, "every confusable concept names what it is NOT",
                bad_detail=f"{len(defects)} undisambiguated confusable concept(s)")


def kpi_grounded(rows: list[dict[str, Any]], in_tree: Callable[[str], bool]) -> dict[str, Any]:
    """The grounding token must REALLY appear in the production corpus. A name nobody
    uses cannot be disambiguated - it is either stale (rename the concept away) or
    invented (this is the ungameable cross-check: you cannot position a fictional name)."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        g = r.get("grounding", "")
        if not _nonempty(g):
            continue  # missing grounding is a well_formed defect, not double-charged here.
        if not in_tree(norm_token(g)):
            defects.append(f"{rid}: grounding '{g}' does not appear in the production corpus")
    return _kpi("grounded", defects, "every concept's grounding token appears in the tree",
                bad_detail=f"{len(defects)} ungrounded concept(s)")


def kpi_anchored(rows: list[dict[str, Any]], exists: Callable[[str], bool]) -> dict[str, Any]:
    """A concept claiming CRYSTAL clarity must have its distinction WRITTEN somewhere
    discoverable: a glossary_anchor that exists. And any non-empty anchor (at any
    verdict) must resolve on disk - a dangling pointer is worse than none."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        anchor = r.get("glossary_anchor", "")
        if r.get("verdict") == "crystal":
            if not _nonempty(anchor):
                defects.append(f"{rid}: verdict 'crystal' but no glossary_anchor - "
                               f"where is the distinction written?")
                continue
        if _nonempty(anchor) and not exists(anchor):
            defects.append(f"{rid}: glossary_anchor '{anchor}' does not exist in the tree")
    return _kpi("anchored", defects, "every crystal concept's distinction is anchored on disk",
                bad_detail=f"{len(defects)} missing/dangling anchor(s)")


def expected_verdict(row: dict[str, Any], *, colliding: bool, exists: Callable[[str], bool],
                     sizes: dict[str, int]) -> tuple[str, str]:
    """The clarity verdict the evidence implies, worst-first. Grounding is graded
    separately (kpi_grounded) so it is not double-charged here.

      undocumented  no definition at all
      colliding     shares a canonical name with another concept
      drifting      defined, but no boundary drawn (only when a sibling exists)
      defined       boundary drawn, but not written in a doc (no/missing anchor)
      crystal       defined + boundary drawn + anchored on disk
    """
    if not _nonempty(row.get("definition")):
        return "undocumented", "no definition"
    if colliding:
        return "colliding", "shares a canonical name with another concept"
    has_sibling = sizes.get(row.get("family"), 0) >= 2
    if has_sibling and not _nonempty(row.get("distinction")):
        return "drifting", "defined but draws no line against its siblings"
    anchor = row.get("glossary_anchor", "")
    if not _nonempty(anchor) or not exists(anchor):
        return "defined", "boundary drawn but not written in a discoverable doc"
    return "crystal", "defined + distinction + anchored on disk"


def kpi_clarity_consistent(rows: list[dict[str, Any]], colliding_ids: set[str],
                           exists: Callable[[str], bool], sizes: dict[str, int]) -> dict[str, Any]:
    """The stated verdict must match what the evidence implies. Calling a drifting
    concept 'crystal', or a colliding one 'defined', is the overclaim this catches -
    the same self-report refusal the rest of the repo runs."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        declared = r.get("verdict")
        exp, why = expected_verdict(r, colliding=(rid in colliding_ids), exists=exists, sizes=sizes)
        if declared != exp:
            defects.append(f"{rid}: claims '{declared}' but evidence implies '{exp}' - {why}")
    return _kpi("clarity_consistent", defects, "every verdict matches its evidence",
                bad_detail=f"{len(defects)} verdict overclaim(s)")


def kpi_kind_grounding_soft(rows: list[dict[str, Any]], doc_verbs: set[str]) -> dict[str, Any]:
    """SOFT: the declared kind should agree with how it is grounded. A 'cli-verb'
    should be a documented verb; a 'metric' should be grounded as a metric. Advisory -
    a judgment nudge, never debt."""
    soft: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        kind, gk = r.get("kind"), r.get("grounding_kind")
        if kind == "cli-verb":
            verb = norm_token(r.get("grounding", ""))
            if gk != "verb":
                soft.append(f"{rid}: kind 'cli-verb' but grounding_kind '{gk}' (expected 'verb')")
            elif verb and verb not in doc_verbs:
                soft.append(f"{rid}: cli-verb '{r.get('grounding')}' not documented in cli-reference")
        if kind == "metric" and gk != "metric":
            soft.append(f"{rid}: kind 'metric' but grounding_kind '{gk}' (expected 'metric')")
    return {"kpi": "kind_grounding_soft", "group": "honesty",
            "score": _clamp(100 - min(40, 6 * len(soft))),
            "detail": "kind agrees with grounding" if not soft else f"{len(soft)} kind/grounding mismatch",
            "defects": [], "soft": soft}


def kpi_hierarchy_soft(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """SOFT: an optional `parent` (the hierarchy / isomorphism grouping) should resolve
    to a real catalog id and not cycle. Advisory - hierarchy is encouraged, not required."""
    ids = {r.get("id") for r in rows if _nonempty(r.get("id"))}
    parent: dict[str, str] = {}
    soft: list[str] = []
    for r in rows:
        rid = r.get("id")
        p = r.get("parent")
        if _nonempty(p):
            if p not in ids:
                soft.append(f"{rid}: parent '{p}' resolves to no catalog id")
            else:
                parent[rid] = p
    # cycle check
    for start in list(parent):
        seen, cur = set(), start
        while cur in parent:
            if cur in seen:
                soft.append(f"{start}: parent chain cycles")
                break
            seen.add(cur)
            cur = parent[cur]
    return {"kpi": "hierarchy_soft", "group": "honesty",
            "score": _clamp(100 - min(30, 6 * len(soft))),
            "detail": "hierarchy parents resolve" if not soft else f"{len(soft)} hierarchy issue(s)",
            "defects": [], "soft": soft}


# ---------------------------------------------------------------------------
# Coverage - discover the confusable universe in the tree, then ask how much of it
# the catalog positions. This is the ungameable engine that lands the honest score.
# ---------------------------------------------------------------------------

def discover_family_tokens(family: dict[str, Any], corpus: dict[str, Any]) -> list[dict[str, Any]]:
    """The distinct confusable tokens of one family that are REAL concepts in the tree.

    A token counts only when it has genuine presence: it spans >= min_files production
    files, OR is a package/dir name, OR is a doc heading / CLAIMS section. A one-off
    local field never inflates the universe. Returns sorted [{token, presence, where}]."""
    roots = [norm_token(x) for x in (family.get("roots") or []) if _nonempty(x)]
    ignore = {norm_token(x) for x in (family.get("ignore") or [])}
    exclude = [norm_token(x) for x in (family.get("exclude") or []) if _nonempty(x)]
    min_files = int(family.get("min_files", 2))

    def _matches(tok: str) -> bool:
        # belongs to this family iff it carries a root and no exclude substring
        # (so the `gate` root does not swallow `gateway`, which is its own family).
        if tok in ignore or not any(rt in tok for rt in roots):
            return False
        return not any(ex in tok for ex in exclude)

    sym_files: dict[str, set[str]] = corpus["sym_files"]   # token -> set(production files)
    structural: set[str] = corpus["structural"]            # dir/package/heading tokens
    found: dict[str, dict[str, Any]] = {}
    for tok, files in sym_files.items():
        if not _matches(tok):
            continue
        is_struct = tok in structural
        if len(files) >= min_files or is_struct:
            found[tok] = {"token": tok, "presence": len(files),
                          "where": "dir/heading" if is_struct else f"{len(files)} files"}
    # structural-only tokens (a dir/heading whose identifier never met min_files but
    # is unmistakably a real concept) still count.
    for tok in structural:
        if tok in found or not _matches(tok):
            continue
        found[tok] = {"token": tok, "presence": len(sym_files.get(tok, set())),
                      "where": "dir/heading"}
    return sorted(found.values(), key=lambda d: (-d["presence"], d["token"]))


def coverage_report(families: list[dict[str, Any]], rows: list[dict[str, Any]],
                    corpus: dict[str, Any]) -> dict[str, Any]:
    """For every watched family, discover the real confusable tokens and mark each
    covered when some row answers to it. Uncovered tokens are coverage_debt: a
    confusable concept the tree has but nobody disambiguated.

    The HEADLINE universe is deduped across families (a cross-cutting token like
    'enginecache' matches both the engine and cache roots but is one concept, counted
    once); per_family keeps its own view for the worst-family backlog."""
    claimed: set[str] = set()
    for r in rows:
        claimed |= row_tokens(r)

    def _covered(tok: str) -> bool:
        # A discovered token is covered only when a row SPECIFICALLY names it
        # (canonical / alias / grounding), by exact normalized identity - not by a
        # loose substring, or one 'cache' row would falsely cover the whole family.
        return tok in claimed

    per_family: list[dict[str, Any]] = []
    global_cov: dict[str, bool] = {}        # token -> covered (deduped)
    global_owner: dict[str, str] = {}       # token -> first family that found it
    for fam in families:
        fid = fam.get("id", "?")
        toks = discover_family_tokens(fam, corpus)
        fam_cov = 0
        fam_unc: list[str] = []
        for t in toks:
            tok = t["token"]
            hit = _covered(tok)
            if hit:
                fam_cov += 1
            else:
                fam_unc.append(tok)
            if tok not in global_cov:
                global_cov[tok] = hit
                global_owner[tok] = fid
        per_family.append({"family": fid, "discovered": len(toks),
                           "covered": fam_cov, "uncovered": fam_unc[:40]})
    total = len(global_cov)
    covered = sum(1 for v in global_cov.values() if v)
    uncovered = [{"family": global_owner[t], "token": t}
                 for t, v in sorted(global_cov.items()) if not v]
    pct = round(100.0 * covered / total, 1) if total else 100.0
    return {"discovered": total, "covered": covered, "coverage_pct": pct,
            "coverage_debt": total - covered, "per_family": per_family,
            "uncovered": uncovered}


# ---------------------------------------------------------------------------
# Fold: KPIs + coverage -> composite score, grade, disambiguation-debt, payload.
# ---------------------------------------------------------------------------

def standing(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {v: 0 for v in VERDICTS}
    for r in rows:
        v = r.get("verdict")
        if v in counts:
            counts[v] += 1
    return counts


def per_row_debt(rows: list[dict[str, Any]], kpis: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {r.get("id", f"row[{i}]"): 0 for i, r in enumerate(rows)}
    for k in kpis:
        for d in k["defects"]:
            rid = d.split(":", 1)[0]
            if rid in out:
                out[rid] += 1
    return out


def leaderboard(rows: list[dict[str, Any]], colliding_ids: set[str],
                exists: Callable[[str], bool], sizes: dict[str, int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        exp, _ = expected_verdict(r, colliding=(r.get("id") in colliding_ids),
                                  exists=exists, sizes=sizes)
        out.append({
            "id": r.get("id"), "canonical": r.get("canonical"), "family": r.get("family"),
            "kind": r.get("kind"), "verdict": r.get("verdict"), "expected_verdict": exp,
            "definition": r.get("definition"), "distinct_from": r.get("distinct_from") or [],
            "grounding": r.get("grounding"), "glossary_anchor": r.get("glossary_anchor", ""),
        })
    return out


def critical_backlog(rows: list[dict[str, Any]], row_debt: dict[str, int]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        rid = r.get("id")
        out.append({"id": rid, "canonical": r.get("canonical"), "family": r.get("family"),
                    "verdict": r.get("verdict"), "debt": row_debt.get(rid, 0),
                    "distance": VERDICT_RANK.get(r.get("verdict"), 9),
                    "gaps": r.get("gaps") or []})
    out.sort(key=lambda x: (-x["debt"], -x["distance"], x["id"] or ""))
    return out


def run_kpis(rows: list[dict[str, Any]], families: set[str], colliding_ids: set[str],
             sizes: dict[str, int], in_tree: Callable[[str], bool],
             exists: Callable[[str], bool], doc_verbs: set[str]) -> list[dict[str, Any]]:
    return [
        kpi_well_formed(rows, families),
        kpi_canonical_unique(rows),
        kpi_defined(rows),
        kpi_disambiguated(rows, sizes),
        kpi_grounded(rows, in_tree),
        kpi_anchored(rows, exists),
        kpi_clarity_consistent(rows, colliding_ids, exists, sizes),
        kpi_kind_grounding_soft(rows, doc_verbs),
        kpi_hierarchy_soft(rows),
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
    fam_defs = [c for c in (data.get("families") or []) if isinstance(c, dict)]
    families = {c.get("id") for c in fam_defs if _nonempty(c.get("id"))}

    corpus = tree.get("corpus") or {"sym_files": {}, "structural": set()}
    in_tree = tree.get("in_tree") or (lambda t: False)
    exists = tree.get("exists") or (lambda p: False)
    doc_verbs = tree.get("doc_verbs") or set()

    colliding_ids = set(find_collisions(rows))
    sizes = cluster_sizes(rows)

    kpis = run_kpis(rows, families, colliding_ids, sizes, in_tree, exists, doc_verbs)
    by_name = {k["kpi"]: k for k in kpis}
    clarity_score = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"]
                              for n in KPI_WEIGHTS if n in by_name), 1)
    clarity_defects = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)

    cov = coverage_report(fam_defs, rows, corpus)
    disambiguation_debt = clarity_defects + cov["coverage_debt"]

    cov_pct = cov["coverage_pct"] if cov["discovered"] else 100.0
    score = round(CLARITY_WEIGHT * clarity_score + COVERAGE_WEIGHT * cov_pct, 1)
    grade = grade_letter(score)

    debt_by_group = {g: 0 for g in GROUPS}
    for k in kpis:
        if k["group"] in debt_by_group:
            debt_by_group[k["group"]] += len(k["defects"])
    breakdown = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    row_debt = per_row_debt(rows, kpis)
    pos = standing(rows)
    lb = leaderboard(rows, colliding_ids, exists, sizes)
    crit = critical_backlog(rows, row_debt)
    n_crystal = pos.get("crystal", 0)

    corpus_out = {
        "score": score, "grade": grade,
        "clarity_score": clarity_score,
        "disambiguation_debt": disambiguation_debt,
        "clarity_defects": clarity_defects,
        "coverage_debt": cov["coverage_debt"],
        "coverage": cov,
        "soft_signals": n_soft,
        "rows": len(rows),
        "crystal_concepts": n_crystal,
        "families": len(fam_defs),
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

    standing_line = (f"{pos['crystal']} crystal / {pos['defined']} defined / "
                     f"{pos['drifting']} drifting / {pos['colliding']} colliding / "
                     f"{pos['undocumented']} undocumented")
    cov_line = (f"coverage {cov['coverage_pct']}% ({cov['covered']}/{cov['discovered']} "
                f"confusable tree tokens positioned)")
    if disambiguation_debt == 0:
        ok, verdict, finding = True, "OK", "namespace_crystal_clear"
        reason = (f"namespace crystal-clear: score {score}/100 (grade {grade}); {cov_line}; "
                  f"zero disambiguation-debt across {len(kpis)} KPIs over {len(rows)} concepts "
                  f"({standing_line}; {n_soft} advisory)")
        next_action = ("hold the line; when a new confusable name lands in the tree, coverage "
                       "drops - position + disambiguate it; re-run to keep debt at 0")
    elif clarity_defects == 0 and cov["coverage_debt"] > 0:
        ok, verdict, finding = False, "ACTION", "coverage_debt"
        reason = (f"{cov['coverage_debt']} confusable tree token(s) not yet positioned; {cov_line}; "
                  f"score {score}/100 (grade {grade}); positioned rows are clean (0 clarity-debt); "
                  f"standing {standing_line}")
        next_action = ("close coverage (see --gaps): add a disambiguated row for each unpositioned "
                       "confusable token, worst family first; re-run")
    else:
        ok, verdict, finding = False, "ACTION", "disambiguation_debt"
        worst = breakdown[0]
        reason = (f"{clarity_defects} clarity defect(s) + {cov['coverage_debt']} coverage gap(s) = "
                  f"disambiguation-debt {disambiguation_debt}; score {score}/100 (grade {grade}); "
                  f"heaviest KPI: {worst['kpi']} ({worst['debt']}); {cov_line}; standing {standing_line}")
        next_action = ("retire disambiguation-debt worst-first (--critical + per-KPI defects): rename "
                       "true collisions, write missing definitions, draw + anchor the distinctions; "
                       "then close coverage (--gaps); re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus_out, "kpis": kpis,
        "_data": {"rows": rows, "families": fam_defs},
    }


# ---------------------------------------------------------------------------
# Disk shell - read the modular data DIRECTORY + the tree-facts the KPIs verify.
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
    """Merge the modular data directory: _meta.json contributes meta + families;
    every other rows-*.json contributes its `rows`."""
    meta_doc, err = _read_json(d / "_meta.json")
    if err:
        return None, err
    if not isinstance(meta_doc, dict):
        return None, "_meta.json is not a JSON object"
    out: dict[str, Any] = {
        "meta": meta_doc.get("meta") or {},
        "families": meta_doc.get("families") or [],
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


_IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_SKIP_DIR = {".git", ".cache", "node_modules", "testdata", "_registry", "__pycache__",
             ".pytest_cache", ".ruff_cache", "vendor", ".dispatch-runs", ".goal-runs"}
_DOC_DIRS = ("docs",)
_ROOT_DOCS = ("CLAIMS.md", "ARCHITECTURE.md", "README.md", "AGENTS.md", "INDEX.md", "GLOSSARY.md")
_MAX_BYTES = 512 * 1024


def _walk_files(root: Path, suffix: str) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob(f"*{suffix}")):
        if any(part in _SKIP_DIR for part in p.parts):
            continue
        out.append(p)
    return out


def load_corpus(root: Path) -> dict[str, Any]:
    """The real-tree facts the coverage + grounded KPIs cross-check against.

      sym_files  : normalized identifier token -> set of PRODUCTION go files it is in
                   (internal/ + cmd/, excluding *_test.go). The presence map.
      structural : normalized tokens that are a package/dir name OR a doc heading OR a
                   CLAIMS section - unmistakably real concepts even if used in one file.
      grounded   : the union (every token that appears anywhere) - for kpi_grounded.
    """
    sym_files: dict[str, set[str]] = {}
    structural: set[str] = set()

    for base in ("internal", "cmd"):
        bdir = root / base
        if not bdir.is_dir():
            continue
        # dir / package names are structural concepts.
        for p in sorted(bdir.rglob("*")):
            if p.is_dir() and not any(part in _SKIP_DIR for part in p.parts):
                structural.add(norm_token(p.name))
        for gp in _walk_files(bdir, ".go"):
            if gp.name.endswith("_test.go"):
                continue
            try:
                if gp.stat().st_size > _MAX_BYTES:
                    continue
                text = gp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(gp.relative_to(root)).replace("\\", "/")
            seen: set[str] = set()
            for m in _IDENT_RE.finditer(text):
                tok = norm_token(m.group(0))
                if tok and tok not in seen:
                    seen.add(tok)
            for tok in seen:
                sym_files.setdefault(tok, set()).add(rel)

    # doc headings + CLAIMS sections are structural concepts.
    heading_re = re.compile(r"^#{1,4}\s+(.*)$")
    doc_files: list[Path] = []
    for dd in _DOC_DIRS:
        ddir = root / dd
        if ddir.is_dir():
            doc_files += _walk_files(ddir, ".md")
    for rd in _ROOT_DOCS:
        p = root / rd
        if p.exists():
            doc_files.append(p)
    for dp in doc_files:
        try:
            if dp.stat().st_size > _MAX_BYTES:
                continue
            text = dp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            m = heading_re.match(line)
            if not m:
                continue
            for w in _IDENT_RE.finditer(m.group(1)):
                structural.add(norm_token(w.group(0)))

    grounded_tokens = set(sym_files) | structural

    def in_tree(tok: str) -> bool:
        # STRICT identity only. A tolerant substring match would let a fabricated
        # grounding ('policyloaded') pass because it contains a real token ('policy') -
        # exactly the gaming the cross-check exists to refuse. A concept's grounding
        # must be a real identifier / dir / heading token, normalized, verbatim.
        return bool(tok) and tok in grounded_tokens

    doc_verbs: set[str] = set()
    cr = root / CLI_REF_REL
    if cr.exists():
        try:
            doc_verbs = {norm_token(w) for w in re.findall(r"[A-Za-z0-9-]+",
                                                           cr.read_text(encoding="utf-8"))}
        except OSError:
            doc_verbs = set()

    return {
        "corpus": {"sym_files": sym_files, "structural": structural},
        "in_tree": in_tree,
        "exists": lambda p: bool(p) and (root / p).exists(),
        "doc_verbs": doc_verbs,
    }


def collect(workspace: Path, *, data_path: Path | None = None) -> dict[str, Any]:
    root = workspace.resolve()
    path = data_path or (root / DATA_DIR_REL)
    data, err = load_data(path)
    tree = load_corpus(root)
    return build_payload(workspace=str(root), data=data, tree=tree, error=err or None)


# ---------------------------------------------------------------------------
# Renderers - terminal, chart, critical backlog, coverage gaps, compare, doc folder.
# ---------------------------------------------------------------------------

_MARK = {"crystal": "*", "defined": "o", "drifting": "~", "colliding": "x", "undocumented": "."}


def _bar(n: int, scale: int, width: int = 28, *, fill: str = "#", empty: str = ".") -> str:
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
        f"concept-disambiguation: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"= {round(c.get('score', 0) / 10.0, 1)}/10 "
         f"- DISAMBIGUATION-DEBT {c.get('disambiguation_debt', 0)} "
         f"(clarity {c.get('clarity_defects', 0)} + coverage {c.get('coverage_debt', 0)}) "
         f"- {c.get('soft_signals', 0)} advisory"),
        (f"coverage: {cov.get('coverage_pct', 0)}% "
         f"({cov.get('covered', 0)}/{cov.get('discovered', 0)} confusable tree tokens positioned) "
         f"- {c.get('rows', 0)} concepts scored - {c.get('crystal_concepts', 0)} crystal"),
        (f"standing: {pos.get('crystal', 0)} crystal - {pos.get('defined', 0)} defined - "
         f"{pos.get('drifting', 0)} drifting - {pos.get('colliding', 0)} colliding - "
         f"{pos.get('undocumented', 0)} undocumented"),
        ("debt by group: " + "  ".join(
            f"{g}:{(c.get('debt_by_group') or {}).get(g, 0)}" for g in GROUPS)),
        "",
        "concepts (best verdict first):",
        f"  {'verdict':<13} {'kind':<10} {'family':<16} canonical",
    ]
    for row in sorted(c.get("leaderboard", []),
                      key=lambda x: (VERDICT_RANK.get(x["verdict"], 9), x.get("family") or "")):
        mark = _MARK.get(row["verdict"], " ")
        flag = "" if row["verdict"] == row["expected_verdict"] else f"  ! expected {row['expected_verdict']}"
        lines.append(f"  {mark} {row['verdict']:<11} {str(row.get('kind')):<10} "
                     f"{str(row.get('family')):<16} {row.get('canonical')}{flag}")
    lines += ["", "per-KPI (worst first):",
              f"  {'score':>5} {'debt':>4}  {'group':<13} {'kpi':<22} detail"]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['group']:<13} "
                     f"{b['kpi']:<22} {b['detail']}")
    lines.append("")
    pf = cov.get("per_family") or []
    if pf:
        lines.append("coverage by family (covered / discovered):")
        for f in sorted(pf, key=lambda x: (x["covered"] - x["discovered"], x["family"])):
            lines.append(f"  {f['family']:<16} {f['covered']:>3}/{f['discovered']:<3}  "
                         f"({f['discovered'] - f['covered']} unpositioned)")
        lines.append("")
    lines.append("disambiguation-debt work-list:")
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
        lines.append("  (none - every positioned concept is clean)")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_critical(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    lines = ["concept-disambiguation critical backlog (clarify worst-first):", ""]
    crit = c.get("critical", [])
    if not crit:
        lines.append("  (no concepts scored)")
        return "\n".join(lines)
    shown = False
    for it in crit:
        if it["debt"] == 0 and it["distance"] <= VERDICT_RANK["defined"]:
            continue
        shown = True
        lines.append(f"  [{it['debt']} debt - {it['verdict']}] {it['id']} ({it['family']})")
        for g in (it.get("gaps") or [])[:4]:
            lines.append(f"      - {g}")
    if not shown:
        lines.append("  (no critical rows - every concept is defined-or-better with 0 debt)")
    lines.append("")
    lines.append("(rows with 0 debt and a crystal/defined verdict are omitted - they are not critical.)")
    return "\n".join(lines)


def render_gaps(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    lines = ["concept-disambiguation coverage backlog (position every confusable token):", ""]
    pf = cov.get("per_family") or []
    for f in sorted(pf, key=lambda x: (x["covered"] - x["discovered"], x["family"])):
        gap = f["discovered"] - f["covered"]
        lines.append(f"FAMILY {f['family']}: {f['covered']}/{f['discovered']} positioned "
                     f"({gap} unpositioned)")
        for tok in f.get("uncovered", []):
            lines.append(f"  - {tok}")
        lines.append("")
    if not pf:
        lines.append("  (no families declared)")
    return "\n".join(lines)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("disambiguation_debt", 0), cur.get("disambiguation_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ratio = "inf (zero)" if cd == 0 else f"{bd / cd:.1f}x"
    lines = [
        f"disambiguation-debt: {bd} -> {cd}   ({ratio} fewer defects+gaps)",
        f"  clarity:    {b.get('clarity_defects', 0)} -> {cur.get('clarity_defects', 0)}",
        f"  coverage:   {b.get('coverage_debt', 0)} -> {cur.get('coverage_debt', 0)}",
        f"score:        {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
        f"crystal:      {b.get('crystal_concepts', 0)} -> {cur.get('crystal_concepts', 0)} crystal concepts",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<13} {gb} -> {gc}")
    target3 = max(0, (bd + 2) // 3)
    target2 = max(0, bd // 2)
    if cd <= target3:
        lines.append(f"VERDICT: >=3x reduction achieved (debt {bd}->{cd}, target <={target3}).")
    elif cd <= target2:
        lines.append(f"VERDICT: >=2x (not yet 3x) - debt {bd}->{cd}; 3x needs <={target3}.")
    else:
        lines.append(f"VERDICT: not yet 2x - need debt <={target2} (now {cd}); 3x target <={target3}.")
    return "\n".join(lines)


def render_chart(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    pos = c.get("standing") or {}
    cov = c.get("coverage") or {}
    lb = c.get("leaderboard") or []
    lines: list[str] = [
        (f"concept-disambiguation chart - {c.get('rows', 0)} concepts - "
         f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) - "
         f"disambiguation-debt {c.get('disambiguation_debt', 0)}"),
        "",
        "clarity ladder (count of concepts, best -> fog):",
    ]
    maxn = max((pos.get(v, 0) for v in VERDICTS), default=0)
    for v in VERDICTS:
        n = pos.get(v, 0)
        lines.append(f"  {_MARK.get(v, ' ')} {v:<13} {_bar(n, maxn)} {n}")
    lines.append("")
    by_fam: dict[str, list[str]] = {}
    for r in lb:
        by_fam.setdefault(r.get("family") or "?", []).append(r.get("verdict"))
    lines.append("clarity mix by family (each cell = one concept):")
    for fam in sorted(by_fam):
        verds = sorted(by_fam[fam], key=lambda v: VERDICT_RANK.get(v, 9))
        spark = "".join(_MARK.get(v, " ") for v in verds)
        crystal = sum(1 for v in verds if v == "crystal")
        lines.append(f"  {fam:<16} {spark:<18} ({len(verds)} concept(s); {crystal} crystal)")
    lines.append("")
    lines.append("coverage by family (positioned / discovered):")
    for f in sorted(cov.get("per_family") or [], key=lambda x: (x["covered"] - x["discovered"], x["family"])):
        lines.append(f"  {f['family']:<16} {_bar(f['covered'], max(1, f['discovered']))} "
                     f"{f['covered']}/{f['discovered']}")
    lines.append("")
    pct = cov.get("coverage_pct", 0.0)
    lines.append(f"namespace coverage  [{_bar(int(round(pct)), 100, width=32)}] {pct}%  "
                 f"({cov.get('covered', 0)}/{cov.get('discovered', 0)} confusable tokens positioned)")
    lines.append("")
    lines.append("legend: " + "   ".join(f"{_MARK[v]} {v}" for v in VERDICTS))
    return "\n".join(lines)


def _front_matter(title: str, desc: str) -> list[str]:
    return ["---", f'title: "{title}"', f'description: "{desc}"', "---", ""]


def render_doc_index(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    out = _front_matter(
        "fak concept-disambiguation scorecard - is every similar-sounding concept crystal-clear",
        "Inward naming scorecard: each confusable fak concept positioned on the grounded / defined / "
        "disambiguated / anchored axes, with one clarity verdict per concept. Two driven numbers: "
        "coverage (of the confusable concept space discovered in the tree) and disambiguation-debt.")
    out.append("# Concept-disambiguation scorecard - crystal clarity across similar-sounding names")
    out.append("")
    if stamp:
        out.append(f"<!-- concept-disambiguation-scorecard: {stamp} - process: "
                   f"tools/concept_disambiguation_scorecard.py - data: {DATA_DIR_REL}/ -->")
        out.append("")
    out.append("The sibling scorecards grade fak's code, docs, and competitive standing. This one asks "
               "the question that bites a reader as the system grows: **of the massive, growing set of "
               "similar-sounding names (cache, vCache, KV cache, cachemeta, the provider prompt-cache), "
               "is each distinct concept crystal-clear - one canonical name, a written definition, and an "
               "explicit line drawn against the siblings it is confused with?** Every number below is "
               "re-derived by `tools/concept_disambiguation_scorecard.py` and cross-checked against the "
               "real tree (the grounding token must appear in the production corpus; the glossary anchor "
               "must exist; a `distinct_from` reference must resolve). No verdict is hand-typed.")
    out.append("")
    out.append("> Regenerate: `python tools/concept_disambiguation_scorecard.py "
               "--markdown-dir docs/concept-disambiguation-scorecard`.")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Score** | **{c.get('score', 0)}/100** (grade {c.get('grade', '?')}) "
               f"= {round(c.get('score', 0) / 10.0, 1)}/10 |")
    out.append(f"| **Coverage** | **{cov.get('coverage_pct', 0)}%** "
               f"({cov.get('covered', 0)}/{cov.get('discovered', 0)} confusable tree tokens positioned) |")
    out.append(f"| **Disambiguation-debt** | **{c.get('disambiguation_debt', 0)}** "
               f"(clarity {c.get('clarity_defects', 0)} + coverage {c.get('coverage_debt', 0)}) |")
    out.append(f"| Crystal-clear concepts | {c.get('crystal_concepts', 0)} of {c.get('rows', 0)} positioned |")
    out.append(f"| As of | {c.get('as_of', '?')} (fak {c.get('fak_version', '?')}) |")
    out.append("")
    out.append("> **Read this right.** The score is deliberately LOW at birth: it grades the WHOLE "
               "confusable namespace discovered in the tree, not the few concepts already catalogued. A "
               "low coverage number is the honest statement that most similar-sounding names are not yet "
               "disambiguated - which is exactly the debt this scorecard exists to retire.")
    out.append("")
    out.append("## Standing at a glance")
    out.append("")
    out.append("```text")
    out.append(render_chart(payload))
    out.append("```")
    out.append("")
    out.append("## The clarity ladder")
    out.append("")
    out.append("| Verdict | Means |")
    out.append("|---|---|")
    out.append("| * crystal | grounded + defined + a line drawn against siblings + that line anchored in a doc that exists |")
    out.append("| o defined | grounded + defined + a distinction line, but the line is not written in a discoverable doc |")
    out.append("| ~ drifting | grounded + defined, but no line drawn against its siblings (you know what it is, not what it is NOT) |")
    out.append("| x colliding | shares a canonical name with another concept - a true ambiguity, fixable only by a rename |")
    out.append("| . undocumented | appears in the tree, but the catalog gives no definition |")
    out.append("")
    out.append("## The concepts (best verdict first)")
    out.append("")
    out.append("| | Verdict | Kind | Family | Canonical - definition |")
    out.append("|---|---|---|---|---|")
    for row in sorted(c.get("leaderboard", []),
                      key=lambda x: (VERDICT_RANK.get(x["verdict"], 9), x.get("family") or "")):
        mark = _MARK.get(row["verdict"], " ")
        out.append(f"| {mark} | {row['verdict']} | {row.get('kind')} | {row.get('family')} | "
                   f"**{row.get('canonical')}** - {row.get('definition')} |")
    out.append("")
    out.append("## Per-KPI (disambiguation-debt = clarity of the rows that exist)")
    out.append("")
    out.append("| Group | KPI | Score | Debt | Detail |")
    out.append("|---|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    pf = cov.get("per_family") or []
    if pf:
        out.append("## Coverage by family (how much of each confusable space is positioned)")
        out.append("")
        out.append("| Family | Positioned | Discovered | Unpositioned |")
        out.append("|---|---:|---:|---:|")
        for f in sorted(pf, key=lambda x: (x["covered"] - x["discovered"], x["family"])):
            out.append(f"| {f['family']} | {f['covered']} | {f['discovered']} | "
                       f"{f['discovered'] - f['covered']} |")
        out.append("")
    return "\n".join(out)


def render_doc_folder(payload: dict[str, Any], *, stamp: str | None = None) -> dict[str, str]:
    return {"README.md": render_doc_index(payload, stamp=stamp)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Concept-disambiguation scorecard (read-only unless --markdown-dir).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--chart", action="store_true", help="an at-a-glance ASCII chart")
    ap.add_argument("--critical", action="store_true", help="the worst-first clarity backlog")
    ap.add_argument("--gaps", action="store_true", help="the coverage backlog (unpositioned tree tokens)")
    ap.add_argument("--compare", default="", help="baseline JSON to prove disambiguation-debt dropped")
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
            print(f"wrote concept-disambiguation doc folder -> {out_dir}")

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
