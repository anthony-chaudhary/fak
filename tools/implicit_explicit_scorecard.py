#!/usr/bin/env python3
"""Implicit-vs-explicit scorecard - is every concept fak RELIES ON directly NAMED?

The sibling ``concept_disambiguation_scorecard`` grades the names that EXIST: of the
similar-SOUNDING names, is each distinct concept crystal-clear? This scorecard asks
the question that comes BEFORE it:

  **Does each concept the system relies on have a DIRECT name at all - one code
  identifier and one written definition - or is it merely IMPLICIT: assumed by
  convention, hinted at in hedged prose ("the so-called warm window"), encoded as a
  magic literal repeated across files, named for the compiler but invisible in the
  docs, or named in prose with no code symbol behind it?**

An implicit concept is invisible debt: you cannot search for it, define it, or
disambiguate it (the sibling scorecard can only grade a name once one exists). This
card DETECTS the implicit concepts and drives NAMING them - each positioned row
carries a ``proposed_name`` (the naming plan) until a real ``named_symbol`` +
``doc_anchor`` make it explicit. Graduating here is the entry ticket to the
disambiguation card: once a concept has a direct name, THAT card ensures the new name
is distinct from its siblings. Related, never overlapping: this card scores the
absence of a name; that card scores the clarity between names.

The coverage UNIVERSE is discovered by four deterministic detectors over the real
tree - the signals a concept leaves when it is assumed or hinted at but not named:

  hinted          a hedge cue in a doc/comment ("so-called", "aka", "what we call",
                  "known as") introducing a quoted phrase - the author flags a
                  concept in their head that has no canonical, searchable name.
  latent-literal  a non-trivial numeric literal repeated on non-const lines across
                  >= min_files production files - an unnamed threshold/constant; the
                  concept exists in the machine but has NO name anywhere.
  code-only       an identifier spanning >= min_files production files that never
                  appears in the doc corpus - explicit to the compiler, implicit to
                  every human reader.
  doc-only        a recurring doc heading (>= min_doc_files files use the phrase)
                  with no matching code identifier - explicit to readers, implicit
                  in the code that implements it under some other, unstated name.

Each catalog row positions one implicit concept on the explicitness ladder:

  explicit     named_symbol resolves in the code corpus AND a definition is written
               at a doc_anchor that exists - directly named in BOTH worlds.
  named-code   a real identifier exists, but no written definition anywhere.
  named-doc    defined in prose at a real anchor, but no code symbol behind it.
  hinted       only hedges/patterns refer to it, but a proposed_name exists -
               the naming work is planned.
  latent       the concept is pure pattern; no name anywhere, no plan yet.

Every check CROSS-CHECKS the row against the real tree: the evidence must actually
appear (a symbol in the production corpus, a literal on non-const lines, a phrase in
the docs), a claimed ``named_symbol`` must resolve, a ``doc_anchor`` must exist on
disk. A row whose ``named_symbol`` resolves is re-grounded by the name itself - so
retiring a magic literal into a named constant never strands its row. The score
cannot be gamed by editing the data alone; to drop debt you NAME the real thing.

Two numbers are driven, mirroring the siblings:

  IMPLICITNESS-DEBT  naming defects of the rows that EXIST + coverage gaps (implicit
                     concepts DISCOVERED in the tree with no row positioning them).
                     Folds into ``scorecard_control_pane`` via ``corpus.implicitness_debt``.
  COVERAGE           of the implicit-concept signals discovered in the tree, how many
                     are positioned (catalogued with a naming plan) at all.

Deterministic + read-only over the data (two clones at one commit score identically);
the only disk writes are the generated doc folder under ``--markdown-dir``. The source
of truth is a DIRECTORY of small JSON files::

    tools/implicit_explicit_scorecard.data/
      _meta.json        meta + detector thresholds/ignores (the declared noise floor)
      rows-*.json       fak's implicit-concept rows, grouped by theme

Run from the repo ROOT::

    python tools/implicit_explicit_scorecard.py                 # human scorecard
    python tools/implicit_explicit_scorecard.py --chart         # at-a-glance ASCII chart
    python tools/implicit_explicit_scorecard.py --json          # machine payload (control-pane)
    python tools/implicit_explicit_scorecard.py --critical      # worst-first naming backlog
    python tools/implicit_explicit_scorecard.py --gaps          # coverage backlog (unpositioned signals)
    python tools/implicit_explicit_scorecard.py --compare base.json   # prove the debt dropped
    python tools/implicit_explicit_scorecard.py --markdown-dir docs/implicit-explicit-scorecard
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

SCHEMA = "fak-implicit-explicit-scorecard/1"
DATA_DIR_REL = "tools/implicit_explicit_scorecard.data"
GENERATED_DOC_DIR = "docs/implicit-explicit-scorecard"

# ---------------------------------------------------------------------------
# Closed vocabularies. Detector thresholds are DATA-defined (_meta.json) so the
# noise floor can be tuned; the vocabularies below ARE the doctrine and fixed.
# ---------------------------------------------------------------------------

# Which detector-signal class the row positions (how the concept stays implicit).
SIGNALS = {
    "hinted": "hedged prose introduces it in quotes; no canonical searchable name",
    "latent-literal": "a magic literal repeated across files; no name anywhere",
    "code-only": "a code identifier never mentioned in any doc",
    "doc-only": "a doc concept with no code identifier behind it",
    "assumed": "a convention everybody relies on that no detector can see (hand-positioned)",
}
# How the row's evidence token is verified against the tree.
EVIDENCE_KINDS = {"symbol", "literal", "phrase", "heading"}

# The explicitness ladder, best -> worst. Rank doubles as the "distance from
# explicit" used to order the worst-first backlog.
VERDICTS = ["explicit", "named-code", "named-doc", "hinted", "latent"]
VERDICT_RANK = {v: i for i, v in enumerate(VERDICTS)}

GROUPS = ("well-formed", "named", "grounded", "honesty")
KPI_GROUP: dict[str, str] = {
    "well_formed": "well-formed",
    "evidenced": "grounded",
    "named_resolves": "named",
    "anchored": "named",
    "naming_planned": "named",
    "explicitness_consistent": "honesty",
}
KPI_WEIGHTS: dict[str, float] = {
    "well_formed": 0.10,
    "evidenced": 0.18,
    "named_resolves": 0.16,
    "anchored": 0.12,
    "naming_planned": 0.22,
    "explicitness_consistent": 0.22,
}
KPI_PENALTY: dict[str, int] = {
    "well_formed": 12,
    "evidenced": 20,
    "named_resolves": 20,
    "anchored": 12,
    "naming_planned": 16,
    "explicitness_consistent": 18,
}
# The composite blends the naming hygiene of the rows that EXIST with how much of
# the discovered implicit space is even positioned. An un-mapped implicit space
# costs the grade HARD: explicitness is a property of the WHOLE system, not of the
# few concepts already catalogued.
CLARITY_WEIGHT = 0.35
COVERAGE_WEIGHT = 0.65

REQUIRED_FIELDS = (
    "id", "canonical", "signal", "evidence", "evidence_kind", "current_name",
    "proposed_name", "named_symbol", "doc_anchor", "definition", "aliases",
    "verdict", "gaps",
)

# Detector defaults - _meta.json `detectors` overrides any of these per detector.
DETECTOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "hinted": {"enabled": True, "ignore": []},
    "latent_literal": {
        "enabled": True, "min_files": 3, "ignore": [],
        # numerals that are idiom, not concept: unit conversions, tiny powers,
        # Go's reference time layout, POSIX file modes, standard HTTP status
        # codes, ANSI escape, calendar-year stamps.
        "trivial": ["100", "1000", "0.0", "1.0", "0.5", "255", "1024", "4096",
                    "10000", "100000", "0.001", "0.01", "0.1",
                    "100.0", "1000.0", "60.0", "24.0", "2.0",
                    "2006", "0102", "1504",
                    "0644", "0755", "0600", "0700", "0750", "0777",
                    "200", "201", "204", "301", "302", "400", "401", "403", "404",
                    "405", "408", "409", "410", "422", "429", "500", "501", "502",
                    "503", "504", "033",
                    "2024", "2025", "2026", "2027"],
    },
    "code_only": {"enabled": True, "min_files": 10, "min_len": 8, "ignore": []},
    "doc_only": {"enabled": True, "min_doc_files": 3, "max_words": 4, "ignore": []},
}


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

    'warm window', 'warmWindow', and 'warm_window' all normalize to 'warmwindow' -
    exactly the collapse that lets a hedged doc phrase match the identifier that
    later names it."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def row_keys(r: dict[str, Any]) -> set[str]:
    """Every key a row answers to in coverage: its normalized names PLUS the raw
    evidence string (so a latent-literal row claims its literal verbatim)."""
    keys = {norm_token(r.get(f, "")) for f in
            ("canonical", "evidence", "current_name", "proposed_name", "named_symbol")}
    for a in r.get("aliases") or []:
        keys.add(norm_token(a))
    ev = r.get("evidence")
    if _nonempty(ev):
        keys.add(ev.strip())
    return {k for k in keys if k}


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each returns
#   {kpi, group, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of implicitness-debt; soft = score-only judgment nudges.
# Every defect string is prefixed `<id>: ` so per-row debt is recoverable.
# ---------------------------------------------------------------------------

def _kpi(name: str, defects: list[str], ok_detail: str, *, soft: list[str] | None = None,
         bad_detail: str | None = None) -> dict[str, Any]:
    soft = soft or []
    pen = KPI_PENALTY[name]
    detail = (bad_detail or f"{len(defects)} defect(s)") if defects else ok_detail
    score = _clamp(100 - pen * len(defects) - min(10, 2 * len(soft)))
    return {"kpi": name, "group": KPI_GROUP[name],
            "score": score, "value": round(score / 100, 3),
            "detail": detail, "defects": defects, "soft": soft}


def kpi_well_formed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A row must be shaped like a naming position: required fields present, every
    enum inside its closed vocabulary, id unique. A malformed row cannot be honestly
    graded, so it is hard debt."""
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
        if r.get("signal") not in SIGNALS:
            defects.append(f"{rid}: signal {r.get('signal')!r} not in {sorted(SIGNALS)}")
        if r.get("evidence_kind") not in EVIDENCE_KINDS:
            defects.append(f"{rid}: evidence_kind {r.get('evidence_kind')!r} not in {sorted(EVIDENCE_KINDS)}")
        if r.get("verdict") not in VERDICT_RANK:
            defects.append(f"{rid}: verdict {r.get('verdict')!r} not in {VERDICTS}")
        for listf in ("aliases", "gaps"):
            if not isinstance(r.get(listf), list):
                defects.append(f"{rid}: {listf} must be a list")
        if not _nonempty(r.get("canonical")):
            defects.append(f"{rid}: missing canonical (the best name we have for it today)")
        if not _nonempty(r.get("evidence")):
            defects.append(f"{rid}: missing evidence (the tree signal that proves it exists)")
    return _kpi("well_formed", defects, f"all {len(rows)} rows well-formed",
                bad_detail=f"{len(defects)} malformed field(s)")


def _evidence_in_tree(r: dict[str, Any], tree: dict[str, Any]) -> bool:
    """Does the row's evidence actually appear in the tree, per its kind? A row
    whose named_symbol already resolves is re-grounded by the name itself - naming
    a magic literal away must never strand its row as 'unevidenced'."""
    in_code = tree.get("in_code") or (lambda t: False)
    if _nonempty(r.get("named_symbol")) and in_code(norm_token(r["named_symbol"])):
        return True
    ev = r.get("evidence", "")
    if not _nonempty(ev):
        return False
    kind = r.get("evidence_kind")
    if kind == "symbol":
        return in_code(norm_token(ev))
    if kind == "literal":
        lits = tree.get("literal_files") or {}
        return ev.strip() in lits
    if kind == "phrase":
        in_doc_text = tree.get("in_doc_text") or (lambda p: False)
        return in_doc_text(ev)
    if kind == "heading":
        headings = tree.get("heading_keys") or set()
        return norm_token(ev) in headings
    return False


def kpi_evidenced(rows: list[dict[str, Any]], tree: dict[str, Any]) -> dict[str, Any]:
    """The evidence must REALLY appear in the tree. You cannot position a fictional
    implicit concept - this is the ungameable cross-check."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        if not _nonempty(r.get("evidence")):
            continue  # missing evidence is a well_formed defect, not double-charged here.
        if r.get("evidence_kind") not in EVIDENCE_KINDS:
            continue  # bad enum is a well_formed defect.
        if not _evidence_in_tree(r, tree):
            defects.append(f"{rid}: evidence '{r.get('evidence')}' ({r.get('evidence_kind')}) "
                           f"does not appear in the tree")
    return _kpi("evidenced", defects, "every row's evidence appears in the tree",
                bad_detail=f"{len(defects)} unevidenced row(s)")


def kpi_named_resolves(rows: list[dict[str, Any]], in_code: Callable[[str], bool]) -> dict[str, Any]:
    """A claimed named_symbol must resolve in the production code corpus. A dangling
    name-claim is worse than none - it reports naming work that never landed."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        sym = r.get("named_symbol", "")
        if _nonempty(sym) and not in_code(norm_token(sym)):
            defects.append(f"{rid}: named_symbol '{sym}' does not resolve in the code corpus")
    return _kpi("named_resolves", defects, "every claimed name resolves in the code",
                bad_detail=f"{len(defects)} dangling name-claim(s)")


def kpi_anchored(rows: list[dict[str, Any]], exists: Callable[[str], bool]) -> dict[str, Any]:
    """A concept claiming EXPLICIT must have its definition WRITTEN at a doc_anchor
    that exists. And any non-empty anchor (at any verdict) must resolve on disk."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        anchor = r.get("doc_anchor", "")
        if r.get("verdict") == "explicit" and not _nonempty(anchor):
            defects.append(f"{rid}: verdict 'explicit' but no doc_anchor - "
                           f"where is the definition written?")
            continue
        if _nonempty(anchor) and not exists(anchor):
            defects.append(f"{rid}: doc_anchor '{anchor}' does not exist in the tree")
    return _kpi("anchored", defects, "every explicit concept's definition is anchored on disk",
                bad_detail=f"{len(defects)} missing/dangling anchor(s)")


def kpi_naming_planned(rows: list[dict[str, Any]], tree: dict[str, Any]) -> dict[str, Any]:
    """THE driver of this scorecard: every concept that is not yet explicit must
    carry a naming plan - a non-empty proposed_name (or an already-resolving
    named_symbol). Positioning an implicit concept without planning its name is
    cataloguing fog, not retiring it."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        exp, _ = expected_verdict(r, tree)
        if exp == "explicit":
            continue
        if _nonempty(r.get("proposed_name")) or _nonempty(r.get("named_symbol")):
            continue
        defects.append(f"{rid}: not yet explicit ({exp}) and no proposed_name - "
                       f"what should this concept be called?")
    return _kpi("naming_planned", defects, "every implicit concept has a naming plan",
                bad_detail=f"{len(defects)} concept(s) with no naming plan")


def expected_verdict(row: dict[str, Any], tree: dict[str, Any]) -> tuple[str, str]:
    """The explicitness verdict the evidence implies, best-first.

      explicit    named_symbol resolves in code AND definition written at a real anchor
      named-code  named_symbol resolves, but no written definition
      named-doc   definition at a real anchor, but no resolving code symbol
      hinted      neither, but a proposed_name exists (the naming work is planned)
      latent      no name anywhere and no plan
    """
    in_code = tree.get("in_code") or (lambda t: False)
    exists = tree.get("exists") or (lambda p: False)
    has_code = _nonempty(row.get("named_symbol")) and in_code(norm_token(row["named_symbol"]))
    has_doc = (_nonempty(row.get("doc_anchor")) and exists(row["doc_anchor"])
               and _nonempty(row.get("definition")))
    if has_code and has_doc:
        return "explicit", "named in code + defined at a real doc anchor"
    if has_code:
        return "named-code", "identifier resolves but no written definition"
    if has_doc:
        return "named-doc", "defined in prose but no code symbol behind it"
    if _nonempty(row.get("proposed_name")):
        return "hinted", "no direct name yet, but the naming is planned"
    return "latent", "no name anywhere and no naming plan"


def kpi_explicitness_consistent(rows: list[dict[str, Any]], tree: dict[str, Any]) -> dict[str, Any]:
    """The stated verdict must match what the evidence implies. Calling a latent
    concept 'explicit' is the overclaim this catches - the same self-report refusal
    the rest of the repo runs."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        declared = r.get("verdict")
        exp, why = expected_verdict(r, tree)
        if declared != exp:
            defects.append(f"{rid}: claims '{declared}' but evidence implies '{exp}' - {why}")
    return _kpi("explicitness_consistent", defects, "every verdict matches its evidence",
                bad_detail=f"{len(defects)} verdict overclaim(s)")


def kpi_name_quality_soft(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """SOFT: a proposed_name should be a NAME (<= 4 words, identifier-shaped), not a
    sentence, and should differ from the hedge phrase it replaces. Advisory - a
    judgment nudge, never debt."""
    soft: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        pn = r.get("proposed_name", "")
        if not _nonempty(pn):
            continue
        if len(pn.split()) > 4:
            soft.append(f"{rid}: proposed_name '{pn}' reads like a sentence, not a name")
        if r.get("signal") == "hinted" and norm_token(pn) and \
                norm_token(pn) == norm_token(r.get("evidence", "")):
            soft.append(f"{rid}: proposed_name just restates the hedge phrase - is that the best name?")
    score = _clamp(100 - min(40, 6 * len(soft)))
    return {"kpi": "name_quality_soft", "group": "honesty",
            "score": score, "value": round(score / 100, 3),
            "detail": "proposed names are name-shaped" if not soft else f"{len(soft)} naming nudge(s)",
            "defects": [], "soft": soft}


# ---------------------------------------------------------------------------
# Discovery - the four detectors over the tree-facts. This is the ungameable
# engine: the universe of implicit-concept signals cannot shrink by editing rows.
# ---------------------------------------------------------------------------

# Words that mark a heading as document STRUCTURE, not a concept name ("What it
# does", "Run it", "The problem"). A phrase containing any of these is skipped.
_STRUCT_WORDS = {
    "it", "its", "this", "that", "these", "those", "you", "your", "we", "our", "my",
    "is", "are", "was", "were", "do", "does", "did", "done", "not", "no", "yes",
    "and", "or", "of", "in", "on", "to", "for", "with", "without", "vs", "versus",
    "what", "why", "how", "when", "where", "who", "which", "one", "two", "three",
    "next", "first", "last", "more", "all", "now", "here", "there", "right", "wrong",
    "changed", "changes", "works", "matters", "read", "run", "use", "get", "see",
}
_ARTICLES = {"the", "a", "an"}


def _concept_phrase(raw: str) -> str:
    """Reduce a doc heading to a candidate concept phrase, or '' when the heading is
    document structure rather than a name. Leading articles are stripped (so 'The
    honest fence' -> 'honest fence' while 'The model' reduces to one word and drops
    out); any structural stopword disqualifies the phrase."""
    words = [w for w in raw.strip().split() if w]
    while words and words[0].lower() in _ARTICLES:
        words = words[1:]
    if len(words) < 2:
        return ""
    for w in words:
        if not re.fullmatch(r"[A-Za-z][A-Za-z-]*", w):
            return ""
        if w.lower() in _STRUCT_WORDS:
            return ""
    return " ".join(words)


def _det_cfg(meta_detectors: dict[str, Any] | None, name: str) -> dict[str, Any]:
    cfg = dict(DETECTOR_DEFAULTS[name])
    for k, v in ((meta_detectors or {}).get(name) or {}).items():
        cfg[k] = v
    return cfg


def discover_signals(tree: dict[str, Any], meta_detectors: dict[str, Any] | None) -> list[dict[str, Any]]:
    """All implicit-concept signals the tree emits, deduped by key. Each finding is
    {key, kind, presence, hint} - `key` is what a row must claim to cover it."""
    found: dict[str, dict[str, Any]] = {}

    def _add(key: str, kind: str, presence: int, hint: str) -> None:
        if key and key not in found:
            found[key] = {"key": key, "kind": kind, "presence": presence, "hint": hint}

    # hinted: hedge-introduced quoted phrases in docs + code comments.
    cfg = _det_cfg(meta_detectors, "hinted")
    if cfg.get("enabled", True):
        ignore = {norm_token(x) for x in (cfg.get("ignore") or [])}
        by_phrase: dict[str, dict[str, Any]] = {}
        for h in tree.get("hedges") or []:
            key = norm_token(h.get("phrase", ""))
            if not key or key in ignore:
                continue
            e = by_phrase.setdefault(key, {"files": set(), "phrase": h.get("phrase", "")})
            e["files"].add(h.get("file", "?"))
        for key, e in by_phrase.items():
            _add(key, "hinted", len(e["files"]),
                 f"hedged as \"{e['phrase']}\" in {len(e['files'])} file(s)")

    # latent-literal: repeated magic literals on non-const production lines.
    cfg = _det_cfg(meta_detectors, "latent_literal")
    if cfg.get("enabled", True):
        trivial = set(cfg.get("trivial") or [])
        ignore = set(cfg.get("ignore") or [])
        min_files = int(cfg.get("min_files", 3))
        for lit, files in (tree.get("literal_files") or {}).items():
            if lit in trivial or lit in ignore or len(files) < min_files:
                continue
            _add(lit, "latent-literal", len(files),
                 f"magic literal {lit} on non-const lines in {len(files)} files")

    # code-only: wide REPO-DECLARED identifiers never mentioned in any doc. Only
    # declarations count as concept candidates - stdlib method names used everywhere
    # (TrimSuffix, MustCompile) are Go idiom, not fak concepts.
    cfg = _det_cfg(meta_detectors, "code_only")
    if cfg.get("enabled", True):
        ignore = {norm_token(x) for x in (cfg.get("ignore") or [])}
        min_files = int(cfg.get("min_files", 10))
        min_len = int(cfg.get("min_len", 8))
        doc_tokens: set[str] = tree.get("doc_tokens") or set()
        sym: dict[str, Any] = tree.get("sym_files") or {}
        for tok in (tree.get("decl_tokens") or set()):
            files = sym.get(tok) or set()
            if len(tok) < min_len or tok in ignore or len(files) < min_files:
                continue
            if tok in doc_tokens:
                continue
            _add(tok, "code-only", len(files),
                 f"declared identifier in {len(files)} production files, zero doc mentions")

    # doc-only: recurring doc headings with no code identifier behind them.
    cfg = _det_cfg(meta_detectors, "doc_only")
    if cfg.get("enabled", True):
        ignore = {norm_token(x) for x in (cfg.get("ignore") or [])}
        min_doc_files = int(cfg.get("min_doc_files", 3))
        max_words = int(cfg.get("max_words", 4))
        sym: dict[str, Any] = tree.get("sym_files") or {}
        structural: set[str] = tree.get("structural") or set()
        count_doc_files = tree.get("count_doc_files") or (lambda p: 0)
        seen_heads: set[str] = set()
        for h in tree.get("headings") or []:
            phrase = _concept_phrase(h.get("phrase") or "")
            if not phrase:
                continue
            words = phrase.split()
            if not (2 <= len(words) <= max_words):
                continue
            key = norm_token(phrase)
            if not key or key in ignore or key in seen_heads:
                continue
            seen_heads.add(key)
            if key in sym or key in structural:
                continue
            n = count_doc_files(phrase)
            if n >= min_doc_files:
                _add(key, "doc-only", n,
                     f"heading \"{phrase}\" recurs in {n} docs, no code identifier")

    return sorted(found.values(), key=lambda d: (-d["presence"], d["key"]))


def coverage_report(signals: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mark each discovered signal covered when some row SPECIFICALLY claims it
    (canonical / evidence / proposed_name / named_symbol / alias), by exact key -
    not by a loose substring, or one broad row would falsely cover the whole space."""
    claimed: set[str] = set()
    for r in rows:
        claimed |= row_keys(r)
    per_kind: dict[str, dict[str, Any]] = {}
    uncovered: list[dict[str, Any]] = []
    covered = 0
    for s in signals:
        k = per_kind.setdefault(s["kind"], {"kind": s["kind"], "discovered": 0,
                                            "covered": 0, "uncovered": []})
        k["discovered"] += 1
        if s["key"] in claimed:
            covered += 1
            k["covered"] += 1
        else:
            k["uncovered"].append(s["key"])
            uncovered.append({"kind": s["kind"], "key": s["key"],
                              "presence": s["presence"], "hint": s["hint"]})
    for k in per_kind.values():
        k["uncovered"] = k["uncovered"][:40]
    total = len(signals)
    pct = round(100.0 * covered / total, 1) if total else 100.0
    return {"discovered": total, "covered": covered, "coverage_pct": pct,
            "coverage_debt": total - covered,
            "per_kind": sorted(per_kind.values(), key=lambda x: x["kind"]),
            "uncovered": uncovered}


# ---------------------------------------------------------------------------
# Fold: KPIs + coverage -> composite score, grade, implicitness-debt, payload.
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


def leaderboard(rows: list[dict[str, Any]], tree: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        exp, _ = expected_verdict(r, tree)
        out.append({
            "id": r.get("id"), "canonical": r.get("canonical"), "signal": r.get("signal"),
            "verdict": r.get("verdict"), "expected_verdict": exp,
            "evidence": r.get("evidence"), "proposed_name": r.get("proposed_name"),
            "named_symbol": r.get("named_symbol"), "doc_anchor": r.get("doc_anchor", ""),
            "definition": r.get("definition"),
        })
    return out


def critical_backlog(rows: list[dict[str, Any]], row_debt: dict[str, int]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        rid = r.get("id")
        out.append({"id": rid, "canonical": r.get("canonical"), "signal": r.get("signal"),
                    "verdict": r.get("verdict"), "debt": row_debt.get(rid, 0),
                    "distance": VERDICT_RANK.get(r.get("verdict"), 9),
                    "proposed_name": r.get("proposed_name", ""),
                    "gaps": r.get("gaps") or []})
    out.sort(key=lambda x: (-x["debt"], -x["distance"], x["id"] or ""))
    return out


def run_kpis(rows: list[dict[str, Any]], tree: dict[str, Any]) -> list[dict[str, Any]]:
    in_code = tree.get("in_code") or (lambda t: False)
    exists = tree.get("exists") or (lambda p: False)
    return [
        kpi_well_formed(rows),
        kpi_evidenced(rows, tree),
        kpi_named_resolves(rows, in_code),
        kpi_anchored(rows, exists),
        kpi_naming_planned(rows, tree),
        kpi_explicitness_consistent(rows, tree),
        kpi_name_quality_soft(rows),
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

    kpis = run_kpis(rows, tree)
    by_name = {k["kpi"]: k for k in kpis}
    clarity_score = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"]
                              for n in KPI_WEIGHTS if n in by_name), 1)
    clarity_defects = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)

    signals = discover_signals(tree, data.get("detectors"))
    cov = coverage_report(signals, rows)
    implicitness_debt = clarity_defects + cov["coverage_debt"]

    cov_pct = cov["coverage_pct"] if cov["discovered"] else 100.0
    score = round(CLARITY_WEIGHT * clarity_score + COVERAGE_WEIGHT * cov_pct, 1)
    grade = grade_letter(score)

    debt_by_group = {g: 0 for g in GROUPS}
    for k in kpis:
        if k["group"] in debt_by_group:
            debt_by_group[k["group"]] += len(k["defects"])
    breakdown = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "value": round(k["score"] / 100, 3),
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    row_debt = per_row_debt(rows, kpis)
    pos = standing(rows)
    lb = leaderboard(rows, tree)
    crit = critical_backlog(rows, row_debt)
    n_explicit = pos.get("explicit", 0)

    corpus_out = {
        "value": round(score / 100, 3), "value_unit": "quality_ratio",
        "score": score, "legacy_score": score, "legacy_score_scale": 100,
        "grade": grade,
        "clarity_score": clarity_score,
        "clarity_value": round(clarity_score / 100, 3),
        "implicitness_debt": implicitness_debt,
        "naming_defects": clarity_defects,
        "coverage_debt": cov["coverage_debt"],
        "coverage": cov,
        "soft_signals": n_soft,
        "rows": len(rows),
        "explicit_concepts": n_explicit,
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

    standing_line = (f"{pos['explicit']} explicit / {pos['named-code']} named-code / "
                     f"{pos['named-doc']} named-doc / {pos['hinted']} hinted / "
                     f"{pos['latent']} latent")
    cov_line = (f"coverage {cov['coverage_pct']}% ({cov['covered']}/{cov['discovered']} "
                f"implicit-concept signals positioned)")
    if implicitness_debt == 0:
        ok, verdict, finding = True, "OK", "every_concept_named"
        reason = (f"every relied-on concept is directly named: score {score}/100 (grade {grade}); "
                  f"{cov_line}; zero implicitness-debt across {len(kpis)} KPIs over {len(rows)} "
                  f"concepts ({standing_line}; {n_soft} advisory)")
        next_action = ("hold the line; when new hedges / magic literals / undocumented identifiers "
                       "land in the tree, coverage drops - position + name them; re-run")
    elif clarity_defects == 0 and cov["coverage_debt"] > 0:
        ok, verdict, finding = False, "ACTION", "coverage_debt"
        reason = (f"{cov['coverage_debt']} implicit-concept signal(s) not yet positioned; {cov_line}; "
                  f"score {score}/100 (grade {grade}); positioned rows are clean (0 naming-debt); "
                  f"standing {standing_line}")
        next_action = ("close coverage (see --gaps): position each unclaimed signal with a row + "
                       "proposed_name, highest presence first; re-run")
    else:
        ok, verdict, finding = False, "ACTION", "implicitness_debt"
        worst = breakdown[0]
        reason = (f"{clarity_defects} naming defect(s) + {cov['coverage_debt']} coverage gap(s) = "
                  f"implicitness-debt {implicitness_debt}; score {score}/100 (grade {grade}); "
                  f"heaviest KPI: {worst['kpi']} ({worst['debt']}); {cov_line}; standing {standing_line}")
        next_action = ("retire implicitness-debt worst-first (--critical + per-KPI defects): plan the "
                       "missing names, land the named symbols + doc anchors, fix overclaimed verdicts; "
                       "then close coverage (--gaps); re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus_out, "kpis": kpis,
        "_data": {"rows": rows},
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
    """Merge the modular data directory: _meta.json contributes meta + detectors;
    every other rows-*.json contributes its `rows`."""
    meta_doc, err = _read_json(d / "_meta.json")
    if err:
        return None, err
    if not isinstance(meta_doc, dict):
        return None, "_meta.json is not a JSON object"
    out: dict[str, Any] = {
        "meta": meta_doc.get("meta") or {},
        "detectors": meta_doc.get("detectors") or {},
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
# floats, or integers of >= 3 digits, standing alone (not inside an identifier).
_LITERAL_RE = re.compile(r"(?<![\w.])(\d+\.\d+|\d{3,})(?![\w.])")
# a hedge cue introducing a quoted phrase - the author naming a concept informally.
_HEDGE_RE = re.compile(
    r"(?:so-called|a\.k\.a\.|aka|informally(?:\s+called|\s+known\s+as)?|colloquially|"
    r"nicknamed|what\s+we\s+call|sometimes\s+called|known\s+as|by\s+convention\s+called)"
    r"[^\n\"'“”`]{0,40}?[\"'“‘`]([A-Za-z][A-Za-z0-9 _./-]{2,40})"
    r"[\"'”’`]", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#{1,4}\s+(.*?)\s*$")
# a top-level func/method/type declaration - what the REPO names, vs what it uses.
_DECL_RE = re.compile(r"^(?:func\s+(?:\([^)]*\)\s*)?|type\s+)([A-Za-z]\w*)", re.MULTILINE)
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


def _scan_go_file(text: str, rel: str, sym_files: dict[str, set[str]],
                  literal_files: dict[str, set[str]], hedges: list[dict[str, str]],
                  decl_tokens: set[str]) -> None:
    """One production go file -> identifier tokens, repo-declared names, non-const
    magic literals, and hedge phrases from its comments."""
    seen_tok: set[str] = set()
    for m in _IDENT_RE.finditer(text):
        tok = norm_token(m.group(0))
        if tok and tok not in seen_tok:
            seen_tok.add(tok)
            sym_files.setdefault(tok, set()).add(rel)
    for m in _DECL_RE.finditer(text):
        tok = norm_token(m.group(1))
        if tok:
            decl_tokens.add(tok)
    in_const_block = False
    seen_lit: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if in_const_block:
            if stripped.startswith(")"):
                in_const_block = False
            continue
        if stripped.startswith("const ") or stripped == "const (":
            if stripped.endswith("("):
                in_const_block = True
            continue
        code, _, comment = line.partition("//")
        if comment:
            for hm in _HEDGE_RE.finditer(comment):
                hedges.append({"phrase": hm.group(1).strip(), "file": rel})
        for lm in _LITERAL_RE.finditer(code):
            lit = lm.group(1)
            if lit not in seen_lit:
                seen_lit.add(lit)
                literal_files.setdefault(lit, set()).add(rel)


def load_corpus(root: Path) -> dict[str, Any]:
    """The real-tree facts the detectors + KPIs cross-check against.

      sym_files     : normalized identifier -> set of PRODUCTION go files (internal/
                      + cmd/, excluding *_test.go).
      structural    : normalized package/dir-name tokens.
      literal_files : magic-literal string -> set of files (non-const lines only).
      hedges        : [{phrase, file}] hedge-introduced quoted phrases (comments + docs).
      headings      : [{phrase, file}] doc headings.
      doc_tokens    : every normalized token appearing anywhere in the doc corpus.
      doc_texts     : rel doc path -> lowercased text (for phrase recurrence).
    """
    sym_files: dict[str, set[str]] = {}
    structural: set[str] = set()
    literal_files: dict[str, set[str]] = {}
    hedges: list[dict[str, str]] = []
    headings: list[dict[str, str]] = []
    decl_tokens: set[str] = set()
    doc_tokens: set[str] = set()
    doc_texts: dict[str, str] = {}

    for base in ("internal", "cmd"):
        bdir = root / base
        if not bdir.is_dir():
            continue
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
            _scan_go_file(text, rel, sym_files, literal_files, hedges, decl_tokens)

    # The card's own generated folder is NOT documentation: counting it would let a
    # regenerated README (which echoes every positioned symbol) silently retire
    # code-only signals from the universe - the exact self-gaming the doctrine bans.
    own_doc_dir = (root / GENERATED_DOC_DIR).resolve()
    doc_files: list[Path] = []
    for dd in _DOC_DIRS:
        ddir = root / dd
        if ddir.is_dir():
            doc_files += [p for p in _walk_files(ddir, ".md")
                          if own_doc_dir not in p.resolve().parents]
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
        rel = str(dp.relative_to(root)).replace("\\", "/")
        doc_texts[rel] = text.lower()
        for m in _IDENT_RE.finditer(text):
            tok = norm_token(m.group(0))
            if tok:
                doc_tokens.add(tok)
        for hm in _HEDGE_RE.finditer(text):
            hedges.append({"phrase": hm.group(1).strip(), "file": rel})
        for line in text.splitlines():
            m = _HEADING_RE.match(line)
            if m:
                headings.append({"phrase": m.group(1), "file": rel})

    # A heading answers both to its raw form and to its article-stripped concept
    # phrase ("The hot path" -> "hot path"), matching the doc_only detector's keys.
    heading_keys: set[str] = set()
    for h in headings:
        heading_keys.add(norm_token(h["phrase"]))
        reduced = _concept_phrase(h["phrase"])
        if reduced:
            heading_keys.add(norm_token(reduced))

    def in_code(tok: str) -> bool:
        # STRICT identity only - a tolerant substring match would let a fabricated
        # named_symbol pass because it contains a real token. A name-claim must be a
        # real identifier / dir token, normalized, verbatim.
        return bool(tok) and (tok in sym_files or tok in structural)

    def in_doc_text(phrase: str) -> bool:
        p = (phrase or "").strip().lower()
        return bool(p) and any(p in t for t in doc_texts.values())

    def count_doc_files(phrase: str) -> int:
        p = (phrase or "").strip().lower()
        if not p:
            return 0
        return sum(1 for t in doc_texts.values() if p in t)

    return {
        "sym_files": sym_files,
        "structural": structural,
        "literal_files": literal_files,
        "hedges": hedges,
        "headings": headings,
        "decl_tokens": decl_tokens,
        "heading_keys": heading_keys,
        "doc_tokens": doc_tokens,
        "in_code": in_code,
        "in_doc_text": in_doc_text,
        "count_doc_files": count_doc_files,
        "exists": lambda p: bool(p) and (root / p).exists(),
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

_MARK = {"explicit": "*", "named-code": "c", "named-doc": "d", "hinted": "~", "latent": "."}


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
        f"implicit-explicit: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"= {round(c.get('score', 0) / 10.0, 1)}/10 "
         f"- IMPLICITNESS-DEBT {c.get('implicitness_debt', 0)} "
         f"(naming {c.get('naming_defects', 0)} + coverage {c.get('coverage_debt', 0)}) "
         f"- {c.get('soft_signals', 0)} advisory"),
        (f"coverage: {cov.get('coverage_pct', 0)}% "
         f"({cov.get('covered', 0)}/{cov.get('discovered', 0)} implicit-concept signals positioned) "
         f"- {c.get('rows', 0)} concepts scored - {c.get('explicit_concepts', 0)} explicit"),
        (f"standing: {pos.get('explicit', 0)} explicit - {pos.get('named-code', 0)} named-code - "
         f"{pos.get('named-doc', 0)} named-doc - {pos.get('hinted', 0)} hinted - "
         f"{pos.get('latent', 0)} latent"),
        ("debt by group: " + "  ".join(
            f"{g}:{(c.get('debt_by_group') or {}).get(g, 0)}" for g in GROUPS)),
        "",
        "concepts (most explicit first):",
        f"  {'verdict':<12} {'signal':<15} {'canonical':<28} proposed / named",
    ]
    for row in sorted(c.get("leaderboard", []),
                      key=lambda x: (VERDICT_RANK.get(x["verdict"], 9), x.get("canonical") or "")):
        mark = _MARK.get(row["verdict"], " ")
        flag = "" if row["verdict"] == row["expected_verdict"] else f"  ! expected {row['expected_verdict']}"
        name = row.get("named_symbol") or row.get("proposed_name") or "-"
        lines.append(f"  {mark} {row['verdict']:<10} {str(row.get('signal')):<15} "
                     f"{str(row.get('canonical')):<28} {name}{flag}")
    lines += ["", "per-KPI (worst first):",
              f"  {'score':>5} {'debt':>4}  {'group':<12} {'kpi':<26} detail"]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['group']:<12} "
                     f"{b['kpi']:<26} {b['detail']}")
    lines.append("")
    pk = cov.get("per_kind") or []
    if pk:
        lines.append("coverage by signal kind (positioned / discovered):")
        for f in sorted(pk, key=lambda x: (x["covered"] - x["discovered"], x["kind"])):
            lines.append(f"  {f['kind']:<15} {f['covered']:>3}/{f['discovered']:<3}  "
                         f"({f['discovered'] - f['covered']} unpositioned)")
        lines.append("")
    lines.append("implicitness-debt work-list:")
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
    lines = ["implicit-explicit critical backlog (name worst-first):", ""]
    crit = c.get("critical", [])
    if not crit:
        lines.append("  (no concepts scored)")
        return "\n".join(lines)
    shown = False
    for it in crit:
        if it["debt"] == 0 and it["distance"] <= VERDICT_RANK["named-doc"]:
            continue
        shown = True
        pn = f" -> propose '{it['proposed_name']}'" if it.get("proposed_name") else ""
        lines.append(f"  [{it['debt']} debt - {it['verdict']}] {it['id']} ({it['signal']}){pn}")
        for g in (it.get("gaps") or [])[:4]:
            lines.append(f"      - {g}")
    if not shown:
        lines.append("  (no critical rows - every concept is named-or-better with 0 debt)")
    lines.append("")
    lines.append("(rows with 0 debt and an explicit/named verdict are omitted - they are not critical.)")
    return "\n".join(lines)


def render_gaps(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    lines = ["implicit-explicit coverage backlog (position every implicit signal):", ""]
    unc = cov.get("uncovered") or []
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for u in unc:
        by_kind.setdefault(u["kind"], []).append(u)
    for kind in sorted(by_kind):
        items = by_kind[kind]
        lines.append(f"SIGNAL {kind}: {len(items)} unpositioned")
        for u in sorted(items, key=lambda x: -x["presence"])[:60]:
            lines.append(f"  - {u['key']}  ({u['hint']})")
        if len(items) > 60:
            lines.append(f"  ... and {len(items) - 60} more")
        lines.append("")
    if not unc:
        lines.append("  (every discovered signal is positioned)")
    return "\n".join(lines)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("implicitness_debt", 0), cur.get("implicitness_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ratio = "inf (zero)" if cd == 0 else f"{bd / cd:.1f}x"
    lines = [
        f"implicitness-debt: {bd} -> {cd}   ({ratio} fewer defects+gaps)",
        f"  naming:     {b.get('naming_defects', 0)} -> {cur.get('naming_defects', 0)}",
        f"  coverage:   {b.get('coverage_debt', 0)} -> {cur.get('coverage_debt', 0)}",
        f"score:        {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
        f"explicit:     {b.get('explicit_concepts', 0)} -> {cur.get('explicit_concepts', 0)} explicit concepts",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<12} {gb} -> {gc}")
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
        (f"implicit-explicit chart - {c.get('rows', 0)} concepts - "
         f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) - "
         f"implicitness-debt {c.get('implicitness_debt', 0)}"),
        "",
        "explicitness ladder (count of concepts, named -> fog):",
    ]
    maxn = max((pos.get(v, 0) for v in VERDICTS), default=0)
    for v in VERDICTS:
        n = pos.get(v, 0)
        lines.append(f"  {_MARK.get(v, ' ')} {v:<12} {_bar(n, maxn)} {n}")
    lines.append("")
    by_sig: dict[str, list[str]] = {}
    for r in lb:
        by_sig.setdefault(r.get("signal") or "?", []).append(r.get("verdict"))
    lines.append("explicitness mix by signal (each cell = one concept):")
    for sig in sorted(by_sig):
        verds = sorted(by_sig[sig], key=lambda v: VERDICT_RANK.get(v, 9))
        spark = "".join(_MARK.get(v, " ") for v in verds)
        explicit = sum(1 for v in verds if v == "explicit")
        lines.append(f"  {sig:<15} {spark:<20} ({len(verds)} concept(s); {explicit} explicit)")
    lines.append("")
    lines.append("coverage by signal kind (positioned / discovered):")
    for f in sorted(cov.get("per_kind") or [], key=lambda x: (x["covered"] - x["discovered"], x["kind"])):
        lines.append(f"  {f['kind']:<15} {_bar(f['covered'], max(1, f['discovered']))} "
                     f"{f['covered']}/{f['discovered']}")
    lines.append("")
    pct = cov.get("coverage_pct", 0.0)
    lines.append(f"signal coverage  [{_bar(int(round(pct)), 100, width=32)}] {pct}%  "
                 f"({cov.get('covered', 0)}/{cov.get('discovered', 0)} implicit signals positioned)")
    lines.append("")
    lines.append("legend: " + "   ".join(f"{_MARK[v]} {v}" for v in VERDICTS))
    return "\n".join(lines)


def _front_matter(title: str, desc: str) -> list[str]:
    return ["---", f'title: "{title}"', f'description: "{desc}"', "---", ""]


def render_doc_index(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    out = _front_matter(
        "fak implicit-explicit scorecard - is every relied-on concept directly named",
        "Inward naming scorecard: detects concepts that are assumed or hinted at but never "
        "directly named (hedged prose, magic literals, code-only identifiers, doc-only headings) "
        "and drives naming them. Two driven numbers: coverage of the discovered implicit-signal "
        "space and implicitness-debt.")
    out.append("# Implicit-explicit scorecard - naming what the system only assumes")
    out.append("")
    if stamp:
        out.append(f"<!-- implicit-explicit-scorecard: {stamp} - process: "
                   f"tools/implicit_explicit_scorecard.py - data: {DATA_DIR_REL}/ -->")
        out.append("")
    out.append("The sibling [concept-disambiguation scorecard](../concept-disambiguation-scorecard/) "
               "grades the names that EXIST - of the similar-sounding names, is each distinct concept "
               "crystal-clear? This card asks the question that comes BEFORE it: **does each concept "
               "the system relies on have a direct name at all, or is it merely implicit - assumed by "
               "convention, hedged in prose (\"the so-called warm window\"), encoded as a magic literal "
               "repeated across files, named for the compiler but invisible in the docs, or named in "
               "prose with no code symbol behind it?** Every number below is re-derived by "
               "`tools/implicit_explicit_scorecard.py` and cross-checked against the real tree (the "
               "evidence must appear; a claimed named_symbol must resolve; a doc_anchor must exist). "
               "No verdict is hand-typed.")
    out.append("")
    out.append("> Regenerate: `python tools/implicit_explicit_scorecard.py "
               "--markdown-dir docs/implicit-explicit-scorecard`.")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Score** | **{c.get('score', 0)}/100** (grade {c.get('grade', '?')}) "
               f"= {round(c.get('score', 0) / 10.0, 1)}/10 |")
    out.append(f"| **Coverage** | **{cov.get('coverage_pct', 0)}%** "
               f"({cov.get('covered', 0)}/{cov.get('discovered', 0)} implicit-concept signals positioned) |")
    out.append(f"| **Implicitness-debt** | **{c.get('implicitness_debt', 0)}** "
               f"(naming {c.get('naming_defects', 0)} + coverage {c.get('coverage_debt', 0)}) |")
    out.append(f"| Explicit concepts | {c.get('explicit_concepts', 0)} of {c.get('rows', 0)} positioned |")
    out.append(f"| As of | {c.get('as_of', '?')} (fak {c.get('fak_version', '?')}) |")
    out.append("")
    out.append("> **Read this right.** The score is deliberately LOW at birth: it grades the WHOLE "
               "implicit-signal space discovered in the tree, not the few concepts already catalogued. "
               "A low coverage number is the honest statement that most assumed/hinted concepts are "
               "not yet named - which is exactly the debt this scorecard exists to retire.")
    out.append("")
    out.append("## The explicitness ladder")
    out.append("")
    out.append("| Verdict | Means |")
    out.append("|---|---|")
    out.append("| * explicit | a code identifier resolves AND a definition is written at a doc anchor that exists - named in both worlds |")
    out.append("| c named-code | a real identifier exists, but no written definition anywhere |")
    out.append("| d named-doc | defined in prose at a real anchor, but no code symbol behind it |")
    out.append("| ~ hinted | only hedges/patterns refer to it, but a proposed_name exists - naming planned |")
    out.append("| . latent | pure pattern; no name anywhere, no plan yet |")
    out.append("")
    out.append("## Standing at a glance")
    out.append("")
    out.append("```text")
    out.append(render_chart(payload))
    out.append("```")
    out.append("")
    out.append("## The concepts (most explicit first)")
    out.append("")
    out.append("| | Verdict | Signal | Canonical - definition | Proposed / named |")
    out.append("|---|---|---|---|---|")
    for row in sorted(c.get("leaderboard", []),
                      key=lambda x: (VERDICT_RANK.get(x["verdict"], 9), x.get("canonical") or "")):
        mark = _MARK.get(row["verdict"], " ")
        name = row.get("named_symbol") or row.get("proposed_name") or "-"
        out.append(f"| {mark} | {row['verdict']} | {row.get('signal')} | "
                   f"**{row.get('canonical')}** - {row.get('definition')} | `{name}` |")
    out.append("")
    out.append("## Per-KPI (implicitness-debt = naming hygiene of the rows that exist)")
    out.append("")
    out.append("| Group | KPI | Score | Debt | Detail |")
    out.append("|---|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    pk = cov.get("per_kind") or []
    if pk:
        out.append("## Coverage by signal kind (how much of each implicit space is positioned)")
        out.append("")
        out.append("| Signal | Positioned | Discovered | Unpositioned |")
        out.append("|---|---:|---:|---:|")
        for f in sorted(pk, key=lambda x: (x["covered"] - x["discovered"], x["kind"])):
            out.append(f"| {f['kind']} | {f['covered']} | {f['discovered']} | "
                       f"{f['discovered'] - f['covered']} |")
        out.append("")
    return "\n".join(out)


def render_doc_folder(payload: dict[str, Any], *, stamp: str | None = None) -> dict[str, str]:
    return {"README.md": render_doc_index(payload, stamp=stamp)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Implicit-explicit scorecard (read-only unless --markdown-dir).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--chart", action="store_true", help="an at-a-glance ASCII chart")
    ap.add_argument("--critical", action="store_true", help="the worst-first naming backlog")
    ap.add_argument("--gaps", action="store_true", help="the coverage backlog (unpositioned signals)")
    ap.add_argument("--compare", default="", help="baseline JSON to prove implicitness-debt dropped")
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
            print(f"wrote implicit-explicit doc folder -> {out_dir}")

    if args.json:
        print(json.dumps(payload, indent=2, default=sorted))
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
