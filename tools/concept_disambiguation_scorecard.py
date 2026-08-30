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
  SEPARATED     - the boundary is drawn against the concept it is genuinely mistakable
                  for, from BOTH sides. Per-row clarity is not pairwise separation: in a
                  250-member family, naming ANY sibling satisfies DISAMBIGUATED while the
                  twin whose SPELLING collides goes undrawn. See "separation" below.
  ANCHORED      - the distinction is written down somewhere DISCOVERABLE (a glossary
                  anchor that exists), so a newcomer can find it - not tribal knowledge.
  INDEXED       - a reader who meets any SPELLING of the name (canonical / alias /
                  grounding token) lands on exactly one concept. See "indexing" below.

The clarity **verdict** ladder folds those into one honest label per concept:

  crystal       grounded + defined + distinction + distinct_from resolves + anchor exists
  defined       grounded + defined + distinction, but the line is not written in a doc
  drifting      grounded + defined, but NO line drawn against its siblings
  entangled     defined and draws SOME line, but not against the twin its own NAME is
                confusable with - per-row clean, pairwise still fog
  colliding     shares a canonical name with another concept (a true ambiguity)
  undocumented  appears in the tree, but the catalog gives no definition

SEPARATION - disambiguating concepts FROM EACH OTHER. The catalog's own names are swept
for the pairs a reader cannot keep apart: ``permuted`` (identical word multiset in a
different order - ``witnessPath`` / ``PathWitness``) and ``near`` (canonical names within
a couple of edits - ``SessionRef`` / ``SessionRow``, ``q4Kernel`` / ``q8Kernel``). For
each such pair the line must be drawn between THOSE TWO - and drawn MUTUALLY, because a
boundary is directed: ``A.distinct_from = [B]`` says nothing to a reader who arrived at
B, which is exactly as likely when the names are twins. The pair list is DISCOVERED, not
declared, so it grows by itself the moment a peer lands a near-twin name.

INDEXING - the reverse map. The catalog is organised by concept; a reader arrives with a
spelling. ``build_index`` turns every surface (canonical / alias / grounding) into a
lookup key carrying the concept it denotes plus its contrast set, and refuses a key that
lands on two concepts which do not separate from each other - an ambiguity the canonical
names cannot show, because both canonicals can be unique while one row quietly claims the
other's name as an ALIAS. Rendered to the generated ``INDEX.md``.

Every check CROSS-CHECKS the row against the real tree: the grounding token must
appear in the production corpus, the glossary anchor must exist on disk, a
``distinct_from`` reference must resolve to a real catalog id. So the score CANNOT be
gamed by editing the data alone; to drop debt you fix the real thing (rename a true
collision, write a definition, draw + anchor the distinction).

For scale, an optional ``parent`` edge rolls a concept up to the abstraction that HEADS
it, and ``--rollup`` folds the forest into a higher-level operator view. The fold is
WEAKEST-LINK: an abstraction reads as crystal-clear only when EVERY concept beneath it
is - one ``defined`` leaf keeps the whole head from rolling up to crystal. That keeps the
collapsed view honest (it cannot hide fog it contains) and ungameable (it is derived from
the same cross-checked per-row verdicts); a head that reads clearer than its subtree
supports is flagged as an abstraction ``overclaim`` (advisory - hierarchy is optional).

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
    python tools/concept_disambiguation_scorecard.py --rollup        # hierarchy roll-up (abstraction heads, weakest-link)
    python tools/concept_disambiguation_scorecard.py --gaps          # coverage backlog (unpositioned tree tokens)
    python tools/concept_disambiguation_scorecard.py --pairs         # pairwise separation backlog (confusable name-pairs)
    python tools/concept_disambiguation_scorecard.py --index         # name -> concept lookup index
    python tools/concept_disambiguation_scorecard.py --lookup "KV cache"  # resolve ONE name a reader met
    python tools/concept_disambiguation_scorecard.py --compare base.json   # prove the debt dropped
    fak concept generate  # canonical writer; preserves classifications
"""
from __future__ import annotations

import argparse
import functools
import itertools
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
# crystal clarity" used to order the worst-first backlog. `entangled` sits between
# `drifting` and `colliding`: the concept HAS a definition and draws a line at some
# sibling, but the twin its NAME is actually mistakable for is not the one it
# separates from - a per-row-clean concept the namespace still cannot keep apart.
VERDICTS = ["crystal", "defined", "drifting", "entangled", "colliding", "undocumented"]
VERDICT_RANK = {v: i for i, v in enumerate(VERDICTS)}

GROUPS = ("well-formed", "distinctness", "separation", "indexed", "grounded", "honesty")
KPI_GROUP: dict[str, str] = {
    "well_formed": "well-formed",
    "canonical_unique": "distinctness",
    "defined": "distinctness",
    "disambiguated": "distinctness",
    "reference_resolves": "separation",
    "pair_separated": "separation",
    "pair_mutual": "separation",
    "grounded": "grounded",
    "anchored": "grounded",
    "index_resolves": "indexed",
    "clarity_consistent": "honesty",
}
KPI_WEIGHTS: dict[str, float] = {
    "well_formed": 0.08,
    "canonical_unique": 0.14,
    "defined": 0.10,
    "disambiguated": 0.15,
    "reference_resolves": 0.07,
    "pair_separated": 0.12,
    "pair_mutual": 0.05,
    "grounded": 0.10,
    "anchored": 0.05,
    "index_resolves": 0.07,
    "clarity_consistent": 0.07,
}
KPI_PENALTY: dict[str, int] = {
    "well_formed": 12,
    "canonical_unique": 25,
    "defined": 16,
    "disambiguated": 20,
    "reference_resolves": 14,
    "pair_separated": 20,
    "pair_mutual": 10,
    "grounded": 16,
    "anchored": 12,
    "index_resolves": 18,
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


_NORM_TOKEN_RE = re.compile(r"[^a-z0-9]")


@functools.lru_cache(maxsize=32768)
def _norm_token_str(s: str) -> str:
    return _NORM_TOKEN_RE.sub("", s.lower())


def norm_token(s: Any) -> str:
    """Collapse a name to its comparable token: lowercase, keep only [a-z0-9].

    So 'kv_cache', 'KV cache', and 'kvCache' all normalize to 'kvcache' - exactly
    the spelling-variant collapse that lets a catalog row match its tree token and
    that surfaces a true canonical-name collision between two rows."""
    if not isinstance(s, str):
        return ""
    return _norm_token_str(s)


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
# SEPARATION - disambiguating concepts FROM EACH OTHER (not just annotating each
# one on its own).
#
# `kpi_disambiguated` asks a PER-ROW question: does this concept say what it is
# NOT, naming at least one sibling? In a family of 250 members that bar is met by
# naming ANY sibling - including a distant one - while the twin a reader actually
# confuses it with goes undrawn. Clarity per row does not imply the namespace is
# separated pairwise.
#
# So this layer works on PAIRS. It discovers, from the catalog itself, the pairs
# whose NAMES a reader cannot keep apart, and then asks whether the catalog draws
# the line between THOSE two specifically:
#
#   permuted  identical word multiset, different order - `witnessPath` /
#             `PathWitness` are the same two words and denote different things.
#   near      normalized canonical names within a couple of edits - `SessionRef` /
#             `SessionRow`, `CacheGiB` / `CacheBit`, `q4Kernel` / `q8Kernel`.
#
# A boundary is DIRECTED: `A.distinct_from = [B]` helps a reader who arrived at A
# and does nothing for one who arrived at B. For a confusable pair the line must
# therefore be MUTUAL - drawn from both sides - or half the readers stay lost.
# ---------------------------------------------------------------------------

# camelCase / ALLCAPS / snake / dotted -> word pieces. `KVCacheShape` splits to
# ('kv','cache','shape') so a permutation of the same pieces is detectable.
_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
_NONWORD_RE = re.compile(r"[^A-Za-z0-9]+")
# Names carry a trailing gloss in parens (`q4Kernel (compute)`); the gloss is prose
# for the reader, not part of the name, so it is stripped before comparison.
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
MAX_PAIR_EDITS = 2      # names this close are one typo apart - a reader will confuse them.
MIN_PAIR_LEN = 6        # below this, an edit or two is a whole different word.
# A couple of edits alone is too loose on short names: `ffngate` and `rungate` are two
# substitutions apart yet share only the family root every member shares. What makes a
# pair genuinely mistakable is a long common HEAD or TAIL - `cachettl5m`/`cachettl1h`,
# `kernel`/`kernelkv`, `sessionref`/`sessionrow` - so a near pair must also share this
# many leading or trailing characters. Without it the sweep would flag most of a family.
MIN_SHARED_AFFIX = 5


def bare_name(name: Any) -> str:
    """A canonical name with its trailing parenthetical gloss removed."""
    if not isinstance(name, str):
        return ""
    return _PAREN_RE.sub("", name).strip()


def split_words(name: Any) -> list[str]:
    """Lowercase word pieces of a name, across camelCase / snake / dotted / spaced."""
    out: list[str] = []
    for chunk in _NONWORD_RE.split(bare_name(name)):
        out.extend(w.lower() for w in _WORD_RE.findall(chunk))
    return out


def edit_distance_within(a: str, b: str, cap: int) -> int:
    """Levenshtein(a, b) when it is <= cap, else cap + 1.

    Length-gated and row-pruned so the all-pairs sweep over a few thousand names
    stays cheap: any row whose best cell already exceeds `cap` can only grow."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def separation_edges(rows: list[dict[str, Any]],
                     index: dict[str, Any] | None = None
                     ) -> tuple[set[tuple[str, str]], list[str]]:
    """The DIRECTED graph of boundaries the catalog actually draws.

    An edge (a, b) means row `a` names `b` in its `distinct_from`. A reference is
    resolved the way a READER would resolve it - through the name index, not only
    through the exact catalog id. The ladder is: catalog id, then exact canonical,
    then the index (which also keys bare canonical names, aliases and grounding
    tokens). This is the indexing half paying for the separation half: an author who
    writes the name a concept is actually known by gets a real boundary, not a dangle.

    Returns (edges, unresolved). `unresolved` describes the two ways a reference can
    fail to name one concept:
      dangling   - it points at nothing, sending a reader to compare against a
                   concept that is not in the catalog, which is worse than silence;
      ambiguous  - it points at several concepts at once, so which boundary is being
                   drawn is exactly the question the reference was supposed to answer.
    """
    ids = {r.get("id") for r in rows if _nonempty(r.get("id"))}
    by_canon = {norm_token(r.get("canonical", "")): r.get("id") for r in rows
                if _nonempty(r.get("canonical"))}
    by_key = (index or {}).get("by_key") or {}
    edges: set[tuple[str, str]] = set()
    unresolved: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id") if _nonempty(r.get("id")) else f"row[{i}]"
        for ref in r.get("distinct_from") or []:
            if not _nonempty(ref):
                continue
            target = ref if ref in ids else by_canon.get(norm_token(ref))
            if target is None:
                landing = [t for t in by_key.get(norm_token(bare_name(ref)), []) if t != rid]
                if len(landing) == 1:
                    target = landing[0]
                elif len(landing) > 1:
                    unresolved.append(
                        f"{rid}: distinct_from {ref!r} names {len(landing)} concepts at once "
                        f"({', '.join(landing[:4])}{'...' if len(landing) > 4 else ''}) - "
                        f"an ambiguous boundary; use the catalog id of the one you mean")
                    continue
            if target is None:
                unresolved.append(f"{rid}: distinct_from {ref!r} resolves to no catalog id, "
                                  f"canonical name or index key - a dangling boundary")
            elif target != rid:
                edges.add((rid, target))
    return edges, unresolved


def shared_affix(a: str, b: str) -> int:
    """The longer of the two names' common leading / trailing run, in characters."""
    head = 0
    for x, y in zip(a, b):
        if x != y:
            break
        head += 1
    tail = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        tail += 1
    return max(head, tail)


def _character_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in value:
        counts[char] = counts.get(char, 0) + 1
    return counts


def _character_multiset_may_be_within(a: dict[str, int], b: dict[str, int],
                                       max_length: int, cap: int) -> bool:
    """Conservative Levenshtein lower bound used only to reject candidates."""
    if len(a) > len(b):
        a, b = b, a
    common = sum(min(count, b.get(char, 0)) for char, count in a.items())
    return max_length - common <= cap


PAIR_KINDS = ("homonym", "permuted", "near")   # most confusable first
_PAIR_RANK = {k: i for i, k in enumerate(PAIR_KINDS)}


def confusable_pairs(rows: list[dict[str, Any]], *, indexed: bool = True,
                     _stats: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """The pairs of DISTINCT concepts whose names a reader cannot tell apart.

    Discovered from the catalog's own names - not declared - so the pair list grows
    by itself the moment a peer lands a near-twin name, and it cannot be shrunk by
    editing the data (dropping a row does not make the tree token go away; it makes
    coverage fall instead).

      homonym   the bare names are THE SAME once the catalog's parenthetical gloss is
                stripped - `Decision (kernel)` and `Decision (scheduler)` are both just
                `Decision` in the tree. Legitimate across Go packages, and precisely
                why the reader who meets the bare word needs to be handed both.
      permuted  same word multiset, different spelling (`witnessPath`/`PathWitness`)
      near      within MAX_PAIR_EDITS edits AND sharing a long head/tail
                (`SessionRef`/`SessionRow`, `kernel`/`kernelkv`, `CacheTTL5m`/`CacheTTL1h`)

    The strongest kind wins when a pair qualifies as several. Returns a deterministic
    list sorted by (kind, a, b)."""
    named = [r for r in rows if _nonempty(r.get("id")) and _nonempty(r.get("canonical"))]
    prepared = [(ordinal, r, norm_token(bare_name(r.get("canonical"))),
                 split_words(r.get("canonical")))
                for ordinal, r in enumerate(named)]
    found: dict[tuple[str, str], dict[str, Any]] = {}

    def _add(a: str, b: str, kind: str, why: str) -> None:
        key = (a, b) if a < b else (b, a)
        cur = found.get(key)
        if cur is None or _PAIR_RANK[kind] < _PAIR_RANK[cur["kind"]]:
            found[key] = {"a": key[0], "b": key[1], "kind": kind, "why": why}

    # homonym: bucket by the BARE normalized name. The gloss in `Decision (kernel)` is
    # the catalog explaining itself to a reader; the tree only ever says `Decision`.
    by_bare: dict[str, list[str]] = {}
    for _ordinal, r, tok, _words in prepared:
        if tok:
            by_bare.setdefault(tok, []).append(r["id"])
    for tok, ids in by_bare.items():
        if len(ids) < 2:
            continue
        for i, a in enumerate(sorted(ids)):
            for b in sorted(ids)[i + 1:]:
                _add(a, b, "homonym", f"both are plainly named {tok!r} once the gloss is stripped")

    # permuted: bucket by the sorted word multiset; anything sharing a bucket is a
    # rearrangement of the same words.
    by_words: dict[tuple[str, ...], list[str]] = {}
    for _ordinal, r, _tok, ws in prepared:
        if len(ws) >= 2:
            by_words.setdefault(tuple(sorted(ws)), []).append(r["id"])
    for ws, ids in by_words.items():
        if len(ids) < 2:
            continue
        for i, a in enumerate(sorted(ids)):
            for b in sorted(ids)[i + 1:]:
                _add(a, b, "permuted", f"same words in a different order: {' '.join(ws)}")

    # near: bucket normalized names by length; a pair within `cap` edits cannot differ
    # in length by more than `cap`, so only neighbouring buckets need comparing.
    by_len: dict[int, list[tuple[int, str, str]]] = {}
    near_by_ordinal: dict[int, tuple[str, str]] = {}
    affix_buckets: dict[tuple[str, str, int], list[int]] = {}
    indexed_tokens: dict[int, str] = {}
    indexed_counts: dict[int, dict[str, int]] = {}
    candidates: set[tuple[int, int]] | None = set() if indexed else None
    for ordinal, r, tok, _words in prepared:
        if len(tok) >= MIN_PAIR_LEN:
            by_len.setdefault(len(tok), []).append((ordinal, r["id"], tok))
            near_by_ordinal[ordinal] = (r["id"], tok)
            if candidates is not None:
                prior: set[int] = set()
                for candidate_length in range(len(tok) - MAX_PAIR_EDITS,
                                              len(tok) + MAX_PAIR_EDITS + 1):
                    prior.update(affix_buckets.get(
                        ("head", tok[:MIN_SHARED_AFFIX], candidate_length), ()))
                    prior.update(affix_buckets.get(
                        ("tail", tok[-MIN_SHARED_AFFIX:], candidate_length), ()))
                counts = _character_counts(tok)
                candidates.update(
                    (other, ordinal) for other in prior
                    if _character_multiset_may_be_within(
                        indexed_counts[other], counts,
                        max(len(indexed_tokens[other]), len(tok)), MAX_PAIR_EDITS))
                indexed_tokens[ordinal] = tok
                indexed_counts[ordinal] = counts
                affix_buckets.setdefault(
                    ("head", tok[:MIN_SHARED_AFFIX], len(tok)), []).append(ordinal)
                affix_buckets.setdefault(
                    ("tail", tok[-MIN_SHARED_AFFIX:], len(tok)), []).append(ordinal)
    if _stats is not None:
        _stats["near_legacy_pairs"] = sum(
            len(bucket) * (len(bucket) - 1) // 2 +
            len(bucket) * sum(len(by_len.get(length + delta, ()))
                              for delta in range(1, MAX_PAIR_EDITS + 1))
            for length, bucket in by_len.items())
        _stats["near_candidate_pairs"] = (
            len(candidates) if candidates is not None else _stats["near_legacy_pairs"])
        _stats["near_distance_checks"] = 0

    def _consider_near(ida: str, ta: str, idb: str, tb: str) -> None:
        if ida == idb or ta == tb:
            return
        affix = shared_affix(ta, tb)
        if affix < MIN_SHARED_AFFIX:
            return
        if _stats is not None:
            _stats["near_distance_checks"] += 1
        distance = edit_distance_within(ta, tb, MAX_PAIR_EDITS)
        if distance <= MAX_PAIR_EDITS:
            _add(ida, idb, "near",
                 f"{distance} edit(s) apart sharing {affix} characters: {ta} / {tb}")

    if candidates is not None:
        for left_ordinal, right_ordinal in sorted(candidates):
            ida, ta = near_by_ordinal[left_ordinal]
            idb, tb = near_by_ordinal[right_ordinal]
            if len(ta) > len(tb):
                ida, ta, idb, tb = idb, tb, ida, ta
            _consider_near(ida, ta, idb, tb)
    else:
        for length in sorted(by_len):
            left = by_len[length]
            right = [x for delta in range(1, MAX_PAIR_EDITS + 1)
                     for x in by_len.get(length + delta, [])]
            for i, (_oa, ida, ta) in enumerate(left):
                for _ob, idb, tb in left[i + 1:] + right:
                    _consider_near(ida, ta, idb, tb)
    return sorted(found.values(), key=lambda p: (_PAIR_RANK[p["kind"]], p["a"], p["b"]))


def separation_report(rows: list[dict[str, Any]], edges: set[tuple[str, str]],
                      pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Grade every discovered confusable pair against the drawn boundaries.

      mutual    both rows name each other - a reader arriving from EITHER side is told
      one_sided exactly one direction is drawn - half the readers still collide
      undrawn   neither names the other - the twin names sit in the catalog unseparated

    Also folds the whole-graph mutuality view (all boundaries, not just confusable
    pairs), which is advisory: a general one-sided edge is a nudge, an unseparated
    TWIN is debt."""
    byid = {r.get("id"): r for r in rows if _nonempty(r.get("id"))}
    graded: list[dict[str, Any]] = []
    counts = {"mutual": 0, "one_sided": 0, "undrawn": 0}
    for p in pairs:
        a, b = p["a"], p["b"]
        ab, ba = (a, b) in edges, (b, a) in edges
        state = "mutual" if (ab and ba) else ("one_sided" if (ab or ba) else "undrawn")
        counts[state] += 1
        graded.append({**p, "state": state, "a_to_b": ab, "b_to_a": ba,
                       "a_canonical": (byid.get(a) or {}).get("canonical"),
                       "b_canonical": (byid.get(b) or {}).get("canonical"),
                       "a_family": (byid.get(a) or {}).get("family"),
                       "b_family": (byid.get(b) or {}).get("family")})
    one_way = sorted(f"{a} -> {b}" for (a, b) in edges if (b, a) not in edges)
    return {
        "pairs": graded, "counts": counts, "discovered": len(graded),
        "separated": counts["mutual"] + counts["one_sided"],
        "edges": len(edges),
        "mutual_edges": sum(1 for (a, b) in edges if (b, a) in edges),
        "one_way_edges": one_way,
    }


# ---------------------------------------------------------------------------
# INDEXING - can a reader who meets a NAME find the concept it denotes?
#
# The catalog is organised by concept; a reader meets a SPELLING. The index turns
# it around: every surface a name can arrive as - the canonical, every alias, the
# grounding token - becomes a lookup key pointing at the concept(s) it denotes,
# carrying that concept's contrast set so the answer to "which one is this?" and
# "how is it not the other one?" arrive together.
#
# A key that lands on TWO concepts is an index defect, not a catalog defect: the
# canonical names may be perfectly distinct while one row quietly claims the other's
# name as an alias, so the lookup is ambiguous even though `canonical_unique` is
# clean. That is only acceptable when the two concepts separate from each other -
# then the index can honestly answer "both, and here is the difference".
# ---------------------------------------------------------------------------

def build_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The NAME -> CONCEPT lookup index over every surface (canonical / alias /
    grounding). Deterministic: entries sorted by key, targets sorted by id."""
    entries: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(rows):
        rid = r.get("id") if _nonempty(r.get("id")) else f"row[{i}]"
        surfaces: list[tuple[str, str]] = []
        for name in [r.get("canonical")] + list(r.get("aliases") or []) + [r.get("grounding")]:
            if not _nonempty(name):
                continue
            source = ("canonical" if name == r.get("canonical")
                      else ("grounding" if name == r.get("grounding") else "alias"))
            surfaces.append((name.strip(), source))
        for name, source in surfaces:
            key = norm_token(bare_name(name)) or norm_token(name)
            if not key:
                continue
            e = entries.setdefault(key, {"key": key, "spellings": set(), "targets": {}})
            e["spellings"].add(name)
            e["targets"].setdefault(rid, set()).add(source)
    out: list[dict[str, Any]] = []
    for key in sorted(entries):
        e = entries[key]
        out.append({
            "key": key,
            "spellings": sorted(e["spellings"]),
            "targets": sorted(e["targets"]),
            "via": {rid: sorted(src) for rid, src in sorted(e["targets"].items())},
            "ambiguous": len(e["targets"]) > 1,
        })
    ambiguous = [e for e in out if e["ambiguous"]]
    return {"entries": out, "keys": len(out), "ambiguous": ambiguous,
            "ambiguous_keys": len(ambiguous),
            "by_key": {e["key"]: list(e["targets"]) for e in out}}


def index_pairs(index: dict[str, Any],
                already: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pairs discovered by LOOKUP rather than by spelling: two concepts answer to the
    same name, so a reader who meets that name lands on both.

    An ambiguous lookup key is not a defect by itself - `Decision` is genuinely the
    name of four different Go types, and the catalog does not get to rename them. It
    is a defect when the concepts it lands on are not separated FROM EACH OTHER,
    because then the index hands the reader a fog instead of a choice. Pairs already
    discovered by spelling are left to the spelling checks so one missing boundary is
    never counted as two defects."""
    seen = {(p["a"], p["b"]) for p in already} | {(p["b"], p["a"]) for p in already}
    out: list[dict[str, Any]] = []
    emitted: set[tuple[str, str]] = set()
    for e in index.get("ambiguous") or []:
        targets = list(e["targets"])
        for a, b in itertools.combinations(targets, 2):
            if (a, b) in seen or (a, b) in emitted:
                continue
            emitted.add((a, b))
            emitted.add((b, a))
            via = sorted(set(e["via"].get(a, [])) | set(e["via"].get(b, [])))
            out.append({"a": a, "b": b, "kind": "shared-name",
                        "why": (f"both answer to the lookup name {e['key']!r} "
                                f"(via {'/'.join(via)}; {len(targets)} concepts share it)")})
    return sorted(out, key=lambda p: (p["a"], p["b"]))


def unseparated_pairs(sep: dict[str, Any], ipairs: list[dict[str, Any]],
                      edges: set[tuple[str, str]]) -> list[dict[str, Any]]:
    """Every pair a reader can confuse whose boundary is not yet drawn BOTH ways,
    from either discovery route (spelling and lookup), in one machine-readable list.

    This is the "what would I have to write" view, and it is what the authoring path
    consumes: `fak concept position` refuses to land a name that collides with an
    existing one until the new row and its twin each say what the other is not. The
    rule lives here, once, so the authoring gate and the scorecard can never drift
    apart into two different definitions of confusable."""
    out: list[dict[str, Any]] = []
    for p in sep.get("pairs") or []:
        if p.get("state") != "mutual":
            out.append({"a": p["a"], "b": p["b"], "kind": p.get("kind", ""),
                        "why": p.get("why", ""), "state": p.get("state", ""),
                        "a_to_b": bool(p.get("a_to_b")), "b_to_a": bool(p.get("b_to_a"))})
    for p in ipairs:
        ab, ba = (p["a"], p["b"]) in edges, (p["b"], p["a"]) in edges
        if ab and ba:
            continue
        out.append({"a": p["a"], "b": p["b"], "kind": p["kind"], "why": p["why"],
                    "state": "one_sided" if (ab or ba) else "undrawn",
                    "a_to_b": ab, "b_to_a": ba})
    return sorted(out, key=lambda p: (p["a"], p["b"]))


def index_contrast(rows: list[dict[str, Any]], edges: set[tuple[str, str]],
                   pairs: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Per concept, the CONTRAST set the index shows a reader: every concept it
    declares a boundary against, plus every confusable twin discovered by name. The
    union is what "not to be confused with" must list for the index to be useful."""
    out: dict[str, set[str]] = {}
    for a, b in edges:
        out.setdefault(a, set()).add(b)
    for p in pairs:
        out.setdefault(p["a"], set()).add(p["b"])
        out.setdefault(p["b"], set()).add(p["a"])
    return {k: sorted(v) for k, v in out.items()}


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
    score = _clamp(100 - pen * len(defects) - min(10, 2 * len(soft)))
    return {"kpi": name, "group": KPI_GROUP[name],
            "score": score, "value": round(score / 100, 3),
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


def kpi_reference_resolves(unresolved: list[str]) -> dict[str, Any]:
    """EVERY `distinct_from` reference must resolve to a real catalog entry.

    `kpi_disambiguated` only needs ONE reference to resolve, so a row listing two
    siblings passes while the second silently dangles. A dangling boundary is worse
    than a missing one: it sends the reader to compare against a concept that is not
    there, and it makes the separation graph under-count the lines actually drawn."""
    return _kpi("reference_resolves", list(unresolved),
                "every distinct_from reference resolves to a real concept",
                bad_detail=f"{len(unresolved)} dangling distinct_from reference(s)")


def kpi_pair_separated(sep: dict[str, Any]) -> dict[str, Any]:
    """THE pairwise test: for every pair of concepts whose NAMES are confusable,
    the catalog must draw the line between THOSE TWO - in at least one direction.

    This is what per-row disambiguation cannot see. A row in a 250-member family
    satisfies `disambiguated` by naming any one sibling; this asks whether the
    sibling it is genuinely mistakable for is the one it separates from."""
    defects = [
        f"{p['a']}: confusable with {p['b']} ({p['why']}) but neither names the other - "
        f"add {p['b']} to {p['a']}'s distinct_from (and back)"
        for p in sep["pairs"] if p["state"] == "undrawn"
    ]
    return _kpi("pair_separated", defects,
                f"all {sep['discovered']} confusable name-pair(s) are separated",
                bad_detail=f"{len(defects)} confusable pair(s) with no line drawn")


def kpi_pair_mutual(sep: dict[str, Any]) -> dict[str, Any]:
    """A boundary is DIRECTED, and for a confusable pair it must be drawn from BOTH
    sides. `A.distinct_from = [B]` only helps a reader who arrived at A; one who
    arrived at B - just as likely, since the names are twins - is told nothing. Only
    confusable pairs are held to this bar; general one-way edges stay advisory."""
    defects = []
    for p in sep["pairs"]:
        if p["state"] != "one_sided":
            continue
        src, dst = (p["a"], p["b"]) if p["a_to_b"] else (p["b"], p["a"])
        defects.append(f"{dst}: {src} separates from it, but it does not separate back "
                       f"({p['why']}) - a reader arriving at {dst} is not warned")
    return _kpi("pair_mutual", defects,
                "every confusable pair draws its line from both sides",
                bad_detail=f"{len(defects)} one-sided boundary on a confusable pair")


def kpi_index_resolves(ipairs: list[dict[str, Any]], rows: list[dict[str, Any]],
                       edges: set[tuple[str, str]], index: dict[str, Any]) -> dict[str, Any]:
    """A lookup name that lands on several concepts must land on a CHOICE.

    Distinct from `canonical_unique`, which only guards canonical names: this guards
    the whole lookup surface, so it also catches one row claiming another row's name
    as an ALIAS, or two rows grounded on the same tree token. Those names are then
    genuinely ambiguous even though both canonicals are unique.

    Ambiguity is not by itself a defect - `Decision` really is the name of four
    different Go types and the catalog does not get to rename them. It is a defect
    when the concepts sharing the name do not separate from each other, because then
    the index answers the reader's question with a fog instead of "both, and here is
    the difference". Pairs already discovered by spelling belong to the separation
    checks; this KPI owns only the ones that lookup alone reveals."""
    byid = {r.get("id"): r for r in rows if _nonempty(r.get("id"))}
    defects: list[str] = []
    for p in ipairs:
        if (p["a"], p["b"]) in edges and (p["b"], p["a"]) in edges:
            continue
        a, b = p["a"], p["b"]
        defects.append(f"{a}: shares a lookup name with {b} "
                       f"({(byid.get(a) or {}).get('canonical')} vs "
                       f"{(byid.get(b) or {}).get('canonical')}) - {p['why']} - but they do "
                       f"not separate from each other; add {b} to {a}'s distinct_from (and back)")
    return _kpi("index_resolves", defects,
                f"every one of {index['keys']} lookup name(s) resolves - "
                f"{index['ambiguous_keys']} land on several concepts, all separated",
                bad_detail=f"{len(defects)} unresolvable shared lookup name(s)")


def kpi_mutuality_soft(sep: dict[str, Any]) -> dict[str, Any]:
    """SOFT: across the WHOLE boundary graph (not only the confusable pairs), how many
    lines are drawn from one side only. Advisory by construction - demanding mutuality
    of every one of thousands of edges would be noise, and the pairs where it actually
    bites are already hard debt in `kpi_pair_mutual`. Reported so the trend is visible."""
    one_way = sep["one_way_edges"]
    score = _clamp(100 - min(20, len(one_way) // 50))
    detail = ("every boundary is drawn from both sides" if not one_way else
              f"{len(one_way)}/{sep['edges']} boundaries drawn one-way only")
    return {"kpi": "mutuality_soft", "group": "separation",
            "score": score, "value": round(score / 100, 3),
            "detail": detail, "defects": [], "soft": one_way[:12]}


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


def entangled_rows(sep: dict[str, Any], edges: set[tuple[str, str]],
                   extra_pairs: list[dict[str, Any]] | None = None) -> dict[str, str]:
    """Concepts that do not separate from a twin their own NAME is mistakable for.

    A row is entangled when it takes part in a confusable pair and does not itself
    name the other side. That covers both the pair nobody drew and the receiving end
    of a one-sided line - in both cases a reader who arrives HERE is not warned. The
    only stable fix is to draw the boundary, which is exactly the intended incentive:
    the verdict cannot be cleared by re-labelling the row."""
    out: dict[str, str] = {}
    for p in list(sep["pairs"]) + list(extra_pairs or []):
        for me, other in ((p["a"], p["b"]), (p["b"], p["a"])):
            if (me, other) not in edges and me not in out:
                out[me] = f"names no boundary against its twin {other} ({p['why']})"
    return out


def expected_verdict(row: dict[str, Any], *, colliding: bool, exists: Callable[[str], bool],
                     sizes: dict[str, int], entangled: str = "") -> tuple[str, str]:
    """The clarity verdict the evidence implies, worst-first. Grounding is graded
    separately (kpi_grounded) so it is not double-charged here.

      undocumented  no definition at all
      colliding     shares a canonical name with another concept
      entangled     defined and draws SOME line, but not against the twin its own name
                    is confusable with - per-row clean, pairwise still fog
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
    if entangled:
        return "entangled", entangled
    anchor = row.get("glossary_anchor", "")
    if not _nonempty(anchor) or not exists(anchor):
        return "defined", "boundary drawn but not written in a discoverable doc"
    return "crystal", "defined + distinction + anchored on disk"


def kpi_clarity_consistent(rows: list[dict[str, Any]], colliding_ids: set[str],
                           exists: Callable[[str], bool], sizes: dict[str, int],
                           entangled: dict[str, str] | None = None,
                           expected_by_row: dict[int, tuple[str, str]] | None = None
                           ) -> dict[str, Any]:
    """The stated verdict must match what the evidence implies. Calling a drifting
    concept 'crystal', a colliding one 'defined', or an ENTANGLED one either, is the
    overclaim this catches - the same self-report refusal the rest of the repo runs."""
    entangled = entangled or {}
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        declared = r.get("verdict")
        expected = expected_by_row.get(id(r)) if expected_by_row is not None else None
        exp, why = expected or expected_verdict(
            r, colliding=(rid in colliding_ids), exists=exists,
            sizes=sizes, entangled=entangled.get(rid, ""))
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
    score = _clamp(100 - min(40, 6 * len(soft)))
    return {"kpi": "kind_grounding_soft", "group": "honesty",
            "score": score, "value": round(score / 100, 3),
            "detail": "kind agrees with grounding" if not soft else f"{len(soft)} kind/grounding mismatch",
            "defects": [], "soft": soft}


# ---------------------------------------------------------------------------
# Hierarchical roll-up - fold the optional `parent` forest into an honest
# higher-level view so an operator can read the namespace at the abstraction
# heads instead of one leaf at a time. The fold is WEAKEST-LINK: an abstraction
# is only as crystal-clear as its foggiest descendant. No averaging hides a
# drifting leaf, and it is ungameable - it is derived from the same cross-checked
# per-row verdicts, so a head cannot roll up to crystal until every concept
# beneath it actually is.
# ---------------------------------------------------------------------------

def resolve_parents(rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, list[str]], list[str]]:
    """Resolve the optional `parent` edges into a forest. Keeps every edge whose
    parent resolves to a real catalog id (subtree traversal below is cycle-safe, so
    a cycle is reported but not silently dropped). Returns (parent_map, children_map,
    soft_issues) where soft_issues names each unresolved / self / cyclic edge."""
    ids = {r.get("id") for r in rows if _nonempty(r.get("id"))}
    parent: dict[str, str] = {}
    soft: list[str] = []
    for r in rows:
        rid = r.get("id")
        p = r.get("parent")
        if not _nonempty(p):
            continue
        if p not in ids:
            soft.append(f"{rid}: parent '{p}' resolves to no catalog id")
        elif p == rid:
            soft.append(f"{rid}: parent points at itself")
        else:
            parent[rid] = p
    for start in list(parent):  # report (do not remove) cycles; traversal guards itself.
        seen, cur = set(), start
        while cur in parent:
            if cur in seen:
                soft.append(f"{start}: parent chain cycles")
                break
            seen.add(cur)
            cur = parent[cur]
    children: dict[str, list[str]] = {}
    for c, p in parent.items():
        children.setdefault(p, []).append(c)
    for p in children:
        children[p].sort()
    return parent, children, soft


def _subtree_ids(root: str, children: dict[str, list[str]]) -> set[str]:
    """Every id in `root`'s subtree, including `root`. Cycle-safe (a `seen` set), so a
    malformed parent cycle can never loop the fold."""
    seen: set[str] = set()
    stack = [root]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        for k in children.get(x, ()):  # deterministic: children are pre-sorted.
            if k not in seen:
                stack.append(k)
    return seen


def _verdict_rank(v: Any) -> int:
    return VERDICT_RANK.get(v, len(VERDICTS))  # an unknown verdict is treated as foggiest.


def abstraction_rollup(rows: list[dict[str, Any]], row_debt: dict[str, int],
                       parent: dict[str, str], children: dict[str, list[str]]) -> list[dict[str, Any]]:
    """One record per abstraction HEAD (a concept with >= 1 child), summarizing its
    whole subtree WEAKEST-LINK:

      rolled_verdict  the WORST verdict anywhere in the subtree (incl. the head) - the
                      honest 'is this whole abstraction crystal-clear?' answer.
      verdict_mix     how many concepts sit at each verdict beneath the head.
      subtree_debt    the clarity-debt summed over the subtree (0 in the clean state).
      weakest         the foggiest concept in the subtree (the drill-down target).
      overclaim       the head's DECLARED verdict reads clearer than the subtree supports
                      (a descendant is worse than the head claims) - abstraction overclaim.

    Sorted worst-first: heaviest subtree-debt, then foggiest roll-up, then largest subtree."""
    byid = {r.get("id"): r for r in rows if _nonempty(r.get("id"))}
    recs: list[dict[str, Any]] = []
    for head in children:
        if head not in byid:
            continue
        subtree = _subtree_ids(head, children)
        mix = {v: 0 for v in VERDICTS}
        debt = 0
        worst_id, worst_rank = head, -1
        for n in sorted(subtree):
            r = byid.get(n)
            if r is None:
                continue
            v = r.get("verdict")
            if v in mix:
                mix[v] += 1
            debt += row_debt.get(n, 0)
            if _verdict_rank(v) > worst_rank:
                worst_rank, worst_id = _verdict_rank(v), n
        rolled = VERDICTS[worst_rank] if 0 <= worst_rank < len(VERDICTS) else "undocumented"
        declared = byid[head].get("verdict")
        depth, cur = 0, head
        while cur in parent and depth <= len(rows):  # bounded even on a malformed cycle.
            depth += 1
            cur = parent[cur]
        recs.append({
            "id": head, "canonical": byid[head].get("canonical"), "family": byid[head].get("family"),
            "declared_verdict": declared, "rolled_verdict": rolled,
            "subtree_size": len(subtree), "child_count": len(children.get(head, [])),
            "subtree_debt": debt, "verdict_mix": mix, "depth": depth,
            "weakest": {"id": worst_id, "verdict": byid[worst_id].get("verdict")},
            "overclaim": _verdict_rank(declared) < worst_rank,
        })
    recs.sort(key=lambda a: (-a["subtree_debt"], -_verdict_rank(a["rolled_verdict"]),
                             -a["subtree_size"], a["id"] or ""))
    return recs


def roll_up(rows: list[dict[str, Any]], row_debt: dict[str, int]) -> dict[str, Any]:
    """Fold the parent forest into the payload roll-up: the per-head abstraction
    records, the count of top-level (depth-0) abstraction roots, the forest size, and
    the abstraction-overclaim list. Deterministic + pure (no disk, no tree)."""
    parent, children, _soft = resolve_parents(rows)
    recs = abstraction_rollup(rows, row_debt, parent, children)
    overclaims = [
        f"{a['id']}: abstraction declares '{a['declared_verdict']}' but rolls up to "
        f"'{a['rolled_verdict']}' (weakest: {a['weakest']['id']} = {a['weakest']['verdict']})"
        for a in recs if a["overclaim"]
    ]

    def _depth(node: str) -> int:  # chain length to a root; bounded even on a cycle.
        d, cur, guard = 0, node, 0
        while cur in parent and guard <= len(parent):
            d += 1
            cur = parent[cur]
            guard += 1
        return d

    all_nodes = set(parent) | set(children)
    return {
        "abstractions": recs,
        "heads": len(recs),
        "roots": sum(1 for a in recs if a["depth"] == 0),
        "forest_nodes": len(all_nodes),
        "max_depth": max((_depth(n) for n in all_nodes), default=0),  # leaf-inclusive forest depth
        "overclaims": overclaims,
    }


def kpi_hierarchy_soft(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """SOFT: the optional `parent` hierarchy should (a) resolve to a real catalog id and
    not cycle, and (b) not OVERCLAIM - a head marked crystal while a descendant still
    drifts means the abstraction, read at the head, is clearer than the subtree supports.
    Advisory - hierarchy is encouraged, not required, so this never becomes hard debt;
    the weakest-link roll-up (see `roll_up`) is what makes the higher-level view honest."""
    parent, children, soft = resolve_parents(rows)
    recs = abstraction_rollup(rows, {}, parent, children)
    soft = list(soft)
    for a in recs:
        if a["overclaim"]:
            soft.append(f"{a['id']}: rolls up to '{a['rolled_verdict']}' but head declares "
                        f"'{a['declared_verdict']}' (weakest descendant {a['weakest']['id']})")
    score = _clamp(100 - min(30, 6 * len(soft)))
    return {"kpi": "hierarchy_soft", "group": "honesty",
            "score": score, "value": round(score / 100, 3),
            "detail": "hierarchy resolves + rolls up honestly" if not soft else f"{len(soft)} hierarchy issue(s)",
            "defects": [], "soft": soft}


# ---------------------------------------------------------------------------
# Coverage - discover the confusable universe in the tree, then ask how much of it
# the catalog positions. This is the ungameable engine that lands the honest score.
# ---------------------------------------------------------------------------


def _presence(files_or_count: Any) -> int:
    return files_or_count if isinstance(files_or_count, int) else len(files_or_count)


class _CorpusFoldIndex:
    """Exact root-substring candidates shared by every watched family."""

    def __init__(self, corpus: dict[str, Any], stats: dict[str, int] | None = None):
        self.sym_files = corpus["sym_files"]
        self.structural = corpus["structural"]
        self.tokens = set(self.sym_files) | set(self.structural)
        self._root_hits: dict[str, tuple[str, ...]] = {}
        self._stats = stats
        if stats is not None:
            stats["coverage_index_tokens"] = len(self.tokens)
            stats["coverage_trigram_windows"] = 0
            stats["coverage_root_candidate_probes"] = 0
            stats["coverage_candidate_tokens"] = 0

    def prepare_roots(self, roots: set[str]) -> None:
        roots = roots - self._root_hits.keys()
        if not roots:
            return
        grams = {root[:3] for root in roots if len(root) >= 3}
        gram_hits: dict[str, list[str]] = {gram: [] for gram in grams}
        for token in self.tokens:
            if self._stats is not None:
                self._stats["coverage_trigram_windows"] += max(0, len(token) - 2)
            seen: set[str] = set()
            for offset in range(len(token) - 2):
                gram = token[offset:offset + 3]
                if gram in gram_hits and gram not in seen:
                    gram_hits[gram].append(token)
                    seen.add(gram)
        for root in roots:
            source = gram_hits.get(root[:3]) if len(root) >= 3 else None
            candidates = self.tokens if source is None else source
            if self._stats is not None:
                self._stats["coverage_root_candidate_probes"] += len(candidates)
            self._root_hits[root] = tuple(token for token in candidates if root in token)

    def root_hits(self, root: str) -> tuple[str, ...]:
        hits = self._root_hits.get(root)
        if hits is None:
            hits = tuple(token for token in self.tokens if root in token)
            self._root_hits[root] = hits
        return hits


def discover_family_tokens(family: dict[str, Any], corpus: dict[str, Any],
                           fold_index: _CorpusFoldIndex | None = None,
                           _stats: dict[str, int] | None = None) -> list[dict[str, Any]]:
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

    sym_files = corpus["sym_files"]
    structural: set[str] = corpus["structural"]
    found: dict[str, dict[str, Any]] = {}
    if fold_index is None:
        candidates = set(sym_files) | set(structural)
    else:
        candidates: set[str] = set()
        for root in roots:
            candidates.update(fold_index.root_hits(root))
    if _stats is not None:
        _stats["coverage_candidate_tokens"] = (
            _stats.get("coverage_candidate_tokens", 0) + len(candidates))
    for tok in candidates:
        if not _matches(tok):
            continue
        presence = _presence(sym_files.get(tok, 0))
        is_struct = tok in structural
        if presence >= min_files or is_struct:
            found[tok] = {"token": tok, "presence": presence,
                          "where": "dir/heading" if is_struct else f"{presence} files"}
    return sorted(found.values(), key=lambda d: (-d["presence"], d["token"]))


def coverage_report(families: list[dict[str, Any]], rows: list[dict[str, Any]],
                    corpus: dict[str, Any], *, indexed: bool = True,
                    _stats: dict[str, int] | None = None) -> dict[str, Any]:
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
    if _stats is not None:
        _stats["coverage_legacy_family_token_probes"] = (
            len(families) * len(set(corpus["sym_files"]) | set(corpus["structural"])))
    fold_index = _CorpusFoldIndex(corpus, _stats) if indexed else None
    if fold_index is not None:
        fold_index.prepare_roots({
            token for family in families for raw in (family.get("roots") or [])
            if _nonempty(raw) and (token := norm_token(raw))
        })
    for fam in families:
        fid = fam.get("id", "?")
        toks = discover_family_tokens(fam, corpus, fold_index, _stats)
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
                exists: Callable[[str], bool], sizes: dict[str, int],
                entangled: dict[str, str] | None = None,
                expected_by_row: dict[int, tuple[str, str]] | None = None
                ) -> list[dict[str, Any]]:
    entangled = entangled or {}
    out: list[dict[str, Any]] = []
    for r in rows:
        expected = expected_by_row.get(id(r)) if expected_by_row is not None else None
        exp, _ = expected or expected_verdict(
            r, colliding=(r.get("id") in colliding_ids), exists=exists, sizes=sizes,
            entangled=entangled.get(r.get("id"), ""))
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
             exists: Callable[[str], bool], doc_verbs: set[str], *,
              sep: dict[str, Any], unresolved: list[str], index: dict[str, Any],
              ipairs: list[dict[str, Any]], edges: set[tuple[str, str]],
              entangled: dict[str, str],
              expected_by_row: dict[int, tuple[str, str]] | None = None
              ) -> list[dict[str, Any]]:
    return [
        kpi_well_formed(rows, families),
        kpi_canonical_unique(rows),
        kpi_defined(rows),
        kpi_disambiguated(rows, sizes),
        kpi_reference_resolves(unresolved),
        kpi_pair_separated(sep),
        kpi_pair_mutual(sep),
        kpi_grounded(rows, in_tree),
        kpi_anchored(rows, exists),
        kpi_index_resolves(ipairs, rows, edges, index),
        kpi_clarity_consistent(rows, colliding_ids, exists, sizes, entangled,
                               expected_by_row),
        kpi_kind_grounding_soft(rows, doc_verbs),
        kpi_hierarchy_soft(rows),
        kpi_mutuality_soft(sep),
    ]


def build_payload(*, workspace: str, data: dict[str, Any] | None, tree: dict[str, Any],
                  error: str | None = None, indexed: bool = True) -> dict[str, Any]:
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
    if indexed:
        # One fold observes one immutable tree. Cache repeated anchor probes so the KPI,
        # verdict consistency check, and leaderboard consume exactly the same fact.
        raw_exists = exists
        exists_cache: dict[str, bool] = {}

        def exists(path: str) -> bool:
            if path not in exists_cache:
                exists_cache[path] = raw_exists(path)
            return exists_cache[path]

    colliding_ids = set(find_collisions(rows))
    sizes = cluster_sizes(rows)

    # Separation + index: the pairwise / lookup layers. Both are DERIVED from the
    # catalog's own names, so neither can be satisfied by adding prose to a row.
    index = build_index(rows)
    edges, unresolved = separation_edges(rows, index)
    pairs = confusable_pairs(rows, indexed=indexed)
    ipairs = index_pairs(index, pairs)
    sep = separation_report(rows, edges, pairs)
    contrast = index_contrast(rows, edges, pairs + ipairs)
    entangled = entangled_rows(sep, edges, ipairs)
    unseparated = unseparated_pairs(sep, ipairs, edges)
    expected_by_row = None
    if indexed:
        expected_by_row = {
            id(row): expected_verdict(
                row, colliding=(row.get("id") in colliding_ids), exists=exists,
                sizes=sizes, entangled=entangled.get(row.get("id"), ""))
            for row in rows
        }

    kpis = run_kpis(rows, families, colliding_ids, sizes, in_tree, exists, doc_verbs,
                    sep=sep, unresolved=unresolved, index=index, ipairs=ipairs,
                    edges=edges, entangled=entangled,
                    expected_by_row=expected_by_row)
    by_name = {k["kpi"]: k for k in kpis}
    clarity_score = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"]
                              for n in KPI_WEIGHTS if n in by_name), 1)
    clarity_defects = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)

    cov = coverage_report(fam_defs, rows, corpus, indexed=indexed)
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
          "value": round(k["score"] / 100, 3),
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    row_debt = per_row_debt(rows, kpis)
    pos = standing(rows)
    lb = leaderboard(rows, colliding_ids, exists, sizes, entangled, expected_by_row)
    crit = critical_backlog(rows, row_debt)
    rollup = roll_up(rows, row_debt)
    n_crystal = pos.get("crystal", 0)

    corpus_out = {
        "value": round(score / 100, 3), "value_unit": "quality_ratio",
        "score": score, "legacy_score": score, "legacy_score_scale": 100,
        "grade": grade,
        "clarity_score": clarity_score,
        "clarity_value": round(clarity_score / 100, 3),
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
        "rollup": rollup,
        "separation": {
            "confusable_pairs": sep["discovered"],
            "separated": sep["separated"],
            "mutual": sep["counts"]["mutual"],
            "one_sided": sep["counts"]["one_sided"],
            "undrawn": sep["counts"]["undrawn"],
            "entangled_concepts": len(entangled),
            "boundaries": sep["edges"],
            "mutual_boundaries": sep["mutual_edges"],
            "one_way_boundaries": len(sep["one_way_edges"]),
            "dangling_references": len(unresolved),
            "pairs": sep["pairs"],
            "unseparated": unseparated,
        },
        "index": {
            "keys": index["keys"],
            "ambiguous_keys": index["ambiguous_keys"],
            "shared_name_pairs": len(ipairs),
            "unresolved_shared_names": sum(
                1 for x in ipairs
                if (x["a"], x["b"]) not in edges or (x["b"], x["a"]) not in edges),
            "ambiguous": index["ambiguous"],
            "contrast_concepts": len(contrast),
        },
    }

    standing_line = (f"{pos['crystal']} crystal / {pos['defined']} defined / "
                     f"{pos['drifting']} drifting / {pos['entangled']} entangled / "
                     f"{pos['colliding']} colliding / {pos['undocumented']} undocumented")
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
        # The full lookup index rides OUTSIDE `corpus` so the control-pane payload stays
        # lean: `corpus.index` keeps the counts + the ambiguous entries, which is all a
        # dashboard needs, while the renderers get every entry from here.
        "_index": index,
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

_MARK = {"crystal": "*", "defined": "o", "drifting": "~", "entangled": "=",
         "colliding": "x", "undocumented": "."}


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
    sp = c.get("separation") or {}
    ix = c.get("index") or {}
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
         f"{pos.get('drifting', 0)} drifting - {pos.get('entangled', 0)} entangled - "
         f"{pos.get('colliding', 0)} colliding - {pos.get('undocumented', 0)} undocumented"),
        (f"separation: {sp.get('confusable_pairs', 0)} confusable name-pair(s) - "
         f"{sp.get('mutual', 0)} mutual, {sp.get('one_sided', 0)} one-sided, "
         f"{sp.get('undrawn', 0)} undrawn - {sp.get('boundaries', 0)} boundaries drawn "
         f"({sp.get('one_way_boundaries', 0)} one-way); see --pairs"),
        (f"index: {ix.get('keys', 0)} lookup name(s) -> {c.get('rows', 0)} concept(s) - "
         f"{ix.get('ambiguous_keys', 0)} shared by several, "
         f"{ix.get('unresolved_shared_names', 0)} of those unseparated; see --index"),
        ("debt by group: " + "  ".join(
            f"{g}:{(c.get('debt_by_group') or {}).get(g, 0)}" for g in GROUPS)),
        (f"roll-up: {(c.get('rollup') or {}).get('roots', 0)} top-level abstraction(s) over "
         f"{(c.get('rollup') or {}).get('forest_nodes', 0)} concepts (weakest-link) - "
         f"{len((c.get('rollup') or {}).get('overclaims') or [])} overclaim(s); see --rollup"),
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


def render_pairs(payload: dict[str, Any]) -> str:
    """The pairwise separation backlog: every pair of concepts whose NAMES a reader
    cannot keep apart, worst-first (undrawn, then one-sided, then mutual)."""
    c = payload.get("corpus") or {}
    sp = c.get("separation") or {}
    order = {"undrawn": 0, "one_sided": 1, "mutual": 2}
    mark = {"undrawn": "x", "one_sided": ">", "mutual": "="}
    lines = [
        (f"concept-disambiguation pairwise separation: {sp.get('confusable_pairs', 0)} "
         f"confusable name-pair(s) discovered - {sp.get('undrawn', 0)} undrawn, "
         f"{sp.get('one_sided', 0)} one-sided, {sp.get('mutual', 0)} mutual"),
        "",
        "A pair is CONFUSABLE when its two names collide by spelling alone, in one of three",
        "ways: homonym (identical once the parenthetical gloss is stripped), permuted (the",
        "same words in a different order), or near (within "
        f"{MAX_PAIR_EDITS} edits AND sharing a head or",
        f"tail run of {MIN_SHARED_AFFIX}+ characters, so a bare family root is not enough).",
        "Pairs are DISCOVERED from the names themselves, never declared, so landing a",
        "near-twin name raises the bar automatically. Being clean per-row does not separate",
        "a pair: a concept in a large family satisfies `disambiguated` by naming ANY sibling,",
        "while the twin it is genuinely mistakable for goes undrawn.",
        "",
        "  x undrawn    neither names the other - the twins sit in the catalog unseparated",
        "  > one-sided  drawn from one side only - a reader arriving at the other is unwarned",
        "  = mutual     both name each other - a reader arriving from EITHER side is told",
        "",
    ]
    pairs = sorted(sp.get("pairs") or [],
                   key=lambda p: (order.get(p["state"], 9), p["kind"], p["a"], p["b"]))
    if not pairs:
        lines.append("  (no confusable name-pairs discovered - every canonical name is far")
        lines.append("   from every other; the namespace separates itself by spelling alone)")
        return "\n".join(lines)
    for p in pairs:
        arrow = ("<->" if p["state"] == "mutual"
                 else ("-->" if p.get("a_to_b") else ("<--" if p.get("b_to_a") else "   ")))
        lines.append(f"  {mark.get(p['state'], ' ')} [{p['kind']:<8}] {p.get('a_canonical')!r} "
                     f"{arrow} {p.get('b_canonical')!r}")
        lines.append(f"      {p['why']}   ({p['a']} / {p['b']})")
    lines.append("")
    lines.append(f"whole-graph mutuality: {sp.get('mutual_boundaries', 0)}/"
                 f"{sp.get('boundaries', 0)} boundaries are drawn from both sides "
                 f"({sp.get('one_way_boundaries', 0)} one-way; advisory outside confusable pairs)")
    lines.append(f"dangling references: {sp.get('dangling_references', 0)}")
    return "\n".join(lines)


def _index_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Join the raw index entries with each target's catalog facts + contrast set, so
    one lookup answers BOTH 'which concept is this?' and 'how is it not the other one?'."""
    c = payload.get("corpus") or {}
    byid = {r.get("id"): r for r in c.get("leaderboard") or []}
    contrast: dict[str, list[str]] = {}
    for p in (c.get("separation") or {}).get("pairs") or []:
        contrast.setdefault(p["a"], []).append(p["b"])
        contrast.setdefault(p["b"], []).append(p["a"])
    out: list[dict[str, Any]] = []
    for e in (payload.get("_index") or {}).get("entries") or []:
        targets = []
        for tid in e["targets"]:
            r = byid.get(tid) or {}
            # The other concepts this very key lands on are twins by LOOKUP, whether or
            # not their spellings are close - so the entry warns even before the
            # boundary is drawn.
            twins = sorted(set(contrast.get(tid) or []) | (set(e["targets"]) - {tid}))
            not_confused = sorted(set((r.get("distinct_from") or []) + twins))
            targets.append({"id": tid, "canonical": r.get("canonical"),
                            "family": r.get("family"), "kind": r.get("kind"),
                            "verdict": r.get("verdict"), "definition": r.get("definition"),
                            "anchor": r.get("glossary_anchor", ""),
                            "not_to_be_confused_with": not_confused,
                            "via": e["via"].get(tid, [])})
        out.append({**e, "targets_full": targets})
    return out


def index_lookup(rows: list[dict[str, Any]], name: str) -> tuple[list[dict[str, Any]], bool]:
    """Resolve one spelling against the index the way a reader arrives at it: the same
    normalization the index was built with, so `KV cache`, `kv_cache` and `kvCache` are
    one question. Falls back to substring matches, flagged as such, rather than
    answering "no" to a name the catalog nearly holds."""
    key = norm_token(bare_name(name)) or norm_token(name)
    if not key:
        return [], False
    exact = [e for e in rows if e["key"] == key]
    if exact:
        return exact, True
    return [e for e in rows if key in e["key"] or e["key"] in key], False


def render_index(payload: dict[str, Any], *, limit: int = 0, lookup: str = "") -> str:
    """The NAME -> CONCEPT lookup index. A reader who meets any spelling - the canonical
    name, an alias, or the raw grounding token - finds the one concept it denotes and
    the concepts it must not be confused with."""
    c = payload.get("corpus") or {}
    ix = c.get("index") or {}
    rows = _index_rows(payload)
    if lookup:
        hits, exact = index_lookup(rows, lookup)
        if not hits:
            return (f"no lookup name matches {lookup!r} among {ix.get('keys', 0)} name(s) "
                    f"over {c.get('rows', 0)} concept(s).\n"
                    "The name is unpositioned: `--gaps` lists the tree tokens still waiting "
                    "for a concept, and `fak concept position` lands one.")
        rows = hits
        lines = [
            (f"{lookup!r} -> {len(rows)} lookup name(s)"
             + ("" if exact else " (no exact match; nearest keys)")),
            "",
        ]
    else:
        lines = [
            (f"concept-disambiguation name index: {ix.get('keys', 0)} lookup name(s) over "
             f"{c.get('rows', 0)} concept(s) - {ix.get('ambiguous_keys', 0)} ambiguous"),
            "",
            "Every surface a name can arrive as (canonical / alias / grounding token) resolves",
            "to the concept it denotes plus that concept's contrast set. '!' marks a name that",
            "lands on more than one concept - a LOOKUP ambiguity the canonical names do not show.",
            "",
        ]
    shown = rows[:limit] if limit else rows
    for e in shown:
        flag = "!" if e["ambiguous"] else " "
        lines.append(f"{flag} {e['spellings'][0]}"
                     + (f"   (also: {', '.join(e['spellings'][1:])})" if len(e["spellings"]) > 1 else ""))
        for t in e["targets_full"]:
            lines.append(f"    -> {t['canonical']}  [{t['family']} / {t['kind']} / "
                         f"{t['verdict']}]  via {'+'.join(t['via'])}")
            if t.get("definition"):
                lines.append(f"       {t['definition']}")
            if t["not_to_be_confused_with"]:
                lines.append(f"       not to be confused with: "
                             f"{', '.join(t['not_to_be_confused_with'])}")
    if limit and len(rows) > limit:
        lines.append(f"  ... and {len(rows) - limit} more name(s) (omit --limit for all)")
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
        (f"pairs undrawn: {(b.get('separation') or {}).get('undrawn', 0)} -> "
         f"{(cur.get('separation') or {}).get('undrawn', 0)}   "
         f"(mutual {(b.get('separation') or {}).get('mutual', 0)} -> "
         f"{(cur.get('separation') or {}).get('mutual', 0)})"),
        (f"ambiguous names: {(b.get('index') or {}).get('ambiguous_keys', 0)} -> "
         f"{(cur.get('index') or {}).get('ambiguous_keys', 0)}"),
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
    sp = c.get("separation") or {}
    npair = sp.get("confusable_pairs", 0)
    lines.append("pairwise separation (of the name-pairs a reader cannot keep apart):")
    for state, label in (("mutual", "mutual"), ("one_sided", "one-sided"), ("undrawn", "undrawn")):
        n = sp.get(state, 0)
        lines.append(f"  {label:<12} {_bar(n, max(1, npair))} {n}")
    lines.append(f"  pairs separated   [{_bar(sp.get('separated', 0), max(1, npair), width=32)}] "
                 f"{sp.get('separated', 0)}/{npair}")
    ix = c.get("index") or {}
    lines.append("")
    lines.append(f"name index: {ix.get('keys', 0)} lookup name(s) -> {c.get('rows', 0)} concept(s), "
                 f"{ix.get('ambiguous_keys', 0)} ambiguous")
    lines.append("")
    lines.append("legend: " + "   ".join(f"{_MARK[v]} {v}" for v in VERDICTS))
    return "\n".join(lines)


def _mix_spark(mix: dict[str, int], cap: int = 24) -> str:
    """A per-verdict sparkline for one subtree: one mark per concept, best-first. If the
    subtree is larger than `cap`, fall back to a compact `*3 o5 ~1` count form so the
    column stays readable."""
    total = sum(mix.get(v, 0) for v in VERDICTS)
    if total <= cap:
        return "".join(_MARK.get(v, " ") * mix.get(v, 0) for v in VERDICTS)
    return " ".join(f"{_MARK.get(v, '?')}{mix.get(v, 0)}" for v in VERDICTS if mix.get(v, 0))


def render_rollup(payload: dict[str, Any]) -> str:
    """Operator view of the namespace at its abstraction HEADS. Each row is one
    top-level abstraction rolled up WEAKEST-LINK - it reads as crystal only when every
    concept beneath it is. `!` flags a head whose declared verdict overclaims the
    subtree (the whole point: a collapsed view that cannot lie about what it hides)."""
    c = payload.get("corpus") or {}
    ru = c.get("rollup") or {}
    recs = ru.get("abstractions") or []
    lines = [
        (f"concept-disambiguation roll-up: {ru.get('roots', 0)} top-level abstraction(s), "
         f"{ru.get('heads', 0)} head(s) total, max depth {ru.get('max_depth', 0)} "
         f"({ru.get('forest_nodes', 0)} concepts in the forest)"),
        "",
        "Each abstraction rolls up WEAKEST-LINK: only as crystal-clear as its foggiest",
        "descendant. '!' = the head verdict reads clearer than the subtree supports.",
        "",
        f"     {'rolled':<12} {'head':<12} {'size':>4} {'debt':>4}  {'mix':<20} family / canonical",
    ]
    if not recs:
        lines.append("  (no hierarchy positioned - add a `parent` edge to a row to roll it up)")
        return "\n".join(lines)
    roots = [a for a in recs if a.get("depth", 0) == 0]
    for a in roots:
        spark = _mix_spark(a.get("verdict_mix") or {})
        flag = "!" if a.get("overclaim") else " "
        lines.append(f"  {flag}{_MARK.get(a['rolled_verdict'], ' ')} {a['rolled_verdict']:<11} "
                     f"{str(a['declared_verdict']):<12} {a['subtree_size']:>4} {a['subtree_debt']:>4}  "
                     f"{spark:<20} {a.get('family')} / {a.get('canonical')}")
    over = ru.get("overclaims") or []
    lines.append("")
    if over:
        lines.append(f"abstraction overclaims ({len(over)}) - head reads clearer than its subtree supports:")
        for o in over[:20]:
            lines.append(f"  ! {o}")
        if len(over) > 20:
            lines.append(f"  ... and {len(over) - 20} more")
    else:
        lines.append("abstraction overclaims: none - every head verdict is backed by its whole subtree")
    lines.append("")
    lines.append("(weakest-link: one 'defined' leaf keeps its abstraction from rolling up to crystal - "
                 "anchor the leaf, or accept the honest lower roll-up.)")
    return "\n".join(lines)


def _front_matter(title: str, desc: str) -> list[str]:
    return ["---", f'title: "{title}"', f'description: "{desc}"', "---", ""]


def provenance_row(card: dict[str, Any]) -> str:
    """Render the `As of` row, refusing an empty provenance instead of emitting one.

    This row is the one line that dates every other number on the page, and it silently
    rendered as `| As of |  (fak ) |` for as long as the data directory's `_meta.json` carried
    no `meta` block (#5609). The `'?'` fallback that was supposed to catch it could never fire:
    the card builder writes `meta.get("as_of", "")`, so a MISSING key arrives here as a present
    empty string and `.get(key, '?')` returns `''`, not the default. The table still looked
    well-formed, which is why it survived.

    That is the grounding epic's class inside a generator — an absent value rendered as a
    valid-looking empty one. A document asserting that every number below it is re-derived
    cannot decline to say WHEN. So empty is treated as missing, and missing refuses.
    """
    as_of = str(card.get("as_of") or "").strip()
    version = str(card.get("fak_version") or "").strip()
    missing = [name for name, value in (("as_of", as_of), ("fak_version", version)) if not value]
    if missing:
        raise ValueError(
            "concept-disambiguation scorecard: refusing to render an undated scorecard — "
            + ", ".join(missing)
            + " is empty or absent in the data directory's _meta.json \"meta\" block. "
            "Populate it the way the sibling scorecards do "
            "(docs/industry-scorecard, docs/persona-fit-scorecard): an undated derived claim "
            "is not a derived claim."
        )
    return f"| As of | {as_of} (fak {version}) |"


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
    out.append("> Regenerate: `fak concept generate` (the canonical writer preserves "
               "classifications while refreshing every generated page).")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("The driver is the UNBOUNDED disambiguation-debt (drive it to 0) plus the positive "
               "counters that climb without a ceiling. The old bounded /100 score SATURATES near the "
               "top - it hides the real, still-open work - so it is demoted to a labeled legacy line "
               "below and is NOT the metric to optimize.")
    out.append("")
    out.append("| Metric (primary = unbounded driver) | Value |")
    out.append("|---|---|")
    out.append(f"| **Disambiguation-debt (drive to 0)** | **{c.get('disambiguation_debt', 0)}** "
               f"(clarity {c.get('clarity_defects', 0)} + coverage {c.get('coverage_debt', 0)}) |")
    out.append(f"| **Crystal-clear concepts (and climbing)** | **{c.get('crystal_concepts', 0)}** "
               f"crystal of {c.get('rows', 0)} positioned |")
    out.append(f"| **Confusable tokens positioned (covered / discovered)** | "
               f"**{cov.get('covered', 0)} / {cov.get('discovered', 0)}** "
               f"({cov.get('coverage_pct', 0)}% of the discovered confusable space) |")
    out.append(f"| **Undrawn twin-pairs (drive to 0)** | "
               f"**{(c.get('separation') or {}).get('undrawn', 0)}** of "
               f"{(c.get('separation') or {}).get('confusable_pairs', 0)} confusable name-pairs |")
    out.append(f"| **Ambiguous lookup names (drive to 0)** | "
               f"**{(c.get('index') or {}).get('ambiguous_keys', 0)}** of "
               f"{(c.get('index') or {}).get('keys', 0)} indexed names |")
    out.append(provenance_row(c))
    out.append(f"| Legacy bounded score (saturates; not the driver) | {c.get('score', 0)}/100 "
               f"(grade {c.get('grade', '?')}) |")
    out.append("")
    out.append("> **Read this right.** The metric to optimize is the UNBOUNDED disambiguation-debt "
               "(drive it toward 0) and the counters that climb without a ceiling (crystal concepts, "
               "confusable tokens positioned). The bounded /100 score SATURATES - once the catalogued "
               "namespace is clean it sits near 100 and can no longer tell you how much confusable "
               "space is still un-disambiguated - so it is kept only as a labeled legacy line, not the "
               "driver.")
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
    out.append("| = entangled | defined and draws SOME line, but not against the twin its own NAME is confusable with - per-row clean, pairwise still fog |")
    out.append("| x colliding | shares a canonical name with another concept - a true ambiguity, fixable only by a rename |")
    out.append("| . undocumented | appears in the tree, but the catalog gives no definition |")
    out.append("")
    sp = c.get("separation") or {}
    ix = c.get("index") or {}
    out.append("## Separation - is each concept disambiguated FROM THE OTHERS?")
    out.append("")
    out.append("Per-concept clarity is not the same question as pairwise separation. A concept "
               "in a 250-member family satisfies `disambiguated` by naming **any one** sibling - "
               "while the twin its name is genuinely mistakable for goes undrawn. So the "
               "scorecard discovers, from the catalog's own names, the pairs a reader cannot "
               "keep apart - **permuted** (`witnessPath` / `PathWitness`: the same words in a "
               "different order) and **near** (`SessionRef` / `SessionRow`: a couple of edits "
               "apart) - and asks whether the line is drawn between *those two specifically*. "
               "Boundaries are directed, so for a confusable pair the line must be drawn from "
               "**both** sides: `A.distinct_from = [B]` does nothing for a reader who arrived "
               "at B. A concept that does not separate from its own twin is `entangled` - "
               "clean per row, still fog pairwise.")
    out.append("")
    out.append("| Separation metric | Value |")
    out.append("|---|---|")
    out.append(f"| Confusable name-pairs discovered | {sp.get('confusable_pairs', 0)} |")
    out.append(f"| **Separated from each other (drive to all)** | **{sp.get('separated', 0)} / "
               f"{sp.get('confusable_pairs', 0)}** ({sp.get('mutual', 0)} mutual, "
               f"{sp.get('one_sided', 0)} one-sided) |")
    out.append(f"| **Undrawn twin-pairs (drive to 0)** | **{sp.get('undrawn', 0)}** |")
    out.append(f"| Entangled concepts (own twin undrawn) | {sp.get('entangled_concepts', 0)} |")
    out.append(f"| Boundaries drawn (mutual / total) | {sp.get('mutual_boundaries', 0)} / "
               f"{sp.get('boundaries', 0)} |")
    out.append(f"| Dangling `distinct_from` references (drive to 0) | "
               f"{sp.get('dangling_references', 0)} |")
    out.append("")
    out.append("## Indexing - can a reader who meets a NAME find the concept?")
    out.append("")
    out.append("The catalog is organised by concept; a reader arrives with a **spelling**. "
               "[`INDEX.md`](INDEX.md) turns it around: every surface a name can arrive as - the "
               "canonical, every alias, the raw grounding token - is a lookup key pointing at the "
               "concept it denotes, carrying that concept's contrast set so *which one is this?* "
               "and *how is it not the other one?* are answered together. A key that lands on two "
               "concepts is an **index** defect the canonical names cannot show: both canonicals "
               "may be unique while one row quietly claims the other's name as an alias.")
    out.append("")
    out.append("| Index metric | Value |")
    out.append("|---|---|")
    out.append(f"| Lookup names indexed | {ix.get('keys', 0)} over {c.get('rows', 0)} concepts |")
    out.append(f"| Lookup names landing on several concepts | {ix.get('ambiguous_keys', 0)} |")
    out.append(f"| **Shared names whose concepts stay unseparated (drive to 0)** | "
               f"**{ix.get('unresolved_shared_names', 0)}** |")
    out.append(f"| Concepts carrying a contrast set | {ix.get('contrast_concepts', 0)} |")
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
    ru = c.get("rollup") or {}
    recs = ru.get("abstractions") or []
    if recs:
        out.append("## Concept roll-up (the namespace at its abstraction heads)")
        out.append("")
        out.append("Most readers cannot hold every concept at once. The optional `parent` forest lets "
                   "the catalog roll concepts up to the abstraction that HEADS them - and the roll-up is "
                   "**weakest-link**: an abstraction reads as crystal only when *every* concept beneath it "
                   "is. A single `defined` leaf keeps the whole head from rolling up to crystal, so the "
                   "collapsed view can never hide fog it contains. `!` marks a head whose declared verdict "
                   "reads clearer than its subtree supports.")
        out.append("")
        out.append("```text")
        out.append(render_rollup(payload))
        out.append("```")
        out.append("")
        out.append("| | Abstraction | Rolled | Head declares | Subtree | Debt | Weakest descendant |")
        out.append("|---|---|---|---|---:|---:|---|")
        for a in [x for x in recs if x.get("depth", 0) == 0]:
            mark = _MARK.get(a["rolled_verdict"], " ")
            flag = "!" if a.get("overclaim") else ""
            weak = a.get("weakest") or {}
            out.append(f"| {mark}{flag} | **{a.get('canonical')}** (`{a.get('id')}`) | {a['rolled_verdict']} "
                       f"| {a['declared_verdict']} | {a['subtree_size']} | {a['subtree_debt']} "
                       f"| {weak.get('id')} = {weak.get('verdict')} |")
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


def _md_cell(s: Any) -> str:
    """Make a value safe inside a Markdown table cell."""
    return str(s if s is not None else "").replace("|", "\\|").replace("\n", " ").strip()


def render_doc_name_index(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    """The generated A-Z name index: every lookup surface -> the concept it denotes and
    the concepts it must not be confused with.

    An index POINTS; it does not copy. The definitions live in README.md, so each entry
    stays one line and the whole lookup surface fits in one scannable artifact."""
    c = payload.get("corpus") or {}
    ix = c.get("index") or {}
    rows = _index_rows(payload)
    out = _front_matter(
        "fak concept name index - every spelling, the concept it denotes, and its contrast set",
        "A-Z lookup over every surface a fak concept name can arrive as (canonical, alias, "
        "grounding token), pointing at the one concept it denotes plus the concepts it must "
        "not be confused with. Generated - do not hand-edit.")
    out.append("# fak concept name index - which concept is this name, and what is it NOT?")
    out.append("")
    if stamp:
        out.append(f"<!-- concept-disambiguation-scorecard: {stamp} - process: "
                   f"tools/concept_disambiguation_scorecard.py - data: {DATA_DIR_REL}/ -->")
        out.append("")
    out.append("The [scorecard](README.md) is organised by concept. A reader arrives with a "
               "**spelling** - something they read in code, a metric name, a flag - and needs "
               "the reverse map. This is it: every surface a name can arrive as (the canonical "
               "name, every alias, the raw grounding token) resolves to the concept it denotes "
               "and to that concept's **contrast set**, so *which one is this?* and *how is it "
               "not the other one?* are answered in the same line.")
    out.append("")
    out.append("> Generated by `python tools/concept_disambiguation_scorecard.py "
               "--markdown-dir docs/concept-disambiguation-scorecard`. Do not hand-edit: edit "
               f"`{DATA_DIR_REL}/` and regenerate.")
    out.append("")
    out.append(f"**{ix.get('keys', 0)}** lookup names over **{c.get('rows', 0)}** concepts; "
               f"**{ix.get('ambiguous_keys', 0)}** land on more than one concept (marked **!** - "
               "tolerated only when the two concepts separate from each other in both "
               "directions, so the index can honestly answer *both, and here is the difference*).")
    out.append("")
    out.append("The **not to be confused with** column is the union of the boundaries the concept "
               "declares (`distinct_from`) and the twins the scorecard discovered by name - the "
               "pairs whose spellings a reader cannot keep apart.")
    out.append("")

    def _bucket(key: str) -> str:
        ch = key[:1]
        if not ch:
            return "#"
        return ch.upper() if ch.isalpha() else "0-9"

    buckets: dict[str, list[dict[str, Any]]] = {}
    for e in rows:
        buckets.setdefault(_bucket(e["key"]), []).append(e)
    for b in sorted(buckets):
        out.append(f"## {b}")
        out.append("")
        out.append("| | Name (and other spellings) | Concept | Family / kind | Not to be confused with |")
        out.append("|---|---|---|---|---|")
        for e in buckets[b]:
            flag = "**!**" if e["ambiguous"] else ""
            spellings = _md_cell(e["spellings"][0])
            if len(e["spellings"]) > 1:
                spellings += " <br><small>" + _md_cell(", ".join(e["spellings"][1:])) + "</small>"
            for t in e["targets_full"]:
                nc = t["not_to_be_confused_with"]
                out.append(f"| {flag} | `{spellings}` | **{_md_cell(t['canonical'])}** "
                           f"| {_md_cell(t['family'])} / {_md_cell(t['kind'])} "
                           f"| {_md_cell(', '.join(nc)) if nc else '-'} |")
        out.append("")
    return "\n".join(out)


def render_doc_folder(payload: dict[str, Any], *, stamp: str | None = None) -> dict[str, str]:
    return {"README.md": render_doc_index(payload, stamp=stamp),
            "INDEX.md": render_doc_name_index(payload, stamp=stamp)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Concept-disambiguation scorecard (read-only unless --markdown-dir).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--chart", action="store_true", help="an at-a-glance ASCII chart")
    ap.add_argument("--critical", action="store_true", help="the worst-first clarity backlog")
    ap.add_argument("--rollup", action="store_true", help="the hierarchy roll-up (namespace at its abstraction heads)")
    ap.add_argument("--gaps", action="store_true", help="the coverage backlog (unpositioned tree tokens)")
    ap.add_argument("--pairs", action="store_true", help="the pairwise separation backlog (confusable name-pairs)")
    ap.add_argument("--index", action="store_true", help="the name -> concept lookup index")
    ap.add_argument("--lookup", default="", help="resolve ONE name through the index (implies --index)")
    ap.add_argument("--limit", type=int, default=0, help="cap the --index listing (0 = all)")
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
            (out_dir / rel).write_text(content + "\n", encoding="utf-8", newline="\n")
        if not args.json:
            print(f"wrote concept-disambiguation doc folder -> {out_dir}")

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.chart:
        print(render_chart(payload))
    elif args.critical:
        print(render_critical(payload))
    elif args.rollup:
        print(render_rollup(payload))
    elif args.gaps:
        print(render_gaps(payload))
    elif args.pairs:
        print(render_pairs(payload))
    elif args.index or args.lookup:
        print(render_index(payload, limit=args.limit, lookup=args.lookup))
    elif not args.markdown_dir:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
