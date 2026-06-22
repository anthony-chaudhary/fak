#!/usr/bin/env python3
"""Repo-hygiene scorecard — the measuring stick for a repo that stays lean as it grows.

The sibling scorecards each watch ONE surface: ``docs_scorecard`` grades the
curated core docs, ``doc_appeal_scorecard`` grades one doc's prose,
``code_quality_scorecard`` grades the Go module, ``seo_aeo_scorecard`` grades
discoverability. None of them watch the thing that rots fastest when many agents
commit in parallel: the **shape of the whole repository**. A second FAQ appears
beside the first. A dated note lands at the root instead of ``docs/notes/``. A
third ``benchmark`` directory is born next to the two that exist. A doc gets
written but linked from no index, so no reader ever finds it. None of that is a
dead link or an overlong sentence, so nothing measured it — "keep the repo clean"
was a vibe.

This is that number. It scores the tracked tree on twelve mechanical KPIs in four
groups, folds them into a weighted score and an A-F grade, and — the lever that
makes "3x cleaner" a checkable target instead of a vibe — counts **hygiene-debt**:
the total of concrete, re-derivable structural defects you fix by *deleting or
consolidating*, not by adding more.

  VERBOSITY     — say it once, briefly
    redundancy     near-duplicate docs (one concept, two files) to consolidate
    bloat          a doc so long no one reads it whole; split or trim it

  ORGANIZATION  — a place for everything
    root_hygiene   the repo root holds only the front-door / meta files
    placement      dated/research notes live in docs/notes/, not scattered
    dir_discipline no near-duplicate sibling directories (benchmark vs benchmarks)

  INDEXING      — everything findable
    index_presence the hub files a reader is pointed at actually exist
    index_integrity every link IN an index resolves on disk (no 404 index entry)
    orphans        every reader-facing doc is reachable from some index

  ACCESSIBILITY — anyone can read it (incl. neurodivergent readers, non-experts)
    alt_text       every doc image carries descriptive alt-text (HARD)
    ai_tells       cliché / LLM-scaffolding phrases that read machine-written (HARD)
    jargon         expert terms with no nearby plain gloss (SOFT)
    plain_language dense reading-ease, undefined acronyms, literal-reader idioms (SOFT)

The accessibility group's HARD defects (``alt_text`` + ``ai_tells``) roll up into
**a11y-debt** — the accessibility counterpart of hygiene-debt: an integer you
drive down to make the tree readable by non-experts, screen-reader users,
translators, and answer engines. It is reported next to hygiene-debt and tracked
by ``--compare`` the same way. The two remaining KPIs (``jargon``,
``plain_language``) are deliberately SOFT: they score (a jargon-dense,
hard-to-read tree grades lower) but emit no hard debt, because the cheap way to
move them is gaming, not clarity — the same WARN/HARD split the sibling
scorecards draw.

It does NOT drift from the commit gates. The root allowlist comes straight from
``check_doc_placement.ALLOWED_ROOT_MD`` and the junk patterns from
``check_committed_files``; the cliché/jargon lists come from
``doc_appeal_scorecard`` / ``docs_scorecard``. This tool is the tree-wide
*measurement* of the same closed-vocabulary reasons (DOC_PLACEMENT,
FILE_ADMISSION, BROKEN_LINK) the pre-commit hook refuses one commit at a time —
so the number it reports and the gate that blocks you can never say different
things.

Deterministic + read-only by construction: it reads the git-tracked tree (so two
clones of the same commit score identically) and edits nothing. Untracked scratch
a parallel session left behind is reported as an advisory worktree signal, never
as debt, so the headline number stays reproducible. Run from the repo ROOT::

    python tools/repo_hygiene_scorecard.py                 # human scorecard
    python tools/repo_hygiene_scorecard.py --json          # machine payload
    python tools/repo_hygiene_scorecard.py --markdown      # the committed snapshot body
    python tools/repo_hygiene_scorecard.py --compare base.json   # prove the debt moved (the Nx gate)

The companion process is the repo-3x program: each HARD defect is one unit of
hygiene-debt to retire by deleting, consolidating, moving, or indexing;
re-running proves the number dropped.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

SCHEMA = "fak-repo-hygiene-scorecard/1"

# ---------------------------------------------------------------------------
# Single-source reuse: pull the ENFORCEMENT rules from the gates that own them,
# so the measurement here can never drift from the refusal at commit time. Each
# import falls back to a small inline copy so the tool still stands alone if a
# sibling is absent (e.g. shipped on its own).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # the root-md allowlist the DOC_PLACEMENT gate enforces
    from check_doc_placement import ALLOWED_ROOT_MD, NOTES_DIR
except Exception:  # noqa: BLE001
    ALLOWED_ROOT_MD = {
        "README.md", "START-HERE.md", "INSTALL.md", "INDEX.md", "CONTRIBUTING.md",
        "CLA.md", "AGENTS.md", "CLAUDE.md", "SECURITY.md", "ARCHITECTURE.md",
        "EXTENDING.md", "GETTING-STARTED.md", "POLICY.md", "STATUS.md", "CLAIMS.md",
    }
    NOTES_DIR = "docs/notes"

try:  # the file-admission junk patterns
    from check_committed_files import HARD_JUNK, SOFT_JUNK, EXEMPT_DATA_DIRS
except Exception:  # noqa: BLE001
    HARD_JUNK = [re.compile(r"\.(exe|dll|so|dylib|pyc|pyo|class|o|a|obj)$"),
                 re.compile(r"~$"), re.compile(r"(^|/)__pycache__/")]
    SOFT_JUNK = [re.compile(r"\.log$"), re.compile(r"\.tmp$"),
                 re.compile(r"(^|/)(report|agent-report)\.json$")]
    EXEMPT_DATA_DIRS = ("experiments/", "testdata/", "internal/")

try:  # the vetted AI/LLM-tell phrase lists (corpus-wide accessibility)
    from doc_appeal_scorecard import CLICHE_PHRASES, LLM_SCAFFOLD_PHRASES
except Exception:  # noqa: BLE001
    CLICHE_PHRASES = ["delve", "leverage", "seamless", "robust", "cutting-edge",
                      "in today's", "landscape of", "tapestry", "boasts", "myriad",
                      "supercharge", "effortless", "world-class", "paradigm shift"]
    LLM_SCAFFOLD_PHRASES = ["here's the thing", "the key insight", "at its core",
                            "in a nutshell", "the bottom line", "it's important to note",
                            "when it comes to", "the reality is", "let's be clear"]

try:  # the jargon list the docs scorecard glosses against
    from docs_scorecard import JARGON_TERMS
except Exception:  # noqa: BLE001
    JARGON_TERMS = ["vDSO", "context-MMU", "RadixAttention", "KV cache", "KV-cache",
                    "idempoten", "quiesce", "default-deny", "fail-open", "adjudicat"]

# ---------------------------------------------------------------------------
# Calibration. Each is a deliberate threshold with a stated reason, generous on
# purpose: only an EGREGIOUS case is hard debt; the soft ceiling is an advisory.
# ---------------------------------------------------------------------------
DOC_SOFT_LINES = 600       # advisory: a doc this long is worth a second look
DOC_HARD_LINES = 1000      # HARD: a doc no one reads whole — split or trim it
DUP_HARD_JACCARD = 0.80    # HARD: two docs this similar are one doc in two files
DUP_SOFT_JACCARD = 0.55    # SOFT: a consolidation candidate (judgment call)
SHINGLE_K = 8              # word-window for the similarity shingles
DIR_MANY_CHILDREN = 18     # SOFT: a flat dir with this many .md children wants an index
ROOT_OTHER_SAMPLE = 40     # cap on listed root-clutter items
EMDASH_PER_100W_BUDGET = 0.6   # em-dashes per 100 prose words before a SOFT nudge
FLESCH_FLOOR = 30.0        # below this a doc reads like a spec, not an invitation
AITELL_PER_DOC_CAP = 8     # cap one doc's ai-tell debt so it can't dominate
ACCESS_SAMPLE = 30         # cap on listed soft accessibility signals

# Reader-facing discovery surface = the root front-door docs + docs/** minus the
# archival / evidence / journal subtrees (history, proofs with their own ledger,
# dated notes, launch planning, benchmark output). A directory rule, fully
# reproducible — nothing is excluded by hand.
ARCHIVE_DIRS = {"releases", "stable-releases", "notes", "proofs", "launch",
                "benchmarks", "benchmark", "benchmarking", "testing", "serving"}

# Root contract/platform meta docs: GitHub and the contributor flow surface these
# directly, so they need no CONTENT index — they are exempt from the orphan check
# (a reader reaches CLA/SECURITY/GOVERNANCE via the platform, not a docs hub). The
# CONTENT docs at root (ARCHITECTURE, POLICY, STATUS, …) are NOT exempt: if no
# index links them, that is real discoverability debt.
ROOT_META_EXEMPT = {
    "README.md", "START-HERE.md", "INDEX.md", "INSTALL.md", "CONTRIBUTING.md",
    "CLA.md", "AGENTS.md", "CLAUDE.md", "SECURITY.md", "GOVERNANCE.md",
    "CODE_OF_CONDUCT.md", "MAINTAINERS.md", "AUTHORS.md", "NOTICE.md", "SUPPORT.md",
    "HISTORY.md", "CHANGELOG.md", "ROADMAP.md", "TRADEMARK.md", "LICENSING.md",
    "PUBLIC-SCRUB-POLICY.md",
}

# Index/hub surfaces a reader is steered to. EXPECTED ones must exist (a missing
# one is debt); ALL existing front doors + every README hub seed the reachability
# BFS (a section README indexes its own directory).
EXPECTED_INDEXES = ["INDEX.md", "llms.txt", "docs/index.md"]
FRONT_DOORS = ["README.md", "llms.txt", "INDEX.md", "docs/index.md",
               "START-HERE.md", "AGENTS.md"]
# The explicit index MANIFESTS whose every local link must resolve (the "an index
# entry must not 404" check). README hubs are linted by check_links; here we hold
# the hand-curated manifests to a higher bar.
INDEX_MANIFESTS = ["llms.txt", "INDEX.md", "docs/index.md"]

# Files legitimately tracked at the repo root that are not .md (everything else
# tracked at root is clutter). The .md allowlist is ALLOWED_ROOT_MD (imported).
ROOT_ALLOWED_OTHER = {
    "go.mod", "go.sum", "Makefile", "LICENSE", "VERSION", "Dockerfile",
    ".dockerignore", ".gitignore", ".gitattributes", ".editorconfig",
    "install.sh", "test.sh", "test.ps1", "dos.toml", "opencode.json",
    "CITATION.cff", "llms.txt", "llms-full.txt", ".golangci.yml", ".golangci.yaml",
    # agent/editor config a contributor is expected to find at the root
    ".cursorrules", ".mcp.json", ".gitmodules", ".markdownlint.json",
}

# This tool's own published snapshot: never scored (it would oscillate — the report
# literally contains the defect strings it lists) and never counted as an orphan.
# You don't lint your linter's output. It stays linked from INDEX (discoverable).
GENERATED_SNAPSHOT = "docs/REPO-HYGIENE-SCORECARD.md"

GROUPS = ("verbosity", "organization", "indexing", "accessibility")

KPI_WEIGHTS: dict[str, float] = {
    # verbosity
    "redundancy": 0.10,
    "bloat": 0.06,
    # organization
    "root_hygiene": 0.12,
    "placement": 0.10,
    "dir_discipline": 0.06,
    # indexing
    "index_presence": 0.10,
    "index_integrity": 0.12,
    "orphans": 0.12,
    # accessibility (group weight 0.22 unchanged; alt_text carved from the others
    # so the composite still tops out at 100 with no rescale)
    "alt_text": 0.06,
    "ai_tells": 0.06,
    "jargon": 0.04,
    "plain_language": 0.06,
}
KPI_GROUP: dict[str, str] = {
    "redundancy": "verbosity", "bloat": "verbosity",
    "root_hygiene": "organization", "placement": "organization",
    "dir_discipline": "organization",
    "index_presence": "indexing", "index_integrity": "indexing", "orphans": "indexing",
    "alt_text": "accessibility", "ai_tells": "accessibility",
    "jargon": "accessibility", "plain_language": "accessibility",
}

_LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)")
# A dated / issue-numbered research doc: a YYYY-MM-DD stamp or a trailing -NNN
# issue suffix in the basename. These belong under docs/notes/, not scattered.
_DATED_RE = re.compile(r"(20\d\d-\d\d-\d\d)|(-\d{3,}\.md$)")
_FENCE_RE = re.compile(r"^(```|~~~)")
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,5})s?\b")
# The corpus-wide AI-tell HARD set starts from the vetted doc_appeal lists but
# drops two bare words that are just as often plain technical English ("a robust
# parser", "the high-leverage fix") — punishing those corpus-wide would be a false
# positive, and a scorer that cries wolf gets ignored. doc_appeal keeps them
# because it grades ONE marketing surface (the README), where they ARE tells.
CONTEXT_SENSITIVE_TELLS = {"robust", "leverage"}
AI_TELL_PHRASES = [p for p in (list(CLICHE_PHRASES) + list(LLM_SCAFFOLD_PHRASES))
                   if p.lower() not in CONTEXT_SENSITIVE_TELLS]

# Idioms / figurative phrases a literal reader (and a translator) stumbles on.
# Kept to phrases that are unambiguously figurative in prose (after code is
# stripped) so a literal-reader / translator stumble is real, not a false positive.
LITERAL_IDIOMS = [
    "rule of thumb", "low-hanging fruit", "boils down to", "under the hood",
    "out of the box", "move the needle", "bite the bullet", "the elephant in the room",
    "ballpark", "a far cry", "hand-wavy", "bread and butter", "silver bullet",
    "wild goose chase", "back of the envelope",
    # figurative-language expansion (issue #510): each is a translator/literal-reader
    # trap with a plainer equivalent.
    "tip of the iceberg", "apples to apples", "apples to oranges", "moving target",
    "north star", "happy path", "raise the bar", "drop in the bucket",
    "throw over the wall", "cut corners", "in the weeds", "on the same page",
    "draw a line in the sand", "the lay of the land", "barking up the wrong tree",
    "piece of cake", "the whole nine yards", "foot the bill", "yak shaving",
    "the long pole in the tent", "boiling the ocean", "secret sauce",
]

# Doc images: a markdown image carries its alt-text in the brackets; an HTML <img>
# in its alt= attribute. An image with empty/missing alt is invisible to a screen
# reader, a translator, and an answer engine — the alt_text HARD signal.
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)")
_HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE | re.DOTALL)
_HTML_ALT_RE = re.compile(r"\balt\s*=\s*([\"'])(?P<alt>.*?)\1", re.IGNORECASE | re.DOTALL)
_HTML_SRC_RE = re.compile(r"\bsrc\s*=\s*([\"'])(?P<src>.*?)\1", re.IGNORECASE | re.DOTALL)


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


def is_dated_doc(name: str) -> bool:
    """A basename that carries a YYYY-MM-DD stamp or a trailing issue number."""
    return bool(_DATED_RE.search(name))


def is_reader_facing(rel: str) -> bool:
    """A discovery-surface doc: a root front-door .md, or a docs/** .md outside the
    archival/evidence/journal subtrees. A directory rule, not a hand-picked list."""
    if not rel.endswith(".md") or rel == GENERATED_SNAPSHOT:
        return False
    parts = rel.split("/")
    if len(parts) == 1:
        return parts[0] in ALLOWED_ROOT_MD
    if parts[0] != "docs":
        return False
    return not (len(parts) >= 3 and parts[1] in ARCHIVE_DIRS)


def _prose_only(text: str) -> str:
    """Drop fenced code, inline code, and YAML front-matter so prose checks see
    only what a reader reads."""
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    start = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                start = j + 1
                break
    for raw in lines[start:]:
        if _FENCE_RE.match(raw.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        out.append(re.sub(r"`[^`]*`", "", raw))
    return "\n".join(out)


def shingles(text: str, k: int = SHINGLE_K) -> set[int]:
    """A set of hashed k-word shingles over the prose. Used for content-similarity;
    deterministic (same text -> same set)."""
    words = re.findall(r"[a-z0-9]+", _prose_only(text).lower())
    if len(words) < k:
        return {hash(tuple(words))} if words else set()
    return {hash(tuple(words[i:i + k])) for i in range(len(words) - k + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _tell_regex(phrase: str) -> "re.Pattern[str]":
    """A word-boundary matcher for an AI-tell phrase. A SINGLE word is guarded
    against hyphen compounds (so 'leverage' never fires inside 'high-leverage');
    a multi-word / hyphenated phrase ('cutting-edge', 'first-class citizen') is
    matched whole on plain word boundaries."""
    esc = re.escape(phrase.lower())
    if " " in phrase or "-" in phrase:
        return re.compile(r"\b" + esc + r"\b")
    return re.compile(r"(?<![\w-])" + esc + r"(?![\w-])")


_AI_TELL_RES = [(ph, _tell_regex(ph)) for ph in AI_TELL_PHRASES]


def _wordcount(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'×%/-]*", text))


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def flesch(text: str) -> float:
    sents = [s for s in re.split(r"[.!?]+\s", text) if s.strip()]
    words = re.findall(r"[A-Za-z]+", text)
    if not sents or not words:
        return 100.0
    syll = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / len(sents)) - 84.6 * (syll / len(words))


def root_md_violations(root_md: list[str]) -> list[str]:
    """Root-level .md basenames not on the DOC_PLACEMENT allowlist."""
    return sorted(n for n in root_md if n not in ALLOWED_ROOT_MD)


def root_other_violations(root_other: list[str]) -> list[str]:
    """Root-level non-.md tracked files that are neither essential nor allowlisted."""
    return sorted(n for n in root_other if n not in ROOT_ALLOWED_OTHER)


def classify_junk(rel: str) -> str | None:
    """Junk reason for a tracked path, or None — reuses the FILE_ADMISSION rules."""
    for rx in HARD_JUNK:
        if rx.search(rel):
            return "build artifact / cache / compiled output"
    if not rel.startswith(tuple(EXEMPT_DATA_DIRS)):
        for rx in SOFT_JUNK:
            if rx.search(rel):
                return "log / temp / report output (regenerable)"
    return None


def reachable_md(seeds: list[str], links_by_doc: dict[str, list[str]]) -> set[str]:
    """BFS over local .md links from the seed docs; returns the reachable rel set.
    ``links_by_doc`` maps each doc to the repo-relative .md targets it links to."""
    visited: set[str] = set()
    q: deque[str] = deque(s for s in seeds if s in links_by_doc)
    while q:
        cur = q.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        for nxt in links_by_doc.get(cur, []):
            if nxt not in visited:
                q.append(nxt)
    return visited


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each returns
#   {kpi, group, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of hygiene-debt; soft = score-only judgment nudges.
# ---------------------------------------------------------------------------

def kpi_redundancy(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Near-duplicate reader-facing docs: one concept living in two files. Each
    high-overlap pair is one unit of debt — consolidate to one. ``docs`` is a list
    of {path, shingles, words}."""
    defects: list[str] = []
    soft: list[str] = []
    n = len(docs)
    for i in range(n):
        di = docs[i]
        for j in range(i + 1, n):
            dj = docs[j]
            wi, wj = di["words"], dj["words"]
            if wi == 0 or wj == 0:
                continue
            if not (0.5 <= wi / wj <= 2.0):  # cheap size prefilter
                continue
            sim = jaccard(di["shingles"], dj["shingles"])
            if sim >= DUP_HARD_JACCARD:
                defects.append(f"near-duplicate ({sim:.0%}): {di['path']} ≈ {dj['path']} "
                               f"— consolidate to one")
            elif sim >= DUP_SOFT_JACCARD:
                soft.append(f"consolidation candidate ({sim:.0%}): {di['path']} ~ {dj['path']}")
    return {"kpi": "redundancy", "group": "verbosity",
            "score": _clamp(100 - 14 * len(defects) - min(20, 3 * len(soft))),
            "detail": (f"{len(defects)} near-duplicate pair(s), {len(soft)} candidate(s)"
                       if defects or soft else "no near-duplicate docs"),
            "defects": defects, "soft": soft}


def kpi_bloat(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """A reader-facing doc past the hard line ceiling is a wall no one reads whole.
    ``docs`` is a list of {path, n_lines}."""
    defects: list[str] = []
    soft: list[str] = []
    for d in docs:
        if d["n_lines"] > DOC_HARD_LINES:
            defects.append(f"oversized doc {d['path']} ({d['n_lines']} lines > {DOC_HARD_LINES}) "
                           f"— split into sections or trim")
        elif d["n_lines"] > DOC_SOFT_LINES:
            soft.append(f"long doc {d['path']} ({d['n_lines']} lines)")
    return {"kpi": "bloat", "group": "verbosity",
            "score": _clamp(100 - 12 * len(defects) - min(20, 2 * len(soft))),
            "detail": (f"{len(defects)} oversized, {len(soft)} long"
                       if defects or soft else "no oversized docs"),
            "defects": defects, "soft": soft}


def kpi_root_hygiene(root_md: list[str], root_other: list[str]) -> dict[str, Any]:
    """The repo root is a public front door. Only the front-door / meta files
    belong there; every other tracked root file is clutter."""
    defects: list[str] = []
    bad_md = root_md_violations(root_md)
    bad_other = root_other_violations(root_other)
    for n in bad_md:
        defects.append(f"non-front-door doc at root: {n} → move to {NOTES_DIR}/ "
                       f"(or add to the allowlist if genuinely a root doc)")
    for n in bad_other[:ROOT_OTHER_SAMPLE]:
        why = classify_junk(n) or "not a front-door / build-essential file"
        defects.append(f"clutter at root: {n} — {why}")
    extra = max(0, len(bad_other) - ROOT_OTHER_SAMPLE)
    soft = [f"... and {extra} more root non-doc file(s)"] if extra else []
    return {"kpi": "root_hygiene", "group": "organization",
            "score": _clamp(100 - 10 * len(defects)),
            "detail": (f"{len(bad_md)} stray root doc(s), {len(bad_other)} stray root file(s)"
                       if defects else "root holds only front-door / meta files"),
            "defects": defects, "soft": soft}


def kpi_placement(dated_misplaced: list[str]) -> dict[str, Any]:
    """Dated/issue-numbered research notes belong under docs/notes/ (reached via
    INDEX.md), not scattered at the root or across docs/. ``dated_misplaced`` is
    the list of such docs found outside the sanctioned locations."""
    defects = [f"dated/research doc outside {NOTES_DIR}/: {p} → move it and index it"
               for p in sorted(dated_misplaced)]
    return {"kpi": "placement", "group": "organization",
            "score": _clamp(100 - 10 * len(defects)),
            "detail": (f"{len(defects)} misplaced dated doc(s)"
                       if defects else f"dated docs live under {NOTES_DIR}/"),
            "defects": defects, "soft": []}


def kpi_dir_discipline(dirs: list[str]) -> dict[str, Any]:
    """Near-duplicate sibling directory names (benchmark / benchmarking /
    benchmarks) fracture a reader's map. ``dirs`` is the list of directory rels."""
    defects: list[str] = []
    soft: list[str] = []
    # Group sibling dirs by parent, then by a normalized stem (singular-ish).
    by_parent: dict[str, list[str]] = {}
    for d in dirs:
        parent = d.rsplit("/", 1)[0] if "/" in d else ""
        by_parent.setdefault(parent, []).append(d)
    for parent, kids in by_parent.items():
        stems: dict[str, list[str]] = {}
        for k in kids:
            base = k.rsplit("/", 1)[-1].lower()
            stem = re.sub(r"(ing|s|es)$", "", base)
            stems.setdefault(stem, []).append(k)
        for stem, group in stems.items():
            if len(group) > 1 and stem:
                defects.append(f"near-duplicate sibling dirs: {sorted(group)} — merge into one")
    return {"kpi": "dir_discipline", "group": "organization",
            "score": _clamp(100 - 12 * len(defects)),
            "detail": (f"{len(defects)} near-duplicate dir group(s)"
                       if defects else "no near-duplicate sibling directories"),
            "defects": defects, "soft": soft}


def kpi_index_presence(present: dict[str, bool]) -> dict[str, Any]:
    """The hub/index files a reader is steered to must actually exist. ``present``
    maps each EXPECTED_INDEXES entry to a bool."""
    defects = [f"missing index surface: {name} (readers and guards are pointed here)"
               for name in EXPECTED_INDEXES if not present.get(name)]
    return {"kpi": "index_presence", "group": "indexing",
            "score": _clamp(100 - 25 * len(defects)),
            "detail": (f"{len(defects)} missing index surface(s)"
                       if defects else "all expected index surfaces present"),
            "defects": defects, "soft": []}


def kpi_index_integrity(dead_by_index: dict[str, list[str]]) -> dict[str, Any]:
    """Every local link inside a curated index manifest must resolve on disk — an
    index that points at a deleted doc is worse than no index. ``dead_by_index``
    maps each manifest to its list of dead local targets."""
    defects: list[str] = []
    for idx, dead in dead_by_index.items():
        for t in sorted(dead):
            defects.append(f"dead index entry in {idx}: {t}")
    return {"kpi": "index_integrity", "group": "indexing",
            "score": _clamp(100 - 15 * len(defects)),
            "detail": (f"{len(defects)} dead index entr(y/ies)"
                       if defects else "every index entry resolves"),
            "defects": defects, "soft": []}


def kpi_orphans(orphans: list[str], n_reader: int) -> dict[str, Any]:
    """A reader-facing doc reachable from no index or hub is written but unfindable.
    Each is one unit of debt — link it from an index or delete it."""
    defects = [f"orphan (reachable from no index/hub): {p} — index it or delete it"
               for p in sorted(orphans)]
    indexed = max(0, n_reader - len(orphans))
    pct = round(100 * indexed / max(1, n_reader), 1)
    return {"kpi": "orphans", "group": "indexing",
            "score": _clamp(pct),
            "detail": f"{indexed}/{n_reader} reader-facing docs reachable from an index ({pct}%)",
            "defects": defects, "soft": []}


def image_alt_defects(text: str) -> list[str]:
    """Doc images whose alt-text is missing or empty — returns the offending image
    sources. Markdown ``![alt](src)`` reads the bracket text; an HTML ``<img>``
    reads its ``alt=`` attribute (the tag may span lines). The caller passes
    prose-only text, so an image-syntax example inside a code fence is never
    flagged. Deterministic: same text -> same list."""
    out: list[str] = []
    for m in _MD_IMG_RE.finditer(text):
        if not m.group("alt").strip():
            out.append(m.group("src").strip())
    for m in _HTML_IMG_RE.finditer(text):
        tag = m.group(0)
        alt = _HTML_ALT_RE.search(tag)
        if alt is None or not alt.group("alt").strip():
            src = _HTML_SRC_RE.search(tag)
            out.append(src.group("src").strip() if src else "<img>")
    return out


def kpi_alt_text(per_doc: list[dict[str, Any]]) -> dict[str, Any]:
    """Every doc image needs alt-text — the surface a screen-reader user, a
    translator, and an answer engine read instead of the pixels. Each image with
    empty/missing alt is one unit of a11y-debt: add a descriptive alt, don't
    delete the image. ``per_doc`` is a list of {path, missing:[src]}."""
    defects: list[str] = []
    for d in per_doc:
        for src in d["missing"]:
            defects.append(f"image without alt-text in {d['path']}: {src} "
                           f"— add descriptive alt-text")
    return {"kpi": "alt_text", "group": "accessibility",
            "score": _clamp(100 - 6 * len(defects)),
            "detail": (f"{len(defects)} image(s) missing alt-text"
                       if defects else "every doc image carries alt-text"),
            "defects": defects, "soft": []}


def kpi_ai_tells(per_doc: list[dict[str, Any]]) -> dict[str, Any]:
    """Cliché and LLM-scaffolding phrases read machine-written and push away a
    reader who wants plain language. Each occurrence (capped per doc) is one unit
    of debt. ``per_doc`` is a list of {path, hits:[phrase], emdash_over}."""
    defects: list[str] = []
    soft: list[str] = []
    for d in per_doc:
        hits = d["hits"][:AITELL_PER_DOC_CAP]
        for ph in hits:
            defects.append(f"AI-tell phrase in {d['path']}: “{ph}” — say it plainly")
        if len(d["hits"]) > AITELL_PER_DOC_CAP:
            soft.append(f"{len(d['hits']) - AITELL_PER_DOC_CAP} more AI-tells in {d['path']} (capped)")
        if d.get("emdash_over", 0) > 0:
            soft.append(f"em-dash flood in {d['path']} ({d['emdash_over']} past budget)")
    return {"kpi": "ai_tells", "group": "accessibility",
            "score": _clamp(100 - 3 * len(defects) - min(15, len(soft))),
            "detail": (f"{len(defects)} AI-tell phrase(s) across {len(per_doc)} doc(s)"
                       if defects else "no AI-tell phrases"),
            "defects": defects, "soft": soft}


def kpi_jargon(naked: list[str], n_reader: int) -> dict[str, Any]:
    """SOFT: expert terms on a doc's first screen with no nearby plain gloss. It
    scores (a jargon-dense surface grades lower) but emits no hard debt — writing
    a gloss to move a number is gaming. Scored on a per-doc RATE so it does not
    mechanically sink as the repo grows. ``naked`` is a list of '<path>: <term>'."""
    soft = [f"first-screen jargon, no gloss: {n}" for n in naked[:ACCESS_SAMPLE]]
    if len(naked) > ACCESS_SAMPLE:
        soft.append(f"... and {len(naked) - ACCESS_SAMPLE} more naked jargon term(s)")
    rate = len(naked) / max(1, n_reader)
    return {"kpi": "jargon", "group": "accessibility",
            "score": _clamp(100 - min(60, round(45 * rate))),
            "detail": (f"{len(naked)} naked first-screen jargon term(s) "
                       f"({rate:.1f}/doc)" if naked
                       else "first-screen terms carry plain glosses"),
            "defects": [], "soft": soft}


def kpi_plain_language(signals: list[str], n_dense: int, n_acro_docs: int,
                       n_idiom: int, n_reader: int) -> dict[str, Any]:
    """SOFT: dense reading-ease, docs that use an acronym before defining it, and
    literal-reader idioms — the accessibility nudges for non-experts and
    neurodivergent readers. Scored on a per-doc RATE (growth-invariant); never
    gates (clarity is a judgment call)."""
    soft = list(signals[:ACCESS_SAMPLE])
    if len(signals) > ACCESS_SAMPLE:
        soft.append(f"... and {len(signals) - ACCESS_SAMPLE} more accessibility signal(s)")
    total = n_dense + n_acro_docs + n_idiom
    rate = total / max(1, n_reader)
    return {"kpi": "plain_language", "group": "accessibility",
            "score": _clamp(100 - min(60, round(45 * rate))),
            "detail": (f"{n_dense} dense doc(s), {n_acro_docs} doc(s) with undefined acronyms, "
                       f"{n_idiom} literal-reader idiom(s)"
                       if total else "reads plainly (ease, acronyms, idioms)"),
            "defects": [], "soft": soft}


# ---------------------------------------------------------------------------
# Fold: KPIs -> composite score, grade, hygiene-debt, control-pane payload.
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, kpis: list[dict[str, Any]],
                  worktree_clutter: list[str] | None = None,
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
    hygiene_debt = sum(len(k["defects"]) for k in kpis)
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
        "score": score, "grade": grade, "hygiene_debt": hygiene_debt,
        # a11y-debt: the accessibility group's HARD defects, broken out as a
        # first-class integer to drive down (issue #510) — the accessibility
        # counterpart of hygiene-debt. Always a slice of hygiene_debt, never extra.
        "a11y_debt": debt_by_group["accessibility"],
        "soft_signals": n_soft,
        "debt_by_group": debt_by_group,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
        "worktree_clutter": list(worktree_clutter or []),
    }

    if hygiene_debt == 0:
        ok, verdict, finding = True, "OK", "repo_clean"
        reason = (f"repo clean: score {score}/100 (grade {grade}), zero hygiene-debt "
                  f"across {len(kpis)} KPIs ({n_soft} advisory signal(s))")
        next_action = "no required edit; re-run after the next structural change"
    else:
        ok, verdict, finding = False, "ACTION", "hygiene_debt"
        worst = breakdown[0]
        reason = (f"{hygiene_debt} unit(s) of hygiene-debt; score {score}/100 (grade {grade}); "
                  f"heaviest: {worst['kpi']} ({worst['debt']} defect(s))")
        next_action = ("retire hygiene-debt worst-first (see corpus.breakdown + per-KPI defects): "
                       "consolidate duplicates, split/trim oversized docs, clear root clutter, "
                       "move dated docs to docs/notes/, index orphans, cut AI-tell phrases; "
                       "re-run to prove the drop")

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


def _local_md_targets(text: str, doc_rel: str, root: Path) -> list[str]:
    """The repo-relative .md targets a doc links to (local links only)."""
    base = (root / doc_rel).parent
    out: list[str] = []
    for m in _LINK_RE.finditer(text):
        t = m.group("target").strip()
        if t.startswith(("http://", "https://", "mailto:", "#", "tel:")):
            continue
        path_part = t.split("#", 1)[0].split("?", 1)[0].strip()
        if not path_part.endswith(".md"):
            continue
        resolved = (root / path_part.lstrip("/")) if path_part.startswith("/") \
            else (base / path_part)
        try:
            out.append(resolved.resolve().relative_to(root.resolve()).as_posix())
        except (ValueError, OSError):
            continue
    return out


def _all_local_links(text: str, doc_rel: str, root: Path) -> list[tuple[str, bool]]:
    """Every local link in a doc as (target_display, exists). Non-.md included —
    an index manifest can point at any file."""
    base = (root / doc_rel).parent
    out: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for m in _LINK_RE.finditer(text):
        t = m.group("target").strip()
        if t.startswith(("http://", "https://", "mailto:", "#", "tel:")):
            continue
        path_part = t.split("#", 1)[0].split("?", 1)[0].strip()
        if not path_part or path_part in seen:
            continue
        seen.add(path_part)
        resolved = (root / path_part.lstrip("/")) if path_part.startswith("/") \
            else (base / path_part)
        out.append((path_part, resolved.exists()))
    return out


def gather(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read the git-tracked tree and run every pure KPI. Returns (kpis, clutter)."""
    tracked = _git_lines(["ls-files"], root)
    tracked_set = set(tracked)
    md_files = [f for f in tracked if f.endswith(".md")]
    reader = [f for f in md_files if is_reader_facing(f)]

    # --- read reader-facing docs once ---
    texts: dict[str, str] = {f: _safe_read(root / f) for f in reader}

    # verbosity: redundancy + bloat (reader-facing only)
    dup_docs = [{"path": f, "shingles": shingles(texts[f]),
                 "words": _wordcount(_prose_only(texts[f]))} for f in reader]
    bloat_docs = [{"path": f, "n_lines": texts[f].count("\n") + 1} for f in reader]

    # organization: root hygiene + placement + dir discipline
    root_md = [f for f in md_files if "/" not in f]
    root_other = [f for f in tracked if "/" not in f and not f.endswith(".md")]
    # placement judges NON-root docs only — a root .md is root_hygiene's domain
    # (and the DOC_PLACEMENT allowlist may deliberately keep a dated one, e.g. the
    # per-release HERO doc; contradicting that allowlist would be a false positive).
    dated_misplaced = sorted(
        f for f in md_files
        if "/" in f and is_dated_doc(f.rsplit("/", 1)[-1])
        and not f.startswith((NOTES_DIR + "/", "docs/releases/", "docs/stable-releases/",
                              "blog/", "experiments/"))
    )
    tracked_dirs = sorted({"/".join(f.split("/")[:-1]) for f in tracked if "/" in f}
                          - {""})

    # indexing: presence + integrity + orphans
    present = {name: (root / name).exists() for name in EXPECTED_INDEXES}
    dead_by_index: dict[str, list[str]] = {}
    for idx in INDEX_MANIFESTS:
        if (root / idx).exists():
            dead = [t for t, ok in _all_local_links(_safe_read(root / idx), idx, root)
                    if not ok]
            if dead:
                dead_by_index[idx] = dead
    # reachability BFS: seed from existing front doors + every README hub.
    seeds = [d for d in FRONT_DOORS if (root / d).exists()]
    seeds += [f for f in md_files if f.rsplit("/", 1)[-1] == "README.md"]
    links_by_doc: dict[str, list[str]] = {}
    for f in md_files:
        links_by_doc[f] = [t for t in _local_md_targets(_safe_read(root / f), f, root)
                           if t in tracked_set]
    for s in seeds:  # a seed that is not a tracked .md (e.g. llms.txt) still seeds
        if s not in links_by_doc:
            links_by_doc[s] = [t for t in _local_md_targets(_safe_read(root / s), s, root)
                               if t in tracked_set]
    reachable = reachable_md(seeds, links_by_doc)
    # orphan candidates: reader-facing docs minus the contract/platform meta the
    # platform surfaces directly (those need no content index).
    orphan_pool = [f for f in reader if f.rsplit("/", 1)[-1] not in ROOT_META_EXEMPT]
    orphans = [f for f in orphan_pool if f not in reachable and f not in seeds]

    # accessibility: alt_text (HARD) + ai_tells (HARD) + jargon (SOFT) + plain_language (SOFT)
    alt_per_doc: list[dict[str, Any]] = []
    ai_per_doc: list[dict[str, Any]] = []
    naked_jargon: list[str] = []
    plain_signals: list[str] = []
    n_dense = n_acro_docs = n_idiom = 0
    for f in reader:
        prose = _prose_only(texts[f])
        low = prose.lower()
        missing_alt = image_alt_defects(prose)
        if missing_alt:
            alt_per_doc.append({"path": f, "missing": missing_alt})
        hits: list[str] = []
        for ph, rx in _AI_TELL_RES:
            hits.extend(ph for _ in rx.finditer(low))
        words = _wordcount(prose)
        n_dash = prose.count("—")
        budget = max(2, int(words * EMDASH_PER_100W_BUDGET / 100))
        if hits or n_dash > budget:
            ai_per_doc.append({"path": f, "hits": hits,
                               "emdash_over": max(0, n_dash - budget)})
        # jargon on the first screen (top 60 lines)
        head = "\n".join(texts[f].splitlines()[:60])
        for term in JARGON_TERMS:
            for line in head.splitlines():
                if term.lower() in line.lower():
                    if not (("(" in line) or ("—" in line) or (" - " in line)):
                        naked_jargon.append(f"{f}: {term}")
                    break
        # plain-language: reading ease, undefined acronyms, literal idioms
        if words > 120:
            ease = flesch(prose)
            if ease < FLESCH_FLOOR:
                n_dense += 1
                plain_signals.append(f"dense reading-ease {ease:.0f} (< {FLESCH_FLOOR:.0f}): {f}")
        # an acronym is a barrier only if it is RECURRING (used 2+ times), not in
        # the broad common set, and never expanded anywhere in the doc — a one-off
        # mention or a glossed term is not flagged (keeps this honest, not noisy).
        all_acros = _ACRONYM_RE.findall(prose)
        counts: dict[str, int] = {}
        for a in all_acros:
            counts[a] = counts.get(a, 0) + 1
        undefined = [a for a in sorted(counts)
                     if counts[a] >= 2 and a not in _COMMON_ACRONYMS
                     and f"({a})" not in prose
                     and not re.search(rf"\b\w+(?:\s+\w+){{1,4}}\s*\(\s*{a}\b", prose)]
        if undefined:
            n_acro_docs += 1
            plain_signals.append(f"acronym(s) used before definition in {f}: {', '.join(undefined[:6])}")
        for idiom in LITERAL_IDIOMS:
            if idiom in low:
                n_idiom += 1
                plain_signals.append(f"literal-reader idiom “{idiom}” in {f}")

    kpis = [
        kpi_redundancy(dup_docs),
        kpi_bloat(bloat_docs),
        kpi_root_hygiene(root_md, root_other),
        kpi_placement(dated_misplaced),
        kpi_dir_discipline(tracked_dirs),
        kpi_index_presence(present),
        kpi_index_integrity(dead_by_index),
        kpi_orphans(orphans, len(orphan_pool)),
        kpi_alt_text(alt_per_doc),
        kpi_ai_tells(ai_per_doc),
        kpi_jargon(naked_jargon, len(reader)),
        kpi_plain_language(plain_signals, n_dense, n_acro_docs, n_idiom, len(reader)),
    ]

    # advisory only: untracked, non-ignored scratch a parallel session left (the
    # concurrency canary). Never debt — it varies by working tree.
    clutter = _worktree_clutter(root)
    return kpis, clutter


# Acronyms common enough that using them unexpanded does not block a reader —
# general computing + this project's domain vocabulary. Kept broad on purpose:
# the goal is to flag a GENUINELY opaque recurring acronym, not common shorthand.
_COMMON_ACRONYMS = {
    # general computing
    "API", "CLI", "URL", "URI", "HTTP", "HTTPS", "JSON", "YAML", "TOML", "HTML",
    "CSS", "CPU", "GPU", "RAM", "OS", "SDK", "ID", "UI", "UX", "IO", "TLS", "SSL",
    "DNS", "TCP", "UDP", "IP", "RPC", "REST", "RPS", "QPS", "VM", "FS", "DB", "SQL",
    "OK", "FAQ", "TOC", "TODO", "FIXME", "CI", "CD", "PR", "VCS", "DSL", "AST",
    "SIMD", "AVX", "NUMA", "OOM", "GEMM", "TLB", "FFI", "JIT", "AOT", "EOF", "UTF",
    "ASCII", "RNG", "UUID", "SHA", "HMAC", "RSA", "AES", "JWT", "CORS", "RBAC",
    "ACL", "PII", "E2E", "SLA", "SLO", "HA", "P0", "P1", "P2", "OSS",
    # licenses / legal / process
    "MIT", "BSD", "GNU", "LGPL", "GPL", "DCO", "CLA", "GDPR",
    # this project's domain
    "ABI", "DAG", "WAL", "CAS", "TTL", "MMU", "DOS", "SOTA", "AEO", "SEO", "RSI",
    "FAK", "AMD", "ARM", "DGX", "TP", "PP", "DP", "AI", "ML", "LLM", "KV", "MCP",
    "GGUF", "AWQ", "GPTQ", "RoPE", "ROPE", "MoE", "LoRA", "RAG", "NLP", "OWASP",
    "CUDA", "HAL", "IFC", "CFI", "KPI", "MVP", "ETA", "AKA", "FYI", "TLDR",
}


def _worktree_clutter(root: Path) -> list[str]:
    """Untracked, non-ignored scratch files (the concurrency canary). Limited to
    scratch-shaped names so genuine in-progress source is not flagged."""
    others = _git_lines(["ls-files", "--others", "--exclude-standard"], root)
    scratch_re = re.compile(r"\.(txt|csv|log|tmp|out|err)$|(^|/)(report|agent-report)\.json$")
    out: list[str] = []
    for f in others:
        if f.startswith(tuple(EXEMPT_DATA_DIRS)):
            continue
        # a data file at the repo root, or a scratch-shaped name anywhere
        is_root_data = "/" not in f and f.split(".")[-1] in {"csv", "json", "txt", "log", "out", "err"}
        if is_root_data or scratch_re.search(f):
            out.append(f)
    return sorted(out)


def collect(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    if not (root / ".git").exists() and not _git_lines(["rev-parse", "--git-dir"], root):
        return build_payload(workspace=str(root), kpis=[],
                             error=f"not a git repo at {root} — run from the repo ROOT")
    kpis, clutter = gather(root)
    return build_payload(workspace=str(root), kpis=kpis, worktree_clutter=clutter)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    lines = [
        f"repo-hygiene-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· HYGIENE-DEBT {c.get('hygiene_debt', 0)} (a11y-debt {c.get('a11y_debt', 0)}) "
         f"· {c.get('soft_signals', 0)} advisory"),
        ("debt by group: " + "  ".join(
            f"{g}:{c.get('debt_by_group', {}).get(g, 0)}" for g in GROUPS)),
        "",
        "per-KPI (worst first):",
        f"  {'score':>5} {'debt':>4}  {'group':<13} {'kpi':<15} detail",
    ]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['group']:<13} "
                     f"{b['kpi']:<15} {b['detail']}")
    lines.append("")
    lines.append("hygiene-debt work-list:")
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
        lines.append("  (none — zero hygiene-debt)")
    clutter = c.get("worktree_clutter") or []
    if clutter:
        lines.append("")
        lines.append(f"worktree clutter (advisory, not debt — {len(clutter)} untracked scratch file(s)):")
        for f in clutter[:12]:
            lines.append(f"      · {f}")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    out: list[str] = []
    out.append("---")
    out.append('title: "fak repo-hygiene scorecard — the hygiene-debt measuring stick"')
    out.append('description: "fak\'s deterministic repo-hygiene scorecard: eleven KPIs across '
               'verbosity, organization, indexing, and accessibility, folded into a composite '
               'score and the headline hygiene-debt metric, re-derived from the git-tracked tree."')
    out.append("---")
    out.append("")
    out.append("# Repo-hygiene scorecard")
    out.append("")
    if stamp:
        out.append(f"<!-- repo-hygiene-scorecard: {stamp} · process: tools/repo_hygiene_scorecard.py -->")
        out.append("")
    out.append("This is the measuring stick for the repo-3x program — the structural counterpart "
               "of the docs and code scorecards. Every number below is re-derived from the "
               "git-tracked tree by `tools/repo_hygiene_scorecard.py` — no hand-entry. The headline "
               "metric is **hygiene-debt**: the count of concrete, mechanical structural defects "
               "you fix by *deleting, consolidating, moving, or indexing* — a duplicate doc, an "
               "oversized doc, root clutter, a misplaced dated note, an orphaned doc no index "
               "links, an AI-tell phrase. Driving hygiene-debt toward zero is what keeps the repo "
               "lean and findable as it grows.")
    out.append("")
    out.append("> Regenerate: `python tools/repo_hygiene_scorecard.py --markdown --stamp DATE > docs/REPO-HYGIENE-SCORECARD.md`")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Hygiene-debt (total HARD defects)** | **{c.get('hygiene_debt', 0)}** |")
    out.append(f"| **a11y-debt (accessibility HARD defects)** | **{c.get('a11y_debt', 0)}** |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append(f"| Advisory (soft) signals | {c.get('soft_signals', 0)} |")
    g = c.get("debt_by_group", {})
    out.append(f"| Debt by group | verbosity:{g.get('verbosity',0)} · organization:{g.get('organization',0)} "
               f"· indexing:{g.get('indexing',0)} · accessibility:{g.get('accessibility',0)} |")
    out.append("")
    out.append("## Per-KPI")
    out.append("")
    out.append("Twelve KPIs, each 0–100, in four groups. `debt` = units of HARD hygiene-debt. "
               "The accessibility group's HARD KPIs (`alt_text`, `ai_tells`) sum to **a11y-debt**. "
               "`jargon` and `plain_language` are advisory (they score but emit no hard debt — "
               "gaming a gloss is not clarity).")
    out.append("")
    out.append("| Group | KPI | Score | Debt | Detail |")
    out.append("|---|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    out.append("## Hygiene-debt work-list")
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
        out.append("No hygiene-debt: the tree is lean, well-placed, fully indexed, and reads plainly. 🎉")
        out.append("")
    return "\n".join(out)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("hygiene_debt", 0), cur.get("hygiene_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ba, ca = b.get("a11y_debt", 0), cur.get("a11y_debt", 0)
    ratio = "∞ (zero)" if cd == 0 else f"{bd / cd:.1f}×"
    lines = [
        f"hygiene-debt: {bd} -> {cd}   ({ratio} fewer defects)",
        f"a11y-debt:    {ba} -> {ca}",
        f"score:        {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<13} {gb} -> {gc}")
    target = max(1, bd // 3)
    if cd <= target:
        lines.append(f"VERDICT: >=3x hygiene-debt reduction achieved ({bd} -> {cd}).")
    else:
        lines.append(f"VERDICT: not yet 3x — need hygiene-debt <= {target} (now {cd}).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repo-hygiene scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the snapshot markdown body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the hygiene-debt delta vs a prior baseline JSON (proves the Nx)")
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
