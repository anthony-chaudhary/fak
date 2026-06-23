#!/usr/bin/env python3
"""Core-docs scorecard — the measuring stick that makes "better docs" provable.

``README.md`` already has a checking layer (``readme_freshness_audit.py``). The
problem that layer does *not* solve: the README is one page, and the project
points adopters at ~two dozen *other* surfaces — the tutorial, the architecture
doc, the policy guide, the FAQ, the explainers. Those rot silently. A dead link
in ``fak/ARCHITECTURE.md`` or a stale version pin in ``fak/GETTING-STARTED.md``
is just as bad for a cold reader as one on the front page, but nothing watches
them. "Make the docs better" was, until now, an unfalsifiable claim — there was
no number to move.

This is that number. It scores every doc in a declared *core set* on five
mechanical KPIs, folds those into a per-doc grade and a corpus aggregate, and —
crucially — counts the corpus's **doc-debt**: the total number of concrete,
re-derivable *defects* (a dead link, a stale pin, an unresolved ``TODO``, a
missing title, a headline that leads with a strawman, an orphaned doc, an
uncovered topic). Doc-debt is an integer you can drive toward zero, so an
improvement program can state an honest, checkable target ("cut doc-debt 100×,
from N to N/100") instead of a vibe.

The five per-doc KPIs (each 0–100, content-only, deterministic — no model,
no network):

  freshness     no stale version pin, no unresolved TODO/TBD/WIP placeholder
  links         every local Markdown link target resolves on disk
  structure     exactly one H1 title, real sections, not a navigational dead-end
  readability   first-screen jargon carries a plain-language gloss (6th-grade law)
  evidence      no bolded headline leads with a "naive" baseline; claims are backed

Per-doc ``score`` is the weighted mean (links and freshness weigh most — they are
what rots). ``grade`` maps it to A–F. The corpus payload adds **coverage**: what
fraction of the core set is reachable from the front door (``README.md`` by link
BFS — an unreachable core doc is an orphan) and what fraction of the *expected
topics* (install, quickstart, architecture, security, …) a non-stub doc covers.

``ok`` is False iff any HARD defect exists (dead link, naive-led headline,
missing H1, orphan core doc, uncovered topic) — these are required edits, the
work-list. Soft signals (jargon, a headline number not yet mirrored in the
authority doc) are WARN/ADVISORY, never a gate, because voice and prose are
judgment calls — the same split ``readme_freshness_audit`` draws.

Read-only by construction: it reads docs, the ``VERSION`` file, and
``fak/BENCHMARK-AUTHORITY.md``; it edits nothing. Run from the repo ROOT::

    python tools/docs_scorecard.py            # human scorecard
    python tools/docs_scorecard.py --json     # machine payload (control-pane)
    python tools/docs_scorecard.py --markdown # the committed DOCS-SCORECARD.md body

The companion process is the docs-100x program: each FAIL is one unit of
doc-debt to retire; re-running proves the number moved.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any

SCHEMA = "fleet-docs-scorecard/1"

# Repo-root-relative inputs (best-effort; a missing one degrades a check, never errors).
VERSION_REL = "VERSION"
AUTHORITY_REL = "fak/BENCHMARK-AUTHORITY.md"
# Reachability is measured by link-BFS from the front doors. A core doc not
# reachable from any of them is an "orphan". The doors THEMSELVES are entry
# points, not destinations, so they are exempt from the orphan check (README is
# the repo home, INDEX is the full map, docs/index.md is the Pages landing).
FRONT_DOOR_REL = "README.md"  # primary BFS root
FRONT_DOORS = ["README.md", "INDEX.md", "docs/index.md"]
FRONT_DOOR_EXEMPT = {"README.md", "INDEX.md", "docs/index.md"}

# ---------------------------------------------------------------------------
# The CORE SET: the surfaces the project itself points adopters at. Derived from
# README's "Go deeper" table + START-HERE's "pick your path" + the project-meta
# files every reader expects. NOT every .md in the tree (the long tail of dated
# plan/results docs is deliberately out of scope) — only the docs a cold reader,
# reviewer, or skeptic is steered to. Edit this when the front door changes.
# ---------------------------------------------------------------------------
CORE_DOCS: list[str] = [
    # front door / navigation
    "README.md",
    "START-HERE.md",
    "INDEX.md",
    "INSTALL.md",
    "docs/index.md",
    # onboarding
    "fak/GETTING-STARTED.md",
    "docs/fak/tutorial.md",
    "fak/cmd/simpledemo/README.md",
    "docs/repro-packet.md",
    "docs/adoption-playbook.md",
    # conceptual
    "docs/concepts-and-story.md",
    "docs/explainers/policy-in-the-kernel.md",
    "docs/explainers/addressable-kv-cache.md",
    "docs/explainers/sota-optimizations.md",
    "EXPLAINER-trust-floor-two-lenses-2026-06-17.md",
    "docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md",
    # reference
    "fak/ARCHITECTURE.md",
    "fak/POLICY.md",
    "fak/CLAIMS.md",
    "fak/STATUS.md",
    "fak/EXTENDING.md",
    "fak/BENCHMARK-AUTHORITY.md",
    "docs/FAQ.md",
    # project meta
    "CONTRIBUTING.md",
    "SECURITY.md",
    "fak/README.md",
]

# Expected-topic coverage: a topic is COVERED if at least one of its candidate
# paths exists AND is non-stub. An uncovered topic is one unit of doc-debt — a
# hole in what a docs set must answer. Paths, not content-sniffing, so it is
# deterministic and obvious to audit.
EXPECTED_TOPICS: list[tuple[str, list[str]]] = [
    ("install", ["INSTALL.md", "fak/GETTING-STARTED.md", "install.sh"]),
    ("quickstart", ["START-HERE.md", "docs/fak/tutorial.md"]),
    ("architecture", ["fak/ARCHITECTURE.md"]),
    ("security_threat_model", ["SECURITY.md", "fak/POLICY.md"]),
    ("contributing", ["CONTRIBUTING.md"]),
    ("faq", ["docs/FAQ.md"]),
    ("benchmarks_evidence", ["fak/BENCHMARK-AUTHORITY.md", "fak/CLAIMS.md"]),
    ("extending_api", ["fak/EXTENDING.md"]),
    ("concepts", ["docs/concepts-and-story.md"]),
    ("license_cite", ["LICENSE", "CITATION.cff"]),
]

# Per-KPI weights for the per-doc score. links + freshness weigh most: they are
# the surfaces that ROT (a dead link, a stale pin) and the ones a cold reader
# hits hardest. Readability is real but a judgment call, so it weighs least.
KPI_WEIGHTS: dict[str, float] = {
    "freshness": 0.20,
    "links": 0.30,
    "structure": 0.20,
    "readability": 0.10,
    "evidence": 0.20,
}

# Directories whose markdown is NOT reader-facing project documentation:
# vendored deps, VCS internals, agent scratch, run archives. Excluded from the
# `reachable` and `all` scopes so the corpus is project docs only.
DOC_EXCLUDE_DIRS = {
    ".git", "node_modules", ".opencode", ".pytest_cache", "__pycache__",
    ".goal-runs", ".dos", ".claude", "visuals",
}

# Tool-generated artifacts: never scored. The scorecard's own output is a list of
# defect *descriptions* (it literally contains the strings "dead link", "naive
# baseline", stale-looking pins from the work-list) — scoring it is circular and
# its headline would oscillate as it re-reads a stale copy of itself. You don't
# lint your linter's output. It stays linked in nav (discoverable), just unscored.
GENERATED_DOCS = {"docs/DOCS-SCORECARD.md"}

MIN_NONSTUB_CHARS = 400  # below this a doc is a stub for coverage purposes
FIRST_SCREEN_LINES = 60  # where we measure jargon density (the cold-read top)

# Unresolved-placeholder markers: content a reader should never meet on a core
# doc. Each DISTINCT marker in a doc is one unit of doc-debt. Deliberately the
# unambiguous set — XXX / WIP / the bare word "placeholder" are excluded because
# they fire on template rows ("XXX ms | YYY ms"), CI labels, and meta-prose.
PLACEHOLDER_RE = re.compile(
    r"\b(TODO|TBD|FIXME|coming soon|to be written|lorem ipsum)\b",
    re.IGNORECASE,
)

# A version pin is only "stale install advice" when it sits on an install/pin
# line. A bare "v0.2.1" in a changelog or a status table is HISTORY, not debt —
# flagging it was the noisiest false positive, so we gate on context.
# Assignment- and URL-anchored only. A version is "stale install advice" when it
# is ASSIGNED to a version var, passed as a build-arg, named in a package-install
# verb, or sits in a GitHub release-download URL. Deliberately NOT matched (these
# were the false positives): a benchmark-lineage field ("app_version 0.22.0" —
# version-at-run-time, no '='), a version inside a filename ("docs/releases/
# v0.9.0-candidate-strategy.md"), and incidental prose ("verified installed …
# v0.28.0"). The literal "FAK_VERSION" env-var name and "pin a version" phrasing
# stay because they only ever appear in install instructions.
_INSTALL_CTX_RE = re.compile(
    r"(FAK_VERSION|[A-Za-z_]*VERSION\s*=|@v?\d+\.\d|releases/(?:download|tag|latest)|"
    r"pip install|brew install|go install|pin a version|--build-arg)",
    re.IGNORECASE,
)

# Same jargon list the README auditor uses — expert terms that stumble a
# 6th-grade reader on the first screen if they appear with no nearby gloss.
JARGON_TERMS = [
    "vDSO", "context-MMU", "IPC", "RadixAttention", "KV cache", "KV-cache",
    "prefix reuse", "append-only", "core dump", "address space",
    "fail-open", "default-deny", "adjudicat", "idempoten", "quiesce",
]

# Regexes shared with the README auditor (kept local so this tool stands alone).
_LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)")
_VERSION_RE = re.compile(r"\bv?(\d+)\.(\d+)\.(?:(\d+)|x)\b")
_BOLD_RE = re.compile(r"\*\*(?P<body>[^*]+)\*\*")
_MULT_RE = re.compile(r"~?\d[\d.,]*\s*(?:[–-]\s*\d[\d.,]*\s*)?[×x]")
_H1_RE = re.compile(r"^#\s+\S", re.MULTILINE)
_H2PLUS_RE = re.compile(r"^#{2,6}\s+\S", re.MULTILINE)
# A quantitative claim a doc might make: a multiplier, a percent, or a speed word.
_CLAIM_RE = re.compile(r"(~?\d[\d.,]*\s*[×x]\b|\b\d[\d.,]*\s*%|\bfaster\b|\bslower\b|\bspeedup\b)",
                       re.IGNORECASE)


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each takes already-read text (+ context) and returns
#   {kpi, score (0-100 int), detail, defects: [str], soft: [str]}
# where every item in `defects` is one HARD unit of doc-debt and every item in
# `soft` is a judgment-call nudge (never gates `ok`). This is the testable seam:
# tests pass fixture strings, no disk needed.
# ---------------------------------------------------------------------------

def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


def kpi_freshness(text: str, version: str | None) -> dict[str, Any]:
    defects: list[str] = []
    soft: list[str] = []
    cur = _parse_version(version or "")
    stale_versions: set[str] = set()
    if cur is not None:
        cur_major, cur_minor, _ = cur
        for line in text.splitlines():
            if not _INSTALL_CTX_RE.search(line):
                continue  # history/changelog mention, not install advice
            for m in _VERSION_RE.finditer(line):
                # Skip dotted-quad IPs like 0.0.0.0:8080 — the match is followed
                # by ".<digit>", so it is an address octet, not a version.
                tail = line[m.end():m.end() + 2]
                if tail[:1] == "." and tail[1:2].isdigit():
                    continue
                major, minor = int(m.group(1)), int(m.group(2))
                if major == cur_major and (major, minor) < (cur_major, cur_minor):
                    stale_versions.add(m.group(0))
    for v in sorted(stale_versions):
        defects.append(f"stale install pin {v} (VERSION is {version.strip()})")
    placeholders = sorted({m.group(0).lower() for m in PLACEHOLDER_RE.finditer(text)})
    for ph in placeholders:
        defects.append(f"unresolved placeholder: {ph!r}")
    score = 100 - 20 * len(stale_versions) - 12 * len(placeholders)
    return {
        "kpi": "freshness", "score": _clamp(score),
        "detail": ("no stale pin or placeholder" if not defects
                   else f"{len(defects)} freshness defect(s)"),
        "defects": defects, "soft": soft,
    }


def kpi_links(text: str, root: Path, doc_rel: str) -> dict[str, Any]:
    """Resolve every LOCAL link relative to the doc's own directory.

    A Markdown link is relative to the file it lives in, not the repo root —
    ``fak/GETTING-STARTED.md`` linking ``examples/x.json`` means ``fak/examples``.
    Network links, ``mailto:`` and bare ``#anchors`` are out of scope (the
    network is not ours to witness).
    """
    base = (root / doc_rel).parent
    dead: list[str] = []
    seen: set[str] = set()
    total = 0
    for m in _LINK_RE.finditer(text):
        target = m.group("target").strip()
        if target.startswith(("http://", "https://", "mailto:", "#", "tel:")):
            continue
        path_part = target.split("#", 1)[0].split("?", 1)[0].strip()
        if not path_part or path_part in seen:
            continue
        seen.add(path_part)
        total += 1
        # Resolve absolute-from-root ("/foo") at root, else relative to the doc.
        resolved = (root / path_part.lstrip("/")) if path_part.startswith("/") \
            else (base / path_part)
        if not resolved.exists():
            dead.append(path_part)
    defects = [f"dead link: {d}" for d in sorted(dead)]
    score = 100 - 20 * len(dead)
    return {
        "kpi": "links", "score": _clamp(score),
        "detail": (f"all {total} local link(s) resolve" if not dead
                   else f"{len(dead)}/{total} local link(s) dead"),
        "defects": defects, "soft": [],
    }


def kpi_structure(text: str, doc_rel: str) -> dict[str, Any]:
    defects: list[str] = []
    soft: list[str] = []
    n_lines = text.count("\n") + 1
    h1s = _H1_RE.findall(text)
    h2s = _H2PLUS_RE.findall(text)
    score = 100
    if len(h1s) == 0:
        defects.append("no H1 title (a '# Title' line)")
        score -= 30
    elif len(h1s) > 1:
        soft.append(f"{len(h1s)} H1 titles (expected exactly one)")
        score -= 10
    if n_lines > 40 and len(h2s) == 0:
        defects.append("long doc with no section headings (## …)")
        score -= 20
    # Navigational dead-end: a substantial doc that links to no OTHER local doc.
    out_links = _local_doc_links(text)
    if n_lines > 30 and not out_links:
        soft.append("no outbound links to other docs (navigational dead-end)")
        score -= 15
    # Wall of text: a very long stretch with no heading break.
    longest = _longest_unbroken_run(text)
    if longest > 160:
        soft.append(f"{longest}-line stretch with no heading (wall of text)")
        score -= 10
    return {
        "kpi": "structure", "score": _clamp(score),
        "detail": (f"{len(h1s)} H1 / {len(h2s)} sections / {len(out_links)} cross-link(s)"),
        "defects": defects, "soft": soft,
    }


def kpi_readability(text: str) -> dict[str, Any]:
    """Voice is a judgment call, so this KPI emits NO hard defects — only soft
    nudges. It still scores, so a jargon-dense doc grades lower, but it never
    flips ``ok``. Mirrors the README auditor's jargon_density = ADVISORY rule.
    """
    soft: list[str] = []
    head = "\n".join(text.splitlines()[:FIRST_SCREEN_LINES])
    naked: list[str] = []
    for term in JARGON_TERMS:
        for line in head.splitlines():
            if term.lower() in line.lower():
                glossed = ("(" in line) or ("—" in line) or (" - " in line)
                if not glossed:
                    naked.append(term)
                break
    naked = sorted(set(naked))
    score = 100 - 6 * len(naked)
    # A plain-language opener: the first real prose line should be a sentence,
    # not a heading/table/code dive. Soft nudge if the top is all scaffolding.
    if not _has_prose_opener(text):
        soft.append("no plain-language opening sentence before the first heading/code/table")
        score -= 15
    for t in naked:
        soft.append(f"first-screen jargon with no nearby gloss: {t}")
    return {
        "kpi": "readability", "score": _clamp(score),
        "detail": (f"{len(naked)} naked first-screen term(s)" if naked
                   else "first-screen reads with plain glosses"),
        "defects": [], "soft": soft,
    }


def kpi_evidence(text: str, authority: str) -> dict[str, Any]:
    defects: list[str] = []
    soft: list[str] = []
    score = 100
    # HARD: a bolded headline that LEADS with a 'naive' baseline (SOTA-not-naive
    # law). Exempt the HONEST two-number framing: a span that names naive AND its
    # tuned/SOTA counterpart (e.g. "60.3× vs naive · 4.1× vs tuned") is the ledger
    # showing both, not a strawman lead — that is exactly what the authority doc
    # is for. Only a naive multiplier with NO counter present is the violation.
    _counter = re.compile(r"\b(tuned|sota|state[- ]of[- ]the[- ]art|warm[- ]?cache|"
                          r"warm stack|best[- ]shipped)\b", re.IGNORECASE)
    for line in text.splitlines():
        for m in _BOLD_RE.finditer(line):
            body = m.group("body")
            if (_MULT_RE.search(body) and re.search(r"\bnaive\b", body, re.IGNORECASE)
                    and not _counter.search(body)):
                defects.append(f"bolded headline leads with a naive baseline: {body.strip()!r}")
                score -= 30
    # SOFT: a bolded multiplier not mirrored in the authority doc (reconcile by hand).
    auth_nums = {_norm_num(x) for x in _MULT_RE.findall(authority)} if authority else set()
    if auth_nums:
        unmirrored: list[str] = []
        for m in _BOLD_RE.finditer(text):
            for num in _MULT_RE.findall(m.group("body")):
                if _norm_num(num) not in auth_nums:
                    unmirrored.append(num.strip())
        for u in sorted(set(unmirrored)):
            soft.append(f"bolded number {u} not found in BENCHMARK-AUTHORITY")
            score -= 8
    # SOFT: a quantitative claim with zero backing (no link, no code, no number-source).
    if _CLAIM_RE.search(text) and not _LINK_RE.search(text) and "```" not in text:
        soft.append("makes a quantitative claim with no link or code block to back it")
        score -= 15
    return {
        "kpi": "evidence", "score": _clamp(score),
        "detail": ("no naive-led headline" if not defects
                   else f"{len(defects)} naive-led headline(s)"),
        "defects": defects, "soft": soft,
    }


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _parse_version(text: str) -> tuple[int, int, int] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text.strip())
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def _norm_num(s: str) -> str:
    return re.sub(r"[~\s]", "", s).replace("x", "×").replace("X", "×")


def _local_doc_links(text: str) -> list[str]:
    """Local links that point at another markdown doc (for navigability)."""
    out: list[str] = []
    for m in _LINK_RE.finditer(text):
        t = m.group("target").strip()
        if t.startswith(("http://", "https://", "mailto:", "#", "tel:")):
            continue
        path_part = t.split("#", 1)[0].split("?", 1)[0]
        if path_part.endswith(".md"):
            out.append(path_part)
    return out


def _longest_unbroken_run(text: str) -> int:
    longest = run = 0
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            run = 0
        else:
            run += 1
            longest = max(longest, run)
    return longest


def _has_prose_opener(text: str) -> bool:
    """True if a sentence-like prose line appears before the first heading/code/
    table after the H1. A cold reader should meet plain language, not scaffolding.
    """
    seen_h1 = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# "):
            seen_h1 = True
            continue
        if not seen_h1:
            continue
        if line.startswith(("#", "```", "|", "<!--", "![", "- ", "* ", "1.")):
            # scaffolding before any prose — keep looking only a little
            if line.startswith(("#", "```", "|")):
                return False
            continue
        # a prose line: has letters and ends like a sentence or is long enough
        if re.search(r"[A-Za-z]", line) and (len(line) > 40 or line.endswith((".", ":", "!"))):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-doc fold
# ---------------------------------------------------------------------------

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


def score_doc(text: str, doc_rel: str, root: Path, *, version: str | None,
              authority: str) -> dict[str, Any]:
    kpis = [
        kpi_freshness(text, version),
        kpi_links(text, root, doc_rel),
        kpi_structure(text, doc_rel),
        kpi_readability(text),
        kpi_evidence(text, authority),
    ]
    by_name = {k["kpi"]: k for k in kpis}
    score = sum(KPI_WEIGHTS[name] * by_name[name]["score"] for name in KPI_WEIGHTS)
    defects = [f"{k['kpi']}: {d}" for k in kpis for d in k["defects"]]
    soft = [f"{k['kpi']}: {s}" for k in kpis for s in k["soft"]]
    return {
        "path": doc_rel,
        "score": round(score, 1),
        "grade": grade_letter(score),
        "kpis": {k["kpi"]: k["score"] for k in kpis},
        "kpi_detail": {k["kpi"]: k["detail"] for k in kpis},
        "defects": defects,
        "soft": soft,
        "n_defects": len(defects),
    }


def missing_doc_entry(doc_rel: str) -> dict[str, Any]:
    """A core doc that does not exist on disk is the worst defect: a 0 with one
    hard unit of doc-debt for being absent."""
    return {
        "path": doc_rel, "score": 0.0, "grade": "F",
        "kpis": {k: 0 for k in KPI_WEIGHTS}, "kpi_detail": {},
        "defects": [f"missing: core doc {doc_rel} does not exist on disk"],
        "soft": [], "n_defects": 1,
    }


# ---------------------------------------------------------------------------
# Coverage: reachability (orphans) + expected-topic holes
# ---------------------------------------------------------------------------

def reachable_md(root: Path, starts: list[str] | str) -> set[str]:
    """BFS over local .md links from the front door(s); returns the reachable set
    (repo-relative POSIX paths). An existing core doc NOT in this set (and not a
    front door itself) is an orphan — present but unreachable from any entry."""
    start_list = [starts] if isinstance(starts, str) else starts
    seeds = [(root / s).resolve() for s in start_list if (root / s).exists()]
    if not seeds:
        return set()
    visited: set[str] = set()
    q: deque[Path] = deque(seeds)
    while q:
        cur = q.popleft()
        try:
            rel = cur.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        if rel in visited:
            continue
        visited.add(rel)
        try:
            txt = cur.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in _LINK_RE.finditer(txt):
            t = m.group("target").strip()
            if t.startswith(("http://", "https://", "mailto:", "#", "tel:")):
                continue
            path_part = t.split("#", 1)[0].split("?", 1)[0]
            if not path_part.endswith(".md"):
                continue
            nxt = (root / path_part.lstrip("/")) if path_part.startswith("/") \
                else (cur.parent / path_part)
            if nxt.exists():
                q.append(nxt)
    return visited


def coverage(root: Path, core_docs: list[str]) -> dict[str, Any]:
    reachable = reachable_md(root, FRONT_DOORS)
    # Orphan check applies to destination docs only; the entry pages are exempt.
    existing = [d for d in core_docs if (root / d).exists() and d not in FRONT_DOOR_EXEMPT]
    orphans = sorted(d for d in existing if d not in reachable)
    reach_pct = round(100 * (len(existing) - len(orphans)) / max(1, len(existing)), 1)

    covered: list[str] = []
    missing_topics: list[str] = []
    for topic, candidates in EXPECTED_TOPICS:
        ok = any((root / c).exists() and _nonstub(root / c) for c in candidates)
        (covered if ok else missing_topics).append(topic)
    topic_pct = round(100 * len(covered) / max(1, len(EXPECTED_TOPICS)), 1)

    overall = round((reach_pct + topic_pct) / 2, 1)
    # Coverage defects: each orphan + each uncovered topic is one unit of doc-debt.
    cov_defects = [f"orphan core doc (unreachable from {FRONT_DOOR_REL}): {o}" for o in orphans] \
        + [f"uncovered expected topic: {t}" for t in missing_topics]
    return {
        "reachability_pct": reach_pct,
        "topic_pct": topic_pct,
        "overall_pct": overall,
        "orphans": orphans,
        "missing_topics": missing_topics,
        "covered_topics": covered,
        "defects": cov_defects,
    }


def _nonstub(path: Path) -> bool:
    try:
        return len(path.read_text(encoding="utf-8").strip()) >= MIN_NONSTUB_CHARS
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Grader: fold per-doc scores + coverage into the standard control-pane payload
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, docs: list[dict[str, Any]],
                  cov: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error,
            "next_action": "fix the read (run from repo ROOT), then re-run",
            "workspace": workspace, "corpus": {}, "coverage": cov, "docs": docs,
        }
    n = len(docs)
    scores = [d["score"] for d in docs]
    doc_defects = sum(d["n_defects"] for d in docs)
    total_defects = doc_defects + len(cov.get("defects", []))
    mean_score = round(sum(scores) / max(1, n), 1)
    grades = {g: 0 for g in "ABCDF"}
    for d in docs:
        grades[d["grade"]] = grades.get(d["grade"], 0) + 1
    worst = sorted(docs, key=lambda d: (d["score"], -d["n_defects"]))[:8]

    corpus = {
        "n_docs": n,
        "mean_score": mean_score,
        "median_score": round(sorted(scores)[n // 2], 1) if n else 0.0,
        "min_score": round(min(scores), 1) if scores else 0.0,
        "max_score": round(max(scores), 1) if scores else 0.0,
        "grade_distribution": grades,
        "doc_debt": total_defects,
        "doc_debt_in_docs": doc_defects,
        "doc_debt_in_coverage": len(cov.get("defects", [])),
        "worst": [{"path": d["path"], "score": d["score"], "grade": d["grade"],
                   "n_defects": d["n_defects"]} for d in worst],
    }

    if total_defects == 0:
        ok, verdict, finding = True, "OK", "docs_clean"
        reason = (f"core docs clean: {n} docs, mean {mean_score}/100, "
                  f"coverage {cov['overall_pct']}%, zero doc-debt")
        next_action = "no required edit; re-run after the next docs change"
    else:
        ok, verdict, finding = False, "ACTION", "doc_debt"
        reason = (f"{total_defects} unit(s) of doc-debt across {n} core docs "
                  f"({doc_defects} in-doc + {len(cov.get('defects', []))} coverage); "
                  f"mean {mean_score}/100, coverage {cov['overall_pct']}%")
        next_action = ("retire doc-debt worst-first (see corpus.worst + coverage.defects): "
                       "fix dead links, stale pins, missing titles, orphans, uncovered topics; re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "coverage": cov, "docs": docs,
    }


# ---------------------------------------------------------------------------
# Wiring + CLI
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _excluded(rel: str) -> bool:
    if rel in GENERATED_DOCS:
        return True
    parts = set(Path(rel).parts)
    return bool(parts & DOC_EXCLUDE_DIRS)


def enumerate_docs(root: Path, scope: str) -> list[str]:
    """The doc set for a scope:
      core      — the curated CORE_DOCS manifest (must-be-perfect surfaces).
      reachable — every project .md a reader can navigate to from a front door
                  (link-BFS closure) — the real reader-facing surface.
      all       — every project .md in the tree (minus vendored/scratch dirs).
    """
    if scope == "core":
        return list(CORE_DOCS)
    if scope == "reachable":
        reach = reachable_md(root, FRONT_DOORS)
        docs = sorted(r for r in reach if not _excluded(r))
        # the front doors are always in the set even if self-referential
        for d in FRONT_DOORS:
            if (root / d).exists() and d not in docs and not _excluded(d):
                docs.append(d)
        return sorted(docs)
    if scope == "all":
        out: list[str] = []
        for p in sorted(root.rglob("*.md")):
            rel = p.relative_to(root).as_posix()
            if not _excluded(rel):
                out.append(rel)
        return out
    raise ValueError(f"unknown scope {scope!r} (core|reachable|all)")


def collect(workspace: Path, *, core_docs: list[str] | None = None,
            scope: str = "core") -> dict[str, Any]:
    root = workspace.resolve()
    core = core_docs if core_docs is not None else enumerate_docs(root, scope)
    version = _safe_read(root / VERSION_REL) or None
    authority = _safe_read(root / AUTHORITY_REL)
    docs: list[dict[str, Any]] = []
    for rel in core:
        path = root / rel
        if not path.exists():
            docs.append(missing_doc_entry(rel))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            d = missing_doc_entry(rel)
            d["defects"] = [f"unreadable: {rel}: {exc}"]
            docs.append(d)
            continue
        docs.append(score_doc(text, rel, root, version=version, authority=authority))
    cov = coverage(root, core)
    payload = build_payload(workspace=str(root), docs=docs, cov=cov)
    payload["scope"] = scope
    return payload


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    cov = payload.get("coverage") or {}
    lines = [
        f"docs-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"corpus: {c.get('n_docs', 0)} docs · mean {c.get('mean_score', 0)}/100 "
         f"· min {c.get('min_score', 0)} · DOC-DEBT {c.get('doc_debt', 0)}"),
        ("grades: " + " ".join(f"{g}:{c.get('grade_distribution', {}).get(g, 0)}"
                                for g in "ABCDF")),
        (f"coverage: {cov.get('overall_pct', 0)}% "
         f"(reach {cov.get('reachability_pct', 0)}% · topics {cov.get('topic_pct', 0)}%)"),
        f"next: {payload.get('next_action')}",
        "",
        "per-doc (worst first):",
        f"  {'score':>5} {'gr':>2} {'def':>3}  path",
    ]
    for d in sorted(payload.get("docs", []), key=lambda x: (x["score"], -x["n_defects"])):
        lines.append(f"  {d['score']:>5} {d['grade']:>2} {d['n_defects']:>3}  {d['path']}")
    # Defect detail for the worst few + coverage holes.
    lines.append("")
    lines.append("doc-debt detail (top defectful docs):")
    worst = sorted(payload.get("docs", []), key=lambda x: -x["n_defects"])[:8]
    for d in worst:
        if not d["defects"]:
            continue
        lines.append(f"  {d['path']} ({d['n_defects']}):")
        for it in d["defects"][:8]:
            lines.append(f"      - {it}")
    if cov.get("defects"):
        lines.append("  coverage:")
        for it in cov["defects"]:
            lines.append(f"      - {it}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    """The committed DOCS-SCORECARD.md body — a human-facing snapshot."""
    c = payload.get("corpus") or {}
    cov = payload.get("coverage") or {}
    scope = payload.get("scope", "core")
    gd = c.get("grade_distribution", {})
    out: list[str] = []
    out.append("# Core-docs scorecard")
    out.append("")
    if stamp:
        out.append(f"<!-- docs-scorecard: {stamp} · scope={scope} · process: tools/docs_scorecard.py -->")
        out.append("")
    out.append(f"> Scope: **{scope}** "
               f"({'curated must-be-perfect set' if scope == 'core' else 'every doc a reader can reach from a front door' if scope == 'reachable' else 'whole tree, minus vendored/scratch dirs'}). "
               f"Regenerate: `python tools/docs_scorecard.py --scope {scope} --markdown --stamp DATE > docs/DOCS-SCORECARD.md`")
    out.append("")
    out.append("> Other scopes: `--scope core` (the 26 curated surfaces), "
               "`--scope all` (the full tree, surfaces archival orphans).")
    out.append("")
    out.append("This is the measuring stick for the docs-100x program. Every number "
               "below is re-derived from disk by `tools/docs_scorecard.py` — no hand-entry. "
               "The headline metric is **doc-debt**: the count of concrete, mechanical "
               "defects (dead links, stale version pins, unresolved placeholders, missing "
               "titles, strawman-led headlines, orphaned docs, uncovered topics). Driving "
               "doc-debt toward zero is what makes \"better docs\" provable.")
    out.append("")
    out.append("## Corpus")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Core docs scored | {c.get('n_docs', 0)} |")
    out.append(f"| **Doc-debt (total defects)** | **{c.get('doc_debt', 0)}** "
               f"({c.get('doc_debt_in_docs', 0)} in-doc + {c.get('doc_debt_in_coverage', 0)} coverage) |")
    out.append(f"| Mean score | {c.get('mean_score', 0)}/100 |")
    out.append(f"| Median / min / max | {c.get('median_score', 0)} / {c.get('min_score', 0)} / {c.get('max_score', 0)} |")
    out.append(f"| Grade distribution | A:{gd.get('A',0)} B:{gd.get('B',0)} C:{gd.get('C',0)} D:{gd.get('D',0)} F:{gd.get('F',0)} |")
    out.append(f"| Coverage (overall) | {cov.get('overall_pct', 0)}% |")
    out.append(f"| — reachable from README | {cov.get('reachability_pct', 0)}% |")
    out.append(f"| — expected topics covered | {cov.get('topic_pct', 0)}% |")
    out.append("")
    out.append("## Per-doc scores")
    out.append("")
    out.append("Five KPIs, each 0–100 (freshness · links · structure · readability · evidence), "
               "weighted into a score and an A–F grade. `def` = units of doc-debt in that doc.")
    out.append("")
    out.append("| Score | Grade | Debt | fresh | link | struct | read | evid | Doc |")
    out.append("|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|")
    for d in sorted(payload.get("docs", []), key=lambda x: (x["score"], -x["n_defects"])):
        k = d.get("kpis", {})
        out.append(
            f"| {d['score']} | {d['grade']} | {d['n_defects']} | "
            f"{k.get('freshness','-')} | {k.get('links','-')} | {k.get('structure','-')} | "
            f"{k.get('readability','-')} | {k.get('evidence','-')} | `{d['path']}` |")
    out.append("")
    # Defect work-list.
    out.append("## Doc-debt work-list")
    out.append("")
    any_defect = False
    for d in sorted(payload.get("docs", []), key=lambda x: -x["n_defects"]):
        if not d["defects"]:
            continue
        any_defect = True
        out.append(f"### `{d['path']}` — {d['n_defects']} defect(s), score {d['score']} ({d['grade']})")
        for it in d["defects"]:
            out.append(f"- {it}")
        out.append("")
    if cov.get("defects"):
        any_defect = True
        out.append("### coverage")
        for it in cov["defects"]:
            out.append(f"- {it}")
        out.append("")
    if not any_defect:
        out.append("No doc-debt: every core doc is clean. 🎉")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Core-docs scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--scope", default="core", choices=["core", "reachable", "all"],
                    help="doc set to score (default: core)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true",
                    help="emit the DOCS-SCORECARD.md body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    args = ap.parse_args(argv)

    # Docs carry Unicode (×, ≈, ·, —); force UTF-8 stdout so a Windows cp1252
    # console can't crash the audit on an em-dash.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, scope=args.scope)

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    else:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
