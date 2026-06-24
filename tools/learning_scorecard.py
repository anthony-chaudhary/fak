#!/usr/bin/env python3
"""Learning-docs scorecard — the measuring stick for whether the *teaching* docs
actually teach.

``docs_scorecard.py`` already scores the front-door surfaces on mechanical
HYGIENE (dead links, stale pins, structure). It answers "does this doc rot?".
It does NOT answer the question this tool exists for: **does this doc teach?**
A tutorial can have zero dead links, a perfect H1, and a fresh version pin — and
still drop a cold reader cold: no audience cue, no runnable command, no worked
output to compare against, a speed number stated as fact with no honesty marker.
Mechanically an A; pedagogically a wall. The learning corpus — the university
(``LEARNING-PATH.md``), the how-to set (``docs/fak/*``), and the concept
explainers (``docs/explainers/*``) — was, until now, unmeasured on the only axis
that matters for it: whether someone can learn from it.

This is that measuring stick. It scores every doc in a declared *learning set* on
seven KPIs — four pedagogy-specific, three mechanical (reused from the hygiene
scorecard so the two never disagree) — folds them into a per-doc grade and a
corpus aggregate, and counts the corpus's **learning-debt**: the total number of
concrete, re-derivable teaching *defects* (a how-to with no runnable command, a
tutorial with no worked output, a teaching doc with no orientation signpost, an
orphan lesson unreachable from any front door, an uncovered learning topic).
Learning-debt is an integer you can drive toward zero, so a "make the docs teach
2× better" program can state an honest, checkable target ("halve learning-debt,
then halve it again") instead of a vibe.

The four pedagogy KPIs (each 0–100, content-only, deterministic — no model,
no network):

  orientation   tells the reader where they are: an audience / prereq / TL;DR /
                "you'll be able to" signpost up top, and a way to navigate onward
                (a link to another learning doc; a "where to go next" pointer)
  runnable      a how-to / tutorial / course teaches at least one runnable command
                or code block — prose-only "how-to" is the cardinal sin
  worked        it SHOWS, not just tells: a lab, a checkpoint, a worked example, or
                expected output a learner can compare their own run against
  honesty       a speed / multiplier claim carries the fak honesty discipline — a
                SIMULATED/OPEN/vs-naive/vs-tuned marker or an authority link — so a
                learner is never taught a strawman number as if it were the bar

The three mechanical KPIs (``structure``, ``links``, ``freshness``) are imported
verbatim from ``docs_scorecard`` — a broken link in a lab or a ``TODO`` in a
lesson is a teaching defect too, and reusing them guarantees the learning grade
never contradicts the hygiene grade.

Per-doc ``score`` is the weighted mean (orientation + runnable weigh most — they
are what makes or breaks a first read). ``grade`` maps it to A–F. The corpus
payload adds **importance**: each learning doc is ranked by how central it is to
the funnel — link-hop proximity to the front doors plus in-degree from other
learning docs — so "the most important ones" is an objective ordering, not a
guess. ``priorities`` is the importance×debt-ranked work-list: fix the docs that
are both most-read and most-broken first.

``ok`` is False iff any HARD defect exists (a required teaching affordance is
missing) — these are the work-list. Soft signals (a single-block how-to, a
missing "next step", a naked perf number) are WARN/ADVISORY, never a gate,
because voice and depth are judgment calls — the same split the hygiene scorer
draws.

Read-only by construction. Run from the repo ROOT::

    python tools/learning_scorecard.py            # human scorecard
    python tools/learning_scorecard.py --json     # machine payload (control-pane)
    python tools/learning_scorecard.py --markdown # the committed LEARNING-SCORECARD.md body

The companion process is the learning-2× program: each FAIL is one unit of
learning-debt to retire; re-running proves the number moved. The pedagogy
counterpart of ``docs_scorecard`` (hygiene), ``doc_appeal_scorecard`` (voice),
and ``agent_readiness_scorecard`` (agent affordances).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docs_scorecard as dsc  # noqa: E402  reuse the hygiene scorer's pure helpers

SCHEMA = "fleet-learning-scorecard/1"

VERSION_REL = "VERSION"
AUTHORITY_REL = "fak/BENCHMARK-AUTHORITY.md"

# Front doors a learner could plausibly enter through. A learning doc reachable
# from none of them is an orphan lesson (present but un-findable). The doors
# themselves are entry points, not destinations, so they are orphan-exempt.
FRONT_DOORS = [
    "README.md", "START-HERE.md", "LEARNING-PATH.md",
    "docs/index.md", "docs/fak/README.md", "INDEX.md",
]
FRONT_DOOR_EXEMPT = set(FRONT_DOORS)

# ---------------------------------------------------------------------------
# The LEARNING SET: the curriculum, the how-to/operator set, the concept
# explainers, and the onboarding roots. NOT every .md — the reference dumps,
# release notes, dated plan/results notes, and proofs (the graduate seminar) are
# out of scope here; they are scored by docs_scorecard on hygiene. This is the
# set whose JOB is to teach a human. Edit when a new teaching surface lands.
# ---------------------------------------------------------------------------
LEARNING_ROOTS: list[str] = [
    "LEARNING-PATH.md",            # the university (98 courses)
    "START-HERE.md",               # pick-your-path front door
    "GETTING-STARTED.md",          # install + four usage tiers
    "INSTALL.md",                  # install reference
    "docs/run-the-demos.md",       # guided demo walk
    "docs/FAQ.md",                 # honest short answers
    "docs/glossary.md",            # the vocabulary
    "docs/concepts-and-story.md",  # the parable + the two gates
    "docs/CONTEXT-IS-NOT-MEMORY.md",
    "docs/MEMORY-LAYERS-EXPLAINER.md",
    "docs/prefill-elimination-explained.md",
]
# Every markdown file under these directories joins the set automatically.
LEARNING_DIRS: list[str] = ["docs/fak", "docs/explainers"]

# Files under the learning dirs that are NOT themselves lessons (nav/meta) — kept
# in the set (they are still scored for hygiene) but typed `reference`, which
# relaxes the runnable/worked HARD requirements that only make sense for a lesson.
META_NAMES = {
    "readme.md", "index.md", "documentation-roadmap.md", "related-items.md",
    "glossary.md", "api-reference.md", "serve-config.md", "server-config.md",
    "mcp-tool-result.md", "video-content-plan.md",
}

# Expected learning-topic coverage: a topic is COVERED iff at least one candidate
# learning doc exists AND is non-stub. An uncovered topic is one unit of
# learning-debt — a hole in what the teaching set must answer.
EXPECTED_TOPICS: list[tuple[str, list[str]]] = [
    ("orientation_path", ["LEARNING-PATH.md", "START-HERE.md"]),
    ("install", ["INSTALL.md", "GETTING-STARTED.md"]),
    ("first_run_tutorial", ["docs/fak/tutorial.md"]),
    ("policy_authoring", ["docs/fak/policy-guide.md"]),
    ("framework_integration", ["docs/fak/agent-framework-integration.md"]),
    ("client_examples", ["docs/fak/multi-language-examples.md"]),
    ("observability", ["docs/fak/observability.md"]),
    ("security_hardening", ["docs/fak/security.md"]),
    ("troubleshooting", ["docs/fak/server-troubleshooting.md"]),
    ("faq", ["docs/fak/faq.md", "docs/FAQ.md"]),
    ("concepts", ["docs/concepts-and-story.md"]),
    ("migration", ["docs/fak/migration-guide.md"]),
]

# Per-KPI weights. orientation + runnable weigh most: a learner who can't tell
# where to start or has no command to run never reaches the rest. The three
# mechanical KPIs together weigh ~0.34 — real, but a clean lesson is more than a
# link check.
KPI_WEIGHTS: dict[str, float] = {
    "orientation": 0.20,
    "runnable": 0.18,
    "worked": 0.15,
    "honesty": 0.13,
    "structure": 0.12,
    "links": 0.14,
    "freshness": 0.08,
}

MIN_NONSTUB_CHARS = 400
FIRST_SCREEN_LINES = 80  # learning docs front-load orientation; look a bit deeper

# Which doc-types must satisfy which pedagogy KPI as a HARD requirement (else the
# defect is soft). A reference/index page is exempt from "needs a runnable
# command"; a tutorial is not.
RUNNABLE_HARD = {"tutorial", "howto", "curriculum"}
WORKED_HARD = {"tutorial", "curriculum"}
ORIENT_HARD = {"tutorial", "howto", "curriculum", "explainer"}

# Orientation signposts: any one of these in the first screen means the reader is
# told where they are / what they need / what they'll get.
_SIGNPOSTS = [
    "audience", "prereq", "you'll be able", "you will be able", "by the end",
    "who joins", "who this is for", "assumes you", "start here", "what you'll",
    "what you will", "before you", "time:", "tl;dr", "this guide", "this page",
    "this tutorial", "this explainer", "read this if", "in this tutorial",
    "in this guide", "what you need",
]
# Onward-navigation cues (a teaching doc should point somewhere next).
_NEXT_CUES = [
    "next step", "where to go", "go deeper", "what's next", "what is next",
    "read next", "see also", "further reading", "from here", "continue to",
    "you've finished", "you have finished", "next:",
]
# Show-don't-tell markers: a lab, a checkpoint, worked output, a numbered walk.
_WORKED_MARKERS = [
    "lab:", "**lab", "checkpoint:", "**checkpoint", "you'll have seen",
    "you will have seen", "expected output", "example output", "worked example",
    "walkthrough", "try it", "step 1", "part 0", "part 1", "output:",
    "what you'll have", "you should see", "you'll see", "real, unedited",
]

# Honesty markers + authority pointers that make a perf number honest teaching.
_HONESTY_RE = re.compile(
    r"\b(SIMULATED|OPEN|PROVEN|REFUTED|SCOPED|PARTIAL|honest|caveat|tombstone|"
    r"vs\.?\s*naive|vs\.?\s*tuned|vs\.?\s*sota|measured|modeled|modelled|"
    r"not\s+a\s+win|does\s+not\s+win)\b",
    re.IGNORECASE,
)
_AUTHORITY_RE = re.compile(
    r"(BENCHMARK-AUTHORITY|CLAIMS\.md|/proofs/|authority\s+doc|claims\s+ledger)",
    re.IGNORECASE,
)

# Install-context regex — a TIGHTER variant of docs_scorecard's. The hygiene
# scorer matches a lowercase `version=` so it can never miss a pin; but learning
# docs are dense with *captured metric output* (Prometheus `version="0.30.0"`,
# the `text/plain; version=0.0.4` exposition-format constant) where a lowercase
# `version=` is the honest record of a build, NOT stale install advice — flagging
# it would push an editor to corrupt true output. So here the env-var-assignment
# alternative is case-SENSITIVE uppercase (the real convention: `FAK_VERSION=`,
# `VERSION=`, `APP_VERSION=`); a lowercase `version=` no longer counts. Every
# genuine install pin (uppercase env var, `go/pip/brew install`, a release URL,
# a build-arg, the literal "pin a version") still matches. No global IGNORECASE,
# so no scoped-flag dependency — portable to any 3.10+ runtime.
_INSTALL_CTX_RE = re.compile(
    r"(FAK_VERSION|[A-Z_]*VERSION\s*=|@v?\d+\.\d|releases/(?:download|tag|latest)|"
    r"pip install|brew install|go install|[Pp]in a version|--build-arg)"
)


def kpi_freshness(text: str, version: str | None) -> dict[str, Any]:
    """Stale-install-pin + unresolved-placeholder check, mirroring
    docs_scorecard.kpi_freshness but with the tighter install-context gate above
    so captured metric/exposition version strings are not miscounted as stale."""
    defects: list[str] = []
    cur = dsc._parse_version(version or "")
    stale_versions: set[str] = set()
    if cur is not None:
        cur_major, cur_minor, _ = cur
        for line in text.splitlines():
            if not _INSTALL_CTX_RE.search(line):
                continue
            for m in dsc._VERSION_RE.finditer(line):
                tail = line[m.end():m.end() + 2]
                if tail[:1] == "." and tail[1:2].isdigit():
                    continue  # dotted-quad IP octet, not a version
                major, minor = int(m.group(1)), int(m.group(2))
                if major == cur_major and (major, minor) < (cur_major, cur_minor):
                    stale_versions.add(m.group(0))
    for v in sorted(stale_versions):
        defects.append(f"stale install pin {v} (VERSION is {(version or '').strip()})")
    placeholders = sorted({m.group(0).lower() for m in dsc.PLACEHOLDER_RE.finditer(text)})
    for ph in placeholders:
        defects.append(f"unresolved placeholder: {ph!r}")
    score = 100 - 20 * len(stale_versions) - 12 * len(placeholders)
    return {
        "kpi": "freshness", "score": dsc._clamp(score),
        "detail": ("no stale pin or placeholder" if not defects
                   else f"{len(defects)} freshness defect(s)"),
        "defects": defects, "soft": [],
    }


# ---------------------------------------------------------------------------
# doc-type: deterministic from the repo-relative path
# ---------------------------------------------------------------------------

def doc_type(rel: str) -> str:
    low = rel.replace("\\", "/").lower()
    name = low.rsplit("/", 1)[-1]
    if low == "learning-path.md":
        return "curriculum"
    if name in META_NAMES:
        return "reference"
    if name == "faq.md" or low == "docs/faq.md":
        return "reference"
    if any(k in name for k in ("tutorial", "getting-started", "start-here",
                               "quickstart", "run-the-demos", "install")):
        return "tutorial"
    if ("/explainers/" in low or name in (
            "concepts-and-story.md", "context-is-not-memory.md",
            "memory-layers-explainer.md", "prefill-elimination-explained.md")):
        return "explainer"
    if "/fak/" in low:
        return "howto"
    return "explainer"


# ---------------------------------------------------------------------------
# The four pedagogy KPIs. Each returns the standard
#   {kpi, score(0-100), detail, defects:[HARD], soft:[ADVISORY]}
# and takes only already-read text (+ doc-type), so tests need no disk.
# ---------------------------------------------------------------------------

def kpi_orientation(text: str, dtype: str) -> dict[str, Any]:
    defects: list[str] = []
    soft: list[str] = []
    score = 100
    head = "\n".join(text.splitlines()[:FIRST_SCREEN_LINES]).lower()
    n_lines = text.count("\n") + 1
    if not any(s in head for s in _SIGNPOSTS):
        msg = "no orientation signpost (audience / prereq / TL;DR / 'you'll be able to')"
        if dtype in ORIENT_HARD:
            defects.append(msg)
            score -= 30
        else:
            soft.append(msg)
            score -= 10
    out_links = dsc._local_doc_links(text)
    if n_lines > 40 and not out_links:
        msg = "no outbound link to another doc (a navigational dead-end in the curriculum)"
        if dtype in ("tutorial", "howto", "curriculum"):
            defects.append(msg)
            score -= 25
        else:
            soft.append(msg)
            score -= 12
    low = text.lower()
    if n_lines > 60 and dtype in ("tutorial", "howto", "curriculum", "explainer"):
        if not any(s in low for s in _NEXT_CUES):
            soft.append("no explicit 'next step / where to go from here' signpost")
            score -= 8
    return {
        "kpi": "orientation", "score": dsc._clamp(score),
        "detail": ("oriented: signpost + onward nav" if not defects
                   else f"{len(defects)} orientation defect(s)"),
        "defects": defects, "soft": soft,
    }


def kpi_runnable(text: str, dtype: str) -> dict[str, Any]:
    defects: list[str] = []
    soft: list[str] = []
    score = 100
    blocks = text.count("```") // 2
    if blocks == 0:
        msg = "teaches no runnable command or code block (prose-only)"
        if dtype in RUNNABLE_HARD:
            defects.append(msg)
            score -= 40
        else:
            soft.append(msg)
            score -= 10
    elif dtype in ("tutorial", "howto") and blocks < 2:
        soft.append("only one code block — a command shown with no expected output to compare against")
        score -= 12
    return {
        "kpi": "runnable", "score": dsc._clamp(score),
        "detail": (f"{blocks} code block(s)" if blocks else "no code blocks"),
        "defects": defects, "soft": soft,
    }


def kpi_worked(text: str, dtype: str) -> dict[str, Any]:
    defects: list[str] = []
    soft: list[str] = []
    score = 100
    low = text.lower()
    has_worked = any(m in low for m in _WORKED_MARKERS)
    if not has_worked:
        msg = "no worked example / lab / checkpoint / expected-output (tells but never shows)"
        if dtype in WORKED_HARD:
            defects.append(msg)
            score -= 35
        else:
            soft.append(msg)
            score -= 12
    return {
        "kpi": "worked", "score": dsc._clamp(score),
        "detail": ("shows a worked example / lab / checkpoint" if has_worked
                   else "no worked example"),
        "defects": defects, "soft": soft,
    }


def kpi_honesty(text: str, authority: str) -> dict[str, Any]:
    # Reuse the hygiene scorer's evidence check for the HARD naive-led-headline
    # law and the soft unmirrored/unsourced nudges, then add the teaching-specific
    # "naked perf number" soft nudge.
    ev = dsc.kpi_evidence(text, authority)
    defects = list(ev["defects"])
    soft = list(ev["soft"])
    score = ev["score"]
    if dsc._MULT_RE.search(text) and not _HONESTY_RE.search(text) \
            and not _AUTHORITY_RE.search(text):
        soft.append("states a speed/multiplier number with no honesty marker "
                    "(SIMULATED / OPEN / vs-naive / vs-tuned) or authority link")
        score = max(0, score - 15)
    return {
        "kpi": "honesty", "score": dsc._clamp(score),
        "detail": ("no naive-led headline; perf claims marked or sourced" if not defects
                   else f"{len(defects)} honesty defect(s)"),
        "defects": defects, "soft": soft,
    }


# ---------------------------------------------------------------------------
# Per-doc fold
# ---------------------------------------------------------------------------

def score_doc(text: str, doc_rel: str, root: Path, *, version: str | None,
              authority: str) -> dict[str, Any]:
    dtype = doc_type(doc_rel)
    kpis = [
        kpi_orientation(text, dtype),
        kpi_runnable(text, dtype),
        kpi_worked(text, dtype),
        kpi_honesty(text, authority),
        dsc.kpi_structure(text, doc_rel),
        dsc.kpi_links(text, root, doc_rel),
        kpi_freshness(text, version),
    ]
    by_name = {k["kpi"]: k for k in kpis}
    score = sum(KPI_WEIGHTS[name] * by_name[name]["score"] for name in KPI_WEIGHTS)
    defects = [f"{k['kpi']}: {d}" for k in kpis for d in k["defects"]]
    soft = [f"{k['kpi']}: {s}" for k in kpis for s in k["soft"]]
    return {
        "path": doc_rel,
        "type": dtype,
        "score": round(score, 1),
        "grade": dsc.grade_letter(score),
        "kpis": {k["kpi"]: k["score"] for k in kpis},
        "kpi_detail": {k["kpi"]: k["detail"] for k in kpis},
        "defects": defects,
        "soft": soft,
        "n_defects": len(defects),
    }


def missing_doc_entry(doc_rel: str) -> dict[str, Any]:
    return {
        "path": doc_rel, "type": doc_type(doc_rel), "score": 0.0, "grade": "F",
        "kpis": {k: 0 for k in KPI_WEIGHTS}, "kpi_detail": {},
        "defects": [f"missing: learning doc {doc_rel} does not exist on disk"],
        "soft": [], "n_defects": 1,
    }


# ---------------------------------------------------------------------------
# Corpus enumeration + importance + coverage
# ---------------------------------------------------------------------------

def enumerate_learning_docs(root: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for rel in LEARNING_ROOTS:
        if (root / rel).exists() and rel not in seen:
            seen.add(rel)
            out.append(rel)
    for d in LEARNING_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.md")):
            rel = p.relative_to(root).as_posix()
            if rel not in seen:
                seen.add(rel)
                out.append(rel)
    return sorted(out)


def _hop_distances(root: Path, starts: list[str]) -> dict[str, int]:
    """BFS over the whole .md link graph from the front doors; hop count per doc."""
    seeds = [(root / s) for s in starts if (root / s).exists()]
    dist: dict[str, int] = {}
    q: deque[Path] = deque()
    for s in seeds:
        try:
            rel = s.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        if rel not in dist:
            dist[rel] = 0
            q.append(s)
    while q:
        cur = q.popleft()
        try:
            rel = cur.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        d = dist[rel]
        try:
            txt = cur.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in dsc._LINK_RE.finditer(txt):
            t = m.group("target").strip()
            if t.startswith(("http://", "https://", "mailto:", "#", "tel:")):
                continue
            pp = t.split("#", 1)[0].split("?", 1)[0]
            if not pp.endswith(".md"):
                continue
            nxt = (root / pp.lstrip("/")) if pp.startswith("/") else (cur.parent / pp)
            if not nxt.exists():
                continue
            try:
                nrel = nxt.resolve().relative_to(root.resolve()).as_posix()
            except (ValueError, OSError):
                continue
            if nrel not in dist:
                dist[nrel] = d + 1
                q.append(nxt)
    return dist


def _indegree(root: Path, corpus: list[str]) -> dict[str, int]:
    """How many OTHER learning docs link to each doc (centrality within the set)."""
    cset = set(corpus)
    indeg: dict[str, int] = {c: 0 for c in corpus}
    for c in corpus:
        txt = dsc._safe_read(root / c)
        if not txt:
            continue
        base = (root / c).parent
        seen: set[str] = set()
        for m in dsc._LINK_RE.finditer(txt):
            t = m.group("target").strip()
            if t.startswith(("http://", "https://", "mailto:", "#", "tel:")):
                continue
            pp = t.split("#", 1)[0].split("?", 1)[0]
            if not pp.endswith(".md"):
                continue
            resolved = (root / pp.lstrip("/")) if pp.startswith("/") else (base / pp)
            try:
                rel = resolved.resolve().relative_to(root.resolve()).as_posix()
            except (ValueError, OSError):
                continue
            if rel in cset and rel != c and rel not in seen:
                seen.add(rel)
                indeg[rel] = indeg.get(rel, 0) + 1
    return indeg


def importance_scores(root: Path, corpus: list[str]) -> dict[str, float]:
    """0–100 funnel-centrality per doc: 60% link-hop proximity to a front door,
    40% normalized in-degree from other learning docs. Deterministic."""
    dist = _hop_distances(root, FRONT_DOORS)
    indeg = _indegree(root, corpus)
    max_indeg = max(indeg.values()) if indeg else 0
    out: dict[str, float] = {}
    for c in corpus:
        hop = dist.get(c)
        proximity = 1.0 / (1 + hop) if hop is not None else 0.0
        indeg_norm = (indeg[c] / max_indeg) if max_indeg else 0.0
        out[c] = round(100 * (0.6 * proximity + 0.4 * indeg_norm), 1)
    return out


def coverage(root: Path, corpus: list[str]) -> dict[str, Any]:
    reachable = dsc.reachable_md(root, FRONT_DOORS)
    existing = [d for d in corpus if (root / d).exists() and d not in FRONT_DOOR_EXEMPT]
    orphans = sorted(d for d in existing if d not in reachable)
    reach_pct = round(100 * (len(existing) - len(orphans)) / max(1, len(existing)), 1)

    covered: list[str] = []
    missing_topics: list[str] = []
    for topic, candidates in EXPECTED_TOPICS:
        ok = any((root / c).exists() and dsc._nonstub(root / c) for c in candidates)
        (covered if ok else missing_topics).append(topic)
    topic_pct = round(100 * len(covered) / max(1, len(EXPECTED_TOPICS)), 1)
    overall = round((reach_pct + topic_pct) / 2, 1)
    cov_defects = [f"orphan lesson (unreachable from any front door): {o}" for o in orphans] \
        + [f"uncovered learning topic: {t}" for t in missing_topics]
    return {
        "reachability_pct": reach_pct, "topic_pct": topic_pct, "overall_pct": overall,
        "orphans": orphans, "missing_topics": missing_topics, "covered_topics": covered,
        "defects": cov_defects,
    }


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, docs: list[dict[str, Any]],
                  cov: dict[str, Any], importance: dict[str, float]) -> dict[str, Any]:
    n = len(docs)
    scores = [d["score"] for d in docs]
    doc_defects = sum(d["n_defects"] for d in docs)
    total_defects = doc_defects + len(cov.get("defects", []))
    mean_score = round(sum(scores) / max(1, n), 1)
    grades = {g: 0 for g in "ABCDF"}
    for d in docs:
        grades[d["grade"]] = grades.get(d["grade"], 0) + 1
    for d in docs:
        d["importance"] = importance.get(d["path"], 0.0)
    worst = sorted(docs, key=lambda d: (d["score"], -d["n_defects"]))[:10]
    # priorities: most-read AND most-broken first. importance×(debt+soft-pressure).
    def prio(d: dict[str, Any]) -> float:
        pressure = d["n_defects"] + 0.25 * len(d["soft"]) + (100 - d["score"]) / 100.0
        return round(d["importance"] * pressure / 100.0, 3)
    prioritized = sorted(docs, key=lambda d: (-prio(d), -d["importance"]))
    priorities = [{"path": d["path"], "type": d["type"], "importance": d["importance"],
                   "score": d["score"], "grade": d["grade"], "n_defects": d["n_defects"],
                   "n_soft": len(d["soft"]), "priority": prio(d)}
                  for d in prioritized[:10]]

    corpus = {
        "n_docs": n, "mean_score": mean_score,
        "median_score": round(sorted(scores)[n // 2], 1) if n else 0.0,
        "min_score": round(min(scores), 1) if scores else 0.0,
        "max_score": round(max(scores), 1) if scores else 0.0,
        "grade_distribution": grades,
        "learning_debt": total_defects,
        "learning_debt_in_docs": doc_defects,
        "learning_debt_in_coverage": len(cov.get("defects", [])),
        "soft_signals": sum(len(d["soft"]) for d in docs),
        "worst": [{"path": d["path"], "score": d["score"], "grade": d["grade"],
                   "n_defects": d["n_defects"]} for d in worst],
        "priorities": priorities,
    }

    if total_defects == 0:
        ok, verdict, finding = True, "OK", "learning_clean"
        reason = (f"learning set clean: {n} docs, mean {mean_score}/100, "
                  f"coverage {cov['overall_pct']}%, zero learning-debt "
                  f"({corpus['soft_signals']} soft signal(s) remain — judgment calls)")
        next_action = ("no required edit; raise the most-important docs (see corpus.priorities) "
                       "by retiring soft signals, then re-run")
    else:
        ok, verdict, finding = False, "ACTION", "learning_debt"
        reason = (f"{total_defects} unit(s) of learning-debt across {n} learning docs "
                  f"({doc_defects} in-doc + {len(cov.get('defects', []))} coverage); "
                  f"mean {mean_score}/100, coverage {cov['overall_pct']}%")
        next_action = ("retire learning-debt by importance×debt (see corpus.priorities): add the "
                       "missing orientation signpost, runnable command, worked example, or honesty "
                       "marker; re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "coverage": cov, "docs": docs,
    }


def collect(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    corpus = enumerate_learning_docs(root)
    version = dsc._safe_read(root / VERSION_REL) or None
    authority = dsc._safe_read(root / AUTHORITY_REL)
    docs: list[dict[str, Any]] = []
    for rel in corpus:
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
    cov = coverage(root, corpus)
    importance = importance_scores(root, corpus)
    return build_payload(workspace=str(root), docs=docs, cov=cov, importance=importance)


# ---------------------------------------------------------------------------
# Renderers + CLI
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    cov = payload.get("coverage") or {}
    lines = [
        f"learning-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"corpus: {c.get('n_docs', 0)} docs · mean {c.get('mean_score', 0)}/100 "
         f"· min {c.get('min_score', 0)} · LEARNING-DEBT {c.get('learning_debt', 0)} "
         f"· soft {c.get('soft_signals', 0)}"),
        ("grades: " + " ".join(f"{g}:{c.get('grade_distribution', {}).get(g, 0)}"
                                for g in "ABCDF")),
        (f"coverage: {cov.get('overall_pct', 0)}% "
         f"(reach {cov.get('reachability_pct', 0)}% · topics {cov.get('topic_pct', 0)}%)"),
        f"next: {payload.get('next_action')}",
        "",
        "priorities (most important × most broken, fix first):",
        f"  {'prio':>5} {'imp':>5} {'score':>5} {'def':>3} {'soft':>4}  path",
    ]
    for p in c.get("priorities", []):
        lines.append(f"  {p['priority']:>5} {p['importance']:>5} {p['score']:>5} "
                     f"{p['n_defects']:>3} {p['n_soft']:>4}  {p['path']}")
    lines += ["", "per-doc (worst score first):",
              f"  {'score':>5} {'gr':>2} {'def':>3}  {'type':<11} path"]
    for d in sorted(payload.get("docs", []), key=lambda x: (x["score"], -x["n_defects"])):
        lines.append(f"  {d['score']:>5} {d['grade']:>2} {d['n_defects']:>3}  "
                     f"{d.get('type', ''):<11} {d['path']}")
    lines += ["", "learning-debt detail (top defectful docs):"]
    worst = sorted(payload.get("docs", []), key=lambda x: -x["n_defects"])[:10]
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
    c = payload.get("corpus") or {}
    cov = payload.get("coverage") or {}
    gd = c.get("grade_distribution", {})
    out: list[str] = ["# Learning-docs scorecard", ""]
    if stamp:
        out += [f"<!-- learning-scorecard: {stamp} · process: tools/learning_scorecard.py -->", ""]
    out += [
        "> Regenerate: `python tools/learning_scorecard.py --markdown --stamp DATE > docs/LEARNING-SCORECARD.md`",
        "",
        "This is the measuring stick for the learning-2× program: does the *teaching* "
        "set actually teach? Every number below is re-derived from disk by "
        "`tools/learning_scorecard.py` — no hand-entry. The headline metric is "
        "**learning-debt**: the count of concrete teaching defects (a how-to with no "
        "runnable command, a tutorial with no worked output, a lesson with no orientation "
        "signpost, an orphan lesson, an uncovered learning topic). Driving learning-debt "
        "toward zero — then raising the most-important docs — is what makes \"the docs "
        "teach 2× better\" provable. Pedagogy counterpart of `docs_scorecard` (hygiene) "
        "and `doc_appeal_scorecard` (voice).",
        "",
        "## Corpus",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Learning docs scored | {c.get('n_docs', 0)} |",
        f"| **Learning-debt (total HARD defects)** | **{c.get('learning_debt', 0)}** "
        f"({c.get('learning_debt_in_docs', 0)} in-doc + {c.get('learning_debt_in_coverage', 0)} coverage) |",
        f"| Soft signals (judgment calls) | {c.get('soft_signals', 0)} |",
        f"| Mean score | {c.get('mean_score', 0)}/100 |",
        f"| Median / min / max | {c.get('median_score', 0)} / {c.get('min_score', 0)} / {c.get('max_score', 0)} |",
        f"| Grade distribution | A:{gd.get('A',0)} B:{gd.get('B',0)} C:{gd.get('C',0)} D:{gd.get('D',0)} F:{gd.get('F',0)} |",
        f"| Coverage (overall) | {cov.get('overall_pct', 0)}% |",
        f"| — reachable from a front door | {cov.get('reachability_pct', 0)}% |",
        f"| — expected learning topics covered | {cov.get('topic_pct', 0)}% |",
        "",
        "## Priorities — fix the most-important, most-broken first",
        "",
        "Ranked by importance (funnel-centrality: link-hop proximity to a front door + "
        "in-degree from other learning docs) × teaching pressure (defects + soft signals + "
        "score gap). These are the 2× targets.",
        "",
        "| Priority | Importance | Score | Grade | Debt | Soft | Type | Doc |",
        "|---:|---:|---:|:--:|:--:|:--:|:--|---|",
    ]
    for p in c.get("priorities", []):
        out.append(f"| {p['priority']} | {p['importance']} | {p['score']} | {p['grade']} | "
                   f"{p['n_defects']} | {p['n_soft']} | {p['type']} | `{p['path']}` |")
    out += ["", "## Per-doc scores", "",
            "Seven KPIs, each 0–100 — pedagogy (orientation · runnable · worked · honesty) "
            "+ mechanical (structure · links · freshness) — weighted into a score and an "
            "A–F grade. `def` = units of learning-debt in that doc.", "",
            "| Score | Grade | Debt | orient | run | work | hon | struct | link | fresh | Imp | Doc |",
            "|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---:|---|"]
    for d in sorted(payload.get("docs", []), key=lambda x: (x["score"], -x["n_defects"])):
        k = d.get("kpis", {})
        out.append(
            f"| {d['score']} | {d['grade']} | {d['n_defects']} | "
            f"{k.get('orientation','-')} | {k.get('runnable','-')} | {k.get('worked','-')} | "
            f"{k.get('honesty','-')} | {k.get('structure','-')} | {k.get('links','-')} | "
            f"{k.get('freshness','-')} | {d.get('importance','-')} | `{d['path']}` |")
    out += ["", "## Learning-debt work-list", ""]
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
        out += ["No learning-debt: every learning doc clears the HARD bar. 🎉", "",
                "Remaining work is soft-signal polish on the priority docs above.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Learning-docs scorecard (read-only, pedagogy-aware).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the LEARNING-SCORECARD.md body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else dsc.repo_root()
    payload = collect(workspace)

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
