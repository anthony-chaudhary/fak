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
import subprocess
import sys
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docs_scorecard as dsc  # noqa: E402  reuse the hygiene scorer's pure helpers

SCHEMA = "fleet-learning-scorecard/1"
DETECT_LATENCY_SCHEMA = "fleet-learning-scorecard.detect-latency-bench/1"

VERSION_REL = "VERSION"
AUTHORITY_REL = "fak/BENCHMARK-AUTHORITY.md"
GENERATED_SNAPSHOT_REL = "docs/LEARNING-SCORECARD.md"
STAMP_CADENCE_DAYS = 1
MANUAL_DETECT_BASELINE_DAYS = 5.0
MANUAL_DETECT_BASELINE_DEFECTS = 34
MANUAL_DETECT_BASELINE_COMMIT = "cf1deba4"

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

# Dynamic coverage catches a different class of debt: a shipped surface that
# landed after the generated scorecard stamp but never entered the course.
_INTERNAL_PLUMBING_RE = re.compile(
    r"(?:cmdutil|demoutil|pathutil|maputil|mathx|intlist|numfmt|test|fixture)$"
)
_CMD_DIR_PLUMBING_RE = re.compile(r"(?:demo|bench)$")
_HIDDEN_FAK_VERBS = {
    "-h", "--help", "help", "-v", "--version",
    "guard-precompact", "guard-stophook",
}

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
# Generated scorecard stamp freshness. The generated docs/LEARNING-SCORECARD.md
# is itself a teaching artifact: if the corpus moved after its stamp and the
# stamp missed the loop cadence, that stale stamp is one unit of learning debt.
# ---------------------------------------------------------------------------

_SNAPSHOT_STAMP_RE = re.compile(
    r"learning-scorecard:\s*(?P<stamp>\d{4}-\d{2}-\d{2})"
)


def snapshot_stamp(text: str) -> str:
    m = _SNAPSHOT_STAMP_RE.search(text)
    return m.group("stamp").strip() if m else ""


def _parse_date(raw: str) -> date | None:
    try:
        return date.fromisoformat(raw.strip()[:10])
    except (TypeError, ValueError):
        return None


def _parse_git_date(raw: str) -> date | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return _parse_date(raw)


def _git_line(args: list[str], root: Path) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _git_head_date(root: Path) -> date | None:
    return _parse_git_date(_git_line(["show", "-s", "--format=%cI", "HEAD"], root))


def _latest_corpus_change_date(root: Path, corpus: list[str]) -> date | None:
    existing = [p for p in corpus if (root / p).exists()]
    if not existing:
        return None
    out = _git_line(["log", "-1", "--format=%cI", "--", *existing], root)
    return _parse_git_date(out.splitlines()[0] if out else "")


def _dirty_corpus_paths(root: Path, corpus: list[str]) -> list[str]:
    existing = [p for p in corpus if (root / p).exists()]
    if not existing:
        return []
    out = _git_line(["status", "--porcelain", "--", *existing], root)
    dirty: list[str] = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        dirty.append(line[3:].strip().replace("\\", "/"))
    return sorted(dirty)


def stamp_freshness_from_dates(*, stamp: str, reference_date: date,
                               last_corpus_change_date: date | None,
                               dirty_paths: list[str] | None = None,
                               cadence_days: int = STAMP_CADENCE_DAYS,
                               doc: str = GENERATED_SNAPSHOT_REL) -> dict[str, Any]:
    cadence = max(1, int(cadence_days))
    dirty = sorted({p.replace("\\", "/") for p in (dirty_paths or [])})
    stamp_date = _parse_date(stamp)
    if stamp_date is None:
        return {
            "doc": doc,
            "stamp": stamp or None,
            "stamp_age_days": None,
            "cadence_days": cadence,
            "last_corpus_change": (last_corpus_change_date.isoformat()
                                   if last_corpus_change_date else None),
            "dirty_corpus_count": len(dirty),
            "dirty_corpus_paths": dirty[:20],
            "corpus_changed_since_stamp": bool(dirty),
            "stale_stamp": False,
            "flag": "",
            "reason": f"{doc} has no parseable learning-scorecard stamp",
        }

    age = max(0, (reference_date - stamp_date).days)
    committed_drift = bool(last_corpus_change_date and last_corpus_change_date > stamp_date)
    corpus_changed = committed_drift or bool(dirty)
    stale = age >= cadence and corpus_changed
    if stale:
        reason = (f"stale-stamp: {doc} stamp {stamp_date.isoformat()} is {age}d old "
                  f"(cadence {cadence}d) and the learning corpus changed since")
    elif corpus_changed:
        reason = (f"stamp {stamp_date.isoformat()} is {age}d old; corpus changed, "
                  f"but still inside the {cadence}d cadence")
    else:
        reason = f"stamp {stamp_date.isoformat()} is {age}d old; no corpus drift since stamp"
    return {
        "doc": doc,
        "stamp": stamp_date.isoformat(),
        "stamp_age_days": age,
        "cadence_days": cadence,
        "last_corpus_change": (last_corpus_change_date.isoformat()
                               if last_corpus_change_date else None),
        "dirty_corpus_count": len(dirty),
        "dirty_corpus_paths": dirty[:20],
        "corpus_changed_since_stamp": corpus_changed,
        "stale_stamp": stale,
        "flag": "stale-stamp" if stale else "",
        "reason": reason,
    }


def stamp_freshness(root: Path, corpus: list[str],
                    cadence_days: int = STAMP_CADENCE_DAYS) -> dict[str, Any]:
    doc_path = root / GENERATED_SNAPSHOT_REL
    try:
        doc_text = doc_path.read_text(encoding="utf-8")
    except OSError:
        doc_text = ""
    reference = _git_head_date(root) or datetime.now(timezone.utc).date()
    return stamp_freshness_from_dates(
        stamp=snapshot_stamp(doc_text),
        reference_date=reference,
        last_corpus_change_date=_latest_corpus_change_date(root, corpus),
        dirty_paths=_dirty_corpus_paths(root, corpus),
        cadence_days=cadence_days,
    )


# ---------------------------------------------------------------------------
# Shipped-surface coverage. Static EXPECTED_TOPICS catches durable curriculum
# holes; this fold catches new surfaces that shipped after the generated
# scorecard stamp and have not entered the course.
# ---------------------------------------------------------------------------

def _rel(path: str) -> str:
    return path.strip().replace("\\", "/").lstrip("./")


def _git_added_paths_since_stamp(root: Path, stamp: str) -> list[str]:
    stamp_date = _parse_date(stamp)
    if stamp_date is None:
        return []
    # The generated stamp is day-granular, not a timestamp. Same-day additions
    # are still inside that stamp's freshness window, so only count later dates.
    since = (stamp_date + timedelta(days=1)).isoformat()
    out = _git_line([
        "log", "--diff-filter=A", "--name-only", "--format=",
        f"--since={since}T00:00:00",
        "--",
        "internal", "cmd/fak", "cmd",
    ], root)
    return sorted({_rel(line) for line in out.splitlines() if line.strip()})


def _git_blob(root: Path, ref: str, rel: str) -> str:
    return _git_line(["show", f"{ref}:{_rel(rel)}"], root)


def _git_first_added_date(root: Path, rel: str) -> date | None:
    out = _git_line([
        "log", "--diff-filter=A", "--reverse", "--format=%cI",
        "--", _rel(rel),
    ], root)
    return _parse_git_date(out.splitlines()[0] if out else "")


def _added_on_or_after_stamp(root: Path, rel: str, stamp_date: date | None,
                             *, require_git_date: bool) -> bool:
    if stamp_date is None:
        return True
    first = _git_first_added_date(root, rel)
    if first is None:
        return not require_git_date
    return first > stamp_date


def _internal_lane_allowlist(root: Path) -> set[str] | None:
    text = dsc._safe_read(root / "dos.toml")
    if not text:
        return None
    names = {
        m.group(1)
        for m in re.finditer(r"^([a-z][a-z0-9]*)\s*=\s*\[\s*\"internal/\1/\*\*\"\s*\]",
                             text, re.MULTILINE)
    }
    return names or None


def _has_nontest_go(root: Path, rel_dir: str) -> bool:
    base = root / rel_dir
    if not base.is_dir():
        return False
    return any(p.suffix == ".go" and not p.name.endswith("_test.go")
               for p in base.glob("*.go"))


def _fak_verb_handlers(main_text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    lines = main_text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("case ") or ":" not in stripped:
            continue
        labels = [s for s in re.findall(r'"([^"]+)"', stripped.split(":", 1)[0])
                  if s not in _HIDDEN_FAK_VERBS and not s.startswith("-")]
        if not labels:
            continue
        handler = ""
        for look in lines[i + 1:i + 8]:
            m = re.search(r"\b(cmd[A-Z][A-Za-z0-9_]*)\s*\(", look)
            if m:
                handler = m.group(1)
                break
        if not handler:
            continue
        for label in labels:
            out[label] = handler
    return out


def _cmd_fak_handler_files(root: Path) -> dict[str, str]:
    base = root / "cmd" / "fak"
    out: dict[str, str] = {}
    if not base.is_dir():
        return out
    for path in sorted(base.glob("*.go")):
        if path.name.endswith("_test.go"):
            continue
        rel = path.relative_to(root).as_posix()
        text = dsc._safe_read(path)
        for m in re.finditer(r"^func\s+(cmd[A-Z][A-Za-z0-9_]*)\s*\(", text,
                             re.MULTILINE):
            out[m.group(1)] = rel
    return out


def _git_added_fak_verbs_since_stamp(root: Path, stamp: str) -> list[str]:
    stamp_date = _parse_date(stamp)
    if stamp_date is None:
        return []
    baseline = _git_line([
        "rev-list", "-1", f"--before={stamp_date.isoformat()}T23:59:59",
        "HEAD", "--", "cmd/fak/main.go",
    ], root)
    if not baseline:
        return []
    before = set(_fak_verb_handlers(_git_blob(root, baseline, "cmd/fak/main.go")))
    current = set(_fak_verb_handlers(dsc._safe_read(root / "cmd" / "fak" / "main.go")))
    return sorted(current - before)


def _surface(kind: str, name: str, ref: str, source_paths: list[str]) -> dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "ref": ref,
        "source_paths": sorted({_rel(p) for p in source_paths}),
    }


def derive_teachable_surfaces(root: Path, added_paths: list[str], *,
                              added_fak_verbs: list[str] | None = None,
                              stamp_date: date | None = None,
                              require_git_dates: bool = False) -> list[dict[str, Any]]:
    added = sorted({_rel(p) for p in added_paths})
    added_verbs = {v.strip() for v in (added_fak_verbs or []) if v.strip()}
    by_key: dict[tuple[str, str], dict[str, Any]] = {}

    allow_internal = _internal_lane_allowlist(root)
    internal_paths: dict[str, list[str]] = {}
    cmd_dirs: dict[str, list[str]] = {}
    for rel in added:
        m = re.match(r"^internal/([a-z][a-z0-9]*)/[^/]+\.go$", rel)
        if m and not rel.endswith("_test.go"):
            pkg = m.group(1)
            if (allow_internal is None or pkg in allow_internal) \
                    and not _INTERNAL_PLUMBING_RE.search(pkg):
                internal_paths.setdefault(pkg, []).append(rel)
            continue
        m = re.match(r"^cmd/([^/]+)/[^/]+\.go$", rel)
        if m and not rel.endswith("_test.go"):
            name = m.group(1)
            if name != "fak" and not _CMD_DIR_PLUMBING_RE.search(name):
                cmd_dirs.setdefault(name, []).append(rel)

    for pkg, paths in internal_paths.items():
        ref = f"internal/{pkg}"
        if _has_nontest_go(root, ref) and _added_on_or_after_stamp(
                root, ref, stamp_date, require_git_date=require_git_dates):
            by_key[("internal", pkg)] = _surface("internal", pkg, ref, paths)

    main_text = dsc._safe_read(root / "cmd" / "fak" / "main.go")
    verb_to_handler = _fak_verb_handlers(main_text)
    handler_to_file = _cmd_fak_handler_files(root)
    added_set = set(added)
    for verb, handler in verb_to_handler.items():
        rel = handler_to_file.get(handler, "")
        if not rel:
            continue
        handler_added = rel in added_set and _added_on_or_after_stamp(
            root, rel, stamp_date, require_git_date=require_git_dates,
        )
        verb_added = verb in added_verbs
        if handler_added or verb_added:
            by_key[("cmd/fak", verb)] = _surface("cmd/fak", verb, f"fak {verb}", [rel])

    for name, paths in cmd_dirs.items():
        ref = f"cmd/{name}"
        if _has_nontest_go(root, ref) and _added_on_or_after_stamp(
                root, ref, stamp_date, require_git_date=require_git_dates):
            by_key[("cmd", name)] = _surface("cmd", name, ref, paths)

    return [by_key[k] for k in sorted(by_key)]


def surface_taught_by_course(surface: dict[str, Any], course_text: str) -> bool:
    name = re.escape(str(surface.get("name", "")))
    kind = surface.get("kind")
    patterns: list[str]
    if kind == "internal":
        patterns = [
            rf"(?<![\w/-])internal/{name}(?:/|\b)",
            rf"`{name}`",
            rf"(?<![\w-]){name}(?![\w-])",
        ]
    elif kind == "cmd/fak":
        patterns = [
            rf"(?<![\w-])fak\s+{name}(?:\s|`|$)",
            rf"cmd/fak\s+{name}(?:\s|`|$)",
            rf"go\s+run\s+\./cmd/fak\s+{name}(?:\s|`|$)",
        ]
    else:
        patterns = [
            rf"(?<![\w/-])cmd/{name}(?:/|\b)",
            rf"go\s+run\s+\./cmd/{name}(?:\s|`|$)",
            rf"`{name}`",
        ]
    return any(re.search(p, course_text, re.IGNORECASE) for p in patterns)


def shipped_surface_coverage(root: Path, *, stamp: str,
                             added_paths: list[str] | None = None,
                             added_fak_verbs: list[str] | None = None,
                             course_text: str | None = None,
                             require_git_dates: bool = True) -> dict[str, Any]:
    stamp_date = _parse_date(stamp)
    added = (_git_added_paths_since_stamp(root, stamp)
             if added_paths is None else sorted({_rel(p) for p in added_paths}))
    verbs = (_git_added_fak_verbs_since_stamp(root, stamp)
             if added_fak_verbs is None else sorted({v for v in added_fak_verbs if v}))
    surfaces = derive_teachable_surfaces(
        root, added, added_fak_verbs=verbs,
        stamp_date=stamp_date, require_git_dates=require_git_dates,
    )
    course = dsc._safe_read(root / "LEARNING-PATH.md") if course_text is None else course_text
    rows: list[dict[str, Any]] = []
    for s in surfaces:
        row = dict(s)
        row["taught"] = surface_taught_by_course(s, course)
        rows.append(row)
    covered = [r for r in rows if r["taught"]]
    missing = [r for r in rows if not r["taught"]]
    return {
        "stamp": stamp or None,
        "added_paths": added,
        "added_fak_verbs": verbs,
        "surfaces": rows,
        "covered": covered,
        "missing": missing,
        "defects": [
            f"shipped-but-untaught surface: {r['ref']} "
            f"(added since learning-scorecard stamp {stamp or 'unknown'})"
            for r in missing
        ],
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


def coverage(root: Path, corpus: list[str], *, added_paths: list[str] | None = None,
             added_fak_verbs: list[str] | None = None,
             course_text: str | None = None,
             require_git_dates: bool = True) -> dict[str, Any]:
    reachable = dsc.reachable_md(root, FRONT_DOORS)
    existing = [d for d in corpus if (root / d).exists() and d not in FRONT_DOOR_EXEMPT]
    orphans = sorted(d for d in existing if d not in reachable)
    reach_pct = round(100 * (len(existing) - len(orphans)) / max(1, len(existing)), 1)

    covered: list[str] = []
    missing_topics: list[str] = []
    for topic, candidates in EXPECTED_TOPICS:
        ok = any((root / c).exists() and dsc._nonstub(root / c) for c in candidates)
        (covered if ok else missing_topics).append(topic)
    scorecard_text = dsc._safe_read(root / GENERATED_SNAPSHOT_REL)
    surface_cov = shipped_surface_coverage(
        root, stamp=snapshot_stamp(scorecard_text), added_paths=added_paths,
        added_fak_verbs=added_fak_verbs,
        course_text=course_text, require_git_dates=require_git_dates,
    )
    covered_surfaces = [s["ref"] for s in surface_cov["covered"]]
    missing_surfaces = [s["ref"] for s in surface_cov["missing"]]
    topic_total = len(EXPECTED_TOPICS) + len(surface_cov["surfaces"])
    topic_covered = len(covered) + len(covered_surfaces)
    topic_pct = round(100 * topic_covered / max(1, topic_total), 1)
    overall = round((reach_pct + topic_pct) / 2, 1)
    cov_defects = [f"orphan lesson (unreachable from any front door): {o}" for o in orphans] \
        + [f"uncovered learning topic: {t}" for t in missing_topics] \
        + list(surface_cov["defects"])
    return {
        "reachability_pct": reach_pct, "topic_pct": topic_pct, "overall_pct": overall,
        "orphans": orphans, "missing_topics": missing_topics, "covered_topics": covered,
        "shipped_surface_coverage": surface_cov,
        "missing_shipped_surfaces": missing_surfaces,
        "covered_shipped_surfaces": covered_surfaces,
        "defects": cov_defects,
    }


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, docs: list[dict[str, Any]],
                  cov: dict[str, Any], importance: dict[str, float],
                  stamp: dict[str, Any] | None = None) -> dict[str, Any]:
    n = len(docs)
    scores = [d["score"] for d in docs]
    doc_defects = sum(d["n_defects"] for d in docs)
    stamp = stamp or {}
    stamp_defects = 1 if stamp.get("stale_stamp") else 0
    total_defects = doc_defects + len(cov.get("defects", [])) + stamp_defects
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
        "learning_debt_in_stamp": stamp_defects,
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
                  f"({doc_defects} in-doc + {len(cov.get('defects', []))} coverage "
                  f"+ {stamp_defects} stamp); "
                  f"mean {mean_score}/100, coverage {cov['overall_pct']}%")
        next_action = ("retire learning-debt by importance×debt (see corpus.priorities): add the "
                       "missing orientation signpost, runnable command, worked example, or honesty "
                       "marker; re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "coverage": cov, "stamp_freshness": stamp, "docs": docs,
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
    stamp = stamp_freshness(root, corpus)
    return build_payload(workspace=str(root), docs=docs, cov=cov, importance=importance,
                         stamp=stamp)


# ---------------------------------------------------------------------------
# Net-true detect-latency bench
# ---------------------------------------------------------------------------

def detect_latency_bench_from_cost(cost_seconds: float,
                                   cadence_days: int = STAMP_CADENCE_DAYS) -> dict[str, Any]:
    """Fixture-backed net-true bench for "catches drift sooner than a human".

    The fixture encodes the issue's manual baseline: a human process left a
    5-day, 34-defect gap (cf1deba4). The loop detects a stale generated
    scorecard on the first daily run after a corpus change, then charges the
    measured scorecard runtime as cost.
    """
    stamp_date = date(2026, 6, 1)
    corpus_change = date(2026, 6, 3)
    loop_run = date(2026, 6, 4)
    witness = stamp_freshness_from_dates(
        stamp=stamp_date.isoformat(),
        reference_date=loop_run,
        last_corpus_change_date=corpus_change,
        cadence_days=cadence_days,
    )
    loop_detect_days = float((loop_run - corpus_change).days)
    cost_days = max(0.0, float(cost_seconds)) / 86400.0
    net_loop_days = loop_detect_days + cost_days
    days_saved = MANUAL_DETECT_BASELINE_DAYS - net_loop_days
    verdict = "net-true" if witness.get("stale_stamp") and days_saved > 0 else "not-yet"
    return {
        "schema": DETECT_LATENCY_SCHEMA,
        "verdict": verdict,
        "ok": verdict == "net-true",
        "fixture": {
            "manual_baseline_commit": MANUAL_DETECT_BASELINE_COMMIT,
            "manual_detect_latency_days": MANUAL_DETECT_BASELINE_DAYS,
            "manual_gap_defects": MANUAL_DETECT_BASELINE_DEFECTS,
            "stamp": stamp_date.isoformat(),
            "corpus_changed": corpus_change.isoformat(),
            "loop_run": loop_run.isoformat(),
            "cadence_days": max(1, int(cadence_days)),
        },
        "stamp_witness": witness,
        "loop": {
            "detect_latency_days": loop_detect_days,
            "scorecard_cost_seconds": round(float(cost_seconds), 6),
            "scorecard_cost_days": cost_days,
            "net_detect_latency_days": net_loop_days,
        },
        "manual": {
            "detect_latency_days": MANUAL_DETECT_BASELINE_DAYS,
            "defects": MANUAL_DETECT_BASELINE_DEFECTS,
            "source_commit": MANUAL_DETECT_BASELINE_COMMIT,
        },
        "net": {
            "days_saved": days_saved,
            "manual_minus_loop_days": MANUAL_DETECT_BASELINE_DAYS - loop_detect_days,
            "cost_seconds_charged": round(float(cost_seconds), 6),
            "claim": ("loop catches drift sooner than the manual baseline"
                      if days_saved > 0 else "loop has not beaten the manual baseline net of cost"),
        },
    }


def detect_latency_bench(root: Path) -> dict[str, Any]:
    start = time.perf_counter()
    collect(root)
    cost_seconds = time.perf_counter() - start
    return detect_latency_bench_from_cost(cost_seconds)


def render_detect_latency_bench(payload: dict[str, Any]) -> str:
    fx = payload.get("fixture") or {}
    loop = payload.get("loop") or {}
    manual = payload.get("manual") or {}
    net = payload.get("net") or {}
    witness = payload.get("stamp_witness") or {}
    return "\n".join([
        f"learning-scorecard detect-latency bench: {payload.get('verdict')}",
        (f"  fixture: stamp {fx.get('stamp')} · corpus changed {fx.get('corpus_changed')} "
         f"· loop run {fx.get('loop_run')} · cadence {fx.get('cadence_days')}d"),
        (f"  stamp witness: stamp_age_days={witness.get('stamp_age_days')} "
         f"flag={witness.get('flag') or 'none'}"),
        (f"  loop: detects in {loop.get('detect_latency_days')}d + "
         f"{loop.get('scorecard_cost_seconds')}s scorecard cost "
         f"= {loop.get('net_detect_latency_days'):.6f}d net"),
        (f"  manual: detects in {manual.get('detect_latency_days')}d "
         f"with {manual.get('defects')} defects ({manual.get('source_commit')})"),
        (f"  net: saves {net.get('days_saved'):.6f}d after cost; "
         f"{net.get('claim')}"),
    ])


# ---------------------------------------------------------------------------
# Renderers + CLI
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    cov = payload.get("coverage") or {}
    stamp = payload.get("stamp_freshness") or {}
    stamp_flag = f" · {stamp.get('flag')}" if stamp.get("flag") else ""
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
        (f"stamp: age {stamp.get('stamp_age_days')}d / cadence "
         f"{stamp.get('cadence_days', STAMP_CADENCE_DAYS)}d"
         f"{stamp_flag} · {stamp.get('reason', 'no stamp witness')}"),
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
    if stamp.get("stale_stamp"):
        lines.append("  stamp:")
        lines.append(f"      - {stamp.get('reason')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    cov = payload.get("coverage") or {}
    freshness = payload.get("stamp_freshness") or {}
    gd = c.get("grade_distribution", {})
    out: list[str] = [
        "---",
        'title: "fak learning-docs scorecard — the learning-debt measuring stick"',
        'description: "fak\'s deterministic learning-docs scorecard: does the teaching set '
        "actually teach? Pedagogy KPIs (a how-to with no runnable command, a tutorial with no "
        "worked output, an orphan lesson, an uncovered learning topic) folded into a composite "
        'score and the headline learning-debt metric, re-derived from disk."',
        "---",
        "",
        "# Learning-docs scorecard",
        "",
    ]
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
        f"({c.get('learning_debt_in_docs', 0)} in-doc + "
        f"{c.get('learning_debt_in_coverage', 0)} coverage + "
        f"{c.get('learning_debt_in_stamp', 0)} stamp) |",
        f"| Soft signals (judgment calls) | {c.get('soft_signals', 0)} |",
        f"| Mean score | {c.get('mean_score', 0)}/100 |",
        f"| Median / min / max | {c.get('median_score', 0)} / {c.get('min_score', 0)} / {c.get('max_score', 0)} |",
        f"| Grade distribution | A:{gd.get('A',0)} B:{gd.get('B',0)} C:{gd.get('C',0)} D:{gd.get('D',0)} F:{gd.get('F',0)} |",
        f"| Coverage (overall) | {cov.get('overall_pct', 0)}% |",
        f"| — reachable from a front door | {cov.get('reachability_pct', 0)}% |",
        f"| — expected learning topics covered | {cov.get('topic_pct', 0)}% |",
        f"| Stamp freshness | age {freshness.get('stamp_age_days')}d / "
        f"cadence {freshness.get('cadence_days', STAMP_CADENCE_DAYS)}d"
        f"{' — **stale-stamp**' if freshness.get('stale_stamp') else ''} |",
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
    if freshness.get("stale_stamp"):
        any_defect = True
        out.append("### stamp freshness")
        out.append(f"- {freshness.get('reason')}")
        out.append("")
    if not any_defect:
        out += ["No learning-debt: every learning doc clears the HARD bar. 🎉", "",
                "Remaining work is soft-signal polish on the priority docs above.", ""]
    return "\n".join(out)


def check_gate(payload: dict[str, Any], baseline: int = 0) -> tuple[int, str]:
    """The CI ratchet decision over a learning payload (pure: exit code + message).

    learning-debt is an integer the corpus drives toward zero. The default exit
    code is green only at ZERO debt; this gate is GREEN while the corpus holds
    at-or-below the pinned ``baseline`` ceiling (0 by default) and RED only when
    debt *regresses* above it — the same shape as ``scorecard_control_pane.py
    --check``. It is the **keep-bit witness** every #1278 child reads: without it
    the tool had ``--json`` / ``--markdown`` / ``--stamp`` and no gate, which is
    why the corpus drifted 34 defects deep before a human noticed (cf1deba4).

      exit 0  held       — learning-debt at-or-below the pinned baseline
      exit 1  regressed  — learning-debt rose above the baseline ceiling
    """
    debt = int(payload.get("corpus", {}).get("learning_debt", 0))
    if debt > baseline:
        return 1, (f"LEARNING-DEBT RATCHET RED: {debt} unit(s) of learning-debt "
                   f"exceed the pinned ceiling of {baseline}; retire worst-first "
                   f"(see corpus.priorities), then re-run")
    return 0, (f"LEARNING-DEBT RATCHET OK: {debt} unit(s) held at-or-below the "
               f"pinned ceiling of {baseline}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Learning-docs scorecard (read-only, pedagogy-aware).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the LEARNING-SCORECARD.md body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--check", action="store_true",
                    help="CI ratchet gate: exit non-zero only if learning-debt exceeds --baseline (#1280)")
    ap.add_argument("--baseline", type=int, default=0,
                    help="allowed learning-debt ceiling for --check (default: 0)")
    ap.add_argument("--detect-latency-bench", action="store_true",
                    help="measure stale-stamp detect latency vs the 5-day manual baseline")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else dsc.repo_root()
    if args.detect_latency_bench:
        bench = detect_latency_bench(workspace)
        if args.json:
            print(json.dumps(bench, indent=2))
        else:
            print(render_detect_latency_bench(bench))
        return 0 if bench.get("ok") else 1

    payload = collect(workspace)

    if args.check:
        code, message = check_gate(payload, args.baseline)
        if args.json:
            # Under --check the tool's contract IS the ratchet, not the raw fold:
            # ok/verdict reflect "did learning-debt hold at-or-below baseline?"
            # (green even with residual debt), not "is debt zero?". gate_exit /
            # gate_message carry the literal CI verdict for a loop runner. #1280.
            gated = {
                **payload,
                "ok": code == 0,
                "verdict": "OK" if code == 0 else "ACTION",
                "gate_exit": code,
                "gate_message": message,
                "gate_baseline": args.baseline,
            }
            print(json.dumps(gated, indent=2))
        else:
            print(message)
        return code

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
