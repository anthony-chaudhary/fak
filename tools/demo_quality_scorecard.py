#!/usr/bin/env python3
"""Demo-quality scorecard — the measuring stick for *demos a skeptic can run*.

The repo ships several runnable demos (``examples/adjudication-demo``,
``examples/agentdojo-redteam``, ``examples/wire-proof``, ``examples/mcp``,
``cmd/simpledemo``) and bets a lot on them: a demo is where a skeptical adopter
*first* decides whether the central claim is real. The doc scorecards already
watch the demos' prose — ``docs_scorecard.py`` (dead links, stale pins, missing
titles) and ``doc_appeal_scorecard.py`` (does the writing land). Neither answers
the question a demo actually has to answer: **can I run it in one command,
reproduce exactly what the README promises, trust what it claims, and understand
what I just saw — without a model, a key, or a babysitter?** "Make the demos
better" was, until now, an unfalsifiable vibe — there was no number to move.

This is that number. It discovers every runnable demo and scores each on five
demo-quality axes, each 0-100, deterministic, content-only (no model, no network,
no build):

  runnable       a copy-paste one-command entry exists, and it points at a real file
  reproducible   a captured example run exists, and the run is checkable (exit code)
  honest_scope   the README states what the demo does NOT claim, and links the ledger
  self_contained prerequisites are stated; anything it creates, it cleans up
  documented     one H1 title, a run/usage section, and an output-explainer section

The axes fold into a weighted **demo-score** (0-100, A-F) AND — the lever that
turns "10x better demos" into a checkable target instead of a vibe — a
**demo-debt** integer: the count of concrete, re-derivable demo defects (no
runnable entry, a dead runner reference, no captured output, no scope statement,
state created without cleanup, no H1, …). Demo-debt is an integer you can drive
toward zero, so an improvement program can promise "cut demo-debt to zero" and
then *prove* it by re-running.

The **skeptical-adopter lens** sets the weights: a demo's first job is to RUN
(weight: most) and to let you CHECK it ran right (reproducible, next); honesty is
this project's signature, so it weighs heavily; friction and prose weigh least
because a demo that runs and reproduces survives a thin README, but a beautiful
README that won't run is not a demo.

HARD defects (each one unit of demo-debt, the work-list) are unambiguous and worth
fixing: no runnable entry, a README that names a ``run.sh``/``verify.py`` that does
not exist, no captured example output anywhere, no "what it does not claim" scope
statement, a script that creates files/dirs with no cleanup, a missing or absent
H1. SOFT signals (no stated prerequisites, no exit-code/determinism claim, no
deeper-doc cross-link, a missing run heading, more than one H1) lower the score but
never count as debt — they are judgment, not mechanical fact, the same FAIL vs
ADVISORY split the sibling scorecards draw.

Read-only by construction: it reads each demo's ``README.md``, its
``EXAMPLE-OUTPUT.md``, and its shell/python entry scripts; it edits nothing. Run
from the repo ROOT::

    python tools/demo_quality_scorecard.py            # human scorecard
    python tools/demo_quality_scorecard.py --json     # machine payload (control-pane)
    python tools/demo_quality_scorecard.py --markdown # the committed snapshot body

The companion process is the demos-to-zero program: each defect is one unit to
retire; re-running proves the number moved.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fleet-demo-quality-scorecard/1"

# ---------------------------------------------------------------------------
# The demo set. Demos are auto-discovered as immediate subdirectories of
# DEMO_ROOTS that carry a README.md (the adopter-facing demo surface AGENTS.md
# points at: "examples/ — Policy manifests AND runnable demos"). EXTRA_DEMOS adds
# known runnable demos that live elsewhere in the tree. Edit these when the demo
# surface moves; everything else is derived from disk.
# ---------------------------------------------------------------------------
DEMO_ROOTS: list[str] = ["examples"]
EXTRA_DEMOS: list[str] = ["cmd/simpledemo"]

# Per-axis weights for the composite demo-score. The skeptical-adopter lens leans
# the weights toward "does it run" (runnable) and "can I check it" (reproducible);
# honesty is the project's signature so it weighs next; friction and prose weigh
# least — a demo that runs and reproduces survives a thin README.
AXIS_WEIGHTS: dict[str, float] = {
    "runnable": 0.24,
    "reproducible": 0.22,
    "honest_scope": 0.20,
    "self_contained": 0.18,
    "documented": 0.16,
}

# Cap on how much of a script we read (entry scripts are tiny; this is a guard).
MAX_SCRIPT_BYTES = 200_000

# Script suffixes we read. `.go` is included so a Go demo (cmd/simpledemo's
# `func main`) is recognized as a real entry; the create/cleanup scan, however,
# stays shell/python-only (Go's `>`/`>>` are operators, not file redirects).
SCRIPT_SUFFIXES = {".sh", ".py", ".ps1", ".bash", ".go"}
# Suffixes whose body we scan for create-without-cleanup (self_contained).
TEARDOWN_SCAN_SUFFIXES = (".sh", ".py", ".ps1", ".bash")

# Captured-output content floor (anti-gaming). A "captured run" must be
# substantive — a 1-byte EXAMPLE-OUTPUT.md or a fenced block holding a single
# stray glyph does not let a reader tell what a correct run looks like.
MIN_OUTPUT_LINES = 3   # non-blank lines a captured run must carry
MIN_RUN_TELLS = 2      # ...or distinct run-signals in a multi-line fence

# Canonical runner basenames a README commonly references. If a README names one
# of these but the file is absent from the demo dir, that is a dead runner (HARD).
CANONICAL_RUNNERS = ("run.sh", "run.ps1", "verify.py", "demo.py", "run.py")

# ---------------------------------------------------------------------------
# Regexes (compiled once). Each is a deliberate, commented heuristic.
# ---------------------------------------------------------------------------
_H1_RE = re.compile(r"^#\s+\S", re.MULTILINE)
_LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)")

# A Go entry point: `func main(` in a non-test .go file makes a cmd/<x> dir a
# real runnable demo, so the runnable axis doesn't hinge on the README prose.
_GO_ENTRY_RE = re.compile(r"(?m)^\s*func\s+main\s*\(")

# A paste-able run command anywhere in the README. Deliberately matches the ways
# this repo actually launches a demo: a run script (bare, `./`-prefixed, or
# path-qualified like `./examples/foo/run.sh`), a named `verify.py`/`demo.py`,
# `go run`, `python3 x.py`, or `make <target>`. NOT `go build` (compiles, doesn't
# run). The `make` arm excludes English prose ("make sure …") via a negative
# lookahead — an over-broad `make\s+\w` was masking the no-runnable-entry defect.
# A server launch (`serve` / `--stdio` / `--addr`) is filtered out separately in
# `_is_run_command_line`: a long-running server is not a self-terminating demo run.
_RUN_CMD_RE = re.compile(
    r"(?mi)^\s*(?:\$\s*|>\s*)?(?:"
    r"(?:bash\s+|sh\s+)?[\w./~$-]*\b(?:run\.(?:sh|ps1|py)|verify\.py|demo\.py)\b"
    r"|go\s+run\b"
    r"|python3?\s+\S+\.py\b"
    r"|make\s+(?!(?:sure|it|a|an|the|your|my|me|this|that|them|us|do|some|certain|note|"
    r"changes|room)\b)[a-z][\w-]*\b"
    r")"
)
# A server launch is not a demo run — filtered from the run-command check.
_SERVER_LAUNCH_RE = re.compile(r"(?i)(\bserve\b|--stdio|--addr)")

# A captured-output section: a heading that promises a sample run. Run over the
# DE-FENCED README so a `## …` inside a code fence can't masquerade as a heading.
_OUTPUT_HEADING_RE = re.compile(
    r"(?im)^#{1,6}\s+.*("
    r"what you('?ll)? see|example output|sample (output|run)|reading the (output|verdict|"
    r"results?|run)|what you get|what this (shows|prints|demonstrates)|expected output"
    r").*$"
)

# Run-tell glyphs/words that mark a fenced block as a captured run rather than a
# config snippet or a command to type.
_RUN_TELL_RE = re.compile(
    r"(✓|✗|→|\bPASS\b|\bFAIL\b|\bDENY\b|\bALLOW\b|summary:|\bok \(|exit 0|"
    r"kernel test passed|reproduced|ASR|caught|MISSED)"
)

# A reproducibility statement: the demo says it is checkable / deterministic.
_REPRO_STMT_RE = re.compile(
    r"(?i)(exit code|exit 0\b|exit-code|CI-?usable|deterministic|byte-identical|"
    r"byte-for-byte|reproducib|same .{0,20}every (run|time)|returns? (0|1)\b|"
    r"\bgate(s)? the exit)"
)

# A scope / honesty statement: the demo states what it does NOT claim. Only
# genuine boundary ASSERTIONS count — a bare `## Scope` heading or an incidental
# "limitation"/"caveat" mention is not stating a boundary, and was the cheapest
# way to clear this HARD defect without honest framing. Run over emphasis-stripped,
# de-fenced prose so `does **not** demonstrate` matches.
_SCOPE_RE = re.compile(
    r"(?i)("
    r"does not (claim|demonstrate|prove|test|show|exercise|guarantee|cover)"
    r"|do not (claim|prove|demonstrate)"
    r"|what (this|it)( demo)? does not"
    r"|not (claimed|tested|demonstrated|exercised|shown|proven)(?:\s+(?:here|in this|by))?"
    r"|out of scope"
    r"|(deliberately|intentionally) (not|non|left|made)"
    r"|what (this )?does ?n'?t (do|work|claim|cover|prove)"
    r"|what doesn'?t work"
    r"|honest (note|position|scope)"
    r"|it does not (claim|exercise|prove|demonstrate|cover|run)"
    r"|is not (the|a) (security|complete|full|quality|benchmark|guarantee|portable)"
    r")"
)

# A prerequisites statement: the demo tells you what you need before running.
# (Dropped the bare `a `/`the ` arms — "needs a database" is not a prereq stmt.)
_PREREQ_RE = re.compile(
    r"(?i)(##\s*.{0,30}(prerequisite|requirement)|requires?:|prerequisite|prereq\b|"
    r"you('?ll)? need|you need\b|depends on|stdlib only|standard library|zero[- ]dep|"
    r"no dependenc|no (model|network|api key|gpu|ollama)|go-?only|requires go|"
    r"python ?3\b|needs? (go|python))"
)

# A run / usage heading (documented axis, soft). Run over the DE-FENCED README.
_RUN_HEADING_RE = re.compile(
    r"(?im)^#{1,6}\s+.*\b(run|usage|quick ?start|getting started|try it|"
    r"one-paste|setup|install|how to)\b.*$"
)

# State-creation verbs in a script (self_contained axis). If a script creates
# durable state and never tears it down, that is a defect.
_CREATE_RE = re.compile(
    r"(mktemp\b|mkdir\b|tempfile\.(mkdtemp|NamedTemporaryFile)|"
    r"os\.makedirs|Path\([^)]*\)\.mkdir|touch\b|\btee\b|>>?\s*[\"']?[\w./~$-]+\.\w)"
)
# Cleanup verbs that discharge the creation above. Only concrete teardown ACTIONS
# count — the bare words `cleanup` and `finally:` were dropped (a comment saying
# "cleanup" or a `finally:` that removes nothing is not teardown); a real
# `trap cleanup EXIT` / `atexit.register(cleanup)` still matches via trap/atexit.
_CLEANUP_RE = re.compile(
    r"(\btrap\b|atexit\.|shutil\.rmtree|tempfile\.TemporaryDirectory|"
    r"\brm\s+-[rf]|\brmdir\b|os\.remove|os\.rmdir|\.unlink\(|\bdefer\b|"
    r"ignore_errors=True|cleanup\s*\()"
)


# ---------------------------------------------------------------------------
# Demo model + discovery. This is the testable seam: every axis takes a `Demo`,
# and tests build one directly from fixture strings — no disk needed.
# ---------------------------------------------------------------------------

class Demo:
    """A discovered demo: its README, entry scripts, file list, captured output."""

    def __init__(self, rel: str, *, readme: str = "", scripts: dict[str, str] | None = None,
                 files: set[str] | None = None, example_output: str = "",
                 exists: bool = True) -> None:
        self.rel = rel
        self.readme = readme
        self.scripts = scripts or {}
        self.files = files if files is not None else set(self.scripts) | (
            {"README.md"} if readme else set())
        self.example_output = example_output
        self.exists = exists

    # --- derived views the axes lean on -----------------------------------
    @property
    def script_text(self) -> str:
        return "\n".join(self.scripts.values())

    @property
    def has_entry_script(self) -> bool:
        for name, body in self.scripts.items():
            low = name.lower()
            if low.endswith((".sh", ".ps1", ".bash")):
                return True
            if low.endswith(".py") and "__main__" in body:
                return True
            if low.endswith(".go") and not low.endswith("_test.go") and _GO_ENTRY_RE.search(body):
                return True
        return "Makefile" in self.files

    @property
    def has_run_command(self) -> bool:
        return any(_is_run_command_line(ln) for ln in self.readme.splitlines())


def discover_demos(root: Path) -> list[str]:
    """Auto-discover demos: README-bearing subdirs of DEMO_ROOTS, plus EXTRA_DEMOS."""
    out: list[str] = []
    for r in DEMO_ROOTS:
        base = root / r
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "README.md").exists():
                rel = child.relative_to(root).as_posix()
                if rel not in out:
                    out.append(rel)
    for e in EXTRA_DEMOS:
        if (root / e).is_dir() and e not in out:
            out.append(e)
    return out


def load_demo(root: Path, rel: str) -> Demo:
    d = root / rel
    if not d.is_dir():
        return Demo(rel, exists=False)
    readme = _safe_read(d / "README.md")
    files: set[str] = set()
    scripts: dict[str, str] = {}
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        files.add(p.name)
        if p.name == "README.md":
            continue
        if p.suffix.lower() in SCRIPT_SUFFIXES:
            scripts[p.name] = _safe_read(p)[:MAX_SCRIPT_BYTES]
    example_output = _safe_read(d / "EXAMPLE-OUTPUT.md")
    return Demo(rel, readme=readme, scripts=scripts, files=files,
                example_output=example_output, exists=True)


def _safe_read(path: Path) -> str:
    # errors="replace": a non-UTF-8 / binary file decodes tolerantly instead of
    # raising UnicodeDecodeError (a ValueError, not an OSError) that would crash
    # the whole corpus run. except is broadened to match, belt-and-suspenders.
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


# ---------------------------------------------------------------------------
# Captured-output detection (shared by reproducible + documented).
# ---------------------------------------------------------------------------

def _fenced_blocks(text: str) -> list[str]:
    """Return the contents of every ``` / ~~~ fenced block."""
    out: list[str] = []
    buf: list[str] | None = None
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            if buf is None:
                buf = []
            else:
                out.append("\n".join(buf))
                buf = None
            continue
        if buf is not None:
            buf.append(line)
    return out


def _strip_fences(text: str) -> str:
    """Drop fenced code blocks. Heading-shaped regexes run over this view so a
    shell `# comment` or a `## …` line inside a fence can't masquerade as a real
    Markdown heading (the simpledemo false '5 H1 titles' + the spoof vector)."""
    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


def _prose_norm(text: str) -> str:
    """De-fenced text with emphasis/backtick markers stripped, so a boundary
    statement written `does **not** demonstrate` is matched as `does not demonstrate`."""
    return re.sub(r"[`*_]+", "", _strip_fences(text))


def _is_run_command_line(line: str) -> bool:
    """A README line that launches the demo: a run command, but NOT a long-running
    server (`serve`/`--stdio`/`--addr`) — a server is not a self-terminating run."""
    return bool(_RUN_CMD_RE.search(line)) and not _SERVER_LAUNCH_RE.search(line)


def _readme_shows_a_run(demo: Demo) -> bool:
    """True only if the README displays a SUBSTANTIVE captured run: a fenced block
    of >= MIN_OUTPUT_LINES non-blank lines that either carries >= MIN_RUN_TELLS
    distinct run-signals OR follows an output-promising heading. This closes two
    gaming holes: a lone-glyph fence, and a config fence sitting before a later
    prose-only 'what you see' heading (the heading must precede the fence)."""
    lines = demo.readme.split("\n")
    out_heading_idxs: list[int] = []
    blocks: list[tuple[int, list[str]]] = []
    in_fence = False
    buf: list[str] = []
    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            if not in_fence:
                in_fence, buf, start = True, [], i
            else:
                in_fence = False
                blocks.append((start, buf))
            continue
        if in_fence:
            buf.append(line)
        elif _OUTPUT_HEADING_RE.search(line):
            out_heading_idxs.append(i)
    for blk_start, content in blocks:
        if len([ln for ln in content if ln.strip()]) < MIN_OUTPUT_LINES:
            continue
        tells = {m.group(0) for m in _RUN_TELL_RE.finditer("\n".join(content))}
        follows_heading = any(h < blk_start for h in out_heading_idxs)
        if len(tells) >= MIN_RUN_TELLS or follows_heading:
            return True
    return False


def has_captured_output(demo: Demo) -> bool:
    eo_lines = [ln for ln in demo.example_output.split("\n") if ln.strip()]
    if len(eo_lines) >= MIN_OUTPUT_LINES:
        return True
    return _readme_shows_a_run(demo)


# ---------------------------------------------------------------------------
# The five axes. Each returns
#   {axis, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of demo-debt (the work-list); soft = score-only nudges.
# ---------------------------------------------------------------------------

def axis_runnable(demo: Demo) -> dict[str, Any]:
    """A skeptic can launch the demo with one copy-paste command."""
    defects: list[str] = []
    soft: list[str] = []
    score = 100.0

    has_entry = demo.has_entry_script
    has_cmd = demo.has_run_command
    if not has_entry and not has_cmd:
        defects.append("no runnable entry: no run script / `go run` / `python x.py` / make "
                       "command, and no `__main__` script — there is no one-command way to run it")
        score -= 55
    elif has_entry and not has_cmd:
        soft.append("a runnable script exists but the README shows no paste-able command to launch it")
        score -= 14

    # Dead runner: the README presents a runner that is not on disk. Covers both
    # the canonical names AND any `./…`-prefixed script invocation (e.g. an
    # `./start.sh` the demo dir doesn't ship) — checked by basename against the
    # dir's files, so it stays a pure function of the Demo (no disk in the axis).
    referenced: set[str] = set()
    for name in CANONICAL_RUNNERS:
        if re.search(r"\b" + re.escape(name) + r"\b", demo.readme):
            referenced.add(name)
    for m in re.finditer(r"\./([\w./-]+\.(?:sh|ps1|py))\b", demo.readme):
        referenced.add(m.group(1).rsplit("/", 1)[-1])
    for base in sorted(referenced):
        if base not in demo.files:
            defects.append(f"dead runner reference: README presents `{base}` but no such file in the demo dir")
            score -= 25

    detail = (f"entry-script={has_entry} · run-cmd-in-readme={has_cmd} · "
              f"scripts={sorted(demo.scripts) or '—'}")
    return {"axis": "runnable", "score": _clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


def axis_reproducible(demo: Demo) -> dict[str, Any]:
    """You can confirm it reproduced: a captured run, and a checkable exit."""
    defects: list[str] = []
    soft: list[str] = []
    score = 100.0

    captured = has_captured_output(demo)
    if not captured:
        defects.append("no captured example output: no EXAMPLE-OUTPUT.md and no in-README "
                       "sample run — a reader cannot tell what a correct run looks like")
        score -= 38

    repro_stmt = bool(_REPRO_STMT_RE.search(demo.readme) or _REPRO_STMT_RE.search(demo.script_text))
    if not repro_stmt:
        soft.append("no exit-code / determinism statement — the demo doesn't say how to tell "
                    "pass from fail (a CI gate needs this)")
        score -= 14

    detail = (f"captured-output={'EXAMPLE-OUTPUT.md' if demo.example_output.strip() else captured} · "
              f"checkable={repro_stmt}")
    return {"axis": "reproducible", "score": _clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


def axis_honest_scope(demo: Demo) -> dict[str, Any]:
    """The demo states what it does NOT claim, and points at the honesty ledger."""
    defects: list[str] = []
    soft: list[str] = []
    score = 100.0

    # Scope must be an actual boundary STATEMENT in the README (the demo's front
    # door a runner reads), matched on emphasis-stripped, de-fenced prose — not a
    # bare `## Scope` heading, an incidental "limitation", or scope text hidden in
    # a sibling integration guide.
    has_scope = bool(_SCOPE_RE.search(_prose_norm(demo.readme)))
    if not has_scope:
        defects.append("no scope / 'what this does not claim' statement — a demo that only "
                       "shows the win overclaims; state the boundary honestly")
        score -= 42

    # SOFT: a demo making a claim should link a deeper / honesty doc (CLAIMS, STATUS,
    # an explainer). A demo with no outbound .md link is a navigational dead-end.
    md_links = [m.group("target") for m in _LINK_RE.finditer(demo.readme)
                if m.group("target").split("#")[0].strip().lower().endswith(".md")]
    if not md_links:
        soft.append("no link to a deeper doc (CLAIMS / STATUS / an explainer) to back the claim")
        score -= 10

    detail = f"scope-statement={has_scope} · deeper-doc-links={len(md_links)}"
    return {"axis": "honest_scope", "score": _clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


def axis_self_contained(demo: Demo) -> dict[str, Any]:
    """Low friction to run, and it leaves no mess behind."""
    defects: list[str] = []
    soft: list[str] = []
    score = 100.0

    # HARD: a shell/python script that creates durable state but never tears it
    # down. (.go is excluded — its `>`/`>>` are operators, not file redirects.)
    for name, body in demo.scripts.items():
        if name.lower().endswith(TEARDOWN_SCAN_SUFFIXES):
            if _CREATE_RE.search(body) and not _CLEANUP_RE.search(body):
                defects.append(f"`{name}` creates files/dirs (mktemp/mkdir/redirect) with no "
                               f"cleanup (no trap/atexit/rm/TemporaryDirectory) — it can leave a mess")
                score -= 22

    # SOFT: a runnable demo with no stated prerequisites.
    if (demo.has_entry_script or demo.has_run_command) and not _PREREQ_RE.search(demo.readme):
        soft.append("no stated prerequisites — a cold runner can't tell what to install first")
        score -= 12

    detail = (f"scripts={len(demo.scripts)} · prereqs-stated="
              f"{bool(_PREREQ_RE.search(demo.readme))}")
    return {"axis": "self_contained", "score": _clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


def axis_documented(demo: Demo) -> dict[str, Any]:
    """One title, a run/usage section, an output-explainer section."""
    defects: list[str] = []
    soft: list[str] = []
    score = 100.0

    # Heading-shaped checks run over the DE-FENCED README so shell `# comments` /
    # in-fence `## …` lines can't be counted as Markdown headings.
    defenced = _strip_fences(demo.readme)
    h1s = _H1_RE.findall(defenced)
    if len(h1s) == 0:
        defects.append("no H1 title (a '# Title' line) in the README")
        score -= 30
    elif len(h1s) > 1:
        soft.append(f"{len(h1s)} H1 titles in the README (expected exactly one)")
        score -= 8

    has_run_section = bool(_RUN_HEADING_RE.search(defenced) or demo.has_run_command)
    if not has_run_section:
        soft.append("no run/usage section and no visible run command — hard to find how to start")
        score -= 12

    has_output_section = bool(_OUTPUT_HEADING_RE.search(defenced))
    if not has_output_section:
        soft.append("no 'what you see' / output-explainer section — the reader is left to "
                    "interpret the run alone")
        score -= 10

    detail = (f"H1={len(h1s)} · run-section={has_run_section} · "
              f"output-section={has_output_section}")
    return {"axis": "documented", "score": _clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


# ---------------------------------------------------------------------------
# Per-demo fold + grader
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


def score_demo(demo: Demo) -> dict[str, Any]:
    if not demo.exists:
        return missing_demo_entry(demo.rel)
    axes = [
        axis_runnable(demo),
        axis_reproducible(demo),
        axis_honest_scope(demo),
        axis_self_contained(demo),
        axis_documented(demo),
    ]
    by_name = {a["axis"]: a for a in axes}
    composite = sum(AXIS_WEIGHTS[name] * by_name[name]["score"] for name in AXIS_WEIGHTS)
    defects = [f"{a['axis']}: {d}" for a in axes for d in a["defects"]]
    soft = [f"{a['axis']}: {s}" for a in axes for s in a["soft"]]
    return {
        "path": demo.rel,
        "score": round(composite, 1),
        "grade": grade_letter(composite),
        "axes": {a["axis"]: a["score"] for a in axes},
        "axis_detail": {a["axis"]: a["detail"] for a in axes},
        "axis_debt": {a["axis"]: len(a["defects"]) for a in axes},
        "defects": defects,
        "soft": soft,
        "n_defects": len(defects),
    }


def missing_demo_entry(rel: str) -> dict[str, Any]:
    return {
        "path": rel, "score": 0.0, "grade": "F",
        "axes": {a: 0 for a in AXIS_WEIGHTS}, "axis_detail": {}, "axis_debt": {},
        "defects": [f"missing: demo {rel} does not exist on disk"],
        "soft": [], "n_defects": 1,
    }


# ---------------------------------------------------------------------------
# Corpus fold -> the standard control-pane payload
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, demos: list[dict[str, Any]],
                  error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error,
            "next_action": "fix the read (run from repo ROOT), then re-run",
            "workspace": workspace, "corpus": {}, "demos": demos,
        }
    n = len(demos)
    if n == 0:
        # An empty corpus is NOT a clean pass — a scorecard that finds nothing to
        # score and reports OK/exit-0 is a silent false-pass (wrong cwd, bad
        # --workspace, or a moved DEMO_ROOTS). Fail loud instead.
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "no_demos",
            "reason": "no demos discovered — run from repo ROOT, or fix DEMO_ROOTS/EXTRA_DEMOS",
            "next_action": "run from the repository root so examples/ and cmd/simpledemo resolve",
            "workspace": workspace, "corpus": {"n_demos": 0, "demo_debt": 0}, "demos": demos,
        }
    scores = [d["score"] for d in demos]
    demo_debt = sum(d["n_defects"] for d in demos)
    mean_score = round(sum(scores) / max(1, n), 1)
    grades = {g: 0 for g in "ABCDF"}
    for d in demos:
        grades[d["grade"]] = grades.get(d["grade"], 0) + 1
    worst = sorted(demos, key=lambda d: (d["score"], -d["n_defects"]))

    corpus = {
        "n_demos": n,
        "mean_score": mean_score,
        "median_score": round(statistics.median(scores), 1) if scores else 0.0,
        "min_score": round(min(scores), 1) if scores else 0.0,
        "max_score": round(max(scores), 1) if scores else 0.0,
        "grade_distribution": grades,
        "demo_debt": demo_debt,
        "worst": [{"path": d["path"], "score": d["score"], "grade": d["grade"],
                   "n_defects": d["n_defects"]} for d in worst],
    }

    if demo_debt == 0:
        ok, verdict, finding = True, "OK", "demos_clean"
        reason = (f"demos clean: {n} demos, mean {mean_score}/100, zero demo-debt — "
                  f"every demo runs, reproduces, scopes itself honestly, and cleans up")
        next_action = "no required edit; re-run after the next demo change"
    else:
        ok, verdict, finding = False, "ACTION", "demo_debt"
        worst_demo = worst[0]
        reason = (f"{demo_debt} unit(s) of demo-debt across {n} demos; mean {mean_score}/100; "
                  f"weakest: {worst_demo['path']} ({worst_demo['score']}/100, "
                  f"{worst_demo['n_defects']} defect(s))")
        next_action = ("retire demo-debt worst-first (see corpus.worst + demo.defects): add a "
                       "one-command runner, capture EXAMPLE-OUTPUT.md, state what the demo does "
                       "NOT claim, add cleanup; re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "demos": demos,
    }


# ---------------------------------------------------------------------------
# Wiring + CLI
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def collect(workspace: Path, *, demo_rels: list[str] | None = None) -> dict[str, Any]:
    root = workspace.resolve()
    rels = demo_rels if demo_rels is not None else discover_demos(root)
    demos = [score_demo(load_demo(root, rel)) for rel in rels]
    return build_payload(workspace=str(root), demos=demos)


def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    lines = [
        f"demo-quality scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"corpus: {c.get('n_demos', 0)} demos · mean {c.get('mean_score', 0)}/100 "
         f"· min {c.get('min_score', 0)} · DEMO-DEBT {c.get('demo_debt', 0)}"),
        ("grades: " + " ".join(f"{g}:{c.get('grade_distribution', {}).get(g, 0)}"
                               for g in "ABCDF")),
        f"next: {payload.get('next_action')}",
        "",
        "per-demo (worst first):",
        f"  {'score':>5} {'gr':>2} {'def':>3}  run rep hon slf doc  demo",
    ]
    for d in sorted(payload.get("demos", []), key=lambda x: (x["score"], -x["n_defects"])):
        a = d.get("axes", {})
        lines.append(
            f"  {d['score']:>5} {d['grade']:>2} {d['n_defects']:>3}  "
            f"{a.get('runnable','-'):>3} {a.get('reproducible','-'):>3} "
            f"{a.get('honest_scope','-'):>3} {a.get('self_contained','-'):>3} "
            f"{a.get('documented','-'):>3}  {d['path']}")
    lines.append("")
    lines.append("demo-debt work-list:")
    any_defect = False
    for d in sorted(payload.get("demos", []), key=lambda x: -x["n_defects"]):
        if not d["defects"]:
            continue
        any_defect = True
        lines.append(f"  {d['path']} ({d['n_defects']}):")
        for it in d["defects"]:
            lines.append(f"      - {it}")
    if not any_defect:
        lines.append("  (none — demos clean)")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    """The committed DEMO-QUALITY-SCORECARD.md body — a human-facing snapshot."""
    c = payload.get("corpus") or {}
    gd = c.get("grade_distribution", {})
    n_demos = c.get("n_demos", 0)
    # YAML front-matter (SEO/AEO: a <title> tag + meta description on the published
    # page). Emitted by the GENERATOR so it survives a regen — the doc is a generated
    # artifact, so a hand-added front-matter block would be wiped on the next
    # `--markdown` write. Title/description are derived from the live corpus.
    out: list[str] = [
        "---",
        'title: "fak Demo-Quality Scorecard: Demos a Skeptic Can Run"',
        ('description: "fak\'s demo-quality scorecard grades '
         f'{n_demos} demos on five deterministic axes into a demo-score (0-100, A-F) '
         'and a re-derivable demo-debt count."'),
        "---",
        "",
        "# Demo-quality scorecard",
        "",
    ]
    if stamp:
        out.append(f"<!-- demo-quality-scorecard: {stamp} · process: tools/demo_quality_scorecard.py -->")
        out.append("")
    out.append("> Regenerate: `python tools/demo_quality_scorecard.py --markdown --stamp DATE "
               "> docs/DEMO-QUALITY-SCORECARD.md`")
    out.append("")
    out.append("> The measuring stick for **demos a skeptic can run**: can I run it in one "
               "command, reproduce what the README promises, trust what it claims, and "
               "understand what I saw — without a model, a key, or a babysitter? Five "
               "deterministic axes (runnable · reproducible · honest_scope · self_contained · "
               "documented), folded into a **demo-score** (0–100, A–F) and a **demo-debt** "
               "integer (the count of concrete, re-derivable demo defects). Every number below "
               "is re-derived from disk by `tools/demo_quality_scorecard.py` — no hand-entry. "
               "Drive demo-debt to zero to make \"better demos\" provable.")
    out.append("")
    out.append("## Corpus")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Demos scored | {c.get('n_demos', 0)} |")
    out.append(f"| **Demo-debt (total defects)** | **{c.get('demo_debt', 0)}** |")
    out.append(f"| Mean score | {c.get('mean_score', 0)}/100 |")
    out.append(f"| Median / min / max | {c.get('median_score', 0)} / {c.get('min_score', 0)} / {c.get('max_score', 0)} |")
    out.append(f"| Grade distribution | A:{gd.get('A',0)} B:{gd.get('B',0)} C:{gd.get('C',0)} D:{gd.get('D',0)} F:{gd.get('F',0)} |")
    out.append("")
    out.append("## Per-demo scores")
    out.append("")
    out.append("Five axes, each 0–100 (runnable · reproducible · honest_scope · self_contained · "
               "documented), weighted into a score and an A–F grade. `def` = units of demo-debt.")
    out.append("")
    out.append("| Score | Grade | Debt | run | repro | scope | self | docs | Demo |")
    out.append("|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|")
    for d in sorted(payload.get("demos", []), key=lambda x: (x["score"], -x["n_defects"])):
        a = d.get("axes", {})
        out.append(
            f"| {d['score']} | {d['grade']} | {d['n_defects']} | "
            f"{a.get('runnable','-')} | {a.get('reproducible','-')} | {a.get('honest_scope','-')} | "
            f"{a.get('self_contained','-')} | {a.get('documented','-')} | `{d['path']}` |")
    out.append("")
    out.append("## Demo-debt work-list")
    out.append("")
    any_defect = False
    for d in sorted(payload.get("demos", []), key=lambda x: -x["n_defects"]):
        if not d["defects"]:
            continue
        any_defect = True
        out.append(f"### `{d['path']}` — {d['n_defects']} defect(s), score {d['score']} ({d['grade']})")
        for it in d["defects"]:
            out.append(f"- {it}")
        out.append("")
    if not any_defect:
        out.append("No demo-debt: every demo runs, reproduces, scopes itself, and cleans up. 🎉")
        out.append("")
    # Soft signals (score only) for the worst few — judgment calls, not gates.
    soft_demos = [d for d in sorted(payload.get("demos", []), key=lambda x: x["score"]) if d.get("soft")]
    if soft_demos:
        out.append("## Soft signals (score only, not debt)")
        out.append("")
        for d in soft_demos:
            out.append(f"### `{d['path']}`")
            for s in d["soft"]:
                out.append(f"- {s}")
            out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Demo-quality scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true",
                    help="emit the DEMO-QUALITY-SCORECARD.md body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    args = ap.parse_args(argv)

    # Demos carry Unicode (✓, →, ·, —, 🤖); force UTF-8 stdout so a Windows cp1252
    # console can't crash the scorer on a glyph.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
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
