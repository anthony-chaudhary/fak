#!/usr/bin/env python3
"""Demo-robustness scorecard — the measuring stick for demos that stay *simple,
fast, and durable*.

The sibling ``demo_quality_scorecard.py`` answers the skeptical-adopter's *first*
question — can I run this in one command, reproduce what the README promises, trust
the claim, and understand what I saw? That scorecard is at zero debt. This one
answers the *second* question, the one a maintainer and a returning user care about:
**will it still run next month, on a fresh box, in one obvious command, in seconds —
without surprises?** "Make the demos more robust" was an unfalsifiable vibe; this is
the number.

It discovers the same demo corpus as the quality scorecard (it reuses that module's
discovery + loader, so the two never drift apart) and scores each demo on three
robustness axes, each 0-100, deterministic, content-only (no model, no network, no
build, no execution):

  simplicity   few moving parts: one obvious command, a small surface to read,
               not a maze of prerequisites/steps before the headline result.
  speed        fast feedback with no surprise waits: a stated runtime, no
               unbounded polling loop that can hang forever, a `go run` fast path.
  durability   still works later, everywhere, every time: pinned external deps,
               a stated stability/determinism guarantee, no run-time fetch of a
               mutable remote, error-stopping shell, a cross-platform note.

The axes fold into a weighted **robustness-score** (0-100, A-F) AND — the lever that
turns "3x more robust demos" into a checkable target instead of a vibe — a
**robustness-debt** integer: the count of concrete, re-derivable robustness defects
(an unbounded wait loop, an unpinned remote pull, no stated runtime, no stability
guarantee, …). Robustness-debt is an integer you can drive down, so an improvement
program can promise "cut robustness-debt 3x" and then *prove* it by re-running.

HARD defects (each one unit of robustness-debt, the work-list) are unambiguous and
genuinely retireable through honest, additive edits — the same discipline the quality
scorecard uses (it makes "no scope statement" a defect retired by writing the true
scope sentence): an unbounded `until…; sleep` loop with no deadline, an `ollama pull`
/ `go get …@latest` / unpinned `pip install` (a run-time fetch of a mutable remote),
no stated expected runtime, no stated stability/determinism guarantee, a shell entry
without `set -e`, a multi-step run with no single command. SOFT signals (builds the
whole binary instead of `go run`, a network/model default path, shell-only with no
cross-platform note, many prerequisites/files) lower the score but never count as
debt — judgment, not mechanical fact, the same FAIL vs ADVISORY split the sibling
scorecards draw.

Read-only by construction: it reads each demo's README, its entry scripts, and its
captured output; it edits nothing. Run from the repo ROOT::

    python tools/demo_robustness_scorecard.py            # human scorecard
    python tools/demo_robustness_scorecard.py --json     # machine payload (control-pane)
    python tools/demo_robustness_scorecard.py --markdown # the committed snapshot body

The companion process is the robustness-3x program: each defect is one unit to
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

# Reuse the quality scorecard's demo discovery, loader, Demo model, and de-fencing
# helpers so the two scorecards score the IDENTICAL corpus and never drift. This is
# the single source of "what is a demo" for the whole demo-scorecard family.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_quality_scorecard as dq  # noqa: E402

SCHEMA = "fleet-demo-robustness-scorecard/1"

# Per-axis weights for the composite robustness-score. Speed and durability lean
# heaviest — a returning user is bitten by a hang or a rotted dependency far more
# than by an extra prerequisite — and simplicity weighs a touch less because a demo
# that runs fast and keeps working survives a slightly larger surface.
AXIS_WEIGHTS: dict[str, float] = {
    "simplicity": 0.30,
    "speed": 0.35,
    "durability": 0.35,
}

# Simplicity: a generous LOC budget for the entry surface a reader must trust. Above
# it WITHOUT a short "start here" pointer (a QUICKSTART file or heading) is a defect —
# a big demo is fine if it tells you the small core to read first.
ENTRY_LOC_BUDGET = 700
MANY_PREREQS = 3      # > this many distinct required tools is a soft simplicity nudge
MANY_FILES = 8        # > this many files in the demo dir is a soft nudge

# ---------------------------------------------------------------------------
# Regexes (compiled once). Each is a deliberate, commented heuristic, calibrated
# against the live demo corpus.
# ---------------------------------------------------------------------------

# A stated expected RUNTIME (speed). A demo should tell you how long a run takes.
# Deliberately strict to avoid matching prose like "second turn" / "a second role"
# / "13 tokens": it requires a number+time-unit sitting next to a run/timing word,
# OR an explicit speed phrase. Run over de-fenced prose.
_RUNTIME_RE = re.compile(
    r"(?i)("
    # "<word> ... ~12s" / "completes in 2 seconds" / "runs in under a minute"
    r"\b(runs?|run|completes?|finish(es|ed)?|takes?|executes?|elapsed|wall[- ]?clock|"
    r"end[- ]to[- ]end|total|latency)\b[^.\n]{0,40}?"
    r"(\d+(\.\d+)?\s?(ms|millisecond|sec|secs|second|seconds|s\b|min|mins|minute|minutes|hour|hours)"
    r"|sub-?second|a few seconds|under (a|one) (second|minute)|instant(ly)?|near-?instant|in seconds)"
    # ...or a number+unit immediately followed by a run/timing word
    r"|\b\d+(\.\d+)?\s?(ms|sec|secs|second|seconds|min|mins|minute|minutes)\b[^.\n]{0,18}"
    r"(to run|per run|run|wall|total|end[- ]to[- ]end|complete|elapsed)"
    # ...or a bare strong speed phrase
    r"|\b(sub-?second|near-?instant|instant(ly)?|in (a )?few seconds|under (a|one) second|"
    r"runs? in seconds|seconds to run|milliseconds to run)\b"
    r")"
)

# A stability / determinism guarantee (durability): a reader can tell the run is
# repeatable / safe to re-run. Run over de-fenced, emphasis-stripped prose.
_STABILITY_RE = re.compile(
    r"(?i)("
    r"determinist|byte-identical|byte-for-byte|reproducib|idempotent|"
    r"same (output|result|verdict|report|bytes)[^.\n]{0,25}(each|every)\b|"
    r"(each|every) (run|time|invocation)[^.\n]{0,25}(identical|same|byte)|"
    r"identical[^.\n]{0,20}every|frozen (corpus|matrix|set)|pinned|"
    r"stable across (runs|versions|platforms)|no (randomness|nondeterminism)|"
    r"safe to re-?run|re-?runnable"
    r")"
)

# A floating / UNPINNED run-time external dependency (durability): a fetch whose
# version is not nailed down, so the demo silently changes when the remote moves.
# Deliberately strict: `go get/install …@latest|@main`, `pip install X` with no
# `==`/digest, `npm i pkg` with no `@ver`, and `curl|wget … | sh` bootstraps. A
# version-tagged, cache-guarded local-model fetch (`ollama pull qwen2.5:14b`, only
# pulled if absent) is NOT this — it is a pinned conventional path, scored by the
# softer net-fetch signal instead. Scanned over SHELL scripts AND README fences.
_FLOATING_DEP_RE = re.compile(
    r"(?im)("
    r"\bgo\s+(get|install)\b[^\n]*@(latest|main|master|head)\b"
    r"|\bpip3?\s+install\b(?![^\n]*(==|@[\da-f]))[^\n]*\b[a-z][\w.-]+"
    r"|\bnpm\s+(i|install)\b(?![^\n]*@\d)[^\n]*\b[a-z][\w.-]+"
    r"|\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(sh|bash)\b"
    r")"
)

# A run-time network fetch of a (presumably remote) asset — softer than a floating
# pin: a `curl`/`wget`/`ollama pull`/`go get` that reaches out at run time. Localhost
# fetches (the demos' own health checks) are excluded by the caller.
_NET_FETCH_RE = re.compile(
    r"(?im)\b(curl|wget|ollama\s+pull|go\s+(get|install)|pip3?\s+install|npm\s+(i|install))\b"
)
_LOCALHOST_RE = re.compile(r"(127\.0\.0\.1|localhost|0\.0\.0\.0|\$\{?(OLLAMA|FAK_DEMO_KERNEL))")

# Builds the whole binary instead of `go run` (soft speed): a leftover artifact +
# a slower cold start than `go run` of the entry package.
_GO_BUILD_RE = re.compile(r"(?m)\bgo\s+build\b")
_GO_RUN_RE = re.compile(r"(?m)\bgo\s+run\b")

# A shell error-stopping guard (durability). Any of `set -e` / `set -euo` / `set -o
# errexit` discharges the no-set-e defect.
_SET_E_RE = re.compile(r"(?m)^\s*set\s+-([a-z]*e[a-z]*|o\s+errexit)\b")

# Loop headers + sleeps, for the unbounded-wait scan.
_SH_LOOP_HEAD_RE = re.compile(r"^\s*(until|while)\b")
_SH_DONE_RE = re.compile(r"^\s*done\b")
_PY_WHILE_RE = re.compile(r"^(\s*)while\b.*:\s*(#.*)?$")
_SLEEP_RE = re.compile(r"(?i)(\bsleep\s+[\d.]|\btime\.sleep\s*\(|\basyncio\.sleep\s*\(|\bsleep\s*\()")
# Tokens that make a polling loop BOUNDED (a deadline / counter / cap exists).
_BOUND_TOKEN_RE = re.compile(
    r"(?i)(\bseq\b|\btimeout\b|date\s+\+%s|\$\(\(|--?ge\b|--?gt\b|--?lt\b|--?le\b|"
    r"\btries\b|\battempt|\bdeadline\b|\bmax[_-]?(tries|attempts|wait|iter)|"
    r"\belapsed\b|time\.(time|monotonic)\s*\(|\brange\s*\(|\bfor\b|_TIMEOUT\b|"
    r"\bretries?\b|\bbudget\b|\bcount(er)?\b\s*[<>=])"
)

# A QUICKSTART / "start here" pointer (simplicity): lets a big demo stay simple by
# naming the small core to read first.
_QUICKSTART_RE = re.compile(
    r"(?im)(^#{1,6}\s+.*\b(quick ?start|start here|tl;?dr|in 30 seconds|prove it first)\b"
    r"|\bsee\s+`?QUICKSTART|\bread\s+[\w./`]+\s+first\b)"
)

# A required multi-step RUN (simplicity): an ordered "1. … 2. …" list whose first
# two items each carry a shell-command token, OR explicit "first run/build X … then
# run/build Y" prose. The command-token requirement is deliberate: a conceptual
# ordered list that *explains the demo's behavior* ("1. the kernel decides … 2. the
# harness routes …") is NOT a run procedure and must not trip this. Run over prose.
_CMD_TOKEN = r"(\./|\bgo\s|\bpython3?\b|\bmake\s|\bnpm\s|\bcargo\s|\bexport\s|\bcd\s|\bbash\s|\bsh\s)"
_MULTISTEP_RE = re.compile(
    r"(?im)(^\s*1\.[^\n]*" + _CMD_TOKEN + r"[^\n]*\n(?:.*\n)*?^\s*2\.[^\n]*" + _CMD_TOKEN
    + r"|\bfirst\b[^.\n]{0,40}\b(run|build|start|export|launch|install)\b[^.\n]{0,50}"
    + r"\bthen\b[^.\n]{0,40}\b(run|build|start|export|launch|install)\b)"
)

# A cross-platform / shell note (durability soft): tells a non-bash user how to run.
_XPLAT_RE = re.compile(
    r"(?i)(\bgit ?bash\b|\bwsl\b|\bmsys|\bpowershell\b|\b\.ps1\b|cross-?platform|"
    r"on windows|windows users?|\bbash\b (is )?required|requires? bash|use bash)"
)

# Required-tool detection (simplicity soft). Each pattern requires a COMMAND form
# or an explicit requires-context, never a bare mention — so a README that says
# "No ollama" or "no GPU" is not miscounted as needing that tool. Read off README
# prose + scripts.
_TOOL_PATTERNS: dict[str, re.Pattern[str]] = {
    "go": re.compile(r"(?i)(\bgo\s+(build|run|test)\b|\bgo\.mod\b|requires? go|\bGo toolchain\b|\bto build\b[^\n]*\bfak\b)"),
    "python": re.compile(r"(?i)(\bpython3?\s+\S|\.py\b|stdlib only|standard library)"),
    "ollama": re.compile(r"(?i)\bollama\s+(serve|pull|list|run|ps|show)\b"),
    "node": re.compile(r"(?i)(\bnpm\s+\w|\bnpx\s+\w|\bnode\s+\S)"),
    "docker": re.compile(r"(?i)\bdocker\s+(run|build|compose|pull|exec)\b"),
    "curl": re.compile(r"(?i)\bcurl\s+(-|http)"),
}


# ---------------------------------------------------------------------------
# Small derived views over the (reused) Demo model.
# ---------------------------------------------------------------------------

def _readme_prose(demo: dq.Demo) -> str:
    """De-fenced, emphasis-stripped README — heading/prose regexes run over this."""
    return dq._prose_norm(demo.readme)


def _entry_loc(demo: dq.Demo) -> int:
    """Lines of the entry scripts a reader must trust (test files excluded)."""
    total = 0
    for name, body in demo.scripts.items():
        low = name.lower()
        if low.endswith("_test.go") or low.endswith("_test.py"):
            continue
        total += len(body.splitlines())
    return total


def _required_tools(demo: dq.Demo) -> list[str]:
    hay = demo.readme + "\n" + demo.script_text
    return sorted(t for t, pat in _TOOL_PATTERNS.items() if pat.search(hay))


def _shell_script_text(demo: dq.Demo) -> str:
    """Concatenated bodies of the SHELL entry scripts only (.sh/.bash/.ps1).

    The floating-dep / network-fetch scans run over this — never over .py/.go
    bodies, where a `curl … | sh` or `pip install` is almost always string data
    (an attack scenario the kernel blocks, a comment, a help string), not a real
    run-time command. The README's command fences are scanned separately.
    """
    return "\n".join(body for name, body in demo.scripts.items()
                     if name.lower().endswith((".sh", ".bash", ".ps1")))


def _scan_text_for_floating_deps(text: str) -> list[str]:
    return [m.group(0).strip() for m in _FLOATING_DEP_RE.finditer(text)]


def _readme_fence_text(demo: dq.Demo) -> str:
    """Concatenated contents of the README's fenced code blocks (the commands)."""
    return "\n".join(dq._fenced_blocks(demo.readme))


def _has_unbounded_shell_loop(body: str) -> bool:
    """A shell `until`/`while` loop whose body sleeps but has no deadline/counter."""
    lines = body.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        if _SH_LOOP_HEAD_RE.search(lines[i]):
            block = [lines[i]]
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if _SH_LOOP_HEAD_RE.search(lines[j]):
                    depth += 1
                elif _SH_DONE_RE.search(lines[j]):
                    depth -= 1
                    if depth == 0:
                        break
                block.append(lines[j])
                j += 1
            blob = "\n".join(block)
            if _SLEEP_RE.search(blob) and not _BOUND_TOKEN_RE.search(blob):
                return True
            i = j + 1
            continue
        i += 1
    return False


def _has_unbounded_py_loop(body: str) -> bool:
    """A python `while` loop whose (indentation) body sleeps with no time/iter bound."""
    lines = body.split("\n")
    n = len(lines)
    for i, line in enumerate(lines):
        m = _PY_WHILE_RE.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        block = [line]
        j = i + 1
        while j < n:
            nxt = lines[j]
            if nxt.strip() == "":
                block.append(nxt)
                j += 1
                continue
            cur_indent = len(nxt) - len(nxt.lstrip())
            if cur_indent <= indent:
                break
            block.append(nxt)
            j += 1
        blob = "\n".join(block)
        if _SLEEP_RE.search(blob) and not _BOUND_TOKEN_RE.search(blob):
            return True
    return False


def _unbounded_wait_scripts(demo: dq.Demo) -> list[str]:
    """Entry scripts that contain an unbounded polling loop (can hang forever)."""
    hits: list[str] = []
    for name, body in demo.scripts.items():
        low = name.lower()
        if low.endswith((".sh", ".bash", ".ps1")):
            if _has_unbounded_shell_loop(body):
                hits.append(name)
        elif low.endswith(".py"):
            if _has_unbounded_py_loop(body):
                hits.append(name)
    return hits


def _shell_scripts_without_set_e(demo: dq.Demo) -> list[str]:
    out: list[str] = []
    for name, body in demo.scripts.items():
        if name.lower().endswith((".sh", ".bash")) and not _SET_E_RE.search(body):
            out.append(name)
    return out


def _has_run_time_net_fetch(demo: dq.Demo) -> bool:
    """A non-localhost network fetch on the run path (README fences + shell scripts).

    Scans shell scripts (not .py/.go) for the same reason as the floating-dep scan:
    a curl/pip in a Python body is overwhelmingly string data, not a run command.
    """
    for blob in [_readme_fence_text(demo), _shell_script_text(demo)]:
        for line in blob.split("\n"):
            if _NET_FETCH_RE.search(line) and not _LOCALHOST_RE.search(line):
                # An `ollama pull` / `go get` / `pip install` is inherently remote.
                if re.search(r"(?i)(ollama\s+pull|go\s+(get|install)|pip3?\s+install|npm\s+(i|install))", line):
                    return True
                # A bare curl/wget to a non-localhost http(s) URL.
                if re.search(r"(?i)\b(curl|wget)\b.*https?://", line):
                    return True
    return False


# ---------------------------------------------------------------------------
# The three axes. Each returns
#   {axis, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of robustness-debt (the work-list); soft = score-only nudges.
# ---------------------------------------------------------------------------

def axis_simplicity(demo: dq.Demo) -> dict[str, Any]:
    """Few moving parts: one obvious command, a small surface to read."""
    defects: list[str] = []
    soft: list[str] = []
    score = 100.0

    # HARD: a required multi-step run with no single command. The headline result
    # should be one paste-able command, not a "first build X, then export Y, then
    # run Z" sequence — wrap it in a runner.
    if _MULTISTEP_RE.search(_readme_prose(demo)) and not demo.has_run_command:
        defects.append("multi-step run with no single command — the README sequences several "
                       "manual steps; wrap the headline run in one command")
        score -= 45

    # HARD: an oversized entry surface with no "start here" pointer. A big demo is
    # fine IF it names the small core to read first (a QUICKSTART file or heading).
    eloc = _entry_loc(demo)
    has_quickstart = bool(_QUICKSTART_RE.search(demo.readme)) or "QUICKSTART.md" in demo.files
    if eloc > ENTRY_LOC_BUDGET and not has_quickstart:
        defects.append(f"oversized entry surface ({eloc} lines) with no quickstart / 'start here' "
                       f"pointer — name the small core a reader should read first")
        score -= 35

    # SOFT: many prerequisites / many files.
    tools = _required_tools(demo)
    if len(tools) > MANY_PREREQS:
        soft.append(f"{len(tools)} prerequisites ({', '.join(tools)}) — more moving parts than a "
                    f"one-tool demo")
        score -= 12
    if len(demo.files) > MANY_FILES:
        soft.append(f"{len(demo.files)} files in the demo dir — a larger surface to skim")
        score -= 8

    detail = (f"entry-loc={eloc} · quickstart={has_quickstart} · prereqs={tools or '—'} · "
              f"files={len(demo.files)}")
    return {"axis": "simplicity", "score": dq._clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


def axis_speed(demo: dq.Demo) -> dict[str, Any]:
    """Fast feedback with no surprise waits."""
    defects: list[str] = []
    soft: list[str] = []
    score = 100.0

    # HARD: no stated expected runtime. A demo should tell you whether it is 2s or
    # 20min, so a runner knows whether to wait.
    has_runtime = bool(_RUNTIME_RE.search(_readme_prose(demo)) or
                       _RUNTIME_RE.search(demo.example_output))
    if not has_runtime:
        defects.append("no stated expected runtime — the README never says how long a run takes; "
                       "state it (e.g. 'runs in ~Ns', 'completes in seconds')")
        score -= 34

    # HARD: an unbounded polling loop — `until/while … sleep` with no deadline or
    # iteration cap. If the awaited condition never arrives the demo hangs forever.
    waits = _unbounded_wait_scripts(demo)
    if waits:
        defects.append(f"unbounded wait loop in {', '.join(f'`{w}`' for w in waits)} — a polling "
                       f"loop sleeps with no timeout / max-attempts; it can hang forever (bound it)")
        score -= 34

    # SOFT: builds the whole binary instead of `go run` (a leftover artifact + a
    # slower cold start), with no `go run` fast path anywhere.
    st = demo.script_text + "\n" + _readme_fence_text(demo)
    if _GO_BUILD_RE.search(st) and not _GO_RUN_RE.search(st):
        soft.append("builds the whole binary (`go build`) with no `go run` fast path — slower cold "
                    "start and a leftover artifact")
        score -= 12

    # SOFT: the default run path reaches the network / a model (slow first run),
    # with no offline tell.
    if _has_run_time_net_fetch(demo) and not re.search(
            r"(?i)(no model|no network|offline|zero dep|stdlib|standard library|no ollama)",
            _readme_prose(demo)):
        soft.append("default run path fetches over the network / needs a model, with no offline "
                    "fast path called out")
        score -= 12

    detail = (f"runtime-stated={has_runtime} · unbounded-waits={waits or '—'} · "
              f"go-run={bool(_GO_RUN_RE.search(st))}")
    return {"axis": "speed", "score": dq._clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


def axis_durability(demo: dq.Demo) -> dict[str, Any]:
    """Still works later, everywhere, every time."""
    defects: list[str] = []
    soft: list[str] = []
    score = 100.0

    # HARD: an unpinned run-time external dependency — a fetch of a mutable remote
    # (ollama pull <tag> / go get @latest / pip install w/o ==), which rots when the
    # remote moves. Scanned over scripts AND the README's command fences.
    floating = _scan_text_for_floating_deps(_shell_script_text(demo))
    floating += _scan_text_for_floating_deps(_readme_fence_text(demo))
    if floating:
        sample = floating[0][:60]
        defects.append(f"unpinned run-time dependency (`{sample}` …) — a fetch of a mutable remote; "
                       f"pin it (digest/version) or note the run is offline after first fetch")
        score -= 34

    # HARD: no stability / determinism guarantee. A reader cannot tell whether re-
    # running is safe or whether the output is stable across runs/time.
    has_stability = bool(_STABILITY_RE.search(_readme_prose(demo)))
    if not has_stability:
        defects.append("no stability / determinism guarantee — the README doesn't say whether a "
                       "re-run is repeatable (deterministic / byte-identical / pinned); state it")
        score -= 34

    # HARD: a shell entry without an error-stopping guard (`set -e`). A silent mid-
    # script failure otherwise prints a misleading 'success'.
    no_set_e = _shell_scripts_without_set_e(demo)
    if no_set_e:
        defects.append(f"shell entry without `set -e` in {', '.join(f'`{s}`' for s in no_set_e)} — "
                       f"a mid-script failure passes silently; add `set -euo pipefail`")
        score -= 30

    # SOFT: shell-only with no cross-platform note (Windows users are stuck).
    has_sh = any(n.lower().endswith((".sh", ".bash")) for n in demo.scripts)
    has_ps1 = any(n.lower().endswith(".ps1") for n in demo.scripts)
    if has_sh and not has_ps1 and not _XPLAT_RE.search(demo.readme):
        soft.append("shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a "
                    "Windows user can't tell how to run it")
        score -= 12

    detail = (f"floating-deps={len(floating)} · stability-stated={has_stability} · "
              f"set-e-missing={no_set_e or '—'}")
    return {"axis": "durability", "score": dq._clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


# ---------------------------------------------------------------------------
# Per-demo fold + grader
# ---------------------------------------------------------------------------

def score_demo(demo: dq.Demo) -> dict[str, Any]:
    if not demo.exists:
        return missing_demo_entry(demo.rel)
    axes = [axis_simplicity(demo), axis_speed(demo), axis_durability(demo)]
    by_name = {a["axis"]: a for a in axes}
    composite = sum(AXIS_WEIGHTS[name] * by_name[name]["score"] for name in AXIS_WEIGHTS)
    defects = [f"{a['axis']}: {d}" for a in axes for d in a["defects"]]
    soft = [f"{a['axis']}: {s}" for a in axes for s in a["soft"]]
    return {
        "path": demo.rel,
        "score": round(composite, 1),
        "grade": dq.grade_letter(composite),
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
        # An empty corpus is NOT a clean pass — a scorecard that finds nothing and
        # reports OK/exit-0 is a silent false-pass (wrong cwd or a moved corpus).
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "no_demos",
            "reason": "no demos discovered — run from repo ROOT (corpus comes from demo_quality_scorecard)",
            "next_action": "run from the repository root so examples/ and cmd/simpledemo resolve",
            "workspace": workspace, "corpus": {"n_demos": 0, "robustness_debt": 0}, "demos": demos,
        }
    scores = [d["score"] for d in demos]
    robustness_debt = sum(d["n_defects"] for d in demos)
    mean_score = round(sum(scores) / max(1, n), 1)
    grades = {g: 0 for g in "ABCDF"}
    for d in demos:
        grades[d["grade"]] = grades.get(d["grade"], 0) + 1
    worst = sorted(demos, key=lambda d: (d["score"], -d["n_defects"]))

    # Per-axis debt totals — the work-list rolls up by axis so a program can pick a
    # lever (e.g. "retire all unbounded-wait debt") instead of one defect at a time.
    axis_debt = {a: 0 for a in AXIS_WEIGHTS}
    for d in demos:
        for a, c in (d.get("axis_debt") or {}).items():
            axis_debt[a] = axis_debt.get(a, 0) + c

    corpus = {
        "n_demos": n,
        "mean_score": mean_score,
        "median_score": round(statistics.median(scores), 1) if scores else 0.0,
        "min_score": round(min(scores), 1) if scores else 0.0,
        "max_score": round(max(scores), 1) if scores else 0.0,
        "grade_distribution": grades,
        "robustness_debt": robustness_debt,
        "axis_debt": axis_debt,
        "worst": [{"path": d["path"], "score": d["score"], "grade": d["grade"],
                   "n_defects": d["n_defects"]} for d in worst],
    }

    if robustness_debt == 0:
        ok, verdict, finding = True, "OK", "demos_robust"
        reason = (f"demos robust: {n} demos, mean {mean_score}/100, zero robustness-debt — "
                  f"every demo is simple to run, fast with no surprise waits, and durable")
        next_action = "no required edit; re-run after the next demo change"
    else:
        ok, verdict, finding = False, "ACTION", "robustness_debt"
        worst_demo = worst[0]
        reason = (f"{robustness_debt} unit(s) of robustness-debt across {n} demos; mean "
                  f"{mean_score}/100; weakest: {worst_demo['path']} ({worst_demo['score']}/100, "
                  f"{worst_demo['n_defects']} defect(s))")
        next_action = ("retire robustness-debt worst-first (see corpus.worst + demo.defects): bound "
                       "unbounded waits, pin/offline run-time deps, state runtime + a stability "
                       "guarantee; re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "demos": demos,
    }


# ---------------------------------------------------------------------------
# Wiring + CLI
# ---------------------------------------------------------------------------

def collect(workspace: Path, *, demo_rels: list[str] | None = None) -> dict[str, Any]:
    root = workspace.resolve()
    rels = demo_rels if demo_rels is not None else dq.discover_demos(root)
    demos = [score_demo(dq.load_demo(root, rel)) for rel in rels]
    return build_payload(workspace=str(root), demos=demos)


def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    ad = c.get("axis_debt", {})
    lines = [
        f"demo-robustness scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"corpus: {c.get('n_demos', 0)} demos · mean {c.get('mean_score', 0)}/100 "
         f"· min {c.get('min_score', 0)} · ROBUSTNESS-DEBT {c.get('robustness_debt', 0)}"),
        ("axis-debt: " + " ".join(f"{a}:{ad.get(a, 0)}" for a in AXIS_WEIGHTS)),
        ("grades: " + " ".join(f"{g}:{c.get('grade_distribution', {}).get(g, 0)}"
                               for g in "ABCDF")),
        f"next: {payload.get('next_action')}",
        "",
        "per-demo (worst first):",
        f"  {'score':>5} {'gr':>2} {'def':>3}  sim spd dur  demo",
    ]
    for d in sorted(payload.get("demos", []), key=lambda x: (x["score"], -x["n_defects"])):
        a = d.get("axes", {})
        lines.append(
            f"  {d['score']:>5} {d['grade']:>2} {d['n_defects']:>3}  "
            f"{a.get('simplicity','-'):>3} {a.get('speed','-'):>3} "
            f"{a.get('durability','-'):>3}  {d['path']}")
    lines.append("")
    lines.append("robustness-debt work-list:")
    any_defect = False
    for d in sorted(payload.get("demos", []), key=lambda x: -x["n_defects"]):
        if not d["defects"]:
            continue
        any_defect = True
        lines.append(f"  {d['path']} ({d['n_defects']}):")
        for it in d["defects"]:
            lines.append(f"      - {it}")
    if not any_defect:
        lines.append("  (none — demos robust)")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    """The committed DEMO-ROBUSTNESS-SCORECARD.md body — a human-facing snapshot."""
    c = payload.get("corpus") or {}
    gd = c.get("grade_distribution", {})
    ad = c.get("axis_debt", {})
    out: list[str] = ["# Demo-robustness scorecard", ""]
    if stamp:
        out.append(f"<!-- demo-robustness-scorecard: {stamp} · process: tools/demo_robustness_scorecard.py -->")
        out.append("")
    out.append("> Regenerate: `python tools/demo_robustness_scorecard.py --markdown --stamp DATE "
               "> docs/DEMO-ROBUSTNESS-SCORECARD.md`")
    out.append("")
    out.append("> The measuring stick for **demos that stay simple, fast, and durable**: will it "
               "still run next month, on a fresh box, in one obvious command, in seconds — without "
               "surprises? Three deterministic axes (simplicity · speed · durability), folded into "
               "a **robustness-score** (0–100, A–F) and a **robustness-debt** integer (the count of "
               "concrete, re-derivable robustness defects). The sibling `demo_quality_scorecard.py` "
               "asks *can a skeptic trust the claim*; this one asks *will it keep running*. Both "
               "score the same corpus. Every number below is re-derived from disk by "
               "`tools/demo_robustness_scorecard.py` — no hand-entry. Drive robustness-debt down to "
               "make \"more robust demos\" provable.")
    out.append("")
    out.append("## Corpus")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Demos scored | {c.get('n_demos', 0)} |")
    out.append(f"| **Robustness-debt (total defects)** | **{c.get('robustness_debt', 0)}** |")
    out.append(f"| Axis-debt | simplicity:{ad.get('simplicity',0)} · speed:{ad.get('speed',0)} · durability:{ad.get('durability',0)} |")
    out.append(f"| Mean score | {c.get('mean_score', 0)}/100 |")
    out.append(f"| Median / min / max | {c.get('median_score', 0)} / {c.get('min_score', 0)} / {c.get('max_score', 0)} |")
    out.append(f"| Grade distribution | A:{gd.get('A',0)} B:{gd.get('B',0)} C:{gd.get('C',0)} D:{gd.get('D',0)} F:{gd.get('F',0)} |")
    out.append("")
    out.append("## Per-demo scores")
    out.append("")
    out.append("Three axes, each 0–100 (simplicity · speed · durability), weighted into a score and "
               "an A–F grade. `def` = units of robustness-debt.")
    out.append("")
    out.append("| Score | Grade | Debt | simplicity | speed | durability | Demo |")
    out.append("|---:|:--:|:--:|:--:|:--:|:--:|---|")
    for d in sorted(payload.get("demos", []), key=lambda x: (x["score"], -x["n_defects"])):
        a = d.get("axes", {})
        out.append(
            f"| {d['score']} | {d['grade']} | {d['n_defects']} | "
            f"{a.get('simplicity','-')} | {a.get('speed','-')} | {a.get('durability','-')} | "
            f"`{d['path']}` |")
    out.append("")
    out.append("## Robustness-debt work-list")
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
        out.append("No robustness-debt: every demo is simple, fast, and durable. 🎉")
        out.append("")
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
    ap = argparse.ArgumentParser(description="Demo-robustness scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true",
                    help="emit the DEMO-ROBUSTNESS-SCORECARD.md body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    args = ap.parse_args(argv)

    # Demos carry Unicode (✓, →, ·, —, 🤖); force UTF-8 stdout so a Windows cp1252
    # console can't crash the scorer on a glyph.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else dq.repo_root()
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
