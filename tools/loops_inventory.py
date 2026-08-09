#!/usr/bin/env python3
r"""loops_inventory — the one surface that answers "what recurring loops run in this
repo, how often does each fire, and where does each one report?".

The fleet's recurring work is declared across TWO surfaces that never meet:

  * OS Scheduled Tasks — one per ``tools/register_*.ps1`` (operator-local, Windows):
    each installs a ``Fleet*`` / ``Fak*`` task that ticks a python fold every N minutes
    (the dispatch loop, the watchdogs, the slack-status card, the doc renderers).
  * GitHub Actions — one per ``.github/workflows/*.yml`` carrying a ``schedule:`` cron
    (the daily feeds, the weekly cadence, the signal gates that post to Slack or commit
    a doc).

There is no single place an operator can open to see the WHOLE recurring surface: which
loops exist, how frequently each fires, and whether each reports to Slack, commits a repo
doc, or only writes operator-local telemetry. A hand-maintained list rots the moment a
``register_*.ps1`` or a workflow is added or removed. This tool DISCOVERS the loops from
their declaration files — so the inventory is always as current as the tree — and folds
them into three surfaces:

    python tools/loops_inventory.py --md docs/loops-inventory.md   # report to the repo
    python tools/loops_inventory.py --slack                        # report to Slack
    python tools/loops_inventory.py --json                         # machine-readable
    python tools/loops_inventory.py --ledger .fak/nightrun/loops-history.jsonl

It is a PURE READ-ONLY FOLD: it parses declaration files, renders text, posts an opt-in
Slack line, and appends its own trend ledger. It launches no loop, installs no task, and
git-commits NOTHING — an operator commits the rendered doc by path when ready, the same
contract as ``bench_plan`` / ``dispatch_status``. The discovery walkers (``discover``)
touch the disk; every parser and renderer below them is pure (text in, record out) so the
fold is table-tested with no clock and no disk.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Sibling tools/ modules (slack_post, fleet_trend) — resolve them whether the tool is run
# from tools/ or as `python tools/loops_inventory.py` from the workspace root (the shape a
# scheduled task invokes it with). Same idiom fleet_slack_status uses.
sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = "loops-inventory/1"

# The committed doc the "report to repo" half renders, workspace-relative. A public,
# non-sensitive list of loop names + cadences — it belongs in the committed tree (like
# docs/bench-plan.md), NOT under the gitignored state root.
DEFAULT_DOC = os.path.join("docs", "loops-inventory.md")

# The trend ledger, under the gitignored state root so a scheduled tick never dirties the
# shared tree (same home as fleet_trend's ledger).
DEFAULT_LEDGER = os.path.join(".fak", "nightrun", "loops-history.jsonl")

# Ring size for the trend ledger — a daily tick stays well under this for years; the cap
# only guards against unbounded growth.
DEFAULT_CAP = 500

# The Slack surface a loop's declaration is inferred to report to, in priority order. The
# first signal that matches wins, so a slack-posting doc renderer tags as "slack" (its
# louder sink) rather than "repo-doc".
SINK_SLACK = "slack"
SINK_REPO = "repo-doc"
SINK_LOCAL = "local-doc"
SINK_GITHUB = "github"
SINK_ACTION = "action"
SINK_UNKNOWN = "?"


# ----- cadence normalization (pure) ------------------------------------------

def _slash_every(field: str) -> int | None:
    """N for a ``*/N`` cron field, else None."""
    m = re.fullmatch(r"\*/(\d+)", field)
    return int(m.group(1)) if m else None


def _isint(field: str) -> bool:
    return field.isdigit()


def cron_cadence(expr: str) -> tuple[str, int | None]:
    """Fold a 5-field cron expression into a ``(label, minutes)`` pair.

    ``label`` is the human cadence for the inventory ("daily 07:41 UTC", "every 6h");
    ``minutes`` is an APPROXIMATE fire period used only to sort loops most-frequent-first
    (None when the shape isn't one of the recognized periodic forms, so it sorts last).
    Recognizes the forms the repo's workflows actually use: ``*/N`` minute/hour, a plain
    daily ``MM HH * * *``, and a day-of-week-qualified weekly/weekday schedule.
    """
    parts = expr.split()
    if len(parts) != 5:
        return expr, None
    minute, hour, dom, month, dow = parts

    n = _slash_every(minute)
    if n and hour == "*" and dom == "*" and dow == "*":
        return f"every {n} min", n

    h = _slash_every(hour)
    if h and dom == "*" and dow == "*":
        return f"every {h}h", h * 60

    if hour == "*" and dom == "*" and dow == "*" and minute != "*":
        return "hourly", 60

    if _isint(minute) and _isint(hour) and dom == "*" and dow == "*":
        return f"daily {int(hour):02d}:{int(minute):02d} UTC", 1440

    if _isint(minute) and _isint(hour) and dom == "*" and dow != "*":
        at = f"{int(hour):02d}:{int(minute):02d} UTC"
        if dow in ("1-5", "1,2,3,4,5"):
            return f"weekdays {at}", 2016  # ~5 fires/week, between daily and weekly
        return f"weekly {at}", 10080

    return expr, None


def humanize_minutes(m: int | None) -> str:
    """Human cadence for a scheduled task's ``$EveryMinutes`` interval."""
    if m is None:
        return "?"
    if m < 60:
        return f"every {m} min"
    if m % 1440 == 0:
        days = m // 1440
        return "daily" if days == 1 else f"every {days}d"
    if m % 60 == 0:
        return f"every {m // 60}h"
    return f"every {m} min"


# ----- register_*.ps1 parsing (pure) -----------------------------------------

_HEADER_BLOCK = re.compile(r"<#(.*?)#>", re.DOTALL)
_TASKNAME = re.compile(r"\$TaskName\s*=\s*['\"]([^'\"]+)['\"]")
_TICK_TOOL = re.compile(r"tools[\\/]([A-Za-z0-9_]+\.(?:py|ps1))")

# A published task name has to be a STATIC literal. PowerShell interpolates a double-quoted
# string, so a captured body carrying ``$`` or a backtick (``"Fleet$($Suffix)Doctor"``)
# names no task that exists on the host; resolving it would mean RUNNING the registrar,
# which this read-only fold never does. Refuse the guess and publish the marker instead —
# a task name nothing answers to is worse in a ground-truth doc than an admitted gap. The
# installer end of the same value was pinned to a literal by #5409; this is the reader end.
_INTERPOLATED = re.compile(r"[$`]")
NAME_UNRESOLVED = "(unresolved: interpolated name)"


def _comment_start(line: str) -> int | None:
    """Index of the ``#`` that opens a trailing comment on ``line``, else None. A ``#``
    inside a quoted string is a character, not a comment, so only one standing at even
    quote parity counts."""
    single = double = 0
    for i, ch in enumerate(line):
        if ch == "'" and double % 2 == 0:
            single += 1
        elif ch == '"' and single % 2 == 0:
            double += 1
        elif ch == "#" and single % 2 == 0 and double % 2 == 0:
            return i
    return None


def _strip_ps_comments(text: str) -> str:
    """Blank every PowerShell comment, preserving line and column shape.

    This is what anchors the task-name read to the script's real param/assignment site: a
    ``$TaskName = '...'`` written in the ``<# ... #>`` usage header or in a ``#`` comment is
    documentation of an override, not the default the task installs under, and must not win
    the first-match race against it.
    """
    blanked = _HEADER_BLOCK.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    out: list[str] = []
    for line in blanked.split("\n"):
        cut = _comment_start(line)
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)

# Cadence is declared several ways across the registrars; resolve them all from the actual
# trigger so a loop's real fire period is read, not a hard-coded guess:
#   -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes)   ($EveryMinutes/$EveryMin/literal)
#   -RepetitionInterval (New-TimeSpan -Hours   $EveryHours)     (worktree-doctor, stale-work)
#   New-ScheduledTaskTrigger -Daily -At $At                     (the daily audit/scout tasks)
#   schtasks /Create ... /SC MINUTE /MO 5                       (supervisor/dos watchdogs)
_REP_INTERVAL = re.compile(
    r"-RepetitionInterval\s*\(\s*New-TimeSpan\s+-(Minutes|Hours)\s+(\$?\w+)\s*\)")
_DAILY_TRIGGER = re.compile(
    r"New-ScheduledTaskTrigger\s+-Daily\s+-At\s+(\$?\w+|'[^']*'|\"[^\"]*\")")
_SCHTASKS = re.compile(r"/SC\s+(MINUTE|HOURLY|DAILY|WEEKLY)(?:\s+/MO\s+(\d+))?", re.IGNORECASE)
_SCHTASKS_UNIT_MIN = {"MINUTE": 1, "HOURLY": 60, "DAILY": 1440, "WEEKLY": 10080}


def _resolve_int_operand(text: str, operand: str) -> int | None:
    """An interval operand is either a literal (``10``) or a ``$Var`` whose ``param``
    default sets it (``$EveryMinutes = 30``). Resolve to the int, or None if unknown."""
    if operand.isdigit():
        return int(operand)
    if operand.startswith("$"):
        m = re.search(re.escape(operand) + r"\s*=\s*(\d+)", text)
        if m:
            return int(m.group(1))
    return None


def _resolve_str_operand(text: str, operand: str) -> str:
    """A ``-At`` time is a quoted literal (``'09:50'``) or a ``$Var`` whose param default
    holds it. Return the resolved string, or "" if unknown."""
    if operand[:1] in ("'", '"'):
        return operand.strip("'\"")
    if operand.startswith("$"):
        m = re.search(re.escape(operand) + r"\s*=\s*['\"]([^'\"]+)['\"]", text)
        if m:
            return m.group(1)
    return ""


def extract_cadence(text: str) -> tuple[int | None, str]:
    """Fold a registrar's trigger into ``(minutes, label)``. ``minutes`` is the effective
    fire period for sorting (None => unknown, sorts last); ``label`` is the human cadence,
    preserving a daily run's ``HH:MM`` and a combined daily+repetition schedule."""
    rep_minutes: int | None = None
    m = _REP_INTERVAL.search(text)
    if m:
        val = _resolve_int_operand(text, m.group(2))
        if val:  # 0 ("daily only") counts as no repetition
            rep_minutes = val if m.group(1) == "Minutes" else val * 60

    daily = _DAILY_TRIGGER.search(text)
    daily_at = _resolve_str_operand(text, daily.group(1)) if daily else ""

    sch_minutes: int | None = None
    ms = _SCHTASKS.search(text)
    if ms:
        mo = int(ms.group(2)) if ms.group(2) else 1
        sch_minutes = _SCHTASKS_UNIT_MIN[ms.group(1).upper()] * mo

    if daily and rep_minutes:
        at = f" {daily_at}" if daily_at else ""
        return rep_minutes, f"{humanize_minutes(rep_minutes)} (+ daily{at})"
    if rep_minutes is not None:
        return rep_minutes, humanize_minutes(rep_minutes)
    if sch_minutes is not None:
        return sch_minutes, humanize_minutes(sch_minutes)
    if daily:
        return 1440, f"daily {daily_at}".rstrip()

    # Fallback: a param default with no matching trigger (an odd script or a fixture).
    pm = re.search(r"\$EveryMin(?:utes)?\s*=\s*(\d+)", text)
    if pm:
        return int(pm.group(1)), humanize_minutes(int(pm.group(1)))
    ph = re.search(r"\$EveryHours\s*=\s*(\d+)", text)
    if ph:
        return int(ph.group(1)) * 60, humanize_minutes(int(ph.group(1)) * 60)
    return None, "?"

# Wrapper/plumbing scripts a register_*.ps1 references that are NOT the loop's real work —
# skip them when picking the tick tool so we report the fold, not the scheduler shim.
_WRAPPER_TOOLS = {"fak_loop_task.ps1"}


def _first_sentence(text: str) -> str:
    """The first sentence of a blob, whitespace-collapsed, trimmed to a sane length."""
    flat = " ".join(text.split())
    dot = flat.find(". ")
    if dot != -1:
        flat = flat[:dot]
    flat = flat.rstrip(". ")
    return flat[:200]


def _purpose_from_header(header: str, filename: str) -> str:
    """Pull a one-line purpose from a register_*.ps1 ``<# ... #>`` header. The header's
    lead line is ``<filename> -- <purpose ...>``; take the text after the first ``--``
    (or ``—``), across wrapped lines, up to the first sentence end."""
    body = header.strip()
    # Drop a leading "filename.ps1" token if present so the purpose isn't prefixed by it.
    for sep in (" -- ", " — ", " – "):
        idx = body.find(sep)
        if idx != -1 and idx < 200:
            return _first_sentence(body[idx + len(sep):])
    return _first_sentence(body)


def _infer_sink(text: str, tick_tool: str, name: str) -> str:
    """Best-effort tag for where a loop reports, from signals in its declaration, ranked
    strongest-first so a weak prose mention never outvotes a structural signal. Clearly
    labeled 'inferred' downstream; the Source column is the ground truth."""
    hay = (text + " " + tick_tool + " " + name).lower()
    if "slack" in hay:
        return SINK_SLACK
    # A doc renderer — keyed on an ACTUAL doc-writing token (--md / $DocPath), not a stray
    # prose mention of a path. Local when the target is gitignored/under a dot-dir, else
    # a committed repo doc.
    writes_doc = ("--md" in hay or "docpath" in hay
                  or re.search(r"-md\b", hay) is not None)
    if writes_doc:
        if ("gitignored" in hay or ".dispatch-runs" in hay
                or re.search(r"\.fak[\\/]", hay)
                or re.search(r"-docpath\s+['\"]?\.", hay)):
            return SINK_LOCAL
        return SINK_REPO
    # Reports to GitHub — files or closes issues. Keyed on the write verbs (create/reopen/
    # --file-issues) so the *dispatch* loop, which only reads the backlog, isn't caught.
    if any(k in hay for k in ("gh issue create", "gh issue reopen", "--file-issues")):
        return SINK_GITHUB
    # A watchdog/guard/reaper that acts on the host rather than reporting anywhere.
    if any(k in hay for k in ("watchdog", "reaper", "guard", "heal", "respawn",
                              "sweep", "doctor", "reap")):
        return SINK_ACTION
    if "gitignored" in hay or "operator-local" in hay:
        return SINK_LOCAL
    return SINK_UNKNOWN


def parse_register_ps1(text: str, filename: str) -> dict[str, Any] | None:
    """Parse one ``register_*.ps1`` into a loop record, or None when it declares no task.

    Pure: takes the file text + its basename, returns
    ``{surface, name, cadence, cadence_minutes, sink, tick, purpose, source, name_unresolved}``.

    The name is read from the COMMENT-STRIPPED text (a header-block mention cannot shadow
    the param default) and is refused when it is not a static literal — the record still
    carries the loop, with ``name`` set to :data:`NAME_UNRESOLVED` and ``name_unresolved``
    true, so the doc, the Slack card and the exit code all show which registrar drifted.
    """
    mname = _TASKNAME.search(_strip_ps_comments(text))
    if not mname:
        return None
    name = mname.group(1)
    name_unresolved = _INTERPOLATED.search(name) is not None
    if name_unresolved:
        name = NAME_UNRESOLVED

    minutes, cadence = extract_cadence(text)

    tick = ""
    for m in _TICK_TOOL.finditer(text):
        cand = m.group(1)
        if cand not in _WRAPPER_TOOLS and cand != filename:
            tick = cand
            break

    hb = _HEADER_BLOCK.search(text)
    header = hb.group(1) if hb else ""
    purpose = _purpose_from_header(header, filename)
    sink = _infer_sink(text, tick, name)

    return {
        "surface": "scheduled-task",
        "name": name,
        "cadence": cadence,
        "cadence_minutes": minutes,
        "sink": sink,
        "tick": ("tools/" + tick) if tick else "",
        "purpose": purpose,
        "source": "tools/" + filename,
        "name_unresolved": name_unresolved,
    }


# ----- workflow yml parsing (pure) -------------------------------------------

_WF_NAME = re.compile(r"^name:\s*(.+?)\s*$", re.MULTILINE)
_WF_CRON = re.compile(r"-\s*cron:\s*['\"]([^'\"]+)['\"]")


def parse_workflow_yml(text: str, filename: str) -> dict[str, Any] | None:
    """Parse one ``.github/workflows/*.yml`` into a loop record, or None when it carries
    no ``schedule:`` cron (a push/PR gate is not a recurring loop). Pure.

    When a workflow declares several crons, the most frequent one sets its cadence (that
    is how often it actually fires); all cron labels are kept in ``crons``.
    """
    crons = _WF_CRON.findall(text)
    if not crons:
        return None

    folded = [cron_cadence(c) for c in crons]
    # Most-frequent cron wins the headline cadence; unknown-period ones sort last.
    best = min(folded, key=lambda lm: (lm[1] is None, lm[1] if lm[1] is not None else 0))
    label, minutes = best

    mname = _WF_NAME.search(text)
    name = mname.group(1).strip().strip("'\"") if mname else filename.rsplit(".", 1)[0]

    hay = (name + " " + filename).lower()
    if any(k in hay for k in ("slack", "feed", "beat", "signal", "notify", "watchdog")):
        sink = SINK_SLACK
    elif "board-sync" in hay or hay.endswith("-doc") or "doc" in hay:
        sink = SINK_REPO
    else:
        sink = "ci"

    return {
        "surface": "github-actions",
        "name": name,
        "cadence": label,
        "cadence_minutes": minutes,
        "sink": sink,
        "crons": crons,
        "tick": "",
        "purpose": name,
        "source": ".github/workflows/" + filename,
    }


# ----- discovery (impure: reads the tree) ------------------------------------

def discover(workspace: str) -> dict[str, Any]:
    """Walk the two declaration surfaces under ``workspace`` and fold them into the full
    inventory. The ONLY disk-touching function; everything it calls is pure."""
    root = Path(workspace)
    loops: list[dict[str, Any]] = []

    reg_dir = root / "tools"
    if reg_dir.is_dir():
        for p in sorted(reg_dir.glob("register_*.ps1")):
            try:
                rec = parse_register_ps1(p.read_text(encoding="utf-8", errors="replace"), p.name)
            except OSError:
                continue
            if rec:
                loops.append(rec)

    wf_dir = root / ".github" / "workflows"
    if wf_dir.is_dir():
        for p in sorted(list(wf_dir.glob("*.yml")) + list(wf_dir.glob("*.yaml"))):
            try:
                rec = parse_workflow_yml(p.read_text(encoding="utf-8", errors="replace"), p.name)
            except OSError:
                continue
            if rec:
                loops.append(rec)

    loops.sort(key=_sort_key)
    return {"schema": SCHEMA, "loops": loops, "summary": summarize(loops)}


def _sort_key(rec: dict[str, Any]) -> tuple[Any, ...]:
    """Sort surface-grouped, then most-frequent-first (unknown cadence last), then name."""
    surface_order = 0 if rec["surface"] == "scheduled-task" else 1
    m = rec.get("cadence_minutes")
    return (surface_order, m is None, m if m is not None else 0, rec["name"].lower())


def summarize(loops: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll the loop list into the headline counts the Slack card + doc lead with."""
    tasks = [lp for lp in loops if lp["surface"] == "scheduled-task"]
    workflows = [lp for lp in loops if lp["surface"] == "github-actions"]
    by_sink: dict[str, int] = {}
    for lp in loops:
        by_sink[lp["sink"]] = by_sink.get(lp["sink"], 0) + 1
    return {
        "total": len(loops),
        "tasks": len(tasks),
        "workflows": len(workflows),
        "slack": by_sink.get(SINK_SLACK, 0),
        "repo": by_sink.get(SINK_REPO, 0),
        "local": by_sink.get(SINK_LOCAL, 0),
        "github": by_sink.get(SINK_GITHUB, 0),
        "by_sink": by_sink,
        # Loops whose declared task name could not be read as a literal — a refusal count,
        # carried so every sink (doc, Slack, --json, exit code) can surface it.
        "unresolved": sum(1 for lp in loops if lp.get("name_unresolved")),
    }


# ----- renderers (pure) ------------------------------------------------------

def _fastest(loops: list[dict[str, Any]], n: int = 4) -> list[dict[str, Any]]:
    ranked = [lp for lp in loops if lp.get("cadence_minutes") is not None]
    ranked.sort(key=lambda lp: lp["cadence_minutes"])
    return ranked[:n]


def render_md(inv: dict[str, Any], now: str) -> str:
    """Render the committed markdown inventory: a headline, then one table per surface,
    each most-frequent-first. Stable output for a given inventory + ``now``."""
    s = inv["summary"]
    loops = inv["loops"]
    out: list[str] = []
    out.append("# Recurring loops inventory")
    out.append("")
    out.append(f"_Generated {now} by `tools/loops_inventory.py` — a read-only fold over "
               "`tools/register_*.ps1` (OS Scheduled Tasks) and `.github/workflows/*.yml` "
               "(cron). Do not hand-edit; re-run the tool._")
    out.append("")
    out.append(f"**{s['total']} loops**: {s['tasks']} scheduled tasks · "
               f"{s['workflows']} cron workflows. "
               f"Reporting to (inferred): {s['slack']} → Slack, {s['github']} → GitHub "
               f"issues, {s['repo']} → repo doc, {s['local']} → operator-local.")
    out.append("")

    if s.get("unresolved"):
        out.append(f"**Refused {s['unresolved']} task name(s).** A `$TaskName` default "
                   "built from an interpolated string names no task that exists on the "
                   f"host, so it is published as `{NAME_UNRESOLVED}` rather than guessed. "
                   "Pin the default in the Source script to a quoted literal.")
        out.append("")

    tasks = [lp for lp in loops if lp["surface"] == "scheduled-task"]
    if tasks:
        out.append("## OS Scheduled Tasks (operator-local, Windows)")
        out.append("")
        out.append("| Task | Cadence | Reports to | Runs | Purpose | Source |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        for lp in tasks:
            out.append(f"| {lp['name']} | {lp['cadence']} | {lp['sink']} | "
                       f"{_md_cell(lp['tick'])} | {_md_cell(lp['purpose'])} | "
                       f"`{lp['source']}` |")
        out.append("")

    workflows = [lp for lp in loops if lp["surface"] == "github-actions"]
    if workflows:
        out.append("## GitHub Actions (cron)")
        out.append("")
        out.append("| Workflow | Cadence | Reports to | Purpose | Source |")
        out.append("| --- | --- | --- | --- | --- |")
        for lp in workflows:
            out.append(f"| {lp['name']} | {lp['cadence']} | {lp['sink']} | "
                       f"{_md_cell(lp['purpose'])} | `{lp['source']}` |")
        out.append("")

    out.append("_\"Reports to\" is inferred from each loop's declaration text; the Source "
               "column is the ground truth._")
    return "\n".join(out) + "\n"


def _md_cell(text: str) -> str:
    """Escape a value for a markdown table cell (pipes would break the column)."""
    return (text or "—").replace("|", "\\|").replace("\n", " ")


def render_slack(inv: dict[str, Any], trend: str = "") -> str:
    """A compact rollup for the channel: totals, sink split, the fastest few loops, and
    an optional trend line. Small enough to read in a phone push."""
    s = inv["summary"]
    lines = [f"🔁 *Recurring loops:* {s['total']} total "
             f"({s['tasks']} scheduled tasks · {s['workflows']} cron workflows)"]
    lines.append(f"report → slack {s['slack']} · github {s['github']} · "
                 f"repo-doc {s['repo']} · local {s['local']}")
    if s.get("unresolved"):
        lines.append(f"refused {s['unresolved']} task name(s): an interpolated $TaskName "
                     "names no real task — pin the default to a quoted literal")
    fast = _fastest(inv["loops"])
    if fast:
        parts = " · ".join(f"{lp['name']} {lp['cadence']}" for lp in fast)
        lines.append("fastest: " + parts)
    if trend:
        lines.append(trend)
    return "\n".join(lines)


# ----- trend ledger (append-only ring; own metrics) --------------------------

# The load-bearing scalars a loops-inventory trend is about — "is the recurring surface
# growing or shrinking, and is more of it wired to Slack/the repo over time".
TREND_METRICS: list[tuple[str, str]] = [
    ("total", "total"),
    ("tasks", "tasks"),
    ("workflows", "workflows"),
    ("slack", "slack"),
    ("repo", "repo-doc"),
]


def trend_row(summary: dict[str, Any], now: str) -> dict[str, Any]:
    row: dict[str, Any] = {"ts": now}
    for k, _ in TREND_METRICS:
        row[k] = int(summary.get(k, 0) or 0)
    return row


def _read_rows(path: str) -> list[dict[str, Any]]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue  # tolerate a torn line rather than brick the history
        if isinstance(row, dict):
            out.append(row)
    return out


def trend_append(path: str, row: dict[str, Any], *, cap: int = DEFAULT_CAP) -> list[dict[str, Any]]:
    """Append ``row`` to the JSONL ledger, trim to the last ``cap`` rows, return the tail.
    Best-effort ring, same idiom as fleet_trend.append; the directory is created if absent."""
    rows = _read_rows(path)
    rows.append(row)
    if cap > 0 and len(rows) > cap:
        rows = rows[-cap:]
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    return rows


def trend_line(rows: list[dict[str, Any]]) -> str:
    """One compact trend line: per metric ``label first→last spark (Δ over N)``. Empty on
    a first-ever tick so the caller drops the line entirely. Reuses fleet_trend's sparkline."""
    if len(rows) < 2:
        return ""
    try:
        import fleet_trend  # sibling; reuse the sparkline renderer
        spark = fleet_trend.spark
    except Exception:  # noqa: BLE001 — a missing sibling must not break the tick
        spark = None
    parts: list[str] = []
    for key, label in TREND_METRICS:
        series = [int(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        # Show only metrics that actually moved, so the line stays about change, not a wall
        # of unchanged counts on every quiet tick.
        if len(series) < 2 or series[0] == series[-1]:
            continue
        delta = series[-1] - series[0]
        sign = "+" if delta > 0 else ""
        sp = (" " + spark(series)) if spark else ""
        parts.append(f"{label} {series[0]}→{series[-1]}{sp} ({sign}{delta} over {len(series)})")
    return "trend: " + " · ".join(parts) if parts else ""


# ----- CLI -------------------------------------------------------------------

def _iso_now() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Inventory the repo's recurring loops (OS Scheduled Tasks + cron "
                    "workflows) and report to the repo doc, Slack, and a trend ledger.")
    ap.add_argument("--workspace", default=".", help="repo root to inventory")
    ap.add_argument("--md", metavar="PATH", nargs="?", const=DEFAULT_DOC, default="",
                    help=f"render the markdown inventory to PATH (default {DEFAULT_DOC})")
    ap.add_argument("--json", action="store_true", help="emit the machine-readable inventory")
    ap.add_argument("--slack", action="store_true", help="post the rollup via slack_post")
    ap.add_argument("--channel", default="", help="Slack channel id (else resolved from env)")
    ap.add_argument("--dry-run", action="store_true", help="with --slack, resolve + show, send nothing")
    ap.add_argument("--ledger", metavar="PATH", nargs="?", const=DEFAULT_LEDGER, default="",
                    help=f"append a trend row to PATH (default {DEFAULT_LEDGER})")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP, help="max ledger rows (ring size)")
    ap.add_argument("--now", default="", help="override the timestamp (tests)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

    now = args.now or _iso_now()
    inv = discover(args.workspace)
    s = inv["summary"]

    # Trend: append first (if asked) so the rendered doc/slack can carry the fresh line.
    trend = ""
    if args.ledger:
        rows = trend_append(args.ledger, trend_row(s, now), cap=args.cap)
        trend = trend_line(rows)

    if args.md:
        doc_path = args.md if os.path.isabs(args.md) else os.path.join(args.workspace, args.md)
        p = Path(doc_path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_md(inv, now), encoding="utf-8")

    slack_result = None
    if args.slack:
        try:
            import slack_post  # sibling module in tools/
            slack_result = slack_post.send(render_slack(inv, trend),
                                           channel=args.channel, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001 — a slack failure is a verdict, not a crash
            slack_result = {"posted": False, "error": f"slack_post: {exc}"}

    if args.json:
        payload = dict(inv)
        if slack_result is not None:
            payload["slack"] = slack_result
        if trend:
            payload["trend"] = trend
        print(json.dumps(payload, indent=2))
    else:
        print(f"loops_inventory: {s['total']} loops "
              f"({s['tasks']} scheduled tasks · {s['workflows']} cron workflows) — "
              f"slack {s['slack']} · github {s['github']} · repo-doc {s['repo']} · "
              f"local {s['local']}")
        if s.get("unresolved"):
            print(f"  refused {s['unresolved']} task name(s): an interpolated $TaskName "
                  f"names no real task — pin the default to a quoted literal")
        if args.md:
            print(f"  wrote {args.md}")
        if trend:
            print("  " + trend)
        if slack_result is not None:
            if slack_result.get("posted"):
                print(f"  slack: posted to {slack_result.get('channel')}")
            elif slack_result.get("dry_run"):
                print(f"  slack (dry-run): would post to "
                      f"{slack_result.get('channel') or '(unset)'}")
            else:
                print(f"  slack: {slack_result.get('skipped') or slack_result.get('error')}")

    # Exit non-zero when a LIVE post was asked for but didn't land, so the scheduled task's
    # LastTaskResult surfaces the misconfiguration (a dry-run posting nothing is expected).
    if args.slack and not args.dry_run and slack_result is not None \
            and not slack_result.get("posted"):
        return 1
    # Same contract for a registrar whose task name is not a static literal: that is a
    # declaration bug in the tree, not a transient hiccup, so LastTaskResult must show it.
    # The doc is still rendered (carrying the marker) — this reports, it does not abort.
    if s.get("unresolved"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
