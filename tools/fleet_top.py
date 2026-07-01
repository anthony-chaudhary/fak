#!/usr/bin/env python3
r"""fleet_top — the live session/account watchdog, the session-health peer of `dos top`.

`dos top` answers "which dos.toml lane is held right now?" (the collision/lease
plane). It says nothing about the Claude Code workers themselves. This view answers
the other half an operator needs at a glance: "which sessions stopped and why, which
accounts can I actually resume on right now, and what needs me?" — the session-health
plane that until now only `fleet_status.ps1` showed, Windows-only and one-shot.

It is built to the SAME contract as `dos top` on purpose, so the two read like one
tool side by side: a boxed live-refreshing header, chip glyphs, and `--once` / `--json`
/ `--interval`. It is cross-platform and dependency-free — the signal layer is the
existing `fleet_sessions.py json` (transcript-derived dispositions + account policy),
so this module only folds, ranks, and renders; it invents no new state.

The thing the old card buried and this surfaces: ATTENTION — the ranked "what needs
me now" (a dead autonomous session on an account that is actually available, with the
exact account-correct resume command; an account that needs an interactive /login; a
throttle that means a resume would instantly re-die).

Usage:
  python tools/fleet_top.py                 # live, refresh every 5s (Ctrl-C to quit)
  python tools/fleet_top.py --once          # render one frame and exit (CI / pipe)
  python tools/fleet_top.py --json          # machine snapshot and exit
  python tools/fleet_top.py --interval 10   # live, 10s cadence
  python tools/fleet_top.py --window 24     # widen the session lookback window
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_version  # noqa: E402

SCHEMA = "fleet-top/1"

# Category render order + chip. Mirrors fleet_sessions' INFRA/AGENT/USER/HANGING/LIVE
# buckets; LIVE and the two things that block throughput (INFRA, HANGING) lead.
CATEGORY_ORDER = ["LIVE", "INFRA", "HANGING", "AGENT", "USER"]
CATEGORY_CHIP = {
    "LIVE": "green",
    "INFRA": "red",
    "HANGING": "yellow",
    "AGENT": "white",
    "USER": "white",
}
# disposition -> human cause, mirroring fleet_sessions.CATEGORY's second element so the
# per-category breakdown reads the same as `fleet_sessions summary` without importing it.
DISP_CAUSE = {
    "LIVE": "live",
    "DONE": "completed",
    "DEAD_MIDTOOL": "crash_mid_tool",
    "DEAD_KILLED": "killed_mid_turn",
    "USER_CLOSED": "user_stopped",
    "STOPPED_LIMIT": "rate_limit",
    "STOPPED_APIERR": "api_error",
    "INFRA_AUTH": "auth",
    "PARKED_WAIT": "parked_on_task",
    "STOPPED_QUIET": "ambiguous_quiet",
}

CHIP = {"green": "🟢", "red": "🔴", "yellow": "🟡", "white": "⚪", "blue": "🔵"}
CAUSE_SHORT = {
    "live": "",
    "completed": "done",
    "crash_mid_tool": "crash",
    "killed_mid_turn": "killed",
    "user_stopped": "stopped",
    "rate_limit": "rate",
    "api_error": "api",
    "auth": "auth",
    "parked_on_task": "parked",
    "ambiguous_quiet": "quiet",
}
# A HANGING session older than this many minutes is worth an operator glance (it has
# been parked / ambiguously quiet long enough that it probably is not coming back on
# its own). Tunable so a host with slower background tasks can raise the floor.
HANGING_ATTENTION_MIN = float(os.environ.get("FLEET_TOP_HANGING_MIN", "30"))
# A parked/quiet session past THIS floor is no longer routine churn the janitor or a
# replace can be expected to clear — it is genuinely stuck and worth escalating. Below
# it, a park is a self-healing state (the lifecycle still owns it); above it, it is an
# operator signal. Default 90m ≈ the reap/replace window; tunable per host.
STUCK_ATTENTION_MIN = float(os.environ.get("FLEET_TOP_STUCK_MIN", "90"))
# Whether the standing fleet actually runs an auto-resume watchdog, i.e. whether a
# DEAD_MIDTOOL row marked AUTO_RESUME is owned by the lifecycle (self-healing) or is
# stranded work an operator must resume by hand (escalate). Default on: the standing
# dispatch fleet resumes automatically. Set FLEET_TOP_AUTORESUME_OWNED=0 on a host with
# no watchdog so a resumable session escalates instead of reading self-healing.
AUTORESUME_OWNED = os.environ.get("FLEET_TOP_AUTORESUME_OWNED", "1").strip().lower() not in ("0", "false", "no", "")
_COMMAND_INLINE_LIMIT = 220

# The system's own health verdict — the point of the card is that the SYSTEM states
# whether it needs a human, so the operator never has to guess from a wall of rows.
LIFECYCLE_SELF_HEALING = "self_healing"  # a lifecycle owns the remediation with an ETA
LIFECYCLE_ESCALATE = "escalate"          # nothing self-heals this — a human must act
VERDICT_HEALTHY = "HEALTHY"              # nothing outstanding
VERDICT_SELF_HEALING = "SELF_HEALING"   # items in flight, all lifecycle-owned
VERDICT_NEEDS_YOU = "NEEDS_YOU"         # ≥1 escalation a human must act on
VERDICT_GLYPH = {VERDICT_HEALTHY: "🟢", VERDICT_SELF_HEALING: "🔵", VERDICT_NEEDS_YOU: "🔴"}
VERDICT_WORD = {VERDICT_HEALTHY: "HEALTHY", VERDICT_SELF_HEALING: "SELF-HEALING", VERDICT_NEEDS_YOU: "NEEDS YOU"}


def _tag(account: str) -> str:
    """The short account tag fleet_sessions uses (".claude-worker7" -> "worker7")."""
    return account.replace(".claude-", "").replace(".claude", "default") or "default"


def load_sessions_doc(
    root: Path, window_h: float, *, timeout_s: int = 90
) -> tuple[dict[str, Any], str]:
    """Run `fleet_sessions.py json` and parse it. Returns (doc, error).

    A watchdog must never crash on a flaky probe, so a failure is captured as an
    error string and the frame renders a clean "signal unavailable" line instead.
    """
    script = root / "tools" / "fleet_sessions.py"
    if not script.exists():
        return {}, f"missing {script}"
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "json", "--window", str(window_h)],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, f"fleet_sessions did not return: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {}, f"fleet_sessions exit {proc.returncode}: {detail[-1] if detail else 'no output'}"
    try:
        return json.loads(proc.stdout), ""
    except ValueError as exc:
        return {}, f"fleet_sessions emitted non-JSON: {exc}"


def build_snapshot(
    doc: dict[str, Any],
    *,
    workspace: str,
    window_h: float,
    now: str,
    error: str = "",
) -> dict[str, Any]:
    """Fold a fleet_sessions json doc into the render-ready top snapshot.

    Pure: no clock, no disk, no subprocess — everything comes from `doc` and the
    passed-in `now`, so the renderer and the tests share one deterministic shape.
    """
    rows = list(doc.get("rows") or [])
    accounts = list(doc.get("accounts") or [])
    throttle = dict(doc.get("throttle") or {})

    by_category: dict[str, int] = {}
    causes: dict[str, dict[str, int]] = {}
    for r in rows:
        cat = str(r.get("category") or "?")
        by_category[cat] = by_category.get(cat, 0) + 1
        cause = DISP_CAUSE.get(str(r.get("disp") or ""), str(r.get("disp") or "?"))
        causes.setdefault(cat, {})[cause] = causes.setdefault(cat, {}).get(cause, 0) + 1

    available = [a for a in accounts if a.get("available")]
    # "throttled" here means unusable *because* of a throttle: an account whose banner
    # has since expired reads as available, so it belongs only in the available line.
    throttled = [a for a in accounts if a.get("throttled") and not a.get("available")]
    blocked_other = [a for a in accounts if a.get("blocked") and not a.get("throttled")]

    attention = _attention(rows, accounts, available, throttle)

    return {
        "schema": SCHEMA,
        "generated_utc": now,
        "workspace": workspace,
        "window_h": window_h,
        "error": error,
        "system": system_fold(attention),
        "sessions": {
            "total": len(rows),
            "by_category": by_category,
            "causes": causes,
        },
        "accounts": {
            "total": len(accounts),
            "usable": len(available),
            "available": [a.get("tag") for a in available],
            # Drive the throttled LIST off the accounts block (throttled AND not available)
            # rather than the raw throttle map: the accounts block has already cleared a
            # throttle that a newer successful turn superseded (the day24 stale-throttle
            # false-positive), so an account whose `available` is True can never appear
            # here. Freshness (verdict_source/age) is carried so a 5-min-old expired-but-
            # cached reset reads visibly differently from a live one.
            "throttled": [
                {
                    "tag": a.get("tag"),
                    "reset": (throttle.get(a.get("account"), {}) or {}).get("reset")
                    or a.get("reset"),
                    "verdict_source": a.get("verdict_source"),
                    "verdict_age_min": a.get("verdict_age_min"),
                }
                for a in throttled
            ],
            "blocked": [
                {"tag": a.get("tag"), "reason": a.get("block_reason"), "kind": a.get("block_kind")}
                for a in blocked_other
            ],
        },
        "attention": attention,
    }


def _attention(
    rows: list[dict[str, Any]],
    accounts: list[dict[str, Any]],
    available: list[dict[str, Any]],
    throttle: dict[str, Any],
) -> list[dict[str, Any]]:
    """The ranked "what needs me now" list — critical (a free action) before warn."""
    items: list[dict[str, Any]] = []

    # Each item carries a `lifecycle`: LIFECYCLE_SELF_HEALING when a running lifecycle
    # owns the remediation with a bounded ETA (auto-resume watchdog, a throttle reset,
    # the reap/replace window) — the system is working and the operator need not act; or
    # LIFECYCLE_ESCALATE when nothing self-heals it (an auth/access wall, an indefinite
    # throttle, a park aged past the reap window) and a human genuinely must step in.
    # Conservative by design: when a self-heal cannot be PROVEN owned, escalate.

    # 1. Resumable: a dead/stopped autonomous session whose account IS available now.
    #    Owned by the auto-resume watchdog on the standing fleet (AUTORESUME_OWNED) —
    #    self-healing there; stranded, operator-only work where no watchdog runs.
    resumable = [r for r in rows if r.get("action") == "AUTO_RESUME"]
    if resumable:
        first = resumable[0]
        items.append({
            "kind": "resume",
            "count": len(resumable),
            "level": "crit",
            "lifecycle": LIFECYCLE_SELF_HEALING if AUTORESUME_OWNED else LIFECYCLE_ESCALATE,
            "title": f"{len(resumable)} session(s) resumable on an available account",
            "detail": f"[{first.get('disp')}] {first.get('project')} "
                      f"age={first.get('age_min')}m"
                      + (f"  (+{len(resumable) - 1} more)" if len(resumable) > 1 else ""),
            "command": first.get("resume_cmd") or "",
        })

    # 2. Accounts that need an interactive /login (auth wall — a person must act). No
    #    automation clears an auth wall, so this always escalates.
    auth = [a for a in accounts if a.get("blocked") and a.get("block_kind") == "auth"]
    for a in auth:
        items.append({
            "kind": "login",
            "tag": a.get("tag"),
            "level": "crit",
            "lifecycle": LIFECYCLE_ESCALATE,
            "title": f"account {a.get('tag')} needs /login",
            "detail": str(a.get("block_reason") or "auth blocked"),
            "command": f"CLAUDE_CONFIG_DIR={a.get('config_dir')} claude  # then /login",
        })

    # 3. Access walls — blocked, but /login will NOT fix it (subscription/admin). A
    #    billing/admin action, never self-healing → escalate.
    access = [a for a in accounts if a.get("blocked") and a.get("block_kind") == "access"]
    for a in access:
        items.append({
            "kind": "access",
            "tag": a.get("tag"),
            "level": "warn",
            "lifecycle": LIFECYCLE_ESCALATE,
            "title": f"account {a.get('tag')} access wall (not fixed by /login)",
            "detail": str(a.get("block_reason") or "access disabled"),
            "command": "",
        })

    # 4. No usable account but work wants to resume — every resume would instantly
    #    re-die. Self-healing when a throttled account carries a known future reset (the
    #    system recovers at the reset); escalate when the stall is indefinite (no reset).
    if resumable and not available:
        throttled_reset = [
            a for a in accounts
            if a.get("throttled") and not a.get("available")
            and str((throttle.get(a.get("account"), {}) or {}).get("reset")
                    or a.get("reset") or "").strip() not in ("", "?")
        ]
        recoverable = bool(throttled_reset)
        items.append({
            "kind": "wait",
            "level": "warn",
            "lifecycle": LIFECYCLE_SELF_HEALING if recoverable else LIFECYCLE_ESCALATE,
            "title": "no account available — resumes will re-die on the throttle",
            "detail": ("all worker accounts throttled; the system recovers at the reset"
                       if recoverable else
                       "all worker accounts blocked with no known reset — indefinite stall"),
            "command": "",
        })

    # 5. Long-parked / ambiguously-quiet sessions. Under the stuck ceiling the janitor /
    #    replace window still owns it (self-healing churn); past the ceiling it is
    #    genuinely stuck and escalates. Split so only the truly-stuck ones alarm.
    parked = [
        r for r in rows
        if r.get("category") == "HANGING"
        and isinstance(r.get("age_min"), (int, float))
        and r["age_min"] >= HANGING_ATTENTION_MIN
    ]
    stuck = [r for r in parked if r["age_min"] >= STUCK_ATTENTION_MIN]
    healing = [r for r in parked if r["age_min"] < STUCK_ATTENTION_MIN]
    if stuck:
        items.append({
            "kind": "quiet",
            "count": len(stuck),
            "level": "warn",
            "lifecycle": LIFECYCLE_ESCALATE,
            "title": f"{len(stuck)} session(s) stuck >{STUCK_ATTENTION_MIN:.0f}m "
                     "(past the reap/replace window)",
            "detail": ", ".join(
                f"{_tag(r.get('account', ''))}/{(r.get('session') or '')[:8]}" for r in stuck[:4]
            ) + (f", +{len(stuck) - 4} more" if len(stuck) > 4 else ""),
            "command": "",
        })
    if healing:
        items.append({
            "kind": "quiet",
            "count": len(healing),
            "level": "warn",
            "lifecycle": LIFECYCLE_SELF_HEALING,
            "title": f"{len(healing)} session(s) parked/quiet >{HANGING_ATTENTION_MIN:.0f}m",
            "detail": ", ".join(
                f"{_tag(r.get('account', ''))}/{(r.get('session') or '')[:8]}" for r in healing[:4]
            ) + (f", +{len(healing) - 4} more" if len(healing) > 4 else ""),
            "command": "",
        })

    return items


def system_fold(attention: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold the attention list into the system's OWN health verdict, so the card leads
    with a self-assessment instead of a wall of rows the operator must judge.

    NEEDS_YOU when any item is an escalation a human must act on; SELF_HEALING when
    items are in flight but every one is owned by a lifecycle with a bounded ETA (the
    system is working — no action); HEALTHY when nothing is outstanding. Pure given the
    attention list."""
    escalate = sum(1 for i in attention if i.get("lifecycle") == LIFECYCLE_ESCALATE)
    healing = sum(1 for i in attention if i.get("lifecycle") == LIFECYCLE_SELF_HEALING)
    if escalate:
        verdict = VERDICT_NEEDS_YOU
    elif healing:
        verdict = VERDICT_SELF_HEALING
    else:
        verdict = VERDICT_HEALTHY
    return {"verdict": verdict, "escalate": escalate, "self_healing": healing}


# ----- rendering -------------------------------------------------------------

class _Ink:
    """Tiny ANSI helper; a no-op when color is off so frames stay diffable in tests."""

    def __init__(self, color: bool) -> None:
        self.color = color

    def __call__(self, text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.color else text


def _box_top(title: str, width: int) -> str:
    head = f"┌─ {title} "
    return head + "─" * max(0, width - _vis_len(head))


def _box_bottom(note: str, width: int) -> str:
    tail = f"└─ {note} "
    return tail + "─" * max(0, width - _vis_len(tail))


def _vis_len(text: str) -> int:
    """Display width good enough for our rule-padding: emoji chips render ~2 cells."""
    n = 0
    for ch in text:
        n += 2 if ord(ch) >= 0x1F000 else 1
    return n


def render_frame(
    snap: dict[str, Any],
    *,
    width: int = 78,
    color: bool = False,
    interval: float | None = None,
) -> str:
    ink = _Ink(color)
    out: list[str] = []
    title = f"fleet top · {snap.get('workspace')} · {snap.get('generated_utc')}"
    out.append(ink(_box_top(title, width), "36"))
    out.append("")

    if snap.get("error"):
        out.append(ink(f"  {CHIP['red']} signal unavailable: {snap['error']}", "31"))
        out.append("")
        out.append(ink(_box_bottom(_footer(snap, interval), width), "36"))
        return "\n".join(out)

    sess = snap.get("sessions", {})
    out.append(ink(f"SESSIONS  {sess.get('total', 0)} in {snap.get('window_h')}h window", "1"))
    if sess.get("total"):
        for cat in CATEGORY_ORDER + sorted(set(sess.get("by_category", {})) - set(CATEGORY_ORDER)):
            n = sess.get("by_category", {}).get(cat)
            if not n:
                continue
            chip = CHIP[CATEGORY_CHIP.get(cat, "white")]
            causes = sess.get("causes", {}).get(cat, {})
            detail = ", ".join(f"{k} {v}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1]))
            out.append(f"  {chip} {cat:<8}{n:>3}   {ink(detail, '2')}")
    else:
        out.append(ink("  (no sessions in window)", "2"))
    out.append("")

    acc = snap.get("accounts", {})
    out.append(ink(f"ACCOUNTS  {acc.get('usable', 0)}/{acc.get('total', 0)} usable", "1"))
    avail = acc.get("available") or []
    out.append(f"  {CHIP['green']} available  " + (", ".join(avail) if avail else ink("(none)", "31")))
    for t in acc.get("throttled") or []:
        fresh = ""
        age = t.get("verdict_age_min")
        if isinstance(age, (int, float)):
            # how stale the throttle evidence is + where it came from; a carried verdict
            # with no fresh row is the stale-latch case worth an operator's eye.
            src = t.get("verdict_source") or "?"
            fresh = ink(f"  ({src}, seen {age:g}m ago)", "2")
        out.append(f"  {CHIP['red']} throttled  {t.get('tag')}  resets {t.get('reset') or '?'}{fresh}")
    for b in acc.get("blocked") or []:
        out.append(f"  {CHIP['red']} blocked    {b.get('tag')}  ({b.get('reason')})")
    out.append("")

    attn = snap.get("attention") or []
    sysv = snap.get("system") or {}
    verdict = sysv.get("verdict", VERDICT_HEALTHY)
    if attn:
        out.append(ink(f"ATTENTION  {len(attn)}  · system "
                       f"{VERDICT_GLYPH.get(verdict, '')} {VERDICT_WORD.get(verdict, verdict)}"
                       f"  ({sysv.get('escalate', 0)} need you, "
                       f"{sysv.get('self_healing', 0)} self-healing)", "1"))
        for item in attn:
            healing = item.get("lifecycle") == LIFECYCLE_SELF_HEALING
            crit = item.get("level") == "crit"
            # A self-healing item reads blue (the system owns it); a real escalation
            # keeps the red/yellow alarm chip.
            chip = CHIP["blue"] if healing else (CHIP["red"] if crit else CHIP["yellow"])
            colour = "34" if healing else ("31" if crit else "33")
            out.append(f"  {chip} " + ink(item.get("title", ""), colour))
            if item.get("detail"):
                out.append(ink(f"       {item['detail']}", "2"))
            if item.get("command"):
                out.append(ink(f"       $ {item['command']}", "36"))
        out.append("")
    else:
        out.append(ink("ATTENTION  none — fleet is quiet", "32"))
        out.append("")

    out.append(ink(_box_bottom(_footer(snap, interval), width), "36"))
    return "\n".join(out)


def _footer(snap: dict[str, Any], interval: float | None) -> str:
    win = f"{snap.get('window_h')}h window"
    if interval is None:
        return f"{win} · snapshot"
    return f"{win} · refresh {interval:g}s · Ctrl-C to quit"


# ----- slack -----------------------------------------------------------------

def _sess_summary(sess: dict[str, Any]) -> str:
    """One compact line of the session breakdown — chip + count + dominant cause per
    non-empty category, in render order, suppressing the zero rows the boxed frame
    still prints a header for."""
    by_cat = sess.get("by_category") or {}
    causes = sess.get("causes") or {}
    bits: list[str] = []
    for cat in CATEGORY_ORDER + sorted(set(by_cat) - set(CATEGORY_ORDER)):
        n = by_cat.get(cat)
        if not n:
            continue
        chip = CHIP[CATEGORY_CHIP.get(cat, "white")]
        cz = causes.get(cat) or {}
        top = max(cz.items(), key=lambda kv: kv[1])[0] if cz else ""
        top = CAUSE_SHORT.get(top, top)
        label = "running" if cat == "LIVE" else cat.lower()
        bits.append(f"{chip}{n} {label}" + (f"({top})" if top and top != label else ""))
    return " · ".join(bits)


_BLOCK_SHORT = {"auth": "/login", "access": "access-wall"}


def _acct_summary(acc: dict[str, Any]) -> str:
    """One compact GLANCE line of account health: available tags, then each
    throttled / blocked account with a terse status (reset time, ``/login``,
    ``access-wall``). The full reason + the fix command live in the ATTENTION
    lines, so the summary never restates them — it points, the attention acts."""
    bits: list[str] = []
    avail = acc.get("available") or []
    if avail:
        bits.append(f"🟢 {', '.join(avail)}")
    for t in acc.get("throttled") or []:
        bits.append(f"🔴 {t.get('tag')} throttled→{t.get('reset') or '?'}")
    for b in acc.get("blocked") or []:
        short = _BLOCK_SHORT.get(b.get("kind")) or b.get("reason")
        bits.append(f"🔴 {b.get('tag')} {short}")
    return " · ".join(bits)


def _attention_summary(item: dict[str, Any]) -> str:
    kind = item.get("kind")
    if kind == "resume":
        line = f"resume {item.get('count') or '?'}"
    elif kind == "login":
        line = f"login {item.get('tag') or '?'}"
    elif kind == "access":
        line = f"access {item.get('tag') or '?'}"
    elif kind == "quiet":
        verb = "stuck" if item.get("lifecycle") == LIFECYCLE_ESCALATE else "quiet"
        line = f"{verb} {item.get('count') or '?'}"
    elif kind == "wait":
        line = "wait"
    else:
        line = f"{item.get('title', '')}"
    detail = item.get("detail")
    if detail:
        line += f" — {detail}"
    cmd = str(item.get("command") or "")
    if cmd:
        if len(cmd) <= _COMMAND_INLINE_LIMIT:
            line += f" → `{cmd}`"
        else:
            line += " → command omitted from Slack summary; run `python tools/fleet_top.py --once` for the copyable resume command"
    return line


def slack_compact(snap: dict[str, Any]) -> str:
    r"""The COMPACT Slack body for a fleet-top snapshot — the signal-dense peer of
    ``render_frame`` (which keeps the boxed live-dashboard chrome for the terminal).

    ``render_frame`` is built for a refreshing terminal: a ``┌─ … ─┐`` rule padded to
    78 cells, blank lines between sections, a two-space indent on every row, and a
    ``└─ … snapshot ─┘`` footer. Posted to Slack — read on a phone, NOT a refreshing
    terminal — that box-drawing, padding, and blank-line scaffolding is pure noise the
    reader scans past to reach the one thing that needs them: the ATTENTION items and
    their resume command.

    This keeps the signal and drops the chrome: a one-line session breakdown, a
    one-line account-health line (both suppressing the zero rows the boxed frame still
    prints a header for), then ONE line per attention item carrying its command as
    inline code (copy-paste on mobile). A quiet fleet collapses to the two summary
    lines plus a single ``✓``. Pure given the snapshot; no fence, no box, no rules."""
    if snap.get("error"):
        return f"🔴 signal unavailable: {snap['error']}"
    sess = snap.get("sessions") or {}
    acc = snap.get("accounts") or {}
    attn = snap.get("attention") or []
    lines: list[str] = [
        "plane: sessions/accounts, not dispatch slots",
    ]
    ssum = _sess_summary(sess)
    if ssum:
        lines.append("session states: " + ssum)
    asum = _acct_summary(acc)
    if asum:
        lines.append("accounts: " + asum)
    # Partition by whether a lifecycle owns the remediation, NOT by crit/warn: an
    # escalation is the only thing that needs a human; self-healing items are folded
    # into one quiet line so the operator sees the system IS handling them without an
    # alarm per item (the resume command is still carried, in case a human wants it).
    escalate = [i for i in attn if i.get("lifecycle") == LIFECYCLE_ESCALATE]
    healing = [i for i in attn if i.get("lifecycle") == LIFECYCLE_SELF_HEALING]
    if escalate:
        lines.append("action: " + _join_attention(escalate))
    if healing:
        lines.append("self-healing: " + _join_attention(healing) + " — system recovering, no action")
    if not escalate and not healing:
        lines.append("healthy: no session/account action")
    return "\n".join(lines) if lines else "(no fleet signal)"


def _join_attention(items: list[dict[str, Any]], *, limit: int = 2) -> str:
    rows = [_attention_summary(i) for i in items[:limit]]
    if len(items) > limit:
        rows.append(f"+{len(items) - limit} more")
    return "; ".join(rows)


def slack_text(snap: dict[str, Any]) -> str:
    """The Slack body for a fleet-top snapshot: a one-line headline (sessions /
    usable accounts / attention count, so the channel preview carries the state)
    above the COMPACT, signal-dense body (``slack_compact``). The boxed
    ``render_frame`` stays the terminal surface; Slack gets mrkdwn, not a monospace
    box, so the channel/phone reader scans state, not chrome (see the fleet-slack
    signal scorecard in tools/fleet_slack_status.py)."""

    sess = snap.get("sessions") or {}
    acc = snap.get("accounts") or {}
    sysv = snap.get("system") or {}
    verdict = sysv.get("verdict", VERDICT_HEALTHY)
    esc = int(sysv.get("escalate", 0) or 0)
    heal = int(sysv.get("self_healing", 0) or 0)
    # Lead with the system's OWN verdict so the channel preview / mobile push answers
    # "does this need me?" without opening the card — the operator never guesses whether
    # the fleet is working. The verdict word lives ONLY here (never restated in the
    # body) so the signal/noise scorecard counts it once.
    tail = []
    if esc:
        tail.append(f"{esc} need you")
    if heal:
        tail.append(f"{heal} self-healing")
    status = (" — " + ", ".join(tail)) if tail else ""
    headline = (f"{VERDICT_GLYPH.get(verdict, '🟢')} *agent session health — "
                f"{VERDICT_WORD.get(verdict, verdict)}*{status} · "
                f"{sess.get('total', 0)} session(s), "
                f"{acc.get('usable', 0)}/{acc.get('total', 0)} accounts usable")
    return headline + "\n" + slack_compact(snap)


def post_to_slack(snap: dict[str, Any], *, channel: str = "",
                  dry_run: bool = False, transport: Any | None = None) -> dict[str, Any]:
    """Post the session/account-health snapshot to Slack via tools/slack_post. Never
    raises — a missing poster or a Slack failure becomes a typed verdict the caller logs.
    Channel/token resolve through slack_post ($FAK_DISPATCH_CHANNEL / the shared
    scoreboard token) unless ``channel`` is set, so fleet status lands in the same ops
    channel as the dispatch card."""
    try:
        import slack_post  # sibling module in tools/
    except Exception as exc:  # noqa: BLE001
        return {"posted": False, "error": f"slack_post unavailable: {exc}", "skipped": None}
    return slack_post.send(slack_text(snap), channel=channel, dry_run=dry_run,
                           transport=transport, include_signal_noise=False)


# ----- runtime ---------------------------------------------------------------

def _enable_vt() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:  # pragma: no cover - best effort, never fatal
        pass


def _want_color(args: argparse.Namespace) -> bool:
    if args.no_color or os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def snapshot(root: Path, window_h: float, *, now: str | None = None) -> dict[str, Any]:
    doc, error = load_sessions_doc(root, window_h)
    stamp = now or (doc.get("now") if not error else None) or _iso_now()
    return build_snapshot(
        doc, workspace=str(root), window_h=window_h, now=stamp, error=error
    )


def _iso_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="The live session/account watchdog — the session-health peer of `dos top`."
    )
    ap.add_argument("--workspace", default="", help="repo root to report (default: this repo)")
    ap.add_argument("--once", action="store_true", help="render one frame and exit (CI / pipe)")
    ap.add_argument("--json", action="store_true", help="emit the machine snapshot and exit")
    ap.add_argument("--interval", type=float, default=5.0, help="live-refresh cadence seconds")
    ap.add_argument("--window", type=float, default=10.0, help="session lookback window hours")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI color")
    ap.add_argument("--slack", nargs="?", const="__env__", default=None, metavar="CHANNEL",
                    help="post one snapshot to Slack and exit (optional channel id; "
                         "default: $FAK_DISPATCH_CHANNEL via tools/slack_post)")
    ap.add_argument("--slack-dry-run", action="store_true",
                    help="with --slack: resolve the channel/token and report what WOULD "
                         "be posted without sending")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    root = Path(args.workspace).resolve() if args.workspace else fleet_version.repo_root(Path(__file__))

    if args.json:
        print(json.dumps(snapshot(root, args.window), indent=2))
        return 0

    if args.slack is not None:
        # A Slack post is a one-shot snapshot, never the 5s live loop.
        snap = snapshot(root, args.window)
        channel = "" if args.slack == "__env__" else args.slack
        verdict = post_to_slack(snap, channel=channel, dry_run=args.slack_dry_run)
        if verdict.get("posted"):
            print(f"slack: posted fleet status to {verdict.get('channel')} "
                  f"(ts={verdict.get('ts')})")
            return 0
        if verdict.get("dry_run"):
            print(f"slack (dry-run): would post to {verdict.get('channel') or '(unset)'} "
                  f"[{verdict.get('channel_source')}]")
            return 0
        if verdict.get("skipped"):
            print(f"slack: skipped — {verdict.get('skipped')}")
            return 1
        print(f"slack: FAILED — {verdict.get('error')}")
        return 1

    color = _want_color(args)
    if args.once:
        print(render_frame(snapshot(root, args.window), color=color, interval=None))
        return 0

    _enable_vt()
    try:
        while True:
            frame = render_frame(snapshot(root, args.window), color=color, interval=args.interval)
            sys.stdout.write("\x1b[H\x1b[2J\x1b[3J" + frame + "\n")
            sys.stdout.flush()
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
