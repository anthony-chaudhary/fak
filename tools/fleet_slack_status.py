#!/usr/bin/env python3
r"""fleet_slack_status — post the WHOLE fleet status to Slack in one scheduled tick.

The operator wants one channel (e.g. $FAK_DISPATCH_CHANNEL) to carry the fleet's
heartbeat: the always-on dispatcher + its supervisor + the watchdog-installed state
(the dispatch_status card) AND the session/account-health plane (the fleet_top
snapshot). Rather than schedule two Slack messages, this folds BOTH planes into one
roll-up post so the channel carries one heartbeat per tick.

It is a thin orchestrator over the two tools that already know how to render status:
``dispatch_status.slack_text`` and ``fleet_top.slack_text`` remain the per-plane
surfaces, while this module posts one operator roll-up through the shared
``slack_post`` resolver: FAK_DISPATCH_TOKEN -> the scoreboard token, channel from
--channel / FAK_DISPATCH_CHANNEL. It invents no new transport and holds no token or
channel id in source.

  python tools/fleet_slack_status.py                 # post one roll-up (full fold)
  python tools/fleet_slack_status.py --dry-run       # resolve + report, send nothing
  python tools/fleet_slack_status.py --fast          # dispatch card skips gh folds
  python tools/fleet_slack_status.py --json          # machine-readable combined verdict
  python tools/fleet_slack_status.py --channel C0ABC123

Exit 0 when every requested post landed (or it was a dry-run); 1 when a live post
failed or was skipped for a missing precondition, so a scheduled tick's LastResult
flags a misconfiguration rather than a silent no-op.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dispatch_status  # noqa: E402
import fleet_top  # noqa: E402


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


# ===========================================================================
# Fleet-Slack signal/noise scorecard — the measuring stick that makes "the
# Slack fleet status is too noisy" a falsifiable, gateable number.
#
# What this tool posts to Slack is TWO cards (dispatch_status + fleet_top). They
# were originally the box-drawn TERMINAL renderers (``render`` / ``render_frame``)
# dumped verbatim into a ``` code fence — so a phone reader scanned past ``╔═ ║``
# rails, an 78-wide ``┌─ … ─┐`` rule, blank-line padding, a column label on every
# row, and a ``╚═`` footer that RESTATES every row again in prose, to reach the one
# value they act on. That chrome + restatement is noise.
#
# This scores the signal/noise of the Slack body with ZERO hand-curated word lists:
# the only thing called "noise" is what is provably not first-occurrence content —
#   * fence   the ``` code-fence delimiters
#   * box     box-drawing glyphs (pure decoration)
#   * space   whitespace (layout: indentation, column padding, blank lines)
#   * redundant   a content token that already appeared earlier (restatement)
# and signal is the first occurrence of every content token (the values an operator
# reads). The headline is **noise-debt** (the count of those noise chars); the goal
# the user set is to make the Slack status 3x less noisy, i.e. cut noise-debt >=3x.
#
# It is self-contained and re-derivable: it renders ONE canonical fixture BOTH ways
# — the boxed renderer fenced (the "before" Slack format) and the current compact
# ``slack_text`` (the "after") — and reports the reduction. No frozen baseline file
# to drift; the boxed renderers stay the terminal surface, so the comparison is
# always live. ``--signal-check`` gates the reduction so the compact card can never
# silently regress back toward chrome.
# ===========================================================================

SIGNAL_SCHEMA = "fleet-slack-signal/1"
# The 3x-less-noise target the goal states; the gate (``--signal-check``) holds the
# compact renderer to at least this reduction vs the boxed-and-fenced baseline.
SIGNAL_TARGET_MULTIPLE = 3.0

# Box-drawing glyphs the terminal renderers use for rails/rules — pure decoration in
# a Slack body. (Unicode box-drawing + the double-line set the cards actually emit.)
BOX_DRAWING = set("─│┌┐└┘├┤┬┴┼╴╵╶╷╮╭╯╰╱╲╳"
                  "═║╔╗╚╝╠╣╦╩╬╒╓╕╖╘╙╛╜╞╟╡╢╤╥╧╨╪╫")

# A per-message meta self-score footer (``_S/N self-score …_``) may be appended by
# slack_post.append_signal_noise. That line is instrumentation ABOUT the message, not
# the status content the goal is about — so the content S/N measure excludes it (you do
# not count the ruler's own label as part of what it measures). ``signal_score`` also
# reports the with-footer number so the meta overhead is never hidden. Mirrors
# slack_post.SN_MARKER (kept as a literal so the classifier needs no import to strip it).
_SN_META_MARKER = "S/N self-score"


def classify_signal_noise(text: str, *, exclude_meta: bool = True) -> dict[str, Any]:
    """Partition a Slack message body into signal vs noise — pure, list-free.

    Every character lands in exactly one bucket: ``fence`` (a ``` delimiter line),
    ``box`` (a box-drawing glyph), ``space`` (any whitespace — layout, not content),
    ``redundant`` (a whitespace-separated content token that already appeared), or
    ``signal`` (the first occurrence of a content token — the value a reader acts on).

    When ``exclude_meta`` (default) a self-score meta-footer line is dropped before
    counting — it is instrumentation about the message, not status content.

    Returns the bucket counts plus ``total``, ``noise`` (fence+box+space+redundant),
    ``signal_ratio`` (signal/total), and ``signal_to_noise`` (signal/noise). The
    headline operability metric is ``noise`` itself — the noise-debt to drive down."""
    fence = box = space = redundant = signal = 0
    seen: set[str] = set()
    for line in text.split("\n"):
        if exclude_meta and _SN_META_MARKER in line:
            continue  # meta self-score footer — not part of the status content
        if line.strip() == "```":
            fence += len(line) + 1  # the fence delimiter + its newline
            continue
        kept: list[str] = []
        for ch in line:
            if ch in BOX_DRAWING:
                box += 1
            else:
                kept.append(ch)
        clean = "".join(kept)
        space += sum(1 for ch in clean if ch.isspace())
        space += 1  # the newline that joined this line
        for tok in clean.split():
            if tok in seen:
                redundant += len(tok)
            else:
                seen.add(tok)
                signal += len(tok)
    total = fence + box + space + redundant + signal
    noise = fence + box + space + redundant
    return {
        "total": total, "fence": fence, "box": box, "space": space,
        "redundant": redundant, "signal": signal, "noise": noise,
        "signal_ratio": round(signal / total, 4) if total else 0.0,
        "signal_to_noise": round(signal / max(1, noise), 4),
    }


# --- the canonical fixture: a realistically BUSY fleet (so the score reflects a
#     status worth posting, not an all-quiet card). Built through the real folds
#     (dispatch_status.build_payload / fleet_top.build_snapshot) so the score tracks
#     the production renderers, never a hand-copied string that drifts. ---

_FIXTURE_FLEET_DOC: dict[str, Any] = {
    "now": "2026-06-29T18:00:00+00:00",
    "throttle": {".claude-bravo": {"reset": "Jun 30, 6pm", "age_min": 120.0}},
    "accounts": [
        {"account": ".claude-alpha", "tag": "alpha", "available": True, "blocked": False,
         "throttled": False, "block_kind": "", "block_reason": "",
         "config_dir": "/h/.claude-alpha"},
        {"account": ".claude-bravo", "tag": "bravo", "available": False, "blocked": True,
         "throttled": True, "block_kind": "throttle", "block_reason": "rate limited",
         "reset": "Jun 30, 6pm", "verdict_source": "passive", "verdict_age_min": 120.0,
         "config_dir": "/h/.claude-bravo"},
        {"account": ".claude-charlie", "tag": "charlie", "available": False, "blocked": True,
         "throttled": False, "block_kind": "auth", "block_reason": "please run /login",
         "config_dir": "/h/.claude-charlie"},
        {"account": ".claude-delta", "tag": "delta", "available": False, "blocked": True,
         "throttled": False, "block_kind": "access", "block_reason": "subscription disabled",
         "config_dir": "/h/.claude-delta"},
    ],
    "rows": [
        {"category": "LIVE", "disp": "LIVE", "action": "SKIP_DONE", "age_min": 1.0,
         "account": ".claude-alpha", "project": "C--work-fak", "session": "aaaaaaaa-1",
         "resume_cmd": ""},
        {"category": "AGENT", "disp": "DONE", "action": "SKIP_DONE", "age_min": 12.0,
         "account": ".claude-alpha", "project": "C--work-fak", "session": "bbbbbbbb-2",
         "resume_cmd": ""},
        {"category": "AGENT", "disp": "DEAD_MIDTOOL", "action": "AUTO_RESUME", "age_min": 7.0,
         "account": ".claude-alpha", "project": "C--work-fleet", "session": "cccccccc-3",
         "resume_cmd": "claude --resume cccccccc-3 -p 'go'"},
        {"category": "INFRA", "disp": "STOPPED_LIMIT", "action": "DEFER_THROTTLED",
         "age_min": 30.0, "account": ".claude-bravo", "project": "C--work-fleet",
         "session": "dddddddd-4", "resume_cmd": ""},
        {"category": "HANGING", "disp": "PARKED_WAIT", "action": "SURFACE", "age_min": 45.0,
         "account": ".claude-alpha", "project": "C--work-fak", "session": "eeeeeeee-5",
         "last": "awaiting task", "resume_cmd": ""},
    ],
}


def fixture_fleet_snapshot() -> dict[str, Any]:
    """The canonical fleet-top snapshot for the scorecard (a busy mixed fleet)."""
    return fleet_top.build_snapshot(
        _FIXTURE_FLEET_DOC, workspace="C:/work/fak", window_h=10.0,
        now="2026-06-29T18:00:00Z")


def fixture_dispatch_payload(root: Path) -> dict[str, Any]:
    """The canonical dispatch-status payload for the scorecard (a healthy dispatcher
    with a below-target rate, two silent workers, and a stub/unhooked backend — the
    real mix a status post carries)."""
    pre = {"cap": 2, "live": 1, "host": {"safe": True},
           "account": {"tag": "worker7", "tier": 2, "model": "opus", "available": True},
           "verdict": "SPAWN_OK"}
    sup = {"verdict": "READY", "supervise": {"target": 2, "alive": 1},
           "plans": {"total_plans": 3, "total_units": 12}}
    wd = {"installed": True, "status": "Ready"}
    backlog = {"lanes": {"compute": {"issues": [1, 2, 3]}, "docs": {"issues": [4, 5]},
                         "gateway": {"issues": [6]}},
               "counts": {"open": 47, "routed": 44, "unrouted": 3}}
    closure = {"counts": {"TRUE_RESOLVED": 120, "DATA_RESOLVED": 15,
                          "CLAIMED_CLOSED": 30, "OPEN_WITNESSED": 4},
               "closure_rate": 0.8, "honest_close_rate": 0.85}
    throughput = {"schema": "x", "verdict": "BELOW_TARGET",
                  "completed_rate_per_hour": 0.5, "target_per_hour": 2.0,
                  "primary_window_hours": 24,
                  "gh": {"per_window": {"24h": {"closed": 12, "completed": 12,
                                                "completed_rate_per_hour": 0.5}}},
                  "loop": {"per_window": {"24h": {"loop_closed": 3,
                                                  "loop_rate_per_hour": 0.12}},
                           "last_loop_close_age_min": 90}}
    silent = [{"issue": 123, "stamp": "20260629-101010", "log": "resolve-123.log",
               "pid": 1, "size": 0, "kind": "empty"},
              {"issue": 124, "stamp": "20260629-090909", "log": "resolve-124.log",
               "pid": 2, "size": 122, "kind": "stub"}]
    backend_stub_rate = [{"product": "codex", "lookback_min": 90, "total": 5,
                          "productive": 1, "stub": 4, "stub_rate": 0.8,
                          "majority_stub": True, "evidence_logs": ["resolve-200.log"]}]
    hook_failures = [{"product": "codex", "lookback_min": 90, "sessions": 5,
                      "sessions_with_hook_failures": 5, "hook_failures": 40,
                      "evidence_logs": ["resolve-200.log"],
                      "failure_session_rate": 1.0, "all_sessions_unhooked": True}]
    return dispatch_status.build_payload(
        root=root, pre=pre, sup=sup, wd=wd, backlog=backlog, closure=closure,
        max_workers=2, fast=False, silent=silent, weekly_cap=None,
        throughput=throughput, backend_health=[],
        backend_stub_rate=backend_stub_rate, hook_failures=hook_failures,
        run_status=[])


def _default_history(root: Path) -> str:
    return str(root / "docs" / "nightrun" / "fleet-status-history.jsonl")


def _boxed_dispatch_body(payload: dict[str, Any]) -> str:
    """Reconstruct the PRE-compaction Slack body for the dispatch card: the boxed
    terminal ``render`` dumped into a code fence — exactly the format Slack carried
    before the compact renderer. The baseline the compact card is measured against."""
    import slack_post
    verdict = payload.get("verdict")
    headline = f"*dispatch status:* `{verdict}` ({'ok' if payload.get('ok') else 'ACTION'})"
    return headline + "\n" + slack_post.wrap_code(dispatch_status.render(payload))


def _boxed_fleet_body(snap: dict[str, Any]) -> str:
    """The PRE-compaction Slack body for the fleet card: the boxed ``render_frame``
    in a code fence — the baseline the compact ``fleet_top.slack_text`` is measured
    against."""
    import slack_post
    sess = snap.get("sessions") or {}
    acc = snap.get("accounts") or {}
    attn = snap.get("attention") or []
    crit = sum(1 for a in attn if a.get("level") == "crit")
    headline = (f"*fleet status:* {sess.get('total', 0)} session(s), "
                f"{acc.get('usable', 0)}/{acc.get('total', 0)} accounts usable, "
                f"{len(attn)} attention" + (f" ({crit} critical)" if crit else ""))
    return headline + "\n" + slack_post.wrap_code(
        fleet_top.render_frame(snap, color=False, interval=None))


def signal_score(root: Path) -> dict[str, Any]:
    """Render the canonical fixture BOTH ways (boxed-and-fenced baseline vs the
    current compact ``slack_text``), classify each, and fold into the noise-debt
    reduction verdict. Deterministic and network-free (pure folds + pure renderers).

    The unit that matters is the COMBINED post — ``fleet_slack_status`` posts one
    roll-up message, so the operator reads it as one heartbeat; the headline
    ``multiple`` is the combined noise-debt reduction. Per-card numbers are carried
    for the work-list."""
    payload = fixture_dispatch_payload(root)
    snap = fixture_fleet_snapshot()

    cards: dict[str, Any] = {}
    combined_before: list[str] = []
    for name, before, after in (
        ("dispatch", _boxed_dispatch_body(payload), dispatch_status.slack_text(payload)),
        ("fleet", _boxed_fleet_body(snap), fleet_top.slack_text(snap)),
    ):
        b = classify_signal_noise(before)
        a = classify_signal_noise(after)
        cards[name] = {
            "before": b, "after": a,
            "noise_multiple": round(b["noise"] / max(1, a["noise"]), 2),
            "signal_to_noise_multiple": round(
                a["signal_to_noise"] / max(1e-9, b["signal_to_noise"]), 2),
        }
        combined_before.append(before)

    before_body = "\n".join(combined_before)
    after_body = rollup_text(payload, snap)
    cb = classify_signal_noise(before_body)
    ca = classify_signal_noise(after_body)
    multiple = round(cb["noise"] / max(1, ca["noise"]), 2)
    ok = multiple >= SIGNAL_TARGET_MULTIPLE
    # Transparency: the with-meta noise of the actually-posted body (including any
    # self-score footer slack_post.append_signal_noise tacked on) — so the meta
    # overhead the content measure excludes is reported, never hidden.
    ca_with_meta = classify_signal_noise(after_body, exclude_meta=False)
    meta_overhead = ca_with_meta["noise"] - ca["noise"]
    combined = {
        "before": cb, "after": ca,
        "noise_multiple": multiple,
        "signal_to_noise_multiple": round(
            ca["signal_to_noise"] / max(1e-9, cb["signal_to_noise"]), 2),
        "signal_ratio_before": cb["signal_ratio"],
        "signal_ratio_after": ca["signal_ratio"],
        "noise_after_with_meta": ca_with_meta["noise"],
        "meta_footer_overhead": meta_overhead,
    }
    return {
        "schema": SIGNAL_SCHEMA,
        "ok": ok,
        "verdict": "SIGNAL_3X" if ok else "BELOW_TARGET",
        "target_multiple": SIGNAL_TARGET_MULTIPLE,
        "noise_debt_before": cb["noise"],
        "noise_debt_after": ca["noise"],
        "noise_multiple": multiple,
        "meta_footer_overhead": meta_overhead,
        "combined": combined,
        "cards": cards,
        "after_body": after_body,
        "workspace": str(root),
    }


def render_signal_score(p: dict[str, Any]) -> str:
    c = p.get("combined") or {}
    cb = c.get("before") or {}
    ca = c.get("after") or {}
    lines = [
        f"fleet-slack signal scorecard: {p.get('verdict')} "
        f"({'ok' if p.get('ok') else 'ACTION'})",
        f"  goal: make the Slack fleet status >={p.get('target_multiple')}x less noisy",
        "",
        f"NOISE-DEBT  {p.get('noise_debt_before')} -> {p.get('noise_debt_after')} chars "
        f"= {p.get('noise_multiple')}x less noise (target {p.get('target_multiple')}x)",
        f"signal/noise   {cb.get('signal_to_noise')} -> {ca.get('signal_to_noise')} "
        f"({c.get('signal_to_noise_multiple')}x)",
        f"signal density {c.get('signal_ratio_before')} -> {c.get('signal_ratio_after')} "
        f"(higher = less chrome per value; densified, not truncated)",
    ]
    if p.get("meta_footer_overhead"):
        lines.append(f"  (excludes the S/N self-score meta-footer: +{p.get('meta_footer_overhead')} "
                     "noise chars/post of message-about-message instrumentation)")
    lines += [
        "",
        f"  {'card':<10} {'noise→':>14} {'×':>6}   {'sig/noise→':>14}",
    ]
    for name, cd in (p.get("cards") or {}).items():
        b = cd.get("before") or {}
        a = cd.get("after") or {}
        lines.append(
            f"  {name:<10} {str(b.get('noise')) + '→' + str(a.get('noise')):>14} "
            f"{str(cd.get('noise_multiple')) + 'x':>6}   "
            f"{str(b.get('signal_to_noise')) + '→' + str(a.get('signal_to_noise')):>14}")
    lines += [
        "",
        "  noise = fence + box-drawing + whitespace + redundant (restated) tokens",
        "  signal = first occurrence of each content token (the values an operator acts on)",
    ]
    if not p.get("ok"):
        lines.append("")
        lines.append(f"  NOT YET {p.get('target_multiple')}x — the compact slack_text "
                     "renderers have drifted back toward chrome; restore the density.")
    return "\n".join(lines)


def post_dispatch(root: Path, *, channel: str, dry_run: bool, fast: bool,
                  max_workers: int = 2, closure_commits: int = 2500,
                  transport: Any | None = None) -> dict[str, Any]:
    """Build the dispatch status card and post it via dispatch_status.post_to_slack."""
    payload = dispatch_status.collect(root, max_workers=max_workers, fast=fast,
                                      closure_commits=closure_commits)
    verdict = dispatch_status.post_to_slack(payload, channel=channel, dry_run=dry_run,
                                            transport=transport)
    verdict["card_verdict"] = payload.get("verdict")
    return verdict


def post_fleet(root: Path, *, channel: str, dry_run: bool,
               window_h: float = 10.0, history_path: str = "",
               trend_window: int = 24, transport: Any | None = None) -> dict[str, Any]:
    """Build the fleet session/account-health snapshot and post it via
    fleet_top.post_to_slack."""
    snap = fleet_top.snapshot(root, window_h)
    verdict = fleet_top.post_to_slack(snap, channel=channel, dry_run=dry_run,
                                      history_path=history_path,
                                      trend_window=trend_window,
                                      transport=transport)
    verdict["sessions"] = (snap.get("sessions") or {}).get("total")
    return verdict


def collect_rollup(root: Path, *, fast: bool, window_h: float,
                   history_path: str = "", trend_window: int = 24,
                   dry_run: bool = False,
                   max_workers: int = 2,
                   closure_commits: int = 2500,
                   do_dispatch: bool = True,
                   do_fleet: bool = True) -> dict[str, Any]:
    """Collect both status planes for one Slack roll-up. The fleet trend ledger is
    appended once per live roll-up tick; dry-run previews only read existing history."""
    dispatch_payload = None
    fleet_snap = None
    if do_dispatch:
        dispatch_payload = dispatch_status.collect(
            root, max_workers=max_workers, fast=fast, closure_commits=closure_commits)
    if do_fleet:
        fleet_snap = fleet_top.snapshot(root, window_h)
        if history_path:
            fleet_top.attach_trend(
                fleet_snap, history_path, window=trend_window, record=not dry_run)
    return {"dispatch": dispatch_payload, "fleet": fleet_snap}


def _strip_prefix(line: str, prefix: str) -> str:
    return line[len(prefix):].strip() if line.startswith(prefix) else line


def _limited_join(rows: list[str], *, limit: int = 3) -> str:
    kept = [r for r in rows if str(r).strip()][:limit]
    if len(rows) > limit:
        kept.append(f"+{len(rows) - limit} more")
    return "; ".join(kept)


def _friendly_time(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return "the reset time"
    try:
        stamp = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        return stamp.astimezone().strftime("%Y-%m-%d %H:%M %Z").strip()
    except ValueError:
        return text


def _friendly_minutes(raw: Any) -> str:
    try:
        mins = float(raw)
    except (TypeError, ValueError):
        return str(raw)
    if mins < 60:
        return f"{mins:g}m"
    hours = mins / 60
    if hours < 24:
        return f"{hours:.1f}".rstrip("0").rstrip(".") + "h"
    days = hours / 24
    return f"{days:.1f}".rstrip("0").rstrip(".") + "d"


def _operator_trend(line: str) -> str:
    return re.sub(
        r"last loop close ([0-9.]+)m ago",
        lambda m: f"last loop close {_friendly_minutes(m.group(1))} ago",
        line,
    )


def _operator_action(row: str) -> str:
    row = str(row or "").strip()
    m = re.match(
        r"^auth_failed=\d+ \[([^\]]+)\]; next action: run `fak accounts status` "
        r"and re-login or remove the named seat\(s\)$", row)
    if m:
        return f"login failed for {m.group(1)}; run `fak accounts status`, then re-login/remove it"
    m = re.match(r"^([A-Za-z0-9_.-]+) majority-stub \(([^)]+)\); inspect backend output$", row)
    if m:
        product = m.group(1)
        return f"{product} returned no output in {m.group(2)}; inspect logs before adding capacity"
    m = re.match(
        r"^([A-Za-z0-9_.-]+) guard hooks unbound \(([^)]+)\); workers ran unhooked$", row)
    if m:
        return f"{m.group(1)} guard not attached ({m.group(2)}); restart before trusting them"
    m = re.match(r"^throughput BELOW_TARGET: ([^ ]+) vs target ([^ ]+)$", row)
    if m:
        return f"ticket closes below target: {m.group(1)} vs {m.group(2)}"
    m = re.match(
        r"^worker/lease orphans: clean=([0-9]+), orphan-process=([0-9]+), orphan-lease=([0-9]+)$",
        row,
    )
    if m:
        op = int(m.group(2))
        ol = int(m.group(3))
        bits: list[str] = []
        if op:
            bits.append(f"{op} orphan process" + ("" if op == 1 else "es"))
        if ol:
            bits.append(f"{ol} stale lease" + ("" if ol == 1 else "s"))
        what = ", ".join(bits) if bits else "stale worker state"
        return f"cleanup needed: {what}; inspect dispatch status before launching more"
    return row


def _operator_login_tags(rows: list[str]) -> set[str]:
    tags: set[str] = set()
    for row in rows:
        m = re.match(
            r"^auth_failed=\d+ \[([^\]]+)\]; next action: run `fak accounts status` "
            r"and re-login or remove the named seat\(s\)$",
            str(row or "").strip(),
        )
        if not m:
            continue
        for tag in re.split(r",\s*", m.group(1)):
            tag = tag.strip()
            if tag and not tag.startswith("+"):
                tags.add(tag)
    return tags


def _operator_handled(row: str) -> str:
    row = str(row or "").strip()
    m = re.match(r"^([A-Za-z0-9_.-]+) held dead; lane ([^;]+) reallocated(; re-probe every .+)?$", row)
    if m:
        product, lane, cadence = m.group(1), m.group(2), m.group(3) or ""
        cadence = cadence.replace("; re-probe", "; rechecks")
        return f"{product} paused after failed starts; {lane} work reallocated{cadence}"
    return row


def _operator_waiting(payload: dict[str, Any] | None, rows: list[str]) -> list[str]:
    out: list[str] = []
    cap = (payload or {}).get("weekly_cap") or {}
    if cap:
        product = str(cap.get("product") or "worker").strip()
        account = str(cap.get("account") or (
            ((payload or {}).get("dispatcher") or {}).get("account") or {}).get("tag")
            or "account").strip()
        reset = _friendly_time(cap.get("until") or cap.get("reset_text"))
        out.append(
            f"{product} account {account} is capped until {reset}; wait for auto recheck")

    for row in rows:
        text = str(row or "").strip()
        if not text:
            continue
        if "weekly-capped until" in text:
            continue
        if text == "at configured worker-slot cap":
            out.append("worker slots are full; wait for a worker to finish")
            continue
        m = re.match(
            r"^([0-9]+) active lane lease\(s\), ([0-9]+) blocking current candidates(?: \((.*)\))?$",
            text,
        )
        if m:
            blocking = int(m.group(2))
            detail = m.group(3) or "current candidates"
            if blocking:
                out.append(f"lane lease is blocking work: {detail}; wait or pick another lane")
            continue
        if (text.startswith("supervisor PLAN_SURFACE_EMPTY")
                or text.startswith("scheduler liveness says STALLED")
                or text.startswith("worker/lease cross-check clean")
                or "none blocking current candidates" in text
                or "0 blocking current candidates" in text):
            continue
        out.append(text)
    return out


def _dispatch_state(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "skipped"
    try:
        state = dispatch_status._dispatch_headline_state(payload)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        state = "healthy" if payload.get("ok") else "needs you"
    verdict = str(payload.get("verdict") or "UNKNOWN")
    if state == "ACTION":
        return "needs operator"
    if state == "auto-solving":
        return "being handled"
    if state == "expected":
        return "waiting"
    if verdict == "READY_TO_GROW":
        return "ready for more workers"
    return "healthy" if state == "healthy" else state.lower()


def _fleet_state(snap: dict[str, Any] | None) -> str:
    if not snap:
        return "skipped"
    sysv = snap.get("system") or {}
    verdict = str(sysv.get("verdict") or fleet_top.VERDICT_HEALTHY)
    word = fleet_top.VERDICT_WORD.get(verdict, verdict).lower()
    if word == "needs you":
        word = "need attention"
    esc = int(sysv.get("escalate", 0) or 0)
    heal = int(sysv.get("self_healing", 0) or 0)
    tail: list[str] = []
    if esc:
        tail.append(f"{esc} operator")
    if heal:
        tail.append(f"{heal} auto-handled")
    return word + (f" ({', '.join(tail)})" if tail else "")


def _rollup_severity(dispatch_payload: dict[str, Any] | None,
                     fleet_snap: dict[str, Any] | None) -> tuple[str, str]:
    if dispatch_payload:
        dispatch_buckets = dispatch_status._dispatch_slack_buckets(  # type: ignore[attr-defined]
            dispatch_payload)
    else:
        dispatch_buckets = {"action": [], "auto-solving": [], "expected": []}
    fleet_sys = (fleet_snap or {}).get("system") or {}
    if dispatch_buckets.get("action") or int(fleet_sys.get("escalate", 0) or 0):
        return "🔴", "NEEDS YOU"
    if (dispatch_buckets.get("auto-solving") or dispatch_buckets.get("expected")
            or int(fleet_sys.get("self_healing", 0) or 0)):
        return "🔵", "BEING HANDLED"
    return "🟢", "HEALTHY"


def _issue_work_line(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "issue work: skipped"
    d = payload.get("dispatcher") or {}
    a = d.get("account") or {}
    b = payload.get("backlog") or {}
    c = payload.get("closure") or {}
    parts: list[str] = []
    if d.get("live") is not None or d.get("cap") is not None:
        headroom = d.get("headroom")
        parts.append(f"{d.get('live')}/{d.get('cap')} workers active"
                     + (f", {headroom} slots open" if isinstance(headroom, int) else ""))
    if not b.get("na") and b.get("open_issues") is not None:
        ticket = f"{b.get('open_issues')} open tickets"
        if b.get("unrouted"):
            ticket += f", {b.get('unrouted')} need routing"
        parts.append(ticket)
    if a.get("tag"):
        parts.append(f"next account {a.get('tag')}")
    if not c.get("na") and c.get("closure_rate") is not None:
        honest = c.get("honest_close_rate")
        rate = dispatch_status._rate_str(c.get("closure_rate"))  # type: ignore[attr-defined]
        if honest is not None:
            verified = dispatch_status._rate_str(honest)  # type: ignore[attr-defined]
            if verified != rate:
                parts.append(f"close rate {rate}/h, verified {verified}/h")
            else:
                parts.append(f"verified close rate {verified}/h")
        else:
            parts.append(f"close rate {rate}/h")
    return "issue work: " + ("; ".join(parts) if parts else "no local signal")


def _session_line(snap: dict[str, Any] | None) -> str:
    if not snap:
        return "agent sessions: skipped"
    sess = snap.get("sessions") or {}
    acc = snap.get("accounts") or {}
    return (f"agent sessions: {sess.get('total', 0)} in the last {snap.get('window_h')}h; "
            f"{acc.get('usable', 0)}/{acc.get('total', 0)} accounts usable")


def _trend_lines(dispatch_payload: dict[str, Any] | None,
                 fleet_snap: dict[str, Any] | None) -> list[str]:
    lines: list[str] = []
    if dispatch_payload:
        trend = dispatch_status._dispatch_trend_line(dispatch_payload.get("throughput") or {})  # type: ignore[attr-defined]
        if trend:
            lines.append("ticket trend: " + _operator_trend(_strip_prefix(trend, "trend:")))
    if fleet_snap:
        trend = str(fleet_snap.get("trend") or "").strip()
        if trend:
            lines.append("capacity trend: " + _strip_prefix(trend, "trend:"))
    return lines


def _attention_line(prefix: str, items: list[dict[str, Any]]) -> str:
    try:
        joined = fleet_top._join_attention(items)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        joined = _limited_join([str(i.get("title") or i) for i in items], limit=2)
    replacements = {
        "DEAD_MIDTOOL": "stuck mid-tool",
        "DEAD_KILLED": "killed mid-turn",
        "USER_CLOSED": "user stopped",
        "STOPPED_LIMIT": "rate limit",
        "STOPPED_APIERR": "api error",
        "INFRA_AUTH": "auth needed",
        "INFRA_ORG_DISABLED": "org disabled",
        "PARKED_WAIT": "waiting on task",
        "STOPPED_QUIET": "quiet",
    }
    for code, label in replacements.items():
        joined = joined.replace(f"[{code}]", label)
    return f"{prefix}: {joined}" if joined else ""


def rollup_text(dispatch_payload: dict[str, Any] | None,
                fleet_snap: dict[str, Any] | None) -> str:
    """One Slack message for the operator: state, trend, what needs a human, and what
    the automation is already handling. This deliberately avoids the per-plane
    "plane:" labels and boxed terminal chrome."""
    glyph, severity = _rollup_severity(dispatch_payload, fleet_snap)
    lines: list[str] = [
        f"{glyph} *fleet roll-up — {severity}*",
        f"status: issue work {_dispatch_state(dispatch_payload)}; sessions {_fleet_state(fleet_snap)}",
        _issue_work_line(dispatch_payload),
        _session_line(fleet_snap),
    ]
    lines.extend(_trend_lines(dispatch_payload, fleet_snap))

    needs: list[str] = []
    handled: list[str] = []
    waiting: list[str] = []
    if dispatch_payload:
        buckets = dispatch_status._dispatch_slack_buckets(dispatch_payload)  # type: ignore[attr-defined]
        action_rows = buckets.get("action", [])
        dispatch_login_tags = _operator_login_tags(action_rows)
        needs.extend(_operator_action(r) for r in action_rows)
        handled.extend(_operator_handled(r) for r in buckets.get("auto-solving", []))
        waiting.extend(_operator_waiting(dispatch_payload, buckets.get("expected", [])))
    else:
        dispatch_login_tags = set()
    if fleet_snap:
        attn = fleet_snap.get("attention") or []
        escalate = [i for i in attn if i.get("lifecycle") == fleet_top.LIFECYCLE_ESCALATE]
        if dispatch_login_tags:
            escalate = [
                i for i in escalate
                if not (i.get("kind") == "login" and str(i.get("tag") or "") in dispatch_login_tags)
            ]
        healing = [i for i in attn if i.get("lifecycle") == fleet_top.LIFECYCLE_SELF_HEALING]
        line = _attention_line("agent sessions", escalate)
        if line:
            needs.append(line)
        line = _attention_line("agent sessions", healing)
        if line:
            handled.append(line)

    lines.append("operator moves: " + (_limited_join(needs, limit=4) if needs else "none"))
    if handled:
        lines.append("being handled: " + _limited_join(handled, limit=3))
    if waiting:
        lines.append("waiting: " + _limited_join(waiting, limit=2))
    return "\n".join(line for line in lines if line)


def post_rollup(dispatch_payload: dict[str, Any] | None,
                fleet_snap: dict[str, Any] | None, *,
                channel: str = "", dry_run: bool = False,
                transport: Any | None = None) -> dict[str, Any]:
    try:
        import slack_post  # sibling module in tools/
    except Exception as exc:  # noqa: BLE001
        return {"posted": False, "error": f"slack_post unavailable: {exc}", "skipped": None}
    return slack_post.send(rollup_text(dispatch_payload, fleet_snap), channel=channel,
                           dry_run=dry_run, transport=transport,
                           include_signal_noise=False)


def run(root: Path, *, channel: str = "", dry_run: bool = False, fast: bool = False,
        window_h: float = 10.0, do_dispatch: bool = True,
        do_fleet: bool = True, separate: bool = False,
        history_path: str | None = None, trend_window: int = 24,
        transport: Any | None = None) -> dict[str, Any]:
    """Post one fleet roll-up by default. ``separate`` keeps the legacy two-message
    mode for operators who explicitly want separate cards."""
    out: dict[str, Any] = {"schema": "fleet-slack-status/1", "workspace": str(root),
                           "mode": "separate" if separate else "rollup",
                           "dispatch": None, "fleet": None, "rollup": None}
    history = _default_history(root) if history_path is None else history_path
    if separate:
        if do_dispatch:
            out["dispatch"] = post_dispatch(root, channel=channel, dry_run=dry_run,
                                            fast=fast, transport=transport)
        if do_fleet:
            out["fleet"] = post_fleet(root, channel=channel, dry_run=dry_run,
                                      window_h=window_h, history_path=history,
                                      trend_window=trend_window, transport=transport)
        parts = [v for v in (out["dispatch"], out["fleet"]) if v is not None]
        out["ok"] = all(bool(v.get("posted") or v.get("dry_run")) for v in parts) if parts else False
        return out

    collected = collect_rollup(root, fast=fast, window_h=window_h, history_path=history,
                               trend_window=trend_window, dry_run=dry_run,
                               do_dispatch=do_dispatch, do_fleet=do_fleet)
    dp = collected.get("dispatch")
    fs = collected.get("fleet")
    out["dispatch"] = (
        {"card_verdict": dp.get("verdict"), "ok": dp.get("ok")}
        if isinstance(dp, dict) else None
    )
    out["fleet"] = (
        {"sessions": (fs.get("sessions") or {}).get("total"),
         "system": (fs.get("system") or {}).get("verdict")}
        if isinstance(fs, dict) else None
    )
    out["rollup"] = post_rollup(dp, fs, channel=channel, dry_run=dry_run,
                                transport=transport)
    out["ok"] = bool((out["rollup"] or {}).get("posted")
                     or (out["rollup"] or {}).get("dry_run"))
    return out


def _line(name: str, v: dict[str, Any] | None) -> str:
    if v is None:
        return f"{name}: skipped (not requested)"
    if v.get("posted"):
        return f"{name}: posted to {v.get('channel')} (ts={v.get('ts')})"
    if v.get("dry_run"):
        return (f"{name} (dry-run): would post to {v.get('channel') or '(unset)'} "
                f"[{v.get('channel_source')}]")
    if v.get("skipped"):
        return f"{name}: skipped — {v.get('skipped')}"
    return f"{name}: FAILED — {v.get('error')}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Post one fleet status roll-up (dispatch + session health) to Slack.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--channel", default="",
                    help="target channel id (default: $FAK_DISPATCH_CHANNEL via slack_post)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve channel/token and report what WOULD post; send nothing")
    ap.add_argument("--fast", action="store_true",
                    help="dispatch card skips the gh-backed folds (backlog/closure/throughput)")
    ap.add_argument("--window", type=float, default=10.0, help="fleet session lookback hours")
    ap.add_argument("--no-dispatch", action="store_true", help="skip the dispatch status card")
    ap.add_argument("--no-fleet", action="store_true", help="skip the fleet session-health card")
    ap.add_argument("--separate", action="store_true",
                    help="legacy mode: post dispatch and session health as two messages")
    ap.add_argument("--history", default="",
                    help="trend ledger path (default: docs/nightrun/fleet-status-history.jsonl)")
    ap.add_argument("--no-trend", action="store_true",
                    help="do not append to / show the fleet trend ledger")
    ap.add_argument("--trend-window", type=int, default=24,
                    help="how many trailing trend ticks the roll-up folds")
    ap.add_argument("--json", action="store_true", help="emit the combined verdict as JSON")
    ap.add_argument("--signal-score", action="store_true",
                    help="score the Slack signal/noise (compact vs boxed-and-fenced) on "
                         "the canonical fixture and exit; post nothing")
    ap.add_argument("--signal-json", action="store_true",
                    help="with --signal-score: emit the score as JSON")
    ap.add_argument("--signal-check", action="store_true",
                    help="with --signal-score: exit non-zero unless the compact card cuts "
                         f"noise-debt >={SIGNAL_TARGET_MULTIPLE:g}x (the regression gate)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()

    # The signal scorecard is a read-only measurement (no Slack post): render the
    # canonical fixture both ways and report the noise-debt reduction.
    if args.signal_score or args.signal_json or args.signal_check:
        score = signal_score(root)
        if args.signal_json:
            print(json.dumps(score, indent=2))
        else:
            print(render_signal_score(score))
        # --signal-check gates on the 3x target; a bare --signal-score is informational
        # (exit 0) so it can be run for the number without failing a script.
        return 0 if (score.get("ok") or not args.signal_check) else 1

    history = "" if args.no_trend else (args.history or _default_history(root))
    out = run(root, channel=args.channel, dry_run=args.dry_run, fast=args.fast,
              window_h=args.window, do_dispatch=not args.no_dispatch,
              do_fleet=not args.no_fleet, separate=args.separate,
              history_path=history, trend_window=args.trend_window)

    if args.json:
        print(json.dumps(out, indent=2))
    elif args.separate:
        print(_line("dispatch", out["dispatch"]))
        print(_line("fleet", out["fleet"]))
    else:
        print(_line("fleet roll-up", out["rollup"]))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
