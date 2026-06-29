#!/usr/bin/env python3
r"""fleet_slack_status — post the WHOLE fleet status to Slack in one scheduled tick.

The operator wants one channel (e.g. $FAK_DISPATCH_CHANNEL) to carry the fleet's
heartbeat: the always-on dispatcher + its supervisor + the watchdog-installed state
(the dispatch_status card) AND the session/account-health plane (the fleet_top
snapshot). Rather than schedule two tasks, this folds BOTH posts into one process so
they land together, on one cadence, in one place.

It is a thin orchestrator over the two tools that already know how to post — it reuses
``dispatch_status.post_to_slack`` and ``fleet_top.post_to_slack`` (and therefore the
shared ``slack_post`` resolver: FAK_DISPATCH_TOKEN -> the scoreboard token, channel
from --channel / FAK_DISPATCH_CHANNEL). It invents no new transport and holds no token
or channel id in source.

  python tools/fleet_slack_status.py                 # post both cards (full fold)
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
import json
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

    The unit that matters is the COMBINED post — ``fleet_slack_status`` posts both
    cards together, so the operator reads them as one heartbeat; the headline
    ``multiple`` is the combined noise-debt reduction. Per-card numbers are carried
    for the work-list."""
    payload = fixture_dispatch_payload(root)
    snap = fixture_fleet_snapshot()

    cards: dict[str, Any] = {}
    combined_before: list[str] = []
    combined_after: list[str] = []
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
        combined_after.append(after)

    before_body = "\n".join(combined_before)
    after_body = "\n".join(combined_after)
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
                  max_workers: int = 2, closure_commits: int = 2500) -> dict[str, Any]:
    """Build the dispatch status card and post it via dispatch_status.post_to_slack."""
    payload = dispatch_status.collect(root, max_workers=max_workers, fast=fast,
                                      closure_commits=closure_commits)
    verdict = dispatch_status.post_to_slack(payload, channel=channel, dry_run=dry_run)
    verdict["card_verdict"] = payload.get("verdict")
    return verdict


def post_fleet(root: Path, *, channel: str, dry_run: bool,
               window_h: float = 10.0) -> dict[str, Any]:
    """Build the fleet session/account-health snapshot and post it via
    fleet_top.post_to_slack."""
    snap = fleet_top.snapshot(root, window_h)
    verdict = fleet_top.post_to_slack(snap, channel=channel, dry_run=dry_run)
    verdict["sessions"] = (snap.get("sessions") or {}).get("total")
    return verdict


def run(root: Path, *, channel: str = "", dry_run: bool = False, fast: bool = False,
        window_h: float = 10.0, do_dispatch: bool = True,
        do_fleet: bool = True) -> dict[str, Any]:
    """Post the requested cards and fold the per-card verdicts into one record. Each
    post is independent: a failure in one is reported, never aborts the other."""
    out: dict[str, Any] = {"schema": "fleet-slack-status/1", "workspace": str(root),
                           "dispatch": None, "fleet": None}
    if do_dispatch:
        out["dispatch"] = post_dispatch(root, channel=channel, dry_run=dry_run, fast=fast)
    if do_fleet:
        out["fleet"] = post_fleet(root, channel=channel, dry_run=dry_run, window_h=window_h)
    # ok iff every attempted post either landed or was a dry-run.
    parts = [v for v in (out["dispatch"], out["fleet"]) if v is not None]
    out["ok"] = all(bool(v.get("posted") or v.get("dry_run")) for v in parts) if parts else False
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
        description="Post the whole fleet status (dispatch card + session health) to Slack.")
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

    out = run(root, channel=args.channel, dry_run=args.dry_run, fast=args.fast,
              window_h=args.window, do_dispatch=not args.no_dispatch,
              do_fleet=not args.no_fleet)

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(_line("dispatch", out["dispatch"]))
        print(_line("fleet", out["fleet"]))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
