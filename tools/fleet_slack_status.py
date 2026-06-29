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
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
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
