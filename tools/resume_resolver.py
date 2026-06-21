#!/usr/bin/env python3
r"""resume_resolver.py -- resolve WHICH account ``claude --resume <sid>`` should
run under, re-homing the transcript onto a healthy account when the owning
account is rate-limited / blocked.

``claude --resume <sid>`` is CLAUDE_CONFIG_DIR + cwd scoped: it only finds the
conversation under ``<config>/projects/<sanitized-cwd>/<sid>.jsonl``, and ONLY
ever under the *active* ``CLAUDE_CONFIG_DIR``. The ``c`` launcher rotates
accounts per launch, so a bare ``c --resume <id>`` would 404 unless it PINS to
the owning account. The existing PowerShell finder (Find-ClaudeSessionAccount)
pins to the owner -- but when that owner is THROTTLED, pinning to it yields a
*dead* resume: every model call is refused until the limit resets, which for a
weekly cap is days. That is the gap the operator hit on
``c --resume <sid>`` against a session owned by a rate-limited account.

This resolver closes it. It locates the owner host-LAST, newest-mtime (the same
selection rule as the PowerShell finder), checks the owner's LIVE availability
via :mod:`fleet_accounts`, and decides:

  * owner available  -> ``PIN`` to the owner (no copy; the safe default, and the
                        same answer the PS finder already gives).
  * owner blocked    -> ``REHOME``: copy the transcript (+ its ``<sid>/`` sidecar)
                        onto the least-loaded healthy Claude worker and pin THERE.
  * no healthy acct  -> ``PIN_BLOCKED``: pin to the owner anyway (best effort --
                        nothing better exists; the resume waits for the reset).

It is the interactive ``c --resume`` analogue of the headless
:mod:`fleet_resume_watchdog` re-home: the SAME locate-the-owner mechanism, the
SAME copy primitive (``rehome_transcript``), the SAME target ranking
(``_rehome_targets``). The two notes
``c-resume-cross-account-recovery`` / ``account-resume-rehome-and-dryrun``
describe exactly this split: owner reachable -> pin, don't copy; owner throttled
-> copy on purpose. Until now only the watchdog did the second half; this brings
it to the interactive path.

Output contract (so the PowerShell ``c`` can consume it):
  stdout   ONE line: the config dir to set ``CLAUDE_CONFIG_DIR`` to (nothing on
           error). With ``--json``, the full decision record instead.
  stderr   human diagnostics (the action + why, duplicate-account notes).
  exit     0 resolved, 1 session not found, 2 internal error.
  --dry-run  decide + report but do NOT copy. stdout still shows the intended
             pin dir; ``--json`` marks it ``would_rehome`` so a caller never
             pins to a target it has not actually populated.

CLI:
  python tools/resume_resolver.py <session-id>            # print pin config dir
  python tools/resume_resolver.py <session-id> --json     # full record
  python tools/resume_resolver.py <session-id> --dry-run  # decide, don't copy
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fleet_accounts  # noqa: E402
import fleet_resume_watchdog  # noqa: E402  (rehome_transcript -- the canonical copy)
import fleet_sessions  # noqa: E402  (_rehome_targets -- the canonical target ranking)


def _is_host(config_dir: str) -> bool:
    """True for the host ``~/.claude`` login -- the account ``c`` keeps OFF the
    rotation, so it is only ever chosen as an owner when it is the SOLE one."""
    return os.path.basename(config_dir.rstrip("\\/")) == ".claude"


def locate_owner(sid: str, home: str) -> dict | None:
    """Return the on-disk owner record for session ``sid``, or ``None``.

    Scans every ``<home>/.claude*`` account dir for
    ``projects/*/<sid>.jsonl`` and selects the owner host-LAST, newest-mtime:
    among non-host accounts the freshest ``.jsonl`` wins; the host (``~/.claude``)
    is chosen only when it is the SOLE owner. This mirrors the PowerShell
    ``Find-ClaudeSessionAccount`` selection exactly, so the resolver and the
    launcher's fallback never disagree about WHO owns a session. Scanning every
    ``projects/*`` (not just the current cwd slug) also makes the lookup robust
    to a session created under a different working directory.

    The record carries the discovered ``project`` dir name (so the re-home copy
    lands at the exact path ``claude --resume`` will look under) plus a
    ``dup_count`` / ``all_accounts`` summary for the duplicate-fork note.
    """
    matches: list[dict] = []
    for acct_dir in glob.glob(os.path.join(home, ".claude*")):
        if not os.path.isdir(acct_dir):
            continue
        proj_root = os.path.join(acct_dir, "projects")
        if not os.path.isdir(proj_root):
            continue
        for f in glob.glob(os.path.join(proj_root, "*", sid + ".jsonl")):
            try:
                mtime = os.path.getmtime(f)
            except OSError:
                continue
            matches.append({
                "config_dir": acct_dir,
                "account": os.path.basename(acct_dir),
                "project": os.path.basename(os.path.dirname(f)),
                "mtime": mtime,
                "is_host": _is_host(acct_dir),
            })
    if not matches:
        return None
    non_host = [m for m in matches if not m["is_host"]]
    pool = non_host or matches
    pool.sort(key=lambda m: m["mtime"], reverse=True)
    owner = dict(pool[0])
    owner["dup_count"] = len(matches)
    owner["all_accounts"] = sorted(m["account"] for m in matches)
    return owner


def _discover_availability(home: str) -> list[dict]:
    """Live availability records for the routable Claude/opencode workers, shaped
    for :func:`fleet_sessions._rehome_targets` (account / available / live_sessions
    / active_sessions / tag / config_dir). Built from the same roster + runtime
    status the account switcher uses, so a re-home target is never an account the
    switcher itself would refuse to offer."""
    rows = fleet_accounts.annotated_roster(home)
    return [
        {
            "account": r["account"],
            "available": r.get("available"),
            "live_sessions": r.get("live_sessions"),
            "active_sessions": r.get("active_sessions"),
            "tag": r.get("tag"),
            "config_dir": r.get("dir"),
        }
        for r in rows
        if fleet_accounts.routable_worker(r)
    ]


def resolve(sid: str, home: str | None = None, *,
            availability: list[dict] | None = None,
            owner_status: dict | None = None,
            dry_run: bool = False,
            rehome_fn=None) -> dict:
    """Decide where ``claude --resume <sid>`` should run.

    ``availability`` / ``owner_status`` / ``rehome_fn`` are injectable so the
    decision is unit-testable without a live registry or real account dirs;
    production passes none and they are read from :mod:`fleet_accounts` and
    copied with :func:`fleet_resume_watchdog.rehome_transcript`.
    """
    home = home or fleet_accounts.USER
    rehome_fn = rehome_fn or fleet_resume_watchdog.rehome_transcript

    owner = locate_owner(sid, home)
    if owner is None:
        return {
            "ok": False, "action": "NOT_FOUND", "session": sid,
            "pin_config_dir": None,
            "reason": "no ~/.claude* account holds this session id",
        }

    if owner_status is None:
        owner_status = fleet_accounts.runtime_status(owner["account"])
    owner_available = bool(owner_status.get("available", True))
    block_reason = str(owner_status.get("block_reason") or "blocked")

    rec = {
        "ok": True, "session": sid, "project": owner["project"],
        "owner_account": owner["account"], "owner_config_dir": owner["config_dir"],
        "owner_available": owner_available,
        "owner_block_reason": owner_status.get("block_reason", ""),
        "dup_count": owner.get("dup_count", 1),
        "all_accounts": owner.get("all_accounts", [owner["account"]]),
    }

    # Owner reachable -> pin to it, no copy. This is the unchanged, safe default
    # (and exactly what the PS finder already returns).
    if owner_available:
        rec.update({
            "action": "PIN", "rehomed": False,
            "pin_account": owner["account"], "pin_config_dir": owner["config_dir"],
            "reason": "owner account is available -- pin to it (no copy)",
        })
        return rec

    # Owner blocked/throttled -> re-home onto the least-loaded healthy Claude
    # worker, the same ranking the headless watchdog uses.
    if availability is None:
        availability = _discover_availability(home)
    targets = fleet_sessions._rehome_targets(availability, owner["account"])
    if not targets:
        rec.update({
            "action": "PIN_BLOCKED", "rehomed": False,
            "pin_account": owner["account"], "pin_config_dir": owner["config_dir"],
            "reason": (f"owner blocked ({block_reason}) and no healthy Claude "
                       "worker available -- pin to owner; resume waits for reset"),
        })
        return rec

    tgt = targets[0]
    tgt_cfg = tgt.get("config_dir") or os.path.join(home, tgt["account"])
    if not dry_run:
        copied = rehome_fn(owner["config_dir"], tgt_cfg, owner["project"], sid)
        if not copied:
            rec.update({
                "action": "PIN_BLOCKED", "rehomed": False,
                "pin_account": owner["account"], "pin_config_dir": owner["config_dir"],
                "reason": "re-home source transcript missing -- pin to owner",
            })
            return rec
        # shutil.copy2 preserves the SOURCE mtime, so the re-homed copy would tie
        # the throttled original -- and the "host-last, newest-mtime" owner pick
        # (here AND the PowerShell fallback finder) could then re-select the walled
        # account on the next launch. Stamp the re-homed copy as newest so the
        # healthy target is the unambiguous owner from now on (it also stops a
        # redundant re-copy each invocation until the live resume writes to it).
        dst_jsonl = os.path.join(tgt_cfg, "projects", owner["project"], sid + ".jsonl")
        try:
            os.utime(dst_jsonl, None)
        except OSError:
            pass

    tgt_tag = tgt.get("tag") or tgt["account"]
    rec.update({
        "action": "REHOME",
        "rehomed": not dry_run,
        "would_rehome": dry_run,
        "pin_account": tgt["account"], "pin_config_dir": tgt_cfg,
        "source_config_dir": owner["config_dir"],
        "reason": (f"owner blocked ({block_reason}) -- "
                   f"{'would re-home' if dry_run else 're-homed'} transcript onto "
                   f"{tgt_tag} and pin there"),
    })
    return rec


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="resume_resolver",
        description="Resolve the CLAUDE_CONFIG_DIR for `claude --resume <sid>`, "
                    "re-homing onto a healthy account when the owner is throttled.")
    ap.add_argument("session", help="session id to resume")
    ap.add_argument("--home", default=None,
                    help="user home holding the .claude* dirs (default: ~)")
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and report but do NOT copy the transcript")
    ap.add_argument("--json", action="store_true",
                    help="emit the full decision record instead of the bare dir")
    args = ap.parse_args(argv)

    try:
        rec = resolve(args.session, args.home, dry_run=args.dry_run)
    except Exception as exc:  # never crash the launcher -- it falls back on rc!=0
        print(f"[resume-resolver] internal error: {exc}", file=sys.stderr)
        return 2

    if not rec.get("ok"):
        print(f"[resume-resolver] {rec.get('reason')}", file=sys.stderr)
        if args.json:
            print(json.dumps(rec, indent=1))
        return 1

    print(f"[resume-resolver] {rec['action']}: {rec['reason']}", file=sys.stderr)
    if rec.get("dup_count", 1) > 1:
        print(f"[resume-resolver] session in {rec['dup_count']} accounts "
              f"({', '.join(rec['all_accounts'])})", file=sys.stderr)

    if args.json:
        print(json.dumps(rec, indent=1))
    else:
        print(rec["pin_config_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
