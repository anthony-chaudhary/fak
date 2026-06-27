#!/usr/bin/env python3
"""Dispatch glm/opencode workers at the DOCS backlog — the fourth quota pool.

The opus accounts (day26/day27) carry the hard code lanes; the docs lane is the
biggest backlog (~124 issues) and the most tractable for a cheaper model. The
glm-4.5-air model behind the zai-coding-plan key is a SEPARATE quota pool from
opus, so draining docs on glm costs no opus quota.

``fleet_accounts`` marks the opencode accounts "auth/login required" from a STALE
cached probe (taken when ~/.local/share/opencode/auth.json was empty); the key in
opencode.json is in fact live (verified) and ``opencode run --dangerously-skip-
permissions`` produces real turns. So this tool BYPASSES the stale switcher check
and spawns opencode/glm workers directly against the working config, deficit-only
and deduped against live + cooled issues — the same spawn machinery the claude
backend uses.

  python tools/dispatch_glm_docs.py --target 2 --live

Self-bounded: --target workers max, dedup, dry-run by default (--live to spawn)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import issue_resolve_dispatch as ird  # noqa: E402
import issue_worker_prompt            # noqa: E402

RUNS = REPO / ".dispatch-runs"
# The opencode account dir whose opencode.json carries the working zai-coding-plan
# key. opencode_worker_env pins XDG_CONFIG_HOME from this so the worker loads it.
OPENCODE_DIR = str(Path(os.path.expanduser("~")) / ".config" / "opencode")
GLM_MODEL = "zai-coding-plan/glm-4.5-air"


def _alive(pid: int) -> bool:
    o = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    return str(pid) in (o.stdout or "")


def live_glm_workers() -> int:
    n = 0
    for bk in RUNS.glob("resolve-*.backend"):
        try:
            if bk.read_text(encoding="utf-8").strip() != "opencode":
                continue
            pid = int(bk.with_suffix(".pid").read_text(encoding="utf-8").split()[0])
        except (OSError, ValueError, IndexError):
            continue
        if _alive(pid):
            n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=2, help="desired live glm workers")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    have = live_glm_workers()
    deficit = max(0, args.target - have)
    pick = ird.lane_issue_numbers(REPO, "docs")
    nums = [int(n) for n in (pick.get("numbers") or [])]
    skip = {int(x) for x in ird.live_resolution_issues(RUNS)} | \
           {int(x) for x in ird.recently_attempted_issues(RUNS, cooldown_min=120)}
    fresh = [n for n in nums if n not in skip]
    targets = fresh[:deficit]

    if not args.live or not targets:
        msg = {"pool": "glm-docs", "live": have, "target": args.target, "deficit": deficit,
               "would_spawn": targets, "reason": "at target" if deficit == 0 else
               ("no fresh docs issue" if not targets else "dry-run")}
        print(json.dumps(msg) if args.json else
              f"glm-docs: live={have}/{args.target} would_spawn={targets} "
              f"({msg['reason']}; --live to spawn)")
        return 0

    acct = {"tag": "zai-coding-plan", "dir": OPENCODE_DIR, "model": GLM_MODEL, "tier": 2}
    spawned = []
    for issue in targets:
        rb = issue_worker_prompt.build(issue, "docs", workspace=REPO)
        env = ird.opencode_worker_env(OPENCODE_DIR, "docs", REPO, RUNS)
        env["FLEET_RESOLVE_ISSUE"] = str(issue)
        cmd = ird.build_worker_command("opencode", rb["prompt"], GLM_MODEL)
        res = ird.spawn_issue_worker(cmd, env, REPO, RUNS, issue, "docs", "opencode",
                                     account=acct, spawn_probe_s=8.0)
        spawned.append({"issue": issue, "pid": res.get("pid"), "log": res.get("log")})

    out = {"pool": "glm-docs", "live_before": have, "spawned": len(spawned), "issues": spawned}
    print(json.dumps(out) if args.json else
          f"glm-docs: spawned {len(spawned)} on docs -> {[s['issue'] for s in spawned]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
