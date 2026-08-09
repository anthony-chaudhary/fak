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

Self-bounded: --target workers max, dedup, dry-run by default (--live to spawn).

Why a side-channel and not the account switcher? The zai-coding-plan/GLM seat carries
``route_weight`` 0 in the roster, so the tier-aware switcher (fleet_accounts.route) never
routes tier-2 docs work to it -- ON PURPOSE. The seat's opencode.json auth was historically
mis-probed as "login required" (a stale claude-shaped probe under an opencode dir), which
would have made switcher-routed spawns flap; and the docs backlog is a single, bounded pool
best drained by ONE dedicated, self-limiting scheduled task rather than fanned across the
general switcher. This tool is that task: it targets the docs lane directly, dedups against
live+cooled issues, gates on a real gateway preflight (below), and caps every worker's wall
clock -- the guarantees the generic switcher path does not give a cheap-model bulk drain.
The account_probe opencode probe now returns a fresh, gateway-aware verdict for this seat,
so the roster no longer shows the stale claude-shaped block; routing it through the switcher
remains a deliberate non-goal (raise its route_weight in dos.toml to opt in)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import account_probe                  # noqa: E402
import dispatch_worker as dw          # noqa: E402
import issue_resolve_dispatch as ird  # noqa: E402
import issue_worker_prompt            # noqa: E402

RUNS = REPO / ".dispatch-runs"
# The opencode account dir whose opencode.json carries the working zai-coding-plan
# key. opencode_worker_env pins XDG_CONFIG_HOME from this so the worker loads it.
OPENCODE_DIR = str(Path(os.path.expanduser("~")) / ".config" / "opencode")
GLM_MODEL = "zai-coding-plan/glm-4.5-air"

# --- Local guard gateway on :18080 (issue #4771) ------------------------------
# The glm/opencode docs worker MUST front ``fak guard`` -> a LOCAL pure-fak gateway
# (default ``127.0.0.1:18080``), never talk to the upstream provider directly -- that
# local hop is the dogfood-the-kernel property every guarded lane relies on. But the
# seat's opencode.json ``options.baseURL`` (the public provider) OUT-RANKS the legacy
# global guard base in ``opencode_guard_base_url``, so the route to :18080 was lost:
# the pre-spawn reachability probe hit the keyed provider bare, 401'd, and the docs
# fleet refused EVERY launch (GATEWAY_DOWN in the incident, AUTH once the provider
# answered). Pin the PROVIDER-SCOPED guard base -- the highest-precedence override in
# ``opencode_guard_base_url`` -- so BOTH the probe and the spawned worker route through
# the supervised local gateway again.
GLM_PROVIDER = GLM_MODEL.split("/", 1)[0]  # "zai-coding-plan"
GLM_GUARD_BASEURL_ENV = f"{dw.OPENCODE_GUARD_BASE_URL_ENV}_{dw._env_key_suffix(GLM_PROVIDER)}"
DEFAULT_GLM_GUARD_GATEWAY = "http://127.0.0.1:18080/v1"


def glm_guard_gateway() -> str:
    """The local pure-fak guard-gateway base URL the glm worker fronts (#4771).

    ``FAK_GLM_GUARD_GATEWAY`` overrides the default endpoint; an EMPTY value opts out
    of the forced local route entirely (fall back to the seat's own resolution)."""
    v = os.environ.get("FAK_GLM_GUARD_GATEWAY")
    return DEFAULT_GLM_GUARD_GATEWAY if v is None else v.strip()


def apply_glm_guard_gateway(env: dict) -> str:
    """Pin the provider-scoped guard base so the glm seat routes through the local
    :18080 gateway for BOTH the reachability probe and the spawned worker (#4771).

    Highest precedence in ``opencode_guard_base_url`` (over opencode.json's provider
    baseURL), so the probe/worker hit the supervised local gateway:
      * gateway up   -> probe ``/models`` 200 -> OK -> spawn or a *real* capacity refusal
      * gateway down -> probe GATEWAY_DOWN -> truthful skip, no wasted spawn
    An operator value already present in ``env`` WINS (never overridden). Returns the
    effective guard base (``''`` = not pinned, e.g. FAK_GLM_GUARD_GATEWAY exported empty)."""
    existing = (env.get(GLM_GUARD_BASEURL_ENV) or "").strip()
    if existing:
        return existing
    gw = glm_guard_gateway()
    if gw:
        env[GLM_GUARD_BASEURL_ENV] = gw
    return gw


def _alive(pid: int) -> bool:
    try:
        o = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True,
                           text=True, timeout=10, creationflags=ird.no_window_creationflags())
    except subprocess.TimeoutExpired:
        # A wedged tasklist must not hang the glm-docs tick. Assume alive so we do
        # NOT over-spawn onto a live worker's pool; the next tick re-probes.
        return True
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


def glm_provider_exhausted(runs: Path, *, lookback: int = 16) -> str | None:
    """Reset hint if the zai-coding-plan pool is drained, else None.

    The glm-4.5-air key is a weekly/monthly quota pool. When it is exhausted
    every spawn just retry-loops on 'Weekly/Monthly Limit Exhausted', burning a
    worker slot and ~30 host threads until the documented reset. Detect that from
    the freshest worker logs so the scheduled task is a clean no-op until the pool
    resets, instead of flooding the docs lane with dead workers."""
    try:
        logs = sorted(runs.glob("resolve-*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)[:lookback]
    except OSError:
        return None
    today = dt.date.today()
    for log in logs:
        try:
            tail = log.read_text(errors="replace")[-4000:]
        except OSError:
            continue
        if "Limit Exhausted" not in tail or "zai-coding-plan" not in tail:
            continue
        m = re.search(r"reset at (\d{4}-\d{2}-\d{2})", tail)
        if not m:
            return "unknown"  # exhausted but no parseable reset -> back off conservatively
        try:
            if dt.date.fromisoformat(m.group(1)) >= today:
                return m.group(1)
        except ValueError:
            return "unknown"
    return None


# Probe verdicts that mean "a worker spawned right now cannot connect / cannot auth" --
# spawning into any of these just burns a slot on a worker that immediately fails. LIMIT is
# handled separately (and more precisely) by glm_provider_exhausted from the worker logs.
_PREFLIGHT_REFUSE = ("GATEWAY_DOWN", "AUTH", "ACCESS", "CREDIT")


def gateway_preflight(acct: dict, *, probe=None) -> dict:
    """Ping the SAME guard gateway a glm worker would route through, before spawning.

    The quota guard (glm_provider_exhausted) catches a drained provider from worker logs,
    but says nothing about the LOCAL guard->gateway hop. The docs-lane incident (2026-07-09)
    was exactly that hop being DOWN: every spawned glm worker connection-refused and retry-
    looped for its whole lifespan. This is the missing reachability check -- a real,
    opencode-aware probe of the base URL the worker uses. ``probe`` is injectable for tests.
    Returns the raw account_probe verdict (status in the closed vocabulary)."""
    row = {"account": "opencode", "dir": acct.get("dir") or OPENCODE_DIR,
           "product": "opencode", "tag": acct.get("tag") or "zai-coding-plan",
           "model": acct.get("model") or GLM_MODEL}
    fn = probe or account_probe.probe_opencode_account
    try:
        return fn(row, workspace=REPO, runs_dir=RUNS)
    except Exception as exc:  # a broken probe must not wedge the tick -- treat as inconclusive
        return {"status": "APIERR", "block_kind": "apierr",
                "block_reason": f"gateway preflight raised: {exc}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=2, help="desired live glm workers")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    # Restore the :18080 local guard-gateway route (#4771) BEFORE the reachability probe
    # and any spawn: pin the provider-scoped guard base in the process env so both the
    # preflight and the (os.environ-inheriting) worker front the supervised local gateway.
    apply_glm_guard_gateway(os.environ)

    # Reap timed-out glm workers BEFORE counting capacity. The detached docs-lane spawn path
    # does not sit inside the main dispatcher's reap loop, so a runaway (e.g. one that outran
    # its guard --max-duration, or a pre-cap worker) would otherwise hold a slot forever. This
    # mirrors dispatch_tick's own pre-count reap; --live actually kills, dry-run just reports.
    reaped = ird.reap_timed_out_workers(
        RUNS, timeout_s=ird.DEFAULT_WORKER_TIMEOUT_S, live=args.live)
    n_reaped = len(reaped.get("reaped") or reaped.get("would_reap") or [])

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
               "reaped": n_reaped, "would_spawn": targets,
               "reason": "at target" if deficit == 0 else
               ("no fresh docs issue" if not targets else "dry-run")}
        print(json.dumps(msg) if args.json else
              f"glm-docs: live={have}/{args.target} would_spawn={targets} "
              f"({msg['reason']}; --live to spawn)")
        return 0

    reset = glm_provider_exhausted(RUNS)
    if reset is not None:
        msg = {"pool": "glm-docs", "live": have, "target": args.target,
               "deficit": deficit, "would_spawn": [],
               "reason": f"zai-coding-plan quota exhausted (resets {reset}); skipping spawn"}
        print(json.dumps(msg) if args.json else
              f"glm-docs: provider quota exhausted (resets {reset}); "
              f"skipping {len(targets)} spawn(s) until reset")
        return 0

    acct = {"tag": "zai-coding-plan", "dir": OPENCODE_DIR, "model": GLM_MODEL, "tier": 2}

    # Gateway reachability preflight (Defect 1): the quota guard above cannot see a DOWN
    # local guard->gateway hop, so probe it before committing any spawn. If the gateway is
    # unreachable (or the seat is hard auth-blocked), refuse to spawn -- exactly the failure
    # class that previously spawned workers that immediately connection-refused and looped.
    pf = gateway_preflight(acct)
    if str(pf.get("status") or "").upper() in _PREFLIGHT_REFUSE:
        msg = {"pool": "glm-docs", "live": have, "target": args.target,
               "deficit": deficit, "would_spawn": [],
               "preflight": pf.get("status"),
               "reason": f"gateway preflight {pf.get('status')}: "
                         f"{pf.get('block_reason') or 'seat not usable'}; skipping spawn"}
        print(json.dumps(msg) if args.json else
              f"glm-docs: gateway preflight {pf.get('status')} "
              f"({pf.get('block_reason') or 'seat not usable'}); "
              f"skipping {len(targets)} spawn(s)")
        return 0
    spawned = []
    held = []
    for issue in targets:
        rb = issue_worker_prompt.build(issue, "docs", workspace=REPO)
        contract = ird.issue_contract_review(REPO, rb.get("issue_record"), issue)
        if (contract.get("unavailable") or not contract.get("ok") or
                int(contract.get("score") or 0) < ird.DEFAULT_ISSUE_CONTRACT_MIN_SCORE):
            held.append({
                "issue": issue,
                "verdict": "ISSUE_CONTRACT_HOLD",
                "score": int(contract.get("score") or 0),
                "reason": ird.issue_contract_hold_reason(contract),
            })
            continue
        env = ird.opencode_worker_env(OPENCODE_DIR, "docs", REPO, RUNS)
        env["FLEET_RESOLVE_ISSUE"] = str(issue)
        cmd = ird.build_worker_command("opencode", rb["prompt"], GLM_MODEL)
        cmd, guarded = dw.guarded_launch_command(
            cmd, "docs", "opencode", REPO, env=env)
        if guarded:
            dw.guard_env_augment(env)
        res = ird.spawn_issue_worker(cmd, env, REPO, RUNS, issue, "docs", "opencode",
                                     account=acct, spawn_probe_s=8.0,
                                     prompt_payload=rb["prompt"])
        spawned.append({"issue": issue, "pid": res.get("pid"), "log": res.get("log")})

    out = {"pool": "glm-docs", "live_before": have, "reaped": n_reaped,
           "spawned": len(spawned), "issues": spawned, "held": held}
    print(json.dumps(out) if args.json else
          f"glm-docs: spawned {len(spawned)} on docs -> {[s['issue'] for s in spawned]} "
          f"(held={len(held)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
