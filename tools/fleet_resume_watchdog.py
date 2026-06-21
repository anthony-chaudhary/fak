#!/usr/bin/env python3
"""fleet_resume_watchdog.py -- macOS/Linux port of fleet_resume_watchdog.ps1.

The cross-account resume layer for ALL autonomous Claude sessions on this host
(not just the supervisor's job workers). Designed to be run on a ~5-minute cron
schedule. Safe to run by hand any time.

Each tick:
  1. EXTRACT-IN-ADVANCE: refresh the on-disk session registry
     (tools/_registry/sessions.json) and the AUTO_RESUME plan
     (tools/_registry/resume_plan.json) via fleet_sessions.py.
  2. Resume each AUTO_RESUME session ONCE EVER, under its owning account's
     CLAUDE_CONFIG_DIR. "Once ever" is enforced by a durable ledger
     (tools/_registry/resume_ledger.jsonl): a session that dies again after
     being auto-resumed is left for a human, never re-resumed in a loop.
  3. Notify (macOS Notification Center via osascript + notifications.log) when
     an account needs a human re-login (BLOCKED_AUTH) -- once per session.

Safety rails (faithful to the .ps1):
  * DRY-RUN by default. Set FAK_LIVE=1 (or pass --live) to actually resume.
  * Interactive sessions are SURFACE (never auto-resumed); supervisor workers are
    SUPERVISED (left to run_supervise_loop); throttled accounts are deferred --
    all decided upstream by fleet_sessions.py when it writes the plan.
  * RESUME ONCE: ledger-gated, survives state-file loss.
  * Per-tick launch cap.

Usage:
  python3 fleet_resume_watchdog.py            # dry-run: log what it WOULD resume
  python3 fleet_resume_watchdog.py --live      # actually resume (once per session)
  FAK_LIVE=1 python3 fleet_resume_watchdog.py  # same, for cron
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET_DIR = os.path.dirname(HERE)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


LIVE = ("--live" in sys.argv) or _env_flag("FAK_LIVE")
WINDOW_H = float(os.environ.get("FAK_WINDOW_H", "6"))
MAX_PER_TICK = int(os.environ.get("FAK_MAX_PER_TICK", "2"))
PYTHON = sys.executable or "python3"
# Source the claude binary from the fleet-wide convention (FLEET_CLAUDE_EXE), like
# account_probe.py and the rest of the fleet; FAK_CLAUDE_EXE stays a back-compat
# fallback. Resolve `claude`/`claude.exe` on PATH before the hardcoded default -- the
# old bare `~/.local/bin/claude` had no `.exe`, so it pointed at a non-existent path
# on Windows (the primary platform).
CLAUDE_EXE = (
    os.environ.get("FLEET_CLAUDE_EXE")
    or os.environ.get("FAK_CLAUDE_EXE")
    or shutil.which("claude")
    or shutil.which("claude.exe")
    or os.path.expanduser("~/.local/bin/claude")
)
LOG_DIR = os.environ.get("FAK_WATCHDOG_LOG_DIR", os.path.join(HERE, "_watchdog"))
# Honor FLEET_REG_DIR exactly as fleet_sessions.py does (same tools/_registry default),
# so the watchdog reads the plan/ledger/sessions from the dir the refresh child WRITES.
# The .ps1 pins this with `$env:FLEET_REG_DIR = $regDir`; without it an ambient
# FLEET_REG_DIR (set by fleet_control_pane.py or an operator) makes fleet_sessions.py
# write to $FLEET_REG_DIR while this watchdog reads HERE/_registry -> stale/empty plan
# (silent no-op) and a split resume-once ledger (latent double-resume).
REG_DIR = os.environ.get("FLEET_REG_DIR", os.path.join(HERE, "_registry"))
# Active-probe parity with the .ps1 (-Probe auto): on a live tick, re-probe blocked
# accounts so a silently-recovered one re-enters the pool instead of riding a stale
# carried verdict. auto -> blocked on --live, none on dry-run (keeps dry-run
# side-effect-free); override with FAK_PROBE=blocked|stale|all|none.
PROBE_MODE = os.environ.get("FAK_PROBE", "auto").strip().lower()
PROBE_MIN_INTERVAL_MIN = int(os.environ.get("FAK_PROBE_MIN_INTERVAL_MIN", "20"))


def resolve_probe_mode(setting: str, live: bool) -> str:
    """auto -> blocked on a live tick, none on dry-run; else the explicit setting.

    Gating to the live tick keeps the default dry-run side-effect-free (no probe
    spend), matching fleet_resume_watchdog.ps1's -Probe auto behavior."""
    if setting == "auto":
        return "blocked" if live else "none"
    return setting


RESUME_PROMPT = (
    "Resume where you left off; re-establish any /goal or /loop and continue toward it."
)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def note(msg: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    line = f"{now_iso()}  {msg}"
    with open(os.path.join(LOG_DIR, "resume_watchdog.log"), "a") as fh:
        fh.write(line + "\n")
    print(line)


def toast(title: str, message: str, level: str = "info") -> None:
    """macOS Notification Center + a durable notifications.log (best-effort)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "notifications.log"), "a") as fh:
        fh.write(f"{now_iso()}  [{level}] {title} -- {message}\n")
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification {json.dumps(message)} with title {json.dumps(title)}',
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        pass


def load_json(path: str, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def rehome_transcript(src_cfg: str, dst_cfg: str, project: str, sid: str) -> bool:
    """Copy a session's transcript (and its sidecar subagents/workflows dir) from
    the throttled owner's config dir into the healthy target account's config dir.

    `claude --resume <sid>` is CLAUDE_CONFIG_DIR + cwd scoped: it only finds the
    conversation under <config>/projects/<sanitized-cwd>/<sid>.jsonl. So to resume
    on a different account the transcript must physically live there first. Returns
    False (caller skips the resume) when the source transcript is missing."""
    src = os.path.join(src_cfg, "projects", project, sid + ".jsonl")
    if not os.path.isfile(src):
        return False
    dst_dir = os.path.join(dst_cfg, "projects", project)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, sid + ".jsonl"))
    side = os.path.join(src_cfg, "projects", project, sid)
    if os.path.isdir(side):
        try:
            shutil.copytree(side, os.path.join(dst_dir, sid), dirs_exist_ok=True)
        except Exception:
            pass
    return True


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. refresh registry + plan (extract in advance). On a live tick, also ACTIVELY
    # probe blocked accounts so a silently-recovered account (re-login / access
    # re-enabled / throttle expired) re-enters the pool instead of riding a stale
    # carried verdict -- parity with fleet_resume_watchdog.ps1.
    probe_mode = resolve_probe_mode(PROBE_MODE, LIVE)
    reg_argv = [PYTHON, os.path.join(HERE, "fleet_sessions.py"), "registry",
                "--window", str(WINDOW_H)]
    if probe_mode != "none":
        reg_argv += ["--probe", probe_mode, "--min-interval-min", str(PROBE_MIN_INTERVAL_MIN)]
    # Pin FLEET_REG_DIR so the child writes where we read; pass FLEET_CLAUDE_EXE (when it
    # resolves to a real file) so the probe spends its tiny `say pong` on the SAME binary
    # the resume uses, mirroring fleet_resume_watchdog.ps1.
    refresh_env = {**os.environ, "FLEET_REG_DIR": REG_DIR}
    if CLAUDE_EXE and os.path.exists(CLAUDE_EXE):
        refresh_env["FLEET_CLAUDE_EXE"] = CLAUDE_EXE
    subprocess.run(
        reg_argv,
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=refresh_env,
    )
    plan = (load_json(os.path.join(REG_DIR, "resume_plan.json"), {}) or {}).get("plan", []) or []
    mode = "LIVE" if LIVE else "DRY-RUN"
    note(f"TICK {mode} plan={len(plan)} window={WINDOW_H}h cap={MAX_PER_TICK}")

    # defense-in-depth: the set of account dir-basenames policy still treats as
    # workers. fleet_sessions.py already excludes non-workers when it writes the
    # plan, but a stale plan file could predate the policy -- re-check here too.
    worker_accts = set()
    try:
        acct_doc = json.loads(
            subprocess.run(
                [PYTHON, os.path.join(HERE, "fleet_accounts.py"), "json"],
                check=False, capture_output=True, text=True,
            ).stdout
            or "{}"
        )
        for a in acct_doc.get("accounts", []):
            if a.get("kind") == "worker":
                worker_accts.add(a.get("account"))
    except Exception:
        pass

    # durable resume-once ledger
    ledger_path = os.path.join(REG_DIR, "resume_ledger.jsonl")
    resumed = {}
    if os.path.exists(ledger_path):
        with open(ledger_path) as fh:
            for ln in fh:
                try:
                    resumed[json.loads(ln)["session"]] = True
                except Exception:
                    pass

    launched = 0
    for p in plan:
        if launched >= MAX_PER_TICK:
            note(f"  per-tick cap reached ({MAX_PER_TICK})")
            break
        sid = p.get("session", "")
        sid8 = sid[:8]
        acct = (p.get("account", "") or "").replace(".claude-", "").replace(".claude", "") or "default"
        if worker_accts and p.get("account") not in worker_accts:
            note(f"  SKIP {sid8} -- account {p.get('account')} is not an offered worker (policy/tombstoned)")
            continue
        if sid in resumed:
            note(f"  SKIP {sid8} -- already resumed once (ledger)")
            continue
        if not LIVE:
            note(f"  WOULD RESUME {sid8} acct={acct} proj={p.get('project')}")
            continue

        env = dict(os.environ)
        resume_cfg = p.get("resume_config_dir") or p.get("config_dir", "")
        # re-home: copy the transcript into the target account first, else
        # `claude --resume` won't find it under the new CLAUDE_CONFIG_DIR.
        if p.get("rehomed"):
            src_cfg = p.get("source_config_dir") or p.get("config_dir", "")
            if not rehome_transcript(src_cfg, resume_cfg, p.get("project", ""), sid):
                note(f"  SKIP {sid8} -- re-home source transcript missing")
                continue
            note(f"  RE-HOME {sid8} {p.get('account')} -> {p.get('resume_account')} "
                 f"(transcript copied; resuming on healthy account)")
        env["CLAUDE_CONFIG_DIR"] = resume_cfg
        env.pop("JOB_SUPERVISED_WORKER", None)
        out = os.path.join(LOG_DIR, f"resume-{sid8}-{int(time.time())}.log")
        wd = p.get("cwd") if p.get("cwd") and os.path.isdir(p.get("cwd")) else FLEET_DIR
        with open(out, "ab") as so, open(out + ".err", "ab") as se:
            proc = subprocess.Popen(
                [CLAUDE_EXE, "--resume", sid, "-p", RESUME_PROMPT,
                 "--dangerously-skip-permissions"],
                cwd=wd, env=env, stdout=so, stderr=se,
                start_new_session=True,
            )
        # record in the durable ledger BEFORE anything else -- a crash can't double-resume
        rec = {"ts": now_iso(), "session": sid, "account": p.get("account"),
               "resume_account": p.get("resume_account"), "rehomed": bool(p.get("rehomed")),
               "project": p.get("project"), "pid": proc.pid, "cause": p.get("disp")}
        with open(ledger_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        resumed[sid] = True
        launched += 1
        note(f"  RESUMED {sid8} acct={acct} pid={proc.pid} (ledger-recorded; will not resume again)")
        toast("Resumed dead session", f"{sid8}  ({acct} / {p.get('project')})", "info")

    # 2. alert on true login-blocked accounts -- once per account blocker.
    notified_path = os.path.join(REG_DIR, "_notified.json")
    notified = load_json(notified_path, {}) or {}
    registry = load_json(os.path.join(REG_DIR, "sessions.json"), {}) or {}
    changed = False
    for a in (registry.get("accounts", []) or []):
        if not a.get("blocked") or a.get("throttled") or a.get("block_kind") != "auth":
            continue
        key = f"auth-account:{a.get('account')}:{a.get('block_reason')}"
        if key in notified:
            continue
        acct = a.get("tag") or (a.get("account", "") or "").replace(".claude-", "").replace(".claude", "")
        reason = a.get("block_reason") or "auth/login required"
        sessions = int(a.get("auth_blocked_sessions") or 0)
        session_text = f" / {sessions} stopped session(s)" if sessions else ""
        toast("Account needs re-login", f"{acct} : {reason}{session_text}", "warn")
        note(f"  ALERT auth-blocked acct={acct} reason={reason} (notified)")
        notified[key] = True
        changed = True
    if changed:
        with open(notified_path, "w") as fh:
            json.dump(notified, fh)

    note(f"  done: launched={launched} resumed_total={len(resumed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
