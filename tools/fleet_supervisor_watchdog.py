#!/usr/bin/env python3
"""fleet_supervisor_watchdog.py -- macOS/Linux port of fleet_supervisor_watchdog.ps1.

Keep the job-fleet supervisor (scripts/run_supervise_loop.py, in the `job` repo)
alive, forever, independent of any Claude Code session -- the missing PID-1.

Root cause this fixes: the standing supervisor is the thing that respawns stopped
dispatch-loop workers. When IT dies (crash, host sleep, the session that launched
it ending), nothing respawns IT and the whole fleet silently stalls. This
watchdog re-launches it as a DETACHED process iff it is not already running.

Designed for a ~5-minute cron schedule. Safe to run by hand. Never starts a
second supervisor.

Opt-in by design -- respawning the supervisor launches autonomous workers, so it
stays OFF until you turn it on for the host that should run the fleet:
  * The `job` repo must be present (FAK_JOB_DIR or ~/Documents/GitHub/job).
  * FAK_SUPERVISOR_ENABLE=1 must be set, else this only REPORTS and exits 0.

Exit codes: 0 = alive / disabled / job repo absent (no-op) | 10 = respawned it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET_DIR = os.path.dirname(HERE)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


JOB_DIR = os.environ.get("FAK_JOB_DIR", os.path.expanduser("~/Documents/GitHub/job"))
TARGET = int(os.environ.get("FAK_SUPERVISOR_TARGET", "4"))
ENABLED = _env_flag("FAK_SUPERVISOR_ENABLE")
LOG_DIR = os.environ.get("FAK_WATCHDOG_LOG_DIR", os.path.join(HERE, "_watchdog"))


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def note(msg: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    line = f"{now_iso()}  {msg}"
    with open(os.path.join(LOG_DIR, "watchdog.log"), "a") as fh:
        fh.write(line + "\n")
    print(line)


def toast(title: str, message: str) -> None:
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification {json.dumps(message)} with title {json.dumps(title)}'],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
    except Exception:
        pass


def supervisor_alive() -> list[int]:
    """A supervisor is alive iff a run_supervise_loop.py process exists."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "run_supervise_loop.py"],
            check=False, capture_output=True, text=True,
        ).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def main() -> int:
    run_loop = os.path.join(JOB_DIR, "scripts", "run_supervise_loop.py")

    alive = supervisor_alive()
    if alive:
        note(f"ALIVE   pid(s)={','.join(map(str, alive))} target={TARGET} -- no action")
        return 0

    if not os.path.exists(run_loop):
        note(f"NOOP    job repo / supervisor absent ({run_loop}) -- nothing to keep alive")
        return 0
    if not ENABLED:
        note("NOOP    supervisor DOWN but FAK_SUPERVISOR_ENABLE not set -- reporting only")
        return 0

    # Not alive, enabled, and the supervisor script exists -> respawn it detached.
    py = os.path.join(JOB_DIR, ".venv", "bin", "python")
    if not os.path.exists(py):
        py = sys.executable or "python3"

    verdict = "?"
    try:
        j = json.loads(
            subprocess.run(
                [py, os.path.join(JOB_DIR, "scripts", "supervise_now.py"), "--json"],
                check=False, capture_output=True, text=True,
            ).stdout
            or "{}"
        )
        verdict = j.get("verdict", "?")
    except Exception:
        pass

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(LOG_DIR, f"supervisor-{ts}.log")

    # Launch with JOB_SUPERVISED_WORKER cleared (that flag is for workers, not the
    # supervisor) and fully detached so it outlives this watchdog tick.
    env = dict(os.environ)
    env.pop("JOB_SUPERVISED_WORKER", None)
    with open(out, "ab") as so, open(out + ".err", "ab") as se:
        proc = subprocess.Popen(
            [py, "scripts/run_supervise_loop.py", "--target", str(TARGET)],
            cwd=JOB_DIR, env=env, stdout=so, stderr=se, start_new_session=True,
        )

    note(f"RESPAWN verdict_was={verdict} launched pid={proc.pid} target={TARGET} log={out}")
    toast("Fleet supervisor respawned", f"was {verdict}; relaunched pid={proc.pid} target={TARGET}")
    return 10


if __name__ == "__main__":
    sys.exit(main())
