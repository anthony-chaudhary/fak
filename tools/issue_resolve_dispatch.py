#!/usr/bin/env python3
r"""One guarded, switcher-routed dispatch TICK that spawns an ISSUE-RESOLUTION
worker — the arm that moves the open-issue counter on a plan-empty repo.

``issue_dispatch.py`` spawns the generic ``/dos-kernel:dos-dispatch-loop``
worker, which resolves units from the *plan portfolio*. This public repo ships
no ``PLAN-*.md`` (``PLAN_SURFACE_EMPTY``), so that worker has no work surface and
closes nothing — the backlog lives in GitHub *issues*. This tick is the
issue-driven sibling: it picks ONE concrete open issue on the busiest lane,
renders a scoped resolution prompt (``issue_worker_prompt.py`` — with the
``#N``-in-subject rule that lets the closure auditor witness the resulting
commit), and launches one detached ``claude -p "<prompt>"`` worker to land it.

It shares every safety primitive with ``issue_dispatch.py`` (imported, not
re-implemented): the per-tick registry refresh (route off current account
evidence), the ``dispatch_preflight`` DoS gate (host clean ∧ account free ∧ live
< cap), the switcher-pinned account env, and the detached spawn. DRY-RUN BY
DEFAULT — prints the issue, account, command, and prompt path. ``--live`` spawns.

In-flight de-dup: it skips an issue that already has a live resolution worker (a
dispatch log naming ``#N`` whose process is still alive) so two ticks never storm
the same issue.

Loop ledger: the CLI path appends this dispatcher tick to fak's durable loop
ledger by default (``fak loop append``): every tick records ``fire`` and
admitted/refused ``admit`` rows, live spawns record ``start``, and successful ticks
record ``end``. Disable with ``--no-loop-ledger`` for hermetic/manual probes.

    python tools/issue_resolve_dispatch.py                 # plan one tick (dry-run)
    python tools/issue_resolve_dispatch.py --live          # spawn one issue worker
    python tools/issue_resolve_dispatch.py --lane gateway --live
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue_dispatch  # noqa: E402  (refresh_registry/preflight/worker_env/spawn_detached)
import issue_worker_prompt  # noqa: E402  (render the per-issue resolution prompt)
import dispatch_worker  # noqa: E402  (child_env for the opencode backend)
import dispatch_preflight  # noqa: E402  (pid-sidecar identity probe)

SCHEMA = "fleet-issue-resolve-dispatch/1"
RUNS_DIRNAME = ".dispatch-runs"
# Per-worker membership sidecar (rank/wave/size/shortfall), written next to the
# .pid/.backend sidecars so an auditor enumerates a wave straight from disk.
WAVE_SIDECAR_SUFFIX = ".wave"
ACCOUNT_SIDECAR_SUFFIX = ".account"
_LOG_ISSUE_RE = re.compile(r"resolve-(\d+)-")
DEFAULT_WORKER_TIMEOUT_S = dispatch_worker.DEFAULT_TIMEOUT_S
DEFAULT_SPAWN_PROBE_S = 5.0
LOOP_ID_PREFIX = "issue-resolve-dispatch"

# Worker backends this tick can launch:
#   claude   = opus (t1) -- the reference path, the established quota pool.
#   opencode = glm-5.2 via the zai-coding-plan accounts (t2) -- a SEPARATE quota
#              pool that sits idle. Routing a lane (e.g. docs, where glm is proven)
#              to opencode relieves the opus weekly-quota throughput ceiling.
BACKENDS = ("claude", "opencode")
_BACKEND_PRODUCT = {"claude": "claude", "opencode": "opencode"}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _py() -> str:
    return sys.executable or "python"


def lane_issue_numbers(root: Path, explicit_lane: str | None,
                       exclude: set[str] | None = None) -> dict[str, Any]:
    """Pick the lane (busiest, or explicit) and return its OPEN issue numbers,
    most-recent first. Reuses the same router fold issue_dispatch.pick_lane uses,
    but keeps the per-issue numbers (which pick_lane discards). ``exclude`` drops
    lanes from the busiest-pick (e.g. an opus task excludes 'docs' so the glm task
    owns it) -- ignored when an explicit lane is named."""
    router = issue_dispatch.run_json(
        [_py(), str(root / "tools" / "issue_lane_router.py"), "--json"],
        root, timeout=130)
    lanes = router.get("lanes") or {}
    nums_by_lane: dict[str, list[int]] = {}
    for ln, info in lanes.items():
        iss = info.get("issues") if isinstance(info, dict) else info
        nums: list[int] = []
        for it in (iss or []):
            n = it.get("number") if isinstance(it, dict) else it
            try:
                nums.append(int(n))
            except (TypeError, ValueError):
                continue
        nums_by_lane[ln] = sorted(nums, reverse=True)
    exclude = exclude or set()
    if explicit_lane:
        chosen = explicit_lane
    else:
        eligible = {k: v for k, v in nums_by_lane.items() if k not in exclude}
        chosen = max(eligible, key=lambda k: len(eligible[k])) if eligible else None
    return {"lane": chosen, "numbers": nums_by_lane.get(chosen or "", []),
            "by_lane_count": {k: len(v) for k, v in nums_by_lane.items()},
            "excluded_lanes": sorted(exclude),
            "router_error": router.get("_error")}


def live_resolution_issues(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> set[int]:
    """Issue numbers that already have a LIVE resolution worker — read from the
    dispatch logs (``resolve-<N>-<stamp>.log``) whose pid is still alive. Best
    effort: a log without a recoverable pid is treated as not-live."""
    live: set[int] = set()
    if not runs_dir.is_dir():
        return live
    if alive is None and probe is None:
        try:
            import psutil  # type: ignore
            alive = {p.pid for p in psutil.process_iter()}
        except ImportError:
            alive = None
    for log in runs_dir.glob("resolve-*.log"):
        m = _LOG_ISSUE_RE.search(log.name)
        if not m:
            continue
        pid_file = log.with_suffix(".pid")
        if pid_file.exists():
            if dispatch_preflight.resolve_sidecar_pid_is_live(
                pid_file, alive=alive, probe=probe):
                live.add(int(m.group(1)))
    return live


def recently_attempted_issues(runs_dir: Path, *, cooldown_min: int,
                              now_ts: float | None = None) -> set[int]:
    """Issue numbers attempted within the last ``cooldown_min`` minutes — read from
    the mtime of their ``resolve-<N>-*.log``. This is the anti-churn gate: a hard
    issue (e.g. a mislabeled epic) that a worker could not land must NOT be re-picked
    every tick — re-dispatching it re-storms a known drain while the rest of the
    lane's backlog goes untouched. After the cooldown it becomes eligible again (the
    repo may have changed, or a fresh worker may get further). 0 disables the gate."""
    if cooldown_min <= 0 or not runs_dir.is_dir():
        return set()
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - cooldown_min * 60
    recent: set[int] = set()
    for log in runs_dir.glob("resolve-*.log"):
        m = _LOG_ISSUE_RE.search(log.name)
        if not m:
            continue
        try:
            if log.stat().st_mtime >= horizon:
                recent.add(int(m.group(1)))
        except OSError:
            continue
    return recent


def terminate_issue_worker_tree(pid: int) -> dict[str, Any]:
    """Kill one detached issue worker and its descendants.

    The resolver's Windows path launches a hidden-console cmd/opencode tree, so
    `taskkill /T` is the honest process-tree reaper. On POSIX the worker is
    spawned with `start_new_session=True`, making its pid the process-group root.
    """
    if pid <= 0:
        return {"ok": False, "returncode": None, "error": "invalid pid"}
    try:
        if os.name == "nt":
            proc = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                                  capture_output=True, text=True, timeout=30)
            return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                    "stdout": (proc.stdout or "").strip()[-500:],
                    "stderr": (proc.stderr or "").strip()[-500:]}
        os.killpg(os.getpgid(pid), signal.SIGKILL)
        return {"ok": True, "returncode": 0}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"ok": False, "returncode": None, "error": str(exc)}


def reap_timed_out_workers(
    runs_dir: Path,
    *,
    timeout_s: int | None,
    live: bool,
    now_ts: float | None = None,
    probe: Any | None = None,
    killer: Any | None = None,
) -> dict[str, Any]:
    """Find resolver workers older than the wall-clock cap and optionally reap.

    This is deliberately sidecar-identity gated: a stale `.pid` file can only
    nominate a process if `dispatch_preflight.resolve_sidecar_pid_is_live`
    proves the current PID still belongs to that sidecar. Dry-run reports
    `would_reap`; the scheduled live tick kills the worker tree before preflight
    counts capacity.
    """
    if not timeout_s or timeout_s <= 0:
        return {"timeout_s": timeout_s, "live": live, "candidates": [],
                "reaped": [], "would_reap": [], "disabled": True}
    import time
    now = now_ts if now_ts is not None else time.time()
    kill = killer or terminate_issue_worker_tree
    candidates: list[dict[str, Any]] = []
    reaped: list[dict[str, Any]] = []
    would_reap: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return {"timeout_s": timeout_s, "live": live, "candidates": [],
                "reaped": [], "would_reap": []}
    for pid_file in sorted(runs_dir.glob("resolve-*.pid")):
        m = _LOG_ISSUE_RE.search(pid_file.name)
        if not m:
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            st = pid_file.stat()
        except (OSError, ValueError):
            continue
        age_s = max(0.0, now - st.st_mtime)
        if age_s < timeout_s:
            continue
        if not dispatch_preflight.resolve_sidecar_pid_is_live(pid_file, probe=probe):
            continue
        rec: dict[str, Any] = {
            "issue": int(m.group(1)),
            "pid": pid,
            "pid_file": pid_file.name,
            "log": pid_file.with_suffix(".log").name,
            "age_s": round(age_s, 1),
        }
        candidates.append(dict(rec))
        if live:
            outcome = kill(pid)
            rec["kill"] = outcome
            reaped.append(rec)
        else:
            would_reap.append(rec)
    return {"timeout_s": timeout_s, "live": live, "candidates": candidates,
            "reaped": reaped, "would_reap": would_reap}


def pick_target_issue(numbers: list[int], skip: set[int]) -> int | None:
    """The first lane issue not in ``skip`` (live workers ∪ recently-attempted)."""
    for n in numbers:
        if n not in skip:
            return n
    return None


def build_worker_command(backend: str, prompt: str, model: str | None) -> list[str]:
    """The argv for one detached issue-resolution worker, per backend. Both forms
    run the SAME issue-resolution prompt (with its ``#N``-in-subject rule), so the
    resulting commit is witnessable by the closure auditor regardless of backend."""
    if backend == "claude":
        return ["claude", "-p", "--permission-mode", "bypassPermissions", prompt]
    if backend == "opencode":
        cmd = ["opencode", "run", "--dangerously-skip-permissions"]
        if model:
            cmd += ["-m", model]  # pin the exact model so the run is reproducible/traced
        cmd.append(prompt)
        return cmd
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")


def _opencode_config_home(account_dir: str, runs_dir: Path) -> str:
    """opencode loads its config from ``$XDG_CONFIG_HOME/opencode``. The switcher's
    account dir is either the canonical ``<config_home>/opencode`` (the default
    account) or a sibling ``<config_home>/opencode-<tag>``. For the canonical dir the
    parent IS the XDG home; for a sibling we mint a tiny pin dir whose ``opencode``
    entry is a junction to the account, so opencode loads exactly that account -- we
    never silently mis-attribute a run to the wrong account."""
    p = Path(account_dir)
    if p.name == "opencode":
        return str(p.parent)
    pin = runs_dir / ".opencode-pins" / p.name
    link = pin / "opencode"
    pin.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        try:
            os.symlink(account_dir, link, target_is_directory=True)
        except OSError:
            # Windows without symlink privilege: a directory junction needs none.
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), account_dir],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    if not link.exists():
        raise RuntimeError(f"could not pin opencode account dir {account_dir!r}")
    return str(pin)


def opencode_worker_env(account_dir: str | None, lane: str, workspace: Path,
                        runs_dir: Path) -> dict[str, str]:
    """Child env for an opencode/glm worker: the account pinned via XDG_CONFIG_HOME
    (NOT CLAUDE_CONFIG_DIR / oauth token, which are claude-only), plus the same
    self-describing dispatch vars the claude path stamps."""
    env = dispatch_worker.child_env(lane, "opencode", workspace)
    env.pop("CLAUDE_CONFIG_DIR", None)
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    if account_dir:
        env["XDG_CONFIG_HOME"] = _opencode_config_home(account_dir, runs_dir)
    return env


# Windows process-creation flags for a detached worker.
#   DETACHED_PROCESS  — the child gets NO console at all.
#   CREATE_NO_WINDOW  — the child gets a console, but it is HIDDEN (no window).
# Every detached worker — batch shim OR native exe — gets CREATE_NO_WINDOW, never
# DETACHED_PROCESS, for TWO reasons:
#   1. A .cmd/.bat launcher (opencode.CMD, the npm shim) is run BY cmd.exe, which
#      REQUIRES a console: under DETACHED_PROCESS the batch wrapper dies at the
#      "Terminate batch job (Y/N)?" prompt producing ZERO output — every dispatched
#      glm/opencode worker was DOA (0-byte log) until it got a (hidden) console.
#   2. A native worker (claude.exe) given DETACHED_PROCESS has no console, so every
#      console tool it spawns — git, gh, fak, the shell — pops its OWN visible
#      window: the "random popup windows" seen during a dispatched run. CREATE_NO_WINDOW
#      gives the worker one HIDDEN console the whole tool subtree inherits, so it
#      still outlives this dispatcher but never flashes a window. (Same conclusion as
#      claude_agent_chat.detached_creationflags.)
_DETACHED_PROCESS = 0x00000008  # retained to document the rejected alternative
_CREATE_NO_WINDOW = 0x08000000


def win_creationflags(exe: str) -> int:
    """The Windows creationflags for spawning ``exe`` detached: ALWAYS a HIDDEN
    console (CREATE_NO_WINDOW). A .cmd/.bat shim needs a console to live; a native
    exe needs one so its console grandchildren don't each pop a visible window.
    ``exe`` is accepted for call-site stability but no longer branches the result."""
    del exe  # both batch shims and native exes take the hidden-console path
    return _CREATE_NO_WINDOW


def wave_membership_env(rank: int, wave_id: str, size: int,
                        shortfall: int) -> dict[str, str]:
    """The env vars that stamp a worker's place in its wave: which ``rank`` it is
    in ``[0, size)``, which ``wave_id`` it belongs to, the granted ``size`` of the
    group, and the recorded ``shortfall`` (lanes the wave fell short of the
    request). The deterministic counterpart of ``fleet_accounts.allocate_wave``'s
    per-lane membership stamp, carried into ``child_env``.

    NOT a collective: these vars LABEL an otherwise-independent detached worker —
    they grant no barrier/gather/all-to-all. A wave stays N lanes whose only shared
    fabric is git + the ``dos arbitrate`` lease."""
    return {
        "FLEET_WAVE_ID": str(wave_id),
        "FLEET_WAVE_RANK": str(int(rank)),
        "FLEET_WAVE_SIZE": str(int(size)),
        "FLEET_WAVE_SHORTFALL": str(int(shortfall)),
    }


def write_wave_sidecar(out_log: Path, rank: int, wave_id: str, size: int,
                       shortfall: int) -> dict[str, Any]:
    """Write the per-worker membership sidecar next to ``.pid``/``.backend`` so an
    auditor enumerates the whole wave from the filesystem — without trusting any
    worker's self-report. Mirrors the env stamp; deterministic at allocation."""
    rec = {"wave_id": str(wave_id), "rank": int(rank), "size": int(size),
           "shortfall": int(shortfall)}
    out_log.with_suffix(WAVE_SIDECAR_SUFFIX).write_text(
        json.dumps(rec, sort_keys=True), encoding="utf-8")
    return rec


def write_account_sidecar(out_log: Path, account: dict[str, Any] | None) -> dict[str, Any]:
    """Write the switcher account selected for this worker next to its log.

    A 0-byte worker log used to tell us only which issue died. Stamping the account
    at launch time makes later silent-death scans attributable without trusting the
    worker to start and self-report.
    """
    src = account or {}
    rec = {k: src.get(k) for k in ("tag", "tier", "model", "dir")
           if src.get(k) is not None}
    if rec:
        out_log.with_suffix(ACCOUNT_SIDECAR_SUFFIX).write_text(
            json.dumps(rec, sort_keys=True), encoding="utf-8")
    return rec


def probe_spawned_worker(proc: subprocess.Popen, out_log: Path,
                         wait_s: float = 0.0) -> dict[str, Any]:
    """Briefly check whether a just-spawned worker died before it could log.

    A healthy issue worker can be silent for many seconds while Claude starts, so a
    live process with a 0-byte log is not a failure. The failure we can witness here
    is narrower: the process has already exited AND the log is still empty.
    """
    if wait_s <= 0:
        return {"checked": False}
    try:
        returncode = proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        return {"checked": True, "alive": True, "wait_s": wait_s}
    try:
        log_bytes = out_log.stat().st_size
    except OSError:
        log_bytes = 0
    rec: dict[str, Any] = {
        "checked": True,
        "alive": False,
        "wait_s": wait_s,
        "returncode": returncode,
        "log_bytes": log_bytes,
        "silent": log_bytes == 0,
    }
    if log_bytes:
        try:
            rec["tail"] = out_log.read_text(encoding="utf-8", errors="replace")[-500:]
        except OSError:
            pass
    return rec


def enumerate_wave(runs_dir: Path, wave_id: str) -> dict[str, Any]:
    """Enumerate a wave from its ``.wave`` sidecars on disk — the auditor's view of
    a typed group. Returns ``{wave_id, granted, size, shortfall, ranks, members}``
    where ``ranks`` is the sorted list of stamped ranks (a complete wave reads
    ``0..granted-1``) and ``shortfall`` is the recorded under-fill. Reads only the
    filesystem, never a worker's self-report."""
    runs_dir = Path(runs_dir)
    want = str(wave_id)
    members: list[dict[str, Any]] = []
    for side in runs_dir.glob(f"resolve-*{WAVE_SIDECAR_SUFFIX}"):
        try:
            rec = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(rec.get("wave_id")) != want:
            continue
        members.append(rec)
    members.sort(key=lambda r: int(r.get("rank", -1)))
    ranks = [int(r.get("rank", -1)) for r in members]
    size = int(members[0].get("size", 0)) if members else 0
    shortfall = int(members[0].get("shortfall", 0)) if members else 0
    return {"wave_id": want, "granted": len(members), "size": size,
            "shortfall": shortfall, "ranks": ranks, "members": members}


def spawn_issue_worker(command: list[str], env: dict[str, str], cwd: Path,
                       log_dir: Path, issue: int, lane: str,
                       backend: str,
                       membership: dict[str, Any] | None = None,
                       account: dict[str, Any] | None = None,
                       spawn_probe_s: float = 0.0) -> dict[str, Any]:
    """Launch a detached worker (claude or opencode) on one issue; record pid.

    The log keeps the backend-neutral ``resolve-<N>-<stamp>.log`` name so the close
    arm, silent-worker scan, and in-flight de-dup all see it uniformly.

    ``membership`` (``{rank, wave_id, size, shortfall}``, as ``allocate_wave`` stamps
    each lane) is OPTIONAL: when given, the worker's rank/wave identity is stamped
    into ``child_env`` (``FLEET_WAVE_*``) AND a ``.wave`` sidecar so the wave is a
    legible typed group enumerable from disk. It does NOT make the wave a collective
    — the worker stays a detached lane."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_log = log_dir / f"resolve-{issue}-{stamp}.log"
    exe = shutil.which(command[0]) or command[0]
    argv = [exe, *command[1:]]
    if membership is not None:
        env = {**env, **wave_membership_env(**membership)}
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = win_creationflags(exe)
    else:
        kwargs["start_new_session"] = True
    fh = open(out_log, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                stdout=fh, stderr=subprocess.STDOUT, **kwargs)
    finally:
        fh.close()
    (out_log.with_suffix(".pid")).write_text(str(proc.pid), encoding="utf-8")
    # Per-worker backend sidecar: makes each run's backend traceable from disk,
    # independent of the (multi-task, last-writer) tick record.
    (out_log.with_suffix(".backend")).write_text(backend, encoding="utf-8")
    result: dict[str, Any] = {"pid": proc.pid, "log": str(out_log), "issue": issue,
                              "lane": lane, "backend": backend}
    acct = write_account_sidecar(out_log, account)
    if acct:
        result["account"] = acct
    if membership is not None:
        result["membership"] = write_wave_sidecar(out_log, **membership)
    early = probe_spawned_worker(proc, out_log, spawn_probe_s)
    if early.get("checked"):
        result["early_exit"] = early
    return result


# --- Weekly-cap gate -------------------------------------------------------
# A worker spawned on a quota-exhausted (but still logged-in) account dies
# immediately with a ~65-byte banner, e.g.:
#   You've hit your weekly limit · resets Jun 25, 1pm (America/Los_Angeles)
#   You've hit your weekly limit · resets 4am (America/Los_Angeles)
# The spawn preflight cannot see this — it checks host/headroom/account-logged-in,
# NOT remaining quota — so absent this gate every tick re-spawns a doomed worker
# for the entire (multi-day) reset window: it ships nothing and floods the logs.
# These helpers read the banner back from recent worker logs and persist a hold so
# the dispatcher stops spawning until the quota resets, turning a silent spin into
# an honest WEEKLY_CAPPED hold. Everything here is FAIL-OPEN: any error resolves to
# "not capped", so the gate can only ever ADD a refusal, never wedge the loop.

_CAP_BANNER_RE = re.compile(r"hit your[\w\s]*limit", re.IGNORECASE)
_CAP_RESET_RE = re.compile(r"resets\s+(.+?)\s*\(America/Los_Angeles\)", re.IGNORECASE)
_CAP_RESET_FALLBACK_RE = re.compile(r"resets\s+([^\r\n]+)", re.IGNORECASE)
_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
    start=1)}
# America/Los_Angeles is PDT (UTC-7) over the summer reset windows this gate sees;
# a PT wall-clock time + this offset == UTC. (A ±1h DST error only nudges the hold
# edge and self-corrects on the next probe — fine for a throttle.)
_PT_TO_UTC = dt.timedelta(hours=7)


def _backend_of_log(log: Path) -> str:
    """The backend that produced a worker log, from its ``.backend`` sidecar;
    legacy logs without one are the original ``claude`` backend."""
    try:
        return log.with_suffix(".backend").read_text(encoding="utf-8").strip() or "claude"
    except OSError:
        return "claude"


def _scan_recent_cap_banner(runs_dir: Path, *, product: str, lookback_min: int,
                            now_ts: float) -> dict[str, Any] | None:
    """The most-recent worker log (this backend, mtime within ``lookback_min``,
    tiny) whose body is the quota-limit banner → ``{reset_text, evidence_log}``,
    else None. The size cap (a real worker log is large; the banner is ~65 bytes)
    plus the specific phrase keep a prose log that merely mentions "limit" from
    false-matching."""
    if not runs_dir.is_dir():
        return None
    horizon = now_ts - lookback_min * 60
    best: tuple[float, str, str] | None = None
    for log in runs_dir.glob("resolve-*.log"):
        try:
            st = log.stat()
        except OSError:
            continue
        if st.st_mtime < horizon or st.st_size > 512:
            continue
        if _backend_of_log(log) != product:
            continue
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not _CAP_BANNER_RE.search(text):
            continue
        m = _CAP_RESET_RE.search(text) or _CAP_RESET_FALLBACK_RE.search(text)
        reset_text = m.group(1).strip() if m else ""
        if best is None or st.st_mtime > best[0]:
            best = (st.st_mtime, reset_text, log.name)
    if best is None:
        return None
    return {"reset_text": best[1], "evidence_log": best[2]}


def _parse_reset_to_utc(reset_text: str, now_utc: dt.datetime) -> dt.datetime | None:
    """Best-effort parse of a banner reset clause ('Jun 25, 1pm', '4am',
    '11:30pm') as America/Los_Angeles wall-clock → a naive-UTC datetime, clamped to
    (now, now+8d]. None when no time-of-day is present (caller falls back to a short
    cooldown)."""
    if not reset_text:
        return None
    t = reset_text.lower()
    tm = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", t)
    if not tm:
        return None
    hour = int(tm.group(1)) % 12 + (12 if tm.group(3) == "pm" else 0)
    minute = int(tm.group(2) or 0)
    mo = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})", t)
    if mo:
        month, day = _MONTHS[mo.group(1)], int(mo.group(2))
        try:
            cand = dt.datetime(now_utc.year, month, day, hour, minute) + _PT_TO_UTC
        except ValueError:
            return None
        if cand < now_utc - dt.timedelta(days=1):
            try:
                cand = dt.datetime(now_utc.year + 1, month, day, hour, minute) + _PT_TO_UTC
            except ValueError:
                return None
    else:
        now_pt = now_utc - _PT_TO_UTC
        cand_pt = now_pt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if cand_pt <= now_pt:
            cand_pt += dt.timedelta(days=1)
        cand = cand_pt + _PT_TO_UTC
    if cand <= now_utc:
        return None
    return min(cand, now_utc + dt.timedelta(days=8))


def _iso_to_utc(s: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat((s or "").replace("Z", ""))
    except (ValueError, AttributeError):
        return None


def check_weekly_cap(runs_dir: Path, *, product: str, account_tag: str | None,
                     now_ts: float | None = None, lookback_min: int = 45,
                     fallback_min: int = 60) -> dict[str, Any]:
    """Is ``account_tag`` quota-capped right now? Detects the limit banner from
    recent worker logs and persists a hold in ``account-cap-<product>.json`` so it
    survives the ticks AFTER spawns stop (no fresh banner is produced once we stop
    launching doomed workers). Returns ``{"capped": bool, ...}``. FAIL-OPEN: any
    exception → ``{"capped": False}``."""
    try:
        import time
        now_ts = time.time() if now_ts is None else now_ts
        now_utc = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=now_ts)  # naive UTC
        state_path = runs_dir / f"account-cap-{product}.json"
        hit = _scan_recent_cap_banner(runs_dir, product=product,
                                      lookback_min=lookback_min, now_ts=now_ts)
        if hit:
            until = (_parse_reset_to_utc(hit["reset_text"], now_utc)
                     or now_utc + dt.timedelta(minutes=fallback_min))
            state = {"product": product, "account": account_tag,
                     "reset_text": hit["reset_text"], "evidence_log": hit["evidence_log"],
                     "detected": now_utc.isoformat() + "Z", "until": until.isoformat() + "Z"}
            try:
                runs_dir.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            except OSError:
                pass
            return {"capped": True, "until": state["until"],
                    "reset_text": hit["reset_text"], "source": "banner"}
        # No fresh banner: honor a persisted, unexpired hold for THIS account.
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {"capped": False}
            until = _iso_to_utc(state.get("until") or "")
            if state.get("account") == account_tag and until and now_utc < until:
                return {"capped": True, "until": state.get("until"),
                        "reset_text": state.get("reset_text", ""), "source": "state"}
            if until and now_utc >= until:
                try:
                    state_path.unlink()
                except OSError:
                    pass
        return {"capped": False}
    except Exception:
        return {"capped": False}


def loop_id_for_payload(payload: dict[str, Any]) -> str:
    backend = str(payload.get("backend") or "claude").strip() or "claude"
    return f"{LOOP_ID_PREFIX}/{backend}"


def default_loop_ledger(root: Path) -> Path:
    configured = os.environ.get("FAK_LOOP_LEDGER")
    if configured:
        return Path(configured)
    return root / ".fak" / "loops.jsonl"


def fak_loop_cmd(root: Path) -> list[str]:
    configured = os.environ.get("FAK_BIN")
    if configured:
        return shlex.split(configured)
    return ["go", "run", "./cmd/fak"]


def _loop_metric_args(metrics: dict[str, int]) -> list[str]:
    out: list[str] = []
    for key in sorted(metrics):
        out += ["--metric", f"{key}={int(metrics[key])}"]
    return out


def _loop_evidence_args(evidence: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for kind, ref in evidence:
        if kind and ref:
            out += ["--evidence", f"{kind}={ref}"]
    return out


def append_loop_event(root: Path, ledger: Path, event: dict[str, Any],
                      *, source: str = "issue_resolve_dispatch") -> dict[str, Any]:
    """Append one canonical loop-ledger row through `fak loop append`.

    Fail-open by design: a dispatcher tick must not fail just because the optional
    observability append could not find a binary. The error is returned into the
    tick payload for audit, while the dispatch verdict stays grounded in preflight.
    """
    cmd = fak_loop_cmd(root) + [
        "loop", "append",
        "--ledger", str(ledger),
        "--loop", str(event["loop_id"]),
        "--kind", str(event["kind"]),
        "--source", source,
        "--principal", str(event.get("principal") or event.get("backend") or "dispatcher"),
    ]
    for flag, key in (("--run", "run_id"), ("--status", "status"),
                      ("--verified-state", "verified_state"),
                      ("--reason", "reason"), ("--summary", "summary")):
        if event.get(key):
            cmd += [flag, str(event[key])]
    cmd += _loop_metric_args(event.get("metrics") or {})
    cmd += _loop_evidence_args(event.get("evidence") or [])
    try:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "kind": event.get("kind"), "error": str(exc)}
    return {
        "ok": proc.returncode == 0,
        "kind": event.get("kind"),
        "returncode": proc.returncode,
        "stderr": (proc.stderr or "").strip()[-500:],
        "stdout": (proc.stdout or "").strip()[-500:],
    }


def loop_run_id(payload: dict[str, Any]) -> str:
    spawned = payload.get("spawned") or {}
    if spawned.get("pid"):
        return f"resolve-{payload.get('target_issue') or 'none'}-{spawned.get('pid')}"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"resolve-tick-{payload.get('backend') or 'claude'}-{stamp}"


def record_loop_tick(root: Path, payload: dict[str, Any],
                     *,
                     ledger: Path | None = None,
                     append: Any | None = None) -> dict[str, Any]:
    """Lower one issue-resolve dispatcher tick into loop ledger events.

    The rows describe the DISPATCHER tick, not the spawned worker's eventual issue
    resolution. A successful spawn is therefore `claimed_done` for the tick only;
    the independent commit/issue witness remains the close/audit arm.
    """
    ledger = ledger or default_loop_ledger(root)
    append = append or append_loop_event
    run_id = str(payload.get("run_id") or loop_run_id(payload))
    payload["run_id"] = run_id
    loop_id = loop_id_for_payload(payload)
    pre = payload.get("preflight") or {}
    spawned = payload.get("spawned") or {}
    metrics: dict[str, int] = {
        "live": 1 if payload.get("live") else 0,
        "lane_issue_count": int(payload.get("lane_issue_count") or 0),
        "max_workers": int(payload.get("max_workers") or 0),
        "preflight_live": int(pre.get("live") or 0),
        "preflight_cap": int(pre.get("cap") or 0),
    }
    if payload.get("target_issue") is not None:
        metrics["target_issue"] = int(payload["target_issue"])
    if payload.get("prompt_chars") is not None:
        metrics["prompt_chars"] = int(payload["prompt_chars"])
    if spawned.get("pid"):
        metrics["pid"] = int(spawned["pid"])
    evidence: list[tuple[str, str]] = []
    if payload.get("target_issue") is not None:
        evidence.append(("issue", str(payload["target_issue"])))
    if spawned.get("log"):
        evidence.append(("log", str(spawned["log"])))
    if (payload.get("account") or {}).get("tag"):
        evidence.append(("account", str((payload.get("account") or {}).get("tag"))))

    events: list[dict[str, Any]] = [{
        "loop_id": loop_id, "run_id": run_id, "kind": "fire",
        "backend": payload.get("backend"), "summary": f"issue dispatch tick lane={payload.get('lane') or '-'}",
        "metrics": metrics, "evidence": evidence,
    }]
    admitted = bool(payload.get("ok")) and payload.get("action") in {"would_spawn", "spawned"}
    events.append({
        "loop_id": loop_id, "run_id": run_id, "kind": "admit",
        "backend": payload.get("backend"),
        "status": "admitted" if admitted else "refused",
        "reason": str(payload.get("verdict") or payload.get("action") or ""),
        "summary": str(payload.get("reason") or "")[:200],
        "metrics": metrics, "evidence": evidence,
    })
    if payload.get("action") == "spawned":
        events.append({
            "loop_id": loop_id, "run_id": run_id, "kind": "start",
            "backend": payload.get("backend"),
            "status": "running",
            "reason": "SPAWNED",
            "summary": str(payload.get("reason") or "")[:200],
            "metrics": metrics, "evidence": evidence,
        })
    if payload.get("ok"):
        events.append({
            "loop_id": loop_id, "run_id": run_id, "kind": "end",
            "backend": payload.get("backend"),
            "status": "claimed_done",
            "reason": str(payload.get("verdict") or payload.get("action") or ""),
            "summary": str(payload.get("reason") or "")[:200],
            "metrics": metrics, "evidence": evidence,
        })
    rows = [append(root, ledger, ev) for ev in events]
    return {
        "ledger": str(ledger),
        "loop_id": loop_id,
        "run_id": run_id,
        "events": rows,
        "ok": all(r.get("ok") for r in rows) if rows else True,
    }


def evaluate(root: Path, *, max_workers: int, work_kind: str, lane: str | None,
              live: bool, refresh: bool = True, cooldown_min: int = 120,
              backend: str = "claude",
              exclude_lanes: set[str] | None = None,
              worker_timeout_s: int | None = DEFAULT_WORKER_TIMEOUT_S,
              spawn_probe_s: float = DEFAULT_SPAWN_PROBE_S,
              record_loop: bool = False,
              loop_ledger: Path | None = None) -> dict[str, Any]:
    def finish(payload: dict[str, Any]) -> dict[str, Any]:
        if record_loop:
            payload["loop_ledger"] = record_loop_tick(root, payload, ledger=loop_ledger)
        return payload

    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    product = _BACKEND_PRODUCT[backend]
    runs_dir = root / RUNS_DIRNAME
    reg = issue_dispatch.refresh_registry(root) if refresh else {"skipped": True}
    reaped = reap_timed_out_workers(runs_dir, timeout_s=worker_timeout_s, live=live)
    pre = issue_dispatch.preflight(root, max_workers=max_workers, work_kind=work_kind,
                                   product=product)
    pre_ok = pre.get("verdict") == "SPAWN_OK"
    acct = pre.get("account") or {}

    # Weekly-cap gate — BEFORE the lane router's gh work. A logged-in account can
    # still be quota-exhausted; the preflight returns SPAWN_OK regardless. Without
    # this, every tick spawns a worker that instantly dies on the limit banner,
    # ships nothing, and floods the logs for the whole reset window. Read that
    # banner back from recent worker logs and HOLD until the stated reset instead.
    # Fail-open (check_weekly_cap can only return capped on positive evidence).
    cap = (check_weekly_cap(runs_dir, product=product, account_tag=acct.get("tag"))
           if pre_ok else {"capped": False})
    if pre_ok and cap.get("capped"):
        return finish({
            "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
            "max_workers": max_workers, "registry_refresh": reg,
            "timed_out_workers": reaped,
            "preflight": {"verdict": pre.get("verdict"), "reason": pre.get("reason"),
                          "cap": pre.get("cap"), "live": pre.get("live")},
            "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
            "weekly_cap": cap, "ok": False, "action": "weekly_capped",
            "verdict": "WEEKLY_CAPPED",
            "reason": (f"account '{acct.get('tag')}' is weekly-capped (resets "
                       f"{cap.get('reset_text') or '?'}); holding spawn until "
                       f"{cap.get('until')} — re-arm is automatic at the reset"),
        })

    pick = lane_issue_numbers(root, lane, exclude=exclude_lanes)
    chosen_lane = pick.get("lane")
    live_issues = live_resolution_issues(runs_dir)
    cooled = recently_attempted_issues(runs_dir, cooldown_min=cooldown_min)
    # Skip both a live worker's issue AND a recently-attempted one (anti-churn), so
    # the picker advances down the lane instead of re-storming one un-landable issue.
    skip = live_issues | cooled
    target = pick_target_issue(pick.get("numbers") or [], skip)

    payload: dict[str, Any] = {
        "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
        "max_workers": max_workers, "registry_refresh": reg,
        "timed_out_workers": reaped,
        "preflight": {"verdict": pre.get("verdict"), "reason": pre.get("reason"),
                      "cap": pre.get("cap"), "live": pre.get("live")},
        "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
        "lane": chosen_lane, "lane_issue_count": len(pick.get("numbers") or []),
        "cooled_recently": sorted(cooled), "target_issue": target,
        "already_live": sorted(live_issues),
    }

    if not pre_ok:
        payload.update({"ok": False, "action": "refused",
                        "verdict": pre.get("verdict") or "REFUSE",
                        "reason": f"preflight refused: {pre.get('reason')}"})
        return finish(payload)
    if not chosen_lane:
        payload.update({"ok": False, "action": "no_lane", "verdict": "NO_LANE",
                        "reason": "no lane has open issues (router empty/error)"})
        return finish(payload)
    if target is None:
        payload.update({"ok": False, "action": "no_issue", "verdict": "NO_ISSUE",
                        "reason": (f"every open issue on lane '{chosen_lane}' is "
                                   f"either live ({sorted(live_issues)}) or in "
                                   f"cooldown ({sorted(cooled)}) — nothing fresh to "
                                   f"dispatch this tick")})
        return finish(payload)

    rec = issue_worker_prompt.build(target, chosen_lane, workspace=root)
    payload["prompt_chars"] = rec.get("prompt_chars")
    payload["issue_title"] = rec.get("title")
    model = acct.get("model") if backend == "opencode" else None
    preview_prompt = f"<resolve #{target} prompt, {rec.get('prompt_chars')} chars>"
    payload["command"] = build_worker_command(backend, preview_prompt, model)

    if not live:
        payload.update({"ok": True, "action": "would_spawn", "verdict": "WOULD_SPAWN",
                        "reason": (f"safe to spawn 1 {backend} issue-resolution worker on "
                                   f"#{target} (lane '{chosen_lane}') under account "
                                   f"'{acct.get('tag')}' (t{acct.get('tier')}, "
                                   f"{acct.get('model') or backend})")})
        return finish(payload)

    if backend == "claude":
        env = issue_dispatch.worker_env(acct.get("dir"), chosen_lane, root)
    else:
        env = opencode_worker_env(acct.get("dir"), chosen_lane, root, runs_dir)
    env["FLEET_RESOLVE_ISSUE"] = str(target)
    command = build_worker_command(backend, rec["prompt"], model)
    spawned = spawn_issue_worker(command, env, root, runs_dir, target, chosen_lane, backend,
                                 account=acct, spawn_probe_s=spawn_probe_s)
    early = spawned.get("early_exit") or {}
    if early.get("checked") and not early.get("alive") and early.get("silent"):
        payload.update({"ok": False, "action": "spawn_failed",
                        "verdict": "SPAWN_FAILED",
                        "spawned": spawned,
                        "reason": (f"{backend} worker pid {spawned['pid']} for "
                                   f"#{target} exited within {early.get('wait_s')}s "
                                   "and produced an empty log")})
        _record(runs_dir, payload)
        return finish(payload)
    payload.update({"ok": True, "action": "spawned", "verdict": "SPAWNED",
                    "spawned": spawned,
                    "reason": (f"spawned {backend} issue-resolution worker pid "
                               f"{spawned['pid']} on #{target} (lane '{chosen_lane}') "
                               f"under '{acct.get('tag')}' ({acct.get('model') or backend})")})
    _record(runs_dir, payload)
    return finish(payload)


def _record(runs_dir: Path, payload: dict[str, Any]) -> None:
    # Scope the tick record by backend so concurrent tasks (e.g. the opus
    # FleetIssueDispatch and the glm FleetIssueDispatchGlm) don't clobber each
    # other's trace; also keep the legacy unscoped file (last-writer) for any
    # manual reader that still expects it.
    backend = payload.get("backend") or "claude"
    blob = json.dumps(payload, indent=2)
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / f"last-resolve-tick-{backend}.json").write_text(blob, encoding="utf-8")
        (runs_dir / "last-resolve-tick.json").write_text(blob, encoding="utf-8")
    except OSError:
        pass


def render(p: dict[str, Any]) -> str:
    a = p.get("account") or {}
    pf = p.get("preflight") or {}
    lines = [
        f"issue-resolve-dispatch: {p.get('verdict')} ({'ok' if p.get('ok') else 'refuse'})  "
        f"backend={p.get('backend')}  live={p.get('live')}",
        f"  preflight : {pf.get('verdict')} ({pf.get('live')}/{pf.get('cap')} live)",
        f"  account   : {a.get('tag') or '-'} (t{a.get('tier')})  {a.get('model') or ''}",
        f"  lane      : {p.get('lane') or '-'}  ({p.get('lane_issue_count')} issues)",
        f"  target    : #{p.get('target_issue')}  {(p.get('issue_title') or '')[:54]}",
        f"  -> {p.get('reason')}",
    ]
    if p.get("spawned"):
        s = p["spawned"]
        lines.append(f"  spawned pid={s.get('pid')} issue=#{s.get('issue')} log={s.get('log')}")
    timed_out = p.get("timed_out_workers") or {}
    reaped = timed_out.get("reaped") or []
    would_reap = timed_out.get("would_reap") or []
    if reaped:
        lines.append(f"  reaped timed-out workers: {len(reaped)}")
    elif would_reap:
        lines.append(f"  would reap timed-out workers: {len(would_reap)}")
    if not p.get("live") and p.get("action") == "would_spawn":
        lines.append("  DRY-RUN — re-run with --live to spawn the issue worker")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="One guarded tick that spawns an issue-RESOLUTION worker (dry-run by default).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="hard cap on live workers, enforced by the preflight (default: 2)")
    ap.add_argument("--work-kind", default=None,
                    help="switcher work kind (engineering->t1, gardening->t2). Default "
                         "follows --backend: engineering for claude (t1 opus pool), "
                         "gardening for opencode/glm (t2 pool). The opencode pool has NO "
                         "t1 account, so an engineering route there REFUSE_NO_ACCOUNTs — "
                         "which silently stalled the docs lane until this default landed.")
    ap.add_argument("--lane", default=None,
                    help="explicit lane (default: the lane with the most open issues)")
    ap.add_argument("--backend", choices=BACKENDS, default="claude",
                    help="worker backend: claude (opus, t1) or opencode (glm-5.2, t2, "
                         "a separate quota pool). Default claude.")
    ap.add_argument("--exclude-lane", default="",
                    help="comma-separated lanes to drop from the busiest-pick (e.g. an "
                         "opus task excludes 'docs' so a glm task owns it). Ignored when "
                         "--lane is set.")
    ap.add_argument("--live", action="store_true",
                    help="actually spawn the worker (default: dry-run / plan only)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip the per-tick account-registry refresh")
    ap.add_argument("--cooldown-min", type=int, default=120,
                    help="skip an issue attempted within this many minutes (anti-churn; "
                         "0 disables). Default 120 — stops re-storming one un-landable issue.")
    ap.add_argument("--worker-timeout-s", type=int, default=DEFAULT_WORKER_TIMEOUT_S,
                    help=f"wall-clock cap for live resolver workers before this tick "
                         f"reaps their process tree (default: {DEFAULT_WORKER_TIMEOUT_S}; "
                         "0 disables)")
    ap.add_argument("--spawn-probe-s", type=float, default=DEFAULT_SPAWN_PROBE_S,
                    help=f"seconds to wait after a live spawn to catch immediate "
                         f"empty-log exits (default: {DEFAULT_SPAWN_PROBE_S}; 0 disables)")
    ap.add_argument("--loop-ledger", default="",
                    help="append this tick to a fak loop ledger (default: FAK_LOOP_LEDGER or .fak/loops.jsonl)")
    ap.add_argument("--no-loop-ledger", action="store_true",
                    help="disable loop-ledger append for this tick")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    exclude_lanes = {s.strip() for s in args.exclude_lane.split(",") if s.strip()}
    # The opencode/glm pool is tier 2 only; engineering routes to t1 and finds
    # nothing there. Derive the work-kind from the backend unless set explicitly.
    work_kind = args.work_kind or ("gardening" if args.backend == "opencode" else "engineering")
    payload = evaluate(root, max_workers=args.max_workers, work_kind=work_kind,
                       lane=args.lane, live=args.live, refresh=not args.no_refresh,
                       cooldown_min=args.cooldown_min, backend=args.backend,
                       exclude_lanes=exclude_lanes,
                       worker_timeout_s=dispatch_worker.normalize_timeout(args.worker_timeout_s),
                       spawn_probe_s=max(0.0, args.spawn_probe_s),
                       record_loop=not args.no_loop_ledger,
                       loop_ledger=(Path(args.loop_ledger).resolve()
                                    if args.loop_ledger else None))
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
