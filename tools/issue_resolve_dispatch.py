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

Pre-spawn lane-lease gate (#1310): it also HOLDS the target lane before launching
rather than trusting the worker to self-arbitrate. fak's taxonomy is one worker
per leaf tree, so a second worker on a lane that already has a live one co-edits
the same files. The auto-pick reroutes around held lanes (busiest FREE lane); an
explicitly-named lane that is held is refused ``LANE_BUSY`` (COLLISION_RISK)
instead of raced. This is the upstream half of the verified loop — deny the
collision by structure, *before* the spawn — paired with the downstream
commit-time closure audit (the ``#N``-in-subject witness, still the close arm).

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

# Re-export the shared console-window suppressor so the account-topup / glm-docs
# entry scripts (which import THIS module as `ird`) route every helper subprocess
# through the one canonical guard. See dispatch_worker.no_window_creationflags.
no_window_creationflags = dispatch_worker.no_window_creationflags

SCHEMA = "fleet-issue-resolve-dispatch/1"
RUNS_DIRNAME = ".dispatch-runs"
# Per-worker membership sidecar (rank/wave/size/shortfall), written next to the
# .pid/.backend sidecars so an auditor enumerates a wave straight from disk.
WAVE_SIDECAR_SUFFIX = ".wave"
ACCOUNT_SIDECAR_SUFFIX = ".account"
# Fenced lane-lease (residual of #1310): the dispatcher ATOMICALLY acquires
# refs/fak/locks/resolve-<lane> via `fak leaseref acquire` before launching, so a
# tick on ANOTHER machine (and a same-host TOCTOU race the log-scan can't close)
# refuses the collision instead of co-editing the leaf tree. The same-host log
# scan (live_resolution_lanes) stays the fast path UNDER this lease. The lease id
# is one safe ref segment; fak's lane names are already that shape.
LEASE_ID_PREFIX = "resolve-"
# A held lease outlives its worker by this margin past the wall-clock worker
# timeout, so a crashed worker that never releases is reaped by `fak leaseref
# reap` on a later tick rather than wedging the lane forever. Generous: the TTL is
# a backstop, not the primary release (that is the witnessed release on exit).
LEASE_TTL_MARGIN_S = 600
_LOG_ISSUE_RE = re.compile(r"resolve-(\d+)-")
# The `lane=<L>` field of the `# fak-spawn` header `spawn_issue_worker` flushes as
# the first line of every worker log (used by the pre-spawn lane-lease gate, #1310).
_SPAWN_LANE_RE = re.compile(r"\blane=(\S+)")
_RID_RE = re.compile(r"^RID-[A-Z0-9]+$")
DEFAULT_WORKER_TIMEOUT_S = dispatch_worker.DEFAULT_TIMEOUT_S
DEFAULT_SPAWN_PROBE_S = 5.0
# Hard ceiling on how many extra worker slots a healthy backend may claim from dead
# siblings in one tick — bounds the blast radius so a transient mass-death can't blow
# the healthy backend's cap past what its account can actually serve.
DEFAULT_REALLOC_CEILING = 2
LOOP_ID_PREFIX = "issue-resolve-dispatch"

# Worker backends this tick can launch:
#   claude   = opus (t1) -- the reference path, the established quota pool.
#   opencode = glm-5.2 via the zai-coding-plan accounts (t2) -- a SEPARATE quota
#              pool that sits idle. Routing a lane (e.g. docs, where glm is proven)
#              to opencode relieves the opus weekly-quota throughput ceiling.
BACKENDS = ("claude", "opencode", "codex")
_BACKEND_PRODUCT = {"claude": "claude", "opencode": "opencode", "codex": "codex"}


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


def _spawn_header_lane(log: Path) -> str | None:
    """Parse ``lane=<L>`` from a worker log's ``# fak-spawn`` header — its first
    line, flushed before exec by :func:`spawn_issue_worker`. Returns ``None`` when
    the log is unreadable or has no recoverable lane field (best effort)."""
    try:
        with open(log, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.readline()
    except OSError:
        return None
    if not head.startswith("# fak-spawn"):
        return None
    m = _SPAWN_LANE_RE.search(head)
    return m.group(1) if m else None


def live_resolution_lanes(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> set[str]:
    """Lanes that already hold a LIVE resolution worker — the pre-spawn collision
    set (#1310). fak's lane taxonomy is ONE worker per leaf tree (``dos.toml``: a
    named lane serializes even on disjoint sub-trees), so a second worker on a
    held lane co-edits the same files. Read the ``lane=`` field the spawn header
    stamps into each ``resolve-<N>-<stamp>.log`` and keep only the lanes whose
    worker pid is still alive — the SAME identity gate as
    :func:`live_resolution_issues`. Best effort: a log we cannot parse, or whose
    pid is gone, contributes no lane."""
    lanes: set[str] = set()
    if not runs_dir.is_dir():
        return lanes
    if alive is None and probe is None:
        try:
            import psutil  # type: ignore
            alive = {p.pid for p in psutil.process_iter()}
        except ImportError:
            alive = None
    for log in runs_dir.glob("resolve-*.log"):
        if not _LOG_ISSUE_RE.search(log.name):
            continue
        pid_file = log.with_suffix(".pid")
        if not pid_file.exists():
            continue
        if not dispatch_preflight.resolve_sidecar_pid_is_live(
                pid_file, alive=alive, probe=probe):
            continue
        lane = _spawn_header_lane(log)
        if lane:
            lanes.add(lane)
    return lanes


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
                                  capture_output=True, text=True, timeout=30,
                                  creationflags=no_window_creationflags())
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


def prune_dead_sidecars(
    runs_dir: Path,
    *,
    live: bool,
    min_age_s: float = 60.0,
    now_ts: float | None = None,
    probe: Any | None = None,
) -> dict[str, Any]:
    """Sweep ``resolve-*.pid`` sidecars whose worker is no longer alive.

    A worker that exits NORMALLY leaves its ``.pid`` (and sibling ``.log`` /
    ``.backend`` / ``.wave`` / ``.account``) sidecars behind forever — the reaper
    only KILLS live runaways, it never deletes the corpses. Left alone they pile
    up (379 were found on one host) and become landmines for the create-time
    spawn-window liveness fallback: a recycled PID landing in a stale sidecar's
    window was miscounted as a live worker, pinning the dispatcher at cap against
    ghosts. Sweeping every tick keeps the worker count the preflight reads honest
    and the directory bounded.

    A sidecar is pruned only when its identity-gated liveness check says the
    process is gone, and only once it is at least ``min_age_s`` old (so a sidecar
    written microseconds ago, before its child has fully materialised, is never
    swept out from under a just-spawned worker). Dry-run reports ``would_prune``;
    ``live`` actually unlinks. The matching non-``.pid`` sidecars are removed with
    the ``.pid`` so a half-pruned run never confuses the wave auditor.
    """
    import time
    now = now_ts if now_ts is not None else time.time()
    pruned: list[str] = []
    would_prune: list[str] = []
    if not runs_dir.is_dir():
        return {"live": live, "pruned": pruned, "would_prune": would_prune}
    sibling_suffixes = (".log", ".backend", ".wave", ".account")
    for pid_file in sorted(runs_dir.glob("resolve-*.pid")):
        if not _LOG_ISSUE_RE.search(pid_file.name):
            continue
        try:
            age_s = max(0.0, now - pid_file.stat().st_mtime)
        except OSError:
            continue
        if age_s < min_age_s:
            continue
        if dispatch_preflight.resolve_sidecar_pid_is_live(pid_file, probe=probe):
            continue
        if not live:
            would_prune.append(pid_file.name)
            continue
        stem = pid_file.with_suffix("")
        try:
            pid_file.unlink()
            pruned.append(pid_file.name)
        except OSError:
            continue
        for suf in sibling_suffixes:
            sib = stem.with_suffix(suf)
            try:
                sib.unlink()
            except OSError:
                pass
    return {"live": live, "pruned": pruned, "would_prune": would_prune}


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
    if backend == "codex":
        # `codex exec` is the headless analogue of `claude -p` / `opencode run`;
        # --dangerously-bypass-approvals-and-sandbox is the full-access mode (we run
        # in the repo, already externally bounded by fak guard + the trunk guard).
        # --skip-git-repo-check keeps it from refusing in a worktree edge case.
        cmd = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
               "--skip-git-repo-check"]
        if model:
            cmd += ["-m", model]
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
                           encoding="utf-8", errors="replace",
                           creationflags=no_window_creationflags())
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


def codex_worker_env(account_dir: str | None, lane: str, workspace: Path) -> dict[str, str]:
    """Child env for a codex (`codex exec`) worker. Codex authenticates from the
    ambient ``~/.codex`` (one ChatGPT login) rather than the multi-account switcher,
    so there is no per-account dir to pin — just clear the claude-only vars and stamp
    the self-describing dispatch vars. When a per-account CODEX_HOME dir IS supplied
    (future multi-account codex), honor it so the run is attributed to that account."""
    env = dispatch_worker.child_env(lane, "codex", workspace)
    env.pop("CLAUDE_CONFIG_DIR", None)
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    if account_dir:
        env["CODEX_HOME"] = account_dir
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
        # Flush a one-line spawn header BEFORE exec so a later 0-byte log means
        # "died before exec" (the OS never ran the child) — distinguishable from
        # "spawned, exec'd, then wrote nothing". The child inherits this fd and
        # appends after the header. Kept short (well under _STUB_LOG_MAX_BYTES) and
        # banner-free so it never trips the stub/cap-banner classifiers.
        fh.write("# fak-spawn %s issue=%s lane=%s backend=%s argv0=%s\n" % (
            stamp, issue, lane, backend, os.path.basename(exe)))
        fh.flush()
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
# The codex (OpenAI/ChatGPT) backend hits its own quota wall with a different banner
# than Claude's: "You've hit your usage limit. Visit https://chatgpt.com/codex/...
# purchase more credits or try again at Jul 1st, 2026 8:41 PM." It matches the phrase
# regex above ("hit your usage limit"), but its reset clause is "try again at <date>"
# (no "(America/Los_Angeles)" suffix), so the Claude reset parsers below miss it and
# the hold would fall back to the short cooldown — re-spawning doomed codex workers
# until the real (Jul-1) reset. _CODEX_RESET_RE recovers the codex reset moment so the
# hold runs to the real wall instead.
_CODEX_RESET_RE = re.compile(
    r"try again at\s+([A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\s+\d{1,2}(?::\d{2})?\s*(?:am|pm))",
    re.IGNORECASE)
# A SESSION limit is a short rolling window (resets in hours); a WEEKLY limit is a
# multi-day quota wall. They share the "hit your … limit" banner but must NOT share
# the hold: treating a transient session limit as a weekly hold — and projecting a
# bare time-of-day reset that already passed a full day forward — falsely walls an
# account that actually has room (the gem8 false-cap). Classify the banner so a
# session limit gets a short cooldown bounded to its real reset.
_CAP_WEEKLY_RE = re.compile(r"weekly\s+limit", re.IGNORECASE)
_CAP_SESSION_RE = re.compile(r"session\s+limit", re.IGNORECASE)
_CAP_RESET_RE = re.compile(r"resets\s+(.+?)\s*\(America/Los_Angeles\)", re.IGNORECASE)
_CAP_RESET_FALLBACK_RE = re.compile(r"resets\s+([^\r\n]+)", re.IGNORECASE)
# A session-limit hold never exceeds this even if its reset clause parses to a far
# future moment (a stale, already-passed bare time-of-day must not become a ~24h
# wall). Weekly limits keep the full parsed reset.
_SESSION_HOLD_MAX_MIN = 90
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


# How many trailing bytes of a worker log to inspect for the quota banner. A walled
# worker emits the banner as the LAST thing it writes, so the tail is where it lives;
# bounding the read keeps a multi-MB productive log cheap to scan and stops its early
# agent prose from false-matching the phrase.
_CAP_TAIL_BYTES = 4096


def _log_tail_text(log: Path, *, nbytes: int = _CAP_TAIL_BYTES) -> str:
    """The last ``nbytes`` of ``log`` decoded as utf-8 (errors replaced), or "" on
    any read error. Used to inspect a log's ending for the quota banner without
    reading the whole (possibly large) file."""
    try:
        size = log.stat().st_size
        with log.open("rb") as fh:
            if size > nbytes:
                fh.seek(-nbytes, os.SEEK_END)
            raw = fh.read()
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


def _log_is_cap_banner(log: Path) -> bool:
    """True when ``log``'s tail is the quota-limit banner (Claude or codex). Lets the
    backend-health classifier treat a credit-walled worker as a stub regardless of
    byte size — a codex wall log clears the size floor on its startup banner alone."""
    return bool(_CAP_BANNER_RE.search(_log_tail_text(log)))


def _scan_recent_cap_banner(runs_dir: Path, *, product: str, lookback_min: int,
                            now_ts: float) -> dict[str, Any] | None:
    """The most-recent worker log (this backend, mtime within ``lookback_min``)
    whose TAIL is the quota-limit banner → ``{reset_text, evidence_log}``, else None.
    Scanning only the tail (not the whole file, and not size-gated) catches a codex
    wall log — which clears any byte floor on its ~700-byte startup banner before the
    error — while keeping a real multi-MB worker log cheap to inspect; the specific
    phrase still keeps a prose log that merely mentions "limit" mid-run from
    false-matching, because the banner only appears as the worker's final output."""
    if not runs_dir.is_dir():
        return None
    horizon = now_ts - lookback_min * 60
    best: tuple[float, str, str, str] | None = None
    for log in runs_dir.glob("resolve-*.log"):
        try:
            st = log.stat()
        except OSError:
            continue
        if st.st_mtime < horizon:
            continue
        if _backend_of_log(log) != product:
            continue
        text = _log_tail_text(log)
        if not _CAP_BANNER_RE.search(text):
            continue
        # Codex's "try again at <date>" reset wins when present (it has no
        # "(America/Los_Angeles)" clause the Claude parsers key on); otherwise fall
        # back to the Claude reset clauses.
        m = (_CODEX_RESET_RE.search(text) or _CAP_RESET_RE.search(text)
             or _CAP_RESET_FALLBACK_RE.search(text))
        reset_text = m.group(1).strip() if m else ""
        # weekly is the conservative (longer) hold; default to weekly only when the
        # banner explicitly says so, treat an explicit "session limit" as session,
        # and an unlabelled "limit" as session too (the safer short hold — a real
        # weekly cap will re-banner and upgrade on the next doomed spawn).
        if _CAP_WEEKLY_RE.search(text):
            kind = "weekly"
        elif _CAP_SESSION_RE.search(text):
            kind = "session"
        else:
            kind = "session"
        if best is None or st.st_mtime > best[0]:
            best = (st.st_mtime, reset_text, log.name, kind)
    if best is None:
        return None
    return {"reset_text": best[1], "evidence_log": best[2], "kind": best[3]}


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
            kind = hit.get("kind") or "session"
            until = (_parse_reset_to_utc(hit["reset_text"], now_utc)
                     or now_utc + dt.timedelta(minutes=fallback_min))
            # A session limit is a short rolling window — never hold longer than
            # _SESSION_HOLD_MAX_MIN even if its bare time-of-day reset parsed to a
            # far-future moment (the stale-banner-pushed-a-day-forward false cap).
            # A weekly limit keeps the full parsed reset.
            if kind == "session":
                session_cap = now_utc + dt.timedelta(minutes=_SESSION_HOLD_MAX_MIN)
                until = min(until, session_cap)
            state = {"product": product, "account": account_tag, "kind": kind,
                     "reset_text": hit["reset_text"], "evidence_log": hit["evidence_log"],
                     "detected": now_utc.isoformat() + "Z", "until": until.isoformat() + "Z"}
            try:
                runs_dir.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            except OSError:
                pass
            return {"capped": True, "until": state["until"], "kind": kind,
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


# --- backend-health reallocation gate -----------------------------------------
# The weekly-cap gate above catches ONE dead-backend shape: the explicit "hit your
# limit" banner. But a backend can spin just as silently WITHOUT that banner — a
# codex spawn that dies credit-walled (0-byte log) or a glm/opencode worker that
# prints only its startup banner ("> build · glm-5.2", ~32 bytes) and exits without
# a turn. Those pass the 5s early-exit probe (alive then) and slip the cap regex, so
# the dispatcher keeps feeding the backend budget while it ships nothing — and the
# lane it owned (e.g. docs, the biggest backlog) is abandoned rather than handed to
# a backend that IS producing turns.
#
# check_backend_health() is the sibling gate, shaped exactly like check_weekly_cap:
# a fast on-disk signal (a STREAK of stub logs for this backend) declares it DEAD,
# persisted in backend-health-<product>.json with the same write/honor/expire and
# FAIL-OPEN discipline (any error -> healthy, so the gate can only ever ADD a hold).
# The dead backend self-suppresses (BACKEND_UNHEALTHY) but lets ONE re-probe worker
# through per interval so it can prove recovery; the healthy backend reads the
# sidecar and claims the freed lane + budget (see evaluate()).

# A worker log at/under this size produced no real turn: a 0-byte credit-wall death
# or a banner-only (~32-byte) exit. A real turn streams kilobytes. Matches the size
# floor the weekly-cap banner scanner already trusts (_scan_recent_cap_banner: 512).
_STUB_LOG_MAX_BYTES = 512
# DEAD only on a sustained streak — the last N attempted spawns for this backend ALL
# stubs — never a single blip (a real worker can rarely exit early for benign
# reasons). N is small so the gate reacts within a few ticks, not hours.
_BACKEND_DEAD_STREAK = 3
# How long a DEAD hold suppresses spawns before letting one re-probe worker through.
# A credit wall lifts on its own (codex resets nightly); the re-probe is how the
# backend earns its budget back without an operator.
_BACKEND_REPROBE_MIN = 30


def _classify_backend_logs(runs_dir: Path, *, product: str, lookback_min: int,
                           now_ts: float, alive: set[int] | None,
                           probe: Any | None) -> list[dict[str, Any]]:
    """The recent resolve logs for ONE backend, newest first, each tagged
    ``{"stamp", "productive", "size"}``. Productive = a log over the real-turn floor;
    a stub (<= floor) only counts as evidence of death when its pid is provably dead
    (a still-running claude worker writes nothing until its final message, so a live
    0-byte log is NOT a stub — reuses the silent_workers dead-pid guard)."""
    out: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return out
    horizon = now_ts - lookback_min * 60
    for log in runs_dir.glob("resolve-*.log"):
        m = _LOG_ISSUE_RE.search(log.name)
        if not m:
            continue
        try:
            st = log.stat()
        except OSError:
            continue
        if st.st_mtime < horizon:
            continue
        if _backend_of_log(log) != product:
            continue
        # Size alone is not productivity. A codex credit-wall log carries a ~700-byte
        # startup banner (workdir/model/session id/hook lines) BEFORE the quota error,
        # so it clears the byte floor while landing zero real turns. A log whose body
        # is the quota banner is never a real turn regardless of size -> read the tail
        # and treat a _CAP_BANNER_RE hit as a stub, so a credit-walled codex worker
        # joins the dead-streak instead of falsely breaking it.
        productive = st.st_size > _STUB_LOG_MAX_BYTES and not _log_is_cap_banner(log)
        if not productive:
            # Only a DEAD pid over a stub log is evidence of a doomed spawn; a live
            # one is still running (claude streams nothing until the end).
            pid_file = log.with_suffix(".pid")
            if pid_file.exists() and dispatch_preflight.resolve_sidecar_pid_is_live(
                    pid_file, alive=alive, probe=probe):
                continue  # still running -> not (yet) evidence either way
        out.append({"stamp": st.st_mtime, "productive": productive, "size": st.st_size,
                    "log": log.name})
    out.sort(key=lambda r: r["stamp"], reverse=True)
    return out


def check_backend_health(runs_dir: Path, *, product: str, lane: str | None = None,
                         now_ts: float | None = None, lookback_min: int = 90,
                         alive: set[int] | None = None,
                         probe: Any | None = None) -> dict[str, Any]:
    """Is ``product`` producing real turns right now, or spinning dead? Reads the
    backend's recent worker logs: the last ``_BACKEND_DEAD_STREAK`` attempts all
    stubs (banner-only/0-byte over a dead pid) -> DEAD; any productive log in the
    window breaks the streak -> HEALTHY (and clears a stale hold). Persists a hold in
    ``backend-health-<product>.json`` so it survives the ticks after spawns stop, and
    gates ONE re-probe spawn per ``_BACKEND_REPROBE_MIN`` so a dead backend can earn
    its budget back. Returns ``{state, ...}``. FAIL-OPEN: any error -> healthy."""
    try:
        import time
        now_ts = time.time() if now_ts is None else now_ts
        now_utc = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=now_ts)
        state_path = runs_dir / f"backend-health-{product}.json"
        if alive is None and probe is None:
            try:
                import psutil  # type: ignore
                alive = {p.pid for p in psutil.process_iter()}
            except ImportError:
                alive = None  # cannot prove a pid dead -> classify no stubs (no false death)
        logs = _classify_backend_logs(runs_dir, product=product, lookback_min=lookback_min,
                                      now_ts=now_ts, alive=alive, probe=probe)
        recent = logs[:_BACKEND_DEAD_STREAK]
        # A productive log anywhere in the window means the backend HAS landed a turn
        # recently -> healthy, clear any stale hold (the witnessed restore: a real
        # turn is the slow-layer confirmation a re-probe worked).
        if any(r["productive"] for r in logs):
            if state_path.exists():
                try:
                    state_path.unlink()
                except OSError:
                    pass
            return {"state": "healthy", "source": "productive"}
        dead = (len(recent) >= _BACKEND_DEAD_STREAK
                and all(not r["productive"] for r in recent))
        if not dead:
            return {"state": "healthy", "source": "no-streak"}
        # DEAD. Honor an existing hold (keep its since/lane); otherwise open one.
        prior: dict[str, Any] = {}
        if state_path.exists():
            try:
                prior = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                prior = {}
        since = prior.get("since") if prior.get("state") == "dead" else None
        since = since or (now_utc.isoformat() + "Z")
        # The lane this backend would otherwise own — recorded so the healthy tick can
        # claim it. Keep a prior non-empty lane if this tick didn't resolve one.
        abandoned = lane or prior.get("abandoned_lane") or ""
        last_probe = _iso_to_utc(prior.get("last_reprobe") or "") if prior else None
        due = (last_probe is None
               or now_utc - last_probe >= dt.timedelta(minutes=_BACKEND_REPROBE_MIN))
        state = {
            "product": product, "state": "dead", "since": since,
            "abandoned_lane": abandoned,
            "evidence_logs": [r["log"] for r in recent],
            "detected": now_utc.isoformat() + "Z",
            "last_reprobe": (now_utc.isoformat() + "Z") if due else prior.get("last_reprobe"),
            "reprobe_min": _BACKEND_REPROBE_MIN,
        }
        try:
            runs_dir.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except OSError:
            pass
        # reprobe_due: caller (the DEAD backend's own tick) may let ONE worker through
        # to test recovery. We stamped last_reprobe above so the next tick holds again.
        return {"state": "dead", "since": since, "abandoned_lane": abandoned,
                "reprobe_due": bool(due), "evidence_logs": state["evidence_logs"],
                "source": "streak"}
    except Exception:
        return {"state": "healthy", "source": "error"}


def read_dead_backends(runs_dir: Path, *, exclude: str | None = None,
                       now_ts: float | None = None) -> list[dict[str, Any]]:
    """Sibling backends currently held DEAD, read from backend-health-*.json — the
    healthy tick's view of what budget/lane is free to claim. ``exclude`` drops the
    caller's own product. Read-only / best-effort (a corrupt sidecar is skipped)."""
    out: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return out
    for path in runs_dir.glob("backend-health-*.json"):
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if st.get("state") != "dead":
            continue
        if exclude and st.get("product") == exclude:
            continue
        out.append(st)
    return out


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


# --- Fenced lane-lease gate (residual of #1310) ------------------------------
# The same-host log-scan gate (live_resolution_lanes) serializes two ticks ON ONE
# HOST run sequentially, but it is blind across machines and has a TOCTOU window
# (the lane= header is written near the END of a tick; two evaluate() processes
# racing both read "lane free"). These helpers close both gaps by holding the
# fenced refs/fak/locks/resolve-<lane> lease via `fak leaseref acquire` — an
# atomic update-ref compare-and-swap that rides ordinary git fetch/push, so a peer
# on another clone SEES the lease after a fetch. The log scan stays the fast path;
# this is the authoritative pre-spawn admission. EVERYTHING here is FAIL-OPEN: any
# error (no fak binary, a broken store, an unparseable verdict) resolves to "not
# held / acquired" so a lease-layer fault can only ever DROP the extra protection,
# never wedge the loop — the same discipline as check_weekly_cap.

# A structured fence refusal (LEASE_HELD / LEASE_CONTENDED / STALE_LEASE /
# NO_LEASE) exits 3 from `fak leaseref` (cmd/fak/leaseref.go: leaserefRefused); 0
# is acquired, 1 a git/store failure, 2 a usage error. We fence-refuse (fail
# CLOSED) only on 3, and fail OPEN on everything else.
LEASE_REFUSED_RC = 3


def _fak_bin(root: Path) -> list[str] | None:
    """The on-disk `fak` argv for a lease subprocess, or None when only the
    `go run ./cmd/fak` fallback is available. We DELIBERATELY refuse the go-run
    fallback for the lease path: building cmd/fak on a peer-dirty shared trunk
    routinely fails (the cmd-lane-undispatchable law), and a lease gate that
    shells a 30s flaky `go run` every tick would be worse than no gate. With no
    FAK_BIN the gate fails open (the same-host log scan still applies)."""
    configured = os.environ.get("FAK_BIN")
    if configured:
        return shlex.split(configured)
    exe = shutil.which("fak")
    return [exe] if exe else None


def lease_holder() -> str:
    """A holder identity stable across this session's separate dispatcher ticks.
    Mirrors dos_fleet_lease.default_owner: the harness sets CLAUDE_CODE_SESSION_ID
    once per session; fall back to host:pid so two unrelated sessions never share a
    holder (a shared holder would let one RENEW the other's lease)."""
    sid = os.environ.get("FAK_LEASE_OWNER") or os.environ.get("CLAUDE_CODE_SESSION_ID")
    if sid:
        return sid.strip()
    host = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "host"
    return f"{host}:{os.getpid()}"


def lease_id_for_lane(lane: str) -> str:
    return f"{LEASE_ID_PREFIX}{lane}"


def _run_lease(root: Path, args: list[str], *, timeout: int = 30) -> dict[str, Any]:
    """Run one `fak leaseref` subcommand, returning {rc, stdout, verdict}. Never
    raises: an exec failure is reported as rc 127 so the caller fails open."""
    bin_argv = _fak_bin(root)
    if not bin_argv:
        return {"rc": 127, "stdout": "", "verdict": None, "skipped": "no fak binary"}
    cmd = [*bin_argv, "leaseref", *args]
    kwargs: dict[str, Any] = {
        "cwd": str(root), "capture_output": True, "text": True,
        "encoding": "utf-8", "errors": "replace", "timeout": timeout,
    }
    if os.name == "nt":
        kwargs["creationflags"] = no_window_creationflags()
    try:
        proc = subprocess.run(cmd, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"rc": 127, "stdout": "", "verdict": None, "error": str(exc)}
    out = (proc.stdout or "").strip()
    doc: Any = None
    try:
        doc = json.loads(out) if out else None
    except ValueError:
        doc = None
    return {"rc": proc.returncode, "stdout": out, "verdict": doc}


def acquire_lane_lease(root: Path, lane: str, *, tree: list[str], ttl_s: int,
                       holder: str | None = None,
                       runner: Any | None = None) -> dict[str, Any]:
    """ATOMICALLY take the fenced refs/fak/locks/resolve-<lane> lease before a
    spawn. Returns {"acquired": bool, "refused": bool, "id", "holder", ...}.

    - acquired=True  -> exit 0: we hold the lease; carry the id+holder so a later
                        tick can renew/reap it. There is no early-release verb yet
                        (`fak leaseref` exposes only acquire/reap), so the lane is
                        freed by TTL + reap, not an explicit release on worker exit.
    - refused=True   -> exit 3: a LIVE peer (this host OR another clone after a
                        fetch) holds the lane; the caller refuses LANE_LEASE_HELD.
    - acquired=False, refused=False -> FAIL OPEN: no fak binary / a git-store
                        failure / an unparseable verdict; the caller proceeds on
                        the same-host log scan alone, exactly as before the lease.
    """
    run = runner or _run_lease
    holder = holder or lease_holder()
    lease_id = lease_id_for_lane(lane)
    args = ["acquire", "--id", lease_id, "--holder", holder, "--ttl", str(int(ttl_s))]
    for t in tree:
        args += ["--tree", t]
    res = run(root, args)
    rc = res.get("rc")
    verdict = res.get("verdict") or {}
    if rc == 0:
        rec = verdict.get("record") if isinstance(verdict, dict) else None
        return {"acquired": True, "refused": False, "id": lease_id, "holder": holder,
                "generation": (rec or {}).get("generation") if isinstance(rec, dict) else None,
                "tree": tree}
    if rc == LEASE_REFUSED_RC:
        v = verdict.get("verdict") if isinstance(verdict, dict) else None
        return {"acquired": False, "refused": True, "id": lease_id, "holder": holder,
                "reason": (v or {}).get("reason") if isinstance(v, dict) else None,
                "fence_verdict": v, "tree": tree}
    # rc in {1,2,127} or unparseable -> fail open (no protection added, loop never wedges).
    return {"acquired": False, "refused": False, "id": lease_id, "holder": holder,
            "fail_open": True, "rc": rc, "detail": res.get("error") or res.get("skipped"),
            "tree": tree}


def reap_expired_leases(root: Path, *, runner: Any | None = None) -> dict[str, Any]:
    """Delete expired (reapable) refs/fak/locks/* leases — the crashed-holder
    backstop. A worker (or a same-tick spawn that died at exec) that never gets to
    drop its lane lease has it swept here once the TTL lapses, so a crash can't
    wedge a lane forever. Fail-open: a reap failure is reported, never raised.

    NOTE: there is deliberately no early-release helper here. `fak leaseref` exposes
    only acquire/reap (no delete-one-live-lease verb), so a held lane is freed by
    TTL+reap, not an explicit release. Binding the release to a worker's
    diff-witness (proposal #2 of #1324) is a separate increment that first needs
    that release verb and per-worker commit-sha tracking — see the issue residual."""
    run = runner or _run_lease
    res = run(root, ["reap"])
    return {"ok": res.get("rc") == 0, "rc": res.get("rc"), "stdout": res.get("stdout")}


# A per-process cache of the dos.toml lane->trees map so a tick that probes several
# lanes shells `dos doctor` at most once. Keyed by workspace string.
_LANE_TREE_CACHE: dict[str, dict[str, list[str]]] = {}


def lane_tree(root: Path, lane: str) -> list[str]:
    """The repo-relative file-tree globs the fenced lease should cover for `lane`,
    from dos.toml's [lanes.trees] (reused via issue_lane_router.lane_taxonomy), with
    an `internal/<lane>/**` fallback. The lease tree only matters for the arbiter's
    disjointness reasoning; the lease id (resolve-<lane>) is what actually
    serializes. Fail-open: a `dos doctor` failure falls back to the convention so
    the gate still acquires SOMETHING rather than refusing to protect."""
    key = str(root)
    trees = _LANE_TREE_CACHE.get(key)
    if trees is None:
        trees = {}
        try:
            import issue_lane_router  # noqa: PLC0415  (lazy: keep the import optional)
            _concurrent, trees = issue_lane_router.lane_taxonomy(root)
        except Exception:  # noqa: BLE001  (fail-open: any failure -> convention fallback)
            trees = {}
        _LANE_TREE_CACHE[key] = trees
    globs = trees.get(lane)
    if globs:
        return list(globs)
    return [f"internal/{lane}/**"]


def is_dos_run_id(run_id: object) -> bool:
    return bool(_RID_RE.fullmatch(str(run_id or "")))


def mint_dos_run_id(root: Path, process: str, parent: str | None = None) -> str | None:
    """Mint the DOS RID that `dos status` accepts, fail-open to caller fallback."""
    cmd = ["dos", "run-id", "mint", process, "--root", str(root)]
    if parent:
        cmd += ["--parent", parent]
    kwargs: dict[str, Any] = {
        "cwd": root,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 30,
    }
    if os.name == "nt":
        kwargs["creationflags"] = no_window_creationflags()
    try:
        proc = subprocess.run(cmd, **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        doc = json.loads(proc.stdout)
    except ValueError:
        return None
    rid = str(doc.get("run_id") or "")
    return rid if proc.returncode == 0 and is_dos_run_id(rid) else None


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
                              encoding="utf-8", errors="replace", timeout=60,
                              creationflags=no_window_creationflags())
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "kind": event.get("kind"), "error": str(exc)}
    return {
        "ok": proc.returncode == 0,
        "kind": event.get("kind"),
        "returncode": proc.returncode,
        "stderr": (proc.stderr or "").strip()[-500:],
        "stdout": (proc.stdout or "").strip()[-500:],
    }


def loop_run_id(payload: dict[str, Any], root: Path | None = None,
                mint: Any | None = None) -> str:
    existing = str(payload.get("run_id") or "")
    if is_dos_run_id(existing):
        return existing
    backend = str(payload.get("backend") or "claude").strip() or "claude"
    if root is not None:
        minted = (mint or mint_dos_run_id)(root, f"issue-resolve-dispatch-{backend}")
        if is_dos_run_id(minted):
            return str(minted)
    if existing:
        return existing
    spawned = payload.get("spawned") or {}
    if spawned.get("pid"):
        return f"resolve-{payload.get('target_issue') or 'none'}-{spawned.get('pid')}"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"resolve-tick-{backend}-{stamp}"


def record_loop_tick(root: Path, payload: dict[str, Any],
                     *,
                     ledger: Path | None = None,
                     append: Any | None = None,
                     mint: Any | None = None) -> dict[str, Any]:
    """Lower one issue-resolve dispatcher tick into loop ledger events.

    The rows describe the DISPATCHER tick, not the spawned worker's eventual issue
    resolution. A successful spawn is therefore `claimed_done` for the tick only;
    the independent commit/issue witness remains the close/audit arm.
    """
    ledger = ledger or default_loop_ledger(root)
    append = append or append_loop_event
    run_id = str(loop_run_id(payload, root=root, mint=mint))
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
              realloc_ceiling: int = DEFAULT_REALLOC_CEILING,
              record_loop: bool = False,
              loop_ledger: Path | None = None,
              lease_runner: Any | None = None) -> dict[str, Any]:
    # lease_runner is the injectable `fak leaseref` seam (default: a real
    # subprocess). Tests pass a canned runner returning {rc, verdict} so the whole
    # acquire/refuse/release path runs with no real git and no real fak binary.
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
    # Sweep the dead sidecars the reaper leaves behind, BEFORE preflight counts
    # capacity — otherwise stale `.pid` files accumulate and a recycled PID landing
    # in one's spawn window is miscounted as a live worker, pinning the cap.
    pruned = prune_dead_sidecars(runs_dir, live=live)
    # Sweep EXPIRED fenced lane leases (residual of #1310) — the crashed-holder
    # backstop. A worker that died without releasing its refs/fak/locks/resolve-<lane>
    # lease has it deleted here once its TTL lapses, so a crash can't wedge a lane
    # forever. Live ticks only (a dry-run never mutates lease state); fail-open.
    leases_reaped = reap_expired_leases(root, runner=lease_runner) if live else {"skipped": True}

    # Backend-health reallocation (the cross-read half). A HEALTHY backend claims the
    # budget + lane a DEAD sibling abandoned: it raises its effective cap by one slot
    # per dead sibling (bounded by realloc_ceiling) and prefers the freed lane in the
    # busiest-pick. Fail-open: no sibling held dead -> realloc is a no-op and behavior
    # is exactly as before. The DEAD backend's own self-suppress gate is below, after
    # preflight/weekly-cap, so a dead backend never also tries to claim from itself.
    own_health = check_backend_health(runs_dir, product=product, lane=lane)
    realloc: dict[str, Any] = {"claimed_lanes": [], "bonus": 0, "from": []}
    eff_max_workers = max_workers
    if own_health.get("state") != "dead":
        dead_siblings = read_dead_backends(runs_dir, exclude=product)
        if dead_siblings:
            bonus = min(len(dead_siblings), max(0, realloc_ceiling))
            eff_max_workers = max_workers + bonus
            realloc = {
                "claimed_lanes": [d.get("abandoned_lane") for d in dead_siblings
                                  if d.get("abandoned_lane")],
                "bonus": bonus,
                "from": [d.get("product") for d in dead_siblings],
            }

    pre = issue_dispatch.preflight(root, max_workers=eff_max_workers, work_kind=work_kind,
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
            "timed_out_workers": reaped, "pruned_sidecars": pruned,
            "preflight": {"verdict": pre.get("verdict"), "reason": pre.get("reason"),
                          "cap": pre.get("cap"), "live": pre.get("live")},
            "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
            "weekly_cap": cap, "ok": False, "action": "weekly_capped",
            "verdict": "WEEKLY_CAPPED",
            "reason": (f"account '{acct.get('tag')}' is weekly-capped (resets "
                       f"{cap.get('reset_text') or '?'}); holding spawn until "
                       f"{cap.get('until')} — re-arm is automatic at the reset"),
        })

    # Backend-health gate (the self-suppress half) — AFTER weekly-cap, same shape. If
    # THIS backend is spinning dead (a streak of banner-only/0-byte deaths the cap
    # regex misses), hold the spawn so we stop feeding budget to a doomed worker. But
    # let ONE re-probe worker through per interval (reprobe_due) so the backend can
    # earn its budget back the moment it recovers — a fully-held backend can never
    # come back. Fail-open: check_backend_health returns dead only on positive
    # streak evidence. (own_health was read above for the realloc no-op decision.)
    if pre_ok and own_health.get("state") == "dead" and not own_health.get("reprobe_due"):
        return finish({
            "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
            "max_workers": max_workers, "registry_refresh": reg,
            "timed_out_workers": reaped, "pruned_sidecars": pruned,
            "preflight": {"verdict": pre.get("verdict"), "reason": pre.get("reason"),
                          "cap": pre.get("cap"), "live": pre.get("live")},
            "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
            "backend_health": own_health, "ok": False, "action": "backend_unhealthy",
            "verdict": "BACKEND_UNHEALTHY",
            "reason": (f"backend '{backend}' is spinning dead (banner-only/0-byte "
                       f"streak since {own_health.get('since')}); holding spawn — its "
                       f"lane '{own_health.get('abandoned_lane') or '?'}' is reallocated "
                       f"to a healthy backend, and one re-probe worker is admitted every "
                       f"{_BACKEND_REPROBE_MIN} min to detect recovery"),
        })

    # Healthy backend: claim any lane a dead sibling abandoned. Dropping the freed
    # lane from the dead sibling's exclude set lets this backend's busiest-pick own it
    # (it was kept off this backend by --exclude-lane lane partitioning). The freed
    # budget was already folded into eff_max_workers above.
    effective_exclude = set(exclude_lanes or set())
    if realloc["claimed_lanes"]:
        effective_exclude -= {lc for lc in realloc["claimed_lanes"] if lc}
    # Pre-spawn lane-lease gate (#1310): the dispatcher HOLDS the target lane
    # before launching instead of trusting the worker to self-arbitrate. fak's
    # taxonomy is one worker per leaf tree, so a second worker on a lane that
    # already has a live one co-edits the same files. Fold the held lanes into the
    # busiest-pick exclude so the auto-pick reroutes to a FREE lane (queued
    # elsewhere); an explicitly-named lane that is held is refused below
    # (COLLISION_RISK) rather than raced.
    held_lanes = live_resolution_lanes(runs_dir)
    pick = lane_issue_numbers(root, lane, exclude=effective_exclude | held_lanes)
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
        # Only surface the realloc block when it actually changed something, so the
        # common (no dead sibling) path's payload is byte-identical to before.
        **({"reallocation": {"effective_max_workers": eff_max_workers, **realloc}}
           if realloc["bonus"] or realloc["claimed_lanes"] else {}),
        # Surface the lease reap only on a live tick where it ran, so a dry-run
        # payload stays byte-identical to before the lease gate.
        **({"leases_reaped": leases_reaped} if live else {}),
        "preflight": {"verdict": pre.get("verdict"), "reason": pre.get("reason"),
                      "cap": pre.get("cap"), "live": pre.get("live")},
        "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
        "lane": chosen_lane, "lane_issue_count": len(pick.get("numbers") or []),
        "cooled_recently": sorted(cooled), "target_issue": target,
        "already_live": sorted(live_issues), "held_lanes": sorted(held_lanes),
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
    if chosen_lane in held_lanes:
        # The lane lease is held by a live worker (only reachable via an explicit
        # --lane, since the auto-pick excluded held lanes above). Refuse instead of
        # racing a second worker onto the same leaf tree — the upstream half of the
        # verified loop (#1310): deny the collision by structure, before the spawn.
        payload.update({"ok": False, "action": "lane_busy", "verdict": "LANE_BUSY",
                        "reason": (f"lane '{chosen_lane}' already holds a live "
                                   f"resolution worker (held lanes "
                                   f"{sorted(held_lanes)}); refusing COLLISION_RISK "
                                   f"— the dispatcher holds the lane lease before "
                                   f"launching, it does not race a second worker "
                                   f"onto the same leaf tree")})
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

    # Fenced lane-lease ACQUIRE (residual of #1310) — the authoritative pre-spawn
    # admission, layered atop the same-host log scan above. ATOMICALLY take
    # refs/fak/locks/resolve-<lane> before launching; a structured fence refusal
    # (a live peer holds the lane, on THIS host or another clone after a fetch)
    # returns LANE_LEASE_HELD instead of racing a second worker onto the leaf tree.
    # Fail-open: a missing fak binary / broken store proceeds on the log scan
    # alone (lease.acquired False, lease.refused False) so the loop never wedges.
    ttl_s = int((worker_timeout_s or DEFAULT_WORKER_TIMEOUT_S) + LEASE_TTL_MARGIN_S)
    lease = acquire_lane_lease(root, chosen_lane, tree=lane_tree(root, chosen_lane),
                               ttl_s=ttl_s, runner=lease_runner)
    payload["lease"] = lease
    if lease.get("refused"):
        payload.update({"ok": False, "action": "lane_leased", "verdict": "LANE_LEASE_HELD",
                        "reason": (f"lane '{chosen_lane}' lease is held by a live peer "
                                   f"(fence {lease.get('reason') or lease.get('fence_verdict') or '?'}); "
                                   f"refusing COLLISION_RISK — the fenced "
                                   f"refs/fak/locks/{lease.get('id')} lease serializes "
                                   f"this lane across machines, not just this host")})
        _record(runs_dir, payload)
        return finish(payload)

    if backend == "claude":
        env = issue_dispatch.worker_env(acct.get("dir"), chosen_lane, root)
    elif backend == "codex":
        env = codex_worker_env(acct.get("dir"), chosen_lane, root)
    else:
        env = opencode_worker_env(acct.get("dir"), chosen_lane, root, runs_dir)
    env["FLEET_RESOLVE_ISSUE"] = str(target)
    command = build_worker_command(backend, rec["prompt"], model)
    spawned = spawn_issue_worker(command, env, root, runs_dir, target, chosen_lane, backend,
                                 account=acct, spawn_probe_s=spawn_probe_s)
    early = spawned.get("early_exit") or {}
    if early.get("checked") and not early.get("alive") and early.get("silent"):
        # The spawn died immediately, so the lane lease we just took now guards a
        # worker that never ran. There is no early-release verb yet (`fak leaseref`
        # exposes only acquire/reap, not a delete-one-live-lease), so the lane is
        # freed the way the issue sanctions for a crashed holder: the lease TTL
        # lapses and a later tick's `reap` sweeps it (reap_expired_leases above).
        # Surface the held lease so the SPAWN_FAILED record is honest about it.
        if lease.get("acquired"):
            payload["lease_held_until_ttl"] = {"id": lease.get("id"),
                                               "holder": lease.get("holder")}
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
    rl = p.get("reallocation") or {}
    if rl.get("bonus") or rl.get("claimed_lanes"):
        frm = ",".join(x for x in (rl.get("from") or []) if x)
        claimed = ",".join(x for x in (rl.get("claimed_lanes") or []) if x)
        lines.append(f"  realloc   : +{rl.get('bonus')} workers, lane(s) [{claimed}] "
                     f"claimed from dead [{frm}] -> eff cap {rl.get('effective_max_workers')}")
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
    ap.add_argument("--max-realloc-workers", type=int, default=DEFAULT_REALLOC_CEILING,
                    help=f"ceiling on extra worker slots a HEALTHY backend may claim "
                         f"from dead siblings in one tick (backend-health reallocation; "
                         f"default {DEFAULT_REALLOC_CEILING}, 0 disables the budget bump)")
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
                       realloc_ceiling=max(0, args.max_realloc_workers),
                       record_loop=not args.no_loop_ledger,
                       loop_ledger=(Path(args.loop_ledger).resolve()
                                    if args.loop_ledger else None))
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
