#!/usr/bin/env python3
"""Worker scanning, reaping, pruning, command/env construction, sidecars, and failure streaks."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from issue_resolve_dispatch_core import (
    ACCOUNT_SIDECAR_SUFFIX,
    BACKENDS,
    BASE_SHA_SIDECAR_SUFFIX,
    BENIGN_ACTIONS,
    CLAUDE_PROMPT_FILE_SUFFIX,
    DEFAULT_FIRST_EDIT_DEADLINE_S,
    LEASE_SIDECAR_SUFFIX,
    NO_COMMIT_BANNER_NOOP,
    OPENCODE_PROMPT_FILE_SUFFIX,
    OPENCODE_PROMPT_NOTICE,
    REPAIR_ISSUES_SIDECAR_SUFFIX,
    SEAT_STREAK_REPROBE_MIN,
    SPAWN_FAILED_RED_STREAK,
    WAVE_SIDECAR_SUFFIX,
    WITNESS_SIDECAR_SUFFIX,
    _ANY_WORKER_LOG_RE,
    _COMMIT_EVENT_RE,
    _EDIT_EVENT_RE,
    _LOG_ISSUE_RE,
    _PLANNING_EVENT_RE,
    _REPAIR_ATTEMPT_LEDGER,
    _REPAIR_LOG_RE,
    _TEST_EVENT_RE,
    _TOOL_EVENT_RE,
    _TYPED_ESCALATION_RE,
    _spawn_header_lane,
    dispatch_preflight,
    dispatch_worker,
    ird_attr,
    no_window_creationflags,
    tier_launch,
    win_creationflags,
    worker_worktree,
)
from issue_resolve_dispatch_witness import classify_no_commit_reason


def live_resolution_issues(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> set[int]:
    """Issue numbers that already have a LIVE resolution worker."""
    live: set[int] = set()
    if not runs_dir.is_dir():
        return live
    dp_mod = ird_attr("dispatch_preflight", dispatch_preflight)
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
            if dp_mod.resolve_sidecar_pid_is_live(
                    pid_file, alive=alive, probe=probe):
                live.add(int(m.group(1)))
    return live


def live_resolution_lanes(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> set[str]:
    """Lanes that already hold a LIVE resolution worker (#1310)."""
    lanes: set[str] = set()
    if not runs_dir.is_dir():
        return lanes
    dp_mod = ird_attr("dispatch_preflight", dispatch_preflight)
    classify_nc = ird_attr("classify_no_commit_reason", classify_no_commit_reason)
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
        if not dp_mod.resolve_sidecar_pid_is_live(
                pid_file, alive=alive, probe=probe):
            continue
        lane = _spawn_header_lane(log)
        if not lane:
            continue
        if classify_nc(log) == NO_COMMIT_BANNER_NOOP:
            continue
        lanes.add(lane)
    return lanes


def recently_attempted_issues(runs_dir: Path, *, cooldown_min: int,
                              now_ts: float | None = None) -> set[int]:
    """Issue numbers attempted within the last ``cooldown_min`` minutes."""
    if cooldown_min <= 0 or not runs_dir.is_dir():
        return set()
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - cooldown_min * 60
    recent: set[int] = set()
    attempts = (*runs_dir.glob("resolve-*.log"),
                *runs_dir.glob(f"resolve-*{WITNESS_SIDECAR_SUFFIX}"))
    for attempt in attempts:
        m = _LOG_ISSUE_RE.search(attempt.name)
        if not m:
            continue
        try:
            if attempt.stat().st_mtime >= horizon:
                recent.add(int(m.group(1)))
        except OSError:
            continue
    return recent


def live_repair_workers(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> list[dict[str, Any]]:
    """Contract-repair workers still ALIVE (max one in flight)."""
    out: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return out
    dp_mod = ird_attr("dispatch_preflight", dispatch_preflight)
    for pid_file in sorted(runs_dir.glob("repair-*.pid")):
        m = _REPAIR_LOG_RE.search(pid_file.name)
        if not m:
            continue
        if not dp_mod.resolve_sidecar_pid_is_live(
                pid_file, alive=alive, probe=probe):
            continue
        try:
            pid: int | None = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = None
        out.append({"pid": pid, "pid_file": pid_file.name, "issue": int(m.group(1))})
    return out


def record_repair_attempt(runs_dir: Path, issues: list[int] | set[int], *,
                          live: bool, now_ts: float | None = None) -> None:
    """Append one repair worker's whole batch to the durable attempt ledger."""
    nums = sorted({int(n) for n in issues if int(n or 0) > 0})
    if not live or not nums:
        return
    import time
    now = now_ts if now_ts is not None else time.time()
    iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with (runs_dir / _REPAIR_ATTEMPT_LEDGER).open("a", encoding="utf-8") as f:
            f.write(json.dumps({"utc": iso, "ts": now, "issues": nums}) + "\n")
    except OSError:
        pass


def recently_repaired_issues(runs_dir: Path, *, cooldown_min: int,
                             now_ts: float | None = None) -> set[int]:
    """Issues a contract-repair worker attempted within ``cooldown_min`` minutes."""
    if cooldown_min <= 0 or not runs_dir.is_dir():
        return set()
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - cooldown_min * 60
    recent: set[int] = set()
    ledger = runs_dir / _REPAIR_ATTEMPT_LEDGER
    if ledger.is_file():
        try:
            lines = ledger.read_text(encoding="utf-8",
                                     errors="replace").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                row = json.loads(line)
                if float(row.get("ts") or 0) < horizon:
                    continue
                recent.update(int(n) for n in (row.get("issues") or []))
            except (TypeError, ValueError):
                continue
    for log in runs_dir.glob("repair-*.log"):
        m = _REPAIR_LOG_RE.search(log.name)
        if not m:
            continue
        try:
            if log.stat().st_mtime < horizon:
                continue
        except OSError:
            continue
        recent.add(int(m.group(1)))
        try:
            extra = log.with_suffix(REPAIR_ISSUES_SIDECAR_SUFFIX).read_text(
                encoding="utf-8")
            recent.update(int(x) for x in extra.split(",") if x.strip())
        except (OSError, ValueError):
            pass
    return recent


def terminate_issue_worker_tree(pid: int) -> dict[str, Any]:
    """Kill one detached issue worker and its descendants."""
    if pid <= 0:
        return {"ok": False, "returncode": None, "error": "invalid pid"}
    os_mod = ird_attr("os", os)
    sub_mod = ird_attr("subprocess", subprocess)
    nowin = ird_attr("no_window_creationflags", no_window_creationflags)
    try:
        if os_mod.name == "nt":
            proc = sub_mod.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True, text=True, timeout=30,
                               creationflags=nowin())
            return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                    "stdout": (proc.stdout or "").strip()[-500:],
                    "stderr": (proc.stderr or "").strip()[-500:]}
        os_mod.killpg(os_mod.getpgid(pid), signal.SIGKILL)
        return {"ok": True, "returncode": 0}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"ok": False, "returncode": None, "error": str(exc)}


def bounded_worker_progress(log_text: str, *, age_s: float,
                            first_edit_deadline_s: int) -> dict[str, Any]:
    """Derive bounded-worker progress only from transcript events, never prose status."""
    edit = _EDIT_EVENT_RE.search(log_text)
    test = _TEST_EVENT_RE.search(log_text)
    commit = _COMMIT_EVENT_RE.search(log_text)
    planning = len(_PLANNING_EVENT_RE.findall(log_text))
    tools = len(_TOOL_EVENT_RE.findall(log_text))
    escalation = _TYPED_ESCALATION_RE.search(log_text)
    if commit:
        status = "committing"
    elif test:
        status = "testing"
    elif edit:
        status = "editing"
    elif age_s >= first_edit_deadline_s:
        status = "stalled"
    else:
        status = "investigating"
    terminal = "STALLED_NO_EDIT" if status == "stalled" else None
    return {
        "status": status,
        "terminal_state": terminal,
        "time_to_first_edit_s": None if edit is None else round(age_s, 1),
        "planning_tool_ratio": round(planning / tools, 3) if tools else 0.0,
        "tool_events": tools,
        "planning_events": planning,
        "typed_escalation_need": escalation.group(1).lower() if escalation else None,
    }


def _compact_investigation_summary(log_text: str, *, limit: int = 12) -> list[str]:
    """Keep the useful tail without replaying a worker's full transcript."""
    useful = [line.strip() for line in log_text.splitlines()
              if line.strip() and not line.startswith("fak guard:")]
    return useful[-limit:]


def reap_timed_out_workers(
    runs_dir: Path,
    *,
    timeout_s: int | None,
    live: bool,
    now_ts: float | None = None,
    probe: Any | None = None,
    killer: Any | None = None,
    first_edit_deadline_s: int = DEFAULT_FIRST_EDIT_DEADLINE_S,
) -> dict[str, Any]:
    """Find resolver workers older than the wall-clock cap and optionally reap."""
    if not timeout_s or timeout_s <= 0:
        return {"timeout_s": timeout_s, "live": live, "candidates": [],
                "reaped": [], "would_reap": [], "disabled": True}
    import time
    now = now_ts if now_ts is not None else time.time()
    kill = killer or ird_attr("terminate_issue_worker_tree", terminate_issue_worker_tree)
    dp_mod = ird_attr("dispatch_preflight", dispatch_preflight)
    candidates: list[dict[str, Any]] = []
    reaped: list[dict[str, Any]] = []
    would_reap: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return {"timeout_s": timeout_s, "live": live, "candidates": [],
                "reaped": [], "would_reap": []}
    for pid_file in sorted((*runs_dir.glob("resolve-*.pid"),
                            *runs_dir.glob("repair-*.pid"))):
        m = _ANY_WORKER_LOG_RE.search(pid_file.name)
        if not m:
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            st = pid_file.stat()
        except (OSError, ValueError):
            continue
        age_s = max(0.0, now - st.st_mtime)
        log_path = pid_file.with_suffix(".log")
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
        progress = bounded_worker_progress(
            log_text, age_s=age_s, first_edit_deadline_s=first_edit_deadline_s)
        no_edit_timeout = progress["terminal_state"] == "STALLED_NO_EDIT"
        wall_timeout = age_s >= timeout_s
        if not (no_edit_timeout or wall_timeout):
            continue
        if not dp_mod.resolve_sidecar_pid_is_live(pid_file, probe=probe):
            continue
        rec: dict[str, Any] = {
            "issue": int(m.group(1)),
            "pid": pid,
            "pid_file": pid_file.name,
            "log": log_path.name,
            "age_s": round(age_s, 1),
            "reason": "STALLED_NO_EDIT" if no_edit_timeout else "WORKER_TIMEOUT",
            "progress": progress,
            "investigation_summary": _compact_investigation_summary(log_text),
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
    """Sweep ``resolve-*.pid`` sidecars whose worker is no longer alive."""
    import time
    now = now_ts if now_ts is not None else time.time()
    dp_mod = ird_attr("dispatch_preflight", dispatch_preflight)
    pruned: list[str] = []
    would_prune: list[str] = []
    if not runs_dir.is_dir():
        return {"live": live, "pruned": pruned, "would_prune": would_prune}
    resolve_sibling_suffixes = (
        ".backend", ".wave", ".account", LEASE_SIDECAR_SUFFIX,
        BASE_SHA_SIDECAR_SUFFIX,
    )
    repair_sibling_suffixes = (
        ".log", ".backend", ".account", LEASE_SIDECAR_SUFFIX,
        BASE_SHA_SIDECAR_SUFFIX, REPAIR_ISSUES_SIDECAR_SUFFIX,
    )
    for pid_file in sorted((*runs_dir.glob("resolve-*.pid"),
                            *runs_dir.glob("repair-*.pid"))):
        if not _ANY_WORKER_LOG_RE.search(pid_file.name):
            continue
        try:
            age_s = max(0.0, now - pid_file.stat().st_mtime)
        except OSError:
            continue
        if age_s < min_age_s:
            continue
        if dp_mod.resolve_sidecar_pid_is_live(pid_file, probe=probe):
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
        sibling_suffixes = (
            repair_sibling_suffixes
            if pid_file.name.startswith("repair-")
            else resolve_sibling_suffixes
        )
        for suf in sibling_suffixes:
            sib = stem.with_suffix(suf)
            try:
                sib.unlink()
            except OSError:
                pass
    return {"live": live, "pruned": pruned, "would_prune": would_prune}


def build_worker_command(backend: str, prompt: str, model: str | None,
                         effort: str | None = None, ultracode: bool = False) -> list[str]:
    """The argv for one detached issue-resolution worker, per backend."""
    del prompt
    tl_mod = ird_attr("tier_launch", tier_launch)
    if backend == "claude":
        cmd = ["claude", "-p", "--permission-mode", "bypassPermissions"]
        if model:
            cmd += ["--model", model]
        if ultracode:
            cmd += ["--settings", tl_mod.ULTRACODE_SETTINGS_ARG]
        elif effort:
            cmd += ["--effort", effort]
        return cmd
    if backend == "opencode":
        cmd = ["opencode", "run", "--pure", "--print-logs",
               "--dangerously-skip-permissions"]
        if model:
            cmd += ["-m", model]
        cmd.append(OPENCODE_PROMPT_NOTICE)
        return cmd
    if backend == "codex":
        cmd = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
               "--skip-git-repo-check"]
        if model:
            cmd += ["-m", model]
        return cmd
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")


def _command_basename(path: str) -> str:
    """Return a command basename for POSIX or Windows-shaped paths."""
    return os.path.basename(str(path).replace("\\", "/")).lower()


def unwrap_opencode_npm_shim(exe: str) -> str:
    """Return the real opencode executable behind the Windows npm shim, if present."""
    if _command_basename(exe) not in {"opencode", "opencode.exe", "opencode.cmd",
                                      "opencode.bat", "opencode.ps1"}:
        return ""
    raw = os.fspath(exe)
    parent_idx = max(raw.rfind("/"), raw.rfind("\\"))
    if parent_idx < 0:
        return ""
    parent = raw[:parent_idx]
    if parent in ("", "."):
        return ""
    sep = raw[parent_idx]
    target = sep.join((parent, "node_modules", "opencode-ai", "bin", "opencode.exe"))
    try:
        return target if os.path.isfile(target) else ""
    except OSError:
        return ""


def resolve_worker_executable(backend: str, name: str) -> str:
    shutil_mod = ird_attr("shutil", shutil)
    os_mod = ird_attr("os", os)
    exe = shutil_mod.which(name) or name
    if backend == "opencode" and os_mod.name == "nt":
        target = unwrap_opencode_npm_shim(exe)
        if target:
            return target
    return exe


def attach_opencode_prompt_file(command: list[str], prompt_path: Path | str) -> list[str]:
    """Attach the staged dispatch prompt while keeping the final notice as text."""
    out = list(command)
    prompt = str(prompt_path)
    if not out or not prompt.strip():
        return out
    if len(out) == 1:
        return [*out, "--file", prompt]
    last = out[-1]
    return [*out[:-1], "--file", prompt, "--", last]


def resolve_opencode_command(command: list[str]) -> list[str]:
    """Resolve the opencode token even when it is nested behind ``fak guard --``."""
    out = list(command)
    for idx, token in enumerate(out):
        if _command_basename(token) in {"opencode", "opencode.exe", "opencode.cmd",
                                        "opencode.bat", "opencode.ps1"} \
                and idx + 1 < len(out) and out[idx + 1] == "run":
            out[idx] = resolve_worker_executable("opencode", token)
            break
    return out


def _opencode_config_home(account_dir: str, runs_dir: Path) -> str:
    """Resolve or mint an XDG_CONFIG_HOME for one opencode account."""
    p = Path(account_dir)
    if p.name == "opencode":
        return str(p.parent)
    pin = runs_dir / ".opencode-pins" / p.name
    link = pin / "opencode"
    pin.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        os_mod = ird_attr("os", os)
        sub_mod = ird_attr("subprocess", subprocess)
        nowin = ird_attr("no_window_creationflags", no_window_creationflags)
        try:
            os_mod.symlink(account_dir, link, target_is_directory=True)
        except OSError:
            sub_mod.run(["cmd", "/c", "mklink", "/J", str(link), account_dir],
                        capture_output=True, text=True,
                        encoding="utf-8", errors="replace",
                        creationflags=nowin())
    if not link.exists():
        raise RuntimeError(f"could not pin opencode account dir {account_dir!r}")
    return str(pin)


def _github_cli_config_dir(base: dict[str, str] | None = None) -> str | None:
    """Return the ambient gh config dir, if one exists."""
    e = base if base is not None else ird_attr("os", os).environ
    explicit = e.get("GH_CONFIG_DIR")
    if explicit:
        return explicit
    candidates: list[Path] = []
    appdata = e.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "GitHub CLI")
    home = e.get("HOME") or e.get("USERPROFILE")
    if home:
        candidates.append(Path(home) / ".config" / "gh")
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _opencode_temp_dir(runs_dir: Path, lane: str) -> Path:
    safe_lane = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(lane or "worker")).strip("-")
    if not safe_lane:
        safe_lane = "worker"
    return runs_dir / ".opencode-tmp" / safe_lane


def opencode_worker_env(account_dir: str | None, lane: str, workspace: Path,
                        runs_dir: Path) -> dict[str, str]:
    """Child env for an opencode/glm worker."""
    dw_mod = ird_attr("dispatch_worker", dispatch_worker)
    env = dw_mod.child_env(lane, "opencode", workspace)
    env.pop("CLAUDE_CONFIG_DIR", None)
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    if account_dir:
        env["XDG_CONFIG_HOME"] = _opencode_config_home(account_dir, runs_dir)
    gh_config = _github_cli_config_dir()
    if gh_config:
        env.setdefault("GH_CONFIG_DIR", gh_config)
    scratch = _opencode_temp_dir(runs_dir, lane)
    scratch.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR"):
        env[key] = str(scratch)
    return env


def codex_worker_env(account_dir: str | None, lane: str, workspace: Path) -> dict[str, str]:
    """Child env for a codex (`codex exec`) worker."""
    dw_mod = ird_attr("dispatch_worker", dispatch_worker)
    env = dw_mod.child_env(lane, "codex", workspace)
    env.pop("CLAUDE_CONFIG_DIR", None)
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    if account_dir:
        env["CODEX_HOME"] = account_dir
    return env


def wave_membership_env(rank: int, wave_id: str, size: int,
                        shortfall: int) -> dict[str, str]:
    """The env vars that stamp a worker's place in its wave."""
    return {
        "FLEET_WAVE_ID": str(wave_id),
        "FLEET_WAVE_RANK": str(int(rank)),
        "FLEET_WAVE_SIZE": str(int(size)),
        "FLEET_WAVE_SHORTFALL": str(int(shortfall)),
    }


def write_wave_sidecar(out_log: Path, rank: int, wave_id: str, size: int,
                       shortfall: int) -> dict[str, Any]:
    """Write the per-worker membership sidecar next to ``.pid``/``.backend``."""
    rec = {"wave_id": str(wave_id), "rank": int(rank), "size": int(size),
           "shortfall": int(shortfall)}
    out_log.with_suffix(WAVE_SIDECAR_SUFFIX).write_text(
        json.dumps(rec, sort_keys=True), encoding="utf-8")
    return rec


def write_account_sidecar(out_log: Path, account: dict[str, Any] | None) -> dict[str, Any]:
    """Write the switcher account selected for this worker next to its log."""
    src = account or {}
    rec = {k: src.get(k) for k in ("tag", "tier", "model", "dir")
           if src.get(k) is not None}
    if rec:
        out_log.with_suffix(ACCOUNT_SIDECAR_SUFFIX).write_text(
            json.dumps(rec, sort_keys=True), encoding="utf-8")
    return rec


def dispatch_account_id(account: dict[str, Any] | None) -> str:
    """The stable identity of the seat a worker is dispatched onto."""
    src = account or {}
    tag = str(src.get("tag") or "").strip()
    if tag:
        return tag
    acct_dir = str(src.get("dir") or "").strip()
    if acct_dir:
        return os.path.basename(os.path.normpath(acct_dir))
    return ""


def stamp_dispatch_account(env: dict[str, str],
                           account: dict[str, Any] | None) -> dict[str, str]:
    """Return ``env`` with ``DISPATCH_ACCOUNT`` naming the seat this worker serves on."""
    out = dict(env)
    ident = dispatch_account_id(account)
    if ident:
        out["DISPATCH_ACCOUNT"] = ident
    else:
        out.pop("DISPATCH_ACCOUNT", None)
    return out


def write_lease_sidecar(out_log: Path, lease: dict[str, Any] | None) -> dict[str, Any]:
    """Write the fenced lease token selected for this worker."""
    src = lease or {}
    if not src.get("acquired"):
        return {}
    rec = {k: src.get(k) for k in ("id", "holder", "generation", "session_id", "tree")
           if src.get(k) is not None}
    if rec.get("id") and rec.get("holder"):
        out_log.with_suffix(LEASE_SIDECAR_SUFFIX).write_text(
            json.dumps(rec, sort_keys=True), encoding="utf-8")
        return rec
    return {}


def probe_spawned_worker(proc: subprocess.Popen, out_log: Path,
                         wait_s: float = 0.0) -> dict[str, Any]:
    """Briefly check whether a just-spawned worker died before it could log."""
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
            rec["tail"] = out_log.read_text(encoding="utf-8", errors="replace")[-8192:]
        except OSError:
            pass
    return rec


def enumerate_wave(runs_dir: Path, wave_id: str) -> dict[str, Any]:
    """Enumerate a wave from its ``.wave`` sidecars on disk."""
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


def _spawn_issue_worker_impl(
    command: list[str], env: dict[str, str], cwd: Path,
    log_dir: Path, issue: int, lane: str,
    backend: str,
    membership: dict[str, Any] | None = None,
    account: dict[str, Any] | None = None,
    lease: dict[str, Any] | None = None,
    base_sha: str | None = None,
    spawn_probe_s: float = 0.0,
    log_prefix: str = "resolve",
    prompt_payload: str | None = None,
    worktree_git: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Launch a detached worker (claude/opencode/codex) on one issue; record pid."""
    os_mod = ird_attr("os", os)
    sub_mod = ird_attr("subprocess", subprocess)
    ww_mod = ird_attr("worker_worktree", worker_worktree)
    win_flags_fn = ird_attr("win_creationflags", win_creationflags)
    probe_fn = ird_attr("probe_spawned_worker", probe_spawned_worker)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_log = log_dir / f"{log_prefix}-{issue}-{stamp}.log"
    spawn_cwd, env, wt_info = ww_mod.isolate_spawn(
        Path(cwd), lane, str(issue), cwd, env, base_sha=base_sha, git=worktree_git)
    env = stamp_dispatch_account(env, account)
    prompt_file: Path | None = None
    prompt_stdin = None
    if backend == "opencode":
        if prompt_payload is None:
            raise ValueError("opencode worker spawn requires prompt_payload")
        prompt_file = out_log.with_suffix(OPENCODE_PROMPT_FILE_SUFFIX)
        prompt_file.write_text(prompt_payload, encoding="utf-8")
        command = resolve_opencode_command(
            attach_opencode_prompt_file(command, prompt_file))
    elif backend in ("claude", "codex") and prompt_payload is not None:
        prompt_file = out_log.with_suffix(CLAUDE_PROMPT_FILE_SUFFIX)
        prompt_file.write_text(prompt_payload, encoding="utf-8")
        prompt_stdin = open(prompt_file, "r", encoding="utf-8")
    exe = resolve_worker_executable(backend, command[0])
    argv = [exe, *command[1:]]
    if membership is not None:
        env = {**env, **wave_membership_env(**membership)}
    kwargs: dict[str, Any] = {}
    if os_mod.name == "nt":
        kwargs["creationflags"] = win_flags_fn(exe)
    else:
        kwargs["start_new_session"] = True
    fh = open(out_log, "w", encoding="utf-8")
    try:
        fh.write("# fak-spawn %s issue=%s lane=%s backend=%s argv0=%s\n" % (
            stamp, issue, lane, backend, os.path.basename(exe)))
        fh.flush()
        proc = sub_mod.Popen(argv, cwd=str(spawn_cwd), env=env,
                             stdin=prompt_stdin if prompt_stdin is not None else subprocess.DEVNULL,
                             stdout=fh, stderr=subprocess.STDOUT, **kwargs)
    finally:
        if prompt_stdin is not None:
            prompt_stdin.close()
        fh.close()
    (out_log.with_suffix(".pid")).write_text(str(proc.pid), encoding="utf-8")
    (out_log.with_suffix(".backend")).write_text(backend, encoding="utf-8")
    if base_sha:
        (out_log.with_suffix(BASE_SHA_SIDECAR_SUFFIX)).write_text(base_sha, encoding="utf-8")
    if wt_info.get("worktree"):
        try:
            out_log.with_suffix(".worktree").write_text(
                str(wt_info["worktree"]), encoding="utf-8")
            if wt_info.get("base_sha"):
                out_log.with_suffix(BASE_SHA_SIDECAR_SUFFIX).write_text(
                    str(wt_info["base_sha"]), encoding="utf-8")
            lease_tree = [t for t in ((lease or {}).get("tree") or [])
                          if isinstance(t, str)]
            if lease_tree:
                out_log.with_suffix(".lease-tree.json").write_text(
                    json.dumps(lease_tree), encoding="utf-8")
        except OSError:
            pass
    result: dict[str, Any] = {"pid": proc.pid, "log": str(out_log), "issue": issue,
                              "lane": lane, "backend": backend}
    if wt_info.get("worktree"):
        result["worktree"] = wt_info["worktree"]
    if prompt_file is not None:
        result["prompt_file"] = str(prompt_file)
    acct = write_account_sidecar(out_log, account)
    if acct:
        result["account"] = acct
    lease_rec = write_lease_sidecar(out_log, lease)
    if lease_rec:
        result["lease"] = lease_rec
    if membership is not None:
        result["membership"] = write_wave_sidecar(out_log, **membership)
    early = probe_fn(proc, out_log, spawn_probe_s)
    if early.get("checked"):
        result["early_exit"] = early
    return result


def _spawn_failure_streak_path(runs_dir: Path, backend: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(backend or "claude")) or "claude"
    return runs_dir / f"spawn-failure-streak-{safe}.json"


def _read_spawn_failure_streaks(runs_dir: Path, backend: str) -> dict[str, int]:
    try:
        doc = json.loads(
            _spawn_failure_streak_path(runs_dir, backend).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in doc.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _write_spawn_failure_streaks(runs_dir: Path, backend: str,
                                 streaks: dict[str, int]) -> None:
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        _spawn_failure_streak_path(runs_dir, backend).write_text(
            json.dumps(streaks, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def bump_spawn_failure_streak(runs_dir: Path, target: int | str, backend: str) -> int:
    """Record one more consecutive SPAWN_FAILED for ``target`` on ``backend``."""
    streaks = _read_spawn_failure_streaks(runs_dir, backend)
    count = streaks.get(str(target), 0) + 1
    streaks[str(target)] = count
    _write_spawn_failure_streaks(runs_dir, backend, streaks)
    return count


def clear_spawn_failure_streak(runs_dir: Path, target: int | str, backend: str) -> None:
    """A successful spawn of ``target`` broke the streak."""
    streaks = _read_spawn_failure_streaks(runs_dir, backend)
    if str(target) in streaks:
        del streaks[str(target)]
        _write_spawn_failure_streaks(runs_dir, backend, streaks)


def _spawn_failure_streak_seat_path(runs_dir: Path, backend: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(backend or "claude")) or "claude"
    return runs_dir / f"spawn-failure-streak-seat-{safe}.json"


def _read_seat_failure_streaks(runs_dir: Path, backend: str) -> dict[str, dict[str, float]]:
    try:
        doc = json.loads(
            _spawn_failure_streak_seat_path(runs_dir, backend).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for k, v in doc.items():
        if isinstance(v, dict):
            try:
                out[str(k)] = {"count": int(v.get("count") or 0),
                               "last_ts": float(v.get("last_ts") or 0.0)}
            except (TypeError, ValueError):
                continue
        else:
            try:
                out[str(k)] = {"count": int(v), "last_ts": 0.0}
            except (TypeError, ValueError):
                continue
    return out


def _write_seat_failure_streaks(runs_dir: Path, backend: str,
                                streaks: dict[str, dict[str, float]]) -> None:
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        _spawn_failure_streak_seat_path(runs_dir, backend).write_text(
            json.dumps(streaks, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def bump_spawn_failure_streak_seat(runs_dir: Path, seat_tag: str | None,
                                   backend: str,
                                   now_ts: float | None = None) -> int:
    """Record one more consecutive SPAWN_FAILED for ``seat_tag`` on ``backend`` (#4591)."""
    import time
    tag = str(seat_tag or "").strip()
    if not tag:
        return 0
    streaks = _read_seat_failure_streaks(runs_dir, backend)
    prior = streaks.get(tag) or {}
    row = {"count": int(prior.get("count") or 0) + 1,
           "last_ts": float(time.time() if now_ts is None else now_ts)}
    streaks[tag] = row
    _write_seat_failure_streaks(runs_dir, backend, streaks)
    return int(row["count"])


def clear_spawn_failure_streak_seat(runs_dir: Path, seat_tag: str | None,
                                    backend: str) -> None:
    """A clean launch ON THIS SEAT broke the seat's streak (#4591)."""
    tag = str(seat_tag or "").strip()
    if not tag:
        return
    streaks = _read_seat_failure_streaks(runs_dir, backend)
    if tag in streaks:
        del streaks[tag]
        _write_seat_failure_streaks(runs_dir, backend, streaks)


def seat_spawn_failure_state(runs_dir: Path, seat_tag: str | None, backend: str,
                             now_ts: float | None = None) -> dict[str, Any]:
    """The routed seat's current seat-keyed streak, folded for the cool gate (#4591)."""
    import time
    tag = str(seat_tag or "").strip()
    row = _read_seat_failure_streaks(runs_dir, backend).get(tag) if tag else None
    count = int((row or {}).get("count") or 0)
    last_ts = float((row or {}).get("last_ts") or 0.0)
    now = float(time.time() if now_ts is None else now_ts)
    cooled = count >= SPAWN_FAILED_RED_STREAK
    return {"seat": tag or None, "backend": backend, "streak": count,
            "last_ts": last_ts or None, "cooled": cooled,
            "reprobe_min": SEAT_STREAK_REPROBE_MIN,
            "reprobe_due": bool(cooled and last_ts
                                and now - last_ts >= SEAT_STREAK_REPROBE_MIN * 60)}


def tick_exit_code(payload: dict[str, Any]) -> int:
    """Process exit code for one dispatch tick (#2636)."""
    if not isinstance(payload, dict):
        return 1
    action = str(payload.get("action") or "")
    if action in BENIGN_ACTIONS:
        return 0
    if action == "spawn_failed":
        try:
            streak = int(payload.get("spawn_failed_streak") or 1)
        except (TypeError, ValueError):
            streak = 1
        return 1 if streak >= SPAWN_FAILED_RED_STREAK else 0
    return 1


def _record(runs_dir: Path, payload: dict[str, Any]) -> None:
    backend = payload.get("backend") or "claude"
    blob = json.dumps(payload, indent=2)
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / f"last-resolve-tick-{backend}.json").write_text(blob, encoding="utf-8")
        (runs_dir / "last-resolve-tick.json").write_text(blob, encoding="utf-8")
    except OSError:
        pass
