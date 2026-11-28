#!/usr/bin/env python3
"""Core constants, atomic JSON/file helpers, config normalization, and process runners for fleet_control_pane."""
from __future__ import annotations

import collections  # noqa: F401 - kept available for callers importing shared stdlib helpers
import contextlib
import datetime as dt
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import appversion  # noqa: E402


SCHEMA = "fleet-control-pane/1"
CONFIG_SCHEMA = "fleet-control-pane.config/1"
SETUP_SCHEMA = "fleet-control-pane.setup/1"
TICK_SCHEMA = "fleet-control-pane.tick/1"
RECOVER_SCHEMA = "fleet-control-pane.recover/1"
SUPERVISOR_SCHEMA = "fleet-control-pane.supervisor/1"
FLEET_SCHEMA = "fleet-control-pane.fleet/1"
COMMIT_SCHEMA = "fleet-control-pane.commit/1"
BOOTSTRAP_SCHEMA = "fleet-control-pane.bootstrap/1"
SYNC_SCHEMA = "fleet-control-pane.sync/1"
PUBLISH_SCHEMA = "fleet-control-pane.publish/1"
LOOP_CONFIG_SCHEMA = "fleet-control-pane.loop-config/1"
LOOP_LIST_SCHEMA = "fleet-control-pane.loop-list/1"
LOOP_CHECK_SCHEMA = "fleet-control-pane.loop-check/1"
LOOP_AUDIT_SCHEMA = "fleet-control-pane.loop-audit/1"
LOOP_SCAFFOLD_SCHEMA = "fleet-control-pane.loop-scaffold/1"
LOOP_SET_SCHEMA = "fleet-control-pane.loop-set/1"
LOOP_CATALOG_SCHEMA = "fleet-control-pane.loop-catalog/1"
ATOMIC_WRITE_REPLACE_ATTEMPTS = 8
ATOMIC_WRITE_REPLACE_SLEEP_S = 0.05

HEALTHY_LOOP_VALUES = {"OK", "HEALTHY", "RUNNING", "ALIVE", "READY", "PASS", "GREEN"}
ACTION_LOOP_VALUES = {"ACTION", "DEAD", "STALLED", "UNHEALTHY", "FAIL", "FAILED", "ERROR", "RED", "MISSING"}
SUPERVISOR_ACTION_VERDICTS = {"DEAD", "STALLED"}
SUPERVISOR_NON_ACTION_VERDICTS = {
    "SUPERVISING",
    "READY_TO_CANARY",
    "AT_TARGET",
    "READY",
    "WALL",
    "DRAINED",
    "STOPPED",
    "IDLE",
    "OK",
    "HEALTHY",
}
SUPERVISOR_WATCHDOG_ACTION_RESULT = 10
WINDOWS_TASK_RUNNING_RESULT = 267009
WINDOWS_TASK_REQUEST_REFUSED_RESULT = 2147946720
WINDOWS_TASK_RESULT_LABELS = {
    SUPERVISOR_WATCHDOG_ACTION_RESULT: "supervisor watchdog reported an action-status verdict",
    WINDOWS_TASK_RUNNING_RESULT: "task is currently running",
    WINDOWS_TASK_REQUEST_REFUSED_RESULT: "request refused, usually because an earlier run is still active",
}
SCHEDULED_TASK_ACTION_STATES = {"disabled"}
CONFIG_OVERRIDE_KEYS = (
    "python",
    "user_home",
    "job_dir",
    "claude_exe",
    "registry_dir",
    "machine_dir",
    "machine_id",
    "watchdog_log_dir",
    "git_remote",
    "target",
    "session_window_h",
)
TRACKED_DEFAULT_DRIFT_KEYS = (
    "registry_dir",
    "machine_dir",
    "watchdog_log_dir",
    "git_remote",
    "target",
    "session_window_h",
)

PANE_SOURCE_PATHS = [
    "VERSION",
    "tools/appversion.py",
    "tools/fleet_control_pane.py",
    "tools/fleet_control_pane_test.py",
    "tools/control_pane.example.json",
    "tools/control_pane.loops.json",
    "tools/install_trunk_guard.py",
    "tools/githooks/reference-transaction",
    "tools/register_control_pane_tick.ps1",
    "tools/register_control_pane_tick.sh",
    "tools/RESUME-NEVER-GUESS.md",
]

DIRTY_GROUP_SUBJECTS = {
    "docs": "docs: update documentation",
    "tools/fleet-control-pane": "tools: update fleet control pane",
    "tools/loops": "tools: update loop scripts",
    "tools/proofs": "tools: update pane proof images",
}

DEFAULT_WORKTREE_MASTER_REF = "origin/master"
DEFAULT_WORKTREE_ALLOW_BRANCHES = ("fak-v0.1",)
TRUNK_GUARD_INSTALLER_REL = "tools/install_trunk_guard.py"
TRUNK_GUARD_HOOKS_REL = "tools/githooks"
TRUNK_GUARD_HOOK_REL = f"{TRUNK_GUARD_HOOKS_REL}/reference-transaction"

# The aggregate-audit loop must not audit itself when run from the tick, or a
# single broken loop would be reported twice (once directly, once via the audit
# loop's own non-zero exit). loop_audit skips this name when enumerating.
LOOP_AUDIT_SELF_NAME = "gardening-loops-audit"
LOOP_BROKEN_STATES = {"UNAVAILABLE", "TIMEOUT"}

_ACTIVE_PANE: list[Any] = []


@contextlib.contextmanager
def bind_pane(mod_ns: Any):
    _ACTIVE_PANE.append(mod_ns)
    try:
        yield mod_ns
    finally:
        _ACTIVE_PANE.pop()


def pane_attr(name: str, fallback: Any) -> Any:
    """Resolve an attribute through the active fleet_control_pane facade when loaded."""
    if _ACTIVE_PANE:
        return getattr(_ACTIVE_PANE[-1], name, fallback)
    mod = sys.modules.get("fleet_control_pane")
    return getattr(mod, name, fallback) if mod is not None else fallback


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_now() -> str:
    return now_utc().isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso_age_min(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        stamp = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=dt.timezone.utc)
        return round((now_utc() - stamp).total_seconds() / 60.0, 1)
    except ValueError:
        return None


def repo_root(start: Path | None = None) -> Path:
    return appversion.repo_root(start or Path(__file__))


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json(path: Path, doc: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(doc, indent=2) + "\n")


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        replace_with_retry(tmp_path, path)
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def replace_with_retry(src: Path, dst: Path) -> None:
    delay = ATOMIC_WRITE_REPLACE_SLEEP_S
    time_mod = pane_attr("time", time)
    for attempt in range(ATOMIC_WRITE_REPLACE_ATTEMPTS):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == ATOMIC_WRITE_REPLACE_ATTEMPTS - 1:
                raise
            time_mod.sleep(delay)
            delay = min(delay * 2, 0.5)


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def resolve_path(value: str | os.PathLike[str] | None, root: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    raw = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def display_path(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    path = path.resolve(strict=False)
    root = root.resolve(strict=False)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def default_config(root: Path) -> dict[str, Any]:
    sibling_job = root.parent / "job"
    claude_exe = shutil.which("claude") or shutil.which("claude.exe") or ""
    return {
        "schema": CONFIG_SCHEMA,
        "session_window_h": 6,
        "target": 4,
        "python": sys.executable,
        "user_home": str(Path.home()),
        "job_dir": str(sibling_job) if sibling_job.exists() else "",
        "claude_exe": claude_exe,
        "registry_dir": "tools/_registry",
        "machine_dir": "tools/_registry/machines",
        "machine_id": socket.gethostname(),
        "machine_stale_min": 15,
        "watchdog_log_dir": "tools/_watchdog",
        "refresh_timeout_s": 90,
        "git_remote": "origin",
        "worktree_master_ref": DEFAULT_WORKTREE_MASTER_REF,
        "worktree_allow_branches": list(DEFAULT_WORKTREE_ALLOW_BRANCHES),
        "loops": {},
        "control_tick": {
            "task_name": "FleetControlPaneTick",
            "register_script": "tools/register_control_pane_tick.ps1",
            "posix_register_script": "tools/register_control_pane_tick.sh",
            "interval_min": 5,
        },
        "watchdogs": {
            "supervisor": {
                "task_name": "FleetSupervisorWatchdog",
                "script": "tools/fleet_supervisor_watchdog.ps1",
                "register_script": "tools/register_supervisor_watchdog.ps1",
                "interval_min": 5,
            },
            "resume": {
                "task_name": "FleetResumeWatchdog",
                "script": "tools/fleet_resume_watchdog.ps1",
                "register_script": "tools/register_resume_watchdog.ps1",
                # 5-min cadence (matches the supervisor) so the AUTO_RESUME queue drains
                # faster; actual launch rate is still bounded by MaxPerTick + the
                # per-source concurrency gate + REHOME_CAP.
                "interval_min": 5,
            },
        },
        "supervisor_status_cmd": [
            "{python}",
            "tools/dos_supervisor_status.py",
            "--json",
        ],
    }


def apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    env_map = {
        "FLEET_PYTHON": "python",
        "FLEET_USER_HOME": "user_home",
        "FLEET_JOB_DIR": "job_dir",
        "FLEET_CLAUDE_EXE": "claude_exe",
        "FLEET_REG_DIR": "registry_dir",
        "FLEET_MACHINE_DIR": "machine_dir",
        "FLEET_MACHINE_ID": "machine_id",
    }
    out = dict(config)
    for env_name, key in env_map.items():
        value = os.environ.get(env_name)
        if value:
            out[key] = value
    return out


def normalize_config(config: dict[str, Any], root: Path) -> dict[str, Any]:
    out = dict(config)
    out["root"] = str(root)
    for key in ("job_dir", "user_home", "registry_dir", "machine_dir", "watchdog_log_dir"):
        out[key] = str(resolve_path(out.get(key), root) or "")
    for key in ("python", "claude_exe"):
        val = out.get(key, "")
        if val:
            resolved = resolve_path(val, root)
            if resolved and resolved.exists():
                out[key] = str(resolved)
            else:
                out[key] = str(val)
    try:
        out["session_window_h"] = float(out.get("session_window_h", 6))
    except (TypeError, ValueError):
        out["session_window_h"] = 6.0
    try:
        out["target"] = int(out.get("target", 4))
    except (TypeError, ValueError):
        out["target"] = 4
    try:
        out["refresh_timeout_s"] = int(out.get("refresh_timeout_s", 90))
    except (TypeError, ValueError):
        out["refresh_timeout_s"] = 90
    try:
        out["machine_stale_min"] = float(out.get("machine_stale_min", 15))
    except (TypeError, ValueError):
        out["machine_stale_min"] = 15.0
    return out


def config_paths(root: Path) -> dict[str, Path]:
    return {
        "example": root / "tools" / "control_pane.example.json",
        "loop_catalog": root / "tools" / "control_pane.loops.json",
        "local": root / "tools" / "_registry" / "control_pane.local.json",
    }


def load_config(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    paths = config_paths(root)
    config = default_config(root)
    config = deep_merge(config, read_json(paths["example"]))
    config = deep_merge(config, read_json(paths["loop_catalog"]))
    config = deep_merge(config, read_json(paths["local"]))
    config = apply_env_overrides(config)
    config = normalize_config(config, root)
    config["_paths"] = {k: str(v) for k, v in paths.items()}
    config["_local_exists"] = paths["local"].exists()
    return config


def tracked_default_config(root: Path) -> dict[str, Any]:
    paths = config_paths(root)
    config = default_config(root)
    config = deep_merge(config, read_json(paths["example"]))
    config = deep_merge(config, read_json(paths["loop_catalog"]))
    return normalize_config(config, root)


def clean_config_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in CONFIG_OVERRIDE_KEYS:
        value = (overrides or {}).get(key)
        if value is None or value == "":
            continue
        if key == "target":
            value = int(value)
        elif key == "session_window_h":
            value = float(value)
        out[key] = value
    return out


def apply_runtime_overrides(config: dict[str, Any], root: Path, overrides: dict[str, Any] | None) -> dict[str, Any]:
    clean = clean_config_overrides(overrides)
    if not clean:
        return config
    paths = dict(config.get("_paths") or {})
    local_exists = bool(config.get("_local_exists"))
    out = deep_merge(config, clean)
    out = normalize_config(out, root)
    out["_paths"] = paths
    out["_local_exists"] = local_exists
    return out


def init_command(config: dict[str, Any], overrides: dict[str, Any] | None, *, force: bool = False) -> list[str]:
    args = [str(config.get("python") or sys.executable), "tools/fleet_control_pane.py", "init"]
    if force:
        args.append("--force")
    flag_map = {
        "python": "--python",
        "user_home": "--user-home",
        "job_dir": "--job-dir",
        "claude_exe": "--claude-exe",
        "registry_dir": "--registry-dir",
        "machine_dir": "--machine-dir",
        "machine_id": "--machine-id",
        "watchdog_log_dir": "--watchdog-log-dir",
        "git_remote": "--git-remote",
        "target": "--target",
        "session_window_h": "--session-window-h",
    }
    for key, value in clean_config_overrides(overrides).items():
        args.extend([flag_map[key], str(value)])
    return args


def local_config_mismatches(local_path: Path, overrides: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    clean = clean_config_overrides(overrides)
    if not clean or not local_path.exists():
        return {}
    existing = read_json(local_path)
    mismatches: dict[str, dict[str, Any]] = {}
    for key, expected in clean.items():
        if existing.get(key) != expected:
            mismatches[key] = {"current": existing.get(key), "requested": expected}
    return mismatches


def local_config_default_drift(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    local_path = Path((config.get("_paths") or {}).get("local") or config_paths(root)["local"])
    local_raw = read_json(local_path)
    if not local_path.exists() or not local_raw:
        return {"count": 0, "items": [], "command": []}
    tracked = tracked_default_config(root)
    local_effective = normalize_config(deep_merge(tracked, local_raw), root)
    items = []
    for key in TRACKED_DEFAULT_DRIFT_KEYS:
        if key not in local_raw:
            continue
        current = local_effective.get(key)
        expected = tracked.get(key)
        if str(current) == str(expected):
            continue
        items.append({
            "key": key,
            "current": current,
            "tracked": expected,
        })
    overrides = {item["key"]: item["tracked"] for item in items}
    detail = ""
    if items:
        shown = ", ".join(
            f"{item['key']} local={item['current']} tracked={item['tracked']}"
            for item in items[:3]
        )
        omitted = max(0, len(items) - 3)
        detail = shown + (f", +{omitted} more" if omitted else "")
    return {
        "count": len(items),
        "items": items,
        "detail": detail,
        "command": init_command(config, overrides, force=True) if items else [],
    }


def config_runtime_dirs(config: dict[str, Any]) -> list[tuple[str, Path]]:
    dirs: list[tuple[str, Path]] = []
    for key in ("registry_dir", "machine_dir", "watchdog_log_dir"):
        value = config.get(key)
        if value:
            dirs.append((key, Path(str(value))))
    return dirs


def directory_probe(path: Path, *, create: bool = False) -> dict[str, Any]:
    try:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            return {"ok": False, "exists": False, "writable": False, "reason": "missing"}
        if not path.is_dir():
            return {"ok": False, "exists": True, "writable": False, "reason": "not a directory"}
        fd, probe_raw = tempfile.mkstemp(prefix=".control-pane-write-test-", dir=path, text=True)
        probe = Path(probe_raw)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("ok\n")
        finally:
            try:
                probe.unlink()
            except OSError:
                pass
        return {"ok": True, "exists": True, "writable": True}
    except OSError as exc:
        return {
            "ok": False,
            "exists": path.exists(),
            "writable": False,
            "reason": str(exc),
        }


def ensure_runtime_dirs(config: dict[str, Any], *, apply: bool = False) -> dict[str, Any]:
    checks = []
    changed = False
    for name, path in config_runtime_dirs(config):
        existed = path.exists()
        probe = directory_probe(path, create=apply)
        checks.append({"name": name, "path": str(path), **probe})
        changed = changed or (apply and not existed and probe.get("ok"))
    ok = all(check.get("ok") for check in checks)
    return {"ok": ok, "changed": changed, "checks": checks}


def init_config_doc(root: Path, base_config: dict[str, Any] | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    base = base_config or default_config(root)
    doc = {
        "schema": CONFIG_SCHEMA,
        "_doc": "Machine-local fleet control pane config. This file is ignored by git.",
    }
    for key in CONFIG_OVERRIDE_KEYS:
        value = base.get(key)
        if value is None or value == "":
            continue
        doc[key] = value
    doc.setdefault("session_window_h", 6)
    doc.setdefault("target", 4)
    doc.setdefault("python", sys.executable)
    doc.setdefault("user_home", str(Path.home()))
    doc.setdefault("job_dir", str(root.parent / "job") if (root.parent / "job").exists() else "")
    doc.setdefault("claude_exe", shutil.which("claude") or shutil.which("claude.exe") or "")
    doc.setdefault("registry_dir", "tools/_registry")
    doc.setdefault("machine_dir", "tools/_registry/machines")
    doc.setdefault("machine_id", socket.gethostname())
    doc.setdefault("watchdog_log_dir", "tools/_watchdog")
    doc.setdefault("git_remote", "origin")
    doc.update(clean_config_overrides(overrides))
    if not doc.get("machine_id"):
        doc["machine_id"] = socket.gethostname()
    return doc


def init_config(
    root: Path,
    force: bool = False,
    overrides: dict[str, Any] | None = None,
    base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = config_paths(root)
    if paths["local"].exists() and not force:
        return {"written": False, "path": str(paths["local"]), "reason": "already exists"}
    doc = init_config_doc(root, base_config=base_config, overrides=overrides)
    write_json(paths["local"], doc)
    return {"written": True, "path": str(paths["local"])}


def env_for_tools(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    if config.get("user_home"):
        env["FLEET_USER_HOME"] = str(config["user_home"])
    if config.get("registry_dir"):
        env["FLEET_REG_DIR"] = str(config["registry_dir"])
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def machine_id(config: dict[str, Any]) -> str:
    raw = str(config.get("machine_id") or socket.gethostname() or "machine").strip()
    chars = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_", "."):
            chars.append(ch.lower())
        else:
            chars.append("-")
    slug = "".join(chars).strip("-.")
    return slug or "machine"


def command_exists(command: str) -> bool:
    if not command:
        return False
    p = Path(command)
    if p.exists():
        return True
    return shutil.which(command) is not None


def step_id_slug(value: str) -> str:
    chars = [
        ch.lower() if ch.isalnum() or ch in ("-", "_", ".") else "-"
        for ch in str(value)
    ]
    slug = "".join(chars).strip("-.")
    return slug or "item"


def command_input_missing(part: str, root: Path) -> bool:
    if not part or part.startswith("-"):
        return False
    if Path(part).suffix.lower() not in {".py", ".ps1", ".sh", ".cmd", ".bat"}:
        return False
    path = resolve_path(part, root)
    return bool(path and not path.exists())


def command_readiness(cmd: list[str], root: Path) -> dict[str, Any]:
    if not cmd:
        return {"ok": False, "detail": "command is not configured"}
    exists_fn = pane_attr("command_exists", command_exists)
    missing = []
    if not exists_fn(cmd[0]):
        missing.append(f"command not found: {cmd[0]}")
    for part in cmd[1:]:
        if command_input_missing(part, root):
            missing.append(f"missing input: {part}")
    detail = display_command(cmd)
    if missing:
        detail = f"{detail}; {'; '.join(missing)}" if detail else "; ".join(missing)
    return {
        "ok": not missing,
        "detail": detail,
        "command": cmd,
    }


def expand_cmd(parts: list[str], config: dict[str, Any]) -> list[str]:
    mapping = {
        "repo": config["root"],
        "root": config["root"],
        "job_dir": config.get("job_dir", ""),
        "python": config.get("python", sys.executable),
        "target": str(config.get("target", 4)),
        "claude_exe": config.get("claude_exe", ""),
        "user_home": config.get("user_home", ""),
        "registry_dir": config.get("registry_dir", ""),
        "machine_id": machine_id(config),
    }
    return [str(part).format(**mapping) for part in parts]


def expand_value(value: Any, config: dict[str, Any]) -> str:
    return str(value).format(
        repo=config["root"],
        root=config["root"],
        job_dir=config.get("job_dir", ""),
        python=config.get("python", sys.executable),
        target=str(config.get("target", 4)),
        claude_exe=config.get("claude_exe", ""),
        user_home=config.get("user_home", ""),
        registry_dir=config.get("registry_dir", ""),
        machine_id=machine_id(config),
    )


def run(
    args: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 30,
    *,
    via_files: bool = False,
) -> subprocess.CompletedProcess[str]:
    if not via_files:
        return subprocess.run(
            args,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    # File-captured variant for spawns that may launch a DETACHED grandchild (the
    # supervisor watchdog relaunching run_supervise_loop; the resume watchdog launching
    # claude). Such a grandchild inherits this child's stdout/stderr handles; with
    # capture_output=True those are OS pipes, so our read blocks on EOF until the
    # long-lived grandchild also closes them -- which never happens, hanging the tick.
    # `timeout` cannot rescue it: after it kills the (already-exited) immediate child, the
    # final drain re-blocks on the same still-open pipe. Redirecting to temp files removes
    # the pipe entirely -- we wait only for the immediate child to exit (bounded by
    # timeout), then read the files; an inherited file handle can't block us.
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            env=env,
            stdout=out_f,
            stderr=err_f,
            timeout=timeout,
            check=False,
        )
        out_f.seek(0)
        err_f.seek(0)
        out = out_f.read().decode("utf-8", "replace")
        err = err_f.read().decode("utf-8", "replace")
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def git(args: list[str], root: Path, check: bool = False) -> subprocess.CompletedProcess[str]:
    run_fn = pane_attr("run", run)
    proc = run_fn(["git", *args], root, timeout=30)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc


def git_stdout(args: list[str], root: Path, check: bool = True) -> str:
    git_fn = pane_attr("git", git)
    proc = git_fn(args, root, check=check)
    return proc.stdout


def display_command(args: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(args)
    import shlex

    return shlex.join(args)


def powershell_exe() -> str | None:
    return shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh")


def platform_is_windows() -> bool:
    return platform.system().lower() == "windows"


def sh_exe() -> str:
    return shutil.which("sh") or "sh"


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def read_json_from_text(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    try:
        doc = json.loads(raw)
        return doc if isinstance(doc, dict) else {}
    except ValueError:
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(raw):
            if ch != "{":
                continue
            try:
                doc, _ = decoder.raw_decode(raw[idx:])
            except ValueError:
                continue
            return doc if isinstance(doc, dict) else {}
    return {}
