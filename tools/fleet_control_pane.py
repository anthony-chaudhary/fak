#!/usr/bin/env python3
"""One control pane for the fleet's local agent-loop runtime.

The existing fleet recovery layer already has the hard parts: transcript-derived
session state, account availability, watchdog scripts, and a safe git sync check.
This module composes those pieces behind one host-portable command so each
machine can answer the same questions without hand-editing local paths.

Commands:
  init          write tools/_registry/control_pane.local.json for this machine
  status        print the pane; optionally refresh sessions and write snapshots
  tick          one recurring keep-alive tick; schedule this on each machine
  recover       dry-run/apply one live recovery pass for autonomous loops
  supervisor    inspect or restart the standing supervisor
  fleet         aggregate every published machine snapshot this pane can see
  sync          inspect/apply safe fast-forward sync without plain git pull
  publish       push this branch only when it is safely ahead of the remote
  commit        stage and commit an explicit path set, refusing foreign staged files
  bootstrap     apply the machine setup plan (local config + control tick)
  doctor        check prerequisites and installation state
  setup-plan    print the idempotent commands this host should run
  loop-list     list configured loops, their source, and command readiness
  loop-check    run one configured loop check and optional recovery
  loop-audit    run every enabled loop once and bucket each healthy/action/broken
  loop-scaffold create starter loop status/recovery files and registration
  loop-set      enable/disable or tune an existing loop without retyping commands
  loop-add      add a loop to the tracked catalog or machine-local config
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import platform
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_version  # noqa: E402


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
    "tools/fleet_version.py",
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


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_now() -> str:
    return now_utc().isoformat(timespec="seconds").replace("+00:00", "Z")


def repo_root(start: Path | None = None) -> Path:
    return fleet_version.repo_root(start or Path(__file__))


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
    for attempt in range(ATOMIC_WRITE_REPLACE_ATTEMPTS):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == ATOMIC_WRITE_REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(delay)
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


def parse_command_arg(value: str | None) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not all(isinstance(part, str) for part in parsed):
            raise ValueError("command JSON must be a list of strings")
        return parsed
    import shlex

    parts = shlex.split(raw, posix=(os.name != "nt"))
    if os.name == "nt":
        parts = [part.strip('"') for part in parts]
    return parts


def loop_config_command(config: dict[str, Any], loop_name: str, spec: dict[str, Any], *, scope: str = "local") -> list[str]:
    args = [
        str(config.get("python") or sys.executable),
        "tools/fleet_control_pane.py",
        "loop-add",
        loop_name,
    ]
    if scope != "local":
        args.extend(["--scope", scope])
    args.extend(["--status-cmd", json.dumps(spec.get("status_cmd") or [])])
    if spec.get("recover_cmd"):
        args.extend(["--recover-cmd", json.dumps(spec["recover_cmd"])])
    if spec.get("auto_recover"):
        args.append("--auto-recover")
    elif spec.get("auto_recover") is False:
        args.append("--no-auto-recover")
    if spec.get("action"):
        args.extend(["--action", str(spec["action"])])
    if spec.get("timeout_s") is not None:
        args.extend(["--timeout-s", str(spec["timeout_s"])])
    if spec.get("recover_timeout_s") is not None:
        args.extend(["--recover-timeout-s", str(spec["recover_timeout_s"])])
    if spec.get("cwd"):
        args.extend(["--cwd", str(spec["cwd"])])
    for code in spec.get("ok_returncodes") or []:
        args.extend(["--ok-returncode", str(code)])
    if spec.get("enabled") is False:
        args.append("--disabled")
    args.append("--apply")
    return args


def loop_config_plan(
    config: dict[str, Any],
    *,
    loop_name: str,
    scope: str = "local",
    status_cmd: list[str],
    recover_cmd: list[str] | None = None,
    auto_recover: bool | None = None,
    action: str = "",
    timeout_s: int | None = None,
    recover_timeout_s: int | None = None,
    cwd: str = "",
    ok_returncodes: list[int] | None = None,
    enabled: bool = True,
    apply: bool = False,
) -> dict[str, Any]:
    name = str(loop_name or "").strip()
    scope = str(scope or "local").strip().lower()
    root_paths = config_paths(Path(config["root"]))
    paths = config.get("_paths") or {}
    path_key = "loop_catalog" if scope == "repo" else "local"
    target_path = Path(paths.get(path_key) or root_paths[path_key])
    existing = read_json(target_path)
    if not existing:
        existing = {
            "schema": LOOP_CATALOG_SCHEMA if scope == "repo" else CONFIG_SCHEMA,
            "_doc": (
                "Tracked fleet control pane loop catalog. Machine-local config may override any loop."
                if scope == "repo"
                else "Machine-local fleet control pane config. This file is ignored by git."
            ),
        }
    loops = existing.get("loops") if isinstance(existing.get("loops"), dict) else {}
    current = loops.get(name) if isinstance(loops.get(name), dict) else {}
    spec = dict(current)
    spec["enabled"] = bool(enabled)
    spec["status_cmd"] = status_cmd
    if recover_cmd is not None:
        if recover_cmd:
            spec["recover_cmd"] = recover_cmd
        else:
            spec.pop("recover_cmd", None)
    if auto_recover is not None:
        spec["auto_recover"] = bool(auto_recover)
    elif "auto_recover" not in spec:
        spec["auto_recover"] = False
    if action:
        spec["action"] = action
    if timeout_s is not None:
        spec["timeout_s"] = int(timeout_s)
    if recover_timeout_s is not None:
        spec["recover_timeout_s"] = int(recover_timeout_s)
    if cwd:
        spec["cwd"] = cwd
    if ok_returncodes:
        spec["ok_returncodes"] = [int(code) for code in ok_returncodes]

    new_doc = dict(existing)
    new_loops = dict(loops)
    new_loops[name] = spec
    new_doc["loops"] = new_loops
    changed = existing.get("loops") != new_loops
    ok = bool(scope in {"local", "repo"} and name and status_cmd)
    if apply and ok:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(target_path, new_doc)
    command = display_command(loop_config_command(config, name, spec, scope=scope))
    followup_commands = []
    if scope == "repo" and ok and changed:
        commit_action = "update" if current else "add"
        followup_commands.append(display_command([
            str(config.get("python") or sys.executable),
            "tools/fleet_control_pane.py",
            "commit",
            "--dirty-group",
            "tools/fleet-control-pane",
            "-m",
            f"tools: {commit_action} {name} loop",
        ]))
    target_label = "repo loop catalog" if scope == "repo" else "local config"
    return {
        "schema": LOOP_CONFIG_SCHEMA,
        "generated_utc": iso_now(),
        "apply": apply,
        "ok": ok,
        "scope": scope,
        "changed": changed,
        "config_path": str(target_path),
        "local_config": str(target_path) if scope == "local" else None,
        "loop": name,
        "spec": spec,
        "reason": (
            "scope must be 'local' or 'repo'" if scope not in {"local", "repo"}
            else "loop name and --status-cmd are required" if not (name and status_cmd)
            else f"loop registered in {target_label}" if apply and changed
            else (f"loop already matched in {target_label}" if apply else f"dry run; pass --apply to write {target_label}")
        ),
        "command": command,
        "followup_commands": followup_commands,
    }


def loop_add_template_command(config: dict[str, Any]) -> str:
    return display_command([
        str(config.get("python") or sys.executable),
        "tools/fleet_control_pane.py",
        "loop-add",
        "NAME",
        "--scope",
        "repo",
        "--status-cmd",
        '["{python}","tools/NAME_status.py","--json"]',
    ])


def loop_scaffold_template_command(config: dict[str, Any]) -> str:
    return display_command([
        str(config.get("python") or sys.executable),
        "tools/fleet_control_pane.py",
        "loop-scaffold",
        "NAME",
    ])


def loop_script_slug(value: str) -> str:
    chars = []
    for ch in str(value).strip().lower():
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars).strip("-_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "loop"


def loop_status_scaffold_text(loop_name: str) -> str:
    loop_literal = json.dumps(loop_name)
    return (
        "#!/usr/bin/env python3\n"
        '"""Starter status check for a fleet-control-pane loop."""\n'
        "from __future__ import annotations\n\n"
        "import argparse\n"
        "import json\n\n"
        f"LOOP_NAME = {loop_literal}\n\n\n"
        "def main() -> int:\n"
        "    parser = argparse.ArgumentParser(description=f\"Status check for {LOOP_NAME}.\")\n"
        "    parser.add_argument(\"--json\", action=\"store_true\")\n"
        "    args = parser.parse_args()\n"
        "    payload = {\n"
        "        \"ok\": False,\n"
        "        \"reason\": f\"TODO: implement {LOOP_NAME} status check\",\n"
        "    }\n"
        "    if args.json:\n"
        "        print(json.dumps(payload))\n"
        "    else:\n"
        "        print(payload[\"reason\"])\n"
        "    return 0\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    raise SystemExit(main())\n"
    )


def loop_recover_scaffold_text(loop_name: str) -> str:
    loop_literal = json.dumps(loop_name)
    return (
        "#!/usr/bin/env python3\n"
        '"""Starter recovery action for a fleet-control-pane loop."""\n'
        "from __future__ import annotations\n\n"
        "import json\n\n"
        f"LOOP_NAME = {loop_literal}\n\n\n"
        "def main() -> int:\n"
        "    payload = {\n"
        "        \"ok\": False,\n"
        "        \"reason\": f\"TODO: implement {LOOP_NAME} recovery action\",\n"
        "    }\n"
        "    print(json.dumps(payload))\n"
        "    return 1\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    raise SystemExit(main())\n"
    )


def loop_scaffold_command(
    config: dict[str, Any],
    name: str,
    *,
    enabled: bool = False,
    auto_recover: bool = False,
    force: bool = False,
) -> list[str]:
    args = [
        str(config.get("python") or sys.executable),
        "tools/fleet_control_pane.py",
        "loop-scaffold",
        name,
    ]
    if enabled:
        args.append("--enabled")
    if auto_recover:
        args.append("--auto-recover")
    if force:
        args.append("--force")
    args.append("--apply")
    return args


def loop_scaffold_plan(
    config: dict[str, Any],
    name: str,
    *,
    enabled: bool = False,
    auto_recover: bool = False,
    force: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    loop_name = str(name or "").strip()
    slug = loop_script_slug(loop_name)
    root = Path(config["root"])
    status_rel = f"tools/loops/{slug}_status.py"
    recover_rel = f"tools/loops/{slug}_recover.py"
    catalog_rel = "tools/control_pane.loops.json"
    files = [
        {
            "role": "status",
            "path": status_rel,
            "exists": (root / status_rel).exists(),
            "will_write": apply,
        },
        {
            "role": "recover",
            "path": recover_rel,
            "exists": (root / recover_rel).exists(),
            "will_write": apply,
        },
    ]
    conflicts = [
        file["path"] for file in files
        if file["exists"] and not force
    ]
    loop_exists = loop_name in (config.get("loops") or {})
    spec = {
        "enabled": bool(enabled),
        "status_cmd": ["{python}", status_rel, "--json"],
        "recover_cmd": ["{python}", recover_rel],
        "auto_recover": bool(auto_recover),
        "timeout_s": 20,
        "action": f"inspect or finish {loop_name} loop",
    }
    ok = bool(loop_name and not conflicts and (force or not loop_exists))
    reason = (
        "loop name is required" if not loop_name
        else "refusing to overwrite existing file(s): " + ", ".join(conflicts) if conflicts
        else f"loop already exists; pass --force to update {loop_name}" if loop_exists and not force
        else "dry run; pass --apply to write scaffold files and repo loop catalog" if not apply
        else "loop scaffold written"
    )
    register_doc = None
    if apply and ok:
        write_text_atomic(root / status_rel, loop_status_scaffold_text(loop_name))
        write_text_atomic(root / recover_rel, loop_recover_scaffold_text(loop_name))
        register_doc = loop_config_plan(
            config,
            loop_name=loop_name,
            scope="repo",
            status_cmd=list(spec["status_cmd"]),
            recover_cmd=list(spec["recover_cmd"]),
            auto_recover=auto_recover,
            action=str(spec["action"]),
            timeout_s=int(spec["timeout_s"]),
            enabled=enabled,
            apply=True,
        )
    commands = []
    followup_commands = []
    if not apply and ok:
        commands.append(display_command(loop_scaffold_command(
            config,
            loop_name or "NAME",
            enabled=enabled,
            auto_recover=auto_recover,
            force=force,
        )))
    if apply and ok:
        followup_commands.append(display_command([
            str(config.get("python") or sys.executable),
            "tools/fleet_control_pane.py",
            "loop-check",
            loop_name,
            "--recover",
        ]))
        followup_commands.append(display_command([
            str(config.get("python") or sys.executable),
            "tools/fleet_control_pane.py",
            "commit",
            "--path",
            status_rel,
            "--path",
            recover_rel,
            "--path",
            catalog_rel,
            "-m",
            f"tools: scaffold {loop_name} loop",
        ]))
    return {
        "schema": LOOP_SCAFFOLD_SCHEMA,
        "generated_utc": iso_now(),
        "ok": ok,
        "apply": apply,
        "loop": loop_name,
        "slug": slug,
        "enabled": bool(enabled),
        "auto_recover": bool(auto_recover),
        "force": bool(force),
        "reason": reason,
        "files": files,
        "spec": spec,
        "register": register_doc,
        "commands": commands,
        "followup_commands": followup_commands,
    }


def loop_set_command(
    config: dict[str, Any],
    name: str,
    *,
    scope: str = "repo",
    enabled: bool | None = None,
    auto_recover: bool | None = None,
) -> list[str]:
    args = [
        str(config.get("python") or sys.executable),
        "tools/fleet_control_pane.py",
        "loop-set",
        name,
    ]
    if scope != "repo":
        args.extend(["--scope", scope])
    if enabled is True:
        args.append("--enable")
    elif enabled is False:
        args.append("--disable")
    if auto_recover is True:
        args.append("--auto-recover")
    elif auto_recover is False:
        args.append("--no-auto-recover")
    args.append("--apply")
    return args


def loop_set_plan(
    config: dict[str, Any],
    name: str,
    *,
    scope: str = "repo",
    enabled: bool | None = None,
    auto_recover: bool | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    loop_name = str(name or "").strip()
    scope = str(scope or "repo").strip().lower()
    root = Path(config["root"])
    root_paths = config_paths(root)
    paths = config.get("_paths") or {}
    path_key = "loop_catalog" if scope == "repo" else "local"
    target_path = Path(paths.get(path_key) or root_paths[path_key])
    available = sorted(
        str(candidate)
        for candidate, spec in (config.get("loops") or {}).items()
        if isinstance(spec, dict)
    )
    effective = (config.get("loops") or {}).get(loop_name)
    if scope == "repo":
        tracked = tracked_default_config(root)
        effective = (tracked.get("loops") or {}).get(loop_name)
    existing = read_json(target_path)
    if not existing:
        existing = {
            "schema": LOOP_CATALOG_SCHEMA if scope == "repo" else CONFIG_SCHEMA,
            "_doc": (
                "Tracked fleet control pane loop catalog. Machine-local config may override any loop."
                if scope == "repo"
                else "Machine-local fleet control pane config. This file is ignored by git."
            ),
        }
    loops = existing.get("loops") if isinstance(existing.get("loops"), dict) else {}
    current = loops.get(loop_name) if isinstance(loops.get(loop_name), dict) else {}
    base_spec = current or (effective if isinstance(effective, dict) else {})
    known = bool(base_spec)
    spec = dict(base_spec) if known else {}
    if known and enabled is not None:
        spec["enabled"] = bool(enabled)
    if known and auto_recover is not None:
        spec["auto_recover"] = bool(auto_recover)
    changed = False
    new_doc = dict(existing)
    new_loops = dict(loops)
    if spec:
        new_loops[loop_name] = spec
        new_doc["loops"] = new_loops
        changed = existing.get("loops") != new_loops
    ok = bool(
        scope in {"local", "repo"}
        and loop_name
        and spec
        and (enabled is not None or auto_recover is not None)
    )
    reason = (
        "scope must be 'local' or 'repo'" if scope not in {"local", "repo"}
        else "loop name is required" if not loop_name
        else "loop is not known in this scope" if not spec
        else "select at least one loop setting" if enabled is None and auto_recover is None
        else "loop already matched requested settings" if not changed
        else "dry run; pass --apply to update loop settings" if not apply
        else "loop settings updated"
    )
    if apply and ok and changed:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(target_path, new_doc)
    commands = []
    if not apply and ok and changed:
        commands.append(display_command(loop_set_command(
            config,
            loop_name,
            scope=scope,
            enabled=enabled,
            auto_recover=auto_recover,
        )))
    followup_commands = []
    if ok and changed:
        followup_commands.append(display_command([
            str(config.get("python") or sys.executable),
            "tools/fleet_control_pane.py",
            "loop-check",
            loop_name,
            "--recover",
        ]))
        if scope == "repo":
            commit_path = (display_path(target_path, root) or str(target_path)).replace("\\", "/")
            followup_commands.append(display_command([
                str(config.get("python") or sys.executable),
                "tools/fleet_control_pane.py",
                "commit",
                "--path",
                commit_path,
                "-m",
                f"tools: update {loop_name} loop settings",
            ]))
    return {
        "schema": LOOP_SET_SCHEMA,
        "generated_utc": iso_now(),
        "ok": ok,
        "apply": apply,
        "scope": scope,
        "changed": changed,
        "loop": loop_name,
        "config_path": str(target_path),
        "local_config": str(target_path) if scope == "local" else None,
        "available": available,
        "reason": reason,
        "spec": spec,
        "commands": commands,
        "followup_commands": followup_commands,
    }


def loop_source_entries(root: Path) -> dict[str, list[dict[str, Any]]]:
    paths = config_paths(root)
    source_defs = [
        ("example", paths["example"]),
        ("repo", paths["loop_catalog"]),
        ("local", paths["local"]),
    ]
    sources: dict[str, list[dict[str, Any]]] = {}
    for source, path in source_defs:
        doc = read_json(path)
        loops = doc.get("loops") if isinstance(doc.get("loops"), dict) else {}
        for name, spec in sorted(loops.items()):
            if not isinstance(spec, dict):
                continue
            sources.setdefault(str(name), []).append({
                "source": source,
                "path": str(path),
                "display_path": display_path(path, root),
                "enabled": spec.get("enabled", True) is not False,
                "has_status_cmd": bool(spec.get("status_cmd")),
                "has_recover_cmd": bool(spec.get("recover_cmd")),
                "auto_recover": bool(spec.get("auto_recover")),
            })
    return sources


def loop_list(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    sources = loop_source_entries(root)
    loops: list[dict[str, Any]] = []
    for name, spec in sorted((config.get("loops") or {}).items()):
        if not isinstance(spec, dict):
            continue
        entries = sources.get(str(name), [])
        source = entries[-1]["source"] if entries else "effective"
        status_cmd = expand_cmd(list(spec.get("status_cmd") or []), config)
        recover_cmd = expand_cmd(list(spec.get("recover_cmd") or []), config)
        status_ready = command_readiness(status_cmd, root)
        recover_ready = command_readiness(recover_cmd, root) if recover_cmd else {
            "ok": not bool(spec.get("auto_recover")),
            "detail": "recover command is not configured",
            "command": [],
        }
        enabled = spec.get("enabled") is not False
        ready = bool(
            not enabled
            or (
                status_ready.get("ok")
                and (not spec.get("auto_recover") or recover_ready.get("ok"))
            )
        )
        loops.append({
            "name": str(name),
            "enabled": enabled,
            "ready": ready,
            "source": source,
            "sources": entries,
            "overridden": len(entries) > 1,
            "status_cmd": status_cmd,
            "status_ready": status_ready,
            "recover_cmd": recover_cmd,
            "recover_ready": recover_ready,
            "auto_recover": bool(spec.get("auto_recover")),
            "action": str(spec.get("action") or ""),
            "cwd": str(spec.get("cwd") or ""),
            "timeout_s": spec.get("timeout_s"),
            "recover_timeout_s": spec.get("recover_timeout_s"),
        })
    enabled_count = sum(1 for loop in loops if loop.get("enabled"))
    blocked = [
        loop for loop in loops
        if loop.get("enabled") and not loop.get("ready")
    ]
    return {
        "schema": LOOP_LIST_SCHEMA,
        "generated_utc": iso_now(),
        "ok": not blocked,
        "count": len(loops),
        "enabled": enabled_count,
        "disabled": len(loops) - enabled_count,
        "blocked": len(blocked),
        "paths": {
            key: str(path)
            for key, path in config_paths(root).items()
        },
        "loops": loops,
        "commands": [
            loop_scaffold_template_command(config),
            loop_add_template_command(config),
        ] if enabled_count == 0 else [],
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
    missing = []
    if not command_exists(cmd[0]):
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
    proc = run(["git", *args], root, timeout=30)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc


def git_stdout(args: list[str], root: Path, check: bool = True) -> str:
    proc = git(args, root, check=check)
    return proc.stdout


def path_compare_key(path: Path) -> str:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    return os.path.normcase(os.path.normpath(str(resolved)))


def configured_hooks_path(root: Path, raw: str) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else root / path


def trunk_guard_status(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    python = str(config.get("python") or sys.executable)
    installer = root / TRUNK_GUARD_INSTALLER_REL
    hook = root / TRUNK_GUARD_HOOK_REL
    hooks_dir = root / TRUNK_GUARD_HOOKS_REL
    command = [python, TRUNK_GUARD_INSTALLER_REL] if installer.exists() else []
    status: dict[str, Any] = {
        "supported": False,
        "installed": False,
        "ok": True,
        "setup_ok": True,
        "state": "skip",
        "detail": "not a git worktree",
        "installer": TRUNK_GUARD_INSTALLER_REL,
        "installer_exists": installer.exists(),
        "hook": TRUNK_GUARD_HOOK_REL,
        "hook_exists": hook.exists(),
        "expected": TRUNK_GUARD_HOOKS_REL,
        "actual": "",
        "command": command,
    }
    if not (root / ".git").exists():
        return status

    git_probe = git(["rev-parse", "--is-inside-work-tree"], root)
    if git_probe.returncode != 0:
        status.update({
            "ok": False,
            "setup_ok": False,
            "state": "blocked",
            "detail": git_probe.stderr.strip() or "git worktree probe failed",
        })
        return status
    status["supported"] = True

    if not installer.exists():
        status.update({
            "ok": False,
            "setup_ok": False,
            "state": "blocked",
            "detail": f"missing {TRUNK_GUARD_INSTALLER_REL}",
        })
        return status
    if not hook.exists():
        status.update({
            "ok": False,
            "setup_ok": False,
            "state": "blocked",
            "detail": f"missing {TRUNK_GUARD_HOOK_REL}",
        })
        return status

    configured = git(["config", "--get", "core.hooksPath"], root)
    actual = configured.stdout.strip() if configured.returncode == 0 else ""
    actual_path = configured_hooks_path(root, actual)
    installed = bool(
        actual_path
        and path_compare_key(actual_path) == path_compare_key(hooks_dir)
    )
    status.update({
        "installed": installed,
        "ok": installed,
        "setup_ok": True,
        "state": "done" if installed else "todo",
        "actual": actual or "unset",
        "detail": (
            f"core.hooksPath={actual}"
            if installed
            else f"core.hooksPath={actual or 'unset'}; expected {TRUNK_GUARD_HOOKS_REL}"
        ),
    })
    return status


def parse_git_status_porcelain(raw: str) -> dict[str, Any]:
    entries: list[dict[str, str]] = []
    fields = [f for f in raw.split("\0") if f]
    i = 0
    while i < len(fields):
        item = fields[i]
        code = item[:2]
        path = item[3:] if len(item) > 3 else ""
        if code and code[0] in ("R", "C") and i + 1 < len(fields):
            old_path = fields[i + 1]
            entries.append({"code": code, "path": path, "old_path": old_path})
            i += 2
            continue
        entries.append({"code": code, "path": path})
        i += 1

    counts = collections.Counter()
    for entry in entries:
        code = entry["code"]
        if code == "??":
            counts["untracked"] += 1
        if "M" in code:
            counts["modified"] += 1
        if "A" in code:
            counts["added"] += 1
        if "D" in code:
            counts["deleted"] += 1
        if "R" in code:
            counts["renamed"] += 1
        if "C" in code:
            counts["copied"] += 1
        if "U" in code:
            counts["unmerged"] += 1
    return {
        "dirty": bool(entries),
        "dirty_total": len(entries),
        "counts": dict(counts),
        "entries": entries,
    }


def git_worktree_summary(root: Path) -> dict[str, Any]:
    proc = git(["status", "--porcelain=v1", "-z"], root, check=False)
    if proc.returncode != 0:
        return {
            "status_available": False,
            "dirty": False,
            "dirty_total": 0,
            "counts": {},
            "entries": [],
            "unmerged_paths": [],
            "merge_in_progress": False,
            "reason": proc.stderr.strip() or proc.stdout.strip() or "git status failed",
        }
    status = parse_git_status_porcelain(proc.stdout)
    entries = status.get("entries") or []
    status["status_available"] = True
    status["unmerged_paths"] = sorted(
        str(entry.get("path"))
        for entry in entries
        if entry.get("path") and "U" in str(entry.get("code") or "")
    )
    status["merge_in_progress"] = git(["rev-parse", "--verify", "-q", "MERGE_HEAD"], root, check=False).returncode == 0
    return status


def worktree_allow_branches(config: dict[str, Any]) -> list[str]:
    raw = config.get("worktree_allow_branches", DEFAULT_WORKTREE_ALLOW_BRANCHES)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(branch) for branch in raw if str(branch).strip()]


def worktree_doctor_command(config: dict[str, Any], *, prune: bool = False, fetch: bool = True) -> list[str]:
    cmd = [str(config.get("python") or sys.executable), "tools/worktree_doctor.py"]
    if prune:
        cmd.append("--prune")
    if fetch:
        cmd.append("--fetch")
    master_ref = str(config.get("worktree_master_ref") or DEFAULT_WORKTREE_MASTER_REF)
    if master_ref != DEFAULT_WORKTREE_MASTER_REF:
        cmd.extend(["--master-ref", master_ref])
    for branch in worktree_allow_branches(config):
        cmd.extend(["--allow-branch", branch])
    return cmd


def collect_worktree_doctor(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    master_ref = str(config.get("worktree_master_ref") or DEFAULT_WORKTREE_MASTER_REF)
    try:
        import worktree_doctor  # type: ignore

        sigs = worktree_doctor.collect(str(root), master_ref, fetch=False)
        plan = worktree_doctor.make_plan(
            sigs,
            master_ref,
            allow_branches=worktree_allow_branches(config),
        )
        prune = list(plan.get("prune") or [])
        blocked = list(plan.get("blocked") or [])
        retained = list(plan.get("retained") or [])
        return {
            "available": True,
            "master_ref": master_ref,
            "total": len(sigs),
            "converged": bool(plan.get("converged")),
            "needs_human": bool(plan.get("needs_human")),
            "primary_offtrack": bool(plan.get("primary_offtrack")),
            "keeper": plan.get("keeper"),
            "prune_count": len(prune),
            "blocked_count": len(blocked),
            "retained_count": len(retained),
            "prune": prune[:10],
            "prune_omitted": max(0, len(prune) - 10),
            "blocked": blocked[:10],
            "blocked_omitted": max(0, len(blocked) - 10),
            "retained": retained[:10],
            "commands": {
                "inspect": display_command(worktree_doctor_command(config, prune=False, fetch=True)),
                "prune": display_command(worktree_doctor_command(config, prune=True, fetch=True)),
            },
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "commands": {
                "inspect": display_command(worktree_doctor_command(config, prune=False, fetch=True)),
            },
        }


def git_mutation_guard(root: Path) -> dict[str, Any]:
    worktree = git_worktree_summary(root)
    unmerged_paths = [str(path) for path in worktree.get("unmerged_paths") or []]
    blockers: list[str] = []
    if worktree.get("merge_in_progress"):
        blockers.append("merge is in progress")
    if unmerged_paths:
        blockers.append(f"worktree has {len(unmerged_paths)} unmerged path(s)")
    return {
        "blocked": bool(blockers),
        "blockers": blockers,
        "reason": "; ".join(blockers),
        "worktree": {
            "status_available": worktree.get("status_available", False),
            "dirty": worktree.get("dirty", False),
            "dirty_total": worktree.get("dirty_total", 0),
            "counts": worktree.get("counts") or {},
            "unmerged_paths": unmerged_paths,
            "merge_in_progress": bool(worktree.get("merge_in_progress")),
        },
    }


def normalize_repo_path(root: Path, raw: str, *, allow_dir: bool = False) -> str:
    if not raw or not str(raw).strip():
        raise ValueError("empty path")
    root = root.resolve()
    candidate = Path(os.path.expandvars(os.path.expanduser(str(raw))))
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes repo: {raw}") from exc
    if ".git" in rel.parts:
        raise ValueError(f"path enters .git: {raw}")
    if resolved.exists() and resolved.is_dir() and not allow_dir:
        raise ValueError(f"path is a directory; pass explicit files instead: {raw}")
    return rel.as_posix()


def split_z(raw: str) -> list[str]:
    return [p for p in raw.split("\0") if p]


def staged_paths(root: Path) -> list[str]:
    return split_z(git_stdout(["diff", "--cached", "--name-only", "-z"], root))


def path_status(root: Path, paths: list[str]) -> list[dict[str, str]]:
    if not paths:
        return []
    return parse_git_status_porcelain(
        git_stdout(["status", "--porcelain=v1", "-z", "--", *paths], root)
    )["entries"]


def commit_plan(
    config: dict[str, Any],
    *,
    paths: list[str],
    message: str,
    apply: bool = False,
    allow_dir: bool = False,
) -> dict[str, Any]:
    root = Path(config["root"]).resolve()
    selected = sorted(dict.fromkeys(
        normalize_repo_path(root, p, allow_dir=allow_dir) for p in paths
    ))
    if not selected:
        return {
            "schema": COMMIT_SCHEMA,
            "ok": False,
            "applied": False,
            "reason": "no paths selected",
        }
    if not message or not message.strip():
        return {
            "schema": COMMIT_SCHEMA,
            "ok": False,
            "applied": False,
            "paths": selected,
            "reason": "commit message is required",
        }

    before_staged = staged_paths(root)
    foreign_before = [p for p in before_staged if p not in selected]
    status_before = path_status(root, selected)
    merge_in_progress = git_worktree_summary(root).get("merge_in_progress", False)
    unmerged_selected = sorted(
        {str(entry.get("path")) for entry in status_before if "U" in str(entry.get("code") or "")}
    )
    changed_selected = sorted({e["path"] for e in status_before})
    commands = [
        ["git", "add", "--", *selected],
        ["git", "commit", "-s", "-m", message.strip()],
    ]
    base: dict[str, Any] = {
        "schema": COMMIT_SCHEMA,
        "generated_utc": iso_now(),
        "ok": False,
        "applied": False,
        "root": str(root),
        "paths": selected,
        "message": message.strip(),
        "status": status_before,
        "changed_selected": changed_selected,
        "merge_in_progress": merge_in_progress,
        "preexisting_staged": before_staged,
        "foreign_staged": foreign_before,
        "commands": [display_command(c) for c in commands],
    }
    if merge_in_progress:
        return {
            **base,
            "reason": "merge is in progress; finish or abort the merge before using the pane commit helper",
        }
    if unmerged_selected:
        return {
            **base,
            "unmerged_paths": unmerged_selected,
            "reason": "selected paths contain unresolved merge conflicts",
        }
    if foreign_before:
        return {
            **base,
            "reason": (
                "refusing because unrelated paths are already staged; "
                "commit or unstage them before using the pane commit helper"
            ),
        }
    if not changed_selected and not any(p in before_staged for p in selected):
        return {**base, "reason": "none of the selected paths have changes"}
    if not apply:
        return {**base, "ok": True, "reason": "dry run; pass --apply to stage and commit"}

    add_proc = git(["add", "--", *selected], root)
    after_staged = staged_paths(root)
    foreign_after = [p for p in after_staged if p not in selected]
    if add_proc.returncode != 0:
        return {
            **base,
            "reason": add_proc.stderr.strip() or add_proc.stdout.strip() or "git add failed",
            "git_add_returncode": add_proc.returncode,
        }
    if foreign_after:
        return {
            **base,
            "preexisting_staged": before_staged,
            "post_add_staged": after_staged,
            "foreign_staged": foreign_after,
            "reason": "refusing because staging selected paths exposed foreign staged paths",
        }
    if not after_staged:
        return {**base, "reason": "selected paths produced no staged changes"}
    commit_proc = git(["commit", "-s", "-m", message.strip()], root)
    if commit_proc.returncode != 0:
        return {
            **base,
            "post_add_staged": after_staged,
            "reason": commit_proc.stderr.strip() or commit_proc.stdout.strip() or "git commit failed",
            "git_commit_returncode": commit_proc.returncode,
        }
    sha = git_stdout(["rev-parse", "--short=12", "HEAD"], root).strip()
    return {
        **base,
        "ok": True,
        "applied": True,
        "commit": sha,
        "post_add_staged": after_staged,
        "stdout": commit_proc.stdout.strip()[-2000:],
        "stderr": commit_proc.stderr.strip()[-2000:],
    }


def dirty_group_key(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized in PANE_SOURCE_PATHS or normalized.startswith("tools/fleet_control_pane"):
        return "tools/fleet-control-pane"
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return "root"
    if len(parts) == 1:
        return "root"
    if parts[0] == "tools":
        if len(parts) == 2 and parts[1].startswith("_") and parts[1].endswith("_proof.png"):
            return "tools/proofs"
        return f"tools/{parts[1]}"
    if parts[0] == "fak":
        if len(parts) >= 3 and parts[1] in {"experiments", "internal"}:
            return f"fak/{parts[1]}/{parts[2]}"
        return "fak"
    return parts[0]


def dirty_group_commit_subject(group: str) -> str:
    return DIRTY_GROUP_SUBJECTS.get(group, f"TODO: describe {group} changes")


def dirty_commit_plan(
    entries: list[dict[str, str]],
    config: dict[str, Any],
    *,
    merge_in_progress: bool = False,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = {}
    for entry in entries:
        path = entry.get("path")
        if not path:
            continue
        groups.setdefault(dirty_group_key(path), []).append(entry)
    out = []
    python = str(config.get("python") or sys.executable)
    for key in sorted(groups):
        group_entries = groups[key]
        paths = sorted(dict.fromkeys(str(entry.get("path")) for entry in group_entries if entry.get("path")))
        has_unmerged = any("U" in str(entry.get("code") or "") for entry in group_entries)
        blocked = merge_in_progress or has_unmerged
        command: list[str] = [
            python,
            "tools/fleet_control_pane.py",
            "commit",
            "--dirty-group",
            key,
        ]
        subject = dirty_group_commit_subject(key)
        command.extend(["-m", subject])
        item: dict[str, Any] = {
            "group": key,
            "count": len(paths),
            "paths": paths,
            "suggested_subject": subject,
        }
        if blocked:
            item["blocked"] = True
            item["reason"] = (
                "finish or abort the merge before using pane commit"
                if merge_in_progress
                else "resolve merge conflicts before using pane commit"
            )
        else:
            item["command"] = display_command(command)
        out.append(item)
    return {"groups": out, "count": len(out)}


def dirty_group_selection(config: dict[str, Any], groups: list[str]) -> dict[str, Any]:
    requested = [group for group in groups if group]
    git_status = collect_git(config)
    plan = git_status.get("dirty_plan") or {}
    by_group = {
        str(group.get("group")): group
        for group in plan.get("groups") or []
        if group.get("group")
    }
    missing = [group for group in requested if group not in by_group]
    paths: list[str] = []
    for group in requested:
        paths.extend(str(path) for path in (by_group.get(group) or {}).get("paths") or [])
    return {
        "ok": not missing,
        "requested": requested,
        "missing": missing,
        "available": sorted(by_group),
        "paths": sorted(dict.fromkeys(paths)),
    }


def collect_git(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    if not (root / ".git").exists() or shutil.which("git") is None:
        return {"available": False, "reason": "not a git repo or git not on PATH"}
    branch = git(["rev-parse", "--abbrev-ref", "HEAD"], root).stdout.strip()
    head = git(["rev-parse", "--short=12", "HEAD"], root).stdout.strip()
    status = git_worktree_summary(root)
    entries = status.pop("entries", [])
    merge_in_progress = bool(status.get("merge_in_progress"))
    out: dict[str, Any] = {
        "available": True,
        "branch": branch,
        "head": head,
        **status,
        "dirty_sample": entries[:25],
        "dirty_omitted": max(0, len(entries) - 25),
        "dirty_plan": dirty_commit_plan(entries, config, merge_in_progress=merge_in_progress),
    }
    try:
        import safe_ff_sync  # type: ignore

        info = safe_ff_sync.assess(
            str(root),
            str(config.get("git_remote", "origin")),
            branch,
            do_fetch=False,
        )
        out["safe_ff"] = {
            k: info.get(k)
            for k in ("ok", "state", "reason", "target_ref", "write_count")
            if k in info
        }
        out["safe_ff"]["divergent_count"] = len(info.get("divergent", []) or [])
    except Exception as exc:  # git may have no remote, no upstream, etc.
        out["safe_ff"] = {"ok": None, "state": "unavailable", "reason": str(exc)}
    out["worktrees"] = collect_worktree_doctor(config)
    return out


def sync_plan(config: dict[str, Any], *, fetch: bool = False, apply: bool = False) -> dict[str, Any]:
    root = Path(config["root"])
    remote = str(config.get("git_remote", "origin"))
    mutation_guard = git_mutation_guard(root) if apply else None
    cmd = [
        str(config.get("python") or sys.executable),
        "tools/safe_ff_sync.py",
        "apply" if apply else "check",
        "--repo",
        str(root),
        "--remote",
        remote,
    ]
    if fetch:
        cmd.append("--fetch")
    cmd.append("--json")
    base: dict[str, Any] = {
        "schema": SYNC_SCHEMA,
        "generated_utc": iso_now(),
        "apply": apply,
        "fetch": fetch,
        "remote": remote,
        "commands": [display_command(cmd)],
    }
    try:
        import safe_ff_sync  # type: ignore

        branch = safe_ff_sync.current_branch(str(root))
        info = safe_ff_sync.assess(str(root), remote, branch, fetch)
        info["branch"] = branch
        applied = False
        if apply:
            if mutation_guard and mutation_guard.get("blocked"):
                return {
                    **base,
                    "ok": False,
                    "state": info.get("state"),
                    "branch": branch,
                    "info": info,
                    "applied": False,
                    "sync_blocked": True,
                    "blockers": mutation_guard.get("blockers") or [],
                    "worktree": mutation_guard.get("worktree") or {},
                    "reason": mutation_guard.get("reason") or "worktree is not safe for sync",
                }
            if info.get("state") == "in-sync":
                applied = False
            elif info.get("state") == "behind" and info.get("ok"):
                info["new_head"] = safe_ff_sync.apply_ff(str(root), remote, branch, info)
                info["applied"] = True
                applied = True
            else:
                info["applied"] = False
        state = str(info.get("state") or "unknown")
        ok = state == "in-sync" or (state == "behind" and bool(info.get("ok")))
        if apply and state == "behind":
            ok = applied
        return {
            **base,
            "ok": ok,
            "state": state,
            "branch": branch,
            "info": info,
            "applied": applied,
            "reason": info.get("reason", ""),
        }
    except Exception as exc:
        return {
            **base,
            "ok": False,
            "state": "unavailable",
            "applied": False,
            "reason": str(exc),
        }


def publish_plan(
    config: dict[str, Any],
    *,
    fetch: bool = True,
    apply: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    root = Path(config["root"])
    remote = str(config.get("git_remote", "origin"))
    base: dict[str, Any] = {
        "schema": PUBLISH_SCHEMA,
        "generated_utc": iso_now(),
        "apply": apply,
        "fetch": fetch,
        "allow_dirty": allow_dirty,
        "remote": remote,
        "applied": False,
    }
    try:
        import safe_ff_sync  # type: ignore

        branch = safe_ff_sync.current_branch(str(root))
        info = safe_ff_sync.assess(str(root), remote, branch, fetch)
        info["branch"] = branch
        push_cmd = ["git", "push", remote, f"HEAD:{branch}"]
        ahead = ahead_commits(root, remote, branch) if info.get("state") == "ahead" else {
            "target": f"{remote}/{branch}",
            "count": 0,
            "shown": 0,
            "limit": 20,
            "commits": [],
            "ok": True,
            "reason": "",
        }
        worktree = git_worktree_summary(root)
        worktree_doc = {
            "status_available": worktree.get("status_available", False),
            "dirty": worktree.get("dirty", False),
            "dirty_total": worktree.get("dirty_total", 0),
            "counts": worktree.get("counts") or {},
            "unmerged_paths": worktree.get("unmerged_paths") or [],
            "merge_in_progress": worktree.get("merge_in_progress", False),
        }
        doc: dict[str, Any] = {
            **base,
            "branch": branch,
            "state": info.get("state"),
            "info": info,
            "ahead_commits": ahead,
            "worktree": worktree_doc,
            "commands": [display_command(push_cmd)],
            "reason": info.get("reason", ""),
        }
        if info.get("state") == "in-sync":
            return {**doc, "ok": True, "reason": "branch already published"}
        if info.get("state") != "ahead":
            return {
                **doc,
                "ok": False,
                "reason": info.get("reason") or f"publish requires ahead state, got {info.get('state')}",
            }
        blockers: list[str] = []
        if worktree_doc["merge_in_progress"]:
            blockers.append("merge is in progress")
        if worktree_doc["unmerged_paths"]:
            blockers.append(f"worktree has {len(worktree_doc['unmerged_paths'])} unmerged path(s)")
        if int(worktree_doc["dirty_total"] or 0) > 0 and not allow_dirty:
            blockers.append(f"worktree has {worktree_doc['dirty_total']} dirty path(s)")
        if blockers:
            return {
                **doc,
                "ok": False,
                "publish_blocked": True,
                "blockers": blockers,
                "reason": "; ".join(blockers),
            }
        if not apply:
            return {
                **doc,
                "ok": True,
                "reason": "dry run; pass --apply to push the ahead commits",
            }
        proc = git(["push", remote, f"HEAD:{branch}"], root)
        return {
            **doc,
            "ok": proc.returncode == 0,
            "applied": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[-2000:],
            "stderr": proc.stderr.strip()[-2000:],
            "reason": proc.stderr.strip() or proc.stdout.strip() or ("published" if proc.returncode == 0 else "git push failed"),
        }
    except Exception as exc:
        return {
            **base,
            "ok": False,
            "state": "unavailable",
            "reason": str(exc),
        }


def ahead_commits(root: Path, remote: str, branch: str, limit: int = 20) -> dict[str, Any]:
    target = f"{remote}/{branch}"
    proc = git(["log", f"--max-count={limit}", "--format=%H%x00%h%x00%s", f"{target}..HEAD"], root, check=False)
    commits: list[dict[str, str]] = []
    if proc.returncode != 0:
        return {
            "target": target,
            "count": 0,
            "shown": 0,
            "limit": limit,
            "commits": commits,
            "ok": False,
            "reason": (proc.stderr or proc.stdout).strip()[-1000:],
        }
    for line in proc.stdout.splitlines():
        parts = line.split("\0", 2)
        if len(parts) != 3:
            continue
        sha, short, subject = parts
        commits.append({"sha": sha, "short": short, "subject": subject})
    count_proc = git(["rev-list", "--count", f"{target}..HEAD"], root, check=False)
    try:
        count = int((count_proc.stdout or "").strip())
    except ValueError:
        count = len(commits)
    return {
        "target": target,
        "count": count,
        "shown": len(commits),
        "limit": limit,
        "commits": commits,
        "ok": count_proc.returncode == 0,
        "reason": "" if count_proc.returncode == 0 else (count_proc.stderr or count_proc.stdout).strip()[-1000:],
    }


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


def summarize_registry(registry: dict[str, Any] | None) -> dict[str, Any]:
    if not registry:
        return {"exists": False, "sessions": 0, "actions": {}, "categories": {}}
    sessions = registry.get("sessions") or []
    accounts = registry.get("accounts") or []
    actions = collections.Counter(str(s.get("action", "")) for s in sessions)
    categories = collections.Counter(str(s.get("category", "")) for s in sessions)
    dispositions = collections.Counter(str(s.get("disp", "")) for s in sessions)
    available = [a for a in accounts if a.get("available")]
    blocked = [a for a in accounts if not a.get("available")]
    blocked_accounts = [blocked_account_snapshot(a) for a in blocked]
    blocked_kinds = collections.Counter(str(a.get("block_kind") or "unknown") for a in blocked_accounts)
    auth_blocked_sessions = dispositions.get("INFRA_AUTH", 0)
    return {
        "exists": True,
        "schema": registry.get("schema"),
        "generated_utc": registry.get("generated_utc"),
        "age_min": parse_iso_age_min(registry.get("generated_utc")),
        "sessions": len(sessions),
        "categories": dict(categories),
        "actions": dict(actions),
        "dispositions": dict(dispositions),
        "auto_resume": actions.get("AUTO_RESUME", 0),
        "surface": actions.get("SURFACE", 0),
        "auth_blocked": auth_blocked_sessions,
        "auth_blocked_actions": actions.get("BLOCKED_AUTH", 0),
        "supervised": actions.get("SUPERVISED", 0),
        "accounts": {
            "total": len(accounts),
            "available": len(available),
            "blocked_count": len(blocked),
            "blocked_kinds": dict(blocked_kinds),
            "auth_blocked_count": sum(1 for a in blocked_accounts if account_block_is_auth_like(a)),
            "usage_blocked_count": sum(1 for a in blocked_accounts if account_block_is_usage_like(a)),
            "available_tags": [str(a.get("tag") or a.get("account")) for a in available],
            "blocked": blocked_accounts,
        },
    }


def account_block_is_auth_like(account: dict[str, Any]) -> bool:
    kind = str(account.get("block_kind") or "").lower()
    reason = str(account.get("reason") or "").lower()
    return kind in {"auth", "credit"} or "auth" in reason or "login" in reason or "credit" in reason


def account_block_needs_login(account: dict[str, Any]) -> bool:
    kind = str(account.get("block_kind") or "").lower()
    reason = str(account.get("reason") or "").lower()
    if kind in {"access", "usage"} or "subscription access" in reason or "access disabled" in reason:
        return False
    return kind == "auth" or "auth/login" in reason or "login required" in reason


def account_block_is_usage_like(account: dict[str, Any]) -> bool:
    kind = str(account.get("block_kind") or "").lower()
    reason = str(account.get("reason") or "").lower()
    return bool(account.get("throttled")) or kind == "usage" or "usage limit" in reason


def account_block_is_access_wall(account: dict[str, Any]) -> bool:
    kind = str(account.get("block_kind") or "").lower()
    reason = str(account.get("reason") or "").lower()
    return kind == "access" or "subscription access" in reason or "access disabled" in reason


def account_profile_command(config_dir: str) -> str:
    if not config_dir:
        return ""
    if os.name == "nt":
        escaped = config_dir.replace("'", "''")
        return f"$env:CLAUDE_CONFIG_DIR='{escaped}'; claude"
    import shlex

    return f"CLAUDE_CONFIG_DIR={shlex.quote(config_dir)} claude"


def blocked_account_snapshot(account: dict[str, Any]) -> dict[str, Any]:
    config_dir = str(account.get("config_dir") or "")
    command = account_profile_command(config_dir)
    out = {
        "account": str(account.get("account") or ""),
        "tag": str(account.get("tag") or account.get("account") or "unknown"),
        "config_dir": config_dir,
        "reason": str(account.get("block_reason") or account.get("reason") or ""),
        "block_kind": account.get("block_kind"),
        "reset": account.get("reset"),
        "throttled": bool(account.get("throttled")),
        "active_sessions": int(account.get("active_sessions") or 0),
        "live_sessions": int(account.get("live_sessions") or 0),
        "auth_blocked_sessions": int(account.get("auth_blocked_sessions") or 0),
        "status_source": account.get("status_source"),
        "registry_age_min": account.get("registry_age_min"),
    }
    if command:
        if not account_block_needs_login(out):
            command = ""
    if command:
        out["command"] = command
    return out


def summarize_resume_plan(config: dict[str, Any], *, limit: int = 10) -> dict[str, Any]:
    root = Path(config["root"])
    registry_dir = resolve_path(config.get("registry_dir"), root) or root / "tools" / "_registry"
    path = registry_dir / "resume_plan.json"
    summary: dict[str, Any] = {
        "path": str(path),
        "display_path": display_path(path, root),
        "exists": path.exists(),
        "count": 0,
        "shown": 0,
        "limit": limit,
        "sessions": [],
        "omitted": 0,
    }
    if not path.exists():
        return summary

    doc = read_json(path)
    summary["generated_utc"] = doc.get("generated_utc")
    plan = doc.get("plan")
    if not isinstance(plan, list):
        summary["reason"] = "missing plan list"
        return summary

    sessions: list[dict[str, Any]] = []
    for item in plan[:limit]:
        if not isinstance(item, dict):
            continue
        session = str(item.get("session") or "")
        sessions.append({
            "account": str(item.get("account") or ""),
            "session": session,
            "session_short": session[:8],
            "project": str(item.get("project") or ""),
            "cwd": str(item.get("cwd") or ""),
            "config_dir": str(item.get("config_dir") or ""),
            "resume_cmd": str(item.get("resume_cmd") or ""),
        })

    summary["count"] = len(plan)
    summary["shown"] = len(sessions)
    summary["sessions"] = sessions
    summary["omitted"] = max(0, len(plan) - len(sessions))
    return summary


def refresh_registry(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    script = root / "tools" / "fleet_sessions.py"
    if not script.exists():
        return {"ok": False, "reason": f"missing {display_path(script, root)}"}
    python = str(config.get("python") or sys.executable)
    if not command_exists(python):
        return {"ok": False, "reason": f"python not found: {python}"}
    args = [
        python,
        str(script),
        "registry",
        "--window",
        str(config.get("session_window_h", 6)),
    ]
    try:
        proc = run(
            args,
            root,
            env=env_for_tools(config),
            timeout=int(config.get("refresh_timeout_s", 90)),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "fleet_sessions.py registry timed out"}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip()[-2000:],
        "stderr": proc.stderr.strip()[-2000:],
    }


def powershell_exe() -> str | None:
    return shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh")


def platform_is_windows() -> bool:
    return platform.system().lower() == "windows"


def sh_exe() -> str:
    return shutil.which("sh") or "sh"


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def scheduled_task_status(task_name: str) -> dict[str, Any]:
    if not platform_is_windows():
        return {"supported": False, "installed": False, "reason": "not Windows"}
    ps = powershell_exe()
    if not ps:
        return {"supported": False, "installed": False, "reason": "PowerShell not found"}
    q = ps_quote(task_name)
    script = (
        "$ErrorActionPreference='SilentlyContinue';"
        f"$t=Get-ScheduledTask -TaskName {q};"
        f"if($null -eq $t){{[pscustomobject]@{{task_name={q};installed=$false}}|ConvertTo-Json -Compress;exit 0}};"
        f"$i=Get-ScheduledTaskInfo -TaskName {q};"
        "[pscustomobject]@{"
        f"task_name={q};installed=$true;"
        "state=[string]$t.State;"
        "last_run=[string]$i.LastRunTime;"
        "last_result=[int64]$i.LastTaskResult;"
        "next_run=[string]$i.NextRunTime;"
        "arguments=[string](($t.Actions|Select-Object -First 1).Arguments)"
        "}|ConvertTo-Json -Compress"
    )
    try:
        proc = run([ps, "-NoProfile", "-Command", script], Path.cwd(), timeout=15)
    except subprocess.TimeoutExpired:
        return {
            "supported": True,
            "installed": None,
            "reason": f"scheduled task query timed out for {task_name}",
        }
    except OSError as exc:
        return {
            "supported": True,
            "installed": None,
            "reason": str(exc),
        }
    if proc.returncode != 0:
        return {
            "supported": True,
            "installed": None,
            "reason": (proc.stderr or proc.stdout).strip(),
        }
    return {"supported": True, **read_json_from_text(proc.stdout)}


def scheduler_result_needs_action(task: dict[str, Any]) -> bool:
    result = task.get("last_result")
    if result in (None, 0):
        return False
    try:
        result_int = scheduler_result_int(result)
    except (TypeError, ValueError):
        return True
    state = str(task.get("state") or "").lower()
    if result_int in {WINDOWS_TASK_RUNNING_RESULT, WINDOWS_TASK_REQUEST_REFUSED_RESULT} and state == "running":
        return False
    return True


def scheduler_state_needs_action(task: dict[str, Any]) -> bool:
    if task.get("supported") is not True or task.get("installed") is not True:
        return False
    return str(task.get("state") or "").strip().lower() in SCHEDULED_TASK_ACTION_STATES


def scheduled_task_needs_action(task: dict[str, Any]) -> bool:
    return scheduler_state_needs_action(task) or scheduler_result_needs_action(task)


def scheduler_result_int(result: Any) -> int:
    value = int(result)
    if value < 0:
        value += 2**32
    return value


def scheduler_result_text(task: dict[str, Any]) -> str:
    result = task.get("last_result")
    if result is None:
        return "unknown"
    try:
        result_int = scheduler_result_int(result)
    except (TypeError, ValueError):
        return str(result)
    hex_code = f"0x{result_int & 0xFFFFFFFF:08X}"
    label = WINDOWS_TASK_RESULT_LABELS.get(result_int)
    if label:
        return f"{result_int} ({hex_code}; {label})"
    return f"{result_int} ({hex_code})"


def scheduled_task_snapshot(task: dict[str, Any], *, accepted_results: set[int] | None = None) -> dict[str, Any]:
    accepted_results = accepted_results or set()
    result_accepted = False
    if task.get("last_result") is not None:
        try:
            result_accepted = scheduler_result_int(task.get("last_result")) in accepted_results
        except (TypeError, ValueError):
            result_accepted = False
    state_action = scheduler_state_needs_action(task)
    result_action = scheduler_result_needs_action(task) and not result_accepted
    out = {
        "supported": task.get("supported"),
        "installed": task.get("installed"),
        "state": task.get("state"),
        "last_result": task.get("last_result"),
        "reason": task.get("reason"),
        "task_name": task.get("task_name"),
        "needs_action": state_action or result_action,
    }
    if task.get("last_result") is not None:
        out["last_result_text"] = scheduler_result_text(task)
        out["last_result_accepted"] = result_accepted
    return out


def scheduled_task_summary_needs_action(task: dict[str, Any]) -> bool:
    if not task or task.get("supported") is False:
        return False
    if "needs_action" in task:
        if task.get("needs_action"):
            return True
    elif scheduled_task_needs_action(task):
        return True
    if task.get("supported") is True and task.get("installed") is not True:
        return True
    return task.get("installed") is False


def scheduled_task_label(task: dict[str, Any]) -> str:
    if task.get("supported") is False:
        return "unsupported"
    if task.get("installed") is True:
        state = str(task.get("state") or "installed")
        if scheduler_state_needs_action(task):
            return f"{state}/disabled"
        if task.get("last_result_accepted"):
            result = task.get("last_result_text") or task.get("last_result") or "unknown"
            return f"{state}/result={result}"
        if task.get("needs_action") or scheduler_result_needs_action(task):
            result = task.get("last_result_text") or task.get("last_result") or "unknown"
            return f"{state}/result={result}"
        return state
    if task.get("installed") is False:
        return "missing"
    return "unknown"


def scheduled_task_has_data(task: dict[str, Any]) -> bool:
    return any(task.get(key) is not None for key in ("supported", "installed", "state", "last_result", "reason"))


def task_is_installed(task: dict[str, Any]) -> bool:
    return task.get("supported") and task.get("installed") is True


def supervisor_payload_verdict(supervisor: dict[str, Any]) -> str:
    payload = supervisor.get("payload") or {}
    return str(payload.get("verdict", "")).upper()


def supervisor_payload_diagnosis_health(supervisor: dict[str, Any]) -> str:
    payload = supervisor.get("payload") or {}
    diagnose = payload.get("diagnose") or {}
    return str(diagnose.get("health", "")).upper()


def supervisor_payload_run_health(supervisor: dict[str, Any]) -> str:
    payload = supervisor.get("payload") or {}
    for section in (payload.get("last_decide") or {}, payload.get("ships") or {}):
        run_health = str(section.get("run_health") or "").strip().upper()
        if run_health:
            return run_health
    return ""


def supervisor_run_health_is_actionable(supervisor: dict[str, Any]) -> bool:
    run_health = supervisor_payload_run_health(supervisor)
    return bool(run_health and run_health not in {"OK", "HEALTHY", "RUNNING", "GREEN"})


def supervisor_diagnosis_is_actionable(supervisor: dict[str, Any]) -> bool:
    verdict = supervisor_payload_verdict(supervisor)
    if verdict in SUPERVISOR_ACTION_VERDICTS:
        return True
    if supervisor_run_health_is_actionable(supervisor):
        return True
    if verdict in SUPERVISOR_NON_ACTION_VERDICTS:
        return False
    health = supervisor_payload_diagnosis_health(supervisor)
    return bool(health and health not in {"HEALTHY", "OK"})


def supervisor_log_hint(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    run_id = str(payload.get("run") or "").strip()
    if not run_id:
        return {}
    job_dir_raw = config.get("job_dir") or ""
    if not job_dir_raw:
        return {"run": run_id, "run_dir": f"output/{run_id}"}
    run_dir = Path(str(job_dir_raw)) / "output" / run_id
    hint = {"run": run_id, "run_dir": str(run_dir)}
    try:
        logs = [p for p in run_dir.glob("worker-*/run.log") if p.is_file()]
    except OSError:
        logs = []
    if logs:
        newest = max(logs, key=lambda p: p.stat().st_mtime)
        hint["run_log"] = str(newest)
    return hint


def supervisor_health_summary(
    supervisor: dict[str, Any],
    config: dict[str, Any],
    *,
    watchdog_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = supervisor.get("payload") or {}
    diagnose = payload.get("diagnose") or {}
    process = payload.get("process") or {}
    last_decide = payload.get("last_decide") or {}
    summary: dict[str, Any] = {
        "available": bool(supervisor.get("available")),
        "verdict": supervisor_payload_verdict(supervisor),
        "alive": process.get("alive"),
        "run_health": supervisor_payload_run_health(supervisor),
        "diagnose": supervisor_payload_diagnosis_health(supervisor),
        "run": payload.get("run"),
        "stop_reason": last_decide.get("stop_reason"),
        "primary_cause": diagnose.get("primary_cause"),
        "primary_action": diagnose.get("primary_action"),
        "next_action": payload.get("next_action"),
        "why": payload.get("why"),
    }
    summary.update(supervisor_log_hint(payload, config))
    if watchdog_task:
        result = watchdog_task.get("last_result")
        if result not in (None, 0):
            summary["watchdog_last_result"] = result
            summary["watchdog_last_result_text"] = scheduler_result_text(watchdog_task)
            try:
                summary["watchdog_last_result_accepted"] = (
                    scheduler_result_int(result) == SUPERVISOR_WATCHDOG_ACTION_RESULT
                )
            except (TypeError, ValueError):
                summary["watchdog_last_result_accepted"] = False
    return summary


def supervisor_health_needs_action(summary: dict[str, Any]) -> bool:
    if not summary.get("available"):
        return True
    if summary.get("alive") is False:
        return True
    verdict = str(summary.get("verdict") or "").upper()
    if verdict in SUPERVISOR_ACTION_VERDICTS:
        return True
    run_health = str(summary.get("run_health") or "").upper()
    if run_health and run_health not in {"OK", "HEALTHY", "RUNNING", "GREEN"}:
        return True
    diagnose = str(summary.get("diagnose") or "").upper()
    if verdict not in SUPERVISOR_NON_ACTION_VERDICTS and diagnose and diagnose not in {"OK", "HEALTHY"}:
        return True
    if summary.get("watchdog_last_result") not in (None, 0):
        return True
    return False


def supervisor_health_action_text(summary: dict[str, Any]) -> str:
    verdict = str(summary.get("verdict") or "UNKNOWN").upper()
    run_health = str(summary.get("run_health") or "").upper()
    diagnose = str(summary.get("diagnose") or "").upper()
    run_id = summary.get("run") or "unknown-run"
    bits = [f"Supervisor run {run_id}"]
    if run_health:
        bits.append(f"health={run_health}")
    bits.append(f"verdict={verdict}")
    if diagnose:
        bits.append(f"diagnosis={diagnose}")
    if summary.get("stop_reason"):
        bits.append(f"stop={summary['stop_reason']}")
    if summary.get("primary_cause"):
        bits.append(f"cause={summary['primary_cause']}")
    if summary.get("watchdog_last_result") not in (None, 0):
        result = summary.get("watchdog_last_result_text") or summary.get("watchdog_last_result")
        accepted = "accepted watchdog status, not setup failure" if summary.get("watchdog_last_result_accepted") else "inspect scheduler result"
        bits.append(f"watchdog={result}; {accepted}")
    inspect = summary.get("run_log") or summary.get("run_dir")
    if inspect:
        bits.append(f"inspect `{inspect}`")
    primary = summary.get("primary_action") or summary.get("next_action") or summary.get("why")
    if primary:
        bits.append(str(primary))
    command = "inspect `python tools/fleet_control_pane.py supervisor --json`"
    if summary.get("alive") is False:
        command += " or restart with `python tools/fleet_control_pane.py supervisor --restart --apply`"
    elif verdict in SUPERVISOR_ACTION_VERDICTS:
        command += " or dry-run restart with `python tools/fleet_control_pane.py supervisor --restart`"
    return "; ".join(bits) + f"; {command}."


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


def control_tick_register_script_value(config: dict[str, Any]) -> Any:
    spec = config.get("control_tick") or {}
    if platform_is_windows():
        return spec.get("register_script")
    return spec.get("posix_register_script") or spec.get("register_script")


def control_tick_runner_path(config: dict[str, Any]) -> Path:
    suffix = ".cmd" if platform_is_windows() else ".sh"
    return Path(str(config["registry_dir"])) / f"control_pane_tick{suffix}"


def posix_tick_status(register_script: Path | None, config: dict[str, Any]) -> dict[str, Any]:
    if platform_is_windows():
        return {"supported": False, "installed": False, "reason": "Windows host"}
    if not register_script or not register_script.exists():
        return {"supported": True, "installed": False, "reason": "register script missing"}
    cmd = [sh_exe(), str(register_script), "status", "--json"]
    try:
        proc = run(cmd, Path(config["root"]), env=env_for_tools(config), timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"supported": False, "installed": False, "reason": str(exc), "cmd": cmd}
    doc = read_json_from_text(proc.stdout)
    if doc:
        return {"supported": True, **doc, "returncode": proc.returncode, "cmd": cmd}
    return {
        "supported": True,
        "installed": None,
        "reason": (proc.stderr or proc.stdout).strip()[-1000:],
        "returncode": proc.returncode,
        "cmd": cmd,
    }


def collect_watchdogs(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    out: dict[str, Any] = {}
    for name, spec in sorted((config.get("watchdogs") or {}).items()):
        script = resolve_path(spec.get("script"), root)
        register_script = resolve_path(spec.get("register_script"), root)
        task_name = str(spec.get("task_name") or name)
        out[name] = {
            "task_name": task_name,
            "script": display_path(script, root),
            "script_exists": bool(script and script.exists()),
            "register_script": display_path(register_script, root),
            "register_script_exists": bool(register_script and register_script.exists()),
            "interval_min": spec.get("interval_min"),
            "task": scheduled_task_status(task_name),
        }
    return out


def collect_control_tick(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    spec = config.get("control_tick") or {}
    register_script = resolve_path(control_tick_register_script_value(config), root)
    task_name = str(spec.get("task_name") or "FleetControlPaneTick")
    task = scheduled_task_status(task_name) if platform_is_windows() else posix_tick_status(register_script, config)
    runner = resolve_path(task.get("runner"), root) if task.get("runner") else control_tick_runner_path(config)
    return {
        "task_name": task_name,
        "register_script": display_path(register_script, root),
        "register_script_exists": bool(register_script and register_script.exists()),
        "runner": display_path(runner, root),
        "runner_exists": bool(runner and runner.exists()),
        "runner_required": bool(task.get("runner") or task.get("arguments")),
        "interval_min": spec.get("interval_min"),
        "task": task,
    }


def control_tick_register_script_path(config: dict[str, Any], tick: dict[str, Any] | None = None) -> Path | None:
    root = Path(config["root"])
    if tick and tick.get("register_script"):
        return resolve_path(str(tick.get("register_script")), root)
    return resolve_path(control_tick_register_script_value(config), root)


def control_tick_runner_missing(tick: dict[str, Any], tick_task: dict[str, Any]) -> bool:
    if not task_is_installed(tick_task) or tick.get("runner_exists") is not False:
        return False
    return bool(tick.get("runner_required", True))


def collect_supervisor(config: dict[str, Any]) -> dict[str, Any]:
    raw_cmd = list(config.get("supervisor_status_cmd") or [])
    uses_job_dir = any("{job_dir}" in str(part) for part in raw_cmd)
    cmd = expand_cmd(raw_cmd, config)
    if not cmd:
        return {"available": False, "reason": "no supervisor_status_cmd configured"}
    if not command_exists(cmd[0]):
        return {"available": False, "reason": f"command not found: {cmd[0]}", "cmd": cmd}
    if uses_job_dir and not str(config.get("job_dir", "")):
        return {"available": False, "reason": "job_dir is not configured", "cmd": cmd}
    try:
        proc = run(cmd, Path(config["root"]), timeout=20)
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": "supervisor status timed out", "cmd": cmd}
    doc = read_json_from_text(proc.stdout)
    if doc:
        return {"available": True, "cmd": cmd, "returncode": proc.returncode, "payload": doc}
    if proc.returncode != 0:
        return {
            "available": False,
            "returncode": proc.returncode,
            "reason": (proc.stderr or proc.stdout).strip()[-1000:],
            "cmd": cmd,
        }
    return {"available": False, "reason": "supervisor status was not JSON", "cmd": cmd}


def loop_cwd(spec: dict[str, Any], config: dict[str, Any]) -> Path:
    raw = spec.get("cwd") or config["root"]
    return resolve_path(expand_value(raw, config), Path(config["root"])) or Path(config["root"])


def loop_expected_returncodes(spec: dict[str, Any]) -> set[int]:
    raw = spec.get("ok_returncodes", [0])
    if isinstance(raw, int):
        return {raw}
    out: set[int] = set()
    if isinstance(raw, list):
        for item in raw:
            try:
                out.add(int(item))
            except (TypeError, ValueError):
                continue
    return out or {0}


def classify_loop_status(proc: subprocess.CompletedProcess[str], spec: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    payload = read_json_from_text(proc.stdout)
    ok_codes = loop_expected_returncodes(spec)
    if proc.returncode not in ok_codes:
        detail = (proc.stderr or proc.stdout).strip()[-1000:]
        return "ACTION", payload, detail or f"returncode {proc.returncode}"

    if payload:
        ok_value = payload.get("ok")
        if isinstance(ok_value, bool):
            return ("OK" if ok_value else "ACTION"), payload, str(payload.get("reason") or payload.get("detail") or "")
        for key in ("verdict", "status", "state", "health"):
            value = payload.get(key)
            if value is None:
                continue
            folded = str(value).upper()
            if folded in HEALTHY_LOOP_VALUES:
                return "OK", payload, f"{key}={value}"
            if folded in ACTION_LOOP_VALUES:
                return "ACTION", payload, f"{key}={value}"
            return "UNKNOWN", payload, f"{key}={value}"
        return "UNKNOWN", payload, "JSON did not include ok/verdict/status/state/health"

    if (proc.stdout or "").strip():
        return "UNKNOWN", payload, proc.stdout.strip()[-1000:]
    return "OK", payload, ""


def collect_loop(name: str, spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if spec.get("enabled") is False:
        return {"name": name, "enabled": False, "state": "SKIPPED", "reason": "disabled"}
    cmd = expand_cmd(list(spec.get("status_cmd") or []), config)
    base: dict[str, Any] = {
        "name": name,
        "enabled": True,
        "cmd": cmd,
        "auto_recover": bool(spec.get("auto_recover")),
        "has_recover_cmd": bool(spec.get("recover_cmd")),
        "action": spec.get("action") or "",
    }
    if not cmd:
        return {**base, "state": "UNAVAILABLE", "reason": "no status_cmd configured"}
    if not command_exists(cmd[0]):
        return {**base, "state": "UNAVAILABLE", "reason": f"command not found: {cmd[0]}"}
    timeout = int(spec.get("timeout_s") or 20)
    try:
        proc = run(cmd, loop_cwd(spec, config), env=env_for_tools(config), timeout=timeout)
    except subprocess.TimeoutExpired:
        return {**base, "state": "TIMEOUT", "reason": f"status command timed out after {timeout}s"}
    except OSError as exc:
        return {**base, "state": "UNAVAILABLE", "reason": str(exc)}
    state, payload, detail = classify_loop_status(proc, spec)
    return {
        **base,
        "state": state,
        "returncode": proc.returncode,
        "payload": payload,
        "detail": detail,
        "stdout": proc.stdout.strip()[-1000:],
        "stderr": proc.stderr.strip()[-1000:],
    }


def collect_loops(config: dict[str, Any], *, skip_names: set[str] | None = None) -> dict[str, Any]:
    skip_names = skip_names or set()
    loops = config.get("loops") or {}
    loop_specs = [
        (str(name), spec)
        for name, spec in sorted(loops.items())
        if isinstance(spec, dict)
    ]
    enabled_specs = [
        (name, spec)
        for name, spec in loop_specs
        if spec.get("enabled") is not False and name not in skip_names
    ]
    checks = [
        collect_loop(name, spec, config)
        for name, spec in enabled_specs
    ]
    counts = collections.Counter(check.get("state", "UNKNOWN") for check in checks)
    return {
        "count": len(checks),
        "configured": len(loop_specs),
        "enabled": len(enabled_specs),
        "disabled": len(loop_specs) - len(enabled_specs),
        "states": dict(counts),
        "checks": checks,
        "commands": [
            loop_scaffold_template_command(config),
            loop_add_template_command(config),
        ] if not enabled_specs else [],
    }


def loop_check_needs_action(check: dict[str, Any]) -> bool:
    return check.get("state") not in {"OK", "SKIPPED"}


def loop_check_snapshot(check: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": check.get("name"),
        "state": check.get("state", "UNKNOWN"),
        "detail": check.get("detail") or check.get("reason") or "",
        "action": check.get("action") or "",
        "auto_recover": bool(check.get("auto_recover")),
        "has_recover_cmd": bool(check.get("has_recover_cmd")),
        "returncode": check.get("returncode"),
    }


def recovery_invokes_control_bootstrap(recover_cmd: list[str]) -> bool:
    normalized = [Path(str(part)).name.lower() for part in recover_cmd]
    return "fleet_control_pane.py" in normalized and "bootstrap" in normalized


def loop_recovery_action(
    name: str,
    spec: dict[str, Any],
    check: dict[str, Any],
    config: dict[str, Any],
    *,
    dry_run: bool,
    force: bool = False,
    include_not_needed: bool = False,
) -> dict[str, Any] | None:
    state = str(check.get("state") or "UNKNOWN")
    recover_cmd = expand_cmd(list((spec or {}).get("recover_cmd") or []), config)
    action: dict[str, Any] = {
        "name": f"loop:{name}",
        "dry_run": dry_run,
        "command": display_command(recover_cmd),
        "loop_state": state,
    }
    if state in {"OK", "SKIPPED"}:
        if include_not_needed:
            return {**action, "ok": True, "skipped": True, "reason": f"loop state {state}; recovery not needed"}
        return None
    if not recover_cmd:
        if include_not_needed or force:
            return {**action, "ok": False, "skipped": True, "reason": "recover_cmd not configured"}
        return None
    if not force and not bool((spec or {}).get("auto_recover")):
        return None
    if not force and recovery_invokes_control_bootstrap(recover_cmd):
        return {
            **action,
            "ok": True,
            "skipped": True,
            "reason": "control bootstrap recovery is skipped inside the recurring tick",
        }
    if dry_run:
        return {**action, "ok": True, "skipped": True, "reason": "dry run"}
    if not command_exists(recover_cmd[0]):
        return {**action, "ok": False, "skipped": True, "reason": f"command not found: {recover_cmd[0]}"}
    timeout = int((spec or {}).get("recover_timeout_s") or (spec or {}).get("timeout_s") or 90)
    try:
        proc = run(recover_cmd, loop_cwd(spec or {}, config), env=env_for_tools(config), timeout=timeout)
    except subprocess.TimeoutExpired:
        return {**action, "ok": False, "skipped": False, "reason": f"recover command timed out after {timeout}s"}
    except OSError as exc:
        return {**action, "ok": False, "skipped": False, "reason": str(exc)}
    return {
        **action,
        "ok": proc.returncode == 0,
        "skipped": False,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip()[-2000:],
        "stderr": proc.stderr.strip()[-2000:],
    }


def loop_check_plan(
    config: dict[str, Any],
    name: str,
    *,
    recover: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    loop_name = str(name or "").strip()
    loops = config.get("loops") or {}
    available = sorted(
        str(candidate)
        for candidate, spec in loops.items()
        if isinstance(spec, dict)
    )
    spec = loops.get(loop_name) if isinstance(loops.get(loop_name), dict) else None
    if not loop_name or spec is None:
        return {
            "schema": LOOP_CHECK_SCHEMA,
            "generated_utc": iso_now(),
            "ok": False,
            "verdict": "UNKNOWN",
            "needs_action": True,
            "loop": loop_name,
            "recover": recover,
            "apply": apply,
            "reason": "unknown loop" if loop_name else "loop name is required",
            "available": available,
            "check": {},
            "recovery": None,
        }

    check = collect_loop(loop_name, spec, config)
    needs_action = loop_check_needs_action(check)
    recovery = (
        loop_recovery_action(
            loop_name,
            spec,
            check,
            config,
            dry_run=not apply,
            force=True,
            include_not_needed=True,
        )
        if recover or apply
        else None
    )
    fatal_status = check.get("state") in {"UNAVAILABLE", "TIMEOUT"}
    ok = bool(not fatal_status and (recovery is None or recovery.get("ok")))
    return {
        "schema": LOOP_CHECK_SCHEMA,
        "generated_utc": iso_now(),
        "ok": ok,
        "verdict": "ACTION" if needs_action else "OK",
        "needs_action": needs_action,
        "loop": loop_name,
        "recover": recover,
        "apply": apply,
        "check": check,
        "recovery": recovery,
    }


# The aggregate-audit loop must not audit itself when run from the tick, or a
# single broken loop would be reported twice (once directly, once via the audit
# loop's own non-zero exit). loop_audit skips this name when enumerating.
LOOP_AUDIT_SELF_NAME = "gardening-loops-audit"
LOOP_BROKEN_STATES = {"UNAVAILABLE", "TIMEOUT"}


def loop_audit_detail(check: dict[str, Any]) -> str:
    """One clean line summarizing a loop check for the audit row.

    A loop that surfaces ACTION by exiting non-zero gets its raw stdout tail in
    ``detail`` (classify_loop_status fallback), which is a multi-line JSON blob.
    Prefer a real summary field from the parsed payload; only then fall back to
    the first non-empty line of the raw detail, clamped, so neither the JSON nor
    the text rendering carries an embedded document.
    """
    payload = check.get("payload")
    if isinstance(payload, dict):
        for key in ("reason", "summary", "verdict", "why", "headline", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:200]
    raw = str(check.get("detail") or check.get("reason") or "")
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""


def loop_audit(config: dict[str, Any], *, names: list[str] | None = None) -> dict[str, Any]:
    """Run every enabled loop once and bucket each as healthy / action / broken.

    The audit keeps two judgments separate, exactly as loop_check_plan does:
      - *broken* (state UNAVAILABLE/TIMEOUT) means the loop's status command could
        not run — the loop itself is the failure.
      - *action* means the loop ran fine and is surfacing a real condition it was
        built to catch (e.g. a low closure_rate). That is the loop WORKING, not
        failing, so it must not fail the audit by default.
      - *healthy* (state OK/SKIPPED) means nothing to do.

    Top-level ``ok`` is true iff no loop is broken; a caller wanting a hard gate on
    surfaced conditions can additionally key off ``counts['action']``.
    """
    loops = config.get("loops") or {}
    requested = {str(n).strip() for n in (names or []) if str(n).strip()}
    selected: list[tuple[str, dict[str, Any]]] = []
    for name, spec in sorted(loops.items()):
        if not isinstance(spec, dict) or spec.get("enabled") is False:
            continue
        if name == LOOP_AUDIT_SELF_NAME:
            continue
        if requested and name not in requested:
            continue
        selected.append((str(name), spec))

    results: list[dict[str, Any]] = []
    counts = {"healthy": 0, "action": 0, "broken": 0}
    for name, spec in selected:
        check = collect_loop(name, spec, config)
        state = str(check.get("state") or "UNKNOWN")
        if state in LOOP_BROKEN_STATES:
            bucket = "broken"
        elif loop_check_needs_action(check):
            bucket = "action"
        else:
            bucket = "healthy"
        counts[bucket] += 1
        results.append({
            "name": name,
            "bucket": bucket,
            "state": state,
            "detail": loop_audit_detail(check),
            "action_hint": spec.get("action") or "",
            "returncode": check.get("returncode"),
        })

    missing = sorted(requested - {name for name, _ in selected}) if requested else []
    counts["total"] = len(results)
    return {
        "schema": LOOP_AUDIT_SCHEMA,
        "generated_utc": iso_now(),
        "ok": counts["broken"] == 0,
        "counts": counts,
        "loops": results,
        "requested": sorted(requested) if requested else [],
        "missing": missing,
    }


def invoke_loop_recoveries(
    config: dict[str, Any],
    loops_doc: dict[str, Any],
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    specs = config.get("loops") or {}
    actions: list[dict[str, Any]] = []
    for check in loops_doc.get("checks") or []:
        name = str(check.get("name") or "")
        spec = specs.get(name) if isinstance(specs.get(name), dict) else {}
        action = loop_recovery_action(name, spec, check, config, dry_run=dry_run)
        if action:
            actions.append(action)
    return actions


def registry_path(config: dict[str, Any]) -> Path:
    return Path(config["registry_dir"]) / "sessions.json"


def collect_status(
    config: dict[str, Any],
    refresh: bool = False,
    *,
    skip_loop_names: set[str] | None = None,
) -> dict[str, Any]:
    refresh_result = refresh_registry(config) if refresh else None
    registry = read_json(registry_path(config))
    control_tick = collect_control_tick(config)
    status: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": iso_now(),
        "app_version": fleet_version.app_version(Path(config["root"])),
        "machine": {
            "id": machine_id(config),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "config": {
            "root": config["root"],
            "local_config": config["_paths"]["local"],
            "local_config_exists": config.get("_local_exists", False),
            "registry_dir": config.get("registry_dir"),
            "machine_dir": config.get("machine_dir"),
            "user_home": config.get("user_home"),
            "job_dir": config.get("job_dir"),
            "target": config.get("target"),
            "session_window_h": config.get("session_window_h"),
        },
        "refresh": refresh_result,
        "git": collect_git(config),
        "registry": summarize_registry(registry),
        "control_tick": control_tick,
        "watchdogs": collect_watchdogs(config),
        "loops": collect_loops(config, skip_names=skip_loop_names),
        "supervisor": collect_supervisor(config),
        "setup_plan": setup_plan(config, tick=control_tick),
    }
    status["actions"] = recommended_actions(status, config)
    status["verdict"] = "ACTION" if status["actions"] else "OK"
    return status


def recommended_actions(status: dict[str, Any], config: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    root = Path(config["root"])
    if not status["config"].get("local_config_exists"):
        actions.append("Run `python tools/fleet_control_pane.py init` once on this machine.")
    reg = status.get("registry", {})
    if not reg.get("exists"):
        actions.append("Run `python tools/fleet_control_pane.py status --refresh --write` to create the session registry.")
    elif reg.get("age_min") is not None and reg["age_min"] > 15:
        actions.append("Refresh stale session state with `python tools/fleet_control_pane.py status --refresh --write`.")
    account_action = blocked_account_recommended_action(reg)
    if account_action:
        actions.append(account_action)
    if reg.get("auto_resume", 0):
        actions.append(f"{reg['auto_resume']} autonomous session(s) are eligible for auto-resume; inspect `python tools/fleet_control_pane.py recover` or apply with `python tools/fleet_control_pane.py recover --apply`.")

    tick = status.get("control_tick") or {}
    tick_task = tick.get("task") or {}
    pane_tick_installed = task_is_installed(tick_task)
    if tick_task.get("supported") and tick_task.get("installed") is False:
        script = tick.get("register_script")
        if script:
            actions.append(f"Install the control-pane tick with `{display_command(register_command(root / script, 'control_tick', config))}`.")
    elif tick_task.get("supported") and tick_task.get("installed") and scheduler_state_needs_action(tick_task):
        actions.append(
            "Control-pane tick task is "
            f"{tick_task.get('state')}; repair it with `python tools/fleet_control_pane.py bootstrap --apply`."
        )
    elif tick_task.get("supported") and tick_task.get("installed") and scheduler_result_needs_action(tick_task):
        actions.append(
            "Control-pane tick last scheduler result was "
            f"{scheduler_result_text(tick_task)}; check it with "
            "`python tools/fleet_control_pane.py tick --dry-run --no-write` and `python tools/fleet_control_pane.py doctor`."
        )

    for name, wd in (status.get("watchdogs") or {}).items():
        if not wd.get("script_exists"):
            actions.append(f"Missing watchdog script for {name}: {wd.get('script')}")
        task = wd.get("task") or {}
        if task.get("supported") and task.get("installed") is False:
            if not pane_tick_installed:
                reg_script = wd.get("register_script")
                if reg_script:
                    actions.append(f"Install {name} watchdog with `{display_command(register_command(root / reg_script, name, config))}`.")
        elif task.get("supported") and task.get("installed") and scheduled_task_needs_action(task):
            if not pane_tick_installed:
                if scheduler_state_needs_action(task):
                    actions.append(f"{name} watchdog task is {task.get('state')}; inspect its registered command.")
                else:
                    actions.append(f"{name} watchdog last scheduler result was {scheduler_result_text(task)}; inspect its registered command.")

    loops_doc = status.get("loops") or {}
    if loops_doc and "enabled" in loops_doc and int(loops_doc.get("enabled") or 0) == 0:
        commands = loops_doc.get("commands") or [loop_scaffold_template_command(config)]
        actions.append(
            "No enabled loops configured; start a shared loop with "
            f"`{commands[0]}` or inspect `python tools/fleet_control_pane.py loop-list`."
        )

    for check in loops_doc.get("checks") or []:
        state = str(check.get("state") or "UNKNOWN")
        if state in {"OK", "SKIPPED"}:
            continue
        name = check.get("name")
        action = check.get("action") or check.get("detail") or check.get("reason") or "inspect loop status"
        spec = (config.get("loops") or {}).get(name) if name is not None else None
        recover_cmd = expand_cmd(list((spec or {}).get("recover_cmd") or []), config) if isinstance(spec, dict) else []
        if recovery_invokes_control_bootstrap(recover_cmd):
            actions.append(f"Loop {name} is {state}; run `python tools/fleet_control_pane.py bootstrap --apply`. {action}")
        elif check.get("auto_recover") and check.get("has_recover_cmd"):
            actions.append(f"Loop {name} is {state}; it will be recovered by the next live control tick. {action}")
        else:
            actions.append(f"Loop {name} is {state}: {action}.")

    supervisor = status.get("supervisor") or {}
    if not supervisor.get("available"):
        reason = supervisor.get("reason", "unavailable")
        actions.append(f"Supervisor status unavailable: {reason}.")
    else:
        supervisor_task = ((status.get("watchdogs") or {}).get("supervisor") or {}).get("task") or {}
        summary = supervisor_health_summary(supervisor, config, watchdog_task=supervisor_task)
        if supervisor_health_needs_action(summary):
            actions.append(supervisor_health_action_text(summary))

    git_status = status.get("git") or {}
    safe_ff = git_status.get("safe_ff") or {}
    counts = git_status.get("counts") or {}
    unmerged_count = int(counts.get("unmerged") or 0)
    unmerged_paths = [str(path) for path in git_status.get("unmerged_paths") or []]
    merge_in_progress = bool(git_status.get("merge_in_progress"))
    if unmerged_count:
        shown = ", ".join(unmerged_paths[:4])
        omitted = max(0, len(unmerged_paths) - 4)
        suffix = f" (+{omitted} more)" if omitted else ""
        detail = f": {shown}{suffix}" if shown else ""
        actions.append(f"Resolve {unmerged_count} unmerged path(s) before commit/publish{detail}.")
    elif merge_in_progress:
        actions.append("Finish or abort the in-progress merge before commit/publish.")
    if safe_ff.get("state") == "ahead":
        if unmerged_count or merge_in_progress or int(git_status.get("dirty_total") or 0) > 0:
            actions.append("Branch is ahead of remote, but publish is blocked until the worktree is clean.")
        else:
            actions.append("Branch is ahead of remote; inspect `python tools/fleet_control_pane.py publish --json` before pushing.")
    elif safe_ff.get("state") == "diverged" or safe_ff.get("ok") is False:
        actions.append("Do not use plain `git pull`; inspect `python tools/fleet_control_pane.py sync --fetch --json`.")
    worktrees = git_status.get("worktrees") or {}
    if worktrees.get("available"):
        commands = worktrees.get("commands") or {}
        prune_count = int(worktrees.get("prune_count") or 0)
        if prune_count > 0:
            actions.append(
                f"{prune_count} safe extra worktree(s) can be pruned; "
                f"run `{commands.get('prune') or display_command(worktree_doctor_command(config, prune=True, fetch=True))}`."
            )
        if int(worktrees.get("blocked_count") or 0) > 0 or worktrees.get("primary_offtrack"):
            actions.append(
                "Worktree doctor is not converged; inspect blocked/off-track worktree state with "
                f"`{commands.get('inspect') or display_command(worktree_doctor_command(config, prune=False, fetch=True))}`."
            )
    if git_status.get("dirty_total", 0) > 0:
        if unmerged_count:
            actions.append(
                f"Worktree has {git_status['dirty_total']} dirty path(s), including merge conflicts; "
                "resolve conflicts before using dirty-plan commit commands."
            )
        else:
            actions.append(f"Worktree has {git_status['dirty_total']} dirty path(s); use the dirty-plan commit commands before broad sync/release.")
    return actions


def supervisor_needs_action(supervisor: dict[str, Any]) -> bool:
    if not supervisor.get("available"):
        return True
    payload = supervisor.get("payload") or {}
    alive = (payload.get("process") or {}).get("alive")
    if alive is False:
        return True
    return supervisor_diagnosis_is_actionable(supervisor)


def register_command(script: Path, name: str, config: dict[str, Any], *, live_resume: bool = False) -> list[str]:
    if name == "control_tick" and not platform_is_windows():
        tick = config.get("control_tick") or {}
        args = [
            sh_exe(),
            str(script),
            "install",
            "--task-name",
            str(tick.get("task_name") or "FleetControlPaneTick"),
            "--python",
            str(config.get("python") or sys.executable),
        ]
        if tick.get("interval_min"):
            args.extend(["--interval-min", str(tick.get("interval_min"))])
        if live_resume:
            args.append("--live-resume")
        return args

    ps = "powershell.exe" if platform_is_windows() else "powershell"
    args = [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    if name == "supervisor":
        args.extend(["-Target", str(config.get("target", 4))])
    elif name == "control_tick":
        args.extend(["-Python", str(config.get("python") or sys.executable)])
        tick = config.get("control_tick") or {}
        if tick.get("interval_min"):
            args.extend(["-IntervalMin", str(tick.get("interval_min"))])
        if live_resume:
            args.append("-LiveResume")
    return args


def watchdog_command(name: str, config: dict[str, Any], live_resume: bool = False) -> list[str] | None:
    if not platform_is_windows():
        return None
    ps = powershell_exe()
    if not ps:
        return None
    spec = (config.get("watchdogs") or {}).get(name) or {}
    script = resolve_path(spec.get("script"), Path(config["root"]))
    if not script:
        return None
    args = [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    if name == "supervisor":
        if config.get("job_dir"):
            args.extend(["-JobDir", str(config["job_dir"])])
        args.extend(["-Target", str(config.get("target", 4))])
        if config.get("watchdog_log_dir"):
            args.extend(["-LogDir", str(config["watchdog_log_dir"])])
    elif name == "resume":
        args.extend(["-FleetDir", str(config["root"])])
        args.extend(["-WindowH", str(config.get("session_window_h", 6))])
        if config.get("registry_dir"):
            args.extend(["-RegistryDir", str(config["registry_dir"])])
        if config.get("claude_exe"):
            args.extend(["-ClaudeExe", str(config["claude_exe"])])
        if config.get("watchdog_log_dir"):
            args.extend(["-LogDir", str(config["watchdog_log_dir"])])
        if live_resume:
            args.append("-Live")
    return args


def invoke_watchdog(name: str, config: dict[str, Any], *, dry_run: bool, live_resume: bool = False) -> dict[str, Any]:
    cmd = watchdog_command(name, config, live_resume=live_resume)
    root = Path(config["root"])
    spec = (config.get("watchdogs") or {}).get(name) or {}
    script = resolve_path(spec.get("script"), root)
    result: dict[str, Any] = {
        "name": name,
        "dry_run": dry_run,
        "script": display_path(script, root),
        "script_exists": bool(script and script.exists()),
        "command": display_command(cmd or []),
    }
    if not platform_is_windows():
        return {**result, "ok": None, "skipped": True, "reason": "PowerShell watchdog scripts are Windows-only"}
    if not cmd:
        return {**result, "ok": False, "skipped": True, "reason": "watchdog command unavailable"}
    if script and not script.exists():
        return {**result, "ok": False, "skipped": True, "reason": "watchdog script missing"}
    if dry_run:
        return {**result, "ok": True, "skipped": True, "reason": "dry run"}
    try:
        # via_files: the watchdog may relaunch a long-lived detached daemon
        # (run_supervise_loop / claude) that inherits our stdout pipe; capturing through
        # a pipe would then deadlock the tick despite `timeout` (see run()).
        proc = run(cmd, root, env=env_for_tools(config), timeout=90, via_files=True)
    except subprocess.TimeoutExpired:
        return {**result, "ok": False, "skipped": False, "reason": "watchdog timed out"}
    return {
        **result,
        "ok": proc.returncode in (0, 10),
        "skipped": False,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip()[-2000:],
        "stderr": proc.stderr.strip()[-2000:],
    }


def terminate_pid(pid: int, root: Path) -> dict[str, Any]:
    if platform_is_windows():
        proc = run(["taskkill", "/PID", str(pid), "/F"], root, timeout=20)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[-2000:],
            "stderr": proc.stderr.strip()[-2000:],
        }
    try:
        os.kill(pid, signal.SIGTERM)
        return {"ok": True}
    except OSError as exc:
        return {"ok": False, "reason": str(exc)}


def supervisor_plan(config: dict[str, Any], *, restart: bool = False, apply: bool = False) -> dict[str, Any]:
    root = Path(config["root"])
    before = collect_supervisor(config)
    actions: list[dict[str, Any]] = []
    payload = before.get("payload") or {}
    proc_info = payload.get("process") or {}
    pid = proc_info.get("pid")
    alive = proc_info.get("alive")
    if restart:
        if pid and alive is not False:
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                pid_int = 0
            command = ["taskkill", "/PID", str(pid_int), "/F"] if platform_is_windows() else ["kill", "-TERM", str(pid_int)]
            action = {
                "name": "terminate-supervisor",
                "pid": pid_int,
                "dry_run": not apply,
                "command": display_command(command),
            }
            if not pid_int:
                actions.append({**action, "ok": False, "skipped": True, "reason": f"invalid supervisor pid: {pid}"})
            elif not apply:
                actions.append({**action, "ok": True, "skipped": True, "reason": "dry run"})
            else:
                result = terminate_pid(pid_int, root)
                actions.append({**action, **result, "skipped": False})
        else:
            actions.append({
                "name": "terminate-supervisor",
                "dry_run": not apply,
                "ok": True,
                "skipped": True,
                "reason": "supervisor is not alive; no terminate step needed",
            })
        actions.append(invoke_watchdog("supervisor", config, dry_run=not apply))
    after = collect_supervisor(config) if restart and apply else before
    ok = before.get("available", False) if not restart else all(action.get("ok") is not False for action in actions)
    return {
        "schema": SUPERVISOR_SCHEMA,
        "generated_utc": iso_now(),
        "restart": restart,
        "apply": apply,
        "ok": ok,
        "needs_action": supervisor_needs_action(after),
        "before": before,
        "after": after,
        "actions": actions,
    }


def control_tick(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    live_resume: bool = False,
    skip_supervisor: bool = False,
    skip_resume: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    mutation_guard = git_mutation_guard(Path(config["root"]))
    git_mutation_blocked = bool(mutation_guard.get("blocked"))
    if git_mutation_blocked:
        actions.append({
            "name": "git-merge-guard",
            "ok": True,
            "skipped": True,
            "reason": (
                "merge/unmerged worktree detected; skipping watchdog and loop recovery actions"
            ),
            "blockers": mutation_guard.get("blockers") or [],
            "worktree": mutation_guard.get("worktree") or {},
        })
    if not git_mutation_blocked and not skip_supervisor:
        actions.append(invoke_watchdog("supervisor", config, dry_run=dry_run))
    if not git_mutation_blocked and not skip_resume:
        actions.append(invoke_watchdog("resume", config, dry_run=dry_run, live_resume=live_resume))

    tick_skip_loops = {"control-pane-doctor"}
    status = collect_status(config, refresh=True, skip_loop_names=tick_skip_loops)
    loop_actions = [] if git_mutation_blocked else invoke_loop_recoveries(config, status.get("loops") or {}, dry_run=dry_run)
    actions.extend(loop_actions)
    if loop_actions and not dry_run and any(not action.get("skipped") for action in loop_actions):
        status = collect_status(config, refresh=False, skip_loop_names=tick_skip_loops)
    written = write_status(status, config) if write else None
    ok = all(a.get("ok") is not False for a in actions)
    return {
        "schema": TICK_SCHEMA,
        "generated_utc": iso_now(),
        "dry_run": dry_run,
        "live_resume": live_resume,
        "ok": ok,
        "mutation_guard": mutation_guard,
        "actions": actions,
        "status": status,
        "written": written,
    }


def recover_plan(
    config: dict[str, Any],
    *,
    apply: bool = False,
    skip_supervisor: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    tick = control_tick(
        config,
        dry_run=not apply,
        live_resume=True,
        skip_supervisor=skip_supervisor,
        skip_resume=False,
        write=write,
    )
    resume_plan = summarize_resume_plan(config)
    return {
        "schema": RECOVER_SCHEMA,
        "generated_utc": iso_now(),
        "apply": apply,
        "ok": tick.get("ok"),
        "resume_plan": resume_plan,
        "tick": tick,
        "actions": tick.get("actions") or [],
        "status": tick.get("status"),
        "written": tick.get("written"),
    }


def display_command(args: list[str]) -> str:
    if not args:
        return ""
    if os.name == "nt":
        return subprocess.list2cmdline(args)
    import shlex

    return shlex.join(args)


def bootstrap(
    config: dict[str, Any],
    *,
    apply: bool = False,
    live_resume: bool = False,
    init_overrides: dict[str, Any] | None = None,
    force_local_config: bool = False,
) -> dict[str, Any]:
    root = Path(config["root"])
    actions: list[dict[str, Any]] = []

    local_path = Path(config["_paths"]["local"])
    mismatches = local_config_mismatches(local_path, init_overrides)
    has_init_overrides = bool(clean_config_overrides(init_overrides))
    if local_path.exists() and force_local_config and apply and has_init_overrides:
        result = init_config(root, force=True, overrides=init_overrides, base_config=config)
        actions.append({
            "id": "local-config",
            "ok": bool(result.get("written")),
            "changed": bool(result.get("written")),
            "skipped": False,
            "detail": result.get("path"),
        })
    elif local_path.exists() and mismatches:
        actions.append({
            "id": "local-config",
            "ok": False,
            "changed": False,
            "skipped": True,
            "detail": "local config already exists with different requested value(s); pass --force-local-config to rewrite",
            "mismatches": mismatches,
            "command": display_command(init_command(config, init_overrides, force=True)),
        })
    elif local_path.exists():
        actions.append({
            "id": "local-config",
            "ok": True,
            "changed": False,
            "skipped": True,
            "detail": str(local_path),
        })
    elif apply:
        result = init_config(root, overrides=init_overrides, base_config=config)
        actions.append({
            "id": "local-config",
            "ok": bool(result.get("written")),
            "changed": bool(result.get("written")),
            "skipped": False,
            "detail": result.get("path"),
        })
    else:
        actions.append({
            "id": "local-config",
            "ok": True,
            "changed": False,
            "skipped": True,
            "detail": f"would write {local_path}",
            "command": display_command(init_command(config, init_overrides)),
        })

    dirs = ensure_runtime_dirs(config, apply=apply)
    missing = [check for check in dirs["checks"] if not check.get("exists")]
    failed_existing = [check for check in dirs["checks"] if not check.get("ok") and check.get("exists")]
    if apply:
        actions.append({
            "id": "runtime-dirs",
            "ok": dirs["ok"],
            "changed": dirs["changed"],
            "skipped": False,
            "detail": "runtime directories ready" if dirs["ok"] else "runtime directory check failed",
            "checks": dirs["checks"],
        })
    elif failed_existing:
        actions.append({
            "id": "runtime-dirs",
            "ok": False,
            "changed": False,
            "skipped": True,
            "detail": "runtime directories are not writable",
            "checks": dirs["checks"],
        })
    elif missing:
        actions.append({
            "id": "runtime-dirs",
            "ok": True,
            "changed": False,
            "skipped": True,
            "detail": "would create runtime directories",
            "checks": dirs["checks"],
        })
    else:
        actions.append({
            "id": "runtime-dirs",
            "ok": True,
            "changed": False,
            "skipped": True,
            "detail": "runtime directories ready",
            "checks": [],
        })

    guard = trunk_guard_status(config)
    if guard.get("supported") or guard.get("installer_exists") or guard.get("hook_exists"):
        if guard.get("installed"):
            actions.append({
                "id": "trunk-guard",
                "ok": True,
                "changed": False,
                "skipped": True,
                "detail": guard.get("detail"),
            })
        elif not guard.get("setup_ok"):
            actions.append({
                "id": "trunk-guard",
                "ok": False,
                "changed": False,
                "skipped": True,
                "detail": guard.get("detail"),
            })
        elif not apply:
            actions.append({
                "id": "trunk-guard",
                "ok": True,
                "changed": False,
                "skipped": True,
                "detail": "would install local trunk guard",
                "command": display_command(list(guard.get("command") or [])),
            })
        else:
            cmd = list(guard.get("command") or [])
            proc = run(cmd, root, env=env_for_tools(config), timeout=30) if cmd else subprocess.CompletedProcess(cmd, 1, "", "installer command missing")
            actions.append({
                "id": "trunk-guard",
                "ok": proc.returncode == 0,
                "changed": proc.returncode == 0,
                "skipped": False,
                "detail": proc.stdout.strip()[-2000:] or proc.stderr.strip()[-2000:],
                "returncode": proc.returncode,
                "command": display_command(cmd),
            })

    tick = collect_control_tick(config)
    tick_task = tick.get("task") or {}
    runner_missing = control_tick_runner_missing(tick, tick_task)
    tick_disabled = scheduler_state_needs_action(tick_task)
    if tick_task.get("supported") and tick_task.get("installed") and not runner_missing and not tick_disabled:
        actions.append({
            "id": "control-tick",
            "ok": True,
            "changed": False,
            "skipped": True,
            "detail": f"{tick.get('task_name')} already installed",
        })
    else:
        register_script = control_tick_register_script_path(config, tick)
        cmd = register_command(register_script, "control_tick", config, live_resume=live_resume) if register_script else []
        if not register_script or not register_script.exists():
            actions.append({
                "id": "control-tick",
                "ok": False,
                "changed": False,
                "skipped": True,
                "detail": "control-pane tick register script not found",
            })
        elif not apply:
            actions.append({
                "id": "control-tick",
                "ok": True,
                "changed": False,
                "skipped": True,
                "detail": (
                    f"would repair missing control-pane tick runner {tick.get('runner')}"
                    if runner_missing
                    else f"would re-enable control-pane tick task {tick.get('task_name')}"
                    if tick_disabled
                    else "would install control-pane tick"
                ),
                "command": display_command(cmd),
            })
        else:
            proc = run(cmd, root, env=env_for_tools(config), timeout=90)
            actions.append({
                "id": "control-tick",
                "ok": proc.returncode == 0,
                "changed": proc.returncode == 0,
                "skipped": False,
                "detail": proc.stdout.strip()[-2000:] or proc.stderr.strip()[-2000:],
                "returncode": proc.returncode,
                "command": display_command(cmd),
            })

    ok = all(action.get("ok") for action in actions)
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "generated_utc": iso_now(),
        "apply": apply,
        "live_resume": live_resume,
        "ok": ok,
        "actions": actions,
    }


def loop_setup_steps(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["root"])
    steps: list[dict[str, Any]] = []
    for name, spec in sorted((config.get("loops") or {}).items()):
        if not isinstance(spec, dict) or spec.get("enabled") is False:
            continue
        loop_id = step_id_slug(name)
        status_cmd = expand_cmd(list(spec.get("status_cmd") or []), config)
        status = command_readiness(status_cmd, root)
        steps.append({
            "id": f"loop-{loop_id}-status",
            "state": "done" if status["ok"] else "blocked",
            "ok": status["ok"],
            "detail": status["detail"],
            "why": "Let the pane check this loop's health without a host-specific watchdog.",
            "command": status_cmd,
        })
        recover_cmd = expand_cmd(list(spec.get("recover_cmd") or []), config)
        if spec.get("auto_recover") and not recover_cmd:
            steps.append({
                "id": f"loop-{loop_id}-recover",
                "state": "blocked",
                "ok": False,
                "detail": "auto_recover is true but recover_cmd is not configured",
                "why": "Give the recurring control tick an explicit recovery command for this loop.",
                "command": [],
            })
        elif recover_cmd:
            recovery = command_readiness(recover_cmd, root)
            steps.append({
                "id": f"loop-{loop_id}-recover",
                "state": "done" if recovery["ok"] else "blocked",
                "ok": recovery["ok"],
                "detail": recovery["detail"],
                "why": "Let the recurring control tick recover this loop without a separate machine setup step.",
                "command": recover_cmd,
            })
    return steps


def setup_plan(config: dict[str, Any], *, tick: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(config["root"])
    python = str(config.get("python") or sys.executable)
    local_path = Path((config.get("_paths") or {}).get("local") or config_paths(root)["local"])
    runtime_dirs = ensure_runtime_dirs(config, apply=False)
    missing_dirs = [check["name"] for check in runtime_dirs["checks"] if not check.get("exists")]
    failed_dirs = [check for check in runtime_dirs["checks"] if not check.get("ok") and check.get("exists")]
    tick = tick or collect_control_tick(config)
    tick_task = tick.get("task") or {}
    bootstrap_cmd = [python, "tools/fleet_control_pane.py", "bootstrap", "--apply"]
    config_drift = local_config_default_drift(config)
    steps = [
        {
            "id": "local-config",
            "state": "done" if local_path.exists() else "todo",
            "ok": True,
            "detail": str(local_path) if local_path.exists() else f"missing {local_path}",
            "why": "Persist host-specific paths under the ignored registry directory.",
            "command": [python, "tools/fleet_control_pane.py", "init"],
        },
    ]
    if config_drift.get("count"):
        steps.append({
            "id": "local-config-tracked-defaults",
            "state": "todo",
            "ok": False,
            "detail": str(config_drift.get("detail") or "local config shadows tracked defaults"),
            "why": "Refresh shared setup defaults in the ignored local config without hand-editing JSON.",
            "command": list(config_drift.get("command") or []),
        })
    guard = trunk_guard_status(config)
    steps.extend([
        {
            "id": "runtime-dirs",
            "state": "blocked" if failed_dirs else ("todo" if missing_dirs else "done"),
            "ok": not failed_dirs,
            "detail": (
                "not writable: " + ", ".join(f"{c['name']}={c.get('reason')}" for c in failed_dirs)
                if failed_dirs
                else ("missing: " + ", ".join(missing_dirs) if missing_dirs else "registry, machine, and watchdog directories are ready")
            ),
            "why": "Ensure registry, machine snapshot, and watchdog log directories exist and are writable.",
            "command": bootstrap_cmd,
        },
        {
            "id": "control-tick",
            "state": "manual",
            "ok": True,
            "detail": "safe to run any time; scheduled task covers the recurring cadence when installed",
            "why": "Run one keep-alive tick now: refresh sessions, invoke watchdogs, and write the pane.",
            "command": [
                python,
                "tools/fleet_control_pane.py",
                "tick",
            ],
        },
    ])
    if guard.get("supported") or guard.get("installer_exists") or guard.get("hook_exists"):
        steps.append({
            "id": "install-trunk-guard",
            "state": guard.get("state"),
            "ok": bool(guard.get("setup_ok")),
            "detail": guard.get("detail"),
            "why": "Install the repo-local git hook that blocks accidental off-master branch creation below the agent layer.",
            "command": list(guard.get("command") or []),
        })
    script = control_tick_register_script_path(config, tick)
    install_state = "todo"
    install_ok = True
    install_detail = "install the recurring control-pane tick"
    install_command = register_command(script, "control_tick", config) if script else []
    runner_missing = control_tick_runner_missing(tick, tick_task)
    tick_disabled = scheduler_state_needs_action(tick_task)
    if task_is_installed(tick_task) and not runner_missing and not tick_disabled:
        install_state = "done"
        install_detail = f"{tick.get('task_name')} already installed"
    elif runner_missing:
        install_state = "todo"
        install_detail = f"control-pane tick runner missing: {tick.get('runner')}"
    elif tick_disabled:
        install_state = "todo"
        install_detail = f"control-pane tick task is {tick_task.get('state')}"
    elif not script or not script.exists():
        install_state = "blocked"
        install_ok = False
        install_detail = "control-pane tick register script not found"
        install_command = []
    elif tick_task.get("supported") is False:
        install_state = "blocked"
        install_ok = False
        install_detail = f"scheduler unsupported: {tick_task.get('reason') or 'unknown'}"
    if script:
        steps.append({
            "id": "install-control-pane-tick",
            "state": install_state,
            "ok": install_ok,
            "detail": install_detail,
            "why": "Let the OS run one pane tick on a cadence instead of installing per-loop tasks by hand.",
            "command": install_command,
        })
    steps.extend(loop_setup_steps(config))
    return {
        "schema": SETUP_SCHEMA,
        "generated_utc": iso_now(),
        "machine": {"host": socket.gethostname(), "platform": platform.platform()},
        "steps": [
            {**step, "display": display_command(step["command"])}
            for step in steps
        ],
    }


def doctor(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("repo-root", root.exists(), str(root))
    add("local-config", bool(config.get("_local_exists")), config["_paths"]["local"])
    config_drift = local_config_default_drift(config)
    if config_drift.get("count"):
        add("local-config-tracked-defaults", False, str(config_drift.get("detail") or "local config shadows tracked defaults"))
    add("python", command_exists(str(config.get("python"))), str(config.get("python")))
    add("user-home", Path(str(config.get("user_home", ""))).exists(), str(config.get("user_home")))
    job_dir = Path(str(config.get("job_dir", ""))) if config.get("job_dir") else None
    add("job-dir", bool(job_dir and job_dir.exists()), str(job_dir or "not configured"))
    claude = str(config.get("claude_exe", ""))
    add("claude-exe", bool(claude and command_exists(claude)), claude or "not configured")
    for check in ensure_runtime_dirs(config, apply=False)["checks"]:
        add(f"{check['name']}-writable", bool(check.get("ok")), str(check.get("path")) if check.get("ok") else f"{check.get('path')}: {check.get('reason')}")
    for rel in (
        "tools/fleet_sessions.py",
        "tools/fleet_accounts.py",
        "tools/safe_ff_sync.py",
        "tools/fleet_control_pane.py",
        "tools/register_control_pane_tick.ps1",
        "tools/register_control_pane_tick.sh",
    ):
        add(rel, (root / rel).exists(), rel)
    guard = trunk_guard_status(config)
    if guard.get("supported") or guard.get("installer_exists") or guard.get("hook_exists"):
        add("trunk-guard", bool(guard.get("ok")), str(guard.get("detail") or "inspect trunk guard"))
    tick = collect_control_tick(config)
    add("control-tick-register-script", bool(tick.get("register_script_exists")), str(tick.get("register_script")))
    tick_task = tick.get("task") or {}
    pane_tick_installed = task_is_installed(tick_task)
    if tick_task.get("supported"):
        add("control-tick-scheduled-task", bool(tick_task.get("installed")), str(tick_task.get("task_name")))
    if pane_tick_installed and "runner_exists" in tick:
        add("control-tick-runner", bool(tick.get("runner_exists")), str(tick.get("runner")))
    for name, wd in (collect_watchdogs(config) or {}).items():
        add(f"{name}-watchdog-script", bool(wd.get("script_exists")), str(wd.get("script")))
        task = wd.get("task") or {}
        if task.get("supported"):
            if pane_tick_installed:
                state = "installed" if task.get("installed") else "not installed"
                reason = task.get("reason")
                detail = f"covered by control-pane tick {tick.get('task_name')}; standalone {state}"
                if reason:
                    detail = f"{detail}: {reason}"
                add(f"{name}-scheduled-task", True, detail)
            else:
                add(f"{name}-scheduled-task", bool(task.get("installed")), str(task.get("task_name")))
    for name, spec in sorted((config.get("loops") or {}).items()):
        if not isinstance(spec, dict) or spec.get("enabled") is False:
            continue
        status_cmd = expand_cmd(list(spec.get("status_cmd") or []), config)
        recover_cmd = expand_cmd(list(spec.get("recover_cmd") or []), config)
        status_ready = command_readiness(status_cmd, root)
        add(f"{name}-loop-status-cmd", bool(status_ready["ok"]), str(status_ready["detail"]))
        if recover_cmd:
            recover_ready = command_readiness(recover_cmd, root)
            add(f"{name}-loop-recover-cmd", bool(recover_ready["ok"]), str(recover_ready["detail"]))

    ok = all(c["ok"] for c in checks if c["name"] not in {"local-config", "job-dir", "claude-exe"})
    return {
        "schema": "fleet-control-pane.doctor/1",
        "generated_utc": iso_now(),
        "ok": ok,
        "checks": checks,
        "setup_plan": setup_plan(config),
    }


def pane_text(status: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"FLEET CONTROL PANE @ {status['generated_utc']}")
    lines.append(f"verdict: {status['verdict']}  host={status['machine']['host']}")

    git_status = status.get("git") or {}
    if git_status.get("available"):
        counts = git_status.get("counts") or {}
        lines.append(
            "git: "
            f"{git_status.get('branch')} {git_status.get('head')}  "
            f"dirty={git_status.get('dirty_total', 0)} "
            f"M={counts.get('modified', 0)} D={counts.get('deleted', 0)} "
            f"?={counts.get('untracked', 0)}"
        )
        if counts.get("unmerged"):
            lines[-1] += f" U={counts.get('unmerged', 0)}"
        if git_status.get("merge_in_progress"):
            lines.append("merge: in-progress")
        safe_ff = git_status.get("safe_ff") or {}
        lines.append(
            "safe-ff: "
            f"state={safe_ff.get('state')} ok={safe_ff.get('ok')} "
            f"divergent={safe_ff.get('divergent_count', 0)}"
        )
        worktrees = git_status.get("worktrees") or {}
        if worktrees.get("available"):
            lines.append(
                "worktrees: "
                f"total={worktrees.get('total', 0)} "
                f"converged={worktrees.get('converged')} "
                f"prune={worktrees.get('prune_count', 0)} "
                f"blocked={worktrees.get('blocked_count', 0)} "
                f"retained={worktrees.get('retained_count', 0)}"
            )
            for item in (worktrees.get("blocked") or [])[:3]:
                reasons = ", ".join(str(reason) for reason in item.get("reasons") or [])
                lines.append(
                    f"  blocked-worktree: {item.get('path')} "
                    f"({item.get('branch') or 'detached'}): {reasons or 'inspect'}"
                )
        elif worktrees:
            lines.append(f"worktrees: unavailable ({worktrees.get('reason')})")
        dirty_plan = git_status.get("dirty_plan") or {}
        dirty_groups = dirty_plan.get("groups") or []
        if dirty_groups:
            lines.append("dirty-plan:")
            for group in dirty_groups[:6]:
                lines.append(f"  {group.get('group')}: {group.get('count', 0)} path(s)")
                if group.get("command"):
                    lines.append(f"    {group['command']}")
                elif group.get("blocked"):
                    lines.append(f"    blocked: {group.get('reason', 'inspect dirty-plan JSON')}")
            omitted = max(0, len(dirty_groups) - 6)
            if omitted:
                lines.append(f"  ... {omitted} more group(s); use `status --json` for all dirty-plan paths")
    else:
        lines.append(f"git: unavailable ({git_status.get('reason')})")

    reg = status.get("registry") or {}
    if reg.get("exists"):
        lines.append(
            "sessions: "
            f"{reg.get('sessions', 0)} age={reg.get('age_min')}m "
            f"categories={compact_counts(reg.get('categories') or {})}"
        )
        lines.append(
            "actions: "
            f"{compact_counts(reg.get('actions') or {})}"
        )
        accts = reg.get("accounts") or {}
        lines.append(
            "accounts: "
            f"available={accts.get('available', 0)}/{accts.get('total', 0)} "
            f"blocked={len(accts.get('blocked', []))} "
            f"tags={','.join(accts.get('available_tags', [])[:8]) or '(none)'}"
        )
        blocked_text = blocked_account_reasons(accts)
        if blocked_text:
            lines.append(f"blocked-accounts: {blocked_text}")
            for account_line in blocked_account_action_lines(accts):
                lines.append(f"account-action: {account_line}")
    else:
        lines.append("sessions: registry missing")

    sup = status.get("supervisor") or {}
    if sup.get("available"):
        payload = sup.get("payload") or {}
        proc = payload.get("process") or {}
        sup_summary = supervisor_health_summary(
            sup,
            status.get("config") or {},
            watchdog_task=(((status.get("watchdogs") or {}).get("supervisor") or {}).get("task") or {}),
        )
        lines.append(
            "supervisor: "
            f"verdict={payload.get('verdict')} alive={proc.get('alive')} "
            f"pid={proc.get('pid')} hb_age={proc.get('heartbeat_age_s')} "
            f"run={sup_summary.get('run') or '(none)'} "
            f"health={sup_summary.get('run_health') or sup_summary.get('diagnose') or 'OK'}"
        )
        if supervisor_health_needs_action(sup_summary):
            lines.append(f"supervisor-action: {supervisor_health_action_text(sup_summary)}")
    else:
        lines.append(f"supervisor: unavailable ({sup.get('reason')})")

    tick = status.get("control_tick") or {}
    tick_task = tick.get("task") or {}
    if tick_task.get("supported"):
        state = "installed" if tick_task.get("installed") else "missing"
        extra = f" state={tick_task.get('state')}" if tick_task.get("state") else ""
    else:
        state = "unsupported"
        extra = f" ({tick_task.get('reason')})"
    lines.append(f"control-tick: {state}{extra}")

    for name, wd in sorted((status.get("watchdogs") or {}).items()):
        task = wd.get("task") or {}
        if task.get("supported"):
            state = "installed" if task.get("installed") else "missing"
            extra = f" state={task.get('state')}" if task.get("state") else ""
        else:
            state = "unsupported"
            extra = f" ({task.get('reason')})"
        lines.append(f"watchdog[{name}]: {state}{extra}")

    loops = status.get("loops") or {}
    if loops.get("count"):
        states = compact_counts(loops.get("states") or {})
        lines.append(f"loops: {states}")
        for check in (loops.get("checks") or [])[:5]:
            if check.get("state") != "OK":
                detail = check.get("detail") or check.get("reason") or ""
                lines.append(f"loop[{check.get('name')}]: {check.get('state')} {detail}".rstrip())
    elif "enabled" in loops:
        lines.append(
            "loops: "
            f"enabled={loops.get('enabled', 0)} "
            f"disabled={loops.get('disabled', 0)} "
            f"configured={loops.get('configured', 0)}"
        )

    if status.get("actions"):
        lines.append("recommended:")
        for action in status["actions"]:
            lines.append(f"  - {action}")
    return "\n".join(lines)


def blocked_account_reasons(accounts: dict[str, Any], *, limit: int = 5) -> str:
    blocked = accounts.get("blocked") or []
    blocked_accounts = []
    for account in blocked[:limit]:
        if not isinstance(account, dict):
            continue
        tag = str(account.get("tag") or "unknown")
        reason = str(account.get("reason") or "").strip()
        kind = str(account.get("block_kind") or "").strip()
        sessions = int(account.get("auth_blocked_sessions") or 0)
        suffix = ""
        if kind:
            suffix += f"/{kind}"
        if sessions:
            suffix += f"/{sessions}s"
        label = f"{tag}[{suffix.lstrip('/')}]" if suffix else tag
        blocked_accounts.append(f"{label}={reason}" if reason else label)
    if not blocked_accounts:
        return ""
    omitted = max(0, len(blocked) - len(blocked_accounts))
    suffix = f" | +{omitted} more" if omitted else ""
    return f"{' | '.join(blocked_accounts)}{suffix}"


def blocked_account_action_lines(accounts: dict[str, Any], *, limit: int = 3) -> list[str]:
    lines = []
    for account in (accounts.get("blocked") or [])[:limit]:
        if not isinstance(account, dict):
            continue
        tag = str(account.get("tag") or "unknown")
        reason = str(account.get("reason") or "blocked").strip()
        config_dir = str(account.get("config_dir") or "").strip()
        command = str(account.get("command") or "").strip()
        detail = f"{tag}: {reason}"
        if config_dir:
            detail += f"; profile={config_dir}"
        if command and account_block_needs_login(account):
            detail += f"; login/check with `{command}`"
        lines.append(detail)
    blocked = accounts.get("blocked") or []
    omitted = max(0, len(blocked) - len(lines))
    if omitted:
        lines.append(f"... {omitted} more blocked account(s); inspect `python tools/fleet_accounts.py list`")
    return lines


def blocked_account_recommended_action(registry_summary: dict[str, Any]) -> str:
    accounts = (registry_summary.get("accounts") or {}).get("blocked") or []
    if not accounts and not int(registry_summary.get("auth_blocked") or 0):
        return ""
    auth_accounts = [a for a in accounts if isinstance(a, dict) and account_block_needs_login(a)]
    credit_accounts = [
        a for a in accounts
        if isinstance(a, dict) and str(a.get("block_kind") or "").lower() == "credit"
    ]
    usage_accounts = [a for a in accounts if isinstance(a, dict) and account_block_is_usage_like(a)]
    access_accounts = [a for a in accounts if isinstance(a, dict) and account_block_is_access_wall(a)]
    other_accounts = [
        a for a in accounts
        if isinstance(a, dict)
        and a not in auth_accounts
        and a not in credit_accounts
        and a not in usage_accounts
        and a not in access_accounts
    ]
    session_count = (
        sum(int(a.get("auth_blocked_sessions") or 0) for a in auth_accounts)
        if auth_accounts else int(registry_summary.get("auth_blocked") or 0)
    )
    parts = []
    if auth_accounts:
        parts.append(
            f"{len(auth_accounts)} login-blocked account(s)"
            f"{f' covering {session_count} auth-stopped session(s)' if session_count else ''}: "
            f"{blocked_account_tags(auth_accounts)}"
        )
    elif session_count and not accounts:
        parts.append(f"{session_count} auth-stopped session(s) but no account-level auth blocker is currently active")
    if credit_accounts:
        parts.append(f"{len(credit_accounts)} credit-blocked account(s): {blocked_account_tags(credit_accounts)}")
    if usage_accounts:
        # Usage limits are expected time walls. They stay visible in the account
        # summary, but should not create an operator remediation prompt.
        pass
    if other_accounts:
        parts.append(f"{len(other_accounts)} other blocked account(s): {blocked_account_tags(other_accounts)}")
    first_command = next((str(a.get("command")) for a in auth_accounts if a.get("command")), "")
    command_text = f"; first remediation command: `{first_command}`" if first_command else ""
    if not parts:
        return ""
    return (
        "Account remediation needed: "
        + "; ".join(parts)
        + command_text
        + "; inspect `python tools/fleet_accounts.py list`."
    )


def blocked_account_tags(accounts: list[dict[str, Any]], *, limit: int = 4) -> str:
    parts = []
    for account in accounts[:limit]:
        tag = str(account.get("tag") or "unknown")
        reason = str(account.get("reason") or "").strip()
        sessions = int(account.get("auth_blocked_sessions") or 0)
        session_text = f", {sessions} session(s)" if sessions else ""
        parts.append(f"{tag}={reason}{session_text}" if reason else f"{tag}{session_text}")
    omitted = max(0, len(accounts) - len(parts))
    if omitted:
        parts.append(f"+{omitted} more")
    return " | ".join(parts)


def dirty_group_summary(groups: list[dict[str, Any]], *, limit: int = 4) -> str:
    parts = [
        f"{group.get('group')}={group.get('count', 0)}"
        for group in groups[:limit]
        if group.get("group")
    ]
    if not parts:
        return "(none)"
    omitted = max(0, len(groups) - len(parts))
    suffix = f" +{omitted} more" if omitted else ""
    return f"{', '.join(parts)}{suffix}"


def machine_setup_action_count(machine: dict[str, Any]) -> int:
    count = 1 if scheduled_task_summary_needs_action(machine.get("control_tick") or {}) else 0
    for task in (machine.get("watchdogs") or {}).values():
        count += 1 if scheduled_task_summary_needs_action(task or {}) else 0
    return count


def machine_setup_summary(machine: dict[str, Any]) -> str:
    parts: list[str] = []
    tick = machine.get("control_tick") or {}
    if scheduled_task_has_data(tick):
        parts.append(f"tick={scheduled_task_label(tick)}")
    watchdogs = machine.get("watchdogs") or {}
    watchdog_parts = [
        f"{name}={scheduled_task_label(task or {})}"
        for name, task in sorted(watchdogs.items())
        if scheduled_task_has_data(task or {})
    ]
    if watchdog_parts:
        parts.append(f"watchdogs={','.join(watchdog_parts)}")
    return " ".join(parts)


def setup_step_snapshot(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": step.get("id"),
        "state": step.get("state"),
        "ok": step.get("ok"),
        "detail": step.get("detail"),
        "display": step.get("display") or display_command(list(step.get("command") or [])),
        "command": list(step.get("command") or []),
    }


def setup_plan_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    steps = [
        setup_step_snapshot(step)
        for step in plan.get("steps") or []
        if isinstance(step, dict)
    ]
    states = collections.Counter(str(step.get("state") or "unknown") for step in steps)
    actionable = [
        step
        for step in steps
        if step.get("state") in {"todo", "blocked"} or step.get("ok") is False
    ]
    return {
        "states": dict(states),
        "action_count": len(actionable),
        "steps": steps,
    }


def setup_plan_action_steps(machine: dict[str, Any]) -> list[dict[str, Any]]:
    plan = machine.get("setup_plan") or {}
    return [
        step
        for step in (plan.get("steps") or [])
        if step.get("state") in {"todo", "blocked"} or step.get("ok") is False
    ]


def machine_git_action(git_status: dict[str, Any]) -> str:
    state = str(git_status.get("safe_ff_state") or "")
    if state == "ahead":
        if int(git_status.get("dirty_total") or 0) > 0 or git_status.get("merge_in_progress"):
            return ""
        return "python tools/fleet_control_pane.py publish --json"
    if state in {"behind", "diverged"} or git_status.get("safe_ff_ok") is False:
        return "python tools/fleet_control_pane.py sync --fetch --json"
    return ""


def machine_version_drift(machine: dict[str, Any], current_version: str) -> dict[str, Any] | None:
    if not current_version:
        return None
    machine_version = str(machine.get("app_version") or "unknown")
    if machine_version == current_version:
        return None
    git_status = machine.get("git") or {}
    git_state = str(git_status.get("safe_ff_state") or "unknown")
    command = machine_git_action(git_status) or "python tools/fleet_control_pane.py sync --fetch --json"
    detail = "pane version differs from this operator pane"
    if git_state == "in-sync":
        detail += "; publish the newer pane first if it is only local, then sync this host"
    return {
        "machine_version": machine_version,
        "current_version": current_version,
        "git_state": git_state,
        "command": command,
        "detail": detail,
    }


def loop_issue_summary(check: dict[str, Any]) -> str:
    name = check.get("name") or "unknown"
    state = check.get("state") or "UNKNOWN"
    detail = check.get("detail") or check.get("action") or ""
    flags = []
    if check.get("auto_recover"):
        flags.append("auto-recover")
    elif check.get("has_recover_cmd"):
        flags.append("recoverable")
    suffix = f" ({', '.join(flags)})" if flags else ""
    body = f" {detail}" if detail else ""
    return f"loop[{name}]: {state}{suffix}{body}".rstrip()


def compact_counts(counts: dict[str, Any]) -> str:
    parts = [f"{k}={v}" for k, v in sorted(counts.items()) if k and v]
    return " ".join(parts) if parts else "(none)"


def write_status(status: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    reg_dir = Path(config["registry_dir"])
    json_path = reg_dir / "control_pane.json"
    text_path = reg_dir / "CONTROL-PANE.txt"
    machine_path = Path(config["machine_dir"]) / f"{machine_id(config)}.json"
    write_json(json_path, status)
    write_text_atomic(text_path, pane_text(status) + "\n")
    write_json(machine_path, status)
    return {"json": str(json_path), "text": str(text_path), "machine": str(machine_path)}


def summarize_machine_snapshot(path: Path, status: dict[str, Any], stale_min: float) -> dict[str, Any]:
    if not status:
        return {
            "id": path.stem,
            "source": str(path),
            "state": "INVALID",
            "reason": "missing or malformed JSON",
        }
    machine = status.get("machine") or {}
    reg = status.get("registry") or {}
    accts = reg.get("accounts") or {}
    blocked_accounts = []
    for account in accts.get("blocked") or []:
        if not isinstance(account, dict):
            continue
        blocked_accounts.append({
            **account,
            "source_machine": str(machine.get("id") or path.stem),
            "source_host": str(machine.get("host") or path.stem),
            "source_snapshot": str(path),
        })
    supervisor = status.get("supervisor") or {}
    sup_payload = supervisor.get("payload") or {}
    sup_proc = sup_payload.get("process") or {}
    status_config = status.get("config") or {}
    supervisor_summary = supervisor_health_summary(
        supervisor,
        status_config,
        watchdog_task=(((status.get("watchdogs") or {}).get("supervisor") or {}).get("task") or {}),
    )
    tick = status.get("control_tick") or {}
    tick_task = tick.get("task") or {}
    watchdogs = status.get("watchdogs") or {}
    loops = status.get("loops") or {}
    loop_checks = loops.get("checks") or []
    loop_summaries = [loop_check_snapshot(check) for check in loop_checks]
    setup_summary = setup_plan_snapshot(status.get("setup_plan") or {})
    git_status = status.get("git") or {}
    safe_ff = git_status.get("safe_ff") or {}
    worktrees = git_status.get("worktrees") or {}
    dirty_plan = git_status.get("dirty_plan") or {}
    dirty_groups = dirty_plan.get("groups") or []
    age = parse_iso_age_min(status.get("generated_utc"))
    verdict = str(status.get("verdict") or "UNKNOWN")
    state = verdict
    if age is None:
        state = "UNKNOWN"
    elif age > stale_min:
        state = "STALE"
    return {
        "id": str(machine.get("id") or path.stem),
        "host": str(machine.get("host") or path.stem),
        "app_version": status.get("app_version"),
        "source": str(path),
        "generated_utc": status.get("generated_utc"),
        "age_min": age,
        "state": state,
        "verdict": verdict,
        "actions_count": len(status.get("actions") or []),
        "actions": list(status.get("actions") or [])[:12],
        "sessions": reg.get("sessions", 0),
        "categories": reg.get("categories") or {},
        "session_actions": reg.get("actions") or {},
        "auth_blocked": reg.get("auth_blocked", 0),
        "auto_resume": reg.get("auto_resume", 0),
        "surface": reg.get("surface", 0),
        "accounts": {
            "available": accts.get("available", 0),
            "total": accts.get("total", 0),
            "available_tags": accts.get("available_tags") or [],
            "blocked": blocked_accounts,
            "blocked_count": accts.get("blocked_count", len(blocked_accounts)),
            "auth_blocked_count": accts.get("auth_blocked_count", sum(1 for a in blocked_accounts if account_block_is_auth_like(a))),
            "usage_blocked_count": accts.get("usage_blocked_count", sum(1 for a in blocked_accounts if account_block_is_usage_like(a))),
            "blocked_kinds": accts.get("blocked_kinds") or {},
        },
        "supervisor": {
            "available": supervisor.get("available", False),
            "verdict": sup_payload.get("verdict"),
            "alive": sup_proc.get("alive"),
            "pid": sup_proc.get("pid"),
            "heartbeat_age_s": sup_proc.get("heartbeat_age_s"),
            "diagnose": (sup_payload.get("diagnose") or {}).get("health"),
            "run": supervisor_summary.get("run"),
            "run_health": supervisor_summary.get("run_health"),
            "primary_cause": supervisor_summary.get("primary_cause"),
            "primary_action": supervisor_summary.get("primary_action"),
            "run_dir": supervisor_summary.get("run_dir"),
            "run_log": supervisor_summary.get("run_log"),
            "watchdog_last_result": supervisor_summary.get("watchdog_last_result"),
            "watchdog_last_result_text": supervisor_summary.get("watchdog_last_result_text"),
            "watchdog_last_result_accepted": supervisor_summary.get("watchdog_last_result_accepted"),
            "needs_action": supervisor_health_needs_action(supervisor_summary),
            "action": supervisor_health_action_text(supervisor_summary)
            if supervisor_health_needs_action(supervisor_summary)
            else "",
        },
        "control_tick": scheduled_task_snapshot(tick_task or {}),
        "watchdogs": {
            name: scheduled_task_snapshot(
                (wd or {}).get("task") or {},
                accepted_results={SUPERVISOR_WATCHDOG_ACTION_RESULT} if name == "supervisor" else None,
            )
            for name, wd in sorted(watchdogs.items())
        },
        "setup_plan": setup_summary,
        "loops": {
            "count": loops.get("count", 0),
            "configured": loops.get("configured", loops.get("count", 0)),
            "enabled": loops.get("enabled", loops.get("count", 0)),
            "disabled": loops.get("disabled", 0),
            "states": loops.get("states") or {},
            "action": sum(1 for check in loop_summaries if loop_check_needs_action(check)),
            "checks": loop_summaries,
            "commands": list(loops.get("commands") or [])[:3],
        },
        "git": {
            "branch": git_status.get("branch"),
            "head": git_status.get("head"),
            "dirty_total": git_status.get("dirty_total", 0),
            "merge_in_progress": git_status.get("merge_in_progress", False),
            "dirty_plan": {
                "count": dirty_plan.get("count", len(dirty_groups)),
                "groups": list(dirty_groups),
            },
            "safe_ff_state": safe_ff.get("state"),
            "safe_ff_ok": safe_ff.get("ok"),
            "safe_ff_reason": safe_ff.get("reason"),
            "worktrees": {
                "available": worktrees.get("available", False),
                "total": worktrees.get("total", 0),
                "converged": worktrees.get("converged"),
                "needs_human": worktrees.get("needs_human", False),
                "primary_offtrack": worktrees.get("primary_offtrack", False),
                "prune_count": worktrees.get("prune_count", 0),
                "blocked_count": worktrees.get("blocked_count", 0),
                "retained_count": worktrees.get("retained_count", 0),
                "prune": list(worktrees.get("prune") or []),
                "blocked": list(worktrees.get("blocked") or []),
                "commands": worktrees.get("commands") or {},
                "reason": worktrees.get("reason"),
            },
        },
    }


def fleet_view(
    config: dict[str, Any],
    *,
    include_live_local: bool = False,
    refresh_live_local: bool = False,
    write_live_local: bool = False,
) -> dict[str, Any]:
    machine_dir = Path(config["machine_dir"])
    stale_min = float(config.get("machine_stale_min", 15))
    machines = [
        summarize_machine_snapshot(path, read_json(path), stale_min)
        for path in sorted(machine_dir.glob("*.json"))
    ] if machine_dir.exists() else []
    local_written = None
    if include_live_local:
        local_path = machine_dir / f"{machine_id(config)}.json"
        local_status = collect_status(config, refresh=refresh_live_local)
        if write_live_local:
            local_written = write_status(local_status, config)
        local_machine = summarize_machine_snapshot(local_path, local_status, stale_min)
        local_machine["live_local"] = True
        local_machine["refresh_local"] = refresh_live_local
        local_id = local_machine.get("id")
        machines = [m for m in machines if m.get("id") != local_id]
        machines.append(local_machine)
        machines.sort(key=lambda m: str(m.get("id") or ""))
    current_version = fleet_version.app_version(Path(config["root"])) if config.get("root") else ""
    for machine in machines:
        drift = machine_version_drift(machine, current_version)
        if drift:
            machine["version_drift"] = drift
    states = collections.Counter(m.get("state", "UNKNOWN") for m in machines)
    versions = collections.Counter(str(m.get("app_version") or "unknown") for m in machines)
    version_mismatches = sum(1 for m in machines if m.get("version_drift"))
    totals = {
        "sessions": sum(int(m.get("sessions") or 0) for m in machines),
        "actions": sum(int(m.get("actions_count") or 0) for m in machines),
        "auth_blocked": sum(int(m.get("auth_blocked") or 0) for m in machines),
        "blocked_accounts": sum(len(((m.get("accounts") or {}).get("blocked") or [])) for m in machines),
        "auth_blocked_accounts": sum(int(((m.get("accounts") or {}).get("auth_blocked_count") or 0)) for m in machines),
        "auto_resume": sum(int(m.get("auto_resume") or 0) for m in machines),
        "surface": sum(int(m.get("surface") or 0) for m in machines),
        "dirty_paths": sum(int((m.get("git") or {}).get("dirty_total") or 0) for m in machines),
        "worktree_prune": sum(int(((m.get("git") or {}).get("worktrees") or {}).get("prune_count") or 0) for m in machines),
        "worktree_blocked": sum(int(((m.get("git") or {}).get("worktrees") or {}).get("blocked_count") or 0) for m in machines),
        "setup_actions": sum(machine_setup_action_count(m) for m in machines),
        "setup_plan_actions": sum(len(setup_plan_action_steps(m)) for m in machines),
        "loop_actions": sum(int((m.get("loops") or {}).get("action") or 0) for m in machines),
        "version_mismatches": version_mismatches,
    }
    verdict = "OK"
    if not machines:
        verdict = "EMPTY"
    elif version_mismatches or any(m.get("state") in {"ACTION", "STALE", "UNKNOWN", "INVALID"} for m in machines):
        verdict = "ACTION"
    return {
        "schema": FLEET_SCHEMA,
        "generated_utc": iso_now(),
        "machine_dir": str(machine_dir),
        "stale_min": stale_min,
        "include_live_local": include_live_local,
        "refresh_live_local": refresh_live_local,
        "current_version": current_version,
        "written": local_written,
        "verdict": verdict,
        "states": dict(states),
        "versions": dict(versions),
        "totals": totals,
        "machines": machines,
    }


def fleet_text(doc: dict[str, Any]) -> str:
    lines = [
        f"FLEET CONTROL PANE AGGREGATE @ {doc['generated_utc']}",
        f"verdict: {doc['verdict']}  machines={len(doc.get('machines') or [])}  states={compact_counts(doc.get('states') or {})}",
    ]
    versions = compact_counts(doc.get("versions") or {})
    if versions != "(none)":
        lines.append(f"versions: {versions}")
    totals = doc.get("totals") or {}
    lines.append(
        "totals: "
        f"sessions={totals.get('sessions', 0)} actions={totals.get('actions', 0)} "
        f"auth_sessions={totals.get('auth_blocked', 0)} "
        f"auth_accounts={totals.get('auth_blocked_accounts', 0)} "
        f"blocked_accounts={totals.get('blocked_accounts', 0)} "
        f"auto_resume={totals.get('auto_resume', 0)} "
        f"setup={totals.get('setup_actions', 0)} setup_plan={totals.get('setup_plan_actions', 0)} "
        f"version={totals.get('version_mismatches', 0)} "
        f"loops={totals.get('loop_actions', 0)} "
        f"dirty={totals.get('dirty_paths', 0)} "
        f"wt_prune={totals.get('worktree_prune', 0)} "
        f"wt_blocked={totals.get('worktree_blocked', 0)}"
    )
    if doc.get("written"):
        written = doc["written"]
        lines.append(f"local-written: {written.get('json')} ; {written.get('machine')}")
    for m in doc.get("machines") or []:
        sup = m.get("supervisor") or {}
        git_status = m.get("git") or {}
        accounts = m.get("accounts") or {}
        loops = m.get("loops") or {}
        source = " live" if m.get("live_local") else ""
        lines.append(
            f"- {m.get('id')}{source}: state={m.get('state')} age={m.get('age_min')}m "
            f"ver={m.get('app_version') or 'unknown'} "
            f"sessions={m.get('sessions', 0)} actions={m.get('actions_count', 0)} "
            f"acct={accounts.get('available', 0)}/{accounts.get('total', 0)} "
            f"sup={sup.get('verdict')}/{sup.get('diagnose')} "
            f"loops={loops.get('action', 0)}/{loops.get('count', 0)} "
            f"git={git_status.get('branch')} sync={git_status.get('safe_ff_state') or 'unknown'} "
            f"dirty={git_status.get('dirty_total', 0)}"
        )
        blocked_text = blocked_account_reasons(accounts, limit=3)
        if blocked_text:
            lines.append(f"    blocked-accounts: {blocked_text}")
            for account_line in blocked_account_action_lines(accounts, limit=2):
                lines.append(f"    account-action: {account_line}")
        setup_text = machine_setup_summary(m)
        if setup_text:
            lines.append(f"    setup: {setup_text}")
        if sup.get("action"):
            lines.append(f"    supervisor-action: {sup['action']}")
        drift = m.get("version_drift") or {}
        if drift:
            lines.append(
                "    version-action: "
                f"pane {drift.get('machine_version')} -> {drift.get('current_version')}; "
                f"{drift.get('command')}"
            )
            if drift.get("detail"):
                lines.append(f"      {drift['detail']}")
        git_action = machine_git_action(git_status)
        if git_action:
            reason = f" ({git_status.get('safe_ff_reason')})" if git_status.get("safe_ff_reason") else ""
            lines.append(f"    git-action: {git_action}{reason}")
        worktrees = git_status.get("worktrees") or {}
        if int(worktrees.get("prune_count") or 0) > 0:
            cmd = (worktrees.get("commands") or {}).get("prune") or "python tools/worktree_doctor.py --prune --fetch"
            lines.append(f"    worktree-action: prune {worktrees.get('prune_count')} safe extra worktree(s); {cmd}")
        if int(worktrees.get("blocked_count") or 0) > 0 or worktrees.get("primary_offtrack"):
            cmd = (worktrees.get("commands") or {}).get("inspect") or "python tools/worktree_doctor.py --fetch"
            lines.append(f"    worktree-action: inspect blocked/off-track worktree state; {cmd}")
        setup_steps = setup_plan_action_steps(m)
        for step in setup_steps[:3]:
            detail = f" {step.get('detail')}" if step.get("detail") else ""
            lines.append(f"    setup-step[{step.get('id')}]: {step.get('state')}{detail}".rstrip())
            if step.get("display"):
                lines.append(f"      {step['display']}")
        omitted_setup = max(0, len(setup_steps) - 3)
        if omitted_setup:
            lines.append(f"    ... {omitted_setup} more setup step(s); use `fleet --json` for all setup-plan commands")
        loop_issues = [
            check
            for check in (loops.get("checks") or [])
            if loop_check_needs_action(check)
        ]
        for check in loop_issues[:3]:
            lines.append(f"    {loop_issue_summary(check)}")
        omitted_loops = max(0, len(loop_issues) - 3)
        if omitted_loops:
            lines.append(f"    ... {omitted_loops} more loop issue(s); use `fleet --json` for all loop checks")
        if int(loops.get("enabled", loops.get("count") or 0) or 0) == 0 and loops.get("commands"):
            lines.append(f"    loop-action: {loops['commands'][0]}")
        dirty_plan = git_status.get("dirty_plan") or {}
        dirty_groups = dirty_plan.get("groups") or []
        if dirty_groups:
            lines.append(f"    dirty-groups: {dirty_group_summary(dirty_groups, limit=4)}")
            shown_commands = 0
            for group in dirty_groups:
                command = group.get("command")
                if not command:
                    continue
                lines.append(f"      {command}")
                shown_commands += 1
                if shown_commands >= 3:
                    break
            omitted = max(0, len(dirty_groups) - shown_commands)
            if omitted:
                lines.append(f"      ... {omitted} more group(s); use `fleet --json` for all dirty-plan paths")
        for action in (m.get("actions") or [])[:3]:
            lines.append(f"    - {action}")
    if not doc.get("machines"):
        lines.append(f"no machine snapshots found in {doc.get('machine_dir')}")
    return "\n".join(lines)


def print_doc(doc: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(doc, indent=2))
    else:
        if doc.get("schema") == SETUP_SCHEMA:
            for step in doc["steps"]:
                state = str(step.get("state") or "step").upper()
                display = step.get("display") or "(no command)"
                print(f"{state} {step['id']}: {display}")
                print(f"  {step['why']}")
                if step.get("detail"):
                    print(f"  {step['detail']}")
        elif doc.get("schema") == "fleet-control-pane.doctor/1":
            print(f"doctor: {'OK' if doc['ok'] else 'ACTION'}")
            for c in doc["checks"]:
                print(f"  {'OK' if c['ok'] else 'MISS'}  {c['name']}: {c['detail']}")
        elif doc.get("schema") == TICK_SCHEMA:
            print(f"tick: {'OK' if doc['ok'] else 'ACTION'} dry_run={doc['dry_run']} live_resume={doc['live_resume']}")
            for action in doc.get("actions", []):
                state = "SKIP" if action.get("skipped") else ("OK" if action.get("ok") else "FAIL")
                detail = action.get("reason") or f"rc={action.get('returncode')}"
                print(f"  {state}  {action.get('name')}: {detail}")
            print(pane_text(doc["status"]))
            if doc.get("written"):
                print(f"written: {doc['written']['json']} ; {doc['written']['text']} ; {doc['written']['machine']}")
        elif doc.get("schema") == RECOVER_SCHEMA:
            print(f"recover: {'OK' if doc.get('ok') else 'ACTION'} apply={doc.get('apply')}")
            resume_plan = doc.get("resume_plan") or {}
            plan_path = resume_plan.get("display_path") or resume_plan.get("path") or "resume_plan.json"
            if resume_plan:
                if resume_plan.get("exists"):
                    print(f"resume-plan: {resume_plan.get('count', 0)} eligible session(s) from {plan_path}")
                    if resume_plan.get("reason"):
                        print(f"  {resume_plan['reason']}")
                    for session in resume_plan.get("sessions") or []:
                        sid = session.get("session_short") or session.get("session") or "unknown"
                        account = session.get("account") or "unknown-account"
                        project = session.get("project") or session.get("cwd") or "unknown-project"
                        print(f"  session={sid} account={account} project={project}")
                        if session.get("resume_cmd"):
                            print(f"       {session['resume_cmd']}")
                    if resume_plan.get("omitted"):
                        print(f"  ... {resume_plan['omitted']} more")
                else:
                    print(f"resume-plan: missing ({plan_path})")
            for action in doc.get("actions", []):
                state = "SKIP" if action.get("skipped") else ("OK" if action.get("ok") else "FAIL")
                detail = action.get("reason") or f"rc={action.get('returncode')}"
                print(f"  {state}  {action.get('name')}: {detail}")
                if action.get("command"):
                    print(f"       {action.get('command')}")
            if doc.get("status"):
                print(pane_text(doc["status"]))
            if doc.get("written"):
                print(f"written: {doc['written']['json']} ; {doc['written']['text']} ; {doc['written']['machine']}")
        elif doc.get("schema") == SUPERVISOR_SCHEMA:
            before = doc.get("before") or {}
            payload = before.get("payload") or {}
            proc = payload.get("process") or {}
            state = "ACTION" if doc.get("needs_action") else ("OK" if doc.get("ok") else "ERROR")
            print(f"supervisor: {state} verdict={payload.get('verdict')} alive={proc.get('alive')} pid={proc.get('pid')} restart={doc.get('restart')} apply={doc.get('apply')}")
            diagnose = payload.get("diagnose") or {}
            if diagnose:
                print(f"diagnose: {diagnose.get('health')} {diagnose.get('primary_action') or diagnose.get('primary_cause') or ''}".rstrip())
            if before.get("reason"):
                print(f"reason: {before.get('reason')}")
            for action in doc.get("actions", []):
                state = "SKIP" if action.get("skipped") else ("OK" if action.get("ok") else "FAIL")
                detail = action.get("reason") or f"rc={action.get('returncode')}"
                print(f"  {state}  {action.get('name')}: {detail}")
                if action.get("command"):
                    print(f"       {action.get('command')}")
        elif doc.get("schema") == FLEET_SCHEMA:
            print(fleet_text(doc))
        elif doc.get("schema") == COMMIT_SCHEMA:
            print(f"commit: {'OK' if doc.get('ok') else 'REFUSE'} applied={doc.get('applied')}")
            if doc.get("commit"):
                print(f"commit_sha: {doc['commit']}")
            if doc.get("reason"):
                print(f"reason: {doc['reason']}")
            if doc.get("paths"):
                print("paths:")
                for path in doc["paths"]:
                    print(f"  - {path}")
            dirty_groups = doc.get("dirty_groups") or {}
            if dirty_groups.get("requested"):
                print("dirty groups:")
                for group in dirty_groups.get("requested") or []:
                    print(f"  - {group}")
                if dirty_groups.get("available") and not dirty_groups.get("ok"):
                    print("available dirty groups:")
                    for group in dirty_groups["available"]:
                        print(f"  - {group}")
            if doc.get("commands"):
                print("commands:")
                for command in doc["commands"]:
                    print(f"  {command}")
            if doc.get("foreign_staged"):
                print("foreign staged:")
                for path in doc["foreign_staged"]:
                    print(f"  - {path}")
        elif doc.get("schema") == SYNC_SCHEMA:
            state = doc.get("state")
            print(f"sync: {'OK' if doc.get('ok') else 'REFUSE'} state={state} apply={doc.get('apply')} fetch={doc.get('fetch')}")
            if doc.get("branch"):
                print(f"branch: {doc.get('branch')} remote={doc.get('remote')}")
            if doc.get("reason"):
                print(f"reason: {doc.get('reason')}")
            info = doc.get("info") or {}
            if info.get("target_ref"):
                print(f"target: {info.get('target_ref')}")
            if info.get("write_count") is not None:
                print(f"write_set: {info.get('write_count')} path(s) divergent={len(info.get('divergent') or [])}")
            if info.get("new_head"):
                print(f"new_head: {info.get('new_head')}")
            if doc.get("commands"):
                print("commands:")
                for command in doc["commands"]:
                    print(f"  {command}")
        elif doc.get("schema") == PUBLISH_SCHEMA:
            print(f"publish: {'OK' if doc.get('ok') else 'REFUSE'} state={doc.get('state')} applied={doc.get('applied')} fetch={doc.get('fetch')}")
            if doc.get("branch"):
                print(f"branch: {doc.get('branch')} remote={doc.get('remote')}")
            if doc.get("reason"):
                print(f"reason: {doc.get('reason')}")
            ahead = doc.get("ahead_commits") or {}
            if ahead.get("count"):
                suffix = "" if ahead.get("shown") == ahead.get("count") else f" (showing {ahead.get('shown')})"
                print(f"ahead_commits: {ahead.get('count')} vs {ahead.get('target')}{suffix}")
                for commit in ahead.get("commits", []):
                    print(f"  {commit.get('short')} {commit.get('subject')}")
            if doc.get("commands"):
                print("commands:")
                for command in doc["commands"]:
                    print(f"  {command}")
        elif doc.get("schema") == BOOTSTRAP_SCHEMA:
            print(f"bootstrap: {'OK' if doc.get('ok') else 'ACTION'} apply={doc.get('apply')} live_resume={doc.get('live_resume')}")
            for action in doc.get("actions", []):
                if action.get("skipped"):
                    state = "SKIP"
                else:
                    state = "OK" if action.get("ok") else "FAIL"
                print(f"  {state}  {action.get('id')}: {action.get('detail')}")
                if action.get("command"):
                    print(f"       {action.get('command')}")
        elif doc.get("schema") == LOOP_CONFIG_SCHEMA:
            print(f"loop-add: {'OK' if doc.get('ok') else 'REFUSE'} apply={doc.get('apply')} changed={doc.get('changed')}")
            print(f"loop: {doc.get('loop')}")
            print(f"scope: {doc.get('scope') or 'local'}")
            print(f"config: {doc.get('config_path') or doc.get('local_config')}")
            if doc.get("reason"):
                print(f"reason: {doc.get('reason')}")
            spec = doc.get("spec") or {}
            if spec.get("status_cmd"):
                print(f"status_cmd: {display_command(list(spec.get('status_cmd') or []))}")
            if spec.get("recover_cmd"):
                print(f"recover_cmd: {display_command(list(spec.get('recover_cmd') or []))}")
            if doc.get("command") and not doc.get("apply"):
                print("commands:")
                print(f"  {doc['command']}")
            if doc.get("followup_commands"):
                print("next:")
                for command in doc["followup_commands"]:
                    print(f"  {command}")
        elif doc.get("schema") == LOOP_LIST_SCHEMA:
            print(
                f"loops: {'OK' if doc.get('ok') else 'ACTION'} "
                f"count={doc.get('count', 0)} enabled={doc.get('enabled', 0)} "
                f"disabled={doc.get('disabled', 0)} blocked={doc.get('blocked', 0)}"
            )
            for loop in doc.get("loops") or []:
                if not loop.get("enabled"):
                    state = "DISABLED"
                else:
                    state = "OK" if loop.get("ready") else "ACTION"
                suffix = " overridden" if loop.get("overridden") else ""
                print(f"- {loop.get('name')}: {state} source={loop.get('source')}{suffix}")
                if not loop.get("enabled"):
                    print("  readiness: skipped because loop is disabled")
                    continue
                status_ready = loop.get("status_ready") or {}
                print(f"  status: {'OK' if status_ready.get('ok') else 'MISS'} {status_ready.get('detail') or ''}".rstrip())
                recover_cmd = loop.get("recover_cmd") or []
                if loop.get("auto_recover") or recover_cmd:
                    recover_ready = loop.get("recover_ready") or {}
                    print(f"  recover: {'OK' if recover_ready.get('ok') else 'MISS'} {recover_ready.get('detail') or ''}".rstrip())
            if doc.get("commands"):
                print("commands:")
                for command in doc["commands"]:
                    print(f"  {command}")
        elif doc.get("schema") == LOOP_CHECK_SCHEMA:
            print(
                f"loop-check: {doc.get('verdict')} loop={doc.get('loop')} "
                f"recover={doc.get('recover')} apply={doc.get('apply')}"
            )
            if doc.get("reason"):
                print(f"reason: {doc['reason']}")
            check = doc.get("check") or {}
            if check:
                if check.get("cmd"):
                    print(f"status_cmd: {display_command(list(check.get('cmd') or []))}")
                detail = check.get("detail") or check.get("reason") or ""
                print(f"status: {check.get('state', 'UNKNOWN')} {detail}".rstrip())
                if check.get("returncode") is not None:
                    print(f"returncode: {check.get('returncode')}")
            recovery = doc.get("recovery") or {}
            if recovery:
                if recovery.get("command"):
                    print(f"recover_cmd: {recovery.get('command')}")
                state = "SKIP" if recovery.get("skipped") else ("OK" if recovery.get("ok") else "FAIL")
                detail = recovery.get("reason") or f"rc={recovery.get('returncode')}"
                print(f"recover: {state} {detail}")
            if doc.get("available"):
                print("available loops:")
                for loop in doc["available"]:
                    print(f"  - {loop}")
        elif doc.get("schema") == LOOP_AUDIT_SCHEMA:
            counts = doc.get("counts") or {}
            print(
                f"loop-audit: {'OK' if doc.get('ok') else 'BROKEN'} "
                f"total={counts.get('total', 0)} healthy={counts.get('healthy', 0)} "
                f"action={counts.get('action', 0)} broken={counts.get('broken', 0)}"
            )
            label = {"healthy": "HEALTHY", "action": "ACTION ", "broken": "BROKEN "}
            for loop in doc.get("loops") or []:
                tag = label.get(loop.get("bucket"), loop.get("bucket", "?"))
                detail = loop.get("detail") or loop.get("state") or ""
                print(f"  {tag} {loop.get('name')}: {detail}".rstrip())
            if doc.get("missing"):
                print("requested but not enabled: " + ", ".join(doc["missing"]))
        elif doc.get("schema") == LOOP_SCAFFOLD_SCHEMA:
            print(
                f"loop-scaffold: {'OK' if doc.get('ok') else 'REFUSE'} "
                f"apply={doc.get('apply')} loop={doc.get('loop') or '(none)'} "
                f"enabled={doc.get('enabled')}"
            )
            if doc.get("reason"):
                print(f"reason: {doc['reason']}")
            if doc.get("files"):
                print("files:")
                for file in doc["files"]:
                    marker = "exists" if file.get("exists") else "new"
                    print(f"  - {file.get('role')}: {file.get('path')} ({marker})")
            if doc.get("commands"):
                print("commands:")
                for command in doc["commands"]:
                    print(f"  {command}")
            if doc.get("followup_commands"):
                print("next:")
                for command in doc["followup_commands"]:
                    print(f"  {command}")
        elif doc.get("schema") == LOOP_SET_SCHEMA:
            print(
                f"loop-set: {'OK' if doc.get('ok') else 'REFUSE'} "
                f"apply={doc.get('apply')} changed={doc.get('changed')} "
                f"loop={doc.get('loop') or '(none)'} scope={doc.get('scope')}"
            )
            print(f"config: {doc.get('config_path') or doc.get('local_config')}")
            if doc.get("reason"):
                print(f"reason: {doc['reason']}")
            spec = doc.get("spec") or {}
            if spec:
                print(
                    f"settings: enabled={spec.get('enabled', True) is not False} "
                    f"auto_recover={bool(spec.get('auto_recover'))}"
                )
            if doc.get("commands"):
                print("commands:")
                for command in doc["commands"]:
                    print(f"  {command}")
            if doc.get("followup_commands"):
                print("next:")
                for command in doc["followup_commands"]:
                    print(f"  {command}")
            if doc.get("available") and not doc.get("ok"):
                print("available loops:")
                for loop in doc["available"]:
                    print(f"  - {loop}")
        else:
            print(json.dumps(doc, indent=2))


def add_config_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", dest="python", help="Python executable stored in local config")
    parser.add_argument("--user-home", help="user home used for account/session discovery")
    parser.add_argument("--job-dir", help="job repo/path used by supervisor status commands")
    parser.add_argument("--claude-exe", help="Claude executable path for resume watchdogs")
    parser.add_argument("--registry-dir", help="local registry directory")
    parser.add_argument("--machine-dir", help="directory where this host publishes machine snapshots")
    parser.add_argument("--machine-id", help="stable id for this machine snapshot")
    parser.add_argument("--watchdog-log-dir", help="watchdog log directory")
    parser.add_argument("--git-remote", help="git remote used for safe-ff checks")
    parser.add_argument("--target", type=int, help="target number of supervised workers")
    parser.add_argument("--session-window-h", type=float, help="session lookback window in hours")


def config_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: getattr(args, key, None)
        for key in CONFIG_OVERRIDE_KEYS
        if hasattr(args, key) and getattr(args, key, None) is not None
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One control pane for fleet agent loops.")
    parser.add_argument("--root", default=None, help="repo root (default: discovered from VERSION)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="write machine-local config")
    p_init.add_argument("--force", action="store_true")
    p_init.add_argument("--json", action="store_true")
    add_config_override_args(p_init)

    p_status = sub.add_parser("status", help="print the pane")
    p_status.add_argument("--refresh", action="store_true", help="refresh tools/_registry/sessions.json first")
    p_status.add_argument("--write", action="store_true", help="write control_pane.json and CONTROL-PANE.txt")
    p_status.add_argument("--fail-on-action", action="store_true", help="exit 1 when the pane verdict is ACTION")
    p_status.add_argument("--json", action="store_true")

    p_tick = sub.add_parser("tick", help="run one keep-alive control tick")
    p_tick.add_argument("--dry-run", action="store_true", help="show watchdog commands without invoking them")
    p_tick.add_argument("--live-resume", action="store_true", help="pass -Live to the resume watchdog")
    p_tick.add_argument("--skip-supervisor", action="store_true")
    p_tick.add_argument("--skip-resume", action="store_true")
    p_tick.add_argument("--no-write", action="store_true", help="do not write control_pane.json / CONTROL-PANE.txt")
    p_tick.add_argument("--json", action="store_true")

    p_recover = sub.add_parser("recover", help="dry-run/apply one live recovery pass")
    p_recover.add_argument("--apply", action="store_true", help="actually resume eligible sessions and configured loop recoveries")
    p_recover.add_argument("--skip-supervisor", action="store_true", help="skip supervisor watchdog during recovery")
    p_recover.add_argument("--no-write", action="store_true", help="do not write control_pane.json / CONTROL-PANE.txt")
    p_recover.add_argument("--json", action="store_true")

    p_supervisor = sub.add_parser("supervisor", help="inspect or restart the standing supervisor")
    p_supervisor.add_argument("--restart", action="store_true", help="plan or apply a supervisor restart")
    p_supervisor.add_argument("--apply", action="store_true", help="actually terminate and respawn; default is dry-run")
    p_supervisor.add_argument("--json", action="store_true")

    p_fleet = sub.add_parser("fleet", help="aggregate published machine snapshots")
    p_fleet.add_argument("--fail-on-action", action="store_true", help="exit 1 when any machine needs action")
    p_fleet.add_argument("--snapshot-only", action="store_true", help="read published snapshots only; do not overlay this host live")
    p_fleet.add_argument("--refresh-local", action="store_true", help="refresh sessions and write this host snapshot before aggregating")
    p_fleet.add_argument("--json", action="store_true")

    p_sync = sub.add_parser("sync", help="check/apply safe fast-forward sync; never plain git pull")
    p_sync.add_argument("--fetch", action="store_true", help="fetch remote branch before checking")
    p_sync.add_argument("--apply", action="store_true", help="apply safe fast-forward when every ff path is safe")
    p_sync.add_argument("--json", action="store_true")

    p_publish = sub.add_parser("publish", help="push current branch only when it is ahead of remote")
    p_publish.add_argument("--apply", action="store_true", help="actually run git push; default is dry-run")
    p_publish.add_argument("--no-fetch", action="store_true", help="skip pre-push fetch/assessment refresh")
    p_publish.add_argument("--allow-dirty", action="store_true", help="allow ordinary dirty paths while still refusing merges/unmerged paths")
    p_publish.add_argument("--json", action="store_true")

    p_commit = sub.add_parser("commit", help="stage and commit only explicitly selected files")
    p_commit.add_argument("--path", action="append", default=[], help="repo path to include (repeatable)")
    p_commit.add_argument("--dirty-group", action="append", default=[], help="include paths from a dirty-plan group")
    p_commit.add_argument("--pane-files", action="store_true", help="include the pane source/doc files")
    p_commit.add_argument("--message", "-m", default="", help="commit subject/body")
    p_commit.add_argument("--apply", action="store_true", help="actually git add and git commit")
    p_commit.add_argument("--allow-dir", action="store_true", help="allow directory pathspecs")
    p_commit.add_argument("--json", action="store_true")

    p_bootstrap = sub.add_parser("bootstrap", help="apply machine setup: local config + control tick")
    p_bootstrap.add_argument("--apply", action="store_true", help="write config/install task; default is dry-run")
    p_bootstrap.add_argument("--live-resume", action="store_true", help="install tick with resume watchdog in live mode")
    p_bootstrap.add_argument("--force-local-config", action="store_true", help="rewrite existing local config when override flags differ")
    p_bootstrap.add_argument("--json", action="store_true")
    add_config_override_args(p_bootstrap)

    p_doctor = sub.add_parser("doctor", help="check host setup")
    p_doctor.add_argument("--json", action="store_true")

    p_setup = sub.add_parser("setup-plan", help="print setup commands for this host")
    p_setup.add_argument("--json", action="store_true")

    p_loop_list = sub.add_parser("loop-list", help="list configured loops, their source, and command readiness")
    p_loop_list.add_argument("--json", action="store_true")

    p_loop_check = sub.add_parser("loop-check", help="run one configured loop check and optional recovery")
    p_loop_check.add_argument("name", help="loop name")
    p_loop_check.add_argument("--recover", action="store_true", help="show the loop's recovery command when action is needed")
    p_loop_check.add_argument("--apply", action="store_true", help="actually run recovery; requires --recover")
    p_loop_check.add_argument("--fail-on-action", action="store_true", help="exit 1 when the loop reports ACTION")
    p_loop_check.add_argument("--json", action="store_true")

    p_loop_audit = sub.add_parser(
        "loop-audit",
        help="run every enabled loop once and bucket each healthy/action/broken (exit 1 only on broken)",
    )
    p_loop_audit.add_argument("--names", help="comma-separated subset of loops to audit; default is all enabled")
    p_loop_audit.add_argument(
        "--fail-on-action",
        action="store_true",
        help="also exit 1 when any loop is in the action bucket (a loop correctly surfacing a condition)",
    )
    p_loop_audit.add_argument("--json", action="store_true")

    p_loop_scaffold = sub.add_parser("loop-scaffold", help="create starter loop files and repo catalog registration")
    p_loop_scaffold.add_argument("name", help="loop name")
    p_loop_scaffold.add_argument("--enabled", action="store_true", help="register the new loop enabled; default is disabled")
    p_loop_scaffold.add_argument("--auto-recover", action="store_true", help="mark the loop for automatic recovery when enabled")
    p_loop_scaffold.add_argument("--force", action="store_true", help="overwrite existing scaffold files or loop entry")
    p_loop_scaffold.add_argument("--apply", action="store_true", help="write scaffold files and catalog entry; default is dry-run")
    p_loop_scaffold.add_argument("--json", action="store_true")

    p_loop_set = sub.add_parser("loop-set", help="enable/disable or tune an existing loop without retyping commands")
    p_loop_set.add_argument("name", help="loop name")
    p_loop_set.add_argument(
        "--scope",
        choices=("local", "repo"),
        default="repo",
        help="write ignored machine-local config or the tracked repo loop catalog",
    )
    set_enabled_group = p_loop_set.add_mutually_exclusive_group()
    set_enabled_group.add_argument("--enable", action="store_true", dest="enabled", default=None)
    set_enabled_group.add_argument("--disable", action="store_false", dest="enabled")
    set_recover_group = p_loop_set.add_mutually_exclusive_group()
    set_recover_group.add_argument("--auto-recover", action="store_true", dest="auto_recover", default=None)
    set_recover_group.add_argument("--no-auto-recover", action="store_false", dest="auto_recover")
    p_loop_set.add_argument("--apply", action="store_true", help="write config; default is dry-run")
    p_loop_set.add_argument("--json", action="store_true")

    p_loop = sub.add_parser("loop-add", help="add or update an enabled loop in local config or the repo catalog")
    p_loop.add_argument("name", help="loop name")
    p_loop.add_argument(
        "--scope",
        choices=("local", "repo"),
        default="local",
        help="write ignored machine-local config or the tracked repo loop catalog",
    )
    p_loop.add_argument("--status-cmd", required=True, help="status command as a shell string or JSON string array")
    p_loop.add_argument("--recover-cmd", default=None, help="recovery command as a shell string or JSON string array")
    recover_group = p_loop.add_mutually_exclusive_group()
    recover_group.add_argument("--auto-recover", action="store_true", dest="auto_recover", default=None)
    recover_group.add_argument("--no-auto-recover", action="store_false", dest="auto_recover")
    p_loop.add_argument("--action", default="", help="operator-facing action text when the loop needs attention")
    p_loop.add_argument("--timeout-s", type=int, default=None)
    p_loop.add_argument("--recover-timeout-s", type=int, default=None)
    p_loop.add_argument("--cwd", default="", help="working directory for loop commands")
    p_loop.add_argument("--ok-returncode", action="append", type=int, default=[])
    p_loop.add_argument("--disabled", action="store_true", help="write the loop entry disabled")
    p_loop.add_argument("--apply", action="store_true", help="write local config; default is dry-run")
    p_loop.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    root = repo_root(Path(args.root)) if args.root else repo_root()

    if args.cmd == "init":
        base_config = load_config(root)
        overrides = config_overrides_from_args(args)
        base_config = apply_runtime_overrides(base_config, root, overrides)
        result = init_config(root, force=args.force, overrides=overrides, base_config=base_config)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(("wrote " if result["written"] else "kept ") + result["path"])
        return 0

    config = load_config(root)
    if args.cmd == "loop-add":
        try:
            status_cmd = parse_command_arg(args.status_cmd)
            recover_cmd = parse_command_arg(args.recover_cmd) if args.recover_cmd is not None else None
        except ValueError as exc:
            doc = {
                "schema": LOOP_CONFIG_SCHEMA,
                "generated_utc": iso_now(),
                "apply": args.apply,
                "ok": False,
                "changed": False,
                "scope": args.scope,
                "loop": args.name,
                "config_path": config["_paths"].get("loop_catalog" if args.scope == "repo" else "local"),
                "local_config": config["_paths"].get("local") if args.scope == "local" else None,
                "reason": str(exc),
                "spec": {},
            }
            print_doc(doc, args.json)
            return 1
        doc = loop_config_plan(
            config,
            loop_name=args.name,
            scope=args.scope,
            status_cmd=status_cmd,
            recover_cmd=recover_cmd,
            auto_recover=args.auto_recover,
            action=args.action,
            timeout_s=args.timeout_s,
            recover_timeout_s=args.recover_timeout_s,
            cwd=args.cwd,
            ok_returncodes=args.ok_returncode,
            enabled=not args.disabled,
            apply=args.apply,
        )
        print_doc(doc, args.json)
        return 0 if doc.get("ok") else 1
    if args.cmd == "loop-list":
        doc = loop_list(config)
        print_doc(doc, args.json)
        return 0 if doc.get("ok") else 1
    if args.cmd == "loop-check":
        if args.apply and not args.recover:
            parser.error("loop-check --apply requires --recover")
        doc = loop_check_plan(config, args.name, recover=args.recover, apply=args.apply)
        print_doc(doc, args.json)
        if not doc.get("ok"):
            return 1
        return 1 if args.fail_on_action and doc.get("needs_action") else 0
    if args.cmd == "loop-audit":
        names = [n.strip() for n in (args.names or "").split(",") if n.strip()] or None
        doc = loop_audit(config, names=names)
        print_doc(doc, args.json)
        if not doc.get("ok"):
            return 1
        return 1 if args.fail_on_action and doc.get("counts", {}).get("action") else 0
    if args.cmd == "loop-scaffold":
        doc = loop_scaffold_plan(
            config,
            args.name,
            enabled=args.enabled,
            auto_recover=args.auto_recover,
            force=args.force,
            apply=args.apply,
        )
        print_doc(doc, args.json)
        return 0 if doc.get("ok") else 1
    if args.cmd == "loop-set":
        doc = loop_set_plan(
            config,
            args.name,
            scope=args.scope,
            enabled=args.enabled,
            auto_recover=args.auto_recover,
            apply=args.apply,
        )
        print_doc(doc, args.json)
        return 0 if doc.get("ok") else 1
    if args.cmd == "bootstrap":
        overrides = config_overrides_from_args(args)
        config = apply_runtime_overrides(config, root, overrides)
    else:
        overrides = {}
    if args.cmd == "status":
        status = collect_status(config, refresh=args.refresh)
        if args.write:
            status["written"] = write_status(status, config)
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(pane_text(status))
            if args.write:
                print(f"written: {status['written']['json']} ; {status['written']['text']}")
        return 1 if args.fail_on_action and status["verdict"] != "OK" else 0
    if args.cmd == "tick":
        doc = control_tick(
            config,
            dry_run=args.dry_run,
            live_resume=args.live_resume,
            skip_supervisor=args.skip_supervisor,
            skip_resume=args.skip_resume,
            write=not args.no_write,
        )
        print_doc(doc, args.json)
        return 0 if doc["ok"] else 1
    if args.cmd == "recover":
        doc = recover_plan(
            config,
            apply=args.apply,
            skip_supervisor=args.skip_supervisor,
            write=not args.no_write,
        )
        print_doc(doc, args.json)
        return 0 if doc.get("ok") else 1
    if args.cmd == "supervisor":
        doc = supervisor_plan(config, restart=args.restart, apply=args.apply)
        print_doc(doc, args.json)
        return 0 if doc.get("ok") else 1
    if args.cmd == "fleet":
        if args.snapshot_only and args.refresh_local:
            parser.error("fleet --refresh-local cannot be combined with --snapshot-only")
        doc = fleet_view(
            config,
            include_live_local=not args.snapshot_only,
            refresh_live_local=args.refresh_local,
            write_live_local=args.refresh_local,
        )
        print_doc(doc, args.json)
        return 1 if args.fail_on_action and doc["verdict"] != "OK" else 0
    if args.cmd == "sync":
        doc = sync_plan(config, fetch=args.fetch, apply=args.apply)
        print_doc(doc, args.json)
        return 0 if doc.get("ok") else 1
    if args.cmd == "publish":
        doc = publish_plan(config, fetch=not args.no_fetch, apply=args.apply, allow_dirty=args.allow_dirty)
        print_doc(doc, args.json)
        return 0 if doc.get("ok") else 1
    if args.cmd == "commit":
        paths = list(args.path or [])
        dirty_selection = None
        if args.dirty_group:
            dirty_selection = dirty_group_selection(config, list(args.dirty_group))
            if not dirty_selection.get("ok"):
                doc = {
                    "schema": COMMIT_SCHEMA,
                    "generated_utc": iso_now(),
                    "ok": False,
                    "applied": False,
                    "reason": "unknown dirty group(s): " + ", ".join(dirty_selection.get("missing") or []),
                    "dirty_groups": dirty_selection,
                }
                print_doc(doc, args.json)
                return 1
            paths.extend(dirty_selection.get("paths") or [])
        if args.pane_files:
            paths.extend(PANE_SOURCE_PATHS)
        try:
            doc = commit_plan(
                config,
                paths=paths,
                message=args.message,
                apply=args.apply,
                allow_dir=args.allow_dir,
            )
            if dirty_selection is not None:
                doc["dirty_groups"] = dirty_selection
        except ValueError as exc:
            doc = {
                "schema": COMMIT_SCHEMA,
                "generated_utc": iso_now(),
                "ok": False,
                "applied": False,
                "reason": str(exc),
            }
        print_doc(doc, args.json)
        return 0 if doc.get("ok") else 1
    if args.cmd == "bootstrap":
        doc = bootstrap(
            config,
            apply=args.apply,
            live_resume=args.live_resume,
            init_overrides=overrides,
            force_local_config=args.force_local_config,
        )
        print_doc(doc, args.json)
        return 0 if doc.get("ok") else 1
    if args.cmd == "doctor":
        doc = doctor(config)
        print_doc(doc, args.json)
        return 0 if doc["ok"] else 1
    if args.cmd == "setup-plan":
        print_doc(setup_plan(config), args.json)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
