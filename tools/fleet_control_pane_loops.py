#!/usr/bin/env python3
"""Loop catalog configuration, scaffolding, execution, check, and audit helpers for fleet_control_pane."""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fleet_control_pane_core import (
    ACTION_LOOP_VALUES,
    CONFIG_SCHEMA,
    HEALTHY_LOOP_VALUES,
    LOOP_AUDIT_SCHEMA,
    LOOP_AUDIT_SELF_NAME,
    LOOP_BROKEN_STATES,
    LOOP_CATALOG_SCHEMA,
    LOOP_CHECK_SCHEMA,
    LOOP_CONFIG_SCHEMA,
    LOOP_LIST_SCHEMA,
    LOOP_SCAFFOLD_SCHEMA,
    LOOP_SET_SCHEMA,
    command_exists,
    command_readiness,
    config_paths,
    display_command,
    display_path,
    env_for_tools,
    expand_cmd,
    expand_value,
    iso_now,
    pane_attr,
    read_json,
    read_json_from_text,
    resolve_path,
    run,
    step_id_slug,
    tracked_default_config,
    write_json,
    write_text_atomic,
)


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
    exists_fn = pane_attr("command_exists", command_exists)
    if not exists_fn(cmd[0]):
        return {**base, "state": "UNAVAILABLE", "reason": f"command not found: {cmd[0]}"}
    timeout = int(spec.get("timeout_s") or 20)
    run_fn = pane_attr("run", run)
    try:
        proc = run_fn(cmd, loop_cwd(spec, config), env=env_for_tools(config), timeout=timeout)
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
    exists_fn = pane_attr("command_exists", command_exists)
    if not exists_fn(recover_cmd[0]):
        return {**action, "ok": False, "skipped": True, "reason": f"command not found: {recover_cmd[0]}"}
    timeout = int((spec or {}).get("recover_timeout_s") or (spec or {}).get("timeout_s") or 90)
    run_fn = pane_attr("run", run)
    try:
        proc = run_fn(recover_cmd, loop_cwd(spec or {}, config), env=env_for_tools(config), timeout=timeout)
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


def loop_audit_detail(check: dict[str, Any]) -> str:
    """One clean line summarizing a loop check for the audit row."""
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
    """Run every enabled loop once and bucket each as healthy / action / broken."""
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
