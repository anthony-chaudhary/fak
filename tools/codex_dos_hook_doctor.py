"""Inspect or repair Codex DOS hook manifests for launch-safe native-hook wiring.

The recent-session DOS audit can prove that Codex hooks are routed through the
Python CLI path, but it should not rewrite the user's Codex cache implicitly.
This helper is the explicit repair surface: dry-run by default, `--apply` to
rewrite cached DOS hook manifests to call the bundled POSIX `dos-hook` launcher
via Git Bash, with Python preserved as the delegate fallback.

Reports are privacy-preserving: they include manifest-relative paths and counts,
not hook command bodies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


SCHEMA = "fak-codex-dos-hook-doctor/1"
PYTHON_HOOK_RE = re.compile(
    r"-m\s+dos\.cli\s+hook\s+(?P<verb>[a-z0-9-]+)(?P<flags>.*?)(?:;|$)",
    re.IGNORECASE,
)
NATIVE_HOOK_RE = re.compile(
    r"dos-hook(?:\.ps1)?(?:['\"])?\s+(?P<verb>[a-z0-9-]+)(?P<flags>.*?)(?:;|$)",
    re.IGNORECASE,
)
PS_NATIVE_HOOK_RE = re.compile(
    r"&\s+\$dosHook\s+(?P<verb>[a-z0-9-]+)(?P<flags>.*?)(?:;|$)",
    re.IGNORECASE,
)


def default_codex_home(env: dict[str, str] = os.environ) -> Path:
    configured = env.get("CODEX_HOME")
    if configured:
        return Path(configured)
    return Path.home() / ".codex"


def home_relpath(path: Path, home: Path) -> str:
    try:
        return path.resolve().relative_to(home.resolve()).as_posix()
    except (OSError, ValueError):
        return path.name


def discover_manifests(home: Path) -> list[Path]:
    plugin_root = home / "plugins" / "cache" / "dos" / "dos-kernel"
    manifests = sorted(plugin_root.glob("*/hooks/hooks.json"))
    if manifests:
        return manifests
    cache_root = home / "plugins" / "cache"
    if not cache_root.exists():
        return []
    return sorted(
        p for p in cache_root.glob("**/hooks/hooks.json")
        if "dos-kernel" in p.as_posix()
    )


def hook_command_mode(command: str) -> str:
    lower = command.lower()
    if "dos-hook.ps1" in lower:
        return "powershell_native_launcher"
    if "dos-hook" in lower:
        return "native_launcher"
    if "dos.cli hook" in lower:
        return "python_cli"
    return "other"


def iter_hook_nodes(manifest: dict[str, Any]):
    hooks = manifest.get("hooks")
    if not isinstance(hooks, dict):
        return
    for event, entries in hooks.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            inner = entry.get("hooks")
            if not isinstance(inner, list):
                continue
            for hook in inner:
                if isinstance(hook, dict):
                    yield str(event), hook


def parse_python_hook(command: str) -> tuple[str, str] | None:
    match = PYTHON_HOOK_RE.search(command)
    if match is None:
        return None
    verb = match.group("verb").strip()
    flags = " ".join(match.group("flags").split())
    return verb, flags


def parse_native_launcher(command: str) -> tuple[str, str] | None:
    match = PS_NATIVE_HOOK_RE.search(command) or NATIVE_HOOK_RE.search(command)
    if match is None:
        return None
    verb = match.group("verb").strip()
    flags = " ".join(match.group("flags").split())
    return verb, flags


def bash_single_quoted(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def native_command(verb: str, flags: str) -> str:
    hook_args = f"{verb} {flags}".strip()
    quoted_hook_args = " ".join(bash_single_quoted(arg) for arg in hook_args.split())
    return (
        'root="${CLAUDE_PLUGIN_ROOT:-${CODEX_PLUGIN_ROOT:-}}"; '
        'if [ -n "$root" ]; then '
        f'"$root/bin/dos-hook" {quoted_hook_args} 2>/dev/null; '
        "rc=$?; [ \"$rc\" -eq 0 ] && exit 0; "
        "fi; "
        f"python -m dos.cli hook {quoted_hook_args} 2>/dev/null || "
        f"python3 -m dos.cli hook {quoted_hook_args} 2>/dev/null || "
        f"powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "
        f"{bash_single_quoted('python -m dos.cli hook ' + hook_args)} 2>/dev/null || true"
    )


def inspect_manifest(path: Path, home: Path, *, apply: bool) -> dict[str, Any]:
    manifest_rel = home_relpath(path, home)
    launcher = path.parent.parent / "bin" / "dos-hook"
    if not launcher.exists():
        return {
            "manifest": manifest_rel,
            "status": "UNKNOWN",
            "reason": "native POSIX launcher missing beside cached DOS hook manifest",
            "command_modes": {},
            "codex_command_modes": {},
            "projected_command_modes": {},
            "projected_codex_command_modes": {},
            "replacements_available": 0,
            "codex_replacements_available": 0,
            "unrepairable_python_cli_hooks": 0,
            "codex_unrepairable_python_cli_hooks": 0,
            "applied": False,
        }

    try:
        original = path.read_text(encoding="utf-8")
        manifest = json.loads(original)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "manifest": manifest_rel,
            "status": "UNKNOWN",
            "reason": f"cannot read hook manifest: {type(exc).__name__}",
            "command_modes": {},
            "codex_command_modes": {},
            "projected_command_modes": {},
            "projected_codex_command_modes": {},
            "replacements_available": 0,
            "codex_replacements_available": 0,
            "unrepairable_python_cli_hooks": 0,
            "codex_unrepairable_python_cli_hooks": 0,
            "applied": False,
        }
    if not isinstance(manifest, dict):
        return {
            "manifest": manifest_rel,
            "status": "UNKNOWN",
            "reason": "hook manifest root is not an object",
            "command_modes": {},
            "codex_command_modes": {},
            "projected_command_modes": {},
            "projected_codex_command_modes": {},
            "replacements_available": 0,
            "codex_replacements_available": 0,
            "unrepairable_python_cli_hooks": 0,
            "codex_unrepairable_python_cli_hooks": 0,
            "applied": False,
        }

    command_modes: Counter[str] = Counter()
    codex_command_modes: Counter[str] = Counter()
    projected_command_modes: Counter[str] = Counter()
    projected_codex_command_modes: Counter[str] = Counter()
    replacements = 0
    codex_replacements = 0
    unrepairable_python = 0
    codex_unrepairable_python = 0
    for _event, hook in iter_hook_nodes(manifest):
        command = hook.get("command")
        if not isinstance(command, str) or not command:
            continue
        mode = hook_command_mode(command)
        command_modes[mode] += 1
        is_codex = "--dialect codex" in command.lower()
        if is_codex:
            codex_command_modes[mode] += 1
        projected_mode = mode
        if mode not in {"python_cli", "powershell_native_launcher"}:
            projected_command_modes[projected_mode] += 1
            if is_codex:
                projected_codex_command_modes[projected_mode] += 1
            continue
        parsed = parse_python_hook(command) if mode == "python_cli" else parse_native_launcher(command)
        if parsed is None:
            unrepairable_python += 1
            if is_codex:
                codex_unrepairable_python += 1
            projected_command_modes[projected_mode] += 1
            if is_codex:
                projected_codex_command_modes[projected_mode] += 1
            continue
        verb, flags = parsed
        replacements += 1
        if is_codex:
            codex_replacements += 1
        projected_mode = "native_launcher"
        projected_command_modes[projected_mode] += 1
        if is_codex:
            projected_codex_command_modes[projected_mode] += 1
        if apply:
            hook["shell"] = "bash"
            hook["command"] = native_command(verb, flags)

    applied = False
    backup_rel = None
    if apply and replacements:
        backup = path.with_name(path.name + ".before-native-dos-hook.bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        backup_rel = home_relpath(backup, home)
        applied = True

    if replacements:
        status = "CHANGED" if applied else "WARN"
        reason = "hook commands can be routed through the bundled POSIX native launcher"
    elif int(codex_command_modes.get("native_launcher") or 0):
        status = "PASS"
        reason = "Codex hook commands already use the native launcher"
    else:
        status = "UNKNOWN"
        reason = "no Python hook commands or Codex native hook commands were found"

    return {
        "manifest": manifest_rel,
        "status": status,
        "reason": reason,
        "command_modes": {k: command_modes[k] for k in sorted(command_modes)},
        "codex_command_modes": {k: codex_command_modes[k] for k in sorted(codex_command_modes)},
        "projected_command_modes": {k: projected_command_modes[k] for k in sorted(projected_command_modes)},
        "projected_codex_command_modes": {k: projected_codex_command_modes[k] for k in sorted(projected_codex_command_modes)},
        "replacements_available": replacements,
        "codex_replacements_available": codex_replacements,
        "unrepairable_python_cli_hooks": unrepairable_python,
        "codex_unrepairable_python_cli_hooks": codex_unrepairable_python,
        "applied": applied,
        "backup": backup_rel,
    }


def build_report(home: Path, *, apply: bool) -> dict[str, Any]:
    manifests = discover_manifests(home)
    manifest_reports = [inspect_manifest(path, home, apply=apply) for path in manifests]
    command_modes: Counter[str] = Counter()
    codex_command_modes: Counter[str] = Counter()
    projected_command_modes: Counter[str] = Counter()
    projected_codex_command_modes: Counter[str] = Counter()
    replacements = 0
    codex_replacements = 0
    unrepairable_python = 0
    codex_unrepairable_python = 0
    applied = 0
    for report in manifest_reports:
        command_modes.update(report.get("command_modes") or {})
        codex_command_modes.update(report.get("codex_command_modes") or {})
        projected_command_modes.update(report.get("projected_command_modes") or {})
        projected_codex_command_modes.update(report.get("projected_codex_command_modes") or {})
        replacements += int(report.get("replacements_available") or 0)
        codex_replacements += int(report.get("codex_replacements_available") or 0)
        unrepairable_python += int(report.get("unrepairable_python_cli_hooks") or 0)
        codex_unrepairable_python += int(report.get("codex_unrepairable_python_cli_hooks") or 0)
        if report.get("applied"):
            applied += 1

    if not manifests:
        status = "UNKNOWN"
    elif applied:
        status = "CHANGED"
    elif replacements:
        status = "WARN"
    elif int(codex_command_modes.get("native_launcher") or 0):
        status = "PASS"
    else:
        status = "UNKNOWN"

    return {
        "schema": SCHEMA,
        "status": status,
        "applied": bool(applied),
        "codex_home": home.name,
        "manifest_count": len(manifests),
        "manifests": manifest_reports,
        "summary": {
            "command_modes": {k: command_modes[k] for k in sorted(command_modes)},
            "codex_command_modes": {k: codex_command_modes[k] for k in sorted(codex_command_modes)},
            "projected_command_modes": {k: projected_command_modes[k] for k in sorted(projected_command_modes)},
            "projected_codex_command_modes": {k: projected_codex_command_modes[k] for k in sorted(projected_codex_command_modes)},
            "replacements_available": replacements,
            "codex_replacements_available": codex_replacements,
            "unrepairable_python_cli_hooks": unrepairable_python,
            "codex_unrepairable_python_cli_hooks": codex_unrepairable_python,
            "applied_manifests": applied,
        },
        "privacy": {
            "copied_fields": ["manifest-relative paths", "hook mode counts", "replacement counts"],
            "dropped": ["hook command bodies", "Codex prompts", "tool arguments", "tool results"],
        },
    }


def render(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        f"codex DOS hook doctor: {report.get('status')}",
        f"  manifests: {report.get('manifest_count')}",
        f"  modes: {json.dumps(summary.get('command_modes') or {}, sort_keys=True)}",
        f"  codex modes: {json.dumps(summary.get('codex_command_modes') or {}, sort_keys=True)}",
        f"  projected codex modes after apply: {json.dumps(summary.get('projected_codex_command_modes') or {}, sort_keys=True)}",
        f"  replacements: {summary.get('replacements_available')} total, {summary.get('codex_replacements_available')} Codex",
        f"  unrepairable python hooks: {summary.get('unrepairable_python_cli_hooks')} total, {summary.get('codex_unrepairable_python_cli_hooks')} Codex",
        f"  applied manifests: {summary.get('applied_manifests')}",
    ]
    if report.get("status") == "WARN":
        lines.append("  next: rerun with --apply to route cached DOS hooks through the native launcher")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--codex-home", type=Path, default=default_codex_home())
    p.add_argument("--apply", action="store_true", help="rewrite cached manifests; dry-run is the default")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    report = build_report(args.codex_home.resolve(), apply=args.apply)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))
    if report["status"] in {"PASS", "CHANGED"}:
        return 0
    if report["status"] == "WARN":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
