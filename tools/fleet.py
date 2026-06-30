#!/usr/bin/env python3
r"""fleet — the one operator console for the agent fleet (and it just works).

Typing `fleet` shows the fleet. The fleet_* tools were each useful but each had its
own name and flags; this is the thin front door over them so an operator has one verb
to remember. `fleet` with no args is the live session/account watchdog (the
session-health peer of `dos top`); the rest are one-shot views and the install hook.

Install it once so it is always on, from anywhere:

    python tools/fleet.py install      # drops a `fleet` wrapper on PATH
    fleet                              # ... then this just works

Verbs (everything after the verb is passed straight through to the tool):
    (none) / top   live session/account watchdog          tools/fleet_top.py
    status         one-shot snapshot of the same view      tools/fleet_top.py --once
    json           machine snapshot                        tools/fleet_top.py --json
    sessions       per-session disposition table           tools/fleet_sessions.py summary
    resume         account-correct resume commands         tools/fleet_sessions.py resume
    accounts       worker-account availability             tools/fleet_accounts.py
    pane           the full control pane                    tools/fleet_control_pane.py status
    install        drop a `fleet` command on PATH (and add the bin dir to PATH; --no-path opts out)
    uninstall      remove the installed wrapper
    help           this list
"""
from __future__ import annotations

import argparse
import os
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
from pathlib import Path
install_no_window_subprocess_defaults(subprocess)

# Verb -> (tool script, fixed leading args). Everything the operator types after the
# verb is appended, so `fleet status --window 24` becomes fleet_top.py --once --window 24.
VERBS: dict[str, tuple[str, list[str]]] = {
    "top": ("fleet_top.py", []),
    "status": ("fleet_top.py", ["--once"]),
    "json": ("fleet_top.py", ["--json"]),
    "sessions": ("fleet_sessions.py", ["summary"]),
    "resume": ("fleet_sessions.py", ["resume"]),
    "accounts": ("fleet_accounts.py", []),
    "pane": ("fleet_control_pane.py", ["status"]),
}
HELP_VERBS = {"help", "-h", "--help"}


def repo_root() -> Path:
    """The repo whose tools we drive. The installed wrapper pins FLEET_ROOT; otherwise
    we are running from within the clone, so the tools live next to this file."""
    env = os.environ.get("FLEET_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def route(argv: list[str]) -> dict:
    """Pure arg router: argv -> a dispatch intent. No subprocess, no disk — so the
    mapping is unit-testable and the runtime only has to act on the verdict."""
    argv = list(argv)
    if not argv:
        verb, rest = "top", []
    else:
        verb, rest = argv[0], argv[1:]
    if verb in HELP_VERBS:
        return {"kind": "help"}
    if verb in ("install", "uninstall"):
        return {"kind": verb, "argv": rest}
    if verb in VERBS:
        tool, lead = VERBS[verb]
        return {"kind": "exec", "tool": tool, "argv": lead + rest}
    # A bare flag (e.g. `fleet --once`) is shorthand for the default view.
    if verb.startswith("-"):
        return {"kind": "exec", "tool": "fleet_top.py", "argv": argv}
    return {"kind": "unknown", "verb": verb}


def exec_command(root: Path, tool: str, argv: list[str]) -> list[str]:
    return [sys.executable, str(root / "tools" / tool), *argv]


def _norm_path_entry(value: str, *, windows: bool) -> str:
    value = value.strip().strip('"').rstrip("\\/")
    return value.lower() if windows else value


def _path_has(path_value: str, bin_dir: Path, *, sep: str, windows: bool) -> bool:
    """Is bin_dir already one of the entries in a PATH-shaped string? Pure, so the
    'already there, do nothing' decision is unit-testable without touching the env."""
    target = _norm_path_entry(str(bin_dir), windows=windows)
    return any(
        _norm_path_entry(part, windows=windows) == target
        for part in (path_value or "").split(sep)
        if part.strip()
    )


def _ps(expr: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", expr],
            capture_output=True, text=True, timeout=25,
        )
    except (OSError, subprocess.SubprocessError):
        return False, ""
    return proc.returncode == 0, proc.stdout.strip()


def _ensure_on_path_windows(bin_dir: Path, *, apply: bool, in_process: bool) -> dict:
    # Check the PERSISTENT user+machine PATH, not this process's PATH: a Git Bash (or
    # other) launcher can inject ~/.local/bin into the process env without it being in
    # the registry, which is what the operator's PowerShell/cmd shells actually read.
    ok_u, user = _ps("[Environment]::GetEnvironmentVariable('Path','User')")
    if not ok_u:
        return {"ok": False, "detail": "could not read the user PATH"}
    ok_m, machine = _ps("[Environment]::GetEnvironmentVariable('Path','Machine')")
    persistent = ";".join(p for p in ((machine if ok_m else ""), user) if p)
    if _path_has(persistent, bin_dir, sep=";", windows=True):
        return {"ok": True, "already": True, "changed": False, "needs_new_shell": not in_process}
    if not apply:
        return {"ok": True, "already": False, "changed": False, "would": True}
    new = (user.rstrip(";") + ";" + str(bin_dir)) if (user and user.strip()) else str(bin_dir)
    safe = new.replace("'", "''")
    set_ok, _ = _ps(f"[Environment]::SetEnvironmentVariable('Path','{safe}','User')")
    return {"ok": set_ok, "already": False, "changed": set_ok, "needs_new_shell": True,
            "detail": "" if set_ok else "SetEnvironmentVariable failed"}


def _ensure_on_path_posix(bin_dir: Path, *, apply: bool) -> dict:
    profile = Path.home() / ".profile"
    existing = profile.read_text(encoding="utf-8", errors="replace") if profile.exists() else ""
    if str(bin_dir) in existing:
        return {"ok": True, "already": True, "changed": False, "needs_new_shell": True}
    if not apply:
        return {"ok": True, "already": False, "changed": False, "would": True}
    line = f'export PATH="{bin_dir}:$PATH"  # added by fleet install'
    prefix = "" if (not existing or existing.endswith("\n")) else "\n"
    with profile.open("a", encoding="utf-8") as fh:
        fh.write(prefix + line + "\n")
    return {"ok": True, "already": False, "changed": True, "needs_new_shell": True,
            "detail": f"appended to {profile}"}


def ensure_on_path(bin_dir: Path, *, platform: str | None = None, apply: bool = True) -> dict:
    """Make bin_dir reachable as PATH so a bare `fleet` resolves from any shell.

    A no-op if it is already visible to this process (and therefore to the shells it
    spawns). Otherwise it appends to the persistent user PATH (Windows) or ~/.profile
    (POSIX); the change is picked up by the NEXT shell, never the current process."""
    platform = platform or sys.platform
    windows = platform.startswith("win")
    bin_dir = Path(bin_dir)
    in_process = _path_has(os.environ.get("PATH", ""), bin_dir, sep=os.pathsep, windows=windows)
    if windows:
        # The persistent registry PATH is the source of truth on Windows, so always ask it.
        return _ensure_on_path_windows(bin_dir, apply=apply, in_process=in_process)
    if in_process:
        return {"ok": True, "already": True, "changed": False, "needs_new_shell": False}
    return _ensure_on_path_posix(bin_dir, apply=apply)


def _run_tool(root: Path, tool: str, argv: list[str]) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        return subprocess.run(exec_command(root, tool, argv), cwd=str(root), env=env).returncode
    except KeyboardInterrupt:
        return 0  # the live view exits cleanly on Ctrl-C; so do we.


def _install(root: Path, argv: list[str], *, uninstall: bool) -> int:
    sys.path.insert(0, str(root / "tools"))
    import install_agent_command as iac  # noqa: E402  (sibling tool, reused verbatim)

    ap = argparse.ArgumentParser(prog="fleet install", add_help=False)
    ap.add_argument("--system", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bin-dir", type=Path, default=None)
    ap.add_argument("--no-path", action="store_true", help="install the wrapper but leave PATH alone")
    opts, _ = ap.parse_known_args(argv)

    common = dict(name="fleet", bin_dir=opts.bin_dir, system=opts.system,
                  force=opts.force, dry_run=opts.dry_run)
    if uninstall:
        result = iac.uninstall(**common)
    else:
        result = iac.install(launcher=root / "tools" / "fleet.py", **common)
    if not result.get("ok"):
        print(f"fleet {'uninstall' if uninstall else 'install'}: {result.get('reason')}",
              file=sys.stderr)
        return 1
    for row in result.get("installed") or result.get("removed") or []:
        print(f"{row['action']}: {row['path']}")
    if result.get("installed"):
        if not opts.no_path:
            _report_path(ensure_on_path(Path(result["bin_dir"]), apply=not opts.dry_run),
                         Path(result["bin_dir"]))
        print("try:    fleet            # the live fleet view")
        print("        fleet status     # one-shot snapshot")
    return 0


def _report_path(res: dict, bin_dir: Path) -> None:
    if res.get("already"):
        tail = "  (open a new terminal)" if res.get("needs_new_shell") else ""
        print(f"PATH:   {bin_dir} already on PATH{tail}")
    elif res.get("would"):
        print(f"PATH:   would add {bin_dir} to your user PATH")
    elif res.get("changed"):
        print(f"PATH:   added {bin_dir} to your user PATH — open a NEW terminal, then `fleet` works anywhere")
    else:
        print(f"PATH:   could not update automatically ({res.get('detail') or 'unknown'}); "
              f"add {bin_dir} to PATH yourself")


def _print_help() -> None:
    print(__doc__.strip())


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = repo_root()
    intent = route(argv)
    kind = intent["kind"]
    if kind == "help":
        _print_help()
        return 0
    if kind in ("install", "uninstall"):
        return _install(root, intent["argv"], uninstall=(kind == "uninstall"))
    if kind == "exec":
        return _run_tool(root, intent["tool"], intent["argv"])
    print(f"fleet: unknown verb '{intent.get('verb')}'\n", file=sys.stderr)
    _print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
