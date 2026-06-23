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
    install        drop a `fleet` command on PATH
    uninstall      remove the installed wrapper
    help           this list
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

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
        print("try:    fleet            # the live fleet view")
        print("        fleet status     # one-shot snapshot")
        if not result.get("on_path"):
            print(f"note:   {result['bin_dir']} is not on PATH yet — add it to use `fleet` from anywhere")
    return 0


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
