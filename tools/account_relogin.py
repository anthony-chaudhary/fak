#!/usr/bin/env python3
r"""account_relogin -- fix Claude config dirs logged into the WRONG account.

The fleet roster keys an account on its dir name, but the account actually *logged into*
a dir is whatever the last ``/login`` flow signed in as. When several dirs get logged into
the same account by mistake (e.g. ``.claude-gem5/gem7/c10-netra`` all signed into one
account), the roster sees one account masquerading as several workers. This tool fixes the
mismatch the only way it can be fixed -- per dir -- while automating everything around the
one interactive step:

  1. read each dir's ACTUAL login via ``claude auth status --json`` (the CLI's own truth);
  2. compare to the operator-supplied INTENDED email for that dir;
  3. for each mismatch: ``claude auth logout`` (clears the wrong account), then hand the
     operator the exact ``claude auth login`` command (interactive -- a browser/device flow
     this tool cannot drive headlessly);
  4. after the operator logs in, VERIFY ``auth status`` now reports the intended email.

Nothing destructive runs without ``--apply``; ``--apply`` only logs OUT (reversible by
logging back in) and never deletes a dir. The intended map is operator-supplied -- the tool
never GUESSES which account a dir should hold (the dir-name heuristic is unreliable: a
dir whose tag differs from its intended login must not be auto-"corrected" on a guess).

Usage:
  # show current login vs intended for every dir in the map (dry-run)
  python tools/account_relogin.py status --map gem5=gem5@x.ai,gem7=gem7@x.ai

  # log out the mismatched dirs and print the per-dir login commands
  python tools/account_relogin.py fix --map gem5=gem5@x.ai,gem7=gem7@x.ai --apply

  # after logging in, confirm each dir now holds its intended account
  python tools/account_relogin.py verify --map gem5=gem5@x.ai,gem7=gem7@x.ai

The map may also be a JSON file: --map-file dir_owners.json  ({"gem5": "gem5@x.ai", ...}).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_accounts  # noqa: E402

SCHEMA = "fleet.account-relogin.v1"


def default_claude_exe() -> str:
    return (os.environ.get("FLEET_CLAUDE_EXE")
            or shutil.which("claude") or shutil.which("claude.exe")
            or os.path.join(os.path.expanduser("~"), ".local", "bin", "claude.exe"))


def _run(argv: list[str], *, config_dir: str, timeout: float = 30.0,
         runner: Callable[..., tuple[int, str, str]] | None = None,
         ) -> tuple[int, str, str]:
    if runner is not None:
        return runner(argv, config_dir=config_dir, timeout=timeout)
    env = os.environ.copy()
    env["CLAUDE_CONFIG_DIR"] = config_dir
    try:
        p = subprocess.run(argv, env=env, input="", capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except OSError as exc:
        return 127, "", str(exc)
    return p.returncode, p.stdout or "", p.stderr or ""


def auth_status(config_dir: str, *, claude_exe: str | None = None,
                runner: Callable[..., tuple[int, str, str]] | None = None) -> dict[str, Any]:
    """Parse ``claude auth status --json`` for one config dir.

    Returns {logged_in, email, org_id, subscription, raw}. A dir not logged in (or any
    error) yields logged_in=False with email="" -- never raises.
    """
    exe = claude_exe or default_claude_exe()
    code, out, err = _run([exe, "auth", "status", "--json"],
                          config_dir=config_dir, runner=runner)
    text = (out or "").strip() or (err or "").strip()
    doc: dict[str, Any] = {}
    try:
        doc = json.loads(text)
    except ValueError:
        doc = {}
    return {
        "logged_in": bool(doc.get("loggedIn")),
        "email": str(doc.get("email") or ""),
        "org_id": str(doc.get("orgId") or ""),
        "subscription": str(doc.get("subscriptionType") or ""),
        "exit_code": code,
        "raw": text[:300],
    }


def auth_logout(config_dir: str, *, claude_exe: str | None = None,
                runner: Callable[..., tuple[int, str, str]] | None = None) -> dict[str, Any]:
    exe = claude_exe or default_claude_exe()
    code, out, err = _run([exe, "auth", "logout"], config_dir=config_dir, runner=runner)
    return {"ok": code == 0, "exit_code": code,
            "detail": ((out or "") + (err or "")).strip()[:200]}


def login_command(config_dir: str, *, claude_exe: str | None = None) -> str:
    """The exact PowerShell line the operator runs to log a dir into the right account.

    Interactive (browser/device flow) -- this tool cannot complete it headlessly, so it
    hands the command over instead of pretending to run it."""
    exe = claude_exe or default_claude_exe()
    return f"$env:CLAUDE_CONFIG_DIR='{config_dir}'; & '{exe}' auth login"


def parse_map(spec: str = "", map_file: str = "") -> dict[str, str]:
    """Parse a dir->intended-email map from a CSV spec (tag=email,tag=email) or a JSON file."""
    out: dict[str, str] = {}
    if map_file:
        try:
            with open(map_file, encoding="utf-8") as f:
                doc = json.load(f)
            if isinstance(doc, dict):
                out.update({str(k): str(v) for k, v in doc.items()})
        except (OSError, ValueError) as exc:
            raise SystemExit(f"account_relogin: cannot read --map-file {map_file}: {exc}")
    for pair in (spec or "").split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        tag, email = pair.split("=", 1)
        out[tag.strip()] = email.strip()
    return out


def _dir_for_tag(tag: str, rows: list[dict[str, Any]]) -> str | None:
    for r in rows:
        if r.get("product") == "claude" and r.get("kind") == "worker" \
                and str(r.get("tag")) == tag:
            return str(r.get("dir") or "")
    return None


def assess(intended: dict[str, str], rows: list[dict[str, Any]], *,
           claude_exe: str | None = None,
           runner: Callable[..., tuple[int, str, str]] | None = None,
           ) -> list[dict[str, Any]]:
    """For each tag in the intended map, compare current login to intended."""
    out = []
    for tag, want_email in intended.items():
        acct_dir = _dir_for_tag(tag, rows)
        if not acct_dir:
            out.append({"tag": tag, "dir": None, "intended": want_email,
                        "current": "", "logged_in": False, "match": False,
                        "note": "no such Claude worker dir"})
            continue
        st = auth_status(acct_dir, claude_exe=claude_exe, runner=runner)
        match = bool(st["email"]) and st["email"].lower() == want_email.lower()
        out.append({
            "tag": tag, "dir": acct_dir, "intended": want_email,
            "current": st["email"], "logged_in": st["logged_in"],
            "subscription": st["subscription"], "match": match,
            "note": "" if match else ("not logged in" if not st["logged_in"]
                                      else f"wrong account ({st['email'] or 'unknown'})"),
        })
    return out


def fix(intended: dict[str, str], rows: list[dict[str, Any]], *, apply: bool = False,
        claude_exe: str | None = None,
        runner: Callable[..., tuple[int, str, str]] | None = None) -> dict[str, Any]:
    """Log out the mismatched dirs (only with apply=True) and emit the per-dir login steps."""
    assessment = assess(intended, rows, claude_exe=claude_exe, runner=runner)
    mismatched = [a for a in assessment if a["dir"] and not a["match"]]
    steps = []
    for a in mismatched:
        step = {"tag": a["tag"], "dir": a["dir"], "intended": a["intended"],
                "current": a["current"], "logout": None,
                "login_command": login_command(a["dir"], claude_exe=claude_exe)}
        if apply and a["logged_in"]:
            step["logout"] = auth_logout(a["dir"], claude_exe=claude_exe, runner=runner)
        steps.append(step)
    return {"assessment": assessment, "mismatched": len(mismatched),
            "applied": apply, "steps": steps}


# ----------------------------------------------------------------------------- CLI

def _print_assessment(assessment: list[dict[str, Any]]) -> int:
    bad = 0
    print(f"{'tag':<14} {'intended':<28} {'current login':<32} verdict")
    for a in assessment:
        if a["match"]:
            verdict = "OK"
        else:
            verdict = "MISMATCH -> " + a["note"]
            bad += 1
        print(f"{a['tag']:<14} {a['intended']:<28} {(a['current'] or '(none)'):<32} {verdict}")
    print(f"\n{len(assessment)} dir(s), {bad} need re-login.")
    return bad


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fix Claude config dirs logged into the wrong account.")
    p.add_argument("cmd", choices=("status", "fix", "verify"), help="action")
    p.add_argument("--map", default="", help="dir->email, e.g. gem5=gem5@x.ai,gem7=gem7@x.ai")
    p.add_argument("--map-file", default="", help="JSON file {tag: email}")
    p.add_argument("--apply", action="store_true", help="actually log out mismatched dirs (fix)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    intended = parse_map(args.map, args.map_file)
    if not intended:
        print("account_relogin: supply --map tag=email,... or --map-file", file=sys.stderr)
        return 2
    rows = fleet_accounts.discover_accounts()

    if args.cmd in ("status", "verify"):
        assessment = assess(intended, rows)
        if args.json:
            print(json.dumps({"schema": SCHEMA, "cmd": args.cmd, "assessment": assessment}, indent=1))
            bad = sum(1 for a in assessment if not a["match"])
        else:
            bad = _print_assessment(assessment)
            if args.cmd == "verify" and bad == 0:
                print("All dirs now hold their intended account. Fixed.")
        return 1 if bad else 0

    # fix
    result = fix(intended, rows, apply=args.apply)
    if args.json:
        print(json.dumps({"schema": SCHEMA, **result}, indent=1))
        return 1 if result["mismatched"] else 0
    _print_assessment(result["assessment"])
    if not result["steps"]:
        print("\nNothing to fix -- every dir already holds its intended account.")
        return 0
    if not args.apply:
        print("\nDRY-RUN. Re-run with --apply to log out the mismatched dirs, then run each "
              "login command below.\nMismatched dirs that WOULD be logged out:")
        for s in result["steps"]:
            print(f"  {s['tag']}  (currently {s['current'] or 'none'})")
        return 0
    print("\nLogged out the mismatched dirs. Now run EACH command below in a PowerShell "
          "window, completing /login with the INTENDED account:\n")
    for s in result["steps"]:
        lo = s.get("logout") or {}
        state = "logged out" if lo.get("ok") else f"logout: {lo.get('detail','?')}"
        print(f"# {s['tag']}  -> sign in as {s['intended']}   ({state})")
        print(s["login_command"])
        print()
    print("Then verify:  python tools/account_relogin.py verify --map <same map>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
