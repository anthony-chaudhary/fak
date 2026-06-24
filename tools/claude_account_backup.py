#!/usr/bin/env python3
"""Snapshot + restore the per-account Claude credentials the roster points at.

The DOS roster (~/.claude/accounts.yaml) maps each account name to a
CLAUDE_CONFIG_DIR. The login flow OVERWRITES the credentials inside a dir in
place: a `/login` against one account's config dir silently replaced its token
with another account's. Claude's own `backups/` only snapshots the MAIN `.claude.json` and
never the per-account `.credentials.json`/`.oauth-token`, so an overwrite is
unrecoverable.

This tool closes that gap. It copies the auth-bearing files of every roster
account (plus the main dir) into a timestamped, per-account backup tree, keyed
by the EMAIL the credentials actually authenticate as — so even if a dir is
later repointed to a different account, the prior account's token is still on
disk under its own email and can be restored.

  backup           snapshot every roster account (default)
  list             show what's been backed up, newest first, per email
  restore EMAIL    restore the newest backup for EMAIL into a target config dir

Backups live under ~/.claude-account-backups/ (host-local, never committed).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
ROSTER = Path(os.environ.get("CLAUDE_ACCOUNTS_FILE", HOME / ".claude" / "accounts.yaml"))
BACKUP_ROOT = HOME / ".claude-account-backups"

# The files that carry identity/auth for a config dir. Either of the first two is
# load-bearing on its own: .credentials.json holds the interactive refresh token,
# .oauth-token holds the long-lived setup token (the gem* workers carry only this
# one). The rest give a clean restore.
AUTH_FILES = (".credentials.json", ".oauth-token", ".claude.json", "settings.json")


def _load_roster_dirs() -> dict[str, Path]:
    """name -> config_dir for every roster account, plus the main dir.

    Parsed without a YAML dependency: the roster's `accounts:` block is a flat
    list of `name:`/`config_dir:` pairs (see accounts.yaml).
    """
    dirs: dict[str, Path] = {"_main": HOME / ".claude"}
    if not ROSTER.is_file():
        return dirs
    name = None
    in_accounts = False
    for raw in ROSTER.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("accounts:"):
            in_accounts = True
            continue
        if in_accounts and line and not line[0].isspace():
            break  # left the accounts block (e.g. `rotation:`)
        if not in_accounts:
            continue
        s = line.strip()
        if s.startswith("- name:"):
            name = s.split(":", 1)[1].strip()
        elif s.startswith("config_dir:") and name:
            dirs[name] = Path(s.split(":", 1)[1].strip())
            name = None
    return dirs


def _email_of(config_dir: Path) -> str | None:
    f = config_dir / ".claude.json"
    if not f.is_file():
        return None
    try:
        return (json.loads(f.read_text(encoding="utf-8"))
                .get("oauthAccount", {})
                .get("emailAddress"))
    except (json.JSONDecodeError, OSError):
        return None


def _safe(email: str) -> str:
    return email.replace("@", "_at_").replace("/", "_").replace("\\", "_")


def cmd_backup(_args: argparse.Namespace) -> int:
    dirs = _load_roster_dirs()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    n = 0
    seen: dict[str, str] = {}
    for name, cdir in dirs.items():
        if not cdir.is_dir():
            continue
        # Protect a dir that carries EITHER auth file. .credentials.json holds the
        # interactive refresh token; .oauth-token holds the long-lived setup token.
        # The gem* worker dirs authenticate via .oauth-token ALONE (no
        # .credentials.json), so gating on .credentials.json silently skipped them
        # -- the exact accounts whose only credential is the more durable one.
        if not (cdir / ".credentials.json").is_file() and not (cdir / ".oauth-token").is_file():
            continue  # nothing to protect
        email = _email_of(cdir) or f"unknown-{name}"
        dest = BACKUP_ROOT / _safe(email) / stamp
        dest.mkdir(parents=True, exist_ok=True)
        copied = []
        for fn in AUTH_FILES:
            src = cdir / fn
            if src.is_file():
                shutil.copy2(src, dest / fn)
                copied.append(fn)
        (dest / "_source.json").write_text(json.dumps({
            "roster_name": name,
            "config_dir": str(cdir),
            "email": email,
            "stamp": stamp,
            "files": copied,
        }, indent=2), encoding="utf-8")
        seen.setdefault(email, name)
        print(f"  backed up {email:<32} <- {name} ({cdir})")
        n += 1
    print(f"snapshot {stamp}: {n} account(s) -> {BACKUP_ROOT}")
    return 0


def _backups_by_email() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    if not BACKUP_ROOT.is_dir():
        return out
    for emaildir in sorted(BACKUP_ROOT.iterdir()):
        if emaildir.is_dir():
            snaps = sorted((p for p in emaildir.iterdir() if p.is_dir()),
                           reverse=True)
            if snaps:
                out[emaildir.name] = snaps
    return out


def _live_block_status() -> dict[str, dict]:
    """Map login-email -> live block status from the fleet roster, best-effort.

    A blocked (usage-limited / auth-disabled) account still EXISTS and still needs
    its credentials protected, so `list` must show it -- not drop it. The status is
    a courtesy column; a missing/broken fleet_accounts must never break the backup
    audit, so any failure degrades to 'no status known' (empty dict).
    """
    try:
        import fleet_accounts  # local sibling; imported lazily so backup never hard-deps it
        rows = fleet_accounts.annotated_roster()
    except Exception:  # noqa: BLE001 -- status is advisory; never crash the audit
        return {}
    out: dict[str, dict] = {}
    for r in rows:
        email = str(r.get("login_email") or "")
        if not email:
            continue
        # a canonical/unique dir wins over a duplicate when both carry the same email
        prior = out.get(email)
        if prior is None or (prior.get("blocked") and not r.get("blocked")):
            out[email] = {"blocked": bool(r.get("blocked")),
                          "reason": str(r.get("block_reason") or "")}
    return out


def cmd_list(_args: argparse.Namespace) -> int:
    """Audit EVERY roster account: its live creds, its backup state, its block status.

    Roster-driven on purpose. The old list iterated only over emails that already
    had a backup, so an account with nothing backed up (or no creds at all) silently
    vanished -- you could not tell 'gem8 has no backup' from 'gem8 does not exist'.
    Now every roster account shows a row, with NO BACKUP / blocked called out.
    """
    by = _backups_by_email()
    # index existing backups by the resolved email in their newest _source.json
    backup_by_email: dict[str, list[Path]] = {}
    for email_safe, snaps in by.items():
        email = email_safe.replace("_at_", "@")
        try:
            meta = json.loads((snaps[0] / "_source.json").read_text(encoding="utf-8"))
            email = meta.get("email", email)
        except (OSError, json.JSONDecodeError):
            pass
        backup_by_email[email] = snaps

    blocks = _live_block_status()
    dirs = _load_roster_dirs()
    seen_emails: set[str] = set()
    print(f"account backup audit  ({len(dirs)} roster dir(s)) -> {BACKUP_ROOT}")
    for name, cdir in dirs.items():
        email = _email_of(cdir) or f"unknown-{name}"
        seen_emails.add(email)
        # live credentials present on disk right now
        have = [fn for fn in (".credentials.json", ".oauth-token") if (cdir / fn).is_file()]
        # compact creds label: 'cred' / 'oauth' / 'cred+oauth' / 'NO CREDS'
        short = {".credentials.json": "cred", ".oauth-token": "oauth"}
        creds = "+".join(short[f] for f in have) if have else "NO CREDS"
        # backup state
        snaps = backup_by_email.get(email)
        if snaps:
            backup = f"{len(snaps)} snapshot(s), newest {snaps[0].name}"
        else:
            backup = "NO BACKUP"
        # live block status
        bs = blocks.get(email)
        if bs and bs.get("blocked"):
            status = f"BLOCKED: {bs['reason']}" if bs.get("reason") else "BLOCKED"
        elif bs is not None:
            status = "available"
        else:
            status = "status unknown"
        print(f"  {email:<40} [{creds:<18}] {backup:<34} {status}")

    # backups whose email is no longer in the roster (a removed/renamed account) --
    # still on disk and restorable, so surface them rather than hide them.
    for email, snaps in backup_by_email.items():
        if email not in seen_emails:
            print(f"  {email:<40} [roster: gone        ] "
                  f"{len(snaps)} snapshot(s), newest {snaps[0].name}  (not in roster)")
    if not dirs and not backup_by_email:
        print(f"(no roster accounts and no backups yet under {BACKUP_ROOT})")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    by = _backups_by_email()
    match = None
    for email_safe, snaps in by.items():
        meta = json.loads((snaps[0] / "_source.json").read_text(encoding="utf-8"))
        if meta.get("email") == args.email or email_safe == _safe(args.email):
            match = (meta, snaps[0])
            break
    if not match:
        print(f"no backup found for {args.email}. available:")
        cmd_list(args)
        return 1
    meta, snap = match
    target = Path(args.into) if args.into else Path(meta["config_dir"])
    print(f"restore {meta['email']} from {snap}")
    print(f"     -> {target}")
    if not args.yes:
        print("DRY RUN. re-run with --yes to write. files that would be restored:")
        for fn in AUTH_FILES:
            if (snap / fn).is_file():
                print(f"     {fn}")
        return 0
    target.mkdir(parents=True, exist_ok=True)
    for fn in AUTH_FILES:
        src = snap / fn
        if src.is_file():
            shutil.copy2(src, target / fn)
            print(f"     restored {fn}")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("backup", help="snapshot every roster account (default)")
    sub.add_parser("list", help="show backups, newest per email")
    r = sub.add_parser("restore", help="restore newest backup for an email")
    r.add_argument("email", help="account email to restore, e.g. worker@example.com")
    r.add_argument("--into", help="target config dir (default: the email's original dir)")
    r.add_argument("--yes", action="store_true", help="actually write (else dry-run)")
    args = p.parse_args(argv)
    cmd = args.cmd or "backup"
    return {"backup": cmd_backup, "list": cmd_list, "restore": cmd_restore}[cmd](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
