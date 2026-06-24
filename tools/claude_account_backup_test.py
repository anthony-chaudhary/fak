#!/usr/bin/env python3
"""Tests for claude_account_backup -- the per-account credential snapshotter.

The load-bearing rule under test: a config dir is protected when it carries
EITHER auth file. The gem* worker dirs authenticate via .oauth-token ALONE (no
.credentials.json), so an earlier .credentials.json-only gate silently skipped
exactly the accounts whose only credential was the more durable setup token.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claude_account_backup as b  # noqa: E402


def _write_account(cdir: Path, email: str, *, cred: bool, oauth: bool) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / ".claude.json").write_text(
        json.dumps({"oauthAccount": {"emailAddress": email}}), encoding="utf-8")
    if cred:
        (cdir / ".credentials.json").write_text('{"refresh":"x"}', encoding="utf-8")
    if oauth:
        (cdir / ".oauth-token").write_text("sk-ant-oat01-token", encoding="utf-8")


def _run_backup(tmp_path: Path, monkeypatch, roster: dict[str, Path]) -> Path:
    backup_root = tmp_path / "backups"
    monkeypatch.setattr(b, "BACKUP_ROOT", backup_root)
    monkeypatch.setattr(b, "_load_roster_dirs", lambda: roster)
    assert b.cmd_backup(None) == 0
    return backup_root


def test_oauth_token_only_account_is_backed_up(tmp_path, monkeypatch):
    """A gem-style dir with only .oauth-token must be protected (the regression)."""
    gem8 = tmp_path / ".claude-gem8-netra"
    _write_account(gem8, "gem8@example.com", cred=False, oauth=True)
    root = _run_backup(tmp_path, monkeypatch, {"gem8-netra": gem8})

    snaps = list((root / "gem8_at_example.com").iterdir())
    assert len(snaps) == 1
    saved = {p.name for p in snaps[0].iterdir()}
    assert ".oauth-token" in saved
    assert ".claude.json" in saved


def test_credentials_only_account_is_backed_up(tmp_path, monkeypatch):
    """The historical case (interactive creds, no setup token) still works."""
    acct = tmp_path / ".claude-smith-netra"
    _write_account(acct, "smith@example.com", cred=True, oauth=False)
    root = _run_backup(tmp_path, monkeypatch, {"smith-netra": acct})

    snaps = list((root / "smith_at_example.com").iterdir())
    assert len(snaps) == 1
    assert ".credentials.json" in {p.name for p in snaps[0].iterdir()}


def test_dir_with_no_auth_files_is_skipped(tmp_path, monkeypatch):
    """A dir carrying neither auth file has nothing to protect -> no snapshot."""
    acct = tmp_path / ".claude-empty-netra"
    _write_account(acct, "empty@example.com", cred=False, oauth=False)
    root = _run_backup(tmp_path, monkeypatch, {"empty-netra": acct})

    assert not (root / "empty_at_example.com").exists()


def test_list_shows_every_roster_account(tmp_path, monkeypatch, capsys):
    """list is roster-driven: a backed-up acct, an un-backed-up acct, and a
    blocked acct must ALL appear -- none may silently vanish (the gem8 ask)."""
    backed = tmp_path / ".claude-gem7-netra"
    _write_account(backed, "gem7@example.com", cred=True, oauth=True)
    nobackup = tmp_path / ".claude-gem8-netra"
    _write_account(nobackup, "gem8@example.com", cred=False, oauth=True)

    backup_root = tmp_path / "backups"
    monkeypatch.setattr(b, "BACKUP_ROOT", backup_root)
    # only gem7 gets a backup; gem8 is on the roster but never backed up
    monkeypatch.setattr(b, "_load_roster_dirs", lambda: {"gem7-netra": backed})
    assert b.cmd_backup(None) == 0
    capsys.readouterr()

    # now list against the full roster, with gem8 reported BLOCKED by the fleet
    monkeypatch.setattr(b, "_load_roster_dirs",
                        lambda: {"gem7-netra": backed, "gem8-netra": nobackup})
    monkeypatch.setattr(b, "_live_block_status", lambda: {
        "gem8@example.com": {"blocked": True, "reason": "usage limit; resets 8pm"}})
    assert b.cmd_list(None) == 0
    out = capsys.readouterr().out

    assert "gem7@example.com" in out
    assert "gem8@example.com" in out          # the un-backed-up account still shows
    assert "NO BACKUP" in out                  # and is flagged as such
    assert "BLOCKED" in out                    # with its live block status


def test_list_degrades_when_fleet_status_unavailable(tmp_path, monkeypatch, capsys):
    """A broken/absent fleet_accounts must not crash the audit -- status degrades to
    'status unknown' and every account still lists."""
    import fleet_accounts
    monkeypatch.setattr(fleet_accounts, "annotated_roster",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    # _live_block_status swallows the failure -> empty map -> 'status unknown'
    assert b._live_block_status() == {}

    acct = tmp_path / ".claude-gem8-netra"
    _write_account(acct, "gem8@example.com", cred=False, oauth=True)
    monkeypatch.setattr(b, "BACKUP_ROOT", tmp_path / "backups")
    monkeypatch.setattr(b, "_load_roster_dirs", lambda: {"gem8-netra": acct})
    assert b.cmd_list(None) == 0
    out = capsys.readouterr().out
    assert "gem8@example.com" in out
    assert "status unknown" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
