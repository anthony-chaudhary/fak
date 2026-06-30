#!/usr/bin/env python3
"""Tests for the release version bumper (tools/release_bump.py).

Covers the three pure bump primitives against a tmp tree:
  * bump_plain_file   — VERSION-style single-token file (change / idempotent /
    dry-run / missing).
  * bump_install_doc  — only semvers on an install/pin LINE are rewritten; a bare
    semver in prose is left alone (the no-over-reach contract).
  * bump_dist_manifest — only semvers on a manifest pin line are rewritten, and a
    CITATION `date-released:` is updated only when a date is supplied.

Run: `python tools/release_bump_test.py`  (exit 0 = all pass),
or `python -m pytest tools/release_bump_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_bump as rb  # noqa: E402


# --- bump_plain_file -------------------------------------------------------

def test_bump_plain_file_changes_and_writes(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("0.33.0\n", encoding="utf-8")
    res = rb.bump_plain_file(tmp_path, "VERSION", "0.34.0", dry_run=False)
    assert res["ok"] and res["changed"] and res["old"] == "0.33.0"
    assert (tmp_path / "VERSION").read_text(encoding="utf-8") == "0.34.0\n"


def test_bump_plain_file_dry_run_does_not_write(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("0.33.0\n", encoding="utf-8")
    res = rb.bump_plain_file(tmp_path, "VERSION", "0.34.0", dry_run=True)
    assert res["changed"] is True
    # dry-run reports the change but the file is untouched.
    assert (tmp_path / "VERSION").read_text(encoding="utf-8") == "0.33.0\n"


def test_bump_plain_file_idempotent(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("0.34.0\n", encoding="utf-8")
    res = rb.bump_plain_file(tmp_path, "VERSION", "0.34.0", dry_run=False)
    assert res["ok"] and res["changed"] is False


def test_bump_plain_file_missing(tmp_path: Path) -> None:
    res = rb.bump_plain_file(tmp_path, "NOPE", "0.34.0", dry_run=False)
    assert res["ok"] is False and "not found" in res["reason"]


# --- bump_install_doc ------------------------------------------------------

def test_bump_install_doc_rewrites_only_install_lines(tmp_path: Path) -> None:
    doc = "Install: `FAK_VERSION=0.33.0`\nIn the 0.33.0 release we shipped X.\n"
    (tmp_path / "INSTALL.md").write_text(doc, encoding="utf-8")
    res = rb.bump_install_doc(tmp_path, "INSTALL.md", "0.34.0", dry_run=False)
    out = (tmp_path / "INSTALL.md").read_text(encoding="utf-8")
    assert res["changed"] and res["replaced"] == ["0.33.0->0.34.0"]
    assert "FAK_VERSION=0.34.0" in out          # install pin bumped
    assert "In the 0.33.0 release" in out        # prose semver left alone


def test_bump_install_doc_idempotent(tmp_path: Path) -> None:
    (tmp_path / "INSTALL.md").write_text("pip install fak==0.34.0\n", encoding="utf-8")
    res = rb.bump_install_doc(tmp_path, "INSTALL.md", "0.34.0", dry_run=False)
    assert res["changed"] is False and res["replaced"] == []


# --- bump_dist_manifest ----------------------------------------------------

def test_bump_dist_manifest_bumps_version_pin(tmp_path: Path) -> None:
    (tmp_path / "server.json").write_text(
        '{\n  "version": "0.33.0",\n  "description": "fak 0.33.0 server"\n}\n',
        encoding="utf-8")
    res = rb.bump_dist_manifest(tmp_path, "server.json", "0.34.0", date=None, dry_run=False)
    out = (tmp_path / "server.json").read_text(encoding="utf-8")
    assert res["changed"] and '"version": "0.34.0"' in out
    assert '"description": "fak 0.33.0 server"' in out   # prose version untouched


def test_bump_dist_manifest_updates_citation_date(tmp_path: Path) -> None:
    (tmp_path / "CITATION.cff").write_text(
        "version: 0.33.0\ndate-released: 2026-01-01\n", encoding="utf-8")
    rb.bump_dist_manifest(tmp_path, "CITATION.cff", "0.34.0",
                          date="2026-06-29", dry_run=False)
    out = (tmp_path / "CITATION.cff").read_text(encoding="utf-8")
    assert "version: 0.34.0" in out and "date-released: 2026-06-29" in out


def test_bump_dist_manifest_no_date_leaves_date_alone(tmp_path: Path) -> None:
    (tmp_path / "CITATION.cff").write_text(
        "version: 0.33.0\ndate-released: 2026-01-01\n", encoding="utf-8")
    rb.bump_dist_manifest(tmp_path, "CITATION.cff", "0.34.0", date=None, dry_run=False)
    out = (tmp_path / "CITATION.cff").read_text(encoding="utf-8")
    assert "date-released: 2026-01-01" in out and "version: 0.34.0" in out


def _run_all() -> int:
    import tempfile
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            if fn.__code__.co_argcount == 1:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
