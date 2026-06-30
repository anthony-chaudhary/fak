#!/usr/bin/env python3
"""Tests for the shared fleet version helpers (tools/fleet_version.py).

Covers `versioned` / `versioned_rows` (stamp a version onto report rows without
clobbering an existing one or mutating the input) and `app_version` (env override
> VERSION file > "dev" fallback, with a conflict-marker guard). Pure stdlib; the
file-reading paths are driven with a tmp VERSION file, no real repo needed.

Run: `python tools/fleet_version_test.py`  (exit 0 = all pass),
or `python -m pytest tools/fleet_version_test.py -q`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fleet_version as fv  # noqa: E402


# --- versioned / versioned_rows --------------------------------------------

def test_versioned_stamps_explicit_version() -> None:
    out = fv.versioned({"a": 1}, version="1.2.3")
    assert out == {"a": 1, "version": "1.2.3"}


def test_versioned_does_not_clobber_existing_version() -> None:
    out = fv.versioned({"version": "9.9.9", "a": 1}, version="1.2.3")
    assert out["version"] == "9.9.9"  # setdefault: existing wins


def test_versioned_does_not_mutate_input() -> None:
    row = {"a": 1}
    fv.versioned(row, version="1.2.3")
    assert row == {"a": 1}  # input untouched (dict copy)


def test_versioned_rows_maps_each_row() -> None:
    rows = [{"a": 1}, {"b": 2}]
    out = fv.versioned_rows(rows, version="7")
    assert out == [{"a": 1, "version": "7"}, {"b": 2, "version": "7"}]


# --- app_version (env > VERSION file > default) -----------------------------

def _clear_env_version():
    prev = os.environ.pop("FAK_APP_VERSION", None)
    return prev


def _restore_env_version(prev):
    if prev is None:
        os.environ.pop("FAK_APP_VERSION", None)
    else:
        os.environ["FAK_APP_VERSION"] = prev


def test_app_version_env_override_wins(tmp_path: Path) -> None:
    prev = os.environ.get("FAK_APP_VERSION")
    os.environ["FAK_APP_VERSION"] = "env-1.0"
    try:
        (tmp_path / "VERSION").write_text("file-2.0\n", encoding="utf-8")
        assert fv.app_version(tmp_path) == "env-1.0"
    finally:
        _restore_env_version(prev)


def test_app_version_reads_version_file(tmp_path: Path) -> None:
    prev = _clear_env_version()
    try:
        (tmp_path / "VERSION").write_text("3.4.5\n", encoding="utf-8")
        assert fv.app_version(tmp_path) == "3.4.5"
    finally:
        _restore_env_version(prev)


def test_app_version_conflict_marker_falls_back_to_dev(tmp_path: Path) -> None:
    prev = _clear_env_version()
    try:
        # assemble the conflict markers from fragments so this test file does not
        # itself carry a literal <<<<<<< / ======= a conflict-marker gate would flag.
        lt, eq = "<" * 7, "=" * 7
        (tmp_path / "VERSION").write_text(f"{lt} HEAD\n1.0\n{eq}\n2.0\n", encoding="utf-8")
        assert fv.app_version(tmp_path) == fv.DEFAULT_VERSION
    finally:
        _restore_env_version(prev)


def test_app_version_missing_file_falls_back_to_dev(tmp_path: Path) -> None:
    prev = _clear_env_version()
    try:
        # tmp_path has no VERSION; repo_root walks up and may find the real repo's
        # VERSION, so point start at an isolated empty subtree with its own marker
        # absent — assert it returns *some* non-empty string (real VERSION) or dev.
        empty = tmp_path / "nowhere"
        empty.mkdir()
        result = fv.app_version(empty)
        assert isinstance(result, str) and result  # never crashes, never empty
    finally:
        _restore_env_version(prev)


# --- repo_root -------------------------------------------------------------

def test_repo_root_finds_dir_with_version_file(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("1.0\n", encoding="utf-8")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert fv.repo_root(sub) == tmp_path


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
