#!/usr/bin/env python3
"""Tests for the doc-placement gate (tools/check_doc_placement.py).

Drives the pure rule `_violations` (which root *.md files are off the front-door
allowlist) and the INDEX-completeness scan `_unindexed_new_notes` (a new dated
docs/notes/ note must be listed in INDEX.md). The first is pure; the second is
driven with a tmp tree + a stubbed staged-file list, so no real git is needed.

Run: `python tools/check_doc_placement_test.py`  (exit 0 = all pass),
or `python -m pytest tools/check_doc_placement_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_doc_placement as cdp  # noqa: E402


# --- _violations (pure) ----------------------------------------------------

def test_stray_root_md_is_a_violation() -> None:
    assert cdp._violations(["PLAN-2026-06-29.md"]) == ["PLAN-2026-06-29.md"]


def test_allowlisted_root_md_is_clean() -> None:
    assert cdp._violations(["README.md", "AGENTS.md", "CLAUDE.md"]) == []


def test_nested_md_is_not_a_root_violation() -> None:
    # a path with "/" is not at the root, so placement doesn't apply here.
    assert cdp._violations(["docs/notes/PLAN-2026-06-29.md"]) == []


def test_non_md_files_are_ignored() -> None:
    assert cdp._violations(["build.py", "Makefile", "go.mod"]) == []


def test_violations_are_sorted_and_filtered() -> None:
    names = ["ZED.md", "README.md", "ALPHA.md", "docs/x.md", "tool.sh"]
    assert cdp._violations(names) == ["ALPHA.md", "ZED.md"]


def test_allowlist_holds_the_core_front_door() -> None:
    for name in ("README.md", "INDEX.md", "CONTRIBUTING.md", "AGENTS.md",
                 "CLAUDE.md", "SECURITY.md"):
        assert name in cdp.ALLOWED_ROOT_MD


# --- _unindexed_new_notes (tmp tree + stubbed staged list) ------------------

def _with_staged(monkeypatch_list: list[str]):
    """Swap _staged_added for a fixed list; returns a restore callable."""
    orig = cdp._staged_added
    cdp._staged_added = lambda root: monkeypatch_list  # type: ignore[assignment]
    return orig


def test_new_dated_note_absent_from_index_is_flagged(tmp_path: Path) -> None:
    (tmp_path / "INDEX.md").write_text("# Index\n- only this\n", encoding="utf-8")
    orig = _with_staged(["docs/notes/PLAN-2026-06-29.md"])
    try:
        assert cdp._unindexed_new_notes(str(tmp_path)) == ["docs/notes/PLAN-2026-06-29.md"]
    finally:
        cdp._staged_added = orig  # type: ignore[assignment]


def test_new_dated_note_present_in_index_passes(tmp_path: Path) -> None:
    (tmp_path / "INDEX.md").write_text(
        "# Index\n- [plan](docs/notes/PLAN-2026-06-29.md)\n", encoding="utf-8")
    orig = _with_staged(["docs/notes/PLAN-2026-06-29.md"])
    try:
        assert cdp._unindexed_new_notes(str(tmp_path)) == []
    finally:
        cdp._staged_added = orig  # type: ignore[assignment]


def test_undated_note_is_not_required_in_index(tmp_path: Path) -> None:
    # only DATED / PLAN- notes must be indexed; a plain note is not flagged.
    (tmp_path / "INDEX.md").write_text("# Index\n", encoding="utf-8")
    orig = _with_staged(["docs/notes/some-explainer.md"])
    try:
        assert cdp._unindexed_new_notes(str(tmp_path)) == []
    finally:
        cdp._staged_added = orig  # type: ignore[assignment]


def test_notes_readme_is_exempt(tmp_path: Path) -> None:
    (tmp_path / "INDEX.md").write_text("# Index\n", encoding="utf-8")
    orig = _with_staged(["docs/notes/README.md"])
    try:
        assert cdp._unindexed_new_notes(str(tmp_path)) == []
    finally:
        cdp._staged_added = orig  # type: ignore[assignment]


def _run_all() -> int:
    import tempfile
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            # supply a tmp_path for the tests that take one (mimics pytest fixture).
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
