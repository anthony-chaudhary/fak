#!/usr/bin/env python3
"""Tests for the secret-shape leak gate (tools/check_secret_shapes.py).

Drives the pure scanner `_scan_text` — the part that decides whether a blob
carries an operator-leak SHAPE (a real home path or an internal lab host) the
literal needle list would miss — across every class it claims to catch and every
placeholder it claims to exempt. Read-only; no git, no disk.

NOTE: the shape fixtures are ASSEMBLED FROM FRAGMENTS at runtime (see the
builders below) so this test file does not itself contain a contiguous leak-shape
literal — otherwise the very gate under test would flag it under `--audit-tree`.
The runtime strings ARE the full shapes, so `_scan_text` sees exactly what it
would in real content; only the on-disk source is fragmented.

Run: `python tools/check_secret_shapes_test.py`  (exit 0 = all pass),
or `python -m pytest tools/check_secret_shapes_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_secret_shapes as css  # noqa: E402

_BS = chr(92)  # backslash, kept out of literal form


def _win(user: str, sep: str = _BS) -> str:
    return f"C:{sep}Users{sep}{user}"


def _mac(user: str) -> str:
    return f"/Users/{user}"


def _msl(name: str) -> str:
    return "msl" + "-" + name


def _lab(name: str) -> str:
    return name + "." + "lab"


def _shapes(text: str) -> list[tuple[str, str]]:
    return list(css._scan_text(text))


# --- operator home paths ---------------------------------------------------

def test_windows_backslash_operator_path_is_caught() -> None:
    p = _win("alice")
    assert ("operator-path", p) in _shapes(f"see {p}{_BS}work for the log")


def test_windows_forwardslash_operator_path_is_caught() -> None:
    # forward-slash form must not slip past (the exact motivating miss).
    p = _win("alice", "/")
    assert {s for s, _ in _shapes(f"path {p}/work")} == {"operator-path"}


def test_mac_operator_path_is_caught() -> None:
    p = _mac("bob")
    assert ("operator-path", p) in _shapes(f"cache at {p}/Library")


def test_placeholder_users_are_exempt() -> None:
    # the redaction placeholders must NOT be flagged or the gate cries wolf.
    for blob in (_win("USER"), _mac("runner"), _win("Public"), _mac("you")):
        assert _shapes(blob) == [], f"placeholder wrongly flagged: {blob}"


def test_placeholder_match_is_case_insensitive() -> None:
    # group is "User" -> lower "user" -> in PLACEHOLDER_USERS -> exempt.
    assert _shapes(_win("User")) == []


# --- internal lab hosts ----------------------------------------------------

def test_msl_host_is_caught_any_case() -> None:
    assert ("internal-host", _msl("gpu01")) in _shapes(f"ssh {_msl('gpu01')}")
    assert any(s == "internal-host" for s, _ in _shapes(_msl("box").upper()))


def test_lab_host_is_caught() -> None:
    host = _lab("testbox")
    assert ("internal-host", host) in _shapes(f"host {host} down")


def test_example_lab_placeholder_is_exempt() -> None:
    # the documented placeholder host must pass clean.
    assert _shapes(f"connect to {_lab('example')}") == []


# --- clean / negative ------------------------------------------------------

def test_clean_text_yields_nothing() -> None:
    assert _shapes("a perfectly ordinary sentence with no secrets") == []


def test_relative_users_path_without_drive_is_not_an_operator_path() -> None:
    # a bare "Users<sep>alice" with no drive letter and no leading slash is not
    # the shape (no C: prefix, no leading /).
    assert _shapes(f"the Users{_BS}alice folder") == []


# --- the rule's self-reference allowlist ------------------------------------

def test_self_ref_includes_this_gate_and_policy() -> None:
    # the files that DEFINE the rule must be self-exempt or the tree audit
    # would forever flag its own regexes.
    assert "tools/check_secret_shapes.py" in css.SELF_REF
    assert "PUBLIC-SCRUB-POLICY.md" in css.SELF_REF


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
