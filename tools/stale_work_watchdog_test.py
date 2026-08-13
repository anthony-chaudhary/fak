#!/usr/bin/env python3
"""Compatibility-shim tests; behavior is witnessed by cmd/fak Go tests."""
from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stale_work_watchdog as watchdog  # noqa: E402


class FakeResult:
    def __init__(self, returncode: int):
        self.returncode = returncode


def test_command_forwards_every_argument_to_go_verb():
    old = os.environ.get("FAK_BIN")
    os.environ["FAK_BIN"] = "C:/tools/fak.exe"
    try:
        assert watchdog.command(["--live", "--json"]) == [
            "C:/tools/fak.exe", "garden", "watchdog", "--live", "--json"
        ]
    finally:
        if old is None:
            os.environ.pop("FAK_BIN", None)
        else:
            os.environ["FAK_BIN"] = old


def test_main_returns_go_verbs_exit_code():
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return FakeResult(7)

    assert watchdog.main(["--fail-on-stale"], runner=runner) == 7
    assert calls[0][0][1:3] == ["garden", "watchdog"]
    assert calls[0][0][-1] == "--fail-on-stale"
    assert calls[0][1]["check"] is False


def _run_all() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
