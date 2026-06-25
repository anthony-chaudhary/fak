#!/usr/bin/env python3
"""Tests for demo_headless_smoke.py.

These tests exercise registry and output matching logic; the tool itself runs the
dynamic Go witnesses.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_headless_smoke as dhs  # noqa: E402


def test_registry_names_are_unique_and_commands_are_go_runs() -> None:
    names = [w.name for w in dhs.WITNESSES]
    assert len(names) == len(set(names)), names
    for witness in dhs.WITNESSES:
        assert witness.argv[:3] == ("go", "run", witness.argv[2]), witness
        assert witness.argv[2].startswith("./cmd/"), witness
        assert witness.must_contain, witness


def test_select_witnesses_keeps_known_and_reports_unknown() -> None:
    witnesses, unknown = dhs.select_witnesses(["guarddemo-selfcheck", "missing"])
    assert [w.name for w in witnesses] == ["guarddemo-selfcheck"]
    assert unknown == ["missing"]


def test_check_output_requires_substrings_and_exit_zero() -> None:
    witness = dhs.Witness("x", ("go", "run", "./cmd/x"), ("needle",), ("bad",))
    assert dhs.check_output(witness, 0, "prefix needle suffix") == []
    assert "exit status 2" in dhs.check_output(witness, 2, "needle")
    assert "missing required output: 'needle'" in dhs.check_output(witness, 0, "other")
    assert "forbidden output present: 'bad'" in dhs.check_output(witness, 0, "needle bad")


def test_documented_headless_go_commands_extracts_only_headless_section() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        docs = root / "docs"
        docs.mkdir()
        (docs / "run-the-demos.md").write_text(
            """
go run ./cmd/outside
## 2. Headless
go run ./cmd/guarddemo -selfcheck   # comment
go run ./cmd/tokendemo   -print
bash tools/run_comparison_demos.sh -q
## 3. With a real model
go run ./cmd/demorace
""",
            encoding="utf-8",
        )
        assert dhs.documented_headless_go_commands(root) == [
            "go run ./cmd/guarddemo -selfcheck",
            "go run ./cmd/tokendemo -print",
        ]


def test_registry_command_defects_flags_missing_and_stale_witnesses() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        docs = root / "docs"
        docs.mkdir()
        (docs / "run-the-demos.md").write_text(
            """
## 2. Headless
go run ./cmd/guarddemo -selfcheck
go run ./cmd/newdemo
## 3. With a real model
""",
            encoding="utf-8",
        )
        witnesses = (
            dhs.Witness("guarddemo-selfcheck", ("go", "run", "./cmd/guarddemo", "-selfcheck"), ("ok",)),
            dhs.Witness("stale", ("go", "run", "./cmd/stale"), ("ok",)),
        )
        defects = dhs.registry_command_defects(root, witnesses=witnesses)
        assert "documented headless command has no witness: go run ./cmd/newdemo" in defects
        assert "headless witness command is not documented in docs/run-the-demos.md: go run ./cmd/stale" in defects


def test_documented_readme_go_commands_keeps_only_deterministic_commands() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        readme = root / "cmd" / "guarddemo" / "README.md"
        readme.parent.mkdir(parents=True)
        readme.write_text(
            """
go run ./cmd/guarddemo
go run ./cmd/guarddemo -print
FAK_DEMO_BASE_PATH=/guarddemo go run ./cmd/guarddemo
go run ./cmd/simpledemo
go run ./cmd/memqdemo -report memqdemo-report.json
""",
            encoding="utf-8",
        )
        assert dhs.documented_readme_go_commands(root) == [
            "go run ./cmd/guarddemo -print",
            "go run ./cmd/memqdemo -report memqdemo-report.json",
        ]


def test_listed_docs_witnesses_are_present() -> None:
    names = set(dhs.witness_map())
    for required in {
        "guarddemo-selfcheck",
        "turntaxdemo-selfcheck",
        "tokendemo-selfcheck",
        "ctxdemo-bars",
        "unseedemo-selfcheck",
        "a2ademo",
        "ctxplandemo-selfcheck",
        "hwcachedemo",
        "cxlpooldemo",
        "memqdemo",
        "poisonedmcpdemo",
        "causalbench-selfcheck",
        "deletioncert-selfcheck",
    }:
        assert required in names


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("demo_headless_smoke_test: OK")
