#!/usr/bin/env python3
"""Planning tests for the private companion repository scaffold."""

import tempfile
from pathlib import Path

import init_private_repo as init_private


def test_plan_private_paths_unifies_existing_explicit_and_globbed_paths():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "secret").mkdir()
        (root / "secret" / "token.txt").write_text("x", encoding="utf-8")
        (root / "logs").mkdir()
        log = root / "logs" / "private.log"
        log.write_text("private", encoding="utf-8")

        class Scrub:
            DELETE_PATHS = ["secret", "missing"]
            DELETE_GLOBS = ["logs/*.log"]

            @staticmethod
            def expand_glob(source_root, pattern):
                return Path(source_root).glob(pattern)

        assert init_private.plan_private_paths(Scrub, td) == [
            "logs/private.log",
            "secret",
        ]


def test_dir_bytes_sums_nested_files_and_ignores_directories():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.bin").write_bytes(b"123")
        (root / "nested").mkdir()
        (root / "nested" / "b.bin").write_bytes(b"45678")
        assert init_private._dir_bytes(td) == 8


def test_generated_readme_names_public_source_and_private_boundary():
    readme = init_private._readme("C:/private-source")
    assert init_private.PUBLIC_REMOTE_HINT in readme
    assert "ONLY the" in readme and "operator-private material" in readme
    assert "C:/private-source" in readme

