#!/usr/bin/env python3
"""Filesystem behavior tests for the getting-started dogfood harness."""

import tempfile
from pathlib import Path

import dogfood_getting_started as dogfood


def test_overlay_copies_only_existing_files_and_preserves_layout():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        repo = root / "repo"
        base = root / "base"
        source = repo / "docs" / "guide.md"
        source.parent.mkdir(parents=True)
        source.write_text("working-tree guide", encoding="utf-8")

        copied = dogfood._overlay(repo, base, ["docs/guide.md", "docs/missing.md"])

        assert copied == ["docs/guide.md"]
        assert (base / "docs" / "guide.md").read_text(encoding="utf-8") == (
            "working-tree guide"
        )


def test_strip_artifacts_removes_binaries_without_touching_sources():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        for rel in dogfood.ARTIFACTS:
            path = base / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"binary")
        keep = base / "cmd" / "fak" / "main.go"
        keep.parent.mkdir(parents=True)
        keep.write_text("package main", encoding="utf-8")

        dogfood._strip_artifacts(base)

        assert all(not (base / rel).exists() for rel in dogfood.ARTIFACTS)
        assert keep.read_text(encoding="utf-8") == "package main"


def test_lane_and_prompt_paths_are_deterministic():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        prompt = root / "lanes" / "laneB.txt"
        prompt.parent.mkdir()
        prompt.write_text("run lane B", encoding="utf-8")
        assert dogfood._lane_dir(root, "B") == root / "laneB"
        assert dogfood._prompt_for("B", root) == prompt

