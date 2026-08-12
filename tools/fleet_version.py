#!/usr/bin/env python3
"""Shared application/concept version helpers for fleet report generators."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


BENCHMARK_CONCEPT_VERSION = "fak.benchmark-concept.v1"
DEFAULT_VERSION = "dev"
CONFLICT_MARKER_PREFIXES = ("<<<<<<<", "=======", ">>>>>>>")


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    if here.is_file():
        here = here.parent
    for path in [here, *here.parents]:
        if (path / "VERSION").exists():
            return path
    return Path(__file__).resolve().parents[1]


def app_version(start: Path | None = None) -> str:
    env_version = os.environ.get("FAK_APP_VERSION", "").strip()
    if env_version:
        return env_version
    version_file = repo_root(start) / "VERSION"
    try:
        version = version_file.read_text(encoding="utf-8").strip()
    except OSError:
        return DEFAULT_VERSION
    if any(line.startswith(CONFLICT_MARKER_PREFIXES) for line in version.splitlines()):
        return DEFAULT_VERSION
    return version or DEFAULT_VERSION


def versioned(row: Mapping[str, Any], version: str | None = None) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("version", version or app_version())
    return out


def versioned_rows(rows: list[Mapping[str, Any]], version: str | None = None) -> list[dict[str, Any]]:
    return [versioned(row, version=version) for row in rows]
