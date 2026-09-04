#!/usr/bin/env python3
"""
cama_patchlib.py -- shared helpers for the CAMA/SGLang patch tooling.

Used by deploy.py, upgrade-sglang.py, and the scripts/ tools so they all agree
on (a) which SGLang files CAMA patches, (b) where the upstream base is pinned,
and (c) how to compare files without tripping over line-ending noise.

WHY LINE-ENDING NORMALIZATION MATTERS
-------------------------------------
The in-tree pre-packaged tree is committed with LF. A fresh `git clone` of
upstream on a machine with `core.autocrlf=true` (the Windows default) checks the
upstream files out with CRLF. A naive byte diff then reports *every line* as
changed -- e.g. environ.py looked like 1181 changed lines when the real delta is
59. Every comparison here normalizes to LF first so the tools work identically on
Linux and Windows checkouts.
"""

from __future__ import annotations

import json
from pathlib import Path

# ── locations ──────────────────────────────────────────────────────────
# cama-connector/  (this file lives at its root)
CONNECTOR_DIR = Path(__file__).resolve().parent
# monorepo root (parent of cama-connector/)
REPO_ROOT = CONNECTOR_DIR.parent

MANIFEST_PATH = CONNECTOR_DIR / "patch_manifest.json"
PATCHES_DIR = CONNECTOR_DIR / "patches"
# In-tree pre-packaged SGLang tree = source of truth for patch CONTENT.
INTREE_SGLANG = REPO_ROOT / "sglang-with-cama-connector"
UPSTREAM_TXT = INTREE_SGLANG / "UPSTREAM.txt"


# ── manifest ───────────────────────────────────────────────────────────
def load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def patch_entries() -> list[dict]:
    """Return the manifest's patch list: [{path, why}, ...]."""
    return load_manifest()["patches"]


def patch_map() -> dict[str, str]:
    """Return {leaf_filename: path_within_sglang_tree}.

    Raises if two patched files share a leaf name (patches/ is a flat dir keyed
    by leaf, so a collision would silently clobber one patch).
    """
    out: dict[str, str] = {}
    for entry in patch_entries():
        rel = entry["path"]
        leaf = Path(rel).name
        if leaf in out:
            raise ValueError(
                f"leaf-name collision in patch_manifest.json: '{leaf}' maps to "
                f"both '{out[leaf]}' and '{rel}'. patches/ is keyed by leaf name; "
                f"give one a distinct name or extend the tooling to nest paths."
            )
        out[leaf] = rel
    return out


# ── upstream pin ───────────────────────────────────────────────────────
def read_upstream() -> dict[str, str]:
    """Parse UPSTREAM.txt (key=value lines, '#' comments) into a dict."""
    info: dict[str, str] = {}
    if not UPSTREAM_TXT.is_file():
        return info
    for line in UPSTREAM_TXT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        info[key.strip()] = val.strip()
    return info


# ── line-ending-safe comparison ────────────────────────────────────────
def read_lf(path: Path) -> str:
    """Read a text file with all line endings normalized to LF.

    open(newline=None) enables universal-newline mode: CRLF/CR/LF all decode to
    '\\n'. This is the single chokepoint that makes diffs CRLF-proof.
    """
    with open(path, "r", encoding="utf-8", newline=None, errors="replace") as fh:
        return fh.read()


def files_differ(a: Path, b: Path) -> bool:
    """True if the two files differ ignoring line-ending style.

    A missing file counts as differing from a present one.
    """
    if a.is_file() != b.is_file():
        return True
    if not a.is_file():
        return False  # neither exists
    return read_lf(a) != read_lf(b)
