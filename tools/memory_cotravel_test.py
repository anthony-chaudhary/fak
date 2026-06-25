#!/usr/bin/env python3
"""Unit tests for memory_cotravel.py -- the slug-scoped agent-memory co-travel that
rides along with a transcript re-home.

Covers the two axes that make this an experimentation surface:
  * the pluggable per-file MERGE strategy (additive / source_wins / newest_mtime), each
    a pure (src,dst)->action function;
  * the FAK_MEMORY_COTRAVEL gate (off / shadow / live) -- shadow decides + ledgers but
    copies NOTHING, the prove-before-trust default.

Pure stdlib; no process spawn, no network. The ledger is redirected into tmp via
FAK_MEMORY_COTRAVEL_LEDGER so a test never touches the host store.

Run:  python -m pytest tools/memory_cotravel_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import memory_cotravel as mc  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers: build a src/dst account-dir pair with a slug-scoped memory store.
# --------------------------------------------------------------------------- #
SID = "deadbeef-0000-1111-2222-333344445555"
SLUG = "C--work-fak"


def _mem_dir(cfg: str, slug: str = SLUG) -> str:
    d = os.path.join(cfg, "projects", slug, "memory")
    os.makedirs(d, exist_ok=True)
    return d


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _ledger_into(tmp_path) -> str:
    p = str(tmp_path / "cotravel-ledger.jsonl")
    os.environ["FAK_MEMORY_COTRAVEL_LEDGER"] = p
    return p


def _clear_env():
    for k in ("FAK_MEMORY_COTRAVEL", "FAK_MEMORY_MERGE", "FAK_MEMORY_COTRAVEL_LEDGER"):
        os.environ.pop(k, None)


# --------------------------------------------------------------------------- #
# Pure strategy table -- no I/O beyond the byte/mtime compare.
# --------------------------------------------------------------------------- #
def test_additive_copies_only_missing(tmp_path):
    src = _mem_dir(str(tmp_path / "A"))
    dst = _mem_dir(str(tmp_path / "B"))
    _write(os.path.join(src, "note.md"), "from-A")
    # missing on dst -> copy
    assert mc._additive(os.path.join(src, "note.md"), os.path.join(dst, "note.md")) == "copy"
    # present on dst (any content) -> skip, never clobber
    _write(os.path.join(dst, "note.md"), "from-B")
    assert mc._additive(os.path.join(src, "note.md"), os.path.join(dst, "note.md")) == "skip"


def test_source_wins_overwrites_on_diff(tmp_path):
    src = _mem_dir(str(tmp_path / "A"))
    dst = _mem_dir(str(tmp_path / "B"))
    _write(os.path.join(src, "note.md"), "from-A")
    _write(os.path.join(dst, "note.md"), "from-B")
    assert mc._source_wins(os.path.join(src, "note.md"), os.path.join(dst, "note.md")) == "copy"
    # identical bytes -> no needless copy
    _write(os.path.join(dst, "note.md"), "from-A")
    assert mc._source_wins(os.path.join(src, "note.md"), os.path.join(dst, "note.md")) == "skip"


def test_newest_mtime_respects_mtime(tmp_path):
    src = _mem_dir(str(tmp_path / "A"))
    dst = _mem_dir(str(tmp_path / "B"))
    s = os.path.join(src, "note.md")
    d = os.path.join(dst, "note.md")
    _write(s, "from-A")
    _write(d, "from-B")
    os.utime(s, (3_000_000, 3_000_000))  # src newer
    os.utime(d, (1_000_000, 1_000_000))
    assert mc._newest_mtime(s, d) == "copy"
    os.utime(s, (1_000_000, 1_000_000))  # src older
    os.utime(d, (3_000_000, 3_000_000))
    assert mc._newest_mtime(s, d) == "skip"


# --------------------------------------------------------------------------- #
# Gate behavior end-to-end through cotravel_memory.
# --------------------------------------------------------------------------- #
def test_live_additive_lands_and_never_clobbers(tmp_path):
    _clear_env()
    _ledger_into(tmp_path)
    A, B = str(tmp_path / "A"), str(tmp_path / "B")
    src = _mem_dir(A)
    dst = _mem_dir(B)
    _write(os.path.join(src, "fresh.md"), "carry-me")     # missing on dst -> should land
    _write(os.path.join(src, "conflict.md"), "A-version")
    _write(os.path.join(dst, "conflict.md"), "B-version")  # exists on dst -> keep B's
    _write(os.path.join(dst, "dest-only.md"), "B-private")  # must NOT be pruned

    rec = mc.cotravel_memory(A, B, SLUG, SID, gate_value="live", strategy="additive")

    assert sorted(rec["copied"]) == ["fresh.md"]
    assert open(os.path.join(dst, "fresh.md"), encoding="utf-8").read() == "carry-me"
    # conflict kept B's bytes (never clobber)
    assert open(os.path.join(dst, "conflict.md"), encoding="utf-8").read() == "B-version"
    # dest-only survives (no prune)
    assert os.path.isfile(os.path.join(dst, "dest-only.md"))
    _clear_env()


def test_shadow_decides_but_copies_nothing(tmp_path):
    _clear_env()
    ledger = _ledger_into(tmp_path)
    A, B = str(tmp_path / "A"), str(tmp_path / "B")
    src = _mem_dir(A)
    _mem_dir(B)
    _write(os.path.join(src, "fresh.md"), "carry-me")

    rec = mc.cotravel_memory(A, B, SLUG, SID, gate_value="shadow", strategy="additive")

    # decided to copy, but in shadow NOTHING is written to the dst store
    assert rec["copied"] == []
    assert rec["would_copy"] == ["fresh.md"]
    assert not os.path.isfile(os.path.join(B, "projects", SLUG, "memory", "fresh.md"))
    # the decision IS recorded in the off-repo ledger
    assert os.path.isfile(ledger)
    rows = mc.read_ledger()
    assert rows and rows[-1]["session"] == SID and rows[-1]["gate"] == "shadow"
    _clear_env()


def test_off_is_a_noop(tmp_path):
    _clear_env()
    ledger = _ledger_into(tmp_path)
    A, B = str(tmp_path / "A"), str(tmp_path / "B")
    src = _mem_dir(A)
    _mem_dir(B)
    _write(os.path.join(src, "fresh.md"), "carry-me")

    rec = mc.cotravel_memory(A, B, SLUG, SID, gate_value="off")

    assert rec["copied"] == [] and rec["plan"] == []
    assert not os.path.isfile(os.path.join(B, "projects", SLUG, "memory", "fresh.md"))
    assert not os.path.isfile(ledger)  # off writes no ledger
    _clear_env()


def test_cross_dir_dst_slug_carries_owner_memory(tmp_path):
    # The cross-directory resume case: src memory lives under the OWNER slug, but the
    # resume runs from a different cwd whose slug is dst_slug -> the owner-slug memory
    # must land under dst_slug on the target.
    _clear_env()
    _ledger_into(tmp_path)
    A, B = str(tmp_path / "A"), str(tmp_path / "B")
    src = _mem_dir(A, SLUG)
    _write(os.path.join(src, "note.md"), "owner-memory")
    other = "C--work-slack-helpers"

    rec = mc.cotravel_memory(A, B, SLUG, SID, dst_slug=other,
                             gate_value="live", strategy="additive")

    assert rec["dst_slug"] == other
    landed = os.path.join(B, "projects", other, "memory", "note.md")
    assert os.path.isfile(landed)
    assert open(landed, encoding="utf-8").read() == "owner-memory"
    _clear_env()


def test_gate_and_strategy_default_to_shadow_additive(tmp_path):
    _clear_env()
    assert mc.gate() == "shadow"
    assert mc.strategy_name() == "additive"
    os.environ["FAK_MEMORY_COTRAVEL"] = "bogus"   # invalid -> falls back to default
    os.environ["FAK_MEMORY_MERGE"] = "bogus"
    assert mc.gate() == "shadow"
    assert mc.strategy_name() == "additive"
    _clear_env()


def test_no_source_memory_is_clean(tmp_path):
    # A session with no memory dir under the owner slug must produce an empty, harmless plan.
    _clear_env()
    _ledger_into(tmp_path)
    A, B = str(tmp_path / "A"), str(tmp_path / "B")
    os.makedirs(os.path.join(A, "projects", SLUG), exist_ok=True)  # no memory/ subdir
    rec = mc.cotravel_memory(A, B, SLUG, SID, gate_value="live")
    assert rec["src_has_memory"] is False
    assert rec["plan"] == [] and rec["copied"] == []
    _clear_env()
