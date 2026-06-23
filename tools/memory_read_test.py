#!/usr/bin/env python3
"""Hermetic tests for tools/memory_read.py.

render_digest() is exercised over a synthetic store built in a tempdir — no repo
mirror, no network. We assert: an absent store degrades to a harmless one-line note
(exit-0 contract), the index is emitted verbatim, per-fact bodies are expanded with
frontmatter stripped, non-fact docs are never expanded, --index-only skips bodies,
and --max-bytes bounds the per-fact output and reports the omission.
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "memory_read.py"


def load():
    spec = importlib.util.spec_from_file_location("memory_read", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load()


def _fact(name: str, desc: str, body: str) -> str:
    return (f"---\nname: {name}\ndescription: {desc}\nmetadata:\n  type: project\n"
            f"---\n\n{body}\n")


def build_store(d: Path) -> None:
    (d / "MEMORY.md").write_text(
        "- [First fact](first-fact.md) — the hook one\n"
        "- [Second fact](second-fact.md) — the hook two\n"
        "- [Archive index](MEMORY_archive.md) — cold tier\n",
        encoding="utf-8")
    (d / "first-fact.md").write_text(
        _fact("first-fact", "desc one", "BODY-ONE is the durable fact."), encoding="utf-8")
    (d / "second-fact.md").write_text(
        _fact("second-fact", "desc two", "BODY-TWO is the other fact."), encoding="utf-8")
    (d / "MEMORY_archive.md").write_text("- [old](old.md) — should not expand\n",
                                         encoding="utf-8")


class ParseIndex(unittest.TestCase):
    def test_links_extracted_dedup_and_nonfact_dropped(self):
        text = ("- [A](a.md) — h\n- [B](b.md) — h\n- [A again](a.md) — dup\n"
                "- [Idx](MEMORY.md) — self\n- [Sub](sub/c.md) — path\n")
        self.assertEqual(M.parse_index(text), [("A", "a.md"), ("B", "b.md")])


class StripFrontmatter(unittest.TestCase):
    def test_removes_leading_block(self):
        self.assertEqual(
            M.strip_frontmatter("---\nname: x\n---\n\nhello\n"), "hello\n")

    def test_no_frontmatter_passthrough(self):
        self.assertEqual(M.strip_frontmatter("just text\n"), "just text\n")


class RenderDigest(unittest.TestCase):
    def test_absent_store_is_harmless_note(self):
        with tempfile.TemporaryDirectory() as t:
            out = M.render_digest(Path(t) / "nope")
            self.assertIn("no committed memory mirror", out)
            self.assertNotIn("BODY-ONE", out)

    def test_full_digest_expands_facts(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            build_store(d)
            out = M.render_digest(d)
            self.assertIn("committed mirror", out)
            self.assertIn("First fact", out)            # index line present
            self.assertIn("BODY-ONE is the durable fact.", out)
            self.assertIn("BODY-TWO is the other fact.", out)
            self.assertNotIn("name: first-fact", out)   # frontmatter stripped
            self.assertNotIn("should not expand", out)   # non-fact doc not expanded

    def test_index_only_skips_bodies(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            build_store(d)
            out = M.render_digest(d, index_only=True)
            self.assertIn("First fact", out)
            self.assertNotIn("BODY-ONE", out)

    def test_max_bytes_bounds_and_reports_omission(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            build_store(d)
            out = M.render_digest(d, max_bytes=1)   # only the first fact fits
            self.assertIn("BODY-ONE", out)
            self.assertNotIn("BODY-TWO", out)
            self.assertIn("omitted", out)


if __name__ == "__main__":
    unittest.main()
