#!/usr/bin/env python3
"""Tests for tools/doc_reap.py: archive moves, banners, link repointing, check gate."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_reap  # noqa: E402


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


class DocReapTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _git(self.root, "init", "-q")
        _git(self.root, "config", "user.email", "t@example.invalid")
        _git(self.root, "config", "user.name", "t")
        files = {
            "README.md": "See [old](docs/old.md) and `docs/old.md`.\n",
            "docs/old.md": "---\ntitle: old\n---\n\n# Old\n\nBack to [readme](../README.md) or [new](new.md#top).\n",
            "docs/new.md": "# New\n\nReplaces [old](old.md).\n",
            "docs/sub/deep.md": "Up: [old](../old.md).\n",
            "docs/_witnesses/w.json": "{\"path\": \"docs/old.md\"}\n",
            "docs/archive/README.md": "# Archive\n",
        }
        for rel, body in files.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8", newline="\n")
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-qm", "seed")
        self._cwd = os.getcwd()
        os.chdir(self.root)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")

    def test_archive_moves_banners_and_repoints(self) -> None:
        rc = doc_reap.main(["archive", "docs/old.md", "--reason", "superseded",
                            "--successor", "docs/new.md", "--today", "2026-11-23"])
        self.assertEqual(rc, 0)
        self.assertFalse((self.root / "docs/old.md").exists())
        moved = self.read("docs/archive/old.md")
        self.assertIn("> **Archived 2026-11-23** · reason: superseded · replacement: [docs/new.md](../new.md)", moved)
        self.assertTrue(moved.startswith("---\ntitle: old\n---\n"), "front matter stays first")
        self.assertIn("[readme](../../README.md)", moved)
        self.assertIn("[new](../new.md#top)", moved)
        self.assertIn("[old](docs/archive/old.md)", self.read("README.md"))
        self.assertIn("`docs/archive/old.md`", self.read("README.md"))
        self.assertIn("[old](archive/old.md)", self.read("docs/new.md"))
        self.assertIn("[old](../archive/old.md)", self.read("docs/sub/deep.md"))
        self.assertIn("\"docs/old.md\"", self.read("docs/_witnesses/w.json"), "evidence stays frozen")
        self.assertEqual(doc_reap.main(["check"]), 0)

    def test_links_between_archived_docs_survive_a_second_archive(self) -> None:
        self.assertEqual(doc_reap.main(["archive", "docs/old.md", "--reason", "stale-snapshot",
                                        "--successor", "none"]), 0)
        _git(self.root, "add", "-A")
        self.assertEqual(doc_reap.main(["archive", "docs/new.md", "--reason", "retired",
                                        "--successor", "none"]), 0)
        self.assertIn("[new](new.md#top)", self.read("docs/archive/old.md"))
        self.assertEqual(doc_reap.main(["check"]), 0)

    def test_check_flags_archived_doc_without_banner(self) -> None:
        (self.root / "docs/archive/bare.md").write_text("# Bare\n", encoding="utf-8")
        _git(self.root, "add", "-A")
        self.assertEqual(doc_reap.main(["check"]), 1)

    def test_archive_refuses_missing_successor(self) -> None:
        rc = doc_reap.main(["archive", "docs/old.md", "--reason", "superseded",
                            "--successor", "docs/nope.md"])
        self.assertEqual(rc, 1)
        self.assertTrue((self.root / "docs/old.md").exists())

    def test_rank_lists_orphan_first(self) -> None:
        rows = doc_reap.rank(self.root, recursive=False, today=doc_reap.dt.date(2027, 6, 1))
        paths = [r["path"] for r in rows]
        self.assertIn("docs/old.md", paths)
        self.assertNotIn("docs/archive/README.md", paths)


if __name__ == "__main__":
    unittest.main()
