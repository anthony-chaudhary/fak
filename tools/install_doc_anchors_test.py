#!/usr/bin/env python3
"""Regression guard for install.sh's documentation anchors (issue #368).

`install.sh`'s unsupported-OS/arch error messages used to point a stranded user at
a non-existent anchor ("the manual-download table in README §7 / GETTING-STARTED"):
that table never existed in README.md, and GETTING-STARTED.md is about run tiers,
not a per-OS/arch asset table. The fix points the error at a destination that is
real on disk — `INSTALL.md §2 (Manual download)` plus a build-from-source fallback.

This test pins that invariant so the papercut can't silently come back: it fails if
install.sh ever cites the phantom anchor again, and it fails if the anchor install.sh
*does* cite stops resolving in INSTALL.md.
"""
from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO / "install.sh"
INSTALL_MD = REPO / "INSTALL.md"


class InstallDocAnchorsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sh = INSTALL_SH.read_text(encoding="utf-8")
        self.md = INSTALL_MD.read_text(encoding="utf-8")

    def test_no_phantom_anchor(self) -> None:
        # The exact dead references from #368 must never reappear in install.sh.
        for phantom in ("README §7", "manual-download table"):
            self.assertNotIn(
                phantom,
                self.sh,
                f"install.sh cites the phantom anchor {phantom!r} (issue #368)",
            )

    def test_unsupported_target_errors_cite_real_anchor(self) -> None:
        # Both the unsupported-OS and unsupported-arch err() lines must point the
        # user at INSTALL.md §2 (Manual download) — the destination that exists.
        err_lines = [
            ln
            for ln in self.sh.splitlines()
            if "unsupported" in ln and "err " in ln
        ]
        self.assertEqual(
            len(err_lines),
            2,
            "expected exactly the unsupported-OS and unsupported-arch err() lines",
        )
        for ln in err_lines:
            self.assertIn("INSTALL.md §2 (Manual download)", ln, ln)
            self.assertIn("build from source", ln, ln)

    def test_cited_anchor_resolves_in_install_md(self) -> None:
        # "INSTALL.md §2 (Manual download)" must resolve to a real heading, and the
        # build-from-source fallback the error promises must exist too.
        self.assertRegex(
            self.md,
            r"(?m)^##\s+2\.\s+Manual download\s*$",
            "install.sh cites INSTALL.md §2 (Manual download) but no such heading exists",
        )
        self.assertRegex(
            self.md,
            r"(?m)^##\s+Build from source\s*$",
            "install.sh promises a build-from-source fallback but INSTALL.md has no such section",
        )


if __name__ == "__main__":
    unittest.main()
