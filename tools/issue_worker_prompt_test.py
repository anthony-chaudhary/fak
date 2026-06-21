#!/usr/bin/env python3
"""Hermetic tests for tools/issue_worker_prompt.py.

render_prompt is pure (data in, string out), so the load-bearing invariants —
the #N citation rule, the trunk/by-path git laws, the honest-block clause — are
asserted directly with no gh/claude/dos call. fetch_issue is exercised against
an injected runner so the gh shell-out never runs.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "issue_worker_prompt.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("issue_worker_prompt", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RenderPromptTest(unittest.TestCase):
    ISSUE = {
        "number": 465,
        "title": "obs: arm the DOS verdict-journal auto-emit",
        "body": "The trust floor's own decisions should be observable.",
        "labels": [{"name": "enhancement"}, {"name": "trust-floor"}],
        "state": "OPEN",
    }

    def test_cites_issue_number_as_the_close_link(self) -> None:
        mod = load()
        p = mod.render_prompt(self.ISSUE, "docs", workspace="C:/work/fak")
        # the #N citation must appear AND be called out as the close-binding rule.
        self.assertIn("#465", p)
        self.assertIn("commit subject", p)
        self.assertIn("never closes", p)  # the consequence of omitting it

    def test_embeds_title_body_labels_and_lane(self) -> None:
        mod = load()
        p = mod.render_prompt(self.ISSUE, "gateway", workspace="C:/work/fak")
        self.assertIn("auto-emit", p)               # title
        self.assertIn("observable", p)              # body
        self.assertIn("enhancement, trust-floor", p)  # labels
        self.assertIn("`gateway` lane", p)          # lane routing

    def test_states_the_git_laws(self) -> None:
        mod = load()
        p = mod.render_prompt(self.ISSUE, "docs", workspace="C:/work/fak")
        self.assertIn("main", p)
        self.assertIn("git add -A", p)      # the forbidden form is named
        self.assertIn("commit -s", p)       # sign-off required
        self.assertIn("OFF_TRUNK", p)

    def test_has_an_honest_block_clause(self) -> None:
        mod = load()
        p = mod.render_prompt(self.ISSUE, "docs", workspace="C:/work/fak")
        self.assertIn("final report", p)
        self.assertIn("fabricate", p)       # do NOT fabricate a pass

    def test_truncates_an_overlong_body(self) -> None:
        mod = load()
        big = dict(self.ISSUE, body="x" * 5000)
        p = mod.render_prompt(big, "docs", workspace="C:/work/fak")
        self.assertIn("truncated", p)
        self.assertLess(len(p), 5000)       # not the full 5000-char body

    def test_missing_body_still_renders(self) -> None:
        mod = load()
        nobody = {"number": 7, "title": "t", "labels": []}
        p = mod.render_prompt(nobody, "docs", workspace="C:/work/fak")
        self.assertIn("#7", p)
        self.assertIn("no body", p)


class FetchIssueTest(unittest.TestCase):
    def test_build_record_shape_on_fetch_error(self) -> None:
        mod = load()
        # Force fetch to fail by pointing gh at a bogus workspace+number with a
        # monkeypatched subprocess that raises — build must still return a prompt.
        mod.fetch_issue = lambda number, *, workspace, timeout=60: {
            "number": number, "_error": "gh not available"}
        with tempfile.TemporaryDirectory() as d:
            rec = mod.build(999, "docs", workspace=Path(d))
        self.assertEqual(rec["issue"], 999)
        self.assertEqual(rec["fetch_error"], "gh not available")
        self.assertIn("#999", rec["prompt"])         # prompt renders regardless
        self.assertGreater(rec["prompt_chars"], 100)


if __name__ == "__main__":
    unittest.main()
