#!/usr/bin/env python3
"""Corpus tests for tools/skill_slop_scorecard.py (#2911).

Hermetic: every candidate skill is an in-memory string or a tmp file; the gate
never touches the network, git, or the real skill library. The corpus encodes the
acceptance: each named slop shape REJECTs (including the #3006 74KB verbatim-dump
shape) and terse-but-legitimate ADMITs (the calibration risk the issue names).
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "skill_slop_scorecard.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("skill_slop_scorecard", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GOOD = """---
name: refresh-index
description: Regenerate the search index after editing docs.
---

# refresh-index

Run the generator, then prove the index is fresh.

1. `python tools/gen_index.py --workspace .`
2. Verify: `python tools/gen_index.py --check` — exit 0 means the committed
   index matches the tree; nonzero names the stale file.
"""

TERSE_LEGIT = """---
name: lint-fast
description: Quick lint pass.
---

Run `python tools/lint.py --changed`; exit 0 means clean, nonzero lists offenders.
"""


def _signals(card):
    return {f["signal"] for f in card["findings"]}


class CorpusTest(unittest.TestCase):
    def test_good_skill_admits(self) -> None:
        mod = load()
        card = mod.grade_text(GOOD)
        self.assertTrue(card["ok"], card)
        self.assertEqual(card["verdict"], "ADMIT")
        self.assertEqual(card["score"], 100)

    def test_terse_but_legitimate_admits(self) -> None:
        # The calibration risk the issue names: the gate rejects SLOP, not brevity.
        mod = load()
        card = mod.grade_text(TERSE_LEGIT)
        self.assertTrue(card["ok"], card)

    def test_3006_shape_verbatim_dump_rejects(self) -> None:
        # bug #3006: a 74KB transcript stored verbatim as one skill entry.
        mod = load()
        dump = GOOD + "\n```\n" + ("x" * 74_000) + "\n```\n"
        card = mod.grade_text(dump)
        self.assertFalse(card["ok"])
        self.assertIn("verbatim_dump", _signals(card))

    def test_oversized_body_without_fences_still_rejects(self) -> None:
        mod = load()
        card = mod.grade_text(GOOD + ("padding line with words\n" * 3_000))
        self.assertIn("verbatim_dump", _signals(card))

    def test_vacuous_body_rejects(self) -> None:
        mod = load()
        card = mod.grade_text("---\nname: x\n---\n\n# x\n\n## Usage\n")
        self.assertFalse(card["ok"])
        self.assertIn("vacuous_body", _signals(card))

    def test_marketing_voice_rejects(self) -> None:
        mod = load()
        card = mod.grade_text(GOOD + "\nThis seamless, revolutionary flow is "
                                     "best-in-class and effortless.\n")
        self.assertFalse(card["ok"])
        self.assertIn("marketing", _signals(card))

    def test_one_off_narrative_rejects(self) -> None:
        mod = load()
        card = mod.grade_text(GOOD + "\nI then ran the build. I noticed a flake, "
                                     "and I fixed the test at 2026-07-07T06:52.\n")
        self.assertFalse(card["ok"])
        self.assertIn("one_off_narrative", _signals(card))

    def test_missing_verification_rejects(self) -> None:
        mod = load()
        card = mod.grade_text("""---
name: vibes
---

# vibes

Open the dashboard and look around the panels.
Consider the layout and think about the colors.
Move things until the page feels balanced overall.
""")
        self.assertFalse(card["ok"])
        self.assertIn("missing_verification", _signals(card))

    def test_frontmatter_never_counts_toward_the_grade(self) -> None:
        mod = load()
        body = mod._strip_frontmatter(GOOD)
        self.assertNotIn("name: refresh-index", body)


class CliGateTest(unittest.TestCase):
    def test_exit_codes_are_the_gate(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            skills = Path(d)
            good = skills / "good" / "SKILL.md"
            good.parent.mkdir(parents=True)
            good.write_text(GOOD, encoding="utf-8")
            self.assertEqual(mod.main([str(skills)]), 0)          # all admit
            bad = skills / "bad" / "SKILL.md"
            bad.parent.mkdir(parents=True)
            bad.write_text("---\nname: bad\n---\n# bad\n", encoding="utf-8")
            self.assertEqual(mod.main([str(skills)]), 1)          # HARD gate
            self.assertEqual(mod.main([str(skills / "missing")]), 2)  # contract error

    def test_json_payload_names_every_card(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "SKILL.md"
            p.write_text(TERSE_LEGIT, encoding="utf-8")
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.main([str(p), "--json"])
            self.assertEqual(rc, 0)
            import json as _json
            payload = _json.loads(buf.getvalue())
            self.assertEqual(payload["schema"], mod.SCHEMA)
            self.assertEqual(payload["graded"], 1)
            self.assertEqual(payload["cards"][0]["verdict"], "ADMIT")


if __name__ == "__main__":
    unittest.main()
