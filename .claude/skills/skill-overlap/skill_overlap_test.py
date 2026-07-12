#!/usr/bin/env python3
"""Witness tests for skill_overlap.py (#3930).

The issue's named witness: a fixture skill pack containing one deliberately-
overlapping pair asserts that pair is surfaced above the threshold while two
unrelated skills are not — plus the front-matter-stripped, proposal-only,
deterministic, archived-skipped invariants around it.

Run: python .claude/skills/skill-overlap/skill_overlap_test.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill_overlap as so  # noqa: E402

# Two near-identical bodies: an original and a would-be-merged fork of it. Same
# vocabulary, so their body cosine similarity is very high.
TWINS_BODY = """
# Cache reuse floor gate

Guard the prompt-cache reuse ratio against regressions. Read the cache ledger,
compute the reuse ratio per session, and fail the gate when the ratio falls
below the configured floor. Report the offending sessions and the floor value.
"""

TWINS_FORK_BODY = """
# Cache reuse floor guard

Guard the prompt-cache reuse ratio against regressions. Read the cache ledger,
compute the reuse ratio for each session, and fail the gate when the ratio drops
below the configured floor. Report the offending sessions with the floor value.
"""

# Two unrelated skills with disjoint vocabularies — neither should pair with
# anything above the threshold.
BROWSER_BODY = """
# Browser action mediation

Drive a headless browser through a mediated action queue: click, type, scroll,
screenshot. Every navigation is gated by an allowlist and every download lands
in a quarantine directory the operator reviews before anything is trusted.
"""

RELEASE_BODY = """
# Versioned release cut

Bump the VERSION file, draft the changelog notes from merged pull requests, tag
the release commit, and publish the artifact bundle. The ordering gotchas are
enforced by refusing a cut whose working tree is dirty outside the bump.
"""


class SkillOverlapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="skill-overlap-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def make_skill(self, name: str, body: str, *, description: str = "test skill") -> Path:
        d = self.tmp / name
        d.mkdir(parents=True)
        fm = "---\nname: %s\ndescription: %s\n---\n%s" % (name, description, body)
        (d / "SKILL.md").write_text(fm, encoding="utf-8")
        return d

    def seed_pack(self):
        self.make_skill("cache-floor", TWINS_BODY)
        self.make_skill("cache-floor-fork", TWINS_FORK_BODY)
        self.make_skill("browser-mediate", BROWSER_BODY)
        self.make_skill("release-cut", RELEASE_BODY)

    # -- the issue's named witness ---------------------------------------

    def test_overlapping_pair_surfaced_unrelated_not(self):
        self.seed_pack()
        pairs = so.overlap_pairs(self.tmp, threshold=so.DEFAULT_THRESHOLD)

        surfaced = {frozenset((p["a"], p["b"])) for p in pairs}
        self.assertIn(frozenset(("cache-floor", "cache-floor-fork")), surfaced,
                      "the deliberately-overlapping pair must be flagged as a merge candidate")

        # No unrelated skill is dragged into a candidate pair.
        flagged_names = {n for pair in surfaced for n in pair}
        self.assertNotIn("browser-mediate", flagged_names)
        self.assertNotIn("release-cut", flagged_names)
        self.assertEqual(len(pairs), 1, "only the twin pair should clear the threshold")

        top = pairs[0]
        self.assertGreater(top["similarity"], so.DEFAULT_THRESHOLD)
        self.assertLessEqual(top["similarity"], 1.0)
        # The 'why' is surfaced for the human deciding the merge.
        self.assertIn("cache", top["shared_top"])

    def test_unrelated_pair_below_threshold(self):
        self.seed_pack()
        vecs = so.load_vectors(self.tmp)
        sim = so.cosine(vecs["browser-mediate"], vecs["release-cut"])
        self.assertLess(sim, so.DEFAULT_THRESHOLD,
                        "two unrelated skills must score below the merge threshold")

    # -- front-matter is not counted --------------------------------------

    def test_frontmatter_shape_does_not_inflate_similarity(self):
        # Identical YAML skeleton, totally different bodies: must NOT pair.
        self.make_skill("alpha", BROWSER_BODY, description="a skill that does alpha things")
        self.make_skill("beta", RELEASE_BODY, description="a skill that does beta things")
        pairs = so.overlap_pairs(self.tmp, threshold=so.DEFAULT_THRESHOLD)
        self.assertEqual(pairs, [], "shared front-matter shape must not create a false candidate")

    def test_body_of_strips_frontmatter(self):
        text = "---\nname: x\ndescription: y\n---\n# Heading\n\nreal body text\n"
        body = so.body_of(text)
        self.assertNotIn("description:", body)
        self.assertIn("real body text", body)

    # -- proposal-only + determinism + hygiene ----------------------------

    def test_detector_never_mutates_the_pack(self):
        self.seed_pack()
        before = sorted(p.relative_to(self.tmp).as_posix() for p in self.tmp.rglob("*"))
        so.overlap_pairs(self.tmp, threshold=so.DEFAULT_THRESHOLD)
        # main() with the default (human) output path too.
        so.main(["--skills-root", str(self.tmp)])
        so.main(["--skills-root", str(self.tmp), "--json"])
        after = sorted(p.relative_to(self.tmp).as_posix() for p in self.tmp.rglob("*"))
        self.assertEqual(before, after, "the detector is proposal-only: it must move/create/delete nothing")

    def test_deterministic_sorted_by_similarity(self):
        self.make_skill("a-twin", TWINS_BODY)
        self.make_skill("b-twin", TWINS_FORK_BODY)
        self.make_skill("c-twin", TWINS_BODY)  # a third near-identical body
        p1 = so.overlap_pairs(self.tmp, threshold=so.DEFAULT_THRESHOLD)
        p2 = so.overlap_pairs(self.tmp, threshold=so.DEFAULT_THRESHOLD)
        self.assertEqual(p1, p2, "same pack must yield the same candidate list")
        sims = [p["similarity"] for p in p1]
        self.assertEqual(sims, sorted(sims, reverse=True), "candidates must be sorted by similarity desc")

    def test_archived_and_dotdir_skills_skipped(self):
        self.seed_pack()
        # An archived copy under a dot-dir must never be proposed for a merge.
        arch = self.tmp / ".archive" / "old-skill"
        arch.mkdir(parents=True)
        (arch / "SKILL.md").write_text("---\nname: old-skill\n---\n" + TWINS_BODY, encoding="utf-8")
        names = sorted(n for n, _ in so.discover_skills(self.tmp))
        # The dot-dir skill is skipped entirely; only the four live skills remain.
        self.assertEqual(names,
                         ["browser-mediate", "cache-floor", "cache-floor-fork", "release-cut"])

    def test_empty_root_is_clean(self):
        self.assertEqual(so.overlap_pairs(self.tmp), [])
        self.assertEqual(so.main(["--skills-root", str(self.tmp)]), 0)

    def test_json_mode_exit_zero(self):
        self.seed_pack()
        self.assertEqual(so.main(["--skills-root", str(self.tmp), "--json"]), 0)
        self.assertEqual(so.main(["--skills-root", str(self.tmp), "--top", "1"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
