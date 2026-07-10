#!/usr/bin/env python3
"""Anti-drift contract for the one canonical `fak` build entrypoint (issue #3709, epic #3708).

The release stamp recipe — `go build -trimpath -ldflags "-s -w -X ...BuildVersion=…"
-o … ./cmd/fak` — used to be copy-pasted into every build site: both Dockerfiles and
the release-artifacts workflow. Three copies means three chances to drift (a dropped
-trimpath, a diverged ldflag, a renamed output). C1 collapses them onto
scripts/build.sh; these tests red the moment a consumer grows its own inline copy of
the stamp again instead of routing through the script — WITHOUT building anything.
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_SH = ROOT / "scripts" / "build.sh"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# The four build sites the epic DRYs onto scripts/build.sh. Each must route through
# the script; none may re-inline the stamp recipe.
CONSUMERS = {
    "Dockerfile": ROOT / "Dockerfile",
    "Dockerfile.cuda": ROOT / "Dockerfile.cuda",
    "release-artifacts.yml": ROOT / ".github" / "workflows" / "release-artifacts.yml",
    "Makefile": ROOT / "Makefile",
}

# The fak release stamp uniquely identifies the recipe: a bare `go build -trimpath`
# for some other cmd/ target (turntaxdemo, repoguard, …) is NOT it — the BuildVersion
# ldflag is. Keying on the ldflag keeps the guard specific to the release build.
STAMP = "appversion.BuildVersion="


class BuildEntrypointExistsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(BUILD_SH.exists(), f"missing {BUILD_SH}")
        self.text = BUILD_SH.read_text(encoding="utf-8")

    def test_carries_the_canonical_recipe(self) -> None:
        # The single source of the trim/strip/stamp flags for the release binary.
        self.assertIn("go build", self.text)
        self.assertIn("-trimpath", self.text)
        self.assertIn(STAMP, self.text)
        self.assertIn("./cmd/fak", self.text)

    def test_is_parameterized_by_env(self) -> None:
        # Callers vary these; a missed knob silently changes a shipped binary's name,
        # stamp, or tag set, so pin that the seams the consumers rely on exist.
        for knob in ("OUT", "VERSION", "TAGS"):
            self.assertIn(knob, self.text, f"build.sh must honor ${knob}")

    def test_cgo_agnostic(self) -> None:
        # The script must NOT pin CGO_ENABLED: the static builds set 0 and the cuda
        # build sets 1, each via its own env. Pinning it here would break one path.
        # Check code only — the header comment legitimately names both values.
        code = "\n".join(
            ln for ln in self.text.splitlines() if not ln.lstrip().startswith("#")
        )
        self.assertNotIn("CGO_ENABLED=", code)


class NoInlineRecipeDriftTest(unittest.TestCase):
    def test_consumers_route_through_the_script(self) -> None:
        for name, path in CONSUMERS.items():
            self.assertTrue(path.exists(), f"missing {name}")
            self.assertIn(
                "scripts/build.sh",
                path.read_text(encoding="utf-8"),
                f"{name} must route through scripts/build.sh, not inline the recipe",
            )

    def test_no_consumer_reinlines_the_stamp(self) -> None:
        # The stamp ldflag must live ONLY in scripts/build.sh among the build sites.
        for name, path in CONSUMERS.items():
            self.assertNotIn(
                STAMP,
                path.read_text(encoding="utf-8"),
                f"{name} re-inlined the stamp recipe; route it through scripts/build.sh (#3709)",
            )

    def test_wired_into_ci(self) -> None:
        # This guard is only worth anything if CI runs it — pin its own registration
        # so it can never silently become a test blackhole.
        ci = CI_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("python tools/build_entrypoint_test.py", ci)


if __name__ == "__main__":
    unittest.main()
