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

import re
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


class ProfileSelectorTest(unittest.TestCase):
    """#3710 — the one recipe grew named dev/race profiles beside the shipped release
    profile. These pin the profile posture so a debuggable `make build` cannot silently
    regress into a stripped build (or the reverse)."""

    def setUp(self) -> None:
        self.text = BUILD_SH.read_text(encoding="utf-8")
        # Code only: the header comment documents every profile, so posture assertions
        # must look at the actual `case` branches, not the prose that describes them.
        self.code = "\n".join(
            ln for ln in self.text.splitlines() if not ln.lstrip().startswith("#")
        )

    def test_implements_the_three_profiles(self) -> None:
        # Each profile must be a selectable `case` branch, not just documented in prose.
        for branch in ("release)", "dev)", "race)"):
            self.assertIn(branch, self.code,
                          f"build.sh must implement the {branch[:-1]} profile as a case branch")

    def test_strip_is_release_only(self) -> None:
        # The strip flags belong to the SHIPPED profile only; a debuggable `make build`
        # must keep DWARF. Exactly one code occurrence of `-s -w` (the release branch).
        self.assertEqual(self.code.count("-s -w"), 1,
                         "the strip flags -s -w must appear once, in the release profile only")

    def test_trimpath_is_release_only(self) -> None:
        # -trimpath removes host paths (reproducible ship) but also the source paths a
        # debugger needs — so it, too, is release-only. One code occurrence.
        self.assertEqual(self.code.count("-trimpath"), 1,
                         "-trimpath must appear once, in the release profile only")

    def test_race_profile_passes_the_detector(self) -> None:
        self.assertIn("-race", self.code, "the race profile must pass -race to go build")

    def test_makefile_routes_dev_and_race_through_the_script(self) -> None:
        # `make build` (dev) and `make build-race` (race) must both select a profile via
        # the entrypoint, never re-type a bare `go build -o fak ./cmd/fak`.
        mk = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("PROFILE=dev sh scripts/build.sh", mk,
                      "`make build` must emit ./fak through the dev profile of the entrypoint")
        self.assertIn("PROFILE=race sh scripts/build.sh", mk,
                      "`make build-race` must build through the race profile of the entrypoint")


class MakeBuildGraphTest(unittest.TestCase):
    """#9672 — keep the artifact build narrow without weakening the whole-tree CI gate."""

    def setUp(self) -> None:
        self.text = (ROOT / "Makefile").read_text(encoding="utf-8")

    def header_prerequisites(self, target: str) -> list[str]:
        match = re.search(rf"(?m)^{re.escape(target)}:(.*)$", self.text)
        self.assertIsNotNone(match, f"Makefile is missing target {target!r}")
        return match.group(1).split()

    def recipe(self, target: str) -> str:
        lines = self.text.splitlines()
        header = f"{target}:"
        start = next(
            (index for index, line in enumerate(lines) if line.startswith(header)),
            None,
        )
        self.assertIsNotNone(start, f"Makefile is missing target {target!r}")
        body: list[str] = []
        for line in lines[start + 1:]:
            if line.startswith("\t"):
                body.append(line[1:])
                continue
            if not line.strip() and body:
                break
            if line.strip() and not line.lstrip().startswith("#"):
                break
        return "\n".join(body)

    def test_build_emits_only_named_operator_artifacts(self) -> None:
        recipe = self.recipe("build")
        self.assertNotIn("go build ./...", recipe)
        self.assertIn("OUT=fak PROFILE=dev sh scripts/build.sh", recipe)
        self.assertIn("-o tools/.bin/repoguard ./cmd/repoguard", recipe)
        self.assertIn("-o tools/.bin/repoguard.exe ./cmd/repoguard", recipe)
        self.assertIn("-o tools/.bin/dispatchworker ./cmd/dispatchworker", recipe)

    def test_build_all_owns_the_whole_tree_compile(self) -> None:
        self.assertIn("build-all", self.header_prerequisites(".PHONY"))
        self.assertEqual(self.recipe("build-all"), "go build ./...")

    def test_ci_keeps_both_build_contracts_direct(self) -> None:
        prerequisites = self.header_prerequisites("ci")
        self.assertIn("build", prerequisites)
        self.assertIn("build-all", prerequisites)


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
