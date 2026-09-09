#!/usr/bin/env python3
"""Anti-drift guard for the release build recipe (#3709, epic #3708, #12490).

The canonical release `go build` flag set --

    -trimpath -ldflags "-s -w -X <appversion>.BuildVersion=<stamp>"

-- is owned by the canonical build entrypoint scripts/build.sh (#3709). Every shipping
consumer (the release-artifacts matrix, the distroless `Dockerfile`, and the cuda
`Dockerfile.cuda`) routes through scripts/build.sh rather than duplicating the flag set.

This test verifies the real ownership boundary: consumers invoke the canonical entrypoint,
direct inline duplicate flags are not required, and the entrypoint owns trimpath and
BuildVersion stamping.
"""
from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUILD_SH = ROOT / "scripts" / "build.sh"

CONSUMERS = {
    "release-artifacts.yml": ROOT / ".github" / "workflows" / "release-artifacts.yml",
    "Dockerfile": ROOT / "Dockerfile",
    "Dockerfile.cuda": ROOT / "Dockerfile.cuda",
}

APPVERSION_STAMP = "github.com/anthony-chaudhary/fak/internal/appversion.BuildVersion="
CANONICAL_ENTRYPOINT = "scripts/build.sh"


def verify_consumer_routes_through_entrypoint(name: str, text: str) -> None:
    """Assert a consumer routes through scripts/build.sh and does not re-inline release flags."""
    if CANONICAL_ENTRYPOINT not in text:
        raise AssertionError(f"{name} bypasses {CANONICAL_ENTRYPOINT}")
    if APPVERSION_STAMP in text:
        raise AssertionError(f"{name} re-inlines duplicate release stamp {APPVERSION_STAMP}")


def verify_canonical_entrypoint(text: str) -> None:
    """Assert the canonical build entrypoint owns trimpath, strip, and BuildVersion stamping."""
    if "-trimpath" not in text:
        raise AssertionError(f"{CANONICAL_ENTRYPOINT} dropped -trimpath")
    if APPVERSION_STAMP not in text and "appversion.BuildVersion=" not in text:
        raise AssertionError(f"{CANONICAL_ENTRYPOINT} dropped BuildVersion stamping")
    if "-s -w" not in text:
        raise AssertionError(f"{CANONICAL_ENTRYPOINT} dropped release strip flags -s -w")


class ReleaseBuildFlagsDriftTest(unittest.TestCase):
    def test_all_consumers_route_through_canonical_entrypoint(self) -> None:
        # Every consumer must route through the canonical entrypoint (scripts/build.sh).
        # Direct inline duplicate flags are NOT required.
        for name, path in CONSUMERS.items():
            self.assertTrue(path.exists(), f"{name} missing at {path}")
            text = path.read_text(encoding="utf-8")
            verify_consumer_routes_through_entrypoint(name, text)

        # Profile-specific flags: static profiles (release archive + distroless)
        # build CGO_ENABLED=0; cuda profile builds CGO_ENABLED=1 -tags cuda.
        rel = CONSUMERS["release-artifacts.yml"].read_text(encoding="utf-8")
        docker = CONSUMERS["Dockerfile"].read_text(encoding="utf-8")
        cuda = CONSUMERS["Dockerfile.cuda"].read_text(encoding="utf-8")
        self.assertRegex(rel, r'CGO_ENABLED:\s*"?0"?')
        self.assertRegex(docker, r"CGO_ENABLED=0")
        self.assertRegex(cuda, r"CGO_ENABLED=1")
        self.assertIn("-tags cuda", cuda)
        self.assertNotIn("-tags cuda", docker)

    def test_canonical_entrypoint_owns_release_flags(self) -> None:
        # The single source of the trim/strip/stamp flags for the release binary.
        self.assertTrue(BUILD_SH.exists(), f"missing {BUILD_SH}")
        text = BUILD_SH.read_text(encoding="utf-8")
        verify_canonical_entrypoint(text)
        self.assertIn("go build", text)
        self.assertIn("./cmd/fak", text)

    def test_consumer_bypass_fixture_fails(self) -> None:
        # A consumer that bypasses scripts/build.sh must fail validation.
        bypass_fixture = "go build -trimpath -o /out/fak ./cmd/fak"
        with self.assertRaises(AssertionError):
            verify_consumer_routes_through_entrypoint("bypass_fixture", bypass_fixture)

        # A consumer that re-inlines duplicate release stamp must fail validation.
        reinlined_fixture = f"sh scripts/build.sh\n-ldflags \"{APPVERSION_STAMP}1.0.0\""
        with self.assertRaises(AssertionError):
            verify_consumer_routes_through_entrypoint("reinlined_fixture", reinlined_fixture)

    def test_canonical_flag_removal_fixture_fails(self) -> None:
        canonical_text = BUILD_SH.read_text(encoding="utf-8")

        # Dropping -trimpath must fail.
        without_trimpath = canonical_text.replace("-trimpath", "")
        with self.assertRaises(AssertionError):
            verify_canonical_entrypoint(without_trimpath)

        # Dropping BuildVersion stamping must fail.
        without_stamp = canonical_text.replace("BuildVersion=", "UnusedFlag=")
        with self.assertRaises(AssertionError):
            verify_canonical_entrypoint(without_stamp)

        # Dropping strip flags must fail.
        without_strip = canonical_text.replace("-s -w", "")
        with self.assertRaises(AssertionError):
            verify_canonical_entrypoint(without_strip)


if __name__ == "__main__":
    unittest.main()
