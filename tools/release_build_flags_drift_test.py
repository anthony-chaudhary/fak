#!/usr/bin/env python3
"""Anti-drift guard for the release build recipe (#3709, epic #3708).

The canonical release `go build` flag set --

    -trimpath -ldflags "-s -w -X <appversion>.BuildVersion=<stamp>"

-- is, until the single build entrypoint lands, hand-copied into three files: the
release-artifacts matrix, the distroless `Dockerfile`, and the cuda `Dockerfile.cuda`.
Nothing asserts the copies agree, so editing one and missing another ships a divergent
release behind a green trunk -- the exact "editing one and missing another produces a
divergent release with no red trunk" failure #3709 exists to close.

This is that ticket's anti-drift acceptance check ("a CI check fails the build if any
consumer hardcodes a divergent flag set"), expressed against the current topology: it
fails the moment any consumer's release ldflags core drifts from the others. It does NOT
own the single-entrypoint de-dup itself (that is the rest of #3709) -- it is the safety
net that makes that de-dup landable without silently regressing a consumer.

When #3709's entrypoint de-dup lands and every consumer sources ONE recipe, collapse
`CONSUMERS` to that single entrypoint; the single-source assertion then holds trivially.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# Every file carrying an inline release `go build ... -ldflags` today. Keep this the
# exhaustive set the ticket enumerates so a dropped/renamed consumer surfaces as a
# missing-file failure rather than a silently-narrowed check.
CONSUMERS = {
    "release-artifacts.yml": ROOT / ".github" / "workflows" / "release-artifacts.yml",
    "Dockerfile": ROOT / "Dockerfile",
    "Dockerfile.cuda": ROOT / "Dockerfile.cuda",
}

APPVERSION_STAMP = "github.com/anthony-chaudhary/fak/internal/appversion.BuildVersion="

# -ldflags "<...>" -- the release flag core that must be single-sourced.
_LDFLAGS = re.compile(r'-ldflags\s+"([^"]*)"')
# The stamp VALUE is a per-consumer shell var (${VERSION} in the workflow vs
# ${APP_VERSION} in the Dockerfiles); that difference is legitimate, so normalize it
# out before comparing the cores.
_STAMP_VALUE = re.compile(r'(BuildVersion=)\S+')


def _release_ldflags_core(text: str) -> str | None:
    """The normalized release ldflags string, or None if this file has no release build."""
    for m in _LDFLAGS.finditer(text):
        core = m.group(1)
        if APPVERSION_STAMP in core:
            return _STAMP_VALUE.sub(r"\1<STAMP>", core)
    return None


class ReleaseBuildFlagsDriftTest(unittest.TestCase):
    def _cores(self) -> dict[str, str]:
        cores: dict[str, str] = {}
        for name, path in CONSUMERS.items():
            self.assertTrue(path.exists(), f"{name} missing at {path}")
            core = _release_ldflags_core(path.read_text(encoding="utf-8"))
            self.assertIsNotNone(
                core,
                f'{name} has no release `-ldflags "... appversion.BuildVersion=..."`',
            )
            cores[name] = core  # type: ignore[assignment]  # asserted non-None above
        return cores

    def test_every_consumer_stamps_build_version(self) -> None:
        # Non-vacuous: the scan must actually find the stamp in each consumer, so the
        # single-source assertion below can never pass by matching nothing.
        cores = self._cores()
        self.assertEqual(set(cores), set(CONSUMERS))

    def test_release_ldflags_are_single_sourced(self) -> None:
        # The drift guard: the normalized release ldflags core is identical across every
        # consumer. Edit one file's `-s -w -X ...` set and miss another -> this goes red.
        cores = self._cores()
        distinct = set(cores.values())
        self.assertEqual(
            len(distinct),
            1,
            "release ldflags drift across build consumers: "
            + "; ".join(f"{n}={c!r}" for n, c in sorted(cores.items())),
        )
        (canonical,) = distinct
        self.assertIn("-s -w", canonical)
        self.assertIn(APPVERSION_STAMP + "<STAMP>", canonical)

    def test_all_consumers_keep_trimpath(self) -> None:
        # -trimpath is part of the reproducible release recipe; it must not drift either.
        for name, path in CONSUMERS.items():
            self.assertIn(
                "-trimpath",
                path.read_text(encoding="utf-8"),
                f"{name} release build dropped -trimpath",
            )

    def test_profile_cgo_and_tags(self) -> None:
        # The static profile (release archive + distroless) builds CGO_ENABLED=0; the
        # cuda profile is the one sanctioned cgo path (CGO_ENABLED=1 -tags cuda). This
        # pins the profile-specific flags that legitimately DIFFER, so a drift check can
        # not be satisfied by flattening cuda onto the static recipe.
        rel = CONSUMERS["release-artifacts.yml"].read_text(encoding="utf-8")
        docker = CONSUMERS["Dockerfile"].read_text(encoding="utf-8")
        cuda = CONSUMERS["Dockerfile.cuda"].read_text(encoding="utf-8")
        self.assertRegex(rel, r'CGO_ENABLED:\s*"?0"?')
        self.assertRegex(docker, r"CGO_ENABLED=0")
        self.assertRegex(cuda, r"CGO_ENABLED=1")
        self.assertIn("-tags cuda", cuda)
        self.assertNotIn("-tags cuda", docker)


if __name__ == "__main__":
    unittest.main()
