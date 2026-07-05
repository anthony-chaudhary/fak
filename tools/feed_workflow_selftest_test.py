#!/usr/bin/env python3
"""Contract test: no Slack *-feed.yml gates its post on the whole cmd/fak test package.

This pins the fix from `bc76e06f` (Refs #2583). The #cache-value cadence went
~58h STALE because `cachevalue-feed.yml` self-tested with `go test ./cmd/fak
-run TestCachevalue`. `-run` only filters which tests EXECUTE — Go still COMPILES
the entire, heavily-developed `cmd/fak` TEST package first, so an unrelated
in-flight compile break in ANY `cmd/fak/*_test.go` file failed the step, failed
the job, and SILENTLY dropped the daily post with perfectly green cachevalue code.

The durable rule the fix established, and this test guards against regression:

  A daily Slack feeder must never gate its outward post on `go test ./cmd/fak`
  (or `./cmd/fak/...`). Narrow package targets (`internal/...`) are fine; the
  `go run ./cmd/fak <verb> --dry-run` render step is the real end-to-end smoke.

A future edit re-introducing a broad `cmd/fak` test gate into ANY `*-feed.yml`
would re-arm the exact silent-drop fragility — this test fails loudly instead.

Hermetic: reads workflow YAML text only. No gh, no Slack, no subprocess.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"

# `go test` targeting the broad cmd/fak TEST package: `./cmd/fak` itself, or the
# recursive `./cmd/fak/...`. A narrow sub-package like `./cmd/fak/internal/x` is
# NOT this pattern (it compiles only that leaf's test files), so we anchor on the
# package token ending right after `cmd/fak` (end, whitespace, or the `/...` glob).
_BROAD_CMD_FAK_TEST = re.compile(r"\bgo\s+test\b[^\n]*\.\/cmd\/fak(?:\/\.\.\.)?(?:\s|$)")


def _strip_yaml_comments(text: str) -> str:
    """Drop full-line YAML comments so the prose in cachevalue-feed.yml — which
    deliberately QUOTES `go test ./cmd/fak` to explain why it is banned — does
    not read as a live command. Inline `#` after code is left alone (rare in run
    bodies and risky to strip against `#channel` literals)."""
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


def _feed_workflows() -> list[Path]:
    files = sorted(WORKFLOWS.glob("*-feed.yml"))
    assert files, f"no *-feed.yml workflows found under {WORKFLOWS}"
    return files


class FeedWorkflowSelfTestContract(unittest.TestCase):
    def test_no_feeder_gates_post_on_whole_cmd_fak_test_package(self) -> None:
        offenders = []
        for wf in _feed_workflows():
            body = _strip_yaml_comments(wf.read_text(encoding="utf-8"))
            for m in _BROAD_CMD_FAK_TEST.finditer(body):
                offenders.append((wf.name, m.group(0).strip()))
        self.assertEqual(
            offenders,
            [],
            "A Slack feeder must not gate its post on the whole cmd/fak test "
            "package (compiles every cmd/fak test file — an unrelated break "
            "silently drops the post; see #2583 / bc76e06f). Offenders: "
            + "; ".join(f"{name}: {cmd!r}" for name, cmd in offenders),
        )

    def test_cachevalue_feeder_keeps_the_narrow_selftest(self) -> None:
        # The fix replaced the broad gate with a package-scoped self-test. Pin
        # that the narrow form is still present, so a future edit cannot quietly
        # drop the self-test to zero coverage while satisfying the ban above.
        wf = WORKFLOWS / "cachevalue-feed.yml"
        body = _strip_yaml_comments(wf.read_text(encoding="utf-8"))
        self.assertIn(
            "go test ./internal/cachevaluereport ./internal/cachevaluepost",
            body,
            "cachevalue-feed.yml lost its narrow cachevalue-package self-test",
        )

    def test_guard_regex_catches_the_regressed_form(self) -> None:
        # Guard the guard: the exact line that caused the 58h STALE must match,
        # and the narrow replacement must NOT — otherwise the test is inert.
        self.assertRegex("          go test ./cmd/fak -run TestCachevalue", _BROAD_CMD_FAK_TEST)
        self.assertRegex("          go test ./cmd/fak/...", _BROAD_CMD_FAK_TEST)
        self.assertNotRegex(
            "          go test ./internal/cachevaluereport ./internal/cachevaluepost",
            _BROAD_CMD_FAK_TEST,
        )
        self.assertNotRegex("          go test ./cmd/fak/internal/leaf", _BROAD_CMD_FAK_TEST)


if __name__ == "__main__":
    unittest.main()
