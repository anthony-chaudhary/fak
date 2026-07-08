#!/usr/bin/env python3
"""Hermetic tests for the cache-headline gate (#1564, Next-50 item 46).

Proves the issue's default/evidence target: a cache headline ("99% cache" /
"cache win") WITHOUT a provider/kernel/context plane + provenance label FAILS the
lint, while a plane+provenance-labeled headline PASSES — and the honest carve-outs
(the legacy phrasing quoted to remove it, a genuine "cache window", a hyphenated
hit-rate stat) are not flagged. No real repo is touched — each case is a throwaway
git tree in a tempdir.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_cache_headlines as cch  # noqa: E402


def _scan(text_by_file: dict[str, str]) -> int:
    """Write files into a throwaway git repo, run --audit-tree, return exit code."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init", "-q"], cwd=d, check=True)
        for name, body in text_by_file.items():
            (Path(d) / name).write_text(body, encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=d, check=True,
                       capture_output=True)
        return cch.main(["--audit-tree", "--root", d])


def _commit_all(root: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c", "user.name=cache-headline-test",
            "-c", "user.email=cache-headline-test@example.invalid",
            "commit", "-q", "-m", "seed",
        ],
        cwd=root, check=True, capture_output=True,
    )


class CacheHeadlineGate(unittest.TestCase):
    # ---- the issue's default/evidence target: unlabeled headline FAILS ----

    def test_flags_bare_99pct_cache(self):
        # the exact legacy shape: a blended "99% cache" with no plane named
        self.assertEqual(
            _scan({"bad.md": "We hit a 99% cache — the biggest win of the quarter!\n"}),
            1, "a bare '99% cache' headline without a plane label must FAIL")

    def test_flags_bare_cache_win(self):
        self.assertEqual(
            _scan({"b.md": "Ship it: this is a huge cache win.\n"}),
            1, "a bare 'cache win' headline without a plane label must FAIL")

    def test_flags_cache_is_99pct_of_the_story(self):
        # the DEFAULT-ENABLEMENT-NEXT-50.md legacy assumption phrasing, ASSERTED
        self.assertEqual(
            _scan({"s.md": "Honestly the cache is 99% of the speedup story here.\n"}),
            1, "'cache is 99% …' asserted without a plane label must FAIL")

    # ---- a plane+provenance-labeled headline PASSES ----

    def test_passes_provider_labeled(self):
        self.assertEqual(
            _scan({"ok.md": "provider prompt-cache rebate: 99% cache read, OBSERVED "
                            "(cost/latency only, not fak-owned reuse)\n"}),
            0, "a headline naming the provider plane + OBSERVED provenance is honest")

    def test_passes_kernel_labeled(self):
        self.assertEqual(
            _scan({"ok.md": "kernel KV reuse was the cache win here — WITNESSED, "
                            "bit-identical to a full re-prefill\n"}),
            0, "a headline naming the kernel plane + WITNESSED provenance is honest")

    def test_passes_context_labeled(self):
        self.assertEqual(
            _scan({"ok.md": "O(1) context saved 99% cache prefill work (WITNESSED "
                            "resident-view elision)\n"}),
            0, "a headline naming the context plane is honest")

    # ---- honest carve-outs are not flagged ----

    def test_passes_legacy_quote_to_remove(self):
        # the doc's own "Legacy Assumptions To Remove" list quotes the phrasing
        self.assertEqual(
            _scan({"legacy.md": '- "cache win" as a bare headline is legacy language '
                                "to remove.\n"}),
            0, "a line quoting the phrasing to REMOVE it is not asserting it")

    def test_passes_cache_window_word_boundary(self):
        # "cache window/windowed" is a real term, not "cache win"
        self.assertEqual(
            _scan({"w.md": "the bounded cache window is argmax-identical to the "
                           "full-cache windowed decode\n"}),
            0, "'cache window(ed)' must not be read as 'cache win'")

    def test_passes_hyphenated_hit_rate_stat(self):
        # "99%-cache-hit" is a specific hit-rate stat, not a blended win headline
        self.assertEqual(
            _scan({"h.md": "the corpus had two ~99%-cache-hit sessions in the tail\n"}),
            0, "a hyphenated hit-rate stat is out of scope")

    def test_clean_tree_is_zero(self):
        self.assertEqual(
            _scan({"plain.md": "fak treats the model like an untrusted program.\n"}),
            0)

    # ---- staged mode: only NEW additions are judged ----

    def test_audit_staged_flags_added_bad_line(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q"], cwd=d, check=True)
            path = Path(d) / "bad.md"
            path.write_text("plain line\n", encoding="utf-8")
            _commit_all(d)

            path.write_text("plain line\na fresh 99% cache win, the quarter's best\n",
                            encoding="utf-8")
            subprocess.run(["git", "add", "bad.md"], cwd=d, check=True,
                           capture_output=True)

            self.assertEqual(cch.main(["--audit-staged", "--root", d]), 1)

    def test_audit_staged_ignores_preexisting_bad_line(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q"], cwd=d, check=True)
            path = Path(d) / "legacy.md"
            path.write_text("a bare 99% cache win headline\n", encoding="utf-8")
            _commit_all(d)

            path.write_text("a bare 99% cache win headline\nnew clean line\n",
                            encoding="utf-8")
            subprocess.run(["git", "add", "legacy.md"], cwd=d, check=True,
                           capture_output=True)

            self.assertEqual(cch.main(["--audit-staged", "--root", d]), 0)


if __name__ == "__main__":
    unittest.main()
