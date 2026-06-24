#!/usr/bin/env python3
"""Hermetic tests for tools/scrub_hardware_names.py — the prose hardware-name leak gate.

This is a load-bearing CI gate (the PUBLIC_LEAK floor family): it must rewrite the
operator's private lab GPU names out of DOC PROSE while leaving every CODE/DATA
IDENTIFIER that contains 'dgx'/'a100' untouched. A regression either leaks a private
name or corrupts an identifier (breaking the build / bench-data joins). These tests
pin that prose/identifier boundary; the pure transforms need no git/files.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "scrub_hardware_names.py"


def load():
    spec = importlib.util.spec_from_file_location("scrub_hardware_names", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()


class ProseRewriteTest(unittest.TestCase):
    def test_bare_dgx_prose_rewritten(self):
        self.assertEqual(m.transform("ran on the DGX box\n"), "ran on the GPU server box\n")

    def test_multi_gpu_phrase_rewritten(self):
        self.assertEqual(m.transform("8×A100-SXM4-40GB server\n"),
                         "8-GPU datacenter server server\n")

    def test_bare_a100_prose_rewritten_when_no_competitor(self):
        self.assertEqual(m.transform("measured on the A100 here\n"),
                         "measured on the datacenter GPU here\n")


class IdentifierPreservationTest(unittest.TestCase):
    def test_inline_code_identifier_untouched(self):
        # `cmd/dgxbridge` is an identifier in a code span -> never rewritten.
        self.assertEqual(m.transform("see `cmd/dgxbridge` for the bridge\n"),
                         "see `cmd/dgxbridge` for the bridge\n")

    def test_uppercase_underscore_identifier_untouched(self):
        # \bDGX\b does NOT match inside FAK_DGX_REQ_ (underscore is a word char).
        self.assertEqual(m.transform("the FAK_DGX_REQ_ marker\n"),
                         "the FAK_DGX_REQ_ marker\n")

    def test_lowercase_dgx_never_matched(self):
        self.assertEqual(m.transform("the dgxbridge command\n"), "the dgxbridge command\n")

    def test_fenced_code_block_passes_through(self):
        src = "```\nDGX A100\n```\n"
        self.assertEqual(m.transform(src), src)


class CompetitorCitationGuardTest(unittest.TestCase):
    def test_bare_a100_kept_on_competitor_line(self):
        # A third-party citation legitimately keeps 'A100' -> must NOT be scrubbed.
        line = "Sarathi-Serve runs on 1xA100 with A100 memory\n"
        self.assertEqual(m.transform(line), line)


class ResidualHitsTest(unittest.TestCase):
    def test_detects_prose_dgx(self):
        hits = m.residual_hits("intro\nran on the DGX box\nend")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0], 2)  # line number

    def test_ignores_code_span_and_fence(self):
        self.assertEqual(m.residual_hits("use `dgxbridge` here"), [])
        self.assertEqual(m.residual_hits("```\nDGX\n```"), [])


class CleanupTest(unittest.TestCase):
    def test_no_doubled_gpu_server(self):
        # '8×A100 DGX' rewrites in two passes; cleanup collapses the duplicate.
        out = m.transform("8×A100 DGX cluster\n")
        self.assertNotIn("GPU server GPU server", out)
        self.assertNotIn("datacenter server GPU server", out)


if __name__ == "__main__":
    unittest.main()
