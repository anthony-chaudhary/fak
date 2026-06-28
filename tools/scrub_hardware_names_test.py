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


class SkuPhraseAccuracyTest(unittest.TestCase):
    """Accuracy: the uppercase SKU/machine forms that previously garbled — the -80GB size
    variant, the 8×DGX-A100-NNGB form, DGX-<word> hyphenated prose, and the "an X" article
    — now soften to clean prose instead of half-rewritten fragments."""

    def test_sxm4_80gb_size_variant(self):
        # previously only -40GB was listed → "-80GB" leaked behind "datacenter GPU".
        self.assertEqual(m.transform("ran on an A100-SXM4-80GB box\n"),
                         "ran on a datacenter GPU box\n")

    def test_dgx_a100_count_phrase(self):
        # "8×DGX-A100-40GB" previously became "8×GPU server-datacenter GPU" (garbled).
        self.assertEqual(m.transform("on 8×DGX-A100-40GB nodes\n"),
                         "on 8-GPU datacenter server nodes\n")

    def test_markdown_bold_size_suffix(self):
        # the DGX-OVERNIGHT table writes the size in bold: 8×A100-SXM4-**80GB**.
        self.assertEqual(m.transform("| 8×A100-SXM4-**80GB**, 886 GiB |\n"),
                         "| 8-GPU datacenter server, 886 GiB |\n")

    def test_bare_a100_size_variant(self):
        # "A100-80GB" (no SXM4) previously left "datacenter GPU-80GB".
        self.assertEqual(m.transform("(GPU server, A100-80GB, sm_80)\n"),
                         "(GPU server, datacenter GPU, sm_80)\n")

    def test_heals_prior_half_scrub_leftover(self):
        # an earlier scrub baked "datacenter GPU-80GB" into committed docs; a re-run heals
        # the leftover tail rather than preserving it.
        self.assertEqual(m.transform("8× datacenter GPU-80GB, sm_80\n"),
                         "8× datacenter GPU, sm_80\n")

    def test_dgx_hyphenated_word(self):
        # "DGX-fleet" previously became "GPU server-fleet"; now "GPU server fleet".
        self.assertEqual(m.transform("the DGX-fleet readiness\n"),
                         "the GPU server fleet readiness\n")

    def test_article_demoted_for_consonant_replacement(self):
        self.assertEqual(m.transform("an A100 here\n"), "a datacenter GPU here\n")
        # sentence-initial capital preserved.
        self.assertEqual(m.transform("An A100 ran it\n"), "A datacenter GPU ran it\n")

    def test_dgx_overnight_link_id_untouched(self):
        # DGX-OVERNIGHT-PLAN-*.md is a path/link token (masked) — uppercase-suffixed
        # DGX-<UPPER> is NOT touched by the DGX-<lowercase-word> rule either way.
        src = "see [plan](docs/nightrun/DGX-OVERNIGHT-PLAN-2026-06-28.md) for detail\n"
        self.assertEqual(m.transform(src), src)


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


class DgxNLabelTest(unittest.TestCase):
    """The lowercase dgxN (dgx3/dgx2/dgx1) machine-label rule: scrub the bare prose
    label, preserve every identifier form (channel name, filename, FQDN, schema id).
    Each pair is verified against the real mask pipeline (transform), covering every
    token shape in the git-grep inventory."""

    SCRUB = [
        ("ran on dgx3\n", "ran on GPU server\n"),
        ("**dgx3**\n", "**GPU server**\n"),
        ("title: dgx2 cross-engine notes\n", "title: GPU server cross-engine notes\n"),
        ("dgx1\n", "GPU server\n"),
        ("end on dgx3, then halt\n", "end on GPU server, then halt\n"),
        # sentence-final period must still scrub (the (?!\.[A-Za-z0-9]) lookahead lets a
        # bare label at a sentence end through, unlike a naive (?![-.\w])):
        ("deployed to dgx1.\n", "deployed to GPU server.\n"),
        ("ran on dgx1. Next we tried it\n", "ran on GPU server. Next we tried it\n"),
        ("done dgx3.)\n", "done GPU server.)\n"),
        # two-digit / zero forms ("dgxN generally"):
        ("scaled to dgx10 overnight\n", "scaled to GPU server overnight\n"),
        # UPPERCASE forms also occur in prose ("serve GLM-5.2 on DGX3", "DGX3's tier") and
        # are NOT caught by the uppercase-only \bDGX\b rule (no boundary before the digit),
        # so the dgxN rule is case-insensitive:
        ("serve GLM-5.2 on DGX3\n", "serve GLM-5.2 on GPU server\n"),
        ("on DGX3 (8 GPUs)\n", "on GPU server (8 GPUs)\n"),
        ("DGX2 cross-engine notes\n", "GPU server cross-engine notes\n"),
        ("DGX3's host-CPU tier\n", "GPU server's host-CPU tier\n"),
    ]

    PRESERVE = [
        # filenames (mask + lookahead):
        "see glm52_stage_serve_dgx3.sh for the run\n",
        "the dgx3-a100-node-state-2026-06-25.json artifact\n",
        "docs/dgx2-cross-engine-data-2026-06-25.md link\n",
        "dgx3_glm_decode_witness.sh\n",
        # FQDN host shortname (dot lookahead is the sole guard):
        "host dgx1.example.lab is the box\n",
        # schema id (hyphen lookahead):
        "schema dgx3-node-state.v1 binds the join\n",
        # Slack channel names — policy-KEPT (not reachable without workspace+bot token):
        "dgx3-control\n",
        "dgx2-control\n",
        "dgx1-control\n",
        # embedded in a larger identifier (leading \b / alnum-suffix guards):
        "xdgx3\n",
        "fak_dgx3\n",
        "dgx3a\n",
        # uppercase identifier with no digit after DGX — the [0-9]+ requirement skips it:
        "the FAK_DGX_REQ_ marker\n",
        # uppercase channel-name form (lookahead refuses the hyphen, case-insensitively):
        "DGX3-CONTROL\n",
    ]

    def test_bare_label_scrubbed(self):
        for src, want in self.SCRUB:
            self.assertEqual(m.transform(src), want, f"SCRUB {src!r}")

    def test_identifier_forms_preserved(self):
        for src in self.PRESERVE:
            self.assertEqual(m.transform(src), src, f"PRESERVE {src!r}")

    def test_mixed_table_cell(self):
        # bare **dgx3** and a trailing bare dgx3 scrub; the dgx3-control channel survives.
        src = "| **dgx3** | dgx3-control | ran on dgx3 |\n"
        want = "| **GPU server** | dgx3-control | ran on GPU server |\n"
        self.assertEqual(m.transform(src), want)

    def test_residual_lint_silent_on_identifiers(self):
        self.assertEqual(m.residual_hits("| col | dgx3-control | col |"), [])
        self.assertEqual(m.residual_hits("host dgx1.example.lab\n"), [])
        self.assertEqual(m.residual_hits("schema dgx3-node-state.v1\n"), [])

    def test_residual_lint_fires_on_bare_label(self):
        hits = m.residual_hits("we ran the eval on dgx3 today\n")
        self.assertEqual(len(hits), 1)

    def test_post_scrub_lints_clean(self):
        src = "| **dgx3** | dgx3-control | ran on dgx3 |\n"
        self.assertEqual(m.residual_hits(m.transform(src)), [])

    def test_idempotent_over_corpus(self):
        for src, _want in self.SCRUB:
            once = m.transform(src)
            self.assertEqual(m.transform(once), once, f"idempotent {src!r}")
        for src in self.PRESERVE:
            self.assertEqual(m.transform(m.transform(src)), m.transform(src))


class MaskNestingTest(unittest.TestCase):
    """A plain markdown link [text](url) where the link text is ITSELF a bare URL nests the
    URL mask inside the link-target mask. The forward-unmask + \\S+ URL mask used to leak the
    inner placeholder as a literal '\\x000\\x00' (rendered '… 0 '), corrupting the link on a
    first --apply. The \\x00-excluding masks + reverse unmask must leave such lines intact."""

    def test_url_link_text_not_corrupted(self):
        src = "- **Aider docs**: [https://aider.chat](https://aider.chat) — official\n"
        self.assertEqual(m.transform(src), src)

    def test_plain_link_passthrough(self):
        src = "[VS Code](https://code.visualstudio.com) docs\n"
        self.assertEqual(m.transform(src), src)

    def test_scrub_with_link_on_same_line(self):
        # the bare dgx3 scrubs; the trailing link survives untouched.
        self.assertEqual(m.transform("ran on dgx3, see [x](https://x.com/a)\n"),
                         "ran on GPU server, see [x](https://x.com/a)\n")


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
