#!/usr/bin/env python3
"""Tests for llamacpp_vulkan_anchor — the same-GPU llama.cpp reference probe for #462.

The probe shells out to `llama-bench` (needs the RX 7600 + a GGUF), so these tests exercise the
pure-Python surface that has no hardware dependency: parsing llama.cpp's own markdown `t/s`
table, folding it into the anchor shape, the gap arithmetic, and the recorded-anchor record.
One opportunistic live smoke SKIPS when llama-bench / the model are absent."""
import os
import shutil
import unittest
from unittest import mock

import llamacpp_vulkan_anchor as a


# A verbatim capture of the real `llama-bench -o md` run on the RX 7600 (Vulkan0) on 2026-06-23,
# including the load_backend/ggml_vulkan preamble the probe must skip past.
_VULKAN_OUT = """load_backend: loaded Vulkan backend from C:\\...\\ggml-vulkan.dll
ggml_vulkan: Found 2 Vulkan devices:
| model                          |       size |     params | backend    | ngl | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------ | --------------: | -------------------: |
| llama 256M Q8_0                | 136.40 MiB |   134.52 M | Vulkan     |  99 | Vulkan0      |            pp16 |     5126.36 ± 183.37 |
| llama 256M Q8_0                | 136.40 MiB |   134.52 M | Vulkan     |  99 | Vulkan0      |            pp64 |      5056.82 ± 38.13 |
| llama 256M Q8_0                | 136.40 MiB |   134.52 M | Vulkan     |  99 | Vulkan0      |           pp256 |   28274.71 ± 8084.95 |
| llama 256M Q8_0                | 136.40 MiB |   134.52 M | Vulkan     |  99 | Vulkan0      |           pp512 |    33723.30 ± 692.30 |
| llama 256M Q8_0                | 136.40 MiB |   134.52 M | Vulkan     |  99 | Vulkan0      |            tg64 |       609.24 ± 12.28 |

build: 9b260fc9e (9673)
"""


class ParseBenchTable(unittest.TestCase):
    def test_parses_means_skipping_preamble_and_separators(self):
        table = a.parse_bench_table(_VULKAN_OUT)
        self.assertEqual(table["tg64"], 609.24)          # decode mean, before the ± stddev
        self.assertEqual(table["pp512"], 33723.30)
        self.assertEqual(set(table), {"pp16", "pp64", "pp256", "pp512", "tg64"})

    def test_no_table_yields_empty(self):
        self.assertEqual(a.parse_bench_table("ggml_vulkan: Found 0 devices\nno table here"), {})

    def test_anchor_shape_splits_decode_and_prefill(self):
        anchor = a.anchor_from_table(a.parse_bench_table(_VULKAN_OUT))
        self.assertEqual(anchor["decode_tok_s"], 609.24)
        self.assertEqual(anchor["prefill_tok_s"]["pp512"], 33723.30)
        self.assertNotIn("tg64", anchor["prefill_tok_s"])  # decode must not leak into prefill


class GapArithmetic(unittest.TestCase):
    def test_gap_is_anchor_over_fak(self):
        # the headline of #462: llama.cpp Vulkan 609.24 vs fak Vulkan Q8 24.6 -> ~24.8x off.
        self.assertEqual(a.gap_vs(609.24, 24.6), 24.77)

    def test_gap_undefined_when_fak_zero_or_anchor_missing(self):
        self.assertIsNone(a.gap_vs(609.24, 0.0))
        self.assertIsNone(a.gap_vs(None, 24.6))


class RunAnchorContract(unittest.TestCase):
    def test_missing_gguf_is_unavailable_not_crash(self):
        with mock.patch.object(a, "_find_bench", return_value="llama-bench"), \
             mock.patch.object(a.shutil, "which", return_value="llama-bench"):
            f = a.run_anchor("/no/such/model.gguf")
        self.assertFalse(f["available"])
        self.assertIn("GGUF not found", f["error"])

    def test_parses_a_stubbed_bench_run(self):
        proc = mock.Mock(stdout=_VULKAN_OUT, stderr="")
        with mock.patch.object(a, "_find_bench", return_value="llama-bench"), \
             mock.patch.object(a.shutil, "which", return_value="llama-bench"), \
             mock.patch.object(a.os.path, "exists", return_value=True), \
             mock.patch.object(a.subprocess, "run", return_value=proc):
            f = a.run_anchor("model.gguf", device="Vulkan0")
        self.assertTrue(f["available"])
        self.assertEqual(f["decode_tok_s"], 609.24)
        self.assertEqual(f["device"], "Vulkan0")

    def test_empty_table_reports_unavailable(self):
        proc = mock.Mock(stdout="no table", stderr="device lost")
        with mock.patch.object(a, "_find_bench", return_value="llama-bench"), \
             mock.patch.object(a.shutil, "which", return_value="llama-bench"), \
             mock.patch.object(a.os.path, "exists", return_value=True), \
             mock.patch.object(a.subprocess, "run", return_value=proc):
            f = a.run_anchor("model.gguf")
        self.assertFalse(f["available"])
        self.assertIn("no t/s table", f["error"])


class RecordedAnchor(unittest.TestCase):
    def test_recorded_anchor_is_self_consistent(self):
        r = a.RECORDED_ANCHOR
        # the recorded same-GPU gap must round to the ~24.8x the docstring and #462 report claim
        self.assertEqual(a.gap_vs(r["vulkan_rx7600"]["decode_tok_s"], r["fak_vulkan_q8_decode_tok_s"]), 24.76)
        # the Vulkan anchor must be the faster one (it's the GPU); CPU is the slower triangulation arm
        self.assertGreater(r["vulkan_rx7600"]["decode_tok_s"], r["cpu_zen4"]["decode_tok_s"])

    def test_main_recorded_flag_exits_zero(self):
        self.assertEqual(a.main(["--recorded"]), 0)


class LiveSmoke(unittest.TestCase):
    def test_live_optional(self):
        """If llama-bench and the SmolLM2 GGUF are really here, the probe should resolve a decode
        number on the RX 7600. Otherwise SKIP — this is a reference probe, not a hard gate."""
        gguf = os.environ.get("FAK_ANCHOR_GGUF", "")
        if not gguf or not os.path.exists(gguf):
            self.skipTest("no FAK_ANCHOR_GGUF model on this box")
        if not (shutil.which("llama-bench") or os.path.exists(a._find_bench())):
            self.skipTest("no llama-bench on this box")
        f = a.run_anchor(gguf, device="Vulkan0")
        if not f.get("available"):
            self.skipTest(f"bench unavailable: {f.get('error')}")
        self.assertGreater(f["decode_tok_s"], 0)


if __name__ == "__main__":
    unittest.main()
