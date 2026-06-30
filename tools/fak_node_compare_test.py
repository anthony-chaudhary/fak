#!/usr/bin/env python3
"""Tests for fak_node_compare — the cross-hardware kernel-result folder.

These exercise the PURE text parsers (parse_q8kernel / parse_batchbench) against
synthetic benchmark output written to a tmp dir, so the regex-extraction contracts
— a q8kernel row maps name->{ms,gbs,xf32}; batchbench yields B1 and the peak batch
— are pinned without needing a real benchmark node. The missing-file fallbacks
(every parser must return its empty shape, never raise, on an absent path) are
asserted too, since load_nodes() relies on that to fold partial nodes."""
import json
import tempfile
import unittest
from pathlib import Path

import fak_node_compare as fnc


Q8_TXT = """\
fak q8 kernel microbench
GOMAXPROCS=32
  f32             3.40 ms     12.0 weight-GB/s   1.00x vs f32
  int8xf32(WO)    1.00 ms     28.5 weight-GB/s   0.39x vs f32
"""


class ParseQ8Kernel(unittest.TestCase):
    def test_parses_gomaxprocs_and_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "q8kernel.txt"
            p.write_text(Q8_TXT, encoding="utf-8")
            out = fnc.parse_q8kernel(str(p))
        self.assertEqual(out["gomaxprocs"], 32)
        self.assertIn("f32", out["rows"])
        self.assertIn("int8xf32(WO)", out["rows"])
        i8 = out["rows"]["int8xf32(WO)"]
        self.assertAlmostEqual(i8["ms"], 1.00)
        self.assertAlmostEqual(i8["gbs"], 28.5)
        self.assertAlmostEqual(i8["xf32"], 0.39)
        # the quantized kernel beating f32 is the headline invariant
        self.assertLess(i8["xf32"], out["rows"]["f32"]["xf32"])

    def test_missing_file_returns_empty_shape(self):
        out = fnc.parse_q8kernel("/no/such/q8kernel.txt")
        self.assertEqual(out, {"gomaxprocs": None, "rows": {}})


class ParseBatchbench(unittest.TestCase):
    def test_prefers_sidecar_json_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            (dd / "batchbench-q8.json").write_text(json.dumps({
                "baseline_b1_tok_per_sec": 11.0,
                "peak": {"batch": 16, "agg_tok_per_sec": 95.0},
            }), encoding="utf-8")
            out = fnc.parse_batchbench(str(dd / "batchbench.txt"))
        self.assertEqual(out["b1_tok_s"], 11.0)
        self.assertEqual(out["bmax"], 16)
        self.assertEqual(out["bmax_tok_s"], 95.0)

    def test_falls_back_to_text_table(self):
        txt = (
            "B=1 step=0.1 agg=  9.0 tok/s\n"
            "B=4 step=0.4 agg= 30.0 tok/s\n"
            "B=8 step=0.8 agg= 48.0 tok/s\n"
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "batchbench.txt"
            p.write_text(txt, encoding="utf-8")
            out = fnc.parse_batchbench(str(p))
        self.assertEqual(out["b1_tok_s"], 9.0)
        # peak picked by the largest batch, not the largest tok/s
        self.assertEqual(out["bmax"], 8)
        self.assertEqual(out["bmax_tok_s"], 48.0)

    def test_missing_file_returns_empty_shape(self):
        out = fnc.parse_batchbench("/no/such/batchbench.txt")
        self.assertEqual(out, {"b1_tok_s": None, "bmax": None, "bmax_tok_s": None})


class LoadNodes(unittest.TestCase):
    def test_load_nodes_folds_q8_and_batch_per_host(self):
        # Point the module's NODES dir at a synthetic tree with one host.
        with tempfile.TemporaryDirectory() as d:
            host = Path(d) / "node-a"
            host.mkdir()
            (host / "node-info.json").write_text(
                json.dumps({"host": "node-a", "arch": "arm64"}), encoding="utf-8")
            (host / "q8kernel.txt").write_text(Q8_TXT, encoding="utf-8")
            orig = fnc.NODES
            try:
                fnc.NODES = str(d)
                nodes = fnc.load_nodes()
            finally:
                fnc.NODES = orig
        self.assertEqual(len(nodes), 1)
        n = nodes[0]
        self.assertEqual(n["host"], "node-a")
        self.assertEqual(n["q8"]["gomaxprocs"], 32)
        self.assertIn("int8xf32(WO)", n["q8"]["rows"])
        # absent batchbench -> empty shape, not a raise
        self.assertEqual(n["batch"], {"b1_tok_s": None, "bmax": None, "bmax_tok_s": None})


if __name__ == "__main__":
    unittest.main()
