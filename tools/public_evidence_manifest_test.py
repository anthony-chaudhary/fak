#!/usr/bin/env python3
"""Hermetic tests for tools/public_evidence_manifest.py.

The manifest deriver's load-bearing pure surface is path normalization and the
citation scanner ``_cited_in`` — the regex fold that decides which
``experiments/*`` artifacts and ``*-RESULTS.md`` docs a piece of prose cites.
These tests pin the real rules the shipping gate depends on: paths normalize
across separators and leading ``./``/``fak/`` prefixes, markdown-link and
inline-code citations are both caught, a longer ``.jsonl`` wins over ``.json``,
and a path written after an ``--output`` flag is a WRITE target, not a citation.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "public_evidence_manifest.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("public_evidence_manifest", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()


class TestNorm(unittest.TestCase):
    def test_backslashes_become_slashes(self):
        self.assertEqual(MOD._norm("experiments\\fanout\\x.json"),
                         "experiments/fanout/x.json")

    def test_leading_dot_slash_stripped(self):
        self.assertEqual(MOD._norm("./experiments/x.json"), "experiments/x.json")


class TestCitedIn(unittest.TestCase):
    def test_markdown_link_experiment_cited(self):
        exp, res = MOD._cited_in("see [data](experiments/fanout/data.json) here")
        self.assertIn("experiments/fanout/data.json", exp)
        self.assertEqual(res, set())

    def test_leading_fak_prefix_normalized(self):
        exp, _ = MOD._cited_in("`fak/experiments/fleet/sweep.csv`")
        self.assertIn("experiments/fleet/sweep.csv", exp)

    def test_jsonl_wins_over_json(self):
        # the trailing (?![A-Za-z]) must let .jsonl match fully, not stop at .json
        exp, _ = MOD._cited_in("path experiments/w/trace.jsonl end")
        self.assertIn("experiments/w/trace.jsonl", exp)
        self.assertNotIn("experiments/w/trace.json", exp)

    def test_results_doc_cited(self):
        _, res = MOD._cited_in("provenance in QWEN36-RESULTS.md is authoritative")
        self.assertIn("QWEN36-RESULTS.md", res)

    def test_output_flag_path_is_not_a_citation(self):
        # a path a command WRITES (after --output) must not be treated as a citation
        exp, _ = MOD._cited_in("run tool --output experiments/tmp/out.json now")
        self.assertNotIn("experiments/tmp/out.json", exp)

    def test_plain_prose_cites_nothing(self):
        exp, res = MOD._cited_in("this paragraph names no artifacts at all")
        self.assertEqual(exp, set())
        self.assertEqual(res, set())


if __name__ == "__main__":
    unittest.main()
