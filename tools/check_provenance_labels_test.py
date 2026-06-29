#!/usr/bin/env python3
"""Hermetic tests for the provenance-label gate.

Proves the gate FLAGS a modeled number labeled "measured" and PASSES the honest
carve-outs (the number labeled modeled, the genuinely-measured 4.1x co-located, a
negation, a meta-critique). No real repo is touched — each case is a throwaway git
tree in a tempdir.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_provenance_labels as cpl  # noqa: E402


def _scan(text_by_file: dict[str, str]) -> int:
    """Write files into a throwaway git repo, run --audit-tree, return exit code."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init", "-q"], cwd=d, check=True)
        for name, body in text_by_file.items():
            (Path(d) / name).write_text(body, encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=d, check=True,
                       capture_output=True)
        return cpl.main(["--audit-tree", "--root", d])


def _commit_all(root: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=provenance-test",
            "-c",
            "user.email=provenance-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "seed",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


class ProvenanceLabelGate(unittest.TestCase):
    def test_flags_modeled_number_called_measured(self):
        # the exact original defect: the webbench 9.7x called "measured"
        self.assertEqual(
            _scan({"bad.md": "measured 9.7x prefill elimination on real "
                             "WebVoyager (643 tasks)\n"}),
            1, "must FLAG a modeled webbench number labeled measured")

    def test_flags_8_8_variant(self):
        self.assertEqual(
            _scan({"b.md": "8.8x-9.7x measured prefill on the real 643-task "
                           "WebVoyager set\n"}),
            1)

    def test_passes_modeled_label(self):
        self.assertEqual(
            _scan({"ok.md": "modeled 9.7x prefill elimination vs the naive "
                            "floor on real WebVoyager (643 tasks)\n"}),
            0, "a number labeled modeled is honest")

    def test_passes_measured_4_1_colocated(self):
        # the README hero alt-text shape: "modeled 9.7x / measured 4.1x"
        self.assertEqual(
            _scan({"hero.md": "multipliers 20,480x / modeled 9.7x / measured "
                              "4.1x on WebVoyager\n"}),
            0, "measured attaches to the genuine live-race 4.1x, not 9.7x")

    def test_passes_negation(self):
        self.assertEqual(
            _scan({"neg.md": "the 9.7x WebVoyager number is modeled geometry, "
                             "not a wall-clock measurement\n"}),
            0)

    def test_passes_meta_critique(self):
        self.assertEqual(
            _scan({"crit.md": "the WebVoyager 9.7x number is mislabeled as "
                              "measured in the draft\n"}),
            0, "a line that calls OUT a false measured claim is not asserting it")

    def test_passes_unrelated_measured(self):
        self.assertEqual(
            _scan({"u.md": "the measured 4.1x live model-reuse race result\n"}),
            0, "no webbench context -> not in scope")

    def test_clean_tree_is_zero(self):
        self.assertEqual(
            _scan({"plain.md": "fak treats the model like an untrusted "
                               "program.\n"}),
            0)

    def test_audit_staged_flags_added_bad_line(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q"], cwd=d, check=True)
            path = Path(d) / "bad.md"
            path.write_text("plain line\n", encoding="utf-8")
            _commit_all(d)

            path.write_text(
                "plain line\nmeasured 9.7x prefill elimination on WebVoyager\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "bad.md"], cwd=d, check=True, capture_output=True)

            self.assertEqual(cpl.main(["--audit-staged", "--root", d]), 1)

    def test_audit_staged_ignores_preexisting_bad_line(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q"], cwd=d, check=True)
            path = Path(d) / "legacy.md"
            path.write_text(
                "measured 9.7x prefill elimination on WebVoyager\n",
                encoding="utf-8",
            )
            _commit_all(d)

            path.write_text(
                "measured 9.7x prefill elimination on WebVoyager\n"
                "new clean line\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "legacy.md"], cwd=d, check=True, capture_output=True)

            self.assertEqual(cpl.main(["--audit-staged", "--root", d]), 0)


if __name__ == "__main__":
    unittest.main()
