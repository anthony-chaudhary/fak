#!/usr/bin/env python3
"""Tests for the public-evidence manifest's pure citation core (tools/public_evidence_manifest.py).

Covers `_norm` (path normalization) and `_cited_in` (extract the experiments/ artifact
paths and *-RESULTS.md provenance docs a doc CITES, while NOT counting a path that sits
right after an output flag — that's a file a command WRITES, not a citation). Pure; no
disk walk needed.

Run: `python tools/public_evidence_manifest_test.py`  (exit 0 = all pass),
or `python -m pytest tools/public_evidence_manifest_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_evidence_manifest as pem  # noqa: E402


# --- _norm -----------------------------------------------------------------

def test_norm_backslashes_to_forward() -> None:
    assert pem._norm("a\\b\\c") == "a/b/c"


def test_norm_strips_leading_dot_slash() -> None:
    assert pem._norm("./experiments/x.json") == "experiments/x.json"


# --- _cited_in -------------------------------------------------------------

def test_cited_in_extracts_experiments_artifact() -> None:
    exp, res = pem._cited_in("see [data](experiments/qwen/run.json) for detail")
    assert "experiments/qwen/run.json" in exp


def test_cited_in_extracts_results_doc() -> None:
    exp, res = pem._cited_in("provenance in 8B-RESULTS.md and more")
    assert "8B-RESULTS.md" in res


def test_cited_in_normalizes_fak_and_dot_prefix() -> None:
    exp, _ = pem._cited_in("link ./fak/experiments/a/b.csv here")
    # the fak/ and ./ prefixes are normalized away to the root-relative path.
    assert "experiments/a/b.csv" in exp


def test_cited_in_skips_output_flag_target() -> None:
    # a path right after --output is a WRITE target, not a citation.
    exp, _ = pem._cited_in("run x --output experiments/tmp/out.json")
    assert "experiments/tmp/out.json" not in exp


def test_cited_in_counts_non_output_path() -> None:
    # the same path NOT behind an output flag IS a citation.
    exp, _ = pem._cited_in("results saved to experiments/tmp/out.json earlier")
    assert "experiments/tmp/out.json" in exp


def test_cited_in_empty_text() -> None:
    exp, res = pem._cited_in("just prose, no artifacts here")
    assert exp == set() and res == set()


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
