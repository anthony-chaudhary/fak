#!/usr/bin/env python3
"""Tests for the vCache scorecard gate (`vcache_scorecard_gate.py`, issue #791).

Drives the gate's assertion logic against a STUB `fak` that emits canned score JSON,
so the test is deterministic and needs no Go build: it proves the gate PASSES on a
2x-ready score, FAILS on each regression shape (sub-2x multiplier, two_x_better false,
wrong schema, a default-path non-zero exit, a negative-path that wrongly passes), and
reports a HARNESS error (exit 2) when the binary cannot run.

Run: `python tools/vcache_scorecard_gate_test.py`  (exit 0 = all pass),
or `python -m pytest tools/vcache_scorecard_gate_test.py -q`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import vcache_scorecard_gate as gate  # noqa: E402

READY = {
    "schema": gate.SCORE_SCHEMA,
    "two_x_better": True,
    "active_multiplier": 3.76,
    "planned": {"status": "PROVEN"},
}


def _patch(monkeypatch_pairs):
    """Replace gate.run_score with a stub that returns (code, payload) keyed on whether
    the `--two-x 99` negative arg is present. monkeypatch_pairs maps "default"/"neg" to
    a (code, payload) tuple."""
    def stub(extra, timeout):
        key = "neg" if "99" in extra else "default"
        return monkeypatch_pairs[key]
    gate.run_score = stub  # type: ignore[assignment]


def _restore(orig):
    gate.run_score = orig  # type: ignore[assignment]


def test_passes_on_2x_ready():
    orig = gate.run_score
    try:
        _patch({"default": (0, READY), "neg": (1, {"two_x_better": False})})
        assert gate.main(["--json"]) == 0
    finally:
        _restore(orig)


def test_fails_on_sub_2x_multiplier():
    orig = gate.run_score
    try:
        bad = dict(READY, active_multiplier=1.5)
        _patch({"default": (0, bad), "neg": (1, {"two_x_better": False})})
        assert gate.main([]) == 1
    finally:
        _restore(orig)


def test_fails_on_two_x_better_false():
    orig = gate.run_score
    try:
        bad = dict(READY, two_x_better=False)
        # If the default already says not-2x, fak would also exit non-zero; model that.
        _patch({"default": (1, bad), "neg": (1, {"two_x_better": False})})
        assert gate.main([]) == 1
    finally:
        _restore(orig)


def test_fails_on_wrong_schema():
    orig = gate.run_score
    try:
        bad = dict(READY, schema="fak.something.else.v1")
        _patch({"default": (0, bad), "neg": (1, {"two_x_better": False})})
        assert gate.main([]) == 1
    finally:
        _restore(orig)


def test_fails_when_default_exits_nonzero():
    orig = gate.run_score
    try:
        _patch({"default": (1, READY), "neg": (1, {"two_x_better": False})})
        assert gate.main([]) == 1
    finally:
        _restore(orig)


def test_fails_when_negative_path_wrongly_passes():
    # The gate must catch an always-pass gate: if --two-x 99 exits 0, that is a
    # regression (the gate stopped gating), even though the default looks fine.
    orig = gate.run_score
    try:
        _patch({"default": (0, READY), "neg": (0, dict(READY))})
        assert gate.main([]) == 1
    finally:
        _restore(orig)


def test_harness_error_returns_2():
    orig = gate.run_score
    try:
        def boom(extra, timeout):
            raise FileNotFoundError("no fak binary")
        gate.run_score = boom  # type: ignore[assignment]
        assert gate.main(["--json"]) == 2
    finally:
        _restore(orig)


def test_json_verdict_shape():
    orig = gate.run_score
    try:
        _patch({"default": (0, READY), "neg": (1, {"two_x_better": False})})
        # Capture stdout to assert the JSON verdict carries the schema + ok=True.
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = gate.main(["--json"])
        assert rc == 0
        out = json.loads(buf.getvalue())
        assert out["schema"] == "fak.vcache-scorecard-gate.v1"
        assert out["ok"] is True
        assert out["failures"] == []
    finally:
        _restore(orig)


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
