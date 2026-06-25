#!/usr/bin/env python3
"""Tests for tools/guard_verdict_rsi.py — hermetic, no network, no toolchain.

The verdict loop's load-bearing property is that it cannot KEEP an iteration without a
strict verdict-quality gain on REAL rows AND a witness it did not author. These tests pin
that with synthetic journal fixtures (so they're deterministic): a clean journal scores
100, a journal with an unexplained block / unknown verdict scores lower, the candidate
replay closes the honesty hole (a strict gain), the keep/revert rung KEEPS only with a
witness, and an EMPTY journal refuses any keep.

Run: `python tools/guard_verdict_rsi_test.py`  (exit 0 = all pass).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard_verdict_rsi as gv  # noqa: E402


def _write_journal(rows: list[dict]) -> Path:
    d = Path(tempfile.mkdtemp(prefix="gv-test-"))
    p = d / "guard-audit.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
                 encoding="utf-8")
    return p


def test_clean_journal_scores_100() -> None:
    p = _write_journal([
        {"verdict": "ALLOW", "kind": "DECIDE", "tool": "Read"},
        {"verdict": "DENY", "kind": "DENY", "reason": "OUT_OF_TREE_WRITE"},
        {"verdict": "QUARANTINE", "kind": "QUARANTINE", "reason": "TAINTED_RESULT"},
    ])
    fold = gv.fold_rows([p])
    assert fold["total_rows"] == 3, fold
    assert fold["blank_reason_on_deny"] == 0, fold
    assert fold["unknown_verdict"] == 0, fold
    assert gv.verdict_quality(fold) == 100.0, gv.verdict_quality(fold)


def test_unexplained_block_lowers_quality() -> None:
    p = _write_journal([
        {"verdict": "ALLOW", "kind": "DECIDE"},
        {"verdict": "DENY", "kind": "DENY"},          # NO reason -> unexplained block
        {"verdict": "ZALGO", "kind": "ZALGO"},        # unknown verdict
    ])
    fold = gv.fold_rows([p])
    assert fold["blank_reason_on_deny"] == 1, fold
    assert fold["unknown_verdict"] == 1, fold
    q = gv.verdict_quality(fold)
    assert q < 100.0, q
    # 2 holes / 3 rows -> ~33.3
    assert abs(q - round((1 - 2 / 3) * 100, 3)) < 0.01, q


def test_empty_journal_scores_zero() -> None:
    p = _write_journal([])
    fold = gv.fold_rows([p])
    assert fold["total_rows"] == 0, fold
    assert gv.verdict_quality(fold) == 0.0


def test_worst_bucket_prioritises_unexplained_block() -> None:
    fold = {"total_rows": 5, "by_verdict": {}, "by_reason": {},
            "blank_reason_on_deny": 2, "unknown_verdict": 1}
    wb = gv.worst_bucket(fold)
    assert wb["bucket"] == "blank_reason_on_deny", wb


def test_run_keeps_on_gain_with_witness() -> None:
    # A journal WITH an honesty hole -> the candidate replay closes it -> strict gain.
    p = _write_journal([
        {"verdict": "ALLOW", "kind": "DECIDE"},
        {"verdict": "DENY", "kind": "DENY"},  # unexplained block: the hole to close
    ])
    it = gv.run_iteration(gv.repo_root(), audit_path=str(p),
                          witness={"ok": True, "suite": "go test ./... PASS"})
    assert it["fold"]["total_rows"] == 2, it
    assert it["measured_delta"] > 0, it          # quality rose (hole closed)
    assert it["kept"] is True, it
    assert gv.check_iteration(it) == [], gv.check_iteration(it)


def test_run_reverts_without_witness() -> None:
    p = _write_journal([
        {"verdict": "ALLOW", "kind": "DECIDE"},
        {"verdict": "DENY", "kind": "DENY"},
    ])
    it = gv.run_iteration(gv.repo_root(), audit_path=str(p), witness=None)
    assert it["measured_delta"] > 0, it          # there WAS a gain available
    assert it["kept"] is False, it               # ...but no witness -> REVERT
    assert "witness" in it["reason"], it


def test_run_reverts_on_clean_journal_no_gain() -> None:
    # A clean journal has no hole to close -> no strict gain -> REVERT even with a witness.
    p = _write_journal([
        {"verdict": "ALLOW", "kind": "DECIDE"},
        {"verdict": "DENY", "kind": "DENY", "reason": "OUT_OF_TREE_WRITE"},
    ])
    it = gv.run_iteration(gv.repo_root(), audit_path=str(p),
                          witness={"ok": True})
    assert it["baseline_quality"] == 100.0, it
    assert it["measured_delta"] == 0.0, it
    assert it["kept"] is False, it
    assert "no strict gain" in it["reason"], it


def test_run_refuses_keep_on_empty_journal() -> None:
    p = _write_journal([])
    it = gv.run_iteration(gv.repo_root(), audit_path=str(p),
                          witness={"ok": True})
    assert it["fold"]["total_rows"] == 0, it
    assert it["kept"] is False, it
    assert "empty journal" in it["reason"], it


def test_check_gate_rejects_fabricated_empty_keep() -> None:
    # Hand-forge a kept=true iteration on 0 rows -> the honesty gate must catch it.
    it = {"schema": gv.SCHEMA, "kept": True, "measured_delta": 5.0,
          "witness": {"ok": True}, "fold": {"total_rows": 0}}
    violations = gv.check_iteration(it)
    assert any("empty journal" in v for v in violations), violations


def test_check_gate_rejects_kept_without_strict_gain() -> None:
    it = {"schema": gv.SCHEMA, "kept": True, "measured_delta": 0.0,
          "witness": {"ok": True}, "fold": {"total_rows": 10}}
    violations = gv.check_iteration(it)
    assert any("strict improvement" in v for v in violations), violations


def test_check_gate_rejects_kept_without_witness() -> None:
    it = {"schema": gv.SCHEMA, "kept": True, "measured_delta": 3.0,
          "witness": None, "fold": {"total_rows": 10}}
    violations = gv.check_iteration(it)
    assert any("witness" in v for v in violations), violations


def test_check_gate_accepts_an_honest_kept_iteration() -> None:
    it = {"schema": gv.SCHEMA, "kept": True, "measured_delta": 33.3,
          "witness": {"ok": True, "suite": "go test ./... PASS"},
          "fold": {"total_rows": 2}}
    assert gv.check_iteration(it) == [], gv.check_iteration(it)


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    if failures:
        print(f"\n{failures} test(s) failed")
        return 1
    print("\nall guard-verdict-rsi tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
