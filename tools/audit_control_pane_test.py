#!/usr/bin/env python3
"""Tests for the audit-family control pane.

Drives the PURE surfaces -- discovery (finds the known auditors on the real
tree, drops *_test.py, honors --only / exclude), the per-auditor row classifier
(exit-code-first verdict; a stubbed FAILING auditor surfaces as ACTION; a
timeout / unrunnable degrades to a soft SKIP that never trips), the rollup fold
(any ACTION trips it, a SKIP never does), and the main wiring (--list discovery,
--check exit code, --json envelope) driven with a stubbed collect so it is fast
and deterministic, not the live 24-auditor sweep. No network, no live subprocess
in the pure tests (hermetic, so the no-blackhole gate runs it).

Run: `python tools/audit_control_pane_test.py`  (exit 0 = all pass),
or `python -m pytest tools/audit_control_pane_test.py -q`.
"""
from __future__ import annotations

import contextlib
import io
import json as _json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_control_pane as acp  # noqa: E402


# --- discovery (against the real tree) --------------------------------------

# A few auditors the issue (#1137) names as ground truth -- if discovery stops
# finding these the contract broke.
KNOWN = ("security_audit", "gofmt_debt_audit", "plan_audit",
         "history_leak_audit", "permission_source_audit")


def test_discover_finds_the_known_auditors() -> None:
    stems = acp.discover(acp.repo_root())
    assert len(stems) >= 20, f"expected the ~24-auditor family, found {len(stems)}"
    for name in KNOWN:
        assert name in stems, f"discovery lost a known auditor: {name}"


def test_discover_drops_paired_tests_and_self() -> None:
    stems = acp.discover(acp.repo_root())
    # No *_test stem should ever be returned as an auditor.
    assert not any(s.endswith("_test") for s in stems)
    # This control pane is not itself an auditor (it is *_pane, not *_audit), so
    # it must not appear; nor may its own test.
    assert "audit_control_pane" not in stems
    assert "audit_control_pane_test" not in stems


def test_discover_is_deterministic_and_sorted() -> None:
    root = acp.repo_root()
    a = acp.discover(root)
    b = acp.discover(root)
    assert a == b == sorted(a)


def test_discover_only_narrows_to_subset() -> None:
    root = acp.repo_root()
    subset = acp.discover(root, only=["security_audit", "plan_audit", "nope_not_real"])
    assert subset == ["plan_audit", "security_audit"]  # discovery order, typo ignored


def test_discover_exclude_drops_named() -> None:
    root = acp.repo_root()
    full = acp.discover(root)
    assert "security_audit" in full
    trimmed = acp.discover(root, exclude=frozenset({"security_audit"}))
    assert "security_audit" not in trimmed
    assert len(trimmed) == len(full) - 1


# --- the per-auditor row classifier (exit-code-first) -----------------------

def test_classify_clean_exit_is_ok() -> None:
    row = acp.classify_row("foo_audit", returncode=0, timed_out=False)
    assert row["verdict"] == "OK" and row["ok"] is True and row["returncode"] == 0


def test_classify_nonzero_exit_is_action() -> None:
    # A stubbed FAILING auditor (exit 2) must surface as a finding the rollup acts on.
    row = acp.classify_row("foo_audit", returncode=2, timed_out=False)
    assert row["verdict"] == "ACTION" and row["ok"] is False
    assert "exit 2" in row["reason"]


def test_classify_timeout_is_soft_skip() -> None:
    row = acp.classify_row("slow_audit", returncode=None, timed_out=True)
    assert row["verdict"] == "SKIP" and row["ok"] is True
    assert "timed out" in row["reason"]


def test_classify_unrunnable_is_soft_skip() -> None:
    row = acp.classify_row("host_audit", returncode=None, timed_out=False,
                           unrunnable="missing tools/host_audit.py")
    assert row["verdict"] == "SKIP" and row["ok"] is True


def test_classify_json_envelope_enriches_but_exit_decides() -> None:
    # A clean exit with a JSON envelope keeps OK but enriches the reason.
    row = acp.classify_row("foo_audit", returncode=0, timed_out=False,
                           payload={"verdict": "OK", "reason": "all green"})
    assert row["verdict"] == "OK" and "all green" in row["reason"]
    # A non-zero exit stays ACTION even if the payload claims OK (exit decides).
    row2 = acp.classify_row("foo_audit", returncode=1, timed_out=False,
                            payload={"verdict": "OK", "reason": "looks fine"})
    assert row2["verdict"] == "ACTION" and row2["ok"] is False


# --- the rollup fold (any ACTION trips; a SKIP never does) ------------------

def _row(stem: str, verdict: str) -> dict:
    ok = verdict != "ACTION"
    return {"key": stem, "label": stem, "verdict": verdict, "ok": ok,
            "returncode": 0 if verdict == "OK" else (None if verdict == "SKIP" else 1),
            "reason": f"{stem} {verdict}"}


def _fold(rows):
    return acp.fold(rows, workspace=".", commit="c0",
                    generated_at="2026-06-28T00:00:00Z")


def test_fold_all_clean_is_ok() -> None:
    out = _fold([_row("a_audit", "OK"), _row("b_audit", "OK")])
    assert out["ok"] is True and out["verdict"] == "OK" and out["finding"] == "all_clear"
    assert out["schema"] == acp.SCHEMA
    assert out["counts"] == {"total": 2, "clean": 2, "action": 0, "skipped": 0}


def test_fold_one_action_trips_the_rollup() -> None:
    out = _fold([_row("a_audit", "OK"), _row("b_audit", "ACTION"),
                 _row("c_audit", "SKIP")])
    assert out["ok"] is False and out["verdict"] == "ACTION"
    assert out["finding"] == "auditor_finding"
    assert "b_audit" in out["reason"] and "b_audit" in out["next_action"]
    assert out["counts"] == {"total": 3, "clean": 1, "action": 1, "skipped": 1}


def test_fold_skip_alone_never_trips() -> None:
    out = _fold([_row("a_audit", "OK"), _row("b_audit", "SKIP")])
    assert out["verdict"] == "OK" and out["ok"] is True
    assert "skipped" in out["reason"] and "b_audit" in out["reason"]


def test_fold_envelope_has_loop_runner_fields() -> None:
    out = _fold([_row("a_audit", "OK")])
    for field in ("schema", "ok", "verdict", "finding", "reason", "next_action",
                  "counts", "auditors", "auditor_order"):
        assert field in out, f"missing {field}"


# --- render -----------------------------------------------------------------

def test_render_lists_every_auditor() -> None:
    text = acp.render(_fold([_row("a_audit", "OK"), _row("b_audit", "ACTION")]))
    assert "a_audit" in text and "b_audit" in text
    assert "audit control pane" in text


def test_render_list_counts() -> None:
    text = acp.render_list(["a_audit", "b_audit", "c_audit"])
    assert "3 auditors discovered" in text and "a_audit" in text


# --- main wiring (stubbed collect; no live sweep) ---------------------------

def test_main_list_is_discovery_only() -> None:
    # --list must not run any auditor; assert by stubbing collect to explode.
    def _boom(*a, **k):
        raise AssertionError("collect must not run under --list")
    orig = acp.collect
    try:
        acp.collect = _boom
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = acp.main(["--list"])
        assert code == 0
        assert "auditors discovered" in buf.getvalue()
    finally:
        acp.collect = orig


def test_main_check_returns_zero_when_clean_nonzero_on_finding() -> None:
    orig = acp.collect
    try:
        acp.collect = lambda root, timeout=acp.DEFAULT_TIMEOUT, only=None: [
            _row("a_audit", "OK"), _row("b_audit", "SKIP")]
        assert acp.main(["--check"]) == 0
        acp.collect = lambda root, timeout=acp.DEFAULT_TIMEOUT, only=None: [
            _row("a_audit", "OK"), _row("b_audit", "ACTION")]
        assert acp.main(["--check"]) == 1
    finally:
        acp.collect = orig


def test_main_json_emits_envelope() -> None:
    orig = acp.collect
    try:
        acp.collect = lambda root, timeout=acp.DEFAULT_TIMEOUT, only=None: [
            _row("a_audit", "OK")]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = acp.main(["--json"])
        out = _json.loads(buf.getvalue())
        assert code == 0 and out["schema"] == acp.SCHEMA
        assert out["counts"]["clean"] == 1
    finally:
        acp.collect = orig


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
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
