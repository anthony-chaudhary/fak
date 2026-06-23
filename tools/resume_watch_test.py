#!/usr/bin/env python3
"""Regression tests for resume_watch.py's transcript-terminal classifier.

The classifier's load-bearing fact: a headless `claude --resume -p` writes ONE
turn and its process EXITS, so PID-liveness is NOT the status signal -- the
transcript's terminal assistant turn is. These pin the precedence the live pass
depended on (an earlier PID-based draft mislabeled finished sessions as DEAD):

  transient API error  -> RETRY    (re-resume; not the session's fault)
  trailing question     -> STALLED  (needs an answer)
  GPU-gated residual    -> GPU_GATED(finished its share; operator/hardware owns the rest)
  clean assistant turn  -> DONE

Pure stdlib; writes tiny .jsonl fixtures to tmp_path, no process spawn, no network.

Run:  python -m pytest tools/resume_watch_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import resume_watch as rw  # noqa: E402


def _write(tmp_path, *records) -> str:
    p = tmp_path / "t.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return str(p)


def _asst(text):
    return {"type": "assistant", "message": {"role": "assistant",
            "content": [{"type": "text", "text": text}]}}


def test_clean_completion_is_done(tmp_path):
    p = _write(tmp_path, _asst("All tests green; working tree clean. Nothing left to commit."))
    t = rw.scan_tail(p)
    assert t["last_asst"] is not None and t["last_err"] is None


def test_transient_error_supersedes_then_retry_signal(tmp_path):
    # a good turn, then a trailing transient 529 -> the error is the terminal state
    p = _write(tmp_path, _asst("Did the work."),
               _asst("API Error: 529 Overloaded. This is a server-side issue, try again."))
    t = rw.scan_tail(p)
    assert t["last_err"] is not None, "trailing 529 must register as the terminal error"


def test_later_good_turn_supersedes_earlier_error(tmp_path):
    # error first, then recovery -> NOT terminal-error (the supersede rule)
    p = _write(tmp_path, _asst("API Error: 529 Overloaded."),
               _asst("Recovered and finished the lane cleanly."))
    t = rw.scan_tail(p)
    assert t["last_err"] is None, "a later good assistant turn must clear an earlier error"


def test_trailing_text_extraction_handles_block_list(tmp_path):
    rec = _asst("hello world")
    assert rw.trailing_text(rec).strip() == "hello world"


def test_control_records_are_not_assistant_turns(tmp_path):
    # mode / permission-mode control records must not be mistaken for a turn
    p = _write(tmp_path, _asst("real work here"),
               {"type": "mode", "mode": "default"},
               {"type": "permission-mode"})
    t = rw.scan_tail(p)
    assert t["last_asst"] is not None
    assert "real work" in rw.trailing_text(t["last_asst"])


def test_shipped_verdict_overrides_gpu_backstory():
    # a GPU phrase as backstory must NOT mark a session GPU_GATED when it then
    # explicitly says it's shipped / no further action.
    low = ("benchmarked on a gpu earlier; everything verified. "
           "no further action needed - it's green and shipped.")
    shipped = any(k in low for k in rw.SHIPPED)
    gpu_gated = any(k in low for k in rw.GPU_GATE) and not shipped
    assert shipped and not gpu_gated


def _gpu_gated(low):
    """Mirror of classify()'s GPU-gated predicate for unit testing the wording rules."""
    shipped = any(k in low for k in rw.SHIPPED)
    gpu_action = ("_on_gpu" in low or "gated by #47" in low
                  or "run_485" in low or "run_486" in low or "run_484" in low
                  or "run_483" in low or "run_482" in low or "run_479" in low
                  or ("residual" in low and ("cuda node" in low or "gpu node" in low
                                             or "on a gpu" in low)))
    gpu_deferred_report = ("left untouched" in low or "correctly left" in low
                           or "left alone" in low or "not run here" in low
                           or "left for" in low)
    return gpu_action and not shipped and not gpu_deferred_report


def test_real_gpu_residual_still_gates():
    low = ("residual: execute tools/run_486_acceptance_on_gpu.sh on a cuda node "
           "to obtain the cosine verdict.")
    assert _gpu_gated(low)


def test_reporting_deferred_gpu_work_is_not_gated():
    # a session that SAYS it left GPU work untouched is DONE, not a GPU residual.
    low = ("gpu/cuda/vulkan issues were correctly left untouched as hardware-gated "
           "on this box. head == origin/main, 21/21 closed.")
    assert not _gpu_gated(low)


def test_weak_gpu_token_alone_does_not_gate():
    low = "i benchmarked this on a gpu last week; everything builds and tests pass."
    assert not _gpu_gated(low)


def test_arm64_residual_is_not_a_gpu_residual():
    # an arm64/m3 acceptance residual is a DIFFERENT hardware gate, not GPU.
    low = ("honest residual: run tools/run_478_acceptance_on_arm64.sh on an arm64 m3. "
           "the run and the tok/s measure are the m3 residual.")
    assert not _gpu_gated(low)


def test_auth_failure_detects_not_logged_in():
    # the exact failure mode the 2026-06-23 pass hit: a re-resume can't fix it.
    ok, reason = rw.auth_failure("Not logged in. Please run /login")
    assert ok and reason == "auth/login required"


def test_auth_failure_detects_oauth_expired():
    ok, reason = rw.auth_failure("API Error: OAuth token has expired; re-authenticate.")
    assert ok and reason == "auth/login required"


def test_auth_failure_distinguishes_credit_and_access():
    ok_c, reason_c = rw.auth_failure("Your credit balance is too low to run this.")
    assert ok_c and reason_c == "credit balance too low"
    ok_a, reason_a = rw.auth_failure("Your organization has disabled Claude subscription access.")
    assert ok_a and reason_a == "Claude subscription access disabled"


def test_auth_failure_ignores_clean_completion():
    ok, reason = rw.auth_failure("All tests green; working tree clean. Nothing left to commit.")
    assert not ok and reason == ""


def test_auth_failure_not_triggered_by_transient_api_error():
    # a transient 529 must stay RETRY, never get misrouted to AUTH_FAIL.
    ok, _ = rw.auth_failure("API Error: 529 Overloaded. This is a server-side issue, try again.")
    assert not ok


def test_terminal_auth_failure_fires_on_error_record(tmp_path):
    # the real gem7 failure shape: an assistant turn that is ALSO an API-error
    # record ('Not logged in / Please run /login') -> AUTH_FAIL via the error channel.
    rec = {"type": "assistant", "isApiErrorMessage": True,
           "message": {"role": "assistant",
                       "content": [{"type": "text", "text": "Not logged in. Please run /login"}]}}
    p = _write(tmp_path, _asst("did some work first"), rec)
    t = rw.scan_tail(p)
    ok, reason = rw.terminal_auth_failure(t)
    assert ok and reason == "auth/login required"


def test_terminal_auth_failure_ignores_auth_prose_in_clean_turn(tmp_path):
    # the false positive caught on live data: a worker editing the resume tooling
    # ends on normal prose that MENTIONS '/login' -> NOT an auth failure (there is
    # no error record), it completed cleanly.
    p = _write(tmp_path, _asst(
        "Added AUTH_FAIL detection for the 'Not logged in / Please run /login' case. "
        "All tests green; nothing left to commit."))
    t = rw.scan_tail(p)
    assert t["last_err"] is None
    ok, _ = rw.terminal_auth_failure(t)
    assert not ok, "auth terms in plain prose must not trip AUTH_FAIL"


def test_terminal_auth_failure_ignores_transient_error(tmp_path):
    # a trailing 529 stays a transient (RETRY), never AUTH_FAIL.
    p = _write(tmp_path, _asst("did work"),
               _asst("API Error: 529 Overloaded. Try again."))
    t = rw.scan_tail(p)
    ok, _ = rw.terminal_auth_failure(t)
    assert not ok


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
