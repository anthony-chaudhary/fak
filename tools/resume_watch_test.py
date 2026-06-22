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


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
