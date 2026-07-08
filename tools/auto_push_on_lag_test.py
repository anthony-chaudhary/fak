#!/usr/bin/env python3
"""Hermetic tests for tools/auto_push_on_lag.py.

The push arm shells `fak sync push`; here that seam (`push_main`) and the git
read (`git_pane`) are replaced with synthetic results on the module, so NOTHING
live (fak/git push) is ever invoked. One guarded integration test builds a real
throwaway git repo to witness the trigger end-to-end, and skips on a git-less box.

Run: python tools/auto_push_on_lag_test.py   (also `python -m pytest -q`)
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_push_on_lag as apl  # noqa: E402

NOW = datetime(2026, 6, 25, 12, 0, 0, tzinfo=timezone.utc)


def _pane(**over):
    base = {
        "branch": "main", "ahead": 3, "behind": 0,
        "push_lag_seconds": 3600, "push_lag_stale": True, "dirty_lag_stale": False,
    }
    base.update(over)
    return base


# --- decide(): the pure admission seam --------------------------------------

def test_decide_on_main_and_stale_pushes() -> None:
    should, reason = apl.decide(_pane())
    assert should is True
    assert reason == "push-lag-60m"


def test_decide_dirty_only_does_not_push() -> None:
    # The common ACTION cause: a merely-dirty tree. push_lag_stale is False, so a
    # dirty tree must NEVER trigger a push regardless of dirty_lag_stale.
    should, reason = apl.decide(_pane(push_lag_stale=False, dirty_lag_stale=True))
    assert should is False
    assert reason == "no-push-lag"


def test_decide_off_main_skips() -> None:
    should, reason = apl.decide(_pane(branch="topic/x"))
    assert should is False
    assert reason == "not-on-main"


def test_decide_nothing_ahead_skips() -> None:
    should, reason = apl.decide(_pane(ahead=0, push_lag_seconds=None))
    assert should is False
    assert reason == "nothing-ahead"


# --- run(): report vs. push, with the seams stubbed -------------------------

class _Patch:
    """Swap module attributes for the duration of a test, then restore."""

    def __init__(self, **attrs):
        self.attrs = attrs
        self.saved: dict = {}

    def __enter__(self):
        for k, v in self.attrs.items():
            self.saved[k] = getattr(apl, k)
            setattr(apl, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(apl, k, v)


def _boom(*a, **k):  # a seam that must NOT be reached
    raise AssertionError("push_main must not be called")


def test_run_dry_run_never_pushes() -> None:
    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=_boom):
        res = apl.run(root, live=False, push_lag_mins=45, now=NOW)
    assert res["verdict"] == "WOULD_PUSH"
    assert res["push_result"] is None
    assert res["ok"] is True


def test_run_live_pushes_once() -> None:
    calls = []

    def fake_push(r, fak):
        calls.append((r, fak))
        return {"ok": True, "pushed": True, "returncode": 0, "reason": "pushed"}

    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=fake_push,
                resolve_fak=lambda r: ["fak"]):
        res = apl.run(root, live=True, push_lag_mins=45, now=NOW)
    assert res["verdict"] == "PUSHED"
    assert res["ok"] is True
    assert len(calls) == 1


def test_run_live_push_failure_is_not_ok() -> None:
    def fake_push(r, fak):
        return {"ok": False, "pushed": False, "returncode": 1, "reason": "push-rejected"}

    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=fake_push,
                resolve_fak=lambda r: ["fak"]):
        res = apl.run(root, live=True, push_lag_mins=45, now=NOW)
    assert res["verdict"] == "PUSH_FAILED"
    assert res["ok"] is False  # exit code will be non-zero -> LastTaskResult surfaces it


def test_run_live_push_failure_surfaces_real_reason() -> None:
    # Regression: a failed push must record its REAL cause (push-rejected / push-timeout /
    # push-oserror), NOT the stale decide reason ("push-lag-NNm"). Else a stalled push reads
    # as a generic lag in auto-push.jsonl and the step that actually died — a rejected
    # pre-push gate vs a timed-out tip-materialize vs a credential hang — stays hidden.
    import json

    def fake_push(r, fak):
        return {"ok": False, "pushed": False, "returncode": 124, "reason": "push-timeout"}

    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=fake_push,
                resolve_fak=lambda r: ["fak"]):
        res = apl.run(root, live=True, push_lag_mins=45, now=NOW)
    assert res["verdict"] == "PUSH_FAILED"
    assert res["reason"] == "push-timeout"  # not "push-lag-60m"
    rec = json.loads((root / apl.LOG_REL).read_text(encoding="utf-8").splitlines()[-1])
    assert rec["reason"] == "push-timeout"  # the breadcrumb carries the real cause


def test_run_live_without_fak_skips_never_bypasses() -> None:
    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=_boom,
                resolve_fak=lambda r: None):
        res = apl.run(root, live=True, push_lag_mins=45, now=NOW)
    assert res["verdict"] == "SKIPPED"
    assert res["reason"] == "fak-unavailable"
    assert res["ok"] is False


def test_run_writes_jsonl_breadcrumb() -> None:
    import json
    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(push_lag_stale=False), push_main=_boom):
        apl.run(root, live=False, push_lag_mins=45, now=NOW)
    log = root / apl.LOG_REL
    assert log.exists()
    rec = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["verdict"] == "NO_LAG" and rec["pushed"] is False


# --- integration: real repo, real git_pane, guarded for a git-less box ------

def _git(repo, *args):
    import subprocess
    base = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
            "-c", "core.hooksPath=", "-c", "commit.gpgsign=false"]
    subprocess.run(base + list(args), check=True, capture_output=True, text=True)


def test_integration_stale_ahead_trips_decide() -> None:
    import subprocess
    try:
        repo = Path(tempfile.mkdtemp())
        _git(repo, "init", "-q", "-b", "main")
        (repo / "a.txt").write_text("a", encoding="utf-8")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "-q", "-m", "A", "--no-gpg-sign")
        # self-referential origin so @{upstream} resolves (per fresh_status_test)
        _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
        _git(repo, "config", "branch.main.remote", "origin")
        _git(repo, "config", "branch.main.merge", "refs/heads/main")
        _git(repo, "config", "remote.origin.url", ".")
        _git(repo, "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*")
        (repo / "b.txt").write_text("b", encoding="utf-8")
        _git(repo, "add", "b.txt")
        _git(repo, "commit", "-q", "-m", "B", "--no-gpg-sign")
    except (OSError, subprocess.SubprocessError):
        return  # git unavailable / sandboxed — skip, don't fail
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    # Real git_pane + real decide: the unpushed B, 2h old, must arm the backstop.
    res = apl.run(repo, live=False, push_lag_mins=45, now=future)
    assert res["push_lag_stale"] is True
    assert res["ahead"] == 1 and res["branch"] == "main"
    assert res["verdict"] == "WOULD_PUSH"
    # A generous threshold keeps the same state a healthy no-op (no false alarm).
    ok = apl.run(repo, live=False, push_lag_mins=10 ** 9, now=future)
    assert ok["verdict"] == "NO_LAG"


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
