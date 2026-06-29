#!/usr/bin/env python3
"""Tests for the cross-domain fresh-status rollup.

Drives the PURE folds — provenance classification (incl. the unknown ->
fail-closed case), catalog freshness/staleness math across both timestamp forms,
the four per-pane folds (git / benchmarks / work / industry), and the rollup
verdict ladder (all_green vs needs_attention, SOFT-SKIP never trips, a HARD pane
ERROR/ACTION does) — over synthetic payloads, then a tolerant live smoke that the
real tree folds to a well-formed envelope. No network, no live sub-tool calls in
the pure tests (hermetic, so the no-blackhole gate runs it).

Run: `python tools/fresh_status_test.py`  (exit 0 = all pass),
or `python -m pytest tools/fresh_status_test.py -q`.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fresh_status as fs  # noqa: E402

NOW = datetime(2026, 6, 25, 12, 0, 0, tzinfo=timezone.utc)


# --- provenance is delegated to bench_provenance (4-way) --------------------
# The classifier itself is exhaustively tested in bench_provenance_test.py
# (taxonomy + the adversarially-verified ground-truth table). Here we only assert
# fold_benchmarks consumes it correctly and reports the 4-way split.


# --- catalog freshness / staleness math -------------------------------------

def test_parse_run_ts_compact_and_iso() -> None:
    assert fs._parse_run_ts("20260625T050017Z") == datetime(2026, 6, 25, 5, 0, 17, tzinfo=timezone.utc)
    assert fs._parse_run_ts("2026-06-25T05:00:17Z") == datetime(2026, 6, 25, 5, 0, 17, tzinfo=timezone.utc)
    assert fs._parse_run_ts("garbage") is None
    assert fs._parse_run_ts("") is None


def _catalog(*runs: dict, machines: int = 2) -> dict:
    return {"runs": list(runs), "machines": {f"m{i}": {} for i in range(machines)}}


def test_fold_benchmarks_counts_and_provenance() -> None:
    # One run of each category, using real tags from the verified taxonomy:
    # radix-benchmark -> measured, turn-tax -> modeled, parity -> functional,
    # bare experiment -> unknown (fail-closed).
    cat = _catalog(
        {"timestamp": "20260625T050017Z", "model": "qwen", "tags": ["radix-benchmark"],
         "run_id": "radix-x"},
        {"timestamp": "20260624T050017Z", "model": "webvoyager", "tags": ["turn-tax"],
         "run_id": "turn-tax-x"},
        {"timestamp": "20260623T050017Z", "model": "experiment", "tags": ["parity"],
         "run_id": "parity-x"},
        {"timestamp": "20260622T050017Z", "model": "experiment", "tags": ["experiment"],
         "run_id": "bare-x"},
    )
    pane = fs.fold_benchmarks(cat, now=NOW)
    assert pane["runs"] == 4 and pane["machines"] == 2
    assert pane["provenance"] == {"measured": 1, "modeled": 1, "functional": 1, "unknown": 1}
    assert pane["verdict"] == "OK" and pane["ok"] is True
    # newest run is the 06-25 one, ~0.3d old
    assert pane["newest"] == "20260625T050017Z" and pane["age_days"] < 1


def test_fold_benchmarks_flags_stale() -> None:
    cat = _catalog({"timestamp": "20260501T000000Z", "model": "x", "tags": []})
    pane = fs.fold_benchmarks(cat, now=NOW, stale_days=30)
    assert pane["stale"] is True and pane["verdict"] == "ACTION" and pane["ok"] is False
    assert "STALE" in pane["reason"]


def test_fold_benchmarks_missing_catalog_is_hard_error() -> None:
    pane = fs.fold_benchmarks(None, now=NOW)
    assert pane["verdict"] == "ERROR" and pane["ok"] is False


def test_fold_benchmarks_empty_runs_is_action() -> None:
    pane = fs.fold_benchmarks(_catalog(machines=1), now=NOW)
    assert pane["runs"] == 0 and pane["verdict"] == "ACTION"
    # the provenance histogram on the error/empty paths must carry all four keys
    assert set(pane["provenance"]) == {"measured", "modeled", "functional", "unknown"}


def test_fold_benchmarks_functional_separates_from_throughput() -> None:
    # The whole point of the 4-way split: an agent-live / parity / load-only run is
    # FUNCTIONAL (a witness), not lumped into measured or hidden in unknown.
    cat = _catalog(
        {"timestamp": "20260625T050017Z", "tags": ["agent-live"], "run_id": "a"},
        {"timestamp": "20260625T050017Z", "tags": ["safetensors-load-rss"], "run_id": "b"},
        {"timestamp": "20260625T050017Z", "tags": ["parity"], "run_id": "c"},
    )
    pane = fs.fold_benchmarks(cat, now=NOW)
    assert pane["provenance"]["functional"] == 3
    assert pane["provenance"]["measured"] == 0 and pane["provenance"]["unknown"] == 0


def test_enrich_catalog_engines_stamps_from_local_artifacts() -> None:
    # enrich_catalog_engines reads a run's local artifact engine fields so a run
    # with a decode artifact classifies measured even if its tags say agent-live --
    # robust to the catalog `provenance` stamp being clobbered by a peer rebuild.
    import json as _json
    import tempfile
    from pathlib import Path as _Path
    with tempfile.TemporaryDirectory() as td:
        root = _Path(td)
        run_dir = root / "experiments" / "benchmark" / "runs" / "by-machine" / "m" / "r"
        run_dir.mkdir(parents=True)
        (run_dir / "01-decode.json").write_text(
            _json.dumps({"engine": "fak-in-kernel Q8_0 ... decode"}), encoding="utf-8")
        catalog = {"runs": [{"run_id": "r", "tags": ["agent-live"],
                             "path": "experiments/benchmark/runs/by-machine/m/r"}],
                   "machines": {"m": {}}}
        enriched = fs.enrich_catalog_engines(root, catalog)
        run = enriched["runs"][0]
        assert run.get("artifact_engines") == ["fak-in-kernel Q8_0 ... decode"]
        # the on-disk catalog object is not mutated (read-only tool)
        assert "artifact_engines" not in catalog["runs"][0]
        # and the classifier now reads the engine (measured), not the agent-live tag
        import bench_provenance as bp
        assert bp.classify(run) == "measured"


# --- work pane: SOFT (zero / absent -> SKIP, never a failure) ---------------

def test_fold_work_zero_plans_is_skip_not_error() -> None:
    pane = fs.fold_work({"counts": {"total_plans": 0}})
    assert pane["verdict"] == "SKIP" and pane["ok"] is True
    assert pane["total_plans"] == 0


def test_fold_work_absent_is_skip() -> None:
    pane = fs.fold_work(None, error="plan_audit unavailable")
    assert pane["verdict"] == "SKIP" and pane["ok"] is True


def test_fold_work_with_plans_reports_ok() -> None:
    pane = fs.fold_work({"counts": {"total_plans": 4, "shipped": 3, "remaining": 1}})
    assert pane["verdict"] == "OK" and pane["total_plans"] == 4
    assert "3 shipped" in pane["reason"] and "1 remaining" in pane["reason"]


# --- industry pane: SOFT ----------------------------------------------------

def test_fold_industry_reports_parity_debt() -> None:
    pane = fs.fold_industry({"corpus": {"parity_debt": 7, "grade": "B"}})
    assert pane["verdict"] == "OK" and pane["parity_debt"] == 7 and pane["grade"] == "B"


def test_fold_industry_absent_is_skip() -> None:
    pane = fs.fold_industry(None, error="timed out")
    assert pane["verdict"] == "SKIP" and pane["ok"] is True


def test_fold_industry_no_debt_key_is_skip() -> None:
    pane = fs.fold_industry({"corpus": {"grade": "A"}})
    assert pane["verdict"] == "SKIP" and pane["grade"] == "A"


# --- git pane ---------------------------------------------------------------

def test_git_pane_smoke_on_real_repo() -> None:
    pane = fs.git_pane(fs.repo_root())
    # In this repo git is available, so the pane must report a sha + branch.
    assert pane["key"] == "git"
    assert pane["verdict"] in ("OK", "ERROR")
    if pane["verdict"] == "OK":
        assert pane["sha"] and isinstance(pane["dirty"], int)
        # Push-lag dimension is always carried; when fully pushed (ahead 0/None)
        # there is nothing waiting, so the lag is None — never a misleading 0.
        assert "push_lag_seconds" in pane
        if not pane.get("ahead"):
            assert pane["push_lag_seconds"] is None


def _git(repo, *args, now_env=None):
    import subprocess
    base = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
            "-c", "core.hooksPath=", "-c", "commit.gpgsign=false"]
    subprocess.run(base + list(args), check=True, capture_output=True, text=True)


def test_git_pane_push_lag_trips_action(tmp_path=None) -> None:
    """Integration witness: a local commit AHEAD of its upstream, older than the
    threshold, trips the git pane to ACTION — the keep-git-up-to-date gate firing
    on committed work that stopped reaching origin. Tolerant of a git-less env."""
    import subprocess
    import tempfile
    from datetime import timedelta
    try:
        td = tempfile.mkdtemp()
        repo = Path(td)
        _git(repo, "init", "-q", "-b", "main")
        (repo / "a.txt").write_text("a", encoding="utf-8")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "-q", "-m", "A", "--no-gpg-sign")
        # Simulate an upstream sitting at commit A, then commit B locally (unpushed).
        # @{upstream} resolves only with a remote url + fetch refspec to map the
        # merge ref onto the tracking ref, so configure a self-referential origin.
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
    # now far past B's commit time -> lag exceeds the 45m default -> ACTION.
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    pane = fs.git_pane(repo, now=future)
    assert pane["ahead"] == 1 and pane["behind"] == 0
    assert pane["push_lag_seconds"] is not None and pane["push_lag_seconds"] > 45 * 60
    assert pane["verdict"] == "ACTION" and pane["ok"] is False
    assert "push to origin" in pane["reason"]
    # A generous threshold keeps the same ahead state green (no false alarm).
    ok_pane = fs.git_pane(repo, now=future, push_lag_action_seconds=10 ** 9)
    assert ok_pane["verdict"] == "OK" and ok_pane["ahead"] == 1


# --- the rollup fold: verdict ladder ----------------------------------------

def _panes(git_v="OK", bench_v="OK", work_v="SKIP", ind_v="OK", *, bench_stale=False) -> list:
    return [
        {"key": "git", "label": "git", "verdict": git_v, "ok": git_v == "OK",
         "reason": "abc (main), clean", "sha": "abc", "branch": "main", "dirty": 0,
         "ahead": 0, "behind": 0},
        {"key": "benchmarks", "label": "benchmarks", "verdict": bench_v,
         "ok": bench_v == "OK", "reason": "55 runs", "stale": bench_stale,
         "provenance": {"measured": 1, "modeled": 1, "functional": 1, "unknown": 1}, "runs": 55,
         "machines": 5, "newest": "x", "age_days": 1},
        {"key": "work", "label": "work", "verdict": work_v, "ok": work_v != "ACTION",
         "reason": "0 plans"},
        {"key": "industry", "label": "industry", "verdict": ind_v, "ok": ind_v == "OK",
         "reason": "parity 0"},
    ]


def _fold(panes):
    return fs.fold(panes, workspace=".", commit="c0", generated_at="2026-06-25T12:00:00Z")


def test_fold_all_green_when_no_action() -> None:
    out = _fold(_panes())
    assert out["ok"] is True and out["verdict"] == "OK" and out["finding"] == "all_green"
    assert out["schema"] == fs.SCHEMA and set(out["panes"]) == {"git", "benchmarks", "work", "industry"}


def test_fold_soft_skip_never_trips() -> None:
    # work SKIP + industry SKIP must still read green.
    out = _fold(_panes(work_v="SKIP", ind_v="SKIP"))
    assert out["verdict"] == "OK" and "skipped" in out["reason"]


def test_fold_hard_git_error_trips_to_action() -> None:
    out = _fold(_panes(git_v="ERROR"))
    assert out["ok"] is False and out["verdict"] == "ACTION" and out["finding"] == "needs_attention"
    assert "git" in out["next_action"]


def test_fold_stale_benchmarks_trips_and_points_to_catalog() -> None:
    out = _fold(_panes(bench_v="ACTION", bench_stale=True))
    assert out["verdict"] == "ACTION"
    assert "bench_catalog.py build" in out["next_action"]


def test_fold_git_push_lag_points_to_push_not_repo_fix() -> None:
    # A git ACTION that is a push LAG (sha present, push_lag_seconds set) must point
    # at pushing, NOT at the "not a repo / git unavailable" ERROR remedy.
    panes = _panes()
    panes[0].update({"verdict": "ACTION", "ok": False, "ahead": 3,
                     "push_lag_seconds": 50 * 60,
                     "reason": "abc (main), clean tree, +3/-0 vs upstream, 3 unpushed, oldest 50m old"})
    out = _fold(panes)
    assert out["verdict"] == "ACTION"
    assert "push to origin" in out["next_action"] and "50m" in out["next_action"]
    assert "not a repo" not in out["next_action"]


def test_render_lists_every_pane() -> None:
    text = fs.render(_fold(_panes()))
    for label in ("git", "benchmarks", "work", "industry"):
        assert label in text
    assert "fresh status" in text


def test_render_doc_has_provenance_and_panes() -> None:
    doc = fs.render_doc(_fold(_panes()), date="2026-06-25")
    assert "Fresh status snapshot (2026-06-25)" in doc
    assert "measured" in doc and "modeled" in doc and "unknown" in doc
    assert "| Pane | Verdict | Detail |" in doc


# --- main wiring ------------------------------------------------------------

def test_main_check_returns_zero_when_green() -> None:
    """`main(['--check'])` returns 0 when the rollup is OK — driven with a stubbed
    collect so it's fast and deterministic, not the live sub-tools."""
    orig = fs.collect
    try:
        fs.collect = lambda root, **_kw: _panes()
        assert fs.main(["--check"]) == 0
        # a hard git error -> non-zero advisory exit
        fs.collect = lambda root, **_kw: _panes(git_v="ERROR")
        assert fs.main(["--check"]) == 1
    finally:
        fs.collect = orig


def test_main_json_emits_envelope() -> None:
    import contextlib
    import io
    import json as _json
    orig = fs.collect
    try:
        fs.collect = lambda root, **_kw: _panes()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = fs.main(["--json"])
        out = _json.loads(buf.getvalue())
        assert code == 0 and out["schema"] == fs.SCHEMA
        assert set(out["panes"]) == {"git", "benchmarks", "work", "industry"}
    finally:
        fs.collect = orig


# --- tolerant live smoke ----------------------------------------------------

def test_live_collect_and_fold() -> None:
    root = fs.repo_root()
    panes = fs.collect(root, timeout=90)
    assert len(panes) == 4
    out = fs.fold(panes, workspace=str(root), commit=fs.head_commit(root),
                  generated_at="now")
    for field in ("schema", "ok", "verdict", "finding", "reason", "next_action", "panes"):
        assert field in out, f"missing {field}"
    # git + benchmarks must be present as HARD panes (no silent drop).
    assert "git" in out["panes"] and "benchmarks" in out["panes"]


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
