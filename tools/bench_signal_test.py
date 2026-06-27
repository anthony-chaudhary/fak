#!/usr/bin/env python3
"""Unit tests for bench-signal's pure core — the machine-scoped key, the
current-value fold, the dual noise threshold (relative % AND absolute floor), the
open-issue dedup, the worst-drop-first cap, and the issue render. In-memory fixtures
only (no gh, no catalog file, no network), so the testable seam runs hermetically.

Dual-runnable (the repo runs the suite pytest-free in CI):
    python tools/bench_signal_test.py
    python -m pytest tools/bench_signal_test.py -q
"""
from __future__ import annotations

import bench_signal as bs


# --- fixtures ---------------------------------------------------------------
def _run(machine: str, model: str, prec: str, peak, ts: str, run_id: str = "") -> dict:
    return {
        "machine_id": machine, "model": model, "precision": prec,
        "peak_tok_per_sec": peak, "baseline_tok_per_sec": None,
        "speedup": None, "timestamp": ts, "tags": [],
        "run_id": run_id or f"{machine}-{model}-{ts}", "path": "",
    }


def _baseline(pins: dict, commit: str = "base123") -> dict:
    return {"schema": bs.BASELINE_SCHEMA, "commit": commit, "pinned": "2026-06-27",
            "baselines": pins}


# --- key + value helpers ----------------------------------------------------
def test_benchmark_key_separates_precision():
    a = bs.benchmark_key({"model": "Qwen2.5-1.5B", "precision": "Q8_0"})
    b = bs.benchmark_key({"model": "Qwen2.5-1.5B", "precision": "Q4_K"})
    assert a != b, "the same model at a different quant tracks apart"
    assert "qwen2.5-1.5b" in a


def test_full_key_is_machine_scoped():
    k1 = bs.full_key("cpu-server-a", "qwen2.5-1.5b-q8")
    k2 = bs.full_key("gcp-g2-l4", "qwen2.5-1.5b-q8")
    assert k1 != k2, "the same benchmark on two machines has two keys"
    assert k1.startswith("cpu-server-a/")


def test_run_value_prefers_peak_skips_null():
    assert bs.run_value(_run("m", "x", "q8", 31.0, "t")) == 31.0
    assert bs.run_value(_run("m", "x", "q8", None, "t")) is None
    # a non-positive number is not a real value
    assert bs.run_value({"peak_tok_per_sec": 0, "baseline_tok_per_sec": 0}) is None


def test_current_by_key_takes_latest_timestamp():
    runs = [_run("m", "x", "q8", 30.0, "20260101T000000Z"),
            _run("m", "x", "q8", 25.0, "20260201T000000Z")]
    cur = bs.current_by_key(runs)
    key = bs.full_key("m", bs.benchmark_key(runs[0]))
    assert cur[key]["value"] == 25.0, "the later run wins"


def test_current_by_key_skips_null_runs():
    runs = [_run("m", "x", "q8", 30.0, "20260101T000000Z"),
            _run("m", "x", "q8", None, "20260301T000000Z")]  # newer but null
    cur = bs.current_by_key(runs)
    key = bs.full_key("m", bs.benchmark_key(runs[0]))
    assert cur[key]["value"] == 30.0, "a newer null does not overwrite a real datum"


# --- threshold: regressions() -----------------------------------------------
def test_regression_fires_past_both_thresholds():
    runs = [_run("m", "x", "q8", 20.0, "t")]  # baseline 30 -> 20 = -33%, -10 tok/s
    cur = bs.current_by_key(runs)
    key = list(cur)[0]
    regs = bs.regressions(cur, _baseline({key: 30.0}), min_drop_pct=15.0, min_abs=1.0)
    assert len(regs) == 1
    assert round(regs[0]["drop_pct"], 1) == 33.3 and regs[0]["drop_abs"] == 10.0


def test_drop_past_pct_but_under_abs_floor_is_skipped():
    # 2.0 -> 1.6 tok/s = -20% (past %) but only -0.4 tok/s (under the 1.0 floor).
    runs = [_run("m", "tiny", "q8", 1.6, "t")]
    cur = bs.current_by_key(runs)
    key = list(cur)[0]
    regs = bs.regressions(cur, _baseline({key: 2.0}), min_drop_pct=15.0, min_abs=1.0)
    assert regs == [], "a big % on a tiny number is below the absolute floor -> noise"


def test_drop_past_abs_but_under_pct_is_skipped():
    # 100 -> 95 tok/s = -5 tok/s (past floor) but only -5% (under the 15% threshold).
    runs = [_run("m", "fast", "q8", 95.0, "t")]
    cur = bs.current_by_key(runs)
    key = list(cur)[0]
    regs = bs.regressions(cur, _baseline({key: 100.0}), min_drop_pct=15.0, min_abs=1.0)
    assert regs == [], "a small % wobble on a fast benchmark is noise"


def test_improvement_is_not_a_regression():
    runs = [_run("m", "x", "q8", 40.0, "t")]
    cur = bs.current_by_key(runs)
    key = list(cur)[0]
    regs = bs.regressions(cur, _baseline({key: 30.0}), min_drop_pct=15.0, min_abs=1.0)
    assert regs == [], "faster than baseline is not a regression"


def test_unpinned_key_is_skipped():
    runs = [_run("m", "x", "q8", 10.0, "t")]
    cur = bs.current_by_key(runs)
    # baseline has no entry for this key -> nothing to compare against.
    regs = bs.regressions(cur, _baseline({}), min_drop_pct=15.0, min_abs=1.0)
    assert regs == [], "a never-pinned benchmark has no baseline to regress against"


def test_machine_scoping_does_not_cross_compare():
    # Same model+prec on two machines; only machine A regressed. B must be untouched.
    runs = [_run("a", "x", "q8", 10.0, "t"),   # A: 30 -> 10, regressed
            _run("b", "x", "q8", 29.0, "t")]   # B: 30 -> 29, flat
    cur = bs.current_by_key(runs)
    keya = bs.full_key("a", bs.benchmark_key(runs[0]))
    keyb = bs.full_key("b", bs.benchmark_key(runs[1]))
    regs = bs.regressions(cur, _baseline({keya: 30.0, keyb: 30.0}),
                          min_drop_pct=15.0, min_abs=1.0)
    assert [r["key"] for r in regs] == [keya], "only machine A's drop fires"


def test_regressions_sorted_worst_drop_first():
    runs = [_run("a", "x", "q8", 20.0, "t"),   # -33%
            _run("b", "y", "q8", 5.0, "t")]    # -83%
    cur = bs.current_by_key(runs)
    regs = bs.regressions(cur, _baseline({list(cur)[0]: 30.0, list(cur)[1]: 30.0}),
                          min_drop_pct=15.0, min_abs=1.0)
    assert regs[0]["drop_pct"] > regs[1]["drop_pct"], "worst drop first"


# --- dedup + cap + render ---------------------------------------------------
def _one_reg():
    runs = [_run("cpu-server-a", "Qwen2.5-1.5B", "Q8_0", 20.0, "t", "run-xyz")]
    cur = bs.current_by_key(runs)
    key = list(cur)[0]
    return bs.regressions(cur, _baseline({key: 30.0}), min_drop_pct=15.0, min_abs=1.0)[0], key


def test_marker_roundtrip_and_dedup():
    _, key = _one_reg()
    body = "x\n" + bs.marker(key) + "\ny"
    assert bs.open_issue_keys([{"number": 1, "body": body}]) == {key}


def test_plan_skips_already_open_key():
    reg, key = _one_reg()
    to_file, stats = bs.plan_issues([reg], open_keys={key}, max_issues=5,
                                    today="2026-06-27")
    assert to_file == [] and stats["already-open"] == 1


def test_steady_state_all_open_files_nothing():
    # Storm bound (the #979 report->arm acceptance): on a day when EVERY current
    # regression is already tracked by an OPEN bench-signal issue, the autonomous
    # run (the daily schedule is dry-run; an armed `--live` dispatch is the only
    # mutator) plans ZERO new issues — no new file — regardless of the cap. So a
    # live run over an unchanged-but-open backlog is idempotent: the no-storm
    # guarantee rests entirely on the planner's label-scoped dedup, not on a human
    # in the loop. Distinct from test_plan_skips_already_open_key (one reg): this
    # locks the MULTI-regression steady state under a cap large enough to file all.
    runs = [_run(f"m{i}", "x", "q8", 30.0 - (i + 2) * 4, "t") for i in range(4)]
    cur = bs.current_by_key(runs)
    regs = bs.regressions(cur, _baseline({k: 30.0 for k in cur}),
                          min_drop_pct=15.0, min_abs=1.0)
    assert len(regs) == 4, "fixture: four real regressions to suppress"
    open_keys = {r["key"] for r in regs}            # every one already tracked open
    to_file, stats = bs.plan_issues(regs, open_keys=open_keys, max_issues=99,
                                    today="2026-06-27")
    assert to_file == [], "all-open steady state files nothing"
    assert stats["already-open"] == 4
    assert stats["over-cap"] == 0, "nothing planned -> nothing capped (idempotent)"


def test_plan_caps_worst_first():
    runs = [_run(f"m{i}", "x", "q8", 30.0 - (i + 2) * 3, "t") for i in range(5)]
    cur = bs.current_by_key(runs)
    pins = {k: 30.0 for k in cur}
    regs = bs.regressions(cur, _baseline(pins), min_drop_pct=15.0, min_abs=1.0)
    to_file, stats = bs.plan_issues(regs, open_keys=set(), max_issues=2,
                                    today="2026-06-27")
    assert len(to_file) == 2 and stats["over-cap"] == len(regs) - 2


def test_render_has_marker_numbers_and_remeasure_path():
    reg, key = _one_reg()
    issue = bs.render_issue(reg, today="2026-06-27")
    assert issue["labels"] == [bs.SIGNAL_LABEL, bs.PERF_LABEL]
    assert bs.marker(key) in issue["body"], "dedup anchor present"
    assert "30.0 -> 20.0 tok/s" in issue["body"], "before->after numbers"
    assert "cpu-server-a" in issue["body"], "machine named"
    assert "tools/bench_signal.py" in issue["body"], "re-measure path (routes lane)"
    assert "#N" in issue["body"], "the worker's #N-stamp contract"
    assert "-33.3%" in issue["title"]


def test_build_baseline_snapshots_current():
    runs = [_run("m", "x", "q8", 31.0, "t")]
    cur = bs.current_by_key(runs)
    base = bs.build_baseline(cur, "abc1234", "2026-06-27")
    key = list(cur)[0]
    assert base["baselines"][key] == 31.0 and base["commit"] == "abc1234"


def test_rendered_body_routes_to_tools_lane():
    import issue_lane_router as router
    reg, _ = _one_reg()
    issue = bs.render_issue(reg, today="2026-06-27")
    routed = router.route_issue(
        {"number": 1, "title": issue["title"], "body": issue["body"],
         "labels": [{"name": lab} for lab in issue["labels"]]},
        ["tools", "ci", "cmd"], {"tools": ["tools/**"], "ci": [".github/**"]})
    assert routed["lane"] == "tools", f"re-measure path routes to tools, got {routed!r}"
    assert routed["confidence"] == "path-confirmed"


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
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
    raise SystemExit(_run_all())
