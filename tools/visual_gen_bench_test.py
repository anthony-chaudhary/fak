#!/usr/bin/env python3
"""Tests for visual_gen_bench.py -- the grade -> DB -> RSI seam.

These pin the three load-bearing behaviors that make the loop trustworthy:
  - the durable DB write produces a scan-able run dir + manifest with the score;
  - the baseline is read BACK from a prior DB run (so RSI compares against the
    durable record, not an in-memory guess);
  - the ACTION/OK verdict keys off "below floor OR regressed", and the keep-bit is
    recorded as evidence, never fabricated in Python.

The catalog-fold (bench_catalog build) and the keep-bit (go run) and the self-test
(pytest) are external subprocesses; tests stub them so the suite is hermetic and
fast. A separate smoke test exercises the real grader on the real deck.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visual_gen_bench as b  # noqa: E402
import visual_gen_grade as grade  # noqa: E402

INIT = "%%{init: {'theme':'base'}}%%"
SVG_GOOD = ('<svg id="s" width="100" height="50" viewBox="0 0 100 50">'
            "<text><tspan>real</tspan></text></svg>")
SVG_BROKEN = ('<svg id="s" width="100%" viewBox="0 0 100 50">'
              "<foreignObject><div>x</div></foreignObject></svg>")
CLEAN_FLOW = (f"{INIT}\nflowchart TD\n  a[\"x\"]:::k\n  b[\"y\"]:::k\n  a --> b\n"
              "  classDef k fill:#fff;\n")


@pytest.fixture
def fake_deck(tmp_path, monkeypatch):
    """A tiny visuals dir + an isolated runs/by-machine dir, wired into the module so
    no real catalog or deck is touched. External subprocesses are stubbed."""
    visuals = tmp_path / "visuals"
    visuals.mkdir()
    runs = tmp_path / "runs" / "by-machine"
    runs.mkdir(parents=True)

    monkeypatch.setattr(grade, "DEFAULT_DIR", visuals)
    monkeypatch.setattr(b, "RUNS_DIR", runs)
    # never shell out in unit tests
    monkeypatch.setattr(b, "fold_into_catalog", lambda: None)
    monkeypatch.setattr(b, "git_context", lambda: {"rev": "test", "branch": "master",
                                                    "dirty": False})
    monkeypatch.setattr(b, "truth_clean", lambda: True)
    monkeypatch.setattr(b, "suite_green", lambda: True)
    # default keep-bit stub: KEEP iff strict gain (mirrors shipgate, no go needed)
    monkeypatch.setattr(b, "run_keep_bit", _fake_keep_bit)
    return visuals, runs


def _fake_keep_bit(before, after, suite_green, truth_ok):
    kept = (after > before) and suite_green and truth_ok
    return {"decision": "KEEP" if kept else "REVERT", "kept": kept,
            "witness": {"before": before, "after": after}}


def write_fig(visuals, base, svg):
    (visuals / f"{base}.mmd").write_text(CLEAN_FLOW, encoding="utf-8")
    (visuals / f"{base}.svg").write_text(svg, encoding="utf-8")


# --------------------------------------------------------------------------------
# clause 2: the durable DB write
# --------------------------------------------------------------------------------

def test_db_write_produces_manifest_and_report(fake_deck):
    visuals, runs = fake_deck
    write_fig(visuals, "good", SVG_GOOD)
    out = b.run_bench(machine="m", now="20260620T000000Z", floor=0.8,
                      do_db=True, do_rsi=True, do_render=False, do_suite=True)
    run_dir = runs / "m" / "20260620T000000Z-visualgen"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "visualgen-report.json").exists()
    man = json.loads((run_dir / "manifest.json").read_text())
    assert man["$schema"] == b.RUN_MANIFEST_SCHEMA
    assert man["run_id"] == "m-visualgen-20260620T000000Z"
    assert "visual-gen" in man["tags"]
    assert man["summary"]["mean_score"] == out["mean_score"]
    # the keep-bit evidence is recorded in the manifest, not acted on
    assert man["rsi"]["decision"] in ("KEEP", "REVERT", "UNAVAILABLE")


def test_no_db_skips_catalog(fake_deck):
    visuals, runs = fake_deck
    write_fig(visuals, "good", SVG_GOOD)
    out = b.run_bench(machine="m", now="20260620T000000Z", floor=0.8,
                      do_db=False, do_rsi=True, do_render=False, do_suite=True)
    assert not (runs / "m").exists()
    assert "run_dir" not in out


# --------------------------------------------------------------------------------
# clause 1: baseline read-back + keep-bit feeds RSI
# --------------------------------------------------------------------------------

def test_first_run_has_null_baseline_and_no_strict_gain(fake_deck):
    visuals, _ = fake_deck
    write_fig(visuals, "good", SVG_GOOD)
    out = b.run_bench(machine="m", now="20260620T000000Z", floor=0.8,
                      do_db=True, do_rsi=True, do_render=False, do_suite=True)
    assert out["baseline"] is None
    # first run: before==after, so no strict gain -> not kept
    assert out["rsi"]["kept"] is False


def test_baseline_is_read_back_from_prior_db_run(fake_deck):
    visuals, _ = fake_deck
    # run 1: a broken figure -> lower score, lands in the DB
    write_fig(visuals, "fig", SVG_BROKEN)
    r1 = b.run_bench(machine="m", now="20260620T000000Z", floor=0.8,
                     do_db=True, do_rsi=True, do_render=False, do_suite=True)
    # run 2: fix the figure -> higher score; baseline must be run 1's score
    write_fig(visuals, "fig", SVG_GOOD)
    r2 = b.run_bench(machine="m", now="20260620T010000Z", floor=0.8,
                     do_db=True, do_rsi=True, do_render=False, do_suite=True)
    assert r2["baseline"] == pytest.approx(r1["mean_score"])
    assert r2["mean_score"] > r1["mean_score"]
    # a genuine improvement over the durable baseline is KEEP
    assert r2["rsi"]["decision"] == "KEEP"
    assert r2["rsi"]["kept"] is True


def test_regression_is_action_and_not_kept(fake_deck):
    visuals, _ = fake_deck
    write_fig(visuals, "fig", SVG_GOOD)
    b.run_bench(machine="m", now="20260620T000000Z", floor=0.8,
                do_db=True, do_rsi=True, do_render=False, do_suite=True)
    # second run regresses the figure
    write_fig(visuals, "fig", SVG_BROKEN)
    r2 = b.run_bench(machine="m", now="20260620T010000Z", floor=0.8,
                     do_db=True, do_rsi=True, do_render=False, do_suite=True)
    assert r2["regressed"] is True
    assert r2["ok"] is False
    assert "regressed" in r2["reason"]
    assert r2["rsi"]["kept"] is False


# --------------------------------------------------------------------------------
# the ACTION/OK verdict contract
# --------------------------------------------------------------------------------

def test_below_floor_is_action(fake_deck):
    visuals, _ = fake_deck
    write_fig(visuals, "broken", SVG_BROKEN)  # render checks fail -> below 0.8
    out = b.run_bench(machine="m", now="20260620T000000Z", floor=0.8,
                      do_db=False, do_rsi=False, do_render=False, do_suite=False)
    assert out["n_below_floor"] >= 1
    assert out["ok"] is False
    assert "below" in out["reason"]


def test_healthy_deck_is_ok(fake_deck):
    visuals, _ = fake_deck
    write_fig(visuals, "good", SVG_GOOD)  # scores 1.0
    out = b.run_bench(machine="m", now="20260620T000000Z", floor=0.8,
                      do_db=False, do_rsi=False, do_render=False, do_suite=False)
    assert out["n_below_floor"] == 0
    assert out["ok"] is True
    assert "healthy" in out["reason"]


def test_keep_bit_not_fabricated_when_go_unavailable(fake_deck, monkeypatch):
    # If rsicycle cannot run, the decision is UNAVAILABLE and kept stays False --
    # the Python side never invents a KEEP.
    visuals, _ = fake_deck
    write_fig(visuals, "good", SVG_GOOD)
    monkeypatch.setattr(b, "run_keep_bit",
                        lambda *a, **k: {"decision": "UNAVAILABLE", "kept": False,
                                         "witness": {}, "note": "no go"})
    out = b.run_bench(machine="m", now="20260620T000000Z", floor=0.8,
                      do_db=True, do_rsi=True, do_render=False, do_suite=True)
    assert out["rsi"]["decision"] == "UNAVAILABLE"
    assert out["rsi"]["kept"] is False


# --------------------------------------------------------------------------------
# smoke: the real deck, real grader (no DB/RSI subprocess)
# --------------------------------------------------------------------------------

def test_main_json_smoke_real_deck():
    rc = b.main(["--json", "--no-db", "--no-rsi", "--no-suite"])
    assert rc == 0


def test_run_keep_bit_parses_rsicycle_output(monkeypatch):
    # Unit the parser against rsicycle's real stdout shape without invoking go.
    class P:
        stdout = ("metric=visual_gen_mean_score before=0.700 after=0.716 ...\n"
                  "DECISION=KEEP kept=true\n")
        returncode = 0
    monkeypatch.setattr(b.subprocess, "run", lambda *a, **k: P())
    res = b.run_keep_bit(0.700, 0.716, True, True)
    assert res["decision"] == "KEEP"
    assert res["kept"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
