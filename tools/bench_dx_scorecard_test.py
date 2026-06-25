#!/usr/bin/env python3
"""Tests for the bench-DX scorecard  -  the benchmarking-developer-experience stick.

Drives the PURE parsing + fold with in-memory fixtures (each KPI's clean and
defect case), then closes with the load-bearing LIVE smoke against the real
tracked tree: the catalog verb + registry must exist, the registry must cover the
cmd/ tree one-to-one, and the headline number must be a low-debt A/B grade  -  the
proof that a developer can actually find, start, learn, read, and compare a
benchmark, and a regression sentinel for the day someone removes an affordance.

Run: `python tools/bench_dx_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/bench_dx_scorecard_test.py -q`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_dx_scorecard as bd  # noqa: E402

ROOT = bd.repo_root()


# --- pure helpers -----------------------------------------------------------

def test_grade_bands() -> None:
    assert bd.grade_letter(92) == "A"
    assert bd.grade_letter(83) == "B"
    assert bd.grade_letter(50) == "F"


def test_cmd_bench_mains_from_paths() -> None:
    tracked = [
        "cmd/modelbench/main.go",
        "cmd/radixbench/main.go",
        "cmd/fak/main.go",          # excluded (the binary itself)
        "cmd/fak/benchmarks.go",
        "cmd/turntaxdemo/main.go",  # not a bench (no 'bench' in name)
        "internal/x/y.go",
    ]
    mains = bd.cmd_bench_mains(tracked)
    assert mains == {"modelbench", "radixbench"}, mains


def test_build_payload_fold_and_verdict() -> None:
    kpis = [
        {"name": "a", "group": "discover", "passed": False, "debt": 3, "detail": ""},
        {"name": "b", "group": "coldstart", "passed": True, "debt": 0, "detail": ""},
    ]
    p = bd.build_payload("w", kpis)
    assert p["corpus"]["bench_dx_debt"] == 3
    assert p["corpus"]["debt_by_group"]["discover"] == 3
    assert not p["ok"]
    clean = bd.build_payload("w", [{"name": "a", "group": "discover", "passed": True, "debt": 0, "detail": ""}])
    assert clean["ok"]


def test_compare_renders_2x_verdict() -> None:
    base = {"corpus": {"bench_dx_debt": 35, "score": 14.0, "debt_by_group": {}}}
    cur = {"corpus": {"bench_dx_debt": 1, "score": 92.0, "debt_by_group": {}}}
    out = bd.render_compare(base, cur)
    assert ">=2x bench-DX-debt reduction achieved" in out
    not_yet = bd.render_compare(base, {"corpus": {"bench_dx_debt": 30, "score": 20, "debt_by_group": {}}})
    assert "not yet 2x" in not_yet


def test_spurious_path_kpi_fixture(tmp_path: Path = None) -> None:
    # The doubled-root bug: a cmd/*bench* default flag pointing at fak/experiments.
    import tempfile, os
    d = Path(tempfile.mkdtemp())
    (d / "go.mod").write_text("module x\n", encoding="utf-8")
    bad = d / "cmd" / "paritybench"
    bad.mkdir(parents=True)
    (bad / "main.go").write_text(
        'package main\nimport "flag"\nfunc main(){ _ = flag.String("ref", "fak/experiments/parity/x.json", "h") }\n',
        encoding="utf-8")
    # Make it a git repo so _tracked works.
    subprocess.run(["git", "init", "-q"], cwd=d)
    subprocess.run(["git", "add", "-A"], cwd=d)
    passed, debt, detail = bd.kpi_no_spurious_paths(d, ["cmd/paritybench/main.go"])
    assert not passed and debt == 1, (passed, debt, detail)
    assert "fak/experiments/parity" in detail


# --- the load-bearing live smoke against the real tracked tree --------------

def test_live_catalog_verb_and_registry_present() -> None:
    payload = bd.collect(ROOT)
    assert not payload.get("error"), payload.get("error")
    by = {k["name"]: k for k in payload["kpis"]}
    assert by["catalog_verb"]["passed"], by["catalog_verb"]["detail"]
    assert by["catalog_registry"]["passed"], by["catalog_registry"]["detail"]


def test_live_registry_covers_tree() -> None:
    payload = bd.collect(ROOT)
    by = {k["name"]: k for k in payload["kpis"]}
    assert by["registry_covers_tree"]["passed"], by["registry_covers_tree"]["detail"]


def test_live_headline_is_low_debt_good_grade() -> None:
    payload = bd.collect(ROOT)
    c = payload["corpus"]
    # The benchmarking DX must stay an A/B with at most one tolerated known-debt
    # item (the -quant polarity convention, deliberately left for the bench owner).
    assert c["bench_dx_debt"] <= 1, f"bench-DX-debt regressed to {c['bench_dx_debt']}: {[k for k in payload['kpis'] if not k['passed']]}"
    assert c["grade"] in ("A", "B"), c


def test_json_roundtrips() -> None:
    payload = bd.collect(ROOT)
    assert json.loads(json.dumps(payload))["schema"] == bd.SCHEMA


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERR  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
