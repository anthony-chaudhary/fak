#!/usr/bin/env python3
"""Tests for the tail-wag audit — the deterministic /tail-wag backing tool.

Drives the PURE core with fixtures (no disk needed): the tier-table and import
parsers, each signal's 1-5 bucket ladder + its boundaries, the candidate scorer, the
deterministic ranking key, and the fold to the control-pane payload across all three
verdict rungs (balanced / advisory / REVIEW) plus the tooling-error case. Closes with
the load-bearing live smoke: the REAL tracked tree must fold to a well-formed payload,
score the SAME way twice (determinism), and rank worst-first — the proof the screen is
clone-deterministic, and a sentinel for the day a parser or ladder drifts.

Run: `python tools/tail_wag_audit_test.py`  (exit 0 = all pass),
or `python -m pytest tools/tail_wag_audit_test.py -q`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tail_wag_audit as ta  # noqa: E402


# --- the parsers ------------------------------------------------------------

TIER_FIXTURE = '''
// preamble
var tier = map[string]int{
\t"abi": 0,

\t"appversion": 1, "blob": 1, "metrics": 1, // a comment, ignored
\t"ctxmmu": 2, "engine": 2,
\t"recall": 3,
\t"agent": 4, "gateway": 4, // trailing comment
}

func unrelated() {}
'''


def test_parse_tiers_extracts_the_map() -> None:
    tiers = ta.parse_tiers(TIER_FIXTURE)
    assert tiers["abi"] == 0 and tiers["appversion"] == 1
    assert tiers["ctxmmu"] == 2 and tiers["recall"] == 3 and tiers["agent"] == 4
    assert "unrelated" not in tiers  # the func, not a map entry
    assert len(tiers) == 9


def test_parse_tiers_empty_on_garbage() -> None:
    assert ta.parse_tiers(None) == {}
    assert ta.parse_tiers("no map here") == {}


def test_parse_internal_imports() -> None:
    src = (
        'package gateway\n'
        'import (\n'
        '\t"context"\n'
        '\t"github.com/anthony-chaudhary/fak/internal/abi"\n'
        '\t"github.com/anthony-chaudhary/fak/internal/engine"\n'
        '\t"github.com/anthony-chaudhary/fak/internal/webbench/browser"\n'
        '\t"github.com/other/pkg/internal/abi"\n'  # different module — must NOT match
        ')\n'
    )
    got = ta.parse_internal_imports(src)
    assert got == {"abi", "engine", "webbench/browser"}


def test_nonblank_lines() -> None:
    assert ta.nonblank_lines("a\n\n  \nb\n") == 2
    assert ta.nonblank_lines("") == 0


# --- the signal ladders + their boundaries ---------------------------------

def test_blast_score_boundaries() -> None:
    assert [ta.blast_score(n) for n in (0, 1, 2, 3, 4, 5, 6, 9, 10, 20)] == \
        [0, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_inversion_score_from_span() -> None:
    assert [ta.inversion_score(s) for s in (0, 1, 2, 3, 4, 5)] == [0, 2, 3, 4, 5, 5]


def test_cheapness_score_smaller_is_higher() -> None:
    assert [ta.cheapness_score(n) for n in (0, 80, 81, 200, 201, 500, 501, 1200, 1201)] == \
        [0, 5, 4, 4, 3, 3, 2, 2, 1]


# --- the candidate scorer + ranking ----------------------------------------

def test_score_candidate_loud() -> None:
    up = [(f"p{i}", 4) for i in range(6)]  # 6 higher-tier importers, span 3
    r = ta.score_candidate("appversion", 1, 75, up)
    assert r["core_fan_in"] == 6 and r["tier_span"] == 3
    assert r["inversion"] == 4 and r["blast"] == 4 and r["cheapness"] == 5
    assert r["product"] == 80 and r["is_finding"] and r["is_loud"]
    assert r["density_per_100loc"] == 8.0
    assert r["dogs"] == sorted(f"p{i}(integrator)" for i in range(6))


def test_score_candidate_advisory_not_loud() -> None:
    r = ta.score_candidate("foo", 1, 300, [("a", 3), ("b", 3)])
    assert r["inversion"] == 3 and r["blast"] == 2 and r["cheapness"] == 3
    assert r["is_finding"] and not r["is_loud"]  # blast 2 < LOUD


def test_score_candidate_not_a_finding() -> None:
    # one importer only one tier up, large module -> blast 1, cheapness 2: below floor
    r = ta.score_candidate("foo", 2, 600, [("a", 3)])
    assert r["blast"] == 1 and not r["is_finding"]


def test_rank_key_orders_by_product_then_fanin() -> None:
    big = ta.score_candidate("big", 1, 75, [(f"p{i}", 4) for i in range(6)])
    small = ta.score_candidate("small", 1, 300, [("a", 3), ("b", 3)])
    assert sorted([small, big], key=ta.rank_key)[0]["pkg"] == "big"


# --- the fold to the control-pane payload ----------------------------------

REQUIRED_FIELDS = ("schema", "ok", "verdict", "finding", "reason",
                   "next_action", "workspace", "corpus", "findings")


def _well_formed(p: dict) -> None:
    for f in REQUIRED_FIELDS:
        assert f in p, f"missing {f}"
    assert p["schema"] == ta.SCHEMA


def test_payload_balanced_when_no_findings() -> None:
    p = ta.build_payload(workspace="x", rows=[])
    _well_formed(p)
    assert p["ok"] and p["verdict"] == "OK" and p["finding"] == "balanced"
    assert p["corpus"]["candidate_findings"] == 0


def test_payload_advisory_when_findings_but_none_loud() -> None:
    rows = [ta.score_candidate("foo", 1, 300, [("a", 3), ("b", 3)])]
    p = ta.build_payload(workspace="x", rows=rows)
    _well_formed(p)
    assert p["ok"] and p["verdict"] == "OK" and p["finding"] == "advisory"
    assert p["corpus"]["candidate_findings"] == 1 and p["corpus"]["loud_findings"] == 0


def test_payload_review_when_loud() -> None:
    loud = ta.score_candidate("appversion", 1, 75, [(f"p{i}", 4) for i in range(6)])
    quiet = ta.score_candidate("foo", 1, 300, [("a", 3), ("b", 3)])
    p = ta.build_payload(workspace="x", rows=[quiet, loud])
    _well_formed(p)
    assert not p["ok"] and p["verdict"] == "REVIEW" and p["finding"] == "inverted_priority"
    assert p["corpus"]["loud_findings"] == 1
    assert p["findings"][0]["pkg"] == "appversion"  # worst first


def test_payload_error_case() -> None:
    p = ta.build_payload(workspace="x", rows=[], error="boom")
    _well_formed(p)
    assert not p["ok"] and p["verdict"] == "AUDIT_ERROR" and "boom" in p["reason"]


def test_renderers_do_not_crash() -> None:
    rows = [ta.score_candidate("appversion", 1, 75, [(f"p{i}", 4) for i in range(6)])]
    p = ta.build_payload(workspace="x", rows=rows)
    assert "tail-wag-audit" in ta.render(p)
    assert "Tail-wag audit" in ta.render_markdown(p, stamp="2026-06-24")
    assert "worst tail-wag index" in ta.render_compare(p, p)


# --- the load-bearing live smoke (the real tracked tree) -------------------

def test_live_payload_well_formed_and_deterministic() -> None:
    root = ta.repo_root()
    p1 = ta.collect(root)
    p2 = ta.collect(root)
    _well_formed(p1)
    # clone-deterministic: scoring the same tree twice is byte-identical.
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)
    assert p1["verdict"] in ("OK", "REVIEW", "AUDIT_ERROR")
    # this IS the fak repo, so the tier table parses and packages get scored.
    if p1["verdict"] != "AUDIT_ERROR":
        assert p1["corpus"]["packages_scored"] > 0
        # findings are ranked worst-first by the deterministic key.
        keys = [ta.rank_key(r) for r in p1["findings"]]
        assert keys == sorted(keys)
        for r in p1["findings"]:
            assert r["tier"] > ta.ROOT_TIER  # tier-0 root is excluded by construction
            assert r["is_finding"]


def main() -> int:
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: unexpected {type(exc).__name__}: {exc}")

    tests = {n: f for n, f in globals().items()
             if n.startswith("test_") and callable(f)}
    for name, fn in tests.items():
        check(name, fn)

    if failures:
        print(f"FAIL ({len(failures)}/{len(tests)}):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"ok ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
