#!/usr/bin/env python3
"""Tests for the managed-cache proving ground (epic #1844).

Fixture ledgers drive the row contracts, the rung fold, and the baseline ratchet;
one live smoke runs the real repo ledgers through the same spine so the committed
evidence can never silently stop parsing.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import managed_cache_proving_ground as pg  # noqa: E402


def _savings_row(**over) -> dict:
    read = over.pop("cache_read_tokens", 1000)
    creation = over.pop("cache_creation_tokens", 100)
    row = {
        "schema": pg.SAVINGS_SCHEMA,
        "date": "2026-07-03",
        "session_type": "guard",
        "provider": "anthropic",
        "mechanism": "provider_prompt_cache",
        "context": "claude",
        "generated_at": "2026-07-03T01:00:00Z",
        "input_tokens": 10,
        "cache_read_tokens": read,
        "cache_creation_tokens": creation,
        "output_tokens": 5,
        "saved_token_equiv": 0.9 * read - 0.25 * creation,
        "net_saved_token_equiv": 0.9 * read - 0.25 * creation,
    }
    row.update(over)
    return row


def _usage_row(**over) -> dict:
    counters = {"submits": 0, "input_tokens": 0, "compaction_fired": 0,
                "kv_prefix_reused_tokens": 0}
    counters.update(over.pop("counters", {}))
    row = {
        "schema": pg.USAGE_SCHEMA,
        "kind": "exit",
        "session_type": "serve",
        "context": "stdio",
        "pid": 1,
        "unix_millis": 1,
        "counters": counters,
        "generated_at": "2026-07-03T02:00:00Z",
    }
    row.update(over)
    return row


def _value_row(**over) -> dict:
    row = {
        "schema": pg.VALUE_SCHEMA,
        "date": "2026-07-03",
        "session_type": "run",
        "context": "smollm2",
        "turns": 2,
        "prompt_tokens": 100,
        "reused_tokens": 40,
        "generated_at": "2026-07-03T03:00:00Z",
    }
    row.update(over)
    return row


def _write_root(tmp: Path, savings: list[dict], usage: list[dict], value: list[dict]) -> Path:
    nightrun = tmp / "docs" / "nightrun"
    nightrun.mkdir(parents=True, exist_ok=True)
    for name, rows in (("cache-savings.jsonl", savings),
                       ("gateway-usage.jsonl", usage),
                       ("cache-value.jsonl", value)):
        (nightrun / name).write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return tmp


def _rung(report: dict, cid: str) -> str:
    for c in report["concepts"]:
        if c["id"] == cid:
            return c["rung"]
    raise AssertionError(f"concept {cid} missing from report")


def test_row_validators():
    assert pg.validate_savings_row(_savings_row(), "x:1") == []
    assert any("SCHEMA_MISMATCH" in v for v in
               pg.validate_savings_row(_savings_row(schema="bogus/9"), "x:1"))
    assert any("UNKNOWN_MECHANISM" in v for v in
               pg.validate_savings_row(_savings_row(mechanism="mystery"), "x:1"))
    assert any("SAVED_FORMULA_MISMATCH" in v for v in
               pg.validate_savings_row(_savings_row(saved_token_equiv=123456.0), "x:1"))
    assert any("NEGATIVE_COUNTER" in v for v in
               pg.validate_savings_row(_savings_row(input_tokens=-1), "x:1"))

    assert pg.validate_usage_row(_usage_row(), "x:1") == []
    assert any("BAD_COUNTER" in v for v in
               pg.validate_usage_row(_usage_row(counters={"submits": -2}), "x:1"))
    assert any("MISSING_COUNTERS" in v for v in
               pg.validate_usage_row({"schema": pg.USAGE_SCHEMA, "kind": "exit",
                                      "session_type": "serve"}, "x:1"))

    assert pg.validate_value_row(_value_row(), "x:1") == []
    assert any("REUSE_EXCEEDS_PROMPT" in v for v in
               pg.validate_value_row(_value_row(prompt_tokens=10, reused_tokens=20), "x:1"))
    print("test_row_validators OK")


def test_rung_fold():
    with tempfile.TemporaryDirectory() as td:
        root = _write_root(Path(td), [_savings_row()], [_usage_row()], [_value_row()])
        report = pg.build_report(root)
        assert report["violation_count"] == 0, report["violations"]
        assert _rung(report, "provider_prompt_cache_passthrough") == "EVIDENCED"
        # Provider rows prove the writer ran, so zero shed rows is a witnessed zero.
        assert _rung(report, "compaction_shed") == "SILENT_ZERO"
        assert _rung(report, "kv_prefix_reuse") == "EVIDENCED"
        # No ttl counter key anywhere -> the C6 lever has no durable channel.
        assert _rung(report, "ttl_upgrade_1h") == "UNWIRED"
        assert _rung(report, "uncached_remainder_shrink") == "UNIMPLEMENTED"
        # Usage ledger carries serve rows only -> guard usage plane not yet landed.
        assert _rung(report, "guard_usage_plane") == "CHANNEL_READY"

        # A ttl-bearing counter key auto-upgrades C6 without a tool change.
        root = _write_root(Path(td), [_savings_row()],
                           [_usage_row(counters={"cache_ttl_upgrades_upgraded": 0})],
                           [_value_row()])
        report = pg.build_report(root)
        assert _rung(report, "ttl_upgrade_1h") == "CHANNEL_READY"

        # A real shed row climbs compaction to EVIDENCED.
        shed = _savings_row(mechanism="compaction_shed", provider="fak",
                            cache_read_tokens=0, cache_creation_tokens=0,
                            saved_token_equiv=500.0, net_saved_token_equiv=500.0,
                            compaction_shed_tokens=500)
        root = _write_root(Path(td), [_savings_row(), shed], [_usage_row()], [_value_row()])
        report = pg.build_report(root)
        assert report["violation_count"] == 0, report["violations"]
        assert _rung(report, "compaction_shed") == "EVIDENCED"

        # Guard usage rows with live counters climb the usage plane to EVIDENCED.
        root = _write_root(Path(td), [_savings_row()],
                           [_usage_row(session_type="guard", counters={"input_tokens": 9})],
                           [_value_row()])
        report = pg.build_report(root)
        assert _rung(report, "guard_usage_plane") == "EVIDENCED"
    print("test_rung_fold OK")


def test_baseline_ratchet():
    with tempfile.TemporaryDirectory() as td:
        root = _write_root(Path(td), [_savings_row(), _savings_row()],
                           [_usage_row()], [_value_row()])
        baseline = pg.baseline_snapshot(pg.build_report(root))

        # Identity holds.
        assert pg.check_against_baseline(pg.build_report(root), baseline) == []

        # Appends keep the ratchet green (concurrent sessions writing rows).
        root = _write_root(Path(td), [_savings_row()] * 3, [_usage_row()] * 2, [_value_row()])
        assert pg.check_against_baseline(pg.build_report(root), baseline) == []

        # A shrunken append-only ledger is a named regression.
        root = _write_root(Path(td), [_savings_row()], [_usage_row()], [_value_row()])
        problems = pg.check_against_baseline(pg.build_report(root), baseline)
        assert any(p.startswith("REGRESSION_ROW_COUNT cache_savings") for p in problems), problems

        # New violations trip the ratchet.
        root = _write_root(Path(td), [_savings_row(), _savings_row(),
                                      _savings_row(saved_token_equiv=1.0)],
                           [_usage_row()], [_value_row()])
        problems = pg.check_against_baseline(pg.build_report(root), baseline)
        assert any(p.startswith("REGRESSION_VIOLATIONS") for p in problems), problems

        # A rung falling down the ladder is a named regression: baseline saw the
        # ttl channel, then the counter key vanishes from the row shape.
        root = _write_root(Path(td), [_savings_row(), _savings_row()],
                           [_usage_row(counters={"cache_ttl_upgrades_upgraded": 0})],
                           [_value_row()])
        with_ttl = pg.baseline_snapshot(pg.build_report(root))
        root = _write_root(Path(td), [_savings_row(), _savings_row()],
                           [_usage_row()], [_value_row()])
        problems = pg.check_against_baseline(pg.build_report(root), with_ttl)
        assert any(p.startswith("REGRESSION_RUNG ttl_upgrade_1h") for p in problems), problems

        # Schema drift is named.
        drifted = dict(baseline)
        drifted["ledger_schemas"] = dict(baseline["ledger_schemas"], cache_value="bogus/2")
        root = _write_root(Path(td), [_savings_row(), _savings_row()], [_usage_row()], [_value_row()])
        problems = pg.check_against_baseline(pg.build_report(root), drifted)
        assert any(p.startswith("SCHEMA_DRIFT cache_value") for p in problems), problems
    print("test_baseline_ratchet OK")


def test_cli_exit_codes():
    with tempfile.TemporaryDirectory() as td:
        root = _write_root(Path(td), [_savings_row()], [_usage_row()], [_value_row()])
        baseline = str(Path(td) / "baseline.json")
        # --check with no baseline is a harness error, never a silent pass.
        assert pg.main(["--root", str(root), "--baseline", baseline, "--check"]) == 2
        assert pg.main(["--root", str(root), "--baseline", baseline, "--write-baseline"]) == 0
        assert pg.main(["--root", str(root), "--baseline", baseline, "--check"]) == 0
        # Shrink the ledger -> ratchet regression.
        _write_root(Path(td), [], [_usage_row()], [_value_row()])
        assert pg.main(["--root", str(root), "--baseline", baseline, "--check"]) == 1
    print("test_cli_exit_codes OK")


def test_live_repo_smoke():
    """The committed ledgers must always parse and fold — the proving ground's floor."""
    report = pg.build_report(ROOT)
    assert report["schema"] == pg.REPORT_SCHEMA
    assert len(report["concepts"]) == 7
    ids = {c["id"] for c in report["concepts"]}
    assert "ttl_upgrade_1h" in ids and "provider_prompt_cache_passthrough" in ids
    for c in report["concepts"]:
        assert c["rung"] in pg.RUNGS
    # The real guard population has been writing provider rows since 2026-07-01;
    # that floor never goes back below EVIDENCED.
    assert _rung(report, "provider_prompt_cache_passthrough") == "EVIDENCED"
    print(f"test_live_repo_smoke OK ({report['ledgers']['cache_savings']['rows']} savings rows, "
          f"{report['violation_count']} violations)")


if __name__ == "__main__":
    test_row_validators()
    test_rung_fold()
    test_baseline_ratchet()
    test_cli_exit_codes()
    test_live_repo_smoke()
    print("managed_cache_proving_ground_test OK")
