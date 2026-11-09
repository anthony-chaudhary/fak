#!/usr/bin/env python3
"""Tests for the concept-coverage drift watcher.

Builds a TINY synthetic workspace (three .go files in internal/demo mentioning
demoCachePolicy / demoCachePolicyHint / demoCacheGate across all three files) plus a
minimal data directory (two families, two rows files), and exercises the watcher end
to end through its CLI: record-baseline writes the file; check-mode is clean (exit 0,
delta 0, receipt written next to the baseline); removing the biggest rows file in a
temp copy re-opens the debt (exit 3, the right tokens uncovered per family); and
--simulate detects its own drift (the self-witness). Also the pure helpers: BOM /
UTF-16 JSON decoding, row-token counting, and victim picking.

Run: `python tools/concept_drift_watcher_test.py`  (exit 0 = all pass),
or `python -m pytest tools/concept_drift_watcher_test.py -q`.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import concept_drift_watcher as dw  # noqa: E402


# NOTE: the comments must not spell the family roots (cache / gate / guard) as
# standalone words - the identifier sweep would discover them as extra tokens.
GO_SRC = '''package demo

// DemoCachePolicy selects the demo policy set.
func DemoCachePolicy() string { return "lru" }

// DemoCachePolicyHint refines the demo policy choice.
func DemoCachePolicyHint() string { return "warm" }

// DemoCacheGate secures the demo entry point.
func DemoCacheGate() bool { return true }
'''


def build_workspace(root: Path) -> tuple[Path, Path]:
    """Tiny synthetic workspace: 3 go files x 3 identifiers + minimal data dir.

    The three identifiers each span all three files (presence 3 >= min_files 2), so
    the scorecard discovers exactly three confusable tokens: democachepolicy,
    democachepolicyhint (family cache) and democachegate (families cache AND
    guard-gate; deduped once in the headline universe). Two rows files position all
    three, so the clean state is coverage-debt 0."""
    ws = root / "ws"
    demo = ws / "internal" / "demo"
    demo.mkdir(parents=True)
    for name in ("a.go", "b.go", "c.go"):
        (demo / name).write_text(GO_SRC, encoding="utf-8")
    data = root / "data"
    data.mkdir(parents=True)
    (data / "_meta.json").write_text(json.dumps({
        "meta": {"as_of": "2026-11-08", "fak_version": "t"},
        "families": [
            {"id": "cache", "roots": ["cache"], "min_files": 2},
            {"id": "guard-gate", "roots": ["gate", "guard"], "min_files": 2},
        ],
    }), encoding="utf-8")
    rows_cache = {"rows": [
        {"id": "dc1", "canonical": "Demo Cache Policy", "family": "cache",
         "kind": "concept", "definition": "the demo cache policy selector.",
         "distinction": "not the hint variant", "distinct_from": ["dc2"],
         "aliases": [], "grounding": "demoCachePolicy", "grounding_kind": "symbol",
         "glossary_anchor": "", "verdict": "defined", "gaps": []},
        {"id": "dc2", "canonical": "Demo Cache Policy Hint", "family": "cache",
         "kind": "concept", "definition": "the demo cache policy hint.",
         "distinction": "not the base policy", "distinct_from": ["dc1"],
         "aliases": [], "grounding": "demoCachePolicyHint", "grounding_kind": "symbol",
         "glossary_anchor": "", "verdict": "defined", "gaps": []},
    ]}
    rows_gate = {"rows": [
        {"id": "dg1", "canonical": "Demo Cache Gate", "family": "guard-gate",
         "kind": "concept", "definition": "the demo cache gate guard.",
         "distinction": "guards the demo cache entry point", "distinct_from": [],
         "aliases": [], "grounding": "demoCacheGate", "grounding_kind": "symbol",
         "glossary_anchor": "", "verdict": "defined", "gaps": []},
    ]}
    (data / "rows-cache.json").write_text(json.dumps(rows_cache), encoding="utf-8")
    (data / "rows-gate.json").write_text(json.dumps(rows_gate), encoding="utf-8")
    return ws, data


def _cli(*args: str) -> int:
    return dw.main(list(args))


# --- pure helpers ----------------------------------------------------------

def test_decode_json_bytes_handles_utf8_bom_and_utf16() -> None:
    obj = {"a": 1}
    assert dw.decode_json_bytes(json.dumps(obj).encode("utf-8")) == obj
    assert dw.decode_json_bytes(b"\xef\xbb\xbf" + json.dumps(obj).encode("utf-8")) == obj
    # Python's utf-16 codec emits a BOM-prefixed payload, like PowerShell 5 '>' does.
    assert dw.decode_json_bytes(json.dumps(obj).encode("utf-16")) == obj


def test_row_tokens_counts_canonical_grounding_aliases() -> None:
    r = {"canonical": "KV cache", "grounding": "kv_cache", "aliases": ["vCache", "kvCache"]}
    assert dw.row_tokens(r) == {"kvcache", "vcache"}
    assert dw.row_tokens({"canonical": "", "grounding": "", "aliases": [None]}) == set()


def test_pick_victim_prefers_most_tokens_then_name(tmp_dir=None) -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "data"
        d.mkdir()
        (d / "_meta.json").write_text(json.dumps({"families": []}), encoding="utf-8")
        (d / "rows-a.json").write_text(json.dumps({"rows": [
            {"canonical": "Alpha", "grounding": "alpha", "aliases": []},
            {"canonical": "Beta", "grounding": "beta", "aliases": []}]}), encoding="utf-8")
        (d / "rows-b.json").write_text(json.dumps({"rows": [
            {"canonical": "Gamma", "grounding": "gamma", "aliases": []}]}), encoding="utf-8")
        victim, toks = dw.pick_victim(d)
        assert victim.name == "rows-a.json" and toks == {"alpha", "beta"}
        # Tie on token count breaks deterministically by file name.
        (d / "rows-a.json").unlink()
        (d / "rows-c.json").write_text(json.dumps({"rows": [
            {"canonical": "Delta", "grounding": "delta", "aliases": []}]}), encoding="utf-8")
        victim, toks = dw.pick_victim(d)
        assert victim.name == "rows-b.json" and toks == {"gamma"}


# --- end to end through the CLI --------------------------------------------

def test_record_baseline_writes_the_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ws, data = build_workspace(root)
        baseline = root / "baseline.json"
        rc = _cli("--workspace", str(ws), "--data", str(data),
                  "--baseline", str(baseline), "--record-baseline")
        assert rc == 0, rc
        doc = json.loads(baseline.read_text(encoding="utf-8"))
        assert doc["schema"] == dw.BASELINE_SCHEMA
        assert doc["discovered"] == 3 and doc["covered"] == 3
        assert doc["coverage_debt"] == 0
        assert doc["recorded_at"] and isinstance(doc["standing"], dict)
        assert doc["standing"]["defined"] == 3


def test_check_mode_clean_exit_0_with_receipt() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ws, data = build_workspace(root)
        baseline = root / "baseline.json"
        assert _cli("--workspace", str(ws), "--data", str(data),
                    "--baseline", str(baseline), "--record-baseline") == 0
        rc = _cli("--workspace", str(ws), "--data", str(data),
                  "--baseline", str(baseline))
        assert rc == 0, rc
        receipts = list(root.glob("drift-receipt-*.json"))
        assert len(receipts) == 1
        rec = json.loads(receipts[0].read_text(encoding="utf-8"))
        assert rec["schema"] == dw.RECEIPT_SCHEMA and rec["kind"] == "check"
        assert rec["verdict"] == "OK" and rec["drift"] is False
        assert rec["delta"] == {"coverage_debt": 0, "discovered": 0}
        assert rec["uncovered_by_family"] == {}


def test_check_mode_removed_rows_file_exit_3() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ws, data = build_workspace(root)
        baseline = root / "baseline.json"
        assert _cli("--workspace", str(ws), "--data", str(data),
                    "--baseline", str(baseline), "--record-baseline") == 0
        # Simulate the merge losing the biggest rows file: copy the data dir, remove it.
        shrunk = root / "shrunk-data"
        shutil.copytree(data, shrunk)
        (shrunk / "rows-cache.json").unlink()   # covers 2 tokens - the most
        rc = _cli("--workspace", str(ws), "--data", str(shrunk),
                  "--baseline", str(baseline))
        assert rc == 3, rc
        rec = json.loads(next(root.glob("drift-receipt-*.json")).read_text(encoding="utf-8"))
        assert rec["verdict"] == "DRIFT" and rec["drift"] is True
        assert rec["delta"]["coverage_debt"] == 2 and rec["delta"]["discovered"] == 0
        assert set(rec["uncovered_by_family"]["cache"]) == \
            {"democachepolicy", "democachepolicyhint"}


def test_simulate_detects_its_own_drift() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ws, data = build_workspace(root)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _cli("--workspace", str(ws), "--data", str(data), "--simulate")
        out = buf.getvalue()
        assert rc == 0, rc
        rec = json.loads(out[out.index("{"):])
        assert rec["kind"] == "simulate" and rec["drift_detected"] is True
        assert rec["verdict"] == "DRIFT-DETECTED"
        assert rec["removed_file"] == "rows-cache.json"
        assert set(rec["removed_tokens"]) == {"democachepolicy", "democachepolicyhint"}
        # The self-witness is a DIFF: clean pre-state, debt re-opened post-removal,
        # and every newly-uncovered token attributable to the removed file.
        assert rec["pre"]["coverage_debt"] == 0 and rec["post"]["coverage_debt"] == 2
        assert rec["delta"]["coverage_debt"] == 2
        assert set(rec["newly_uncovered"]) == {"democachepolicy", "democachepolicyhint"}


def test_check_mode_without_baseline_is_operational_error() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ws, data = build_workspace(root)
        rc = _cli("--workspace", str(ws), "--data", str(data),
                  "--baseline", str(root / "nope.json"))
        assert rc == 2, rc


def test_all_background_spawns_suppress_windows_console() -> None:
    # Mock every launch: this test never creates a ticket or a child process.
    for flags in (0, 0x08000000):
        with mock.patch.object(dw, "no_window_creationflags", return_value=flags):
            with mock.patch.object(dw.subprocess, "run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = b"abc123\n"
                assert dw.git_head(Path(".")) == "abc123"
                assert run.call_args.kwargs["creationflags"] == flags
                assert run.call_args.kwargs["timeout"] == dw.GIT_TIMEOUT_S
            with mock.patch.object(dw.subprocess, "run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = b'{}'
                assert dw.run_scorecard(Path(".")) == ({}, "")
                assert run.call_args.kwargs["creationflags"] == flags
                assert run.call_args.kwargs["timeout"] == dw.SCORECARD_TIMEOUT_S
            with mock.patch.object(dw.shutil, "which", return_value="fak-run"):
                with mock.patch.object(dw.subprocess, "run") as run:
                    run.return_value.returncode = 0
                    dw.create_drift_ticket(Path("."), 1, ["demo"])
                    assert run.call_args.kwargs["creationflags"] == flags
                    assert run.call_args.kwargs["timeout"] == dw.TICKET_TIMEOUT_S


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
