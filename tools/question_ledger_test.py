#!/usr/bin/env python3
"""Self-running tests for tools/question_ledger.py — prove each labeling rule bites.

Run: python3 tools/question_ledger_test.py   (exit 0 = all pass).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import question_ledger as ql

GOOD = {
    "id": "Q-20260708-001",
    "ts": "2026-07-08T00:00:00Z",
    "category": "CONTRARIAN",
    "target": "the witness thesis",
    "question": "Is the witness layer itself witnessed?",
    "why": "trust may have just moved up a level",
    "status": "open",
    "ticket": None,
}


def _rows(objs):
    """Turn a list of dicts (or raw strings) into read_rows-shaped tuples."""
    out = []
    for i, o in enumerate(objs, start=1):
        raw = o if isinstance(o, str) else json.dumps(o)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = None
        out.append((i, obj, raw))
    return out


def test_good_row_is_clean():
    assert ql.lint_rows(_rows([GOOD])) == []


def test_bad_id_rejected():
    bad = {**GOOD, "id": "Q-2026-07-08-1"}
    assert any("Q-YYYYMMDD-NNN" in p for p in ql.lint_rows(_rows([bad])))


def test_duplicate_id_rejected():
    probs = ql.lint_rows(_rows([GOOD, {**GOOD}]))
    assert any("duplicate id" in p for p in probs)


def test_unknown_category_rejected():
    bad = {**GOOD, "category": "SPICY"}
    assert any("category" in p for p in ql.lint_rows(_rows([bad])))


def test_unknown_status_rejected():
    bad = {**GOOD, "status": "parked"}
    assert any("status" in p for p in ql.lint_rows(_rows([bad])))


def test_ticketed_without_number_rejected():
    bad = {**GOOD, "status": "ticketed", "ticket": None}
    assert any("requires a positive int ticket" in p for p in ql.lint_rows(_rows([bad])))


def test_open_with_ticket_rejected():
    bad = {**GOOD, "status": "open", "ticket": 42}
    assert any("must be null" in p for p in ql.lint_rows(_rows([bad])))


def test_ticketed_with_number_ok():
    ok = {**GOOD, "status": "ticketed", "ticket": 42}
    assert ql.lint_rows(_rows([ok])) == []


def test_boolean_ticket_rejected():
    # json true is an int subclass in python — the guard must reject it.
    bad = {**GOOD, "status": "ticketed", "ticket": True}
    assert any("positive int ticket" in p for p in ql.lint_rows(_rows([bad])))


def test_extra_key_rejected():
    bad = {**GOOD, "severity": "high"}
    assert any("unexpected key" in p for p in ql.lint_rows(_rows([bad])))


def test_missing_key_rejected():
    bad = {k: v for k, v in GOOD.items() if k != "why"}
    assert any("missing key" in p for p in ql.lint_rows(_rows([bad])))


def test_non_question_rejected():
    bad = {**GOOD, "question": "This is a statement, not a question."}
    assert any("must end with '?'" in p for p in ql.lint_rows(_rows([bad])))


def test_leak_absolute_path_rejected():
    bad = {**GOOD, "question": "Why does C:\\work\\fak leak here?"}
    assert any("leaks" in p for p in ql.lint_rows(_rows([bad])))


def test_leak_email_rejected():
    bad = {**GOOD, "why": "raised by someone@example.com"}
    assert any("leaks" in p for p in ql.lint_rows(_rows([bad])))


def test_bad_json_line_rejected():
    assert any("not valid JSON" in p for p in ql.lint_rows(_rows(["{not json"])))


def test_next_id_increments():
    rows = _rows([
        {**GOOD, "id": "Q-20260708-001"},
        {**GOOD, "id": "Q-20260708-005"},
    ])
    assert ql.next_id(rows, "20260708") == "Q-20260708-006"


def test_next_id_first_of_day():
    rows = _rows([{**GOOD, "id": "Q-20260708-009"}])
    assert ql.next_id(rows, "20260709") == "Q-20260709-001"


def test_dedupe_detects_identical():
    rows = _rows([GOOD])
    hit = ql.dedupe_match(rows, "Is the witness layer itself witnessed?")
    assert hit and hit["id"] == "Q-20260708-001"


def test_dedupe_passes_novel():
    rows = _rows([GOOD])
    assert ql.dedupe_match(rows, "What breaks if the nightly target is halved?") is None


def test_stats_counts():
    rows = _rows([
        {**GOOD, "id": "Q-20260708-001", "category": "CONTRARIAN", "status": "open"},
        {**GOOD, "id": "Q-20260708-002", "category": "AFRAID", "status": "ticketed", "ticket": 7},
    ])
    s = ql.stats(rows)
    assert s["total"] == 2
    assert s["by_category"]["CONTRARIAN"] == 1
    assert s["by_status"]["ticketed"] == 1


def test_lint_cli_on_real_ledger():
    # The shipped seed ledger must itself be clean through the CLI.
    root = Path(__file__).resolve().parent.parent
    real = root / "docs" / "questions" / "asked.jsonl"
    if real.exists():
        assert ql.main(["lint", "--ledger", str(real)]) == 0


def test_absent_ledger_is_empty_not_error():
    with tempfile.TemporaryDirectory() as d:
        missing = str(Path(d) / "nope.jsonl")
        assert ql.main(["lint", "--ledger", missing]) == 0
        assert ql.main(["next-id", "--ledger", missing, "--date", "20260708"]) == 0


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 — surface harness errors as failures
            failed += 1
            print(f"ERR  {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
