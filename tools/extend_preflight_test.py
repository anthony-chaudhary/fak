#!/usr/bin/env python3
"""Tests for extend_preflight.py — the contributor/extension readiness gate.

Run:  python -m pytest tools/extend_preflight_test.py
  or:  python tools/extend_preflight_test.py   (standalone)

These are deterministic and read-only: they exercise the pure logic (stamp regex,
check shape, summarize/render) and assert that THIS repo passes its own preflight, so a
regression that removes a golden-path entry point (rsicycle, architest, the compute seam)
trips the test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extend_preflight as ep  # noqa: E402

EXPECTED_NAMES = {
    "git-guards-installed",
    "on-master",
    "ship-stamp-convention",
    "gate1-plug-in",
    "gate2-prove-correct",
    "gate3-prove-faster",
    "golden-path-docs",
    "test-path",
}


def test_check_names_stable():
    names = {c["name"] for c in ep.run_checks()}
    assert names == EXPECTED_NAMES, f"unexpected check set: {names ^ EXPECTED_NAMES}"


def test_every_check_has_shape():
    for c in ep.run_checks():
        assert set(c) == {"name", "level", "ok", "detail", "fix"}
        assert c["level"] in (ep.ERROR, ep.WARN, ep.INFO)
        assert isinstance(c["ok"], bool)
        assert c["detail"]


def test_stamp_regex_accepts_the_three_ship_forms():
    assert ep.STAMP_RE.search("fix(gateway): treat same-tick ready (fak gateway)")
    assert ep.STAMP_RE.search("docs(extending): add the golden path (fak docs)")
    assert ep.STAMP_RE.search("fak/architest: extend the no-os/exec rule")  # legacy prefix
    assert ep.STAMP_RE.search("v0.29.0: cut the release")  # release anchor


def test_stamp_regex_rejects_an_unstamped_subject():
    assert not ep.STAMP_RE.search("fix(gateway): treat same-tick ready as positive")
    assert not ep.STAMP_RE.search("just some random commit message")


def test_required_gate_entry_points_present_in_this_repo():
    # The three gates and their docs must resolve in a healthy checkout — this is the
    # regression guard for the golden path itself.
    by_name = {c["name"]: c for c in ep.run_checks()}
    for gate in ("gate1-plug-in", "gate2-prove-correct", "gate3-prove-faster",
                 "golden-path-docs", "git-guards-installed"):
        assert by_name[gate]["ok"], f"{gate} failed: {by_name[gate]['detail']}"


def test_summarize_ok_when_no_required_failure():
    result = ep.summarize(ep.run_checks())
    assert result["ok"] is True, f"required failures: {result['failed_required']}"
    assert result["doc"] == "fak/EXTENDING.md"
    assert len(result["golden_path"]) == 4


def test_summarize_reports_required_failures():
    fake = [
        ep._check("gate1-plug-in", ep.ERROR, False, "missing", "fix it"),
        ep._check("on-master", ep.WARN, False, "off master", "switch"),  # warn != required
    ]
    result = ep.summarize(fake)
    assert result["ok"] is False
    assert result["failed_required"] == ["gate1-plug-in"]  # the WARN does not count


def test_render_contains_golden_path_and_doc():
    text = ep.render(ep.summarize(ep.run_checks()), quiet=False)
    assert "EXTENDING.md" in text
    assert "Gate 1" in text and "Gate 2" in text and "Gate 3" in text


def test_main_json_is_valid_and_exit_zero(capsys):
    rc = ep.main(["--json"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["ok"] is True
    assert {c["name"] for c in parsed["checks"]} == EXPECTED_NAMES
    assert rc == 0


def _run_standalone() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            # the capsys-dependent test is pytest-only; skip it standalone
            if "capsys" in fn.__code__.co_varnames:
                rc = ep.main(["--json"])
                assert rc == 0
            else:
                fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
