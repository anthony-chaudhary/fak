#!/usr/bin/env python3
"""Gate witness for tools/dogfood_coverage.py (issue #731).

`tools/dogfood_coverage_test.py` already proves the *scoring* responds to the live
env (guard-off lowers coverage / raises debt). This file proves the thing CI
actually invokes: the `--check` CLI **exit code** is a real gate — it returns 0
when HARD `dogfood_debt == 0` and 1 the moment a HARD KPI regresses. Without this,
a green `--check` in `.github/workflows/dogfood-coverage.yml` could be a tool that
always passes; here we flip a HARD KPI and watch the gate go RED.

The check is made hermetic by pinning `FAK_BIN` to an existing file (the running
Python), so the HARD `fak_bin_resolvable` / `fleet_leaf_guarded` KPIs are met
WITHOUT needing a built `fak` on the box — the self-test can therefore run BEFORE
the `go build` step in CI. The single variable under test is then
`FLEET_DOGFOOD_GUARD`: unset (default ON) => debt 0 => exit 0; `=0` => guard
disabled => two HARD KPIs unmet => exit 1.

Run: `python tools/dogfood_coverage_gate_test.py`  (exit 0 = all pass),
or `python -m pytest tools/dogfood_coverage_gate_test.py -q`.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dogfood_coverage.py"


def _env(guard: str | None) -> dict[str, str]:
    """A hermetic env: a resolvable FAK_BIN (the Python exe) so the binary KPIs are
    met without a built fak, with FLEET_DOGFOOD_GUARD as the single knob."""
    e = dict(os.environ)
    e["FAK_BIN"] = sys.executable  # an existing file => fak_bin_resolvable=True
    e.pop("FLEET_DOGFOOD_GUARD", None)
    if guard is not None:
        e["FLEET_DOGFOOD_GUARD"] = guard
    return e


def _run_check(guard: str | None) -> int:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=str(ROOT), env=_env(guard),
        capture_output=True, text=True,
    )
    return proc.returncode


def test_check_passes_when_guard_default_on() -> None:
    # Guard unset == default ON; with a resolvable fak bin the HARD KPIs are met.
    assert _run_check(None) == 0, "gate should PASS (exit 0) at debt 0"


def test_check_fails_when_guard_disabled() -> None:
    # FLEET_DOGFOOD_GUARD=0 unwraps the worker => fleet_leaf_guarded + guard_default_on
    # both regress => the gate must FAIL (exit 1). This is the regression CI must catch.
    assert _run_check("0") == 1, "gate should FAIL (exit 1) when guard is disabled"


def test_evaluate_api_agrees_with_cli_gate() -> None:
    # Belt-and-braces: the in-process evaluate() debt must agree with the CLI verdict,
    # so the gate can't drift from the score it claims to enforce.
    import importlib.util

    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dogfood_coverage", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    on = mod.evaluate(ROOT, env=_env(None))
    off = mod.evaluate(ROOT, env=_env("0"))
    assert on["dogfood_debt"] == 0, on["dogfood_debt"]
    assert off["dogfood_debt"] > 0, off["dogfood_debt"]
    assert off["coverage"] < on["coverage"], (off["coverage"], on["coverage"])


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    if failures:
        print(f"\n{failures} test(s) failed")
        return 1
    print("\nall dogfood-coverage gate tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
