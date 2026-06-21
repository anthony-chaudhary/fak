#!/usr/bin/env python3
"""Regression tests for fleet_resume_watchdog.py's cross-account env wiring.

These pin the parity the .ps1 already had and the .py port had drifted from:
  * REG_DIR must follow FLEET_REG_DIR so the watchdog READS the registry/plan/ledger
    from the same dir the fleet_sessions.py refresh child WRITES (the silent-no-op /
    split-resume-once-ledger blocker when an ambient FLEET_REG_DIR is set by the
    control pane or an operator).
  * CLAUDE_EXE must prefer the fleet-wide FLEET_CLAUDE_EXE convention (account_probe.py
    et al.), with FAK_CLAUDE_EXE only a back-compat fallback.
  * Active probing must stay gated to the live tick so a default dry-run spends nothing.

The module reads the env-derived constants at import time, so each test reloads it
under a patched environment. Pure stdlib; no process spawn, no network.

Run:  python -m pytest tools/fleet_resume_watchdog_test.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _reload(env):
    """Reload fleet_resume_watchdog with `env` overlaid, then restore the environment."""
    saved = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        import fleet_resume_watchdog as wd
        return importlib.reload(wd)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_reg_dir_follows_fleet_reg_dir(tmp_path):
    target = str(tmp_path / "host_registry")
    wd = _reload({"FLEET_REG_DIR": target})
    assert wd.REG_DIR == target, "watchdog must read where fleet_sessions.py writes"


def test_reg_dir_defaults_to_local_registry():
    wd = _reload({"FLEET_REG_DIR": None})
    assert wd.REG_DIR == os.path.join(wd.HERE, "_registry")


def test_claude_exe_prefers_fleet_convention(tmp_path):
    fleet_bin = str(tmp_path / "claude-fleet")
    wd = _reload({"FLEET_CLAUDE_EXE": fleet_bin, "FAK_CLAUDE_EXE": str(tmp_path / "fak")})
    assert wd.CLAUDE_EXE == fleet_bin


def test_claude_exe_falls_back_to_fak(tmp_path):
    fak_bin = str(tmp_path / "claude-fak")
    wd = _reload({"FLEET_CLAUDE_EXE": None, "FAK_CLAUDE_EXE": fak_bin})
    assert wd.CLAUDE_EXE == fak_bin


def test_probe_mode_dry_run_is_side_effect_free():
    wd = _reload({})
    # auto resolves to a real probe only on a live tick; dry-run probes nothing.
    assert wd.resolve_probe_mode("auto", live=False) == "none"
    assert wd.resolve_probe_mode("auto", live=True) == "blocked"


def test_probe_mode_explicit_setting_is_honored():
    wd = _reload({})
    assert wd.resolve_probe_mode("all", live=False) == "all"
    assert wd.resolve_probe_mode("none", live=True) == "none"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
