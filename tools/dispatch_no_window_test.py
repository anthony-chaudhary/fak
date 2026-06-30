#!/usr/bin/env python3
"""Regression: every console subprocess on the scheduled-dispatch path suppresses its
window.

The four ``\\JobSearch\\FleetIssueDispatch*`` scheduled tasks launch via ``pythonw.exe``
— a windowless parent with no console of its own. On Windows a console child spawned
WITHOUT ``CREATE_NO_WINDOW`` forces a fresh, VISIBLE console window, so every windowless
dispatch tick that shells out to ``gh`` / ``git`` / ``fak`` / ``taskkill`` / ``tasklist``
/ ``powershell`` pops its OWN window — the "random popup windows" that kept coming back.
The detached-worker spawn already suppressed this (``win_creationflags``); the regression
was the HELPER subprocess calls (issue fetch, liveness probes, account pin, reapers) going
out naked.

This test pins the shared suppressor's per-OS value and proves each helper call site
FORWARDS it instead of spawning naked — so a future naked ``subprocess.run`` on this path
fails CI instead of silently popping windows again.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import dispatch_account_topup  # noqa: E402
import dispatch_glm_docs  # noqa: E402
import dispatch_preflight  # noqa: E402
import dispatch_worker  # noqa: E402
import issue_resolve_dispatch as ird  # noqa: E402
import issue_worker_prompt  # noqa: E402

CREATE_NO_WINDOW = 0x08000000


def _ok(cmd, stdout=""):
    """A benign CompletedProcess so the call site's post-processing doesn't blow up."""
    return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")


class HelperValueTest(unittest.TestCase):
    """The suppressor is CREATE_NO_WINDOW on Windows, 0 on POSIX (where ``creationflags``
    must be 0)."""

    def _check(self, mod, fn):
        with mock.patch.object(mod.os, "name", "nt"):
            self.assertEqual(fn(), CREATE_NO_WINDOW)
        with mock.patch.object(mod.os, "name", "posix"):
            self.assertEqual(fn(), 0)

    def test_canonical_helper(self):
        self._check(dispatch_worker, dispatch_worker.no_window_creationflags)

    def test_ird_reexport_is_the_same_object(self):
        # The topup / glm-docs entry scripts reach the suppressor via `ird`.
        self.assertIs(ird.no_window_creationflags, dispatch_worker.no_window_creationflags)

    def test_leaf_helpers(self):
        self._check(dispatch_preflight, dispatch_preflight._no_window_creationflags)
        self._check(issue_worker_prompt, issue_worker_prompt._no_window_creationflags)

    def test_default_installer_adds_hidden_console_flag(self):
        captured = {}

        def base(*args, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        module = SimpleNamespace(
            run=base,
            Popen=base,
            call=base,
            check_call=base,
            check_output=base,
        )
        with mock.patch.object(dispatch_worker.os, "name", "nt"):
            dispatch_worker.install_no_window_subprocess_defaults(module)

        self.assertEqual(module.run(["git"]), "ok")
        self.assertEqual(captured["kwargs"]["creationflags"], CREATE_NO_WINDOW)

        module.Popen(["git"], creationflags=123)
        self.assertEqual(captured["kwargs"]["creationflags"], 123)


class ForwardsSuppressorTest(unittest.TestCase):
    """Each helper call site passes ``creationflags`` = the suppressor (CREATE_NO_WINDOW
    under a simulated Windows tick), never a naked subprocess."""

    def _assert_forwards(self, mod, invoke):
        captured = {}

        def fake_run(cmd, *a, **kw):
            captured["kw"] = kw
            return _ok(cmd, stdout=kw.get("_stdout", ""))

        # Force the Windows branch so os-guarded call sites (the taskkill reaper) are
        # exercised and the suppressor resolves to CREATE_NO_WINDOW regardless of the CI
        # host. os.name is read in dispatch_worker (the canonical/`ird` suppressor) and,
        # for a leaf module or an os-guarded call site, in the call module itself.
        os_modules = [dispatch_worker.os]
        if getattr(mod, "os", None) is not None and mod.os not in os_modules:
            os_modules.append(mod.os)
        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run):
            patches = [mock.patch.object(m, "name", "nt") for m in os_modules]
            for p in patches:
                p.start()
            try:
                invoke()
            finally:
                for p in patches:
                    p.stop()
        self.assertIn("creationflags", captured.get("kw", {}),
                      f"{mod.__name__}: helper subprocess.run went out NAKED (no creationflags)")
        self.assertEqual(captured["kw"]["creationflags"], CREATE_NO_WINDOW,
                         f"{mod.__name__}: creationflags is not CREATE_NO_WINDOW")

    def test_issue_worker_prompt_gh_view(self):
        self._assert_forwards(
            issue_worker_prompt,
            lambda: issue_worker_prompt.fetch_issue(1, workspace=Path(".")),
        )

    def test_account_topup_tasklist(self):
        self._assert_forwards(dispatch_account_topup, lambda: dispatch_account_topup._alive(999999))

    def test_glm_docs_tasklist(self):
        self._assert_forwards(dispatch_glm_docs, lambda: dispatch_glm_docs._alive(999999))

    def test_preflight_run_json(self):
        self._assert_forwards(
            dispatch_preflight,
            lambda: dispatch_preflight.run_json(["fak", "version"], cwd=Path("."), timeout=5),
        )

    def test_issue_resolve_taskkill_reaper(self):
        self._assert_forwards(ird, lambda: ird.terminate_issue_worker_tree(999999))


class DirectSpawnerIssueContractTest(unittest.TestCase):
    """Ad hoc direct spawners must not bypass the worker-ready issue contract."""

    def test_non_test_spawn_calls_are_contract_gated(self):
        offenders = []
        for path in sorted(TOOLS.glob("*.py")):
            if path.name.endswith("_test.py"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "spawn_issue_worker(" not in text:
                continue
            if path.name == "issue_resolve_dispatch.py":
                required = [
                    "def issue_contract_review(",
                    "issue_contract_gate",
                    "DEFAULT_ISSUE_CONTRACT_MIN_SCORE",
                ]
                missing = [marker for marker in required if marker not in text]
                if missing:
                    offenders.append(f"{path.name}: missing {', '.join(missing)}")
                continue

            review_at = text.find("issue_contract_review(")
            spawn_at = text.find("spawn_issue_worker(")
            if review_at < 0 or review_at > spawn_at:
                offenders.append(f"{path.name}: spawn before issue_contract_review")
            if "ISSUE_CONTRACT_HOLD" not in text:
                offenders.append(f"{path.name}: missing ISSUE_CONTRACT_HOLD report")

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
