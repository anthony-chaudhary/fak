#!/usr/bin/env python3
"""Hermetic tests for tools/dispatch_preflight.py.

The spawn gate composes four INDEPENDENT checks (host_check, account_check,
kernel_alive, proc_worker_count). All of those shell out to other tools, so here
we replace them on the module with synthetic results and assert the verdict
logic — SPAWN_OK and every typed REFUSE_* — plus the pure helpers (_last_json,
_int) and the cap = min(max_workers, dos target) rule. No subprocess runs.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dispatch_preflight.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dispatch_preflight", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_checks(mod, *, host=None, account=None, kernel=None, procs=0):
    """Replace the four shelling-out checks with constant synthetic results."""
    host = host if host is not None else {"safe": True, "flagged": 0, "flagged_names": []}
    account = account if account is not None else {
        "available": True, "tag": "worker-a", "dir": "/acct/a", "tier": 1,
        "model": "claude", "reason": "free", "blocked": []}
    kernel = kernel if kernel is not None else {"alive": 0, "target": 3, "verdict": "FILLING"}
    mod.host_check = lambda root, **kw: host
    mod.account_check = lambda root, **kw: account
    mod.kernel_alive = lambda root: kernel
    mod.proc_worker_count = lambda: procs


def run_eval(mod, **kw):
    defaults = dict(max_workers=2, work_kind="engineering", product="claude")
    defaults.update(kw)
    return mod.evaluate(ROOT, **defaults)


class LastJsonTest(unittest.TestCase):
    def test_parses_whole_text_when_it_is_one_object(self) -> None:
        mod = load()
        self.assertEqual(mod._last_json('{\n "ok": true,\n "n": 2\n}\n'), {"ok": True, "n": 2})

    def test_returns_last_json_object_line_amid_noise(self) -> None:
        mod = load()
        text = 'starting up...\nnot json\n{"a": 1}\n{"verdict": "X"}\n'
        self.assertEqual(mod._last_json(text), {"verdict": "X"})

    def test_empty_or_nonobject_yields_empty_dict(self) -> None:
        mod = load()
        self.assertEqual(mod._last_json(""), {})
        self.assertEqual(mod._last_json("[1,2,3]"), {})
        self.assertEqual(mod._last_json("plain log line"), {})


class IntTest(unittest.TestCase):
    def test_coerces_and_falls_back(self) -> None:
        mod = load()
        self.assertEqual(mod._int("5"), 5)
        self.assertEqual(mod._int(7), 7)
        self.assertIsNone(mod._int(None))
        self.assertIsNone(mod._int("nope"))
        self.assertEqual(mod._int("nope", 0), 0)


class EvaluateVerdictTest(unittest.TestCase):
    def test_spawn_ok_when_host_clean_account_free_under_cap(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 3, "verdict": "FILLING"}, procs=0)
        p = run_eval(mod, max_workers=2)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], mod.OK_VERDICT)
        self.assertEqual(p["cap"], 2)  # min(max_workers=2, target=3)
        self.assertEqual(p["live"], 0)
        self.assertEqual(p["headroom"], 2)

    def test_refuse_host_when_guard_flags_a_process(self) -> None:
        mod = load()
        patch_checks(mod, host={"safe": False, "flagged": 2,
                                "flagged_names": ["llama-cli", "orphan"]})
        p = run_eval(mod)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], mod.REFUSE_HOST)
        self.assertIn("flagged 2 process", p["reason"])
        self.assertIn("llama-cli", p["reason"])

    def test_refuse_no_account_when_switcher_has_none(self) -> None:
        mod = load()
        patch_checks(mod, account={"available": False, "tag": None, "tier": None,
                                   "reason": "all throttled", "blocked": ["worker-x"]})
        p = run_eval(mod)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], mod.REFUSE_NO_ACCOUNT)
        self.assertIn("blocked: worker-x", p["reason"])
        self.assertIn("all throttled", p["reason"])

    def test_refuse_at_cap_when_live_meets_cap(self) -> None:
        mod = load()
        # cap = min(max_workers=2, target=5) = 2; os procs = 2 -> live 2 >= cap 2.
        patch_checks(mod, kernel={"alive": 0, "target": 5, "verdict": "FULL"}, procs=2)
        p = run_eval(mod, max_workers=2)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], mod.REFUSE_AT_CAP)
        self.assertEqual(p["cap"], 2)
        self.assertEqual(p["live"], 2)

    def test_refuse_at_cap_uses_max_of_kernel_and_os_views(self) -> None:
        mod = load()
        # kernel alive=1, os procs=3 -> live = max(1,3) = 3 >= cap 3.
        patch_checks(mod, kernel={"alive": 1, "target": 9, "verdict": "X"}, procs=3)
        p = run_eval(mod, max_workers=3)
        self.assertEqual(p["live"], 3)
        self.assertEqual(p["verdict"], mod.REFUSE_AT_CAP)
        self.assertEqual(p["os_worker_procs"], 3)

    def test_refuse_inspect_when_host_check_errored(self) -> None:
        mod = load()
        patch_checks(mod, host={"safe": False, "error": "guard not found", "flagged": 0})
        p = run_eval(mod)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], mod.REFUSE_INSPECT)
        self.assertIn("guard not found", p["reason"])

    def test_refuse_inspect_when_kernel_check_errored(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": None, "target": None, "error": "dos loop crashed"})
        p = run_eval(mod)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], mod.REFUSE_INSPECT)
        self.assertIn("dos loop crashed", p["reason"])

    def test_cap_is_min_of_max_workers_and_dos_target(self) -> None:
        mod = load()
        # target 1 below max_workers 5 -> cap clamps to the dos target.
        patch_checks(mod, kernel={"alive": 0, "target": 1, "verdict": "X"}, procs=0)
        p = run_eval(mod, max_workers=5)
        self.assertEqual(p["cap"], 1)

    def test_cap_falls_back_to_max_workers_when_target_unknown(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": None, "target": None, "verdict": None}, procs=0)
        p = run_eval(mod, max_workers=4)
        self.assertEqual(p["cap"], 4)


class RenderTest(unittest.TestCase):
    def test_render_does_not_raise_on_evaluate_output(self) -> None:
        mod = load()
        patch_checks(mod)
        text = mod.render(run_eval(mod))
        self.assertIn("dispatch preflight", text)
        self.assertIn("SPAWN_OK", text)


if __name__ == "__main__":
    unittest.main()
