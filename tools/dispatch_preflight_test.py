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
import os
import sys
import tempfile
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
    mod.proc_worker_count = lambda root=None, *, product=None: procs


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

    def test_refuse_at_cap_counts_issue_resolution_sidecars(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 9, "verdict": "X"}, procs=0)
        # Restore the real proc_worker_count, but make its two live-process witnesses
        # hermetic: no command-line workers, two live issue-resolution sidecars.
        mod._cmdline_worker_pids = lambda: set()
        mod.live_resolve_worker_pids = lambda runs_dir, **kw: {101, 102}
        # Exercise the REAL union/scoping logic: an unscoped count is cmdline ∪
        # sidecars; a product-scoped count is the sidecars for that product. Here
        # the patched witnesses ignore product, so both views see {101, 102}.
        def _count(root=None, *, product=None):
            if product is not None:
                return len(mod.live_resolve_worker_pids((root or ROOT) / mod.RUNS_DIRNAME,
                                                        product=product))
            return len(mod._cmdline_worker_pids() | mod.live_resolve_worker_pids(
                (root or ROOT) / mod.RUNS_DIRNAME))
        mod.proc_worker_count = _count
        p = run_eval(mod, max_workers=2)
        self.assertEqual(p["live"], 2)
        self.assertEqual(p["os_worker_procs"], 2)
        self.assertEqual(p["verdict"], mod.REFUSE_AT_CAP)

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

    def test_cap_falls_back_to_max_workers_when_target_is_zero(self) -> None:
        # Regression for the #517 wedge: `dos [supervise].target` is 0 in this repo
        # (the emit-only `dos loop` keeps no standing loop alive), but the cron-armed
        # issue-dispatch self-spawner must still spawn up to its own --max-workers. A
        # zero target must NOT pin the cap to 0 — that silently froze the live
        # issue-closer for ~12h. (A positive target below max_workers still throttles;
        # see test_cap_is_min_of_max_workers_and_dos_target.)
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "AT_TARGET"}, procs=0)
        p = run_eval(mod, max_workers=3)
        self.assertEqual(p["cap"], 3)
        self.assertEqual(p["verdict"], mod.OK_VERDICT)

    def test_zero_target_does_not_count_live_dos_lanes_as_issue_workers(self) -> None:
        # With target=0, `dos loop` reports ordinary live lanes (for example tools,
        # docs, experiments) even though no DOS standing-loop worker population is
        # armed. Those lanes must not consume the issue-dispatcher's process cap.
        mod = load()
        patch_checks(mod, kernel={"alive": 3, "target": 0, "verdict": "OVER_TARGET"}, procs=0)
        p = run_eval(mod, max_workers=2)
        self.assertEqual(p["cap"], 2)
        self.assertEqual(p["live"], 0)
        self.assertEqual(p["verdict"], mod.OK_VERDICT)


class RenderTest(unittest.TestCase):
    def test_render_does_not_raise_on_evaluate_output(self) -> None:
        mod = load()
        patch_checks(mod)
        text = mod.render(run_eval(mod))
        self.assertIn("dispatch preflight", text)
        self.assertIn("SPAWN_OK", text)


class WorkerCountTest(unittest.TestCase):
    def test_is_worker_cmdline_matches_generic_and_issue_resolver(self) -> None:
        mod = load()
        self.assertTrue(mod._is_worker_cmdline("claude -p /dos-kernel:dos-dispatch-loop --lane docs"))
        self.assertTrue(mod._is_worker_cmdline("claude -p your goal: resolve GitHub issue #717"))
        self.assertFalse(mod._is_worker_cmdline("python tools/dispatch_preflight.py --json"))

    def test_collapse_descendant_worker_pids_counts_wrapper_tree_once(self) -> None:
        mod = load()
        # The live opencode shape is a .cmd wrapper whose backend child keeps the same
        # prompt marker in its argv. Both match, but they are one worker tree.
        pids = {8436, 30912, 40388}
        parents = {8436: 47720, 30912: 8436, 40388: 45116}
        self.assertEqual(mod._collapse_descendant_pids(pids, parents), {8436, 40388})

    def test_live_resolve_worker_pids_counts_only_alive_sidecars(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            one = runs / "resolve-717-20260625-062210.pid"
            two = runs / "resolve-718-20260625-060712.pid"
            bad = runs / "resolve-719-20260625-055209.pid"
            one.write_text("101", encoding="utf-8")
            two.write_text("102", encoding="utf-8")
            bad.write_text("not-a-pid", encoding="utf-8")
            os.utime(one, (now, now))
            os.utime(two, (now, now))
            # In-window survivor must also LOOK like a worker backend image; a
            # cmdline-less probe with a claude image is the real "OS hid the
            # cmdline of a live claude worker" case the window fallback exists for.
            probe = lambda pid: {"alive": True, "create_time": now - 1,
                                 "name": "claude.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolve_worker_pids(runs, alive={101}, probe=probe), {101})

    def test_live_resolve_worker_pids_rejects_recycled_shell_in_window(self) -> None:
        # The ghost that pinned the dispatcher at cap: a recycled cmd.exe whose
        # create time happens to fall inside a stale sidecar's spawn window, with
        # no cmdline marker. A bare shell image is NOT a worker even in-window, so
        # it must NOT consume a cap slot.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            side = runs / "resolve-825-20260625-213720.pid"
            side.write_text("58752", encoding="utf-8")
            os.utime(side, (now, now))
            probe = lambda pid: {
                "alive": True,
                "create_time": now - 30,  # well inside the 5-min window
                "name": "cmd.exe",
                "cmdline": "",
            }
            self.assertEqual(mod.live_resolve_worker_pids(runs, probe=probe), set())

    def test_live_resolve_worker_pids_rejects_reused_pid_after_sidecar(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            side = runs / "resolve-717-20260625-062210.pid"
            side.write_text("20032", encoding="utf-8")
            os.utime(side, (now, now))
            probe = lambda pid: {
                "alive": True,
                "create_time": now + 60 * 60,
                "name": "conhost.exe",
                "cmdline": "",
            }
            self.assertEqual(mod.live_resolve_worker_pids(runs, probe=probe), set())

    def test_live_resolve_worker_pids_rejects_unrelated_old_process(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            side = runs / "resolve-717-20260625-062210.pid"
            side.write_text("29520", encoding="utf-8")
            os.utime(side, (now, now))
            probe = lambda pid: {
                "alive": True,
                "create_time": now - 60 * 60,
                "name": "chrome.exe",
                "cmdline": "chrome.exe --type=renderer",
            }
            self.assertEqual(mod.live_resolve_worker_pids(runs, probe=probe), set())

    def test_live_resolve_worker_pids_counts_worker_marker(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            side = runs / "resolve-717-20260625-062210.pid"
            side.write_text("31337", encoding="utf-8")
            os.utime(side, (now, now))
            probe = lambda pid: {
                "alive": True,
                "create_time": now - 60 * 60,
                "cmdline": "claude -p resolve GitHub issue #717",
            }
            self.assertEqual(mod.live_resolve_worker_pids(runs, probe=probe), {31337})

    def test_proc_worker_count_unions_cmdline_and_sidecar_pids(self) -> None:
        mod = load()
        mod._cmdline_worker_pids = lambda: {101, 103}
        mod.live_resolve_worker_pids = lambda runs_dir, **kw: {101, 102}
        self.assertEqual(mod.proc_worker_count(ROOT), 3)

    def test_proc_worker_count_scopes_to_product_pool(self) -> None:
        # A product-scoped count is the issue-resolver sidecars for that product
        # ALONE — the generic cmdline-marked DOS-loop workers (no backend tag) do
        # not pin a specific pool's cap, so the two account pools fill independently.
        mod = load()
        mod._cmdline_worker_pids = lambda: {999}
        seen = {}

        def fake_pids(runs_dir, **kw):
            seen["product"] = kw.get("product")
            return {201, 202} if kw.get("product") == "opencode" else {201, 202, 203}
        mod.live_resolve_worker_pids = fake_pids
        # Unscoped: cmdline ∪ all sidecars = {999, 201, 202, 203} = 4
        self.assertEqual(mod.proc_worker_count(ROOT), 4)
        # Product-scoped: only that pool's sidecars, no cmdline union
        self.assertEqual(mod.proc_worker_count(ROOT, product="opencode"), 2)
        self.assertEqual(seen["product"], "opencode")

    def test_account_check_codex_uses_ambient_login(self) -> None:
        # Codex has no switcher roster — its availability is read from ~/.codex.
        import tempfile, os as _os
        mod = load()
        with tempfile.TemporaryDirectory() as home:
            old = _os.environ.get("USERPROFILE"), _os.environ.get("HOME")
            try:
                _os.environ["USERPROFILE"] = home
                _os.environ["HOME"] = home
                # No auth.json yet -> not available.
                out = mod.account_check(ROOT, work_kind="engineering", product="codex")
                self.assertFalse(out["available"])
                # Create the login -> available, ambient account, switcher NOT consulted.
                codex = Path(home) / ".codex"
                codex.mkdir(parents=True, exist_ok=True)
                (codex / "auth.json").write_text("{}", encoding="utf-8")
                out = mod.account_check(ROOT, work_kind="engineering", product="codex")
                self.assertTrue(out["available"])
                self.assertEqual(out["tag"], "codex-ambient")
            finally:
                for k, v in zip(("USERPROFILE", "HOME"), old):
                    if v is None:
                        _os.environ.pop(k, None)
                    else:
                        _os.environ[k] = v

    def test_live_resolve_worker_pids_filters_by_backend_sidecar(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            cl = runs / "resolve-700-20260625-100000.pid"
            oc = runs / "resolve-701-20260625-100100.pid"
            cl.write_text("701", encoding="utf-8")
            oc.write_text("702", encoding="utf-8")
            cl.with_suffix(".backend").write_text("claude", encoding="utf-8")
            oc.with_suffix(".backend").write_text("opencode", encoding="utf-8")
            for f in (cl, oc):
                os.utime(f, (now, now))
            probe = lambda pid: {"alive": True, "create_time": now - 1,
                                 "name": "claude.exe", "cmdline": ""}
            self.assertEqual(
                mod.live_resolve_worker_pids(runs, product="claude", probe=probe), {701})
            self.assertEqual(
                mod.live_resolve_worker_pids(runs, product="opencode", probe=probe), {702})
            self.assertEqual(
                mod.live_resolve_worker_pids(runs, probe=probe), {701, 702})


if __name__ == "__main__":
    unittest.main()
