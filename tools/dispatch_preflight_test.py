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


def load_fleet_accounts():
    """Import tools/fleet_accounts.py the same hermetic way — the explicit seat pool
    (#1336: ``seat_pool`` / ``live_seat_leases``) lives there, so the SeatPool and
    LiveSeatLeases tests load and exercise it directly."""
    fa = ROOT / "tools" / "fleet_accounts.py"
    sys.path.insert(0, str(fa.parent))
    spec = importlib.util.spec_from_file_location("fleet_accounts", fa)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_checks(mod, *, host=None, account=None, kernel=None, procs=0, host_res=None,
                 seat=None):
    """Replace the shelling-out checks with constant synthetic results.

    ``host_res`` stubs the host-resource probe (#1337); the default is a roomy box
    (64 cores, 128 GB free, 1k threads) whose derived host_cap (32) sits well above
    every small cap the verdict tests assert, so it never perturbs them — a test
    that wants host_cap to BIND passes a scarce host_res of its own.

    ``seat`` stubs the explicit seat-pool view (#1336); the default ``{"total": None}``
    means "no seat view" so the seat fold is SKIPPED and the cap is governed by the
    static/host caps alone — exactly the pre-seat behavior the other verdict tests
    assert. A test exercising the seat pool passes an explicit ``{total, free, leased,
    depleted}`` of its own. Stubbing this keeps evaluate() hermetic: without it the
    seat fold would shell out to fleet_accounts.py and leak the real box's seat count
    into every test."""
    host = host if host is not None else {"safe": True, "flagged": 0, "flagged_names": []}
    account = account if account is not None else {
        "available": True, "tag": "worker-a", "dir": "/acct/a", "tier": 1,
        "model": "claude", "reason": "free", "blocked": []}
    kernel = kernel if kernel is not None else {"alive": 0, "target": 3, "verdict": "FILLING"}
    host_res = host_res if host_res is not None else {
        "cores": 64, "free_ram_mb": 128_000, "total_threads": 1000}
    seat = seat if seat is not None else {"total": None}
    mod.host_check = lambda root, **kw: host
    mod.account_check = lambda root, **kw: account
    mod.kernel_alive = lambda root: kernel
    mod.proc_worker_count = lambda root=None, *, product=None: procs
    mod.host_resources = lambda: host_res
    mod.seat_check = lambda root, *, product=None: seat


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


class HostCapacityPureTest(unittest.TestCase):
    """The pure host-derived cap (#1337): cores, free RAM, and live OS-thread total
    turned into the largest sustainable worker population. No I/O."""

    def test_roomy_box_is_bound_by_cores(self) -> None:
        mod = load()
        info = mod.host_capacity(cores=64, free_ram_mb=128_000, total_threads=1000)
        # cores 64//2=32, ram 128000//1500=85, threads (64*400-1000)//200=123 -> min 32.
        self.assertEqual(info["host_cap"], 32)
        self.assertEqual(info["binding"], "cores")

    def test_thread_saturation_drops_cap_to_the_floor(self) -> None:
        # The exact failure mode this subsystem exists for: the box's live thread
        # total has blown past its budget, so the host can sustain ~no new worker.
        mod = load()
        info = mod.host_capacity(cores=8, free_ram_mb=64_000, total_threads=200_000)
        self.assertEqual(info["host_cap"], 1)        # floored, not 0
        self.assertEqual(info["binding"], "threads")
        self.assertEqual(info["components"]["threads"], 0)

    def test_low_free_ram_binds_the_cap(self) -> None:
        mod = load()
        info = mod.host_capacity(cores=32, free_ram_mb=3000, total_threads=2000)
        self.assertEqual(info["host_cap"], 2)        # 3000//1500
        self.assertEqual(info["binding"], "ram")

    def test_all_dimensions_unknown_yields_no_bound(self) -> None:
        mod = load()
        info = mod.host_capacity(cores=None, free_ram_mb=None, total_threads=None)
        self.assertIsNone(info["host_cap"])
        self.assertEqual(info["components"], {})

    def test_missing_ram_dimension_is_skipped_not_a_breach(self) -> None:
        # macOS-style host where free RAM could not be read: cores+threads still bound.
        mod = load()
        info = mod.host_capacity(cores=8, free_ram_mb=None, total_threads=500)
        self.assertEqual(info["host_cap"], 4)        # cores 8//2, threads big
        self.assertNotIn("ram", info["components"])

    def test_thread_dimension_needs_cores_for_its_budget(self) -> None:
        mod = load()
        info = mod.host_capacity(cores=None, free_ram_mb=6000, total_threads=100)
        self.assertEqual(info["host_cap"], 4)        # ram alone (6000//1500)
        self.assertEqual(list(info["components"].keys()), ["ram"])


class HostCapFoldTest(unittest.TestCase):
    """host_cap folds into the cap via min, the adaptive throttle (#1337)."""

    def test_host_cap_binds_below_the_static_cap(self) -> None:
        mod = load()
        # cores 2//2=1, ram 1000//1500=0 -> host_cap floored to 1.
        patch_checks(mod, kernel={"alive": 0, "target": 3, "verdict": "X"}, procs=0,
                     host_res={"cores": 2, "free_ram_mb": 1000, "total_threads": 100})
        p = run_eval(mod, max_workers=5)
        self.assertEqual(p["host_cap"], 1)
        self.assertEqual(p["cap"], 1)               # min(min(5,3)=3, host_cap=1)
        self.assertEqual(p["verdict"], mod.OK_VERDICT)

    def test_host_cap_throttles_even_when_dos_target_is_zero(self) -> None:
        # The adaptive promise: target=0 (emit-only loop) no longer means "fill to
        # --max-workers" — the live host headroom still throttles the spawn count.
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "AT_TARGET"}, procs=0,
                     host_res={"cores": 4, "free_ram_mb": 3000, "total_threads": 1000})
        p = run_eval(mod, max_workers=5)
        self.assertEqual(p["host_cap"], 2)          # cores 2, ram 2 -> min 2
        self.assertEqual(p["cap"], 2)               # host_cap throttles 5 -> 2

    def test_host_cap_above_static_cap_does_not_raise_it(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 3, "verdict": "X"}, procs=0)
        p = run_eval(mod, max_workers=2)            # roomy default host_res -> host_cap 32
        self.assertEqual(p["host_cap"], 32)
        self.assertEqual(p["cap"], 2)               # min(2, 3, 32) = the static cap

    def test_unreadable_host_probe_leaves_static_cap_intact(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 3, "verdict": "X"}, procs=0,
                     host_res={"cores": None, "free_ram_mb": None, "total_threads": None})
        p = run_eval(mod, max_workers=5)
        self.assertIsNone(p["host_cap"])
        self.assertEqual(p["cap"], 3)               # min(5, target 3); no host bound

    def test_loaded_box_spawns_fewer_than_a_roomy_box(self) -> None:
        # The done-condition behavior in one assertion: same request (max_workers=10,
        # target unset), but a loaded box derives a smaller cap than a roomy one.
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0,
                     host_res={"cores": 64, "free_ram_mb": 128_000, "total_threads": 1000})
        roomy = run_eval(mod, max_workers=10)
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0,
                     host_res={"cores": 8, "free_ram_mb": 64_000, "total_threads": 200_000})
        loaded = run_eval(mod, max_workers=10)
        self.assertEqual(roomy["cap"], 10)          # roomy: host_cap(32) does not bind
        self.assertEqual(loaded["cap"], 1)          # loaded: host_cap(1) throttles hard
        self.assertLess(loaded["cap"], roomy["cap"])


class RenderTest(unittest.TestCase):
    def test_render_does_not_raise_on_evaluate_output(self) -> None:
        mod = load()
        patch_checks(mod)
        text = mod.render(run_eval(mod))
        self.assertIn("dispatch preflight", text)
        self.assertIn("SPAWN_OK", text)

    def test_render_shows_host_cap(self) -> None:
        mod = load()
        patch_checks(mod)
        text = mod.render(run_eval(mod))
        self.assertIn("host_cap=32", text)
        self.assertIn("bound by cores", text)


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
            def probe(pid):
                return {"alive": True, "create_time": now - 1,
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
            def probe(pid):
                return {
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
            def probe(pid):
                return {
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
            def probe(pid):
                return {
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
            def probe(pid):
                return {
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
        import tempfile
        import os as _os
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
            def probe(pid):
                return {"alive": True, "create_time": now - 1,
                                             "name": "claude.exe", "cmdline": ""}
            self.assertEqual(
                mod.live_resolve_worker_pids(runs, product="claude", probe=probe), {701})
            self.assertEqual(
                mod.live_resolve_worker_pids(runs, product="opencode", probe=probe), {702})
            self.assertEqual(
                mod.live_resolve_worker_pids(runs, probe=probe), {701, 702})


def _seat_row(tag, *, uuid="", available=True, product="claude", role="", dir_=None):
    """A synthetic roster row in the shape ``seat_pool`` consumes: a routable worker
    (``kind == worker`` and not a duplicate identity) carrying the pool-key inputs
    (``account_uuid`` / ``account`` / ``dir``) and the display fields."""
    return {
        "kind": "worker", "identity_role": role, "tag": tag,
        "account": (f".claude-{tag}-acct" if product == "claude" else f"opencode-{tag}"),
        "dir": dir_ or f"/home/u/{tag}", "available": available,
        "account_uuid": uuid, "product": product, "model": "m", "model_tier": 1,
    }


class SeatPoolTest(unittest.TestCase):
    """The explicit multi-seat account pool (#1336): seat -> live-worker binding,
    depletion, double-booking, and the no-double-hand (one seat per rate-limit pool)."""

    def test_free_pool_has_full_headroom_and_is_not_depleted(self) -> None:
        fa = load_fleet_accounts()
        pool = fa.seat_pool([_seat_row("a"), _seat_row("b"), _seat_row("c")], [])
        self.assertEqual(pool["total_seats"], 3)
        self.assertEqual(pool["free_seats"], 3)
        self.assertEqual(pool["leased_seats"], 0)
        self.assertFalse(pool["depleted"])
        self.assertEqual(pool["double_booked"], [])

    def test_a_live_lease_binds_its_seat(self) -> None:
        fa = load_fleet_accounts()
        leases = [{"worker": "resolve-101", "pid": 101, "tag": "a", "dir": "/home/u/a"}]
        pool = fa.seat_pool([_seat_row("a"), _seat_row("b")], leases)
        self.assertEqual(pool["leased_seats"], 1)
        self.assertEqual(pool["free_seats"], 1)
        leased = [s for s in pool["seats"] if s["state"] == "leased"]
        self.assertEqual(len(leased), 1)
        self.assertEqual(leased[0]["tag"], "a")
        self.assertEqual(leased[0]["workers"], ["resolve-101"])

    def test_pool_depleted_when_every_seat_leased(self) -> None:
        fa = load_fleet_accounts()
        leases = [{"worker": "w1", "tag": "a", "dir": "/home/u/a"},
                  {"worker": "w2", "tag": "b", "dir": "/home/u/b"}]
        pool = fa.seat_pool([_seat_row("a"), _seat_row("b")], leases)
        self.assertEqual(pool["leased_seats"], 2)
        self.assertEqual(pool["free_seats"], 0)
        self.assertTrue(pool["depleted"])

    def test_double_booking_one_seat_two_live_workers_is_surfaced(self) -> None:
        # The invariant the issue forbids — two live workers on ONE seat — must be
        # OBSERVABLE, not silently assumed away.
        fa = load_fleet_accounts()
        leases = [{"worker": "w1", "tag": "a", "dir": "/home/u/a"},
                  {"worker": "w2", "tag": "a", "dir": "/home/u/a"}]
        pool = fa.seat_pool([_seat_row("a")], leases)
        self.assertEqual(len(pool["double_booked"]), 1)
        self.assertEqual(sorted(pool["double_booked"][0]["workers"]), ["w1", "w2"])

    def test_lease_on_account_not_in_pool_is_unbound(self) -> None:
        fa = load_fleet_accounts()
        leases = [{"worker": "ghost", "tag": "gone", "dir": "/home/u/gone"}]
        pool = fa.seat_pool([_seat_row("a")], leases)
        self.assertEqual(pool["leased_seats"], 0)
        self.assertEqual(len(pool["unbound_leases"]), 1)
        self.assertEqual(pool["unbound_leases"][0]["worker"], "ghost")

    def test_two_dirs_on_one_account_are_one_seat(self) -> None:
        # The no-double-hand core: two dirs sharing one Anthropic account (a duplicate
        # identity) collapse to ONE seat, so the pool never double-counts a rate limit.
        fa = load_fleet_accounts()
        rows = [_seat_row("canon", uuid="U1"),
                _seat_row("copy", uuid="U1", role="duplicate")]
        pool = fa.seat_pool(rows, [])
        self.assertEqual(pool["total_seats"], 1)
        self.assertEqual(pool["seats"][0]["tag"], "canon")

    def test_product_scope_filters_the_pool(self) -> None:
        fa = load_fleet_accounts()
        rows = [_seat_row("a", product="claude"), _seat_row("g", product="opencode")]
        self.assertEqual(fa.seat_pool(rows, [], product="claude")["total_seats"], 1)
        self.assertEqual(fa.seat_pool(rows, [], product="opencode")["total_seats"], 1)


class LiveSeatLeasesTest(unittest.TestCase):
    """The seat -> worker binding is derived from LIVE worker pids, so an exited worker
    frees its seat on the next read with no separate release step (#1336)."""

    def _live_probe(self, now):
        return lambda pid: {"alive": True, "create_time": now - 1,
                            "name": "claude.exe", "cmdline": ""}

    def test_live_worker_account_sidecar_becomes_a_seat_lease(self) -> None:
        fa = load_fleet_accounts()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            pid_file = runs / "resolve-101-20260625-062210.pid"
            pid_file.write_text("101", encoding="utf-8")
            pid_file.with_suffix(".account").write_text(
                '{"tag": "worker-a", "dir": "/home/u/worker-a"}', encoding="utf-8")
            os.utime(pid_file, (now, now))
            leases = fa.live_seat_leases(str(runs), alive={101}, probe=self._live_probe(now))
            self.assertEqual(len(leases), 1)
            self.assertEqual(leases[0]["tag"], "worker-a")
            self.assertEqual(leases[0]["pid"], 101)

    def test_exited_worker_frees_its_seat(self) -> None:
        fa = load_fleet_accounts()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            pid_file = runs / "resolve-102-20260625-062210.pid"
            pid_file.write_text("102", encoding="utf-8")
            pid_file.with_suffix(".account").write_text(
                '{"tag": "worker-a", "dir": "/home/u/worker-a"}', encoding="utf-8")
            os.utime(pid_file, (now, now))
            # Worker 102 is no longer alive -> its sidecar yields NO lease, so the seat
            # it held is free again on this very read.
            leases = fa.live_seat_leases(str(runs), alive=set(), probe=self._live_probe(now))
            self.assertEqual(leases, [])
            pool = fa.seat_pool([_seat_row("worker-a")], leases)
            self.assertEqual(pool["free_seats"], 1)
            self.assertFalse(pool["depleted"])


class SeatRefusalTest(unittest.TestCase):
    """Preflight folds the seat pool into the cap and emits the typed REFUSE_NO_SEAT
    on depletion (#1336): N>M wave -> exactly M, remainder structurally refused."""

    def test_seat_count_is_the_effective_cap(self) -> None:
        # An N>M ask (max_workers=100) with a free pool of M=4 caps at 4, not 100 —
        # the effective concurrency cap becomes the seat count.
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0,
                     seat={"total": 4, "free": 4, "leased": 0, "depleted": False})
        p = run_eval(mod, max_workers=100)
        self.assertEqual(p["cap"], 4)
        self.assertEqual(p["verdict"], mod.OK_VERDICT)

    def test_depleted_seat_pool_refuses_the_remainder_with_no_seat(self) -> None:
        # M=4 seats, all leased to 4 live workers: the 5th preflight in an N>M wave
        # gets the typed REFUSE_NO_SEAT, never a silent double-book.
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=4,
                     seat={"total": 4, "free": 0, "leased": 4, "depleted": True})
        p = run_eval(mod, max_workers=100)
        self.assertEqual(p["cap"], 4)
        self.assertEqual(p["verdict"], mod.REFUSE_NO_SEAT)
        self.assertFalse(p["ok"])

    def test_free_seat_admits(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=1,
                     seat={"total": 3, "free": 2, "leased": 1, "depleted": False})
        p = run_eval(mod, max_workers=10)
        self.assertEqual(p["verdict"], mod.OK_VERDICT)

    def test_all_blocked_pool_is_no_account_not_no_seat(self) -> None:
        # A pool with no free seat because every seat is THROTTLED (none leased) is a
        # REFUSE_NO_ACCOUNT, not a REFUSE_NO_SEAT — depletion-by-lease and
        # depletion-by-block are distinct typed refusals.
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0,
                     account={"available": False, "tag": None, "dir": None, "tier": None,
                              "model": None, "reason": "throttled", "blocked": ["a", "b"]},
                     seat={"total": 2, "free": 0, "leased": 0, "depleted": True})
        p = run_eval(mod, max_workers=10)
        self.assertEqual(p["verdict"], mod.REFUSE_NO_ACCOUNT)

    def test_missing_seat_view_skips_the_fold(self) -> None:
        # total=None (the seat view could not run, or codex's ambient login) -> the
        # fold is SKIPPED and the static/host caps govern; never a fail-closed refusal.
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 3, "verdict": "X"}, procs=0,
                     seat={"total": None})
        p = run_eval(mod, max_workers=2)
        self.assertEqual(p["cap"], 2)
        self.assertEqual(p["verdict"], mod.OK_VERDICT)
        self.assertIn("seat", p)


class DoubledDefaultCeilingTest(unittest.TestCase):
    """The 2x-concurrency change (DEFAULT_MAX_WORKERS 2->4) is safe iff the doubled
    static ceiling is still strictly bounded by the adaptive gates. These tests pin
    the new default AND prove the doubling can never exceed host_cap or the seat pool
    — i.e. raising the operator ceiling only realizes concurrency the box and the
    account roster can actually carry."""

    def test_default_ceiling_is_doubled(self) -> None:
        # The ceiling itself: pin 4 so a later silent revert is caught.
        mod = load()
        self.assertEqual(mod.DEFAULT_MAX_WORKERS, 4)

    def test_default_ceiling_realizes_2x_on_a_roomy_box_with_seats(self) -> None:
        # The win: a roomy box with >=4 free seats and no dos throttle lets the default
        # ceiling fill to 4 — double the old static 2 — governed only by the gates.
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "AT_TARGET"}, procs=0,
                     host_res={"cores": 64, "free_ram_mb": 128_000, "total_threads": 1000},
                     seat={"total": 4, "free": 4, "leased": 0, "depleted": False})
        p = run_eval(mod, max_workers=mod.DEFAULT_MAX_WORKERS)
        self.assertEqual(p["cap"], 4)               # min(4, host_cap=32, seats=4)
        self.assertEqual(p["verdict"], mod.OK_VERDICT)

    def test_default_ceiling_still_throttles_on_a_loaded_box(self) -> None:
        # Safety: doubling the ceiling cannot saturate a loaded host — host_cap pulls
        # the effective cap back below 4 (here to the floor), exactly as before.
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0,
                     host_res={"cores": 8, "free_ram_mb": 64_000, "total_threads": 200_000},
                     seat={"total": 8, "free": 8, "leased": 0, "depleted": False})
        p = run_eval(mod, max_workers=mod.DEFAULT_MAX_WORKERS)
        self.assertEqual(p["host_cap"], 1)
        self.assertEqual(p["cap"], 1)               # doubled ceiling, still throttled
        self.assertLess(p["cap"], mod.DEFAULT_MAX_WORKERS)

    def test_default_ceiling_still_bounded_by_a_smaller_seat_pool(self) -> None:
        # Safety: doubling the ceiling cannot double-book accounts — a 3-seat roster
        # caps the doubled ceiling at 3, not 4 (the live agent-host case).
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0,
                     host_res={"cores": 64, "free_ram_mb": 128_000, "total_threads": 1000},
                     seat={"total": 3, "free": 3, "leased": 0, "depleted": False})
        p = run_eval(mod, max_workers=mod.DEFAULT_MAX_WORKERS)
        self.assertEqual(p["cap"], 3)               # min(4, host_cap=32, seats=3)
        self.assertEqual(p["verdict"], mod.OK_VERDICT)


if __name__ == "__main__":
    unittest.main()
