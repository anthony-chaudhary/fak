#!/usr/bin/env python3
"""Hermetic tests for tools/dispatch_preflight.py.

The spawn gate composes its independent host, account, kernel, process-census,
and system-commit checks. The live checks use OS/tool probes, so here
we replace them on the module with synthetic results and assert the verdict
logic â€” SPAWN_OK and every typed REFUSE_* â€” plus the pure helpers (_last_json,
_int) and the cap = min(max_workers, dos target) rule. No subprocess runs.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dispatch_preflight.py"

# #5879: the fleet-tuning knobs both modules under test read from the ambient
# environment. On a live fleet host these ARE exported (FAK_SESSIONS_PER_ACCOUNT=6,
# FAK_HOST_CORES_PER_WORKER=1, ...), and every one of them feeds a number this module
# asserts on — so an unpinned run reads the OPERATOR's box, not the code, and reports a
# confident false red. Both loaders below clear the whole family and re-apply only what
# a test asks for explicitly, so the assertions are about the defaults in the source.
# KnobDriftTest pins this list against the tools themselves so a knob added later
# cannot quietly re-open the leak.
AMBIENT_ENV_KNOBS = (
    "FAK_MAX_WORKERS",
    "FAK_CODEX_OAUTH_SESSIONS",
    "FAK_HOST_CORES_PER_WORKER",
    "FAK_HOST_RAM_MB_PER_WORKER",
    "FAK_HOST_THREADS_PER_CORE",
    "FAK_HOST_THREADS_PER_WORKER",
    "FAK_SESSIONS_PER_ACCOUNT",
    "FAK_SYSTEM_COMMIT_HEADROOM_MB",
)

# Set by the witness's own re-run of this module (AmbientKnobHermeticityTest) so the
# child process does not recurse into spawning another one.
CHILD_RUN_ENV = "FAK_DISPATCH_PREFLIGHT_TEST_CHILD"


def no_window_creationflags() -> int:
    """CREATE_NO_WINDOW on Windows, ``0`` on POSIX — mirrors
    ``dispatch_preflight._no_window_creationflags`` so this test module's children
    cannot pop a console window when the suite runs windowless."""
    return 0x08000000 if os.name == "nt" else 0


def _pinned_env(overrides: dict) -> dict:
    """The real environment with every ambient knob cleared, then ``overrides`` applied.

    ``None`` in ``overrides`` means "stay cleared", which lets a test say "load with
    FAK_MAX_WORKERS unset" without caring what the host exports."""
    for name in overrides:
        if name not in AMBIENT_ENV_KNOBS:
            # Explicit raise, not `assert`: `python -O` would strip the assert and let a
            # misspelled knob silently fall through to the host's ambient value.
            raise AssertionError("not a pinned knob: %s" % (name,))
    env = {k: v for k, v in os.environ.items() if k not in AMBIENT_ENV_KNOBS}
    env.update({k: v for k, v in overrides.items() if v is not None})
    return env


class _PinnedOS:
    """``os`` with a knob-masked ``environ``, for the module under test to import.

    Clearing the process environment around the import is enough for the constants
    folded at import time (``HOST_CORES_PER_WORKER`` and friends), but NOT for the
    knobs read LAZILY inside the call a test makes — ``fleet_accounts._claude_session_cap``
    re-reads FAK_SESSIONS_PER_ACCOUNT on every ``seat_pool`` row. Rebinding the loaded
    module's ``os`` keeps those reads pinned for the module's whole life without
    editing the tool. ``environ`` is rebuilt per access rather than snapshotted, so a
    test that mutates an UNpinned variable (HOME/USERPROFILE) is still seen; every
    other attribute (``os.path``, ``os.utime``, ...) delegates to the real module."""

    def __init__(self, overrides: dict) -> None:
        self._overrides = overrides

    @property
    def environ(self) -> dict:
        return _pinned_env(self._overrides)

    def __getattr__(self, name):
        return getattr(os, name)


def _load_pinned(name: str, path: Path, overrides: dict):
    """Import ``path`` as ``name`` with the ambient knobs pinned at import AND after."""
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, _pinned_env(overrides), clear=True):
        spec.loader.exec_module(mod)
    mod.os = _PinnedOS(overrides)
    return mod


def load(**knobs):
    """Import tools/dispatch_preflight.py hermetically.

    Keyword knobs are ambient env names (see AMBIENT_ENV_KNOBS): pass a string to load
    as if the host exported it, pass nothing to load against the built-in defaults."""
    return _load_pinned("dispatch_preflight", SCRIPT, knobs)


def load_fleet_accounts(**knobs):
    """Import tools/fleet_accounts.py the same hermetic way â€” the explicit seat pool
    (#1336: ``seat_pool`` / ``live_seat_leases``) lives there, so the SeatPool and
    LiveSeatLeases tests load and exercise it directly."""
    return _load_pinned("fleet_accounts", ROOT / "tools" / "fleet_accounts.py", knobs)


def patch_checks(mod, *, host=None, account=None, kernel=None, procs=0, host_res=None,
                 seat=None, weekly=None, fak_bin=None, commit_headroom=None):
    """Replace the shelling-out checks with constant synthetic results.

    ``host_res`` stubs the host-resource probe (#1337); the default is a roomy box
    (64 cores, 128 GB free, 1k threads) whose derived host_cap (32) sits well above
    every small cap the verdict tests assert, so it never perturbs them â€” a test
    that wants host_cap to BIND passes a scarce host_res of its own.

    ``seat`` stubs the explicit seat-pool view (#1336); the default ``{"total": None}``
    means "no seat view" so the seat fold is SKIPPED and the cap is governed by the
    static/host caps alone â€” exactly the pre-seat behavior the other verdict tests
    assert. A test exercising the seat pool passes an explicit ``{total, free, leased,
    depleted}`` of its own. Stubbing this keeps evaluate() hermetic: without it the
    seat fold would shell out to fleet_accounts.py and leak the real box's seat count
    into every test.

    ``weekly`` stubs the weekly-limit cooldown probe (#2610); the default
    ``{"capped": False}`` means "no cooldown active" so the pre-cooldown verdict logic
    is unchanged. Stubbing keeps evaluate() hermetic: without it the cooldown fold
    would read the real box's `.dispatch-runs/account-cap-*.json` holds into every
    test. A test exercising the cooldown passes an explicit capped hold of its own.

    ``fak_bin`` stubs the binary-provenance table. The default is ONE clean agreeing
    build, so the advisory adds nothing to any verdict test. Stubbing matters twice
    here: unstubbed, evaluate() would spawn a real ``fak version`` per resolver AND
    append a tick to the real repo's ``.dispatch-runs/fak-bin-provenance.json``. The
    recorder is stubbed alongside it for the same reason."""
    host = host if host is not None else {"safe": True, "flagged": 0, "flagged_names": []}
    account = account if account is not None else {
        "available": True, "tag": "worker-a", "dir": "/acct/a", "tier": 1,
        "model": "claude", "reason": "free", "blocked": []}
    kernel = kernel if kernel is not None else {"alive": 0, "target": 3, "verdict": "FILLING"}
    host_res = host_res if host_res is not None else {
        "cores": 64, "free_ram_mb": 128_000, "total_threads": 1000}
    seat = seat if seat is not None else {"total": None}
    weekly = weekly if weekly is not None else {"capped": False}
    mod.host_check = lambda root, **kw: host
    mod.account_check = lambda root, **kw: account
    mod.kernel_alive = lambda root: kernel
    mod.managed_worker_census = lambda root, *, product=None: {
        "pids": list(range(procs)), "count": procs,
        "status": "CONSISTENT", "ambiguous": []}
    mod.host_resources = lambda: host_res
    mod.seat_check = lambda root, *, product=None: seat
    mod.weekly_cap_check = lambda root, **kw: weekly
    commit_headroom = commit_headroom if commit_headroom is not None else {
        "supported": True, "ok": True, "reason": "", "observed_bytes": 100,
        "required_bytes": 10, "system_commit_bytes": 900,
        "system_commit_limit": 1000}
    mod.system_commit_headroom_check = lambda: commit_headroom
    fak_bin = fak_bin if fak_bin is not None else {
        "schema": mod.FAK_BIN_PROVENANCE_SCHEMA,
        "resolvers": {"preflight_gate": {"path": "/x/fak", "resolved": True,
                                         "build": "deadbeef", "dirty": False,
                                         "build_key": "1-2-fak"}},
        "distinct_builds": 1, "builds": ["deadbeef"], "agree": True,
        "dirty": [], "unresolved": []}
    mod.fak_bin_provenance = lambda root, env=None, **kw: fak_bin
    mod.record_fak_bin_provenance = lambda root, prov, **kw: None


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


class SystemCommitHeadroomTest(unittest.TestCase):
    def test_requirement_parser_matches_guard_fail_closed_contract(self) -> None:
        for raw in ("", "0", "-1", "+1", "1GB", "17592186044416"):
            mod = load(FAK_SYSTEM_COMMIT_HEADROOM_MB=raw)
            self.assertEqual(mod.required_system_commit_headroom_bytes(),
                             mod.DEFAULT_SYSTEM_COMMIT_HEADROOM_BYTES, raw)
        mod = load(FAK_SYSTEM_COMMIT_HEADROOM_MB="456")
        self.assertEqual(mod.required_system_commit_headroom_bytes(), 456 << 20)

    def test_pure_low_exact_high_boundary(self) -> None:
        mod = load()
        for used, want_refuse in ((91, True), (90, True), (89, False)):
            got = mod.evaluate_system_commit_headroom(
                system_bytes=used, system_limit=100, required_bytes=10)
            self.assertEqual(not got["ok"], want_refuse, got)
            self.assertEqual(got["observed_bytes"], 100 - used)

    def test_low_headroom_refuses_before_spawn_verdict(self) -> None:
        mod = load()
        patch_checks(mod, commit_headroom={
            "supported": True, "ok": False, "reason": "SYSTEM_COMMIT_HEADROOM",
            "observed_bytes": 9, "required_bytes": 10,
            "system_commit_bytes": 91, "system_commit_limit": 100,
            "physical_available_bytes": 12345678, "physical_total_bytes": 87654321})
        payload = run_eval(mod)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["verdict"], mod.REFUSE_SYSTEM_COMMIT_HEADROOM)
        self.assertEqual(payload["system_commit_headroom"]["observed_bytes"], 9)
        self.assertIn("fak recover SYSTEM_COMMIT_HEADROOM", payload["reason"])
        self.assertIn("physical RAM available: 12345678 bytes", payload["reason"])

    def test_high_headroom_follows_normal_spawn_path(self) -> None:
        mod = load()
        patch_checks(mod, commit_headroom={
            "supported": True, "ok": True, "reason": "",
            "observed_bytes": 11, "required_bytes": 10,
            "system_commit_bytes": 89, "system_commit_limit": 100,
            "physical_available_bytes": 50000000, "physical_total_bytes": 87654321})
        payload = run_eval(mod)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["verdict"], mod.OK_VERDICT)

    def test_evaluate_system_commit_headroom_preserves_physical_ram(self) -> None:
        mod = load()
        got = mod.evaluate_system_commit_headroom(
            system_bytes=50, system_limit=100, required_bytes=10,
            physical_available_bytes=1234, physical_total_bytes=5678)
        self.assertEqual(got["physical_available_bytes"], 1234)
        self.assertEqual(got["physical_total_bytes"], 5678)

    def test_windows_system_commit_snapshot_outputs_physical_available_bytes(self) -> None:
        mod = load()
        if os.name != "nt":
            self.skipTest("Windows-only snapshot counters")
        snap = mod._windows_system_commit_snapshot()
        self.assertIsInstance(snap, dict)
        self.assertIn("physical_available_bytes", snap)
        self.assertIn("physical_total_bytes", snap)
        self.assertIn("system_commit_bytes", snap)
        self.assertIn("system_commit_limit", snap)
        self.assertGreater(snap["physical_available_bytes"], 0)
        self.assertGreater(snap["physical_total_bytes"], 0)
        self.assertLessEqual(snap["physical_available_bytes"], snap["physical_total_bytes"])

    def test_windows_system_commit_snapshot_mock(self) -> None:
        mod = load()

        class FakeFunc:
            argtypes = None
            restype = None

            def __call__(self, byref_info, cb):
                info = byref_info._obj
                info.commit_total = 100
                info.commit_limit = 200
                info.physical_available = 300
                info.physical_total = 400
                info.page_size = 4096
                return 1

        class FakeWinDLL:
            def __init__(self, *args, **kwargs):
                self.GetPerformanceInfo = FakeFunc()

        with unittest.mock.patch.object(mod.ctypes, "WinDLL", FakeWinDLL, create=True):
            snap = mod._windows_system_commit_snapshot()
            self.assertEqual(snap["system_commit_bytes"], 100 * 4096)
            self.assertEqual(snap["system_commit_limit"], 200 * 4096)
            self.assertEqual(snap["physical_available_bytes"], 300 * 4096)
            self.assertEqual(snap["physical_total_bytes"], 400 * 4096)


class HostCheckProtectedTest(unittest.TestCase):
    """host_check must refuse only on ACTIONABLE (non-protected) flags (#2227).

    A protected process (e.g. the operator's terminal) breaching a threshold is
    report-only: the guard's ``ok`` stays true and its reaper refuses the kill,
    so the spawn gate hardening it into REFUSE_HOST wedged every dispatch behind
    an impossible recovery. These are hermetic: ``run_json`` is stubbed."""

    def _host(self, mod, guard_doc):
        mod.run_json = lambda cmd, root, timeout=90, ok_codes=None: guard_doc
        return mod.host_check(ROOT)

    def test_protected_breach_is_advisory_not_a_refusal(self) -> None:
        mod = load()
        host = self._host(mod, {
            "schema": "fleet-proc-resource-guard/1", "ok": True,
            "flagged": [{"pid": 1, "name": "WindowsTerminal",
                         "reasons": ["threads 2768 > 2000"],
                         "protected": True, "action": "report"}]})
        self.assertTrue(host["safe"])
        self.assertEqual(host["flagged"], 0)
        self.assertEqual(host["flagged_names"], [])
        self.assertEqual(host["protected_flagged"], 1)
        self.assertEqual(host["protected_names"], ["WindowsTerminal"])

    def test_actionable_runaway_still_refuses(self) -> None:
        mod = load()
        host = self._host(mod, {
            "schema": "fleet-proc-resource-guard/1", "ok": False,
            "flagged": [{"pid": 2, "name": "llama-cli",
                         "reasons": ["threads 129427 > 2000"],
                         "protected": False, "action": "report"}]})
        self.assertFalse(host["safe"])
        self.assertEqual(host["flagged"], 1)
        self.assertEqual(host["flagged_names"], ["llama-cli"])
        self.assertEqual(host["protected_flagged"], 0)

    def test_mixed_flags_refuse_and_name_only_the_actionable(self) -> None:
        mod = load()
        host = self._host(mod, {
            "schema": "fleet-proc-resource-guard/1", "ok": False,
            "flagged": [
                {"pid": 1, "name": "WindowsTerminal", "protected": True},
                {"pid": 2, "name": "llama-cli", "protected": False}]})
        self.assertFalse(host["safe"])
        self.assertEqual(host["flagged"], 1)
        self.assertEqual(host["flagged_names"], ["llama-cli"])
        self.assertEqual(host["protected_names"], ["WindowsTerminal"])

    def test_protected_fleet_binary_still_refuses(self) -> None:
        # #2252: the advisory demotion is for FOREIGN protected processes only.
        # A fleet agent backend (here a claude worker that landed in the guard's
        # protected pid set) is fleet-spawned and fleet-reapable, so its flag
        # stays actionable â€” safe=False, never an advisory.
        mod = load()
        host = self._host(mod, {
            "schema": "fleet-proc-resource-guard/1", "ok": True,
            "flagged": [{"pid": 3, "name": "claude.exe",
                         "reasons": ["threads 5000 > 2000"],
                         "protected": True, "action": "protected-skip"}]})
        self.assertFalse(host["safe"])
        self.assertEqual(host["flagged"], 1)
        self.assertEqual(host["flagged_names"], ["claude.exe"])
        self.assertEqual(host["protected_flagged"], 0)
        self.assertEqual(host["protected_names"], [])


class HostAdvisoryStateTest(unittest.TestCase):
    """#2252: a PROTECTED-only breach must not freeze spawning, and the status
    card must show a ``host=ADVISORY(name)`` state distinct from ``FLAGGED`` so
    the operator can tell "foreign baseline noted" from "reap before growing"."""

    _advisory_host = {"safe": True, "flagged": 0, "flagged_names": [],
                      "protected_flagged": 1,
                      "protected_names": ["WindowsTerminal"]}

    def test_protected_only_advisory_host_still_spawns(self) -> None:
        # The liveness carve-out end to end: an advisory host is SPAWN_OK, with
        # the advisory rows carried in the payload for the status card/JSON.
        mod = load()
        patch_checks(mod, host=dict(self._advisory_host))
        p = run_eval(mod)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], mod.OK_VERDICT)
        self.assertEqual(p["host"]["protected_names"], ["WindowsTerminal"])

    def test_render_shows_advisory_distinct_from_flagged(self) -> None:
        mod = load()
        patch_checks(mod, host=dict(self._advisory_host))
        text = mod.render(run_eval(mod))
        self.assertIn("host=ADVISORY(WindowsTerminal)", text)
        self.assertNotIn("FLAGGED", text)

    def test_render_flagged_host_stays_flagged(self) -> None:
        mod = load()
        patch_checks(mod, host={"safe": False, "flagged": 1,
                                "flagged_names": ["llama-cli"]})
        text = mod.render(run_eval(mod))
        self.assertIn("host=FLAGGED", text)
        self.assertNotIn("ADVISORY", text)

    def test_render_clean_host_stays_clean(self) -> None:
        mod = load()
        patch_checks(mod)
        text = mod.render(run_eval(mod))
        self.assertIn("host=clean", text)
        self.assertNotIn("ADVISORY", text)


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
        # Restore the real managed census, but make its live-process witnesses
        # hermetic: no command-line workers, two live issue-resolution sidecars.
        mod._cmdline_worker_pids = lambda: set()
        mod.live_resolve_worker_pids = lambda runs_dir, **kw: {101, 102}
        mod.live_goal_worker_pids = lambda runs_dir, **kw: set()
        mod.managed_worker_census = lambda root, *, product: {
            "pids": [101, 102], "count": 2,
            "status": "CONSISTENT", "ambiguous": []}
        p = run_eval(mod, max_workers=2)
        self.assertEqual(p["live"], 2)
        self.assertEqual(p["os_worker_procs"], 2)
        self.assertEqual(p["verdict"], mod.REFUSE_AT_CAP)

    def test_seat_process_disagreement_is_typed_without_weakening_capacity(self) -> None:
        mod = load()
        patch_checks(
            mod, procs=2, kernel={"alive": 0, "target": 20, "verdict": "BELOW_TARGET"},
            seat={"total": 12, "free": 9, "leased": 3, "depleted": False},
        )
        p = run_eval(mod, max_workers=20)
        self.assertEqual(p["os_worker_procs"], 2)
        self.assertEqual(p["live"], 2)
        self.assertEqual(p["seat"]["process_gap"], 1)
        self.assertEqual(p["seat"]["process_consistency"], "SEATS_EXCEED_PROCESS_TREES")
        self.assertEqual(p["seat"]["free"], 9)

    def test_process_tree_excess_is_typed_and_reserves_unattributed_capacity(self) -> None:
        mod = load()
        patch_checks(
            mod, procs=4, kernel={"alive": 0, "target": 20, "verdict": "BELOW_TARGET"},
            seat={"total": 12, "free": 11, "leased": 1, "depleted": False},
        )
        p = run_eval(mod, max_workers=20)
        self.assertEqual(p["seat"]["process_gap"], -3)
        self.assertEqual(p["seat"]["process_consistency"], "PROCESS_TREES_EXCEED_SEATS")
        self.assertEqual(p["seat"]["unattributed_live"], 3)
        self.assertEqual(p["seat"]["free"], 8)

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
        # zero target must NOT pin the cap to 0 â€” that silently froze the live
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
        # --max-workers" â€” the live host headroom still throttles the spawn count.
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


class CapacityLimiterTest(unittest.TestCase):
    """#1803: the preflight status names the primary cap limiter and carries the raw
    terms used to compute it."""

    def test_configured_max_limiter(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0)
        p = run_eval(mod, max_workers=4)
        self.assertEqual(p["capacity_limiter"]["primary"], "configured_max")
        self.assertEqual(p["capacity_limiter"]["term"], "max_workers")
        self.assertEqual(p["capacity_limiter"]["raw"]["max_workers"], 4)

    def test_zero_configured_cap_is_not_misreported_as_leases(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0)
        p = run_eval(mod, max_workers=0)
        self.assertEqual(p["capacity_limiter"]["primary"], "configured_max")

    def test_cpu_limiter(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0,
                     host_res={"cores": 2, "free_ram_mb": 128_000, "total_threads": 100})
        p = run_eval(mod, max_workers=10)
        self.assertEqual(p["capacity_limiter"]["primary"], "cpu")
        self.assertEqual(p["capacity_limiter"]["term"], "host_cap")
        self.assertEqual(p["capacity_limiter"]["raw"]["host_binding"], "cores")

    def test_memory_limiter(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0,
                     host_res={"cores": 64, "free_ram_mb": 3000, "total_threads": 100})
        p = run_eval(mod, max_workers=10)
        self.assertEqual(p["capacity_limiter"]["primary"], "memory")
        self.assertEqual(p["capacity_limiter"]["term"], "host_cap")
        self.assertEqual(p["capacity_limiter"]["raw"]["host_components"]["ram"], 2)

    def test_seat_limiter(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0,
                     seat={"total": 3, "free": 3, "leased": 0, "depleted": False})
        p = run_eval(mod, max_workers=10)
        self.assertEqual(p["capacity_limiter"]["primary"], "seats")
        self.assertEqual(p["capacity_limiter"]["term"], "seat_total")
        self.assertEqual(p["capacity_limiter"]["raw"]["seat_free"], 3)

    def test_live_leases_limiter(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=4,
                     seat={"total": 4, "free": 0, "leased": 4, "depleted": True})
        p = run_eval(mod, max_workers=100)
        self.assertEqual(p["capacity_limiter"]["primary"], "leases")
        self.assertEqual(p["capacity_limiter"]["term"], "live")
        self.assertEqual(p["capacity_limiter"]["raw"]["live"], 4)

    def test_render_shows_limiter_terms(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0,
                     host_res={"cores": 64, "free_ram_mb": 3000, "total_threads": 100})
        text = mod.render(run_eval(mod, max_workers=10))
        self.assertIn("limiter=memory", text)
        self.assertIn("host_cap=2", text)
        self.assertIn("host_binding=ram", text)


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

    def test_image_matches_product_scopes_by_backend(self) -> None:
        mod = load()
        self.assertTrue(mod._image_matches_product("claude.exe", "claude"))
        self.assertTrue(mod._image_matches_product("/usr/bin/claude", "claude"))
        self.assertFalse(mod._image_matches_product("opencode.exe", "claude"))
        self.assertTrue(mod._image_matches_product("opencode", "opencode"))
        self.assertFalse(mod._image_matches_product("", "claude"))

    def test_proc_worker_count_product_counts_cmdline_dispatch_loop_worker(self) -> None:
        # Regression (#preflight-live-detection-gap): a live
        # `claude ... /dos-dispatch-loop --lane docs` worker writes no `.backend`
        # sidecar, so the product-scoped count returned 0 while it was alive and
        # authorized an over-subscribing spawn. It must now be counted via its image.
        mod = load()
        mod.live_resolve_worker_pids = lambda *a, **k: set()
        mod.live_goal_worker_pids = lambda *a, **k: set()
        mod._cmdline_worker_pids = (
            lambda product=None: {4242} if product == "claude" else set())
        self.assertEqual(mod.proc_worker_count(Path("/nonexistent"), product="claude"), 1)
        # a sibling product's cap is unaffected â€” independent per-pool headroom.
        self.assertEqual(mod.proc_worker_count(Path("/nonexistent"), product="opencode"), 0)

    def test_worker_pids_from_process_rows_filters_backend_and_collapses_children(self) -> None:
        mod = load()
        rows = [
            {"ProcessId": 100, "ParentProcessId": 1, "Name": "claude.exe",
             "CommandLine": "claude -p /dos-kernel:dos-dispatch-loop --lane docs"},
            {"ProcessId": 101, "ParentProcessId": 100, "Name": "claude.exe",
             "CommandLine": "claude -p /dos-kernel:dos-dispatch-loop --lane docs"},
            {"ProcessId": 200, "ParentProcessId": 1, "Name": "powershell.exe",
             "CommandLine": "rg dos-dispatch-loop"},
            {"ProcessId": 300, "ParentProcessId": 1, "Name": "opencode.exe",
             "CommandLine": "opencode resolve GitHub issue #717"},
        ]
        self.assertEqual(
            mod._worker_pids_from_process_rows(rows, product="claude"), {100})
        self.assertEqual(
            mod._worker_pids_from_process_rows(rows, product="opencode"), {300})

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

    def test_live_resolve_worker_pids_counts_guarded_fak_root(self) -> None:
        # Guarded Claude workers record the `fak guard` root PID in the sidecar,
        # not the claude child. If Windows hides that root cmdline, the image name
        # plus spawn window must still count it as live.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            side = runs / "resolve-2347-20260703-233432.pid"
            side.write_text("27800", encoding="utf-8")
            side.with_suffix(".backend").write_text("claude", encoding="utf-8")
            os.utime(side, (now, now))
            def probe(pid):
                return {"alive": True, "create_time": now - 1,
                        "name": "fak.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolve_worker_pids(runs, alive={27800}, probe=probe), {27800})

    def test_live_resolve_worker_pids_counts_temp_fak_guard_root(self) -> None:
        # Build-verification and canary guard binaries live outside the tree as
        # fak-*.exe. They are still sidecar-authenticated guarded roots, subject to
        # the same create-time window as fak.exe.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            side = runs / "resolve-2042-20260705-182420.pid"
            side.write_text("54324", encoding="utf-8")
            side.with_suffix(".backend").write_text("opencode", encoding="utf-8")
            os.utime(side, (now, now))
            def probe(pid):
                return {"alive": True, "create_time": now - 1,
                        "name": "fak-verify.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolve_worker_pids(runs, alive={54324}, probe=probe), {54324})

    def test_live_resolve_worker_pids_rejects_old_fak_pid_after_sidecar(self) -> None:
        # `fak.exe` is accepted only as a sidecar-authenticated guarded root.
        # A stale sidecar whose pid was later reused by an unrelated fak command
        # outside the spawn window must not pin capacity.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            side = runs / "resolve-2347-20260703-233432.pid"
            side.write_text("27800", encoding="utf-8")
            os.utime(side, (now, now))
            def probe(pid):
                return {"alive": True, "create_time": now + 60 * 60,
                        "name": "fak.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolve_worker_pids(runs, alive={27800}, probe=probe), set())

    def test_live_repair_worker_pid_counts_toward_cap(self) -> None:
        # A contract-repair worker (repair-<N>-<stamp>.pid, spawned by the
        # dispatcher when its whole contract-scan window fails the gate) burns
        # the same account seat a resolution worker does â€” it must pin the cap.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            side = runs / "repair-1207-20260702-190000.pid"
            side.write_text("303", encoding="utf-8")
            os.utime(side, (now, now))
            def probe(pid):
                return {"alive": True, "create_time": now - 1,
                        "name": "claude.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolve_worker_pids(runs, alive={303}, probe=probe), {303})

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
        mod.live_goal_worker_pids = lambda runs_dir, **kw: set()
        self.assertEqual(mod.proc_worker_count(ROOT), 3)

    def test_proc_worker_count_scopes_to_product_pool(self) -> None:
        # A product-scoped count = that product's sidecars PLUS the generic
        # cmdline-marked dos-dispatch-loop workers whose process image is that
        # product's backend (they carry no `.backend` sidecar, so the image is the
        # only pool signal). A SIBLING product's workers never pin this pool's cap,
        # so the two account pools still fill independently â€” but a claude dos-loop
        # worker now correctly pins the claude pool (closing the undercount that
        # authorized an over-subscribing spawn).
        mod = load()
        # cmdline workers are now product-filtered by image: the live claude
        # dos-loop worker {999} belongs to the claude pool, none to opencode.
        mod._cmdline_worker_pids = (
            lambda product=None: {999} if product in (None, "claude") else set())
        mod.live_goal_worker_pids = lambda runs_dir, **kw: set()
        seen = {}

        def fake_pids(runs_dir, **kw):
            seen["product"] = kw.get("product")
            return {201, 202} if kw.get("product") == "opencode" else {201, 202, 203}
        mod.live_resolve_worker_pids = fake_pids
        # Unscoped: cmdline âˆª all sidecars = {999, 201, 202, 203} = 4
        self.assertEqual(mod.proc_worker_count(ROOT), 4)
        # claude pool: its sidecars âˆª its cmdline worker {999} = 4
        self.assertEqual(mod.proc_worker_count(ROOT, product="claude"), 4)
        # opencode pool: only its sidecars â€” the claude cmdline worker does NOT pin it
        self.assertEqual(mod.proc_worker_count(ROOT, product="opencode"), 2)
        self.assertEqual(seen["product"], "opencode")

    def test_codex_process_rows_collapse_node_wrapper_and_native_child(self) -> None:
        # Windows Codex shape: node.exe runs @openai/codex/bin/codex.js, then spawns
        # codex.exe. The ambient-seat count must read that as ONE live Codex session,
        # not two, while still counting a wrapper whose native child has not appeared.
        mod = load()
        rows = [
            {"pid": 10, "ppid": 1, "name": "node.exe",
             "cmdline": r"C:\node.exe C:\Users\u\AppData\Roaming\npm\node_modules\@openai\codex\bin\codex.js"},
            {"pid": 11, "ppid": 10, "name": "codex.exe",
             "cmdline": r"C:\...\codex.exe"},
            {"pid": 20, "ppid": 1, "name": "node.exe",
             "cmdline": "/usr/bin/node /x/node_modules/@openai/codex/bin/codex.js"},
            {"pid": 30, "ppid": 1, "name": "node.exe",
             "cmdline": "/usr/bin/node /x/not-codex.js"},
        ]
        self.assertEqual(mod._codex_process_pids_from_rows(rows), {11, 20})

    def test_proc_worker_count_codex_excludes_unregistered_interactive_processes(self) -> None:
        mod = load()
        mod.live_resolve_worker_pids = lambda runs_dir, **kw: set()
        mod.live_goal_worker_pids = lambda runs_dir, **kw: set()
        mod._cmdline_worker_pids = lambda product=None: set()
        self.assertEqual(mod.proc_worker_count(ROOT, product="codex"), 0)

    def test_codex_marker_only_process_row_is_managed(self) -> None:
        mod = load()
        rows = [{"pid": 303, "ppid": 1, "name": "codex.exe",
                 "cmdline": "codex exec resolve GitHub issue #9400"}]
        self.assertEqual(mod._worker_pids_from_process_rows(rows, product="codex"),
                         {303})

    def test_codex_registered_sidecar_counts_once_and_stale_sidecar_drops(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / mod.RUNS_DIRNAME
            runs.mkdir()
            side = runs / "resolve-9400-20260827-120000.pid"
            side.write_text("101", encoding="utf-8")
            side.with_suffix(".backend").write_text("codex", encoding="utf-8")
            now = side.stat().st_mtime
            mod._cmdline_worker_pids = lambda product=None: {201}
            mod._ambient_codex_process_rows = lambda: [
                {"pid": 111, "ppid": 101, "name": "node.exe", "cmdline": "codex.js"},
                {"pid": 201, "ppid": 111, "name": "codex.exe",
                 "cmdline": "codex exec resolve GitHub issue #9400"},
            ]
            mod._process_probe = lambda pid: {
                "alive": True, "create_time": now, "name": "codex.exe",
                "cmdline": "codex exec resolve GitHub issue #9400",
            }
            self.assertEqual(mod.proc_worker_count(root, product="codex"), 1)
            self.assertEqual(mod.managed_worker_identity_check(root, product="codex")["status"],
                             "CONSISTENT")

            mod._process_probe = lambda pid: {"alive": False}
            mod._cmdline_worker_pids = lambda product=None: set()
            self.assertEqual(mod.proc_worker_count(root, product="codex"), 0)
            self.assertEqual(mod.managed_worker_identity_check(root, product="codex")["status"],
                             "CONSISTENT")

    def test_codex_stdin_goal_breadcrumb_is_managed(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            goals = root / mod.GOAL_RUNS_DIRNAME
            goals.mkdir()
            side = goals / "issue-9400-20260827-120000.pid"
            side.write_text("404", encoding="utf-8")
            now = side.stat().st_mtime
            mod._cmdline_worker_pids = lambda product=None: set()
            census = mod.managed_worker_census(
                root, product="codex", probe=lambda pid: {
                    "alive": True, "create_time": now, "name": "codex.exe",
                    "cmdline": "codex exec -",
                })
            self.assertEqual(census["pids"], [404])
            self.assertEqual(census["count"], 1)
            self.assertEqual(census["status"], "CONSISTENT")

    def test_codex_live_ambiguous_sidecar_refuses_inspection(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / mod.RUNS_DIRNAME
            runs.mkdir()
            side = runs / "resolve-9400-20260827-120000.pid"
            side.write_text("101", encoding="utf-8")
            now = side.stat().st_mtime
            mod._process_probe = lambda pid: {
                "alive": True, "create_time": now + 3600, "name": "powershell.exe",
                "cmdline": "powershell.exe",
            }
            identity = mod.managed_worker_identity_check(root, product="codex")
            self.assertEqual(identity["status"], "AMBIGUOUS")
            self.assertEqual(identity["ambiguous"][0]["reason"], "missing_backend")

            patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"},
                         procs=0, seat={"total": 10, "free": 10, "leased": 0,
                                        "depleted": False})
            mod.managed_worker_census = lambda root, product=None: {
                "pids": [], "count": 0, **identity}
            p = run_eval(mod, max_workers=1, product="codex")
            self.assertEqual(p["verdict"], mod.REFUSE_INSPECT)
            self.assertEqual(p["worker_identity"]["status"], "AMBIGUOUS")

    def test_codex_malformed_probe_and_contradictory_sidecars_are_ambiguous(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / mod.RUNS_DIRNAME
            runs.mkdir()
            fixtures = ((101, "not-a-pid", "codex"),
                        (202, "202", "codex"),
                        (303, "303", "codex"))
            for issue, pid, backend in fixtures:
                side = runs / f"resolve-{issue}-20260827-120000.pid"
                side.write_text(pid, encoding="utf-8")
                side.with_suffix(".backend").write_text(backend, encoding="utf-8")
            mod._cmdline_worker_pids = lambda product=None: set()
            def probe(pid):
                if pid == 202:
                    raise PermissionError("denied")
                return {"alive": True, "create_time": 0, "name": "claude.exe",
                        "cmdline": "claude resolve GitHub issue #9400"}
            census = mod.managed_worker_census(root, product="codex", probe=probe)
            self.assertEqual(census["count"], 0)
            self.assertEqual({row["reason"] for row in census["ambiguous"]}, {
                "malformed_pid", "probe_inspection_failed",
                "contradictory_backend_image",
            })

    def test_codex_mixed_backend_sidecars_are_isolated(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / mod.RUNS_DIRNAME
            runs.mkdir()
            now = 1_000_000.0
            for pid, backend in ((101, "codex"), (202, "claude")):
                side = runs / f"resolve-{pid}-20260827-120000.pid"
                side.write_text(str(pid), encoding="utf-8")
                side.with_suffix(".backend").write_text(backend, encoding="utf-8")
                os.utime(side, (now, now))
            mod._cmdline_worker_pids = lambda product=None: set()
            mod._process_probe = lambda pid: {
                "alive": True, "create_time": now, "name": f"{101 == pid and 'codex' or 'claude'}.exe",
                "cmdline": f"{101 == pid and 'codex' or 'claude'} resolve GitHub issue #9400",
            }
            self.assertEqual(mod.proc_worker_count(root, product="codex"), 1)
            self.assertEqual(mod.proc_worker_count(root, product="claude"), 1)
            self.assertEqual(mod.managed_worker_identity_check(root, product="codex")["status"],
                             "CONSISTENT")

    def test_codex_seat_keeps_ambient_sessions_as_telemetry_only(self) -> None:
        mod = load()
        mod.DEFAULT_CODEX_OAUTH_SESSIONS = 10
        mod.ambient_codex_pids = lambda: {201, 202, 203}
        seat = mod.seat_check(ROOT, product="codex")
        self.assertEqual(seat["total"], 10)
        self.assertEqual(seat["free"], 10)
        self.assertEqual(seat["leased"], 0)
        self.assertFalse(seat["depleted"])
        self.assertEqual(seat["ambient_live"], 3)

        mod.ambient_codex_pids = lambda: set(range(20))
        seat = mod.seat_check(ROOT, product="codex")
        self.assertEqual(seat["total"], 10)
        self.assertEqual(seat["free"], 10)
        self.assertEqual(seat["leased"], 0)
        self.assertFalse(seat["depleted"])
        self.assertEqual(seat["ambient_live"], 20)

    def test_codex_managed_worker_hits_cap_boundary(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=1,
                     seat={"total": 10, "free": 10, "leased": 0, "depleted": False})
        self.assertEqual(run_eval(mod, max_workers=1, product="codex")["verdict"],
                         mod.REFUSE_AT_CAP)
        self.assertEqual(run_eval(mod, max_workers=2, product="codex")["verdict"],
                         mod.OK_VERDICT)

    def test_codex_registered_worker_folds_final_seat_and_process_gap(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=1,
                     seat={"total": 10, "free": 10, "leased": 0,
                           "depleted": False})
        p = run_eval(mod, max_workers=10, product="codex")
        self.assertEqual(p["live"], 1)
        self.assertEqual(p["seat"]["leased"], 1)
        self.assertEqual(p["seat"]["free"], 9)
        self.assertEqual(p["seat"]["process_consistency"],
                         "PROCESS_TREES_EXCEED_SEATS")

    def test_codex_resolver_and_native_fixture_json_agree(self) -> None:
        """Mirror the native #7827 fixture at the shared JSON boundary."""
        mod = load()
        rows = [
            {"pid": 10, "ppid": 1, "name": "codex.exe", "cmdline": "codex"},
            {"pid": 20, "ppid": 1, "name": "node.exe", "cmdline": "codex.js"},
            {"pid": 21, "ppid": 20, "name": "codex.exe", "cmdline": "codex"},
            {"pid": 200, "ppid": 1, "name": "codex.exe",
             "cmdline": "codex exec resolve GitHub issue #9400"},
        ]
        managed = mod._worker_pids_from_process_rows(rows, product="codex")
        self.assertEqual(managed, {200})
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"},
                     procs=len(managed),
                     seat={"total": 10, "free": 10, "leased": 0,
                           "depleted": False})
        resolver = run_eval(mod, max_workers=1, product="codex")
        resolver_json = json.dumps({
            "live_workers": resolver["live"],
            "os_worker_procs": resolver["worker_identity"]["count"],
            "verdict": resolver["verdict"],
        }, sort_keys=True)
        native_fixture_json = json.dumps({
            "live_workers": 1,
            "os_worker_procs": 1,
            "verdict": mod.REFUSE_AT_CAP,
        }, sort_keys=True)
        self.assertEqual(resolver_json, native_fixture_json)

    def test_codex_foreground_sessions_cannot_deplete_managed_seats(self) -> None:
        mod = load()
        mod.DEFAULT_CODEX_OAUTH_SESSIONS = 10
        mod.ambient_codex_pids = lambda: set(range(20))
        seat = mod.seat_check(ROOT, product="codex")
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"},
                     procs=0, seat=seat)
        p = run_eval(mod, max_workers=10, product="codex")
        self.assertEqual(p["cap"], 10)
        self.assertEqual(p["live"], 0)
        self.assertEqual(p["verdict"], mod.OK_VERDICT)

    def test_fak_command_prefers_installed_binary_over_repo_artifact(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ("fak.exe" if os.name == "nt" else "fak")).write_text(
                "stale developer build", encoding="utf-8")
            bindir = root / "bin"
            bindir.mkdir()
            installed = bindir / ("fak.exe" if os.name == "nt" else "fak")
            installed.write_text("installed build", encoding="utf-8")
            env = dict(os.environ, PATH=str(bindir), FAK_BIN="")
            self.assertEqual(mod._fak_command(root, env), [str(installed)])

    def test_fak_command_explicit_override_still_wins(self) -> None:
        mod = load()
        env = dict(os.environ, FAK_BIN="C:/pinned/fak.exe")
        self.assertEqual(mod._fak_command(ROOT, env), ["C:/pinned/fak.exe"])

    def test_account_check_native_route_rejects_needs_login(self) -> None:
        mod = load()
        # Models an enabled registry home whose credential file exists but whose
        # OAuth accessToken is empty; native resolve folds it to needs_login/false.
        blocked = {"ok": False, "selected_tier": 1,
            "tag": "logged-out", "config_dir": "/tmp/logged-out", "model": "opus",
            "login_status": "needs_login", "can_serve": False,
            "block_reason": "credential has no usable access token"}
        with mock.patch.object(mod, "_fak_command", return_value=["fak"]), \
             mock.patch.object(mod, "run_json", return_value=blocked) as run:
            got = mod.account_check(ROOT, work_kind="issue", product="claude")
        self.assertFalse(got["available"])
        self.assertEqual(got["login_status"], "needs_login")
        self.assertFalse(got["can_serve"])
        self.assertIn("no usable access token", got["reason"])
        self.assertEqual(run.call_args.args[0][:3],
                         ["fak", "fleet-accounts", "resolve"])

    def test_account_check_native_route_selects_ready_account(self) -> None:
        mod = load()
        ready = {"ok": True, "selected_tier": 1,
            "tag": "ready-seat", "config_dir": "/tmp/ready", "model": "opus",
            "login_status": "ready", "can_serve": True}
        with mock.patch.object(mod, "_fak_command", return_value=["fak"]), \
             mock.patch.object(mod, "run_json", return_value=ready):
            got = mod.account_check(ROOT, work_kind="issue", product="claude")
        self.assertTrue(got["available"])
        self.assertEqual(got["tag"], "ready-seat")
        self.assertEqual(got["login_status"], "ready")
        self.assertTrue(got["can_serve"])

    def test_account_check_missing_native_router_fails_closed(self) -> None:
        mod = load()
        with mock.patch.object(mod, "_fak_command", return_value=None):
            got = mod.account_check(ROOT, work_kind="issue", product="claude")
        self.assertFalse(got["available"])
        self.assertIn("not found", got["error"])

    def test_account_check_codex_uses_ambient_login(self) -> None:
        # Codex has no switcher roster â€” its availability is read from ~/.codex.
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


class GoalBreadcrumbTest(unittest.TestCase):
    """#2226: a detached /goal worker is fed its goal via STDIN (`claude -p` with
    no prompt argument), so the cmdline scan is blind to it until it leases a
    lane. The launcher's `.goal-runs/<tag>-<stamp>.pid` breadcrumb must occupy a
    cap slot from the instant of launch â€” and a dead-pid breadcrumb must NEVER
    inflate the count and wedge spawning."""

    NOW = 1_000_000.0

    def _crumb(self, runs: Path, pid,
               name: str = "resolve-tickets-witnessed-20260702-090000.pid") -> Path:
        crumb = runs / name
        crumb.write_text(f"{pid}\n", encoding="utf-8")
        os.utime(crumb, (self.NOW, self.NOW))
        return crumb

    def _live_claude_probe(self):
        # The stdin-fed shape this issue is about: a live claude image whose
        # command line carries NO worker marker (the goal went in via stdin).
        def probe(pid):
            return {"alive": True, "create_time": self.NOW - 1.0,
                    "name": "claude.exe",
                    "cmdline": "claude -p --permission-mode bypassPermissions"}
        return probe

    def test_live_breadcrumb_counts_from_the_instant_of_launch(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._crumb(runs, 4711)
            self.assertEqual(
                mod.live_goal_worker_pids(runs, probe=self._live_claude_probe()),
                {4711})

    def test_dead_pid_breadcrumb_is_ignored(self) -> None:
        # Stale-breadcrumb hygiene: the worker exited, its breadcrumb remains â€”
        # it must contribute NOTHING to the live count.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._crumb(runs, 4712)
            self.assertEqual(
                mod.live_goal_worker_pids(runs, probe=lambda pid: {"alive": False}),
                set())

    def test_recycled_pid_created_after_breadcrumb_is_rejected(self) -> None:
        # Windows pid reuse: a LATER claude session recycled the dead worker's
        # pid. Alive + right image is not enough â€” the create time sits outside
        # the breadcrumb's spawn window, so it must not consume a cap slot.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._crumb(runs, 4713)
            def probe(pid):
                return {"alive": True, "create_time": self.NOW + 60 * 60,
                        "name": "claude.exe", "cmdline": ""}
            self.assertEqual(mod.live_goal_worker_pids(runs, probe=probe), set())

    def test_recycled_shell_pid_in_window_is_rejected(self) -> None:
        # A bare shell that recycled the pid inside the spawn window is NOT a
        # worker â€” same guard as the resolve sidecars.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._crumb(runs, 4714)
            def probe(pid):
                return {"alive": True, "create_time": self.NOW - 30,
                        "name": "cmd.exe", "cmdline": ""}
            self.assertEqual(mod.live_goal_worker_pids(runs, probe=probe), set())

    def test_breadcrumb_scopes_by_live_process_image(self) -> None:
        # Goal workers write no `.backend` sidecar; the live process image is the
        # pool signal, so a claude /goal worker pins the claude cap, not opencode's.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._crumb(runs, 4715)
            probe = self._live_claude_probe()
            self.assertEqual(
                mod.live_goal_worker_pids(runs, product="claude", probe=probe), {4715})
            self.assertEqual(
                mod.live_goal_worker_pids(runs, product="opencode", probe=probe), set())

    def test_malformed_and_foreign_pid_files_are_ignored(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._crumb(runs, "not-a-pid")                    # unparseable pid
            self._crumb(runs, 4716, name="random.pid")        # no <tag>-<stamp> shape
            self.assertEqual(
                mod.live_goal_worker_pids(runs, probe=self._live_claude_probe()),
                set())

    def test_proc_worker_count_folds_goal_breadcrumbs(self) -> None:
        # Union semantics: a worker visible through several witnesses counts once,
        # and the goal breadcrumbs add to the existing cmdline/sidecar rungs.
        mod = load()
        mod._cmdline_worker_pids = lambda: {101}
        mod.live_resolve_worker_pids = lambda runs_dir, **kw: {102}
        mod.live_goal_worker_pids = lambda runs_dir, **kw: {101, 103}
        self.assertEqual(mod.proc_worker_count(ROOT), 3)

    def test_evaluate_counts_goal_breadcrumb_toward_cap(self) -> None:
        # End to end (the issue's done-condition rung): one live stdin-fed /goal
        # worker â€” no lane lease, no cmdline marker, ONLY its breadcrumb â€” and the
        # preflight already sees live=1 and refuses at cap=1.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / mod.GOAL_RUNS_DIRNAME).mkdir()
            self._crumb(root / mod.GOAL_RUNS_DIRNAME, 4717)
            real_census = mod.managed_worker_census
            patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"})
            mod.managed_worker_census = real_census
            mod._cmdline_worker_pids = lambda product=None: set()
            mod._process_probe = self._live_claude_probe()
            p = mod.evaluate(root, max_workers=1, work_kind="engineering",
                             product="claude")
            self.assertEqual(p["live"], 1)
            self.assertEqual(p["os_worker_procs"], 1)
            self.assertEqual(p["verdict"], mod.REFUSE_AT_CAP)

    def test_stale_breadcrumb_never_wedges_spawning(self) -> None:
        # Same wiring, but the breadcrumb's pid is DEAD: live stays 0 and the
        # gate stays SPAWN_OK â€” a stale entry can never pin the cap.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / mod.GOAL_RUNS_DIRNAME).mkdir()
            self._crumb(root / mod.GOAL_RUNS_DIRNAME, 4718)
            real_census = mod.managed_worker_census
            patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"})
            mod.managed_worker_census = real_census
            mod._cmdline_worker_pids = lambda product=None: set()
            mod._process_probe = lambda pid: {"alive": False}
            p = mod.evaluate(root, max_workers=1, work_kind="engineering",
                             product="claude")
            self.assertEqual(p["live"], 0)
            self.assertEqual(p["verdict"], mod.OK_VERDICT)


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
    """The explicit account session-slot pool (#1336): pool -> live-worker binding,
    depletion, over-cap double-booking, and the no-double-count duplicate identity rule."""

    def test_free_pool_has_full_headroom_and_is_not_depleted(self) -> None:
        fa = load_fleet_accounts()
        pool = fa.seat_pool([_seat_row("a"), _seat_row("b"), _seat_row("c")], [])
        self.assertEqual(pool["total_seats"], 12)
        self.assertEqual(pool["free_seats"], 12)
        self.assertEqual(pool["leased_seats"], 0)
        self.assertFalse(pool["depleted"])
        self.assertEqual(pool["double_booked"], [])

    def test_a_live_lease_binds_its_seat(self) -> None:
        fa = load_fleet_accounts()
        leases = [{"worker": "resolve-101", "pid": 101, "tag": "a", "dir": "/home/u/a"}]
        pool = fa.seat_pool([_seat_row("a"), _seat_row("b")], leases)
        self.assertEqual(pool["leased_seats"], 1)
        self.assertEqual(pool["free_seats"], 7)
        leased = [s for s in pool["seats"] if s["state"] == "leased"]
        self.assertEqual(len(leased), 1)
        self.assertEqual(leased[0]["tag"], "a")
        self.assertEqual(leased[0]["session_cap"], 4)
        self.assertEqual(leased[0]["free_slots"], 3)
        self.assertEqual(leased[0]["workers"], ["resolve-101"])

    def test_pool_depleted_when_every_session_slot_leased(self) -> None:
        fa = load_fleet_accounts()
        leases = [
            {"worker": f"a{i}", "tag": "a", "dir": "/home/u/a"} for i in range(4)
        ] + [
            {"worker": f"b{i}", "tag": "b", "dir": "/home/u/b"} for i in range(4)
        ]
        pool = fa.seat_pool([_seat_row("a"), _seat_row("b")], leases)
        self.assertEqual(pool["leased_seats"], 8)
        self.assertEqual(pool["free_seats"], 0)
        self.assertTrue(pool["depleted"])

    def test_overbooking_one_account_beyond_cap_is_surfaced(self) -> None:
        # Four Claude sessions on one account are admitted; the fifth is the
        # over-cap condition that must be OBSERVABLE, not silently assumed away.
        fa = load_fleet_accounts()
        leases = [{"worker": f"w{i}", "tag": "a", "dir": "/home/u/a"} for i in range(5)]
        pool = fa.seat_pool([_seat_row("a")], leases)
        self.assertEqual(len(pool["double_booked"]), 1)
        self.assertEqual(pool["double_booked"][0]["session_cap"], 4)
        self.assertEqual(sorted(pool["double_booked"][0]["workers"]),
                         ["w0", "w1", "w2", "w3", "w4"])

    def test_lease_on_account_not_in_pool_is_unbound(self) -> None:
        fa = load_fleet_accounts()
        leases = [{"worker": "ghost", "tag": "gone", "dir": "/home/u/gone"}]
        pool = fa.seat_pool([_seat_row("a")], leases)
        self.assertEqual(pool["leased_seats"], 0)
        self.assertEqual(pool["free_seats"], 4)
        self.assertEqual(len(pool["unbound_leases"]), 1)
        self.assertEqual(pool["unbound_leases"][0]["worker"], "ghost")

    def test_two_dirs_on_one_account_are_one_seat(self) -> None:
        # The no-double-hand core: two dirs sharing one Anthropic account (a duplicate
        # identity) collapse to ONE seat, so the pool never double-counts a rate limit.
        fa = load_fleet_accounts()
        rows = [_seat_row("canon", uuid="U1"),
                _seat_row("copy", uuid="U1", role="duplicate")]
        pool = fa.seat_pool(rows, [])
        self.assertEqual(pool["total_seats"], 4)
        self.assertEqual(pool["seats"][0]["tag"], "canon")

    def test_same_uuid_collapses_even_before_duplicate_annotation(self) -> None:
        # Defensive collapse: if two offered rows share one account UUID before the
        # duplicate-role annotation lands, the seat pool still counts one rate-limit
        # pool and binds a lease stamped with either row.
        fa = load_fleet_accounts()
        rows = [_seat_row("canon", uuid="U1", dir_="/home/u/canon"),
                _seat_row("copy", uuid="U1", dir_="/home/u/copy")]
        leases = [{"worker": "copy-worker", "tag": "copy", "dir": "/home/u/copy"}]
        pool = fa.seat_pool(rows, leases)
        self.assertEqual(pool["total_seats"], 4)
        self.assertEqual(len(pool["seats"]), 1)
        self.assertEqual(pool["leased_seats"], 1)
        self.assertEqual(pool["free_seats"], 3)
        self.assertEqual(pool["seats"][0]["workers"], ["copy-worker"])

    def test_product_scope_filters_the_pool(self) -> None:
        fa = load_fleet_accounts()
        rows = [_seat_row("a", product="claude"), _seat_row("g", product="opencode")]
        self.assertEqual(fa.seat_pool(rows, [], product="claude")["total_seats"], 4)
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
            self.assertEqual(pool["free_seats"], 4)
            self.assertFalse(pool["depleted"])


class SeatRefusalTest(unittest.TestCase):
    """Preflight folds the seat pool into the cap and emits the typed REFUSE_NO_SEAT
    on depletion (#1336): N>M wave -> exactly M, remainder structurally refused."""

    def test_seat_count_is_the_effective_cap(self) -> None:
        # An N>M ask (max_workers=100) with a free pool of M=4 caps at 4, not 100 â€”
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

    def test_unattributed_live_workers_consume_free_slots(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=11,
                     seat={"total": 20, "free": 20, "leased": 0, "depleted": False})
        p = run_eval(mod, max_workers=20)
        self.assertEqual(p["verdict"], mod.OK_VERDICT)
        self.assertEqual(p["seat"]["free"], 9)
        self.assertEqual(p["seat"]["leased"], 11)
        self.assertEqual(p["seat"]["unattributed_live"], 11)
        self.assertEqual(p["capacity_limiter"]["raw"]["seat_free"], 9)
        self.assertEqual(p["capacity_limiter"]["raw"]["seat_leased"], 11)

    def test_unattributed_live_workers_can_deplete_pool(self) -> None:
        mod = load()
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=4,
                     seat={"total": 4, "free": 4, "leased": 0, "depleted": False})
        p = run_eval(mod, max_workers=100)
        self.assertEqual(p["cap"], 4)
        self.assertEqual(p["verdict"], mod.REFUSE_NO_SEAT)
        self.assertEqual(p["seat"]["free"], 0)
        self.assertEqual(p["seat"]["leased"], 4)
        self.assertTrue(p["seat"]["depleted"])

    def test_all_blocked_pool_is_no_account_not_no_seat(self) -> None:
        # A pool with no free seat because every seat is THROTTLED (none leased) is a
        # REFUSE_NO_ACCOUNT, not a REFUSE_NO_SEAT â€” depletion-by-lease and
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


class RaisedDefaultCeilingTest(unittest.TestCase):
    """The raised static ceiling (DEFAULT_MAX_WORKERS -> Darwin 30, env-tunable via
    FAK_MAX_WORKERS) is safe iff it stays strictly bounded by the adaptive gates.
    These tests pin the new default, prove the env knob retunes it both ways, and
    prove the raise can never exceed host_cap or the seat pool â€” i.e. raising the
    operator ceiling only realizes concurrency the box and roster can carry."""

    def _load_with_env(self, value: str | None):
        """Load the module with FAK_MAX_WORKERS pinned (None = unset), hermetic to
        whatever the real box exports â€” the constant is resolved at import time."""
        return load(FAK_MAX_WORKERS=value)

    def test_platform_defaults_are_deterministic(self) -> None:
        mod = self._load_with_env(None)
        self.assertEqual(mod._built_in_max_workers("darwin"), 30)
        self.assertEqual(mod._built_in_max_workers("linux"), 20)
        self.assertEqual(mod._built_in_max_workers("win32"), 20)

    def test_live_platform_default_matches_policy(self) -> None:
        mod = self._load_with_env(None)
        self.assertEqual(mod.DEFAULT_MAX_WORKERS, mod._built_in_max_workers())

    def test_env_knob_retunes_the_ceiling(self) -> None:
        # FAK_MAX_WORKERS is the dynamic half: retune per host, no code change;
        # garbage / non-positive values fall back to the built-in ceiling.
        self.assertEqual(self._load_with_env("12").DEFAULT_MAX_WORKERS, 12)
        self.assertEqual(self._load_with_env("1").DEFAULT_MAX_WORKERS, 1)
        self.assertEqual(self._load_with_env("garbage").DEFAULT_MAX_WORKERS, self._load_with_env(None)._built_in_max_workers())
        self.assertEqual(self._load_with_env("0").DEFAULT_MAX_WORKERS, self._load_with_env(None)._built_in_max_workers())

    def test_default_ceiling_fills_on_a_roomy_box_with_seats(self) -> None:
        # The win: a roomy box with enough free session slots and no dos throttle lets
        # the live platform default ceiling fill â€” governed only by the adaptive gates.
        mod = self._load_with_env(None)
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "AT_TARGET"}, procs=0,
                     host_res={"cores": 64, "free_ram_mb": 128_000, "total_threads": 1000},
                     seat={"total": 20, "free": 20, "leased": 0, "depleted": False})
        p = run_eval(mod, max_workers=mod.DEFAULT_MAX_WORKERS)
        self.assertEqual(p["cap"], min(mod.DEFAULT_MAX_WORKERS, 20))
        self.assertEqual(p["verdict"], mod.OK_VERDICT)

    def test_default_ceiling_still_throttles_on_a_loaded_box(self) -> None:
        # Safety: raising the ceiling cannot saturate a loaded host â€” host_cap pulls
        # the effective cap back below the static ceiling (here to the floor), exactly as before.
        mod = self._load_with_env(None)
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0,
                     host_res={"cores": 8, "free_ram_mb": 64_000, "total_threads": 200_000},
                     seat={"total": 20, "free": 20, "leased": 0, "depleted": False})
        p = run_eval(mod, max_workers=mod.DEFAULT_MAX_WORKERS)
        self.assertEqual(p["host_cap"], 1)
        self.assertEqual(p["cap"], 1)               # raised ceiling, still throttled
        self.assertLess(p["cap"], mod.DEFAULT_MAX_WORKERS)

    def test_default_ceiling_still_bounded_by_a_smaller_seat_pool(self) -> None:
        # Safety: raising the ceiling cannot overbook accounts â€” a 3-slot roster
        # caps the raised ceiling at 3, not the static ceiling.
        mod = self._load_with_env(None)
        patch_checks(mod, kernel={"alive": 0, "target": 0, "verdict": "X"}, procs=0,
                     host_res={"cores": 64, "free_ram_mb": 128_000, "total_threads": 1000},
                     seat={"total": 3, "free": 3, "leased": 0, "depleted": False})
        p = run_eval(mod, max_workers=mod.DEFAULT_MAX_WORKERS)
        self.assertEqual(p["cap"], 3)               # lower seat bound remains binding
        self.assertEqual(p["verdict"], mod.OK_VERDICT)

    def test_host_budget_env_knobs_retune_the_gradient(self) -> None:
        # The per-worker charges are env knobs too (FAK_HOST_*): a measured box can
        # halve the cores-per-worker guess and double its cores dimension.
        mod = load(FAK_HOST_CORES_PER_WORKER="1")
        cap = mod.host_capacity(cores=8, free_ram_mb=128_000, total_threads=1000)
        self.assertEqual(cap["components"]["cores"], 8)   # 8 // 1, not 8 // 2


class WeeklyCapCooldownTest(unittest.TestCase):
    """#2610: a weekly-limit 429 (`kind=weekly_limit`) cools the routed seat until its
    announced reset window instead of re-offering it. This is DISTINCT from the
    stale-credential cases (#2059/#2075 -> REFUSE_NO_ACCOUNT): the credential is valid,
    only the seat is temporarily quota-capped. The persisted hold
    (`.dispatch-runs/account-cap-*.json`, written by the resolve dispatcher's
    check_weekly_cap and read by dispatch_status) is honored here so a fresh preflight
    refuses the capped seat â€” and, because issue_dispatch.py gates on this verdict, it
    stops offering the seat too."""

    NOW_TS = 1_000_000.0  # naive-UTC 1970-01-12T13:46:40
    ACTIVE_UNTIL = "1970-01-12T14:46:40Z"   # 1h after NOW -> cooldown still active
    EXPIRED_UNTIL = "1970-01-12T12:46:40Z"  # 1h before NOW -> cooldown elapsed

    def _write_hold(self, runs: Path, *, product="claude", account="worker-a",
                    until=ACTIVE_UNTIL, kind="weekly", reset_text="1h7m0s",
                    name=None) -> Path:
        runs.mkdir(parents=True, exist_ok=True)
        path = runs / (name or f"account-cap-{product}-{account}.json")
        path.write_text(json.dumps({
            "product": product, "account": account, "kind": kind,
            "reset_text": reset_text, "evidence_log": "resolve-2610-20260704-000000.log",
            "detected": "1970-01-12T13:40:00Z", "until": until}), encoding="utf-8")
        return path

    # --- the pure reader ---------------------------------------------------- #
    def test_active_hold_marks_account_capped(self) -> None:
        import json as _json  # noqa: F401  (json imported at module top)
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write_hold(root / mod.RUNS_DIRNAME)
            out = mod.weekly_cap_check(root, product="claude", account_tag="worker-a",
                                       now_ts=self.NOW_TS)
            self.assertTrue(out["capped"])
            self.assertEqual(out["until"], self.ACTIVE_UNTIL)
            self.assertEqual(out["reset_text"], "1h7m0s")
            self.assertEqual(out["kind"], "weekly")

    def test_expired_hold_is_ignored_so_seat_is_not_permanently_walled(self) -> None:
        # The cooldown self-expires at its announced window (a confusion risk the issue
        # names): once `until` has passed, the seat is offered again.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write_hold(root / mod.RUNS_DIRNAME, until=self.EXPIRED_UNTIL)
            out = mod.weekly_cap_check(root, product="claude", account_tag="worker-a",
                                       now_ts=self.NOW_TS)
            self.assertFalse(out["capped"])

    def test_hold_for_a_different_account_does_not_wall_this_seat(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write_hold(root / mod.RUNS_DIRNAME, account="worker-b")
            out = mod.weekly_cap_check(root, product="claude", account_tag="worker-a",
                                       now_ts=self.NOW_TS)
            self.assertFalse(out["capped"])

    def test_hold_for_a_different_product_does_not_wall_this_seat(self) -> None:
        # A capped claude seat must not wall an uncapped opencode account of the same
        # tag â€” the pools are independent.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write_hold(root / mod.RUNS_DIRNAME, product="opencode",
                             account="worker-a")
            out = mod.weekly_cap_check(root, product="claude", account_tag="worker-a",
                                       now_ts=self.NOW_TS)
            self.assertFalse(out["capped"])

    def test_legacy_null_account_hold_matches_any_tag(self) -> None:
        # A pre-account-scoped generic hold (account: null) is honored for the routed
        # tag, mirroring dispatch_status.read_active_weekly_cap.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write_hold(root / mod.RUNS_DIRNAME, account=None,
                             name="account-cap-claude.json")
            out = mod.weekly_cap_check(root, product="claude", account_tag="worker-a",
                                       now_ts=self.NOW_TS)
            self.assertTrue(out["capped"])

    def test_malformed_hold_is_fail_open_not_capped(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / mod.RUNS_DIRNAME
            runs.mkdir(parents=True, exist_ok=True)
            (runs / "account-cap-claude-worker-a.json").write_text(
                "{ not json", encoding="utf-8")
            out = mod.weekly_cap_check(root, product="claude", account_tag="worker-a",
                                       now_ts=self.NOW_TS)
            self.assertFalse(out["capped"])

    def test_no_runs_dir_is_not_capped(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            out = mod.weekly_cap_check(Path(d), product="claude",
                                       account_tag="worker-a", now_ts=self.NOW_TS)
            self.assertFalse(out["capped"])

    def test_soonest_expiring_active_hold_wins(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / mod.RUNS_DIRNAME
            self._write_hold(runs, until="1970-01-12T18:00:00Z",
                             reset_text="far", name="account-cap-claude.json")
            self._write_hold(runs, until="1970-01-12T15:00:00Z", reset_text="near",
                             name="account-cap-claude-worker-a.json")
            out = mod.weekly_cap_check(root, product="claude", account_tag="worker-a",
                                       now_ts=self.NOW_TS)
            self.assertTrue(out["capped"])
            self.assertEqual(out["reset_text"], "near")

    # --- the evaluate() verdict --------------------------------------------- #
    def _capped(self, **over):
        hold = {"capped": True, "until": self.ACTIVE_UNTIL, "reset_text": "1h7m0s",
                "kind": "weekly", "account": "worker-a", "evidence_log": "resolve.log"}
        hold.update(over)
        return hold

    def test_evaluate_refuses_a_weekly_capped_routed_seat(self) -> None:
        mod = load()
        patch_checks(mod, weekly=self._capped())
        p = run_eval(mod)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], mod.REFUSE_WEEKLY_CAPPED)
        self.assertIn("1h7m0s", p["reason"])           # the announced reset window
        self.assertIn(self.ACTIVE_UNTIL, p["reason"])  # the cooldown deadline
        self.assertTrue(p["weekly_cap"]["capped"])

    def test_evaluate_spawn_ok_when_no_cooldown_active(self) -> None:
        mod = load()
        patch_checks(mod)  # default weekly = {"capped": False}
        p = run_eval(mod)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], mod.OK_VERDICT)
        self.assertFalse(p["weekly_cap"]["capped"])

    def test_weekly_cap_is_distinct_from_stale_credential(self) -> None:
        # The issue's core distinction: a stale/blocked credential is REFUSE_NO_ACCOUNT
        # (no seat routed at all), while a VALID but quota-capped seat is
        # REFUSE_WEEKLY_CAPPED (routed, but cooling). A no-account state must NEVER be
        # mislabeled as a weekly cap, and vice versa.
        mod = load()
        patch_checks(mod, account={"available": False, "tag": None, "tier": None,
                                   "reason": "all throttled", "blocked": ["worker-a"]},
                     weekly=self._capped())
        stale = run_eval(mod)
        self.assertEqual(stale["verdict"], mod.REFUSE_NO_ACCOUNT)

        patch_checks(mod, weekly=self._capped())  # valid seat, but weekly-capped
        capped = run_eval(mod)
        self.assertEqual(capped["verdict"], mod.REFUSE_WEEKLY_CAPPED)

    def test_render_names_cooldown_reason_and_retry_window(self) -> None:
        mod = load()
        patch_checks(mod, weekly=self._capped())
        text = mod.render(run_eval(mod))
        self.assertIn("REFUSE_WEEKLY_CAPPED", text)
        self.assertIn("weekly-cap:", text)
        self.assertIn(self.ACTIVE_UNTIL, text)  # retry window named on the status line
        self.assertIn("1h7m0s", text)


class FakBinProvenanceTest(unittest.TestCase):
    """Which `fak` build made this decision?

    Three resolvers pick a binary on one dispatch tick under three different rules
    (`<root>/fak.exe` for the preflight/contract gates, `<ws>/tools/.bin/fak.exe` for
    the worker's guard front, bare PATH for the lease gate), so they agree only by
    accident. These tests pin the MEASUREMENT, not a resolution order: nothing here
    may change which binary any resolver picks."""

    CLEAN = "0.43.0\nbuild: abc123def456\ngo: go1.26.5  windows/amd64\n"
    DIRTY = "0.43.0\nbuild: 8af92fbdc366 +uncommitted  (committed 2026-08-07T07:37:02Z)\n"

    def setUp(self) -> None:
        self.mod = load()

    def test_repository_relation_uses_git_ancestry_not_revision_age(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True,
                           creationflags=no_window_creationflags())
            subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=root, check=True,
                           creationflags=no_window_creationflags())
            subprocess.run(["git", "config", "user.name", "Fixture"], cwd=root, check=True,
                           creationflags=no_window_creationflags())
            (root / "row").write_text("one", encoding="utf-8")
            subprocess.run(["git", "add", "row"], cwd=root, check=True,
                           creationflags=no_window_creationflags())
            subprocess.run(["git", "commit", "-qm", "one"], cwd=root, check=True,
                           creationflags=no_window_creationflags())
            old = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True,
                creationflags=no_window_creationflags()).strip()
            (root / "row").write_text("two", encoding="utf-8")
            subprocess.run(["git", "commit", "-qam", "two"], cwd=root, check=True,
                           creationflags=no_window_creationflags())
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True,
                creationflags=no_window_creationflags()).strip()

            self.assertEqual(self.mod.repository_build_relation(root, old),
                             {"expected_head": head, "observed_build": old,
                              "relation": "BEHIND"})
            self.assertEqual(self.mod.repository_build_relation(root, head)["relation"], "MATCH")
            self.assertEqual(self.mod.repository_build_relation(root, "f" * 40)["relation"], "UNKNOWN")

    def test_stale_agreeing_build_refuses_before_capacity_admission(self) -> None:
        stale = {"schema": self.mod.FAK_BIN_PROVENANCE_SCHEMA,
                 "resolvers": {
                     "preflight_gate": {"path": "/x/fak", "resolved": True, "build": "111111111111", "dirty": False},
                     "worker_guard": {"path": "/y/fak", "resolved": True, "build": "111111111111", "dirty": False}},
                 "distinct_builds": 1, "builds": ["111111111111"], "agree": True,
                 "dirty": [], "unresolved": [], "resolved_count": 2,
                 "expected_head": "222222222222", "observed_build": "111111111111",
                 "repository_relation": "BEHIND", "historical_override": False}
        patch_checks(self.mod, fak_bin=stale)
        payload = run_eval(self.mod)
        self.assertEqual(payload["verdict"], self.mod.REFUSE_FAK_BIN_STALE)
        self.assertFalse(payload["ok"])
        self.assertIn("111111111111", payload["reason"])
        self.assertIn("222222222222", payload["reason"])
        self.assertIn("fak self-update --force --root .", payload["reason"])

    def test_build_identity_reads_the_build_id_and_the_uncommitted_flag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "fak.exe"
            exe.write_bytes(b"x" * 7)
            clean = self.mod.fak_build_identity(exe, probe=lambda p: (self.CLEAN, None))
            self.assertEqual(clean["build"], "abc123def456")
            self.assertFalse(clean["dirty"])
            self.assertEqual(clean["size"], 7)
            # build_key is the SAME <size>-<mtime_ns>-<basename> shape as
            # issue_resolve_dispatch.guard_build_id, so a provenance row joins
            # against the guard-probe inventory already in .dispatch-runs.
            self.assertTrue(clean["build_key"].startswith("7-"))
            self.assertTrue(clean["build_key"].endswith("-fak.exe"))
            dirty = self.mod.fak_build_identity(exe, probe=lambda p: (self.DIRTY, None))
            self.assertTrue(dirty["dirty"])
            self.assertEqual(dirty["build"], "8af92fbdc366")

    def test_unrunnable_binary_reports_unknown_never_clean(self) -> None:
        """An undeterminable build stays None. Coercing it to `dirty: False` would
        report an unreviewable binary as reviewed -- the exact lie this exists to stop."""
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "fak"
            exe.write_bytes(b"")
            row = self.mod.fak_build_identity(exe, probe=lambda p: ("", "boom"))
            self.assertIsNone(row["build"])
            self.assertIsNone(row["dirty"])
            self.assertEqual(row["error"], "boom")
        missing = self.mod.fak_build_identity(None)
        self.assertFalse(missing["resolved"])

    def test_provenance_flags_disagreement_and_a_dirty_gate(self) -> None:
        mod = self.mod
        mod.fak_bin_resolutions = lambda root, env=None: {
            "preflight_gate": "/root/fak.exe",     # -> DIRTY banner
            "worker_guard": "/ws/tools/.bin/fak",  # -> clean banner
            "path": None}
        # Two distinct builds; the repo-root one is the `+uncommitted` hand-build.
        rows = {"/root/fak.exe": {"build": "8af92fbdc366", "dirty": True,
                                  "build_key": "10-1-fak.exe"},
                "/ws/tools/.bin/fak": {"build": "b225bb1ca20f", "dirty": False,
                                       "build_key": "20-1-fak"}}

        def ident(path, **kw):
            if not path:
                return {"path": None, "resolved": False}
            return {"path": str(path), "resolved": True, **rows[str(path)]}

        prov = mod.fak_bin_provenance(ROOT, identity=ident)
        self.assertFalse(prov["agree"])
        self.assertEqual(prov["distinct_builds"], 2)
        self.assertEqual(prov["dirty"], ["preflight_gate"])
        self.assertEqual(prov["unresolved"], ["path"])
        warns = mod.fak_bin_warnings(prov)
        self.assertTrue(any(w.startswith("DIRTY_FAK_BIN") for w in warns))
        self.assertTrue(any(w.startswith("FAK_BIN_DISAGREEMENT") for w in warns))

    def test_windows_case_variants_share_one_build_identity(self) -> None:
        mod = self.mod
        mod.fak_bin_resolutions = lambda root, env=None: {
            "preflight_gate": r"C:\fixture\bin\fak.exe",
            "path": r"C:\fixture\bin\fak.EXE"}

        def ident(path, **kw):
            return {"path": path, "resolved": True, "size": 7, "mtime_ns": 11,
                    "build": "abc123def456", "dirty": False,
                    "build_key": mod._fak_build_key(path, 7, 11, platform="nt")}

        prov = mod.fak_bin_provenance(ROOT, identity=ident)
        self.assertTrue(prov["agree"])
        self.assertEqual(prov["distinct_builds"], 1)
        self.assertEqual(mod.fak_bin_warnings(prov), [])

    def test_different_file_or_revision_still_disagrees(self) -> None:
        mod = self.mod
        mod.fak_bin_resolutions = lambda root, env=None: {
            "preflight_gate": r"C:\one\fak.exe", "path": r"C:\two\fak.exe"}
        rows = {
            r"C:\one\fak.exe": {"build_key": "7-11-fak.exe", "build": "rev-a"},
            r"C:\two\fak.exe": {"build_key": "8-12-fak.exe", "build": "rev-b"},
        }

        def ident(path, **kw):
            return {"path": path, "resolved": True, "dirty": False, **rows[path]}

        prov = mod.fak_bin_provenance(ROOT, identity=ident)
        self.assertFalse(prov["agree"])
        self.assertEqual(prov["distinct_builds"], 2)

        rows[r"C:\two\fak.exe"]["build_key"] = "7-11-fak.exe"
        same_metadata = mod.fak_bin_provenance(ROOT, identity=ident)
        self.assertFalse(same_metadata["agree"])
        self.assertEqual(same_metadata["distinct_builds"], 2)

    def test_one_clean_agreeing_build_is_silent(self) -> None:
        """No warning when there is nothing to warn about -- an advisory that fires
        every tick is an advisory nobody reads."""
        prov = {"resolvers": {"a": {"path": "/x", "resolved": True, "build": "b",
                                    "dirty": False, "build_key": "1-1-fak"},
                              "b": {"path": "/y", "resolved": True, "build": "b",
                                    "dirty": False, "build_key": "1-1-fak"}},
                "distinct_builds": 1, "agree": True, "dirty": []}
        self.assertEqual(self.mod.fak_bin_warnings(prov), [])

    def test_record_accumulates_ticks_per_distinct_configuration(self) -> None:
        """Keyed on the resolver->build fingerprint, so a 03:00 disagreement is still
        answerable at 09:00 and the file stays bounded by configurations, not ticks."""
        mod = self.mod
        prov = {"resolvers": {"gate": {"path": "/x", "build": "b1", "dirty": True,
                                       "build_key": "1-1-fak"}},
                "agree": True, "dirty": ["gate"], "builds": ["b1"]}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod.record_fak_bin_provenance(root, prov, now="2026-08-07T01:00:00+00:00")
            path = mod.record_fak_bin_provenance(root, prov, now="2026-08-07T02:00:00+00:00")
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(doc["schema"], mod.FAK_BIN_PROVENANCE_SCHEMA)
            row = doc["gate=1-1-fak"]
            self.assertEqual(row["ticks"], 2)
            self.assertEqual(row["first_utc"], "2026-08-07T01:00:00+00:00")
            self.assertEqual(row["last_utc"], "2026-08-07T02:00:00+00:00")
            self.assertEqual(row["dirty"], ["gate"])
            # A DIFFERENT build lands as its own key -- the old row survives.
            prov2 = {"resolvers": {"gate": {"path": "/x", "build": "b2", "dirty": False,
                                            "build_key": "2-2-fak"}},
                     "agree": True, "dirty": [], "builds": ["b2"]}
            mod.record_fak_bin_provenance(root, prov2, now="2026-08-07T03:00:00+00:00")
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(sorted(k for k in doc if k != "schema"),
                             ["gate=1-1-fak", "gate=2-2-fak"])

    def test_record_never_raises_on_an_unwritable_root(self) -> None:
        """Advisory, so it fails OPEN: provenance must never be able to fail a gate."""
        with mock.patch.object(Path, "mkdir", side_effect=OSError("nope")):
            self.assertIsNone(self.mod.record_fak_bin_provenance(ROOT, {"resolvers": {}}))

    def test_resolutions_call_the_real_resolvers_and_never_raise(self) -> None:
        """The table CALLS each resolver rather than restating its rule, so it cannot
        drift from what dispatch executes; a resolver that blows up degrades to None."""
        got = self.mod.fak_bin_resolutions(ROOT, {"PATH": "", "FAK_BIN": ""})
        self.assertEqual(sorted(got), ["path", "preflight_gate", "worker_guard"])

    def test_evaluate_carries_provenance_and_a_dirty_gate_never_changes_the_verdict(self) -> None:
        mod = self.mod
        dirty = {"schema": mod.FAK_BIN_PROVENANCE_SCHEMA,
                 "resolvers": {"preflight_gate": {"path": "/root/fak.exe", "resolved": True,
                                                  "build": "8af92fbdc366", "dirty": True,
                                                  "build_key": "1-1-fak.exe"}},
                 "distinct_builds": 3, "builds": ["a", "b", "c"], "agree": False,
                 "dirty": ["preflight_gate"], "unresolved": []}
        patch_checks(mod, fak_bin=dirty)
        payload = run_eval(mod)
        self.assertEqual(payload["verdict"], mod.OK_VERDICT)  # advisory, not a gate
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["fak_bin"]["dirty"], ["preflight_gate"])
        text = mod.render(payload)
        self.assertIn("8af92fbdc366 +uncommitted", text)
        self.assertIn("DIRTY_FAK_BIN", text)

    def test_a_provenance_probe_that_explodes_leaves_the_verdict_intact(self) -> None:
        mod = self.mod
        patch_checks(mod)

        def boom(root, env=None, **kw):
            raise RuntimeError("probe exploded")

        mod.fak_bin_provenance = boom
        payload = run_eval(mod)
        self.assertEqual(payload["verdict"], mod.OK_VERDICT)
        self.assertIn("probe exploded", payload["fak_bin"]["error"])


class _FakeCompleted:
    """Stand-in for subprocess.CompletedProcess with just the fields the probe reads."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


class PosixThreadProbeDialectTests(unittest.TestCase):
    """#5541 -- the POSIX thread probe hands `ps` the keyword `nlwp`, which only
    procps-ng knows. That is deliberately NOT treated as the defect the process guard's
    census had, and these tests are the record of why.

    Two things make this site safe where tools/proc_resource_guard.py's census was not:

      * `nlwp` is the SOLE column of the argv. A `ps` that rejects the keyword destroys
        exactly the dimension that keyword names and nothing else. The census argv
        mixed it with pid/rss/comm, so one unknown keyword took the entire scan down
        and the guard then reported a measured-clean host.
      * nothing parses out of that failure, so the probe returns None -- an ABSENT
        reading -- and host_capacity() skips the dimension instead of charging it as
        "0 threads in use", which would fabricate the whole thread budget as headroom.

    BSD `ps` has no thread-count keyword AT ALL, so there is no portable spelling to
    branch to: "unmeasured" is the correct answer on such a host. These tests pin that
    it is the answer actually produced, and they are what earns this file the `lone
    column` allowance in the `ps`-dialect gate in tools/proc_resource_guard_test.py.
    """

    # A real non-procps `ps` rejecting the invocation: empty stdout, non-zero exit,
    # the diagnostic on stderr. Witnessed verbatim from the MSYS2 `ps` shipped with
    # Git for Windows, which does not implement -o at all.
    REJECTED = _FakeCompleted(stdout="", stderr="ps: unknown option -- o\n", returncode=1)

    def test_rejected_ps_leaves_the_thread_reading_absent_not_zero(self) -> None:
        mod = load()
        with mock.patch.object(mod.subprocess, "run", return_value=self.REJECTED):
            _ram, threads = mod._ram_and_threads_posix()
        self.assertIsNone(threads)          # a named absent measurement ...
        self.assertNotEqual(threads, 0)     # ... and specifically not a measured zero

    def test_absent_thread_reading_is_skipped_not_charged_as_headroom(self) -> None:
        mod = load()
        absent = mod.host_capacity(cores=16, free_ram_mb=32000, total_threads=None)
        measured_zero = mod.host_capacity(cores=16, free_ram_mb=32000, total_threads=0)
        # Unmeasured drops the dimension; a measured zero is a real reading and keeps it.
        self.assertNotIn("threads", absent["components"])
        self.assertIn("threads", measured_zero["components"])
        self.assertIsNone(absent["total_threads"])

    def test_working_ps_still_sums_every_thread_count(self) -> None:
        # The green control: the assertions above are not achieved by calling the probe
        # broken on every host.
        mod = load()
        working = _FakeCompleted(stdout="  613\n  328\n   90\n", returncode=0)
        with mock.patch.object(mod.subprocess, "run", return_value=working):
            _ram, threads = mod._ram_and_threads_posix()
        self.assertEqual(threads, 613 + 328 + 90)

    def test_thread_probe_argv_asks_for_exactly_one_column(self) -> None:
        # The property that makes the lone procps-only keyword tolerable here. If a
        # portable column is ever added alongside it, this fails and the site has to be
        # dialect-branched like tools/proc_resource_guard.py::_ps_census_spec.
        mod = load()
        seen: list[list[str]] = []

        def record(cmd, *_a, **_k):
            seen.append(list(cmd))
            return self.REJECTED

        with mock.patch.object(mod.subprocess, "run", side_effect=record):
            mod._ram_and_threads_posix()
        self.assertEqual(len(seen), 1)
        columns = [c for c in seen[0][-1].split(",") if c]
        self.assertEqual(len(columns), 1, "argv %r asks for more than the one column "
                                          "procps-only `nlwp` can lose on its own" % (seen[0],))
        self.assertTrue(columns[0].startswith("nlwp"))


class AmbientKnobHermeticityTest(unittest.TestCase):
    """#5879: the ambient fleet-tuning knobs must not reach the modules under test.

    Every knob in AMBIENT_ENV_KNOBS is folded into a number this file asserts on, so
    an operator's exported tuning silently rewrites the expected values: on the live
    fleet host (FAK_SESSIONS_PER_ACCOUNT=6, FAK_HOST_CORES_PER_WORKER=1,
    FAK_HOST_THREADS_PER_CORE=1000) a clean trunk reported 13 failures. That is a
    check reporting on something other than the thing under test — the same class of
    defect as a stale binary — so it is pinned by the loaders and witnessed here with
    the exact hostile values measured on that host."""

    HOSTILE = {
        "FAK_MAX_WORKERS": "4",
        "FAK_HOST_CORES_PER_WORKER": "1",
        "FAK_HOST_THREADS_PER_CORE": "1000",
        "FAK_SESSIONS_PER_ACCOUNT": "6",
    }

    def test_import_time_constants_ignore_ambient_knobs(self) -> None:
        # The eager half: constants folded by _env_pos_int at import time.
        with mock.patch.dict(os.environ, self.HOSTILE):
            mod = load()
        self.assertEqual(mod.HOST_CORES_PER_WORKER, 2)
        self.assertEqual(mod.HOST_THREADS_PER_CORE, 400)
        self.assertEqual(mod.DEFAULT_MAX_WORKERS, mod._built_in_max_workers())
        # ... and therefore the derived host_cap every verdict test leans on: the roomy
        # 64-core box is worth 32 workers, not the 64 the host's knobs would claim.
        self.assertEqual(mod.host_capacity(cores=64, free_ram_mb=128_000,
                                           total_threads=1000)["host_cap"], 32)

    def test_lazily_read_seat_cap_ignores_ambient_knobs(self) -> None:
        # The lazy half: fleet_accounts._claude_session_cap re-reads the env inside the
        # seat_pool call, so clearing it only around the import would not be enough.
        with mock.patch.dict(os.environ, self.HOSTILE):
            fa = load_fleet_accounts()
            pool = fa.seat_pool([_seat_row("worker-a")], [])
        self.assertEqual(pool["total_seats"], 4)
        self.assertEqual(pool["free_seats"], 4)

    def test_an_explicit_knob_still_reaches_the_module(self) -> None:
        # Pinned is a MASK, not a hardcode: a test that means to tune a knob still can,
        # and gets its value whatever the host exports underneath.
        with mock.patch.dict(os.environ, self.HOSTILE):
            self.assertEqual(load(FAK_HOST_CORES_PER_WORKER="4").HOST_CORES_PER_WORKER, 4)
            fa = load_fleet_accounts(FAK_SESSIONS_PER_ACCOUNT="6")
            self.assertEqual(fa.seat_pool([_seat_row("worker-a")], [])["total_seats"], 6)

    def test_a_mistyped_knob_is_refused(self) -> None:
        # A knob name that is not pinned would silently do nothing (the ambient value
        # would win), so asking for one is an error rather than a no-op.
        with self.assertRaises(AssertionError):
            load(FAK_HOST_CORES_PER_WORKR="4")

    def test_whole_module_is_green_under_the_hostile_host_env(self) -> None:
        # The end-to-end witness: re-run THIS file in a child that exports exactly the
        # knobs the fleet host does. Before the pin that child was 13-red; the child
        # skips this test itself (CHILD_RUN_ENV) so the re-run cannot recurse.
        if os.environ.get(CHILD_RUN_ENV):
            self.skipTest("already the hostile-env child run")
        env = dict(os.environ, **self.HOSTILE)
        env[CHILD_RUN_ENV] = "1"
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "tools/dispatch_preflight_test.py"],
            cwd=str(ROOT), env=env, capture_output=True, text=True,
            creationflags=no_window_creationflags())
        self.assertEqual(proc.returncode, 0,
                         "hostile ambient knobs turned this module red:\n%s"
                         % (proc.stderr[-4000:],))


class KnobDriftTest(unittest.TestCase):
    """AMBIENT_ENV_KNOBS is the pin list, so it has to keep matching the tools. A knob
    added to dispatch_preflight.py (or a renamed seat-cap knob in fleet_accounts.py)
    without being pinned re-opens #5879 silently — here it fails loudly instead."""

    def test_every_env_pos_int_knob_is_pinned(self) -> None:
        src = SCRIPT.read_text(encoding="utf-8")
        found = set(re.findall(r'_env_pos_int\(\s*"([A-Z0-9_]+)"', src))
        self.assertTrue(found, "no _env_pos_int knobs found — did the reader move?")
        self.assertEqual(sorted(found - set(AMBIENT_ENV_KNOBS)), [])

    def test_the_seat_cap_knob_is_pinned(self) -> None:
        fa = load_fleet_accounts()
        self.assertIn(fa.SESSIONS_PER_ACCOUNT_ENV, AMBIENT_ENV_KNOBS)


if __name__ == "__main__":
    unittest.main()
