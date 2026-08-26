#!/usr/bin/env python3
"""Hermetic tests for tools/dispatch_status.py.

build_payload() is a pure FOLD over five already-collected sub-tool dicts
(preflight, supervisor, watchdog, backlog, closure). We feed it synthetic dicts
and assert the overall verdict, the watchdog reason line, and the backlog/closure
"na" degradation — no subprocess, no gh, no schtasks. render() is exercised on a
minimal payload to prove it does not raise.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dispatch_status.py"


def load():
    """Import `tools/dispatch_status.py` fresh.

    `DISPATCH_STATUS_SCRIPT` repoints this at another copy of the script — the hook
    that makes a fail-before/pass-after proof reproducible by anyone: dump the
    pre-fix module with `git show <sha>:tools/dispatch_status.py > /tmp/pristine.py`,
    point the env var at it, and re-run this same suite against it. Unset (the
    normal case) it loads the in-tree script, byte-identically to before.
    """
    script = Path(os.environ.get("DISPATCH_STATUS_SCRIPT") or SCRIPT)
    spec = importlib.util.spec_from_file_location("dispatch_status", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pre(verdict: str = "SPAWN_OK", *, host_safe: bool = True, cap: int = 2,
        live: int = 0, limiter: dict | None = None) -> dict:
    doc = {
        "verdict": verdict,
        "reason": f"synthetic {verdict}",
        "cap": cap,
        "live": live,
        "max_workers": cap,
        "host": {"safe": host_safe},
        "account": {"tag": "worker-a", "tier": 1, "model": "claude", "available": True},
    }
    if limiter is not None:
        doc["capacity_limiter"] = limiter
    return doc


def sup(verdict: str = "READY_TO_CANARY") -> dict:
    return {
        "verdict": verdict,
        "supervise": {"target": 3, "alive": 1},
        "plans": {"total_plans": 2, "total_units": 17},
    }


def backlog_ok() -> dict:
    return {
        "lanes": {"docs": {"issues": [1, 2, 3]}, "agent": {"issues": [4]}},
        "counts": {"open": 4, "routed": 4, "unrouted": 0},
    }


def closure_ok() -> dict:
    return {
        "closure_rate": 0.8,
        "counts": {"TRUE_RESOLVED": 8, "CLAIMED_CLOSED": 10, "OPEN_WITNESSED": 2},
    }


def cap_limiter(primary: str = "configured_max", term: str = "max_workers") -> dict:
    return {
        "primary": primary,
        "term": term,
        "raw": {
            "cap": 2,
            "live": 0,
            "headroom": 2,
            "max_workers": 2,
            "dos_target": 0,
            "host_cap": 32,
            "host_binding": "cores",
            "seat_total": 4,
            "seat_free": 4,
            "seat_leased": 0,
        },
    }


def resolver_tick(*, verdict: str = "WOULD_SPAWN", action: str = "would_spawn",
                  ready: bool | None = True, blocker: str | None = None,
                  next_action: str = "", live: bool = False,
                  seat_adaptive: dict | None = None,
                  spawn_failed_streak: int | None = None,
                  cause: str | None = None) -> dict:
    gate = {}
    if ready is not None:
        gate = {
            "ready": ready,
            "blockers": ([] if ready else [{
                "code": blocker or verdict,
                "reason": "synthetic hold",
                "next_action": next_action or "inspect-last-resolve-tick",
            }]),
        }
    latest = {
        "backend": "claude",
        "verdict": verdict,
        "action": action,
        "ok": ready is True,
        "live": live,
        "max_workers": 4,
        "work_kind": "engineering",
        "force": False,
        "lane": "docs",
        "target_issue": 2042,
        "reason": "synthetic resolver tick",
        "launch_gate": gate,
        "next_action": next_action,
        "age_min": 1.0,
        "fresh": True,
        "seat_adaptive": seat_adaptive or {},
        "spawn_failed_streak": spawn_failed_streak,
        "cause": cause,
    }
    if ready is True and action == "would_spawn":
        latest["live_command"] = [
            "python", "tools\\issue_resolve_dispatch.py", "--backend", "claude",
            "--max-workers", "4", "--work-kind", "engineering",
            "--lane", "docs", "--issue", "2042", "--live", "--json",
        ]
        latest["live_command_text"] = " ".join(latest["live_command"])
    elif ready is True and action == "would_repair":
        latest["contract_scan"] = 8
        latest["repair_batch"] = 5
        latest["live_command"] = [
            "python", "tools\\issue_resolve_dispatch.py", "--backend", "claude",
            "--max-workers", "4", "--work-kind", "engineering",
            "--contract-scan", "8", "--repair-batch", "5", "--live", "--json",
        ]
        latest["live_command_text"] = " ".join(latest["live_command"])
    return {
        "schema": "fleet-resolve-ticks/1",
        "fresh_min": 90.0,
        "count": 1,
        "fresh_count": 1,
        "latest": latest,
        "selected": latest,
        "ticks": [latest],
        "errors": [],
    }


def resolver_preflight(*, backend: str = "opencode",
                       verdict: str = "REFUSE_NO_SEAT") -> dict:
    return {
        "schema": "fleet-dispatch-preflight/1",
        "_backend": backend,
        "verdict": verdict,
        "cap": 2,
        "live": 2,
        "headroom": 0,
        "seat": {
            "total": 2,
            "free": 0,
            "leased": 2,
            "depleted": True,
            "unattributed_live": 1,
        },
        "os_worker_procs": 2,
    }


def spawn_causes(*, spawns: int = 20, stale_cred: int = 0, child_crash: int = 0,
                 lookback_min: int = 24 * 60) -> dict:
    """A synthetic spawn_failed_cause_breakdown (#4590) with a controllable
    stale_cred count so a test can force (or stay under) the drain threshold."""
    failed = stale_cred + child_crash
    by_cause: dict = {}
    for cause, n in (("stale_cred", stale_cred), ("child_crash", child_crash)):
        by_cause[cause] = {
            "count": n,
            "rate_of_failed": round(n / failed, 3) if failed else 0.0,
            "rate_of_spawns": round(n / spawns, 4) if spawns else 0.0,
            "evidence": [],
        }
    return {
        "schema": "fak.spawn-failed-cause-breakdown.v1",
        "lookback_min": lookback_min,
        "spawns": spawns,
        "spawn_failed": failed,
        "rate": round(failed / spawns, 4) if spawns else 0.0,
        "by_cause": by_cause,
        "events": [],
    }


def build(mod, **over):
    kw = dict(
        root=ROOT, pre=pre(), sup=sup(), wd={"installed": True, "status": "Ready"},
        backlog=backlog_ok(), closure=closure_ok(), max_workers=2, fast=False)
    kw.update(over)
    return mod.build_payload(**kw)


class VerdictTest(unittest.TestCase):
    def test_dispatcher_block_helper_pins_payload_schema_and_order(self) -> None:
        mod = load()
        preflight = pre("SPAWN_OK", cap=4, live=1)
        watchdog = {"installed": True, "status": "Ready"}
        limiter = mod._dispatch_limiter(preflight, backlog_ok(), closure_ok(), {})
        expected = {
            "cap": 4,
            "live": 1,
            "headroom": 3,
            "host_safe": True,
            "preflight_verdict": "SPAWN_OK",
            "limiter": limiter,
            "account": {
                "tag": "worker-a",
                "tier": 1,
                "model": "claude",
                "available": True,
            },
            "watchdog": watchdog,
        }

        self.assertEqual(
            mod._dispatcher_block(
                cap=4, live=1, host_safe=True, pre_verdict="SPAWN_OK",
                limiter=limiter, acct=preflight["account"], wd=watchdog),
            expected,
        )
        self.assertEqual(build(mod, pre=preflight, wd=watchdog)["dispatcher"], expected)

    def test_ready_to_grow_when_safe_to_spawn(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"))
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "READY_TO_GROW")
        self.assertEqual(p["dispatcher"]["headroom"], 2)

    def test_run_liveness_does_not_overwrite_spawn_gate_verdict(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"), run_status=[
            {"run_id": "RID-ACTIVE1", "liveness": {"verdict": "STALLED"}},
        ])
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "READY_TO_GROW")
        self.assertEqual(p["run_status"]["liveness"], {"STALLED": 1})

    def test_host_flagged_fails_the_card(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK", host_safe=False))
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "HOST_FLAGGED")
        self.assertTrue(any("host resource guard flagged" in r for r in p["reasons"]))

    def test_at_cap_is_a_healthy_steady_state(self) -> None:
        mod = load()
        p = build(mod, pre=pre("REFUSE_AT_CAP", cap=2, live=2))
        self.assertTrue(p["ok"])  # at cap is normal, not breakage
        self.assertEqual(p["verdict"], "AT_CAP")
        self.assertEqual(p["dispatcher"]["headroom"], 0)

    def test_blocked_on_account_is_a_healthy_steady_state(self) -> None:
        mod = load()
        p = build(mod, pre=pre("REFUSE_NO_ACCOUNT"))
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "BLOCKED_ON_ACCOUNT")

    def test_blocked_on_seat_is_a_healthy_steady_state(self) -> None:
        mod = load()
        p = build(mod, pre=pre("REFUSE_NO_SEAT"))
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "BLOCKED_ON_SEAT")
        self.assertTrue(any("seat free" in r for r in p["reasons"]))
        self.assertFalse(any("safe to spawn" in r for r in p["reasons"]))

    def test_inspect_fails_the_card(self) -> None:
        mod = load()
        p = build(mod, pre=pre("REFUSE_INSPECT"))
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "INSPECT")


class SpawnCauseCardTest(unittest.TestCase):
    """#4590: the trailing spawn-failed cause mix folds onto the DEFAULT card, and a
    stale_cred rate over threshold reddens the verdict — no --spawn-causes needed."""

    def test_default_card_folds_the_spawn_failed_mix(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"),
                  spawn_causes=spawn_causes(spawns=20, child_crash=2))
        # Healthy: 2 child_crash / 20 spawns, zero stale_cred → no drain flip.
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "READY_TO_GROW")
        self.assertFalse(p["spawn_causes"]["na"])
        self.assertFalse(p["spawn_causes"]["stale_cred_alarm"]["red"])
        self.assertTrue(any("spawn-failed mix" in r for r in p["reasons"]))
        self.assertIn("spawn-fail:", mod.render(p))

    def test_high_stale_cred_rate_reddens_the_verdict(self) -> None:
        mod = load()
        # 4 stale_cred / 20 spawns = 20% >= 10% threshold, 20 spawns >= floor.
        p = build(mod, pre=pre("SPAWN_OK"),
                  spawn_causes=spawn_causes(spawns=20, stale_cred=4))
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "SPAWN_STALE_CRED_DRAIN")
        alarm = p["spawn_causes"]["stale_cred_alarm"]
        self.assertTrue(alarm["red"])
        self.assertEqual(alarm["count"], 4)
        self.assertTrue(any("stale-cred spawn-failure drain" in r for r in p["reasons"]))
        self.assertIn("STALE_CRED_DRAIN", mod.render(p))

    def test_small_sample_stale_cred_does_not_trip_the_alarm(self) -> None:
        mod = load()
        # 1 stale_cred / 4 spawns = 25% rate but only 4 spawns (< min floor 8).
        p = build(mod, pre=pre("SPAWN_OK"),
                  spawn_causes=spawn_causes(spawns=4, stale_cred=1))
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "READY_TO_GROW")
        self.assertFalse(p["spawn_causes"]["stale_cred_alarm"]["red"])

    def test_stale_cred_drain_does_not_stomp_a_flagged_host(self) -> None:
        mod = load()
        # An already-failing verdict (host flagged) stays the more urgent one; the
        # drain still adds its reason line so the operator sees both.
        p = build(mod, pre=pre("SPAWN_OK", host_safe=False),
                  spawn_causes=spawn_causes(spawns=20, stale_cred=4))
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "HOST_FLAGGED")
        self.assertTrue(any("stale-cred spawn-failure drain" in r for r in p["reasons"]))

    def test_missing_spawn_causes_folds_to_na_without_reddening(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"))
        self.assertTrue(p["spawn_causes"]["na"])
        self.assertFalse(p["spawn_causes"]["stale_cred_alarm"]["red"])
        self.assertTrue(p["ok"])


def seat_streak_rows(*rows: tuple[str, int]) -> list[dict]:
    """Synthetic read_seat_spawn_failure_streaks rows (#4591)."""
    return [{"seat": tag, "backend": "claude", "streak": n, "last_ts": 1_000_000.0}
            for tag, n in rows]


def seat_inventory_with(*, auth_failed: tuple[str, ...] = ()) -> dict:
    seats = [{"tag": t, "dispatch_state": "unavailable", "hold_reason": "auth_failed"}
             for t in auth_failed]
    return {"schema": "seat-pool/1", "total_seats": max(1, len(seats)),
            "by_dispatch_state": {"available": 0, "busy": 0, "cooling": 0,
                                  "unavailable": len(seats)},
            "seats": seats}


class SeatStreakCardTest(unittest.TestCase):
    """#4591: ONE dead seat with a spawn-fail run-length at threshold flips the
    card verdict with a NAMED seat + operator action — the per-seat streak the
    aggregate #4590 rate alarm can stay under in a big pool."""

    def test_seat_at_threshold_flips_the_verdict_with_named_seat(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"),
                  seat_streaks=seat_streak_rows(("july17", 3)),
                  seat_inventory=seat_inventory_with(auth_failed=("july17",)),
                  spawn_causes=spawn_causes(spawns=4, stale_cred=1))
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "SEAT_SPAWN_FAIL_STREAK")
        alarm = p["seat_streaks"]["alarm"]
        self.assertTrue(alarm["red"])
        self.assertTrue(alarm["seats"][0]["auth_failed"])
        # The named, actionable reason the issue asks for — seat + count + cause
        # + the operator command.
        self.assertTrue(any(
            "seat july17: 3 spawn-fails in a row (stale_cred) -> fak accounts status" in r
            for r in p["reasons"]))

    def test_seat_under_threshold_never_flips(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"),
                  seat_streaks=seat_streak_rows(("july17", 2)),
                  seat_inventory=seat_inventory_with(auth_failed=("july17",)))
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "READY_TO_GROW")
        self.assertFalse(p["seat_streaks"]["alarm"]["red"])

    def test_unjoined_seat_still_flips_without_cause_qualifier(self) -> None:
        # The streak alone is actionable evidence even when the inventory/cause
        # join can't confirm stale_cred — the reason just omits the qualifier.
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"),
                  seat_streaks=seat_streak_rows(("gem5", 4)))
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "SEAT_SPAWN_FAIL_STREAK")
        self.assertTrue(any(
            "seat gem5: 4 spawn-fails in a row -> fak accounts status" in r
            for r in p["reasons"]))

    def test_seat_streak_does_not_stomp_a_flagged_host(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK", host_safe=False),
                  seat_streaks=seat_streak_rows(("july17", 3)))
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "HOST_FLAGGED")
        self.assertTrue(any("seat july17" in r for r in p["reasons"]))

    def test_read_seat_spawn_failure_streaks_reads_the_ledgers(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            (runs / "spawn-failure-streak-seat-claude.json").write_text(json.dumps({
                "july17": {"count": 3, "last_ts": 1_000_000.0},
                "ok-seat": {"count": 1, "last_ts": 1_000_100.0},
            }), encoding="utf-8")
            (runs / "spawn-failure-streak-seat-opencode.json").write_text(
                json.dumps({"glm-a": 2}), encoding="utf-8")  # bare-int tolerated
            (runs / "spawn-failure-streak-claude.json").write_text(
                json.dumps({"1515": 9}), encoding="utf-8")  # TARGET ledger: ignored
            rows = mod.read_seat_spawn_failure_streaks(runs)
        self.assertEqual([(r["seat"], r["backend"], r["streak"]) for r in rows],
                         [("july17", "claude", 3), ("glm-a", "opencode", 2),
                          ("ok-seat", "claude", 1)])


class NetDeclineCardTest(unittest.TestCase):
    """#4591 part 2: `live` declining M consecutive fleet-status-history appends
    raises the net-worker-decline alarm on the card."""

    def test_red_decline_flips_the_verdict(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"),
                  fleet_decline={"red": True, "key": "live", "declines": 3,
                                 "reason": "net worker decline: 'live' fell 3 "
                                           "consecutive ledger appends (5→4→3→2)"})
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "NET_WORKER_DECLINE")
        self.assertTrue(any("net worker decline" in r for r in p["reasons"]))

    def test_absent_or_errored_ledger_never_flips(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"), fleet_decline={"_error": "no ledger"})
        self.assertTrue(p["ok"])
        p = build(mod, pre=pre("SPAWN_OK"),
                  fleet_decline={"red": False, "declines": 1, "reason": ""})
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "READY_TO_GROW")


class SeatSelectionCardTest(unittest.TestCase):
    def test_read_and_render_surface_seat_selection_summary(self) -> None:
        mod = load()
        selection = {
            "winner_tag": "dead",
            "winner_reason": "chosen despite auth_failed",
            "summary": "picked dead over 8 (chosen despite auth_failed)",
            "candidates": [{"rank": 1, "tag": "dead", "skip_reason": ""}],
        }
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td) / mod.RUNS_DIRNAME
            runs.mkdir(parents=True)
            (runs / "last-resolve-tick-claude.json").write_text(json.dumps({
                "backend": "claude", "verdict": "WOULD_SPAWN", "lane": "docs",
                "seat_selection": selection,
            }), encoding="utf-8")
            ticks = mod.read_resolve_ticks(Path(td))
        self.assertEqual(ticks["latest"]["seat_selection"]["winner_reason"], "chosen despite auth_failed")
        card = mod.render(build(mod, resolve_ticks=ticks))
        self.assertIn("picked dead over 8 (chosen despite auth_failed)", card)

class SeatAdaptiveTickCardTest(unittest.TestCase):
    """#4589: read_resolve_ticks stops dropping seat_adaptive / spawn_failed_streak /
    cause, and render()/render_md() surface the ramp cap + spawn-fail streak."""

    def test_read_resolve_ticks_keeps_seat_adaptive_streak_and_cause(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td) / mod.RUNS_DIRNAME
            runs.mkdir(parents=True)
            (runs / "last-resolve-tick-claude.json").write_text(json.dumps({
                "backend": "claude", "verdict": "WOULD_SPAWN", "lane": "docs",
                "seat_adaptive": {"signal_available": True, "binding": "ramp_delta",
                                  "ramp_delta": 2, "effective_target": 6},
                "spawn_failed_streak": 3, "cause": "stale_cred",
            }), encoding="utf-8")
            out = mod.read_resolve_ticks(Path(td))
        row = out["latest"]
        self.assertEqual(row["seat_adaptive"]["binding"], "ramp_delta")
        self.assertEqual(row["spawn_failed_streak"], 3)
        self.assertEqual(row["cause"], "stale_cred")

    def test_render_surfaces_ramp_cap_and_spawn_fail_streak(self) -> None:
        mod = load()
        ticks = resolver_tick(
            seat_adaptive={"signal_available": True, "binding": "ramp_delta",
                           "ramp_delta": 2, "effective_target": 6,
                           "live": 4, "seat_free": 2, "hard_ceiling": 8},
            spawn_failed_streak=3, cause="stale_cred")
        p = build(mod, resolve_ticks=ticks)
        card = mod.render(p)
        self.assertIn("seat cap", card)
        self.assertIn("bound by ramp_delta", card)
        self.assertIn("ramp +2/tick", card)
        self.assertIn("streak 3", card)
        self.assertIn("cause=stale_cred", card)

    def test_render_md_surfaces_ramp_cap_and_streak(self) -> None:
        mod = load()
        ticks = resolver_tick(
            seat_adaptive={"signal_available": True, "binding": "ramp_delta",
                           "ramp_delta": 2, "effective_target": 6,
                           "live": 4, "seat_free": 2, "hard_ceiling": 8},
            spawn_failed_streak=3, cause="stale_cred")
        p = build(mod, resolve_ticks=ticks)
        md = mod.render_md(p, date="2026-07-14")
        self.assertIn("seat cap", md)
        self.assertIn("spawn-fail streak", md)

    def test_no_seat_adaptive_signal_emits_no_ramp_line(self) -> None:
        mod = load()
        # A tick with no adaptive signal and zero streak → neither line renders.
        p = build(mod, resolve_ticks=resolver_tick())
        card = mod.render(p)
        self.assertNotIn("seat cap", card)
        self.assertNotIn("spawn-fail: streak", card)


class LimiterStatusTest(unittest.TestCase):
    """#1803: dispatch status displays one primary limiter plus raw compute terms."""

    def test_preflight_limiter_flows_into_payload_and_render(self) -> None:
        mod = load()
        limiter = cap_limiter("memory", "host_cap")
        limiter["raw"]["host_cap"] = 2
        limiter["raw"]["host_binding"] = "ram"
        p = build(mod, pre=pre("SPAWN_OK", limiter=limiter))
        self.assertEqual(p["dispatcher"]["limiter"]["primary"], "memory")
        self.assertEqual(p["dispatcher"]["limiter"]["raw"]["host_cap"], 2)

        text = mod.render(p)
        self.assertIn("limiter   : memory", text)
        self.assertIn("host_cap=2", text)
        self.assertIn("host_binding=ram", text)

    def test_github_rate_limit_is_primary_status_limiter(self) -> None:
        mod = load()
        p = build(
            mod,
            pre=pre("SPAWN_OK", limiter=cap_limiter("configured_max")),
            backlog={"_error": "gh: API rate limit exceeded"},
        )
        self.assertEqual(p["dispatcher"]["limiter"]["primary"], "github_rate_limit")
        self.assertIn("API rate limit", p["dispatcher"]["limiter"]["raw"]["github_error"])
        self.assertIn("github_error=rate_limit", mod.render(p))
        self.assertIn("(ACTION)", mod.slack_text(p))
        self.assertIn("GitHub rate limit is blocking", mod.slack_text(p))

    def test_blocking_lane_lease_overrides_preflight_limiter(self) -> None:
        mod = load()
        leases = {
            "active_count": 1,
            "blocking_count": 1,
            "active": [{"id": "L1", "lane": "tools", "blocks_candidate": True,
                        "blocking_candidates": [{"issue": 1803}]}],
        }
        p = build(mod, pre=pre("SPAWN_OK", limiter=cap_limiter("configured_max")),
                  leases=leases)
        self.assertEqual(p["dispatcher"]["limiter"]["primary"], "leases")
        self.assertEqual(p["dispatcher"]["limiter"]["term"], "lane_leases_blocking")
        self.assertEqual(p["dispatcher"]["limiter"]["raw"]["lane_leases_blocking"], 1)


class BacklogClosureNaTest(unittest.TestCase):
    def test_backlog_na_on_skipped_and_closure_na_on_skipped(self) -> None:
        mod = load()
        p = build(mod, backlog={"_skipped": "fast"}, closure={"_skipped": "fast"}, fast=True)
        self.assertTrue(p["backlog"]["na"])
        self.assertIsNone(p["backlog"]["open_issues"])
        self.assertTrue(p["closure"]["na"])
        self.assertIsNone(p["closure"]["closure_rate"])

    def test_backlog_na_on_error_with_no_lanes(self) -> None:
        mod = load()
        p = build(mod, backlog={"_error": "gh timed out"})
        self.assertTrue(p["backlog"]["na"])

    def test_backlog_present_folds_lane_counts(self) -> None:
        mod = load()
        p = build(mod)
        self.assertFalse(p["backlog"]["na"])
        self.assertEqual(p["backlog"]["open_issues"], 4)
        self.assertEqual(p["backlog"]["by_lane"], {"docs": 3, "agent": 1})
        self.assertEqual(p["backlog"]["routed"], 4)

    def test_closure_present_surfaces_rate_and_open_witnessed(self) -> None:
        mod = load()
        p = build(mod)
        self.assertFalse(p["closure"]["na"])
        self.assertEqual(p["closure"]["closure_rate"], 0.8)
        self.assertEqual(p["closure"]["open_witnessed_closable"], 2)


class WatchdogReasonTest(unittest.TestCase):
    def test_watchdog_installed_reason_line(self) -> None:
        mod = load()
        p = build(mod, wd={"installed": True, "status": "Ready"})
        self.assertTrue(any("watchdog installed (Ready)" in r for r in p["reasons"]))
        self.assertEqual(p["dispatcher"]["watchdog"]["installed"], True)

    def test_watchdog_not_installed_reason_line(self) -> None:
        mod = load()
        p = build(mod, wd={"installed": False, "status": None})
        self.assertTrue(any("watchdog NOT installed" in r for r in p["reasons"]))

    def test_watchdog_unknown_emits_no_install_line(self) -> None:
        mod = load()
        # installed is None (schtasks couldn't run) -> neither install line appears.
        p = build(mod, wd={"installed": None, "error": "schtasks missing"})
        self.assertFalse(any("watchdog" in r for r in p["reasons"]))


class RunStatusDigestTest(unittest.TestCase):
    def test_loop_ledger_run_ids_are_recent_active_started_rids_only(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            ledger = Path(d) / "loops.jsonl"
            now_ns = 10_000_000_000_000
            recent_ns = now_ns - 30 * 60 * 1_000_000_000
            stale_ns = now_ns - 3 * 60 * 60 * 1_000_000_000
            ledger.write_text("\n".join([
                '{"loop_id":"issue-resolve-dispatch/claude","run_id":"legacy-1"}',
                '{"loop_id":"other","run_id":"RID-OTHER1"}',
                '{"loop_id":"issue-resolve-progress","run_id":"RID-PROGRESS1","kind":"fire"}',
                '{"loop_id":"issue-resolve-progress","run_id":"RID-PROGRESS1","kind":"end","status":"claimed_done"}',
                '{"loop_id":"issue-resolve-dispatch/codex","run_id":"RID-DRYRUN1","kind":"fire"}',
                '{"loop_id":"issue-resolve-dispatch/codex","run_id":"RID-DRYRUN1","kind":"admit","status":"admitted"}',
                '{"loop_id":"issue-resolve-dispatch/codex","run_id":"RID-DRYRUN1","kind":"end","status":"claimed_done"}',
                '{"loop_id":"issue-resolve-dispatch/claude","run_id":"RID-REFUSED1","kind":"fire"}',
                '{"loop_id":"issue-resolve-dispatch/claude","run_id":"RID-REFUSED1","kind":"admit","status":"refused"}',
                '{"loop_id":"issue-resolve-dispatch/opencode","run_id":"RID-ACTIVE1","kind":"fire"}',
                f'{{"loop_id":"issue-resolve-dispatch/opencode","run_id":"RID-ACTIVE1","kind":"start","status":"running","ts_unix_nano":{recent_ns}}}',
                f'{{"loop_id":"issue-resolve-dispatch/claude","run_id":"RID-STALE1","kind":"start","status":"running","ts_unix_nano":{stale_ns}}}',
                f'{{"loop_id":"issue-resolve-dispatch/claude","run_id":"RID-ACTIVE2","kind":"start","status":"running","ts_unix_nano":{recent_ns + 1}}}',
            ]) + "\n", encoding="utf-8")
            self.assertEqual(
                mod.run_ids_from_loop_ledger(ledger, lookback_min=60, now_ns=now_ns),
                ["RID-ACTIVE2", "RID-ACTIVE1"])

    def test_claimed_key_detector_is_recursive(self) -> None:
        mod = load()
        self.assertTrue(mod.has_key_named({"liveness": [{"claimed": "done"}]}, "claimed"))
        self.assertFalse(mod.has_key_named({"liveness": [{"verdict": "STALLED"}]}, "claimed"))

    def test_build_payload_summarizes_dos_status_digests(self) -> None:
        mod = load()
        p = build(mod, run_status=[
            {"run_id": "RID-DISPATCH1", "liveness": {"verdict": "ADVANCING"}},
            {"run_id": "RID-PROGRESS1", "_error": "dos unavailable"},
        ])
        self.assertEqual(p["run_status"]["source"], "dos status")
        self.assertEqual(p["run_status"]["count"], 2)
        self.assertEqual(p["run_status"]["liveness"], {"ADVANCING": 1})
        self.assertEqual(p["run_status"]["errors"], 1)
        self.assertTrue(any("dos status digest" in r for r in p["reasons"]))
        self.assertIn("run truth", mod.render(p))


class LabReadinessTest(unittest.TestCase):
    def test_missing_lab_readiness_fails_closed(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            rec = mod.read_lab_readiness(Path(d) / "missing.json")
        self.assertEqual(rec["schema"], mod.LINK_STATE_SCHEMA)
        self.assertEqual(rec["phase"], "WAITING")
        self.assertEqual(rec["detail"], "indeterminate")
        self.assertFalse(rec["admit_dispatch"])
        self.assertEqual(rec["next_action"], "publish-lab-readiness")
        self.assertEqual(rec["evidence"], "no-readiness-record")
        self.assertFalse(rec["present"])
        self.assertIn("--write-default", rec["commands"]["mark_clear"])

    def test_native_record_derives_admit_bit_and_rejects_private_fields(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "lab-readiness.json"
            # A native fak.link_state/v1 record with a LYING admit_dispatch=False on a
            # CLEAR phase: the reader must re-derive the bit from the phase, not trust it.
            path.write_text(json.dumps({
                "schema": mod.LINK_STATE_SCHEMA,
                "subject": "gpu-server",
                "checked_at": "2026-07-04T14:00:00Z",
                "phase": "CLEAR",
                "detail": "ready",
                "next_action": "admit-dispatch",
                "evidence": "scrubbed-private-readback",
                "admit_dispatch": False,
            }), encoding="utf-8")
            rec = mod.read_lab_readiness(path)
            self.assertEqual(rec["phase"], "CLEAR")
            self.assertTrue(rec["admit_dispatch"])

            path.write_text(json.dumps({
                "schema": mod.LINK_STATE_SCHEMA,
                "subject": "gpu-server",
                "phase": "CLEAR",
                "detail": "ready",
                "next_action": "admit-dispatch",
                "evidence": "scrubbed-private-readback",
                "raw_thread": "secret",
            }), encoding="utf-8")
            bad = mod.read_lab_readiness(path)
            self.assertEqual(bad["phase"], "WAITING")
            self.assertEqual(bad["evidence"], "invalid-readiness-record")
            self.assertIn("unknown field", bad["_error"])

    def test_legacy_record_read_shim_coarsens_onto_phase(self) -> None:
        # Rollover safety: a legacy fak.lab_readiness/v1 record (the private bridge may
        # keep emitting it for one cycle) is still accepted and coarsened onto a phase.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "lab-readiness.json"
            path.write_text(json.dumps({
                "schema": mod.LAB_READINESS_SCHEMA,
                "machine_class": "gpu-server",
                "checked_at": "2026-07-04T14:00:00Z",
                "status": "READY_FOR_DEV_WORK",
                "next_action": "admit-lab-backed-dispatch",
                "evidence": "scrubbed-private-readback",
                "admit_lab_dispatch": True,
            }), encoding="utf-8")
            ready = mod.read_lab_readiness(path)
            self.assertEqual(ready["schema"], mod.LINK_STATE_SCHEMA)
            self.assertEqual(ready["phase"], "CLEAR")
            self.assertEqual(ready["detail"], "ready")
            self.assertTrue(ready["admit_dispatch"])

            # A legacy not-ready status coarsens to WAITING with the mapped detail, and
            # a lying admit_lab_dispatch=True can never re-open the gate.
            path.write_text(json.dumps({
                "schema": mod.LAB_READINESS_SCHEMA,
                "machine_class": "gpu-server",
                "status": "WAIT_PRIVATE_RECOVERY",
                "next_action": "confirm-private-control-session",
                "evidence": "scrubbed-private-readback",
                "admit_lab_dispatch": True,
            }), encoding="utf-8")
            wait = mod.read_lab_readiness(path)
            self.assertEqual(wait["phase"], "WAITING")
            self.assertEqual(wait["detail"], "private-recovery")
            self.assertFalse(wait["admit_dispatch"])

    def test_lab_hold_reaches_payload_render_and_slack_action(self) -> None:
        mod = load()
        lab = {
            "schema": mod.LINK_STATE_SCHEMA,
            "subject": "gpu-server",
            "checked_at": "2026-07-04T14:00:00Z",
            "phase": "WAITING",
            "detail": "private-recovery",
            "next_action": "confirm-private-control-session",
            "evidence": "scrubbed-private-readback",
            "admit_dispatch": False,
            "commands": mod._lab_readiness_commands(),
        }
        p = build(mod, lab_readiness=lab)
        self.assertEqual(p["lab_readiness"]["phase"], "WAITING")
        self.assertTrue(any("lab readiness: WAITING/private-recovery" in r for r in p["reasons"]))
        self.assertIn("lab       : WAITING/private-recovery", mod.render(p))
        self.assertIn("lab cmd", mod.render(p))
        slack = mod.slack_text(p)
        self.assertIn("*dispatch scheduler:* `READY_TO_GROW` (ACTION)", slack)
        self.assertIn("lab readiness WAITING/private-recovery", slack)
        self.assertIn("lab-backed dispatch held", slack)
        self.assertIn("fak lab readiness --phase CLEAR --write-default --json", slack)

    def test_ready_lab_record_is_expected_not_action(self) -> None:
        mod = load()
        lab = {
            "schema": mod.LINK_STATE_SCHEMA,
            "subject": "gpu-server",
            "checked_at": "2026-07-04T14:00:00Z",
            "phase": "CLEAR",
            "detail": "ready",
            "next_action": "admit-dispatch",
            "evidence": "scrubbed-private-readback",
            "admit_dispatch": True,
            "commands": mod._lab_readiness_commands(),
        }
        p = build(mod, lab_readiness=lab)
        slack = mod.slack_text(p)
        self.assertIn("lab readiness CLEAR", slack)
        self.assertNotIn("lab-backed dispatch held", slack)


class ResolverTickTest(unittest.TestCase):
    def test_read_resolve_ticks_summarizes_latest_backend_artifact(self) -> None:
        import os
        import time
        mod = load()
        now = time.time()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / mod.RUNS_DIRNAME
            runs.mkdir()
            tick = runs / "last-resolve-tick-claude.json"
            tick.write_text(json.dumps({
                "backend": "claude",
                "verdict": "SELF_MODIFY_HOLD",
                "action": "no_lane",
                "ok": False,
                "live": False,
                "lane": None,
                "target_issue": None,
                "reason": "all self-source",
                "launch_gate": {
                    "ready": False,
                    "blockers": [{
                        "code": "SELF_MODIFY_HOLD",
                        "reason": "all self-source",
                        "next_action": "enable-worktree-isolated-resolver",
                    }],
                },
            }), encoding="utf-8")
            os.utime(tick, (now - 60, now - 60))

            out = mod.read_resolve_ticks(root, now_ts=now)

        self.assertEqual(out["count"], 1)
        self.assertEqual(out["fresh_count"], 1)
        self.assertEqual(out["latest"]["backend"], "claude")
        self.assertEqual(out["latest"]["launch_gate"]["blockers"][0]["code"],
                         "SELF_MODIFY_HOLD")
        self.assertEqual(out["latest"]["next_action"],
                         "enable-worktree-isolated-resolver")

    def test_read_resolve_ticks_selects_launch_ready_over_newer_unguarded(self) -> None:
        import os
        import time
        mod = load()
        now = time.time()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / mod.RUNS_DIRNAME
            runs.mkdir()
            claude = runs / "last-resolve-tick-claude.json"
            claude.write_text(json.dumps({
                "backend": "claude",
                "verdict": "WOULD_SPAWN",
                "action": "would_spawn",
                "ok": True,
                "live": False,
                "max_workers": 4,
                "work_kind": "engineering",
                "lane": "tools",
                "target_issue": 1672,
                "launch_gate": {"ready": True, "blockers": []},
            }), encoding="utf-8")
            opencode = runs / "last-resolve-tick-opencode.json"
            opencode.write_text(json.dumps({
                "backend": "opencode",
                "verdict": "WOULD_SPAWN",
                "action": "would_spawn",
                "ok": True,
                "live": False,
                "max_workers": 4,
                "lane": "tools",
                "target_issue": 1672,
                "launch_gate": {
                    "ready": False,
                    "blockers": [{
                        "code": "UNGUARDED_WORKER",
                        "next_action": "make-fak-guard-resolvable",
                    }],
                },
            }), encoding="utf-8")
            os.utime(claude, (now - 120, now - 120))
            os.utime(opencode, (now - 30, now - 30))

            out = mod.read_resolve_ticks(root, now_ts=now)

        self.assertEqual(out["latest"]["backend"], "opencode")
        self.assertEqual(out["selected"]["backend"], "claude")
        self.assertEqual(out["selected"]["live_command"],
                         ["python", "tools\\issue_resolve_dispatch.py", "--backend", "claude",
                          "--max-workers", "4", "--work-kind", "engineering",
                          "--lane", "tools", "--issue", "1672", "--live", "--json"])

    def test_read_resolve_ticks_prefers_repair_in_flight_over_refusal_noise(self) -> None:
        import os
        import time
        mod = load()
        now = time.time()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / mod.RUNS_DIRNAME
            runs.mkdir()
            repair = runs / "last-resolve-tick-claude.json"
            repair.write_text(json.dumps({
                "backend": "claude",
                "verdict": "REPAIR_IN_FLIGHT",
                "action": "repair_in_flight",
                "ok": True,
                "live": False,
                "target_issue": 2201,
                "lane": "docs",
            }), encoding="utf-8")
            refusal = runs / "last-resolve-tick-codex.json"
            refusal.write_text(json.dumps({
                "backend": "codex",
                "verdict": "REFUSE_AT_CAP",
                "action": "refused",
                "ok": False,
                "live": False,
            }), encoding="utf-8")
            os.utime(repair, (now - 60, now - 60))
            os.utime(refusal, (now - 10, now - 10))

            out = mod.read_resolve_ticks(root, now_ts=now)

        self.assertEqual(out["latest"]["backend"], "codex")
        self.assertEqual(out["selected"]["backend"], "claude")
        self.assertEqual(mod._resolve_tick_state(out["selected"]), "repair-in-flight")

    def test_launch_ready_tick_reaches_payload_render_and_slack_action(self) -> None:
        mod = load()
        p = build(mod, resolve_ticks=resolver_tick())
        self.assertTrue(any("selected resolver tick launch-ready" in r for r in p["reasons"]))
        self.assertEqual(p["utilization"]["state"], "HEADROOM_LAUNCH_READY")
        self.assertIn("approve-live-launch", p["utilization"]["next_actions"][0])
        self.assertIn("--live", p["utilization"]["launch_command"])
        self.assertIn("planner", mod.render(p))
        self.assertIn("launch-ready", mod.render(p))
        self.assertIn("launch    : python", mod.render(p))
        self.assertIn("use       : HEADROOM_LAUNCH_READY", mod.render(p))
        buckets = mod._dispatch_slack_buckets(p)
        self.assertTrue(any("resolver dry-run launch-ready" in r for r in buckets["action"]))
        self.assertTrue(any("--live --json" in r for r in buckets["action"]))
        self.assertTrue(any("utilization HEADROOM_LAUNCH_READY" in r
                            for r in buckets["action"]))
        self.assertEqual(mod._dispatch_headline_state(p), "ACTION")

    def test_dispatcher_preflight_refusal_suppresses_launch_ready_tick(self) -> None:
        mod = load()
        p = build(mod, pre=pre("REFUSE_NO_SEAT", cap=4, live=0),
                  resolve_ticks=resolver_tick())
        self.assertFalse(any("selected resolver tick launch-ready" in r
                             for r in p["reasons"]))
        self.assertTrue(any("selected resolver tick held by current preflight" in r
                            and "REFUSE_NO_SEAT" in r
                            for r in p["reasons"]))
        self.assertEqual(p["utilization"]["state"], "HEADROOM_HELD")
        self.assertEqual(p["utilization"]["blockers"][0]["code"], "NO_FREE_SEAT")
        rendered = mod.render(p)
        self.assertNotIn("launch    : python", rendered)
        self.assertNotIn("use       : HEADROOM_LAUNCH_READY", rendered)
        buckets = mod._dispatch_slack_buckets(p)
        self.assertFalse(any("resolver dry-run launch-ready" in r
                             for r in buckets["action"]))

    def test_resolver_product_preflight_surfaces_backend_seat_state(self) -> None:
        mod = load()
        tick = resolver_tick()
        tick["selected"]["backend"] = "opencode"
        p = build(mod, resolve_ticks=tick, resolver_preflight=resolver_preflight())
        self.assertEqual(p["resolver_preflight"]["_backend"], "opencode")
        self.assertTrue(any(
            "selected resolver preflight: opencode REFUSE_NO_SEAT" in r
            and "live=2/2" in r
            and "unattributed_live=1" in r
            for r in p["reasons"]))
        self.assertFalse(any("selected resolver tick launch-ready" in r
                             for r in p["reasons"]))
        self.assertTrue(any("selected resolver tick held by current preflight" in r
                            and "REFUSE_NO_SEAT" in r
                            for r in p["reasons"]))
        rendered = mod.render(p)
        self.assertIn("plan gate : opencode REFUSE_NO_SEAT live=2/2", rendered)
        self.assertEqual(p["utilization"]["state"], "HEADROOM_HELD")
        self.assertEqual(p["utilization"]["blockers"][0]["code"], "NO_FREE_SEAT")
        self.assertNotIn("use       : HEADROOM_LAUNCH_READY", rendered)
        self.assertNotIn("launch    : python", rendered)
        md = mod.render_md(p, date="2026-07-05")
        self.assertIn("**resolver product preflight**: opencode REFUSE_NO_SEAT", md)
        buckets = mod._dispatch_slack_buckets(p)
        self.assertTrue(any("opencode REFUSE_NO_SEAT" in r
                            for r in buckets["expected"]))
        self.assertFalse(any("resolver dry-run launch-ready" in r
                             for r in buckets["action"]))
        self.assertTrue(any("resolver launch held by current preflight REFUSE_NO_SEAT" in r
                            for r in buckets["expected"]))

    def test_selected_resolver_preflight_runs_product_scoped_gate(self) -> None:
        mod = load()
        tick = resolver_tick()
        tick["selected"]["backend"] = "opencode"
        tick["selected"]["max_workers"] = 2
        calls = []
        original = mod.run_json

        def fake_run_json(cmd, cwd, timeout, ok_codes=None):
            calls.append(cmd)
            return resolver_preflight()

        mod.run_json = fake_run_json
        try:
            out = mod.selected_resolver_preflight(ROOT, tick, max_workers=4)
        finally:
            mod.run_json = original

        self.assertEqual(out["_backend"], "opencode")
        self.assertIn("--product", calls[0])
        self.assertIn("opencode", calls[0])
        self.assertEqual(calls[0][calls[0].index("--max-workers") + 1], "2")

    def test_repair_ready_tick_reaches_payload_render_and_slack_action(self) -> None:
        mod = load()
        p = build(mod, resolve_ticks=resolver_tick(
            verdict="WOULD_REPAIR",
            action="would_repair",
            ready=True,
        ))
        self.assertEqual(p["utilization"]["state"], "HEADROOM_REPAIR_READY")
        self.assertIn("--repair-batch", p["utilization"]["launch_command"])
        rendered = mod.render(p)
        self.assertIn("use       : HEADROOM_REPAIR_READY", rendered)
        self.assertIn("launch    : python", rendered)
        buckets = mod._dispatch_slack_buckets(p)
        self.assertTrue(any("resolver dry-run launch-ready" in r for r in buckets["action"]))
        self.assertTrue(any("utilization HEADROOM_REPAIR_READY" in r
                            for r in buckets["action"]))

    def test_repair_in_flight_is_productive_utilization(self) -> None:
        mod = load()
        p = build(mod, resolve_ticks=resolver_tick(
            verdict="REPAIR_IN_FLIGHT",
            action="repair_in_flight",
            ready=None,
        ))
        self.assertEqual(p["utilization"]["state"], "REPAIR_IN_FLIGHT")
        self.assertIn("repair-in-flight", mod.render(p))
        self.assertTrue(any("resolver repair-in-flight" in r
                            for r in mod._dispatch_slack_buckets(p)["auto-solving"]))

    def test_held_tick_reaches_payload_render_and_slack_action(self) -> None:
        mod = load()
        p = build(mod, resolve_ticks=resolver_tick(
            verdict="SELF_MODIFY_HOLD",
            action="no_lane",
            ready=False,
            blocker="SELF_MODIFY_HOLD",
            next_action="enable-worktree-isolated-resolver",
        ))
        self.assertTrue(any("selected resolver tick held" in r for r in p["reasons"]))
        self.assertEqual(p["utilization"]["state"], "HEADROOM_HELD")
        self.assertEqual(p["utilization"]["blockers"][0]["code"], "SELF_MODIFY_HOLD")
        self.assertIn("held SELF_MODIFY_HOLD", mod.render(p))
        self.assertIn("use       : HEADROOM_HELD", mod.render(p))
        self.assertTrue(any("last resolver tick held SELF_MODIFY_HOLD" in r
                            for r in mod._dispatch_slack_buckets(p)["action"]))

    def test_safe_lanes_busy_is_primary_utilization_action(self) -> None:
        mod = load()
        tick = resolver_tick(
            verdict="SELF_MODIFY_HOLD",
            action="no_lane",
            ready=False,
            blocker="SELF_MODIFY_HOLD",
            next_action="route-non-self-source-lane-or-enable-worktree-isolated-resolver",
        )
        latest = tick["selected"]
        latest["lane"] = None
        latest["target_issue"] = None
        latest["launch_gate"]["blockers"] = [
            {
                "code": "SAFE_LANES_BUSY",
                "reason": "safe non-self-source lanes already have live workers: docs, tools",
                "next_action": "wait-for-safe-lane-lease",
            },
            {
                "code": "SELF_MODIFY_HOLD",
                "reason": "every remaining open issue lane maps to the shared source tree",
                "next_action": "route-non-self-source-lane-or-enable-worktree-isolated-resolver",
            },
        ]
        latest["next_action"] = "wait-for-safe-lane-lease"
        p = build(mod, resolve_ticks=tick)
        self.assertEqual(p["utilization"]["state"], "HEADROOM_HELD")
        self.assertEqual(p["utilization"]["blockers"][0]["code"], "SAFE_LANES_BUSY")
        self.assertEqual(p["utilization"]["next_actions"][0], "wait-for-safe-lane-lease")
        self.assertIn("held SAFE_LANES_BUSY", mod.render(p))
        self.assertTrue(any("last resolver tick held SAFE_LANES_BUSY" in r
                            for r in mod._dispatch_slack_buckets(p)["action"]))

    def test_lab_hold_is_a_utilization_blocker(self) -> None:
        mod = load()
        lab = {
            "schema": mod.LINK_STATE_SCHEMA,
            "subject": "gpu-server",
            "phase": "WAITING",
            "detail": "indeterminate",
            "next_action": "publish-lab-readiness",
            "evidence": "no-readiness-record",
            "admit_dispatch": False,
            "commands": mod._lab_readiness_commands(),
        }
        p = build(mod, resolve_ticks=resolver_tick(), lab_readiness=lab)
        blockers = {b["code"]: b for b in p["utilization"]["blockers"]}
        self.assertEqual(p["utilization"]["state"], "HEADROOM_LAUNCH_READY")
        self.assertIn("LAB_READINESS_HELD", blockers)
        self.assertIn("publish-lab-readiness", p["utilization"]["next_actions"])
        md = mod.render_md(p, date="2026-07-04")
        self.assertIn("**utilization**", md)
        self.assertIn("publish-lab-readiness", md)
        self.assertIn("lab publish command", md)

    def test_stale_capacity_refusal_is_not_treated_as_current_hold(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK", live=0, cap=16),
                  resolve_ticks=resolver_tick(
                      verdict="REFUSE_AT_CAP",
                      action="refused",
                      ready=None,
                      next_action="",
                  ))
        self.assertEqual(p["utilization"]["state"], "HEADROOM_STALE_PLAN")
        self.assertEqual(p["utilization"]["blockers"][0]["code"], "STALE_PLANNER_REFUSAL")
        self.assertIn("run-issue-resolve-dispatch-dry-run",
                      p["utilization"]["next_actions"])
        self.assertIn("HEADROOM_STALE_PLAN", mod.render(p))


class LeaseStateTest(unittest.TestCase):
    def _backlog(self) -> dict:
        return {
            "lanes": {
                "tools": {"tree": ["tools/**"], "issues": [1762]},
            },
            "issues": [
                {"number": 1762, "lane": "tools", "confidence": "path-confirmed"},
            ],
        }

    @staticmethod
    def _swap_run(mod, fake):
        """Temporarily replace mod.subprocess.run; returns a restore thunk."""
        orig = mod.subprocess.run
        mod.subprocess.run = fake
        return lambda: setattr(mod.subprocess, "run", orig)

    def test_read_leaseref_returns_triple_when_git_read_fails(self) -> None:
        # The git-error return paths must carry the SAME 3-tuple arity as the
        # happy path. When `git for-each-ref` returns non-zero, a 2-tuple return
        # makes read_lease_state's `records, sessions, err = ...` unpack raise
        # ValueError, which crashes the whole status-card render (froze the
        # operator card for ~3 days, 2026-07-02..05).
        mod = load()

        class _Boom:
            returncode = 128
            stdout = ""
            stderr = "fatal: not a git repository"

        restore = self._swap_run(mod, lambda *a, **k: _Boom())
        try:
            triple = mod.read_leaseref_records_and_sessions(Path("."))
        finally:
            restore()
        self.assertEqual(len(triple), 3)
        records, sessions, err = triple  # must not raise
        self.assertEqual(records, [])
        self.assertEqual(sessions, {})
        self.assertIn("not a git repository", err)

    def test_read_leaseref_returns_triple_on_git_oserror(self) -> None:
        mod = load()

        def _raise(*a, **k):
            raise OSError("git binary missing")

        restore = self._swap_run(mod, _raise)
        try:
            triple = mod.read_leaseref_records_and_sessions(Path("."))
        finally:
            restore()
        self.assertEqual(len(triple), 3)
        _records, sessions, err = triple
        self.assertEqual(sessions, {})
        self.assertIn("git binary missing", err)

    def test_read_lease_state_survives_git_read_failure(self) -> None:
        # End-to-end: the caller that actually crashed must degrade to a
        # read_error dict, not raise, when the lease-ref read fails.
        mod = load()

        class _Boom:
            returncode = 128
            stdout = ""
            stderr = "fatal: not a git repository"

        restore = self._swap_run(mod, lambda *a, **k: _Boom())
        try:
            state = mod.read_lease_state(Path("."), self._backlog())
        finally:
            restore()
        self.assertIn("read_error", state)
        self.assertEqual(state["active_count"], 0)
        self.assertEqual(state["blocking_count"], 0)

    def test_summarize_leases_marks_blocking_and_non_blocking_active_leases(self) -> None:
        mod = load()
        state = mod.summarize_leases([
            {
                "id": "resolve-tools-1762",
                "tree_globs": ["tools/**"],
                "holder": "peer-a",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
            {
                "id": "resolve-model-1700",
                "tree_globs": ["internal/model/**"],
                "holder": "peer-b",
                "acquired_unix": 1_100,
                "ttl_seconds": 3_600,
            },
            {
                "id": "resolve-docs",
                "tree_globs": ["docs/**"],
                "holder": "old-peer",
                "acquired_unix": 100,
                "ttl_seconds": 10,
            },
        ], self._backlog(), now_ts=1_600)

        self.assertEqual(state["active_count"], 2)
        self.assertEqual(state["expired_count"], 1)
        self.assertEqual(state["blocking_count"], 1)
        by_id = {row["id"]: row for row in state["active"]}
        self.assertTrue(by_id["resolve-tools-1762"]["blocks_candidate"])
        self.assertEqual(by_id["resolve-tools-1762"]["lane"], "tools")
        self.assertEqual(by_id["resolve-tools-1762"]["age_min"], 10.0)
        self.assertEqual(
            by_id["resolve-tools-1762"]["blocking_candidates"][0]["issue"], 1762)
        self.assertFalse(by_id["resolve-model-1700"]["blocks_candidate"])
        self.assertEqual(state["expired"][0]["id"], "resolve-docs")

    def test_candidate_blocking_is_unknown_when_backlog_fold_unavailable(self) -> None:
        mod = load()
        state = mod.summarize_leases([
            {
                "id": "resolve-tools-1762",
                "tree_globs": ["tools/**"],
                "holder": "peer-a",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
        ], {"_skipped": "fast"}, now_ts=1_600)

        self.assertFalse(state["candidate_source_available"])
        self.assertIsNone(state["active"][0]["blocks_candidate"])

    def test_summarize_leases_annotates_session_liveness(self) -> None:
        mod = load()
        state = mod.summarize_leases([
            {
                "id": "resolve-docs",
                "tree_globs": ["docs/**"],
                "holder": "node-a/s-live",
                "session_id": "s-live",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
            {
                "id": "resolve-tools",
                "tree_globs": ["tools/**"],
                "holder": "node-a/s-stopped",
                "session_id": "s-stopped",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
            {
                "id": "resolve-gateway",
                "tree_globs": ["internal/gateway/**"],
                "holder": "legacy",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
        ], self._backlog(), sessions={
            "s-live": {"id": "s-live", "pcb_state": "RUNNING",
                       "updated_at": 1_500, "ttl_seconds": 600},
            "s-stopped": {"id": "s-stopped", "pcb_state": "STOPPED",
                          "updated_at": 1_500, "ttl_seconds": 600},
        }, now_ts=1_600)

        by_id = {row["id"]: row for row in state["active"]}
        self.assertEqual(by_id["resolve-docs"]["liveness"], "peer-live")
        self.assertFalse(by_id["resolve-docs"]["reclaimable"])
        self.assertEqual(by_id["resolve-docs"]["session_id"], "s-live")
        self.assertEqual(by_id["resolve-tools"]["liveness"], "peer-dead")
        self.assertTrue(by_id["resolve-tools"]["reclaimable"])
        self.assertEqual(by_id["resolve-gateway"]["liveness"], "peer-unknown")
        self.assertFalse(by_id["resolve-gateway"]["reclaimable"])

    def test_summarize_leases_counts_stranded_vs_live_blocking(self) -> None:
        # #4324: LANE_LEASE_HELD refusals against a dead holder ("phantom"
        # collisions) must be countable separately from refusals against a
        # genuinely live peer, so a release-on-exit fix's effect is measurable.
        mod = load()
        state = mod.summarize_leases([
            {
                "id": "resolve-tools-1762",
                "tree_globs": ["tools/**"],
                "holder": "node-a/s-stopped",
                "session_id": "s-stopped",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
            {
                "id": "resolve-tools-1763",
                "tree_globs": ["tools/**"],
                "holder": "node-a/s-live",
                "session_id": "s-live",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
            {
                "id": "resolve-model-1700",
                "tree_globs": ["internal/model/**"],
                "holder": "peer-b",
                "acquired_unix": 1_100,
                "ttl_seconds": 3_600,
            },
        ], self._backlog(), sessions={
            "s-stopped": {"id": "s-stopped", "pcb_state": "STOPPED",
                          "updated_at": 1_500, "ttl_seconds": 600},
            "s-live": {"id": "s-live", "pcb_state": "RUNNING",
                       "updated_at": 1_500, "ttl_seconds": 600},
        }, now_ts=1_600)

        self.assertEqual(state["blocking_count"], 2)
        self.assertEqual(state["blocking_stranded_count"], 1)
        self.assertEqual(state["blocking_live_count"], 1)

    def test_age_text_formats_minutes_and_hours(self) -> None:
        mod = load()
        self.assertEqual(mod._age_text(10.0), "10m")
        self.assertEqual(mod._age_text(120.0), "2h")


class WorkerLeaseCrossCheckTest(unittest.TestCase):
    def _leases(self) -> dict:
        return {
            "active": [
                {"id": "resolve-tools-1765", "lane": "tools", "holder": "peer-a"},
                {"id": "resolve-model-1700", "lane": "model", "holder": "peer-b"},
            ],
        }

    def test_cross_check_buckets_clean_orphan_process_and_orphan_lease(self) -> None:
        mod = load()
        workers = {
            "available": True,
            "workers": [
                {
                    "worker": "resolve-1765-20260701-120000",
                    "issue": 1765,
                    "pid": 111,
                    "lane": "tools",
                    "backend": "opencode",
                    "lease_id": "resolve-tools-1765",
                },
                {
                    "worker": "resolve-1770-20260701-120001",
                    "issue": 1770,
                    "pid": 222,
                    "lane": "docs",
                    "backend": "claude",
                    "lease_id": "resolve-docs-1770",
                },
            ],
        }

        out = mod.cross_check_worker_leases(workers, self._leases())

        self.assertEqual(out["clean_count"], 1)
        self.assertEqual(out["orphan_process_count"], 1)
        self.assertEqual(out["orphan_lease_count"], 1)
        self.assertEqual(out["clean"][0]["worker"]["worker"], "resolve-1765-20260701-120000")
        self.assertEqual(out["orphan_process"][0]["worker"]["worker"], "resolve-1770-20260701-120001")
        self.assertEqual(out["orphan_lease"][0]["lease"]["id"], "resolve-model-1700")
        summary = mod._active_worker_summary(out)
        self.assertIn("#1765 tools/opencode pid=111 lease=resolve-tools-1765", summary)
        self.assertIn("#1770 docs/claude pid=222 lease=resolve-docs-1770", summary)

    def test_payload_render_surfaces_active_resolver_workers(self) -> None:
        mod = load()
        worker_leases = {
            "available": True,
            "clean_count": 1,
            "orphan_process_count": 0,
            "orphan_lease_count": 0,
            "clean": [{
                "worker": {
                    "worker": "resolve-2718-20260705-183239",
                    "issue": 2718,
                    "pid": 70232,
                    "lane": "tools",
                    "backend": "opencode",
                    "lease_id": "resolve-tools",
                },
                "lease": {"id": "resolve-tools"},
            }],
            "orphan_process": [],
            "orphan_lease": [],
        }
        p = build(mod, worker_leases=worker_leases)
        self.assertTrue(any("#2718 tools/opencode pid=70232" in r
                            for r in p["reasons"]))
        self.assertIn("live work : #2718 tools/opencode pid=70232", mod.render(p))
        self.assertIn("**active resolver workers**: #2718 tools/opencode pid=70232",
                      mod.render_md(p, date="2026-07-05"))
        self.assertTrue(any("#2718 tools/opencode pid=70232" in r
                            for r in mod._dispatch_slack_buckets(p)["auto-solving"]))

    def test_cross_check_preserves_unavailable_liveness(self) -> None:
        mod = load()
        out = mod.cross_check_worker_leases(
            {"available": False, "error": "psutil unavailable"}, self._leases())
        self.assertFalse(out["available"])
        self.assertEqual(out["orphan_process_count"], 0)
        self.assertEqual(out["orphan_lease_count"], 0)


# Windows FILETIME (100ns ticks since 1601-01-01) for a known Unix epoch second.
def _filetime(epoch_s: float) -> int:
    return int((epoch_s + 11644473600.0) * 1e7)


class LaneLeaseHolderLivenessTest(unittest.TestCase):
    """The cross-check must cover ALL lane leases, and judge holders on
    (pid, proc_starttime) — never on bare pid existence (#5859).

    Regression guard for the real defect: `dos lease-lane live` returned 24 exclusive
    lane leases, 18 held by processes that no longer existed (the oldest 22 days old,
    fencing `cmd/**` and `internal/modver/**`), while the card printed
    `lease chk : clean=3 orphan-process=0 unmatched-live-lease=0` and folded
    "worker/lease cross-check clean" into its summary. The cross-check only ever saw
    the `refs/fak/locks/*` resolver leases, a different substrate entirely.
    """

    NOW = 1786080000.0
    HOST = "fleet-box"

    def _lane_records(self) -> list[dict]:
        """One live holder, one dead (pid gone), one dead by PID RECYCLING."""
        return [
            {"lane": "docs", "holder": "worker-live", "host_id": self.HOST,
             "mode": "exclusive", "pid": 1001, "tree": ["docs/**"],
             "acquired_at": "2026-08-06T12:00:00Z",
             "proc_starttime": _filetime(self.NOW - 600)},
            {"lane": "cmd", "holder": "claude-5031", "host_id": self.HOST,
             "mode": "exclusive", "pid": 51980, "tree": ["cmd/**"],
             "acquired_at": "2026-07-19T09:00:00Z",
             "proc_starttime": _filetime(self.NOW - 1_600_000)},
            {"lane": "guard", "holder": "worker-2461", "host_id": self.HOST,
             "mode": "exclusive", "pid": 44396, "tree": ["internal/modver/**"],
             "acquired_at": "2026-08-06T11:00:00Z",
             "proc_starttime": _filetime(self.NOW - 4000)},
        ]

    def _starts(self) -> dict[int, float]:
        """The live process table. pid 51980 is GONE. pid 44396 EXISTS but was
        recycled — it started long after the lease recorded it."""
        return {1001: self.NOW - 600, 44396: self.NOW - 100, 2: self.NOW}

    def _fold(self, **over):
        mod = load()
        kwargs = {"starts": self._starts(), "host_id": self.HOST, "now_ts": self.NOW}
        kwargs.update(over)
        return mod, mod.summarize_lane_lease_holders(self._lane_records(), **kwargs)

    def test_dead_holder_counted_including_recycled_pid(self) -> None:
        _, lane = self._fold()
        self.assertTrue(lane["available"])
        self.assertEqual(lane["total"], 3)
        self.assertEqual(lane["live_count"], 1)
        # BOTH dead holders — the missing pid AND the recycled one. A bare
        # pid-existence check would score the recycled holder "alive" and report 1.
        self.assertEqual(lane["dead_count"], 2)
        dead_lanes = sorted(r["lane"] for r in lane["dead"])
        self.assertEqual(dead_lanes, ["cmd", "guard"])
        self.assertIn("RECYCLED", next(
            r["holder_evidence"] for r in lane["dead"] if r["lane"] == "guard"))
        # The action must never hand the operator a blind reap (#5859). The verb may
        # be NAMED — as the thing not to run — but never as a command to execute.
        action = lane["next_action"]
        self.assertIn("do NOT reap", action)
        self.assertNotIn("--owner <holder>", action)
        self.assertNotIn("release --lane", action)

    def test_pid_existence_alone_is_not_proof_of_life(self) -> None:
        """A live pid with no readable proc_starttime is UNKNOWN, never live."""
        mod = load()
        state, evidence = mod.classify_lane_lease_holder(
            {"lane": "cmd", "pid": 1001, "host_id": self.HOST}, self._starts(),
            host_id=self.HOST)
        self.assertEqual(state, "unknown")
        self.assertIn("pid existence alone is not proof of life", evidence)

    def test_verdict_is_not_clean_with_a_dead_holder(self) -> None:
        """THE defect, at the verdict seam: a cross-check whose local worker/lease
        pairs all match must still not read "clean" while a lane is fenced."""
        mod, lane = self._fold()
        out = mod.cross_check_worker_leases(
            {"available": True, "workers": []}, {"active": []}, lane)
        self.assertEqual(out["dead_holder_count"], 2)
        self.assertFalse(mod.lane_lease_verdict_clean(out))
        reasons = " | ".join(mod._worker_lease_reasons(out))
        self.assertNotIn("cross-check clean", reasons)
        self.assertIn("dead-holder=2", reasons)
        self.assertIn("next action:", reasons)

    def test_unreadable_lane_lease_set_is_not_clean(self) -> None:
        """An unread lease set is not a clean one — and neither is one this host has
        no liveness oracle for.

        Note the oracle case is now judged from the WAL, not the process table:
        TTL expiry is time evidence the lease record carries itself, so a host with
        no psutil can still prove a 452-hour-old lease stale. `available` stays
        False (the pid corroboration rung is blind), so the card still withholds
        "clean" — the fail-safe direction is preserved without pretending the WAL
        said nothing.
        """
        mod, lane = self._fold(read_error="dos: command not found")
        self.assertFalse(lane["available"])
        out = mod.cross_check_worker_leases({"available": True, "workers": []},
                                            {"active": []}, lane)
        self.assertFalse(mod.lane_lease_verdict_clean(out))
        no_oracle = mod.summarize_lane_lease_holders(
            self._lane_records(), starts=None, host_id=self.HOST, now_ts=self.NOW)
        self.assertFalse(no_oracle["available"])
        self.assertFalse(mod.lane_lease_verdict_clean(
            mod.cross_check_worker_leases({"available": True, "workers": []},
                                          {"active": []}, no_oracle)))
        # All three fixtures are hours-to-weeks past the 50m TTL, and that is WAL
        # evidence — no process probe involved.
        self.assertEqual(no_oracle["dead_count"], 3)
        self.assertEqual(no_oracle["live_count"], 0)
        for row in no_oracle["rows"]:
            self.assertIn("TTL EXPIRED", row["holder_evidence"])
            self.assertIn("psutil unavailable", row["holder_evidence"])

    def test_foreign_host_holder_is_unknown_not_dead(self) -> None:
        """This host cannot probe another host's pid — and must not call it dead."""
        mod = load()
        state, _ = mod.classify_lane_lease_holder(
            {"lane": "cmd", "pid": 51980, "host_id": "other-box"},
            self._starts(), host_id=self.HOST)
        self.assertEqual(state, "unknown")

    def test_proc_starttime_decodes_windows_filetime(self) -> None:
        mod = load()
        # The real record observed on the reference tree.
        self.assertAlmostEqual(
            mod._proc_starttime_epoch(134305505540666317), 1786076954.07, places=1)
        self.assertIsNone(mod._proc_starttime_epoch(None))
        self.assertIsNone(mod._proc_starttime_epoch(0))

    def test_read_lane_leases_parses_kernel_json(self) -> None:
        mod = load()
        seen = {}

        def runner(cmd, cwd):
            seen["cmd"], seen["cwd"] = cmd, cwd
            return (json.dumps(self._lane_records()), None)

        records, err = mod.read_lane_leases(ROOT, runner=runner)
        self.assertIsNone(err)
        self.assertEqual(len(records), 3)
        self.assertEqual(seen["cmd"], ["dos", "lease-lane", "live"])
        # `--workspace` makes the kernel re-resolve to a different .dos/ and emit
        # non-JSON; the cwd carries the workspace instead.
        self.assertNotIn("--workspace", seen["cmd"])

    def test_render_surfaces_dead_holder_on_every_operator_surface(self) -> None:
        mod, lane = self._fold()
        worker_leases = mod.cross_check_worker_leases(
            {"available": True, "workers": []}, {"active": []}, lane)
        p = build(mod, worker_leases=worker_leases)

        text = mod.render(p)
        self.assertIn("dead-holder=2", text)
        self.assertIn("lane lease: 3 held", text)
        self.assertIn("do NOT reap", text)
        self.assertNotIn("release --lane", text)
        self.assertNotIn("cross-check clean", text)

        md = mod.render_md(p, date="2026-08-06")
        self.assertIn("dead-holder=2", md)
        self.assertIn("Kernel lane leases", md)
        self.assertIn("`cmd`", md)
        self.assertIn("do NOT reap", md)

        slack = mod.slack_text(p)
        self.assertIn("dead-holder", slack)
        self.assertIn("TTL-EXPIRED", slack)
        self.assertIn("do NOT reap", slack)
        self.assertNotIn("release --lane", slack)
        self.assertNotIn("worker/lease cross-check clean", slack)

    def test_all_live_holders_still_read_clean(self) -> None:
        """The guard must not cry wolf: with every holder live the card is clean."""
        mod = load()
        lane = mod.summarize_lane_lease_holders(
            self._lane_records()[:1], starts=self._starts(), host_id=self.HOST,
            now_ts=self.NOW)
        self.assertEqual(lane["dead_count"], 0)
        self.assertEqual(lane["live_count"], 1)
        out = mod.cross_check_worker_leases(
            {"available": True,
             "workers": [{"worker": "w", "issue": 1, "pid": 1001, "lane": "tools",
                          "backend": "claude", "lease_id": "resolve-tools-1765"}]},
            {"active": [{"id": "resolve-tools-1765", "lane": "tools", "holder": "peer-a"}]},
            lane)
        self.assertTrue(mod.lane_lease_verdict_clean(out))
        self.assertIn("worker/lease cross-check clean",
                      " | ".join(mod._worker_lease_reasons(out)))

    def test_lane_lease_verdict_absent_fold_preserves_legacy_behaviour(self) -> None:
        mod = load()
        out = mod.cross_check_worker_leases({"available": True, "workers": []},
                                            {"active": []})
        self.assertTrue(mod.lane_lease_verdict_clean(out))
        self.assertNotIn("lane_leases", out)

    def test_card_must_not_read_clean_with_a_dead_holder_lease(self) -> None:
        """THE reported symptom, asserted on the rendered card.

        Deliberately built from a LITERAL payload rather than the fold's own
        helpers, so it exercises only the pre-existing `worker_lease_check` ->
        render seam. That makes it a real fail-first guard: against the unfixed
        `dispatch_status.py` this reaches `render()` and fails on the assertions
        below (the card printed `clean=3 orphan-process=0 unmatched-live-lease=0`
        and "worker/lease cross-check clean" with 18 lanes fenced), rather than
        erroring out on a missing symbol.
        """
        mod = load()
        worker_leases = {
            "available": True,
            # Every LOCAL worker/lease pair matches — this is the state that used to
            # print "clean" while the kernel's lane leases were all fenced.
            "clean_count": 3,
            "orphan_process_count": 0,
            "orphan_lease_count": 0,
            "clean": [], "orphan_process": [], "orphan_lease": [],
            "dead_holder_count": 2,
            "lane_leases": {
                "schema": "fleet-lane-lease-liveness/1",
                "source": "dos lease-lane live",
                "available": True,
                "total": 3, "live_count": 1, "dead_count": 2, "unknown_count": 0,
                "next_action": ("do NOT reap: `dos lease-lane release` runs NO "
                                "liveness check (#5859) · the admission fold already "
                                "elides these"),
                "rows": [], "dead": [
                    {"lane": "cmd", "holder": "claude-5031", "pid": 51980,
                     "age_min": 31680.0, "holder_state": "dead",
                     "holder_evidence": "TTL EXPIRED: no acquire stamp for 528h"},
                    {"lane": "modver", "holder": "worker-2461", "pid": 56980,
                     "age_min": 17280.0, "holder_state": "dead",
                     "holder_evidence": "TTL EXPIRED: no acquire stamp for 288h"},
                ],
            },
        }
        p = build(mod, worker_leases=worker_leases)

        text = mod.render(p)
        self.assertIn("dead-holder=2", text)
        self.assertNotIn("worker/lease cross-check clean", text)
        self.assertIn("cmd(claude-5031, pid 51980", text)
        self.assertIn("do NOT reap", text)
        self.assertNotIn("release --lane", text)

        self.assertTrue(any("dead-holder=2" in r for r in p["reasons"]),
                        f"no dead-holder counter in reasons: {p['reasons']}")
        self.assertFalse(any("cross-check clean" in r for r in p["reasons"]),
                         f"card still reads clean with 2 dead holders: {p['reasons']}")


class LaneLeaseEphemeralAcquirerTest(unittest.TestCase):
    """The holder predicate must judge a lease by the WAL, not by its acquirer pid.

    The first cut of the lane-lease fold decided deadness from `(pid,
    proc_starttime)` alone. That predicate cannot discriminate: a lane lease's
    recorded pid is the EPHEMERAL `dos lease-lane acquire` subprocess, which
    journals the ACQUIRE and exits immediately (`dos/lane_lease.py:466-492`), and
    the reservation is designed to outlive it (`acquire()` at `:453` says so). So
    a healthy, actively-held lease always probes "dead", and the card rendered
    `lane lease: 25 held - live=0 dead-holder=25` while several of those lanes
    were held by agents running at that instant — four of them acquired MINUTES
    earlier, and five holders self-released their own leases inside one 9-minute
    window while probing "dead" throughout.

    The corrected predicate mirrors the kernel's own live-set fold
    (`dos.lane_lease._lease_is_dead`): heartbeat/TTL staleness is PRIMARY, the pid
    only corroborates.

    Run this class against the pre-fix module to see it fail:

        git show <sha>:tools/dispatch_status.py > $SCRATCH/pristine_dispatch_status.py
        DISPATCH_STATUS_SCRIPT=$SCRATCH/pristine_dispatch_status.py \\
            python tools/dispatch_status_test.py LaneLeaseEphemeralAcquirerTest
    """

    NOW = 1786080000.0            # 2026-08-07T05:20:00Z
    HOST = "fleet-box"

    def _classify(self, rec, starts=None):
        """Classify one record, pinning the clock when the module accepts a pin.

        `now_ts` is passed only if the loaded classifier takes it, so that against a
        PRE-FIX module these tests still reach the real predicate and fail on its
        VERDICT rather than erroring out on a missing keyword. The pre-fix classifier
        reads no clock at all, so omitting the pin changes nothing for it; every
        fixture below is dated relative to `NOW` and judged only on the resulting
        state.
        """
        import inspect
        mod = load()
        fn = mod.classify_lane_lease_holder
        kwargs = {"host_id": self.HOST}
        if "now_ts" in inspect.signature(fn).parameters:
            kwargs["now_ts"] = self.NOW
        return fn(rec, {} if starts is None else starts, **kwargs)

    @staticmethod
    def _iso(now: float, minutes_ago: float) -> str:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(
            now - minutes_ago * 60.0, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def test_absent_pid_with_fresh_heartbeat_is_live(self) -> None:
        """THE decisive assertion. A lease whose pid is gone but whose heartbeat is
        FRESH is live — that is the normal, healthy shape of a held lane lease, not
        a reapable orphan.

        Against the pre-fix classifier this returns `dead` on the strength of the
        absent pid alone, with no reference to the heartbeat at all.
        """
        state, evidence = self._classify({
            "lane": "tools", "holder": "guard-livelock-5858", "host_id": self.HOST,
            "pid": 33784, "proc_starttime": _filetime(self.NOW - 4000),
            "acquired_at": self._iso(self.NOW, 240.0),
            "heartbeat_at": self._iso(self.NOW, 0.5),
        }, starts={})       # pid 33784 is NOT in the process table
        self.assertEqual(state, "live", f"fresh heartbeat judged {state}: {evidence}")
        self.assertIn("fresh", evidence)
        self.assertIn("not a running process", evidence)  # the pid IS reported…
        self.assertIn("ephemeral", evidence)              # …and explicitly discounted

    def test_absent_pid_with_fresh_acquire_stamp_is_live(self) -> None:
        """The same, for the shape this fleet actually writes: no heartbeat op has
        ever been journaled here, so `acquired_at` IS the newest stamp. A lease
        acquired seconds ago must never read dead."""
        state, evidence = self._classify({
            "lane": "docs", "holder": "claude-5842", "host_id": self.HOST,
            "pid": 84412, "proc_starttime": _filetime(self.NOW - 900),
            "acquired_at": self._iso(self.NOW, 0.75),
        }, starts={})
        self.assertEqual(state, "live", f"45s-old lease judged {state}: {evidence}")

    def test_absent_pid_inside_ttl_is_unknown_not_dead(self) -> None:
        """Past the freshness grace but inside the TTL, with the ephemeral acquirer
        gone: unproven. `unknown` — never `live`, and never `dead` either, because
        an absent acquirer pid is the expected state of a healthy lease."""
        state, evidence = self._classify({
            "lane": "session", "holder": "claude-5254", "host_id": self.HOST,
            "pid": 61696, "proc_starttime": _filetime(self.NOW - 3000),
            "acquired_at": self._iso(self.NOW, 20.0),
        }, starts={})
        self.assertEqual(state, "unknown", f"20m-old lease judged {state}: {evidence}")
        self.assertIn("ephemeral", evidence)

    def test_ttl_expiry_is_the_death_evidence(self) -> None:
        """A lease past its TTL + grace IS dead — that is the kernel's own primary
        signal, and it is what `live_leases(expire_dead=True)` elides. The pid rung
        rides along as corroboration only."""
        state, evidence = self._classify({
            "lane": "adjudicator", "holder": "fable-superloop-23874",
            "host_id": self.HOST, "pid": 63712,
            "proc_starttime": _filetime(self.NOW - 2_000_000),
            "acquired_at": self._iso(self.NOW, 31680.0),   # 22 days
        }, starts={})
        self.assertEqual(state, "dead")
        self.assertIn("TTL EXPIRED", evidence)

    def test_declared_ttl_minutes_is_honoured_over_the_backstop(self) -> None:
        """A lease that declares a long `ttl_minutes` is not dead at the 50-minute
        backstop — the backstop only catches a record that declared none."""
        rec = {"lane": "bench", "holder": "long-runner", "host_id": self.HOST,
               "pid": 4242, "acquired_at": self._iso(self.NOW, 300.0),
               "ttl_minutes": 600}
        state, _ = self._classify(rec, starts={})
        self.assertEqual(state, "unknown")
        state, evidence = self._classify(dict(rec, ttl_minutes=60), starts={})
        self.assertEqual(state, "dead")
        self.assertIn("60-minute TTL", evidence)

    def test_dead_pid_alone_never_kills_a_lease(self) -> None:
        """The load-bearing inversion, stated directly: across the whole freshness
        range, a confidently-dead pid on its own produces ZERO `dead` verdicts.

        Pre-fix, every one of these is `dead` — the predicate returned the same
        answer for a 30-second-old lease and a three-week-old one, which is why it
        reported live=0 of 25.
        """
        states = []
        for minutes in (0.1, 1.0, 4.9, 5.0, 9.0, 20.0, 45.0):
            rec = {"lane": f"l{minutes}", "holder": "h", "host_id": self.HOST,
                   "pid": 999, "proc_starttime": _filetime(self.NOW - 5000),
                   "acquired_at": self._iso(self.NOW, minutes)}
            states.append(self._classify(rec, starts={})[0])
        self.assertNotIn("dead", states, f"dead pid alone still kills leases: {states}")
        self.assertEqual(states[:4], ["live", "live", "live", "live"])

    def test_whole_set_does_not_collapse_to_all_dead(self) -> None:
        """The reality check: fold a set shaped like the real one and assert the
        verdict actually discriminates. A classifier that answers `dead` for every
        lease is not fixed, whatever its internals say."""
        mod = load()
        records = [
            # freshly acquired, acquirer already exited — the healthy shape
            {"lane": "docs", "holder": "claude-5842", "host_id": self.HOST,
             "pid": 84412, "acquired_at": self._iso(self.NOW, 0.75)},
            {"lane": "hooks", "holder": "claude-5026", "host_id": self.HOST,
             "pid": 50584, "acquired_at": self._iso(self.NOW, 4.0)},
            # inside the TTL, unprovable either way
            {"lane": "tools", "holder": "guard-5858", "host_id": self.HOST,
             "pid": 33784, "acquired_at": self._iso(self.NOW, 9.0)},
            {"lane": "cmd", "holder": "claude-5254", "host_id": self.HOST,
             "pid": 3388, "acquired_at": self._iso(self.NOW, 20.0)},
            # genuinely aged out
            {"lane": "adjudicator", "holder": "fable-superloop", "host_id": self.HOST,
             "pid": 63712, "acquired_at": self._iso(self.NOW, 31680.0)},
        ]
        lane = mod.summarize_lane_lease_holders(
            records, starts={}, host_id=self.HOST, now_ts=self.NOW)
        self.assertEqual(lane["total"], 5)
        self.assertEqual(lane["live_count"], 2)
        self.assertEqual(lane["unknown_count"], 2)
        self.assertEqual(lane["dead_count"], 1)
        self.assertLess(lane["dead_count"], lane["total"],
                        "every lease still reads dead — the predicate is still wrong")

    def test_card_never_calls_dead_what_the_kernel_still_admits(self) -> None:
        """The safety invariant: the card's `dead` set is a SUBSET of what the
        kernel's admission fold drops, never a superset.

        It holds because `dead` now requires the same TTL+grace expiry the kernel
        uses, and the one rung the card narrows (dead-pid corroboration, which the
        card demands an OBSERVED heartbeat for) only ever moves a verdict toward
        `unknown`. Asserted here as monotonicity over age: the verdict may only walk
        live -> unknown -> dead, and it may not reach `dead` before TTL+grace.

        Verified live at authoring time against `dos lease-lane live` on the
        reference tree: the kernel's `live_leases(expire_dead=True)` dropped 25 of
        26 leases, the card called 18 of them dead, and `card_dead - kernel_dropped`
        was empty — the 7 differences were all card-`unknown`, the safe direction.
        """
        order = {"live": 0, "unknown": 1, "dead": 2}
        seen = []
        for minutes in (0.0, 1.0, 5.0, 5.1, 20.0, 54.9, 55.0, 55.1, 600.0, 31680.0):
            state, _ = self._classify({
                "lane": "cmd", "holder": "h", "host_id": self.HOST, "pid": 3388,
                "proc_starttime": _filetime(self.NOW - 9000),
                "acquired_at": self._iso(self.NOW, minutes),
            }, starts={})
            seen.append((minutes, state))
            if state == "dead":
                self.assertGreater(minutes, 50.0 + 5.0,
                                   f"dead at {minutes}m, before TTL+grace: {seen}")
        ranks = [order[s] for _, s in seen]
        self.assertEqual(ranks, sorted(ranks), f"verdict is not monotone in age: {seen}")
        self.assertEqual(seen[-1][1], "dead")
        self.assertEqual(seen[0][1], "live")

    def test_next_action_never_recommends_a_blind_release(self) -> None:
        """`dos lease-lane release` runs no liveness check and its `--owner ""`
        matches any holder (#5859), so the card must never hand an operator a reap
        loop over the lane set. It must instead say what IS true: the admission
        fold self-heals, and a stale structural lease blocks only
        `lane_lease.acquire()`."""
        mod = load()
        action = mod._LANE_LEASE_NEXT_ACTION
        self.assertNotIn("--owner <holder>", action)
        self.assertNotIn("release --lane", action)
        self.assertNotIn("reap each", action)
        self.assertIn("do NOT reap", action)
        self.assertIn("#5859", action)
        self.assertIn("OP_SCAVENGE", action)
        self.assertIn("adopt()", action)
        self.assertIn("expire_dead=True", action)
        self.assertIn("lane_lease.acquire()", action)
        self.assertIn("dos/lane_lease.py:453", action)


class RenderTest(unittest.TestCase):
    def test_render_does_not_raise_on_minimal_payload(self) -> None:
        mod = load()
        p = build(mod)
        text = mod.render(p)
        self.assertIn("DISPATCHER", text)
        self.assertIn("READY_TO_GROW", text)

    def test_render_does_not_raise_on_na_payload(self) -> None:
        mod = load()
        p = build(mod, backlog={"_skipped": "fast"}, closure={"_skipped": "fast"}, fast=True)
        text = mod.render(p)
        self.assertIn("n/a", text)

    def test_render_surfaces_silent_workers_line(self) -> None:
        mod = load()
        p = build(mod, silent=[{"issue": 465, "stamp": "20260621-232003",
                                "log": "resolve-465-20260621-232003.log", "pid": 39688}])
        text = mod.render(p)
        self.assertIn("1 silent", text)
        self.assertIn("#465", text)

    def test_merge_in_progress_reports_wait_action(self) -> None:
        mod = load()
        p = build(mod, merge={
            "merge_in_progress": True,
            "merge_head": "C:/work/fak/.git/MERGE_HEAD",
            "next_action": "wait for MERGE_HEAD to clear before starting new worker edits",
        })
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "MERGE_IN_PROGRESS")
        self.assertTrue(p["git"]["merge_in_progress"])
        self.assertIn("MERGE_HEAD", p["git"]["next_action"])
        text = mod.render(p)
        self.assertIn("MERGE_HEAD present", text)
        self.assertIn("wait for MERGE_HEAD to clear", text)
        slack = mod.slack_text(p)
        self.assertIn("peer merge in progress", slack)
        self.assertIn("wait for MERGE_HEAD", slack)

    def test_render_surfaces_blocking_and_non_blocking_leases(self) -> None:
        mod = load()
        leases = {
            "source": "refs/fak/locks",
            "candidate_source_available": True,
            "candidate_count": 1,
            "active_count": 2,
            "expired_count": 0,
            "blocking_count": 1,
            "active": [
                {
                    "id": "resolve-tools-1762",
                    "lane": "tools",
                    "holder": "peer-a",
                    "tree": ["tools/**"],
                    "age_min": 10.0,
                    "ttl_seconds": 3600,
                    "blocks_candidate": True,
                    "blocking_candidates": [{"issue": 1762, "lane": "tools"}],
                },
                {
                    "id": "resolve-model-1700",
                    "lane": "model",
                    "holder": "peer-b",
                    "tree": ["internal/model/**"],
                    "age_min": 5.0,
                    "ttl_seconds": 3600,
                    "blocks_candidate": False,
                    "blocking_candidates": [],
                },
            ],
            "expired": [],
        }

        p = build(mod, leases=leases)
        self.assertTrue(any("active lane lease" in r for r in p["reasons"]))
        text = mod.render(p)
        self.assertIn("leases", text)
        self.assertIn("resolve-tools-1762", text)
        self.assertIn("blocks #1762", text)
        self.assertIn("resolve-model-1700", text)
        self.assertIn("no candidate", text)
        slack = mod.slack_text(p)
        self.assertIn("leases 2 active (1 blocking)", slack)
        self.assertIn("active lane lease", slack)

    def test_render_surfaces_worker_lease_cross_check_buckets(self) -> None:
        mod = load()
        worker_leases = {
            "available": True,
            "clean_count": 1,
            "orphan_process_count": 1,
            "orphan_lease_count": 1,
            "clean": [{
                "worker": {"worker": "resolve-1765-20260701-120000", "issue": 1765, "pid": 111},
                "lease": {"id": "resolve-tools-1765"},
            }],
            "orphan_process": [{
                "worker": {"worker": "resolve-1770-20260701-120001", "issue": 1770, "pid": 222,
                           "lease_id": "resolve-docs-1770"},
                "reason": "missing active dispatch lease",
            }],
            "orphan_lease": [{
                "lease": {"id": "resolve-model-1700", "lane": "model", "holder": "peer-b"},
                "reason": "active lease has no local live worker sidecar",
            }],
        }

        p = build(mod, worker_leases=worker_leases)
        self.assertTrue(any("orphan-process=1" in r for r in p["reasons"]))
        self.assertTrue(any("unmatched-live-lease=1" in r for r in p["reasons"]))
        text = mod.render(p)
        self.assertIn(
            "lease chk : clean=1 orphan-process=1 unmatched-live-lease=1 dead-holder=0", text)
        self.assertIn("orphan-process resolve-1770-20260701-120001", text)
        self.assertIn("unmatched-live-lease resolve-model-1700", text)
        slack = mod.slack_text(p)
        self.assertIn("worker/lease mismatch", slack)
        self.assertIn("unmatched-live-lease=1", slack)


class SilentWorkersFoldTest(unittest.TestCase):
    """build_payload folds the injected silent list into payload['workers'] and a reason."""

    def test_silent_workers_fold_and_reason(self) -> None:
        mod = load()
        silent = [{"issue": 465, "stamp": "20260621-232003",
                   "log": "resolve-465-20260621-232003.log", "pid": 39688}]
        p = build(mod, silent=silent)
        self.assertEqual(p["workers"]["silent_count"], 1)
        self.assertEqual(p["workers"]["silent"], silent)
        self.assertTrue(any("exited producing nothing" in r for r in p["reasons"]))

    def test_no_silent_workers_emits_no_reason(self) -> None:
        mod = load()
        p = build(mod, silent=[])
        self.assertEqual(p["workers"]["silent_count"], 0)
        self.assertFalse(any("producing nothing" in r for r in p["reasons"]))

    def test_silent_defaults_to_empty_when_omitted(self) -> None:
        mod = load()
        p = build(mod)  # build() does not pass silent -> defaults to None -> []
        self.assertEqual(p["workers"]["silent_count"], 0)


class SilentWorkersScanTest(unittest.TestCase):
    """silent_workers() classification — hermetic: a tmp runs-dir + injected alive set."""

    def _mk(self, runs: Path, issue: int, stamp: str, *, size: int, pid: int,
            sidecar_mtime: float | None = None) -> None:
        import os
        log = runs / f"resolve-{issue}-{stamp}.log"
        log.write_bytes(b"x" * size)
        pid_file = runs / f"resolve-{issue}-{stamp}.pid"
        pid_file.write_text(str(pid), encoding="utf-8")
        if sidecar_mtime is not None:
            os.utime(pid_file, (sidecar_mtime, sidecar_mtime))

    def test_zero_byte_dead_pid_is_silent(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 465, "20260621-232003", size=0, pid=39688)
            out = mod.silent_workers(runs, alive=set())  # nothing alive
            self.assertEqual([w["issue"] for w in out], [465])
            self.assertEqual(out[0]["pid"], 39688)

    def test_zero_byte_live_pid_is_excluded(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 465, "20260621-232003", size=0, pid=39688)
            def probe(pid):
                return {
                            "alive": True,
                            "cmdline": "claude -p resolve GitHub issue #465",
                        }
            out = mod.silent_workers(runs, alive={39688}, probe=probe)  # still running
            self.assertEqual(out, [])

    def test_zero_byte_reused_pid_is_silent(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            self._mk(runs, 465, "20260621-232003", size=0, pid=39688,
                     sidecar_mtime=now)
            def probe(pid):
                return {
                            "alive": True,
                            "create_time": now + 60 * 60,
                            "cmdline": "chrome.exe --type=renderer",
                        }
            out = mod.silent_workers(runs, alive={39688}, probe=probe)
            self.assertEqual([w["issue"] for w in out], [465])

    def test_non_empty_log_is_excluded(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 464, "20260621-232247", size=2366, pid=18796)
            out = mod.silent_workers(runs, alive=set())
            self.assertEqual(out, [])

    def test_missing_runs_dir_is_empty(self) -> None:
        mod = load()
        out = mod.silent_workers(Path("does-not-exist-xyz"), alive=set())
        self.assertEqual(out, [])

    def test_no_liveness_oracle_reports_nothing(self) -> None:
        # alive=None with psutil unavailable must NOT false-positive a silent worker
        # (we cannot prove the pid dead, so we report nothing rather than a false alarm).
        mod = load()
        import builtins

        real_import = builtins.__import__

        def no_psutil(name, *a, **k):
            if name == "psutil":
                raise ImportError("psutil disabled for this test")
            return real_import(name, *a, **k)

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 465, "20260621-232003", size=0, pid=39688)
            builtins.__import__ = no_psutil
            try:
                out = mod.silent_workers(runs)  # alive=None -> tries psutil -> ImportError
            finally:
                builtins.__import__ = real_import
            self.assertEqual(out, [])

    def test_newest_first_ordering(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 465, "20260621-225432", size=0, pid=36580)
            self._mk(runs, 465, "20260621-232003", size=0, pid=39688)
            out = mod.silent_workers(runs, alive=set())
            self.assertEqual([w["stamp"] for w in out],
                             ["20260621-232003", "20260621-225432"])

    def test_banner_only_stub_dead_pid_is_silent(self) -> None:
        # A detached opencode worker that prints `> build · <model>` and exits is a
        # 122-byte banner-only stub: below the real-turn floor, so it produced
        # nothing even though size != 0 (the #1276 blind spot).
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 1207, "20260629-165930", size=122, pid=16752)
            out = mod.silent_workers(runs, alive=set())  # nothing alive
            self.assertEqual([w["issue"] for w in out], [1207])
            self.assertEqual(out[0]["kind"], "stub")
            self.assertEqual(out[0]["size"], 122)

    def test_banner_only_stub_live_pid_is_excluded(self) -> None:
        # A sub-floor log over a LIVE pid is still-running, not silent — the same
        # dead-pid guard that protects a claude worker streaming nothing until the end.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 1207, "20260629-165930", size=122, pid=16752)
            out = mod.silent_workers(runs, alive={16752},
                                     probe=lambda pid: {"alive": True,
                                                        "cmdline": "opencode run resolve GitHub issue #1207"})
            self.assertEqual(out, [])

    def test_at_floor_is_silent_over_floor_is_not(self) -> None:
        # Boundary: exactly _STUB_LOG_MAX_BYTES is still a stub; one byte over is output.
        mod = load()
        floor = mod._STUB_LOG_MAX_BYTES
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 1, "20260629-100000", size=floor, pid=111)
            self._mk(runs, 2, "20260629-100001", size=floor + 1, pid=222)
            out = mod.silent_workers(runs, alive=set())
            self.assertEqual([w["issue"] for w in out], [1])

    def test_stub_floor_matches_canonical(self) -> None:
        # Drift pin: the status-card floor must equal the dispatcher's classifier floor
        # so the two surfaces agree on what "produced a real turn" means.
        mod = load()
        spec = importlib.util.spec_from_file_location(
            "issue_resolve_dispatch", ROOT / "tools" / "issue_resolve_dispatch.py")
        ird = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ird)
        self.assertEqual(mod._STUB_LOG_MAX_BYTES, ird._STUB_LOG_MAX_BYTES)


class RouteHealthFoldTest(unittest.TestCase):
    """route_health_status() (#3035) — hermetic latest-per-route fold over a tmp
    route-health.jsonl: last probe age, typed failure class, cooldown remaining,
    and the exact recheck command; a failing route is suppressed without touching
    its healthy sibling on the same provider."""

    _RECHECK = ("fak dispatch route-health probe --base-url https://nim.example/v1 "
                "--model deepseek-chat --provider nim")

    def _write(self, runs: Path, rows: list[dict], *, tail: str = "") -> None:
        runs.mkdir(parents=True, exist_ok=True)
        (runs / "route-health.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows) + tail, encoding="utf-8")

    def test_missing_ledger_is_empty_fold(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            out = mod.route_health_status(Path(d) / "no-such-runs")
        self.assertEqual(out, {"probed": 0, "suppressed": 0, "routes": []})

    def test_suppresses_only_failing_route_and_carries_recheck(self) -> None:
        mod = load()
        now = 1_700_000_000
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write(runs, [
                # stale healthy row for the failing route: the later timeout row wins
                {"route": "nim/seat1/deepseek-chat", "class": "healthy",
                 "probed_at_unix": now - 600, "recheck": self._RECHECK},
                {"route": "nim/seat1/deepseek-chat", "class": "timeout",
                 "probed_at_unix": now - 120, "cooldown_until_unix": now + 480,
                 "recheck": self._RECHECK},
                {"route": "nim/seat1/kimi-k2", "class": "healthy",
                 "probed_at_unix": now - 60,
                 "recheck": "fak dispatch route-health probe --base-url "
                            "https://nim.example/v1 --model kimi-k2 --provider nim"},
            ])
            out = mod.route_health_status(runs, now_ts=now)
        self.assertEqual(out["probed"], 2)
        self.assertEqual(out["suppressed"], 1)
        rows = {r["route"]: r for r in out["routes"]}
        bad = rows["nim/seat1/deepseek-chat"]
        self.assertTrue(bad["suppressed"])
        self.assertEqual(bad["class"], "timeout")
        self.assertEqual(bad["probe_age_secs"], 120)
        self.assertEqual(bad["cooldown_remaining_secs"], 480)
        self.assertEqual(bad["recheck"], self._RECHECK)
        sibling = rows["nim/seat1/kimi-k2"]
        self.assertFalse(sibling["suppressed"])
        self.assertEqual(sibling["class"], "healthy")

    def test_elapsed_cooldown_and_corrupt_row_do_not_suppress(self) -> None:
        mod = load()
        now = 1_700_000_000
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write(runs, [
                {"route": "nim/seat1/deepseek-chat", "class": "rate_limited",
                 "probed_at_unix": now - 3600, "cooldown_until_unix": now - 60,
                 "recheck": self._RECHECK},
            ], tail="{not json}\n")
            out = mod.route_health_status(runs, now_ts=now)
        self.assertEqual(out["probed"], 1)
        self.assertEqual(out["suppressed"], 0)
        self.assertFalse(out["routes"][0]["suppressed"])

    def test_payload_threading_and_render_line(self) -> None:
        mod = load()
        rh = {"probed": 2, "suppressed": 1, "routes": [
            {"route": "nim/seat1/deepseek-chat", "class": "timeout",
             "probe_age_secs": 120, "suppressed": True,
             "cooldown_remaining_secs": 480, "recheck": self._RECHECK},
            {"route": "nim/seat1/kimi-k2", "class": "healthy",
             "probe_age_secs": 60, "suppressed": False,
             "cooldown_remaining_secs": 0, "recheck": ""},
        ]}
        p = build(mod, route_health=rh)
        self.assertEqual(p["route_health"], rh)
        text = mod.render(p)
        self.assertIn("routes    : 2 probed, 1 suppressed", text)
        self.assertIn("nim/seat1/deepseek-chat=timeout", text)
        self.assertIn("cooldown=480s left", text)
        self.assertIn("recheck   : " + self._RECHECK, text)

    def test_no_probes_emits_no_render_line(self) -> None:
        mod = load()
        p = build(mod, route_health={"probed": 0, "suppressed": 0, "routes": []})
        self.assertNotIn("routes    :", mod.render(p))


class BackendStubRateScanTest(unittest.TestCase):
    """backend_stub_rates() is a content sweep, not a backend-health sidecar read."""

    def _mk(self, runs: Path, issue: int, stamp: str, *, size: int, pid: int,
            backend: str, mtime: float) -> None:
        import os

        log = runs / f"resolve-{issue}-{stamp}.log"
        log.write_bytes(b"x" * size)
        (runs / f"resolve-{issue}-{stamp}.pid").write_text(str(pid), encoding="utf-8")
        (runs / f"resolve-{issue}-{stamp}.backend").write_text(backend, encoding="utf-8")
        os.utime(log, (mtime, mtime))

    def test_majority_stub_backend_is_flagged_without_health_sidecar(self) -> None:
        mod = load()
        now = 2_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # 8/9 opencode attempts are banner-only stubs, and no
            # backend-health-opencode.json sidecar exists. This is the #1276 blind
            # spot: absence of the sidecar must not render as proof of health.
            for i in range(8):
                self._mk(runs, 1200 + i, f"20260629-1659{i:02d}",
                         size=122, pid=16_000 + i, backend="opencode",
                         mtime=now - i)
            self._mk(runs, 1299, "20260629-170000", size=2048, pid=17_000,
                     backend="opencode", mtime=now - 20)
            self._mk(runs, 1300, "20260629-170001", size=2048, pid=18_000,
                     backend="claude", mtime=now - 30)

            out = mod.backend_stub_rates(runs, now_ts=now, alive=set())
            by_product = {r["product"]: r for r in out}
            self.assertFalse((runs / "backend-health-opencode.json").exists())
            self.assertEqual(by_product["opencode"]["total"], 9)
            self.assertEqual(by_product["opencode"]["stub"], 8)
            self.assertEqual(by_product["opencode"]["productive"], 1)
            self.assertTrue(by_product["opencode"]["majority_stub"])
            self.assertFalse(by_product["claude"]["majority_stub"])

    def test_live_stub_log_is_not_counted_as_backend_evidence_yet(self) -> None:
        mod = load()
        now = 2_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 1207, "20260629-165930", size=122, pid=16752,
                     backend="opencode", mtime=now)
            out = mod.backend_stub_rates(
                runs, now_ts=now, alive={16752},
                probe=lambda pid: {"alive": True,
                                   "cmdline": "opencode run resolve GitHub issue #1207"})
            self.assertEqual(out, [])


class BackendHookFailureScanTest(unittest.TestCase):
    """backend_hook_failures() makes a fully-unhooked backend visible (#1277)."""

    def _mk(self, runs: Path, issue: int, stamp: str, *, backend: str,
            text: str, mtime: float) -> None:
        import os

        log = runs / f"resolve-{issue}-{stamp}.log"
        log.write_text(text, encoding="utf-8")
        log.with_suffix(".backend").write_text(backend, encoding="utf-8")
        os.utime(log, (mtime, mtime))

    def test_all_recent_sessions_with_hook_failures_are_flagged(self) -> None:
        mod = load()
        now = 2_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(
                runs, 65, "20260629-115413", backend="codex", mtime=now - 10,
                text="hook: SessionStart Failed\nhook: PreToolUse Failed\n")
            self._mk(
                runs, 1176, "20260629-043016", backend="codex", mtime=now - 20,
                text="hook: UserPromptSubmit Failed\n")
            self._mk(
                runs, 1207, "20260629-173104", backend="opencode", mtime=now - 30,
                text="> build - glm\n")

            rows = mod.backend_hook_failures(runs, now_ts=now)
            by_product = {row["product"]: row for row in rows}

            self.assertEqual(by_product["codex"]["sessions"], 2)
            self.assertEqual(by_product["codex"]["sessions_with_hook_failures"], 2)
            self.assertEqual(by_product["codex"]["hook_failures"], 3)
            self.assertEqual(by_product["codex"]["failure_session_rate"], 1.0)
            self.assertTrue(by_product["codex"]["all_sessions_unhooked"])
            self.assertEqual(by_product["opencode"]["failure_session_rate"], 0.0)
            self.assertFalse(by_product["opencode"]["all_sessions_unhooked"])


class HookHealthPayloadTest(unittest.TestCase):
    def test_unhooked_backend_reaches_payload_reason_and_render(self) -> None:
        mod = load()
        hook_failures = [{
            "product": "codex",
            "lookback_min": 90,
            "sessions": 3,
            "sessions_with_hook_failures": 3,
            "failure_session_rate": 1.0,
            "hook_failures": 195,
            "evidence_logs": ["resolve-65-20260629-115413.log"],
            "all_sessions_unhooked": True,
        }]

        p = build(mod, hook_failures=hook_failures)

        self.assertEqual(p["hook_health"]["unhooked_count"], 1)
        self.assertEqual(p["hook_health"]["by_backend"], hook_failures)
        self.assertTrue(any("guard hook layer UNBOUND" in r for r in p["reasons"]))
        self.assertIn("hooks", mod.render(p))
        self.assertIn("codex=195 fail/3 sess (100%)", mod.render(p))


class SeatInventoryPayloadTest(unittest.TestCase):
    """#1799: build_payload threads the seat inventory (fleet_accounts.seat_pool
    output) straight through to the JSON payload and surfaces a summary reason line,
    covering all four operator-facing states: available / busy / cooling /
    unavailable-with-reason."""

    def _pool(self) -> dict:
        return {
            "schema": "fleet-seat-pool/1",
            "product": "claude",
            "total_seats": 4,
            "free_seats": 1,
            "leased_seats": 1,
            "blocked_seats": 2,
            "by_dispatch_state": {"available": 1, "busy": 1, "cooling": 1, "unavailable": 1},
            "depleted": False,
            "double_booked": [],
            "unbound_leases": [],
            "seats": [
                {"seat": "dir:.claude-free1", "tag": "free1", "account": ".claude-free1",
                 "product": "claude", "model_tier": 1, "available": True,
                 "state": "free", "dispatch_state": "available", "hold_reason": "",
                 "workers": []},
                {"seat": "dir:.claude-busy1", "tag": "busy1", "account": ".claude-busy1",
                 "product": "claude", "model_tier": 1, "available": True,
                 "state": "leased", "dispatch_state": "busy",
                 "hold_reason": "leased to w1", "workers": ["w1"]},
                {"seat": "dir:.claude-cool1", "tag": "cool1", "account": ".claude-cool1",
                 "product": "claude", "model_tier": 1, "available": False,
                 "state": "blocked", "dispatch_state": "cooling",
                 "hold_reason": "cooldown_until=9pm", "workers": []},
                {"seat": "dir:.claude-auth1", "tag": "auth1", "account": ".claude-auth1",
                 "product": "claude", "model_tier": 1, "available": False,
                 "state": "blocked", "dispatch_state": "unavailable",
                 "hold_reason": "auth_failed", "workers": []},
            ],
        }

    def test_seat_inventory_flows_into_payload_json(self) -> None:
        mod = load()
        p = build(mod, seat_inventory=self._pool())
        self.assertEqual(p["seat_inventory"]["total_seats"], 4)
        by_state = p["seat_inventory"]["by_dispatch_state"]
        self.assertEqual(by_state, {"available": 1, "busy": 1, "cooling": 1, "unavailable": 1})
        states = {s["tag"]: s["dispatch_state"] for s in p["seat_inventory"]["seats"]}
        self.assertEqual(states, {"free1": "available", "busy1": "busy",
                                  "cool1": "cooling", "auth1": "unavailable"})
        # unavailable/cooling seats carry a SPECIFIC reason, never the bare word
        reasons_by_tag = {s["tag"]: s["hold_reason"] for s in p["seat_inventory"]["seats"]}
        self.assertEqual(reasons_by_tag["cool1"], "cooldown_until=9pm")
        self.assertEqual(reasons_by_tag["auth1"], "auth_failed")
        self.assertNotIn("unavailable", reasons_by_tag["cool1"])
        self.assertNotEqual(reasons_by_tag["auth1"], "unavailable")

    def test_seat_inventory_summary_reason_line(self) -> None:
        mod = load()
        p = build(mod, seat_inventory=self._pool())
        self.assertTrue(any(
            "seat inventory" in r and "available=1" in r and "busy=1" in r
            and "cooling=1" in r and "unavailable=1" in r
            for r in p["reasons"]))

    def test_unattributed_live_from_preflight_is_visible_in_summary(self) -> None:
        mod = load()
        inv = mod.annotate_seat_inventory_from_preflight(
            self._pool(), {"seat": {"unattributed_live": 2}})
        self.assertEqual(inv["free_seats"], 0)
        self.assertEqual(inv["leased_seats"], 3)
        self.assertTrue(inv["depleted"])
        line = mod._seat_inventory_summary_line(inv)
        self.assertIn("slots free=0 leased=3", line)
        self.assertIn("unattributed_live=2", line)

    def test_auth_failed_seat_is_actionable_and_excluded_from_capacity(self) -> None:
        mod = load()
        p = build(mod, seat_inventory=self._pool())
        self.assertEqual(p["seat_inventory"]["free_seats"], 1)
        self.assertTrue(any(
            "auth_failed=1 [auth1]" in r and "next action: run `fak accounts status`" in r
            for r in p["reasons"]))

        rendered = mod.render(p)
        self.assertIn("auth_failed=1 [auth1]", rendered)
        self.assertIn("next action: run `fak accounts status`", rendered)

        slack = mod.slack_text(p)
        self.assertIn("*dispatch scheduler:* `READY_TO_GROW` (ACTION)", slack)
        self.assertIn("action: auth_failed=1 [auth1]", slack)

    def test_double_booked_seat_is_actionable(self) -> None:
        mod = load()
        pool = self._pool()
        pool["double_booked"] = [{
            "seat": "dir:opencode",
            "tag": "default",
            "workers": ["resolve-1", "resolve-2"],
            "session_cap": 1,
        }]
        p = build(mod, seat_inventory=pool)
        self.assertTrue(any(
            "double_booked=1 [default:2/1]" in r
            and "reap a dead/stale worker" in r
            for r in p["reasons"]))

        rendered = mod.render(p)
        self.assertIn("double_booked=1 [default:2/1]", rendered)

        slack = mod.slack_text(p)
        self.assertIn("*dispatch scheduler:* `READY_TO_GROW` (ACTION)", slack)
        self.assertIn("double_booked=1 [default:2/1]", slack)

    def test_seat_inventory_error_degrades_gracefully(self) -> None:
        mod = load()
        p = build(mod, seat_inventory={"_error": "boom"})
        self.assertTrue(p["ok"])  # informational fold only — never flips ok
        self.assertTrue(any("seat inventory unavailable: boom" in r for r in p["reasons"]))

    def test_seat_inventory_defaults_to_empty_dict_when_omitted(self) -> None:
        mod = load()
        p = build(mod)
        self.assertEqual(p["seat_inventory"], {})


class RenderMdTest(unittest.TestCase):
    """render_md is pure: a hand-built payload -> the committed-doc markdown."""

    def _payload(self, mod, **over):
        return build(mod, **over)

    def test_md_has_lane_table_and_closure_table(self) -> None:
        mod = load()
        md = mod.render_md(self._payload(mod), date="2026-06-21")
        self.assertIn("# Issue dispatch status — 2026-06-21", md)
        self.assertIn("## Backlog by lane", md)
        self.assertIn("| docs | 3 |", md)          # from backlog_ok(): docs has 3
        self.assertIn("| agent | 1 |", md)
        self.assertIn("## Closure honesty", md)
        self.assertIn("`closure_rate` = **0.8**", md)
        self.assertIn("| TRUE_RESOLVED | 8 |", md)

    def test_md_silent_section_lists_workers(self) -> None:
        mod = load()
        silent = [{"issue": 465, "stamp": "20260621-232003",
                   "log": "resolve-465-20260621-232003.log", "pid": 39688}]
        md = mod.render_md(self._payload(mod, silent=silent), date="2026-06-21")
        self.assertIn("## Workers that produced nothing", md)
        self.assertIn("| #465 | 20260621-232003 |", md)

    def test_md_silent_section_shows_kind_and_bytes(self) -> None:
        mod = load()
        silent = [{"issue": 1207, "stamp": "20260629-165930", "kind": "stub",
                   "size": 122, "log": "resolve-1207-20260629-165930.log", "pid": 16752}]
        md = mod.render_md(self._payload(mod, silent=silent), date="2026-06-29")
        self.assertIn("| kind | bytes |", md)
        self.assertIn("| #1207 | 20260629-165930 | stub | 122 |", md)

    def test_md_backend_stub_rate_flags_majority_without_dead_sidecar(self) -> None:
        mod = load()
        stub_rate = [{
            "product": "opencode",
            "lookback_min": 90,
            "total": 9,
            "productive": 1,
            "stub": 8,
            "stub_rate": 0.889,
            "majority_stub": True,
            "evidence_logs": ["resolve-1207-20260629-165930.log"],
        }]
        p = self._payload(mod, backend_health=[], backend_stub_rate=stub_rate)
        self.assertTrue(any("backend stub-rate majority-stub" in r
                            for r in p["reasons"]))
        md = mod.render_md(p, date="2026-06-29")
        self.assertIn("Backend stub-rate", md)
        self.assertIn("No `backend-health-*.json` sidecar", md)
        self.assertNotIn("All backends healthy", md)
        self.assertIn("| opencode | 90m | 9 | 1 | 8 | 0.889 | **MAJORITY_STUB** |", md)

    def test_md_hook_health_lists_unhooked_backend(self) -> None:
        mod = load()
        hook_failures = [{
            "product": "codex",
            "lookback_min": 90,
            "sessions": 3,
            "sessions_with_hook_failures": 3,
            "failure_session_rate": 1.0,
            "hook_failures": 195,
            "evidence_logs": ["resolve-65-20260629-115413.log"],
            "all_sessions_unhooked": True,
        }]
        md = mod.render_md(self._payload(mod, hook_failures=hook_failures),
                           date="2026-06-29")
        self.assertIn("## Hook health", md)
        self.assertIn("| codex | 90m | 3 | 3 | 1.0 | 195 | **UNHOOKED** |", md)

    def test_md_lists_active_lane_leases(self) -> None:
        mod = load()
        leases = {
            "source": "refs/fak/locks",
            "candidate_source_available": True,
            "candidate_count": 1,
            "active_count": 2,
            "expired_count": 1,
            "blocking_count": 1,
            "active": [
                {
                    "id": "resolve-tools-1762",
                    "lane": "tools",
                    "holder": "peer-a",
                    "tree": ["tools/**"],
                    "age_min": 10.0,
                    "ttl_seconds": 3600,
                    "blocks_candidate": True,
                    "blocking_candidates": [{"issue": 1762, "lane": "tools"}],
                },
                {
                    "id": "resolve-model-1700",
                    "lane": "model",
                    "holder": "peer-b",
                    "tree": ["internal/model/**"],
                    "age_min": 5.0,
                    "ttl_seconds": 3600,
                    "blocks_candidate": False,
                    "blocking_candidates": [],
                },
            ],
            "expired": [{"id": "resolve-docs", "status": "EXPIRED"}],
        }
        md = mod.render_md(self._payload(mod, leases=leases), date="2026-06-30")
        self.assertIn("## Active lane leases", md)
        self.assertIn("| `resolve-tools-1762` | tools | 10m | 3600s | unknown | blocks #1762 |", md)
        self.assertIn("| `resolve-model-1700` | model | 5m | 3600s | unknown | no candidate |", md)
        self.assertIn("1 expired lease record", md)

    def test_md_lists_worker_lease_cross_check_buckets(self) -> None:
        mod = load()
        worker_leases = {
            "available": True,
            "clean_count": 1,
            "orphan_process_count": 1,
            "orphan_lease_count": 1,
            "clean": [{
                "worker": {"worker": "resolve-1765-20260701-120000", "issue": 1765, "pid": 111},
                "lease": {"id": "resolve-tools-1765"},
            }],
            "orphan_process": [{
                "worker": {"worker": "resolve-1770-20260701-120001", "issue": 1770, "pid": 222,
                           "lease_id": "resolve-docs-1770"},
                "reason": "missing active dispatch lease",
            }],
            "orphan_lease": [{
                "lease": {"id": "resolve-model-1700", "lane": "model", "holder": "peer-b"},
                "reason": "active lease has no local live worker sidecar",
            }],
        }
        md = mod.render_md(self._payload(mod, worker_leases=worker_leases),
                           date="2026-07-01")
        self.assertIn("## Worker / lease cross-check", md)
        self.assertIn("unmatched-live-lease=1", md)
        self.assertIn("not automatically safe to reap", md)
        self.assertIn("| `resolve-1765-20260701-120000` | #1765 | 111 | `resolve-tools-1765` |", md)
        self.assertIn("| `resolve-1770-20260701-120001` | #1770 | 222 | `resolve-docs-1770` | missing active dispatch lease |", md)
        self.assertIn("| `resolve-model-1700` | model | `peer-b` | active lease has no local live worker sidecar |", md)

    def test_md_silent_section_says_none_when_clean(self) -> None:
        mod = load()
        md = mod.render_md(self._payload(mod, silent=[]), date="2026-06-21")
        self.assertIn("## Workers that produced nothing", md)
        self.assertIn("None — every spawned worker", md)

    def test_md_handles_na_folds(self) -> None:
        mod = load()
        p = build(mod, backlog={"_skipped": "fast"}, closure={"_skipped": "fast"}, fast=True)
        md = mod.render_md(p, date="2026-06-21")
        self.assertIn("Backlog n/a", md)
        self.assertIn("Closure audit n/a", md)


class SlackPostTest(unittest.TestCase):
    """The --slack wiring: slack_text is pure; post_to_slack resolves + posts via the
    injected transport (no network, no real token/channel)."""

    SLACK_KEYS = ("FAK_DISPATCH_TOKEN", "FAK_DISPATCH_CHANNEL", "FAK_SCOREBOARD_TOKEN")

    def _clear_env(self):
        import os
        saved = {k: os.environ.pop(k, None) for k in self.SLACK_KEYS}
        self.addCleanup(self._restore_env, saved)
        cwd = os.getcwd()
        tmp = tempfile.TemporaryDirectory()
        os.chdir(tmp.name)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, cwd)

    def _restore_env(self, saved):
        import os
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_slack_text_has_headline_and_compact_card(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"))
        text = mod.slack_text(p)
        self.assertIn("*dispatch scheduler:* `READY_TO_GROW` (healthy)", text)
        self.assertIn("plane: scheduler/backlog, not session health", text)
        self.assertIn("worker slots 0/2 active", text)
        self.assertNotIn("S/N self-score", text)
        self.assertNotIn("```", text)         # Slack uses compact mrkdwn, not a terminal box.

    def test_slack_text_buckets_expected_auto_and_action(self) -> None:
        mod = load()
        p = build(
            mod,
            pre=pre("SPAWN_OK", live=0),
            sup={"verdict": "PLAN_SURFACE_EMPTY", "supervise": {"target": 0, "alive": 0}},
            weekly_cap={"reset_text": "tomorrow", "until": "2026-06-30T00:00:00Z"},
            backend_health=[{"product": "claude", "abandoned_lane": "docs", "reprobe_min": 30}],
            backend_stub_rate=[{"product": "claude", "total": 14, "stub": 14, "majority_stub": True}],
            hook_failures=[{
                "product": "codex",
                "sessions": 2,
                "sessions_with_hook_failures": 2,
                "hook_failures": 63,
                "all_sessions_unhooked": True,
            }],
        )
        text = mod.slack_text(p)
        self.assertIn("(ACTION)", text)
        self.assertIn("expected: worker-a weekly-capped until tomorrow", text)
        self.assertIn("supervisor PLAN_SURFACE_EMPTY: expected", text)
        self.assertIn("auto-solving: claude held dead; lane docs reallocated", text)
        self.assertIn("evidence 14/14 recent logs are stubs", text)
        self.assertIn("action: codex guard hooks unbound", text)
        # The dead-backend line carries the stub evidence, so it must not also emit
        # a second majority-stub action for the same backend.
        self.assertNotIn("claude majority-stub", text)

    def test_slack_bucket_helpers_preserve_category_and_message_order(self) -> None:
        mod = load()
        payload = {
            "verdict": "AT_CAP",
            "ok": True,
            "weekly_cap": {"reset_text": "tomorrow"},
            "dispatcher": {
                "account": {"tag": "worker-a"},
                "watchdog": {"installed": False},
                "host_safe": True,
                "preflight_verdict": "REFUSE_AT_CAP",
                "limiter": {},
            },
            "supervisor": {"verdict": "READY"},
            "workers": {"silent_count": 1, "silent": [{"issue": 7}]},
            "low_yield": {"lanes": [{
                "lane": "tools", "verdict": "LOW_YIELD", "turns": 9, "sessions": 2,
            }]},
            "backend_health": {
                "dead": [{"product": "claude", "abandoned_lane": "docs"}],
                "stub_rate": [],
            },
        }
        gate = {
            "expected": [
                "worker-a weekly-capped until tomorrow; scheduler waits",
                "at configured worker-slot cap",
            ],
            "auto-solving": [],
            "action": ["always-on watchdog not installed; register FleetIssueDispatch"],
        }
        health = {
            "expected": [],
            "auto-solving": [
                "1 no-output worker(s) skipped by cooldown (#7); inspect only if the same issue repeats",
                "claude held dead; lane docs reallocated",
            ],
            "action": [
                "lane tools low-yield: 9 turns / 2 session(s), 0 ancestry-closes; re-scope or exclude the lane",
            ],
        }

        self.assertEqual(mod._dispatch_gate_slack_buckets(payload), gate)
        self.assertEqual(mod._dispatch_health_slack_buckets(payload), health)
        self.assertEqual(mod._dispatch_slack_buckets(payload), {
            name: gate[name] + health[name] for name in gate
        })

    def test_slack_text_explains_stalled_ok(self) -> None:
        mod = load()
        p = build(
            mod,
            pre=pre("SPAWN_OK", live=0),
            sup={"verdict": "PLAN_SURFACE_EMPTY", "supervise": {"target": 0, "alive": 0}},
        )
        p["verdict"] = "STALLED"
        p["ok"] = True
        text = mod.slack_text(p)
        self.assertIn("*dispatch scheduler:* `STALLED` (expected)", text)
        self.assertIn("scheduler liveness says STALLED but gate marks it ok", text)
        self.assertNotIn("*dispatch status:* `STALLED` (ok)", text)

    def test_post_to_slack_posts_via_injected_transport(self) -> None:
        import json as _json
        import os
        mod = load()
        self._clear_env()
        os.environ["FAK_SCOREBOARD_TOKEN"] = "xoxb-test-tok"
        os.environ["FAK_DISPATCH_CHANNEL"] = "C0DISPATCH"
        calls = []

        def transport(url, body, headers, timeout):
            calls.append({"url": url, "body": _json.loads(body.decode("utf-8")),
                          "auth": headers.get("Authorization")})
            return 200, _json.dumps({"ok": True, "ts": "9.9", "channel": "C0DISPATCH"})

        p = build(mod, pre=pre("SPAWN_OK"))
        verdict = mod.post_to_slack(p, transport=transport)
        self.assertTrue(verdict["posted"])
        self.assertEqual(verdict["channel"], "C0DISPATCH")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["auth"], "Bearer xoxb-test-tok")
        self.assertIn("READY_TO_GROW", calls[0]["body"]["text"])
        self.assertNotIn("S/N self-score", calls[0]["body"]["text"])

    def test_post_to_slack_dry_run_does_not_call_transport(self) -> None:
        import os
        mod = load()
        self._clear_env()
        os.environ["FAK_SCOREBOARD_TOKEN"] = "xoxb-test-tok"
        os.environ["FAK_DISPATCH_CHANNEL"] = "C0DISPATCH"
        calls = []

        def transport(url, body, headers, timeout):
            calls.append(1)
            return 200, "{}"

        p = build(mod, pre=pre("SPAWN_OK"))
        verdict = mod.post_to_slack(p, dry_run=True, transport=transport)
        self.assertFalse(verdict["posted"])
        self.assertEqual(verdict["skipped"], "dry-run")
        self.assertEqual(calls, [])

    def test_post_to_slack_skips_cleanly_without_channel(self) -> None:
        import os
        mod = load()
        self._clear_env()
        os.environ["FAK_SCOREBOARD_TOKEN"] = "xoxb-test-tok"
        # no channel anywhere -> a clean skip, never a crash
        p = build(mod, pre=pre("SPAWN_OK"))
        verdict = mod.post_to_slack(p, channel="")
        self.assertFalse(verdict["posted"])
        self.assertIn("no channel", verdict["skipped"])


class GuardCoverageScanTest(unittest.TestCase):
    """guard_coverage() folds the per-session .dispatch-runs/guard-audit/*.jsonl
    decision journals into a coverage + decision-mix rollup — hermetic over a tmp dir."""

    def _journal(self, audit_dir: Path, name: str, rows: list[str],
                 *, mtime: float | None = None) -> None:
        import os
        audit_dir.mkdir(parents=True, exist_ok=True)
        jp = audit_dir / name
        jp.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        if mtime is not None:
            os.utime(jp, (mtime, mtime))

    def test_missing_dir_is_zeroed_and_marked_absent(self) -> None:
        mod = load()
        out = mod.guard_coverage(Path("does-not-exist-xyz"))
        self.assertFalse(out["dir_present"])
        self.assertEqual(out["sessions"], 0)
        self.assertEqual(out["rows"], 0)

    def test_counts_sessions_rows_and_decision_mix(self) -> None:
        mod = load()
        now = 3_000_000.0
        with tempfile.TemporaryDirectory() as d:
            audit = Path(d) / mod.GUARD_AUDIT_DIRNAME
            # Session A: an allow (DECIDE), a deny, a quarantine.
            self._journal(audit, "gateway-claude-1-aaaa.jsonl", [
                '{"seq":1,"kind":"DECIDE","verdict":"ALLOW"}',
                '{"seq":2,"kind":"DENY","verdict":"DENY"}',
                '{"seq":3,"kind":"QUARANTINE"}',
            ], mtime=now)
            # Session B: a result-deny + a vDSO hit (recent).
            self._journal(audit, "recall-claude-2-bbbb.jsonl", [
                '{"seq":1,"kind":"RESULT_DENY"}',
                '{"seq":2,"kind":"VDSO_HIT"}',
            ], mtime=now)
            # Session C: empty (booted under guard, no adjudicated call) and OLD.
            self._journal(audit, "docs-claude-3-cccc.jsonl", [],
                          mtime=now - 10 * 24 * 3600)
            out = mod.guard_coverage(Path(d), now_ts=now)
        self.assertTrue(out["dir_present"])
        self.assertEqual(out["sessions"], 3)
        self.assertEqual(out["rows"], 5)
        self.assertEqual(out["empty_sessions"], 1)
        self.assertEqual(out["denied"], 2)        # DENY + RESULT_DENY
        self.assertEqual(out["quarantined"], 1)
        self.assertEqual(out["by_kind"]["DECIDE"], 1)
        self.assertEqual(out["by_kind"]["VDSO_HIT"], 1)
        # Only A and B fall in the recent window; C is 10 days old.
        self.assertEqual(out["recent_sessions"], 2)
        self.assertEqual(out["recent_rows"], 5)
        self.assertIn("gateway-claude-1-aaaa.jsonl", out["evidence"])

    def test_malformed_line_is_bucketed_not_crashing(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            audit = Path(d) / mod.GUARD_AUDIT_DIRNAME
            self._journal(audit, "x-claude-9-dddd.jsonl",
                          ['{"kind":"DECIDE"}', 'not json at all'])
            out = mod.guard_coverage(Path(d))
        self.assertEqual(out["rows"], 2)
        self.assertEqual(out["by_kind"].get("MALFORMED"), 1)

    def test_repeated_allow_none_rows_do_not_become_livelock_candidates(self) -> None:
        mod = load()
        now = 3_000_000.0
        allow = ('{"kind":"DECIDE","verdict":"ALLOW","tool":"Read","reason":"NONE",'
                 '"args_digest":"allowdigest"}')
        quarantine = ('{"kind":"QUARANTINE","verdict":"QUARANTINE","tool":"tool_result",'
                      '"reason":"SECRET_EXFIL","args_digest":"quardigest"}')
        with tempfile.TemporaryDirectory() as d:
            audit = Path(d) / mod.GUARD_AUDIT_DIRNAME
            self._journal(audit, "loop-claude-1-aaaa.jsonl",
                          [allow, allow, allow] + [quarantine] * 10,
                          mtime=now)
            out = mod.guard_coverage(Path(d), now_ts=now)
        candidates = out["livelock_candidates"]
        self.assertFalse(any(c["tool"] == "Read" for c in candidates))
        self.assertTrue(any(c["tool"] == "tool_result" and c["count"] == 10
                            for c in candidates))

    def test_non_candidate_rows_break_livelock_run_length(self) -> None:
        mod = load()
        now = 3_000_000.0
        allow = ('{"kind":"DECIDE","verdict":"ALLOW","tool":"Read","reason":"NONE",'
                 '"args_digest":"allowdigest"}')
        quarantine = ('{"kind":"QUARANTINE","verdict":"QUARANTINE","tool":"tool_result",'
                      '"reason":"SECRET_EXFIL","args_digest":"quardigest"}')
        with tempfile.TemporaryDirectory() as d:
            audit = Path(d) / mod.GUARD_AUDIT_DIRNAME
            rows = []
            for _ in range(10):
                rows.extend([quarantine, allow])
            self._journal(audit, "loop-claude-1-aaaa.jsonl", rows, mtime=now)
            out = mod.guard_coverage(Path(d), now_ts=now)
        candidates = out["livelock_candidates"]
        top = next(c for c in candidates if c["tool"] == "tool_result")
        self.assertEqual(top["count"], 10)
        self.assertEqual(top["longest_run"], 1)


class GuardEmptyCauseSplitTest(unittest.TestCase):
    """`empty=N` on the guard card is split by a CLOSED cause vocabulary (#5862 dc-2).

    An empty guard session used to be one opaque number, so an operator could not tell a
    fleet parked behind a provider wall (benign — the supervisor resumes the same
    command) from one silently failing to spawn. Hermetic over a tmp .dispatch-runs."""

    def _corpus(self, d: Path, *, journal: str, rows: list[str],
                log_tail: str | None = None, at: float = 3_000_000.0) -> None:
        import os
        audit = d / load().GUARD_AUDIT_DIRNAME
        audit.mkdir(parents=True, exist_ok=True)
        jp = audit / journal
        jp.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        os.utime(jp, (at, at))
        if log_tail is not None:
            lp = d / "resolve-1-20260807-000000.log"
            # Name the journal in the log the way the guard's banner does — that string
            # IS the join key.
            lp.write_text(f"  audit log  : {jp}\n{log_tail}\n", encoding="utf-8")
            os.utime(lp, (at, at))

    def test_park_on_a_provider_wall_is_named_not_opaque(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            self._corpus(Path(d), journal="model-claude-1-aaaa.jsonl", rows=[],
                         log_tail=("fak guard: goal parked outside active context budget "
                                   "until 1786402799; reason=LONG_RETRY_AFTER\n"
                                   "── guard · audit ──\n"))
            out = mod.guard_coverage(Path(d), now_ts=3_000_000.0)
        self.assertEqual(out["empty_sessions"], 1)
        self.assertEqual(out["empty_by_cause"],
                         {mod._GUARD_EMPTY_PROVIDER_QUOTA_WALL: 1})

    def test_child_that_never_launched_is_split_from_the_park(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            # A partial epilogue is present, as it is on the real spawn failures: the
            # spawn marker must still win, or the fleet's only "did not launch" signal
            # is booked as a clean exit.
            self._corpus(Path(d), journal="tools-claude-2-bbbb.jsonl", rows=[],
                         log_tail=("── guard · audit ──\n"
                                   'fak guard: could not run "claude": snapshot generated '
                                   "child config: The system cannot find the path specified.\n"))
            out = mod.guard_coverage(Path(d), now_ts=3_000_000.0)
        self.assertEqual(out["empty_by_cause"], {mod._GUARD_EMPTY_SPAWN_FAILED: 1})

    def test_interactive_journal_is_not_counted_as_a_dispatch_spawn(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            # guardDefaultAuditPath's name: an attended `fak guard`, which consumed no
            # dispatch slot, no account rotation and no spawn.
            self._corpus(Path(d), journal="interactive-4242-16daf1ca84d5.jsonl", rows=[])
            out = mod.guard_coverage(Path(d), now_ts=3_000_000.0)
        self.assertEqual(out["empty_by_cause"], {mod._GUARD_EMPTY_INTERACTIVE: 1})

    def test_terminal_witness_row_is_not_a_decision_and_names_the_cause(self) -> None:
        """The regression #5862's own producer fix would otherwise cause.

        ae47d2911d makes a parked teardown write a CHILD_EXIT row. Counting that row as
        a decision would drop the session out of `empty` entirely — the number would
        deflate while explaining nothing. It must stay counted AND carry its reason."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            self._corpus(Path(d), journal="session-claude-3-cccc.jsonl", rows=[
                '{"seq":1,"kind":"CHILD_EXIT","reason":"CLEAN_EXIT","exit_code":0}',
            ])
            out = mod.guard_coverage(Path(d), now_ts=3_000_000.0)
        self.assertEqual(out["rows"], 1)
        self.assertEqual(out["zero_row_sessions"], 0)   # the file is NOT empty ...
        self.assertEqual(out["empty_sessions"], 1)      # ... but it adjudicated nothing
        self.assertEqual(out["empty_by_cause"], {"clean_exit": 1})

    def test_a_real_decision_is_never_counted_empty(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            self._corpus(Path(d), journal="docs-claude-4-dddd.jsonl", rows=[
                '{"seq":1,"kind":"DECIDE","verdict":"ALLOW"}',
                '{"seq":2,"kind":"CHILD_EXIT","reason":"CLEAN_EXIT"}',
            ])
            out = mod.guard_coverage(Path(d), now_ts=3_000_000.0)
        self.assertEqual(out["empty_sessions"], 0)
        self.assertEqual(out["empty_by_cause"], {})

    def test_unrecognised_witness_reason_falls_through_to_better_evidence(self) -> None:
        """"unknown" is the last resort, never a shortcut.

        Older CHILD_CRASH rows predate the Reason field; stamping them "unknown" throws
        away the name/log evidence that still explains them."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            self._corpus(Path(d), journal="interactive-99-16daf1ca84d5.jsonl", rows=[
                '{"seq":1,"kind":"CHILD_CRASH"}',
            ])
            out = mod.guard_coverage(Path(d), now_ts=3_000_000.0)
        self.assertEqual(out["empty_by_cause"], {mod._GUARD_EMPTY_INTERACTIVE: 1})

    def test_no_joinable_log_is_missing_artifact_not_a_fabricated_cause(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            self._corpus(Path(d), journal="repair-claude-5-eeee.jsonl", rows=[])
            out = mod.guard_coverage(Path(d), now_ts=3_000_000.0)
        self.assertEqual(out["empty_by_cause"], {mod._GUARD_EMPTY_MISSING_LOG: 1})

    def test_log_outside_the_mtime_window_is_not_joined(self) -> None:
        """The join is bounded, so the card cannot become a full-corpus scan."""
        mod = load()
        import os
        with tempfile.TemporaryDirectory() as d:
            self._corpus(Path(d), journal="model-claude-6-ffff.jsonl", rows=[],
                         log_tail="reason=LONG_RETRY_AFTER\n")
            lp = Path(d) / "resolve-1-20260807-000000.log"
            far = 3_000_000.0 + mod._GUARD_EMPTY_LOG_WINDOW_S * 3
            os.utime(lp, (far, far))
            out = mod.guard_coverage(Path(d), now_ts=3_000_000.0)
        self.assertEqual(out["empty_by_cause"], {mod._GUARD_EMPTY_MISSING_LOG: 1})

    def test_causes_are_rendered_on_every_guard_surface(self) -> None:
        mod = load()
        bit = mod._guard_empty_cause_bit(
            {"empty_by_cause": {mod._GUARD_EMPTY_PROVIDER_QUOTA_WALL: 25,
                                mod._GUARD_EMPTY_INTERACTIVE: 21}})
        self.assertIn("provider_quota_wall=25", bit)
        self.assertIn("interactive_not_dispatch=21", bit)
        # A fleet with nothing to explain grows no all-zero bucket list.
        self.assertEqual(mod._guard_empty_cause_bit({"empty_by_cause": {}}), "")

    def test_cause_vocabulary_does_not_drift_from_the_go_side(self) -> None:
        """The closed vocabulary is a literal COPY of Go constants (Python cannot import
        Go). Parse those constants back OUT of the Go source so a rename on either side
        fails CI instead of silently producing a cause the other half cannot name.

        This matters concretely: internal/dispatchconservation folds any reason outside
        its registered set to "unknown", so a cause string invented only here would
        vanish on the Go side rather than error."""
        mod = load()
        root = Path(__file__).resolve().parents[1]
        cons = (root / "internal" / "dispatchconservation" / "conservation.go")
        crash = (root / "internal" / "journal" / "crash.go")
        if not cons.is_file() or not crash.is_file():
            self.skipTest("Go sources absent")
        cons_src = cons.read_text(encoding="utf-8", errors="replace")
        crash_src = crash.read_text(encoding="utf-8", errors="replace")
        # 1. Every log-derived cause is a dispatchconservation Reason* constant AND is
        #    registered in noCommitReasons (an unregistered one folds to "unknown").
        for cause in (mod._GUARD_EMPTY_PROVIDER_QUOTA_WALL,
                      mod._GUARD_EMPTY_SPAWN_FAILED,
                      mod._GUARD_EMPTY_DIED_BEFORE_EPILOGUE,
                      mod._GUARD_EMPTY_CLEAN_EXIT_NO_COMMIT,
                      mod._GUARD_EMPTY_MISSING_LOG,
                      mod._GUARD_EMPTY_UNKNOWN):
            self.assertIn(f'"{cause}"', cons_src,
                          f"{cause} is not a reason internal/dispatchconservation knows")
        # 2. Every marker literal is byte-identical to the Go one it copies.
        for marker in (mod._GUARD_PARK_MARKER, mod._GUARD_SPAWN_FAILED_MARKER,
                       mod._GUARD_EPILOGUE_MARKER):
            self.assertIn(f'"{marker}"', cons_src,
                          f"marker {marker!r} drifted from conservation.go")
        # 3. Every witness reason is a Crash* constant in internal/journal/crash.go
        #    (compared upper-cased: the journal stamps them upper, the card renders
        #    them lower).
        for reason in mod._GUARD_EMPTY_WITNESS_REASONS:
            if reason == "crash_restart_exhausted":
                continue  # cmd/fak/guard_crash_restart.go, checked below
            self.assertIn(f'"{reason.upper()}"', crash_src,
                          f"witness reason {reason} is not in the journal's closed set")
        restart = (root / "cmd" / "fak" / "guard_crash_restart.go")
        if restart.is_file():
            self.assertIn('"CRASH_RESTART_EXHAUSTED"',
                          restart.read_text(encoding="utf-8", errors="replace"))
        # 4. The terminal kinds are the journal's own Kind constants.
        for kind in mod._GUARD_TERMINAL_KINDS:
            self.assertIn(f'"{kind}"', crash_src)


class GuardCoverageFoldTest(unittest.TestCase):
    """build_payload + render surface the guard rollup (payload section, reason, card)."""

    def _guard(self, **over) -> dict:
        g = {"dir_present": True, "sessions": 2, "recent_sessions": 2,
             "empty_sessions": 0, "rows": 5, "recent_rows": 5,
             "by_kind": {"DECIDE": 3, "DENY": 1, "QUARANTINE": 1},
             "denied": 1, "quarantined": 1, "lookback_min": 90,
             "evidence": ["gateway-claude-1-aaaa.jsonl"]}
        g.update(over)
        return g

    def test_guard_section_and_reason_when_decisions_present(self) -> None:
        mod = load()
        p = build(mod, guard=self._guard())
        self.assertEqual(p["guard"]["sessions"], 2)
        self.assertTrue(any("fak guard witnessed 5 kernel decision" in r for r in p["reasons"]))
        self.assertIn("guard", mod.render(p))
        self.assertIn("DENY=1", mod.render(p))

    def test_child_crashes_are_rendered_when_present(self) -> None:
        mod = load()
        p = build(mod, guard=self._guard(
            by_kind={"DECIDE": 3, "DENY": 1, "QUARANTINE": 1, "CHILD_CRASH": 2},
        ))
        self.assertTrue(any("2 child crashes" in r for r in p["reasons"]))
        self.assertIn("CRASH=2", mod.render(p))

    def test_livelock_candidate_is_rendered_when_present(self) -> None:
        mod = load()
        p = build(mod, guard=self._guard(livelock_candidates=[{
            "file": "loop-claude-1-aaaa.jsonl",
            "tool": "Read",
            "reason": "NONE",
            "digest": "allowdigest",
            "count": 3,
            "longest_run": 3,
        }]))
        self.assertTrue(any("loop candidate:" in r for r in p["reasons"]))
        self.assertIn("loop=Read/NONE x3 run=3", mod.render(p))

    def test_empty_sessions_reason_when_no_decisions(self) -> None:
        mod = load()
        p = build(mod, guard=self._guard(rows=0, recent_rows=0, empty_sessions=2,
                                         by_kind={}, denied=0, quarantined=0))
        self.assertTrue(any("recorded 0 decisions" in r for r in p["reasons"]))

    def test_guard_defaults_to_empty_when_omitted(self) -> None:
        mod = load()
        p = build(mod)  # build() does not pass guard -> defaults to None -> {}
        self.assertEqual(p["guard"], {})
        self.assertFalse(any("fak guard witnessed" in r for r in p["reasons"]))


class LowYieldLanesScanTest(unittest.TestCase):
    """low_yield_lanes() binds turns-spent to ancestry-closes per lane (#2062) —
    hermetic: a tmp runs-dir of resolve logs + an injected closes_counter (no git)."""

    NOW = 2_000_000.0

    def _mk(self, runs: Path, issue: int, stamp: str, *, lane: str, turns: int,
            mtime: float, tree: list[str] | None = None) -> None:
        import json as _json
        import os
        log = runs / f"resolve-{issue}-{stamp}.log"
        lines = [f"# fak-spawn {stamp} issue={issue} lane={lane} backend=claude"]
        lines += [f"fak-turn trace=guard ok prov={i}k tok" for i in range(turns)]
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.utime(log, (mtime, mtime))
        if tree is not None:
            (runs / f"resolve-{issue}-{stamp}.lease-tree.json").write_text(
                _json.dumps(tree), encoding="utf-8")

    def test_high_turn_zero_close_lane_is_low_yield(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 2062, "20260704-120000", lane="tools", turns=44,
                     mtime=self.NOW - 60, tree=["tools/**"])
            out = mod.low_yield_lanes(
                runs, closes_counter=lambda lane, tree: 0, now_ts=self.NOW)
        self.assertEqual(out["low_yield_count"], 1)
        row = out["lanes"][0]
        self.assertEqual(row["lane"], "tools")
        self.assertEqual(row["turns"], 44)
        self.assertEqual(row["sessions"], 1)
        self.assertEqual(row["closes"], 0)
        self.assertTrue(row["tree_known"])
        self.assertEqual(row["verdict"], "LOW_YIELD")
        self.assertEqual(row["tree"], ["tools"])  # normalized pathspec

    def test_lane_with_a_close_is_ok(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 2100, "20260704-120500", lane="docs", turns=60,
                     mtime=self.NOW - 60, tree=["docs/**"])
            out = mod.low_yield_lanes(
                runs, closes_counter=lambda lane, tree: 3, now_ts=self.NOW)
        self.assertEqual(out["low_yield_count"], 0)
        self.assertEqual(out["lanes"][0]["verdict"], "OK")
        self.assertEqual(out["lanes"][0]["closes"], 3)

    def test_below_floor_lane_never_flags_and_skips_the_git_join(self) -> None:
        mod = load()
        calls: list[str] = []

        def counter(lane, tree):
            calls.append(lane)
            return 0

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 2200, "20260704-121000", lane="model", turns=10,
                     mtime=self.NOW - 60, tree=["internal/model/**"])
            out = mod.low_yield_lanes(runs, closes_counter=counter, now_ts=self.NOW)
        self.assertEqual(out["low_yield_count"], 0)
        self.assertEqual(out["lanes"][0]["verdict"], "OK")
        self.assertIsNone(out["lanes"][0]["closes"])
        self.assertEqual(calls, [])  # a below-floor lane never pays the git join

    def test_candidate_without_known_tree_is_not_flagged(self) -> None:
        # High turns but no lease-tree sidecar and no lane_trees entry: we cannot
        # join to the lane's tree, so we never fabricate a LOW_YIELD verdict.
        mod = load()
        calls: list[str] = []

        def counter(lane, tree):
            calls.append(lane)
            return 0

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 2300, "20260704-121500", lane="mystery", turns=80,
                     mtime=self.NOW - 60)  # no tree sidecar
            out = mod.low_yield_lanes(runs, closes_counter=counter, now_ts=self.NOW)
        row = out["lanes"][0]
        self.assertEqual(row["turns"], 80)
        self.assertFalse(row["tree_known"])
        self.assertEqual(row["verdict"], "OK")
        self.assertIsNone(row["closes"])
        self.assertEqual(calls, [])

    def test_lane_trees_map_overrides_sidecar_for_the_join(self) -> None:
        mod = load()
        seen: list[tuple[str, tuple]] = []

        def counter(lane, tree):
            seen.append((lane, tuple(tree)))
            return 0

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # No sidecar; the router's lane→tree map supplies the tree.
            self._mk(runs, 2400, "20260704-122000", lane="bench", turns=50,
                     mtime=self.NOW - 60)
            out = mod.low_yield_lanes(
                runs, closes_counter=counter,
                lane_trees={"bench": ["internal/bench/**"]}, now_ts=self.NOW)
        self.assertEqual(out["lanes"][0]["verdict"], "LOW_YIELD")
        self.assertEqual(out["lanes"][0]["tree"], ["internal/bench"])
        self.assertEqual(seen, [("bench", ("internal/bench",))])

    def test_sessions_aggregate_turns_across_the_window(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 2500, "20260704-120000", lane="tools", turns=25,
                     mtime=self.NOW - 120, tree=["tools/**"])
            self._mk(runs, 2501, "20260704-121000", lane="tools", turns=20,
                     mtime=self.NOW - 60, tree=["tools/**"])
            out = mod.low_yield_lanes(
                runs, closes_counter=lambda lane, tree: 0, now_ts=self.NOW)
        row = out["lanes"][0]
        self.assertEqual(row["sessions"], 2)
        self.assertEqual(row["turns"], 45)            # 25 + 20 >= floor
        self.assertEqual(row["max_session_turns"], 25)
        self.assertEqual(row["verdict"], "LOW_YIELD")

    def test_stale_logs_outside_the_window_are_ignored(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 2600, "20260701-120000", lane="tools", turns=90,
                     mtime=self.NOW - 10 * 3600, tree=["tools/**"])  # 10h old
            out = mod.low_yield_lanes(
                runs, closes_counter=lambda lane, tree: 0, now_ts=self.NOW,
                lookback_min=180)
        self.assertEqual(out["lanes"], [])
        self.assertEqual(out["low_yield_count"], 0)

    def test_missing_runs_dir_is_empty(self) -> None:
        mod = load()
        out = mod.low_yield_lanes(
            Path("does-not-exist-xyz"), closes_counter=lambda lane, tree: 0)
        self.assertEqual(out["lanes"], [])
        self.assertEqual(out["low_yield_count"], 0)
        self.assertEqual(out["schema"], mod._LOW_YIELD_SCHEMA)

    def test_turn_count_reads_fak_turn_lines(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "resolve-1-20260704-120000.log"
            log.write_text(
                "# fak-spawn header lane=tools\n"
                "fak-turn trace=guard cold prov=0 tok\n"
                "some tool output that is not a turn\n"
                "fak-turn trace=guard ok prov=5k tok\n",
                encoding="utf-8")
            self.assertEqual(mod._log_turn_count(log), 2)


class LowYieldFoldTest(unittest.TestCase):
    """build_payload folds the injected low_yield rollup into payload + a reason,
    and render/render_md/slack surface a flagged lane — never flipping ok (#2062)."""

    def _low_yield(self, **over) -> dict:
        ly = {
            "schema": "fleet-low-yield-lanes/1",
            "turns_floor": 40,
            "lookback_min": 180,
            "low_yield_count": 1,
            "lanes": [
                {"lane": "tools", "sessions": 2, "turns": 118, "max_session_turns": 74,
                 "closes": 0, "tree": ["tools"], "tree_known": True,
                 "verdict": "LOW_YIELD",
                 "evidence_logs": ["resolve-2062-20260704-120000.log"]},
                {"lane": "docs", "sessions": 1, "turns": 60, "max_session_turns": 60,
                 "closes": 4, "tree": ["docs"], "tree_known": True, "verdict": "OK",
                 "evidence_logs": ["resolve-2100-20260704-121000.log"]},
            ],
        }
        ly.update(over)
        return ly

    def test_flagged_lane_reaches_payload_reason_and_render(self) -> None:
        mod = load()
        p = build(mod, low_yield=self._low_yield())
        self.assertTrue(p["ok"])  # informational only — never flips ok
        self.assertEqual(p["low_yield"]["low_yield_count"], 1)
        self.assertTrue(any("low-yield lane" in r and "#2062" in r for r in p["reasons"]))
        rendered = mod.render(p)
        self.assertIn("low-yield :", rendered)
        self.assertIn("tools=118t/2s/0c", rendered)

    def test_flagged_lane_is_a_slack_action(self) -> None:
        mod = load()
        p = build(mod, low_yield=self._low_yield())
        buckets = mod._dispatch_slack_buckets(p)
        self.assertTrue(any("lane tools low-yield" in r and "re-scope or exclude" in r
                            for r in buckets["action"]))
        self.assertEqual(mod._dispatch_headline_state(p), "ACTION")

    def test_no_flagged_lane_emits_no_reason(self) -> None:
        mod = load()
        p = build(mod, low_yield=self._low_yield(
            low_yield_count=0,
            lanes=[{"lane": "docs", "sessions": 1, "turns": 60, "closes": 4,
                    "tree": ["docs"], "tree_known": True, "verdict": "OK",
                    "max_session_turns": 60, "evidence_logs": []}]))
        self.assertFalse(any("low-yield lane" in r for r in p["reasons"]))
        self.assertNotIn("low-yield :", mod.render(p))

    def test_low_yield_defaults_to_empty_when_omitted(self) -> None:
        mod = load()
        p = build(mod)  # build() does not pass low_yield -> defaults to None -> {}
        self.assertEqual(p["low_yield"], {})
        self.assertFalse(any("low-yield lane" in r for r in p["reasons"]))

    def test_md_low_yield_section_lists_flagged_lane(self) -> None:
        mod = load()
        md = mod.render_md(build(mod, low_yield=self._low_yield()), date="2026-07-04")
        self.assertIn("## Low-yield lanes (turns spent vs ancestry closes)", md)
        self.assertIn("| tools | 2 | 118 | 74 | 0 | **LOW_YIELD** |", md)
        self.assertIn("| docs | 1 | 60 | 60 | 4 | ok |", md)


class ShipsPerWorkerFoldTest(unittest.TestCase):
    """parse_ships_per_worker/ships_per_worker fold the best-effort (fak-worker <id>)
    trailer into per-worker ship counts (#2065) — pure over commit records, and the
    impure fold uses an injectable git runner so it stays hermetic."""

    def _rec(self, subject: str, body: str = "") -> str:
        return subject + "\n" + body

    def test_counts_and_sorts_per_worker(self) -> None:
        mod = load()
        recs = [
            self._rec("feat(x): a (fak tools)", "Fixes #1\n(fak-worker acct3)"),
            self._rec("fix(y): b (fak tools)", "Fixes #2\n(fak-worker acct3)"),
            self._rec("feat(z): c (fak gateway)", "Closes #3\n(fak-worker acct7)"),
        ]
        out = mod.parse_ships_per_worker(recs)
        self.assertEqual(out["attributed_ships"], 3)
        self.assertEqual(out["worker_count"], 2)
        self.assertEqual(out["unknown"], 0)
        # richest worker first, then lexicographic
        self.assertEqual(out["workers"][0], {"worker": "acct3", "ships": 2})
        self.assertEqual(out["workers"][1], {"worker": "acct7", "ships": 1})

    def test_zero_trailer_commit_is_unknown(self) -> None:
        mod = load()
        recs = [
            self._rec("feat(x): a (fak tools)", "Fixes #1\n(fak-worker acct3)"),
            self._rec("chore(y): no trailer here", "Fixes #2"),  # matched-but-unparseable
        ]
        out = mod.parse_ships_per_worker(recs)
        self.assertEqual(out["attributed_ships"], 2)
        self.assertEqual(out["unknown"], 1)
        self.assertEqual(out["worker_count"], 1)
        self.assertIn({"worker": "acct3", "ships": 1}, out["workers"])

    def test_empty_records_yield_empty_fold(self) -> None:
        mod = load()
        out = mod.parse_ships_per_worker([])
        self.assertEqual(out["attributed_ships"], 0)
        self.assertEqual(out["workers"], [])
        self.assertEqual(out["schema"], mod._SHIPS_PER_WORKER_SCHEMA)

    def test_impure_fold_uses_injected_runner(self) -> None:
        mod = load()
        recs = [self._rec("feat: a (fak tools)", "Fixes #1\n(fak-worker acct-a)")]
        out = mod.ships_per_worker(ROOT, now_ts=0.0, runner=lambda root, since: recs)
        self.assertEqual(out["attributed_ships"], 1)
        self.assertEqual(out["workers"], [{"worker": "acct-a", "ships": 1}])
        self.assertNotIn("unavailable", out)

    def test_git_unavailable_fails_open(self) -> None:
        mod = load()
        out = mod.ships_per_worker(ROOT, now_ts=0.0, runner=lambda root, since: None)
        self.assertTrue(out.get("unavailable"))
        self.assertEqual(out["attributed_ships"], 0)


class CommitDroughtTest(unittest.TestCase):
    """The loop-level drought witness: zero fleet-attributed commits over the
    window is ``dry``; git-unavailable fails open (no false drought); and the
    Slack render surfaces the alarm only once the caller derives ``droughty``
    (``dry`` AND armed)."""

    def test_zero_commits_is_dry(self) -> None:
        mod = load()
        out = mod.commit_drought(ROOT, now_ts=0.0, runner=lambda root, since: [])
        self.assertTrue(out["dry"])
        self.assertEqual(out["commit_count"], 0)
        self.assertNotIn("unavailable", out)

    def test_commits_present_not_dry(self) -> None:
        mod = load()
        recs = ["feat: a\n(fak-worker acct-a)", "fix: b\n(fak-worker acct-b)"]
        out = mod.commit_drought(ROOT, now_ts=0.0, runner=lambda root, since: recs)
        self.assertFalse(out["dry"])
        self.assertEqual(out["commit_count"], 2)

    def test_git_unavailable_fails_open(self) -> None:
        mod = load()
        out = mod.commit_drought(ROOT, now_ts=0.0, runner=lambda root, since: None)
        self.assertTrue(out.get("unavailable"))
        self.assertNotIn("dry", out)  # caller ANDs dry with armed; no dry -> no alarm

    def test_slack_surfaces_drought_alarm(self) -> None:
        mod = load()
        txt = mod.render_slack({"commit_drought": {"droughty": True, "hours": 3.0}})
        self.assertIn("drought", txt.lower())

    def test_slack_no_alarm_when_not_droughty(self) -> None:
        mod = load()
        txt = mod.render_slack(
            {"commit_drought": {"dry": True, "droughty": False, "hours": 3.0},
             "ok": True})
        self.assertNotIn("drought", txt.lower())


class ShipsPerWorkerPayloadTest(unittest.TestCase):
    """build_payload folds the injected ships-per-worker rollup into payload + a reason,
    and render/render_md surface it — never flipping ok (#2065)."""

    def _ships(self, **over) -> dict:
        s = {
            "schema": "fleet-ships-per-worker/1",
            "attributed_ships": 3,
            "worker_count": 2,
            "unknown": 1,
            "lookback_min": 1440,
            "workers": [{"worker": "acct3", "ships": 2}, {"worker": "acct7", "ships": 1}],
            "note": "best-effort agent-emitted (fak-worker) trailer — attribution aid, not a witness",
        }
        s.update(over)
        return s

    def test_ships_reach_payload_reason_and_render(self) -> None:
        mod = load()
        p = build(mod, ships=self._ships())
        self.assertTrue(p["ok"])  # informational only — never flips ok
        self.assertEqual(p["ships_per_worker"]["attributed_ships"], 3)
        self.assertTrue(any("attributed to" in r and "#2065" in r for r in p["reasons"]))
        rendered = mod.render(p)
        self.assertIn("ships/wkr :", rendered)
        self.assertIn("acct3=2", rendered)
        self.assertIn("unattributed", rendered)

    def test_no_ships_emits_no_reason(self) -> None:
        mod = load()
        p = build(mod, ships=self._ships(attributed_ships=0, worker_count=0,
                                         unknown=0, workers=[]))
        self.assertFalse(any("(fak-worker) trailer" in r for r in p["reasons"]))
        self.assertNotIn("ships/wkr :", mod.render(p))

    def test_ships_defaults_to_empty_when_omitted(self) -> None:
        mod = load()
        p = build(mod)  # build() does not pass ships -> defaults to None -> {}
        self.assertEqual(p["ships_per_worker"], {})

    def test_md_ships_section_lists_workers(self) -> None:
        mod = load()
        md = mod.render_md(build(mod, ships=self._ships()), date="2026-07-04")
        self.assertIn("## Ships per worker (best-effort attribution)", md)
        self.assertIn("| `acct3` | 2 |", md)
        self.assertIn("| _unattributed_ | 1 |", md)


def _prow(utc: str, *, open_now: int, loop_total: int, closed_now: int = 0,
          audit_error=None) -> dict:
    return {"schema": "fleet-issue-resolve-progress/1", "utc": utc,
            "open_now": open_now, "closed_by_loop_total": loop_total,
            "closed_now": closed_now, "audit_error": audit_error}


class WatchDecisionTest(unittest.TestCase):
    """#2642: the trailing-window watch digest that explains WHY the monitor
    (intentionally) takes no action, reproducing the ef59064f OUTPACED shape."""

    def _now(self, mod, utc: str) -> float:
        return mod._iso_epoch(utc)

    def test_ef59064f_shape_is_outpaced_not_stalled(self) -> None:
        # closures keep advancing (772 -> 791) while the backlog rises (700 -> 764)
        # over the 6h watch: arrivals outpace service, NOT a stalled dispatcher.
        mod = load()
        rows = [
            _prow("2026-07-06T18:15:00Z", open_now=700, loop_total=772, closed_now=0),
            _prow("2026-07-06T20:00:00Z", open_now=730, loop_total=780, closed_now=8),
            _prow("2026-07-06T22:00:00Z", open_now=752, loop_total=787, closed_now=7),
            _prow("2026-07-07T00:15:00Z", open_now=764, loop_total=791, closed_now=4),
        ]
        wd = mod.watch_decision(rows, now_ts=self._now(mod, "2026-07-07T00:15:00Z"),
                                window_hours=6.0)
        self.assertEqual(wd["verdict"], "OUTPACED")
        self.assertNotEqual(wd["verdict"], "STALLED")
        self.assertFalse(wd["action_needed"])  # informational — never hand-launch
        c = wd["cited"]
        self.assertEqual(c["open_now_start"], 700)
        self.assertEqual(c["open_now_end"], 764)
        self.assertEqual(c["closed_by_loop_total_start"], 772)
        self.assertEqual(c["closed_by_loop_total_end"], 791)
        self.assertEqual(c["closed_now_sum"], 19)
        self.assertEqual(wd["audit_status"], "clean")
        self.assertIn("outpace", wd["why"])

    def test_stalled_when_backlog_persists_without_closes(self) -> None:
        mod = load()
        rows = [
            _prow("2026-07-06T20:00:00Z", open_now=500, loop_total=772, closed_now=0),
            _prow("2026-07-07T00:00:00Z", open_now=540, loop_total=772, closed_now=0),
        ]
        wd = mod.watch_decision(rows, now_ts=self._now(mod, "2026-07-07T00:00:00Z"))
        self.assertEqual(wd["verdict"], "STALLED")
        self.assertTrue(wd["action_needed"])

    def test_draining_when_backlog_falls_while_closing(self) -> None:
        mod = load()
        rows = [
            _prow("2026-07-06T20:00:00Z", open_now=540, loop_total=772, closed_now=5),
            _prow("2026-07-07T00:00:00Z", open_now=500, loop_total=790, closed_now=6),
        ]
        wd = mod.watch_decision(rows, now_ts=self._now(mod, "2026-07-07T00:00:00Z"))
        self.assertEqual(wd["verdict"], "DRAINING")
        self.assertFalse(wd["action_needed"])

    def test_healthy_idle_when_backlog_drained(self) -> None:
        mod = load()
        rows = [
            _prow("2026-07-06T20:00:00Z", open_now=0, loop_total=800, closed_now=0),
            _prow("2026-07-07T00:00:00Z", open_now=0, loop_total=800, closed_now=0),
        ]
        wd = mod.watch_decision(rows, now_ts=self._now(mod, "2026-07-07T00:00:00Z"))
        self.assertEqual(wd["verdict"], "HEALTHY_IDLE")
        self.assertFalse(wd["action_needed"])

    def test_insufficient_data_with_one_row(self) -> None:
        mod = load()
        rows = [_prow("2026-07-07T00:00:00Z", open_now=500, loop_total=772)]
        wd = mod.watch_decision(rows, now_ts=self._now(mod, "2026-07-07T00:00:00Z"))
        self.assertEqual(wd["verdict"], "INSUFFICIENT_DATA")
        self.assertFalse(wd["action_needed"])

    def test_rows_outside_window_are_excluded(self) -> None:
        mod = load()
        rows = [
            _prow("2026-07-06T10:00:00Z", open_now=999, loop_total=700, closed_now=0),
            _prow("2026-07-06T20:00:00Z", open_now=730, loop_total=780, closed_now=8),
            _prow("2026-07-07T00:00:00Z", open_now=764, loop_total=791, closed_now=4),
        ]
        wd = mod.watch_decision(rows, now_ts=self._now(mod, "2026-07-07T00:00:00Z"),
                                window_hours=6.0)
        self.assertEqual(wd["rows_in_window"], 2)  # the 14h-old row is dropped
        self.assertEqual(wd["cited"]["open_now_start"], 730)

    def test_audit_error_surfaces_in_status(self) -> None:
        mod = load()
        rows = [
            _prow("2026-07-06T20:00:00Z", open_now=730, loop_total=780, closed_now=8),
            _prow("2026-07-07T00:00:00Z", open_now=764, loop_total=791, closed_now=4,
                  audit_error="gh 502"),
        ]
        wd = mod.watch_decision(rows, now_ts=self._now(mod, "2026-07-07T00:00:00Z"))
        self.assertEqual(wd["audit_status"], "error: gh 502")

    def test_sched_task_red_without_recovery_is_unresolved(self) -> None:
        mod = load()
        wd = mod.watch_decision(
            [], now_ts=self._now(mod, "2026-07-07T00:00:00Z"),
            sched_task={"installed": True, "status": "Ready", "last_result": 1})
        self.assertEqual(wd["scheduled_task"]["classification"], "unresolved_unknown")

    def test_sched_task_red_with_recovery_is_self_healing(self) -> None:
        mod = load()
        self.assertEqual(
            mod._classify_sched_task(
                {"installed": True, "last_result": 1, "recovered": True}),
            "self_healing")
        self.assertEqual(
            mod._classify_sched_task({"installed": True, "status": "Ready"}), "clean")

    def test_follow_ups_are_listed_and_deduped(self) -> None:
        mod = load()
        rows = [
            _prow("2026-07-06T20:00:00Z", open_now=730, loop_total=780, closed_now=8),
            _prow("2026-07-07T00:00:00Z", open_now=764, loop_total=791, closed_now=4),
        ]
        wd = mod.watch_decision(rows, now_ts=self._now(mod, "2026-07-07T00:00:00Z"),
                                follow_ups=[2636, 2634, 2635, 2636])
        self.assertEqual(wd["follow_ups"], [2634, 2635, 2636])

    def test_payload_carries_watch_and_renders(self) -> None:
        mod = load()
        watch = mod.watch_decision(
            [_prow("2026-07-06T20:00:00Z", open_now=730, loop_total=780, closed_now=8),
             _prow("2026-07-07T00:00:00Z", open_now=764, loop_total=791, closed_now=4)],
            now_ts=self._now(mod, "2026-07-07T00:00:00Z"), window_hours=6.0,
            follow_ups=[2634])
        p = build(mod, watch=watch)
        self.assertEqual(p["watch_decision"]["verdict"], "OUTPACED")
        rendered = mod.render(p)
        self.assertIn("watch     :", rendered)
        self.assertIn("OUTPACED", rendered)
        self.assertIn("#2634", rendered)
        md = mod.render_md(p, date="2026-07-07")
        self.assertIn("## Watch decision (why no action)", md)
        self.assertIn("intentionally no action", md)
        self.assertIn("#2634", md)

    def test_watch_defaults_to_empty_when_omitted(self) -> None:
        mod = load()
        p = build(mod)  # build() does not pass watch -> defaults to None -> {}
        self.assertEqual(p["watch_decision"], {})
        self.assertNotIn("watch     :", mod.render(p))

    def test_parse_watchdog_query_extracts_status_and_last_result(self) -> None:
        # verbose `schtasks /Query /V /FO LIST` shape: a red LastTaskResult must be
        # captured so the live digest can answer the issue's question 2.
        mod = load()
        out = "\r\n".join([
            "Folder: \\", "HostName:      HOST", "TaskName:      \\fak-watchdog",
            "Status:        Ready", "Last Run Time: 7/7/2026 1:00:00 AM",
            "Last Result:   267009", "Author:        HOST\\op",
        ])
        got = mod._parse_watchdog_query(out)
        self.assertEqual(got["status"], "Ready")
        self.assertEqual(got["last_result"], 267009)

    def test_parse_watchdog_query_green_and_absent_fields(self) -> None:
        mod = load()
        self.assertEqual(
            mod._parse_watchdog_query("Status: Running\nLast Result: 0\n"),
            {"status": "Running", "last_result": 0})
        # no Last Result line (e.g. non-verbose output) -> None, not a raise
        self.assertEqual(
            mod._parse_watchdog_query("Status: Ready\n"),
            {"status": "Ready", "last_result": None})

    def test_sched_recovered_true_when_closures_advance(self) -> None:
        mod = load()
        rows = [
            _prow("2026-07-06T20:00:00Z", open_now=730, loop_total=780, closed_now=8),
            _prow("2026-07-07T00:00:00Z", open_now=764, loop_total=791, closed_now=4),
        ]
        self.assertTrue(mod._sched_recovered(
            rows, now_ts=self._now(mod, "2026-07-07T00:00:00Z")))

    def test_sched_recovered_false_when_loop_flat(self) -> None:
        mod = load()
        rows = [
            _prow("2026-07-06T20:00:00Z", open_now=730, loop_total=780, closed_now=0),
            _prow("2026-07-07T00:00:00Z", open_now=764, loop_total=780, closed_now=0),
        ]
        self.assertFalse(mod._sched_recovered(
            rows, now_ts=self._now(mod, "2026-07-07T00:00:00Z")))

    def test_live_red_result_with_recovery_classifies_self_healing(self) -> None:
        # end-to-end live-wiring shape (what collect() now feeds watch_decision): a
        # red LastTaskResult whose loop kept closing => self_healing, not a
        # malfunction — the distinction that was dead code in the live path before.
        mod = load()
        rows = [
            _prow("2026-07-06T20:00:00Z", open_now=730, loop_total=780, closed_now=8),
            _prow("2026-07-07T00:00:00Z", open_now=764, loop_total=791, closed_now=4),
        ]
        now = self._now(mod, "2026-07-07T00:00:00Z")
        recovered = mod._sched_recovered(rows, now_ts=now)
        wd = mod.watch_decision(rows, now_ts=now, sched_task={
            "installed": True, "status": "Ready",
            "last_result": 267009, "recovered": recovered})
        self.assertEqual(wd["scheduled_task"]["classification"], "self_healing")

    def test_live_red_result_without_recovery_stays_unresolved(self) -> None:
        mod = load()
        rows = [
            _prow("2026-07-06T20:00:00Z", open_now=730, loop_total=780, closed_now=0),
            _prow("2026-07-07T00:00:00Z", open_now=764, loop_total=780, closed_now=0),
        ]
        now = self._now(mod, "2026-07-07T00:00:00Z")
        wd = mod.watch_decision(rows, now_ts=now, sched_task={
            "installed": True, "status": "Ready", "last_result": 267009,
            "recovered": mod._sched_recovered(rows, now_ts=now)})
        self.assertEqual(
            wd["scheduled_task"]["classification"], "unresolved_unknown")


class BacklogRateTest(unittest.TestCase):
    """#2634: the numeric arrival-vs-service fold that tells 'outpaced' (healthy
    but supply-bound) apart from 'stalled' (a malfunction), with a BACKLOG_OUTPACED
    flag over K consecutive windows and a synthetic-progress.jsonl fixture."""

    NOW = "2026-07-07T00:00:00Z"

    def _now(self, mod) -> float:
        return mod._iso_epoch(self.NOW)

    def _outpaced_rows(self) -> list[dict]:
        # Witness-shaped: closes climb continuously (717 -> 797) while the backlog
        # also climbs (700 -> 772) across four consecutive 6h windows.
        return [
            _prow("2026-07-06T00:00:00Z", open_now=700, loop_total=717),
            _prow("2026-07-06T06:00:00Z", open_now=740, loop_total=740),
            _prow("2026-07-06T12:00:00Z", open_now=760, loop_total=774),
            _prow("2026-07-06T18:00:00Z", open_now=770, loop_total=790),
            _prow("2026-07-07T00:00:00Z", open_now=772, loop_total=797),
        ]

    def test_outpaced_sets_backlog_outpaced_over_k_windows(self) -> None:
        mod = load()
        br = mod.backlog_rates(self._outpaced_rows(), now_ts=self._now(mod),
                               window_hours=6.0, windows=4, k_consecutive=2)
        self.assertEqual(br["schema"], "fleet-backlog-rate/1")
        self.assertEqual(br["verdict"], "OUTPACED")
        self.assertTrue(br["backlog_outpaced"])
        self.assertGreaterEqual(br["consecutive_outpaced_windows"], 2)
        self.assertEqual(br["window_hours"], 6.0)
        self.assertEqual(len(br["per_window"]), 4)
        self.assertTrue(all(w["outpaced"] for w in br["per_window"]))
        # arrival must exceed service — that is the whole point of the fold.
        self.assertGreater(br["arrival_rate_per_hour"], br["service_rate_per_hour"])
        self.assertGreater(br["service_rate_per_hour"], 0)  # working, not stalled
        self.assertEqual(br["span_hours"], 24.0)
        self.assertIn("BACKLOG_OUTPACED", br["why"])

    def test_draining_backlog_is_not_outpaced(self) -> None:
        mod = load()
        rows = [
            _prow("2026-07-06T12:00:00Z", open_now=800, loop_total=700),
            _prow("2026-07-06T18:00:00Z", open_now=780, loop_total=720),
            _prow("2026-07-07T00:00:00Z", open_now=760, loop_total=740),
        ]
        br = mod.backlog_rates(rows, now_ts=self._now(mod))
        self.assertEqual(br["verdict"], "DRAINING")
        self.assertFalse(br["backlog_outpaced"])
        self.assertLess(br["arrival_rate_per_hour"], br["service_rate_per_hour"])

    def test_stalled_backlog_is_not_outpaced(self) -> None:
        # closes flat (790) while the backlog rises: STALLED, never BACKLOG_OUTPACED
        # (service == 0 is what separates a malfunction from a supply/demand trend).
        mod = load()
        rows = [
            _prow("2026-07-06T12:00:00Z", open_now=700, loop_total=790),
            _prow("2026-07-06T18:00:00Z", open_now=730, loop_total=790),
            _prow("2026-07-07T00:00:00Z", open_now=760, loop_total=790),
        ]
        br = mod.backlog_rates(rows, now_ts=self._now(mod))
        self.assertEqual(br["verdict"], "STALLED")
        self.assertFalse(br["backlog_outpaced"])
        self.assertEqual(br["service_rate_per_hour"], 0.0)

    def test_insufficient_data_with_one_row(self) -> None:
        mod = load()
        rows = [_prow(self.NOW, open_now=500, loop_total=772)]
        br = mod.backlog_rates(rows, now_ts=self._now(mod))
        self.assertEqual(br["verdict"], "INSUFFICIENT_DATA")
        self.assertFalse(br["backlog_outpaced"])
        self.assertIsNone(br["arrival_rate_per_hour"])

    def test_fixture_on_synthetic_progress_jsonl(self) -> None:
        # The Done-condition witness: pin the fold end-to-end on a synthetic
        # .dispatch-runs/progress.jsonl read back through read_dispatch_progress.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / mod.RUNS_DIRNAME
            runs.mkdir()
            (runs / mod.PROGRESS_LOG).write_text(
                "\n".join(json.dumps(r) for r in self._outpaced_rows()) + "\n",
                encoding="utf-8")
            records = mod.read_dispatch_progress(root)
            self.assertEqual(len(records), 5)
            br = mod.backlog_rates(records, now_ts=self._now(mod),
                                   window_hours=6.0, windows=4, k_consecutive=2)
        self.assertTrue(br["backlog_outpaced"])
        self.assertEqual(br["verdict"], "OUTPACED")

    def test_payload_carries_backlog_rate_and_renders(self) -> None:
        mod = load()
        br = mod.backlog_rates(self._outpaced_rows(), now_ts=self._now(mod),
                               window_hours=6.0, windows=4, k_consecutive=2)
        p = build(mod, backlog_rate=br)
        self.assertEqual(p["backlog_rate"]["verdict"], "OUTPACED")
        self.assertTrue(p["backlog_rate"]["backlog_outpaced"])
        self.assertTrue(any("backlog OUTPACED" in r and "#2634" in r
                            for r in p["reasons"]))
        rendered = mod.render(p)
        self.assertIn("supply    :", rendered)
        self.assertIn("BACKLOG_OUTPACED", rendered)

    def test_backlog_rate_defaults_to_empty_when_omitted(self) -> None:
        mod = load()
        p = build(mod)  # build() does not pass backlog_rate -> None -> {}
        self.assertEqual(p["backlog_rate"], {})
        self.assertNotIn("supply    :", mod.render(p))


class SpawnFailedCauseBreakdownTest(unittest.TestCase):
    """spawn_failed_cause_breakdown folds the silent/stub early-exit population into a
    per-cause SPAWN_FAILED mix with evidence rows (#2635) — hermetic tmp runs-dir."""

    HEADER = "# fak-spawn 20260710-010203 issue={n} lane=cmd backend=claude argv0=claude.exe\n"

    def _mk(self, runs: Path, issue: int, stamp: str, tail_body: str) -> None:
        log = runs / f"resolve-{issue}-{stamp}.log"
        log.write_text(self.HEADER.format(n=issue) + tail_body, encoding="utf-8")
        (runs / f"resolve-{issue}-{stamp}.pid").write_text("39000", encoding="utf-8")

    def test_mix_and_evidence_over_dead_stub_logs(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 1515, "20260703-010101",
                     "You've hit your weekly limit · resets 4am (America/Los_Angeles)\n")
            self._mk(runs, 2382, "20260703-020202",
                     "Missing environment variable: `ANTHROPIC_API_KEY`\n")
            self._mk(runs, 2401, "20260703-030303", "")  # header-only -> exec_race
            # A productive worker (over the stub floor) is a spawn but NOT a failure.
            big = runs / "resolve-999-20260703-040404.log"
            big.write_text("x" * 4000, encoding="utf-8")
            (runs / "resolve-999-20260703-040404.pid").write_text("39001", encoding="utf-8")

            out = mod.spawn_failed_cause_breakdown(runs, alive=set())  # nothing alive

            self.assertEqual(out["schema"], "fak.spawn-failed-cause-breakdown.v1")
            self.assertEqual(out["spawns"], 4)          # all four resolve logs
            self.assertEqual(out["spawn_failed"], 3)    # the three stub early-exits
            self.assertEqual(out["by_cause"]["weekly_limit"]["count"], 1)
            self.assertEqual(out["by_cause"]["stale_cred"]["count"], 1)
            self.assertEqual(out["by_cause"]["exec_race"]["count"], 1)
            self.assertEqual(out["by_cause"]["child_crash"]["count"], 0)
            # per-event evidence rows carry the classified cause
            ev = out["by_cause"]["weekly_limit"]["evidence"][0]
            self.assertEqual(ev["issue"], 1515)
            self.assertEqual(ev["cause"], "weekly_limit")
            self.assertIn("spawn-failed cause breakdown", mod.render_spawn_causes(out))

    def test_empty_runs_dir_is_fail_open(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            out = mod.spawn_failed_cause_breakdown(Path(d), alive=set())
            self.assertEqual(out["spawns"], 0)
            self.assertEqual(out["spawn_failed"], 0)
            self.assertEqual(out["rate"], 0.0)


class WaveSpawnFailedTelemetryTest(unittest.TestCase):
    """#6492: the WAVE path's launch failures must be countable too.

    The #2635 fold only ever enumerated `resolve-<issue>-<stamp>.log` — so a wave whose
    four children all died on launch reported a 0% SPAWN_FAILED rate while the fleet was
    launching nothing. Wave dispatch logs now count as spawns, and the terminal
    `.spawn-failed.json` record each dead wave child leaves behind counts as a failure
    with its classified cause."""

    def _wave_log(self, runs: Path, lane: str, stamp: str, body: str = "") -> Path:
        log = runs / f"dispatch-{lane}-{stamp}.log"
        log.write_text(body, encoding="utf-8")
        return log

    def _failed(self, runs: Path, lane: str, stamp: str, cause: str,
                returncode: int = 1) -> None:
        log = self._wave_log(runs, lane, stamp)
        (runs / f"dispatch-{lane}-{stamp}.spawn-failed.json").write_text(json.dumps({
            "verdict": "SPAWN_FAILED", "cause": cause, "lane": lane,
            "seat": "acct-0", "pid": 4242, "log": str(log), "returncode": returncode,
            "log_bytes": 0, "silent": True, "tail": "", "stamp": stamp,
        }), encoding="utf-8")

    def test_dead_wave_children_are_counted_not_invisible(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._failed(runs, "tools", "20260811-010101", "exec_race")
            self._failed(runs, "docs", "20260811-010102", "weekly_limit")
            # a surviving wave worker: a spawn, but NOT a failure
            self._wave_log(runs, "model", "20260811-010103", "x" * 4000)

            out = mod.spawn_failed_cause_breakdown(runs, alive=set())

            self.assertEqual(out["spawns"], 3)        # wave logs now count as spawns
            self.assertEqual(out["spawn_failed"], 2)  # was: 0 — wave deaths were unseen
            self.assertEqual(out["by_cause"]["exec_race"]["count"], 1)
            self.assertEqual(out["by_cause"]["weekly_limit"]["count"], 1)
            ev = out["by_cause"]["exec_race"]["evidence"][0]
            self.assertEqual(ev["lane"], "tools")
            self.assertEqual(ev["returncode"], 1)
            self.assertEqual(ev["source"], "wave_spawn_failed_record")
            self.assertIn("spawn-failed cause breakdown", mod.render_spawn_causes(out))

    def test_a_malformed_record_is_skipped_not_fatal(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._wave_log(runs, "tools", "20260811-020201")
            (runs / "dispatch-tools-20260811-020201.spawn-failed.json").write_text(
                "{ not json", encoding="utf-8")
            out = mod.spawn_failed_cause_breakdown(runs, alive=set())
            self.assertEqual(out["spawns"], 1)
            self.assertEqual(out["spawn_failed"], 0)  # fail-open: no phantom failure

    def test_a_record_older_than_the_lookback_is_out_of_window(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._failed(runs, "tools", "20260101-030301", "exec_race")
            fresh = mod.spawn_failed_cause_breakdown(runs, alive=set(), lookback_min=60)
            self.assertEqual(fresh["spawn_failed"], 1)
            # same record, read a day later: outside the lookback, so out of the rate
            stale = mod.spawn_failed_cause_breakdown(
                runs, alive=set(), lookback_min=60,
                now_ts=os.stat(runs / "dispatch-tools-20260101-030301.log").st_mtime + 86400)
            self.assertEqual(stale["spawn_failed"], 0)


class CapacityHonestyTest(unittest.TestCase):
    """#4649 residual: the dispatcher card renders capacity honestly — a missing read is
    UNKNOWN (never a literal None), an over-subscribed pool shows the overshoot (never a
    bare negative headroom, and never hidden behind a clamped `headroom 0`), and the two
    capacity scopes are reconciled by name when they disagree. Parity with the Slack
    roll-up's `_worker_slot_phrase`."""

    def test_headroom_note_missing_read_is_unknown(self) -> None:
        mod = load()
        self.assertEqual(mod._slots_headroom_note(None, None), "headroom UNKNOWN")
        self.assertEqual(mod._slots_headroom_note(1, None), "headroom UNKNOWN")
        self.assertEqual(mod._or_unknown(None), "UNKNOWN")
        self.assertEqual(mod._or_unknown(0), 0)  # a real zero survives; only None -> UNKNOWN

    def test_headroom_note_oversubscription_shows_overshoot_not_bare_negative(self) -> None:
        mod = load()
        note = mod._slots_headroom_note(3, 2)  # live 3 > cap 2
        self.assertEqual(note, "1 over the 2-slot target")
        self.assertNotIn("-1", note)

    def test_headroom_note_spare_capacity_keeps_familiar_term(self) -> None:
        mod = load()
        self.assertEqual(mod._slots_headroom_note(1, 4), "headroom 3")
        self.assertEqual(mod._slots_headroom_note(4, 4), "headroom 0")

    def test_dispatcher_clause_missing_read_is_unknown(self) -> None:
        mod = load()
        self.assertEqual(mod._workers_live_clause({}),
                         "UNKNOWN (dispatcher read incomplete)")

    def test_card_renders_oversubscription_across_box_md_and_utilization(self) -> None:
        # A full-card witness: with live 3 > cap 2 the overshoot appears in the box, the
        # markdown workers line, AND the utilization worker-slots line — the last proves
        # the render recomputes headroom from live/cap, so the upstream max(0,…) clamp
        # (which stores a misleading headroom 0) can no longer hide an overshoot.
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK", cap=2, live=3),
                  resolver_preflight=resolver_preflight())
        self.assertEqual((p["utilization"]["worker_slots"])["headroom"], 0)  # clamped upstream
        box = mod.render(p)
        md = mod.render_md(p, date="2026-07-05")
        for text in (box, md):
            self.assertIn("3/2 live (1 over the 2-slot target)", text)  # dispatcher
            self.assertNotIn("headroom -1", text)
        self.assertIn("worker slots 3/2 (1 over the 2-slot target)", md)  # utilization

    def test_reconcile_names_resolver_binding_when_host_has_free_slots(self) -> None:
        # host has 2 free worker slots (live 0/cap 2) but the resolver target refuses:
        # a launch is gated by the resolver, not host slots — say so, so two capacity
        # numbers on the card do not read as a contradiction.
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK", cap=2, live=0),
                  resolver_preflight=resolver_preflight())  # REFUSE_NO_SEAT
        md = mod.render_md(p, date="2026-07-05")
        self.assertIn("capacity reconcile", md)
        self.assertIn("gated by the resolver", md)

    def test_reconcile_names_host_binding_when_resolver_ready_but_slots_full(self) -> None:
        mod = load()
        rp = resolver_preflight()
        rp["verdict"] = "SPAWN_OK"
        p = build(mod, pre=pre("SPAWN_OK", cap=2, live=2), resolver_preflight=rp)
        md = mod.render_md(p, date="2026-07-05")
        self.assertIn("capacity reconcile", md)
        self.assertIn("gated by host slots", md)

    def test_no_reconcile_line_when_the_two_scopes_agree(self) -> None:
        # host at cap AND resolver refuses -> both say "no room"; nothing to reconcile.
        mod = load()
        p = build(mod, pre=pre("REFUSE_AT_CAP", cap=2, live=2),
                  resolver_preflight=resolver_preflight())
        md = mod.render_md(p, date="2026-07-05")
        self.assertNotIn("capacity reconcile", md)


def seat_pool(*, free: int = 1, tag: str = "worker-a", cooling: bool = False,
              cooldown_until: str | None = None, extra: list[dict] | None = None) -> dict:
    """A synthetic `fleet_accounts.seat_pool` inventory with `free` free slot(s) on
    `tag` -- the "nominally free seat" the #6495 failure turned on."""
    seats = [{
        "seat": tag, "tag": tag, "account": f".claude-{tag}", "product": "claude",
        "model": "claude", "available": not cooling,
        "state": "free", "dispatch_state": "cooling" if cooling else "available",
        "hold_reason": "rate_limited" if cooling else "",
        "cooldown": ({"next_eligible": cooldown_until} if cooldown_until else None),
        "session_cap": free, "leased_slots": 0, "free_slots": free, "workers": [],
    }]
    seats += list(extra or [])
    return {
        "schema": "fak.seat-pool/1",
        "total_seats": sum(int(s.get("session_cap") or 0) for s in seats),
        "free_seats": sum(int(s.get("free_slots") or 0) for s in seats),
        "leased_seats": 0, "blocked_seats": 0,
        "depleted": False, "double_booked": [], "unbound_leases": [],
        "seats": seats,
    }


def write_ledger(path: Path, rows: list[tuple[str, "dt.datetime"]]) -> str:
    """Write a real `launch_admission` ledger: (account, launch time) per line."""
    with open(path, "w", encoding="utf-8") as fh:
        for account, ts in rows:
            fh.write(json.dumps({
                "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "account": account, "resume_account": account,
                "phase": "launched", "via": "test",
            }) + "\n")
    return str(path)


class LaunchAdmissionGateTest(unittest.TestCase):
    """#6495: the card folds the LAUNCHER's final admission gate.

    Every verdict here is produced by the real `tools/launch_admission.py` (the one
    seam a launcher passes through) over a real on-disk ledger -- not a stubbed
    verdict -- so a drift between what status folds and what the launcher enforces
    fails these tests.
    """

    def admission(self, mod, ledger_rows, *, seats=None, tag="worker-a",
                  now=None, cooling=False, cooldown_until=None):
        now = now or dt.datetime.now(dt.timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            ledger = write_ledger(Path(td) / "resume_ledger.jsonl", ledger_rows)
            return mod.read_launch_admission(
                ROOT, pre("SPAWN_OK"),
                seats if seats is not None else seat_pool(
                    tag=tag, cooling=cooling, cooldown_until=cooldown_until),
                now=now, ledger_path=ledger)

    def test_free_seat_under_launch_rate_exceeded_does_not_recommend_growth(self) -> None:
        # The #6495 witness: preflight SPAWN_OK, a nominally FREE seat, and a launch
        # ledger already at the per-account ceiling. The launcher would refuse this
        # launch LAUNCH_RATE_EXCEEDED, so the card must not recommend growth.
        mod = load()
        now = dt.datetime(2026, 8, 11, 20, 30, tzinfo=dt.timezone.utc)
        rows = [("worker-a", now - dt.timedelta(minutes=m)) for m in (1, 2, 3)]
        adm = self.admission(mod, rows, now=now)
        self.assertEqual(adm["verdict"], "DEFER")
        self.assertEqual(adm["reason"], "LAUNCH_RATE_EXCEEDED")
        self.assertFalse(adm["would_launch"])
        self.assertTrue(adm["retry_after"])

        p = build(mod, pre=pre("SPAWN_OK"), seat_inventory=seat_pool(),
                  launch_admission=adm)
        self.assertNotEqual(p["verdict"], "READY_TO_GROW")
        self.assertEqual(p["verdict"], "LAUNCH_COOLDOWN")
        self.assertTrue(p["ok"])  # a cooldown is a steady state, not breakage
        self.assertFalse(p["spawn_gate"]["would_launch"])
        self.assertEqual(p["spawn_gate"]["reason"], "LAUNCH_RATE_EXCEEDED")
        self.assertEqual(p["spawn_gate"]["retry_after"], adm["retry_after"])
        # "one free seat" is never mistaken for "one launchable seat" again.
        self.assertEqual(p["accounts"]["free_seats"], 1)
        self.assertEqual(p["accounts"]["launchable_seats"], 0)
        self.assertEqual([d["account"] for d in p["accounts"]["deferred"]], ["worker-a"])
        self.assertIn(adm["retry_after"], p["next_action"])
        self.assertTrue(any("LAUNCH_RATE_EXCEEDED" in r for r in p["reasons"]))

    def test_admitted_launch_keeps_ready_to_grow(self) -> None:
        mod = load()
        adm = self.admission(mod, [])
        self.assertEqual(adm["verdict"], "ADMIT")
        p = build(mod, pre=pre("SPAWN_OK"), seat_inventory=seat_pool(),
                  launch_admission=adm)
        self.assertEqual(p["verdict"], "READY_TO_GROW")
        self.assertTrue(p["spawn_gate"]["would_launch"])
        self.assertEqual(p["accounts"]["launchable_seats"], 1)
        self.assertIn("launch one worker", p["next_action"])

    def test_global_cap_defers_every_candidate_account(self) -> None:
        # Six nominally free seats, a fleet already at the GLOBAL cap: no account
        # can launch, so the fold must refuse growth for ALL of them (the #6495
        # follow-up comment -- global-cap admission, not only per-account cooldown).
        mod = load()
        now = dt.datetime(2026, 8, 11, 20, 30, tzinfo=dt.timezone.utc)
        rows = [(f"other-{i}", now - dt.timedelta(minutes=1)) for i in range(10)]
        extra = [dict(seat_pool(tag="worker-b")["seats"][0])]
        adm = self.admission(mod, rows, seats=seat_pool(extra=extra), now=now)
        self.assertEqual(adm["reason"], "GLOBAL_LAUNCH_CAP")
        self.assertEqual(len(adm["deferred_accounts"]), 2)
        p = build(mod, pre=pre("SPAWN_OK"), launch_admission=adm)
        self.assertEqual(p["verdict"], "LAUNCH_COOLDOWN")

    def test_cooling_seat_is_deferred_on_its_own_reset(self) -> None:
        mod = load()
        adm = self.admission(mod, [], cooling=True,
                             cooldown_until="2026-08-11T21:00:00Z")
        self.assertEqual(adm["reason"], "ACCOUNT_THROTTLED")
        self.assertEqual(adm["retry_after"], "2026-08-11T21:00:00Z")
        p = build(mod, pre=pre("SPAWN_OK"), launch_admission=adm)
        self.assertEqual(p["verdict"], "LAUNCH_COOLDOWN")

    def test_one_admitted_account_among_deferrals_still_grows(self) -> None:
        # Growth is "could ONE worker launch", so a single admitted account is enough
        # even while a rate-limited sibling defers -- the fold must not over-refuse.
        mod = load()
        now = dt.datetime(2026, 8, 11, 20, 30, tzinfo=dt.timezone.utc)
        rows = [("worker-a", now - dt.timedelta(minutes=m)) for m in (1, 2, 3)]
        extra = [dict(seat_pool(tag="worker-b")["seats"][0])]
        adm = self.admission(mod, rows, seats=seat_pool(extra=extra), now=now)
        self.assertTrue(adm["would_launch"])
        self.assertEqual(adm["admitted_accounts"], ["worker-b"])
        p = build(mod, pre=pre("SPAWN_OK"), launch_admission=adm)
        self.assertEqual(p["verdict"], "READY_TO_GROW")
        self.assertEqual(p["accounts"]["launchable_seats"], 1)

    def test_unconsultable_gate_never_reads_as_a_green_light(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"),
                  launch_admission={"schema": "fak.dispatch-launch-admission.v1",
                                    "evaluated": False, "would_launch": None,
                                    "selected": "worker-a",
                                    "_error": "ledger unreadable"})
        self.assertEqual(p["verdict"], "LAUNCH_ADMISSION_UNKNOWN")
        self.assertIsNone(p["spawn_gate"]["would_launch"])
        self.assertIn("launch_admission.py", p["next_action"])

    def test_deferral_does_not_overwrite_a_more_specific_hold(self) -> None:
        mod = load()
        now = dt.datetime(2026, 8, 11, 20, 30, tzinfo=dt.timezone.utc)
        rows = [("worker-a", now - dt.timedelta(minutes=m)) for m in (1, 2, 3)]
        adm = self.admission(mod, rows, now=now)
        p = build(mod, pre=pre("REFUSE_AT_CAP", cap=2, live=2), launch_admission=adm)
        self.assertEqual(p["verdict"], "AT_CAP")  # the binding constraint, still
        self.assertTrue(any("LAUNCH_RATE_EXCEEDED" in r for r in p["reasons"]))
        self.assertFalse(p["spawn_gate"]["would_launch"])

    def test_candidate_accounts_lead_with_the_switcher_pick_and_skip_full_seats(self) -> None:
        mod = load()
        busy = {"seat": "worker-c", "tag": "worker-c", "dispatch_state": "busy",
                "session_cap": 1, "leased_slots": 1, "free_slots": 0}
        inv = seat_pool(tag="worker-b", extra=[busy])
        rows = mod.launch_candidate_accounts(inv, "worker-a")
        self.assertEqual([r["account"] for r in rows], ["worker-a", "worker-b"])
        self.assertIsNone(rows[0]["free_slots"])  # the pick is not itself a pool seat

    def test_no_candidate_account_is_not_a_deferral(self) -> None:
        mod = load()
        adm = mod.read_launch_admission(ROOT, {"account": {}}, {"seats": []})
        self.assertFalse(adm["evaluated"])
        p = build(mod, pre=pre("REFUSE_NO_ACCOUNT"), launch_admission=adm)
        self.assertEqual(p["verdict"], "BLOCKED_ON_ACCOUNT")

    def test_card_renders_the_admission_gate_and_next_action(self) -> None:
        mod = load()
        now = dt.datetime(2026, 8, 11, 20, 30, tzinfo=dt.timezone.utc)
        rows = [("worker-a", now - dt.timedelta(minutes=m)) for m in (1, 2, 3)]
        adm = self.admission(mod, rows, now=now)
        text = mod.render(build(mod, pre=pre("SPAWN_OK"),
                                seat_inventory=seat_pool(), launch_admission=adm))
        self.assertIn("admission :", text)
        self.assertIn("LAUNCH_RATE_EXCEEDED", text)
        self.assertIn("launchable=0/1", text)
        self.assertIn("next      :", text)

    def test_unmodelled_preflight_refusal_never_reads_as_growth(self) -> None:
        # The same #6495 failure one layer up: the card only NAMED four preflight
        # tokens and fell through to READY_TO_GROW on every other one — so a live
        # REFUSE_BIN_SKEW (#6508) host published "safe to spawn".
        mod = load()
        p = build(mod, pre=pre("REFUSE_BIN_SKEW"))
        self.assertNotEqual(p["verdict"], "READY_TO_GROW")
        self.assertEqual(p["verdict"], "BLOCKED_ON_BIN_SKEW")
        self.assertFalse(p["spawn_gate"]["would_launch"] or False)
        self.assertTrue(any("REFUSE_BIN_SKEW" in r for r in p["reasons"]))

    def test_unknown_preflight_token_holds_and_names_itself(self) -> None:
        mod = load()
        p = build(mod, pre=pre("REFUSE_SOMETHING_NEW"))
        self.assertEqual(p["verdict"], "BLOCKED_ON_PREFLIGHT")
        self.assertTrue(any("REFUSE_SOMETHING_NEW" in r for r in p["reasons"]))
        self.assertIn("dispatch_preflight", p["next_action"])

    def test_missing_preflight_verdict_is_not_a_yes(self) -> None:
        mod = load()
        doc = pre("SPAWN_OK")
        doc.pop("verdict")
        p = build(mod, pre=doc)
        self.assertEqual(p["verdict"], "BLOCKED_ON_PREFLIGHT")
        self.assertTrue(any("no preflight verdict" in r for r in p["reasons"]))

    def test_weekly_capped_preflight_keeps_its_specific_name(self) -> None:
        mod = load()
        p = build(mod, pre=pre("REFUSE_WEEKLY_CAPPED"))
        self.assertEqual(p["verdict"], "WEEKLY_CAPPED")

    def test_cooling_switcher_pick_is_asked_with_its_cooldown(self) -> None:
        # A cooling account holds NO free slot in the real seat pool, so it never
        # reaches the free-seat scan — the switcher's pick must still be asked with
        # its own reset, or a throttled pick reads as admitted by omission.
        mod = load()
        cooling = {"seat": "worker-a", "tag": "worker-a", "dispatch_state": "cooling",
                   "hold_reason": "rate_limited", "session_cap": 1, "leased_slots": 0,
                   "free_slots": 0,
                   "cooldown": {"next_eligible": "2026-08-11T22:00:00Z"}}
        rows = mod.launch_candidate_accounts({"seats": [cooling]}, "worker-a")
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["throttled"])
        self.assertEqual(rows[0]["throttle_reset"], "2026-08-11T22:00:00Z")
        adm = self.admission(mod, [], seats={"seats": [cooling]})
        self.assertEqual(adm["reason"], "ACCOUNT_THROTTLED")
        p = build(mod, pre=pre("SPAWN_OK"), launch_admission=adm)
        self.assertEqual(p["verdict"], "LAUNCH_COOLDOWN")

    def test_spawn_gate_carries_immediate_exit_evidence(self) -> None:
        # #6492's false-success signature, carried ON the growth gate: dead-pid
        # workers that produced nothing beside a spawn counter reporting 0 failures.
        mod = load()
        silent = [{"issue": 6492, "stamp": "20260811-2026", "log": "resolve-6492.log",
                   "pid": 15724, "size": 0, "kind": "empty"}]
        p = build(mod, pre=pre("SPAWN_OK"), silent=silent,
                  spawn_causes=spawn_causes(spawns=8, stale_cred=0))
        ev = p["spawn_gate"]["launch_evidence"]
        self.assertEqual(ev["immediate_exits"], 1)
        self.assertEqual(ev["issues"], [6492])
        self.assertEqual(ev["spawns"], 8)
        self.assertEqual(ev["spawn_failed"], 0)

    def test_md_doc_publishes_the_admission_state_and_next_action(self) -> None:
        mod = load()
        now = dt.datetime(2026, 8, 11, 20, 30, tzinfo=dt.timezone.utc)
        rows = [("worker-a", now - dt.timedelta(minutes=m)) for m in (1, 2, 3)]
        adm = self.admission(mod, rows, now=now)
        md = mod.render_md(build(mod, pre=pre("SPAWN_OK"), seat_inventory=seat_pool(),
                                 launch_admission=adm), date="2026-08-11")
        self.assertIn("launch admission", md)
        self.assertIn("LAUNCH_RATE_EXCEEDED", md)
        self.assertIn("next action", md)

    def test_contract_fields_are_present_without_an_admission_fold(self) -> None:
        # Back-compat: a caller that supplies no admission fold still gets the three
        # populated contract fields (#6495 done condition 3) and the old verdict.
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"))
        self.assertEqual(p["verdict"], "READY_TO_GROW")
        self.assertEqual(p["spawn_gate"]["preflight_verdict"], "SPAWN_OK")
        self.assertIsNone(p["spawn_gate"]["would_launch"])
        self.assertFalse(p["accounts"]["admission_evaluated"])
        self.assertEqual(p["accounts"]["selected"]["tag"], "worker-a")
        self.assertTrue(p["next_action"])


if __name__ == "__main__":
    unittest.main()
