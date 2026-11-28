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
