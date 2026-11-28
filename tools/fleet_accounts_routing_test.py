#!/usr/bin/env python3
"""Unit tests for tools/fleet_accounts_routing.py."""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent)


def _routing():
    if _TOOLS_DIR not in sys.path:
        sys.path.insert(0, _TOOLS_DIR)
    return importlib.import_module("fleet_accounts_routing")


def _worker(tag: str, *, tier: int = 1, product: str = "claude",
            available: bool = True, live: int = 0, active: int = 0,
            weight: int = 0, uuid: str = "", block_kind: str | None = None,
            block_reason: str = "", throttled: bool = False,
            reset: str | None = None) -> dict:
    acct = f"opencode-{tag}" if product == "opencode" else f".claude-{tag}"
    return {
        "kind": "worker",
        "account": acct,
        "tag": tag,
        "product": product,
        "dir": f"/tmp/{acct}",
        "model": f"model-t{tier}",
        "model_tier": tier,
        "available": available,
        "live_sessions": live,
        "active_sessions": active,
        "route_weight": weight,
        "account_uuid": uuid,
        "block_kind": block_kind,
        "block_reason": block_reason,
        "throttled": throttled,
        "reset": reset,
    }


class FleetAccountsRoutingTest(unittest.TestCase):
    def test_classify_task_work_kinds_and_heuristics(self) -> None:
        rt = _routing()
        self.assertEqual(rt.classify_task("audit everything", "gardening")["target_tier"], 2)
        self.assertEqual(rt.classify_task("hi", "engineering")["target_tier"], 1)
        self.assertEqual(rt.classify_task("say pong")["target_tier"], 2)
        self.assertEqual(rt.classify_task("implement a cache")["target_tier"], 1)
        self.assertEqual(rt.classify_task("", "tier3")["target_tier"], 3)

    def test_route_account_ranking_and_lease_exclusion(self) -> None:
        rt = _routing()
        rows = [
            _worker("a", tier=1, live=1, active=1),
            _worker("b", tier=1, live=0, active=5, weight=2),
            _worker("dup", tier=1, live=0) | {"identity_role": "duplicate"},
        ]
        res = rt.route_account(rows, "fix bug")
        self.assertTrue(res["ok"])
        self.assertEqual(res["account"]["tag"], "b")

        full_leases = [{"worker": f"w{i}", "tag": "b", "dir": "/tmp/.claude-b"} for i in range(50)]
        res2 = rt.route_account(rows, "fix bug", leases=full_leases)
        self.assertEqual(res2["account"]["tag"], "a")

    def test_resolve_account_pin_faklocal_and_missing(self) -> None:
        rt = _routing()
        with tempfile.TemporaryDirectory() as td:
            local = rt.resolve_account("faklocal", faklocal_ok=True, home=td)
            self.assertTrue(local["ok"])
            self.assertTrue(Path(local["config_dir"], "projects").is_dir())

        rows = [_worker("a", available=False, block_reason="usage limit")]
        blocked = rt.resolve_account("a", roster_fn=lambda *a, **kw: rows)
        self.assertFalse(blocked["ok"])
        self.assertEqual(blocked["block_reason"], "usage limit")

        missing = rt.resolve_account("ghost", roster_fn=lambda *a, **kw: rows)
        self.assertFalse(missing["ok"])

    def test_allocate_wave_spreads_across_pools_and_tracks_shortfall(self) -> None:
        rt = _routing()
        saved = os.environ.get("FAK_SESSIONS_PER_ACCOUNT")
        try:
            os.environ["FAK_SESSIONS_PER_ACCOUNT"] = "2"
            rows = [
                _worker("p1a", tier=1, uuid="u1", live=0),
                _worker("p1b", tier=1, uuid="u1", live=0),
                _worker("p2", tier=1, uuid="u2", live=0),
            ]
            leases = [{"worker": "w0", "tag": "p2", "dir": "/tmp/.claude-p2"}]
            wave = rt.allocate_wave(
                5,
                task_text="implement feature",
                leases=leases,
                roster_fn=lambda *a, **kw: rows,
            )
            self.assertTrue(wave["ok"])
            self.assertEqual(wave["granted"], 3)
            self.assertEqual(wave["shortfall"], 2)
            self.assertEqual(wave["distinct_pools"], 2)
            self.assertTrue(wave["wave_id"].startswith("wave-"))
            self.assertEqual([lane["rank"] for lane in wave["lanes"]], [0, 1, 2])
        finally:
            if saved is None:
                os.environ.pop("FAK_SESSIONS_PER_ACCOUNT", None)
            else:
                os.environ["FAK_SESSIONS_PER_ACCOUNT"] = saved

    def test_seat_pool_dispatch_states_double_booking_and_unbound(self) -> None:
        rt = _routing()
        rows = [
            _worker("free", tier=1, uuid="u1", available=True),
            _worker("cool", tier=1, uuid="u2", available=False, block_kind="usage",
                    throttled=True, reset="5pm") | {"throttled_since": "2026-06-25T10:00:00Z"},
            _worker("auth", tier=1, uuid="u3", available=False, block_kind="auth"),
            _worker("oc", tier=2, product="opencode", available=True),
        ]
        leases = [
            {"worker": "w1", "tag": "oc", "dir": "/tmp/opencode-oc"},
            {"worker": "w2", "tag": "oc", "dir": "/tmp/opencode-oc"},
            {"worker": "orphan", "tag": "gone", "dir": "/tmp/.claude-gone"},
        ]
        pool = rt.seat_pool(rows, leases=leases)
        by_tag = {s["tag"]: s for s in pool["seats"]}
        self.assertEqual(by_tag["cool"]["dispatch_state"], "cooling")
        self.assertEqual(by_tag["cool"]["cooldown"]["until"], "5pm")
        self.assertEqual(by_tag["auth"]["hold_reason"], "auth_failed")
        self.assertEqual(len(pool["double_booked"]), 1)
        self.assertEqual(len(pool["unbound_leases"]), 1)

    def test_live_seat_leases_reads_live_sidecars(self) -> None:
        rt = _routing()
        with tempfile.TemporaryDirectory() as td:
            Path(td, "resolve-1.pid").write_text("111\n", encoding="utf-8")
            Path(td, "resolve-1.account").write_text(
                json.dumps({"tag": "a", "dir": "/tmp/.claude-a"}), encoding="utf-8"
            )
            Path(td, "resolve-2.pid").write_text("222\n", encoding="utf-8")
            Path(td, "resolve-2.account").write_text(
                json.dumps({"tag": "b", "dir": "/tmp/.claude-b"}), encoding="utf-8"
            )
            leases = rt.live_seat_leases(
                runs_dir=td,
                alive={111},
                probe=lambda pid: {"alive": pid == 111, "cmdline": "claude -p 'resolve GitHub issue #1'"},
            )
        self.assertEqual(len(leases), 1)
        self.assertEqual(leases[0]["tag"], "a")

    def test_cli_arg_and_output_helpers(self) -> None:
        rt = _routing()
        self.assertEqual(rt._tier_arg(["--t2"]), ("t2", True))
        self.assertEqual(rt._tier_arg(["--tier=t1"]), ("t1", True))
        self.assertTrue(rt._has_arg(["--explain"], ("--explain",)))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rt._cli_seats({
                "product": "claude", "total_seats": 4, "free_seats": 4,
                "leased_seats": 0, "blocked_seats": 0, "depleted": False,
                "seats": [_worker("a", tier=1) | {"dispatch_state": "available", "workers": []}],
            })
        self.assertIn("seat pool [claude]", buf.getvalue())


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FleetAccountsRoutingTest)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
