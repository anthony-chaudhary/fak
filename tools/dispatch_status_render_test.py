#!/usr/bin/env python3
"""Tests for tools/dispatch_status_render.py (split from dispatch_status_test.py)."""
from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
ROOT = _TEST_DIR.parent
_SHARED_CACHE: list[object] = []


def _shared():
    if not _SHARED_CACHE:
        spec = importlib.util.spec_from_file_location(
            "dispatch_status_test_helpers", _TEST_DIR / "dispatch_status_test.py"
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SHARED_CACHE.append(mod)
    return _SHARED_CACHE[0]


def load():
    return _shared().load()


def build(*args, **kwargs):
    return _shared().build(*args, **kwargs)

def pre(*args, **kwargs):
    return _shared().pre(*args, **kwargs)



class DispatchStatusRenderModuleTest(unittest.TestCase):
    def test_direct_module_import_exports(self) -> None:
        tools_dir = str(_TEST_DIR)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        mod = importlib.import_module("dispatch_status_render")
        self.assertEqual(mod._rate_str(0.5), "0.5")
        self.assertEqual(mod._join_limited(["a", "b"], limit=1), "a; +1 more")


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


if __name__ == "__main__":
    unittest.main()
