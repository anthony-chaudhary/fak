#!/usr/bin/env python3
"""Hermetic tests for dispatch witness closure, no-commit classification, and loop ledger."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue_resolve_dispatch_test import ROOT, _no_window_creationflags, load  # noqa: E402


class LoopLedgerTest(unittest.TestCase):
    def test_record_loop_tick_writes_fire_admit_end_for_dry_run(self) -> None:
        mod = load()
        rows: list[dict[str, object]] = []

        def append(root, ledger, ev):
            rows.append(dict(ev))
            return {"ok": True, "kind": ev["kind"]}

        payload = {
            "schema": mod.SCHEMA,
            "workspace": str(ROOT),
            "live": False,
            "backend": "claude",
            "max_workers": 2,
            "preflight": {"verdict": "SPAWN_OK", "cap": 2, "live": 0},
            "account": {"tag": "worker-a", "tier": 1, "model": "opus"},
            "lane": "gateway",
            "lane_issue_count": 3,
            "target_issue": 717,
            "prompt_chars": 1200,
            "ok": True,
            "action": "would_spawn",
            "verdict": "WOULD_SPAWN",
            "reason": "safe to spawn one worker",
        }
        rec = mod.record_loop_tick(
            ROOT, payload, ledger=Path("loops.jsonl"), append=append,
            mint=lambda root, process: "RID-DISPATCH1")

        self.assertTrue(rec["ok"])
        self.assertEqual(rec["loop_id"], "issue-resolve-dispatch/claude")
        self.assertEqual([r["kind"] for r in rows], ["fire", "admit", "end"])
        self.assertEqual(rows[1]["status"], "admitted")
        self.assertEqual(rows[1]["reason"], "WOULD_SPAWN")
        self.assertEqual(rows[1]["metrics"]["target_issue"], 717)
        self.assertIn(("issue", "717"), rows[1]["evidence"])
        self.assertEqual(payload["run_id"], "RID-DISPATCH1")

    def test_record_loop_tick_refusal_has_fire_and_refused_admit(self) -> None:
        mod = load()
        mod.lane_tree = lambda root, lane: [f"internal/{lane}/**"]
        rows: list[dict[str, object]] = []
        payload = {
            "schema": mod.SCHEMA,
            "workspace": str(ROOT),
            "live": True,
            "backend": "opencode",
            "max_workers": 2,
            "preflight": {"verdict": "REFUSE_NO_ACCOUNT", "cap": 2, "live": 2},
            "account": {},
            "lane": "docs",
            "lane_issue_count": 8,
            "ok": False,
            "action": "refused",
            "verdict": "REFUSE_NO_ACCOUNT",
            "reason": "preflight refused: no account",
        }

        mod.record_loop_tick(
            ROOT,
            payload,
            ledger=Path("loops.jsonl"),
            append=lambda root, ledger, ev: (rows.append(dict(ev)) or {"ok": True, "kind": ev["kind"]}),
            mint=lambda root, process: "RID-DISPATCH2",
        )

        self.assertEqual([r["kind"] for r in rows], ["fire", "admit"])
        self.assertEqual(rows[1]["status"], "refused")
        self.assertEqual(rows[1]["reason"], "REFUSE_NO_ACCOUNT")
        self.assertEqual(rows[1]["metrics"]["preflight_live"], 2)
        # #4322: a refused tick's evidence carries the structured lane fields
        # (not scraped from summary prose) so per-lane collision rate is exact.
        self.assertIn(("lane", "docs"), rows[1]["evidence"])
        self.assertIn(("lane_kind", "cluster"), rows[1]["evidence"])
        self.assertIn(("mode", "shared"), rows[1]["evidence"])
        self.assertIn(("tree", "internal/docs/**"), rows[1]["evidence"])

    def test_record_loop_tick_admitted_tick_has_no_collision_evidence(self) -> None:
        # The structured lane/tree fields are additive on REFUSED ticks only —
        # an admitted (would_spawn/spawned) tick's evidence is unchanged.
        mod = load()
        rows: list[dict[str, object]] = []
        payload = {
            "schema": mod.SCHEMA,
            "workspace": str(ROOT),
            "live": False,
            "backend": "claude",
            "max_workers": 2,
            "preflight": {"verdict": "SPAWN_OK", "cap": 2, "live": 0},
            "account": {"tag": "worker-a"},
            "lane": "gateway",
            "lane_issue_count": 3,
            "target_issue": 717,
            "ok": True,
            "action": "would_spawn",
            "verdict": "WOULD_SPAWN",
            "reason": "safe to spawn one worker",
        }
        mod.record_loop_tick(
            ROOT, payload, ledger=Path("loops.jsonl"),
            append=lambda root, ledger, ev: (rows.append(dict(ev)) or {"ok": True, "kind": ev["kind"]}),
            mint=lambda root, process: "RID-DISPATCH3")

        for r in rows:
            self.assertFalse(any(k == "lane_kind" for k, _ in r["evidence"]))

    def test_record_loop_tick_lane_lease_held_prefers_lease_tree(self) -> None:
        # The fenced-CAS lease refusal carries its OWN requested tree (the
        # dispatch_tick.go convention: lane/lane_kind/mode/tree stamped on the
        # lease map) -- record_loop_tick reads it straight off payload["lease"]
        # rather than re-deriving it from the lane taxonomy.
        mod = load()
        mod.lane_tree = lambda root, lane: [f"SHOULD-NOT-BE-USED/{lane}/**"]
        rows: list[dict[str, object]] = []
        payload = {
            "schema": mod.SCHEMA,
            "workspace": str(ROOT),
            "live": True,
            "backend": "claude",
            "max_workers": 2,
            "preflight": {"verdict": "SPAWN_OK", "cap": 2, "live": 0},
            "account": {"tag": "worker-a"},
            "lane": "cmd",
            "lane_issue_count": 3,
            "target_issue": 900,
            "ok": False,
            "action": "lane_leased",
            "verdict": "LANE_LEASE_HELD",
            "reason": "lane \"cmd\" lease is held by a live peer",
            "lease": {"refused": True, "acquired": False,
                      "tree": ["cmd/fak/**", "internal/dispatchtick/**"]},
        }
        mod.record_loop_tick(
            ROOT, payload, ledger=Path("loops.jsonl"),
            append=lambda root, ledger, ev: (rows.append(dict(ev)) or {"ok": True, "kind": ev["kind"]}),
            mint=lambda root, process: "RID-DISPATCH4")

        self.assertEqual(rows[1]["status"], "refused")
        self.assertIn(("lane", "cmd"), rows[1]["evidence"])
        self.assertIn(("tree", "cmd/fak/**,internal/dispatchtick/**"), rows[1]["evidence"])

    def test_record_loop_tick_dirty_path_collision_carries_paths(self) -> None:
        mod = load()
        mod.lane_tree = lambda root, lane: [f"internal/{lane}/**"]
        rows: list[dict[str, object]] = []
        payload = {
            "schema": mod.SCHEMA,
            "workspace": str(ROOT),
            "live": True,
            "backend": "claude",
            "max_workers": 2,
            "preflight": {"verdict": "SPAWN_OK", "cap": 2, "live": 0},
            "account": {"tag": "worker-a"},
            "lane": "gateway",
            "lane_issue_count": 3,
            "target_issue": 901,
            "ok": False,
            "action": "dirty_path_collision",
            "verdict": "DIRTY_PATH_COLLISION",
            "reason": "dirty path collision",
            "dirty_path_collision": {"collides": True,
                                      "dirty_paths": ["internal/gateway/foo.go"]},
        }
        mod.record_loop_tick(
            ROOT, payload, ledger=Path("loops.jsonl"),
            append=lambda root, ledger, ev: (rows.append(dict(ev)) or {"ok": True, "kind": ev["kind"]}),
            mint=lambda root, process: "RID-DISPATCH5")

        self.assertIn(("paths", "internal/gateway/foo.go"), rows[1]["evidence"])
        self.assertIn(("tree", "internal/gateway/**"), rows[1]["evidence"])


class FleetTrendTickTest(unittest.TestCase):
    """#4594: each live dispatcher tick feeds the fleet-status trend ledger."""

    def test_appends_partial_live_row_from_preflight(self) -> None:
        import json
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            payload = {"backend": "codex", "preflight": {"verdict": "SPAWN_OK", "cap": 4, "live": 3}}
            out = mod.append_fleet_trend_row(root, payload, now="2026-07-14T12:00:00Z")
            self.assertTrue(out["ok"])
            ledger = root / ".fak" / "nightrun" / "fleet-status-history.jsonl"
            self.assertTrue(ledger.exists())
            rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ts"], "2026-07-14T12:00:00Z")
            self.assertEqual(rows[0]["scope"], "backend")
            self.assertEqual(rows[0]["backend"], "codex")
            self.assertEqual(rows[0]["backend_live"], 3.0)
            # Backend-local probes never masquerade as an aggregate fleet census.
            for absent in ("live", "usable", "sessions", "escalate"):
                self.assertNotIn(absent, rows[0])
            mod.append_fleet_trend_row(root, payload, now="2026-07-14T12:05:00Z")
            rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)

    def test_missing_preflight_records_scoped_zero(self) -> None:
        import json
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = mod.append_fleet_trend_row(root, {}, now="2026-07-14T12:00:00Z")
            self.assertTrue(out["ok"])
            row = json.loads(Path(out["ledger"]).read_text(encoding="utf-8").strip())
            self.assertEqual(row["scope"], "backend")
            self.assertEqual(row["backend"], "unknown")
            self.assertEqual(row["backend_live"], 0.0)
            self.assertNotIn("live", row)

    def test_append_failure_never_raises(self) -> None:
        mod = load()

        def boom(path, metrics, now, **kw):
            raise OSError("disk full")

        out = mod.append_fleet_trend_row(Path("z:/nope"), {}, append=boom)
        self.assertFalse(out["ok"])
        self.assertIn("disk full", out["error"])


class SubjectCitesIssueTest(unittest.TestCase):
    """The per-worker commit binding key (#1324 proposal #2): a subject is THIS
    worker's commit only when it names the worker's #issue at a word boundary —
    never a numeric prefix/suffix or a glued token."""

    def test_word_boundary_binding(self) -> None:
        mod = load()
        self.assertTrue(mod._subject_cites_issue("fix(tools): bind slot (#1324) (fak tools)", 1324))
        self.assertTrue(mod._subject_cites_issue("resolve #1324 now", 1324))
        # a longer number that merely CONTAINS the digits is not a match
        self.assertFalse(mod._subject_cites_issue("fix #13240 thing", 1324))
        self.assertFalse(mod._subject_cites_issue("fix #132 thing", 1324))
        # a glued token (leading word char) is not a binding reference
        self.assertFalse(mod._subject_cites_issue("xoxb-secret#1324", 1324))
        self.assertFalse(mod._subject_cites_issue("no reference here", 1324))


class WorkerResolvingShaTest(unittest.TestCase):
    """worker_resolving_sha picks the NEWEST commit citing the worker's issue, scoped
    to the per-worker base..HEAD window when known, and yields None (claims nothing)
    when no resolving commit exists — the wrong-issue / no-commit slot."""

    def test_picks_newest_matching_subject(self) -> None:
        mod = load()
        # git log is newest-first; only the second line cites #1324.
        log = "deadbeef\x1fchore: bump (#99)\ncafef00d\x1ffix: bind (#1324) (fak tools)\n"
        out = mod.worker_resolving_sha(ROOT, 1324, git=lambda root, args: (0, log))
        self.assertEqual(out, "cafef00d")

    def test_wrong_issue_only_yields_none(self) -> None:
        mod = load()
        log = "deadbeef\x1fchore: bump (#99)\ncafef00d\x1ffix: other (#42)\n"
        self.assertIsNone(mod.worker_resolving_sha(ROOT, 1324, git=lambda root, args: (0, log)))

    def test_base_sha_scopes_the_range(self) -> None:
        mod = load()
        seen: dict = {}
        def git(root, args):
            seen["args"] = list(args)
            return (0, "cafef00d\x1ffix: bind (#1324)\n")
        mod.worker_resolving_sha(ROOT, 1324, base_sha="base000", git=git)
        self.assertIn("base000..HEAD", seen["args"])

    def test_no_base_falls_back_to_recent_window(self) -> None:
        mod = load()
        seen: dict = {}
        def git(root, args):
            seen["args"] = list(args)
            return (0, "")
        mod.worker_resolving_sha(ROOT, 1324, git=git, scan_limit=42)
        self.assertIn("-n", seen["args"])
        self.assertIn("42", seen["args"])

    def test_git_error_fails_open_to_none(self) -> None:
        mod = load()
        self.assertIsNone(mod.worker_resolving_sha(ROOT, 1324, git=lambda root, args: (127, "")))


class AuditCommitWitnessTest(unittest.TestCase):
    """audit_commit_witness grades a sha through the DOS witness rung: witnessed only
    on verdict OK AND a diff-witness; everything else (ABSTAIN, subject-only, an empty
    audit) is the conservative not-witnessed verdict."""

    def test_ok_diff_witnessed_is_witnessed(self) -> None:
        mod = load()
        out = mod.audit_commit_witness(
            ROOT, "abc", runner=lambda root, sha: {"verdict": "OK", "witness": "diff-witnessed"})
        self.assertTrue(out["witnessed"])
        self.assertEqual(out["verdict"], "OK")

    def test_abstain_is_not_witnessed(self) -> None:
        mod = load()
        out = mod.audit_commit_witness(
            ROOT, "abc", runner=lambda root, sha: {"verdict": "ABSTAIN", "witness": "abstain"})
        self.assertFalse(out["witnessed"])

    def test_subject_only_is_not_witnessed(self) -> None:
        mod = load()
        out = mod.audit_commit_witness(
            ROOT, "abc", runner=lambda root, sha: {"verdict": "OK", "witness": "subject-only"})
        self.assertFalse(out["witnessed"])

    def test_empty_audit_fails_open_to_not_witnessed(self) -> None:
        mod = load()
        out = mod.audit_commit_witness(ROOT, "abc", runner=lambda root, sha: {})
        self.assertFalse(out["witnessed"])


class CommitAuditAbstainHoldsTest(unittest.TestCase):
    """Recent candidate-bound test commits whose audit ABSTAINS are surfaced as
    closure-witness mismatch holds instead of being selected again."""

    def test_test_scope_abstain_commit_holds_matching_candidate(self) -> None:
        mod = load()
        log = (
            "41f23ea7\x1f"
            "test(boundarylint): cover cmp-diff change-detector lint (#2867) (fak boundarylint)\n"
            "\x1e"
        )

        def git(root, args):
            return (0, log)

        def audit(root, sha):
            return {
                "sha": sha,
                "verdict": "ABSTAIN",
                "witness": "abstain",
                "claim_kind": "none",
                "reason": "subject makes no checkable code/test claim",
                "test_files": ["internal/boundarylint/changedetector_test.go"],
            }

        rows = mod.commit_audit_abstain_holds(
            ROOT, {2867, 9999}, git=git, audit_runner=audit)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["issue"], 2867)
        self.assertEqual(rows[0]["code"], "COMMIT_AUDIT_ABSTAIN")
        self.assertEqual(rows[0]["test_files"],
                         ["internal/boundarylint/changedetector_test.go"])


class WitnessExitedWorkersTest(unittest.TestCase):
    """The commit-time witness sweep (#1324 proposal #2): each FINISHED (dead-pid)
    worker's slot is graded via dos commit-audit and recorded CLAIM_WITNESSED /
    CLAIM_UNWITNESSED / CLAIM_NO_COMMIT — never a silent productive claim. Dead-pid
    gated and write-on-live, exactly like prune_dead_sidecars."""

    def _mk(self, runs: Path, issue: int, stamp: str, *, pid: int,
            base: str | None = "base000") -> Path:
        log = runs / f"resolve-{issue}-{stamp}.log"
        log.write_text("worker output here", encoding="utf-8")
        log.with_suffix(".pid").write_text(str(pid), encoding="utf-8")
        if base is not None:
            log.with_suffix(".basesha").write_text(base, encoding="utf-8")
        return log

    @staticmethod
    def _dead(pid):
        return {"alive": False}

    def test_unwitnessed_commit_is_recorded_not_silently_productive(self) -> None:
        # The issue's acceptance witness #2: a worker that exits with an unwitnessed
        # commit is recorded CLAIM_UNWITNESSED, and a .witness sidecar persists it.
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = self._mk(runs, 1324, "20260629-120000", pid=4242)
            def git(root, args):
                return (0, "cafef00d\x1ffix: bind (#1324)\n")
            def audit(root, sha):
                return {"verdict": "ABSTAIN", "witness": "abstain"}
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=git, audit_runner=audit)
            self.assertEqual(len(out["unwitnessed"]), 1)
            self.assertEqual(out["unwitnessed"][0]["claim"], "CLAIM_UNWITNESSED")
            self.assertEqual(out["unwitnessed"][0]["sha"], "cafef00d")
            side = log.with_suffix(".witness")
            self.assertTrue(side.exists())
            self.assertEqual(json.loads(side.read_text(encoding="utf-8"))["claim"],
                             "CLAIM_UNWITNESSED")

    def test_diff_witnessed_commit_claims_the_slot(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 1324, "20260629-120100", pid=4243)
            def git(root, args):
                return (0, "cafef00d\x1ffix: bind (#1324)\n")
            def audit(root, sha):
                return {"verdict": "OK", "witness": "diff-witnessed"}
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=git, audit_runner=audit)
            self.assertEqual(len(out["witnessed"]), 1)
            self.assertEqual(out["witnessed"][0]["claim"], "CLAIM_WITNESSED")

    def test_witness_carries_exact_lease_session_identity(self) -> None:
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = self._mk(runs, 3330, "20260629-120125", pid=4244)
            log.with_suffix(".lease").write_text(json.dumps({
                "id": "resolve-sessionmine", "holder": "worker",
                "session_id": "sess-3330", "tree": ["internal/sessionmine/**"],
            }), encoding="utf-8")
            def git(root, args):
                return (0, "abc1234\x1ffeat: bind (#3330)\n")
            def audit(root, sha):
                return {"verdict": "OK", "witness": "diff-witnessed"}
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=git, audit_runner=audit)
            self.assertEqual(out["witnessed"][0]["session_id"], "sess-3330")
            self.assertEqual(json.loads(log.with_suffix(".witness").read_text(encoding="utf-8"))["session_id"], "sess-3330")

    def test_dead_worker_releases_recorded_lane_lease(self) -> None:
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = self._mk(runs, 1324, "20260629-120150", pid=4248)
            log.with_suffix(".lease").write_text(json.dumps({
                "id": "resolve-docs",
                "holder": "session-1",
                "generation": 5,
                "tree": ["docs/**"],
            }), encoding="utf-8")
            def git(root, args):
                return (0, "cafef00d\x1ffix: bind (#1324)\n")
            def audit(root, sha):
                return {"verdict": "OK", "witness": "diff-witnessed"}
            seen: dict = {}
            def lease_runner(root, args, **k):
                seen["args"] = list(args)
                return {"rc": 0, "verdict": {"ok": True}}
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=git, audit_runner=audit,
                                             lease_runner=lease_runner)
            self.assertEqual(len(out["lease_released"]), 1)
            self.assertEqual(out["lease_released"][0]["id"], "resolve-docs")
            argv = seen["args"]
            self.assertEqual(argv[0], "release")
            self.assertEqual(argv[argv.index("--holder") + 1], "session-1")
            self.assertEqual(argv[argv.index("--generation") + 1], "5")
            self.assertNotIn("--force", argv)
            side = json.loads(log.with_suffix(".witness").read_text(encoding="utf-8"))
            self.assertTrue(side["lease_release"]["released"])

    def test_no_resolving_commit_is_claim_no_commit(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 1324, "20260629-120200", pid=4244)
            # git finds nothing citing #1324 -> the worker landed no resolving commit.
            def git(root, args):
                return (0, "deadbeef\x1fchore: unrelated (#5)\n")
            def boom(root, sha):
                raise AssertionError("no audit without a sha")
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=git, audit_runner=boom)
            self.assertEqual(len(out["no_commit"]), 1)
            self.assertEqual(out["no_commit"][0]["claim"], "CLAIM_NO_COMMIT")
            self.assertIsNone(out["no_commit"][0]["sha"])

    def test_live_worker_is_not_audited(self) -> None:
        # A still-running worker may not have committed yet — never mis-blame it.
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 1324, "20260629-120300", pid=4245)
            def alive_probe(pid):
                return {"alive": True, "create_time": 0.0,
                        "name": "claude.exe",
                        "cmdline": "claude -p resolve GitHub issue #1324"}
            def boom_git(root, args):
                raise AssertionError("a live worker must not be audited")
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=alive_probe,
                                             git=boom_git)
            self.assertEqual(out["audited"], [])

    def test_already_witnessed_is_skipped(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = self._mk(runs, 1324, "20260629-120400", pid=4246)
            log.with_suffix(".witness").write_text('{"claim": "CLAIM_WITNESSED"}',
                                                   encoding="utf-8")
            def boom_git(root, args):
                raise AssertionError("an already-witnessed worker must not be re-audited")
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=boom_git)
            self.assertEqual(out["audited"], [])

    def test_already_witnessed_dead_worker_releases_recorded_lane_lease(self) -> None:
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = self._mk(runs, 1324, "20260629-120450", pid=4249)
            log.with_suffix(".witness").write_text(
                json.dumps({"claim": "CLAIM_WITNESSED"}), encoding="utf-8")
            log.with_suffix(".lease").write_text(json.dumps({
                "id": "resolve-tools",
                "holder": "session-2",
                "generation": 8,
                "tree": ["tools/**"],
            }), encoding="utf-8")
            def boom_git(root, args):
                raise AssertionError("an already-witnessed worker must not be re-audited")
            seen: dict = {}
            def lease_runner(root, args, **k):
                seen["args"] = list(args)
                return {"rc": 0, "verdict": {"ok": True}}
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=boom_git, lease_runner=lease_runner)
            self.assertEqual(out["audited"], [])
            self.assertEqual(out["lease_release_retried"], [
                {
                    "log": log.name,
                    "id": "resolve-tools",
                    "reason": "already_witnessed_dead_worker",
                    "released": True,
                }
            ])
            self.assertEqual(len(out["lease_released"]), 1)
            self.assertEqual(out["lease_release_failed"], [])
            self.assertEqual(out["lease_released"][0]["id"], "resolve-tools")
            argv = seen["args"]
            self.assertEqual(argv[0], "release")
            self.assertEqual(argv[argv.index("--holder") + 1], "session-2")
            self.assertEqual(argv[argv.index("--generation") + 1], "8")
            side = json.loads(log.with_suffix(".witness").read_text(encoding="utf-8"))
            self.assertTrue(side["lease_release"]["released"])

    def test_dry_run_audits_but_writes_no_sidecar(self) -> None:
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = self._mk(runs, 1324, "20260629-120500", pid=4247)
            log.with_suffix(".lease").write_text(json.dumps({
                "id": "resolve-docs", "holder": "session-1", "generation": 5,
            }), encoding="utf-8")
            def git(root, args):
                return (0, "cafef00d\x1ffix: bind (#1324)\n")
            def audit(root, sha):
                return {"verdict": "ABSTAIN", "witness": "abstain"}
            def boom_release(root, args, **k):
                raise AssertionError("dry-run witness must not release a lease")
            out = mod.witness_exited_workers(runs, ROOT, live=False, probe=self._dead,
                                             git=git, audit_runner=audit,
                                             lease_runner=boom_release)
            self.assertEqual(out["unwitnessed"][0]["claim"], "CLAIM_UNWITNESSED")
            self.assertFalse(log.with_suffix(".witness").exists())
            self.assertEqual(out["lease_released"], [])

    def test_no_pid_sidecar_is_not_auditable(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # a log with no .pid sidecar: we cannot prove it finished.
            (runs / "resolve-1324-20260629-120600.log").write_text("x", encoding="utf-8")
            def boom_git(root, args):
                raise AssertionError("a worker we cannot prove finished must not be audited")
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=boom_git)
            self.assertEqual(out["audited"], [])


class ClassifyNoCommitReasonTest(unittest.TestCase):
    """classify_no_commit_reason tags WHY a finished worker shipped no commit, from
    the log tail. Each documented terminal-block signature maps to its reason; an
    unrecognized log stays UNKNOWN (no false positive)."""

    def _write(self, text: str):
        import tempfile
        d = tempfile.mkdtemp()
        p = Path(d) / "resolve-1338-20260629-221102.log"
        p.write_text(text, encoding="utf-8")
        return p

    def test_self_modify(self) -> None:
        mod = load()
        p = self._write("...\nfinish=end_turn safety=blocked:1 reason=SELF_MODIFY\n"
                        "[fak] refused 1 tool call(s): Bash (SELF_MODIFY/ESCALATE).\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_SELF_MODIFY)

    def test_policy_block(self) -> None:
        mod = load()
        p = self._write("...\nfinish=end_turn safety=blocked:1 reason=POLICY_BLOCK\n"
                        "[fak] refused 1 tool call(s): Bash (POLICY_BLOCK/TERMINAL).\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_POLICY_BLOCK)

    def test_auth_wall_glm(self) -> None:
        mod = load()
        p = self._write("> build · glm-5.2\nWeekly Limit Exhausted. Your limit will "
                        "reset at 2026-07-01.\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_AUTH_WALL)

    def test_auth_wall_claude_cap_banner(self) -> None:
        mod = load()
        # _CAP_BANNER_RE: "hit your … limit" (Claude/codex wording).
        p = self._write("You've hit your weekly usage limit for Claude.\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_AUTH_WALL)

    def test_auth_wall_guarded_provider_429(self) -> None:
        mod = load()
        p = self._write("fak-turn trace=guard FAILED reason=rate_limited\n"
                        "API Error: Request rejected (429) · upstream rate-limited "
                        "the request (HTTP 429)\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_AUTH_WALL)

    def test_auth_wall_missing_openai_api_key(self) -> None:
        mod = load()
        p = self._write("ERROR: Missing environment variable: `OPENAI_API_KEY`.\n"
                        "ERROR: Missing environment variable: `OPENAI_API_KEY`.\n"
                        "fak guard: codex exited abnormally (code 1).\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_AUTH_WALL)

    def test_off_trunk(self) -> None:
        mod = load()
        p = self._write("git commit refused: OFF_TRUNK — work on main only.\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_OFF_TRUNK)

    def test_preview_confirm_feedback(self) -> None:
        mod = load()
        p = self._write(
            "tool call refused: REQUIRE_WITNESS/ESCALATE\n"
            "{\"_fak_confirm\":\"fak-3bff999a1f01335d\"}\n"
            "step=20\n"
            "exiting loop\n")
        self.assertEqual(mod.classify_no_commit_reason(p),
                         mod.NO_COMMIT_PREVIEW_CONFIRM)

    def test_banner_noop_small_log(self) -> None:
        mod = load()
        # The 122-byte opencode no-op: spawn banner + "> build · glm-…" and nothing.
        p = self._write("# fak-spawn 20260629-221102 issue=1338 lane=docs backend=opencode\n"
                        "> build · glm-4.5-air\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_BANNER_NOOP)

    def test_banner_in_large_real_log_is_not_noop(self) -> None:
        mod = load()
        # A real worker also prints "> build · …" at startup, but its log is large and
        # carries real turns — it must NOT be misread as a banner no-op.
        big = "> build · glm-5.2\n" + ("fak-turn trace=guard ok saved=80k tok\n" * 200)
        p = self._write(big)
        # No banner no-op, and (#5870) no bare `unknown` either: the tail carries no
        # guard epilogue, so the run reads as killed mid-turn.
        self.assertEqual(mod.classify_no_commit_reason(p),
                         mod.NO_COMMIT_DIED_BEFORE_EPILOGUE)

    def test_restart_exhausted_is_typed_with_count_and_cause(self) -> None:
        mod = load()
        p = self._write("")
        p.write_text("fak guard: managed-context status restart_exhausted count=3 "
                     "dominant_cause=BUDGET_CONTEXT_EXHAUSTED from_trace=t\n", encoding="utf-8")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_RESTART_EXHAUSTED)
        self.assertEqual(mod.classify_restart_exhaustion(p), {
            "reason": mod.NO_COMMIT_RESTART_EXHAUSTED, "restart_count": 3,
            "dominant_cause": "BUDGET_CONTEXT_EXHAUSTED"})

    def test_live_reset_limit_corpus_is_typed(self) -> None:
        mod = load()
        p = self._write("")
        p.write_text("fak guard: managed-context status reset_limit limit=16 "
                     "reason=BUDGET_CONTEXT_EXHAUSTED continuity=degraded\n", encoding="utf-8")
        self.assertEqual(mod.classify_restart_exhaustion(p)["restart_count"], 16)
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_RESTART_EXHAUSTED)

    def test_live_reset_limit_survives_long_guard_epilogue(self) -> None:
        mod = load()
        p = self._write(
            "fak guard: managed-context status reset_limit limit=16 "
            "reason=BUDGET_CONTEXT_EXHAUSTED continuity=degraded\n"
            + ("resource and audit summary padding\n" * 190)
            + "fak guard: claude exited abnormally (code 1).\n")
        # The live #2788 guard epilogue put the typed terminal more than 4 KiB
        # before EOF. The precise restart scan is larger but remains bounded.
        self.assertGreater(p.stat().st_size, mod._CAP_TAIL_BYTES)
        self.assertEqual(mod.classify_restart_exhaustion(p)["restart_count"], 16)
        self.assertEqual(mod.classify_no_commit_reason(p),
                         mod.NO_COMMIT_RESTART_EXHAUSTED)

    def test_residual_with_no_signature_is_typed_not_unknown(self) -> None:
        mod = load()
        # #5870: the residual is no longer a bare `unknown`. This log matches no
        # failure signature AND carries no guard epilogue -> died_before_epilogue.
        p = self._write("the worker ran a few turns and then exited cleanly\n")
        # Asserted with the PRE-EXISTING constant first, so this fails as a
        # wrong-behavior assertion on the old sweep, not merely as a missing symbol.
        self.assertNotEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_UNKNOWN)
        self.assertEqual(mod.classify_no_commit_reason(p),
                         mod.NO_COMMIT_DIED_BEFORE_EPILOGUE)

    def test_missing_log_artifact(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "resolve-1338-20260629-221102.log"
            self.assertEqual(mod.classify_no_commit_reason(p),
                             mod.NO_COMMIT_MISSING_LOG)

    def test_self_modify_wins_over_banner(self) -> None:
        mod = load()
        # Priority: a structural guard block is reported even if a startup banner is
        # also present (the guard refusal is the actionable cause).
        p = self._write("> build · glm-5.2\n...\nreason=SELF_MODIFY\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_SELF_MODIFY)


class RefineUnknownNoCommitTest(unittest.TestCase):
    """The residual no-commit bucket is TYPED AT WRITE TIME (#5870), by the same two
    markers internal/dispatchconservation applies at read time (#5867), so the .witness
    sidecar the dispatcher routes off is self-describing instead of an opaque blob."""

    # `guardSection` (cmd/fak/guard_format_layout.go) pads the rule to 60 columns.
    def _section(self, name: str) -> str:
        head = "── guard · " + name + " "
        return head + "─" * max(0, 60 - len(head)) + "\n"

    def test_guard_epilogue_means_clean_exit(self) -> None:
        mod = load()
        tail = ("fak-turn trace=guard ok\n" + self._section("audit")
                + "  refused                    0\n")
        self.assertEqual(mod.refine_unknown_no_commit(tail),
                         mod.NO_COMMIT_CLEAN_EXIT)

    def test_epilogue_without_a_cache_window_section_still_counts(self) -> None:
        mod = load()
        # THE #5867 LESSON, pinned. `guard · cache window` is emitted only when the
        # session recorded cache turns, so keying on it books a quiet clean exit as a
        # death. Keying on the SECTION RULE has no such blind spot: an epilogue whose
        # only section is `audit` is still an epilogue.
        tail = self._section("audit") + "  refused                    0\n"
        self.assertNotIn("cache window", tail)
        self.assertEqual(mod.refine_unknown_no_commit(tail),
                         mod.NO_COMMIT_CLEAN_EXIT)

    def test_no_epilogue_means_died_before_epilogue(self) -> None:
        mod = load()
        tail = "fak-turn trace=e3f9 in-flight saved=12k tok\n"
        self.assertEqual(mod.refine_unknown_no_commit(tail),
                         mod.NO_COMMIT_DIED_BEFORE_EPILOGUE)

    def test_spawn_failure_wins_over_a_partial_epilogue(self) -> None:
        mod = load()
        # A guard that could not exec the agent ALSO prints a partial epilogue, and
        # "the child never started" is the more specific — and more fixable — claim.
        tail = (self._section("audit")
                + 'fak guard: could not run "claude": exec: not found\n')
        self.assertEqual(mod.refine_unknown_no_commit(tail),
                         mod.NO_COMMIT_GUARD_SPAWN_FAILED)

    def test_empty_tail_fails_open_to_died_before_epilogue(self) -> None:
        mod = load()
        self.assertEqual(mod.refine_unknown_no_commit(""),
                         mod.NO_COMMIT_DIED_BEFORE_EPILOGUE)

    def test_classifier_never_returns_a_bare_unknown(self) -> None:
        mod = load()
        # The writer's vocabulary is now total over the residual: `unknown` survives
        # only in HISTORIC sidecars, which is what keeps it meaningful as a
        # vocabulary-drift alarm on the reading side.
        self.assertNotIn(mod.NO_COMMIT_UNKNOWN, {
            mod.refine_unknown_no_commit(t) for t in
            ("", "nothing here\n", self._section("audit"),
             "fak guard: could not run \"x\": e\n")})

    def test_refined_classes_are_not_added_to_the_reblock_hold_set(self) -> None:
        mod = load()
        # MEASURED DECISION, pinned so it is not "fixed" by assumption. #5869 predicted
        # that typing this bucket would lift its re-block streak hold from 6 runs/1.5h
        # to 24 runs/7.3h "with no change here". Replaying the same window (2026-08-04
        # .. 08-07, 293 finished .witness records / 95.3h over 121 issues) against the
        # typed sidecars: with the hold set UNCHANGED the yield is bit-identical — 6
        # runs / 1.5h, the same six runs — because a `clean_exit_no_commit` streak is
        # not a streak of a re-blockable reason. Adding clean_exit_no_commit here DOES
        # reach 33 runs / 10.9h, but it suppresses SIX runs that landed a witnessed
        # commit (62b998c75, 9f1558700, 0573fbe1d, b6a80c576, b4d656107, 0d0255ad8),
        # destroying the zero-witnessed-loss property that is #5869's whole safety
        # argument. A clean exit that lands nothing is not a refusal to re-block on.
        for reason in (mod.NO_COMMIT_CLEAN_EXIT, mod.NO_COMMIT_DIED_BEFORE_EPILOGUE,
                       mod.NO_COMMIT_GUARD_SPAWN_FAILED):
            self.assertNotIn(reason, mod._HOLD_NO_COMMIT_REASONS)
        self.assertEqual(mod.held_no_commit_issues({"no_commit": [
            {"issue": 11, "reason": mod.NO_COMMIT_CLEAN_EXIT},
            {"issue": 22, "reason": mod.NO_COMMIT_DIED_BEFORE_EPILOGUE},
            {"issue": 33, "reason": mod.NO_COMMIT_GUARD_SPAWN_FAILED}]}), set())


class HeldNoCommitIssuesTest(unittest.TestCase):
    """held_no_commit_issues reads the recorded no-commit reason and HOLDS only the
    re-blockable guard refusals; a transient auth_wall or an UNKNOWN is left to the
    time cooldown, not structurally held (#1396)."""

    def test_holds_reblockable_guard_feedback(self) -> None:
        mod = load()
        w = {"no_commit": [
            {"issue": 11, "reason": mod.NO_COMMIT_SELF_MODIFY},
            {"issue": 22, "reason": mod.NO_COMMIT_POLICY_BLOCK},
            {"issue": 33, "reason": mod.NO_COMMIT_PREVIEW_CONFIRM}]}
        self.assertEqual(mod.held_no_commit_issues(w), {11, 22, 33})

    def test_does_not_hold_auth_wall_banner_offtrunk_or_unknown(self) -> None:
        mod = load()
        w = {"no_commit": [
            {"issue": 33, "reason": mod.NO_COMMIT_AUTH_WALL},
            {"issue": 44, "reason": mod.NO_COMMIT_BANNER_NOOP},
            {"issue": 55, "reason": mod.NO_COMMIT_OFF_TRUNK},
            {"issue": 66, "reason": mod.NO_COMMIT_UNKNOWN},
            {"issue": 77, "reason": mod.NO_COMMIT_MISSING_LOG}]}
        self.assertEqual(mod.held_no_commit_issues(w), set())

    def test_mixed_holds_only_reblockable(self) -> None:
        mod = load()
        w = {"no_commit": [
            {"issue": 11, "reason": mod.NO_COMMIT_SELF_MODIFY},
            {"issue": 33, "reason": mod.NO_COMMIT_AUTH_WALL}]}
        self.assertEqual(mod.held_no_commit_issues(w), {11})

    def test_failopen_on_skipped_empty_or_odd_record(self) -> None:
        mod = load()
        self.assertEqual(mod.held_no_commit_issues(None), set())
        self.assertEqual(mod.held_no_commit_issues({"skipped": True}), set())
        self.assertEqual(mod.held_no_commit_issues({"no_commit": []}), set())
        # an odd record (no issue / wrong type) is skipped, never raised
        self.assertEqual(mod.held_no_commit_issues(
            {"no_commit": [{"reason": mod.NO_COMMIT_SELF_MODIFY}, "garbage",
                           {"issue": "x", "reason": mod.NO_COMMIT_SELF_MODIFY}]}), set())


class AppendFleetTrendRowTest(unittest.TestCase):
    """Backend dispatcher ticks preserve scoped diagnostics without claiming an
    authoritative all-backend fleet census. The producer remains best-effort and bounded."""

    def test_live_tick_appends_backend_scoped_row(self) -> None:
        mod = load()
        import fleet_trend
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ledger = str(root / fleet_trend.DEFAULT_LEDGER)
            res = mod.append_fleet_trend_row(
                root, {"backend": "codex", "preflight": {"live": 3}}, now="2026-07-14T17:00:00Z")
            self.assertTrue(res["ok"])
            self.assertEqual(res["ledger"], ledger)
            rows = fleet_trend.tail(ledger, 24)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ts"], "2026-07-14T17:00:00Z")
            self.assertEqual(rows[0]["backend_live"], 3.0)
            self.assertEqual(rows[0]["scope"], "backend")
            self.assertEqual(rows[0]["backend"], "codex")
            self.assertNotIn("live", rows[0])
            # a second live tick APPENDS (a trend needs history), never overwrites.
            mod.append_fleet_trend_row(
                root, {"backend": "codex", "preflight": {"live": 1}}, now="2026-07-14T17:05:00Z")
            self.assertEqual([r["backend_live"] for r in fleet_trend.tail(ledger, 24)],
                             [3.0, 1.0])  # both ticks recorded, oldest→newest

    def test_missing_preflight_live_records_scoped_zero(self) -> None:
        # Missing backend-local evidence remains an explicitly scoped zero; it never
        # fabricates an aggregate fleet-wide zero.
        mod = load()
        import fleet_trend
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ledger = str(root / fleet_trend.DEFAULT_LEDGER)
            res = mod.append_fleet_trend_row(root, {}, now="2026-07-14T17:00:00Z")
            self.assertTrue(res["ok"])
            row = fleet_trend.tail(ledger, 24)[0]
            self.assertEqual(row["backend_live"], 0.0)
            self.assertNotIn("live", row)

    def test_append_error_is_swallowed_never_fails_the_tick(self) -> None:
        # Best-effort: an append that raises must be reported, not propagated — a
        # trend-ledger hiccup can never take down the dispatcher tick that feeds it.
        mod = load()

        def boom(*_a, **_k):
            raise OSError("disk full")

        with tempfile.TemporaryDirectory() as d:
            res = mod.append_fleet_trend_row(
                Path(d), {"preflight": {"live": 2}}, append=boom, now="t")
            self.assertFalse(res["ok"])
            self.assertIn("disk full", res["error"])


class SpawnFailedCauseTest(unittest.TestCase):
    """classify_spawn_failed_cause pins each cause bucket on a sample worker-log
    tail — the ~4% SPAWN_FAILED noise made attributable, never a false blame (#2635)."""

    HEADER = "# fak-spawn 20260710-010203 issue=1515 lane=cmd backend=claude argv0=claude.exe\n"

    def test_weekly_limit_banner(self) -> None:
        mod = load()
        tail = self.HEADER + "You've hit your weekly limit · resets 4am (America/Los_Angeles)\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": tail}),
                         mod.SPAWN_CAUSE_WEEKLY_LIMIT)

    def test_glm_wall_is_weekly_limit(self) -> None:
        mod = load()
        tail = self.HEADER + "Limit Exhausted — usage limit reached\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": tail}),
                         mod.SPAWN_CAUSE_WEEKLY_LIMIT)

    def test_stale_cred_auth_gap(self) -> None:
        mod = load()
        tail = self.HEADER + "Missing environment variable: `ANTHROPIC_API_KEY`\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": tail}),
                         mod.SPAWN_CAUSE_STALE_CRED)
        login = self.HEADER + "no ChatGPT subscription auth.json was resolved. Run `codex login`\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": login}),
                         mod.SPAWN_CAUSE_STALE_CRED)

    def test_child_crash_traceback(self) -> None:
        mod = load()
        tail = self.HEADER + "Traceback (most recent call last):\n  File ...\nRuntimeError: boom\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": tail}),
                         mod.SPAWN_CAUSE_CHILD_CRASH)
        oom = self.HEADER + "fatal error: runtime: out of memory\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": oom}),
                         mod.SPAWN_CAUSE_CHILD_CRASH)

    def test_exec_race_header_only_and_empty(self) -> None:
        mod = load()
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": self.HEADER}),
                         mod.SPAWN_CAUSE_EXEC_RACE)
        self.assertEqual(
            mod.classify_spawn_failed_cause({"tail": "", "silent": True, "log_bytes": 0}),
            mod.SPAWN_CAUSE_EXEC_RACE)

    def test_unknown_unrecognized_child_output(self) -> None:
        mod = load()
        tail = self.HEADER + "some novel shutdown line we do not recognize yet\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": tail}),
                         mod.SPAWN_CAUSE_UNKNOWN)

    def test_precedence_quota_beats_crash(self) -> None:
        # A cap banner that also mentions a fatal error is still a weekly_limit,
        # not a child_crash — most-specific (quota) wins.
        mod = load()
        tail = self.HEADER + "HTTP 429 rate-limited\nfatal error: aborted\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": tail}),
                         mod.SPAWN_CAUSE_WEEKLY_LIMIT)


class OpenWitnessedDispositionTest(unittest.TestCase):
    """#5071: the pre-dispatch OPEN_WITNESSED guard against a REAL temporary git
    history — the #2850 fixture (open issue + resolving commit already in trunk
    ancestry) plus a concrete non-witnessed control that remains eligible."""

    @staticmethod
    def _git(repo: Path, *args: str) -> None:
        base = ["git", "-C", str(repo), "-c", "user.email=t@t",
                "-c", "user.name=t", "-c", "core.hooksPath=",
                "-c", "commit.gpgsign=false"]
        subprocess.run(base + list(args), check=True, capture_output=True, text=True,
                       creationflags=_no_window_creationflags())

    @staticmethod
    def _head_sha(repo: Path) -> str:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True,
                             creationflags=_no_window_creationflags())
        return out.stdout.strip()

    def _repo_with_resolving_commit(self, repo: Path) -> str:
        """One trunk commit whose subject cites #2850 (the resolving grammar)."""
        self._git(repo, "init", "-q", "-b", "main")
        (repo / "a.go").write_text("package a\n", encoding="utf-8")
        self._git(repo, "add", "a.go")
        self._git(repo, "commit", "-q", "--no-gpg-sign", "-m",
                  "fix(dispatch): close pre-witnessed guard (#2850) (fak tools)")
        return self._head_sha(repo)

    def test_witnessed_trunk_commit_disposes_candidate_control_stays(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            sha = self._repo_with_resolving_commit(repo)
            audits = {sha: {"sha": sha, "verdict": "OK",
                            "witness": "diff-witnessed",
                            "claim_kind": "code_effect"}}
            rows = mod.open_witnessed_dispositions(
                repo, {2850, 2293},
                audit_runner=lambda root, s, **k: audits.get(s, {}))
            # #2850 is disposed with its witnessing SHA...
            self.assertEqual([r["issue"] for r in rows], [2850])
            self.assertEqual(rows[0]["sha"], sha)
            self.assertEqual(rows[0]["code"], "OPEN_WITNESSED")
            self.assertEqual(rows[0]["witness"], "diff-witnessed")
            # ...and the control (#2293, no resolving commit anywhere) is not.

    def test_unwitnessed_resolving_cite_is_not_shipped(self) -> None:
        """A mere subject mention without the diff-witness keep-bit must not
        dispose the candidate (preserve the existing witness rules)."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            sha = self._repo_with_resolving_commit(repo)
            audits = {sha: {"sha": sha, "verdict": "ABSTAIN", "witness": "abstain"}}
            rows = mod.open_witnessed_dispositions(
                repo, {2850},
                audit_runner=lambda root, s, **k: audits.get(s, {}))
            self.assertEqual(rows, [])

    def test_doc_claim_does_not_dispose_issue(self) -> None:
        """A diff-witnessed doc/triage claim witnesses a note, never the
        candidate's feature — the issue stays eligible (#2998 rule shared)."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            sha = self._repo_with_resolving_commit(repo)
            audits = {sha: {"sha": sha, "verdict": "OK",
                            "witness": "diff-witnessed", "claim_kind": "doc"}}
            rows = mod.open_witnessed_dispositions(
                repo, {2850},
                audit_runner=lambda root, s, **k: audits.get(s, {}))
            self.assertEqual(rows, [])

    def test_git_error_fails_open(self) -> None:
        mod = load()
        rows = mod.open_witnessed_dispositions(
            Path("z:/nope"), {2850},
            git=lambda root, args, **k: (1, ""),
            audit_runner=lambda root, s, **k: {})
        self.assertEqual(rows, [])


class CloseOpenWitnessedTest(unittest.TestCase):
    """#5071: dispositions transition through the EXISTING trusted close path
    (issue_resolve_witnessed.py via a synthetic OPEN_WITNESSED --audit-json)."""

    ROW = {"issue": 2850, "sha": "f8aff29dfd", "code": "OPEN_WITNESSED",
           "subject": "fix(dispatch): guard tick (#2850) (fak tools)",
           "verdict": "OK", "witness": "diff-witnessed",
           "claim_kind": "code_effect"}

    def test_routes_through_close_arm_with_synthetic_audit(self) -> None:
        mod = load()
        calls: list[list[str]] = []

        def fake_run(root, cmd, **_kw):
            calls.append(list(cmd))
            return 0, json.dumps({"ok": True, "verdict": "CLOSED",
                                  "counts": {"closed": 1},
                                  "closed_numbers": [2850]})

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs"
            out = mod.close_open_witnessed(Path(d), runs, [dict(self.ROW)],
                                           live=True, runner=fake_run)
            self.assertTrue(out["invoked"])
            self.assertTrue(out["ok"])
            self.assertEqual(out["verdict"], "CLOSED")
            self.assertEqual(out["closed_numbers"], [2850])
            cmd = calls[0]
            self.assertIn("--audit-json", cmd)
            self.assertIn("--live", cmd)
            self.assertTrue(str(cmd[1]).endswith("issue_resolve_witnessed.py"))
            # The synthetic audit carries the OPEN_WITNESSED bucket + witness.
            audit_path = Path(cmd[cmd.index("--audit-json") + 1])
            doc = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(doc["issues"][0]["number"], 2850)
            self.assertEqual(doc["issues"][0]["bucket"], "OPEN_WITNESSED")
            self.assertEqual(doc["issues"][0]["witnessed_commits"][0]["sha"],
                             "f8aff29dfd")

    def test_dry_run_never_passes_live(self) -> None:
        mod = load()
        calls: list[list[str]] = []

        def fake_run(root, cmd, **_kw):
            calls.append(list(cmd))
            return 0, json.dumps({"ok": True, "verdict": "PLANNED"})

        with tempfile.TemporaryDirectory() as d:
            out = mod.close_open_witnessed(Path(d), Path(d) / "runs",
                                           [dict(self.ROW)],
                                           live=False, runner=fake_run)
            self.assertTrue(out["invoked"])
            self.assertNotIn("--live", calls[0])

    def test_close_arm_failure_fails_open(self) -> None:
        mod = load()

        def fake_run(root, cmd, **_kw):
            return 127, "boom: not json"

        with tempfile.TemporaryDirectory() as d:
            out = mod.close_open_witnessed(Path(d), Path(d) / "runs",
                                           [dict(self.ROW)],
                                           live=True, runner=fake_run)
            self.assertTrue(out["invoked"])
            self.assertFalse(out["ok"])
            self.assertIn("boom", out["error"])

    def test_no_rows_is_a_noop(self) -> None:
        mod = load()
        self.assertEqual(
            mod.close_open_witnessed(Path("."), Path("."), [], live=True,
                                     runner=lambda *a, **k: (0, "{}")),
            {})


class AppendLoopEventArgvTest(unittest.TestCase):
    """The argv `append_loop_event` hands to `fak loop append`.

    This wrapper had no coverage at all — every caller's test stubs `append=`, so the
    only place the argv is actually built was never exercised. It shipped a
    `--verified-state` flag the Go verb does not define, which made the flag parser exit
    2 before appending. Only the WITNESS event carries verified_state, so fire/admit/end
    landed and every witness row was silently dropped (the wrapper is fail-open), and
    `fak loop health` read the loop 0-of-N witnessed with witness_collapse=true.
    """

    def test_fak_loop_cmd_prefers_path_and_never_go_run(self) -> None:
        mod = load()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(mod.shutil, "which", return_value=r"C:\bin\fak.exe"):
            self.assertEqual(mod.fak_loop_cmd(ROOT), [r"C:\bin\fak.exe"])
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(mod.shutil, "which", return_value=None):
            self.assertEqual(mod.fak_loop_cmd(Path(td)), [])

    def _argv(self, event: dict[str, object]) -> list[str]:
        mod = load()
        seen: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            seen.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with mock.patch.object(mod.subprocess, "run", fake_run):
            out = mod.append_loop_event(ROOT, Path("loops.jsonl"), event)
        self.assertTrue(out["ok"])
        self.assertEqual(len(seen), 1)
        return seen[0]

    def _loop_append_flags(self) -> set[str]:
        """The flag names `fak loop append` really defines, read from its flag set.

        Parsed from the Go source rather than shelled out so the test stays hermetic
        (this file runs nothing live). This is what pins the Python argv to Go reality:
        a flag Python invents but Go never declares is a silent exit-2, not an error a
        fail-open caller can see.
        """
        src = (ROOT / "cmd" / "fak" / "loop.go").read_text(encoding="utf-8")
        start = src.index("func runLoopAppend(")
        end = src.index("\nfunc ", start + 1)
        body = src[start:end]
        return set(re.findall(r'fs\.(?:String|Bool|Int|Int64|Var)\(\s*(?:&\w+,\s*)?"([a-z-]+)"', body))

    def test_verified_state_rides_evidence_not_an_undefined_flag(self) -> None:
        argv = self._argv({
            "loop_id": "issue-resolve-progress", "run_id": "RID-1", "kind": "witness",
            "status": "witnessed_done", "verified_state": "verified_done",
            "reason": "OK", "summary": "open_count=479",
            "evidence": [("progress_log", "runs/progress.jsonl")],
        })
        self.assertNotIn("--verified-state", argv)
        self.assertIn("--evidence", argv)
        self.assertIn("verified_state=verified_done", argv)
        # The pre-existing evidence is preserved, not replaced.
        self.assertIn("progress_log=runs/progress.jsonl", argv)
        # And the row is still a witness carrying its status.
        self.assertEqual(argv[argv.index("--kind") + 1], "witness")
        self.assertEqual(argv[argv.index("--status") + 1], "witnessed_done")

    def test_every_flag_is_one_fak_loop_append_defines(self) -> None:
        defined = self._loop_append_flags()
        self.assertIn("evidence", defined)      # the parse found a real flag set
        self.assertNotIn("verified-state", defined)
        for event in (
            {"loop_id": "l", "run_id": "R", "kind": "witness", "status": "witnessed_done",
             "verified_state": "verified_unavailable", "reason": "AUDIT_UNAVAILABLE",
             "summary": "s", "metrics": {"open_now": 479}, "evidence": [("progress_log", "p")]},
            {"loop_id": "l", "run_id": "R", "kind": "end", "status": "claimed_done",
             "reason": "OK", "summary": "s"},
        ):
            argv = self._argv(event)
            for tok in argv:
                if tok.startswith("--"):
                    self.assertIn(tok[2:], defined,
                                  f"{tok} is not a flag `fak loop append` defines; "
                                  f"it would exit 2 and drop the row (argv={argv})")

    def test_no_verified_state_appends_no_evidence_ref(self) -> None:
        argv = self._argv({"loop_id": "l", "run_id": "R", "kind": "end",
                           "status": "claimed_done", "reason": "OK", "summary": "s"})
        self.assertNotIn("--evidence", argv)


class ReblockStreakHoldTest(unittest.TestCase):
    """#5869: hold on REPETITION, not on the reason CLASS.

    Measured over the clean current-fleet window (2026-08-04 .. 08-07, 286 finished
    -worker .witness records / 94.0h over 117 issues), 24 issues hit a re-blockable
    terminal and drew 54 further worker-units / 16.6h — but 8 of those 54 (15%) went
    on to land a WITNESSED commit. So the #1396 docstring premise ("re-dispatching it
    re-blocks identically") is only ~30% true, and a blanket reason-class cooldown
    would refuse a retry that works one time in seven. These tests pin BOTH
    directions: the hold fires on a genuine consecutive-identical repeat, and it stays
    SILENT on the first retry that the measurement says still lands commits.
    """

    # 2026-08-06T12:00:00Z — the fixture epoch, so a stamped sidecar name and the
    # ``now_ts`` a tick would pass in are derived from the same instant.
    EPOCH = 1786363200.0

    @staticmethod
    def _stamp(ts: float) -> str:
        import datetime as dt
        return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime(
            "%Y%m%d-%H%M%S")

    def _witness(self, runs: Path, issue: int, ts: float, *,
                 claim: str = "CLAIM_NO_COMMIT", reason: str | None = None) -> Path:
        """Write one durable ``resolve-<issue>-<UTC>.witness`` record. Note there is
        no ``os.utime`` here and no ``.pid``/``.log`` sibling: the hold keys off the
        LAUNCH STAMP IN THE FILENAME plus the record body, which is exactly what
        survives ``prune_dead_sidecars``."""
        p = runs / f"resolve-{issue}-{self._stamp(ts)}.witness"
        body: dict = {"claim": claim, "issue": issue}
        if reason is not None:
            body["reason"] = reason
        p.write_text(json.dumps(body), encoding="utf-8")
        return p

    def test_first_retry_after_policy_block_is_admitted_second_repeat_is_held(self):
        """BOTH directions, the load-bearing test.

        One ``policy_block`` is NOT enough: that is the 15%-still-lands case, and
        holding it is precisely the blanket cooldown #5869 rejects. A SECOND
        consecutive identical ``policy_block`` is the repetition that proves futility,
        and only then does the issue leave the candidate stream."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # One re-blockable terminal, 1h ago. The first retry MUST stay admitted.
            self._witness(runs, 4568, self.EPOCH - 3600, reason="policy_block")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), set(),
                "a FIRST retry after a policy_block must still be admitted — 15% of "
                "them land a witnessed commit")
            # The retry ran and re-blocked IDENTICALLY. Now hold it.
            self._witness(runs, 4568, self.EPOCH - 1800, reason="policy_block")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), {4568})
            records = mod.reblock_streak_held_records(runs, now_ts=self.EPOCH)
            self.assertEqual(records[0]["reason"], "policy_block")
            self.assertEqual(records[0]["streak"], 2)

    def test_two_different_terminals_are_not_a_streak(self):
        """``policy_block`` then ``self_modify`` is two blocks but NOT the same block,
        so nothing is proven repeatable and the issue stays free. Same for a
        re-blockable terminal separated by an untyped ``unknown`` — over the window
        ``policy_block -> unknown`` (13) actually OUTNUMBERS ``policy_block ->
        policy_block`` (8), and folding those together is the reason-class hold in
        disguise."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._witness(runs, 3695, self.EPOCH - 3600, reason="policy_block")
            self._witness(runs, 3695, self.EPOCH - 1800, reason="self_modify")
            # Different reason interleaved: not identical, not held.
            self._witness(runs, 3267, self.EPOCH - 5400, reason="policy_block")
            self._witness(runs, 3267, self.EPOCH - 3600, reason="unknown")
            self._witness(runs, 3267, self.EPOCH - 1800, reason="policy_block")
            # A transient wall is not a structural block at all.
            self._witness(runs, 5103, self.EPOCH - 3600, reason="auth_wall")
            self._witness(runs, 5103, self.EPOCH - 1800, reason="auth_wall")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), set())

    def test_landed_commit_in_the_tail_clears_the_hold(self):
        """The structural clear. An admitted run that lands a witnessed commit breaks
        the tail, so the hold evaporates with no reset verb — the property the
        lane-livelock / repair-cooldown / goal-park defects all lacked."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._witness(runs, 5586, self.EPOCH - 7200, reason="policy_block")
            self._witness(runs, 5586, self.EPOCH - 5400, reason="policy_block")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), {5586})
            self._witness(runs, 5586, self.EPOCH - 1800, claim="CLAIM_WITNESSED")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), set(),
                "a landed commit must clear the hold")

    def test_ttl_admits_one_probe_and_the_probe_outcome_decides(self):
        """The TTL does not declare the issue retryable — it admits exactly ONE probe
        so the streak CAN break. Without it a held issue could never produce the
        record that frees it (a self-sealing hold). A probe that re-blocks identically
        re-arms the hold for another window."""
        mod = load()
        ttl = mod.DEFAULT_REBLOCK_STREAK_HOLD_TTL_H
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._witness(runs, 4518, self.EPOCH - 7200, reason="policy_block")
            self._witness(runs, 4518, self.EPOCH - 5400, reason="policy_block")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), {4518})
            # Past the TTL (measured from the NEWEST run in the streak) the hold
            # lapses and one probe is admitted.
            lapsed = self.EPOCH - 5400 + ttl * 3600 + 60
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=lapsed), set())
            # That probe re-blocked identically -> the hold RE-ARMS for another window.
            self._witness(runs, 4518, lapsed, reason="policy_block")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=lapsed + 60), {4518})
            # A probe with a DIFFERENT outcome would have cleared it instead.
            self._witness(runs, 4518, lapsed + 120, reason="restart_exhausted")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=lapsed + 180), set())

    # ---- #5869 follow-up: the TTL margin, re-derived over the whole corpus --------
    #
    # The real 10-run trace of #4568 (UTC launch stamps), the ONLY trace in the 2396
    # -record corpus that sets the zero-witnessed-loss boundary. Its 06:52 win is
    # f34f02f84; the load-bearing gap is 17:32 -> 00:30 = 6.97h, between a block and
    # an UNRELATED untyped run.
    _T4568 = [
        ("08-06 02:30", "unknown"), ("08-06 04:56", "policy_block"),
        ("08-06 07:44", "unknown"), ("08-06 11:01", "self_modify"),
        ("08-06 14:00", "policy_block"), ("08-06 17:32", "policy_block"),
        ("08-06 20:49", "policy_block"), ("08-07 00:30", "unknown"),
        ("08-07 04:10", "policy_block"), ("08-07 06:52", None),
    ]

    @staticmethod
    def _utc(label: str) -> float:
        import datetime as dt
        return dt.datetime.strptime("2026-" + label, "%Y-%m-%d %H:%M").replace(
            tzinfo=dt.timezone.utc).timestamp()

    def _replay(self, mod, trace, issue: int, ttl_h: float) -> list[str]:
        """Counterfactual replay of ``trace`` under a ``ttl_h`` hold, driven by the
        REAL gate rather than a re-implementation of it: at each launch instant the
        picker sees only the sidecars of the runs that were not suppressed upstream,
        which is what makes the cascade visible. Returns the SUPPRESSED labels."""
        suppressed: list[str] = []
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for label, reason in trace:
                ts = self._utc(label)
                held = mod.reblock_streak_held_issues(runs, ttl_h=ttl_h, now_ts=ts)
                if issue in held:
                    suppressed.append(label)
                    continue
                if reason is None:
                    self._witness(runs, issue, ts, claim="CLAIM_WITNESSED")
                else:
                    self._witness(runs, issue, ts, reason=reason)
        return suppressed

    def test_default_ttl_keeps_a_margin_under_the_measured_first_loss(self):
        """The shipped TTL must sit a stated distance BELOW the replayed boundary.

        This is the invariant the first cut of the constant lacked: it was justified
        by a quoted "band", and a band cannot be re-checked when the corpus moves.
        Naming the boundary and the margin makes the justification executable."""
        mod = load()
        self.assertLessEqual(
            mod.DEFAULT_REBLOCK_STREAK_HOLD_TTL_H
            + mod.REBLOCK_STREAK_MIN_TTL_MARGIN_H,
            mod.REBLOCK_STREAK_FIRST_LOSS_TTL_H,
            "the default TTL must stay at least "
            f"{mod.REBLOCK_STREAK_MIN_TTL_MARGIN_H}h under the measured "
            f"{mod.REBLOCK_STREAK_FIRST_LOSS_TTL_H}h first-loss boundary")

    def test_boundary_is_a_suppression_cascade_not_a_late_lander(self):
        """Pins the mechanism, because the mechanism is what was mis-read.

        At the measured boundary the hold does NOT eat the win by out-waiting it: the
        win launches 2.7h after that issue's last block, well inside ANY TTL here.
        It eats an untyped `unknown` 6.97h downstream of an EARLIER block, and only
        the DELETION of that run re-forms a [policy_block, policy_block] tail that
        never existed in the real trace. That is why reading the block-to-win
        distribution alone (nothing between 2.69h and 9.33h) cannot find this."""
        mod = load()
        at_boundary = self._replay(
            mod, self._T4568, 4568, mod.REBLOCK_STREAK_FIRST_LOSS_TTL_H)
        self.assertIn("08-07 00:30", at_boundary,
                      "the cascade TRIGGER: the untyped run 6.97h after the 17:32 "
                      "block is what a 7.0h TTL newly swallows")
        self.assertIn("08-07 06:52", at_boundary,
                      "and deleting it costs the f34f02f84 win — the boundary")
        # The win is NOT a late lander: it launches 2.7h after the last block, so a
        # hold that ate it by duration alone would have eaten it at every TTL >= 3h.
        self.assertLess(
            (self._utc("08-07 06:52") - self._utc("08-07 04:10")) / 3600.0, 3.0)

    def test_shipped_default_loses_nothing_on_the_boundary_trace(self):
        """Fail-safe direction: at the SHIPPED TTL the same real trace keeps its win,
        and it keeps the cascade trigger too — the margin is margin in the mechanism,
        not just in the number."""
        mod = load()
        at_default = self._replay(
            mod, self._T4568, 4568, mod.DEFAULT_REBLOCK_STREAK_HOLD_TTL_H)
        self.assertNotIn("08-07 06:52", at_default,
                         "the shipped default must not destroy f34f02f84")
        self.assertNotIn("08-07 00:30", at_default,
                         "nor arm the cascade that destroys it")
        # It is still a HOLD, not a no-op: the repeat 3.3h after the 14:00/17:32
        # streak is exactly the futile re-dispatch the gate exists to refuse.
        self.assertIn("08-06 20:49", at_default)

    def test_updated_at_bump_and_kill_switches(self):
        """A guard refusal is a verdict on what the ISSUE ASKED FOR, so a re-scope
        (fresh gh ``updatedAt``) genuinely changes it and re-admits early — the same
        content-keyed escape the contract/multi-lane ledgers take, and why this hold
        does NOT take ``collision_held_records``' local-tree opt-out. Either knob at 0
        disables the gate, matching the other holds' kill switch."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._witness(runs, 4568, self.EPOCH - 3600, reason="policy_block")
            self._witness(runs, 4568, self.EPOCH - 1800, reason="policy_block")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), {4568})
            # An updatedAt OLDER than the streak does not void it.
            self.assertEqual(mod.reblock_streak_held_issues(
                runs, now_ts=self.EPOCH, updated_ts={4568: self.EPOCH - 2400}),
                {4568})
            # A re-scope AFTER the last block re-admits the issue.
            self.assertEqual(mod.reblock_streak_held_issues(
                runs, now_ts=self.EPOCH, updated_ts={4568: self.EPOCH - 600}), set())
            self.assertEqual(mod.reblock_streak_held_issues(
                runs, streak_n=0, now_ts=self.EPOCH), set())
            self.assertEqual(mod.reblock_streak_held_issues(
                runs, ttl_h=0, now_ts=self.EPOCH), set())
            # A stricter N than the observed streak does not fire.
            self.assertEqual(mod.reblock_streak_held_issues(
                runs, streak_n=3, now_ts=self.EPOCH), set())

    def test_durable_hold_survives_the_process_the_tick_scoped_one_does_not(self):
        """The DEFECT this closes, stated as a contrast.

        ``held_no_commit_issues`` reads the in-memory result of the witness sweep the
        CURRENT tick just ran, so a fresh dispatcher process — the next tick — has no
        memory of the refusal and re-picks the issue. That is the "in-memory tally
        dies with the process and silently never fires" shape. The durable reader
        keys off the retained ``.witness`` sidecars instead, so the same evidence
        still holds after the process that produced it is gone."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._witness(runs, 4568, self.EPOCH - 3600, reason="policy_block")
            self._witness(runs, 4568, self.EPOCH - 1800, reason="policy_block")
            # A NEW dispatcher process has no in-memory witness result to read.
            self.assertEqual(mod.held_no_commit_issues(None), set())
            self.assertEqual(mod.held_no_commit_issues({"no_commit": []}), set())
            # The durable reader still holds it off the very same evidence.
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), {4568})

    def test_history_is_launch_ordered_and_fail_open(self):
        """Ordering comes from the FILENAME stamp, not the mtime the witness sweep
        rewrites at an arbitrary later tick. Unreadable/odd sidecars are skipped, never
        raised, so this can only ever ADD a hold — it can never wedge the picker."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # Written newest-first on disk; history must still come back oldest-first.
            newer = self._witness(runs, 77, self.EPOCH - 1800, reason="policy_block")
            older = self._witness(runs, 77, self.EPOCH - 3600, reason="self_modify")
            os.utime(newer, (self.EPOCH - 9999, self.EPOCH - 9999))
            os.utime(older, (self.EPOCH, self.EPOCH))
            hist = mod.issue_witness_history(runs)
            self.assertEqual([r["reason"] for r in hist[77]],
                             ["self_modify", "policy_block"])
            # Garbage in the runs dir is ignored, not raised.
            (runs / "resolve-nope.witness").write_text("{", encoding="utf-8")
            (runs / "resolve-88-20260806-120000.witness").write_text(
                "not json", encoding="utf-8")
            self.assertNotIn(88, mod.issue_witness_history(runs))
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), set())
            # A missing runs dir is empty, never an exception.
            self.assertEqual(mod.issue_witness_history(runs / "gone"), {})


if __name__ == "__main__":
    unittest.main()
