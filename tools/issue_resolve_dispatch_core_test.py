#!/usr/bin/env python3
"""Hermetic tests for core lane/issue selection, contracts, and collisions."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue_resolve_dispatch_test import ROOT, load  # noqa: E402


class PickTargetTest(unittest.TestCase):
    def test_first_not_in_skip(self) -> None:
        mod = load()
        self.assertEqual(mod.pick_target_issue([497, 496, 495], set()), 497)
        self.assertEqual(mod.pick_target_issue([497, 496, 495], {497}), 496)
        self.assertEqual(mod.pick_target_issue([497, 496], {497, 496}), None)
        self.assertEqual(mod.pick_target_issue([], set()), None)


class CapabilitySkipTest(unittest.TestCase):
    def test_standard_runner_skips_only_explicit_hardware_requirement(self) -> None:
        mod = load()
        skipped, rows = mod.capability_skips(
            [30, 31], {30: [], 31: ["gpu"]}, frozenset())
        self.assertEqual(skipped, {31})
        self.assertEqual(rows, [{"issue": 31, "required_caps": ["gpu"]}])

    def test_capable_runner_admits_explicit_requirement(self) -> None:
        mod = load()
        skipped, rows = mod.capability_skips(
            [31], {31: ["GPU"]}, frozenset({"gpu"}))
        self.assertEqual(skipped, set())
        self.assertEqual(rows, [])


class ContractOverlayTest(unittest.TestCase):
    def test_record_merges_overlay_into_body(self) -> None:
        mod = load()
        rec = mod._issue_record_for_contract(
            {"number": 7, "title": "t", "body": "base body"}, 7,
            "Done condition: the gate passes")
        self.assertIn("base body", rec["body"])
        self.assertIn("Done condition: the gate passes", rec["body"])
        self.assertIn("local contract overlay", rec["body"])  # legible merge marker
        # No overlay -> the record is byte-identical to before.
        self.assertEqual(
            mod._issue_record_for_contract({"number": 7, "body": "base body"}, 7)["body"],
            "base body")

    def test_contract_review_invokes_fak_dev_after_command_split(self) -> None:
        mod = load()
        seen = []

        def runner(command, **_kwargs):
            seen.append(command)
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout=json.dumps({
                    "ok": True,
                    "reviews": [{"ok": True, "score": {"total": 100},
                                 "spine_priority": {"total": 100}}],
                }), stderr="")

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
                mod, "_fak_dev_command_prefix", return_value=["fak-dev"]):
            review = mod.issue_contract_review(
                Path(td), {"number": 7, "title": "t", "body": "body"}, 7,
                runner=runner)
        self.assertTrue(review["ok"])
        self.assertEqual(seen[0][:3], ["fak-dev", "issue", "contract"])

    def test_read_and_times_roundtrip(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            self.assertEqual(mod.read_contract_overlay(runs, 1207), "")
            self.assertEqual(mod.contract_overlay_times(runs), {})
            path = mod.contract_overlay_path(runs, 1207)
            path.parent.mkdir(parents=True)
            path.write_text("Scope: one file\n", encoding="utf-8")
            (path.parent / "not-an-overlay.md").write_text("x", encoding="utf-8")
            self.assertEqual(mod.read_contract_overlay(runs, 1207), "Scope: one file")
            times = mod.contract_overlay_times(runs)
            self.assertEqual(sorted(times), [1207])
            self.assertGreater(times[1207], 0)


class ContractScanStreamTest(unittest.TestCase):
    def test_round_robin_across_lanes_oldest_first_within(self) -> None:
        mod = load()
        eligible = [["docs", [1, 2, 3]], ["tools", [10]], ["policy", [20, 21]]]
        self.assertEqual(
            mod.contract_scan_stream(eligible, skip=set()),
            [("docs", 1), ("tools", 10), ("policy", 20),
             ("docs", 2), ("policy", 21), ("docs", 3)])

    def test_skip_and_empty_lanes_drop_out(self) -> None:
        mod = load()
        eligible = [["docs", [1, 2]], ["tools", [10]], ["bench", []]]
        self.assertEqual(mod.contract_scan_stream(eligible, skip={1, 10}),
                         [("docs", 2)])
        self.assertEqual(mod.contract_scan_stream([], skip=set()), [])
        self.assertEqual(mod.contract_scan_stream(None, skip=set()), [])


class FakCommandPrefixTest(unittest.TestCase):
    """#5856: nothing refreshes the repo-root fak.exe, while FakSelfUpdate rebuilds the
    PATH copy every 20 minutes from a pristine detached origin/main worktree. This prefix
    used to take the repo-root artifact unconditionally and never consult PATH at all --
    so a single dispatcher tick ran two different fak builds, the guarded one of them
    witnessed 34 commits behind HEAD and stamped +dirty."""

    def _layout(self, td: Path):
        exe = "fak.exe" if os.name == "nt" else "fak"
        root, pathdir = td / "root", td / "bin"
        root.mkdir(parents=True)
        pathdir.mkdir(parents=True)
        rootbin, onpath = root / exe, pathdir / exe
        for p in (rootbin, onpath):
            p.write_text("stub", encoding="utf-8")
            p.chmod(0o755)
        return root, rootbin, onpath, pathdir

    def test_explicit_fak_bin_still_wins(self) -> None:
        mod = load()
        with mock.patch.dict(os.environ, {"FAK_BIN": "C:/pinned/fak.exe"}):
            self.assertEqual(mod._fak_command_prefix(Path("/nope")), ["C:/pinned/fak.exe"])

    def test_freshest_build_wins_in_both_directions(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root, rootbin, onpath, pathdir = self._layout(Path(td))
            with mock.patch.dict(os.environ, {"PATH": str(pathdir)}, clear=True):
                # The refreshed PATH copy is newer -> it wins. The fleet case, and the
                # assertion the old repo-root-first prefix fails.
                os.utime(rootbin, (1_000_000, 1_000_000))
                os.utime(onpath, (2_000_000, 2_000_000))
                self.assertEqual(mod._fak_command_prefix(root), [str(onpath)])
                # A just-rebuilt repo-root binary is newer -> it wins. The dev case holds.
                os.utime(rootbin, (3_000_000, 3_000_000))
                self.assertEqual(mod._fak_command_prefix(root), [str(rootbin)])

    def test_no_binary_anywhere_falls_back_to_go_run(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            empty = Path(td) / "empty"
            empty.mkdir()
            with mock.patch.dict(os.environ, {"PATH": str(empty)}, clear=True):
                self.assertEqual(
                    mod._fak_command_prefix(empty), ["go", "run", "./cmd/fak"])


class ExcludeLaneTest(unittest.TestCase):
    ROUTER = {"lanes": {
        "docs": {"issues": [{"number": 9}, {"number": 8}, {"number": 7}]},
        "gateway": {"issues": [{"number": 6}, {"number": 5}]},
    }}

    def test_busiest_pick_skips_excluded_lane(self) -> None:
        mod = load()
        mod.issue_dispatch.run_json = lambda cmd, root, timeout: self.ROUTER
        # without exclude: docs (3) is busiest
        self.assertEqual(mod.lane_issue_numbers(ROOT, None)["lane"], "docs")
        # excluding docs: gateway (2) wins
        picked = mod.lane_issue_numbers(ROOT, None, exclude={"docs"})
        self.assertEqual(picked["lane"], "gateway")
        self.assertEqual(picked["excluded_lanes"], ["docs"])

    def test_explicit_lane_ignores_exclude(self) -> None:
        mod = load()
        mod.issue_dispatch.run_json = lambda cmd, root, timeout: self.ROUTER
        picked = mod.lane_issue_numbers(ROOT, "docs", exclude={"docs"})
        self.assertEqual(picked["lane"], "docs")   # explicit wins over exclude

    def test_numbers_are_oldest_first_despite_router_newest_first_input(self) -> None:
        # gh issue list (and the router's per-lane "issues") is newest-first; this
        # fold must explicitly reverse it so pick_target_issue's "first not in skip"
        # lands on the OLDEST open issue, per the fleet's stated dispatch priority.
        mod = load()
        mod.issue_dispatch.run_json = lambda cmd, root, timeout: self.ROUTER
        picked = mod.lane_issue_numbers(ROOT, "docs")
        self.assertEqual(picked["numbers"], [7, 8, 9])   # ascending, not [9, 8, 7]


class LaneIssueNumbersSelfModifyHoldTest(unittest.TestCase):
    """Proactive pre-route hold on the ACTUAL production picker: this module
    (not issue_dispatch.py) is what the live Scheduled Tasks invoke. Mirrors
    issue_dispatch.pick_lane's fix -- a lane whose tree is fak's trust-critical referee source is excluded from
    the busiest-pick under guard; ordinary cmd/** and internal/** lanes remain
    dispatchable under the narrowed self-modify policy."""

    ROUTER = {"lanes": {
        "docs": {"issues": [{"number": 2}, {"number": 1}], "tree": ["docs/**"]},
        "policy": {"issues": [{"number": 4}, {"number": 3}, {"number": 2}, {"number": 1}],
                   "tree": ["internal/policy/**"]},
        "tools": {"issues": [{"number": 9}], "tree": ["tools/**", "scripts/**"]},
    }}

    def _router(self, mod) -> None:
        mod.issue_dispatch.run_json = lambda cmd, root, timeout: self.ROUTER

    def test_guarded_skips_self_source_lane_for_richest_safe(self) -> None:
        mod = load()
        self._router(mod)
        picked = mod.lane_issue_numbers(ROOT, None, guarded=True)
        self.assertEqual(picked["lane"], "docs")   # policy (4, trust-critical) excluded
        self.assertEqual(picked["self_modify_held"], ["policy"])

    def test_unguarded_does_not_hold_self_source_lane(self) -> None:
        mod = load()
        self._router(mod)
        picked = mod.lane_issue_numbers(ROOT, None, guarded=False)
        self.assertEqual(picked["lane"], "policy")
        self.assertEqual(picked["self_modify_held"], [])

    def test_all_lanes_self_source_yields_no_lane(self) -> None:
        mod = load()
        mod.issue_dispatch.run_json = lambda cmd, root, timeout: {"lanes": {
            "policy": {"issues": [{"number": 2}, {"number": 1}], "tree": ["internal/policy/**"]},
            "adjudicator": {"issues": [{"number": 9}], "tree": ["internal/adjudicator/**"]},
        }}
        picked = mod.lane_issue_numbers(ROOT, None, guarded=True)
        self.assertIsNone(picked["lane"])
        self.assertEqual(picked["numbers"], [])
        self.assertEqual(picked["self_modify_held"], ["adjudicator", "policy"])

    def test_explicit_lane_honored_despite_self_source(self) -> None:
        mod = load()
        self._router(mod)
        picked = mod.lane_issue_numbers(ROOT, "policy", guarded=True)
        self.assertEqual(picked["lane"], "policy")
        self.assertEqual(picked["self_modify_held"], [])   # explicit path skips the fold

    def test_exclude_and_self_source_hold_compose(self) -> None:
        # docs excluded (busy) + gateway held (self-source) -> only tools left.
        mod = load()
        self._router(mod)
        picked = mod.lane_issue_numbers(ROOT, None, exclude={"docs"}, guarded=True)
        self.assertEqual(picked["lane"], "tools")
        self.assertEqual(picked["safe_lanes_busy"], ["docs"])
        self.assertEqual(picked["self_modify_held"], ["policy"])

    def test_safe_lanes_busy_surfaces_when_only_self_source_remains(self) -> None:
        mod = load()
        self._router(mod)
        picked = mod.lane_issue_numbers(ROOT, None, exclude={"docs", "tools"},
                                        guarded=True)
        self.assertIsNone(picked["lane"])
        self.assertEqual(picked["safe_lanes_busy"], ["docs", "tools"])
        self.assertEqual(picked["self_modify_held"], ["policy"])

    def test_default_guarded_reads_dispatch_worker_guard_enabled(self) -> None:
        mod = load()
        self._router(mod)
        mod.dispatch_worker.guard_enabled = lambda *a, **k: True
        picked = mod.lane_issue_numbers(ROOT, None)
        self.assertEqual(picked["self_modify_held"], ["policy"])


class MultiLaneScopeTest(unittest.TestCase):
    """The #2615 guard: refuse an issue whose named file families fall outside the
    chosen lane's LEASE TREE (the collision authority), not just its label. Pure —
    canned trees, no dos/gh."""

    TREES = {
        "ci": [".github/**"],
        "docs": ["docs/**"],
        "claude": [".claude/**"],
        "tools": ["tools/**", "scripts/**"],
        "gateway": ["internal/gateway/**"],
    }
    LANES = set(TREES)

    def test_broad_issue_under_narrow_lane_is_multi_lane(self) -> None:
        # #2233 shape: a ci-leased worker (tree .github/**) told to edit .claude/**
        # and docs/** — the families the narrow lease cannot protect.
        mod = load()
        body = ("Callers live in `.claude/skills/x/SKILL.md`, `.github/workflows/y.yml`, "
                "and `docs/archive/z.md`.")
        scan = mod.multi_lane_scope(body, "ci", [".github/**"], self.TREES, self.LANES)
        self.assertTrue(scan["multi_lane"])
        self.assertEqual(scan["uncovered_lanes"], ["claude", "docs"])
        # The path the lease DOES cover is not flagged.
        self.assertIn(".github/workflows/y.yml", scan["covered_paths"])

    def test_single_lane_issue_is_not_refused(self) -> None:
        # An issue naming only its own lane's tree clears — no false positive.
        mod = load()
        body = "Edit `tools/issue_resolve_dispatch.py` and `tools/issue_lane_router.py`."
        scan = mod.multi_lane_scope(body, "tools", ["tools/**", "scripts/**"],
                                    self.TREES, self.LANES)
        self.assertFalse(scan["multi_lane"])
        self.assertEqual(scan["uncovered"], [])

    def test_lease_covering_every_family_clears(self) -> None:
        # The done-condition escape hatch: when the lease tree covers every named
        # family, the same broad body is admissible (no refusal).
        mod = load()
        body = "Touches `.github/workflows/a.yml` and `docs/b.md`."
        wide = {"wide": [".github/**", "docs/**"]}
        trees = {**self.TREES, **wide}
        scan = mod.multi_lane_scope(body, "wide", [".github/**", "docs/**"],
                                    trees, set(trees))
        self.assertFalse(scan["multi_lane"])

    def test_glob_and_bare_paths_do_not_trip_the_guard(self) -> None:
        # A `tools/*.py` glob or a bare `Makefile`/`dos.toml` is not a rooted path,
        # so it never manufactures a spurious cross-lane family.
        mod = load()
        body = "Rewrite `tools/*.py`, `Makefile`, and `dos.toml` command strings."
        scan = mod.multi_lane_scope(body, "ci", [".github/**"], self.TREES, self.LANES)
        self.assertFalse(scan["multi_lane"])

    def test_reason_names_families_and_split_path(self) -> None:
        mod = load()
        scan = {"chosen_lane": "ci", "chosen_tree": [".github/**"],
                "uncovered": [{"path": ".claude/skills/", "lanes": ["claude"]}],
                "uncovered_lanes": ["claude"]}
        reason = mod.multi_lane_scope_reason(2233, scan)
        self.assertIn("#2233", reason)
        self.assertIn(".claude/skills/", reason)
        self.assertIn("Split into", reason)


class DirtyPathCollisionTest(unittest.TestCase):
    def test_exact_dirty_paths_named_in_issue_collide(self) -> None:
        mod = load()
        scan = mod.dirty_path_collision(
            "Likely files: `cmd/fak/knownbad.go`, `fak/internal/foo/bar.go`.",
            ["cmd/fak/knownbad.go", "internal/foo/bar.go", "docs/clean.md"])
        self.assertTrue(scan["collides"])
        self.assertEqual(scan["dirty_paths"],
                         ["cmd/fak/knownbad.go", "internal/foo/bar.go"])

    def test_punctuation_only_dirty_name_does_not_collide_with_prose(self) -> None:
        mod = load()
        scan = mod.dirty_path_collision(
            "Acceptance: preserve policy; then dispatch.", [";"])
        self.assertFalse(scan["collides"])

    def test_bare_filename_does_not_collide(self) -> None:
        mod = load()
        scan = mod.dirty_path_collision(
            "Knownbad work mentions knownbad.go but no repo-relative path.",
            ["cmd/fak/knownbad.go"])
        self.assertFalse(scan["collides"])

    def test_git_status_parser_handles_renames_and_untracked(self) -> None:
        mod = load()
        paths = mod.parse_git_status_paths(
            " M cmd/fak/knownbad.go\n"
            "?? internal/knownbad/knownbad_test.go\n"
            "R  old/path.go -> new/path.go\n")
        self.assertEqual(paths, [
            "cmd/fak/knownbad.go",
            "internal/knownbad/knownbad_test.go",
            "old/path.go",
            "new/path.go",
        ])


class SameIssueWipCollisionTest(unittest.TestCase):
    def test_recent_same_issue_log_with_dirty_path_collides(self) -> None:
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = runs / "resolve-2718-20260705-191407.log"
            log.write_text(
                "Final report: W6 implementation left as working-tree changes.\n"
                "Uncommitted files:\n"
                "- cmd/fak/knownbad.go\n",
                encoding="utf-8",
            )
            os.utime(log, (now - 60, now - 60))

            scan = mod.same_issue_wip_collision(
                runs, 2718, ["cmd/fak/knownbad.go", "docs/other.md"],
                now_ts=now)

        self.assertTrue(scan["collides"])
        self.assertEqual(scan["dirty_paths"], ["cmd/fak/knownbad.go"])
        self.assertEqual(scan["evidence"][0]["log"],
                         "resolve-2718-20260705-191407.log")

    def test_worktree_wording_collides(self) -> None:
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = runs / "resolve-2718-20260705-191407.log"
            log.write_text(
                "Final report: implementation left as local worktree changes.\n"
                "- cmd/fak/knownbad.go\n",
                encoding="utf-8",
            )
            os.utime(log, (now - 60, now - 60))

            scan = mod.same_issue_wip_collision(
                runs, 2718, ["cmd/fak/knownbad.go"], now_ts=now)

        self.assertTrue(scan["collides"])
        self.assertEqual(scan["dirty_paths"], ["cmd/fak/knownbad.go"])

    def test_ignores_other_issue_and_stale_log(self) -> None:
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            other = runs / "resolve-9999-20260705-191407.log"
            other.write_text("Uncommitted: cmd/fak/knownbad.go\n", encoding="utf-8")
            stale = runs / "resolve-2718-20260701-191407.log"
            stale.write_text("Uncommitted: cmd/fak/knownbad.go\n", encoding="utf-8")
            os.utime(other, (now - 60, now - 60))
            os.utime(stale, (now - 3 * 24 * 3600, now - 3 * 24 * 3600))

            scan = mod.same_issue_wip_collision(
                runs, 2718, ["cmd/fak/knownbad.go"], now_ts=now)

        self.assertFalse(scan["collides"])

    def test_no_commit_witness_sidecar_is_wip_evidence(self) -> None:
        import json
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = runs / "resolve-2718-20260705-191407.log"
            log.write_text("Changed cmd/fak/knownbad.go\n", encoding="utf-8")
            log.with_suffix(mod.WITNESS_SIDECAR_SUFFIX).write_text(
                json.dumps({"claim": "CLAIM_NO_COMMIT"}), encoding="utf-8")
            os.utime(log, (now - 60, now - 60))

            scan = mod.same_issue_wip_collision(
                runs, 2718, ["cmd/fak/knownbad.go"], now_ts=now)
            reason = mod.same_issue_wip_reason(2718, scan)

        self.assertTrue(scan["collides"])
        self.assertEqual(scan["evidence"][0]["witness_claim"], "CLAIM_NO_COMMIT")
        self.assertIn("CLAIM_NO_COMMIT", reason)


class SameIssueWipOrientationDocTest(unittest.TestCase):
    """#4321 -- an orientation doc named in a resolve log is not same-issue WIP.

    The SAME_ISSUE_WIP scan infers "the prior resolver left this path dirty" from the
    path being MENTIONED in that resolver's log. Every worker prompt orders the worker
    to read the repo's orientation docs by name, so those names land in every log
    regardless of what the worker touched -- and those same files are chronically
    dirty in the one shared trunk checkout. The ledger investigation behind #4321
    measured ~175/180 SAME_ISSUE_WIP hits on AGENTS.md alone; this repo's own
    .dispatch-runs/collision-holds.jsonl shows the identical defect under different
    names (README.md 78 / llms.txt 63 / INDEX.md 42 = 183 of 208 same_issue_wip
    path-hits, vs 25 on a real code path). The concentration is therefore NOT a
    property of AGENTS.md, and the issue's proposed fix of splitting AGENTS.md into
    fragments would have relocated the magnet rather than removed it.
    """

    def _log(self, runs, now, body):
        import os
        log = runs / "resolve-2718-20260705-191407.log"
        log.write_text(body, encoding="utf-8")
        os.utime(log, (now - 60, now - 60))
        return log

    def test_orientation_doc_alone_does_not_collide(self) -> None:
        """A log that names ONLY orientation docs no longer refuses the issue."""
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # The backticked form the real worker prompt uses ("orient with
            # `AGENTS.md`", "the doc map `llms.txt`"). Note text_mentions_repo_path's
            # `(?![\\w./-])` lookahead means a name ending a sentence ("AGENTS.md.")
            # does NOT match -- the fixture mirrors the phrasing that does.
            self._log(runs, now,
                      "Read `AGENTS.md` first, then the doc map `llms.txt`, "
                      "plus `README.md`\n"
                      "Final report: left as uncommitted working-tree changes.\n")
            scan = mod.same_issue_wip_collision(
                runs, 2718, ["AGENTS.md", "llms.txt", "README.md"], now_ts=now)

        self.assertFalse(scan["collides"])
        self.assertEqual(scan["dirty_paths"], [])
        # The suppression is auditable, not silent.
        self.assertEqual(sorted(scan["orientation_paths"]),
                         ["AGENTS.md", "README.md", "llms.txt"])

    def test_real_wip_path_still_collides_alongside_orientation_docs(self) -> None:
        """The guard is narrowed, not disabled: a genuine dirty code path still
        refuses, and the orientation doc is dropped from the refusal's path list."""
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._log(runs, now,
                      "Oriented with `AGENTS.md`\n"
                      "Final report: cmd/fak/knownbad.go left uncommitted\n")
            scan = mod.same_issue_wip_collision(
                runs, 2718, ["AGENTS.md", "cmd/fak/knownbad.go"], now_ts=now)

        self.assertTrue(scan["collides"])
        self.assertEqual(scan["dirty_paths"], ["cmd/fak/knownbad.go"])
        self.assertEqual(scan["orientation_paths"], ["AGENTS.md"])

    def test_dirty_path_guard_still_protects_a_real_agents_md_issue(self) -> None:
        """The narrowing is scoped to the log-mention INFERENCE. An issue whose own
        BODY names AGENTS.md as its work site is still refused by the dirty-path
        guard, so a real AGENTS.md edit cannot land on a peer's uncommitted one."""
        mod = load()
        scan = mod.dirty_path_collision(
            "fix(docs): tighten the commit rule in `AGENTS.md`", ["AGENTS.md"])
        self.assertTrue(scan["collides"])
        self.assertEqual(scan["dirty_paths"], ["AGENTS.md"])


class RefusalClassTest(unittest.TestCase):
    """#4321 -- lane-lease and working-tree co-tenancy are separate refusal classes.

    Both mechanisms wear the word "collision" in this launcher (the lane-lease
    refusal's own reason text says "refusing COLLISION_RISK"), which is how ~369
    working-tree refusals got folded in with lease contention during ledger analysis.
    They share no fix: a lane-lease refusal clears itself when the holding peer
    finishes, while a working-tree refusal clears only on a human/peer commit or
    revert. Carrying the class as a FIELD is what stops the two being conflated by
    verdict-name pattern matching.
    """

    def test_classes_are_distinct_and_table_driven(self) -> None:
        mod = load()
        self.assertEqual(mod.refusal_class("LANE_LEASE_HELD"),
                         mod.REFUSAL_CLASS_LANE_LEASE)
        self.assertEqual(mod.refusal_class("SAME_ISSUE_WIP"),
                         mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
        self.assertEqual(mod.refusal_class("DIRTY_PATH_COLLISION"),
                         mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
        self.assertNotEqual(mod.REFUSAL_CLASS_LANE_LEASE,
                            mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
        # Unclassified verdicts stay "" -- a new refusal code must be classified on
        # purpose rather than inherit a class from a substring of its name.
        self.assertEqual(mod.refusal_class("MULTI_LANE_SCOPE"), "")
        self.assertEqual(mod.refusal_class("SOME_NEW_COLLISION"), "")
        self.assertEqual(mod.refusal_class(""), "")

    def test_launch_gate_blocker_carries_the_class(self) -> None:
        mod = load()
        gate = mod.launch_gate_blocked("SAME_ISSUE_WIP", "why", "next")
        self.assertEqual(gate["blockers"][0]["refusal_class"],
                         mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
        # An unclassified hold is unchanged -- no empty field is invented for it.
        plain = mod.launch_gate_blocked("FORCE_REASON_REQUIRED", "why", "next")
        self.assertNotIn("refusal_class", plain["blockers"][0])

    def test_collision_ledger_rows_carry_the_class(self) -> None:
        """Rows on the collision ledger are co-tenancy by construction -- both when
        freshly written and when read back from a pre-#4321 row with no class."""
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            mod.record_collision_holds(runs, [{
                "issue": 4321, "kind": "same_issue_wip", "lane": "tools",
                "title": "t", "reason": "r", "dirty_paths": ["AGENTS.md"],
            }], live=True, now_ts=1_000_000.0)
            ledger = runs / "collision-holds.jsonl"
            written = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(written["refusal_class"],
                             mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
            # A legacy row with no class reads back as co-tenancy, not as "".
            with ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": 1_000_000.0, "issue": 4322,
                                    "kind": "dirty_path"}) + "\n")
            recs = {r["issue"]: r for r in mod.collision_held_records(
                runs, ttl_h=3, now_ts=1_000_100.0)}
            self.assertEqual(recs[4322]["refusal_class"],
                             mod.REFUSAL_CLASS_WORKTREE_COTENANCY)

    def test_loop_ledger_evidence_carries_the_split(self) -> None:
        """#4321 -- the loops.jsonl evidence pairs make the WIP-vs-lease split
        computable as a FIELD. Lane/tree alone cannot: both refusals emit the SAME
        lane and tree, which is why the two were conflated in the first place."""
        mod = load()

        def ev(payload):
            return dict(mod._dispatch_collision_evidence(ROOT, payload))

        wip = ev({"verdict": "SAME_ISSUE_WIP", "lane": "tools"})
        dirty = ev({"verdict": "DIRTY_PATH_COLLISION", "lane": "tools"})
        lease = ev({"verdict": "LANE_LEASE_HELD", "lane": "tools"})

        self.assertEqual(wip["refusal_class"], mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
        self.assertEqual(dirty["refusal_class"], mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
        self.assertEqual(lease["refusal_class"], mod.REFUSAL_CLASS_LANE_LEASE)
        # Same lane on both sides -- the class is the ONLY separator here.
        self.assertEqual(wip["lane"], lease["lane"])
        self.assertNotEqual(wip["refusal_class"], lease["refusal_class"])
        # A non-contention tick is unchanged (additive, no empty field invented).
        self.assertNotIn("refusal_class",
                         ev({"verdict": "WOULD_SPAWN", "lane": "tools"}))
        # The class survives a refusal that never resolved a lane.
        self.assertEqual(ev({"verdict": "SAME_ISSUE_WIP"})["refusal_class"],
                         mod.REFUSAL_CLASS_WORKTREE_COTENANCY)


class CandidatePriorityTest(unittest.TestCase):
    """candidate_priority maps priority/P* labels to the dispatchorder Candidate.priority
    integer (P0>P1>P2, 0 when unlabeled) -- #3222 DoD item 5."""

    def test_label_to_weight(self) -> None:
        mod = load()
        # gh's {"name": ...} dict shape, heaviest label wins.
        self.assertEqual(mod.candidate_priority([{"name": "priority/P0"}]), 1000)
        self.assertEqual(mod.candidate_priority([{"name": "priority/P1"}]), 400)
        self.assertEqual(mod.candidate_priority([{"name": "priority/P2"}]), 150)
        self.assertEqual(
            mod.candidate_priority([{"name": "priority/P2"}, {"name": "priority/P0"}]),
            1000)
        # Plain-string labels are accepted too.
        self.assertEqual(mod.candidate_priority(["priority/P1"]), 400)

    def test_unlabeled_is_zero(self) -> None:
        mod = load()
        # No priority label, an unrelated label, and empty/None all map to 0 (the
        # additive-no-regression floor: an unlabeled candidate orders by recency).
        self.assertEqual(mod.candidate_priority([{"name": "enhancement"}]), 0)
        self.assertEqual(mod.candidate_priority([]), 0)
        self.assertEqual(mod.candidate_priority(None), 0)


class CandidateBlockedByTest(unittest.TestCase):
    """candidate_blocked_by parses "depends-on:/blocked-by: #N" markers from an issue body into the
    dispatchorder Candidate.blocked_by list (empty when the body names no prerequisite) -- #3224."""

    def test_marker_forms(self) -> None:
        mod = load()
        # Both marker verbs, hyphenated or spaced, colon optional.
        self.assertEqual(mod.candidate_blocked_by("depends-on: #120"), ["120"])
        self.assertEqual(mod.candidate_blocked_by("Depends on #120"), ["120"])
        self.assertEqual(mod.candidate_blocked_by("blocked-by: #120"), ["120"])
        self.assertEqual(mod.candidate_blocked_by("Blocked by #120"), ["120"])

    def test_multiple_refs_and_dedup(self) -> None:
        mod = load()
        # Comma/and/&-separated refs on one marker, and across markers, deduped in first-seen order.
        self.assertEqual(
            mod.candidate_blocked_by("blocked-by: #120, #121 and #122"),
            ["120", "121", "122"])
        self.assertEqual(
            mod.candidate_blocked_by("Depends on #7.\nAlso blocked by #7 & #9."),
            ["7", "9"])

    def test_no_marker_is_empty(self) -> None:
        mod = load()
        # Prose that merely contains the words never matches (the marker must be followed by #N);
        # a body-free / marker-free issue carries no prerequisite edge (additive-no-regression).
        self.assertEqual(mod.candidate_blocked_by("it depends on the weather"), [])
        self.assertEqual(mod.candidate_blocked_by("no markers here #notanumber"), [])
        self.assertEqual(mod.candidate_blocked_by(""), [])
        self.assertEqual(mod.candidate_blocked_by(None), [])


class UnresolvedAcceptanceGateTest(unittest.TestCase):
    """#5070: a contract whose ``## Acceptance gate`` explicitly needs operator
    input is held BEFORE worker spawn, even when its aggregate score clears the
    floor. Drives the real admission seam (issue_contract_review ->
    issue_contract_hold_reason) with only the `fak issue contract` shell-out
    stubbed, so the gate is the thing under test, not a mock of it."""

    #: The gate #2806 carried when the live resolver wrongly admitted it.
    UNRESOLVED_GATE = "unknown -- needs operator input"
    #: #5070's own gate: concrete, and it happens to contain the word
    #: "unresolved" -- the confusion risk a naive substring match would trip on.
    CONCRETE_GATE = ("`tools/issue_resolve_dispatch_test.py` regression covering "
                     "unresolved and concrete acceptance-gate controls.")

    @staticmethod
    def _body(gate: str, *, trailer: str = "") -> str:
        return (f"## Working spine\nHold unexecutable contracts.\n\n"
                f"## Acceptance gate\n{gate}\n\n"
                f"## Lane\ntools\n{trailer}")

    @staticmethod
    def _passing_runner(*_a, **_k):
        """A `fak issue contract` run that PASSES the numeric floor, so the only
        thing that can hold the contract is the acceptance-gate check."""
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps({
                "ok": True,
                "reviews": [{"ok": True, "reasons": [], "missing_fields": [],
                             "score": {"total": 100},
                             "spine_priority": {"total": 100}}],
            }), stderr="")

    def _review(self, body: str) -> dict:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            return mod.issue_contract_review(
                Path(td), {"number": 2806, "title": "t", "body": body}, 2806,
                runner=self._passing_runner)

    @staticmethod
    def _holds(mod, contract: dict) -> bool:
        """The module's own admission predicate, mirrored: this is exactly what
        the resolver's `_contract_holds` reads before it spawns or leases."""
        return bool(contract.get("unavailable") or not contract.get("ok") or
                    int(contract.get("score") or 0)
                    < mod.DEFAULT_ISSUE_CONTRACT_MIN_SCORE)

    def test_unresolved_gate_holds_a_score_passing_contract(self) -> None:
        mod = load()
        contract = self._review(self._body(self.UNRESOLVED_GATE))
        # The score cleared the floor -- the hold is the gate's doing alone.
        self.assertEqual(contract["score"], mod.DEFAULT_ISSUE_CONTRACT_MIN_SCORE)
        self.assertFalse(contract["ok"])
        self.assertTrue(self._holds(mod, contract))
        reason = mod.issue_contract_hold_reason(contract)
        self.assertTrue(reason.startswith(mod.ACCEPTANCE_GATE_HOLD_REASON), reason)
        self.assertIn("needs operator input", reason)

    def test_concrete_gate_stays_dispatchable(self) -> None:
        mod = load()
        contract = self._review(self._body(self.CONCRETE_GATE))
        self.assertTrue(contract["ok"])
        self.assertFalse(self._holds(mod, contract))
        self.assertEqual(contract["acceptance_gate_hold"], "")
        self.assertEqual(mod.issue_contract_hold_reason(contract),
                         "issue contract passed")

    def test_operator_prose_outside_the_gate_never_holds(self) -> None:
        """The declared confusion risk: bind the Acceptance gate section only."""
        mod = load()
        body = self._body(
            self.CONCRETE_GATE,
            trailer="\n## Coordination\nPing the operator; needs operator input "
                    "before the milestone is set.\n")
        contract = self._review(body)
        self.assertTrue(contract["ok"])
        self.assertFalse(self._holds(mod, contract))

    def test_unresolved_vocabulary_forms(self) -> None:
        mod = load()
        for gate in ("unknown -- needs operator input", "unknown", "TBD",
                     "To be determined", "unresolved: awaiting operator decision",
                     "- unknown, pending operator sign-off", "???"):
            with self.subTest(gate=gate):
                self.assertTrue(mod.unresolved_acceptance_gate(self._body(gate)),
                                f"expected a hold for {gate!r}")

    def test_concrete_vocabulary_forms_stay_dispatchable(self) -> None:
        mod = load()
        for gate in (self.CONCRETE_GATE,
                     "`go test ./internal/dispatchtick` is green.",
                     "make ci green; the unknown-flake list is unchanged.",
                     "A render witness the operator reviewed already."):
            with self.subTest(gate=gate):
                self.assertEqual(mod.unresolved_acceptance_gate(self._body(gate)),
                                 "", f"unexpected hold for {gate!r}")

    def test_crlf_body_still_binds_the_gate(self) -> None:
        """A GitHub-authored body arrives CRLF; the gate must bind there too, or
        the check silently no-ops on exactly the live issues it exists for."""
        mod = load()
        crlf = self._body(self.UNRESOLVED_GATE).replace("\n", "\r\n")
        self.assertIn("\r\n", crlf)
        self.assertTrue(mod.unresolved_acceptance_gate(crlf))
        self.assertEqual(mod.unresolved_acceptance_gate(
            self._body(self.CONCRETE_GATE).replace("\n", "\r\n")), "")

    def test_no_acceptance_gate_section_is_not_a_gate_hold(self) -> None:
        """A body with no gate section is the score gate's business, not this
        check's -- it must stay additive-no-regression."""
        mod = load()
        self.assertEqual(mod.unresolved_acceptance_gate("## Lane\ntools\n"), "")
        self.assertEqual(mod.unresolved_acceptance_gate(None), "")


if __name__ == "__main__":
    unittest.main()
