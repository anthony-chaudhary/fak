#!/usr/bin/env python3
"""Tests for worktree_doctor — the safe "converge to one worktree on master" sweeper.

These exercise the PURE decision surface (issues_of / safe_to_remove / make_plan /
parse_worktree_list) with synthetic worktree records, so the safety rule — never
remove a worktree that has uncommitted work, an in-progress merge, or commits not
yet on master — is proven without touching a real repo."""
import datetime
import json
import os
import tempfile
import unittest

import worktree_doctor as wd


def sig(path, branch, primary=False, dirty=False, untracked=0, mid_op=None, unmerged=0, unpushed=0):
    return {"path": path, "branch": branch, "is_primary": primary, "dirty": dirty,
            "untracked": untracked, "mid_op": mid_op, "unmerged_to_master": unmerged,
            "unpushed": unpushed}


class IssuesAndSafety(unittest.TestCase):
    def test_clean_merged_nonprimary_is_safe(self):
        s = sig("/wt/clean", "feature-done", unmerged=0)
        self.assertEqual(wd.issues_of(s), [])
        self.assertTrue(wd.safe_to_remove(s))

    def test_each_kind_of_work_blocks_removal(self):
        for label, s in [
            ("dirty", sig("/wt", "b", dirty=True)),
            ("untracked", sig("/wt", "b", untracked=3)),
            ("mid-merge", sig("/wt", "b", mid_op="merge")),
            ("unmerged", sig("/wt", "b", unmerged=2)),
        ]:
            with self.subTest(label):
                self.assertTrue(wd.issues_of(s), f"{label} should be an issue")
                self.assertFalse(wd.safe_to_remove(s), f"{label} must NOT be safe to remove")

    def test_primary_is_never_safe_to_remove(self):
        s = sig("/main", "master", primary=True)  # clean, but primary
        self.assertFalse(wd.safe_to_remove(s))

    def test_is_clean_master(self):
        self.assertTrue(wd.is_clean_master(sig("/m", "master")))
        self.assertFalse(wd.is_clean_master(sig("/m", "master", dirty=True)))
        self.assertFalse(wd.is_clean_master(sig("/m", "feature")))


class PlanComposition(unittest.TestCase):
    def test_converged_single_clean_master_primary(self):
        plan = wd.make_plan([sig("/main", "master", primary=True)])
        self.assertTrue(plan["converged"])
        self.assertEqual(plan["keeper"], "/main")
        self.assertEqual(plan["prune"], [])

    def test_prunes_only_clean_merged_keeps_blocked(self):
        sigs = [
            sig("/main", "master", primary=True),                 # keeper
            sig("/obs", "master"),                                # redundant clean master -> prunable
            sig("/done", "feature-merged", unmerged=0),           # clean + merged -> prunable
            sig("/wip", "integrate", dirty=True),                 # has work -> BLOCKED (pass)
            sig("/stuck", "feat", mid_op="merge"),                # mid-merge -> BLOCKED (pass)
        ]
        plan = wd.make_plan(sigs)
        self.assertEqual(plan["keeper"], "/main")
        pruned = {p["path"] for p in plan["prune"]}
        self.assertEqual(pruned, {"/obs", "/done"})
        blocked = {b["path"] for b in plan["blocked"]}
        self.assertEqual(blocked, {"/wip", "/stuck"})
        self.assertFalse(plan["converged"])

    def test_primary_offtrack_keeps_a_master_worktree_and_flags(self):
        # mirrors the real situation: primary mid-merge on a feature branch, a clean
        # master worktree exists elsewhere. Keep the master worktree; flag the primary.
        sigs = [
            sig("/main", "fak-v0.1", primary=True, mid_op="merge", unmerged=4),
            sig("/obs", "master"),                                # the only clean master
            sig("/integrate", "integrate-to-master", dirty=True),
        ]
        plan = wd.make_plan(sigs)
        self.assertTrue(plan["primary_offtrack"])
        self.assertEqual(plan["keeper"], "/obs")     # keep the clean master
        self.assertEqual(plan["prune"], [])          # /obs is the keeper; /integrate is blocked
        self.assertTrue(any("off master" in n for n in plan["notes"]))

    def test_dirty_on_master_primary_is_not_offtrack_or_needs_human(self):
        # THE shared-worktree norm on this box: the primary is ON master but carries
        # in-flight peer work (dirty + untracked + local-ahead commits). This must NOT
        # be flagged "off master", must NOT cry wolf as needs_human (so a nightly cron
        # exits 0 and an operator never learns to ignore it), and must still prune the
        # safe surplus. Previously this mislabelled the primary "off master" and told
        # the operator to run a no-op `git switch master`.
        sigs = [
            sig("/main", "master", primary=True, dirty=True, untracked=6, unmerged=3),
            sig("/stray-a", None),          # detached, clean, merged -> safe to prune
            sig("/stray-b", None),          # detached, clean, merged -> safe to prune
        ]
        plan = wd.make_plan(sigs)
        self.assertFalse(plan["primary_offtrack"], "dirty-on-master primary is NOT off master")
        self.assertFalse(plan["needs_human"], "shared-worktree in-flight work must not cry wolf")
        self.assertEqual({p["path"] for p in plan["prune"]}, {"/stray-a", "/stray-b"})
        self.assertFalse(any("off master" in n for n in plan["notes"]),
                         "must not emit the false 'off master' note on an on-master primary")

    def test_stuck_merge_on_master_primary_still_needs_human(self):
        # dirty/untracked on master is fine, but a STUCK merge on the primary is a real
        # anomaly even when the primary is on master — it must still surface + exit 1.
        sigs = [sig("/main", "master", primary=True, mid_op="merge")]
        plan = wd.make_plan(sigs)
        self.assertFalse(plan["primary_offtrack"])          # it IS on master
        self.assertTrue(plan["needs_human"])                # but the stuck merge is real
        self.assertTrue(any("stuck" in n for n in plan["notes"]))

    def test_dirty_on_master_primary_not_in_blocked(self):
        # the primary is never a prune candidate, so its soft issues must not appear in
        # `blocked` (which lists work-at-risk NON-primary worktrees the doctor passes on).
        sigs = [sig("/main", "master", primary=True, dirty=True, untracked=2)]
        plan = wd.make_plan(sigs)
        self.assertEqual(plan["blocked"], [])

    def test_never_prunes_the_last_master_when_no_keeper(self):
        # primary off master + dirty, the only master worktree is itself the lone path
        # to master: do not remove it even though it is non-primary + clean.
        sigs = [
            sig("/main", "feature", primary=True, dirty=True),
            sig("/obs", "master"),
        ]
        plan = wd.make_plan(sigs)
        self.assertEqual(plan["keeper"], "/obs")
        self.assertEqual(plan["prune"], [])  # keeper is /obs; nothing else to prune


class AllowListedWorktrees(unittest.TestCase):
    def test_allow_listed_release_line_is_retained_not_blocked(self):
        # the real steady state on this box: a clean master primary + the long-lived
        # fak-v0.1 worktree (commits not on master). With --allow-branch it is RETAINED,
        # not a blocker, so a cron run does not cry wolf.
        sigs = [
            sig("/main", "master", primary=True),
            sig("/rel", "fak-v0.1", unmerged=6),
        ]
        plan = wd.make_plan(sigs, allow_branches=["fak-v0.1"])
        self.assertEqual(plan["blocked"], [])
        self.assertEqual([r["path"] for r in plan["retained"]], ["/rel"])
        self.assertFalse(plan["needs_human"])
        self.assertTrue(plan["converged"])          # only NON-retained worktree is the master primary
        self.assertEqual(plan["prune"], [])         # never prune an allow-listed worktree

    def test_without_allow_list_same_release_line_needs_human(self):
        sigs = [sig("/main", "master", primary=True), sig("/rel", "fak-v0.1", unmerged=6)]
        plan = wd.make_plan(sigs)                    # no allow-list
        self.assertTrue(plan["needs_human"])
        self.assertFalse(plan["converged"])

    def test_allow_list_does_not_rescue_a_real_anomaly(self):
        # a dirty stray worktree on a non-allow-listed branch still needs a human.
        sigs = [
            sig("/main", "master", primary=True),
            sig("/rel", "fak-v0.1", unmerged=6),
            sig("/stray", "wip", dirty=True),
        ]
        plan = wd.make_plan(sigs, allow_branches=["fak-v0.1"])
        self.assertEqual([b["path"] for b in plan["blocked"]], ["/stray"])
        self.assertTrue(plan["needs_human"])


class DeletableBranches(unittest.TestCase):
    def test_only_merged_unprotected_unchecked_out_are_deletable(self):
        local = ["master", "fak-v0.1", "feature-merged", "feature-wip", "old-merged"]
        merged = {"master", "feature-merged", "old-merged"}   # fak-v0.1 + wip not merged
        checked_out = {"master", "fak-v0.1"}
        protected = {"master", "fak-v0.1"}
        got = wd.deletable_branches(local, protected, checked_out, merged)
        self.assertEqual(got, ["feature-merged", "old-merged"])

    def test_protected_and_checked_out_are_never_deletable(self):
        # even a merged branch is spared if it is protected or checked out somewhere.
        local = ["master", "release", "tmp"]
        merged = {"master", "release", "tmp"}
        self.assertEqual(
            wd.deletable_branches(local, {"master"}, {"release"}, merged), ["tmp"])

    def test_nothing_merged_means_nothing_deletable(self):
        self.assertEqual(
            wd.deletable_branches(["a", "b"], {"master"}, set(), set()), [])


class PorcelainParsing(unittest.TestCase):
    SAMPLE = (
        "worktree /repo/main\n"
        "HEAD 1111111111111111111111111111111111111111\n"
        "branch refs/heads/fak-v0.1\n"
        "\n"
        "worktree /repo/obs\n"
        "HEAD 2222222222222222222222222222222222222222\n"
        "branch refs/heads/master\n"
        "\n"
        "worktree /repo/detached\n"
        "HEAD 3333333333333333333333333333333333333333\n"
        "detached\n"
    )

    def test_parse_marks_primary_and_branches(self):
        wts = wd.parse_worktree_list(self.SAMPLE)
        self.assertEqual(len(wts), 3)
        self.assertTrue(wts[0]["is_primary"])
        self.assertFalse(wts[1]["is_primary"])
        self.assertEqual(wts[0]["branch"], "fak-v0.1")
        self.assertEqual(wts[1]["branch"], "master")
        self.assertTrue(wts[2]["detached"])
        self.assertIsNone(wts[2]["branch"])


class TrunkAware(unittest.TestCase):
    """The doctor anchors on the REAL trunk (this repo is `main`), not a hardcoded
    `master`. The pure plan logic takes a trunk= name; these prove `main` works."""

    def test_main_primary_is_converged_on_main_trunk(self):
        plan = wd.make_plan([sig("/repo", "main", primary=True)], trunk="main")
        self.assertTrue(plan["converged"])
        self.assertEqual(plan["keeper"], "/repo")
        self.assertFalse(plan["primary_offtrack"])
        self.assertFalse(plan["needs_human"])

    def test_master_worktree_is_offtrack_when_trunk_is_main(self):
        # a 'master' branch is NOT the trunk here; an on-master primary reads as off-trunk.
        plan = wd.make_plan([sig("/repo", "master", primary=True)], trunk="main")
        self.assertTrue(plan["primary_offtrack"])
        self.assertTrue(any("off main" in n for n in plan["notes"]))

    def test_default_trunk_is_still_master_for_back_compat(self):
        # callers that pass no trunk= keep the original master semantics (existing tests).
        self.assertTrue(wd.is_on_master(sig("/m", "master")))
        self.assertTrue(wd.is_on_master(sig("/m", "main"), trunk="main"))
        self.assertFalse(wd.is_on_master(sig("/m", "main")))


def _swp(path, branch=None, primary=False, dirty=False, untracked=0, mid_op=None,
         age_seconds=10_000, detached=True):
    return {"path": path, "branch": branch, "is_primary": primary, "dirty": dirty,
            "untracked": untracked, "mid_op": mid_op, "age_seconds": age_seconds,
            "detached": detached}


class DisposablePath(unittest.TestCase):
    def test_scratchpad_and_prwork_segments_are_disposable(self):
        self.assertTrue(wd.is_disposable_path("/tmp/claude/abc/scratchpad/rel-wt"))
        self.assertTrue(wd.is_disposable_path("C:/work/pr-work/fak-pr-814"))

    def test_a_normal_checkout_is_not_disposable(self):
        self.assertFalse(wd.is_disposable_path("C:/work/fak"))
        self.assertFalse(wd.is_disposable_path(""))

    def test_explicit_root_makes_a_path_disposable(self):
        self.assertTrue(wd.is_disposable_path("/scratch/x/wt", roots=["/scratch"]))
        self.assertFalse(wd.is_disposable_path("/elsewhere/wt", roots=["/scratch"]))


class SweepCandidates(unittest.TestCase):
    def test_primary_is_never_swept_even_if_disposable(self):
        sigs = [_swp("/tmp/scratchpad/wt", primary=True)]
        self.assertEqual(wd.sweep_candidates(sigs), [])

    def test_stale_disposable_is_reaped_and_flags_work(self):
        sigs = [
            _swp("/tmp/scratchpad/clean", age_seconds=10_000),
            _swp("/tmp/scratchpad/dirty", dirty=True, untracked=2, age_seconds=10_000),
        ]
        got = wd.sweep_candidates(sigs, fresh_seconds=3600)
        bypath = {c["path"]: c for c in got}
        self.assertEqual(set(bypath), {"/tmp/scratchpad/clean", "/tmp/scratchpad/dirty"})
        self.assertFalse(bypath["/tmp/scratchpad/clean"]["has_work"])
        self.assertTrue(bypath["/tmp/scratchpad/dirty"]["has_work"])

    def test_fresh_disposable_is_spared_protecting_live_sessions(self):
        # a worktree touched 1 min ago is a live session — never reap it.
        sigs = [_swp("/tmp/scratchpad/live", age_seconds=60)]
        self.assertEqual(wd.sweep_candidates(sigs, fresh_seconds=3600), [])

    def test_unknown_age_reads_as_stale(self):
        sigs = [_swp("/tmp/scratchpad/old", age_seconds=None)]
        got = wd.sweep_candidates(sigs, fresh_seconds=3600)
        self.assertEqual([c["path"] for c in got], ["/tmp/scratchpad/old"])

    def test_non_disposable_and_kept_paths_are_spared(self):
        sigs = [
            _swp("C:/work/fak/sub", age_seconds=10_000),          # not disposable
            _swp("/tmp/scratchpad/keepme", age_seconds=10_000),   # disposable but kept
        ]
        got = wd.sweep_candidates(sigs, fresh_seconds=0,
                                  keep_paths=["/tmp/scratchpad/keepme"])
        self.assertEqual(got, [])


def _seed_day(base, name, files=("base-commit.txt",)):
    """Create base/<name>/ (a stand-in archive day) with a dummy file, so a survivor is a
    real on-disk directory and a reap actually has content to remove."""
    day = os.path.join(base, name)
    os.makedirs(day, exist_ok=True)
    for f in files:
        with open(os.path.join(day, f), "w", encoding="utf-8") as fh:
            fh.write("x\n")
    return day


class ArchiveDayRetention(unittest.TestCase):
    """The sweep-archive reaper: prune worktree-archive/<date> days older than the window,
    while the current run's day, every day inside the window, and any non-date entry survive.
    Age (with a generous floor) is the signal — an archive day is a recovery convenience, so
    an OLD day past the window is safe to delete and a RECENT one is not."""

    def test_reaps_only_stale_days_keeps_fresh_and_today(self):
        today = datetime.date(2026, 7, 22)
        keep = 7  # cutoff = 2026-07-15; a day strictly older than that is stale
        with tempfile.TemporaryDirectory() as base:
            fresh = [today.isoformat(),                                   # today's run
                     (today - datetime.timedelta(days=1)).isoformat(),    # yesterday
                     (today - datetime.timedelta(days=keep)).isoformat()] # exactly at the floor -> kept
            stale = [(today - datetime.timedelta(days=keep + 1)).isoformat(),  # one day past -> reaped
                     (today - datetime.timedelta(days=30)).isoformat()]        # long past -> reaped
            self.assertIn("2026-07-14", stale)  # the ticket's 4.6GB day (22 - 8) lands in the reap set
            for d in fresh + stale:
                _seed_day(base, d)
            # non-<date> siblings must be left completely alone (a stray file, an unrelated dir)
            _seed_day(base, "notes-not-a-date")
            with open(os.path.join(base, "README.txt"), "w", encoding="utf-8") as fh:
                fh.write("keep me\n")

            archive_dir = os.path.join(base, today.isoformat())
            removed = wd.reap_stale_archive_days(archive_dir, keep_days=keep, today=today)

            self.assertEqual(removed, sorted(stale))
            for d in stale:
                self.assertFalse(os.path.isdir(os.path.join(base, d)), f"{d} should be reaped")
            for d in fresh:
                self.assertTrue(os.path.isdir(os.path.join(base, d)), f"{d} must survive")
            self.assertTrue(os.path.isdir(os.path.join(base, "notes-not-a-date")))
            self.assertTrue(os.path.isfile(os.path.join(base, "README.txt")))

    def test_never_removes_the_current_run_day_even_if_it_parses_old(self):
        # Safety guard: the day THIS run writes into is skipped by path, so even a current
        # archive_dir whose own date is far past the window is never deleted.
        today = datetime.date(2020, 1, 15)  # cutoff = 2020-01-08
        with tempfile.TemporaryDirectory() as base:
            _seed_day(base, "2020-01-01")   # the current run's day: old, but must be spared
            _seed_day(base, "2019-12-01")   # a genuinely stale peer day -> reaped
            archive_dir = os.path.join(base, "2020-01-01")
            removed = wd.reap_stale_archive_days(archive_dir, keep_days=7, today=today)
            self.assertEqual(removed, ["2019-12-01"])
            self.assertTrue(os.path.isdir(os.path.join(base, "2020-01-01")))
            self.assertFalse(os.path.isdir(os.path.join(base, "2019-12-01")))

    def test_missing_base_dir_is_a_noop(self):
        # nothing has ever been swept -> the base doesn't exist -> reap is a clean no-op.
        with tempfile.TemporaryDirectory() as base:
            archive_dir = os.path.join(base, "never", "2026-07-22")
            self.assertEqual(wd.reap_stale_archive_days(archive_dir, keep_days=7,
                                                        today=datetime.date(2026, 7, 22)), [])

    def test_zero_window_still_spares_today(self):
        # a 0-day window reaps every past day but STILL never removes the current run's day.
        today = datetime.date(2026, 7, 22)
        with tempfile.TemporaryDirectory() as base:
            _seed_day(base, today.isoformat())
            _seed_day(base, (today - datetime.timedelta(days=1)).isoformat())
            archive_dir = os.path.join(base, today.isoformat())
            removed = wd.reap_stale_archive_days(archive_dir, keep_days=0, today=today)
            self.assertEqual(removed, [(today - datetime.timedelta(days=1)).isoformat()])
            self.assertTrue(os.path.isdir(os.path.join(base, today.isoformat())))


class MultiRepoAggregate(unittest.TestCase):
    """One machine-global scheduled task maintains SEVERAL checkouts, so the run's verdict
    must fold every repository's own result. The regression these lock down is #6498: the
    live task swept C:\\work\\fleet only, reported one clean repo, and nothing anywhere said
    C:\\work\\fak had never been looked at. A clean result must never stand in for a failing
    or an absent one — in either direction, whatever order the repos are listed in."""

    @staticmethod
    def repo(path, code=0, reason=wd.REASON_OK, **extra):
        return dict({"repo": path, "exit": code, "reason": reason,
                     "paths_scanned": [path], "retained": [], "pruned": [], "blocked": []},
                    **extra)

    def test_all_healthy_is_zero(self):
        got = wd.aggregate_result([self.repo("/fleet"), self.repo("/fak")])
        self.assertEqual((got["exit"], got["reason"], got["failed_repo"]),
                         (0, wd.REASON_OK, None))

    def test_a_clean_repo_cannot_mask_a_blocked_peer_in_either_order(self):
        clean = self.repo("/fleet")
        blocked = self.repo("/fak", 1, wd.REASON_NEEDS_HUMAN)
        for label, rows in (("clean first", [clean, blocked]),
                            ("blocked first", [blocked, clean])):
            with self.subTest(label):
                got = wd.aggregate_result(rows)
                self.assertEqual(got["exit"], 1)
                self.assertEqual(got["reason"], wd.REASON_NEEDS_HUMAN)
                self.assertEqual(got["failed_repo"], "/fak")

    def test_could_not_run_outranks_needs_human_and_names_that_repo(self):
        rows = [self.repo("/fleet"),
                self.repo("/fak", 1, wd.REASON_NEEDS_HUMAN),
                self.repo("/broken", 2, wd.REASON_ERROR, error="not a git repository")]
        got = wd.aggregate_result(rows)
        self.assertEqual((got["exit"], got["reason"], got["failed_repo"]),
                         (2, wd.REASON_ERROR, "/broken"))

    def test_a_clean_repo_cannot_mask_an_UNSWEPT_one(self):
        # the #6498 shape exactly: the one repo in the run is perfectly healthy, and the
        # repo that was never in the run is the failure the exit code must carry.
        coverage = [{"repo": "/fleet", "declared": True, "covered": True,
                     "reason": wd.REASON_OK},
                    {"repo": "/fak", "declared": True, "covered": False,
                     "reason": wd.REASON_NO_COVERAGE}]
        got = wd.aggregate_result([self.repo("/fleet")], coverage)
        self.assertEqual((got["exit"], got["reason"], got["failed_repo"]),
                         (1, wd.REASON_NO_COVERAGE, "/fak"))

    def test_full_coverage_over_healthy_repos_stays_zero(self):
        coverage = [{"repo": "/fleet", "declared": True, "covered": True,
                     "reason": wd.REASON_OK},
                    {"repo": "/fak", "declared": True, "covered": True,
                     "reason": wd.REASON_OK}]
        got = wd.aggregate_result([self.repo("/fleet"), self.repo("/fak")], coverage)
        self.assertEqual(got["exit"], 0)


class OneBadRepoDoesNotSinkTheRun(unittest.TestCase):
    """A --repo that cannot be run at all must come back as ITS OWN typed row. Before the
    multi-repo run this was an uncaught OSError out of `git -C <missing>`, which would have
    taken the sweep of every other repository down with it — the very masking #6498 is about."""

    def test_git_survives_an_unusable_cwd(self):
        with tempfile.TemporaryDirectory() as base:
            rc, out, err = wd._git(["status"], cwd=os.path.join(base, "no-such-dir"))
            self.assertNotEqual(rc, 0)
            self.assertEqual(out, "")
            self.assertTrue(err)

    def test_missing_repo_yields_exit_2_and_never_hides_a_peer(self):
        import types
        with tempfile.TemporaryDirectory() as base:
            args = types.SimpleNamespace(master_ref="origin/main", fetch=False,
                                         archive_dir=os.path.join(base, "archive"))
            result, ctx = wd.run_repo(os.path.join(base, "no-such-dir"), args)
            self.assertIsNone(ctx)
            self.assertEqual((result["exit"], result["reason"]), (2, wd.REASON_ERROR))
            self.assertTrue(result["error"])
            self.assertEqual(result["paths_scanned"], [])
            got = wd.aggregate_result([{"repo": "/fleet", "exit": 0, "reason": wd.REASON_OK},
                                       result])
            self.assertEqual((got["exit"], got["failed_repo"]),
                             (2, result["repo"]))


class DoctorCoverage(unittest.TestCase):
    """A checkout whose AGENTS.md promises a scheduled worktree doctor and that no --repo
    maintains is a typed gap, not silence."""

    def test_agents_md_claim_is_detected_and_a_bare_doc_is_not(self):
        self.assertTrue(wd.agents_md_declares_doctor(
            "A scheduled task runs it (`tools/register_worktree_doctor.ps1`)."))
        self.assertTrue(wd.agents_md_declares_doctor(
            "`tools/worktree_doctor.py` is the janitor: it prunes safe strays."))
        self.assertFalse(wd.agents_md_declares_doctor("Run make ci before you push."))
        self.assertFalse(wd.agents_md_declares_doctor(None))

    def test_declaring_checkout_absent_from_the_run_is_a_gap(self):
        cands = [("/work/fleet", "a scheduled task runs tools/worktree_doctor.py"),
                 ("/work/fak", "the janitor is tools/worktree_doctor.py"),
                 ("/work/unrelated", "nothing to see here")]
        rows = wd.coverage_gaps(cands, ["/work/fleet"])
        by_repo = {r["repo"]: r for r in rows}
        self.assertEqual(len(rows), 2, "only the two DECLARING checkouts get a row")
        self.assertTrue(by_repo[os.path.abspath("/work/fleet")]["covered"])
        gap = by_repo[os.path.abspath("/work/fak")]
        self.assertFalse(gap["covered"])
        self.assertEqual(gap["reason"], wd.REASON_NO_COVERAGE)

    def test_repo_paths_match_across_separator_and_case_spelling(self):
        cands = [("/work/fak", "tools/worktree_doctor.py")]
        spelled = os.path.join(os.path.abspath("/work"), "fak") + os.sep
        self.assertTrue(wd.coverage_gaps(cands, [spelled])[0]["covered"])

    def test_scratch_clones_of_a_maintained_repo_raise_no_gap(self):
        # a busy box carries dozens of throwaway clones of the SAME repo, each with the
        # identical AGENTS.md. Auditing them one by one would report ~50 gaps for one real
        # one, so identity (origin URL) groups them and the canonical path names the group.
        cands = [("/work/fak", "tools/worktree_doctor.py", "git@github:org/fak"),
                 ("/work/.iso5032", "tools/worktree_doctor.py", "git@github:org/fak"),
                 ("/work/fak-verify-1349", "tools/worktree_doctor.py", "git@github:org/fak"),
                 ("/work/fleet", "tools/worktree_doctor.py", "git@github:org/fleet")]
        rows = wd.coverage_gaps(cands, ["/work/fak"], ["git@github:org/fak"])
        self.assertEqual([os.path.basename(r["repo"]) for r in rows], ["fak", "fleet"])
        self.assertEqual([r["covered"] for r in rows], [True, False])

    def test_a_scratch_clone_alone_does_not_cover_its_repository_by_identity(self):
        # the identity match is what covers the group; without it the repo is still a gap.
        cands = [("/work/fak", "tools/worktree_doctor.py", "git@github:org/fak")]
        self.assertFalse(wd.coverage_gaps(cands, ["/work/fleet"], ["git@github:org/fleet"])[0]
                         ["covered"])

    def test_scan_finds_sibling_checkouts_and_skips_non_repos(self):
        with tempfile.TemporaryDirectory() as base:
            for name, agents in (("fleet", "runs tools/worktree_doctor.py"),
                                 ("fak", "A scheduled task runs it (register_worktree_doctor.ps1)"),
                                 ("notes", "just a folder"), ("empty", None)):
                os.makedirs(os.path.join(base, name))
                if agents is not None:
                    with open(os.path.join(base, name, "AGENTS.md"), "w",
                              encoding="utf-8") as fh:
                        fh.write(agents)
            found = wd.scan_coverage_roots([base], identify=lambda p: os.path.basename(p))
            rows = wd.coverage_gaps(found, [os.path.join(base, "fleet")], ["fleet"])
            self.assertEqual([os.path.basename(r["repo"]) for r in rows], ["fak", "fleet"])
            self.assertEqual([r["covered"] for r in rows], [False, True])

    def test_scan_only_pays_for_identity_on_a_declaring_checkout(self):
        with tempfile.TemporaryDirectory() as base:
            for name, agents in (("fak", "tools/worktree_doctor.py"), ("notes", "nothing")):
                os.makedirs(os.path.join(base, name))
                with open(os.path.join(base, name, "AGENTS.md"), "w", encoding="utf-8") as fh:
                    fh.write(agents)
            asked = []

            def identify(p):
                asked.append(os.path.basename(p))
                return "origin"

            wd.scan_coverage_roots([base], identify=identify)
            self.assertEqual(asked, ["fak"])

    def test_missing_scan_root_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as base:
            gone = os.path.join(base, "no-such-dir")
            self.assertEqual(wd.scan_coverage_roots([gone]), [])


class MainEnumeratesRepos(unittest.TestCase):
    """End-to-end through main(): every --repo is run, every one gets its own record in the
    JSON envelope, and the process exit names the repository at fault."""

    def _run(self, argv, runner, identity=None):
        import contextlib
        import io as _io
        buf = _io.StringIO()
        real, real_id = wd.run_repo, wd.repo_identity
        wd.run_repo = runner
        # These temp dirs are not clones, so stand in for the origin-URL lookup the real
        # coverage audit uses to group a repository's scratch copies.
        wd.repo_identity = identity or (lambda p: os.path.basename(os.path.abspath(p)))
        try:
            with contextlib.redirect_stdout(buf):
                code = wd.main(argv)
        finally:
            wd.run_repo, wd.repo_identity = real, real_id
        return code, json.loads(buf.getvalue())

    def test_every_repo_is_run_and_the_failing_one_is_named(self):
        seen = []

        def fake(repo, args):
            seen.append(repo)
            blocked = repo.endswith("fak")
            return ({"repo": repo, "exit": 1 if blocked else 0,
                     "reason": wd.REASON_NEEDS_HUMAN if blocked else wd.REASON_OK,
                     "paths_scanned": [repo], "retained": [], "pruned": [],
                     "blocked": [repo + "/stray"] if blocked else []}, None)

        code, out = self._run(["--repo", "/work/fleet", "--repo", "/work/fak", "--json"], fake)
        self.assertEqual(seen, ["/work/fleet", "/work/fak"], "both repos were maintained")
        self.assertEqual(code, 1)
        self.assertEqual(out["schema"], wd.JSON_SCHEMA)
        self.assertEqual(out["reason"], wd.REASON_NEEDS_HUMAN)
        self.assertEqual(out["failed_repo"], "/work/fak")
        # the healthy repo is still fully reported: the failure hides neither peer.
        self.assertEqual([r["repo"] for r in out["repos"]], ["/work/fleet", "/work/fak"])
        self.assertEqual(out["repos"][0]["paths_scanned"], ["/work/fleet"])
        self.assertEqual(out["repos"][1]["blocked"], ["/work/fak/stray"])

    def test_repeated_repo_is_maintained_once(self):
        seen = []

        def fake(repo, args):
            seen.append(repo)
            return ({"repo": repo, "exit": 0, "reason": wd.REASON_OK,
                     "paths_scanned": [], "retained": [], "pruned": [], "blocked": []}, None)

        code, out = self._run(["--repo", "/work/fak", "--repo", "/work/fak/", "--json"], fake)
        self.assertEqual(len(seen), 1)
        self.assertEqual(code, 0)
        self.assertEqual(len(out["repos"]), 1)

    def test_uncovered_checkout_fails_a_run_whose_repos_are_all_clean(self):
        with tempfile.TemporaryDirectory() as base:
            fleet, fak = os.path.join(base, "fleet"), os.path.join(base, "fak")
            for d in (fleet, fak):
                os.makedirs(d)
                with open(os.path.join(d, "AGENTS.md"), "w", encoding="utf-8") as fh:
                    fh.write("A scheduled task runs tools/worktree_doctor.py")

            def fake(repo, args):
                return ({"repo": os.path.abspath(repo), "exit": 0, "reason": wd.REASON_OK,
                         "paths_scanned": [], "retained": [], "pruned": [],
                         "blocked": []}, None)

            code, out = self._run(["--repo", fleet, "--coverage-scan", base, "--json"], fake)
            self.assertEqual(code, 1, "an unswept declaring checkout must fail the run")
            self.assertEqual(out["reason"], wd.REASON_NO_COVERAGE)
            self.assertEqual(os.path.normcase(out["failed_repo"]), os.path.normcase(fak))
            self.assertEqual([r["covered"] for r in out["coverage"]], [False, True])

            code, out = self._run(["--repo", fleet, "--repo", fak,
                                   "--coverage-scan", base, "--json"], fake)
            self.assertEqual(code, 0, "covering both checkouts clears the gap")
            self.assertEqual(out["reason"], wd.REASON_OK)


if __name__ == "__main__":
    unittest.main()
