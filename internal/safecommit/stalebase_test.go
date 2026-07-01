package safecommit

import (
	"context"
	"strings"
	"testing"
)

// staleBaseReply overlays onTrunkBase with the refs the stale-base guard reads, so a test
// can vary just the three diff/merge-base keys. peerDiff is `git diff <mb> origin/main -- P`
// (lines origin gained since the fork); wtDiff is `git diff origin/main -- P` (lines origin
// holds that the working tree lacks shown as '-'). A non-empty origin SHA + a merge-base that
// DIFFERS from it is what arms the content check.
func staleBaseReply(peerDiff, wtDiff string) map[string]reply {
	rep := onTrunkBase()
	rep["rev-parse --verify --quiet origin/main"] = reply{out: "0000aaaa\n", code: 0}
	rep["merge-base HEAD origin/main"] = reply{out: "ffff1111\n", code: 0}
	rep["diff-peer"] = reply{out: peerDiff, code: 0}
	rep["diff-wt"] = reply{out: wtDiff, code: 0}
	return rep
}

// TestStaleBaseDeletion_refusesDroppedPeerBlock is the bug-reproducing test (#1073): a peer
// pushed a multi-line block to origin/main:P that the working-tree P lacks. Committing the
// stale blob by pathspec would delete it. The guard must refuse BEFORE staging/committing.
func TestStaleBaseDeletion_refusesDroppedPeerBlock(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "") // default = block
	// Peer added a 3-line onUpstreamRetry block since the fork point...
	peerDiff := "@@ -10,0 +11,3 @@\n" +
		"+func onUpstreamRetry(r *Req) {\n" +
		"+\tr.RetryNotify()\n" +
		"+\treturn\n"
	// ...and the working tree (committed-to-be blob) lacks all 3 -> origin-vs-wt shows them removed.
	wtDiff := "@@ -11,3 +10,0 @@\n" +
		"-func onUpstreamRetry(r *Req) {\n" +
		"-\tr.RetryNotify()\n" +
		"-\treturn\n"
	g := &fakeGit{reply: staleBaseReply(peerDiff, wtDiff)}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonStaleBaseDeletion {
		t.Fatalf("want %q, got reason=%q detail=%q", ReasonStaleBaseDeletion, res.Reason, res.Detail)
	}
	if res.Committed {
		t.Fatalf("a stale-base refusal must not commit, got %+v", res)
	}
	// The refusal happens before any add/commit — strictly cleaner than PATHSPEC_RACE.
	for _, forbidden := range []string{"add", "commit"} {
		if g.sawSubcommand(forbidden) {
			t.Fatalf("stale-base refusal must not %q; calls=%v", forbidden, g.calls)
		}
	}
	if !strings.Contains(res.Detail, "internal/foo/bar.go") || !strings.Contains(res.Detail, "git fetch") {
		t.Fatalf("detail should name the path and the fetch+merge remedy, got %q", res.Detail)
	}
}

// TestStaleBaseDeletion_warnModeCommits proves FAK_STALE_BASE_GUARD=warn records the would-be
// refusal in Detail but lets the commit proceed (the documented one-shot escape).
func TestStaleBaseDeletion_warnModeCommits(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "warn")
	peerDiff := "@@ -1,0 +1,2 @@\n+peerLineOne()\n+peerLineTwo()\n"
	wtDiff := "@@ -1,2 +1,0 @@\n-peerLineOne()\n-peerLineTwo()\n"
	g := &fakeGit{reply: staleBaseReply(peerDiff, wtDiff)}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != "" {
		t.Fatalf("warn mode must not set a refusal reason, got %q", res.Reason)
	}
	if !res.Verified {
		t.Fatalf("warn mode should commit and verify, got %+v", res)
	}
	if !strings.Contains(res.Detail, "STALE_BASE_DELETION (warn)") {
		t.Fatalf("warn mode should record the would-be refusal in Detail, got %q", res.Detail)
	}
}

// TestStaleBaseDeletion_offModeSkips proves FAK_STALE_BASE_GUARD=off falls through entirely —
// the guard does not even read the origin ref.
func TestStaleBaseDeletion_offModeSkips(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "off")
	peerDiff := "@@ -1,0 +1,2 @@\n+peerLineOne()\n+peerLineTwo()\n"
	wtDiff := "@@ -1,2 +1,0 @@\n-peerLineOne()\n-peerLineTwo()\n"
	g := &fakeGit{reply: staleBaseReply(peerDiff, wtDiff)}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != "" || !res.Verified {
		t.Fatalf("off mode should skip the guard and commit, got %+v", res)
	}
	if g.sawSubcommand("merge-base") {
		t.Fatalf("off mode must not run the content check (no merge-base); calls=%v", g.calls)
	}
}

// TestStaleBaseDeletion_genuineDeletionNoFalsePositive: the author deletes a block that is
// present on BOTH sides of the merge-base (it pre-dates the fork) — origin did NOT add it
// since the fork, so peerAdded is empty and the guard does not fire. The commit proceeds.
func TestStaleBaseDeletion_genuineDeletionNoFalsePositive(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "")
	// origin gained nothing in P since the fork (empty peer diff)...
	peerDiff := ""
	// ...the author is deleting an old, pre-fork block (shown removed vs origin), but those
	// lines were never peer-added, so no run qualifies.
	wtDiff := "@@ -5,3 +5,0 @@\n-oldPreForkLineOne()\n-oldPreForkLineTwo()\n-oldPreForkLineThree()\n"
	g := &fakeGit{reply: staleBaseReply(peerDiff, wtDiff)}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != "" {
		t.Fatalf("a genuine pre-fork deletion must not fire the guard, got reason=%q detail=%q", res.Reason, res.Detail)
	}
	if !res.Verified {
		t.Fatalf("a genuine deletion should commit and verify, got %+v", res)
	}
}

// TestStaleBaseDeletion_freshBaseSkips: HEAD already descends from origin/main (merge-base ==
// origin tip) -> the working tree is fresh, nothing to drop, guard skips before any diff.
func TestStaleBaseDeletion_freshBaseSkips(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "")
	rep := onTrunkBase()
	rep["rev-parse --verify --quiet origin/main"] = reply{out: "cafe1234\n", code: 0}
	rep["merge-base HEAD origin/main"] = reply{out: "cafe1234\n", code: 0} // == origin tip
	g := &fakeGit{reply: rep}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != "" || !res.Verified {
		t.Fatalf("a fresh base should skip the guard and commit, got %+v", res)
	}
	if sawContentDiff(g.calls) {
		t.Fatalf("fresh base must not run the content diff; calls=%v", g.calls)
	}
}

func sawContentDiff(calls [][]string) bool {
	for _, c := range calls {
		if len(c) < 4 || c[0] != "diff" || c[1] == "--cached" {
			continue
		}
		if c[1] == "origin/main" || c[2] == "origin/main" {
			return true
		}
	}
	return false
}

// TestStaleBaseDeletion_failOpenWhenNoOriginRef: origin/main does not resolve (fresh clone,
// no remote-tracking ref) -> the guard fails open and the commit proceeds, the same posture
// every existing hook takes on such a clone.
func TestStaleBaseDeletion_failOpenWhenNoOriginRef(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "")
	rep := onTrunkBase()
	rep["rev-parse --verify --quiet origin/main"] = reply{out: "", code: 1} // does not resolve
	g := &fakeGit{reply: rep}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != "" || !res.Verified {
		t.Fatalf("a missing origin ref should fail-open and commit, got %+v", res)
	}
	if g.sawSubcommand("merge-base") {
		t.Fatalf("no origin ref must short-circuit before merge-base; calls=%v", g.calls)
	}
}

// TestStaleBaseDeletion_reformatNoFalsePositive: a peer gofmt-reformats P (changes only
// whitespace). The peer-added lines have a trimmed twin already in the working tree, so no
// contiguous peer-added run is genuinely absent — the guard must NOT fire. This proves the
// comparison is whitespace-insensitive, not a raw per-line intersection.
func TestStaleBaseDeletion_reformatNoFalsePositive(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "")
	// Peer reformatted: added the re-indented form (tab-indented)...
	peerDiff := "@@ -3,2 +3,2 @@\n+\t\tcallWidget(a, b)\n+\t\tcallGadget(c)\n"
	// ...working tree has the same logic, only differently indented (4 spaces). vs origin it
	// shows the old-indent lines removed AND the new-indent lines added; but trimmed, the
	// removed lines equal the peer-added lines -> they are NOT genuinely absent.
	wtDiff := "@@ -3,2 +3,2 @@\n" +
		"-\t\tcallWidget(a, b)\n" +
		"-\t\tcallGadget(c)\n" +
		"+    callWidget(a, b)\n" +
		"+    callGadget(c)\n"
	g := &fakeGit{reply: staleBaseReply(peerDiff, wtDiff)}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != "" {
		t.Fatalf("a pure reformat must not fire the guard, got reason=%q detail=%q", res.Reason, res.Detail)
	}
	if !res.Verified {
		t.Fatalf("a reformat-base commit should verify, got %+v", res)
	}
}

// TestStaleBaseDeletion_belowMinRunSkips: a single peer-added line dropped (below the min-run
// threshold) does not fire — suppresses brace/blank coincidence.
func TestStaleBaseDeletion_belowMinRunSkips(t *testing.T) {
	t.Setenv(staleBaseEnvVar, "")
	peerDiff := "@@ -1,0 +1,1 @@\n+loneSemanticLine()\n"
	wtDiff := "@@ -1,1 +1,0 @@\n-loneSemanticLine()\n"
	g := &fakeGit{reply: staleBaseReply(peerDiff, wtDiff)}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != "" {
		t.Fatalf("a single dropped line is below min-run; guard must not fire, got %q", res.Reason)
	}
}

// TestDroppedPeerRun_unit drives the run-detector directly across its decisive branches:
// a long enough run fires, a sub-threshold run does not, an author-deleted (non-peer) line
// breaks contiguity, and trivial lines neither extend nor break a run.
func TestDroppedPeerRun_unit(t *testing.T) {
	peer := map[string]bool{"a()": true, "b()": true, "c()": true}
	none := map[string]bool{} // working tree carries none of the peer lines

	// three contiguous peer-added removed lines, none re-added -> run of 3.
	wt := "@@\n-a()\n-b()\n-c()\n"
	if got := droppedPeerRun(wt, peer, none, 2); got != 3 {
		t.Fatalf("contiguous peer run: want 3, got %d", got)
	}
	// a non-peer deletion in the middle breaks the run: runs of 1 and 1, neither reaches 2.
	wt = "@@\n-a()\n-zzz()\n-b()\n"
	if got := droppedPeerRun(wt, peer, none, 2); got != 0 {
		t.Fatalf("broken run: want 0 (max sub-run is 1), got %d", got)
	}
	// a trivial removed line (a lone brace) does not break a peer run separated by it.
	wt = "@@\n-a()\n-}\n-b()\n"
	if got := droppedPeerRun(wt, peer, none, 2); got != 2 {
		t.Fatalf("brace-separated peer run: want 2, got %d", got)
	}
	// below threshold.
	wt = "@@\n-a()\n"
	if got := droppedPeerRun(wt, peer, none, 2); got != 0 {
		t.Fatalf("sub-threshold run: want 0, got %d", got)
	}
	// a reformat: the peer lines are removed (old layout) but re-added (new layout) in the
	// SAME diff, so wtPresent carries them -> not dropped, run is 0.
	wt = "@@\n-a()\n-b()\n+a()\n+b()\n"
	present := map[string]bool{"a()": true, "b()": true}
	if got := droppedPeerRun(wt, peer, present, 2); got != 0 {
		t.Fatalf("reformat (re-added under new layout): want 0, got %d", got)
	}
}
