package main

import "testing"

// BenchmarkRunsSnapshotOnePassAllProjections measures the coalesced path #3593 ships:
// build ONE runsSnapshot and serve every view (held lanes, live scopes/issues, cooldown
// set + rows, tree-collision, attempt cap) as projections over the captured state. This
// is the discovery cost a dispatch tick actually pays per loop.
func BenchmarkRunsSnapshotOnePassAllProjections(b *testing.B) {
	runsDir, now := snapshotFixture(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		snap := scanRunsSnapshot(runsDir, now)
		_ = snap.liveLanes()
		_ = snap.liveScopes()
		_ = snap.liveIssueDetails()
		_ = snap.liveIssues()
		_ = snap.recentlyAttempted(120)
		_ = snap.cooldownRowMaps(120)
		_, _ = snap.treeCollision([]string{"cmd/**"})
		_ = snap.attemptExhausted(3)
	}
}

// BenchmarkRunsPerViewLegacyScans measures the pre-#3593 path: each view re-globbed and
// re-statted the same sidecars through a separate free function, so one tick walked the
// runs directory several times over identical bytes. Comparing the two benchmarks makes
// the O(N)x(views) -> O(N) reduction the refactor claims measurable, not just asserted.
func BenchmarkRunsPerViewLegacyScans(b *testing.B) {
	runsDir, now := snapshotFixture(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = liveResolutionLanes(runsDir)
		_ = liveResolutionScopes(runsDir)
		_ = liveResolutionIssueDetails(runsDir)
		_ = liveResolutionIssues(runsDir)
		_ = recentlyAttemptedIssuesAt(runsDir, 120, now)
		_ = cooldownIssueRowsAt(runsDir, 120, now)
		_, _ = liveResolutionTreeCollision(runsDir, []string{"cmd/**"})
		_ = attemptExhaustedIssues(runsDir, 3)
	}
}

// TestRunsSnapshotOnePassGlobsFarFewerThanPerView is the benchmark's assertion twin: it
// pins the same claim as a hard inequality the gate enforces (a benchmark alone never
// fails CI). Serving all eight views off one snapshot must glob the runs directory a
// small constant number of times, while the eight legacy per-view calls glob it at least
// once each -- so the coalesced path does strictly, and substantially, less discovery I/O.
func TestRunsSnapshotOnePassGlobsFarFewerThanPerView(t *testing.T) {
	runsDir, now := snapshotFixture(t)

	var globs int
	origGlob := fsGlob
	t.Cleanup(func() { fsGlob = origGlob })
	fsGlob = func(p string) ([]string, error) { globs++; return origGlob(p) }

	globs = 0
	snap := scanRunsSnapshot(runsDir, now)
	_ = snap.liveLanes()
	_ = snap.liveScopes()
	_ = snap.liveIssueDetails()
	_ = snap.liveIssues()
	_ = snap.recentlyAttempted(120)
	_ = snap.cooldownRowMaps(120)
	_, _ = snap.treeCollision([]string{"cmd/**"})
	_ = snap.attemptExhausted(3)
	onePass := globs

	globs = 0
	_ = liveResolutionLanes(runsDir)
	_ = liveResolutionScopes(runsDir)
	_ = liveResolutionIssueDetails(runsDir)
	_ = liveResolutionIssues(runsDir)
	_ = recentlyAttemptedIssuesAt(runsDir, 120, now)
	_ = cooldownIssueRowsAt(runsDir, 120, now)
	_, _ = liveResolutionTreeCollision(runsDir, []string{"cmd/**"})
	_ = attemptExhaustedIssues(runsDir, 3)
	perView := globs

	if onePass != 2 {
		t.Fatalf("one-pass snapshot globbed %d times serving 8 views, want exactly 2", onePass)
	}
	if perView <= onePass*3 {
		t.Fatalf("per-view scans globbed %d times, want far more than the one-pass %d (>= 8, one per view)", perView, onePass)
	}
}
