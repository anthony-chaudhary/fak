package main

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// snapshotFixture writes a runs directory exercising every branch the snapshot must
// classify: a live worker with an in-flight lease tree, a dead-pid worker, a live-pid
// banner-noop worker (a recycled pid pinning a lane it must NOT hold, #1398), a
// poison issue with several recorded attempts, and a witness-only cooled slot. It
// returns the runs dir and the fixed "now" the cooldown projections are measured at.
func snapshotFixture(t *testing.T) (string, time.Time) {
	t.Helper()
	runsDir := t.TempDir()
	now := time.Unix(1700000000, 0).UTC()

	writeLog := func(stem, body string, pid int, mod time.Time) {
		t.Helper()
		logPath := filepath.Join(runsDir, stem+".log")
		if err := os.WriteFile(logPath, []byte(body), 0o644); err != nil {
			t.Fatalf("write %s: %v", logPath, err)
		}
		if err := os.Chtimes(logPath, mod, mod); err != nil {
			t.Fatalf("chtime %s: %v", logPath, err)
		}
		if pid != 0 {
			if err := os.WriteFile(filepath.Join(runsDir, stem+".pid"), []byte(fmt.Sprint(pid)), 0o644); err != nil {
				t.Fatalf("write pid for %s: %v", stem, err)
			}
		}
	}

	// #100: a genuinely live worker (this test's own pid is alive) holding lane cmd with
	// an in-flight lease tree and an explicit lease id.
	live := "resolve-100-20260701-010000"
	writeLog(live, "# fak-spawn issue=100 lane=cmd\nreal streamed work\n", os.Getpid(), now.Add(-5*time.Minute))
	if err := os.WriteFile(filepath.Join(runsDir, live+dispatchLeaseTreeSidecarSuffix), []byte(`["cmd/**"]`), 0o644); err != nil {
		t.Fatalf("write lease tree: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runsDir, live+dispatchLeaseIDSidecarSuffix), []byte("resolve-cmd-100"), 0o644); err != nil {
		t.Fatalf("write lease id: %v", err)
	}

	// #200: a dead-pid worker -- not live, holds no lane.
	writeLog("resolve-200-20260701-010000", "# fak-spawn issue=200 lane=gateway\nreal work\n", deadDispatchPID, now.Add(-5*time.Minute))

	// #300: a live pid but a terminal banner no-op (#1275) -- its lane docs MUST be
	// dropped even though the pid passes the liveness gate (#1398).
	writeLog("resolve-300-20260701-010000", "# fak-spawn issue=300 lane=docs\n> build · glm-4.6\n", os.Getpid(), now.Add(-5*time.Minute))

	// #400: a poison issue with three distinct recorded attempts (all dead) -- the
	// attempt-budget cap counts these regardless of liveness.
	for _, stamp := range []string{"20260701-000100", "20260701-000200", "20260701-000300"} {
		writeLog(fmt.Sprintf("resolve-400-%s", stamp), "# fak-spawn issue=400 lane=cmd\nreal work\n", deadDispatchPID, now.Add(-90*time.Minute))
	}

	// #500: only a durable .witness survives (its .log was swept), touched 30 min ago.
	witPath := filepath.Join(runsDir, "resolve-500-20260701-003000"+dispatchtick.WitnessSidecarSuffix)
	if err := os.WriteFile(witPath, []byte(`{"claim":"CLAIM_NO_COMMIT"}`), 0o644); err != nil {
		t.Fatalf("write witness: %v", err)
	}
	witMod := now.Add(-30 * time.Minute)
	if err := os.Chtimes(witPath, witMod, witMod); err != nil {
		t.Fatalf("chtime witness: %v", err)
	}

	return runsDir, now
}

// TestScanRunsSnapshotSingleScanServesEveryProjection is the #3593 acceptance witness:
// one scanRunsSnapshot walks the runs directory in a single pass (exactly two globs --
// resolve-*.log and resolve-*.witness), and EVERY downstream view (live lanes, live
// scopes, issue details, live issues, tree-collision, cooldown rows/set, attempt cap)
// is a pure projection that touches the disk ZERO further times. It then contrasts this
// with the legacy per-view free functions, each of which re-scans -- proving the tick's
// discovery cost dropped from O(N)x(views) to O(N).
func TestScanRunsSnapshotSingleScanServesEveryProjection(t *testing.T) {
	runsDir, now := snapshotFixture(t)

	var globs, stats, reads int
	origGlob, origStat, origRead := fsGlob, fsStat, fsReadFile
	t.Cleanup(func() { fsGlob, fsStat, fsReadFile = origGlob, origStat, origRead })
	fsGlob = func(p string) ([]string, error) { globs++; return origGlob(p) }
	fsStat = func(p string) (os.FileInfo, error) { stats++; return origStat(p) }
	fsReadFile = func(p string) ([]byte, error) { reads++; return origRead(p) }

	// One pass.
	snap := scanRunsSnapshot(runsDir, now)
	scanGlobs, scanStats, scanReads := globs, stats, reads

	if scanGlobs != 2 {
		t.Fatalf("scan globbed %d times, want exactly 2 (resolve-*.log + resolve-*.witness)", scanGlobs)
	}

	// Every projection reads only captured state.
	_ = snap.liveScopes()
	_ = snap.liveLanes()
	_ = snap.liveIssueDetails()
	_ = snap.liveIssues()
	_, _ = snap.treeCollision([]string{"cmd/**"})
	_ = snap.cooldownRows(120)
	_ = snap.cooldownRowMaps(120)
	_ = snap.recentlyAttempted(120)
	_ = snap.attemptExhausted(3)

	if globs != scanGlobs || stats != scanStats || reads != scanReads {
		t.Fatalf("projections did I/O: glob %d->%d, stat %d->%d, read %d->%d; want zero delta after the single scan",
			scanGlobs, globs, scanStats, stats, scanReads, reads)
	}

	// Contrast: the legacy per-view free functions each re-scan the runs directory, so
	// four views cost four passes (>= 8 globs) versus the single snapshot's 2.
	globs, stats, reads = 0, 0, 0
	_ = liveResolutionScopes(runsDir)
	_ = liveResolutionLanes(runsDir)
	_ = cooldownIssueRowsAt(runsDir, 120, now)
	_ = attemptExhaustedIssues(runsDir, 3)
	if globs <= scanGlobs {
		t.Fatalf("four legacy per-view scans globbed %d times, want strictly more than the single snapshot's %d", globs, scanGlobs)
	}
}

// TestRunsSnapshotProjectionsMatchLegacyScans pins that the snapshot projections are
// byte-identical to the per-loop scans they replace: same live scopes, same held lanes
// (with the banner-noop lane dropped), same live issue set, same tree-collision verdict,
// same cooldown rows, and same attempt-cap set -- both against explicit expectations and
// against the legacy free functions the rest of the tick surface still calls.
func TestRunsSnapshotProjectionsMatchLegacyScans(t *testing.T) {
	runsDir, now := snapshotFixture(t)
	snap := scanRunsSnapshot(runsDir, now)

	// Live scopes: only #100 is live-with-issue; its lease tree/id/lane carry through.
	scopes := snap.liveScopes()
	if len(scopes) != 1 {
		t.Fatalf("live scopes = %+v, want exactly one (#100)", scopes)
	}
	got := scopes[0]
	if got.Issue != 100 || got.Lane != "cmd" || got.LeaseID != "resolve-cmd-100" ||
		got.PID != os.Getpid() || !reflect.DeepEqual(got.Tree, []string{"cmd/**"}) {
		t.Fatalf("live scope = %+v, want issue=100 lane=cmd leaseID=resolve-cmd-100 pid=%d tree=[cmd/**]", got, os.Getpid())
	}

	// Held lanes: cmd is held; docs is NOT (its only worker is a dead banner no-op, #1398).
	lanes := snap.liveLanes()
	if !lanes["cmd"] || lanes["docs"] || lanes["gateway"] || len(lanes) != 1 {
		t.Fatalf("live lanes = %#v, want {cmd:true} only (docs banner-noop dropped)", lanes)
	}

	// Live issue set: only #100.
	if issues := snap.liveIssues(); !issues[100] || len(issues) != 1 {
		t.Fatalf("live issues = %#v, want {100}", issues)
	}

	// Tree-collision: an overlapping request hits #100; a disjoint one does not.
	if live, ok := snap.treeCollision([]string{"cmd/**"}); !ok || live.Issue != 100 {
		t.Fatalf("tree collision for cmd/** = (%+v, %v), want #100 hit", live, ok)
	}
	if _, ok := snap.treeCollision([]string{"docs/**"}); ok {
		t.Fatalf("tree collision for docs/** = hit, want miss")
	}

	// Attempt cap at budget 3: #400 (three attempts) is exhausted, nothing else.
	if ex := snap.attemptExhausted(3); !ex[400] || len(ex) != 1 {
		t.Fatalf("attempt-exhausted = %#v, want {400}", ex)
	}
	if ex := snap.attemptExhausted(4); len(ex) != 0 {
		t.Fatalf("attempt-exhausted at budget 4 = %#v, want empty", ex)
	}

	// Cooldown: #100/#300 (5 min) and #500 (30 min, from its witness) are cooling within
	// the 120-min window; #400 (90 min) is too but still inside; assert #500 cools from
	// the witness and the set matches the legacy scan.
	cooled := snap.recentlyAttempted(120)
	if !cooled[500] {
		t.Fatalf("recently attempted = %#v, want #500 cooling from its .witness", cooled)
	}

	// Parity with the legacy free functions the rest of the tick still calls.
	if want := liveResolutionScopes(runsDir); !reflect.DeepEqual(scopes, want) {
		t.Fatalf("liveScopes projection = %+v, legacy = %+v", scopes, want)
	}
	if want := liveResolutionLanes(runsDir); !reflect.DeepEqual(lanes, want) {
		t.Fatalf("liveLanes projection = %#v, legacy = %#v", lanes, want)
	}
	if want := liveResolutionIssueDetails(runsDir); !reflect.DeepEqual(snap.liveIssueDetails(), want) {
		t.Fatalf("liveIssueDetails projection mismatch vs legacy")
	}
	if want := recentlyAttemptedIssuesAt(runsDir, 120, now); !reflect.DeepEqual(cooled, want) {
		t.Fatalf("recentlyAttempted projection = %#v, legacy = %#v", cooled, want)
	}
	if want := cooldownIssueRowsAt(runsDir, 120, now); !reflect.DeepEqual(snap.cooldownRows(120), want) {
		t.Fatalf("cooldownRows projection mismatch vs legacy")
	}
	if want := attemptExhaustedIssues(runsDir, 3); !reflect.DeepEqual(snap.attemptExhausted(3), want) {
		t.Fatalf("attemptExhausted projection mismatch vs legacy")
	}
}

// TestRunsSnapshotCooldownRowsAreSortedByIssue guards the deterministic scan order the
// JSON cooldown payload depends on: rows come out sorted by issue regardless of glob
// order, matching the legacy cooldownIssueRowsAt sort.
func TestRunsSnapshotCooldownRowsAreSortedByIssue(t *testing.T) {
	runsDir, now := snapshotFixture(t)
	rows := scanRunsSnapshot(runsDir, now).cooldownRows(120)
	issues := make([]int, len(rows))
	for i, r := range rows {
		issues[i] = r.Issue
	}
	if !sort.IntsAreSorted(issues) {
		t.Fatalf("cooldown rows issues = %v, want ascending", issues)
	}
}
