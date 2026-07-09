package main

import (
	"os"
	"testing"
	"time"
)

// prepush_build_slot_test.go — the advisory single-flight (SKIPPED_CONTENDED) rung.
//
// The trunk-build gate's whole-repo archive + go list + importer-cone build is the heaviest
// push-seam step, and in FLEET_BUILD_GUARD=warn (the default) its verdict is advisory. Under a
// concurrent push burst, one gate should build and the rest should skip rather than pile N
// redundant full builds onto the host contention. These tests pin that decision at the seam
// (tryAcquireBuildSlot) and end-to-end (evaluatePrePushBuild with advisory=true), with the
// filesystem marker fully faked so no real disk or clock is touched.

// fakeSlot is an in-memory stand-in for the on-disk single-flight marker.
type fakeSlot struct {
	exists  bool
	modTime time.Time
	created bool
	removed bool
}

// fakeSlotInfo is the minimal os.FileInfo the stat seam returns; only ModTime is consulted.
type fakeSlotInfo struct{ mod time.Time }

func (f fakeSlotInfo) Name() string       { return "fak-prepush-build.slot" }
func (f fakeSlotInfo) Size() int64        { return 0 }
func (f fakeSlotInfo) Mode() os.FileMode  { return 0o644 }
func (f fakeSlotInfo) ModTime() time.Time { return f.mod }
func (f fakeSlotInfo) IsDir() bool        { return false }
func (f fakeSlotInfo) Sys() any           { return nil }

// installFakeSlot wires the slot seams over fs with a frozen clock at now, restoring on cleanup.
func installFakeSlot(t *testing.T, fs *fakeSlot, now time.Time) {
	t.Helper()
	oStat, oCreate, oRemove, oNow := prepushSlotStat, prepushSlotCreate, prepushSlotRemove, prepushSlotNow
	t.Cleanup(func() {
		prepushSlotStat, prepushSlotCreate, prepushSlotRemove, prepushSlotNow = oStat, oCreate, oRemove, oNow
	})
	prepushSlotStat = func(string) (os.FileInfo, error) {
		if !fs.exists {
			return nil, os.ErrNotExist
		}
		return fakeSlotInfo{mod: fs.modTime}, nil
	}
	prepushSlotCreate = func(string) error {
		if fs.exists {
			return os.ErrExist // O_EXCL loser
		}
		fs.exists, fs.created, fs.modTime = true, true, now
		return nil
	}
	prepushSlotRemove = func(string) error {
		fs.exists, fs.removed = false, true
		return nil
	}
	prepushSlotNow = func() time.Time { return now }
}

func TestBuildSlotBlockModeAlwaysRuns(t *testing.T) {
	// A fresh peer marker exists, but block mode (advisory=false) must ignore the marker
	// entirely and always run — a hard-enforced push is never skipped.
	fs := &fakeSlot{exists: true, modTime: time.Unix(1_700_000_000, 0)}
	installFakeSlot(t, fs, time.Unix(1_700_000_001, 0))
	run, release := tryAcquireBuildSlot(false)
	if !run {
		t.Fatal("block mode must always run, even with a fresh peer marker")
	}
	if fs.created || fs.removed {
		t.Fatalf("block mode must not touch the marker: created=%v removed=%v", fs.created, fs.removed)
	}
	release() // must be a safe no-op
}

func TestBuildSlotAdvisoryUncontendedAcquiresAndReleases(t *testing.T) {
	fs := &fakeSlot{}
	installFakeSlot(t, fs, time.Unix(1_700_000_000, 0))
	run, release := tryAcquireBuildSlot(true)
	if !run || !fs.created {
		t.Fatalf("advisory with no peer marker must acquire and run: run=%v created=%v", run, fs.created)
	}
	release()
	if !fs.removed {
		t.Fatal("release must remove the marker it created")
	}
}

func TestBuildSlotAdvisoryContendedSkips(t *testing.T) {
	// A peer holds a FRESH marker → the O_EXCL create loses → skip.
	now := time.Unix(1_700_000_000, 0)
	fs := &fakeSlot{exists: true, modTime: now.Add(-5 * time.Second)}
	installFakeSlot(t, fs, now)
	run, release := tryAcquireBuildSlot(true)
	if run {
		t.Fatal("advisory with a fresh peer marker must skip (run=false)")
	}
	if fs.removed {
		t.Fatal("a skip must not remove the peer's live marker")
	}
	release() // no-op
}

func TestBuildSlotAdvisoryStealsStaleMarker(t *testing.T) {
	// A marker older than prepushBuildSlotStale is a crashed gate → steal it, then run.
	now := time.Unix(1_700_000_000, 0)
	fs := &fakeSlot{exists: true, modTime: now.Add(-(prepushBuildSlotStale + time.Minute))}
	installFakeSlot(t, fs, now)
	run, release := tryAcquireBuildSlot(true)
	if !run {
		t.Fatal("a stale marker must be stolen so the check is never wedged off")
	}
	if !fs.created {
		t.Fatal("stealing must re-create a fresh marker for this gate")
	}
	release()
}

func TestEvaluatePrePushBuildSkipsUnderContention(t *testing.T) {
	// End-to-end: a real Go delta + advisory + a fresh peer marker must return SKIPPED_CONTENDED
	// (exit 0) WITHOUT running the expensive archive/list/build seams.
	setupHappyPrepushSeams(t)
	prepushExtractTip = func(string, string) (string, error) {
		t.Fatal("extractTip must not run when the build is skipped under contention")
		return "", nil
	}
	prepushBuild = func(string, []string) (string, bool) {
		t.Fatal("build must not run when the build is skipped under contention")
		return "", true
	}
	now := time.Unix(1_700_000_000, 0)
	fs := &fakeSlot{exists: true, modTime: now.Add(-2 * time.Second)}
	installFakeSlot(t, fs, now)

	res, code := evaluatePrePushBuild("/repo", "", time.Minute, true)
	if code != 0 || res.Verdict != "SKIPPED_CONTENDED" || !res.OK {
		t.Fatalf("want SKIPPED_CONTENDED/exit0/ok, got verdict=%s code=%d ok=%v", res.Verdict, code, res.OK)
	}
}

func TestEvaluatePrePushBuildAdvisoryUncontendedStillBuilds(t *testing.T) {
	// Advisory but NO peer in flight → the gate builds normally (verdict OK) and releases its
	// marker, so a lone push is never degraded by the single-flight path.
	setupHappyPrepushSeams(t)
	fs := &fakeSlot{}
	installFakeSlot(t, fs, time.Unix(1_700_000_000, 0))
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, true)
	if code != 0 || res.Verdict != "OK" || !res.OK {
		t.Fatalf("advisory uncontended must build normally: verdict=%s code=%d ok=%v", res.Verdict, code, res.OK)
	}
	if !fs.created || !fs.removed {
		t.Fatalf("the gate must acquire then release its marker: created=%v removed=%v", fs.created, fs.removed)
	}
}

func TestEvaluatePrePushBuildNoopNeverContends(t *testing.T) {
	// A docs-only (no Go delta) push must short-circuit as NOOP before ever touching the slot,
	// so a cheap push never blocks a concurrent real build.
	setupHappyPrepushSeams(t)
	prepushChangedFiles = func(string, string) ([]string, error) { return nil, nil }
	fs := &fakeSlot{}
	installFakeSlot(t, fs, time.Unix(1_700_000_000, 0))
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, true)
	if code != 0 || res.Verdict != "NOOP" {
		t.Fatalf("no-Go-delta push must be NOOP: verdict=%s code=%d", res.Verdict, code)
	}
	if fs.created {
		t.Fatal("a NOOP push must not acquire the build slot")
	}
}
