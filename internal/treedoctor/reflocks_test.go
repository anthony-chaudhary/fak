package treedoctor

// reflocks_test.go — the ref-maintenance half of #5335.
//
// The live tree this was written against had a 0-byte `.git/packed-refs.lock` frozen for
// two days, so every `git pack-refs --all` failed with rc=128, so loose refs had grown to
// 7,463 against a 2,337-byte packed-refs. The doctor probed only fak-commit.lock and saw
// none of it.
//
// The gate these tests defend is the CONSERVATIVE one. safecommit/staleindexlock.go
// deliberately excluded ref locks from age-only reaping (a push can legitimately hold one
// for a long time), so a frozen mtime is necessary but never sufficient: the two-sample
// ADVANCING witness must also come back negative. An advancing lock is spared no matter
// how old its mtime reads.

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// seedRefLock writes a ref lock of the given size with the given mtime.
func seedRefLock(t *testing.T, gitDir, name string, mod time.Time, size int) string {
	t.Helper()
	path := filepath.Join(gitDir, name)
	if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, mod, mod); err != nil {
		t.Fatal(err)
	}
	return path
}

// advancingWriter is a Sleep seam that pushes a file's mtime forward on every settle
// window, simulating the LIVE writer the second sample exists to catch — with no goroutine
// and no wall-clock wait. It advances RELATIVE to the current mtime, so repeated probes
// keep seeing motion exactly as they would against a real long-running push.
func advancingWriter(t *testing.T, path string) func(time.Duration) {
	t.Helper()
	return func(time.Duration) {
		fi, err := os.Stat(path)
		if err != nil {
			return // the lock cleared; there is nothing left to advance
		}
		next := fi.ModTime().Add(time.Second)
		if err := os.Chtimes(path, next, next); err != nil {
			t.Fatal(err)
		}
	}
}

// TestDiagnoseRefLockVerdicts is the core Change-3 witness: a frozen orphan is diagnosed,
// an ADVANCING lock is spared however old it looks, a fresh lock is spared, and an absent
// lock is a clean no-op.
func TestDiagnoseRefLockVerdicts(t *testing.T) {
	now := time.Date(2026, 7, 29, 4, 0, 0, 0, time.UTC)

	cases := []struct {
		name       string
		lock       string
		present    bool
		age        time.Duration
		advance    bool
		wantStale  bool
		wantReason string
		why        string
	}{
		{
			name:       "the observed 2-day frozen packed-refs.lock is an orphan",
			lock:       "packed-refs.lock",
			present:    true,
			age:        48 * time.Hour,
			wantStale:  true,
			wantReason: RefLockReapFrozen,
			why:        "0 bytes, mtime frozen 2 days — git pack-refs has been failing rc=128 the whole time",
		},
		{
			name:       "AUTO_MERGE.lock is diagnosed on the same bar",
			lock:       "AUTO_MERGE.lock",
			present:    true,
			age:        48 * time.Hour,
			wantStale:  true,
			wantReason: RefLockReapFrozen,
			why:        "it carried the identical frozen mtime — the same crashed writer left both",
		},
		{
			name:       "an ADVANCING lock is spared no matter how old the first sample reads",
			lock:       "packed-refs.lock",
			present:    true,
			age:        48 * time.Hour,
			advance:    true,
			wantStale:  false,
			wantReason: RefLockKeepAdvancing,
			why:        "a long push legitimately holds a ref lock; a moving mtime proves a live writer",
		},
		{
			name:       "a lock inside the stale window is spared",
			lock:       "packed-refs.lock",
			present:    true,
			age:        5 * time.Minute,
			wantStale:  false,
			wantReason: RefLockKeepFresh,
			why:        "DefaultRefLockStaleAge is an hour — four times the index lock's bar, on purpose",
		},
		{
			name:       "an absent lock is a clean no-op",
			lock:       "packed-refs.lock",
			present:    false,
			wantStale:  false,
			wantReason: RefLockKeepAbsent,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root, gitDir := residueGitDir(t)
			refOpts := RefLockOptions{}
			if tc.present {
				path := seedRefLock(t, gitDir, tc.lock, now.Add(-tc.age), 0)
				if tc.advance {
					refOpts.Sleep = advancingWriter(t, path)
				} else {
					refOpts.Sleep = func(time.Duration) {}
				}
			} else {
				refOpts.Sleep = func(time.Duration) {}
			}

			rep := Diagnose(context.Background(), (&fakeGit{}).run, Options{RepoRoot: root, Now: now, RefLock: refOpts})

			var got *RefLockState
			for i := range rep.RefLocks.Locks {
				if rep.RefLocks.Locks[i].Name == tc.lock {
					got = &rep.RefLocks.Locks[i]
				}
			}
			if got == nil {
				t.Fatalf("%s was not probed at all; RefLockNames=%v", tc.lock, RefLockNames)
			}
			if got.Present != tc.present {
				t.Fatalf("Present=%v, want %v (%+v)", got.Present, tc.present, *got)
			}
			if got.Stale != tc.wantStale {
				t.Fatalf("Stale=%v, want %v (%s): %+v", got.Stale, tc.wantStale, tc.why, *got)
			}
			if got.Reason != tc.wantReason {
				t.Errorf("Reason=%q, want %q (%s)", got.Reason, tc.wantReason, tc.why)
			}
			if tc.advance && !got.Advancing {
				t.Errorf("the settle witness did not see the mtime move: %+v", *got)
			}
			if tc.wantStale && got.Blocks == "" {
				t.Errorf("a stale ref lock must name what it BLOCKS, got %+v", *got)
			}
		})
	}
}

// TestRefLockNeverReapsOnAgeAlone pins the precedent this diagnosis must not override:
// safecommit/staleindexlock.go excluded ref locks from age-only reaping because a
// concurrent push can hold one for a legitimately long window. Age is necessary; the
// advancing witness is what makes it sufficient. Removing EITHER pillar must flip to keep.
func TestRefLockNeverReapsOnAgeAlone(t *testing.T) {
	now := time.Date(2026, 7, 29, 4, 0, 0, 0, time.UTC)

	pillars := map[string]struct {
		age     time.Duration
		advance bool
	}{
		"both pillars hold — reap": {48 * time.Hour, false},
		"remove the freeze":        {time.Minute, false},
		"remove the non-advancing": {48 * time.Hour, true},
	}
	for name, p := range pillars {
		t.Run(name, func(t *testing.T) {
			root, gitDir := residueGitDir(t)
			path := seedRefLock(t, gitDir, "packed-refs.lock", now.Add(-p.age), 0)
			sleep := func(time.Duration) {}
			if p.advance {
				sleep = advancingWriter(t, path)
			}

			rep := Diagnose(context.Background(), (&fakeGit{}).run,
				Options{RepoRoot: root, Now: now, RefLock: RefLockOptions{Sleep: sleep}})

			stale := rep.StaleRefLocks()
			wantReap := name == "both pillars hold — reap"
			if (len(stale) > 0) != wantReap {
				t.Fatalf("%s: stale=%v, wantReap=%v (%+v)", name, len(stale) > 0, wantReap, rep.RefLocks.Locks)
			}
			if rep.RefMaintenanceBlocked() != wantReap {
				t.Errorf("%s: RefMaintenanceBlocked()=%v, want %v", name, rep.RefMaintenanceBlocked(), wantReap)
			}
		})
	}
}

// TestSweepRefLocksReportOnlyVsApply pins that the diagnosis is REPORT-ONLY by default and
// only --apply removes the file — and that an advancing lock survives --apply.
func TestSweepRefLocksReportOnlyVsApply(t *testing.T) {
	now := time.Now()
	root, gitDir := residueGitDir(t)
	orphan := seedRefLock(t, gitDir, "packed-refs.lock", now.Add(-48*time.Hour), 0)
	live := seedRefLock(t, gitDir, "AUTO_MERGE.lock", now.Add(-48*time.Hour), 0)

	// AUTO_MERGE.lock is being written right now; packed-refs.lock is not.
	opts := Options{RepoRoot: root, Now: now, RefLock: RefLockOptions{Sleep: advancingWriter(t, live)}}

	_, planned := Sweep(context.Background(), (&fakeGit{}).run, opts, false)
	joined := strings.Join(planned, "\n")
	if !strings.Contains(joined, "would reap orphaned packed-refs.lock") {
		t.Fatalf("report-only did not plan the ref-lock reap: %v", planned)
	}
	if !strings.Contains(joined, "git pack-refs --all") {
		t.Errorf("the plan must say WHAT the orphan blocks: %v", planned)
	}
	if _, err := os.Stat(orphan); err != nil {
		t.Fatalf("report-only removed the lock: %v", err)
	}

	_, applied := Sweep(context.Background(), (&fakeGit{}).run, opts, true)
	if !strings.Contains(strings.Join(applied, "\n"), "reaped orphaned packed-refs.lock") {
		t.Fatalf("--apply did not record the reap: %v", applied)
	}
	if _, err := os.Stat(orphan); !os.IsNotExist(err) {
		t.Fatalf("--apply did not remove the orphaned packed-refs.lock: %v", err)
	}
	if _, err := os.Stat(live); err != nil {
		t.Fatalf("--apply removed an ADVANCING ref lock — the one outcome the precedent forbids: %v", err)
	}
}

// TestLooseRefPressureIsCountedAndAttributed pins the second half of Change 3: the pile
// that grows while packing is blocked is reported as a count, attributed to the namespaces
// it actually lives in.
func TestLooseRefPressureIsCountedAndAttributed(t *testing.T) {
	now := time.Now()
	root, gitDir := residueGitDir(t)
	if err := os.WriteFile(filepath.Join(gitDir, "packed-refs"), make([]byte, 2337), 0o644); err != nil {
		t.Fatal(err)
	}
	seedRefs := func(ns string, n int) {
		dir := filepath.Join(gitDir, "refs", filepath.FromSlash(ns))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < n; i++ {
			if err := os.WriteFile(filepath.Join(dir, "r"+strconv.Itoa(i)), []byte("deadbeef\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	// Proportional to the observed tree: locks dominate, wip next, a handful of branches.
	seedRefs("fak/locks", 61)
	seedRefs("fak/wip", 13)
	seedRefs("heads", 3)

	opts := Options{RepoRoot: root, Now: now, RefLock: RefLockOptions{
		Sleep:         func(time.Duration) {},
		LoosePressure: 50,
	}}
	rep, actions := Sweep(context.Background(), (&fakeGit{}).run, opts, false)

	loose := rep.RefLocks.LooseRefs
	if loose.Total != 77 {
		t.Fatalf("loose ref total = %d, want 77 (%+v)", loose.Total, loose)
	}
	if loose.ByNamespace["fak/locks"] != 61 || loose.ByNamespace["fak/wip"] != 13 || loose.ByNamespace["heads"] != 3 {
		t.Fatalf("namespace attribution wrong: %+v", loose.ByNamespace)
	}
	if loose.PackedRefsBytes != 2337 {
		t.Errorf("packed-refs size = %d, want 2337", loose.PackedRefsBytes)
	}
	if !loose.Pressure {
		t.Fatalf("77 loose refs against a threshold of 50 must flag pressure: %+v", loose)
	}
	joined := strings.Join(actions, "\n")
	for _, want := range []string{"loose-ref pressure", "77 loose refs", "2337-byte packed-refs", "fak/locks 61"} {
		if !strings.Contains(joined, want) {
			t.Errorf("pressure advisory missing %q: %v", want, actions)
		}
	}
}

// TestLooseRefPressureQuietUnderThreshold pins that a healthy, packed ref store produces no
// advisory at all — the doctor must not cry wolf on every run.
func TestLooseRefPressureQuietUnderThreshold(t *testing.T) {
	now := time.Now()
	root, gitDir := residueGitDir(t)
	heads := filepath.Join(gitDir, "refs", "heads")
	if err := os.MkdirAll(heads, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(heads, "main"), []byte("deadbeef\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	rep, actions := Sweep(context.Background(), (&fakeGit{}).run, Options{
		RepoRoot: root, Now: now, RefLock: RefLockOptions{Sleep: func(time.Duration) {}},
	}, false)

	if rep.RefLocks.LooseRefs.Pressure {
		t.Fatalf("1 loose ref must not flag pressure: %+v", rep.RefLocks.LooseRefs)
	}
	if joined := strings.Join(actions, "\n"); strings.Contains(joined, "loose-ref pressure") {
		t.Fatalf("healthy tree emitted a pressure advisory: %v", actions)
	}
}
