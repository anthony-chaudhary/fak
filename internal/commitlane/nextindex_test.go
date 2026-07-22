package commitlane

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// TestDecideNextIndexReclaim pins the evidence bar for reaping orphaned
// .git/next-index-<pid>.lock residue (#5338): N stale files whose owner pids are dead
// and with no live writer are reaped, while a young one, a live-owner one, a live-writer
// report and an un-probed report are all KEPT.
func TestDecideNextIndexReclaim(t *testing.T) {
	stale := func(path string, pid int) NextIndexLock {
		return NextIndexLock{Path: path, PID: pid, StaleHint: true}
	}

	t.Run("N stale dead-owner files with no live writer are all reaped", func(t *testing.T) {
		rep := Report{
			ProcessProbe: "ok",
			NextIndexLocks: []NextIndexLock{
				stale("/g/next-index-11228.lock", 11228),
				stale("/g/next-index-12640.lock", 12640),
				stale("/g/next-index-14116.lock", 14116),
			},
		}
		got := DecideNextIndexReclaim(rep)
		if len(got) != 3 {
			t.Fatalf("decisions = %d, want 3 (%+v)", len(got), got)
		}
		for _, d := range got {
			if !d.Reap || d.Reason != ReclaimReapStale {
				t.Errorf("%s: got reap=%v reason=%q, want reap with %q", d.Path, d.Reap, d.Reason, ReclaimReapStale)
			}
		}
		if got[0].PID != 11228 {
			t.Errorf("PID = %d, want the pid parsed from the filename", got[0].PID)
		}
	})

	t.Run("a young file is kept while its stale siblings are reaped", func(t *testing.T) {
		rep := Report{
			ProcessProbe: "ok",
			NextIndexLocks: []NextIndexLock{
				stale("/g/next-index-100.lock", 100),
				{Path: "/g/next-index-200.lock", PID: 200, StaleHint: false},
			},
		}
		got := DecideNextIndexReclaim(rep)
		if !got[0].Reap {
			t.Errorf("stale sibling should reap, got %+v", got[0])
		}
		if got[1].Reap || got[1].Reason != ReclaimKeepFresh {
			t.Errorf("young file = %+v, want keep_fresh", got[1])
		}
	})

	t.Run("a stale file whose named owner pid is still alive is kept", func(t *testing.T) {
		rep := Report{
			ProcessProbe: "ok",
			NextIndexLocks: []NextIndexLock{
				{Path: "/g/next-index-300.lock", PID: 300, StaleHint: true, OwnerAlive: true},
			},
		}
		got := DecideNextIndexReclaim(rep)
		if got[0].Reap || got[0].Reason != ReclaimKeepLiveOwner {
			t.Errorf("live owner = %+v, want keep_live_owner", got[0])
		}
	})

	t.Run("a live writer keeps every file even when all are stale", func(t *testing.T) {
		rep := Report{
			ProcessProbe:   "ok",
			LiveWriters:    []ProcessFact{{PID: 4242, Match: "git_writer"}},
			NextIndexLocks: []NextIndexLock{stale("/g/next-index-400.lock", 400)},
		}
		got := DecideNextIndexReclaim(rep)
		if got[0].Reap || got[0].Reason != ReclaimKeepLiveWriter {
			t.Errorf("live writer = %+v, want keep_live_writer", got[0])
		}
	})

	t.Run("a failed process probe fails closed", func(t *testing.T) {
		for _, probe := range []string{"error", "not_run"} {
			rep := Report{
				ProcessProbe:   probe,
				NextIndexLocks: []NextIndexLock{stale("/g/next-index-500.lock", 500)},
			}
			got := DecideNextIndexReclaim(rep)
			if got[0].Reap || got[0].Reason != ReclaimKeepProbeFailed {
				t.Errorf("probe %q = %+v, want keep_probe_failed", probe, got[0])
			}
		}
	})

	t.Run("no residue yields no decisions", func(t *testing.T) {
		if got := DecideNextIndexReclaim(Report{ProcessProbe: "ok"}); got != nil {
			t.Errorf("empty residue = %+v, want nil", got)
		}
	})
}

// TestDecideNextIndexReclaimReapIsSoleFallThrough guards the safety-critical property
// that a reap is reachable ONLY through the full-evidence default branch: flipping any
// single guard away from its reapable value must flip the verdict to keep.
func TestDecideNextIndexReclaimReapIsSoleFallThrough(t *testing.T) {
	reapable := Report{
		ProcessProbe:   "ok",
		NextIndexLocks: []NextIndexLock{{Path: "/g/next-index-7.lock", PID: 7, StaleHint: true}},
	}
	if d := DecideNextIndexReclaim(reapable); !d[0].Reap {
		t.Fatalf("baseline should reap, got %+v", d[0])
	}
	mutators := map[string]func(*Report){
		"probe failed": func(r *Report) { r.ProcessProbe = "error" },
		"live writer":  func(r *Report) { r.LiveWriters = []ProcessFact{{PID: 1}} },
		"live owner":   func(r *Report) { r.NextIndexLocks[0].OwnerAlive = true },
		"not stale":    func(r *Report) { r.NextIndexLocks[0].StaleHint = false },
	}
	for name, mut := range mutators {
		r := reapable
		// Deep-copy the slice so one mutation cannot leak into the next case.
		r.NextIndexLocks = []NextIndexLock{reapable.NextIndexLocks[0]}
		mut(&r)
		if d := DecideNextIndexReclaim(r); d[0].Reap {
			t.Errorf("%s: expected keep, but decision reaped (%+v)", name, d[0])
		}
	}
}

// TestStatusScansNextIndexResidue exercises the observation half end to end against a
// real temp .git directory: the scan must age each file, parse the owner pid out of the
// filename, and consult liveness — so a stale dead-owner file reaps and a fresh one does
// not. It also proves the scan never removes anything.
func TestStatusScansNextIndexResidue(t *testing.T) {
	root, gitDir := testRepoPaths(t)
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	old, young := now.Add(-2*time.Hour), now.Add(-1*time.Minute)

	files := map[string]time.Time{
		filepath.Join(gitDir, "next-index-11228.lock"): old,
		filepath.Join(gitDir, "next-index-12640.lock"): old,
		filepath.Join(gitDir, "next-index-99999.lock"): young,
	}
	paths := make([]string, 0, len(files))
	for p := range files {
		paths = append(paths, p)
	}

	rep, err := Status(context.Background(), Options{
		Runner:    fakeRepoRunner(root, gitDir),
		ProbeLock: func(path string) safecommit.LockProbe { return safecommit.LockProbe{Path: path} },
		Glob:      func(string) ([]string, error) { return paths, nil },
		Stat: func(path string) FileFact {
			mt, ok := files[path]
			if !ok {
				return FileFact{} // index.lock and friends are absent
			}
			return FileFact{Exists: true, ModTime: mt, Size: 4}
		},
		PIDAlive:    func(int) bool { return false },
		ProcessList: func(context.Context) ([]Process, error) { return nil, nil },
		Now:         func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.NextIndexLocks) != 3 {
		t.Fatalf("scanned %d next-index files, want 3 (%+v)", len(rep.NextIndexLocks), rep.NextIndexLocks)
	}
	// Sorted by path, so 11228 < 12640 < 99999.
	if rep.NextIndexLocks[0].PID != 11228 || rep.NextIndexLocks[2].PID != 99999 {
		t.Fatalf("pids not parsed from filenames: %+v", rep.NextIndexLocks)
	}
	if rep.NextIndexLocks[0].AgeSeconds != int64((2*time.Hour)/time.Second) {
		t.Errorf("age = %ds, want 7200", rep.NextIndexLocks[0].AgeSeconds)
	}
	if !rep.NextIndexLocks[0].StaleHint || rep.NextIndexLocks[2].StaleHint {
		t.Errorf("stale hints wrong: %+v", rep.NextIndexLocks)
	}

	decisions := DecideNextIndexReclaim(rep)
	reaped := 0
	for _, d := range decisions {
		if d.Reap {
			reaped++
		}
	}
	if reaped != 2 {
		t.Fatalf("reaped %d, want the 2 stale dead-owner files only (%+v)", reaped, decisions)
	}
	if decisions[2].Reap || decisions[2].Reason != ReclaimKeepFresh {
		t.Errorf("young file = %+v, want keep_fresh", decisions[2])
	}

	// Residue is not a blockage: an otherwise-clear lane stays clear.
	if !rep.OK || rep.Verdict != VerdictClear {
		t.Errorf("residue alone should not change the verdict, got %s/%v", rep.Verdict, rep.OK)
	}
}

// TestStatusNextIndexScanSurfacesGlobErrors proves a broken scan is reported rather than
// silently rendering as "no residue" — the same fail-closed discipline the process probe
// uses, so a reap is never authorized off a probe that did not actually run.
func TestStatusNextIndexScanSurfacesGlobErrors(t *testing.T) {
	root, gitDir := testRepoPaths(t)
	rep, err := Status(context.Background(), Options{
		Runner:      fakeRepoRunner(root, gitDir),
		ProbeLock:   func(path string) safecommit.LockProbe { return safecommit.LockProbe{Path: path} },
		Glob:        func(string) ([]string, error) { return nil, errProcessProbe },
		Stat:        func(string) FileFact { return FileFact{} },
		ProcessList: func(context.Context) ([]Process, error) { return nil, nil },
		Now:         fixedNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.NextIndexLocks) != 0 {
		t.Fatalf("a failed scan must report no residue, got %+v", rep.NextIndexLocks)
	}
	if len(rep.Errors) != 1 {
		t.Fatalf("scan error not surfaced: %+v", rep.Errors)
	}
}

// TestStatusNextIndexSkipsRacedAwayFiles proves a file that vanishes between the glob
// and the stat is dropped rather than reported as reapable residue with a zero age.
func TestStatusNextIndexSkipsRacedAwayFiles(t *testing.T) {
	root, gitDir := testRepoPaths(t)
	gone := filepath.Join(gitDir, "next-index-4242.lock")
	rep, err := Status(context.Background(), Options{
		Runner:      fakeRepoRunner(root, gitDir),
		ProbeLock:   func(path string) safecommit.LockProbe { return safecommit.LockProbe{Path: path} },
		Glob:        func(string) ([]string, error) { return []string{gone}, nil },
		Stat:        func(string) FileFact { return FileFact{} }, // already gone
		ProcessList: func(context.Context) ([]Process, error) { return nil, nil },
		Now:         fixedNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.NextIndexLocks) != 0 {
		t.Fatalf("raced-away file should be dropped, got %+v", rep.NextIndexLocks)
	}
}
