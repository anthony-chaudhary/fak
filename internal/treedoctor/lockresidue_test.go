package treedoctor

// lockresidue_test.go — the residue-matcher half of #5335.
//
// The doctor globbed `*.lock.orphan*`. NOTHING in this repository has ever written that
// name, so the matcher was dead on arrival: `fak tree-doctor --apply` reported a clean
// `.git` while the residue that actually accumulates — `fak-commit.lock.stale-<stamp>`
// files, and a 1.16 MB `index.lock.stale-20260716-044445` that sat there for thirteen
// days — was invisible to it. These tests pin the REAL name, pin that a live-looking
// residue file is spared, and pin that an ACTIVE lock name can never be matched.

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// residueGitDir builds a fake `.git` and returns its path.
func residueGitDir(t *testing.T) (root, gitDir string) {
	t.Helper()
	root = t.TempDir()
	gitDir = filepath.Join(root, ".git")
	if err := os.MkdirAll(gitDir, 0o755); err != nil {
		t.Fatal(err)
	}
	return root, gitDir
}

// TestDiagnoseMatchesTheResidueNameTheFleetActuallyWrites is the core Change-2 witness.
// The names here are transcribed from the live `.git` this ticket was filed against.
func TestDiagnoseMatchesTheResidueNameTheFleetActuallyWrites(t *testing.T) {
	now := time.Date(2026, 7, 29, 4, 0, 0, 0, time.UTC)

	cases := []struct {
		name          string
		file          string
		age           time.Duration
		wantMatched   bool
		wantSweepable bool
		why           string
	}{
		{
			name:          "the 13-day 1.16MB index.lock residue the doctor could not touch",
			file:          "index.lock" + StaleAsideSuffix + "20260716-044445",
			age:           13 * 24 * time.Hour,
			wantMatched:   true,
			wantSweepable: true,
			why:           "the exact file that motivated the ticket",
		},
		{
			name:          "a fak-commit.lock stale-aside file",
			file:          "fak-commit.lock" + StaleAsideSuffix + "20260716-044444",
			age:           13 * 24 * time.Hour,
			wantMatched:   true,
			wantSweepable: true,
		},
		{
			name:          "the older T-stamped variant is still residue",
			file:          "fak-commit.lock" + StaleAsideSuffix + "20260716T023923",
			age:           13 * 24 * time.Hour,
			wantMatched:   true,
			wantSweepable: true,
			why:           "the stamp format drifted; the suffix is the marker, not the layout",
		},
		{
			name:          "a FRESH stale-aside file is reported but spared",
			file:          "packed-refs.lock" + StaleAsideSuffix + "20260729-035959",
			age:           time.Minute,
			wantMatched:   true,
			wantSweepable: false,
			why:           "a rename-aside is instantaneous, so a young one may be an in-flight recovery",
		},
		{
			name:          "a stale-aside file inside the residue floor but past the live window is spared",
			file:          "HEAD.lock" + StaleAsideSuffix + "20260729-034500",
			age:           15 * time.Minute,
			wantMatched:   true,
			wantSweepable: false,
			why:           "DefaultResidueMinAge (30m) is stricter than DefaultLiveWindow (10m) and wins",
		},
		{
			name:          "the legacy orphan spelling still cleans up",
			file:          "HEAD.lock.orphan-recovered-06344",
			age:           time.Hour,
			wantMatched:   true,
			wantSweepable: true,
		},
		{
			name:        "an ACTIVE index.lock is never residue",
			file:        "index.lock",
			age:         13 * 24 * time.Hour,
			wantMatched: false,
			why:         "an active git lock ends in exactly .lock; deleting one races a live transaction",
		},
		{
			name:        "an ACTIVE packed-refs.lock is never residue",
			file:        "packed-refs.lock",
			age:         13 * 24 * time.Hour,
			wantMatched: false,
		},
		{
			name:        "an ACTIVE fak-commit.lock is never residue",
			file:        "fak-commit.lock",
			age:         13 * 24 * time.Hour,
			wantMatched: false,
			why:         "the live commit lock has its own PID-guarded reaper; the glob must not shadow it",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root, gitDir := residueGitDir(t)
			writeAt(t, filepath.Join(gitDir, tc.file), now.Add(-tc.age))

			rep := Diagnose(context.Background(), (&fakeGit{}).run, Options{RepoRoot: root, Now: now})

			var found *LockResidueState
			for i := range rep.LockResidue {
				if filepath.Base(rep.LockResidue[i].Path) == tc.file {
					found = &rep.LockResidue[i]
				}
			}
			if !tc.wantMatched {
				if found != nil {
					t.Fatalf("%s was matched as residue (%s): %+v", tc.file, tc.why, *found)
				}
				return
			}
			if found == nil {
				t.Fatalf("%s was NOT matched as residue (%s); report=%+v", tc.file, tc.why, rep.LockResidue)
			}
			if found.Sweepable != tc.wantSweepable {
				t.Fatalf("%s Sweepable=%v, want %v (%s)", tc.file, found.Sweepable, tc.wantSweepable, tc.why)
			}
			if found.AgeSeconds != int64(tc.age/time.Second) {
				t.Errorf("%s AgeSeconds=%d, want %d", tc.file, found.AgeSeconds, int64(tc.age/time.Second))
			}
		})
	}
}

// TestStaleAsideNameIsMatchedByTheReaper closes the loop the drift opened: whatever a
// writer produces through StaleAsideName is, by construction, something this reaper sees.
// A hand-built name is what made `*.lock.orphan*` outlive every file it was meant to catch.
func TestStaleAsideNameIsMatchedByTheReaper(t *testing.T) {
	now := time.Date(2026, 7, 29, 4, 0, 0, 0, time.UTC)
	root, gitDir := residueGitDir(t)

	for _, lock := range []string{"index.lock", "packed-refs.lock", "fak-commit.lock", "AUTO_MERGE.lock"} {
		aside := StaleAsideName(filepath.Join(gitDir, lock), now.Add(-2*time.Hour))
		if !strings.Contains(filepath.Base(aside), StaleAsideSuffix) {
			t.Fatalf("StaleAsideName(%s) = %s, missing the shared suffix", lock, aside)
		}
		writeAt(t, aside, now.Add(-2*time.Hour))
	}

	rep := Diagnose(context.Background(), (&fakeGit{}).run, Options{RepoRoot: root, Now: now})
	if got := len(rep.SweepableLockResidue()); got != 4 {
		t.Fatalf("SweepableLockResidue() = %d, want 4 — every StaleAsideName result must be reapable: %+v",
			got, rep.LockResidue)
	}
}

// TestResidueThresholdIsTheStricterOfTheTwoKnobs pins the age gate itself: neither knob
// can make the sweep more aggressive than the other permits, and a negative floor is the
// test seam that lets a suite exercise the gate without a 30-minute fixture.
func TestResidueThresholdIsTheStricterOfTheTwoKnobs(t *testing.T) {
	cases := []struct {
		name   string
		window time.Duration
		minAge time.Duration
		want   time.Duration
	}{
		{"zero floor defaults to DefaultResidueMinAge", DefaultLiveWindow, 0, DefaultResidueMinAge},
		{"floor wins when it is the stricter knob", time.Minute, time.Hour, time.Hour},
		{"window wins when IT is the stricter knob", 4 * time.Hour, time.Hour, 4 * time.Hour},
		{"negative floor disables itself, leaving the window", time.Minute, -1, time.Minute},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := residueThreshold(tc.window, tc.minAge); got != tc.want {
				t.Fatalf("residueThreshold(%s, %s) = %s, want %s", tc.window, tc.minAge, got, tc.want)
			}
		})
	}
}

// TestSweepRemovesRealStaleAsideResidueOnlyOnApply pins the end-to-end effect that was
// missing: report-only plans the sweep, --apply performs it, and a fresh sibling survives
// both.
func TestSweepRemovesRealStaleAsideResidueOnlyOnApply(t *testing.T) {
	now := time.Now()
	root, gitDir := residueGitDir(t)
	aged := filepath.Join(gitDir, "index.lock"+StaleAsideSuffix+"20260716-044445")
	fresh := filepath.Join(gitDir, "fak-commit.lock"+StaleAsideSuffix+"20260729-035959")
	// The active-lock sentinel is index.lock: nothing in the doctor ever removes it, so if it
	// disappears here the residue matcher is the only thing that could have taken it.
	active := filepath.Join(gitDir, "index.lock")
	writeAt(t, aged, now.Add(-13*24*time.Hour))
	writeAt(t, fresh, now)
	writeAt(t, active, now.Add(-13*24*time.Hour))

	opts := Options{RepoRoot: root, Now: now}

	_, planned := Sweep(context.Background(), (&fakeGit{}).run, opts, false)
	if !strings.Contains(strings.Join(planned, "\n"), aged) {
		t.Fatalf("report-only did not plan a sweep of %s: %v", aged, planned)
	}
	if _, err := os.Stat(aged); err != nil {
		t.Fatalf("report-only removed the residue: %v", err)
	}

	_, applied := Sweep(context.Background(), (&fakeGit{}).run, opts, true)
	if !strings.Contains(strings.Join(applied, "\n"), aged) {
		t.Fatalf("--apply did not record sweeping %s: %v", aged, applied)
	}
	if _, err := os.Stat(aged); !os.IsNotExist(err) {
		t.Fatalf("--apply did not remove the 13-day residue: %v", err)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Fatalf("--apply removed FRESH residue: %v", err)
	}
	if _, err := os.Stat(active); err != nil {
		t.Fatalf("--apply removed an ACTIVE lock — the one outcome this matcher must never allow: %v", err)
	}
}
