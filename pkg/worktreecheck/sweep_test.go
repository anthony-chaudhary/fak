package worktreecheck

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/testgit"
)

func TestMain(m *testing.M) { os.Exit(testgit.RunGitIsolated(m)) }

var sweepNow = time.Date(2026, 10, 11, 12, 0, 0, 0, time.UTC)

func ownerName(prefix string, pid int) string {
	return prefix + strconv.Itoa(pid) + "-" + strconv.FormatInt(1234, 36) + "-x1"
}

func mkdirAged(t *testing.T, parent, name string, age time.Duration) string {
	t.Helper()
	p := filepath.Join(parent, name)
	if err := os.Mkdir(p, 0o755); err != nil {
		t.Fatal(err)
	}
	mt := sweepNow.Add(-age)
	if err := os.Chtimes(p, mt, mt); err != nil {
		t.Fatal(err)
	}
	return p
}

func gitRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	if out, err := exec.Command("git", "init", "-q", root).CombinedOutput(); err != nil {
		t.Skipf("git init unavailable: %v %s", err, out)
	}
	return root
}

func opts(apply bool, limit int, live ...int) SweepOptions {
	return SweepOptions{
		Now:          sweepNow,
		LegacyMaxAge: time.Hour,
		Apply:        apply,
		Limit:        limit,
		ProcessAlive: func(pid int) bool { return slices.Contains(live, pid) },
		ProcessStart: func(int) (time.Time, bool) { return time.Time{}, false },
	}
}

func byBase(r SweepReport) map[string]SweepItem {
	m := map[string]SweepItem{}
	for _, it := range r.Candidates {
		m[filepath.Base(it.Path)] = it
	}
	return m
}

// fak-test:runtime fast est=20ms
func TestSweepDryRunClassifiesWithoutRemoving(t *testing.T) {
	parent := t.TempDir()
	dead := ownerName(".fak-cand-validate-", 101)
	live := ownerName(".fak-cand-validate-", 202)
	witness := ownerName("fak-exec-witness-", 303)
	mkdirAged(t, parent, dead, time.Minute)
	mkdirAged(t, parent, live, time.Minute)
	mkdirAged(t, parent, witness, time.Minute)
	mkdirAged(t, parent, ".fak-cand-validate-legacyold", 2*time.Hour)
	mkdirAged(t, parent, ".fak-cand-validate-legacynew", time.Minute)
	mkdirAged(t, parent, "unrelated-dir", 48*time.Hour)

	r := Sweep(gitRepo(t), parent, opts(false, 0, 202))

	if r.Mode != "dry-run" || r.Reaped != 0 || r.WouldReap != 3 {
		t.Fatalf("mode=%q reaped=%d would_reap=%d, want dry-run/0/3", r.Mode, r.Reaped, r.WouldReap)
	}
	if s, rp, k := Counts(r); s != 5 || rp != 0 || k != 5 {
		t.Fatalf("Counts(dry-run) = %d,%d,%d, want 5,0,5", s, rp, k)
	}
	items := byBase(r)
	if _, ok := items["unrelated-dir"]; ok {
		t.Fatal("non-candidate directory was classified")
	}
	cases := []struct {
		name     string
		eligible bool
		reason   string
	}{
		{dead, true, "owner_exited"},
		{live, false, "owner_process_live"},
		{witness, true, "owner_exited"},
		{".fak-cand-validate-legacyold", true, ""},
		{".fak-cand-validate-legacynew", false, "legacy_too_young"},
	}
	for _, c := range cases {
		it, ok := items[c.name]
		if !ok {
			t.Fatalf("%s missing from report", c.name)
		}
		if it.Eligible != c.eligible || it.Removed {
			t.Errorf("%s eligible=%v removed=%v, want eligible=%v removed=false", c.name, it.Eligible, it.Removed, c.eligible)
		}
		if c.reason != "" && it.Reason != c.reason {
			t.Errorf("%s reason=%q, want %q", c.name, it.Reason, c.reason)
		}
		if _, err := os.Stat(filepath.Join(parent, c.name)); err != nil {
			t.Errorf("dry run touched %s: %v", c.name, err)
		}
	}
}

// fak-test:runtime fast est=50ms
func TestSweepApplyRemovesOnlyEligibleWithinLimit(t *testing.T) {
	parent := t.TempDir()
	dead1 := ownerName(".fak-cand-validate-", 111)
	dead2 := ownerName(".fak-cand-validate-", 112)
	live := ownerName(".fak-cand-validate-", 222)
	for _, n := range []string{dead1, dead2, live} {
		mkdirAged(t, parent, n, time.Minute)
	}

	r := Sweep(gitRepo(t), parent, opts(true, 1, 222))

	if r.Mode != "apply" || r.WouldReap != 2 || r.Reaped != 1 {
		t.Fatalf("mode=%q would_reap=%d reaped=%d, want apply/2/1", r.Mode, r.WouldReap, r.Reaped)
	}
	if len(r.Failures) != 0 {
		t.Fatalf("failures = %+v, want none", r.Failures)
	}
	removed := 0
	for _, it := range r.Candidates {
		_, statErr := os.Stat(it.Path)
		gone := os.IsNotExist(statErr)
		if it.Removed != gone {
			t.Errorf("%s removed=%v but on-disk gone=%v", filepath.Base(it.Path), it.Removed, gone)
		}
		if it.Removed {
			removed++
			if !it.Eligible {
				t.Errorf("%s removed while ineligible", filepath.Base(it.Path))
			}
		}
	}
	if removed != 1 {
		t.Fatalf("removed %d candidates, want 1", removed)
	}
	if _, err := os.Stat(filepath.Join(parent, live)); err != nil {
		t.Fatalf("live candidate removed: %v", err)
	}
}

// fak-test:runtime fast est=10ms
func TestSweepMissingParentReportsFailure(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "absent")
	r := Sweep(gitRepo(t), missing, opts(true, 0))
	if len(r.Candidates) != 0 || r.Reaped != 0 || len(r.Failures) != 1 || r.Failures[0].Path != missing {
		t.Fatalf("report = %+v, want one failure for %s and no candidates", r, missing)
	}
}

// fak-test:runtime fast est=1ms
func TestCounts(t *testing.T) {
	r := SweepReport{Candidates: make([]SweepItem, 5), Reaped: 2}
	scanned, reaped, kept := Counts(r)
	if scanned != 5 || reaped != 2 || kept != 3 {
		t.Fatalf("Counts = %d,%d,%d, want 5,2,3", scanned, reaped, kept)
	}
	if s, rp, k := Counts(SweepReport{}); s != 0 || rp != 0 || k != 0 {
		t.Fatalf("Counts(empty) = %d,%d,%d, want 0,0,0", s, rp, k)
	}
}
