package treedoctor

// gotmpsweep_test.go — the #6207 witness.
//
// The load-bearing property is the LIVENESS KEY: an entry's age is the newest file found
// ANYWHERE inside it, never the top-level mtime. A long `go build` writes into
// subdirectories of its WORK dir for its whole run without bumping the top-level dir's
// mtime, so a naive mtime sweep deletes a running build's WORK dir. TestSweepGoTmpKeeps...
// pins exactly that case: a dir whose top level is backdated 20 hours but whose nested file
// was written a moment ago must SURVIVE an applied sweep.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// writeAged writes a file and stamps both its times to age before now.
func writeAged(t *testing.T, path string, size int, now time.Time, age time.Duration) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
	stamp := now.Add(-age)
	if err := os.Chtimes(path, stamp, stamp); err != nil {
		t.Fatal(err)
	}
}

// ageDir stamps a directory's own mtime. Called AFTER its contents are written, since
// creating a child bumps the parent.
func ageDir(t *testing.T, path string, now time.Time, age time.Duration) {
	t.Helper()
	stamp := now.Add(-age)
	if err := os.Chtimes(path, stamp, stamp); err != nil {
		t.Fatal(err)
	}
}

// TestPlanGoTmpDecisionTable pins the closed verdict vocabulary over hand-built entries.
// Plan is the pure half — no clock, no filesystem — so every keep case can be stated
// directly instead of being reproduced on disk.
func TestPlanGoTmpDecisionTable(t *testing.T) {
	opts := GoTmpOptions{Root: "/gotmp", MinAge: 2 * time.Hour}

	cases := []struct {
		name  string
		entry GoTmpEntry
		want  GoTmpVerdict
		why   string
	}{
		{
			name:  "an orphaned WORK dir quiet for 17 hours",
			entry: GoTmpEntry{Name: "go-build3064938779", NewestAgeSec: 17 * 60 * 60, Bytes: 1010 << 20},
			want:  GoTmpReap,
			why:   "go removes its own WORK dir on a clean exit; a surviving one was killed",
		},
		{
			name:  "a WORK dir written into a minute ago",
			entry: GoTmpEntry{Name: "go-build123", NewestAgeSec: 60},
			want:  GoTmpKeepLive,
			why:   "a build may still be running here",
		},
		{
			name:  "a WORK dir exactly at the floor",
			entry: GoTmpEntry{Name: "go-build124", NewestAgeSec: 2 * 60 * 60},
			want:  GoTmpReap,
			why:   "the floor is inclusive; anything quieter than MinAge is reapable",
		},
		{
			name:  "a leaked t.TempDir named after its test",
			entry: GoTmpEntry{Name: "TestAccountsNextRoundRobin1234", NewestAgeSec: 40 * 60 * 60},
			want:  GoTmpKeepForeign,
			why:   "not a go WORK dir — belongs to whichever package leaked it",
		},
		{
			name:  "a WORK dir whose walk was truncated",
			entry: GoTmpEntry{Name: "go-build9", NewestAgeSec: 40 * 60 * 60, Truncated: true},
			want:  GoTmpKeepIndeterminate,
			why:   "liveness unproven past the walk bound; a false reap breaks a live build",
		},
		{
			name:  "a WORK dir whose walk errored",
			entry: GoTmpEntry{Name: "go-build10", NewestAgeSec: 40 * 60 * 60, ScanErr: "permission denied"},
			want:  GoTmpKeepIndeterminate,
			why:   "unreadable is not the same as quiet",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rep := PlanGoTmp([]GoTmpEntry{tc.entry}, opts)
			if len(rep.Entries) != 1 {
				t.Fatalf("want 1 entry, got %d", len(rep.Entries))
			}
			if got := rep.Entries[0].Verdict; got != tc.want {
				t.Fatalf("verdict = %q, want %q (%s)", got, tc.want, tc.why)
			}
			wantReaped := 0
			if tc.want == GoTmpReap {
				wantReaped = 1
			}
			if got := rep.ReapCount(); got != wantReaped {
				t.Fatalf("ReapCount = %d, want %d", got, wantReaped)
			}
		})
	}
}

// TestPlanGoTmpAgeSplitsBeforeCallingItALeak pins the measurement caveat the ticket asks be
// preserved: the report always separates in-flight churn from stale mass, so a reader
// cannot repeat the earlier audit pass that sampled mid-sweep and concluded either "6GB
// leak" or "self-collects" from one undifferentiated total.
func TestPlanGoTmpAgeSplitsBeforeCallingItALeak(t *testing.T) {
	entries := []GoTmpEntry{
		{Name: "go-buildA", NewestAgeSec: 10 * 60, Bytes: 600},       // in-flight
		{Name: "go-buildB", NewestAgeSec: 10 * 60 * 60, Bytes: 5000}, // stale
		{Name: "go-buildC", NewestAgeSec: 30 * 60 * 60, Bytes: 12},   // cold
		{Name: "TestLeak1", NewestAgeSec: 30 * 60 * 60, Bytes: 41},   // cold but foreign
	}
	rep := PlanGoTmp(entries, GoTmpOptions{MinAge: 2 * time.Hour})

	want := []GoTmpBand{
		{Name: "in_flight", Entries: 1, Bytes: 600},
		{Name: "stale", Entries: 1, Bytes: 5000},
		{Name: "cold", Entries: 2, Bytes: 53},
	}
	if len(rep.Bands) != len(want) {
		t.Fatalf("bands = %+v, want %d rows", rep.Bands, len(want))
	}
	for i, w := range want {
		if rep.Bands[i] != w {
			t.Fatalf("band %d = %+v, want %+v", i, rep.Bands[i], w)
		}
	}
	if rep.TotalBytes != 5653 {
		t.Fatalf("TotalBytes = %d, want 5653", rep.TotalBytes)
	}
	// The cold band counts the foreign leftover (it IS mass on disk) but the reap does not
	// claim it: bands measure, verdicts decide.
	if rep.ReapedBytes != 5012 {
		t.Fatalf("ReapedBytes = %d, want 5012 (stale+cold go-build only)", rep.ReapedBytes)
	}
}

// TestSweepGoTmpKeepsALiveBuildWhoseTopLevelMtimeIsStale is THE witness for the ticket's
// liveness rule. Both dirs have a 20-hour-old TOP-LEVEL mtime; only one has a file written
// inside a moment ago. A sweep keyed on the top-level mtime deletes both and breaks the
// running build. This one must delete exactly the orphan.
func TestSweepGoTmpKeepsALiveBuildWhoseTopLevelMtimeIsStale(t *testing.T) {
	now := time.Now()
	root := t.TempDir()

	// A LIVE build: top-level backdated 20h, but it is still writing deep inside.
	live := filepath.Join(root, "go-build1111")
	writeAged(t, filepath.Join(live, "b001", "pkg", "_pkg_.a"), 128, now, 0)
	ageDir(t, filepath.Join(live, "b001", "pkg"), now, 20*time.Hour)
	ageDir(t, filepath.Join(live, "b001"), now, 20*time.Hour)
	ageDir(t, live, now, 20*time.Hour)

	// An ORPHAN: nothing anywhere inside has been touched in 20h.
	orphan := filepath.Join(root, "go-build2222")
	writeAged(t, filepath.Join(orphan, "b001", "pkg", "_pkg_.a"), 256, now, 20*time.Hour)
	ageDir(t, filepath.Join(orphan, "b001", "pkg"), now, 20*time.Hour)
	ageDir(t, filepath.Join(orphan, "b001"), now, 20*time.Hour)
	ageDir(t, orphan, now, 20*time.Hour)

	// A leaked test temp dir, equally cold — not this reaper's garbage.
	foreign := filepath.Join(root, "TestSomethingLeaked999")
	writeAged(t, filepath.Join(foreign, "note.txt"), 16, now, 20*time.Hour)
	ageDir(t, foreign, now, 20*time.Hour)

	rep := SweepGoTmp(GoTmpOptions{Root: root, Now: now, MinAge: 2 * time.Hour}, true)
	if rep.Err != "" {
		t.Fatalf("sweep error: %s", rep.Err)
	}
	if rep.Failed() {
		t.Fatalf("sweep reported a removal failure: %+v", rep.Entries)
	}

	if _, err := os.Stat(orphan); !os.IsNotExist(err) {
		t.Fatalf("the orphaned WORK dir survived the sweep (stat err %v)", err)
	}
	if _, err := os.Stat(live); err != nil {
		t.Fatalf("the LIVE build's WORK dir was deleted — the top-level mtime was trusted: %v", err)
	}
	if _, err := os.Stat(foreign); err != nil {
		t.Fatalf("a foreign leftover was deleted: %v", err)
	}

	if got := rep.ReapCount(); got != 1 {
		t.Fatalf("ReapCount = %d, want exactly the one orphan (%+v)", got, rep.Reaped)
	}
	if rep.ReapedBytes != 256 {
		t.Fatalf("ReapedBytes = %d, want 256", rep.ReapedBytes)
	}
	byName := map[string]GoTmpEntry{}
	for _, e := range rep.Entries {
		byName[e.Name] = e
	}
	if v := byName["go-build1111"].Verdict; v != GoTmpKeepLive {
		t.Fatalf("live dir verdict = %q, want %q", v, GoTmpKeepLive)
	}
	if v := byName["go-build2222"].Verdict; v != GoTmpReap {
		t.Fatalf("orphan verdict = %q, want %q", v, GoTmpReap)
	}
	if v := byName["TestSomethingLeaked999"].Verdict; v != GoTmpKeepForeign {
		t.Fatalf("foreign dir verdict = %q, want %q", v, GoTmpKeepForeign)
	}
}

// TestSweepGoTmpReapsAnEmptyWorkDirByItsOwnMtime covers the one case where there is no file
// inside to key on: the entry's own mtime is then the only evidence, and a 20-hour-old
// empty WORK dir is still an orphan.
func TestSweepGoTmpReapsAnEmptyWorkDirByItsOwnMtime(t *testing.T) {
	now := time.Now()
	root := t.TempDir()
	empty := filepath.Join(root, "go-build3333")
	if err := os.MkdirAll(empty, 0o755); err != nil {
		t.Fatal(err)
	}
	ageDir(t, empty, now, 20*time.Hour)

	rep := SweepGoTmp(GoTmpOptions{Root: root, Now: now}, true)
	if got := rep.ReapCount(); got != 1 {
		t.Fatalf("ReapCount = %d, want 1 (%+v)", got, rep.Entries)
	}
	if _, err := os.Stat(empty); !os.IsNotExist(err) {
		t.Fatalf("empty orphan survived (stat err %v)", err)
	}
}

// TestSweepGoTmpDryRunMutatesNothing pins that the diagnosis path reports the same decision
// it would act on, while leaving every byte in place.
func TestSweepGoTmpDryRunMutatesNothing(t *testing.T) {
	now := time.Now()
	root := t.TempDir()
	orphan := filepath.Join(root, "go-build4444")
	writeAged(t, filepath.Join(orphan, "b001", "_pkg_.a"), 64, now, 30*time.Hour)
	ageDir(t, filepath.Join(orphan, "b001"), now, 30*time.Hour)
	ageDir(t, orphan, now, 30*time.Hour)

	rep := SweepGoTmp(GoTmpOptions{Root: root, Now: now}, false)
	if !rep.DryRun {
		t.Fatal("DryRun should be set on a non-apply sweep")
	}
	if got := rep.ReapCount(); got != 1 {
		t.Fatalf("dry run should report the 1 reap it would do, got %d", got)
	}
	if rep.ReapedBytes != 64 {
		t.Fatalf("ReapedBytes = %d, want 64", rep.ReapedBytes)
	}
	if _, err := os.Stat(orphan); err != nil {
		t.Fatalf("dry run removed the dir: %v", err)
	}
}

// TestSweepGoTmpEmptyRootIsANoOp is the wiring safety net: a caller with no redirected
// GOTMPDIR must get a silent no-op, never an error and never a sweep of some default.
func TestSweepGoTmpEmptyRootIsANoOp(t *testing.T) {
	rep := SweepGoTmp(GoTmpOptions{}, true)
	if rep.Err != "" || rep.ReapCount() != 0 || len(rep.Entries) != 0 {
		t.Fatalf("empty root should be inert, got %+v", rep)
	}
	if rep.Failed() {
		t.Fatal("an inert sweep must not report failure")
	}
}

// TestSweepGoTmpMissingRootIsReportedNotPanicked: a configured-but-absent GOTMPDIR is an
// operator-visible fact, not a crash and not a silent success.
func TestSweepGoTmpMissingRootIsReportedNotPanicked(t *testing.T) {
	rep := SweepGoTmp(GoTmpOptions{Root: filepath.Join(t.TempDir(), "nope")}, true)
	if rep.Err == "" {
		t.Fatal("a missing root should surface an error")
	}
	if !rep.Failed() {
		t.Fatal("Failed() should be true when the root could not be read")
	}
	if rep.ReapCount() != 0 {
		t.Fatalf("nothing can be reaped from an unreadable root, got %d", rep.ReapCount())
	}
}

// TestGoTmpRootFromEnv pins the resolution the thin CLI edge uses.
func TestGoTmpRootFromEnv(t *testing.T) {
	if got := GoTmpRootFromEnv(nil); got != "" {
		t.Fatalf("nil lookup should resolve to empty, got %q", got)
	}
	if got := GoTmpRootFromEnv(func(string) string { return "  " }); got != "" {
		t.Fatalf("a blank GOTMPDIR should resolve to empty, got %q", got)
	}
	want := filepath.Join("C:", "work", "fak", "_scratch", "go-tmp")
	got := GoTmpRootFromEnv(func(k string) string {
		if k != GoTmpDirEnv {
			t.Fatalf("looked up %q, want %q", k, GoTmpDirEnv)
		}
		return want
	})
	if got != want {
		t.Fatalf("resolved %q, want %q", got, want)
	}
}

// TestGoTmpReportSummary pins the operator sentence, including the two states an operator
// most needs told apart: a disabled rung and an unreadable root both reclaim nothing, and
// saying "reaped 0" for either would read as a healthy, empty tree.
func TestGoTmpReportSummary(t *testing.T) {
	if got := (GoTmpReport{}).Summary(); got != "go-build WORK dirs: rung disabled (no GOTMPDIR configured)" {
		t.Fatalf("disabled summary = %q", got)
	}
	got := GoTmpReport{Root: "/gotmp", Err: "permission denied"}.Summary()
	if got != "go-build WORK dirs: could not read /gotmp: permission denied" {
		t.Fatalf("error summary = %q", got)
	}

	rep := PlanGoTmp([]GoTmpEntry{
		{Name: "go-buildA", NewestAgeSec: 20 * 60 * 60, Bytes: 2 << 20},
		{Name: "go-buildB", NewestAgeSec: 60, Bytes: 1 << 20},
		{Name: "go-buildC", NewestAgeSec: 20 * 60 * 60, Truncated: true},
		{Name: "TestLeak", NewestAgeSec: 20 * 60 * 60},
	}, GoTmpOptions{Root: "/gotmp"})
	want := "go-build WORK dirs in /gotmp: reaped 1 (2.0 MB), kept 1 live, 1 indeterminate, 1 foreign; 3.0 MB on disk"
	if rep.Summary() != want {
		t.Fatalf("summary = %q, want %q", rep.Summary(), want)
	}
	rep.DryRun = true
	if got := rep.Summary(); !strings.Contains(got, "would reap 1") {
		t.Fatalf("a dry-run summary must not claim a reap happened: %q", got)
	}
}

// TestScanGoTmpSkipsLooseFiles: only directories are candidates. A loose file in GOTMPDIR
// belongs to something else and must never enter the plan.
func TestScanGoTmpSkipsLooseFiles(t *testing.T) {
	now := time.Now()
	root := t.TempDir()
	writeAged(t, filepath.Join(root, "go-build-not-a-dir"), 8, now, 30*time.Hour)

	entries, err := ScanGoTmp(GoTmpOptions{Root: root, Now: now})
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("a loose file must not be scanned as an entry, got %+v", entries)
	}
}

// TestScanGoTmpTruncatedWalkIsIndeterminate proves the bounded walk fails SAFE: when the
// entry cap is hit the sweep cannot claim the dir is quiet, so it keeps it.
func TestScanGoTmpTruncatedWalkIsIndeterminate(t *testing.T) {
	now := time.Now()
	root := t.TempDir()
	dir := filepath.Join(root, "go-build5555")
	for i := 0; i < 5; i++ {
		writeAged(t, filepath.Join(dir, "f"+string(rune('a'+i))), 1, now, 30*time.Hour)
	}
	ageDir(t, dir, now, 30*time.Hour)

	rep := SweepGoTmp(GoTmpOptions{Root: root, Now: now, MaxWalkEntries: 2}, true)
	if got := rep.ReapCount(); got != 0 {
		t.Fatalf("a truncated walk must reap nothing, got %d", got)
	}
	if len(rep.Entries) != 1 || rep.Entries[0].Verdict != GoTmpKeepIndeterminate {
		t.Fatalf("want one INDETERMINATE entry, got %+v", rep.Entries)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("the unproven dir was removed: %v", err)
	}
}
