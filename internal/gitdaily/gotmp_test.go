package gitdaily

// gotmp_test.go — the #6207 WIRING witness.
//
// The reaper itself is proven in internal/treedoctor (gotmpsweep_test.go). What is proven
// HERE is the half that ticket #5344 taught the fleet to distrust: a collector that exists
// but is never called. `fak git-daily` is the once-a-day unattended caller an OS scheduler
// already fires, so these tests pin that one applied tick actually removes an orphaned
// `go-build*` WORK dir, spares a live one, witnesses the bytes it reclaimed into the same
// ledger that witnesses the fold, and stays completely inert when no GOTMPDIR is named.

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// goTmpTree builds a GOTMPDIR holding one ORPHANED WORK dir (nothing inside touched in
// 20h) and one LIVE one (top-level backdated 20h, but still being written into). It
// returns the root and the two paths.
func goTmpTree(t *testing.T, now time.Time) (root, orphan, live string) {
	t.Helper()
	root = t.TempDir()

	write := func(path string, size int, age time.Duration) {
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
	age := func(path string, d time.Duration) {
		stamp := now.Add(-d)
		if err := os.Chtimes(path, stamp, stamp); err != nil {
			t.Fatal(err)
		}
	}

	orphan = filepath.Join(root, "go-build3064938779")
	write(filepath.Join(orphan, "b001", "_pkg_.a"), 4096, 20*time.Hour)
	age(filepath.Join(orphan, "b001"), 20*time.Hour)
	age(orphan, 20*time.Hour)

	live = filepath.Join(root, "go-build1263071031")
	write(filepath.Join(live, "b001", "_pkg_.a"), 512, 0)
	age(filepath.Join(live, "b001"), 20*time.Hour)
	age(live, 20*time.Hour)

	return root, orphan, live
}

// TestTickCollectsOrphanedGoBuildWorkDirs is the #6207 witness: the scheduled tick — not a
// manual verb — is what reclaims the orphans, and it reclaims exactly the orphans.
func TestTickCollectsOrphanedGoBuildWorkDirs(t *testing.T) {
	now := time.Now()
	gotmp, orphan, live := goTmpTree(t, now)

	opts := fakeRepo(t)
	opts.Now = now
	opts.Apply = true
	opts.GoTmpDir = gotmp

	var calls []string
	res := Run(context.Background(), recordingRunner(&calls), opts)
	if res.Skipped != "" {
		t.Fatalf("tick skipped (%s); the rung never ran", res.Skipped)
	}

	if _, err := os.Stat(orphan); !os.IsNotExist(err) {
		t.Fatalf("the orphaned go-build WORK dir survived a scheduled tick (stat err %v) — "+
			"the collector is unwired, which is the #5344 shape this ticket exists to avoid", err)
	}
	if _, err := os.Stat(live); err != nil {
		t.Fatalf("the tick deleted a LIVE build's WORK dir: %v", err)
	}
	if got := res.GoTmp.ReapCount(); got != 1 {
		t.Fatalf("GoTmp.ReapCount = %d, want 1 (%+v)", got, res.GoTmp.Entries)
	}
	if res.GoTmp.ReapedBytes != 4096 {
		t.Fatalf("GoTmp.ReapedBytes = %d, want 4096", res.GoTmp.ReapedBytes)
	}
	if res.GoTmp.Root != gotmp {
		t.Fatalf("GoTmp.Root = %q, want %q", res.GoTmp.Root, gotmp)
	}
	// The age split rides along in every report so a reader can never mistake in-flight
	// churn for a leak (the measurement caveat the ticket asks be preserved).
	if len(res.GoTmp.Bands) != 3 {
		t.Fatalf("the report must carry the age split, got %+v", res.GoTmp.Bands)
	}

	// And the reclaim is witnessed in the SAME ledger that witnesses the fold.
	rows := Status(LedgerPath(opts), 0)
	if len(rows) != 1 {
		t.Fatalf("want 1 ledger row, got %d", len(rows))
	}
	if rows[0].GoTmpReaped != 1 || rows[0].GoTmpReclaimedBytes != 4096 {
		t.Fatalf("ledger row = %+v, want gotmp_reaped=1 gotmp_reclaimed_bytes=4096", rows[0])
	}
}

// TestTickDryRunLeavesTheWorkDirsAlone: --dry-run reports the reclaim it would perform and
// removes nothing, through the same decision path that would remove it.
func TestTickDryRunLeavesTheWorkDirsAlone(t *testing.T) {
	now := time.Now()
	gotmp, orphan, live := goTmpTree(t, now)

	opts := fakeRepo(t)
	opts.Now = now
	opts.Apply = false
	opts.GoTmpDir = gotmp

	var calls []string
	res := Run(context.Background(), recordingRunner(&calls), opts)

	if !res.GoTmp.DryRun {
		t.Fatal("a dry-run tick must mark the GoTmp report as a dry run")
	}
	if got := res.GoTmp.ReapCount(); got != 1 {
		t.Fatalf("dry run should report the 1 reap it would do, got %d", got)
	}
	for _, p := range []string{orphan, live} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("dry run removed %s: %v", p, err)
		}
	}
}

// TestTickWithNoGoTmpDirIsInert pins the disabled state. An empty GoTmpDir must leave the
// rung completely dormant — no error, no incident, nothing walked — so a clone without the
// build-isolation redirection (and every pre-existing caller) is byte-for-byte unchanged.
func TestTickWithNoGoTmpDirIsInert(t *testing.T) {
	opts := fakeRepo(t)
	opts.Now = time.Date(2026, 8, 12, 3, 0, 0, 0, time.UTC)
	opts.Apply = true

	var calls []string
	res := Run(context.Background(), recordingRunner(&calls), opts)

	if res.GoTmp.Err != "" || res.GoTmp.ReapCount() != 0 || len(res.GoTmp.Entries) != 0 {
		t.Fatalf("the rung should be dormant with no GoTmpDir, got %+v", res.GoTmp)
	}
	rows := Status(LedgerPath(opts), 0)
	if len(rows) != 1 {
		t.Fatalf("want 1 ledger row, got %d", len(rows))
	}
	if rows[0].GoTmpReaped != 0 || rows[0].GoTmpReclaimedBytes != 0 {
		t.Fatalf("a dormant rung must not write reclaim counters: %+v", rows[0])
	}
}

// TestTickGoTmpFailureIsNotAnIncident pins the escalation choice. An unreadable GOTMPDIR is
// recorded as evidence, but Incident is reserved for "an operator must act TODAY" (posture
// drift, a lock the tick could not clear, an unwritable ledger). Disk that will be retried
// on tomorrow's tick must not spend that signal, or the daily non-zero exit trains the
// operator to ignore it.
func TestTickGoTmpFailureIsNotAnIncident(t *testing.T) {
	opts := fakeRepo(t)
	opts.Now = time.Date(2026, 8, 12, 3, 0, 0, 0, time.UTC)
	opts.Apply = true
	opts.GoTmpDir = filepath.Join(t.TempDir(), "absent-gotmpdir")

	var calls []string
	res := Run(context.Background(), recordingRunner(&calls), opts)

	if res.GoTmp.Err == "" {
		t.Fatal("an unreadable GOTMPDIR should be recorded on the report")
	}
	if res.Incident {
		t.Fatal("a GoTmp read failure must not raise the operator-must-act signal")
	}
}
