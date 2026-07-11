package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// mustTime parses an RFC3339 timestamp or fails the test.
func mustTime(t *testing.T, s string) time.Time {
	t.Helper()
	got, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatalf("parse time %q: %v", s, err)
	}
	return got
}

// mustDur parses a Go duration or fails the test.
func mustDur(t *testing.T, s string) time.Duration {
	t.Helper()
	got, err := time.ParseDuration(s)
	if err != nil {
		t.Fatalf("parse duration %q: %v", s, err)
	}
	return got
}

// fireArgs builds a `cron fire` argv for one job/slot against a ledger.
func fireArgs(job, ledger, at, interval string) []string {
	argv := []string{"fire", "--job", job, "--ledger", ledger, "--quiet"}
	if at != "" {
		argv = append(argv, "--at", at)
	}
	if interval != "" {
		argv = append(argv, "--interval", interval)
	}
	return argv
}

// countFired reads the ledger and counts fired rows for (job, slot).
func countFired(t *testing.T, ledger, job, slot string) int {
	t.Helper()
	fires, err := cronReadFires(ledger)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	n := 0
	for _, r := range fires {
		if r.Job == job && r.Slot == slot && r.Outcome == cronOutcomeFired {
			n++
		}
	}
	return n
}

// TestCronFireDoubleTickSingleDelivery induces a double-tick sequentially: two
// fires for the same (job, slot) must yield exactly one `fired` (exit 0) and one
// `deduped` (exit cronExitDeduped) — the compare-and-set holds at-most-once.
func TestCronFireDoubleTickSingleDelivery(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "fires.jsonl")
	const job, at, interval = "nightly", "2026-07-10T00:15:00Z", "1h"
	slot := cronFireSlot(mustTime(t, at), mustDur(t, interval))

	var out, errb bytes.Buffer
	if code := runCron(&out, &errb, fireArgs(job, ledger, at, interval)); code != 0 {
		t.Fatalf("first fire: exit %d, want 0 (fired); stderr=%q", code, errb.String())
	}
	out.Reset()
	errb.Reset()
	if code := runCron(&out, &errb, fireArgs(job, ledger, at, interval)); code != cronExitDeduped {
		t.Fatalf("second fire: exit %d, want %d (deduped); stderr=%q", code, cronExitDeduped, errb.String())
	}
	if n := countFired(t, ledger, job, slot); n != 1 {
		t.Fatalf("fired records for (%s,%s) = %d, want exactly 1 (single delivery)", job, slot, n)
	}
}

// TestCronFireConcurrentDoubleTickSingleDelivery is the strong form: two ticks race
// the SAME (job, slot) from separate goroutines. The dup-tick lock serializes them
// so the CAS admits exactly one — one exit 0, one exit cronExitDeduped, one fired
// row on disk. This is the "induce a double-tick, prove single delivery" contract.
func TestCronFireConcurrentDoubleTickSingleDelivery(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "fires.jsonl")
	const job, slot = "hourly", "2026-07-10T00:00:00Z"

	var wg sync.WaitGroup
	codes := make([]int, 2)
	for i := range codes {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			var out, errb bytes.Buffer
			// --slot fixes both ticks onto one slot regardless of scheduling jitter.
			codes[idx] = runCron(&out, &errb, []string{
				"fire", "--job", job, "--ledger", ledger, "--slot", slot, "--quiet",
			})
		}(i)
	}
	wg.Wait()

	fired, deduped := 0, 0
	for _, c := range codes {
		switch c {
		case 0:
			fired++
		case cronExitDeduped:
			deduped++
		default:
			t.Fatalf("unexpected exit %d from a racing tick", c)
		}
	}
	if fired != 1 || deduped != 1 {
		t.Fatalf("racing ticks: fired=%d deduped=%d, want 1 and 1 (single delivery)", fired, deduped)
	}
	if n := countFired(t, ledger, job, slot); n != 1 {
		t.Fatalf("fired records for (%s,%s) = %d, want exactly 1", job, slot, n)
	}
}

// TestCronAuditReportsFiredMissedDeduped drives fires across slots with a cadence
// gap and a duplicate, then asserts `cron audit --json` reports fired / missed /
// deduped per job.
func TestCronAuditReportsFiredMissedDeduped(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "fires.jsonl")
	const job, interval = "nightly", "1h"

	// Three distinct hourly slots fire; the 02:00 slot is skipped (a missed gap);
	// a second tick inside the 00:00 slot dedupes.
	for _, at := range []string{
		"2026-07-10T00:00:00Z", // slot 00:00 fired
		"2026-07-10T00:30:00Z", // slot 00:00 again -> deduped
		"2026-07-10T01:00:00Z", // slot 01:00 fired
		"2026-07-10T03:00:00Z", // slot 03:00 fired (02:00 missed)
	} {
		var out, errb bytes.Buffer
		_ = runCron(&out, &errb, fireArgs(job, ledger, at, interval))
	}

	var out, errb bytes.Buffer
	if code := runCron(&out, &errb, []string{"audit", "--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("audit: exit %d, want 0; stderr=%q", code, errb.String())
	}
	var got []cronJobAudit
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("audit json: %v\noutput=%q", err, out.String())
	}
	if len(got) != 1 {
		t.Fatalf("audit rows = %d, want 1: %+v", len(got), got)
	}
	g := got[0]
	if g.Job != job || g.Fired != 3 || g.Missed != 1 || g.Deduped != 1 {
		t.Fatalf("audit = %+v, want {Job:%s Fired:3 Missed:1 Deduped:1}", g, job)
	}
}

// TestCronAuditEmptyLedger reports cleanly when nothing has fired.
func TestCronAuditEmptyLedger(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "empty.jsonl")
	var out, errb bytes.Buffer
	if code := runCron(&out, &errb, []string{"audit", "--ledger", ledger}); code != 0 {
		t.Fatalf("audit empty: exit %d, want 0; stderr=%q", code, errb.String())
	}
	if got := out.String(); got == "" {
		t.Fatalf("audit empty: want a message, got empty output")
	}
	// A missing ledger must not error out of cronReadFires.
	if _, err := os.Stat(ledger); !os.IsNotExist(err) {
		t.Fatalf("audit must not create the ledger; stat err=%v", err)
	}
}

// TestCronFireRequiresJobAndLedger rejects missing required flags with exit 2.
func TestCronFireRequiresJobAndLedger(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runCron(&out, &errb, []string{"fire", "--ledger", "x.jsonl"}); code != 2 {
		t.Fatalf("fire without --job: exit %d, want 2", code)
	}
	out.Reset()
	errb.Reset()
	if code := runCron(&out, &errb, []string{"fire", "--job", "j"}); code != 2 {
		t.Fatalf("fire without --ledger: exit %d, want 2", code)
	}
}
