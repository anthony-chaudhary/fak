package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/waiting"
)

// appendAt writes one chained loop event stamped at ts (WithClock overrides the
// wall clock so the fixture's filed-at times are deterministic).
func appendAt(t *testing.T, path string, ev loopmgr.Event, ts time.Time) {
	t.Helper()
	if _, err := loopmgr.Append(path, ev, loopmgr.WithClock(func() time.Time { return ts })); err != nil {
		t.Fatalf("append %s@%s: %v", ev.Kind, ev.LoopID, err)
	}
}

// TestWaitingVerbFoldsLedgerRankedByCostOfDelay is the wiring gate: a real
// hash-chained ledger (built via the loopmgr writer) folds into the queue the
// verb prints. An expired blocked notify surfaces its EXPIRED_DEFAULT row with
// the prescribed safe default; an unrelated DONE notify never becomes a row.
func TestWaitingVerbFoldsLedgerRankedByCostOfDelay(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, ".fak", "loops.jsonl")

	// A blocked-on-operator run filed ~3h ago and still in flight: past the 2h
	// deadline -> expired_default, holding a worker seat.
	old := time.Now().UTC().Add(-3 * time.Hour)
	appendAt(t, ledger, loopmgr.Event{LoopID: "dispatch/iss-1", RunID: "run-1", Kind: loopmgr.EventStart}, old)
	appendAt(t, ledger, loopmgr.Event{LoopID: "dispatch/iss-1", RunID: "run-1", Kind: loopmgr.EventNotify, Reason: "ESCALATE_APPROVAL"}, old)
	// An informational completion notify — must NOT become a queue row.
	appendAt(t, ledger, loopmgr.Event{LoopID: "dispatch/iss-2", RunID: "run-2", Kind: loopmgr.EventNotify, Reason: "DONE_WITNESSED"}, time.Now().UTC().Add(-1*time.Minute))

	// --json path: assert the fold shape.
	var out, errb bytes.Buffer
	if code := runWaiting(&out, &errb, []string{"--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("runWaiting --json exit = %d, want 0 (stderr: %s)", code, errb.String())
	}
	var q waiting.Queue
	if err := json.Unmarshal(out.Bytes(), &q); err != nil {
		t.Fatalf("decode queue json: %v\n%s", err, out.String())
	}
	if q.Schema != waiting.Schema {
		t.Fatalf("schema = %q, want %q", q.Schema, waiting.Schema)
	}
	if len(q.Items) != 1 {
		t.Fatalf("Items = %d, want 1 (only the blocked notify is a row; DONE_WITNESSED is not)", len(q.Items))
	}
	it := q.Items[0]
	if it.Status != waiting.StatusExpiredDefault {
		t.Fatalf("status = %q, want expired_default (filed 3h ago, 2h deadline)", it.Status)
	}
	if it.SafeDefault != "release_held_resources_and_requeue" {
		t.Fatalf("SafeDefault = %q, want release_held_resources_and_requeue", it.SafeDefault)
	}
	if !it.Held.WorkerSeat {
		t.Fatal("Held.WorkerSeat = false, want true (run started, no end)")
	}
	if q.AckClosureNotYet == "" || !strings.Contains(q.AckClosureNotYet, "#2271") {
		t.Fatalf("AckClosureNotYet must name the missing R2 ack row (#2271); got %q", q.AckClosureNotYet)
	}

	// human path: the expired row and its safe default are surfaced.
	out.Reset()
	errb.Reset()
	if code := runWaiting(&out, &errb, []string{"--ledger", ledger}); code != 0 {
		t.Fatalf("runWaiting exit = %d, want 0 (stderr: %s)", code, errb.String())
	}
	human := out.String()
	if !strings.Contains(human, "expired_default") || !strings.Contains(human, "release_held_resources_and_requeue") {
		t.Fatalf("human render missing the expired row / safe default:\n%s", human)
	}
	if !strings.Contains(human, "ranked by cost-of-delay") {
		t.Fatalf("human render missing the ranking header:\n%s", human)
	}
}

// TestWaitingVerbEmptyLedgerIsCleanExit proves the non-vacuous edge: an absent
// ledger folds to an empty queue and exits 0 — never invents a row.
func TestWaitingVerbEmptyLedgerIsCleanExit(t *testing.T) {
	dir := t.TempDir()
	var out, errb bytes.Buffer
	code := runWaiting(&out, &errb, []string{"--ledger", filepath.Join(dir, "nope.jsonl")})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 for an absent ledger (stderr: %s)", code, errb.String())
	}
	if !strings.Contains(out.String(), "empty") {
		t.Fatalf("want an empty-queue line; got:\n%s", out.String())
	}
}
