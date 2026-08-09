package gatewayusageledger

import (
	"encoding/json"
	"path/filepath"
	"testing"
	"time"
)

// upstream_error_kinds_write_test.go is the WRITE half of the #5487 witness: it names
// Counters.UpstreamErrorKinds directly, so unlike its sibling file it cannot compile
// against the pre-fix ledger.go. The sibling (upstream_error_kinds_test.go) is the one that
// carries the runnable fail-before proof; this one pins the writer's behaviour once the
// field exists.

// TestUpstreamErrorKindsRoundTripThroughAppendAndRead is the end-to-end durability check on
// the real writer: a session that recorded a stall Appends and Reads back with the stall
// still attributable to its own kind, alongside the other kinds the same session hit.
func TestUpstreamErrorKindsRoundTripThroughAppendAndRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gateway-usage.jsonl")

	row := NewRow("exit", "guard", "claude", "sess-stall", 61*time.Second, nil, Counters{
		Total:         9,
		Allowed:       6,
		Errored:       3,
		ObservedTurns: 9,
		InputTokens:   100,
		OutputTokens:  20,
		UpstreamErrorKinds: map[string]uint64{
			"stalled":    2,
			"status_5xx": 1,
		},
	}, time.Unix(1700000000, 0))
	if err := Append(path, row); err != nil {
		t.Fatalf("Append: %v", err)
	}

	rows := ReadLedgerFile(path)
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	got := rows[0].Counters.UpstreamErrorKinds
	if got["stalled"] != 2 {
		t.Fatalf("UpstreamErrorKinds[stalled] = %d, want 2 — a stall must survive the durable write as its own kind", got["stalled"])
	}
	if got["status_5xx"] != 1 {
		t.Fatalf("UpstreamErrorKinds[status_5xx] = %d, want 1 — a co-occurring kind must stay distinct from the stall", got["status_5xx"])
	}
	if len(got) != 2 {
		t.Fatalf("UpstreamErrorKinds = %v, want exactly the two recorded kinds", got)
	}
	// The neighbouring adjudication counter is untouched. Errored is a DIFFERENT population
	// (kernel ERROR verdicts, not upstream turn failures), so the new map sits beside it
	// rather than restating or replacing it.
	if rows[0].Counters.Errored != 3 {
		t.Fatalf("Errored = %d, want 3", rows[0].Counters.Errored)
	}
}

// TestUpstreamErrorKindsAbsentWhenNothingFailed pins the omitempty contract that keeps the
// addition byte-additive: a session with no upstream errors must write NO key at all. An
// empty object on the wire would read as "measured, zero upstream failures", which is a
// different claim from "this build never measured" — and it would also change the bytes of
// every row, moving RowKey.
func TestUpstreamErrorKindsAbsentWhenNothingFailed(t *testing.T) {
	row := NewRow("exit", "serve", "http", "sess-clean", time.Second, nil, Counters{Total: 4, Allowed: 4}, time.Unix(1700000000, 0))
	b, err := json.Marshal(row)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var generic map[string]any
	if err := json.Unmarshal(b, &generic); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	c := generic["counters"].(map[string]any)
	if _, present := c["upstream_error_kinds"]; present {
		t.Fatalf("upstream_error_kinds present on a row that recorded no upstream failure: %s", b)
	}

	// An explicitly EMPTY (non-nil) map must also stay off the wire, so a caller that
	// snapshots an untouched counter map cannot accidentally assert a measured zero.
	empty := NewRow("exit", "serve", "http", "sess-clean", time.Second, nil,
		Counters{Total: 4, Allowed: 4, UpstreamErrorKinds: map[string]uint64{}}, time.Unix(1700000000, 0))
	b2, err := json.Marshal(empty)
	if err != nil {
		t.Fatalf("marshal empty-map row: %v", err)
	}
	if string(b2) != string(b) {
		t.Fatalf("an empty kind map changed the row bytes:\n nil: %s\nempty: %s", b, b2)
	}
	if empty.RowKey != row.RowKey {
		t.Fatalf("an empty kind map moved RowKey: %q vs %q", empty.RowKey, row.RowKey)
	}
}
