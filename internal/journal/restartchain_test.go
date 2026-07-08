package journal

import (
	"path/filepath"
	"testing"
)

// TestRestartChainAppendChainsAndVerifies pins the emit half of #3057: a
// RESTART_HOP row is a genuine chained row (it consumes a seq and verifies with
// the rest of the chain), carries the agent/session/continuity identity on the
// frozen decision fields, and carries the full correlated record on the
// non-chained Restart payload with its schema defaulted.
func TestRestartChainAppendChainsAndVerifies(t *testing.T) {
	j := OpenMemory()
	j.Emit(testDenyEvent("bash", "gw-1", `{"cmd":"rm"}`)) // a normal decision row ahead of the hop, so the hop chains onto real history

	hop := RestartHop{
		Hop:        1,
		FromTrace:  "gw-1",
		ToTrace:    "gw-2",
		SeedFile:   "/tmp/fak-guard-reset-1/reset-gw-1-to-gw-2.json",
		SeedTokens: 12,
		Handback:   "continue",
		Child:      "gw-2",
		Status:     RestartHopOK,
	}
	row := j.AppendRestartHop("claude", "guard-abc", hop)

	if row.Kind != KindRestartHop {
		t.Fatalf("Kind = %q, want %q", row.Kind, KindRestartHop)
	}
	if row.Tool != "claude" || row.TraceID != "guard-abc" {
		t.Fatalf("identity fields = (%q, %q), want (claude, guard-abc)", row.Tool, row.TraceID)
	}
	if row.Reason != RestartHopOK {
		t.Fatalf("Reason = %q, want the mirrored status %q", row.Reason, RestartHopOK)
	}
	if row.Restart == nil {
		t.Fatal("Restart payload missing from committed row")
	}
	if row.Restart.Schema != RestartChainSchema {
		t.Fatalf("payload schema = %q, want defaulted %q", row.Restart.Schema, RestartChainSchema)
	}
	if row.Restart.FromTrace != "gw-1" || row.Restart.ToTrace != "gw-2" || row.Restart.SeedTokens != 12 {
		t.Fatalf("payload not carried through: %+v", row.Restart)
	}
	if row.Seq == 0 || row.Hash == "" {
		t.Fatalf("row not committed through the chain: seq=%d hash=%q", row.Seq, row.Hash)
	}
	// The load-bearing property: the payload is NOT part of the pre-image, so the
	// chain over (decision row, hop row) verifies exactly like any other journal.
	if n, err := VerifyRows(j.Recent(0)); err != nil || n != 2 {
		t.Fatalf("VerifyRows = (%d, %v), want (2, nil)", n, err)
	}
}

// TestRestartChainNilJournalNoOp pins the caller contract: guardEmitRestartHop
// calls this unconditionally, so a --no-audit session (nil journal) must be a
// safe no-op, not a panic.
func TestRestartChainNilJournalNoOp(t *testing.T) {
	var j *Journal
	row := j.AppendRestartHop("claude", "guard-abc", RestartHop{Status: RestartHopOK})
	if row.Kind != "" || row.Seq != 0 {
		t.Fatalf("nil journal must return the zero Row, got %+v", row)
	}
}

// TestRestartChainRowsPersistAndReload pins the durable half: a hop row written
// through a file-backed journal survives close and comes back through ReadRows
// with its payload intact — the exact read path `fak guard restart-audit` uses.
func TestRestartChainRowsPersistAndReload(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	j, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	j.AppendRestartHop("claude", "guard-abc", RestartHop{
		Hop: 1, FromTrace: "gw-1", ToTrace: "gw-2", Handback: "continue", Status: RestartHopOK,
	})
	if err := j.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	rows, err := ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("len(rows) = %d, want 1", len(rows))
	}
	got := rows[0]
	if got.Kind != KindRestartHop || got.Restart == nil || got.Restart.ToTrace != "gw-2" {
		t.Fatalf("reloaded row lost its payload: %+v (restart=%+v)", got, got.Restart)
	}
}
