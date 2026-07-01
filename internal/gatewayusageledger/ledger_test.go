package gatewayusageledger

import (
	"path/filepath"
	"testing"
	"time"
)

// TestAppendTwoSessionsProducesTwoRows is the #1610 fail-before/pass-after proof: it
// simulates two independent `fak serve` sessions (as if the process restarted between
// them — a fresh PID/UnixMillis/counters, same ledger file) and asserts both land as
// separate JSONL rows in a temp ledger file. Before this package existed there was no
// way to durably observe a served-turn counter snapshot across a restart at all; this
// test is the evidence the ledger now provides that durability.
func TestAppendTwoSessionsProducesTwoRows(t *testing.T) {
	dir := t.TempDir()
	ledgerPath := filepath.Join(dir, "gateway-usage.jsonl")

	session1 := NewRow("exit", "serve", "http", "gw-1", 12*time.Second, Counters{
		Submits:            10,
		VDSOHits:           4,
		EngineCalls:        6,
		Denies:             1,
		Admitted:           9,
		InputTokens:        1000,
		OutputTokens:       200,
		CachedPromptTokens: 400,
	}, time.Unix(1000, 0))
	if err := Append(ledgerPath, session1); err != nil {
		t.Fatalf("Append session1: %v", err)
	}

	// Simulate a restart: a second, independent session appends to the SAME file.
	session2 := NewRow("exit", "serve", "http", "gw-2", 30*time.Second, Counters{
		Submits:            25,
		VDSOHits:           12,
		EngineCalls:        13,
		Denies:             2,
		Admitted:           23,
		InputTokens:        3000,
		OutputTokens:       800,
		CachedPromptTokens: 1200,
	}, time.Unix(2000, 0))
	if err := Append(ledgerPath, session2); err != nil {
		t.Fatalf("Append session2: %v", err)
	}

	rows := ReadLedgerFile(ledgerPath)
	if len(rows) != 2 {
		t.Fatalf("expected 2 rows after two sessions, got %d: %+v", len(rows), rows)
	}
	if rows[0].SessionID != "gw-1" || rows[1].SessionID != "gw-2" {
		t.Fatalf("rows out of expected order/content: %+v", rows)
	}
	if rows[0].Schema != Schema || rows[1].Schema != Schema {
		t.Fatalf("expected schema %q on both rows, got %q and %q", Schema, rows[0].Schema, rows[1].Schema)
	}
	if rows[0].Counters.Submits != 10 || rows[1].Counters.Submits != 25 {
		t.Fatalf("counters did not round-trip: %+v", rows)
	}

	// The reader function: fold >=2 rows into a trend (acceptance criteria).
	trend, ok := FoldTrend(rows)
	if !ok {
		t.Fatalf("FoldTrend: expected ok=true for 2 rows")
	}
	if trend.Sessions != 2 {
		t.Fatalf("trend.Sessions = %d, want 2", trend.Sessions)
	}
	if trend.DeltaSubmits != 15 {
		t.Fatalf("trend.DeltaSubmits = %d, want 15 (25-10)", trend.DeltaSubmits)
	}
	if trend.DeltaInputTokens != 2000 {
		t.Fatalf("trend.DeltaInputTokens = %d, want 2000 (3000-1000)", trend.DeltaInputTokens)
	}
	if trend.DeltaVDSOHits != 8 {
		t.Fatalf("trend.DeltaVDSOHits = %d, want 8 (12-4)", trend.DeltaVDSOHits)
	}
}

// TestFoldTrendInsufficientOnFewerThanTwoRows asserts the fall-open posture: zero or
// one row is not a failure, just "not enough data yet" (ok=false), mirroring
// cachevalueledger's thin-corpus posture.
func TestFoldTrendInsufficientOnFewerThanTwoRows(t *testing.T) {
	if _, ok := FoldTrend(nil); ok {
		t.Fatalf("FoldTrend(nil): expected ok=false")
	}
	one := []Row{NewRow("exit", "serve", "", "gw-1", 0, Counters{}, time.Now())}
	if _, ok := FoldTrend(one); ok {
		t.Fatalf("FoldTrend(1 row): expected ok=false")
	}
}

// TestReadLedgerFileMissingIsEmptyNotError matches ReadLedgerFile's documented
// fall-open posture — a ledger that has never been written to is a clean first-run
// state (nil rows), not an error a caller must special-case.
func TestReadLedgerFileMissingIsEmptyNotError(t *testing.T) {
	dir := t.TempDir()
	rows := ReadLedgerFile(filepath.Join(dir, "does-not-exist.jsonl"))
	if len(rows) != 0 {
		t.Fatalf("expected 0 rows for a missing ledger file, got %d", len(rows))
	}
}

// TestParseLedgerSkipsCorruptLines asserts a malformed or foreign line never aborts
// the whole read — only that line is dropped.
func TestParseLedgerSkipsCorruptLines(t *testing.T) {
	content := `{"schema":"fak-gateway-usage-ledger/1","kind":"exit","session_type":"serve","pid":1,"unix_millis":1000,"counters":{},"generated_at":"2026-01-01T00:00:00Z"}
not json at all
{"schema":"","kind":"exit"}

{"schema":"fak-gateway-usage-ledger/1","kind":"periodic","session_type":"serve","pid":2,"unix_millis":2000,"counters":{},"generated_at":"2026-01-01T00:01:00Z"}
`
	rows := ParseLedger(content)
	if len(rows) != 2 {
		t.Fatalf("expected 2 valid rows out of 5 lines, got %d: %+v", len(rows), rows)
	}
}
