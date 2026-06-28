package cachevalueledger

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

func TestParseLedger(t *testing.T) {
	content := `{"schema":"fak-cache-value-ledger/1","date":"2026-06-27","session_type":"serve","context":"test","pid":12345,"unix_millis":1719494400000,"turns":10,"prompt_tokens":1000,"reused_tokens":800,"frozen_turns":5,"partial_turns":3,"cold_turns":2,"reuse_ratio":4.0}
{"schema":"fak-cache-value-ledger/1","date":"2026-06-27","session_type":"run","context":"test2","pid":12346,"unix_millis":1719494500000,"turns":5,"prompt_tokens":500,"reused_tokens":400,"frozen_turns":2,"partial_turns":1,"cold_turns":2,"reuse_ratio":4.0}
not json
{"date":"","session_type":"serve"}
`
	rows := ParseLedger(content)
	if len(rows) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(rows))
	}
	if rows[0].SessionType != "serve" {
		t.Errorf("expected session_type 'serve', got %s", rows[0].SessionType)
	}
	if rows[1].SessionType != "run" {
		t.Errorf("expected session_type 'run', got %s", rows[1].SessionType)
	}
}

func TestAppendLedgerLine(t *testing.T) {
	row := Row{
		Schema:       Schema,
		Date:         "2026-06-27",
		SessionType:  "serve",
		Context:      "test",
		PID:          12345,
		UnixMillis:   1719494400000,
		Turns:        10,
		PromptTokens: 1000,
		ReusedTokens: 800,
		FrozenTurns:  5,
		PartialTurns: 3,
		ColdTurns:    2,
		ReuseRatio:   4.0,
	}
	line, err := AppendLedgerLine(row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if line == "" {
		t.Fatal("expected non-empty line")
	}
}

func TestNewRow(t *testing.T) {
	now := time.Date(2026, 6, 27, 12, 0, 0, 0, time.UTC)
	stats := cacheobs.Stats{
		Turns:        10,
		PromptTokens: 1000,
		ReusedTokens: 800,
		FrozenTurns:  5,
		PartialTurns: 3,
		ColdTurns:    2,
		ReuseRatio:   4.0,
	}
	row := NewRow("serve", "test", stats, now)
	if row.Schema != Schema {
		t.Errorf("expected schema %s, got %s", Schema, row.Schema)
	}
	if row.Date != "2026-06-27" {
		t.Errorf("expected date 2026-06-27, got %s", row.Date)
	}
	if row.SessionType != "serve" {
		t.Errorf("expected session_type 'serve', got %s", row.SessionType)
	}
	if row.Turns != 10 {
		t.Errorf("expected turns 10, got %d", row.Turns)
	}
	if row.PromptTokens != 1000 {
		t.Errorf("expected prompt_tokens 1000, got %d", row.PromptTokens)
	}
	if row.ReusedTokens != 800 {
		t.Errorf("expected reused_tokens 800, got %d", row.ReusedTokens)
	}
}

func TestAppendAndReadLedgerFile(t *testing.T) {
	tmpDir := t.TempDir()
	ledgerPath := filepath.Join(tmpDir, "test-ledger.jsonl")
	stats := cacheobs.Stats{
		Turns:        10,
		PromptTokens: 1000,
		ReusedTokens: 800,
		FrozenTurns:  5,
		PartialTurns: 3,
		ColdTurns:    2,
		ReuseRatio:   4.0,
	}
	err := Append("serve", "test", ledgerPath, stats)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	rows := ReadLedgerFile(ledgerPath)
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	if rows[0].SessionType != "serve" {
		t.Errorf("expected session_type 'serve', got %s", rows[0].SessionType)
	}
	if rows[0].Turns != 10 {
		t.Errorf("expected turns 10, got %d", rows[0].Turns)
	}
}

func TestScoreLedger(t *testing.T) {
	tmpDir := t.TempDir()
	ledgerPath := filepath.Join(tmpDir, "test-ledger.jsonl")
	stats1 := cacheobs.Stats{
		Turns:        10,
		PromptTokens: 1000,
		ReusedTokens: 800,
		FrozenTurns:  6,
		PartialTurns: 3,
		ColdTurns:    1,
		ReuseRatio:   0.8,
	}
	stats2 := cacheobs.Stats{
		Turns:        5,
		PromptTokens: 500,
		ReusedTokens: 400,
		ReuseRatio:   0.8,
	}
	Append("serve", "test1", ledgerPath, stats1)
	Append("run", "test2", ledgerPath, stats2)

	result, err := ScoreLedger(ledgerPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalSessions != 2 || result.MultiTurnSessions != 2 || result.SingleTurnSessions != 0 {
		t.Errorf("session split = %d/%d/%d, want 2/2/0", result.TotalSessions, result.MultiTurnSessions, result.SingleTurnSessions)
	}
	if result.TotalTurns != 15 || result.MultiTurnTurns != 15 {
		t.Errorf("turns = %d total / %d multi-turn, want 15/15", result.TotalTurns, result.MultiTurnTurns)
	}
	if result.GateReusedTokens != 1200 || result.GatePromptTokens != 1500 {
		t.Errorf("gate tokens = %d/%d, want 1200/1500", result.GateReusedTokens, result.GatePromptTokens)
	}
	if want := 1200.0 / 1500.0; result.RealizedReuseRatio != want {
		t.Errorf("RealizedReuseRatio = %.4f, want %.4f", result.RealizedReuseRatio, want)
	}
	if !result.HasEnoughData() {
		t.Errorf("15 multi-turn turns should be enough data (MinGateTurns=%d)", MinGateTurns)
	}
	// #1066: the gate must NEVER surface the vs-naive re-prefill multiple.
	if !result.VsNaiveMultipleExcluded || result.PublishableValueFamily == "" || result.SingleSessionMarginalX != 1.0 {
		t.Errorf("honesty-fence fields not set: excluded=%v family=%q marginal=%.2f", result.VsNaiveMultipleExcluded, result.PublishableValueFamily, result.SingleSessionMarginalX)
	}
}

// A cold single-turn `fak run` has no previous turn to reuse from, so it must not drag the
// realized-reuse ratio down — folding it in would manufacture a false regression.
func TestScoreLedgerExcludesSingleTurnColdRuns(t *testing.T) {
	tmpDir := t.TempDir()
	ledgerPath := filepath.Join(tmpDir, "test-ledger.jsonl")
	// One healthy multi-turn session...
	Append("serve", "multi", ledgerPath, cacheobs.Stats{Turns: 10, PromptTokens: 1000, ReusedTokens: 800})
	// ...and three cold single-turn runs (no reuse opportunity).
	for i := 0; i < 3; i++ {
		Append("run", "single", ledgerPath, cacheobs.Stats{Turns: 1, PromptTokens: 12, ReusedTokens: 0})
	}
	result, err := ScoreLedger(ledgerPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.SingleTurnSessions != 3 || result.MultiTurnSessions != 1 {
		t.Errorf("session split = multi %d / single %d, want 1/3", result.MultiTurnSessions, result.SingleTurnSessions)
	}
	// The single-turn prompt tokens (3*12) must be excluded from the gate ratio.
	if result.GatePromptTokens != 1000 || result.RealizedReuseRatio != 0.8 {
		t.Errorf("gate prompt %d ratio %.4f, want 1000 / 0.8 (single-turn runs excluded)", result.GatePromptTokens, result.RealizedReuseRatio)
	}
}

// A thin multi-turn corpus is reported INSUFFICIENT (HasEnoughData false), so the gate
// passes rather than fabricating a regression from too little data.
func TestScoreLedgerThinCorpusIsInsufficient(t *testing.T) {
	tmpDir := t.TempDir()
	ledgerPath := filepath.Join(tmpDir, "test-ledger.jsonl")
	Append("serve", "thin", ledgerPath, cacheobs.Stats{Turns: 3, PromptTokens: 300, ReusedTokens: 30})
	result, err := ScoreLedger(ledgerPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.MultiTurnTurns >= MinGateTurns {
		t.Fatalf("test fixture should be below MinGateTurns=%d, got %d", MinGateTurns, result.MultiTurnTurns)
	}
	if result.HasEnoughData() {
		t.Errorf("a %d-turn corpus must report INSUFFICIENT, not enough to gate", result.MultiTurnTurns)
	}
}

func TestScoreLedgerEmpty(t *testing.T) {
	tmpDir := t.TempDir()
	ledgerPath := filepath.Join(tmpDir, "empty-ledger.jsonl")
	result, err := ScoreLedger(ledgerPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalSessions != 0 {
		t.Errorf("expected TotalSessions 0, got %d", result.TotalSessions)
	}
	if result.RealizedReuseRatio != 0 {
		t.Errorf("expected RealizedReuseRatio 0, got %f", result.RealizedReuseRatio)
	}
	if result.HasEnoughData() {
		t.Errorf("empty ledger must not be gateable")
	}
}
