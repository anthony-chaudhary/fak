package cachevalueledger

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

func TestRejectedTierAccessesIsTopLevelIndexed(t *testing.T) {
	stats := cacheobs.Stats{RejectedTierAccesses: 7}
	row := NewSessionRow("serve", "test", "session-7", stats, time.Unix(7, 0).UTC())
	if row.RejectedTierAccesses != 7 || row.Stats.RejectedTierAccesses != 7 {
		t.Fatalf("typed row lost rejected tier accesses: top=%d nested=%d", row.RejectedTierAccesses, row.Stats.RejectedTierAccesses)
	}

	line, err := AppendLedgerLine(row)
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal([]byte(line), &doc); err != nil {
		t.Fatal(err)
	}
	if string(doc["rejected_tier_accesses"]) != "7" {
		t.Fatalf("top-level JSON rejected_tier_accesses=%s, want 7: %s", doc["rejected_tier_accesses"], line)
	}
	var nested map[string]json.RawMessage
	if err := json.Unmarshal(doc["stats"], &nested); err != nil {
		t.Fatal(err)
	}
	if string(nested["RejectedTierAccesses"]) != "7" {
		t.Fatalf("nested Stats JSON rejected count=%s, want 7: %s", nested["RejectedTierAccesses"], line)
	}

	parsed := ParseLedger(line)
	if len(parsed) != 1 || parsed[0].RejectedTierAccesses != 7 || parsed[0].Stats.RejectedTierAccesses != 7 {
		t.Fatalf("parsed row lost rejected count: %+v", parsed)
	}

	delete(doc, "rejected_tier_accesses")
	legacyJSON, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	legacy := ParseLedger(string(legacyJSON))
	if len(legacy) != 1 || legacy[0].RejectedTierAccesses != 0 {
		t.Fatalf("legacy row without top-level field must parse zero: %+v", legacy)
	}
}

func TestRejectedTierIndexEdgeInputs(t *testing.T) {
	valid := func(value string) string {
		return `{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"edge","rejected_tier_accesses":` + value + `}`
	}
	tests := []struct {
		name    string
		content string
		want    []uint64
	}{
		{name: "empty ledger", content: ""},
		{name: "zero", content: valid("0"), want: []uint64{0}},
		{name: "maximum uint64", content: valid("18446744073709551615"), want: []uint64{^uint64(0)}},
		{name: "legacy row without index", content: valid("0")[:strings.Index(valid("0"), `,"rejected_tier_accesses"`)] + "}", want: []uint64{0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rows := ParseLedger(tt.content)
			if len(rows) != len(tt.want) {
				t.Fatalf("ParseLedger() returned %d rows, want %d: %+v", len(rows), len(tt.want), rows)
			}
			for i, want := range tt.want {
				if got := rows[i].RejectedTierAccesses; got != want {
					t.Errorf("row %d rejected_tier_accesses = %d, want %d", i, got, want)
				}
			}
		})
	}
}

func TestRejectedTierIndexAdversarialInputs(t *testing.T) {
	valid := `{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"adversarial","rejected_tier_accesses":9}`
	tests := []struct {
		name    string
		content string
	}{
		{name: "malformed JSON before valid row", content: "{not-json}\n" + valid},
		{name: "negative count before valid row", content: strings.Replace(valid, ":9}", ":-1}", 1) + "\n" + valid},
		{name: "overflow count before valid row", content: strings.Replace(valid, ":9}", ":18446744073709551616}", 1) + "\n" + valid},
		{name: "string count before valid row", content: strings.Replace(valid, ":9}", `:"9"}`, 1) + "\n" + valid},
		{name: "oversized malformed row before valid row", content: `{"junk":"` + strings.Repeat("x", 512*1024) + `"` + "\n" + valid},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rows := ParseLedger(tt.content)
			if len(rows) != 1 || rows[0].RejectedTierAccesses != 9 {
				t.Fatalf("ParseLedger() = %+v, want one valid row with rejected_tier_accesses=9", rows)
			}
		})
	}
}

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
	if rows[0].Provider != "fak" || rows[0].Mechanism != "kv_prefix_reuse" {
		t.Errorf("old ledger row dimensions = provider %q mechanism %q, want fak/kv_prefix_reuse", rows[0].Provider, rows[0].Mechanism)
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
	if row.Provider != "fak" || row.Mechanism != "kv_prefix_reuse" {
		t.Errorf("row dimensions = provider %q mechanism %q, want fak/kv_prefix_reuse", row.Provider, row.Mechanism)
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

// trendRow builds a multi-turn ledger row at an explicit time so a test can control the
// chronological window split deterministically (Append's wall-clock stamp would collide in a
// tight loop).
func trendRow(unixMillis int64, turns, prompt, reused uint64) Row {
	return Row{
		Schema:       Schema,
		Date:         "2026-06-29",
		SessionType:  "guard",
		UnixMillis:   unixMillis,
		Turns:        turns,
		PromptTokens: prompt,
		ReusedTokens: reused,
	}
}

// The trend gate must turn RED when realized reuse drops from the baseline window to the
// most-recent (trailing) window by more than the tolerance — the synthetic regression fixture.
func TestFoldTrendGateRegressed(t *testing.T) {
	rows := []Row{
		// Baseline window: two healthy sessions, reuse 1600/2000 = 0.80.
		trendRow(1000, 5, 1000, 800),
		trendRow(2000, 5, 1000, 800),
		// Trailing window (newer): two regressed sessions, reuse 600/2000 = 0.30.
		trendRow(3000, 5, 1000, 300),
		trendRow(4000, 5, 1000, 300),
	}
	res := FoldTrendGate(rows)
	if res.Verdict != "REGRESSED" {
		t.Fatalf("verdict = %q, want REGRESSED (finding: %s)", res.Verdict, res.Finding)
	}
	if res.OK {
		t.Errorf("a real reuse drop must NOT be CI-green (OK should be false)")
	}
	if res.BaselineReuseRatio != 0.8 || res.RecentReuseRatio != 0.3 {
		t.Errorf("windows = baseline %.3f / recent %.3f, want 0.800 / 0.300", res.BaselineReuseRatio, res.RecentReuseRatio)
	}
	if res.DeltaReuseRatio >= -TrendRegressionTolerance {
		t.Errorf("delta %.3f should be below -tolerance %.3f", res.DeltaReuseRatio, TrendRegressionTolerance)
	}
	// #1066 fence: the regression is measured on realized reuse, never the forbidden multiple.
	if !res.VsNaiveMultipleExcluded || res.PublishableValueFamily == "" {
		t.Errorf("honesty-fence fields not set: excluded=%v family=%q", res.VsNaiveMultipleExcluded, res.PublishableValueFamily)
	}
}

// A thin corpus (too few multi-turn turns to form both a baseline and a trailing window) must fall
// open INSUFFICIENT and stay CI-green — too little evidence is never a regression.
func TestFoldTrendGateThinCorpusIsInsufficient(t *testing.T) {
	// One short multi-turn session: no baseline window can be formed.
	res := FoldTrendGate([]Row{trendRow(1000, 3, 300, 30)})
	if res.Verdict != "INSUFFICIENT" {
		t.Fatalf("verdict = %q, want INSUFFICIENT (finding: %s)", res.Verdict, res.Finding)
	}
	if !res.OK {
		t.Errorf("a thin corpus must stay CI-green (OK should be true)")
	}
}

// A corpus thick enough to trend but with reuse holding steady (drop within tolerance) passes.
func TestFoldTrendGateStableIsOK(t *testing.T) {
	rows := []Row{
		trendRow(1000, 5, 1000, 800), // baseline 0.80
		trendRow(2000, 5, 1000, 800),
		trendRow(3000, 5, 1000, 790), // recent 0.78 — within the 0.05 dead-band
		trendRow(4000, 5, 1000, 790),
	}
	res := FoldTrendGate(rows)
	if res.Verdict != "OK" || !res.OK {
		t.Fatalf("verdict = %q OK = %v, want OK/true (finding: %s)", res.Verdict, res.OK, res.Finding)
	}
}

// An IMPROVING trend (recent reuse above baseline) is never a regression.
func TestFoldTrendGateImprovedIsOK(t *testing.T) {
	rows := []Row{
		trendRow(1000, 5, 1000, 600), // baseline 0.60
		trendRow(2000, 5, 1000, 600),
		trendRow(3000, 5, 1000, 900), // recent 0.90
		trendRow(4000, 5, 1000, 900),
	}
	res := FoldTrendGate(rows)
	if res.Verdict != "OK" || !res.OK {
		t.Fatalf("verdict = %q OK = %v, want OK/true (finding: %s)", res.Verdict, res.OK, res.Finding)
	}
	if res.DeltaReuseRatio <= 0 {
		t.Errorf("delta %.3f should be positive for an improving trend", res.DeltaReuseRatio)
	}
}

// Single-turn cold runs carry no reuse opportunity and must be excluded from BOTH windows, exactly
// as ScoreLedger excludes them — folding them in would manufacture a false regression.
func TestFoldTrendGateExcludesSingleTurnRuns(t *testing.T) {
	rows := []Row{
		trendRow(1000, 5, 1000, 800), // baseline 0.80
		trendRow(2000, 5, 1000, 800),
		{Schema: Schema, Date: "2026-06-29", SessionType: "run", UnixMillis: 2500, Turns: 1, PromptTokens: 12}, // cold single-turn
		trendRow(3000, 5, 1000, 790), // recent 0.78
		trendRow(4000, 5, 1000, 790),
	}
	res := FoldTrendGate(rows)
	if res.BaselineSessions != 2 || res.RecentSessions != 2 {
		t.Fatalf("session split = baseline %d / recent %d, want 2/2 (single-turn run excluded)", res.BaselineSessions, res.RecentSessions)
	}
	if res.Verdict != "OK" {
		t.Errorf("verdict = %q, want OK (finding: %s)", res.Verdict, res.Finding)
	}
}

// ScoreTrendGate is the impure file shell over FoldTrendGate; it must reproduce the same verdict
// from a ledger on disk.
func TestScoreTrendGateReadsLedger(t *testing.T) {
	tmpDir := t.TempDir()
	ledgerPath := filepath.Join(tmpDir, "trend-ledger.jsonl")
	for i := 0; i < 2; i++ {
		Append("guard", "base", ledgerPath, cacheobs.Stats{Turns: 5, PromptTokens: 1000, ReusedTokens: 800})
	}
	for i := 0; i < 2; i++ {
		Append("guard", "recent", ledgerPath, cacheobs.Stats{Turns: 5, PromptTokens: 1000, ReusedTokens: 200})
	}
	res := ScoreTrendGate(ledgerPath)
	if res.Verdict != "REGRESSED" || res.OK {
		t.Fatalf("verdict = %q OK = %v, want REGRESSED/false (finding: %s)", res.Verdict, res.OK, res.Finding)
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

func TestSessionRowStampsOpaqueJoinKeyAndLegacyOmitsIt(t *testing.T) {
	stats := cacheobs.Stats{Turns: 1}
	now := time.Unix(1, 0).UTC()
	row := NewSessionRow("serve", "http", " sess-1 ", stats, now)
	if row.SessionID != "sess-1" {
		t.Fatalf("session=%q", row.SessionID)
	}
	line, err := AppendLedgerLine(row)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(line, `"session_id":"sess-1"`) {
		t.Fatal(line)
	}
	legacy := NewRow("serve", "http", stats, now)
	line, err = AppendLedgerLine(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(line, "session_id") {
		t.Fatalf("legacy shape changed: %s", line)
	}
}
