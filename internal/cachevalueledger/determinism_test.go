package cachevalueledger

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

// TestDeterminism verifies that cache-value ledger parsing, scoring, trend gating,
// and rejected cache-tier indexing are strictly deterministic and race-clean.
func TestDeterminism(t *testing.T) {
	t.Run("LedgerParsing", TestDeterminismLedgerParsing)
	t.Run("Scoring", TestDeterminismScoring)
	t.Run("TrendGating", TestDeterminismTrendGating)
	t.Run("EdgeCases", TestDeterminismEdgeCases)
	t.Run("ConcurrentRaceWitness", TestDeterminismConcurrentRaceWitness)
}

// TestCacheValueLedger_Determinism forwards to TestDeterminism for standard test discovery.
func TestCacheValueLedger_Determinism(t *testing.T) {
	TestDeterminism(t)
}

// TestDeterminismLedgerParsing asserts that ParseLedger, AppendLedgerLine, NewRow,
// and NewSessionRow produce byte-identical and deeply equal outputs across multiple runs.
func TestDeterminismLedgerParsing(t *testing.T) {
	fixedTime := time.Date(2026, 8, 31, 14, 30, 0, 0, time.UTC)
	stats := cacheobs.Stats{
		Turns:                10,
		PromptTokens:         1000,
		ReusedTokens:         800,
		RejectedTierAccesses: 42,
		FrozenTurns:          5,
		PartialTurns:         3,
		ColdTurns:            2,
		ReuseRatio:           0.8,
	}

	// 1. Deterministic row construction (NewRow and NewSessionRow)
	r1 := NewRow("serve", "ctx-det", stats, fixedTime)
	r2 := NewRow("serve", "ctx-det", stats, fixedTime)
	if !reflect.DeepEqual(r1, r2) {
		t.Fatalf("NewRow diverged between identical runs:\n r1: %+v\n r2: %+v", r1, r2)
	}

	s1 := NewSessionRow("serve", "ctx-det", "sess-det-10102", stats, fixedTime)
	s2 := NewSessionRow("serve", "ctx-det", "sess-det-10102", stats, fixedTime)
	if !reflect.DeepEqual(s1, s2) {
		t.Fatalf("NewSessionRow diverged between identical runs:\n s1: %+v\n s2: %+v", s1, s2)
	}
	if s1.RejectedTierAccesses != 42 || s1.Stats.RejectedTierAccesses != 42 {
		t.Fatalf("row lost rejected_tier_accesses: top=%d nested=%d", s1.RejectedTierAccesses, s1.Stats.RejectedTierAccesses)
	}

	// 2. Deterministic serialization (AppendLedgerLine)
	line1, err1 := AppendLedgerLine(s1)
	if err1 != nil {
		t.Fatalf("first AppendLedgerLine: %v", err1)
	}
	line2, err2 := AppendLedgerLine(s2)
	if err2 != nil {
		t.Fatalf("second AppendLedgerLine: %v", err2)
	}
	if line1 != line2 {
		t.Fatalf("AppendLedgerLine strings diverged:\n l1: %s\n l2: %s", line1, line2)
	}
	if !bytes.Equal([]byte(line1), []byte(line2)) {
		t.Fatalf("AppendLedgerLine bytes diverged")
	}

	// 3. Deterministic parsing across 50 iterations
	fixture := fmt.Sprintf(
		"%s\n%s\n"+
			`{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"run","context":"single","unix_millis":1756650600000,"turns":1,"prompt_tokens":100,"reused_tokens":0,"rejected_tier_accesses":5}`+"\n"+
			`{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"legacy","unix_millis":1756650601000,"turns":8,"prompt_tokens":800,"reused_tokens":600}`+"\n"+
			`{invalid-json-line}`+"\n"+
			`{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"high-rejected","unix_millis":1756650602000,"turns":4,"prompt_tokens":400,"reused_tokens":320,"rejected_tier_accesses":999999}`+"\n",
		line1,
		func() string {
			l, _ := AppendLedgerLine(r1)
			return l
		}(),
	)

	refRows := ParseLedger(fixture)
	if len(refRows) != 5 {
		t.Fatalf("expected 5 parsed rows, got %d", len(refRows))
	}
	refJSON, err := json.Marshal(refRows)
	if err != nil {
		t.Fatalf("marshal refRows: %v", err)
	}

	for i := 0; i < 50; i++ {
		gotRows := ParseLedger(fixture)
		if !reflect.DeepEqual(gotRows, refRows) {
			t.Fatalf("iteration %d: ParseLedger diverged:\n got:  %+v\n want: %+v", i, gotRows, refRows)
		}
		gotJSON, err := json.Marshal(gotRows)
		if err != nil {
			t.Fatalf("iteration %d: marshal: %v", i, err)
		}
		if !bytes.Equal(gotJSON, refJSON) {
			t.Fatalf("iteration %d: JSON bytes diverged:\n got:  %s\n want: %s", i, gotJSON, refJSON)
		}
	}

	// 4. Round-trip re-serialization determinism
	var roundTripBuf bytes.Buffer
	for _, row := range refRows {
		line, err := AppendLedgerLine(row)
		if err != nil {
			t.Fatalf("AppendLedgerLine round-trip: %v", err)
		}
		roundTripBuf.WriteString(line)
		roundTripBuf.WriteByte('\n')
	}
	reParsed := ParseLedger(roundTripBuf.String())
	if !reflect.DeepEqual(reParsed, refRows) {
		t.Fatalf("round-trip ParseLedger diverged:\n reParsed: %+v\n refRows:  %+v", reParsed, refRows)
	}
	reParsedJSON, _ := json.Marshal(reParsed)
	if !bytes.Equal(reParsedJSON, refJSON) {
		t.Fatalf("round-trip JSON bytes diverged:\n reParsedJSON: %s\n refJSON:      %s", reParsedJSON, refJSON)
	}
}

// TestDeterminismScoring asserts that ScoreLedger produces identical results and
// byte-identical JSON across repeated runs on identical ledger inputs.
func TestDeterminismScoring(t *testing.T) {
	tmpDir := t.TempDir()
	ledgerPath := filepath.Join(tmpDir, "scoring-determinism.jsonl")

	content := `{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"c1","unix_millis":1000,"turns":10,"prompt_tokens":1000,"reused_tokens":800,"rejected_tier_accesses":10,"frozen_turns":6,"partial_turns":3,"cold_turns":1,"reuse_ratio":0.8}
{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"guard","context":"c2","unix_millis":2000,"turns":6,"prompt_tokens":600,"reused_tokens":480,"rejected_tier_accesses":5,"frozen_turns":4,"partial_turns":1,"cold_turns":1,"reuse_ratio":0.8}
{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"run","context":"c3","unix_millis":3000,"turns":1,"prompt_tokens":100,"reused_tokens":0,"rejected_tier_accesses":2,"cold_turns":1,"reuse_ratio":0.0}
{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"c4","unix_millis":4000,"turns":0,"prompt_tokens":0,"reused_tokens":0}
`
	if err := os.WriteFile(ledgerPath, []byte(content), 0644); err != nil {
		t.Fatalf("write ledger: %v", err)
	}

	refScore, err := ScoreLedger(ledgerPath)
	if err != nil {
		t.Fatalf("initial ScoreLedger: %v", err)
	}
	if refScore.TotalSessions != 3 || refScore.MultiTurnSessions != 2 || refScore.SingleTurnSessions != 1 {
		t.Fatalf("unexpected sessions: total=%d multi=%d single=%d",
			refScore.TotalSessions, refScore.MultiTurnSessions, refScore.SingleTurnSessions)
	}
	if refScore.TotalTurns != 17 || refScore.MultiTurnTurns != 16 {
		t.Fatalf("unexpected turns: total=%d multi=%d", refScore.TotalTurns, refScore.MultiTurnTurns)
	}
	if refScore.GatePromptTokens != 1600 || refScore.GateReusedTokens != 1280 {
		t.Fatalf("unexpected gate tokens: prompt=%d reused=%d", refScore.GatePromptTokens, refScore.GateReusedTokens)
	}
	if want := 1280.0 / 1600.0; refScore.RealizedReuseRatio != want {
		t.Fatalf("RealizedReuseRatio = %f, want %f", refScore.RealizedReuseRatio, want)
	}
	if !refScore.HasEnoughData() {
		t.Fatalf("16 multi-turn turns must have enough data")
	}

	refBytes, err := json.Marshal(refScore)
	if err != nil {
		t.Fatalf("marshal refScore: %v", err)
	}

	for i := 0; i < 50; i++ {
		got, err := ScoreLedger(ledgerPath)
		if err != nil {
			t.Fatalf("iteration %d: ScoreLedger error: %v", i, err)
		}
		if !reflect.DeepEqual(got, refScore) {
			t.Fatalf("iteration %d: ScoreLedger diverged:\n got:  %+v\n want: %+v", i, got, refScore)
		}
		gotBytes, err := json.Marshal(got)
		if err != nil {
			t.Fatalf("iteration %d: marshal error: %v", i, err)
		}
		if !bytes.Equal(gotBytes, refBytes) {
			t.Fatalf("iteration %d: JSON bytes diverged:\n got:  %s\n want: %s", i, gotBytes, refBytes)
		}
	}

	// Thin corpus determinism (insufficient data)
	thinPath := filepath.Join(tmpDir, "thin.jsonl")
	thinContent := `{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"thin","unix_millis":1000,"turns":4,"prompt_tokens":400,"reused_tokens":200}` + "\n"
	if err := os.WriteFile(thinPath, []byte(thinContent), 0644); err != nil {
		t.Fatalf("write thin ledger: %v", err)
	}

	thinScore1, err1 := ScoreLedger(thinPath)
	thinScore2, err2 := ScoreLedger(thinPath)
	if err1 != nil || err2 != nil {
		t.Fatalf("ScoreLedger thin: err1=%v err2=%v", err1, err2)
	}
	if !reflect.DeepEqual(thinScore1, thinScore2) {
		t.Fatalf("thin ScoreLedger diverged:\n s1: %+v\n s2: %+v", thinScore1, thinScore2)
	}
	if thinScore1.HasEnoughData() {
		t.Fatalf("thin score should not have enough data")
	}
	tb1, _ := json.Marshal(thinScore1)
	tb2, _ := json.Marshal(thinScore2)
	if !bytes.Equal(tb1, tb2) {
		t.Fatalf("thin ScoreLedger JSON bytes diverged")
	}
}

// TestDeterminismTrendGating asserts that FoldTrendGate and ScoreTrendGate produce
// deeply equal and byte-identical results across runs under varied trend conditions.
func TestDeterminismTrendGating(t *testing.T) {
	cases := []struct {
		name        string
		rows        []Row
		wantVerdict string
		wantOK      bool
	}{
		{
			name: "regressed trailing window",
			rows: []Row{
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 1000, Turns: 5, PromptTokens: 1000, ReusedTokens: 800, RejectedTierAccesses: 4},
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 2000, Turns: 5, PromptTokens: 1000, ReusedTokens: 800, RejectedTierAccesses: 8},
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 3000, Turns: 5, PromptTokens: 1000, ReusedTokens: 300, RejectedTierAccesses: 12},
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 4000, Turns: 5, PromptTokens: 1000, ReusedTokens: 300, RejectedTierAccesses: 16},
			},
			wantVerdict: "REGRESSED",
			wantOK:      false,
		},
		{
			name: "stable trailing window within tolerance",
			rows: []Row{
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 1000, Turns: 5, PromptTokens: 1000, ReusedTokens: 800},
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 2000, Turns: 5, PromptTokens: 1000, ReusedTokens: 800},
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 3000, Turns: 5, PromptTokens: 1000, ReusedTokens: 780},
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 4000, Turns: 5, PromptTokens: 1000, ReusedTokens: 780},
			},
			wantVerdict: "OK",
			wantOK:      true,
		},
		{
			name: "improving trailing window",
			rows: []Row{
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 1000, Turns: 5, PromptTokens: 1000, ReusedTokens: 600},
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 2000, Turns: 5, PromptTokens: 1000, ReusedTokens: 600},
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 3000, Turns: 5, PromptTokens: 1000, ReusedTokens: 900},
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 4000, Turns: 5, PromptTokens: 1000, ReusedTokens: 900},
			},
			wantVerdict: "OK",
			wantOK:      true,
		},
		{
			name: "insufficient data thin corpus",
			rows: []Row{
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 1000, Turns: 4, PromptTokens: 400, ReusedTokens: 200},
			},
			wantVerdict: "INSUFFICIENT",
			wantOK:      true,
		},
		{
			name: "out-of-order and duplicate timestamps with rejected accesses",
			rows: []Row{
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 4000, Turns: 5, PromptTokens: 1000, ReusedTokens: 300, RejectedTierAccesses: 99},
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 1000, Turns: 5, PromptTokens: 1000, ReusedTokens: 800, RejectedTierAccesses: 10},
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 2000, Turns: 5, PromptTokens: 1000, ReusedTokens: 800, RejectedTierAccesses: 20},
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 2000, Turns: 2, PromptTokens: 200, ReusedTokens: 160, RejectedTierAccesses: 30},
				{Schema: Schema, Date: "2026-08-31", SessionType: "serve", UnixMillis: 3000, Turns: 5, PromptTokens: 1000, ReusedTokens: 300, RejectedTierAccesses: 40},
			},
			wantVerdict: "REGRESSED",
			wantOK:      false,
		},
	}

	tmpDir := t.TempDir()

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			refRes := FoldTrendGate(tc.rows)
			if refRes.Verdict != tc.wantVerdict || refRes.OK != tc.wantOK {
				t.Fatalf("unexpected FoldTrendGate result: verdict=%q ok=%v, want verdict=%q ok=%v (finding: %s)",
					refRes.Verdict, refRes.OK, tc.wantVerdict, tc.wantOK, refRes.Finding)
			}
			refJSON, err := json.Marshal(refRes)
			if err != nil {
				t.Fatalf("marshal refRes: %v", err)
			}

			// Verify fold determinism over 50 iterations
			for i := 0; i < 50; i++ {
				clone := append([]Row(nil), tc.rows...)
				got := FoldTrendGate(clone)
				if !reflect.DeepEqual(got, refRes) {
					t.Fatalf("iteration %d: FoldTrendGate diverged:\n got:  %+v\n want: %+v", i, got, refRes)
				}
				gotJSON, err := json.Marshal(got)
				if err != nil {
					t.Fatalf("iteration %d: marshal: %v", i, err)
				}
				if !bytes.Equal(gotJSON, refJSON) {
					t.Fatalf("iteration %d: JSON bytes diverged:\n got:  %s\n want: %s", i, gotJSON, refJSON)
				}
			}

			// Verify ScoreTrendGate file determinism
			safeName := filepath.Clean(tc.name)
			safeName = filepath.Base(safeName)
			ledgerPath := filepath.Join(tmpDir, safeName+".jsonl")
			var buf bytes.Buffer
			for _, r := range tc.rows {
				line, _ := AppendLedgerLine(r)
				buf.WriteString(line)
				buf.WriteByte('\n')
			}
			if err := os.WriteFile(ledgerPath, buf.Bytes(), 0644); err != nil {
				t.Fatalf("write file for ScoreTrendGate: %v", err)
			}

			fileRef := ScoreTrendGate(ledgerPath)
			if !reflect.DeepEqual(fileRef, refRes) {
				t.Fatalf("ScoreTrendGate diverged from FoldTrendGate:\n fileRef: %+v\n refRes:  %+v", fileRef, refRes)
			}
			fileRefJSON, _ := json.Marshal(fileRef)
			if !bytes.Equal(fileRefJSON, refJSON) {
				t.Fatalf("ScoreTrendGate JSON diverged from FoldTrendGate JSON")
			}

			for i := 0; i < 10; i++ {
				gotFile := ScoreTrendGate(ledgerPath)
				if !reflect.DeepEqual(gotFile, fileRef) {
					t.Fatalf("iteration %d: ScoreTrendGate diverged from fileRef", i)
				}
			}
		})
	}
}

// TestDeterminismEdgeCases verifies determinism across empty ledgers, single-turn only
// ledgers, legacy rows lacking indices, and mixed edge cases.
func TestDeterminismEdgeCases(t *testing.T) {
	tmpDir := t.TempDir()

	// 1. Empty ledger
	t.Run("EmptyLedger", func(t *testing.T) {
		empty1 := ParseLedger("")
		empty2 := ParseLedger("")
		if !reflect.DeepEqual(empty1, empty2) {
			t.Fatalf("ParseLedger on empty string diverged")
		}
		if len(empty1) != 0 {
			t.Fatalf("expected 0 rows for empty ledger, got %d", len(empty1))
		}

		emptyPath := filepath.Join(tmpDir, "empty.jsonl")
		if err := os.WriteFile(emptyPath, []byte(""), 0644); err != nil {
			t.Fatalf("write empty: %v", err)
		}

		score1, err1 := ScoreLedger(emptyPath)
		score2, err2 := ScoreLedger(emptyPath)
		if err1 != nil || err2 != nil {
			t.Fatalf("ScoreLedger empty error: %v, %v", err1, err2)
		}
		if !reflect.DeepEqual(score1, score2) {
			t.Fatalf("ScoreLedger empty diverged:\n s1: %+v\n s2: %+v", score1, score2)
		}
		sb1, _ := json.Marshal(score1)
		sb2, _ := json.Marshal(score2)
		if !bytes.Equal(sb1, sb2) {
			t.Fatalf("ScoreLedger empty JSON diverged")
		}

		trend1 := FoldTrendGate(nil)
		trend2 := FoldTrendGate([]Row{})
		if !reflect.DeepEqual(trend1, trend2) {
			t.Fatalf("FoldTrendGate nil vs empty slice diverged:\n t1: %+v\n t2: %+v", trend1, trend2)
		}
		if trend1.Verdict != "INSUFFICIENT" || !trend1.OK {
			t.Fatalf("empty FoldTrendGate should be INSUFFICIENT/true, got verdict=%s ok=%v", trend1.Verdict, trend1.OK)
		}

		fileTrend1 := ScoreTrendGate(emptyPath)
		fileTrend2 := ScoreTrendGate(emptyPath)
		if !reflect.DeepEqual(fileTrend1, fileTrend2) {
			t.Fatalf("ScoreTrendGate empty diverged")
		}
	})

	// 2. Single-turn only rows
	t.Run("SingleTurnOnly", func(t *testing.T) {
		singleRowsContent := `{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"run","context":"st1","unix_millis":1000,"turns":1,"prompt_tokens":150,"reused_tokens":0,"rejected_tier_accesses":3}
{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"run","context":"st2","unix_millis":2000,"turns":1,"prompt_tokens":200,"reused_tokens":0,"rejected_tier_accesses":0}
{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"guard","context":"st3","unix_millis":3000,"turns":1,"prompt_tokens":250,"reused_tokens":0,"rejected_tier_accesses":7}
`
		singlePath := filepath.Join(tmpDir, "single-turn.jsonl")
		if err := os.WriteFile(singlePath, []byte(singleRowsContent), 0644); err != nil {
			t.Fatalf("write single-turn: %v", err)
		}

		rows1 := ParseLedger(singleRowsContent)
		rows2 := ParseLedger(singleRowsContent)
		if !reflect.DeepEqual(rows1, rows2) {
			t.Fatalf("ParseLedger single-turn diverged")
		}

		score1, err1 := ScoreLedger(singlePath)
		score2, err2 := ScoreLedger(singlePath)
		if err1 != nil || err2 != nil {
			t.Fatalf("ScoreLedger single-turn error: %v, %v", err1, err2)
		}
		if !reflect.DeepEqual(score1, score2) {
			t.Fatalf("ScoreLedger single-turn diverged:\n s1: %+v\n s2: %+v", score1, score2)
		}
		if score1.MultiTurnSessions != 0 || score1.SingleTurnSessions != 3 || score1.GatePromptTokens != 0 || score1.RealizedReuseRatio != 0 {
			t.Fatalf("single-turn score leaked into gate values: %+v", score1)
		}

		trend1 := FoldTrendGate(rows1)
		trend2 := FoldTrendGate(rows2)
		if !reflect.DeepEqual(trend1, trend2) {
			t.Fatalf("FoldTrendGate single-turn diverged")
		}
		if trend1.Verdict != "INSUFFICIENT" || !trend1.OK {
			t.Fatalf("single-turn FoldTrendGate must be INSUFFICIENT/true, got verdict=%s ok=%v", trend1.Verdict, trend1.OK)
		}

		fileTrend1 := ScoreTrendGate(singlePath)
		fileTrend2 := ScoreTrendGate(singlePath)
		if !reflect.DeepEqual(fileTrend1, fileTrend2) {
			t.Fatalf("ScoreTrendGate single-turn diverged")
		}
	})

	// 3. Legacy rows without index or default dimensions
	t.Run("LegacyRowsWithoutIndex", func(t *testing.T) {
		legacyContent := `{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"leg1","unix_millis":1000,"turns":5,"prompt_tokens":500,"reused_tokens":400}
{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"leg2","unix_millis":2000,"turns":5,"prompt_tokens":500,"reused_tokens":400}
`
		parsed1 := ParseLedger(legacyContent)
		parsed2 := ParseLedger(legacyContent)
		if !reflect.DeepEqual(parsed1, parsed2) {
			t.Fatalf("ParseLedger legacy diverged")
		}
		for i, r := range parsed1 {
			if r.RejectedTierAccesses != 0 {
				t.Fatalf("legacy row %d must default rejected_tier_accesses to 0, got %d", i, r.RejectedTierAccesses)
			}
			if r.Provider != "fak" || r.Mechanism != "kv_prefix_reuse" {
				t.Fatalf("legacy row %d dimensions not normalized: provider=%s mechanism=%s", i, r.Provider, r.Mechanism)
			}
		}

		b1, _ := json.Marshal(parsed1)
		b2, _ := json.Marshal(parsed2)
		if !bytes.Equal(b1, b2) {
			t.Fatalf("legacy rows JSON bytes diverged")
		}
	})

	// 4. Mixed boundary inputs and max uint64
	t.Run("BoundaryAndMixedRows", func(t *testing.T) {
		mixedContent := `{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"max","unix_millis":1000,"turns":10,"prompt_tokens":1000,"reused_tokens":800,"rejected_tier_accesses":18446744073709551615}
{junk-line}
{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"run","context":"zero-tokens","unix_millis":2000,"turns":0,"prompt_tokens":0,"reused_tokens":0,"rejected_tier_accesses":0}
{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"normal","unix_millis":3000,"turns":5,"prompt_tokens":500,"reused_tokens":400,"rejected_tier_accesses":50}
`
		mixedPath := filepath.Join(tmpDir, "mixed.jsonl")
		if err := os.WriteFile(mixedPath, []byte(mixedContent), 0644); err != nil {
			t.Fatalf("write mixed: %v", err)
		}

		p1 := ParseLedger(mixedContent)
		p2 := ParseLedger(mixedContent)
		if !reflect.DeepEqual(p1, p2) {
			t.Fatalf("ParseLedger mixed diverged")
		}
		if len(p1) != 3 {
			t.Fatalf("expected 3 valid rows from mixed content, got %d", len(p1))
		}
		if p1[0].RejectedTierAccesses != ^uint64(0) {
			t.Fatalf("max uint64 rejected_tier_accesses = %d, want %d", p1[0].RejectedTierAccesses, ^uint64(0))
		}

		s1, err1 := ScoreLedger(mixedPath)
		s2, err2 := ScoreLedger(mixedPath)
		if err1 != nil || err2 != nil {
			t.Fatalf("ScoreLedger mixed error: %v, %v", err1, err2)
		}
		if !reflect.DeepEqual(s1, s2) {
			t.Fatalf("ScoreLedger mixed diverged")
		}

		t1 := FoldTrendGate(p1)
		t2 := FoldTrendGate(p2)
		if !reflect.DeepEqual(t1, t2) {
			t.Fatalf("FoldTrendGate mixed diverged")
		}

		ft1 := ScoreTrendGate(mixedPath)
		ft2 := ScoreTrendGate(mixedPath)
		if !reflect.DeepEqual(ft1, ft2) {
			t.Fatalf("ScoreTrendGate mixed diverged")
		}
	})
}

// TestDeterminismConcurrentRaceWitness executes concurrent goroutines performing
// parsing, serializing, scoring, and folding across independent copies and read-only
// shared fixtures with sync.WaitGroup to witness race-freedom and absolute determinism.
func TestDeterminismConcurrentRaceWitness(t *testing.T) {
	fixtureContent := `{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"w1","unix_millis":1000,"turns":5,"prompt_tokens":1000,"reused_tokens":800,"rejected_tier_accesses":10}
{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"w2","unix_millis":2000,"turns":5,"prompt_tokens":1000,"reused_tokens":800,"rejected_tier_accesses":20}
{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"run","context":"w3","unix_millis":2500,"turns":1,"prompt_tokens":50,"reused_tokens":0,"rejected_tier_accesses":1}
{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"w4","unix_millis":3000,"turns":5,"prompt_tokens":1000,"reused_tokens":300,"rejected_tier_accesses":30}
{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"serve","context":"w5","unix_millis":4000,"turns":5,"prompt_tokens":1000,"reused_tokens":300,"rejected_tier_accesses":40}
{"schema":"fak-cache-value-ledger/1","date":"2026-08-31","session_type":"guard","context":"w6","unix_millis":5000,"turns":8,"prompt_tokens":800,"reused_tokens":640,"rejected_tier_accesses":50}
`
	tmpDir := t.TempDir()
	sharedLedgerPath := filepath.Join(tmpDir, "shared-ledger.jsonl")
	if err := os.WriteFile(sharedLedgerPath, []byte(fixtureContent), 0644); err != nil {
		t.Fatalf("write shared ledger: %v", err)
	}

	// Reference single-threaded artifacts
	refRows := ParseLedger(fixtureContent)
	refRowsJSON, err := json.Marshal(refRows)
	if err != nil {
		t.Fatalf("marshal refRows: %v", err)
	}

	refRow := refRows[0]
	refLine, err := AppendLedgerLine(refRow)
	if err != nil {
		t.Fatalf("AppendLedgerLine refRow: %v", err)
	}

	refScore, err := ScoreLedger(sharedLedgerPath)
	if err != nil {
		t.Fatalf("ScoreLedger ref: %v", err)
	}
	refScoreJSON, err := json.Marshal(refScore)
	if err != nil {
		t.Fatalf("marshal refScore: %v", err)
	}

	refTrend := FoldTrendGate(refRows)
	refTrendJSON, err := json.Marshal(refTrend)
	if err != nil {
		t.Fatalf("marshal refTrend: %v", err)
	}

	refFileTrend := ScoreTrendGate(sharedLedgerPath)
	refFileTrendJSON, err := json.Marshal(refFileTrend)
	if err != nil {
		t.Fatalf("marshal refFileTrend: %v", err)
	}

	const workers = 32
	const iterations = 25
	var wg sync.WaitGroup
	start := make(chan struct{})
	errCh := make(chan error, workers)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			<-start

			for iter := 0; iter < iterations; iter++ {
				// 1. Concurrent parsing of shared string
				gotRows := ParseLedger(fixtureContent)
				if !reflect.DeepEqual(gotRows, refRows) {
					errCh <- fmt.Errorf("worker %d iter %d: ParseLedger diverged from reference", workerID, iter)
					return
				}
				gotRowsJSON, err := json.Marshal(gotRows)
				if err != nil {
					errCh <- fmt.Errorf("worker %d iter %d: marshal gotRows: %w", workerID, iter, err)
					return
				}
				if !bytes.Equal(gotRowsJSON, refRowsJSON) {
					errCh <- fmt.Errorf("worker %d iter %d: ParseLedger JSON bytes diverged", workerID, iter)
					return
				}

				// 2. Concurrent serialization of shared row
				gotLine, err := AppendLedgerLine(refRow)
				if err != nil {
					errCh <- fmt.Errorf("worker %d iter %d: AppendLedgerLine error: %w", workerID, iter, err)
					return
				}
				if gotLine != refLine {
					errCh <- fmt.Errorf("worker %d iter %d: AppendLedgerLine output diverged", workerID, iter)
					return
				}

				// 3. Concurrent scoring of shared file
				gotScore, err := ScoreLedger(sharedLedgerPath)
				if err != nil {
					errCh <- fmt.Errorf("worker %d iter %d: ScoreLedger error: %w", workerID, iter, err)
					return
				}
				if !reflect.DeepEqual(gotScore, refScore) {
					errCh <- fmt.Errorf("worker %d iter %d: ScoreLedger diverged from reference", workerID, iter)
					return
				}
				gotScoreJSON, err := json.Marshal(gotScore)
				if err != nil {
					errCh <- fmt.Errorf("worker %d iter %d: marshal gotScore: %w", workerID, iter, err)
					return
				}
				if !bytes.Equal(gotScoreJSON, refScoreJSON) {
					errCh <- fmt.Errorf("worker %d iter %d: ScoreLedger JSON bytes diverged", workerID, iter)
					return
				}

				// 4. Concurrent folding of shared rows slice
				gotTrend := FoldTrendGate(refRows)
				if !reflect.DeepEqual(gotTrend, refTrend) {
					errCh <- fmt.Errorf("worker %d iter %d: FoldTrendGate diverged from reference", workerID, iter)
					return
				}
				gotTrendJSON, err := json.Marshal(gotTrend)
				if err != nil {
					errCh <- fmt.Errorf("worker %d iter %d: marshal gotTrend: %w", workerID, iter, err)
					return
				}
				if !bytes.Equal(gotTrendJSON, refTrendJSON) {
					errCh <- fmt.Errorf("worker %d iter %d: FoldTrendGate JSON bytes diverged", workerID, iter)
					return
				}

				// 5. Concurrent trend scoring of shared file
				gotFileTrend := ScoreTrendGate(sharedLedgerPath)
				if !reflect.DeepEqual(gotFileTrend, refFileTrend) {
					errCh <- fmt.Errorf("worker %d iter %d: ScoreTrendGate diverged from reference", workerID, iter)
					return
				}
				gotFileTrendJSON, err := json.Marshal(gotFileTrend)
				if err != nil {
					errCh <- fmt.Errorf("worker %d iter %d: marshal gotFileTrend: %w", workerID, iter, err)
					return
				}
				if !bytes.Equal(gotFileTrendJSON, refFileTrendJSON) {
					errCh <- fmt.Errorf("worker %d iter %d: ScoreTrendGate JSON bytes diverged", workerID, iter)
					return
				}

				// 6. Concurrent edge cases
				if emptyRows := ParseLedger(""); len(emptyRows) != 0 {
					errCh <- fmt.Errorf("worker %d iter %d: ParseLedger empty returned non-zero rows", workerID, iter)
					return
				}
				if emptyTrend := FoldTrendGate(nil); emptyTrend.Verdict != "INSUFFICIENT" || !emptyTrend.OK {
					errCh <- fmt.Errorf("worker %d iter %d: FoldTrendGate nil unexpected result: %+v", workerID, iter, emptyTrend)
					return
				}
			}
		}(w)
	}

	close(start)
	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatal(err)
	}
}
