package gateway

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestTurnLineSplitsPastBudgetNonFire is the diagnostic half of the compaction-solvency work. A
// bare `compact=none` conflated two OPPOSITE readings — "resident is under the budget line, nothing
// to shed" (benign) and "resident is already past the line and the burst gate refused anyway"
// (the pathology). That conflation is precisely how the problem hid: across 3191 real served turns
// 1622 ran over the line without firing (median 23.4k over, max 97.6k over) and every one of them
// rendered as an unremarkable `compact=none`, while the same runs reported "14 fired, shed 241349
// tokens" — which reads as success. The past-budget non-fire must carry its own label.
func TestTurnLineSplitsPastBudgetNonFire(t *testing.T) {
	const budget = 96000
	// resident = prompt + cacheRead + cacheCreate (the same sum compactionBudgetPast folds).
	for _, tc := range []struct {
		name                 string
		prompt, read, create int
		compacted            bool
		wantField            string
		wantAbsent           string
	}{
		{"under the line, did not fire — benign", 20000, 30000, 0, false, "compact=none ", "none-past-budget"},
		{"past the line, did not fire — the alarm", 20000, 120000, 0, false, "compact=none-past-budget ", ""},
		{"past the line and fired — still just fired", 20000, 120000, 0, true, "compact=fired ", "none-past-budget"},
		{"exactly at the line counts as past", 96000, 0, 0, false, "compact=none-past-budget ", ""},
	} {
		line := formatTurnDebugStatsWithBudget("t1", "wire", false, "end_turn",
			tc.prompt, 100, tc.read, tc.create, tc.compacted, budget, ResetDecision{}, false)
		if !strings.Contains(line, tc.wantField) {
			t.Fatalf("%s: line missing %q\n  got: %s", tc.name, tc.wantField, line)
		}
		if tc.wantAbsent != "" && strings.Contains(line, tc.wantAbsent) {
			t.Fatalf("%s: line must not contain %q\n  got: %s", tc.name, tc.wantAbsent, line)
		}
	}

	// With the lever OFF (budget 0) there is no line to be past, so the label never appears —
	// a disabled compactor must not manufacture an alarm.
	off := formatTurnDebugStatsWithBudget("t1", "wire", false, "end_turn", 200000, 100, 0, 0, false, 0, ResetDecision{}, false)
	if strings.Contains(off, "none-past-budget") {
		t.Fatalf("budget 0 (lever off) must render a plain compact=none, got: %s", off)
	}
}

// TestSolvencyForcedFiresCountedApart proves a forced (deliberately unprofitable) fire is counted
// as a SUBSET of fired rather than replacing it, so the exit summary can say "x of N fired" and the
// cache-value ledger never books survival spending as a cache win.
func TestSolvencyForcedFiresCountedApart(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeCompaction(agent.CompactOutcome{Reason: agent.CompactReasonNone, Dropped: 3, ShedTokens: 700}, false)
	m.observeCompaction(agent.CompactOutcome{Reason: agent.CompactReasonNone, Dropped: 5, ShedTokens: 1100, SolvencyForced: true}, false)
	m.observeCompaction(agent.CompactOutcome{Reason: agent.CompactReasonBurstUnprofitable}, false)

	s := m.adjudicationSummary()
	if s.CompactionFired != 2 {
		t.Fatalf("a forced fire is still a fire: CompactionFired = %d, want 2", s.CompactionFired)
	}
	if s.CompactionSolvencyForced != 1 {
		t.Fatalf("CompactionSolvencyForced = %d, want 1", s.CompactionSolvencyForced)
	}
	if s.CompactionShedTokens != 1800 || s.CompactionDroppedTurns != 8 {
		t.Fatalf("forced fires must contribute their real work: shed=%d dropped=%d, want 1800/8",
			s.CompactionShedTokens, s.CompactionDroppedTurns)
	}
	if s.CompactionBailed != 1 {
		t.Fatalf("CompactionBailed = %d, want 1", s.CompactionBailed)
	}
}

// TestSolvencyFloorReachesTheCompactor wires Config → Server → CompactOptions, so the operator flag
// is provably not inert. Without this the floor could be plumbed everywhere except the one call
// that consults it and every other test would still pass.
func TestSolvencyFloorReachesTheCompactor(t *testing.T) {
	s := newTestServerWithConfig(t, Config{
		EngineID: "test", Model: "test-model",
		CompactHistoryBudget:       10000,
		CompactAnchorHead:          true,
		CompactSolvencyFloorTokens: 143000,
	})
	if s.compactSolvencyFloorTokens != 143000 {
		t.Fatalf("Config.CompactSolvencyFloorTokens did not reach the server: got %d, want 143000",
			s.compactSolvencyFloorTokens)
	}
	// The default stays disarmed, so every caller that does not opt in keeps pure economics.
	plain := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model", CompactHistoryBudget: 10000})
	if plain.compactSolvencyFloorTokens != 0 {
		t.Fatalf("the solvency floor must default to disarmed, got %d", plain.compactSolvencyFloorTokens)
	}
}
