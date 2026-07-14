package gatewayusageledger

import (
	"strings"
	"testing"
	"time"
)

// TestFoldCompactionSegmentsByRegime pins the compaction-by-regime fold against the exact
// false-alarm it exists to prevent: an interactive-48k long band that sheds heavily and a
// headless-96k long band that correctly bails `under_budget` must land in SEPARATE
// segments (blending them reads the deliberate budget switch as a regression), and a
// phantom-100% row (shed>0 but cached+input==0) must be QUARANTINED, not folded into the
// shed percentile.
func TestFoldCompactionSegmentsByRegime(t *testing.T) {
	now := time.Unix(4000, 0)
	b48 := &Provenance{CompactHistoryBudget: 48000}
	b96 := &Provenance{CompactHistoryBudget: 96000}

	rows := []Row{
		// 48k regime, 40-80 band: two fired rows with valid denominators → shed 70% and 80%.
		NewRow("exit", "guard", "claude", "", 0, b48, Counters{
			ObservedTurns: 50, CompactionFired: 100, CompactionShedTokens: 700, CachedPromptTokens: 300,
		}, now),
		NewRow("exit", "guard", "claude", "", 0, b48, Counters{
			ObservedTurns: 60, CompactionFired: 100, CompactionShedTokens: 800, CachedPromptTokens: 200,
		}, now),
		// 96k regime, 40-80 band: an all-bail row (0 fired) ...
		NewRow("exit", "guard", "claude", "", 0, b96, Counters{
			ObservedTurns: 50, CompactionBailed: 64,
			CompactionBailReasons: map[string]uint64{"under_budget": 40, "burst_unprofitable": 24},
		}, now),
		// ... and a phantom-100% row in the SAME cell: fired, shed>0, but no cached/input to divide by.
		NewRow("exit", "guard", "claude", "", 0, b96, Counters{
			ObservedTurns: 55, CompactionFired: 5, CompactionShedTokens: 1000,
		}, now),
		// 48k regime, 0-20 band: a short fired session (banding check).
		NewRow("exit", "guard", "claude", "", 0, b48, Counters{
			ObservedTurns: 5, CompactionFired: 3, CompactionShedTokens: 100, CachedPromptTokens: 900,
		}, now),
		// A periodic (non-exit) snapshot must be ignored so a live session is not double-counted.
		NewRow("periodic", "serve", "http", "", 0, b96, Counters{
			ObservedTurns: 70, CompactionFired: 99, CompactionShedTokens: 9999, CachedPromptTokens: 1,
		}, now),
	}

	rep := FoldCompaction(rows, "2026-07-09")

	if rep.ExitRows != 5 {
		t.Fatalf("ExitRows = %d, want 5 (periodic row must be ignored)", rep.ExitRows)
	}
	if rep.QuarantinedRows != 1 {
		t.Fatalf("QuarantinedRows = %d, want 1 (the phantom-100%% row)", rep.QuarantinedRows)
	}
	if rep.Since != "2026-07-09" {
		t.Fatalf("Since = %q, want 2026-07-09", rep.Since)
	}

	seg48long := findSeg(t, rep, 48000, "40-80")
	if seg48long.Sessions != 2 || seg48long.FiredSessions != 2 || seg48long.ValidDenomRows != 2 {
		t.Fatalf("48k/40-80 counts = sess %d fired %d valid %d, want 2/2/2", seg48long.Sessions, seg48long.FiredSessions, seg48long.ValidDenomRows)
	}
	if seg48long.ShedPctMedian != 75.0 {
		t.Fatalf("48k/40-80 ShedPctMedian = %.3f, want 75.0", seg48long.ShedPctMedian)
	}
	if seg48long.BudgetRegime != "interactive" {
		t.Fatalf("48k regime label = %q, want interactive", seg48long.BudgetRegime)
	}
	if seg48long.TopBailReason != "" {
		t.Fatalf("48k/40-80 TopBailReason = %q, want empty (nothing bailed)", seg48long.TopBailReason)
	}

	seg96long := findSeg(t, rep, 96000, "40-80")
	if seg96long.Sessions != 2 || seg96long.FiredSessions != 1 {
		t.Fatalf("96k/40-80 sess/fired = %d/%d, want 2/1", seg96long.Sessions, seg96long.FiredSessions)
	}
	if seg96long.ValidDenomRows != 0 || seg96long.DenomZeroRows != 1 {
		t.Fatalf("96k/40-80 valid/denom0 = %d/%d, want 0/1 (fired row was quarantined, not folded)", seg96long.ValidDenomRows, seg96long.DenomZeroRows)
	}
	if seg96long.ShedPctMedian != 0 {
		t.Fatalf("96k/40-80 ShedPctMedian = %.3f, want 0 (no valid-denom rows → honest absence)", seg96long.ShedPctMedian)
	}
	if seg96long.Fires != 5 || seg96long.Bails != 64 {
		t.Fatalf("96k/40-80 fires/bails = %d/%d, want 5/64", seg96long.Fires, seg96long.Bails)
	}
	if seg96long.TopBailReason != "under_budget" {
		t.Fatalf("96k/40-80 TopBailReason = %q, want under_budget", seg96long.TopBailReason)
	}
	if seg96long.BudgetRegime != "headless" {
		t.Fatalf("96k regime label = %q, want headless", seg96long.BudgetRegime)
	}

	// Segment ordering: budget ascending, then canonical band order (48k/0-20 before 48k/40-80 before 96k/40-80).
	wantOrder := [][2]interface{}{{48000, "0-20"}, {48000, "40-80"}, {96000, "40-80"}}
	if len(rep.Segments) != len(wantOrder) {
		t.Fatalf("segment count = %d, want %d", len(rep.Segments), len(wantOrder))
	}
	for i, w := range wantOrder {
		if rep.Segments[i].Budget != w[0].(int) || rep.Segments[i].Band != w[1].(string) {
			t.Fatalf("segment[%d] = (%d,%s), want (%v,%v)", i, rep.Segments[i].Budget, rep.Segments[i].Band, w[0], w[1])
		}
	}

	// Render: the 48k/40-80 cell shows a percentage, the all-bail/quarantined 96k cell a dash.
	out := RenderCompaction(rep)
	if !strings.Contains(out, "75.0%") {
		t.Fatalf("render missing 48k shed percentile:\n%s", out)
	}
	if !strings.Contains(out, "quarantined") {
		t.Fatalf("render missing quarantine note:\n%s", out)
	}
	if !strings.Contains(out, "interactive(48000)") || !strings.Contains(out, "headless(96000)") {
		t.Fatalf("render missing regime labels:\n%s", out)
	}
}

// TestRenderCompactionEmpty — a report with no exit rows renders the live-ledger hint,
// never a blank or a crash, so a wrong/stale ledger path is diagnosable at a glance.
func TestRenderCompactionEmpty(t *testing.T) {
	out := RenderCompaction(FoldCompaction(nil, ""))
	if !strings.Contains(out, "no exit rows") {
		t.Fatalf("empty render missing hint:\n%s", out)
	}
}

func findSeg(t *testing.T, rep CompactionReport, budget int, band string) CompactionSegment {
	t.Helper()
	for _, s := range rep.Segments {
		if s.Budget == budget && s.Band == band {
			return s
		}
	}
	t.Fatalf("segment (%d,%s) not found in %+v", budget, band, rep.Segments)
	return CompactionSegment{}
}
