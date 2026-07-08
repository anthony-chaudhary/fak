package cachevaluereport

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// clusterFor finds the (length, outcome) cluster in a report, or nil if absent.
func clusterFor(r ShapeReport, l LengthBand, o OutcomeBand) *ShapeCluster {
	for i := range r.Clusters {
		if r.Clusters[i].Length == l && r.Clusters[i].Outcome == o {
			return &r.Clusters[i]
		}
	}
	return nil
}

func TestFoldShapes_EmptyIsInsufficientButOK(t *testing.T) {
	r := FoldShapes(nil, fixedNow)
	if !r.OK {
		t.Fatalf("empty shape roll-up should be OK (a report, not a gate); got OK=false")
	}
	if r.Verdict != "INSUFFICIENT" {
		t.Fatalf("empty shape roll-up verdict = %q, want INSUFFICIENT", r.Verdict)
	}
	if len(r.Clusters) != 0 || r.TotalSessions != 0 {
		t.Fatalf("empty roll-up should have no clusters/sessions; got %d clusters, %d sessions", len(r.Clusters), r.TotalSessions)
	}
	if !r.VsNaiveMultipleExcluded || r.PublishableValueFamily == "" {
		t.Fatalf("#1066 fence self-labels missing: excluded=%v family=%q", r.VsNaiveMultipleExcluded, r.PublishableValueFamily)
	}
	if r.Schema != ShapeSchema {
		t.Fatalf("schema = %q, want %q", r.Schema, ShapeSchema)
	}
}

func TestFoldShapes_SingleTurnIsNAOutcomeNotCold(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekAEarly, SessionType: "run", Turns: 1, PromptTokens: 500, ReusedTokens: 0, ReuseRatio: 0},
		{Date: weekALate, SessionType: "run", Turns: 1, PromptTokens: 500, ReusedTokens: 0, ReuseRatio: 0},
	}
	r := FoldShapes(rows, fixedNow)
	if r.Verdict != "INSUFFICIENT" {
		t.Fatalf("single-turn-only verdict = %q, want INSUFFICIENT (no multi-turn shape)", r.Verdict)
	}
	if r.SingleTurnSessions != 2 || r.MultiTurnSessions != 0 {
		t.Fatalf("single/multi counts = %d/%d, want 2/0", r.SingleTurnSessions, r.MultiTurnSessions)
	}
	if c := clusterFor(r, LengthSingle, OutcomeNA); c == nil || c.Sessions != 2 {
		t.Fatalf("want a single/n-a cluster with 2 sessions; got %+v", c)
	}
	if clusterFor(r, LengthSingle, OutcomeCold) != nil {
		t.Fatalf("single-turn rows must never land in a cold cluster (structurally-impossible reuse)")
	}
}

func TestFoldShapes_LengthAndOutcomeBanding(t *testing.T) {
	rows := []cachevalueledger.Row{
		// short (2..4 turns), warm (reuse >= 0.5)
		{Date: weekAEarly, SessionType: "guard", Turns: 3, PromptTokens: 1000, ReusedTokens: 700, ReuseRatio: 0.70},
		// long (>= 5 turns), partial (0.1..0.5)
		{Date: weekAEarly, SessionType: "serve", Turns: 8, PromptTokens: 2000, ReusedTokens: 600, ReuseRatio: 0.30},
		// long, cold (< 0.1)
		{Date: weekALate, SessionType: "serve", Turns: 6, PromptTokens: 1000, ReusedTokens: 50, ReuseRatio: 0.05},
		// short, warm again — folds into the same cluster as row 1
		{Date: weekB, SessionType: "guard", Turns: 4, PromptTokens: 500, ReusedTokens: 300, ReuseRatio: 0.60},
	}
	r := FoldShapes(rows, fixedNow)
	if r.Verdict != "MEASURED" {
		t.Fatalf("verdict = %q, want MEASURED", r.Verdict)
	}
	if r.MultiTurnSessions != 4 {
		t.Fatalf("MultiTurnSessions = %d, want 4", r.MultiTurnSessions)
	}

	shortWarm := clusterFor(r, LengthShort, OutcomeWarm)
	if shortWarm == nil || shortWarm.Sessions != 2 {
		t.Fatalf("short/warm cluster = %+v, want 2 sessions", shortWarm)
	}
	// reuse ratio is the aggregate over the two short/warm rows: (700+300)/(1000+500).
	if got := shortWarm.RealizedReuseRatio; got < 0.66 || got > 0.67 {
		t.Fatalf("short/warm realized reuse = %.4f, want ~0.6667", got)
	}
	if shortWarm.BySessionType["guard"] != 2 {
		t.Fatalf("short/warm by session_type = %v, want guard:2", shortWarm.BySessionType)
	}

	if c := clusterFor(r, LengthLong, OutcomePartial); c == nil || c.Sessions != 1 {
		t.Fatalf("long/partial cluster = %+v, want 1 session", c)
	}
	if c := clusterFor(r, LengthLong, OutcomeCold); c == nil || c.Sessions != 1 {
		t.Fatalf("long/cold cluster = %+v, want 1 session", c)
	}
}

func TestFoldShapes_SharesSumToOne(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekAEarly, SessionType: "run", Turns: 1, PromptTokens: 100, ReusedTokens: 0, ReuseRatio: 0},
		{Date: weekAEarly, SessionType: "guard", Turns: 3, PromptTokens: 1000, ReusedTokens: 700, ReuseRatio: 0.70},
		{Date: weekALate, SessionType: "serve", Turns: 9, PromptTokens: 2000, ReusedTokens: 1200, ReuseRatio: 0.60},
	}
	r := FoldShapes(rows, fixedNow)
	var sessShare, reuseShare float64
	for _, c := range r.Clusters {
		sessShare += c.ShareOfSessions
		reuseShare += c.ShareOfReusedTokens
	}
	if sessShare < 0.999 || sessShare > 1.001 {
		t.Fatalf("session shares sum to %.4f, want ~1.0", sessShare)
	}
	// Only the two reuse-bearing clusters carry reused tokens; the single/n-a cluster's
	// share is 0, so the reuse shares still sum to 1.
	if reuseShare < 0.999 || reuseShare > 1.001 {
		t.Fatalf("reused-token shares sum to %.4f, want ~1.0", reuseShare)
	}
}

func TestFoldShapes_ZeroTurnRowsSkipped(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekAEarly, SessionType: "guard", Turns: 0, PromptTokens: 999},
		{Date: weekAEarly, SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 600, ReuseRatio: 0.60},
	}
	r := FoldShapes(rows, fixedNow)
	if r.TotalSessions != 1 {
		t.Fatalf("TotalSessions = %d, want 1 (zero-turn row skipped)", r.TotalSessions)
	}
	if c := clusterFor(r, LengthLong, OutcomeWarm); c == nil || c.Sessions != 1 {
		t.Fatalf("want a long/warm cluster with 1 session; got %+v", c)
	}
}

func TestFoldShapes_HealthClassification(t *testing.T) {
	rows := []cachevalueledger.Row{
		// single-turn cold run → n/a outcome, earning health (structurally reuse-free, not a failure)
		{Date: weekAEarly, SessionType: "run", Turns: 1, PromptTokens: 100, ReusedTokens: 0, ReuseRatio: 0},
		// long × cold → the expensive WASTEFUL failure shape
		{Date: weekAEarly, SessionType: "serve", Turns: 9, PromptTokens: 2000, ReusedTokens: 50, ReuseRatio: 0.02},
		// long × partial → underwarmed near-miss
		{Date: weekALate, SessionType: "serve", Turns: 7, PromptTokens: 1000, ReusedTokens: 300, ReuseRatio: 0.30},
		// short × cold → weak (cheap, low stakes)
		{Date: weekALate, SessionType: "guard", Turns: 3, PromptTokens: 500, ReusedTokens: 10, ReuseRatio: 0.02},
		// long × warm → earning
		{Date: weekB, SessionType: "guard", Turns: 8, PromptTokens: 1000, ReusedTokens: 800, ReuseRatio: 0.80},
	}
	r := FoldShapes(rows, fixedNow)

	wants := []struct {
		l LengthBand
		o OutcomeBand
		h ShapeHealth
	}{
		{LengthSingle, OutcomeNA, HealthEarning},
		{LengthLong, OutcomeCold, HealthWasteful},
		{LengthLong, OutcomePartial, HealthUnderwarmed},
		{LengthShort, OutcomeCold, HealthWeak},
		{LengthLong, OutcomeWarm, HealthEarning},
	}
	for _, w := range wants {
		c := clusterFor(r, w.l, w.o)
		if c == nil {
			t.Fatalf("missing %s/%s cluster", w.l, w.o)
		}
		if c.Health != w.h {
			t.Fatalf("%s/%s health = %q, want %q", w.l, w.o, c.Health, w.h)
		}
	}

	if r.WastefulSessions != 1 {
		t.Fatalf("WastefulSessions = %d, want 1 (the long×cold row)", r.WastefulSessions)
	}
	if r.WastefulSessionShare < 0.19 || r.WastefulSessionShare > 0.21 {
		t.Fatalf("WastefulSessionShare = %.4f, want ~0.20 (1 of 5)", r.WastefulSessionShare)
	}
	if !strings.Contains(r.NextAction, "ran cold") {
		t.Fatalf("NextAction should call out the wasteful shape; got %q", r.NextAction)
	}
}

func TestFoldShapes_NoWastefulWhenAllWarm(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekAEarly, SessionType: "guard", Turns: 8, PromptTokens: 1000, ReusedTokens: 800, ReuseRatio: 0.80},
	}
	r := FoldShapes(rows, fixedNow)
	if r.WastefulSessions != 0 {
		t.Fatalf("WastefulSessions = %d, want 0", r.WastefulSessions)
	}
	if strings.Contains(r.NextAction, "ran cold") {
		t.Fatalf("NextAction should not mention wasteful shape when none exist; got %q", r.NextAction)
	}
}

func TestFoldShapes_ClustersAreOrdered(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekB, SessionType: "serve", Turns: 8, PromptTokens: 2000, ReusedTokens: 1400, ReuseRatio: 0.70},
		{Date: weekAEarly, SessionType: "run", Turns: 1, PromptTokens: 100, ReusedTokens: 0, ReuseRatio: 0},
		{Date: weekALate, SessionType: "guard", Turns: 3, PromptTokens: 1000, ReusedTokens: 200, ReuseRatio: 0.20},
	}
	r := FoldShapes(rows, fixedNow)
	// Expect length-major (single, short, long) ordering regardless of input order.
	wantLengths := []LengthBand{LengthSingle, LengthShort, LengthLong}
	if len(r.Clusters) != len(wantLengths) {
		t.Fatalf("cluster count = %d, want %d", len(r.Clusters), len(wantLengths))
	}
	for i, want := range wantLengths {
		if r.Clusters[i].Length != want {
			t.Fatalf("cluster[%d].Length = %q, want %q (unstable ordering)", i, r.Clusters[i].Length, want)
		}
	}
}
