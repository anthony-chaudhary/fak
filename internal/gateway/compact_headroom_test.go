package gateway

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestUnderBudgetBailRecordsHeadroom proves the compactible-span witness survives the bail. The
// compactor computes SuffixTokens on every under_budget bail and the metrics layer used to discard
// it, which left "under_budget xN" unreadable: a short session and a session whose span can never
// reach the line produce the identical counter. Recording last+peak turns the reason into a
// distance from firing.
func TestUnderBudgetBailRecordsHeadroom(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeCompaction(agent.CompactOutcome{Reason: agent.CompactReasonUnderBudget, SuffixTokens: 51000}, false)
	m.observeCompaction(agent.CompactOutcome{Reason: agent.CompactReasonUnderBudget, SuffixTokens: 56000}, false)
	m.observeCompaction(agent.CompactOutcome{Reason: agent.CompactReasonUnderBudget, SuffixTokens: 52000}, false)

	snap := m.compactionSnapshotData()
	if snap.lastSuffixTokens != 52000 {
		t.Fatalf("lastSuffixTokens = %d, want 52000 (the MOST RECENT bail, not the largest)", snap.lastSuffixTokens)
	}
	if snap.peakSuffixTokens != 56000 {
		t.Fatalf("peakSuffixTokens = %d, want 56000 (the high-water span this traffic reached)", snap.peakSuffixTokens)
	}
}

// TestPreEligibilityBailsLeaveHeadroomUntouched pins the one way the witness could lie. A bail that
// resolved no compactible span at all (too_few_msgs and the other pre-eligibility reasons) reports
// SuffixTokens 0; folding that in would drag `last` to zero and read as "0 tokens from firing" —
// the exact opposite of the truth — on any gateway whose traffic mixes short requests with long ones.
func TestPreEligibilityBailsLeaveHeadroomUntouched(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeCompaction(agent.CompactOutcome{Reason: agent.CompactReasonUnderBudget, SuffixTokens: 48000}, false)
	m.observeCompaction(agent.CompactOutcome{Reason: agent.CompactReasonTooFewMsgs}, false)

	snap := m.compactionSnapshotData()
	if snap.lastSuffixTokens != 48000 || snap.peakSuffixTokens != 48000 {
		t.Fatalf("a span-less bail must not overwrite the witness: last=%d peak=%d, want 48000/48000",
			snap.lastSuffixTokens, snap.peakSuffixTokens)
	}
}

// TestDebugVarsSurfacesCompactionHeadroomAndAnchorStarved proves the JSON front door carries what
// only the Prometheus surface used to. anchor_starved in particular was renderable at /metrics but
// absent from /debug/vars, so the operator tools that poll the JSON could not separate the #1407
// anchor-starved pathology from a benign short session.
func TestDebugVarsSurfacesCompactionHeadroomAndAnchorStarved(t *testing.T) {
	s, err := New(Config{EngineID: "mock", Model: "test-model", CompactHistoryBudget: 96000})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	s.metrics.observeCompaction(agent.CompactOutcome{
		Reason: agent.CompactReasonUnderBudget, SuffixTokens: 51000, ProtectedPrefixTokens: 120000, AnchorStarved: true,
	}, false)

	got := s.debugVars(time.Now()).Metrics.Compaction
	if got.Budget != 96000 {
		t.Fatalf("budget = %d, want the RESOLVED 96000 (the flag help advertises 48000, which guard overrides)", got.Budget)
	}
	if got.LastSuffixTokens != 51000 {
		t.Fatalf("last_suffix_tokens = %d, want 51000", got.LastSuffixTokens)
	}
	if got.AnchorStarved != 1 {
		t.Fatalf("anchor_starved = %d, want 1 — the field /debug/vars previously omitted entirely", got.AnchorStarved)
	}
	if got.BailReasons[agent.CompactReasonUnderBudget] != 1 {
		t.Fatalf("anchor-starved must stay a SUBSET of under_budget, not replace it: %v", got.BailReasons)
	}
}
