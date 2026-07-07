package dispatchtick

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// fixtureTierDecisions builds the canonical C8 readout corpus: one over-tier
// fallback, one under-tier refusal, an exact-optimal ship, a below-optimal saving,
// and an escalation — enough to exercise every counter and outcome bucket.
func fixtureTierDecisions() []TierDecisionInput {
	all := []AccountRow{acct("frontier", 1, true), acct("mid", 2, true), acct("small", 3, true)}
	return []TierDecisionInput{
		{ // exact optimal, shipped
			Issue: 100, Lane: "gateway",
			Tier:    tagged(modelroute.TierT1, modelroute.TierT1),
			Rows:    all,
			Outcome: TierOutcomeShipped,
		},
		{ // routine work, only a frontier seat free -> over-tier waste, shipped
			Issue: 101, Lane: "tools",
			Tier:    tagged(modelroute.TierT2, modelroute.TierT2),
			Rows:    []AccountRow{acct("frontier", 1, true), acct("mid", 2, false), acct("small", 3, false)},
			Outcome: TierOutcomeShipped,
		},
		{ // security floor T0 optimal, only a routine seat -> under-tier refusal
			Issue: 102, Lane: "release",
			Tier: tagged(modelroute.TierT1, modelroute.TierT0),
			Rows: []AccountRow{acct("small", 3, true)},
		},
		{ // security floor: required T1, optimal T0, frontier down -> mid, saving
			Issue: 103, Lane: "release",
			Tier:    tagged(modelroute.TierT1, modelroute.TierT0),
			Rows:    []AccountRow{acct("frontier", 1, false), acct("mid", 2, true)},
			Outcome: TierOutcomeShipped,
		},
		{ // chosen tier could not carry it -> escalation
			Issue: 104, Lane: "dispatchtick",
			Tier:      tagged(modelroute.TierT1, modelroute.TierT1),
			Rows:      all,
			Escalated: true,
			Outcome:   TierOutcomeEscalated,
		},
	}
}

// TestTierDecisionStatusReport is the C8 done-condition witness: a status readout
// over fixture workers that shows one over-tier fallback and one under-tier
// refusal, with the counters and modeled cost the working spine names.
func TestTierDecisionStatusReport(t *testing.T) {
	rep := BuildTierStatusReport(fixtureTierDecisions())

	if rep.Schema != TierStatusSchema {
		t.Fatalf("schema = %q, want %q", rep.Schema, TierStatusSchema)
	}
	if rep.Decisions != 5 {
		t.Fatalf("decisions = %d, want 5", rep.Decisions)
	}
	if rep.OverTier != 1 {
		t.Fatalf("over_tier = %d, want 1", rep.OverTier)
	}
	if rep.UnderTierRefused != 1 {
		t.Fatalf("under_tier_refused = %d, want 1", rep.UnderTierRefused)
	}
	if rep.Escalations != 1 {
		t.Fatalf("escalations = %d, want 1", rep.Escalations)
	}
	if rep.Routed != 4 {
		t.Fatalf("routed = %d, want 4 (5 decisions - 1 refusal)", rep.Routed)
	}
	if rep.CostProvenance != CostProvenanceModeled {
		t.Fatalf("cost_provenance = %q, want %q", rep.CostProvenance, CostProvenanceModeled)
	}
	if rep.Note == "" {
		t.Fatalf("report must carry the modeled-cost quality-parity caveat")
	}
}

// TestTierDecisionStatusOverTierRow checks the over-tier fallback row carries the
// verdict, the positive MODELED cost delta (waste), and the reason verbatim.
func TestTierDecisionStatusOverTierRow(t *testing.T) {
	rep := BuildTierStatusReport(fixtureTierDecisions())
	row := findTierRow(t, rep, 101)

	if !row.OverTier {
		t.Fatalf("issue 101 must be flagged over-tier: %+v", row)
	}
	if row.Reason != TierReasonOverTierFallback {
		t.Fatalf("reason = %q, want %q", row.Reason, TierReasonOverTierFallback)
	}
	if row.ChosenModelTier != 1 {
		t.Fatalf("over-tier fallback should land on the frontier seat (tier 1), got %d", row.ChosenModelTier)
	}
	// routine optimal (T2 -> model tier 3, cost 1) vs frontier (tier 1, cost 9) = +8 waste.
	if row.CostDeltaModeled <= 0 {
		t.Fatalf("over-tier row must model positive cost waste, got %d", row.CostDeltaModeled)
	}
	if rep.OverTierCostModeled != row.CostDeltaModeled {
		t.Fatalf("report over_tier_cost_modeled = %d, want the single over-tier row delta %d",
			rep.OverTierCostModeled, row.CostDeltaModeled)
	}
}

// TestTierDecisionStatusUnderTierRow checks the under-tier refusal row: no seat,
// no spend, the refused outcome, and the too-weak seat it turned away named.
func TestTierDecisionStatusUnderTierRow(t *testing.T) {
	rep := BuildTierStatusReport(fixtureTierDecisions())
	row := findTierRow(t, rep, 102)

	if !row.UnderTierRefused {
		t.Fatalf("issue 102 must be under-tier refused: %+v", row)
	}
	if row.Reason != TierReasonUnderTierRefused {
		t.Fatalf("reason = %q, want %q", row.Reason, TierReasonUnderTierRefused)
	}
	if row.Account != "" {
		t.Fatalf("a refusal must not name a chosen seat, got %q", row.Account)
	}
	if row.CostDeltaModeled != 0 {
		t.Fatalf("a refusal spends nothing, cost delta must be 0, got %d", row.CostDeltaModeled)
	}
	if row.Outcome != TierOutcomeRefused {
		t.Fatalf("refusal outcome = %q, want %q", row.Outcome, TierOutcomeRefused)
	}
	if len(row.Blocked) != 1 || row.Blocked[0] != "small" {
		t.Fatalf("refusal should name the below-floor seat it turned away, got %+v", row.Blocked)
	}
}

// TestTierDecisionStatusSavingsRow checks the below-optimal saving: the chosen
// mid seat is cheaper than the T0 optimal, so the MODELED delta is negative and
// rolls up into the report's savings column, kept separate from waste.
func TestTierDecisionStatusSavingsRow(t *testing.T) {
	rep := BuildTierStatusReport(fixtureTierDecisions())
	row := findTierRow(t, rep, 103)

	if row.OverTier || row.UnderTierRefused {
		t.Fatalf("issue 103 is a clean below-optimal saving: %+v", row)
	}
	if row.Reason != TierReasonCheaperThanOptimal {
		t.Fatalf("reason = %q, want %q", row.Reason, TierReasonCheaperThanOptimal)
	}
	if row.CostDeltaModeled >= 0 {
		t.Fatalf("below-optimal row must model a negative (saving) cost delta, got %d", row.CostDeltaModeled)
	}
	if rep.SavingsCostModeled != -row.CostDeltaModeled {
		t.Fatalf("report savings_cost_modeled = %d, want %d", rep.SavingsCostModeled, -row.CostDeltaModeled)
	}
	if rep.OverTierCostModeled != 8 {
		t.Fatalf("waste and savings must not net: over_tier_cost_modeled = %d, want 8", rep.OverTierCostModeled)
	}
}

// TestTierDecisionStatusOutcomeJoin checks the decision->outcome join: every
// outcome the fixtures carry lands in the tally, and the escalation both flags
// its row and increments the escalation counter.
func TestTierDecisionStatusOutcomeJoin(t *testing.T) {
	rep := BuildTierStatusReport(fixtureTierDecisions())

	if got := rep.OutcomeTally[TierOutcomeShipped]; got != 3 {
		t.Fatalf("shipped tally = %d, want 3", got)
	}
	if got := rep.OutcomeTally[TierOutcomeRefused]; got != 1 {
		t.Fatalf("refused tally = %d, want 1", got)
	}
	if got := rep.OutcomeTally[TierOutcomeEscalated]; got != 1 {
		t.Fatalf("escalated tally = %d, want 1", got)
	}
	esc := findTierRow(t, rep, 104)
	if !esc.Escalation || esc.Outcome != TierOutcomeEscalated {
		t.Fatalf("issue 104 must be an escalation: %+v", esc)
	}
}

// TestTierDecisionStatusRenderDeterministic checks the human render names the
// verdict, the modeled-cost caveat, and both the over-tier and refused rows.
func TestTierDecisionStatusRenderDeterministic(t *testing.T) {
	rep := BuildTierStatusReport(fixtureTierDecisions())
	out := rep.Render()
	for _, want := range []string{"OVER-TIER", "REFUSED", "ESCALATED", "modeled cost points", "note:"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
}

func findTierRow(t *testing.T, rep TierStatusReport, issue int) TierDecisionRow {
	t.Helper()
	for _, row := range rep.Rows {
		if row.Issue == issue {
			return row
		}
	}
	t.Fatalf("no row for issue %d in %+v", issue, rep.Rows)
	return TierDecisionRow{}
}
