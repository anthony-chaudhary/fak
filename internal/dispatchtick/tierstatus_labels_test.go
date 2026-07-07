package dispatchtick

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// TestTierDecisionRowFromLabels checks the C4->C8 join: when an input carries raw
// GitHub Labels, BuildTierDecisionRow derives the tier from them (the real
// dispatcher signal) and surfaces the tag flags on the row, so the readout shows
// tagging HEALTH alongside the routing verdict.
func TestTierDecisionRowFromLabels(t *testing.T) {
	all := []AccountRow{acct("frontier", 1, true), acct("mid", 2, true), acct("small", 3, true)}

	// Clean routine tags -> cheapest seat that meets the T2 floor, no flags.
	clean := BuildTierDecisionRow(TierDecisionInput{
		Issue: 200, Lane: "docs",
		Labels: []string{"tier/T2-required", "tier/T2-optimal"},
		Rows:   all,
	})
	if clean.RequiredTier != modelroute.TierT2 || clean.ChosenModelTier != 3 {
		t.Fatalf("clean routine labels should route to the tier-3 seat at floor T2, got %+v", clean)
	}
	if len(clean.TagFlags) != 0 {
		t.Fatalf("clean labels must carry no tag flags, got %v", clean.TagFlags)
	}

	// Contradictory tags (optimal weaker than the required floor) -> HasTier=false ->
	// conservative frontier floor, AND the row names the contradiction.
	contra := BuildTierDecisionRow(TierDecisionInput{
		Issue: 201, Lane: "release",
		Labels: []string{"tier/T0-required", "tier/T1-optimal"},
		Rows:   all,
	})
	if contra.RequiredTier != modelroute.TierT0 || contra.ChosenModelTier != 1 {
		t.Fatalf("contradictory labels must fall back to the frontier floor, got %+v", contra)
	}
	if contra.Reason != TierReasonUntaggedConservative {
		t.Fatalf("conservative route reason = %q, want %q", contra.Reason, TierReasonUntaggedConservative)
	}
	if !hasFlag(contra.TagFlags, TagFlagContradiction) {
		t.Fatalf("contradictory tags must be named on the row, got flags %v", contra.TagFlags)
	}

	// A half-tagged issue (only required) stays conservative and names the gap.
	half := BuildTierDecisionRow(TierDecisionInput{
		Issue: 202, Lane: "tools",
		Labels: []string{"tier/T2-required"},
		Rows:   all,
	})
	if half.ChosenModelTier != 1 {
		t.Fatalf("a half-tagged issue must not route below the frontier floor, got %+v", half)
	}
	if !hasFlag(half.TagFlags, TagFlagOptimalMissing) {
		t.Fatalf("half-tagged issue must name the missing optimal tag, got %v", half.TagFlags)
	}

	// Back-compat: no Labels -> the explicit Tier is used, no flags stamped.
	explicit := BuildTierDecisionRow(TierDecisionInput{
		Issue: 203, Lane: "gateway",
		Tier: tagged(modelroute.TierT1, modelroute.TierT1),
		Rows: all,
	})
	if explicit.ChosenModelTier != 2 || len(explicit.TagFlags) != 0 {
		t.Fatalf("explicit-Tier input must route by Tier with no flags, got %+v", explicit)
	}
}

func hasFlag(flags []string, want string) bool {
	for _, f := range flags {
		if f == want {
			return true
		}
	}
	return false
}
