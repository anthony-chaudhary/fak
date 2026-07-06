package modelroute

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelscore"
)

// The done condition: the policy returns required + optimal tiers plus reasons
// for the representative work classes, and the security/release class never
// floors below T1.
func TestTierPolicyWorkClassFloors(t *testing.T) {
	cases := []struct {
		class        WorkClass
		wantRequired WorkTier
		wantOptimal  WorkTier
	}{
		{ClassUltraHard, TierT0, TierT0},
		{ClassNormalImpl, TierT1, TierT1},
		{ClassRoutine, TierT2, TierT2},
		{ClassSecurityRelease, TierT1, TierT0}, // floor T1, optimal frontier
	}
	for _, tc := range cases {
		t.Run(string(tc.class), func(t *testing.T) {
			p := PolicyFor(tc.class)
			if p.RequiredTier != tc.wantRequired {
				t.Fatalf("%s required = %s, want %s", tc.class, p.RequiredTier, tc.wantRequired)
			}
			if p.OptimalTier != tc.wantOptimal {
				t.Fatalf("%s optimal = %s, want %s", tc.class, p.OptimalTier, tc.wantOptimal)
			}
			// Every fallback is at least as demanding as the floor — never below it.
			for _, f := range p.AllowedFallbacks {
				if !f.MeetsRequirement(p.RequiredTier) {
					t.Fatalf("%s fallback %s is below required floor %s", tc.class, f, p.RequiredTier)
				}
			}
		})
	}
}

// The security/release class specifically must NEVER drop to routine T2, however
// small the task looks.
func TestTierPolicySecurityNeverRoutine(t *testing.T) {
	p := PolicyFor(ClassSecurityRelease)
	if p.RequiredTier == TierT2 {
		t.Fatalf("security/release floor fell to T2")
	}
	if !contains(p.Reasons, ReasonSecurityFloor) {
		t.Fatalf("security policy missing %q reason: %v", ReasonSecurityFloor, p.Reasons)
	}
}

// An unknown class stays conservative (highest floor), never silently down-tiered.
func TestTierPolicyUnknownClassConservative(t *testing.T) {
	p := PolicyFor(WorkClass("mystery"))
	if p.RequiredTier != TierT0 {
		t.Fatalf("unknown class floor = %s, want T0 (conservative)", p.RequiredTier)
	}
	if !contains(p.Reasons, ReasonUnknownClass) {
		t.Fatalf("unknown class missing %q reason", ReasonUnknownClass)
	}
}

// Admit: under-tier is refused with a closed reason; over-tier is admitted and
// flagged as waste; an exact match is admitted as optimal. A high score cannot
// rescue an under-tier choice because capability, not score, is compared.
func TestTierPolicyAdmit(t *testing.T) {
	normal := PolicyFor(ClassNormalImpl) // required T1, optimal T1

	// A T2-only model (routine capability) cannot meet a T1 floor — refused.
	under := normal.Admit("small", TierT2)
	if under.Admitted {
		t.Fatalf("under-tier model admitted for normal work")
	}
	if !contains(under.Reasons, ReasonUnderTierRefused) {
		t.Fatalf("under-tier missing %q: %v", ReasonUnderTierRefused, under.Reasons)
	}

	// A T1 model is the optimal match.
	exact := normal.Admit("mid", TierT1)
	if !exact.Admitted || exact.OverTier || !contains(exact.Reasons, ReasonOptimalMatch) {
		t.Fatalf("T1 model should be optimal match, got %+v", exact)
	}

	// A T0 frontier model can do normal work but it is over-tier waste.
	over := normal.Admit("frontier", TierT0)
	if !over.Admitted || !over.OverTier || !contains(over.Reasons, ReasonOverTierWaste) {
		t.Fatalf("T0 on normal work should be admitted+waste, got %+v", over)
	}
}

// The floor invariant across every class and capability: Admit never returns
// admitted=true when the capability does not meet the required floor.
func TestTierPolicyNeverAdmitsBelowFloor(t *testing.T) {
	classes := []WorkClass{ClassUltraHard, ClassNormalImpl, ClassRoutine, ClassSecurityRelease, WorkClass("unknown")}
	caps := []WorkTier{TierT0, TierT1, TierT2}
	for _, cl := range classes {
		p := PolicyFor(cl)
		for _, cap := range caps {
			c := p.Admit("m", cap)
			if c.Admitted && !cap.MeetsRequirement(p.RequiredTier) {
				t.Fatalf("class %s admitted capability %s below floor %s", cl, cap, p.RequiredTier)
			}
			if !c.Admitted && cap.MeetsRequirement(p.RequiredTier) {
				t.Fatalf("class %s refused capability %s that meets floor %s", cl, cap, p.RequiredTier)
			}
		}
	}
}

// The numbering trap: T0 is the LOWEST number but the MOST demanding. Assert the
// comparison helpers encode capability order, not raw int order.
func TestTierPolicyDemandOrdering(t *testing.T) {
	if !TierT0.MeetsRequirement(TierT1) {
		t.Fatalf("T0 (most capable) must meet a T1 floor")
	}
	if TierT2.MeetsRequirement(TierT1) {
		t.Fatalf("T2 (routine) must NOT meet a T1 floor")
	}
	if !TierT0.MoreDemandingThan(TierT2) {
		t.Fatalf("T0 must be more demanding than T2")
	}
}

// CapabilityFromProfile folds raw evidence into a capability tier — and an
// ILLUSTRATIVE-only score never promotes a model above the routine floor.
func TestTierPolicyCapabilityFromProfile(t *testing.T) {
	ladder := DefaultCapabilityLadder()

	// Measured frontier-swe above the T0 bar -> T0 capable.
	strong := modelscore.Profile{Model: "strong", Benchmarks: []modelscore.BenchScore{
		{Benchmark: "frontier-swe", Score: 15, Unit: "pct-resolved", Provenance: modelscore.Provenance{Source: "s", Confidence: 1}},
	}}
	if tier, _ := CapabilityFromProfile(strong, ladder); tier != TierT0 {
		t.Fatalf("strong measured frontier-swe should be T0, got %s", tier)
	}

	// Same score but ILLUSTRATIVE -> must not promote above T2.
	fixture := modelscore.Profile{Model: "fixture", Benchmarks: []modelscore.BenchScore{
		{Benchmark: "frontier-swe", Score: 15, Unit: "pct-resolved", Provenance: modelscore.Provenance{Source: "s", Confidence: 0.2, Illustrative: true}},
	}}
	if tier, reasons := CapabilityFromProfile(fixture, ladder); tier != TierT2 {
		t.Fatalf("illustrative-only score promoted to %s (reasons %v); want T2", tier, reasons)
	}

	// Measured swe-bench above the T1 bar but no frontier-swe -> T1.
	mid := modelscore.Profile{Model: "mid", Benchmarks: []modelscore.BenchScore{
		{Benchmark: "swe-bench-verified", Score: 50, Unit: "pct-resolved", Provenance: modelscore.Provenance{Source: "s", Confidence: 1}},
	}}
	if tier, _ := CapabilityFromProfile(mid, ladder); tier != TierT1 {
		t.Fatalf("mid measured swe-bench should be T1, got %s", tier)
	}

	// No clearing evidence -> routine T2 floor.
	empty := modelscore.Profile{Model: "empty"}
	if tier, _ := CapabilityFromProfile(empty, ladder); tier != TierT2 {
		t.Fatalf("no evidence should floor at T2, got %s", tier)
	}
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}
