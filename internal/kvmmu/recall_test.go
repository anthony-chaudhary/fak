package kvmmu_test

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvmmu"
)

// recall_test.go — issue #3901: the attention-mass RECALL gauge (hit-QUALITY, not hit-rate).
//
// These are the first-checkable-step witnesses the issue names: given a synthetic attention
// distribution folded through the #855 accumulator and an eviction plan (kept vs dropped),
//   1. retained_mass_fraction == (Σ kept-span cumulative mass) / (Σ total),
//   2. keeping the top-k-spans-by-mass maximizes the fraction vs ANY other k-subset, and
//   3. no observer installed ⇒ no gauge (fail-closed), the observability-only guard.
// No forward pass runs — the gauge reads the accumulator, exactly as the seam prescribes.

// feedSynthetic folds one turn of a synthetic per-span attention distribution into a
// post-hoc accumulator (λ=1, no Forget), so Cumulative == the mass each span drew — the
// #855 reduction the gauge reads. Returns the accumulator.
func feedSynthetic(mass map[string]float64) *kvmmu.AttentionAccumulator {
	acc := kvmmu.NewAttentionAccumulator(1.0, 0)
	acc.Observe(mass)
	return acc
}

// TestRetainedMassFractionEqualsKeptOverTotal is witness 1: the gauge equals the hand-
// computed Σ(kept cumulative)/Σ(total cumulative), and the token fraction expresses the
// "X% of the mass at Y% of the tokens" reading the issue asks for.
func TestRetainedMassFractionEqualsKeptOverTotal(t *testing.T) {
	// Synthetic distribution: four spans, distinct masses summing to 10.
	mass := map[string]float64{"a": 1, "b": 2, "c": 3, "d": 4}
	cost := map[string]int{"a": 10, "b": 10, "c": 10, "d": 10}
	acc := feedSynthetic(mass)

	// Eviction plan: keep the two warmest {c,d}, drop the two coldest {a,b}.
	kept := []string{"c", "d"}
	g := kvmmu.RetainedMass(acc, kept, cost)

	if !g.Available {
		t.Fatal("gauge reported unavailable with observed attention present")
	}
	// Σ kept cumulative = 3+4 = 7; Σ total = 1+2+3+4 = 10.
	if g.KeptMass != 7 || g.TotalMass != 10 {
		t.Fatalf("KeptMass=%.1f TotalMass=%.1f, want 7 and 10", g.KeptMass, g.TotalMass)
	}
	if math.Abs(g.Fraction-0.7) > 1e-9 {
		t.Fatalf("Fraction=%.6f, want 0.7 = (3+4)/(1+2+3+4)", g.Fraction)
	}
	// Token fraction: kept 20 of 40 tokens -> 0.5. "70% of the mass at 50% of the tokens."
	if g.KeptTokens != 20 || g.TotalTokens != 40 {
		t.Fatalf("KeptTokens=%d TotalTokens=%d, want 20 and 40", g.KeptTokens, g.TotalTokens)
	}
	if math.Abs(g.TokenFraction-0.5) > 1e-9 {
		t.Fatalf("TokenFraction=%.6f, want 0.5", g.TokenFraction)
	}
	t.Logf("retained %.0f%% of the observed attention mass at %.0f%% of the tokens",
		g.Fraction*100, g.TokenFraction*100)
}

// TestTopKByMassMaximizesRetainedFraction is witness 2: no k-subset of the observed spans
// captures more mass than the top-k-by-mass set, so RetainedMass over TopKSpansByMass is
// the ceiling any k-span retention can reach. Enumerated exhaustively over every k-subset.
func TestTopKByMassMaximizesRetainedFraction(t *testing.T) {
	mass := map[string]float64{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
	ids := []string{"a", "b", "c", "d", "e"}
	acc := feedSynthetic(mass)

	for k := 1; k <= len(ids); k++ {
		best := kvmmu.RetainedMass(acc, kvmmu.TopKSpansByMass(acc, k), nil).Fraction
		// Every k-subset must have retained-mass fraction <= the top-k choice.
		for _, subset := range combinations(ids, k) {
			f := kvmmu.RetainedMass(acc, subset, nil).Fraction
			if f > best+1e-9 {
				t.Fatalf("k=%d: subset %v fraction %.6f beats top-k %.6f — top-k is not the ceiling",
					k, subset, f, best)
			}
		}
	}
}

// TestRetainedMassFailsClosedWithoutObserver is witness 3, the observability-only guard: an
// accumulator that has observed no attention (the observer is default-off, #852) yields NO
// gauge — Available false, every ratio 0 — never a 0/0 NaN or a spurious 1.0. A nil
// accumulator is the same fail-closed answer.
func TestRetainedMassFailsClosedWithoutObserver(t *testing.T) {
	// Never observed: fresh accumulator.
	if g := kvmmu.RetainedMass(kvmmu.NewAttentionAccumulator(1.0, 0), []string{"a"}, nil); g.Available {
		t.Fatalf("gauge available on an un-observed accumulator: %+v", g)
	}
	// Observed, but every span drew zero mass (rows attributed, all weights 0).
	zero := kvmmu.RetainedMass(feedSynthetic(map[string]float64{"a": 0, "b": 0}), []string{"a"}, nil)
	if zero.Available || zero.Fraction != 0 {
		t.Fatalf("gauge not fail-closed at total mass 0: %+v", zero)
	}
	if math.IsNaN(zero.Fraction) {
		t.Fatal("fraction is NaN — the 0/0 the fail-closed guard exists to prevent")
	}
	// Nil accumulator.
	if g := kvmmu.RetainedMass(nil, []string{"a"}, nil); g.Available {
		t.Fatalf("gauge available on a nil accumulator: %+v", g)
	}
}

// TestRetainedMassKeepsUnknownKeptIdAtZero proves the numerator treats a kept id the
// accumulator never saw (e.g. a pinned-but-never-attended span) as honest zero mass, not
// an error and not a silent denominator change.
func TestRetainedMassKeepsUnknownKeptIdAtZero(t *testing.T) {
	acc := feedSynthetic(map[string]float64{"a": 4, "b": 6})
	// Keep "a" (mass 4) plus a never-observed "ghost": ghost adds 0 to the numerator.
	g := kvmmu.RetainedMass(acc, []string{"a", "ghost"}, nil)
	if g.TotalMass != 10 || g.KeptMass != 4 {
		t.Fatalf("KeptMass=%.1f TotalMass=%.1f, want 4 and 10 (ghost contributes 0)", g.KeptMass, g.TotalMass)
	}
	if math.Abs(g.Fraction-0.4) > 1e-9 {
		t.Fatalf("Fraction=%.6f, want 0.4", g.Fraction)
	}
}

// combinations returns every k-element subset of ids (order within a subset preserved).
func combinations(ids []string, k int) [][]string {
	var out [][]string
	var rec func(start int, cur []string)
	rec = func(start int, cur []string) {
		if len(cur) == k {
			out = append(out, append([]string(nil), cur...))
			return
		}
		for i := start; i < len(ids); i++ {
			rec(i+1, append(cur, ids[i]))
		}
	}
	rec(0, nil)
	return out
}
