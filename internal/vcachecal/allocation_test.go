package vcachecal

import (
	"math"
	"testing"
)

// allocation_test.go pins the issue-#4016 acceptance: a peaky bucket gains share vs
// the flat split while the global budget stays fixed, and the default (topK <= 0)
// path is byte-identical to today's even split regardless of bucket contents. The
// fixture models the memq-render competition: N sessions (buckets) with ranked cell
// values compete for one shared render byte budget.

// bucketOf builds one competitor whose cell values are the given weights, already
// ranked descending as FitConcentration requires (Size/ReuseDensity 1, so Weight ==
// Frequency — the value axis under test).
func bucketOf(key string, weights ...float64) BudgetBucket {
	rows := make([]RankedVBlock, len(weights))
	for i, w := range weights {
		rows[i] = RankedVBlock{Key: key, Frequency: w, Size: 1, ReuseDensity: 1}
	}
	return BudgetBucket{Key: key, Ranked: rows}
}

func TestAllocateByConcentrationPeakyBucketGainsShare(t *testing.T) {
	// Two sessions with the SAME total cell value (100) and the SAME cell count, so
	// concentration is the only differentiator: "peaky" holds 90% of its value in its
	// top cell; "diffuse" spreads it 34/33/33.
	buckets := []BudgetBucket{
		bucketOf("peaky", 90, 5, 5),
		bucketOf("diffuse", 34, 33, 33),
	}
	got := AllocateByConcentration(buckets, 10_000, 1)

	// Top-1 captured fractions: 90/100 and 34/100 — the #4016 concentration weight.
	if math.Abs(got[0].Concentration-0.90) > 1e-12 || math.Abs(got[1].Concentration-0.34) > 1e-12 {
		t.Fatalf("concentrations = %g/%g, want 0.90/0.34", got[0].Concentration, got[1].Concentration)
	}
	// The peaky bucket gains vs the 5000/5000 flat split; the diffuse one is dampened
	// but never starved: 0.90/1.24 and 0.34/1.24 of 10000, largest-remainder rounded.
	if got[0].Budget != 7258 || got[1].Budget != 2742 {
		t.Errorf("budgets = %d/%d, want 7258/2742", got[0].Budget, got[1].Budget)
	}
	if got[0].Budget <= 5_000 || got[1].Budget >= 5_000 {
		t.Errorf("peaky must gain vs flat and diffuse must cede: %d/%d", got[0].Budget, got[1].Budget)
	}
	// The global budget is conserved exactly — the renormalization invariant.
	if sum := got[0].Budget + got[1].Budget; sum != 10_000 {
		t.Errorf("budget sum = %d, want exactly 10000 (total conserved)", sum)
	}
}

func TestAllocateByConcentrationDefaultOffIsFlatSplit(t *testing.T) {
	// topK <= 0 is the opt-out default: the split is the flat even slice regardless
	// of how peaky the buckets are — byte-identical to today's un-weighted division.
	buckets := []BudgetBucket{
		bucketOf("peaky", 1000, 1, 1),
		bucketOf("diffuse", 1, 1, 1),
		bucketOf("empty"),
	}
	got := AllocateByConcentration(buckets, 10, 0)
	// floor(10/3) = 3 each; the leftover unit goes to the lowest input index on the
	// all-equal remainder tie — explicit and deterministic.
	want := []int64{4, 3, 3}
	for i, w := range want {
		if got[i].Budget != w {
			t.Errorf("bucket %d budget = %d, want %d", i, got[i].Budget, w)
		}
		if got[i].Share != got[i].FlatShare {
			t.Errorf("bucket %d share = %g, want the flat %g (weighting must be OFF)", i, got[i].Share, got[i].FlatShare)
		}
		if got[i].Concentration != 0 {
			t.Errorf("bucket %d concentration = %g, want 0 (nothing measured when off)", i, got[i].Concentration)
		}
	}
}

func TestAllocateByConcentrationUnmeasuredBucketKeepsFlatShare(t *testing.T) {
	// A bucket with no positive weight cannot be judged peaky or diffuse: it keeps
	// exactly its flat share — never starved by a measurement it never had (the
	// upstream AdaKV starvation tradeoff, guarded).
	buckets := []BudgetBucket{
		bucketOf("peaky", 8, 1, 1),
		bucketOf("unmeasured"),
		bucketOf("diffuse", 4, 3, 3),
	}
	got := AllocateByConcentration(buckets, 9_000, 1)
	if got[1].Concentration != 0 {
		t.Fatalf("unmeasured concentration = %g, want 0", got[1].Concentration)
	}
	// peaky top-1 = 0.8, diffuse top-1 = 0.4: the measured 2/3 mass splits 2:1.
	want := []int64{4_000, 3_000, 2_000}
	for i, w := range want {
		if got[i].Budget != w {
			t.Errorf("bucket %d budget = %d, want %d", i, got[i].Budget, w)
		}
	}
	if sum := got[0].Budget + got[1].Budget + got[2].Budget; sum != 9_000 {
		t.Errorf("budget sum = %d, want exactly 9000 (total conserved)", sum)
	}
}

func TestAllocateByConcentrationEdges(t *testing.T) {
	if got := AllocateByConcentration(nil, 100, 1); got != nil {
		t.Errorf("nil buckets = %v, want nil", got)
	}
	// A single bucket owns the whole budget; K past the curve clamps to full coverage.
	got := AllocateByConcentration([]BudgetBucket{bucketOf("only", 5)}, 100, 3)
	if len(got) != 1 || got[0].Budget != 100 || got[0].Concentration != 1 {
		t.Errorf("single bucket = %+v, want the whole budget at concentration 1 (K clamped)", got)
	}
	// total <= 0: shares still report, budgets stay 0.
	z := AllocateByConcentration([]BudgetBucket{bucketOf("a", 2, 1), bucketOf("b", 1, 1)}, 0, 1)
	if z[0].Budget != 0 || z[1].Budget != 0 {
		t.Errorf("zero-total budgets = %d/%d, want 0/0", z[0].Budget, z[1].Budget)
	}
}
