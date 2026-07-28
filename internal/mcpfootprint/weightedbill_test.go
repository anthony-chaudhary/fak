package mcpfootprint

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// tf builds a priced per-tool item (name + marginal token cost) for the join tests.
func tf(name string, tokens int) agent.ToolFootprint {
	return agent.ToolFootprint{Name: name, Tokens: tokens}
}

// TestWeightedBillHotCheapOutranks is the doctrine's central case: a cheap-but-frequent
// tool must be able to outrank an expensive-but-rare one once the second axis is real.
// With frequency pinned at 1.0 the expensive tool would win; weighted, the hot tool
// carries the larger contribution and sorts first.
func TestWeightedBillHotCheapOutranks(t *testing.T) {
	items := []agent.ToolFootprint{
		tf("expensive_cold", 1000), // marginal 1000, invoked rarely
		tf("cheap_hot", 100),       // marginal 100, invoked constantly
	}
	freq := map[string]float64{
		"expensive_cold": 0.5, // 1000 x 0.5 = 500
		"cheap_hot":      9,   // 100  x 9   = 900
	}
	got := WeightedBill(items, freq)
	if got[0].Name != "cheap_hot" {
		t.Fatalf("weighted top = %q want cheap_hot (cheap-and-hot must outrank expensive-and-cold)", got[0].Name)
	}
	if got[0].Contribution <= got[1].Contribution {
		t.Fatalf("ranking not by contribution: %+v", got)
	}
}

// TestWeightedBillIsProductPerItem proves the per-item bill is exactly frequency x
// marginal, and that the ranking is contribution-descending with the fallback applied
// to an un-observed tool.
func TestWeightedBillIsProductPerItem(t *testing.T) {
	items := []agent.ToolFootprint{
		tf("a", 40),
		tf("b", 30),
		tf("c", 10), // no frequency entry -> FrequencyFallback (1.0)
	}
	freq := map[string]float64{"a": 2, "b": 5}
	got := WeightedBill(items, freq)

	want := map[string]float64{"a": 80, "b": 150, "c": 10}
	for _, it := range got {
		if it.Contribution != want[it.Name] {
			t.Fatalf("%s contribution=%v want %v", it.Name, it.Contribution, want[it.Name])
		}
	}
	// b (150) > a (80) > c (10)
	if got[0].Name != "b" || got[1].Name != "a" || got[2].Name != "c" {
		t.Fatalf("ranking = %q,%q,%q want b,a,c", got[0].Name, got[1].Name, got[2].Name)
	}
	if got[2].Frequency != FrequencyFallback {
		t.Fatalf("un-observed tool freq=%v want fallback %v", got[2].Frequency, FrequencyFallback)
	}
}

// TestWeightedBillZeroFrequencyZeroContribution proves an OBSERVED-idle tool (present
// in the table with frequency 0) contributes zero — distinct from an un-observed tool,
// which keeps the 1.0 fallback.
func TestWeightedBillZeroFrequencyZeroContribution(t *testing.T) {
	items := []agent.ToolFootprint{tf("idle", 500), tf("unseen", 500)}
	freq := map[string]float64{"idle": 0}
	got := WeightedBill(items, freq)

	by := map[string]WeightedItem{}
	for _, it := range got {
		by[it.Name] = it
	}
	if by["idle"].Contribution != 0 {
		t.Fatalf("observed-idle contribution=%v want 0", by["idle"].Contribution)
	}
	if by["unseen"].Contribution != 500 {
		t.Fatalf("un-observed contribution=%v want 500 (fallback 1.0 x 500)", by["unseen"].Contribution)
	}
}

// TestWeightedBillDeterministicTieBreak proves equal contributions order by name, so the
// ranking is stable regardless of input order.
func TestWeightedBillDeterministicTieBreak(t *testing.T) {
	items := []agent.ToolFootprint{tf("zebra", 100), tf("alpha", 100), tf("mango", 100)}
	freq := map[string]float64{"zebra": 1, "alpha": 1, "mango": 1} // all tie at 100
	got := WeightedBill(items, freq)
	if got[0].Name != "alpha" || got[1].Name != "mango" || got[2].Name != "zebra" {
		t.Fatalf("tie order = %q,%q,%q want alpha,mango,zebra", got[0].Name, got[1].Name, got[2].Name)
	}
}

// TestWeightedBillNegativeFailsClosed proves a broken axis (negative marginal or
// negative frequency) yields a safe zero contribution rather than crediting the bill,
// while the offending value stays visible in the item's fields.
func TestWeightedBillNegativeFailsClosed(t *testing.T) {
	items := []agent.ToolFootprint{tf("neg_marginal", -50), tf("healthy", 20)}
	freq := map[string]float64{"neg_marginal": 3, "healthy": -4} // one bad axis each
	got := WeightedBill(items, freq)
	for _, it := range got {
		if it.Contribution != 0 {
			t.Fatalf("%s contribution=%v want 0 (fail closed on negative axis)", it.Name, it.Contribution)
		}
	}
	// The raw negative values are still surfaced so the breakage is diagnosable.
	by := map[string]WeightedItem{}
	for _, it := range got {
		by[it.Name] = it
	}
	if by["neg_marginal"].Marginal != -50 || by["healthy"].Frequency != -4 {
		t.Fatalf("negative axes not surfaced: %+v", got)
	}
}

// TestWeightedBillEmpty proves an empty item set yields an empty bill and a zero total —
// the degenerate no-tools case fails closed to a safe zero, never a nil-deref.
func TestWeightedBillEmpty(t *testing.T) {
	got := WeightedBill(nil, nil)
	if len(got) != 0 {
		t.Fatalf("empty input bill=%+v want empty", got)
	}
	if TotalBill(got) != 0 {
		t.Fatalf("empty total=%v want 0", TotalBill(got))
	}
}

// TestTotalBillSums proves TotalBill rolls the healthy contributions up and that a
// broken (negative-axis) item adds its safe zero rather than dragging the total down.
func TestTotalBillSums(t *testing.T) {
	items := []agent.ToolFootprint{tf("a", 40), tf("b", 30), tf("broken", -100)}
	freq := map[string]float64{"a": 2, "b": 5, "broken": 9} // a=80, b=150, broken=0
	total := TotalBill(WeightedBill(items, freq))
	if total != 230 {
		t.Fatalf("total=%v want 230 (80 + 150 + 0)", total)
	}
}
