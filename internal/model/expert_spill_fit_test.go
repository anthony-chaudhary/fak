package model

import (
	"errors"
	"testing"
)

// TestAutoFitExpertSpillEndpoints pins the two endpoints: a budget that fits everything spills
// nothing (N=0), and a budget the dense base alone overruns fails closed at the maximal spill
// with Fits=false (never a negative or over-count spill).
func TestAutoFitExpertSpillEndpoints(t *testing.T) {
	// base=100, per-layer=10, 8 layers => full resident = 100 + 80 = 180.
	b := ExpertSpillBudget{MoELayers: 8, ExpertBytesPerLayer: 10, DeviceBaseBytes: 100}

	// Budget fits everything -> N=0.
	b.BudgetBytes = 200
	got, err := AutoFitExpertSpill(b)
	if err != nil {
		t.Fatalf("fits-all: unexpected err: %v", err)
	}
	if got.SpillLayers != 0 || !got.Fits || got.DeviceResidentBytes != 180 || got.HostSpillBytes != 0 {
		t.Fatalf("fits-all: got %+v, want N=0 Fits=true resident=180 host=0", got)
	}

	// Budget below the dense base -> spill all, fail closed.
	b.BudgetBytes = 50 // < base(100)
	got, err = AutoFitExpertSpill(b)
	if err != nil {
		t.Fatalf("fits-none: unexpected err: %v", err)
	}
	if got.SpillLayers != 8 || got.Fits {
		t.Fatalf("fits-none: got %+v, want N=8 Fits=false", got)
	}
	if got.DeviceResidentBytes != 100 || got.HostSpillBytes != 80 {
		t.Fatalf("fits-none: got %+v, want resident=100 (base only) host=80", got)
	}
}

// TestAutoFitExpertSpillPartial pins the graded boundary: the MINIMAL N that fits, and that N-1
// does not fit while N does.
func TestAutoFitExpertSpillPartial(t *testing.T) {
	// base=100, per-layer=10, 8 layers, budget=155. full=180. Need to shed >=25 -> ceil(25/10)=3.
	// resident(3)=100+50=150<=155 (fits); resident(2)=100+60=160>155 (does not).
	b := ExpertSpillBudget{MoELayers: 8, ExpertBytesPerLayer: 10, DeviceBaseBytes: 100, BudgetBytes: 155}
	got, err := AutoFitExpertSpill(b)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got.SpillLayers != 3 || !got.Fits || got.DeviceResidentBytes != 150 {
		t.Fatalf("got %+v, want N=3 Fits=true resident=150", got)
	}
	// Verify minimality against the explicit-N residency at N-1.
	prev, err := ResolveExpertSpill(b, got.SpillLayers-1)
	if err != nil {
		t.Fatalf("prev resolve err: %v", err)
	}
	if prev.Fits {
		t.Fatalf("minimality broken: N-1=%d already fits (%+v)", got.SpillLayers-1, prev)
	}

	// Exact-boundary budget: full=180, budget=160 -> need 20 -> N=2 exactly, resident=160.
	b.BudgetBytes = 160
	got, err = AutoFitExpertSpill(b)
	if err != nil {
		t.Fatalf("boundary err: %v", err)
	}
	if got.SpillLayers != 2 || !got.Fits || got.DeviceResidentBytes != 160 {
		t.Fatalf("boundary: got %+v, want N=2 Fits=true resident=160", got)
	}
}

// TestAutoFitExpertSpillMonotone checks device residency is monotone non-increasing in N and that
// auto-fit returns the least fitting N across a sweep of budgets.
func TestAutoFitExpertSpillMonotone(t *testing.T) {
	b := ExpertSpillBudget{MoELayers: 12, ExpertBytesPerLayer: 7, DeviceBaseBytes: 40}
	prev := int64(1 << 62)
	for n := 0; n <= b.MoELayers; n++ {
		f, err := ResolveExpertSpill(b, n)
		if err != nil {
			t.Fatalf("resolve N=%d: %v", n, err)
		}
		if f.DeviceResidentBytes > prev {
			t.Fatalf("residency not monotone at N=%d: %d > %d", n, f.DeviceResidentBytes, prev)
		}
		prev = f.DeviceResidentBytes
		// Auto-fit for a budget exactly at this residency must return the minimal N <= n.
		bb := b
		bb.BudgetBytes = f.DeviceResidentBytes
		af, err := AutoFitExpertSpill(bb)
		if err != nil {
			t.Fatalf("autofit budget=%d: %v", bb.BudgetBytes, err)
		}
		if !af.Fits {
			t.Fatalf("autofit budget=%d should fit, got %+v", bb.BudgetBytes, af)
		}
		if af.SpillLayers > n {
			t.Fatalf("autofit budget=%d picked N=%d > %d (not minimal)", bb.BudgetBytes, af.SpillLayers, n)
		}
	}
}

// TestResolveExpertSpillExplicit pins explicit-N honoring and range validation.
func TestResolveExpertSpillExplicit(t *testing.T) {
	b := ExpertSpillBudget{MoELayers: 8, ExpertBytesPerLayer: 10, DeviceBaseBytes: 100, BudgetBytes: 130}

	// Honored exactly: N=5 -> resident=100+3*10=130, fits.
	got, err := ResolveExpertSpill(b, 5)
	if err != nil {
		t.Fatalf("N=5 err: %v", err)
	}
	if got.SpillLayers != 5 || got.DeviceResidentBytes != 130 || !got.Fits || got.HostSpillBytes != 50 {
		t.Fatalf("N=5: got %+v, want N=5 resident=130 Fits=true host=50", got)
	}

	// N=0 and N=MoELayers are valid boundaries.
	if _, err := ResolveExpertSpill(b, 0); err != nil {
		t.Fatalf("N=0 should be valid: %v", err)
	}
	if _, err := ResolveExpertSpill(b, b.MoELayers); err != nil {
		t.Fatalf("N=MoELayers should be valid: %v", err)
	}

	// Out-of-range -> typed error, no silent clamp.
	for _, bad := range []int{-1, 9, 100} {
		_, err := ResolveExpertSpill(b, bad)
		var re *ExpertSpillRangeError
		if !errors.As(err, &re) {
			t.Fatalf("N=%d: want *ExpertSpillRangeError, got %v", bad, err)
		}
		if re.N != bad || re.Layers != b.MoELayers {
			t.Fatalf("N=%d: range error fields %+v", bad, re)
		}
	}
}

// TestExpertSpillEdgeCases covers zero-layer, zero-per-layer, and negative-input fail-closed.
func TestExpertSpillEdgeCases(t *testing.T) {
	// Zero MoE layers: nothing to spill; fit is just base vs budget.
	z := ExpertSpillBudget{MoELayers: 0, ExpertBytesPerLayer: 10, DeviceBaseBytes: 50, BudgetBytes: 60}
	got, err := AutoFitExpertSpill(z)
	if err != nil {
		t.Fatalf("zero-layer err: %v", err)
	}
	if got.SpillLayers != 0 || !got.Fits {
		t.Fatalf("zero-layer: got %+v, want N=0 Fits=true", got)
	}
	z.BudgetBytes = 40 // base overruns
	got, _ = AutoFitExpertSpill(z)
	if got.SpillLayers != 0 || got.Fits {
		t.Fatalf("zero-layer overrun: got %+v, want N=0 Fits=false", got)
	}

	// Per-layer cost 0 but base overruns: spilling frees nothing -> spill all, fail closed.
	zc := ExpertSpillBudget{MoELayers: 6, ExpertBytesPerLayer: 0, DeviceBaseBytes: 100, BudgetBytes: 50}
	got, err = AutoFitExpertSpill(zc)
	if err != nil {
		t.Fatalf("zero-cost err: %v", err)
	}
	if got.SpillLayers != 6 || got.Fits || got.DeviceResidentBytes != 100 {
		t.Fatalf("zero-cost: got %+v, want N=6 Fits=false resident=100", got)
	}

	// Negative fields fail closed.
	for _, bad := range []ExpertSpillBudget{
		{MoELayers: -1},
		{ExpertBytesPerLayer: -1},
		{DeviceBaseBytes: -1},
		{BudgetBytes: -1},
	} {
		if _, err := AutoFitExpertSpill(bad); err == nil {
			t.Fatalf("expected error for %+v", bad)
		}
		if _, err := ResolveExpertSpill(bad, 0); err == nil {
			t.Fatalf("expected resolve error for %+v", bad)
		}
	}
}
