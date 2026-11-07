package model

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// sweepRing replays rows layer-major prefill sweeps that touch every expert of every layer (the
// shape a long prefill row routes) and returns the hit fraction of the rows after the first.
func sweepRing(t *testing.T, budget, stable int64, rows int) (float64, ExpertRingStats) {
	t.Helper()
	const layers, experts, projections = 8, 32, 3
	be := compute.Default()
	src := compute.NewF32(be, []int{1}, []float32{1})
	r := newPagedRing(be, budget)
	r.stableCap = stable
	var h0, p0 int
	for row := 0; row < rows; row++ {
		if row == 1 {
			h0, p0 = r.hit, r.pageIn
		}
		for l := 0; l < layers; l++ {
			for e := 0; e < experts; e++ {
				for p := 0; p < projections; p++ {
					name := fmt.Sprintf("blk.%d.ffn_exps.%d.%d", l, e, p)
					if _, err := r.stageWithError(name, func() compute.Tensor { return src }, compute.F32, 1, false); err != nil {
						t.Fatalf("stage %s: %v", name, err)
					}
				}
			}
		}
	}
	hits, pageIns := r.hit-h0, r.pageIn-p0
	return float64(hits) / float64(hits+pageIns), r.stats()
}

// TestExpertRingStableSetSurvivesLayerCyclicSweep is the DS4.1/H200 0%-hit witness: a ring smaller
// than the routed band never serves a cross-row hit under recency eviction (each expert is evicted
// before the sweep returns to its layer), while the stable set retains its share of the band.
func TestExpertRingStableSetSurvivesLayerCyclicSweep(t *testing.T) {
	const band = 8 * 32 * 3
	budget := int64(band * 6 / 10) // a 112 GiB ring over a 183 GiB band
	lru, _ := sweepRing(t, budget, 0, 4)
	if lru != 0 {
		t.Fatalf("plain LRU cross-row hit = %.3f; the cyclic-scan pathology this test pins expects 0", lru)
	}
	stableCap := budget - budget/4
	got, st := sweepRing(t, budget, stableCap, 4)
	if want := 0.9 * float64(stableCap) / band; got < want {
		t.Fatalf("stable-set cross-row hit = %.3f, want >= %.3f (stable %d of band %d)", got, want, stableCap, band)
	}
	if st.StableBytes != stableCap || st.StableCapBytes != stableCap {
		t.Fatalf("stable bytes=%d cap=%d, want both %d", st.StableBytes, st.StableCapBytes, stableCap)
	}
	if st.PeakBytes > budget || st.Refusals != 0 {
		t.Fatalf("peak=%d budget=%d refusals=%d: the stable set must stay inside the budget without refusals", st.PeakBytes, budget, st.Refusals)
	}
}

func TestSharedExpertRingRefusesStableSetAtBudget(t *testing.T) {
	_, err := NewSharedExpertRing(SharedExpertRingConfig{Model: &Model{}, Backend: compute.Default(), BudgetBytes: 10, StableBytes: 10})
	if err == nil {
		t.Fatal("a stable set as large as the budget leaves no churn room and must be refused")
	}
}
