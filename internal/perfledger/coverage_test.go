// fak-test:runtime fast est=1s lane=default
package perfledger

import (
	"math"
	"testing"
)

func TestCoverage(t *testing.T) {
	t.Run("full vs partial TTFT and ITL coverage across phase", func(t *testing.T) {
		nan := math.NaN()
		// Phase spans [0, 1000] across 4 requests.
		// TTFT is observed on all 4 requests ([0, 1000] -> ratio 1.0).
		// ITL is only observed on the first 2 requests ([0, 400] -> ratio 0.40).
		starts := []float64{0, 200, 400, 700}
		ends := []float64{200, 400, 700, 1000}
		ttft := []float64{15, 18, 20, 25}
		itl := []float64{5, 6, nan, nan}

		rep, err := ComputeTTFTITLCoverage(starts, ends, ttft, itl, nil)
		if err != nil {
			t.Fatalf("ComputeTTFTITLCoverage: %v", err)
		}
		if rep.PhaseStartNS != 0 || rep.PhaseEndNS != 1000 || rep.TotalRecords != 4 {
			t.Fatalf("phase meta = %+v", rep)
		}
		if math.Abs(rep.TTFT.CoverageRatio-1.0) > 1e-9 || math.Abs(rep.TTFT.ActiveCoverageRatio-1.0) > 1e-9 {
			t.Fatalf("TTFT coverage = %+v, want 1.0", rep.TTFT)
		}
		if math.Abs(rep.ITL.CoverageRatio-0.4) > 1e-9 || math.Abs(rep.ITL.ActiveCoverageRatio-0.4) > 1e-9 {
			t.Fatalf("ITL coverage = %+v, want 0.4", rep.ITL)
		}
		if math.Abs(rep.ITL.SampleCoverageRatio-0.5) > 1e-9 {
			t.Fatalf("ITL sample ratio = %v, want 0.5", rep.ITL.SampleCoverageRatio)
		}
		if !rep.TTFT.MeetsFloor(0.8) || rep.ITL.MeetsFloor(0.8) {
			t.Fatalf("unexpected MeetsFloor results: ttft=%v itl=%v", rep.TTFT.MeetsFloor(0.8), rep.ITL.MeetsFloor(0.8))
		}
	})

	t.Run("active coverage exposes mid-phase dropout gap", func(t *testing.T) {
		nan := math.NaN()
		// Valid ITL in [0, 200] and [800, 1000]; dead gap [200, 800].
		starts := []float64{0, 200, 800}
		ends := []float64{200, 800, 1000}
		itl := []float64{4, nan, 5}
		cov, err := ComputeMetricCoverage("inter_token_latency", starts, ends, itl, nil)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(cov.CoverageRatio-1.0) > 1e-9 {
			t.Fatalf("span CoverageRatio = %v, want 1.0", cov.CoverageRatio)
		}
		if math.Abs(cov.ActiveCoverageRatio-0.4) > 1e-9 {
			t.Fatalf("ActiveCoverageRatio = %v, want 0.4", cov.ActiveCoverageRatio)
		}
	})

	t.Run("zero valid signals and ColumnStore phase integration", func(t *testing.T) {
		store, err := NewColumnStore(ColumnStoreLimits{InitialRows: 8})
		if err != nil {
			t.Fatal(err)
		}
		// Warmup row excluded by phase mask; profiling rows have TTFT but zero valid ITL (non-streaming).
		if err := store.Ingest(0, 0, 100, map[string]any{DefaultTTFTMetricTag: 10.0, DefaultITLMetricTag: 2.0}); err != nil {
			t.Fatal(err)
		}
		if err := store.Ingest(1, 100, 600, map[string]any{DefaultTTFTMetricTag: 12.0}); err != nil {
			t.Fatal(err)
		}
		if err := store.Ingest(2, 600, 1100, map[string]any{DefaultTTFTMetricTag: 14.0}); err != nil {
			t.Fatal(err)
		}
		_ = store.IngestCategorical(0, DefaultPhaseColumnTag, PhaseWarmup)
		_ = store.IngestCategorical(1, DefaultPhaseColumnTag, PhaseProfiling)
		_ = store.IngestCategorical(2, DefaultPhaseColumnTag, PhaseProfiling)

		rep, err := store.PhaseTTFTITLCoverage(PhaseProfiling, "", "")
		if err != nil {
			t.Fatalf("PhaseTTFTITLCoverage: %v", err)
		}
		if rep.TotalRecords != 2 || rep.PhaseStartNS != 100 || rep.PhaseEndNS != 1100 {
			t.Fatalf("report = %+v", rep)
		}
		if rep.TTFT.CoverageRatio != 1.0 {
			t.Fatalf("TTFT CoverageRatio = %v, want 1.0", rep.TTFT.CoverageRatio)
		}
		if rep.ITL.ValidCount != 0 || rep.ITL.CoverageRatio != 0 || rep.ITL.ActiveCoverageRatio != 0 {
			t.Fatalf("expected 0 ITL coverage, got %+v", rep.ITL)
		}
	})
}
