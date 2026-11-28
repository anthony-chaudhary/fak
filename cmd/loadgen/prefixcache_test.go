// fak-test:runtime fast est=1s lane=default
package main

import (
	"math"
	"testing"
)

func TestPrefixCache(t *testing.T) {
	t.Run("phase-scoped block accounting clamps hits to [0, total]", func(t *testing.T) {
		turns := []PrefixBlockTurn{
			{Phase: "warmup", HitBlocks: 10, TotalBlocks: 10},    // excluded by phase filter
			{Phase: "profiling", HitBlocks: 4, TotalBlocks: 10},  // 0.4
			{Phase: "profiling", HitBlocks: 15, TotalBlocks: 10}, // clamped to 10 (1.0)
			{Phase: "profiling", HitBlocks: -3, TotalBlocks: 20}, // clamped to 0 (0.0)
			{Phase: "profiling", HitBlocks: 0, TotalBlocks: 0},   // 0 blocks -> 0.0
		}

		st, err := ComputeTheoreticalPrefixCache("profiling", turns)
		if err != nil {
			t.Fatalf("ComputeTheoreticalPrefixCache: %v", err)
		}
		if st.Records != 4 || st.SkippedPhaseRecords != 1 || st.ClampedRecords != 2 {
			t.Fatalf("unexpected counts: %+v", st)
		}
		// Total hit blocks: 4 + 10 + 0 + 0 = 14; total blocks: 10 + 10 + 20 + 0 = 40.
		if st.TotalHitBlocks != 14 || st.TotalBlocks != 40 {
			t.Fatalf("block totals = (%d, %d), want (14, 40)", st.TotalHitBlocks, st.TotalBlocks)
		}
		if math.Abs(st.TheoreticalHitRate-0.35) > 1e-9 || math.Abs(st.TheoreticalHitPct-35.0) > 1e-9 {
			t.Fatalf("TheoreticalHitRate/Pct = (%v, %v), want (0.35, 35)", st.TheoreticalHitRate, st.TheoreticalHitPct)
		}
		// Per-record rates: [0.4, 1.0, 0.0, 0.0] -> mean 0.35, sorted [0.0, 0.0, 0.4, 1.0].
		if math.Abs(st.MeanPerRecordHitRate-0.35) > 1e-9 {
			t.Fatalf("MeanPerRecordHitRate = %v, want 0.35", st.MeanPerRecordHitRate)
		}
		if st.P50PerRecordHitRate != 0.0 || st.P95PerRecordHitRate != 1.0 {
			t.Fatalf("percentiles = (%v, %v), want (0.0, 1.0)", st.P50PerRecordHitRate, st.P95PerRecordHitRate)
		}
	})

	t.Run("incremental accountant and negative total_blocks rejection", func(t *testing.T) {
		acc := NewPrefixCacheAccountant("")
		r1, counted1, err := acc.Observe(PrefixBlockTurn{Phase: "profiling", HitBlocks: 6, TotalBlocks: 8})
		if err != nil || !counted1 || math.Abs(r1-0.75) > 1e-9 {
			t.Fatalf("Observe 1 = (%v, %v, %v)", r1, counted1, err)
		}
		r2, counted2, err := acc.Observe(PrefixBlockTurn{Phase: "profiling", HitBlocks: 2, TotalBlocks: 2})
		if err != nil || !counted2 || r2 != 1.0 {
			t.Fatalf("Observe 2 = (%v, %v, %v)", r2, counted2, err)
		}
		st := acc.Stats()
		if st.TotalHitBlocks != 8 || st.TotalBlocks != 10 || math.Abs(st.TheoreticalHitRate-0.8) > 1e-9 {
			t.Fatalf("incremental Stats = %+v", st)
		}
		if _, _, err := acc.Observe(PrefixBlockTurn{TotalBlocks: -1}); err == nil {
			t.Fatal("expected error for negative total_blocks")
		}
	})
}
