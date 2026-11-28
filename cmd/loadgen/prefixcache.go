// Adapted from ai-dynamo/aiperf
// modified from the original): theoretical infinite-cache prefix-hit accounting
// from loader-stamped per-turn (hit_blocks, total_blocks) with clamping to
// [0, total_blocks] and phase scoping, requiring zero runtime hash_ids or
// re-tokenization.
package main

import (
	"fmt"
	"sort"
)

// PrefixBlockTurn carries loader-stamped KV prefix block counts for one turn.
type PrefixBlockTurn struct {
	Phase       string `json:"phase,omitempty"`
	HitBlocks   int    `json:"hit_blocks"`
	TotalBlocks int    `json:"total_blocks"`
}

// PrefixCacheStats summarizes theoretical prefix-cache hit rates across a phase.
type PrefixCacheStats struct {
	TargetPhase          string  `json:"target_phase,omitempty"`
	Records              int     `json:"records"`
	SkippedPhaseRecords  int     `json:"skipped_phase_records,omitempty"`
	ClampedRecords       int     `json:"clamped_records,omitempty"`
	TotalHitBlocks       int64   `json:"total_hit_blocks"`
	TotalBlocks          int64   `json:"total_blocks"`
	TheoreticalHitRate   float64 `json:"theoretical_hit_rate"`
	TheoreticalHitPct    float64 `json:"theoretical_hit_pct"`
	MeanPerRecordHitRate float64 `json:"mean_per_record_hit_rate"`
	P50PerRecordHitRate  float64 `json:"p50_per_record_hit_rate"`
	P95PerRecordHitRate  float64 `json:"p95_per_record_hit_rate"`
}

// PrefixCacheAccountant accumulates theoretical prefix-cache hit statistics
// incrementally via two integer additions per phase-matching turn.
type PrefixCacheAccountant struct {
	targetPhase   string
	records       int
	skippedPhase  int
	clamped       int
	totalHit      int64
	totalBlocks   int64
	perRecordRate []float64
	rateSum       float64
}

// NewPrefixCacheAccountant returns an accountant scoped to targetPhase (or
// all turns when targetPhase is empty).
func NewPrefixCacheAccountant(targetPhase string) *PrefixCacheAccountant {
	return &PrefixCacheAccountant{targetPhase: targetPhase}
}

// Observe folds one turn's loader-stamped block counts into the accountant,
// clamping HitBlocks into [0, TotalBlocks].
func (a *PrefixCacheAccountant) Observe(turn PrefixBlockTurn) (perTurnHitRate float64, counted bool, err error) {
	if a == nil {
		return 0, false, fmt.Errorf("loadgen: PrefixCacheAccountant.Observe: nil accountant")
	}
	if turn.TotalBlocks < 0 {
		return 0, false, fmt.Errorf("loadgen: PrefixCacheAccountant.Observe: negative total_blocks %d", turn.TotalBlocks)
	}
	if a.targetPhase != "" && turn.Phase != a.targetPhase {
		a.skippedPhase++
		return 0, false, nil
	}

	hits := turn.HitBlocks
	if hits < 0 {
		hits = 0
		a.clamped++
	} else if hits > turn.TotalBlocks {
		hits = turn.TotalBlocks
		a.clamped++
	}

	a.records++
	a.totalHit += int64(hits)
	a.totalBlocks += int64(turn.TotalBlocks)

	rate := 0.0
	if turn.TotalBlocks > 0 {
		rate = float64(hits) / float64(turn.TotalBlocks)
	}
	a.perRecordRate = append(a.perRecordRate, rate)
	a.rateSum += rate
	return rate, true, nil
}

// Stats computes the block-weighted theoretical hit rate and per-record summary.
func (a *PrefixCacheAccountant) Stats() PrefixCacheStats {
	if a == nil {
		return PrefixCacheStats{}
	}
	st := PrefixCacheStats{
		TargetPhase:         a.targetPhase,
		Records:             a.records,
		SkippedPhaseRecords: a.skippedPhase,
		ClampedRecords:      a.clamped,
		TotalHitBlocks:      a.totalHit,
		TotalBlocks:         a.totalBlocks,
	}
	if a.totalBlocks > 0 {
		st.TheoreticalHitRate = float64(a.totalHit) / float64(a.totalBlocks)
		st.TheoreticalHitPct = st.TheoreticalHitRate * 100.0
	}
	if a.records > 0 {
		st.MeanPerRecordHitRate = a.rateSum / float64(a.records)
		sorted := append([]float64(nil), a.perRecordRate...)
		sort.Float64s(sorted)
		st.P50PerRecordHitRate = percentile(sorted, 50)
		st.P95PerRecordHitRate = percentile(sorted, 95)
	}
	return st
}

// ComputeTheoreticalPrefixCache evaluates theoretical prefix-cache hit stats
// across a slice of loader-stamped turns.
func ComputeTheoreticalPrefixCache(targetPhase string, turns []PrefixBlockTurn) (PrefixCacheStats, error) {
	acc := NewPrefixCacheAccountant(targetPhase)
	for _, turn := range turns {
		if _, _, err := acc.Observe(turn); err != nil {
			return PrefixCacheStats{}, err
		}
	}
	return acc.Stats(), nil
}
