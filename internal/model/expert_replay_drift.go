package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// ExpertHorizonWindow is one contiguous horizon segment scored by the D1 gauge on its own
// cold cache. Independent per-window replay isolates a segment's intrinsic cacheability
// from warm-up carried in from earlier segments: the question is not "did the cache happen
// to be warm here" but "how well does the residency policy serve THIS segment's workload".
type ExpertHorizonWindow struct {
	Index           int                  `json:"index"`
	StartEvent      int                  `json:"start_event"`
	EndEvent        int                  `json:"end_event"` // exclusive
	Events          int                  `json:"events"`
	LRUGoodDecision float64              `json:"lru_good_decision_ratio"`
	LRUHitBytes     int                  `json:"lru_hit_bytes"`
	OracleHitBytes  int                  `json:"oracle_hit_bytes"`
	OracleExact     bool                 `json:"oracle_exact"`
	Locality        ExpertLocalityReport `json:"locality"`
	// LocalityInverted marks a window where observed recency no longer beats the
	// layer-sequential null: the prior behind LRU is unsupported here even though it may
	// hold over the whole trace. Only asserted when the window has locality samples; a
	// window too small to form a reuse pair is undetermined, not inverted.
	LocalityInverted bool `json:"locality_inverted"`
}

// ExpertReplayDriftReport is the long-horizon sibling of ExpertReplayReport. A single
// whole-trace GoodDecisionRatio averages the residency policy's regret across the session;
// under the workload drift a long, many-agent run actually exhibits (arithmetic->code
// regime shifts; a shared gateway interleaving several agents' working sets) that scalar
// can read "somewhat suboptimal" while an entire late-horizon window has collapsed to pure
// thrashing. This report keeps the per-window regret and locality validity visible and
// summarizes how far they drift, so the collapse is a named window rather than a lost term.
type ExpertReplayDriftReport struct {
	Name        string                `json:"name"`
	Source      string                `json:"source"`
	BudgetBytes int64                 `json:"budget_bytes"`
	Windows     []ExpertHorizonWindow `json:"windows"`
	// WholeTraceGoodDecision is the D1 whole-trace LRU regret this report is contrasted
	// against: the number a single scalar would report, kept so the concealment is legible.
	WholeTraceGoodDecision float64 `json:"whole_trace_lru_good_decision_ratio"`
	GoodDecisionMin        float64 `json:"good_decision_min"`
	GoodDecisionMax        float64 `json:"good_decision_max"`
	GoodDecisionSpread     float64 `json:"good_decision_spread"`
	// GoodDecisionTrendSlope is the least-squares slope of per-window GoodDecisionRatio
	// against window index. Negative means the policy's regret worsens as the horizon
	// advances -- the fault-service-dominates drift a stationary scalar cannot show.
	GoodDecisionTrendSlope float64 `json:"good_decision_trend_slope"`
	WorstWindow            int     `json:"worst_window"`
	LocalityInversions     int     `json:"locality_inversions"`
}

// ReplayExpertAccessTraceDrift splits the trace into `windows` contiguous horizon segments,
// scores each through the existing ReplayExpertAccessTrace (reusing the KV Belady oracle,
// pagedRing-LRU replay, GoodDecisionRatio, and the locality diagnostic unchanged), and
// reports how the LRU regret and locality validity drift across the session. The whole
// trace is validated and replayed once up front so WholeTraceGoodDecision is the exact
// scalar this per-window view is contrasted against.
func ReplayExpertAccessTraceDrift(trace ExpertAccessTrace, windows int, policies ...compute.KVEvictPolicy) (ExpertReplayDriftReport, error) {
	if windows < 1 {
		return ExpertReplayDriftReport{}, fmt.Errorf("model: expert replay windows must be >= 1, got %d", windows)
	}
	whole, err := ReplayExpertAccessTrace(trace, policies...)
	if err != nil {
		return ExpertReplayDriftReport{}, err
	}
	if windows > len(trace.Events) {
		return ExpertReplayDriftReport{}, fmt.Errorf("model: %d horizon windows exceed %d sized events", windows, len(trace.Events))
	}

	report := ExpertReplayDriftReport{
		Name: trace.Name, Source: trace.Source, BudgetBytes: trace.BudgetBytes,
		WholeTraceGoodDecision: whole.PagedRingLRU.GoodDecisionRatio,
		Windows:                make([]ExpertHorizonWindow, 0, windows),
	}
	gdrs := make([]float64, 0, windows)
	for i, start := 0, 0; i < windows; i++ {
		end := start + horizonWindowSize(len(trace.Events), windows, i)
		sub := ExpertAccessTrace{
			Schema: trace.Schema, Name: trace.Name, Source: trace.Source,
			BudgetBytes: trace.BudgetBytes, Events: trace.Events[start:end],
		}
		rep, err := ReplayExpertAccessTrace(sub, policies...)
		if err != nil {
			return ExpertReplayDriftReport{}, fmt.Errorf("model: horizon window %d [%d:%d): %w", i, start, end, err)
		}
		window := ExpertHorizonWindow{
			Index: i, StartEvent: start, EndEvent: end, Events: end - start,
			LRUGoodDecision: rep.PagedRingLRU.GoodDecisionRatio, LRUHitBytes: rep.PagedRingLRU.HitTokens,
			OracleHitBytes: rep.Oracle.HitTokens, OracleExact: rep.Oracle.Exact, Locality: rep.Locality,
			LocalityInverted: rep.Locality.Samples > 0 &&
				rep.Locality.RecencyNextUseCorrelation <= rep.Locality.LayerSequentialNullCorrelation,
		}
		report.Windows = append(report.Windows, window)
		gdrs = append(gdrs, window.LRUGoodDecision)
		if window.LocalityInverted {
			report.LocalityInversions++
		}
		start = end
	}

	report.GoodDecisionMin, report.GoodDecisionMax, report.WorstWindow = gdrs[0], gdrs[0], 0
	for i, gdr := range gdrs {
		if gdr < report.GoodDecisionMin {
			report.GoodDecisionMin, report.WorstWindow = gdr, i
		}
		if gdr > report.GoodDecisionMax {
			report.GoodDecisionMax = gdr
		}
	}
	report.GoodDecisionSpread = report.GoodDecisionMax - report.GoodDecisionMin
	report.GoodDecisionTrendSlope = gdrTrendSlope(gdrs)
	return report, nil
}

// horizonWindowSize returns the length of contiguous window i when n events are split into
// w windows, distributing the remainder to the earliest windows so sizes differ by at most
// one. Callers guarantee 1 <= w <= n, so every window receives at least one event.
func horizonWindowSize(n, w, i int) int {
	base, rem := n/w, n%w
	if i < rem {
		return base + 1
	}
	return base
}

// gdrTrendSlope is the least-squares slope of gdr against its index. A single window (or an
// otherwise degenerate index axis) has no trend and returns 0.
func gdrTrendSlope(gdr []float64) float64 {
	n := len(gdr)
	if n < 2 {
		return 0
	}
	var sx, sy float64
	for i, v := range gdr {
		sx += float64(i)
		sy += v
	}
	mx, my := sx/float64(n), sy/float64(n)
	var cov, vx float64
	for i, v := range gdr {
		dx := float64(i) - mx
		cov += dx * (v - my)
		vx += dx * dx
	}
	if vx == 0 {
		return 0
	}
	return cov / vx
}
