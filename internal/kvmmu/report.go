package kvmmu

import "sort"

// report.go — post-hoc attention reporting and signal-to-noise aggregation.

// TurnSN records one turn's witnessed signal-to-noise ratio, resident cost, and cache-hit ratio.
type TurnSN struct {
	Turn     int     `json:"turn"`
	Ratio    float64 `json:"ratio"`     // witnessed S/N for the turn (ctxplan SignalNoise.Ratio)
	Cost     int     `json:"cost"`      // resident token cost that turn — the integral weight
	CacheHit float64 `json:"cache_hit"` // provider prefix cache-hit ratio that turn (OBSERVED)
}

// SpanReport is one span's line in the report: its identity, the two reductions, its resident token
// cost (when known), and the first/last turn it actually drew attention — so "dead since turn K" is
// legible (LastHot..end is the span's dead-weight tail).
type SpanReport struct {
	ID         string  `json:"id"`
	Cumulative float64 `json:"cumulative"`
	EMA        float64 `json:"ema"`
	Cost       int     `json:"cost,omitempty"`
	FirstHot   int     `json:"first_hot,omitempty"` // first turn with mass > 0 (0 = never attended)
	LastHot    int     `json:"last_hot,omitempty"`  // last turn with mass > 0 (0 = never attended)
}

// SessionAttentionReport summarizes post-hoc session attention metrics.
type SessionAttentionReport struct {
	Turns        int          `json:"turns"`
	Hottest      []SpanReport `json:"hottest"`       // top-N by cumulative attention mass
	DeadWeight   []SpanReport `json:"dead_weight"`   // coldest-but-resident: the bloat (cold, ranked by cost)
	Curve        []TurnSN     `json:"curve"`         // S/N(t) over the run
	IntegratedSN float64      `json:"integrated_sn"` // cost-weighted mean of S/N(t) over the run
	BloatedSince int          `json:"bloated_since"` // first turn from which S/N declines as cache-hit climbs, or -1

	// RetainedCeiling records the retention ceiling gauge for the best topN-span retention.
	RetainedCeiling RetainedMassGauge `json:"retained_ceiling"`
}

// BuildSessionAttentionReport folds an accumulator and recorded per-turn S/N curve into a session report.
func BuildSessionAttentionReport(acc *AttentionAccumulator, curve []TurnSN, cost map[string]int, topN int, deadThreshold float64) SessionAttentionReport {
	r := SessionAttentionReport{
		Turns:        acc.Turns(),
		Curve:        append([]TurnSN(nil), curve...),
		IntegratedSN: integratedSN(curve),
		BloatedSince: bloatedSince(curve),
	}

	// Build a SpanReport for every span the accumulator still holds.
	snap := acc.Snapshot()

	// The retention CEILING (#5123): grade the mass-maximizing topN keep with the same #3901
	// gauge a live eviction emits, so the report carries the bar a decision is measured against.
	// topN <= 0 means "no bound" everywhere else here (clip), so it means every observed span
	// too — a ceiling over the whole set, which retains all of the mass by construction.
	ceilingK := topN
	if ceilingK <= 0 {
		ceilingK = len(snap)
	}
	r.RetainedCeiling = RetainedMass(acc, TopKSpansByMass(acc, ceilingK), cost)
	rows := make([]SpanReport, 0, len(snap))
	for _, s := range snap {
		first, last := hotSpan(s.Trajectory)
		rows = append(rows, SpanReport{
			ID:         s.ID,
			Cumulative: s.Cumulative,
			EMA:        s.EMA,
			Cost:       cost[s.ID],
			FirstHot:   first,
			LastHot:    last,
		})
	}

	hot := append([]SpanReport(nil), rows...)
	sort.Slice(hot, func(i, j int) bool {
		if hot[i].Cumulative != hot[j].Cumulative {
			return hot[i].Cumulative > hot[j].Cumulative
		}
		return hot[i].ID < hot[j].ID
	})
	r.Hottest = clip(hot, topN)

	dead := make([]SpanReport, 0, len(rows))
	for _, row := range rows {
		if row.Cumulative <= deadThreshold {
			dead = append(dead, row)
		}
	}
	sort.Slice(dead, func(i, j int) bool {
		if dead[i].Cost != dead[j].Cost {
			return dead[i].Cost > dead[j].Cost
		}
		if dead[i].Cumulative != dead[j].Cumulative {
			return dead[i].Cumulative < dead[j].Cumulative
		}
		return dead[i].ID < dead[j].ID
	})
	r.DeadWeight = clip(dead, topN)

	return r
}

// integratedSN is the cost-weighted mean of the per-turn S/N: Σ Ratio·Cost / Σ Cost. Returns 0 when
// total cost is 0 (or the curve is empty), so an unweighted/empty run reports an honest 0 rather than
// dividing by zero.
func integratedSN(curve []TurnSN) float64 {
	var num, den float64
	for _, t := range curve {
		num += t.Ratio * float64(t.Cost)
		den += float64(t.Cost)
	}
	if den == 0 {
		return 0
	}
	return num / den
}

// bloatedSince returns the turn id at the start of the maximal trailing window over which S/N is
// non-increasing while cache-hit is non-decreasing AND S/N net-declines — the "bloating even as it
// hits more" pathology. Returns -1 when there is no such window (needs at least two consecutive turns).
func bloatedSince(curve []TurnSN) int {
	n := len(curve)
	if n < 2 {
		return -1
	}
	j := n - 1
	for j > 0 {
		a, b := curve[j-1], curve[j]
		if b.Ratio <= a.Ratio+1e-12 && b.CacheHit >= a.CacheHit-1e-12 {
			j--
			continue
		}
		break
	}
	if j == n-1 {
		return -1 // the last pair did not qualify: no trailing decline
	}
	if curve[n-1].Ratio < curve[j].Ratio-1e-12 {
		return curve[j].Turn
	}
	return -1 // flat window (cache-hit climbed but S/N did not actually fall): not bloat
}

// hotSpan returns the first and last turn in a trajectory with mass > 0 (0,0 if the span was never
// attended). The gap from LastHot to the session end is the span's dead-weight tail.
func hotSpan(traj []TurnMass) (first, last int) {
	for _, e := range traj {
		if e.Mass > 0 {
			if first == 0 {
				first = e.Turn
			}
			last = e.Turn
		}
	}
	return first, last
}

// clip returns the first n elements of s (all of s if n <= 0 or n >= len).
func clip(s []SpanReport, n int) []SpanReport {
	if n <= 0 || n >= len(s) {
		return s
	}
	return s[:n]
}
