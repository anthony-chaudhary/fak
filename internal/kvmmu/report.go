package kvmmu

import "sort"

// report.go — issue #857, rung 6 of the attention-witness epic (#851): the post-hoc analyst surface.
//
// Rung 4 (#855, accumulator.go) keeps, per span, the undecayed cumulative attention mass and a
// per-turn trajectory. This file folds that into the session-integrated picture an operator reads
// to answer "was this 9000-token blob ever worth its residency, and for how many turns was it dead
// weight?": the hottest spans by cumulative mass, the coldest-but-resident spans (the dead weight),
// the S/N(t) curve over the run, and a "bloated since turn K" marker.
//
// This is the WITNESSED counterpart of the density proxy the session auditor computes from raw
// transcripts (tools/session_audit.py). Where the attention witness is available, the integrated
// S/N here is the real thing; offline / API-only sessions fall back to the proxy. Same WITNESSED-
// vs-OBSERVED line the rest of the epic holds.
//
// The report takes the per-turn S/N as plain recorded floats (TurnSN), not a ctxplan type, so it
// stays dependency-free: a session driver computes each turn's S/N via
// ctxplan.SignalNoiseFromAttention (#854) and feeds Ratio() in. Feed the accumulator from a
// post-hoc (λ=1, no-Forget) pass so it carries every span's full-session history — that is what
// makes "cold-but-resident-through-the-session" an honest label.

// TurnSN is one recorded sample of a turn's witnessed signal-to-noise plus the resident token cost
// that turn (the integral weight) and the provider cache-hit ratio. The cache-hit field is what lets
// the report expose the exact pathology from docs/explainers/context-signal-to-noise.md: S/N(t)
// falling while cache-hit climbs (a window that bloats even as it "hits" more).
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

// SessionAttentionReport is the post-hoc session picture (#857).
type SessionAttentionReport struct {
	Turns        int          `json:"turns"`
	Hottest      []SpanReport `json:"hottest"`       // top-N by cumulative attention mass
	DeadWeight   []SpanReport `json:"dead_weight"`   // coldest-but-resident: the bloat (cold, ranked by cost)
	Curve        []TurnSN     `json:"curve"`         // S/N(t) over the run
	IntegratedSN float64      `json:"integrated_sn"` // cost-weighted mean of S/N(t) over the run
	BloatedSince int          `json:"bloated_since"` // first turn from which S/N declines as cache-hit climbs, or -1
}

// BuildSessionAttentionReport folds the accumulator and a recorded per-turn S/N curve into the
// session report. topN bounds the hottest / dead-weight lists; deadThreshold is the cumulative-mass
// ceiling below which a still-tracked span counts as dead weight (cold). cost maps span id to its
// resident token count (may be nil — dead weight then ranks by coldness alone; with cost it ranks the
// big cold blobs first, the ones whose residency cost the most for the least attention).
//
// IntegratedSN is the cost-weighted mean Σ_t Ratio(t)·Cost(t) / Σ_t Cost(t): the run's S/N weighted
// by how much window each turn carried (a lean turn and a bloated turn are not equal votes). With all
// costs zero or an empty curve it is 0.
//
// Pure and deterministic: same (accumulator snapshot, curve, cost) yields the same report.
func BuildSessionAttentionReport(acc *AttentionAccumulator, curve []TurnSN, cost map[string]int, topN int, deadThreshold float64) SessionAttentionReport {
	r := SessionAttentionReport{
		Turns:        acc.Turns(),
		Curve:        append([]TurnSN(nil), curve...),
		IntegratedSN: integratedSN(curve),
		BloatedSince: bloatedSince(curve),
	}

	// Build a SpanReport for every span the accumulator still holds.
	snap := acc.Snapshot()
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

	// Hottest: highest cumulative mass first (ties by id for determinism).
	hot := append([]SpanReport(nil), rows...)
	sort.Slice(hot, func(i, j int) bool {
		if hot[i].Cumulative != hot[j].Cumulative {
			return hot[i].Cumulative > hot[j].Cumulative
		}
		return hot[i].ID < hot[j].ID
	})
	r.Hottest = clip(hot, topN)

	// Dead weight: still-resident spans at or below the cold threshold, ranked by cost desc (the big
	// cold blobs first) then by coldest cumulative, then id. Names the residency that did not earn its keep.
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
