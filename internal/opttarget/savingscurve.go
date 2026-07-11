package opttarget

import "fmt"

// savingscurve.go is the savings-vs-budget curve + diminishing-returns knee over
// one recorded key-log (#3398, borrow D2-savings-vs-budget-curve). The existing
// sweep core (cachesize.go's int-sweep target, rsiloop's HitRate replay) drives a
// HigherBetter metric, so the loop always drifts toward the largest cache; this
// file answers the orthogonal question — where does one MORE unit of budget stop
// paying for itself? — by replaying the same trace at every budget and locating
// the elbow of the resulting savings curve. Pure and deterministic: no
// wall-clock, no RNG, no I/O. rsiloop's replay is closed over its own fixed
// reference trace (HitRate takes no trace argument), so this file carries a
// small local LRU replay generic over the caller's trace instead of importing
// that one.

// kneeFraction is the diminishing-returns threshold KneeBudget applies: a curve
// point still "pays for itself" while its marginal savings per unit of budget is
// at least this fraction of the PEAK marginal gain seen anywhere on the curve.
// Past the last such point, every extra unit of budget returns less than a tenth
// of what the best unit returned — the collapse the knee names.
const kneeFraction = 0.10

// SavingsPoint is one measured point on the savings-vs-budget curve: an LRU
// replay of the whole trace with a cache of Budget entries.
type SavingsPoint struct {
	// Budget is the cache capacity, in entries (keys), this point was replayed at.
	Budget int `json:"budget"`
	// HitRate is hit_keys/total_keys over the full trace at this budget.
	HitRate float64 `json:"hit_rate"`
	// Savings is the $-savings PROXY: the count of hit keys — accesses served
	// from cache instead of recomputed (hit keys treated as tokens saved). A
	// caller with a real cost model multiplies by its per-key cost; the knee is
	// invariant under that constant rescale (the threshold is a FRACTION of the
	// peak marginal gain), so the raw count is a sufficient proxy for knee
	// selection.
	Savings float64 `json:"savings"`
}

// SavingsCurve is the full savings-vs-budget replay of one key-log, plus the
// detected diminishing-returns knee.
type SavingsCurve struct {
	// Points holds one SavingsPoint per requested budget, in budget order.
	Points []SavingsPoint `json:"points"`
	// Knee is the diminishing-returns budget; SavingsVsBudget documents the
	// exact definition.
	Knee int `json:"knee"`
	// TraceLen is the number of accesses replayed — provenance for a journal or
	// scorecard row: the curve is "savings over N accesses", and N belongs in
	// the record.
	TraceLen int `json:"trace_len"`
}

// SavingsVsBudget replays one access trace (a key sequence — e.g. token/block
// ids from a recorded key-log) through an LRU cache at each of the given
// budgets and returns the savings-vs-budget curve plus the diminishing-returns
// knee. It is pure and deterministic (no wall-clock, no RNG), so the same
// (trace, budgets) input yields the identical curve on any platform — the same
// determinism rule rsiloop's HitRate honors.
//
// Knee definition (diminishing returns): for consecutive points the marginal
// gain is (Savings[i]-Savings[i-1]) / (Budget[i]-Budget[i-1]) — extra savings
// per extra unit of budget. The knee is the budget of the LAST point whose
// marginal gain is at least kneeFraction of the peak marginal gain: every
// budget past it buys less than that fraction of the best unit's return, for
// good. A flat curve (no reuse, or the smallest budget already covers the
// working set) has peak 0 and the knee is the smallest budget — spending more
// buys nothing. A single budget has no marginals and is its own knee. The knee
// is grid-limited: it is always one of the REQUESTED budgets, so a saturation
// point falling between two requested budgets reports as the first requested
// budget at or above it.
//
// A malformed request is REFUSED, never silently lowered into a meaningless
// curve: the trace must be non-empty (a hit rate over zero accesses is
// undefined) and budgets must be non-negative and strictly increasing (the
// marginal-gain denominator must be positive). A budget of 0 is legal and
// holds nothing — every access misses.
func SavingsVsBudget(trace []int, budgets []int) (SavingsCurve, error) {
	if len(trace) == 0 {
		return SavingsCurve{}, fmt.Errorf("opttarget savings-curve: empty trace")
	}
	if len(budgets) == 0 {
		return SavingsCurve{}, fmt.Errorf("opttarget savings-curve: no budgets")
	}
	for i, b := range budgets {
		if b < 0 {
			return SavingsCurve{}, fmt.Errorf("opttarget savings-curve: negative budget %d at index %d", b, i)
		}
		if i > 0 && b <= budgets[i-1] {
			return SavingsCurve{}, fmt.Errorf("opttarget savings-curve: budgets must be strictly increasing (budgets[%d]=%d, budgets[%d]=%d)", i-1, budgets[i-1], i, b)
		}
	}
	total := len(trace)
	points := make([]SavingsPoint, 0, len(budgets))
	for _, b := range budgets {
		hits := lruHits(trace, b)
		points = append(points, SavingsPoint{
			Budget:  b,
			HitRate: float64(hits) / float64(total),
			Savings: float64(hits),
		})
	}
	return SavingsCurve{Points: points, Knee: kneeBudget(points), TraceLen: total}, nil
}

// kneeBudget locates the diminishing-returns knee of a budget-ordered curve —
// the definition SavingsVsBudget documents. Callers guarantee at least one
// point with strictly increasing budgets.
func kneeBudget(points []SavingsPoint) int {
	knee := points[0].Budget
	if len(points) == 1 {
		return knee
	}
	// marginal[i] is the savings gained per unit of budget moving from point i-1
	// to point i; marginal[0] stays 0 (no prior point).
	marginal := make([]float64, len(points))
	peak := 0.0
	for i := 1; i < len(points); i++ {
		span := float64(points[i].Budget - points[i-1].Budget) // > 0: strictly increasing
		marginal[i] = (points[i].Savings - points[i-1].Savings) / span
		if marginal[i] > peak {
			peak = marginal[i]
		}
	}
	if peak <= 0 {
		// Flat curve: no budget past the smallest ever bought anything, so the
		// smallest budget is already at diminishing returns.
		return knee
	}
	threshold := kneeFraction * peak
	for i := 1; i < len(points); i++ {
		if marginal[i] >= threshold {
			knee = points[i].Budget
		}
	}
	return knee
}

// lruHits replays trace through an LRU cache of budget entries and returns the
// hit count. It mirrors rsiloop's HitRate replay — a linear scan for the LRU
// victim keeps the code obviously correct (no heap, no generics): determinism
// over cleverness — but is generic over the caller's trace instead of a
// package-fixed one. budget <= 0 holds nothing: every access misses.
func lruHits(trace []int, budget int) int {
	if budget <= 0 {
		return 0
	}
	// recency[k] = the position (monotone counter) of the last access to key k,
	// tracked for resident keys only.
	recency := make(map[int]int, budget+1)
	resident := make(map[int]bool, budget+1)
	hits, clock := 0, 0
	for _, key := range trace {
		clock++
		if resident[key] {
			hits++
			recency[key] = clock
			continue
		}
		// miss: admit, evicting the least-recently-used resident at capacity.
		if len(resident) >= budget {
			victim, victimAt, found := 0, 0, false
			for k := range resident {
				if !found || recency[k] < victimAt {
					victim, victimAt, found = k, recency[k], true
				}
			}
			if found {
				delete(resident, victim)
				delete(recency, victim)
			}
		}
		resident[key] = true
		recency[key] = clock
	}
	return hits
}
