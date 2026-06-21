package rsiloop

// kpi.go is the demo KPI the RSI loop measures: a deterministic LRU cache
// hit-rate over a fixed reference access trace. It is chosen to honor the repo's
// determinism rule for a witness metric (docs/proofs/00-METHOD.md): no wall-clock,
// no RNG, so it reproduces BIT-FOR-BIT on any platform. That is what makes it a
// legal RSI witness — the keep/revert decision is the same on a Mac and a Windows
// box, so a KEEP can't be a one-box fluke.
//
// The metric is monotonically non-decreasing in the cache size (a larger LRU
// window can only turn misses into hits, never the reverse), and strictly rises
// over the candidate range until the cache covers the working set. So the loop has
// a REAL gain to find: raise the cache size, the measured hit-rate goes up, and
// the non-forgeable keep-bit fires — driven by the measurement, not a flag.

// referenceTrace is the fixed access pattern HitRate is measured over. It is built
// once at package init from a deterministic generator (no RNG) whose reuse-distance
// histogram is spread across 1..maxReuse, so HitRate(N) — the fraction of accesses
// whose reuse distance is < N — climbs smoothly with N. buildReferenceTrace
// documents the exact construction.
var referenceTrace = buildReferenceTrace()

// workingSet is the number of distinct keys in the trace; HitRate saturates once
// the cache covers it.
const workingSet = 12

// buildReferenceTrace emits, for each reuse distance d in 1..workingSet-1, several
// episodes "touch X, touch d distinct fillers, touch X again" so the second touch
// is a hit exactly when the cache holds > d distinct keys. Spreading d across the
// range gives a hit-rate curve that rises with cache size instead of jumping. The
// construction is pure and deterministic — the same slice on every platform.
func buildReferenceTrace() []int {
	var tr []int
	// A pool of filler keys disjoint from the "target" key space so a filler never
	// accidentally is the target being measured. Targets use [0, workingSet);
	// fillers use [100, 100+workingSet).
	const episodesPerDistance = 6
	for d := 1; d < workingSet; d++ {
		for e := 0; e < episodesPerDistance; e++ {
			target := (d*7 + e) % workingSet // deterministic spread over the target space
			tr = append(tr, target)
			for f := 0; f < d; f++ {
				tr = append(tr, 100+((d+e+f)%workingSet)) // d distinct-ish fillers
			}
			tr = append(tr, target) // reuse: hit iff cache held > d distinct since first touch
		}
	}
	return tr
}

// HitRate computes the LRU cache hit-rate over the reference trace for the given
// cache size. Deterministic, allocation-light, wall-clock-free. cacheSize <= 0 is
// treated as 0 (everything misses). This is the single source of truth for the
// metric: the kpiprobe prints it, the worktree Measurer parses that print, and the
// tests assert against it — so the loop, the probe, and the gate never disagree.
func HitRate(cacheSize int) float64 {
	if len(referenceTrace) == 0 {
		return 0
	}
	if cacheSize < 0 {
		cacheSize = 0
	}
	// recency[k] = the position (monotone counter) of the last access to key k.
	recency := make(map[int]int, cacheSize*2+4)
	// order holds resident keys; we evict the least-recently-used when over capacity.
	// For a small demo cache a linear scan for the LRU victim is fine and keeps the
	// code obviously correct (no heap, no generics) — determinism over cleverness.
	resident := make(map[int]bool, cacheSize+1)
	hits, total := 0, 0
	clock := 0
	for _, key := range referenceTrace {
		clock++
		total++
		if resident[key] {
			hits++
			recency[key] = clock
			continue
		}
		// miss: admit, evicting the LRU resident if at capacity.
		if cacheSize == 0 {
			continue // a zero cache holds nothing; every access misses.
		}
		if len(resident) >= cacheSize {
			victim, victimAt := -1, int(^uint(0)>>1)
			for k := range resident {
				if recency[k] < victimAt {
					victim, victimAt = k, recency[k]
				}
			}
			if victim != -1 {
				delete(resident, victim)
				delete(recency, victim)
			}
		}
		resident[key] = true
		recency[key] = clock
	}
	return float64(hits) / float64(total)
}

// TraceLen reports the number of accesses in the reference trace (for provenance in
// a journal row — the KPI is "hit-rate over N accesses", and N belongs in the
// record).
func TraceLen() int { return len(referenceTrace) }
