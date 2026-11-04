package gateway

// uncachedPrefillCandidate describes an already-admissible replica in a single
// coherent snapshot. CachedTokens must count confirmed reusable prompt tokens,
// not message segments, optimistic routing observations, or cache capacity.
// PendingTokens counts uncached prefill work already reserved on this replica.
type uncachedPrefillCandidate struct {
	Worker        string
	CachedTokens  int
	PendingTokens int
	CacheKnown    bool
}

type uncachedPrefillChoice struct {
	Worker          string
	UncachedTokens  int
	ProjectedTokens int
	PeakTokens      int
}

// chooseUncachedPrefill minimizes the projected fleet peak after assigning one
// request. Ties prefer lower selected-replica load, then lexical worker identity.
// Unknown or invalid metadata declines the entire snapshot so callers can retain
// their existing policy rather than silently treating missing work as free.
// This pure scorer does not reserve work, establish residency, or change routing.
func chooseUncachedPrefill(promptTokens int, promptKnown bool, candidates []uncachedPrefillCandidate) (uncachedPrefillChoice, bool) {
	if !promptKnown || promptTokens < 0 || len(candidates) == 0 {
		return uncachedPrefillChoice{}, false
	}
	maxInt := int(^uint(0) >> 1)
	peak := 0
	seen := make(map[string]struct{}, len(candidates))
	for _, c := range candidates {
		if c.Worker == "" || !c.CacheKnown || c.CachedTokens < 0 || c.CachedTokens > promptTokens || c.PendingTokens < 0 {
			return uncachedPrefillChoice{}, false
		}
		if _, duplicate := seen[c.Worker]; duplicate {
			return uncachedPrefillChoice{}, false
		}
		seen[c.Worker] = struct{}{}
		// Decline unrepresentable projections instead of wrapping or collapsing
		// distinct excessive loads into a saturated tie.
		if c.PendingTokens > maxInt-(promptTokens-c.CachedTokens) {
			return uncachedPrefillChoice{}, false
		}
		if c.PendingTokens > peak {
			peak = c.PendingTokens
		}
	}
	var best uncachedPrefillChoice
	for i, c := range candidates {
		uncached := promptTokens - c.CachedTokens
		projected := c.PendingTokens + uncached
		projectedPeak := peak
		if projected > projectedPeak {
			projectedPeak = projected
		}
		choice := uncachedPrefillChoice{c.Worker, uncached, projected, projectedPeak}
		if i == 0 || choice.PeakTokens < best.PeakTokens ||
			(choice.PeakTokens == best.PeakTokens && (choice.ProjectedTokens < best.ProjectedTokens ||
				(choice.ProjectedTokens == best.ProjectedTokens && choice.Worker < best.Worker))) {
			best = choice
		}
	}
	return best, true
}
