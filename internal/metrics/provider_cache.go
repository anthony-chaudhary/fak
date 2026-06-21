package metrics

import "github.com/anthony-chaudhary/fak/internal/cachemeta"

// This file emits provider prompt-prefix cache telemetry into the report's Arm as
// a metric DISTINCT from the local KV/tool counters (issue #112). The double-count
// guard lives in cachemeta (SavingsSplit / LocalReuseTokens / ProviderReadTokens):
// a provider-resident entry contributes to ProviderCache* here and never to a
// local-reuse total, so a token-counting benchmark cannot count a provider hit as
// a local win.

// FoldCacheEntry routes one cachemeta entry into the correct Arm counter. A
// provider-resident prompt-prefix hit lands in the provider counters; a local
// entry is left for the local-reuse accounting (VDSOHits / InTokens / etc.),
// which this helper deliberately does NOT touch. Returns true if the entry was a
// provider hit (so the caller knows it was handled as provider telemetry).
func (a *Arm) FoldCacheEntry(e cachemeta.Entry) bool {
	if !cachemeta.IsProviderResidency(e) {
		return false
	}
	a.ProviderCacheReadTokens += cachemeta.ProviderReadTokens(e)
	if cachemeta.IsProviderPrefixHit(e) {
		a.ProviderCacheHits++
	}
	return true
}

// FoldSavingsSplit copies a cachemeta.SavingsSplit's provider side into the Arm's
// provider counters. The local side of the split is intentionally not merged here
// — local reuse is already accounted via VDSOHits/token fields — keeping the two
// kinds of savings reported under separate labels.
func (a *Arm) FoldSavingsSplit(s cachemeta.SavingsSplit) {
	a.ProviderCacheReadTokens += s.ProviderReadTokens
	a.ProviderCacheHits += s.ProviderHits
}
