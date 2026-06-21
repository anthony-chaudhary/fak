package cachemeta

// This file adds the double-count guard for provider prompt-prefix cache hits
// (issue #112). The representation already lives in provider.go:
// FromProviderCache folds Anthropic/OpenAI `cached_tokens` into an Entry with
// residency=provider and MediaType=prompt_prefix, and ProviderCacheVerdict makes
// the no-local-trust-claim rule mechanical. What was missing is a guard that a
// token-counting benchmark cannot accidentally fold a provider savings into a
// LOCAL reuse total. These helpers make that separation a code rule rather than a
// convention a benchmark author has to remember.

// IsProviderResidency reports whether an entry's payload lives on a remote
// provider (Anthropic/OpenAI/etc.) rather than in a locally-owned cache tier. A
// provider-resident entry is cost/latency telemetry only; its tokens were never
// re-served by a fak-local cache, so they must not be attributed to a local
// reuse win.
func IsProviderResidency(e Entry) bool {
	return e.Residency.Tier == TierProvider
}

// IsProviderPrefixHit reports whether an entry is a provider-resident
// prompt-prefix cache hit — the shape FromProviderCache produces. The goal
// language for #112 names this as plane=prompt_prefix, residency=provider; the
// shipped entry carries MediaType=prompt_prefix with provider residency, so this
// predicate identifies it regardless of the Plane label other code already keys
// on (PlaneProvider vs the PlanePrompt="prompt_prefix" alias). Callers and
// benchmarks use this to route an entry to the provider counter, never the local
// one.
func IsProviderPrefixHit(e Entry) bool {
	return IsProviderResidency(e) && e.ID.MediaType == MediaPromptPrefix
}

// LocalReuseTokens returns the prefill tokens an entry may be credited to a
// LOCAL reuse win, and zero for any provider-resident entry. This is the
// double-count guard: a token-counting benchmark that sums LocalReuseTokens over
// a mixed entry stream can never count a provider hit's cached_tokens as a local
// saving, because a provider entry contributes 0 here. The same provider tokens
// are still observable via ProviderReadTokens for the distinct provider metric.
func LocalReuseTokens(e Entry) int64 {
	if IsProviderResidency(e) {
		return 0
	}
	return e.Metrics.PrefillTokensSaved
}

// ProviderReadTokens returns the provider-reported cache-read (cached) tokens for
// a provider-resident prompt-prefix entry, and zero otherwise. This is the
// counterpart to LocalReuseTokens: a metrics sink credits these to the provider
// cache counter, kept DISTINCT from the local KV/tool reuse total. A local entry
// contributes 0 here, so the two accumulators partition the token stream with no
// overlap.
func ProviderReadTokens(e Entry) int64 {
	if !IsProviderResidency(e) {
		return 0
	}
	return e.Metrics.PrefillTokensSaved
}

// SavingsSplit partitions a stream of cache entries into local-reuse tokens and
// provider-cache tokens with no double counting: every entry contributes to at
// most one side. A benchmark folds its observed entries through this to get the
// two totals it may report — a local reuse win and a (separately-labeled)
// provider cost saving — without ever conflating them.
type SavingsSplit struct {
	LocalReuseTokens   int64 // tokens re-served by a fak-local cache (a real local win)
	ProviderReadTokens int64 // provider-reported cached tokens (cost telemetry, NOT a local win)
	ProviderHits       int64 // count of provider prompt-prefix hits observed
}

// Add folds one entry into the split, honoring the double-count guard. A
// provider-resident entry's tokens land only in ProviderReadTokens; a
// locally-resident entry's tokens land only in LocalReuseTokens.
func (s *SavingsSplit) Add(e Entry) {
	if IsProviderResidency(e) {
		s.ProviderReadTokens += ProviderReadTokens(e)
		if IsProviderPrefixHit(e) {
			s.ProviderHits++
		}
		return
	}
	s.LocalReuseTokens += LocalReuseTokens(e)
}
