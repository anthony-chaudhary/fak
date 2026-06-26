package gateway

// cache_pricing.go — the prompt-cache PRICING MODEL for issue #218 (F-002,
// "Prompt Caching Features"), the "Pricing model" acceptance item.
//
// The gateway already OBSERVES the provider's prompt-cache token axes on every
// served turn — `cache_read_input_tokens` (a prefix the upstream served from its
// own cache) and `cache_creation_input_tokens` (a prefix the upstream wrote into
// its cache) — and folds them into the AdjudicationSummary the guard exit summary
// prints. What was missing was a way to turn those token counts into the COST they
// represent, so an operator can see the dollars caching saved rather than a bare
// token count whose economic meaning they have to know by heart.
//
// This is a pure, deterministic, provider-agnostic compute model: the three
// Anthropic prompt-cache price multipliers are stable constants; the model's BASE
// per-MTok input/output price is a PARAMETER the caller supplies, so this file
// never embeds (and never has to chase) a per-model price table. Same inputs →
// same dollars, with no clock, no I/O, and no network.
//
// PROVENANCE: a dollar figure this model derives from CachedPromptTokens (the
// upstream's reported cache_read) is a COST PROJECTION over an OBSERVED quantity —
// fak relays the provider's token counts; it does not author them. The saving is
// therefore reported as cost/latency evidence, never as a fak-WITNESSED claim, in
// keeping with the same OBSERVED-vs-WITNESSED discipline metrics.go applies to the
// raw counters (see [AdjudicationSummary.CachedPromptTokens]).

// Anthropic prompt-cache price multipliers, expressed RELATIVE to the model's base
// input per-token price. These are the published cache economics:
//
//   - a cache READ (cache_read_input_tokens) bills at ~0.1× the base input price;
//   - a cache WRITE bills at a premium over base input — 1.25× for the default
//     5-minute TTL, 2.0× for the 1-hour TTL.
//
// The asymmetry is why caching is a net win only once reads accrue: the first
// write costs MORE than an uncached read (1.25×/2.0× vs 1.0×), and each subsequent
// read recovers 0.9× of base. The break-even is two requests at 5m TTL
// (1.25 + 0.1 = 1.35 < 2.0) and three at 1h TTL (2.0 + 0.2 = 2.2 < 3.0) — a fact
// [CachePricing.SavingsUSD] makes mechanical by pricing the write premium as a
// negative saving rather than hiding it.
const (
	// CacheReadMultiplier is the price of a cached-prefix READ relative to base input.
	CacheReadMultiplier = 0.1
	// CacheWrite5mMultiplier is the price of a 5-minute-TTL cache WRITE relative to base input.
	CacheWrite5mMultiplier = 1.25
	// CacheWrite1hMultiplier is the price of a 1-hour-TTL cache WRITE relative to base input.
	CacheWrite1hMultiplier = 2.0
)

// CacheTTL names the cache_control TTL a write was placed under. It mirrors the
// Anthropic `cache_control` grammar: the bare `{"type":"ephemeral"}` breakpoint is
// the 5-minute tier, and `{"type":"ephemeral","ttl":"1h"}` is the 1-hour tier.
type CacheTTL string

const (
	// CacheTTL5m is the default ephemeral cache tier (5-minute TTL).
	CacheTTL5m CacheTTL = "5m"
	// CacheTTL1h is the extended ephemeral cache tier (1-hour TTL).
	CacheTTL1h CacheTTL = "1h"
)

// WriteMultiplier returns the cache-WRITE price multiplier (relative to base input)
// for this TTL. An unset or unrecognized TTL falls back to the 5-minute tier — the
// default the gateway forwards when a client supplies a bare ephemeral breakpoint —
// so a missing TTL is priced conservatively (the cheaper write), never as a free one.
func (t CacheTTL) WriteMultiplier() float64 {
	if t == CacheTTL1h {
		return CacheWrite1hMultiplier
	}
	return CacheWrite5mMultiplier
}

// CacheUsage is one served turn's token accounting on the four billable axes the
// Anthropic usage block reports. It is a plain-data projection of the upstream
// usage — InputTokens is the uncached remainder billed at full price, CacheReadTokens
// is the prefix served from cache (0.1×), CacheCreationTokens is the prefix written
// to cache (WriteTTL's multiplier), and OutputTokens is the generated completion.
// WriteTTL is the tier the write was placed under (defaults to 5m when zero).
type CacheUsage struct {
	InputTokens         int
	CacheReadTokens     int
	CacheCreationTokens int
	OutputTokens        int
	WriteTTL            CacheTTL
}

// CachePricing is a model's BASE per-million-token price on the two axes a turn is
// billed on. The cache multipliers above are applied ON TOP of InputPerMTokUSD;
// OutputPerMTokUSD prices the completion. The caller supplies the numbers for the
// model in play (e.g. Opus 4.8 = {5, 25}, Sonnet 4.6 = {3, 15}, Haiku 4.5 = {1, 5}),
// so this model stays correct as prices change without re-touching this file.
type CachePricing struct {
	InputPerMTokUSD  float64
	OutputPerMTokUSD float64
}

// perToken converts a per-MTok price to a per-token price.
func perToken(perMTok float64) float64 { return perMTok / 1_000_000 }

// CostUSD is the actual dollar cost of a turn under prompt caching: the uncached
// input at 1.0×, the cache read at 0.1×, the cache write at its TTL multiplier, plus
// the output at the output price. This is what the turn DID cost given the cache hits
// and misses the provider reported.
func (p CachePricing) CostUSD(u CacheUsage) float64 {
	in := perToken(p.InputPerMTokUSD)
	cost := float64(u.InputTokens) * in
	cost += float64(u.CacheReadTokens) * in * CacheReadMultiplier
	cost += float64(u.CacheCreationTokens) * in * u.WriteTTL.WriteMultiplier()
	cost += float64(u.OutputTokens) * perToken(p.OutputPerMTokUSD)
	return cost
}

// UncachedCostUSD is the COUNTERFACTUAL cost of the same turn with no caching: every
// prompt token — the uncached remainder, the would-be cache read, and the would-be
// cache write — billed at the full input price, plus the same output. It is the
// baseline SavingsUSD measures against, and is always computed from the same token
// counts so the two can be differenced exactly.
func (p CachePricing) UncachedCostUSD(u CacheUsage) float64 {
	in := perToken(p.InputPerMTokUSD)
	promptTokens := u.InputTokens + u.CacheReadTokens + u.CacheCreationTokens
	return float64(promptTokens)*in + float64(u.OutputTokens)*perToken(p.OutputPerMTokUSD)
}

// SavingsUSD is UncachedCostUSD − CostUSD: the dollars caching saved on this turn.
// It is honest about the write premium — a turn that only WROTE the cache (a cold
// miss) has CacheCreationTokens priced ABOVE base input, so its saving is NEGATIVE,
// and it takes the later reads to pull the running total positive. Output tokens
// cancel in the difference, so the result is purely the prompt-cache effect:
//
//	savings = [read×(1 − 0.1) + write×(1 − writeMult)] × inputPricePerToken
func (p CachePricing) SavingsUSD(u CacheUsage) float64 {
	return p.UncachedCostUSD(u) - p.CostUSD(u)
}

// ProviderCacheSavingsUSD prices the provider prompt-cache reuse this summary has
// OBSERVED across the session: the cumulative cache_read tokens (CachedPromptTokens),
// each of which billed at 0.1× instead of the full input price, valued at the model's
// base input price the caller supplies. It is the dollar companion to the cached-token
// count the guard exit summary already prints — a COST PROJECTION over an observed
// quantity, not a fak-authored claim — so a consumer can show "$X saved by cache reuse"
// rather than leaving the operator to translate a token count into money.
//
// It folds the READ axis only (the unambiguous win); the write premium is per-turn and
// not retained on the summary, so this never overstates the saving by ignoring a cost.
func (s AdjudicationSummary) ProviderCacheSavingsUSD(inputPerMTokUSD float64) float64 {
	return float64(s.CachedPromptTokens) * perToken(inputPerMTokUSD) * (1 - CacheReadMultiplier)
}
