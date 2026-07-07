package gateway

import "strings"

// deepseek_pricing.go — DeepSeek V4 prompt-cache PRICING METADATA (#3009, under the
// DeepSeek V4 support program #3006).
//
// DeepSeek prices the prompt on TWO explicit axes instead of Anthropic's
// multiplier-over-base model: a cache-HIT input rate and a cache-MISS input rate,
// each a direct per-MTok price, plus the output rate. The token counters they bill
// against are the top-level `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
// fields agent.Usage normalizes (CachedPromptTokens / UncachedPromptTokens), so a
// priced DeepSeek turn is hit×hitRate + miss×missRate + output×outputRate.
//
// PROVENANCE: everything in this file is PROVIDER-OBSERVED / PROVIDER-PRICED.
// DeepSeek's context caching is on by default and the hit counter is the provider's
// own relayed number — a DeepSeek cache hit is NEVER booked as fak-authored savings
// unless a separate fak-authored mechanism demonstrably shaped the request. This is
// the same OBSERVED-vs-WITNESSED discipline cache_pricing.go applies to the
// Anthropic counters.
//
// Rates sourced from https://api-docs.deepseek.com/quick_start/pricing as read on
// 2026-07-07 (per 1M tokens, USD). The table FAILS CLOSED: an unknown model returns
// ok=false so no caller ever prices DeepSeek usage off a stale or guessed row.
const (
	// DeepSeekV4ProCacheHitInputPerMTokUSD is deepseek-v4-pro's cache-HIT input rate.
	DeepSeekV4ProCacheHitInputPerMTokUSD = 0.003625
	// DeepSeekV4ProCacheMissInputPerMTokUSD is deepseek-v4-pro's cache-MISS input rate.
	DeepSeekV4ProCacheMissInputPerMTokUSD = 0.435
	// DeepSeekV4ProOutputPerMTokUSD is deepseek-v4-pro's output rate.
	DeepSeekV4ProOutputPerMTokUSD = 0.87

	// DeepSeekV4FlashCacheHitInputPerMTokUSD is deepseek-v4-flash's cache-HIT input rate.
	DeepSeekV4FlashCacheHitInputPerMTokUSD = 0.0028
	// DeepSeekV4FlashCacheMissInputPerMTokUSD is deepseek-v4-flash's cache-MISS input rate.
	DeepSeekV4FlashCacheMissInputPerMTokUSD = 0.14
	// DeepSeekV4FlashOutputPerMTokUSD is deepseek-v4-flash's output rate.
	DeepSeekV4FlashOutputPerMTokUSD = 0.28

	// DeepSeekPricingSourceV4Pro / ...V4Flash name the price rows in ledgers, in the
	// same source-string convention as CachePricingSourceAnthropicClaudeOpus48.
	DeepSeekPricingSourceV4Pro   = "provider-priced:deepseek/deepseek-v4-pro@2026-07-07"
	DeepSeekPricingSourceV4Flash = "provider-priced:deepseek/deepseek-v4-flash@2026-07-07"
)

// DeepSeekCachePricing is one DeepSeek model's per-million-token price row on the
// three axes DeepSeek bills a turn on. Unlike CachePricing (base price × Anthropic
// cache multipliers), the hit and miss rates are DIRECT provider-published prices —
// there is no fak-chosen multiplier anywhere in the row.
type DeepSeekCachePricing struct {
	CacheHitInputPerMTokUSD  float64
	CacheMissInputPerMTokUSD float64
	OutputPerMTokUSD         float64
}

// DeepSeekCachePricingFor resolves the built-in DeepSeek V4 price table for a model
// id. It FAILS CLOSED: only the two current V4 rows (plus the documented pre-V4
// aliases deepseek-chat / deepseek-reasoner, which DeepSeek maps to the flash tier
// until their 2026-07-24 deprecation) resolve; any other model returns ok=false so
// the caller must stay dollar-blind rather than price against a fabricated row.
func DeepSeekCachePricingFor(model string) (DeepSeekCachePricing, string, bool) {
	switch strings.ToLower(strings.TrimSpace(model)) {
	case "deepseek-v4-pro":
		return DeepSeekCachePricing{
			CacheHitInputPerMTokUSD:  DeepSeekV4ProCacheHitInputPerMTokUSD,
			CacheMissInputPerMTokUSD: DeepSeekV4ProCacheMissInputPerMTokUSD,
			OutputPerMTokUSD:         DeepSeekV4ProOutputPerMTokUSD,
		}, DeepSeekPricingSourceV4Pro, true
	case "deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner":
		return DeepSeekCachePricing{
			CacheHitInputPerMTokUSD:  DeepSeekV4FlashCacheHitInputPerMTokUSD,
			CacheMissInputPerMTokUSD: DeepSeekV4FlashCacheMissInputPerMTokUSD,
			OutputPerMTokUSD:         DeepSeekV4FlashOutputPerMTokUSD,
		}, DeepSeekPricingSourceV4Flash, true
	}
	return DeepSeekCachePricing{}, "", false
}

// CostUSD prices one DeepSeek turn from the provider's own counters: the cache-hit
// tokens at the hit rate, the cache-miss tokens at the miss rate, and the completion
// at the output rate. A COST PROJECTION over OBSERVED provider-relayed counters —
// the hit/miss split is DeepSeek's default context caching at work, so the implied
// discount is provider-observed economics, never a fak-authored savings claim.
func (p DeepSeekCachePricing) CostUSD(hitTokens, missTokens, outputTokens int) float64 {
	return float64(hitTokens)*perToken(p.CacheHitInputPerMTokUSD) +
		float64(missTokens)*perToken(p.CacheMissInputPerMTokUSD) +
		float64(outputTokens)*perToken(p.OutputPerMTokUSD)
}
