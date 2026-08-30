package gateway

import "github.com/anthony-chaudhary/fak/internal/cacheprice"

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

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

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
	// Sourced from cacheprice.ReadMultiplier — the ONE canonical rate the fire gate,
	// resume planner, net-true ledger and Track-2 report all price the same token at (#2798).
	CacheReadMultiplier = cacheprice.ReadMultiplier
	// CacheWrite5mMultiplier is the price of a 5-minute-TTL cache WRITE relative to base input.
	CacheWrite5mMultiplier = cacheprice.Write5mMultiplier
	// CacheWrite1hMultiplier is the price of a 1-hour-TTL cache WRITE relative to base input.
	CacheWrite1hMultiplier = cacheprice.Write1hMultiplier

	// ClaudeOpus48InputPerMTokUSD is the default guarded-Claude base input price.
	ClaudeOpus48InputPerMTokUSD = 5.0
	// ClaudeOpus48OutputPerMTokUSD is the default guarded-Claude base output price.
	ClaudeOpus48OutputPerMTokUSD = 25.0
	// CachePricingSourceAnthropicClaudeOpus48 names the default price source in ledgers.
	CachePricingSourceAnthropicClaudeOpus48 = "default:anthropic/claude-opus-4-8"

	// ClaudeFable5InputPerMTokUSD is the Fable 5 base input price: $10/MTok, Anthropic's
	// published rate (2× Opus 4.8). Cache reads get the same 90% discount every tier does
	// via CacheReadMultiplier, so this base is all the fable row needs to price. Fable is a
	// frontier tier — pricing a fable worker's row as Opus would UNDERstate its savings, so
	// this entry makes that row dollar-real instead of dollar-blind (task #3).
	ClaudeFable5InputPerMTokUSD = 10.0
	// ClaudeFable5OutputPerMTokUSD is the Fable 5 base output price: $50/MTok (5× its input,
	// the standard frontier-Anthropic ratio).
	ClaudeFable5OutputPerMTokUSD = 50.0
	// CachePricingSourceAnthropicClaudeFable5 names the fable price source in ledgers.
	CachePricingSourceAnthropicClaudeFable5 = "default:anthropic/claude-fable-5"
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
	// CacheReadMultiplier overrides the provider default only when positive.
	// Fresh measured calibration may set it; zero preserves the static table.
	CacheReadMultiplier float64
	// CacheWrite*Multiplier override the static provider tier only when positive.
	CacheWrite5mMultiplier float64
	CacheWrite1hMultiplier float64
}

// DefaultCachePricing resolves the small built-in price table used when a caller
// has provider/context but no explicit price env. It is intentionally narrow:
// the guarded Claude Code path is known to default to Opus 4.8, while other
// provider/context pairs stay unpriced so callers can mark them dollar-blind.
func DefaultCachePricing(provider, context string) (CachePricing, string, bool) {
	p := strings.ToLower(strings.TrimSpace(provider))
	c := normalizedPricingContext(context)
	switch p {
	case "anthropic", "claude":
		switch c {
		case "claude", "claude.exe", "claude-opus-4-8":
			return CachePricing{
				InputPerMTokUSD:  ClaudeOpus48InputPerMTokUSD,
				OutputPerMTokUSD: ClaudeOpus48OutputPerMTokUSD,
			}, CachePricingSourceAnthropicClaudeOpus48, true
		case "claude-fable-5":
			return CachePricing{
				InputPerMTokUSD:  ClaudeFable5InputPerMTokUSD,
				OutputPerMTokUSD: ClaudeFable5OutputPerMTokUSD,
			}, CachePricingSourceAnthropicClaudeFable5, true
		}
		if strings.Contains(c, "claude-opus-4-8") || strings.Contains(c, "opus-4-8") {
			return CachePricing{
				InputPerMTokUSD:  ClaudeOpus48InputPerMTokUSD,
				OutputPerMTokUSD: ClaudeOpus48OutputPerMTokUSD,
			}, CachePricingSourceAnthropicClaudeOpus48, true
		}
		if strings.Contains(c, "claude-fable-5") || strings.Contains(c, "fable-5") {
			return CachePricing{
				InputPerMTokUSD:  ClaudeFable5InputPerMTokUSD,
				OutputPerMTokUSD: ClaudeFable5OutputPerMTokUSD,
			}, CachePricingSourceAnthropicClaudeFable5, true
		}
	}
	return CachePricing{}, "", false
}

func normalizedPricingContext(context string) string {
	c := strings.ToLower(strings.TrimSpace(context))
	c = strings.ReplaceAll(c, "\\", "/")
	c = strings.TrimRight(c, "/")
	if i := strings.LastIndex(c, "/"); i >= 0 {
		c = c[i+1:]
	}
	return c
}

// perToken converts a per-MTok price to a per-token price.
func perToken(perMTok float64) float64 { return perMTok / 1_000_000 }

// CostUSD is the actual dollar cost of a turn under prompt caching: the uncached
// input at 1.0×, the cache read at 0.1×, the cache write at its TTL multiplier, plus
// the output at the output price. This is what the turn DID cost given the cache hits
// and misses the provider reported.
func (p CachePricing) writeMultiplier(ttl CacheTTL) float64 {
	if ttl == CacheTTL1h {
		if p.CacheWrite1hMultiplier > 0 {
			return p.CacheWrite1hMultiplier
		}
	} else if p.CacheWrite5mMultiplier > 0 {
		return p.CacheWrite5mMultiplier
	}
	return ttl.WriteMultiplier()
}

func (p CachePricing) CostUSD(u CacheUsage) float64 {
	in := perToken(p.InputPerMTokUSD)
	cost := float64(u.InputTokens) * in
	readMult := p.CacheReadMultiplier
	if readMult <= 0 {
		readMult = CacheReadMultiplier
	}
	cost += float64(u.CacheReadTokens) * in * readMult
	cost += float64(u.CacheCreationTokens) * in * p.writeMultiplier(u.WriteTTL)
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

// ProviderCacheNetSavings prices the NET realized provider prompt-cache effect this
// summary has OBSERVED across the session: the read rebate (each cache_read token
// billed at 0.1x instead of 1x) MINUS the write premium (each cache_creation token
// billed ABOVE 1x). Unlike ProviderCacheSavingsUSD, which folds the READ axis only,
// this accounts for BOTH axes, so a cold-write-only session reads NEGATIVE (REFUTED)
// until the reads repay the writes — the same break-even per-turn SavingsUSD models.
//
// It is computed by the SAME engine `fak vcache observe` uses
// (vcachegov.ProveTelemetrySavings) over ONE aggregate row, so the live gateway numbers
// (saved-token-equiv, hit rate, multiplier) are byte-identical to the offline observe
// Aggregate on the same totals — the model is linear, so one aggregate row reproduces
// the sum of N per-turn rows. The provider wire never splits 5m vs 1h creation tokens,
// but the gateway KNOWS the subset it upgraded (CacheCreationTokensUpgraded, attributed
// per witnessed observeCacheTTLUpgrade), so that slice is priced at the 1h write
// multiplier (2.0x) and the remainder at the 5m tier (#2179); an unattributed caller
// passes 0 and reproduces the old all-5m convention byte-for-byte. This attribution is
// gateway_attributed, not provider-reported — see vcacheProofFromCountersWithUpgrade and
// TestProviderCacheNetSavingsPricesUpgradedCreationAt1h.
//
// PROVENANCE: every input is OBSERVED (provider-relayed); the saving is a realized
// rebate, never a fak trust claim. The result is in INPUT-TOKEN-EQUIVALENTS (the $ dual
// is ProviderCacheSavingsUSD). InputTokens is the provider-normalized uncached remainder:
// the live Usage ingestion seam peels OpenAI/Codex cached_tokens out of their inclusive
// input_tokens, while Anthropic's already-disjoint input/cache_read pair passes through.
// Thus savings, hit_rate, and multiplier all read from the same disjoint axes.
func (s AdjudicationSummary) ProviderCacheNetSavings() vcachegov.TelemetrySavingsProof {
	return vcacheProofFromCountersWithUpgrade(s.InputTokens, s.CachedPromptTokens, s.CacheCreationTokens, s.CacheCreationTokensUpgraded)
}

// MeanTurnLatencySeconds is the session's OBSERVED mean end-to-end turn latency in
// seconds, or 0 when no turn was timed (E2ELatencyCount == 0 — a replayed/offline summary
// or a session that served nothing). OBSERVED (provider round-trip timing), never modeled:
// it is the honest per-turn cost the guard exit line prices WITNESSED turns-saved against,
// so "time saved" is (turns spared) × (a latency the session actually measured), not a
// fabricated tokens/sec constant.
func (s AdjudicationSummary) MeanTurnLatencySeconds() float64 {
	if s.E2ELatencyCount == 0 {
		return 0
	}
	return s.E2ELatencySumSeconds / float64(s.E2ELatencyCount)
}

// MechanismSavings is the owner/mechanism split for cache-like savings that the
// operator-facing surfaces render. Token-equivalent fields all use the same input-token
// currency as ProviderCacheNetSavings and fak_vcache_saved_token_equiv:
//
//   - ProviderPromptCacheReadTokenEquiv is the provider-authored read rebate.
//   - ProviderPromptCacheWritePremiumTokenEquiv is negative when a write billed above the
//     uncached baseline, so cold-write-only sessions cannot look like wins.
//   - FakCompactionShedTokens and FakKVPrefixReusedTokens are fak-authored savings.
//
// FakVDSOAvoidedCalls is kept in the same record for mechanism attribution, but it is not
// folded into token-equivalent totals because the current witness is avoided engine calls,
// not prompt tokens.
type MechanismSavings struct {
	ProviderPromptCacheReadTokenEquiv         float64 `json:"provider_prompt_cache_read_token_equiv"`
	ProviderPromptCacheWritePremiumTokenEquiv float64 `json:"provider_prompt_cache_write_premium_token_equiv"`
	FakCompactionShedTokens                   uint64  `json:"fak_compaction_shed_tokens"`
	// FakCompactionCacheReadTokens is the OBSERVED provider cache_read at this session's
	// compaction fires — the warm witness FakTokenEquiv prices the shed on. It is the
	// warmWitness argument to cacheprice.ShedTokenEquiv: the warm PORTION of the shed,
	// min(shed, this), prices at the 0.1x read marginal (those tokens were already
	// cache-reads); the cold remainder keeps full input. It is NOT a binary flip — a
	// nonzero value discounts only the witnessed-warm slice, not the whole shed (the same
	// proportional blend the Track-2 report's compaction row uses, #2794/#2798).
	FakCompactionCacheReadTokens uint64 `json:"fak_compaction_cache_read_tokens,omitempty"`
	FakKVPrefixReusedTokens      uint64 `json:"fak_kv_prefix_reused_tokens"`
	FakVDSOAvoidedCalls          uint64 `json:"fak_vdso_avoided_calls"`
}

// MechanismSavings folds the summary's existing counters into the owner/mechanism split.
// VDSO avoided calls live on kernel.Counters, so callers that have that witness should set
// FakVDSOAvoidedCalls on the returned value before rendering.
func (s AdjudicationSummary) MechanismSavings() MechanismSavings {
	return MechanismSavings{
		ProviderPromptCacheReadTokenEquiv:         float64(s.CachedPromptTokens) * (1 - CacheReadMultiplier),
		ProviderPromptCacheWritePremiumTokenEquiv: splitCacheCreationPremiumTokenEquiv(s.CacheCreationTokens, s.CacheCreationTokensUpgraded),
		FakCompactionShedTokens:                   s.CompactionShedTokens,
		FakCompactionCacheReadTokens:              s.CompactionCacheReadTokens,
		FakKVPrefixReusedTokens:                   s.KVPrefixReusedTokens,
	}
}

// splitCacheCreationPremiumTokenEquiv is the provider write-premium token-equivalent
// for `total` cache-creation tokens, split across the 1h and 5m tiers: `upgraded` is
// the GATEWAY-ATTRIBUTED subset written while the managed-cache 1h TTL rung was
// active (#2179); the remainder prices at the 5m tier, the same conservative
// default this file has always applied to an unattributed write. upgraded > total
// is clamped so an inconsistent pair never inflates the priced total. Negative
// because a cache write always bills ABOVE the uncached baseline.
func splitCacheCreationPremiumTokenEquiv(total, upgraded uint64) float64 {
	if upgraded > total {
		upgraded = total
	}
	remainder := total - upgraded
	return float64(upgraded)*(1-CacheWrite1hMultiplier) + float64(remainder)*(1-CacheWrite5mMultiplier)
}

// ProviderTokenEquiv is the net OBSERVED provider prompt-cache effect: read rebate minus
// write premium.
func (m MechanismSavings) ProviderTokenEquiv() float64 {
	return m.ProviderPromptCacheReadTokenEquiv + m.ProviderPromptCacheWritePremiumTokenEquiv
}

// FakTokenEquiv is the WITNESSED fak-authored token-equivalent slice: prefix reuse valued at
// its full local marginal (fak realized those tokens itself), plus compaction shed valued at
// its honest PROPORTIONAL warm/cold blend. The warm portion of the shed —
// min(shed, FakCompactionCacheReadTokens), tokens the provider evidenced as cache_reads — is
// worth only the 0.1x read marginal (dropping an already-cached token saves the read cost, not
// fresh input); the remainder is worth full input. Booking all shed at 1.0x over-credited fak's
// compaction ~10x on warm traffic, and the aggregate-warm binary fix that followed then
// UNDER-credited a cold-dominant session ~10x the instant one warm cache_read appeared — the
// blend (cacheprice.ShedTokenEquiv, the same source the Track-2 report prices on) removes both
// cliffs (#2794/#2798). VDSO is deliberately excluded because its current witness is avoided
// calls, not prompt-token equivalents.
func (m MechanismSavings) FakTokenEquiv() float64 {
	return cacheprice.ShedTokenEquiv(m.FakCompactionShedTokens, m.FakCompactionCacheReadTokens) +
		float64(m.FakKVPrefixReusedTokens)
}

// TotalTokenEquiv is the sum of the token-equivalent owner slices.
func (m MechanismSavings) TotalTokenEquiv() float64 {
	return m.ProviderTokenEquiv() + m.FakTokenEquiv()
}

// HasAnyTokenActivity reports whether the token-equivalent attribution line has anything
// nonzero to say, including a negative provider write premium.
func (m MechanismSavings) HasAnyTokenActivity() bool {
	return m.ProviderPromptCacheReadTokenEquiv != 0 ||
		m.ProviderPromptCacheWritePremiumTokenEquiv != 0 ||
		m.FakCompactionShedTokens != 0 ||
		m.FakKVPrefixReusedTokens != 0
}

// vcacheProofFromCounters prices NET realized provider-cache economics over cumulative
// (uncached input, cache_read, cache_creation) token counts using THIS gateway's
// published cache multipliers — the single source the /metrics, /debug/vars, and the
// guard exit summary all read, so the three surfaces cannot drift. The multipliers are
// passed EXPLICITLY: vcachegov defaults the WRITE multipliers but NOT the read one, so
// an unset ReadMult would price cache reads at 0x (free) and overstate the saving. They
// equal vcachegov's defaults (0.1 / 1.25 / 2.0), so the result is byte-identical to
// `fak vcache observe` on the same totals. Callers with no 1h/5m attribution (the
// per-family/debug rolling snapshots) price the whole creation total at the 5m write
// tier, same as vcacheProofFromCountersWithUpgrade(..., 0).
func vcacheProofFromCounters(input, read, creation uint64) vcachegov.TelemetrySavingsProof {
	return vcacheProofFromCountersWithUpgrade(input, read, creation, 0)
}

// vcacheProofFromCountersWithUpgrade is vcacheProofFromCounters with `creationUpgraded`
// (#2179): the GATEWAY-ATTRIBUTED subset of `creation` written while the managed-cache
// 1h TTL rung was active for that turn — the provider wire itself never splits 5m vs
// 1h creation tokens, so this is fak's own witness, never provider-reported. Folded into
// Ephemeral1hInputTokens so ProveTelemetrySavings prices it at the 1h tier; the
// remainder stays unspecified (creationUpgraded=0 reproduces vcacheProofFromCounters's
// all-5m pricing byte-for-byte). creationUpgraded > creation is clamped so an
// inconsistent pair never inflates the priced total.
func vcacheProofFromCountersWithUpgrade(input, read, creation, creationUpgraded uint64) vcachegov.TelemetrySavingsProof {
	if creationUpgraded > creation {
		creationUpgraded = creation
	}
	return vcachegov.ProveTelemetrySavings(vcachegov.TelemetrySavingsInput{
		Rows: []vcachegov.TelemetryRow{{
			InputTokens:              float64(input),
			CacheReadInputTokens:     float64(read),
			CacheCreationInputTokens: float64(creation),
			Ephemeral1hInputTokens:   float64(creationUpgraded),
		}},
		ReadMult:    CacheReadMultiplier,
		Write5mMult: CacheWrite5mMultiplier,
		Write1hMult: CacheWrite1hMultiplier,
	})
}
