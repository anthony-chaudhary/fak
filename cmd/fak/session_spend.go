package main

// session_spend.go — the host-side TURN PRICER for the session spend ceiling
// (#1573's spend axis, made live). internal/session owns the spend BUDGET
// (Budget.SpendMicroCentsLeft, debited by DebitUsage, drained with
// BUDGET_SPEND_EXHAUSTED) but is deliberately price-blind: the per-MTok price
// table must live in exactly one place, and that place is the host, which knows
// which provider a served session talks to. This file resolves that pricing once
// per process and prices each served turn's provider-reported usage into the
// micro-cent (1e-8 USD) cost debitSession hands the table.
//
// RESOLUTION ORDER (mirrors cachevalueSavingsPricing):
//  1. FAK_SPEND_INPUT_PER_MTOK_USD / FAK_SPEND_OUTPUT_PER_MTOK_USD — the explicit
//     operator override, honored on every host path (guard, serve, anything that
//     debits served sessions).
//  2. gateway.DefaultCachePricing(provider, context) — armed at guard boot
//     (resolveGuardUpstream), so the flagship `fak guard -- claude` path prices
//     without configuration.
//  3. dollar-blind: no pricing means NO debit — a configured spend budget is left
//     honestly untouched rather than debited a guessed cost. A `fak serve` local
//     model stays dollar-blind by default because a local turn costs no provider
//     dollars.
//
// HONEST SCOPE. Token axes are priced under the Anthropic usage-block convention
// (PromptTokens = the uncached remainder; cache read at 0.1x input; cache
// creation at the 5-minute write tier, the same unsplit-creation convention every
// other fak $ surface applies — see gateway/cache_pricing.go). On an env-priced
// OpenAI-style wire, whose prompt_tokens already fold in cached tokens, the read
// rebate is not peeled off, so the meter can only OVER-charge there — the
// conservative direction for a ceiling, never a silent overrun.

import (
	"math"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const (
	spendInputPriceEnv    = "FAK_SPEND_INPUT_PER_MTOK_USD"
	spendOutputPriceEnv   = "FAK_SPEND_OUTPUT_PER_MTOK_USD"
	spendEnvPricingSource = "env:FAK_SPEND_INPUT_PER_MTOK_USD/FAK_SPEND_OUTPUT_PER_MTOK_USD"

	// usdPerMTokToMicroCentsPerTok converts a per-MTok USD price into micro-cents
	// per token: (USD/1e6 tok) * (1e8 micro-cents/USD) = 100 micro-cents/tok per
	// USD-per-MTok. At this grain every published Anthropic price, including the
	// 0.1x cache-read multiplier, is an exact integer per token.
	usdPerMTokToMicroCentsPerTok = 100.0
)

// servedSpend is the process-wide pricing state the served-session spend meter
// reads. ok=false is dollar-blind: servedTurnSpendMicroCents returns 0 and a
// configured spend budget is never debited a guessed cost.
var servedSpend struct {
	mu     sync.RWMutex
	armed  bool
	ok     bool
	p      gateway.CachePricing
	source string
}

// armServedSpendPricing resolves and installs the served-session spend pricing
// for this process: explicit env override first, then the built-in default table
// for the provider/context pair, else dollar-blind. Returns the source label and
// whether pricing is live. Safe to call more than once; the latest resolution
// wins (same inputs resolve identically, so repeated boots are idempotent).
func armServedSpendPricing(provider, context string) (string, bool) {
	p, source, ok := resolveSpendPricing(provider, context)
	servedSpend.mu.Lock()
	defer servedSpend.mu.Unlock()
	servedSpend.armed = true
	servedSpend.p, servedSpend.source, servedSpend.ok = p, source, ok
	return source, ok
}

func resolveSpendPricing(provider, context string) (gateway.CachePricing, string, bool) {
	input, inputSet := spendPriceFromEnv(spendInputPriceEnv)
	output, outputSet := spendPriceFromEnv(spendOutputPriceEnv)
	if inputSet || outputSet {
		blind := input == 0 && output == 0
		return gateway.CachePricing{InputPerMTokUSD: input, OutputPerMTokUSD: output},
			spendEnvPricingSource, !blind
	}
	if p, source, ok := gateway.DefaultCachePricing(provider, context); ok {
		return p, source, true
	}
	return gateway.CachePricing{}, "none", false
}

func spendPriceFromEnv(name string) (float64, bool) {
	raw, ok := os.LookupEnv(name)
	raw = strings.TrimSpace(raw)
	if !ok || raw == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil || v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, true
	}
	return v, true
}

// servedTurnSpendMicroCents prices one served turn's provider-reported usage for
// the session spend debit. A host path that never armed pricing gets a one-time
// lazy env-only resolution, so FAK_SPEND_*_PER_MTOK_USD works on every path
// (serve included) with no per-command wiring. Dollar-blind pricing returns 0.
func servedTurnSpendMicroCents(u gateway.SessionUsage) int64 {
	servedSpend.mu.RLock()
	armed, ok, p := servedSpend.armed, servedSpend.ok, servedSpend.p
	servedSpend.mu.RUnlock()
	if !armed {
		_, ok = armServedSpendPricing("", "")
		servedSpend.mu.RLock()
		p = servedSpend.p
		servedSpend.mu.RUnlock()
	}
	if !ok {
		return 0
	}
	return spendTurnMicroCents(p, u)
}

// spendTurnMicroCents is the pure per-turn cost model: uncached input at 1.0x,
// cache read at 0.1x, cache creation at the 5-minute write tier (the unsplit-
// creation convention), output at the output price — the same shape as
// gateway.CachePricing.CostUSD, in integer micro-cents. Rounded UP: for a spend
// CEILING the conservative error direction is charging a fraction more, never a
// fraction less.
func spendTurnMicroCents(p gateway.CachePricing, u gateway.SessionUsage) int64 {
	inPerTok := p.InputPerMTokUSD * usdPerMTokToMicroCentsPerTok
	outPerTok := p.OutputPerMTokUSD * usdPerMTokToMicroCentsPerTok
	cost := float64(u.PromptTokens) * inPerTok
	cost += float64(u.CacheReadInputTokens) * inPerTok * gateway.CacheReadMultiplier
	cost += float64(u.CacheCreationInputTokens) * inPerTok * gateway.CacheWrite5mMultiplier
	cost += float64(u.CompletionTokens) * outPerTok
	if cost <= 0 || math.IsNaN(cost) || math.IsInf(cost, 0) {
		return 0
	}
	return int64(math.Ceil(cost))
}
