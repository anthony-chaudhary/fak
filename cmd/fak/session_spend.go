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
//
// provider/context are retained alongside the resolved price because they are the
// INPUTS the card was chosen from, and a dollar figure whose card is unnamed is
// unfalsifiable (#5483): the exit-summary basis stamp reports the raw context the
// meter was armed with, not just the card it landed on, so a reader can tell an
// Opus-row price that was asked for from one that was defaulted into.
var servedSpend struct {
	mu       sync.RWMutex
	armed    bool
	ok       bool
	p        gateway.CachePricing
	source   string
	provider string
	context  string
}

// armServedSpendPricing resolves and installs the served-session spend pricing
// for this process: explicit env override first, then the built-in default table
// for the provider/context pair, else dollar-blind. Returns the source label and
// whether pricing is live. Safe to call more than once; the latest resolution
// wins (same inputs resolve identically, so repeated boots are idempotent).
func armServedSpendPricing(provider, context string) (string, bool) {
	p, source, ok := resolveSpendPricing(provider, context)
	if cal := loadVCacheRuntimeCalibration(provider, context); cal != nil {
		p = cal.ApplyCachePricing(p)
		if cal.ReadMultMeasured {
			source += "+vcache-calibrated-read"
		}
	}
	servedSpend.mu.Lock()
	defer servedSpend.mu.Unlock()
	servedSpend.armed = true
	servedSpend.p, servedSpend.source, servedSpend.ok = p, source, ok
	servedSpend.provider, servedSpend.context = provider, context
	return source, ok
}

// servedSpendPricingBasis reports the pricing the meter is currently armed with —
// the read half of armServedSpendPricing, whose (source, ok) return both guard call
// sites discard. It is what makes an operator-facing dollar figure falsifiable: the
// card that produced it, the provider/context pair it was resolved from, and whether
// a price was resolved at all. armed=false means no host path ever armed the meter,
// which is reported apart from "armed and dollar-blind" — the first is unmeasured,
// the second is a measured absence of price.
func servedSpendPricingBasis() (p gateway.CachePricing, source, provider, context string, armed, ok bool) {
	servedSpend.mu.RLock()
	defer servedSpend.mu.RUnlock()
	return servedSpend.p, servedSpend.source, servedSpend.provider, servedSpend.context, servedSpend.armed, servedSpend.ok
}

// guardSpendPricingContext picks the served-spend pricing context for a guard
// session. resolveGuardUpstream first arms the meter with the agent name
// (command[0]), which prices every Claude tier as the Opus 4.8 default. When the
// upstream model is statically known — a `fak guard --model` override, or a
// --model/-m in the wrapped child argv — AND that model has a known price, prefer
// the real model id so a non-default tier bills at its own rate (e.g.
// claude-fable-5 is 2x Opus, so an all-`claude` ledger otherwise UNDER-books a
// fable session's spend and hides its larger cache savings). An unknown model
// falls back to the agent name rather than going dollar-blind, so this only ever
// CORRECTS a known misprice and never regresses today's default. An env override
// (FAK_SPEND_*_PER_MTOK) wins inside armServedSpendPricing regardless of the
// context returned here, so the context choice never overrides an operator price.
func guardSpendPricingContext(provider, guardModel string, command []string) string {
	agentName := ""
	if len(command) > 0 {
		agentName = command[0]
	}
	if m := guardStaticClaudeModel(guardModel, command); m != "" {
		if _, _, ok := gateway.DefaultCachePricing(provider, m); ok {
			return m
		}
	}
	return agentName
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
	return priceFromEnv(name, true)
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
