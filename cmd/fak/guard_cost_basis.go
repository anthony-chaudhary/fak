package main

// guard_cost_basis.go — the BASIS STAMP for `fak guard`'s dollar estimate (#5483).
//
// THE DEFECT THIS FIXES IS AN HONESTY ONE, NOT AN ARITHMETIC ONE. Guard already
// prices every served turn (session_spend.go debits the #1573 spend ceiling in
// micro-cents), but two things made that number unusable as evidence:
//
//  1. It was never SURFACED. The exit summary rendered the cache attribution purely
//     in token-equivalents (guard_format.go formatCacheAttribution); the only `$`
//     guard ever printed was the resume projection. Set no --budget-envelope and the
//     pricing still ran, and nothing said what it found.
//  2. It was never ATTRIBUTED. armServedSpendPricing returns (source, ok) and BOTH
//     guard call sites — guard_child.go's resolveGuardUpstream and
//     guard_upstream_posture.go — discard it, so the card that produced the number
//     was resolved once per process and thrown away.
//
// And the tree holds FOUR Opus-class rate cards claiming THREE different bases
// (guardOpusClassRateCards below). Any one of them can be defended; they cannot all
// be the published list price. So a bare `$1.23` from fak is unfalsifiable: the
// reader cannot tell which card it came from, and two honest fak surfaces can differ
// 3x on the same session.
//
// THE FIX IS A LABEL, NOT A NUMBER. This file deliberately does NOT reconcile the
// four cards, does not pick a winner, and does not blend them — which price is
// correct is a published-rate question this repo cannot settle, and inventing one
// would replace a visible ambiguity with an invisible error. It stamps the estimate
// with the card that produced it, the provider/context it was resolved from, the
// rate modifiers that WERE applied, the ones that were NOT, and — when the in-tree
// cards disagree — says so on the face of the output. An estimate labelled with its
// source is worth more than a hidden exact number.
//
// EVERY FIGURE HERE IS ESTIMATED, NEVER BILLED. The token axes are OBSERVED
// (provider-relayed via the AdjudicationSummary); the dollars are a projection over
// them under a stated card. fak never sees an invoice.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

// guardCostBasisStamp is the machine- and human-readable provenance of one dollar
// figure: which card, resolved from what, with which modifiers, and whether it is a
// price at all. Priced=false is DOLLAR-BLIND — an honest "no card matched", which is
// reported rather than silently defaulted, because a substituted default that reads
// like a measurement is the exact failure #5483 names.
type guardCostBasisStamp struct {
	// Priced is false when no rate card resolved: the session is dollar-blind and
	// no dollar figure may be printed for it.
	Priced bool
	// Armed is false when no host path ever resolved pricing in this process — an
	// UNMEASURED state, reported apart from a measured dollar-blind.
	Armed bool
	// Card is the source label of the rate card in force, e.g.
	// "default:anthropic/claude-opus-4-8" or the FAK_SPEND_* env override label.
	Card string
	// Provider / Context are the raw inputs the card was resolved FROM. Context is
	// the un-normalized model id or agent name as guard saw it, so a reader can tell
	// a requested Opus row from one defaulted into via the agent name.
	Provider string
	Context  string
	// InputPerMTokUSD / OutputPerMTokUSD are the base rates the card supplies.
	InputPerMTokUSD  float64
	OutputPerMTokUSD float64
}

// guardCostEnvOverrideCard is the card label the meter reports when the operator
// supplied the price themselves. Named apart from the built-in table's label so an
// operator-set number is never mistaken for a fak-authored default.
const guardCostEnvOverrideCard = spendEnvPricingSource

// guardServedCostBasis reads the basis the served-spend meter is armed with. This is
// the consumer of servedSpendPricingBasis that makes the previously-discarded
// (source, ok) return observable at the one place an operator looks.
func guardServedCostBasis() guardCostBasisStamp {
	p, source, provider, context, armed, ok := servedSpendPricingBasis()
	return guardCostBasisStamp{
		Priced:           ok,
		Armed:            armed,
		Card:             source,
		Provider:         provider,
		Context:          context,
		InputPerMTokUSD:  p.InputPerMTokUSD,
		OutputPerMTokUSD: p.OutputPerMTokUSD,
	}
}

// guardSessionCostUSD prices a session's OBSERVED token axes under the stamped card.
// It delegates to gateway.CachePricing.CostUSD — the ONE canonical per-turn cost
// model the gateway, the Track-2 ledger and the spend ceiling all price on — rather
// than re-deriving the multiplier arithmetic, so this surface cannot drift from the
// meter that debits the budget.
//
// The cache-creation axis is split across the two write tiers exactly the way
// MechanismSavings splits it: CacheCreationTokensUpgraded is the GATEWAY-ATTRIBUTED
// subset written under the managed-cache 1h rung (the provider wire never splits
// them), priced at 2.0x, and the remainder at the conservative 5m 1.25x. The model is
// linear, so two CostUSD calls over disjoint token sets sum exactly to one call over
// the union. An inconsistent upgraded > total pair is clamped so it can never inflate.
func guardSessionCostUSD(sum gateway.AdjudicationSummary, p gateway.CachePricing) float64 {
	upgraded := sum.CacheCreationTokensUpgraded
	if upgraded > sum.CacheCreationTokens {
		upgraded = sum.CacheCreationTokens
	}
	cost := p.CostUSD(gateway.CacheUsage{
		InputTokens:         int(sum.InputTokens),
		CacheReadTokens:     int(sum.CachedPromptTokens),
		CacheCreationTokens: int(sum.CacheCreationTokens - upgraded),
		OutputTokens:        int(sum.OutputTokens),
		WriteTTL:            gateway.CacheTTL5m,
	})
	cost += p.CostUSD(gateway.CacheUsage{
		CacheCreationTokens: int(upgraded),
		WriteTTL:            gateway.CacheTTL1h,
	})
	return cost
}

// guardCostSessionHasTokens reports whether the session has any billable token axis
// to price. A session that served nothing stays silent rather than printing $0.0000 —
// the same clean-run silence every other exit-summary formatter keeps. This is NOT
// the --budget-envelope gate the issue removes: a session that served turns prints
// its estimate with no ceiling configured, and prints DOLLAR-BLIND when unpriced.
func guardCostSessionHasTokens(sum gateway.AdjudicationSummary) bool {
	return sum.InputTokens+sum.OutputTokens+sum.CachedPromptTokens+sum.CacheCreationTokens > 0
}

// ---------------------------------------------------------------------------
// The in-tree rate-card inventory.
// ---------------------------------------------------------------------------

// guardRateCard is one in-tree Opus-class base price: where it lives, what it
// claims to be, and — read from the LIVE symbol, never a transcribed copy — what it
// says. Reading the live value is the point: a transcribed inventory rots the moment
// a card moves, and a rotted provenance table is worse than none.
type guardRateCard struct {
	// Site is the file + symbol the number lives at, so a reader can go check it.
	Site string
	// InPerMTokUSD / OutPerMTokUSD are the card's Opus-class base rates.
	InPerMTokUSD  float64
	OutPerMTokUSD float64
	// Claims is what that site says its own number IS, quoted from its own doc
	// comment — the material fact, because each site documents itself as
	// authoritative and at most one of them can be the published list price.
	Claims string
}

// Base returns the card's rate pair in the "$in/$out" shorthand the divergence line
// reports.
func (c guardRateCard) Base() string {
	return fmt.Sprintf("$%g/$%g", c.InPerMTokUSD, c.OutPerMTokUSD)
}

// guardOpusClassRateCards enumerates every in-tree rate card that can produce an
// OPERATOR-FACING Opus-class dollar figure. It is the evidence behind the
// "cards disagree" warning: four sites, three different bases, each documenting
// itself as authoritative.
//
// SCOPE OF THE INVENTORY. It lists cards that price a real session for a human, not
// every $/MTok constant in the tree. Deliberately excluded, because none of them can
// print a number an operator would read as their session's cost:
// internal/turnbench's CostModel default and internal/bench's "representative"
// 3/15 (both self-declared illustrative bench constants), cmd/radixbench's
// dollarsPerMTok sweep proxy, cmd/fanbench's operator-supplied --dollars-in/out, and
// internal/modelscore's catalog (which carries its own per-row provenance already).
//
// This function does NOT rank the cards and does NOT pick a winner. Which base is
// the published list price is a data question outside this repo; the deliverable is
// that the reader can SEE the disagreement instead of inheriting one arm of it.
func guardOpusClassRateCards() []guardRateCard {
	return []guardRateCard{
		{
			Site:          "internal/gateway/cache_pricing.go ClaudeOpus48{Input,Output}PerMTokUSD",
			InPerMTokUSD:  gateway.ClaudeOpus48InputPerMTokUSD,
			OutPerMTokUSD: gateway.ClaudeOpus48OutputPerMTokUSD,
			Claims:        "the default guarded-Claude base price; the LIVE card the guard spend meter resolves",
		},
		{
			Site:          `internal/sessionaudit/sessionaudit.go Pricing["opus"]`,
			InPerMTokUSD:  sessionaudit.Pricing["opus"].Input,
			OutPerMTokUSD: sessionaudit.Pricing["opus"].Output,
			Claims:        "\"Anthropic's published rates\"; the offline transcript-audit card",
		},
		{
			Site:          "internal/modelroute/cost.go FrontierAnchor",
			InPerMTokUSD:  modelroute.FrontierAnchor.In,
			OutPerMTokUSD: modelroute.FrontierAnchor.Out,
			Claims:        "\"the repo's published $3 in / $15 out convention\"; a routing cost LENS, self-declared never a bill",
		},
		{
			Site:          "cmd/fak/guard_resume.go guardResumeFallback{Input,Output}PerMTokUSD",
			InPerMTokUSD:  guardResumeFallbackInputPerMTokUSD,
			OutPerMTokUSD: guardResumeFallbackOutputPerMTokUSD,
			Claims:        "the resume projection's SUBSTITUTED default when the operator supplies no price",
		},
	}
}

// guardRateCardBases returns the DISTINCT Opus-class bases the inventory holds, in a
// stable ascending order. len > 1 is the disagreement the stamp reports.
func guardRateCardBases() []string {
	seen := map[string]float64{}
	for _, c := range guardOpusClassRateCards() {
		seen[c.Base()] = c.InPerMTokUSD
	}
	out := make([]string, 0, len(seen))
	for b := range seen {
		out = append(out, b)
	}
	sort.Slice(out, func(i, j int) bool { return seen[out[i]] < seen[out[j]] })
	return out
}

// ---------------------------------------------------------------------------
// Rendering.
// ---------------------------------------------------------------------------

// guardCostModifiersApplied names the rate modifiers this estimate DID apply: the
// three published prompt-cache multipliers, which are the only rate-changing settings
// fak models today. They are stated numerically so the reader can reproduce the
// figure by hand from the token axes on the row above it.
func guardCostModifiersApplied() string {
	return fmt.Sprintf("cache read %.2fx, cache write %.2fx (5m) / %.2fx (1h) — applied on top of the base input rate",
		gateway.CacheReadMultiplier, gateway.CacheWrite5mMultiplier, gateway.CacheWrite1hMultiplier)
}

// guardCostModifiersNotApplied names the rate-changing provider settings this estimate
// does NOT model, so their absence is visible instead of silent. Naming an unmodeled
// modifier is the honest move: fak has no published multiplier for any of them, and a
// guessed one would be a fabricated number wearing a measurement's clothes. A detected
// but unpriced long-context variant is called out separately by the caller, because
// there the information IS in the process and only the rate is missing.
func guardCostModifiersNotApplied() string {
	return "premium/\"fast\" output mode, long-context ([1m]) tier, batch/off-peak tier, seat-vs-metered billing — no published multiplier in-tree, so none is guessed"
}

// formatGuardSessionCost renders the exit-summary cost section for a session, given
// an already-resolved basis. Pure over its inputs (no globals, no I/O) so the whole
// grammar — priced, dollar-blind, unarmed, long-context — is unit-tested directly.
// Returns "" when the session has no billable token axis at all.
func formatGuardSessionCost(sum gateway.AdjudicationSummary, basis guardCostBasisStamp) string {
	if !guardCostSessionHasTokens(sum) {
		return ""
	}
	var b strings.Builder
	b.WriteString(guardSection("estimated cost"))

	axes := fmt.Sprintf("%s uncached in + %s cache read + %s cache write + %s out (OBSERVED, provider-relayed)",
		gateway.HumanTokenEquiv(float64(sum.InputTokens)),
		gateway.HumanTokenEquiv(float64(sum.CachedPromptTokens)),
		gateway.HumanTokenEquiv(float64(sum.CacheCreationTokens)),
		gateway.HumanTokenEquiv(float64(sum.OutputTokens)))

	if !basis.Priced {
		reason := "no rate card matched this provider/context pair"
		if !basis.Armed {
			reason = "pricing was never armed on this host path"
		}
		b.WriteString(guardRow("session cost", "DOLLAR-BLIND — "+reason+"; nothing priced, nothing guessed"))
		b.WriteString(guardRow("resolved from", guardCostBasisResolvedFrom(basis)))
		b.WriteString(guardRow("token axes", axes))
		b.WriteString(guardNote("to price it, set " + spendInputPriceEnv + " / " + spendOutputPriceEnv + " (a self-hosted run can hand-set a $/GPU-hour-derived rate here; it is an ESTIMATE, never a bill)"))
		return b.String()
	}

	usd := guardSessionCostUSD(sum, gateway.CachePricing{
		InputPerMTokUSD:  basis.InputPerMTokUSD,
		OutputPerMTokUSD: basis.OutputPerMTokUSD,
	})
	b.WriteString(guardRow("session cost", fmt.Sprintf("~$%.4f ESTIMATED (not a bill — fak never sees an invoice)", usd)))
	b.WriteString(guardRow("basis card", fmt.Sprintf("%s  [$%g/MTok in, $%g/MTok out]",
		orDash(basis.Card), basis.InputPerMTokUSD, basis.OutputPerMTokUSD)))
	if basis.Card == guardCostEnvOverrideCard {
		// An operator-set rate is NOT a fak default and must not read as one: it
		// outranks the built-in table inside resolveSpendPricing, so the in-tree
		// divergence below is context for the reader, not the source of this figure.
		b.WriteString(guardNote("this rate is OPERATOR-SUPPLIED via the environment and outranks every in-tree card listed below"))
	}
	b.WriteString(guardRow("resolved from", guardCostBasisResolvedFrom(basis)))
	b.WriteString(guardRow("modifiers applied", guardCostModifiersApplied()))
	b.WriteString(guardRow("modifiers NOT applied", guardCostModifiersNotApplied()))
	b.WriteString(guardRow("token axes", axes))
	if guardClaudeOneMillionModel(basis.Context) {
		b.WriteString(guardRow("⚠ long-context", "this session declared the [1m] long-context variant, which the card prices at the STANDARD-window rate — providers charge a premium above it, so this estimate is a LOWER BOUND"))
	}
	if line := formatGuardRateCardDivergence(); line != "" {
		b.WriteString(line)
	}
	return b.String()
}

// guardCostBasisResolvedFrom renders the provider/context pair the card was chosen
// from. An empty context is reported as such rather than blank: "resolved from
// nothing" is a real and diagnosable state (the lazy env-only arming path).
func guardCostBasisResolvedFrom(basis guardCostBasisStamp) string {
	provider, context := strings.TrimSpace(basis.Provider), strings.TrimSpace(basis.Context)
	if provider == "" && context == "" {
		return "provider/context unset — pricing resolved with no upstream identity"
	}
	return fmt.Sprintf("provider=%s context=%s (raw, as guard saw it at boot)", orDash(provider), orDash(context))
}

// formatGuardRateCardDivergence warns, on the face of the estimate, that the tree
// holds more than one Opus-class base — so the reader knows the figure above is one
// arm of a live disagreement rather than a settled price. Silent when the cards
// agree, which is what makes it a real signal: the day someone reconciles them, this
// line disappears on its own.
func formatGuardRateCardDivergence() string {
	bases := guardRateCardBases()
	if len(bases) < 2 {
		return ""
	}
	cards := guardOpusClassRateCards()
	var b strings.Builder
	b.WriteString(guardRow("⚠ cards disagree", fmt.Sprintf("%d in-tree Opus-class rate card(s) claim %d different bases (%s /MTok) — the figure above uses the one named on 'basis card'",
		len(cards), len(bases), strings.Join(bases, ", "))))
	for _, c := range cards {
		b.WriteString(guardRow("  "+c.Base(), c.Site+" — "+c.Claims))
	}
	b.WriteString(guardNote("fak does not reconcile these: which base is the published list price is a data question outside this repo, and a blended number would hide the disagreement instead of showing it (#5483)"))
	return b.String()
}
