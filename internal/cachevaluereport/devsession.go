package cachevaluereport

// Track 3 — the real, un-proxied dev-session lens. Track 1/Track 2/FleetBenefit above are all
// sourced from fak's OWN gateway/guard runtime ledgers, so a session that never ran through
// `fak guard`/`fak serve` (a plain `claude` session — the common case for actual dev work)
// contributes nothing to them, even though its local transcript already carries real provider
// cache_read/cache_creation counters (internal/sessionaudit already parses these for the
// session-pressure gate and vcache calibration corpus, just never for THIS report). This track
// prices those same real sessions with the identical Track-2 provider_prompt_cache economics
// (NewSavingsRows), so fak's cache-value story is grounded in the dev sessions that actually
// produce it, not only the minority that happen to be gateway-wrapped.
//
// PROVENANCE FENCE: a session run through fak guard/serve ALSO writes a local Claude Code
// transcript, so this track's rows MAY OVERLAP FleetBenefit's usage rows for the same session.
// Never sum DevSessionBenefit into FleetBenefit or Track 2 — it is a separate, wider lens
// ("how much real cache value did our dev sessions produce, gateway-wrapped or not"), not an
// additional saving on top of what the gateway already counted.

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

const devSessionBenefitProvenance = "OBSERVED provider prompt-cache token axes from real, un-proxied Claude Code session transcripts (internal/sessionaudit over ~/.claude*/projects), priced per-model at sessionaudit's base $/MTok tiers; a session also run through fak guard/serve has a transcript here too, so this MAY OVERLAP the Fleet aggregate above — never sum the two"

// DevSessionBenefitReport is the cumulative Track-3 fold over real dev-session transcripts:
// how much provider prompt-cache value did our own actual Claude Code sessions realize,
// independent of whether any of them ran through fak's gateway. PricedSessions counts only
// sessions with at least one billable (Pricing-known) model; a session using only an unpriced
// model contributes to Sessions but not to any token/dollar axis.
type DevSessionBenefitReport struct {
	Sessions       int            `json:"sessions"`
	PricedSessions int            `json:"priced_sessions"`
	ModelTiers     map[string]int `json:"model_tiers,omitempty"`
	UnpricedModels map[string]int `json:"unpriced_models,omitempty"`

	InputTokens         uint64 `json:"input_tokens"`
	OutputTokens        uint64 `json:"output_tokens"`
	CacheReadTokens     uint64 `json:"cache_read_tokens"`
	CacheCreationTokens uint64 `json:"cache_creation_tokens"`

	SavedTokenEquiv float64 `json:"saved_token_equiv"`

	ObservedActualSpendUSD      float64  `json:"observed_actual_spend_usd"`
	ObservedAPICostAvoidedUSD   float64  `json:"observed_api_cost_avoided_usd"`
	ObservedCounterfactualUSD   float64  `json:"observed_counterfactual_usd"`
	ObservedAPICostReductionPct *float64 `json:"observed_api_cost_reduction_pct,omitempty"`

	Provenance string `json:"provenance"`
	Finding    string `json:"finding"`
}

// FoldDevSessionBenefit prices real dev-session transcripts (already discovered + analyzed by
// the caller via sessionaudit.Discover/Analyze) with the same provider_prompt_cache economics
// Track 2 uses. It is PURE — sessions in, report out — so discovery/analysis I/O stays in the
// CLI layer and this stays unit-testable on fixture sessions.
func FoldDevSessionBenefit(sessions []sessionaudit.Session, now time.Time) DevSessionBenefitReport {
	rep := DevSessionBenefitReport{
		ModelTiers:     map[string]int{},
		UnpricedModels: map[string]int{},
		Provenance:     devSessionBenefitProvenance,
	}
	for _, s := range sessions {
		if s.Error != "" {
			continue
		}
		rep.Sessions++
		priced := false
		for model, c := range s.PerModel {
			tier := sessionaudit.ModelTier(model)
			rates, ok := sessionaudit.PriceFor(model)
			if !ok {
				rep.UnpricedModels[tier]++
				continue
			}
			priced = true
			rep.ModelTiers[tier]++
			rows := NewSavingsRows(SavingsObservation{
				SessionType:         "dev_session",
				Provider:            "anthropic",
				Context:             "claude",
				InputTokens:         nonNegUint64(c.Input),
				CacheReadTokens:     nonNegUint64(c.CacheRead),
				CacheCreationTokens: nonNegUint64(c.CacheCreate),
				OutputTokens:        nonNegUint64(c.Output),
				Pricing: SavingsPricing{
					InputPerMTokUSD:  rates.Input,
					OutputPerMTokUSD: rates.Output,
					Source:           "sessionaudit:" + tier,
				},
			}, now)
			for _, row := range rows {
				rep.InputTokens += row.InputTokens
				rep.OutputTokens += row.OutputTokens
				rep.CacheReadTokens += row.CacheReadTokens
				rep.CacheCreationTokens += row.CacheCreationTokens
				rep.SavedTokenEquiv += row.SavedTokenEquiv
				rep.ObservedActualSpendUSD += row.SpendUSD
				rep.ObservedAPICostAvoidedUSD += row.RebateUSD - row.WritePremiumUSD
			}
		}
		if priced {
			rep.PricedSessions++
		}
	}
	rep.ObservedCounterfactualUSD = rep.ObservedActualSpendUSD + rep.ObservedAPICostAvoidedUSD
	if rep.ObservedCounterfactualUSD != 0 {
		pct := 100 * rep.ObservedAPICostAvoidedUSD / rep.ObservedCounterfactualUSD
		rep.ObservedAPICostReductionPct = &pct
	}
	rep.fillFinding()
	return rep
}

func nonNegUint64(v int64) uint64 {
	if v <= 0 {
		return 0
	}
	return uint64(v)
}

func (r *DevSessionBenefitReport) fillFinding() {
	if r.Sessions == 0 {
		r.Finding = "no real dev-session transcripts discovered"
		return
	}
	parts := []string{fmt.Sprintf("%d session(s) discovered (%d priced)", r.Sessions, r.PricedSessions)}
	if r.ObservedCounterfactualUSD != 0 {
		parts = append(parts, fmt.Sprintf("$%.4f API cost avoided", r.ObservedAPICostAvoidedUSD))
	}
	r.Finding = strings.Join(parts, "; ")
}

// RenderDevSessionBenefit renders the Track-3 dev-session lens as a compact section, matching
// the fenced style RenderFleetBenefit uses.
func RenderDevSessionBenefit(r DevSessionBenefitReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "\nDev-session lens (Track 3, real un-proxied Claude Code transcripts)\n")
	if r.Finding != "" {
		fmt.Fprintf(&b, "  %s\n", r.Finding)
	}
	if r.Sessions == 0 {
		fmt.Fprintf(&b, "  provenance: %s\n", r.Provenance)
		return b.String()
	}
	if len(r.ModelTiers) > 0 {
		fmt.Fprintf(&b, "  model tiers: %s\n", renderSortedCountMap(r.ModelTiers))
	}
	if len(r.UnpricedModels) > 0 {
		fmt.Fprintf(&b, "  unpriced models (dollar-blind, excluded): %s\n", renderSortedCountMap(r.UnpricedModels))
	}
	fmt.Fprintf(&b, "  tokens: input=%d output=%d cache_read=%d cache_write=%d\n",
		r.InputTokens, r.OutputTokens, r.CacheReadTokens, r.CacheCreationTokens)
	fmt.Fprintf(&b, "  saved token-equiv: %.0f\n", r.SavedTokenEquiv)
	if r.ObservedCounterfactualUSD != 0 || r.ObservedActualSpendUSD != 0 || r.ObservedAPICostAvoidedUSD != 0 {
		reduction := "-"
		if r.ObservedAPICostReductionPct != nil {
			reduction = fmt.Sprintf("%.2f%%", *r.ObservedAPICostReductionPct)
		}
		fmt.Fprintf(&b, "  API cost: observed_spend=$%.4f counterfactual=$%.4f avoided=$%.4f reduction=%s\n",
			r.ObservedActualSpendUSD, r.ObservedCounterfactualUSD, r.ObservedAPICostAvoidedUSD, reduction)
	}
	fmt.Fprintf(&b, "  provenance: %s\n", r.Provenance)
	return b.String()
}

func renderSortedCountMap(m map[string]int) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, m[k]))
	}
	return strings.Join(parts, " ")
}
