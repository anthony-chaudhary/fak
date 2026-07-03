package cachevaluereport

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// FleetBenefitOptions configures the cumulative fleet-benefit fold. ContextBudgetTokens
// is optional: when set, the fold normalizes witnessed context-extension tokens into
// "how much of one session window did fak add?" without baking in a model-specific limit.
type FleetBenefitOptions struct {
	ContextBudgetTokens uint64
}

// FleetBenefitReport is the all-time (or caller-filtered) fleet roll-up over the durable
// guard/serve usage ledgers. It answers the operator question "what did fak's cache work buy
// cumulatively?" without blending provenance:
//
//   - usage/session totals come from gatewayusageledger rows (WITNESSED counters);
//   - provider cost reduction comes from Track-2 savings rows (OBSERVED provider counters
//     priced from the row's configured base rates);
//   - session extension comes only from fak-authored compaction shed tokens, never from a
//     provider prompt-cache rebate that reduces spend but does not enlarge the context window.
type FleetBenefitReport struct {
	UsageRows    int            `json:"usage_rows"`
	ExitSessions int            `json:"exit_sessions"`
	SessionTypes map[string]int `json:"session_types,omitempty"`
	UptimeSecs   float64        `json:"uptime_seconds"`

	KernelDecisions uint64 `json:"kernel_decisions"`
	Allowed         uint64 `json:"allowed"`
	Denied          uint64 `json:"denied"`
	Quarantined     uint64 `json:"quarantined"`
	Transformed     uint64 `json:"transformed"`

	InputTokens         uint64 `json:"input_tokens"`
	OutputTokens        uint64 `json:"output_tokens"`
	CacheReadTokens     uint64 `json:"cache_read_tokens"`
	CacheCreationTokens uint64 `json:"cache_creation_tokens"`
	CachedTurns         uint64 `json:"cached_turns"`

	CompactionFired        uint64 `json:"compaction_fired"`
	CompactionBailed       uint64 `json:"compaction_bailed"`
	CompactionShedTokens   uint64 `json:"compaction_shed_tokens"`
	ContextExtensionTokens uint64 `json:"context_extension_tokens"`
	ContextBudgetTokens    uint64 `json:"context_budget_tokens,omitempty"`

	ContextExtensionPct     *float64 `json:"context_extension_pct,omitempty"`
	EquivalentContextWindow *float64 `json:"equivalent_context_windows,omitempty"`

	Track1Sessions              int      `json:"track1_sessions"`
	ProviderPromptCacheTokenEq  float64  `json:"provider_prompt_cache_token_equiv"`
	FakKVPrefixReusedTokens     uint64   `json:"fak_kv_prefix_reused_tokens"`
	FakCompactionTokenEq        float64  `json:"fak_compaction_token_equiv"`
	FakAuthoredTokenEq          float64  `json:"fak_authored_token_equiv"`
	TotalSavedTokenEq           float64  `json:"total_saved_token_equiv"`
	FakSharePct                 *float64 `json:"fak_share_pct,omitempty"`
	ObservedActualSpendUSD      float64  `json:"observed_actual_spend_usd"`
	ObservedAPICostAvoidedUSD   float64  `json:"observed_api_cost_avoided_usd"`
	ObservedCounterfactualUSD   float64  `json:"observed_counterfactual_usd"`
	ObservedAPICostReductionPct *float64 `json:"observed_api_cost_reduction_pct,omitempty"`
	DollarBlindRows             int      `json:"dollar_blind_rows,omitempty"`

	Provenance string `json:"provenance"`
	Finding    string `json:"finding"`
}

const fleetBenefitProvenance = "cumulative over recorded rows: gateway-usage counters are WITNESSED; provider prompt-cache dollars are OBSERVED/provider-relayed projections; context extension counts only WITNESSED fak compaction-shed tokens"

// FoldFleetBenefit folds the Track-1 kernel ledger, Track-2 savings ledger, and
// gateway-usage ledger into one cumulative report. The caller is responsible for any
// time-window filtering before calling this function.
func FoldFleetBenefit(track1 []cachevalueledger.Row, track2 []SavingsRow, usage []gatewayusageledger.Row, opts FleetBenefitOptions) FleetBenefitReport {
	rep := FleetBenefitReport{
		SessionTypes:        map[string]int{},
		ContextBudgetTokens: opts.ContextBudgetTokens,
		Provenance:          fleetBenefitProvenance,
	}

	for _, row := range usage {
		if row.Schema == "" {
			continue
		}
		rep.UsageRows++
		if row.Kind == "exit" {
			rep.ExitSessions++
		}
		st := strings.TrimSpace(row.SessionType)
		if st == "" {
			st = "unknown"
		}
		rep.SessionTypes[st]++
		rep.UptimeSecs += row.UptimeSecs
		c := row.Counters
		rep.KernelDecisions += c.Total
		rep.Allowed += c.Allowed
		rep.Denied += c.Denied
		rep.Quarantined += c.Quarantined
		rep.Transformed += c.Transformed
		rep.InputTokens += c.InputTokens
		rep.OutputTokens += c.OutputTokens
		rep.CacheReadTokens += c.CachedPromptTokens
		rep.CacheCreationTokens += c.CacheCreationTokens
		rep.CachedTurns += c.CachedTurns
		rep.CompactionFired += c.CompactionFired
		rep.CompactionBailed += c.CompactionBailed
		rep.CompactionShedTokens += c.CompactionShedTokens
	}
	rep.ContextExtensionTokens = rep.CompactionShedTokens

	for _, row := range track1 {
		if row.Turns == 0 {
			continue
		}
		rep.Track1Sessions++
		rep.FakKVPrefixReusedTokens += row.ReusedTokens
	}

	var fallbackContextExtension uint64
	for _, row := range track2 {
		normalizeSavingsDimensions(&row)
		if row.DollarStatus == SavingsDollarStatusBlind {
			rep.DollarBlindRows++
		}
		switch {
		case row.Mechanism == "provider_prompt_cache":
			rep.ProviderPromptCacheTokenEq += providerTokenEqFromRow(row)
		case row.Provider == "fak" || strings.HasPrefix(row.Mechanism, "compaction"):
			rep.FakCompactionTokenEq += fakTokenEqFromRow(row)
			fallbackContextExtension += row.CompactionShedTokens
		}
		if row.DollarStatus != SavingsDollarStatusBlind {
			rep.ObservedActualSpendUSD += row.SpendUSD
			rep.ObservedAPICostAvoidedUSD += row.RebateUSD + row.CompactionSavedUSD - row.WritePremiumUSD
		}
	}
	if rep.ContextExtensionTokens == 0 {
		rep.ContextExtensionTokens = fallbackContextExtension
	}

	rep.FakAuthoredTokenEq = float64(rep.FakKVPrefixReusedTokens) + rep.FakCompactionTokenEq
	rep.TotalSavedTokenEq = rep.ProviderPromptCacheTokenEq + rep.FakAuthoredTokenEq
	if rep.TotalSavedTokenEq > 0 {
		pct := 100 * rep.FakAuthoredTokenEq / rep.TotalSavedTokenEq
		rep.FakSharePct = &pct
	}
	rep.ObservedCounterfactualUSD = rep.ObservedActualSpendUSD + rep.ObservedAPICostAvoidedUSD
	if rep.ObservedCounterfactualUSD != 0 {
		pct := 100 * rep.ObservedAPICostAvoidedUSD / rep.ObservedCounterfactualUSD
		rep.ObservedAPICostReductionPct = &pct
	}
	if opts.ContextBudgetTokens > 0 {
		windows := float64(rep.ContextExtensionTokens) / float64(opts.ContextBudgetTokens)
		pct := 100 * windows
		rep.EquivalentContextWindow = &windows
		rep.ContextExtensionPct = &pct
	}
	rep.fillFinding()
	return rep
}

func providerTokenEqFromRow(row SavingsRow) float64 {
	if row.NetSavedTokenEquiv != 0 {
		return row.NetSavedTokenEquiv
	}
	return row.SavedTokenEquiv
}

func fakTokenEqFromRow(row SavingsRow) float64 {
	if row.NetSavedTokenEquiv != 0 {
		return row.NetSavedTokenEquiv
	}
	return float64(row.CompactionShedTokens)
}

func (r *FleetBenefitReport) fillFinding() {
	if r.UsageRows == 0 && r.Track1Sessions == 0 && r.TotalSavedTokenEq == 0 {
		r.Finding = "no fleet usage or savings rows recorded yet"
		return
	}
	parts := []string{
		fmt.Sprintf("%d usage row(s), %d exit session(s)", r.UsageRows, r.ExitSessions),
		fmt.Sprintf("%.0f token-equiv saved", r.TotalSavedTokenEq),
	}
	if r.ObservedCounterfactualUSD != 0 {
		parts = append(parts, fmt.Sprintf("$%.4f API cost avoided", r.ObservedAPICostAvoidedUSD))
	}
	if r.ContextExtensionTokens > 0 {
		parts = append(parts, fmt.Sprintf("%d context token(s) shed", r.ContextExtensionTokens))
	}
	r.Finding = strings.Join(parts, "; ")
}

// RenderFleetBenefit renders the all-time/caller-window aggregate as a compact section
// beneath the two-track report.
func RenderFleetBenefit(r FleetBenefitReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "\nFleet aggregate (cumulative over recorded rows)\n")
	if r.Finding != "" {
		fmt.Fprintf(&b, "  %s\n", r.Finding)
	}
	if r.UsageRows == 0 && r.Track1Sessions == 0 && r.TotalSavedTokenEq == 0 {
		fmt.Fprintf(&b, "  no durable usage/savings rows yet\n")
		return b.String()
	}
	fmt.Fprintf(&b, "  usage: rows=%d exit_sessions=%d uptime=%.0fs decisions=%d allow/deny/repair/quarantine=%d/%d/%d/%d\n",
		r.UsageRows, r.ExitSessions, r.UptimeSecs, r.KernelDecisions, r.Allowed, r.Denied, r.Transformed, r.Quarantined)
	fmt.Fprintf(&b, "  tokens: input=%d output=%d cache_read=%d cache_write=%d cached_turns=%d\n",
		r.InputTokens, r.OutputTokens, r.CacheReadTokens, r.CacheCreationTokens, r.CachedTurns)
	fmt.Fprintf(&b, "  saved token-equiv: provider=%.0f fak=%.0f total=%.0f",
		r.ProviderPromptCacheTokenEq, r.FakAuthoredTokenEq, r.TotalSavedTokenEq)
	if r.FakSharePct != nil {
		fmt.Fprintf(&b, " fak_share=%.4f%%", *r.FakSharePct)
	}
	b.WriteByte('\n')
	if r.ObservedCounterfactualUSD != 0 || r.ObservedActualSpendUSD != 0 || r.ObservedAPICostAvoidedUSD != 0 {
		reduction := "-"
		if r.ObservedAPICostReductionPct != nil {
			reduction = fmt.Sprintf("%.2f%%", *r.ObservedAPICostReductionPct)
		}
		fmt.Fprintf(&b, "  API cost: observed_spend=$%.4f counterfactual=$%.4f avoided=$%.4f reduction=%s\n",
			r.ObservedActualSpendUSD, r.ObservedCounterfactualUSD, r.ObservedAPICostAvoidedUSD, reduction)
	}
	if r.ContextExtensionTokens > 0 || r.ContextBudgetTokens > 0 {
		fmt.Fprintf(&b, "  session extension: %d WITNESSED context token(s) shed", r.ContextExtensionTokens)
		if r.ContextBudgetTokens > 0 && r.EquivalentContextWindow != nil && r.ContextExtensionPct != nil {
			fmt.Fprintf(&b, " = %.2f%% of a %d-token budget (%.4f window-equivalent)",
				*r.ContextExtensionPct, r.ContextBudgetTokens, *r.EquivalentContextWindow)
		} else {
			fmt.Fprintf(&b, " (pass --context-budget-tokens to normalize into window-equivalent)")
		}
		b.WriteByte('\n')
	}
	if r.DollarBlindRows > 0 {
		fmt.Fprintf(&b, "  pricing: %d savings row(s) dollar-blind; token evidence is counted, dollar fields are not\n", r.DollarBlindRows)
	}
	fmt.Fprintf(&b, "  provenance: %s\n", r.Provenance)
	return b.String()
}
