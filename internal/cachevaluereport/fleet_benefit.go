package cachevaluereport

import (
	"fmt"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// observeTime widens [first,last] to include t, skipping the zero time so an
// unparseable/absent row date never collapses the span to the epoch.
func observeTime(first, last *time.Time, t time.Time) {
	if t.IsZero() {
		return
	}
	if first.IsZero() || t.Before(*first) {
		*first = t
	}
	if last.IsZero() || t.After(*last) {
		*last = t
	}
}

// parseRowTime parses an RFC3339 timestamp, falling back to a YYYY-MM-DD date;
// an unparseable value yields the zero time (skipped by observeTime), matching
// the "skip, do not error the fold" convention the Track-2 bucketer uses.
func parseRowTime(rfc3339, date string) time.Time {
	if rfc3339 != "" {
		if t, err := time.Parse(time.RFC3339, rfc3339); err == nil {
			return t.UTC()
		}
	}
	if date != "" {
		if t, err := time.Parse("2006-01-02", date); err == nil {
			return t.UTC()
		}
	}
	return time.Time{}
}

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

	// Owner-split of the avoided dollars. ObservedAPICostAvoidedUSD is kept as the
	// blended sum (= Provider + Fak) so % reduction and the counterfactual are
	// unchanged; the split names WHO earned each dollar so a provider-only corpus
	// cannot read as a fak win. Provider = read rebate − write premium on
	// provider_prompt_cache rows (OBSERVED/provider-relayed); Fak = compaction
	// saving on fak-authored rows (WITNESSED shed, dollar value still projected).
	ProviderAPICostAvoidedUSD float64 `json:"provider_api_cost_avoided_usd"`
	FakAPICostAvoidedUSD      float64 `json:"fak_api_cost_avoided_usd"`

	// ProviderCacheReadTokens is the display "cache_read=" provider-token count,
	// sourced from the Track-2 savings ledger (authoritative and complete) — NOT
	// from the gateway-usage CacheReadTokens field below, which is only wired at
	// guard teardown since 2026-07-03 and is therefore back-incomplete. The two are
	// deliberately kept in SEPARATE fields and never summed; if a future field ever
	// needs the union it must dedup by session identity (a session can appear in
	// both ledgers), or it will double-count provider prompt-cache reads.
	ProviderCacheReadTokens uint64 `json:"provider_cache_read_tokens"`

	// Time span + run-rate (long-horizon lens). Span is derived from the SAVINGS
	// rows that actually contributed dollars (SavingsFirstUTC/SavingsLastUTC), not
	// the union with usage rows whose wider timestamp range would deflate $/day.
	// FirstRowUTC/LastRowUTC cover the whole recorded corpus for context. All rates
	// are OBSERVED (provider-cache economics today) and PROVISIONAL under a thin
	// window; the per-day token rate is split provider/fak so a WITNESSED fak
	// component is never smuggled under an OBSERVED label.
	FirstRowUTC              time.Time `json:"first_row_utc,omitempty"`
	LastRowUTC               time.Time `json:"last_row_utc,omitempty"`
	SavingsFirstUTC          time.Time `json:"savings_first_utc,omitempty"`
	SavingsLastUTC           time.Time `json:"savings_last_utc,omitempty"`
	SpanDays                 float64   `json:"span_days"`
	RateProvisional          bool      `json:"rate_provisional"`
	USDAvoidedPerDay         float64   `json:"usd_avoided_per_day,omitempty"`
	USDAvoidedPerWeek        float64   `json:"usd_avoided_per_week,omitempty"`
	ProviderUSDAvoidedPerDay float64   `json:"provider_usd_avoided_per_day,omitempty"`
	FakUSDAvoidedPerDay      float64   `json:"fak_usd_avoided_per_day,omitempty"`
	ProviderTokenEqPerDay    float64   `json:"provider_token_eq_per_day,omitempty"`
	FakTokenEqPerDay         float64   `json:"fak_token_eq_per_day,omitempty"`

	Provenance string `json:"provenance"`
	Finding    string `json:"finding"`
}

// minHonestSpanDays is the thin-window floor: a run-rate over a shorter savings
// span is still shown but flagged PROVISIONAL so a 30/90-day extrapolation off a
// couple of days is never read as a settled figure.
const minHonestSpanDays = 3.0

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
		observeTime(&rep.FirstRowUTC, &rep.LastRowUTC, parseRowTime(row.GeneratedAt, ""))
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
		// Span for the dollar run-rate rides on the savings rows that actually
		// carry the avoided dollars, kept separate from the usage-row corpus span
		// so a wider usage timestamp range cannot deflate $/day.
		observeTime(&rep.SavingsFirstUTC, &rep.SavingsLastUTC, parseRowTime(row.GeneratedAt, row.Date))
		observeTime(&rep.FirstRowUTC, &rep.LastRowUTC, parseRowTime(row.GeneratedAt, row.Date))
		isProvider := row.Mechanism == "provider_prompt_cache"
		isFak := row.Provider == "fak" || strings.HasPrefix(row.Mechanism, "compaction")
		switch {
		case isProvider:
			rep.ProviderPromptCacheTokenEq += providerTokenEqFromRow(row)
			// Authoritative provider cache-read display source (Track-2 is complete
			// back to the first session; the usage-ledger CacheReadTokens is only
			// wired since 2026-07-03 and back-incomplete). Kept in its own field,
			// never summed with CacheReadTokens — see ProviderCacheReadTokens doc.
			rep.ProviderCacheReadTokens += row.CacheReadTokens
		case isFak:
			rep.FakCompactionTokenEq += fakTokenEqFromRow(row)
			fallbackContextExtension += row.CompactionShedTokens
		}
		if row.DollarStatus != SavingsDollarStatusBlind {
			rep.ObservedActualSpendUSD += row.SpendUSD
			// Split the avoided dollars by owner: provider = read rebate net of the
			// cache-write premium it charges; fak = the compaction saving fak
			// authored. The blended ObservedAPICostAvoidedUSD stays their exact sum.
			switch {
			case isProvider:
				rep.ProviderAPICostAvoidedUSD += row.RebateUSD - row.WritePremiumUSD
			case isFak:
				rep.FakAPICostAvoidedUSD += row.CompactionSavedUSD
			default:
				// An unclassified priced row still contributes to the blended total;
				// attribute it to provider (its rebate/premium are provider-side).
				rep.ProviderAPICostAvoidedUSD += row.RebateUSD - row.WritePremiumUSD
				rep.FakAPICostAvoidedUSD += row.CompactionSavedUSD
			}
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
	rep.fillRunRate()
	rep.fillFinding()
	return rep
}

// fillRunRate normalizes the cumulative avoided dollars and saved token-equiv
// into per-day/per-week rates over the SAVINGS-row span (the rows that carry the
// dollars), splitting provider vs fak so a thin, provider-only corpus reports a
// fak rate of exactly zero rather than a blended headline. A span under
// minHonestSpanDays is still rated but flagged RateProvisional. A degenerate span
// (single row, or all dates unparseable) leaves every rate at zero.
func (r *FleetBenefitReport) fillRunRate() {
	if r.SavingsFirstUTC.IsZero() || !r.SavingsLastUTC.After(r.SavingsFirstUTC) {
		return
	}
	r.SpanDays = r.SavingsLastUTC.Sub(r.SavingsFirstUTC).Hours() / 24
	if r.SpanDays <= 0 {
		return
	}
	r.RateProvisional = r.SpanDays < minHonestSpanDays
	r.ProviderUSDAvoidedPerDay = r.ProviderAPICostAvoidedUSD / r.SpanDays
	r.FakUSDAvoidedPerDay = r.FakAPICostAvoidedUSD / r.SpanDays
	r.USDAvoidedPerDay = r.ObservedAPICostAvoidedUSD / r.SpanDays
	r.USDAvoidedPerWeek = r.USDAvoidedPerDay * 7
	r.ProviderTokenEqPerDay = r.ProviderPromptCacheTokenEq / r.SpanDays
	r.FakTokenEqPerDay = r.FakAuthoredTokenEq / r.SpanDays
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
	fmt.Fprintf(&b, "  usage (WITNESSED operational; usage ledger complete since 2026-07-03): rows=%d exit_sessions=%d uptime=%.0fs decisions=%d allow/deny/repair/quarantine=%d/%d/%d/%d\n",
		r.UsageRows, r.ExitSessions, r.UptimeSecs, r.KernelDecisions, r.Allowed, r.Denied, r.Transformed, r.Quarantined)
	// cache_read is the authoritative provider prompt-cache read count from the
	// Track-2 savings ledger (complete back to the first session); input/output are
	// the usage-ledger operational axes. The two are never summed.
	fmt.Fprintf(&b, "  tokens: input=%d output=%d cache_read=%d (OBSERVED, Track-2) cache_write=%d cached_turns=%d\n",
		r.InputTokens, r.OutputTokens, r.ProviderCacheReadTokens, r.CacheCreationTokens, r.CachedTurns)
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
		fmt.Fprintf(&b, "  API cost: observed_spend=$%.4f counterfactual=$%.4f avoided=$%.4f (provider $%.4f + fak $%.4f) reduction=%s\n",
			r.ObservedActualSpendUSD, r.ObservedCounterfactualUSD, r.ObservedAPICostAvoidedUSD,
			r.ProviderAPICostAvoidedUSD, r.FakAPICostAvoidedUSD, reduction)
	}
	if r.SpanDays > 0 {
		prov := ""
		if r.RateProvisional {
			prov = fmt.Sprintf(" [PROVISIONAL: span < %.0fd — thin window, extrapolate with caution]", minHonestSpanDays)
		}
		fmt.Fprintf(&b, "  run-rate (OBSERVED provider-cache economics, over %.2fd %s..%s): $/day provider $%.2f + fak $%.2f = $%.2f; $%.2f/week; saved-tok-eq/day provider %.0f + fak %.0f%s\n",
			r.SpanDays, r.SavingsFirstUTC.Format("2006-01-02"), r.SavingsLastUTC.Format("2006-01-02"),
			r.ProviderUSDAvoidedPerDay, r.FakUSDAvoidedPerDay, r.USDAvoidedPerDay, r.USDAvoidedPerWeek,
			r.ProviderTokenEqPerDay, r.FakTokenEqPerDay, prov)
		fmt.Fprintf(&b, "  projection (OBSERVED, straight-line at current rate): 30d ~= provider $%.0f + fak $%.0f; 90d ~= provider $%.0f + fak $%.0f%s\n",
			r.ProviderUSDAvoidedPerDay*30, r.FakUSDAvoidedPerDay*30,
			r.ProviderUSDAvoidedPerDay*90, r.FakUSDAvoidedPerDay*90, prov)
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
