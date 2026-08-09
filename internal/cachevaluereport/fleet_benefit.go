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

	// ShedRows counts the ledgered rows (each one session) whose WITNESSED
	// compaction shed actually contributed to ContextExtensionTokens — the artifact
	// citation behind the session-extension figure (#2711). When the usage ledger
	// carries the shed it counts usage rows; when the Track-2 savings fallback
	// supplies it, it counts the contributing savings rows instead (the two sources
	// are never mixed, same precedence as ContextExtensionTokens itself). Zero
	// alongside a non-empty corpus means the formula's precondition (fak compaction
	// fired and shed) is unmet — the render says so explicitly rather than showing
	// a bare zero.
	ShedRows int `json:"shed_rows,omitempty"`

	ContextExtensionPct     *float64 `json:"context_extension_pct,omitempty"`
	EquivalentContextWindow *float64 `json:"equivalent_context_windows,omitempty"`

	Track1Sessions             int      `json:"track1_sessions"`
	ProviderPromptCacheTokenEq float64  `json:"provider_prompt_cache_token_equiv"`
	FakKVPrefixReusedTokens    uint64   `json:"fak_kv_prefix_reused_tokens"`
	FakCompactionTokenEq       float64  `json:"fak_compaction_token_equiv"`
	FakAuthoredTokenEq         float64  `json:"fak_authored_token_equiv"`
	TotalSavedTokenEq          float64  `json:"total_saved_token_equiv"`
	FakSharePct                *float64 `json:"fak_share_pct,omitempty"`

	// FakCompactionShedTokensSavings is the raw fak-authored compaction shed tokens
	// summed from the Track-2 SAVINGS rows — the denominator the #2807 basis sweep
	// reprices. It is DISTINCT from CompactionShedTokens above (the usage-ledger
	// operational count) and never summed with it; the savings rows are the axis the
	// fak_share token-equiv is actually folded from, so the sweep reprices exactly the
	// shed that backs FakCompactionTokenEq.
	FakCompactionShedTokensSavings uint64 `json:"fak_compaction_shed_tokens_savings,omitempty"`

	// FakShareGrossPct / FakShareMarginalPct are the fleet fak_share recomputed with
	// the fak compaction-shed token-equiv valued at two extra price bases, so the
	// valuation assumption behind the headline fak_share (FakSharePct, the honest
	// per-row warm/cold blend) is VISIBLE instead of buried (#2807). WITNESSED
	// KV-prefix reuse and the provider token-equiv are held fixed across all three;
	// only the shed is repriced. Gross books every shed token at full input (1.0x) —
	// the pre-#2794 overstatement; Marginal books it at the 0.1x cache-read marginal
	// (the conservative floor). By construction Marginal <= Net(FakSharePct) <= Gross,
	// and the Gross−Net gap is the overstatement made visible. Each is nil when the
	// recomputed total is not positive, mirroring FakSharePct so an empty/upside-down
	// corpus never renders a 0%/100% claim.
	FakShareGrossPct    *float64 `json:"fak_share_gross_pct,omitempty"`
	FakShareMarginalPct *float64 `json:"fak_share_marginal_pct,omitempty"`

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

	// Billing-seat split of the SAME priced rows (#3664). The blended headline above
	// prices every row at published list $/MTok with no billing input at all, so it is an
	// API-key-EQUIVALENT projection over the whole corpus: real-dollar for the
	// per-token-billed subset, notional for the flat-rate subscription rows sitting in the
	// same ledger. These two columns partition that total by the posture each row stamped
	// at write time, so "real dollars even on API-key billing" is auditable from the
	// ledger rather than asserted over the blend.
	//
	// BOTH columns stay OBSERVED — list-priced projections, never a reconciled invoice.
	// The split does not upgrade either side to a witnessed invoice figure; it only makes
	// the API-key-vs-OAuth attribution visible. Notional carries oauth AND unknown (and
	// therefore every row written before the field existed), because a row that cannot
	// prove its seat is never counted as real money.
	//
	// APIKey* + Notional* sum EXACTLY to the blended Observed* triple above; the reduction
	// percentages do NOT sum, each being its own column's avoided/counterfactual ratio.
	// Each pct is nil when its column has no priced counterfactual, mirroring the blended
	// pointer so an absent column renders "-" instead of a 0% claim.
	APIKeyRows                int      `json:"api_key_rows,omitempty"`
	APIKeyActualSpendUSD      float64  `json:"api_key_actual_spend_usd,omitempty"`
	APIKeyAvoidedUSD          float64  `json:"api_key_avoided_usd,omitempty"`
	APIKeyCounterfactualUSD   float64  `json:"api_key_counterfactual_usd,omitempty"`
	APIKeyReductionPct        *float64 `json:"api_key_reduction_pct,omitempty"`
	NotionalRows              int      `json:"notional_rows,omitempty"`
	NotionalActualSpendUSD    float64  `json:"notional_actual_spend_usd,omitempty"`
	NotionalAvoidedUSD        float64  `json:"notional_avoided_usd,omitempty"`
	NotionalCounterfactualUSD float64  `json:"notional_counterfactual_usd,omitempty"`
	NotionalReductionPct      *float64 `json:"notional_reduction_pct,omitempty"`

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

	// Cost per unit of agentic WORK (#3662). Everything above divides the cumulative
	// totals by TIME (SpanDays) or by WINDOW (ContextBudgetTokens); these divide by
	// the work actually done, using denominators this same fold already aggregates
	// (KernelDecisions, ExitSessions, MultiTurnTurns). Nothing new is measured — the
	// run-rate is only made legible as the three-factor product
	// (per-turn reuse ratio) x (work per turn) x (turn count) instead of a bare
	// calendar rate.
	//
	// PROVENANCE IS SPLIT AND NEVER BLENDED, each field inheriting the fence of the
	// numerator it divides: every *USD*-denominated per-work number is OBSERVED (a
	// projection priced at list $/MTok from provider-relayed counters, never a
	// reconciled invoice) and carries RateProvisional under a thin span; every
	// *token*-denominated per-work number is WITNESSED. A nil pointer means the
	// denominator was zero — the metric is UNDEFINED, not zero — matching FakSharePct
	// and EquivalentContextWindow so an empty corpus never renders a measured-looking 0.
	//
	// MultiTurn* are gated on turns >= 2, the same multi-turn corpus
	// cachevalueledger.ScoreLedger gates its reuse ratio on: a single-turn cold run has
	// no previous turn to reuse from, so folding it in would deflate per-turn work size
	// with turns that could never have reused anything.
	MultiTurnTurns        uint64 `json:"multi_turn_turns,omitempty"`
	MultiTurnPromptTokens uint64 `json:"multi_turn_prompt_tokens,omitempty"`
	MultiTurnReusedTokens uint64 `json:"multi_turn_reused_tokens,omitempty"`

	AvoidedUSDPerDecision        *float64 `json:"avoided_usd_per_decision,omitempty"`
	AvoidedUSDPerExitSession     *float64 `json:"avoided_usd_per_exit_session,omitempty"`
	SpendUSDPerDecision          *float64 `json:"observed_spend_usd_per_decision,omitempty"`
	CounterfactualUSDPerDecision *float64 `json:"counterfactual_usd_per_decision,omitempty"`

	SavedTokenEqPerTurn *float64 `json:"saved_token_eq_per_turn,omitempty"`
	PromptTokensPerTurn *float64 `json:"prompt_tokens_per_turn,omitempty"`
	ReusedTokensPerTurn *float64 `json:"reused_tokens_per_turn,omitempty"`
	ShedTokensPerTurn   *float64 `json:"shed_tokens_per_turn,omitempty"`

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
		if c.CompactionShedTokens > 0 {
			rep.ShedRows++
		}
	}
	rep.ContextExtensionTokens = rep.CompactionShedTokens

	for _, row := range track1 {
		if row.Turns == 0 {
			continue
		}
		rep.Track1Sessions++
		rep.FakKVPrefixReusedTokens += row.ReusedTokens
		// Per-turn work size (#3662) is folded over the MULTI-TURN corpus only, the
		// same turns >= 2 gate cachevalueledger.ScoreLedger uses: a single-turn cold
		// run has no previous turn to reuse from, so counting its prompt would deflate
		// "work per turn" with turns that could never have reused a prefix. Note this
		// is deliberately NARROWER than FakKVPrefixReusedTokens above, which stays the
		// all-sessions KV-reuse total feeding fak_share.
		if row.Turns >= 2 {
			rep.MultiTurnTurns += row.Turns
			rep.MultiTurnPromptTokens += row.PromptTokens
			rep.MultiTurnReusedTokens += row.ReusedTokens
		}
	}

	var fallbackContextExtension uint64
	var fallbackShedRows int
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
			rep.FakCompactionShedTokensSavings += row.CompactionShedTokens
			fallbackContextExtension += row.CompactionShedTokens
			if row.CompactionShedTokens > 0 {
				fallbackShedRows++
			}
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
			avoided := row.RebateUSD + row.CompactionSavedUSD - row.WritePremiumUSD
			rep.ObservedAPICostAvoidedUSD += avoided
			// Partition the SAME dollars by the seat the row stamped (#3664). Only a
			// proven per-token-billed row lands in the real-$ column; oauth, unknown and
			// unstamped (every pre-#3664 row) all fold notional.
			if RealDollarBillingMode(row.BillingMode) {
				rep.APIKeyRows++
				rep.APIKeyActualSpendUSD += row.SpendUSD
				rep.APIKeyAvoidedUSD += avoided
			} else {
				rep.NotionalRows++
				rep.NotionalActualSpendUSD += row.SpendUSD
				rep.NotionalAvoidedUSD += avoided
			}
		}
	}
	if rep.ContextExtensionTokens == 0 {
		rep.ContextExtensionTokens = fallbackContextExtension
		rep.ShedRows = fallbackShedRows
	}

	rep.FakAuthoredTokenEq = float64(rep.FakKVPrefixReusedTokens) + rep.FakCompactionTokenEq
	rep.TotalSavedTokenEq = rep.ProviderPromptCacheTokenEq + rep.FakAuthoredTokenEq
	if rep.TotalSavedTokenEq > 0 {
		pct := 100 * rep.FakAuthoredTokenEq / rep.TotalSavedTokenEq
		rep.FakSharePct = &pct
	}
	// Basis sweep (#2807): reprice the fak compaction shed at 1.0x (gross) and 0.1x
	// (marginal), holding WITNESSED KV reuse and provider token-equiv fixed, so the
	// gap between the headline honest-blend fak_share (FakSharePct) and its full-input
	// overstatement is visible instead of buried. The 0.1x marginal READS the canonical
	// providerCacheReadMultiplier so it can never drift from the value the fold and the
	// fire gate price a cached token at.
	// Populated only when there is a shed to reprice; with no fak compaction shed all
	// three bases collapse to the KV-reuse-only share, so the sweep is left nil (and
	// omitted from JSON) rather than echoing FakSharePct three times.
	if shed := float64(rep.FakCompactionShedTokensSavings); shed > 0 {
		if g, ok := fakShareAtBasis(rep.FakKVPrefixReusedTokens, rep.ProviderPromptCacheTokenEq, shed); ok {
			rep.FakShareGrossPct = &g
		}
		if m, ok := fakShareAtBasis(rep.FakKVPrefixReusedTokens, rep.ProviderPromptCacheTokenEq, shed*providerCacheReadMultiplier); ok {
			rep.FakShareMarginalPct = &m
		}
	}
	rep.ObservedCounterfactualUSD = rep.ObservedActualSpendUSD + rep.ObservedAPICostAvoidedUSD
	if rep.ObservedCounterfactualUSD != 0 {
		pct := 100 * rep.ObservedAPICostAvoidedUSD / rep.ObservedCounterfactualUSD
		rep.ObservedAPICostReductionPct = &pct
	}
	// Each seat column gets its OWN counterfactual denominator, so a corpus that is all
	// one seat reproduces the blended headline exactly in that column and reports the
	// other as absent ("-") rather than as a 0% reduction (#3664).
	rep.APIKeyCounterfactualUSD = rep.APIKeyActualSpendUSD + rep.APIKeyAvoidedUSD
	if rep.APIKeyCounterfactualUSD != 0 {
		pct := 100 * rep.APIKeyAvoidedUSD / rep.APIKeyCounterfactualUSD
		rep.APIKeyReductionPct = &pct
	}
	rep.NotionalCounterfactualUSD = rep.NotionalActualSpendUSD + rep.NotionalAvoidedUSD
	if rep.NotionalCounterfactualUSD != 0 {
		pct := 100 * rep.NotionalAvoidedUSD / rep.NotionalCounterfactualUSD
		rep.NotionalReductionPct = &pct
	}
	if opts.ContextBudgetTokens > 0 {
		windows := float64(rep.ContextExtensionTokens) / float64(opts.ContextBudgetTokens)
		pct := 100 * windows
		rep.EquivalentContextWindow = &windows
		rep.ContextExtensionPct = &pct
	}
	rep.fillRunRate()
	rep.fillPerWorkCost()
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

// fillPerWorkCost folds the cumulative dollars and token-equiv into cost per unit of
// agentic WORK (#3662), dividing by denominators this fold already aggregates —
// kernel decisions, exit sessions, multi-turn turns — rather than by the calendar.
// It measures nothing new: it re-expresses the same totals so "$X/day" is legible as
// what one unit of agentic work cost with vs without fak.
//
// Each ratio is left nil when its denominator is zero, so an empty or single-turn
// corpus reports the metric as UNDEFINED instead of a 0.00 that would read as a
// measured floor. The dollar axis (OBSERVED, list-priced projection) and the token
// axis (WITNESSED) are computed into separate fields and never blended into one
// figure — the render prints them on separate, separately-fenced lines.
func (r *FleetBenefitReport) fillPerWorkCost() {
	if decisions := float64(r.KernelDecisions); decisions > 0 {
		// The counterfactual pair: what one kernel decision actually cost, against
		// what it would have cost with no cache work at all. Both OBSERVED.
		avoided := r.ObservedAPICostAvoidedUSD / decisions
		spend := r.ObservedActualSpendUSD / decisions
		counterfactual := r.ObservedCounterfactualUSD / decisions
		r.AvoidedUSDPerDecision = &avoided
		r.SpendUSDPerDecision = &spend
		r.CounterfactualUSDPerDecision = &counterfactual
	}
	if sessions := float64(r.ExitSessions); sessions > 0 {
		avoided := r.ObservedAPICostAvoidedUSD / sessions
		r.AvoidedUSDPerExitSession = &avoided
	}
	if turns := float64(r.MultiTurnTurns); turns > 0 {
		// Work-per-turn size: the "massiveness" of a turn that MultiTurnTurns alone
		// only counts. Shed rides on ContextExtensionTokens (WITNESSED fak-authored
		// compaction shed), the same axis the session-extension figure cites.
		saved := r.TotalSavedTokenEq / turns
		prompt := float64(r.MultiTurnPromptTokens) / turns
		reused := float64(r.MultiTurnReusedTokens) / turns
		shed := float64(r.ContextExtensionTokens) / turns
		r.SavedTokenEqPerTurn = &saved
		r.PromptTokensPerTurn = &prompt
		r.ReusedTokensPerTurn = &reused
		r.ShedTokensPerTurn = &shed
	}
}

func providerTokenEqFromRow(row SavingsRow) float64 {
	if row.NetSavedTokenEquiv != 0 {
		return row.NetSavedTokenEquiv
	}
	return row.SavedTokenEquiv
}

func fakTokenEqFromRow(row SavingsRow) float64 {
	// Shared with the owner-attribution fold so a legacy (unpriced) warm-fire row is
	// re-priced at the 0.1x cache-read marginal in BOTH roll-ups, never the raw 1.0x
	// this used to book here (#2798).
	return fakAuthoredTokenEquiv(row.NetSavedTokenEquiv, row.CompactionShedTokens, row.CompactionCacheReadTokens)
}

// fakShareAtBasis recomputes the fleet fak_share (percent) with the fak
// compaction-shed token-equiv set to shedTeq, holding the WITNESSED KV-prefix
// reuse and the provider token-equiv fixed — the ONE place the #2807 basis sweep
// reprices the share. It returns ok=false when the recomputed total is not
// positive, mirroring FakSharePct so an empty/upside-down corpus never renders a
// 0%/100% claim.
func fakShareAtBasis(kvReusedTokens uint64, providerTeq, shedTeq float64) (float64, bool) {
	fakAuthored := float64(kvReusedTokens) + shedTeq
	total := providerTeq + fakAuthored
	if total <= 0 {
		return 0, false
	}
	return 100 * fakAuthored / total, true
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

// benefitLensHead opens a benefit-lens section the way every lens opens one: a blank
// separator line, the section title, and the one-line Finding when the fold produced
// one. The caller keeps its OWN "nothing to report" early return, because what counts as
// an empty lens differs per lens (no transcripts vs no durable ledger rows).
func benefitLensHead(b *strings.Builder, title, finding string) {
	fmt.Fprintf(b, "\n%s\n", title)
	if finding != "" {
		fmt.Fprintf(b, "  %s\n", finding)
	}
}

// apiCostReduction is the shared gate and reduction cell of every lens's API-cost line:
// it reports whether ANY observed cost axis is non-zero (an all-zero lens prints no cost
// line at all, rather than a row of zeros that reads like a measurement), and renders the
// reduction cell identically everywhere — "-" when the pointer is nil, because a missing
// counterfactual denominator has no honest percentage, else a 2-decimal percent. The
// Fprintf itself stays with each caller: the lenses print different cost breakdowns (the
// fleet aggregate splits avoided cost into provider and fak halves; the dev-session lens
// has no such split).
func apiCostReduction(counterfactual, spend, avoided float64, pct *float64) (string, bool) {
	if counterfactual == 0 && spend == 0 && avoided == 0 {
		return "", false
	}
	return reductionCell(pct), true
}

// reductionCell is the reduction percentage as every lens prints it: "-" when the pointer
// is nil, because a missing counterfactual denominator has no honest percentage, else a
// 2-decimal percent. Split out of apiCostReduction so the billing-seat columns (#3664)
// render the same cell as the blended headline instead of growing a second format string
// that could drift from it.
func reductionCell(pct *float64) string {
	if pct == nil {
		return "-"
	}
	return fmt.Sprintf("%.2f%%", *pct)
}

// compactionLeverBuckets selects the savings buckets carrying compaction-lever telemetry
// — the fak-authored rows plus any mechanism in the compaction family — so the terminal
// and markdown render paths agree on exactly which rows the fire/starve/shed trend is
// folded from. Returns nil when the ledger has none, which both callers read as "no
// lever health section".
// leverCells is one compaction-lever row's cells in the fixed column order every
// renderer prints: period, sessions, fired, bailed, anchor-starved, shed tokens,
// budget. The markdown P&L and the terminal P&L differ only in their format string
// (pipe table vs padded columns), so the ORDER and the SET of columns live here once.
// That is the part that actually goes wrong twice: a lever counter added to one
// renderer and forgotten in the other reads as an inert lever in exactly one view.
// Each caller keeps its own verbs, so the rendered bytes of both views are unchanged.
func (b SavingsBucket) leverCells() []any {
	return []any{b.Period, b.Sessions, b.CompactionFired, b.CompactionBailed,
		b.CompactionAnchorStarved, b.CompactionShedTokens, b.CompactionBudget}
}

func compactionLeverBuckets(buckets []SavingsBucket) []SavingsBucket {
	var out []SavingsBucket
	for _, b := range buckets {
		if b.Provider == "fak" || strings.HasPrefix(b.Mechanism, "compaction") {
			out = append(out, b)
		}
	}
	return out
}

// RenderFleetBenefit renders the all-time/caller-window aggregate as a compact section
// beneath the two-track report.
func RenderFleetBenefit(r FleetBenefitReport) string {
	var b strings.Builder
	benefitLensHead(&b, "Fleet aggregate (cumulative over recorded rows)", r.Finding)
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
	// Basis sweep (#2807): show the fleet fak_share under the three price bases side
	// by side so the valuation assumption is visible instead of buried. Only rendered
	// when a fak compaction shed exists to reprice (otherwise all three collapse to
	// the KV-reuse-only share); a fully-cold shed honestly makes gross==net (1.0x IS
	// the cold basis) while a warm/blended shed opens the gross−net gap.
	if r.FakCompactionShedTokensSavings > 0 && r.FakShareGrossPct != nil && r.FakShareMarginalPct != nil && r.FakSharePct != nil {
		fmt.Fprintf(&b, "  fak_share basis sweep (shed valued 3 ways; #2807): gross(1.0x)=%.4f%% marginal(0.1x)=%.4f%% net(observed)=%.4f%% — gross−net gap=%.4f pp (overstatement)\n",
			*r.FakShareGrossPct, *r.FakShareMarginalPct, *r.FakSharePct, *r.FakShareGrossPct-*r.FakSharePct)
	}
	if reduction, ok := apiCostReduction(r.ObservedCounterfactualUSD, r.ObservedActualSpendUSD, r.ObservedAPICostAvoidedUSD, r.ObservedAPICostReductionPct); ok {
		fmt.Fprintf(&b, "  API cost: observed_spend=$%.4f counterfactual=$%.4f avoided=$%.4f (provider $%.4f + fak $%.4f) reduction=%s\n",
			r.ObservedActualSpendUSD, r.ObservedCounterfactualUSD, r.ObservedAPICostAvoidedUSD,
			r.ProviderAPICostAvoidedUSD, r.FakAPICostAvoidedUSD, reduction)
		// Billing-seat split (#3664): the blended line above prices every row at list
		// $/MTok with no billing input, so it is an API-key-EQUIVALENT projection over
		// the whole corpus. These two lines say which half of it was actually billed per
		// token. BOTH stay OBSERVED — the split re-attributes the same projected dollars,
		// it does not reconcile either against an invoice.
		fmt.Fprintf(&b, "    billing seat (OBSERVED both columns — list-priced projection, never a reconciled invoice; #3664):\n")
		fmt.Fprintf(&b, "      API-key (REAL-$; %d priced row(s)): spend=$%.4f counterfactual=$%.4f avoided=$%.4f reduction=%s\n",
			r.APIKeyRows, r.APIKeyActualSpendUSD, r.APIKeyCounterfactualUSD, r.APIKeyAvoidedUSD, reductionCell(r.APIKeyReductionPct))
		fmt.Fprintf(&b, "      OAuth/unknown (NOTIONAL — flat-rate seat, no per-token invoice; %d priced row(s)): spend=$%.4f counterfactual=$%.4f avoided=$%.4f reduction=%s\n",
			r.NotionalRows, r.NotionalActualSpendUSD, r.NotionalCounterfactualUSD, r.NotionalAvoidedUSD, reductionCell(r.NotionalReductionPct))
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
	// Cost per unit of agentic work (#3662): the same cumulative totals divided by the
	// work done rather than by the calendar. The dollar lines and the token lines are
	// printed SEPARATELY, each carrying its own OBSERVED/WITNESSED tag, so a
	// list-priced projection can never be read as a witnessed count. The dollar lines
	// inherit the thin-window PROVISIONAL flag the run-rate above already computed.
	if r.AvoidedUSDPerDecision != nil || r.AvoidedUSDPerExitSession != nil || r.SavedTokenEqPerTurn != nil {
		prov := ""
		if r.RateProvisional {
			prov = fmt.Sprintf(" [PROVISIONAL: span < %.0fd]", minHonestSpanDays)
		}
		fmt.Fprintf(&b, "  cost per unit of agentic work:\n")
		if r.SpendUSDPerDecision != nil && r.CounterfactualUSDPerDecision != nil && r.AvoidedUSDPerDecision != nil {
			fmt.Fprintf(&b, "    $/decision (OBSERVED, list-priced projection; over %d decision(s)): spend $%.6f vs counterfactual $%.6f = avoided $%.6f%s\n",
				r.KernelDecisions, *r.SpendUSDPerDecision, *r.CounterfactualUSDPerDecision, *r.AvoidedUSDPerDecision, prov)
		}
		if r.AvoidedUSDPerExitSession != nil {
			fmt.Fprintf(&b, "    $/exit-session (OBSERVED, list-priced projection; over %d session(s)): avoided $%.6f%s\n",
				r.ExitSessions, *r.AvoidedUSDPerExitSession, prov)
		}
		if r.SavedTokenEqPerTurn != nil {
			fmt.Fprintf(&b, "    saved-tok-eq/turn (WITNESSED; over %d multi-turn turn(s)): %.2f\n",
				r.MultiTurnTurns, *r.SavedTokenEqPerTurn)
		}
		if r.PromptTokensPerTurn != nil && r.ReusedTokensPerTurn != nil && r.ShedTokensPerTurn != nil {
			fmt.Fprintf(&b, "    work per turn (WITNESSED): prompt=%.2f reused_prefix=%.2f shed=%.2f token(s)\n",
				*r.PromptTokensPerTurn, *r.ReusedTokensPerTurn, *r.ShedTokensPerTurn)
			// The explicit ratio x size x count decomposition. It is an identity by
			// construction (reuse ratio x prompt/turn x turns == reused tokens), which
			// is the point: the aggregate is shown to BE the three-factor product, so
			// the volume thesis is legible instead of implicit inside a summed total.
			if *r.PromptTokensPerTurn > 0 {
				ratio := *r.ReusedTokensPerTurn / *r.PromptTokensPerTurn
				fmt.Fprintf(&b, "    three-factor decomposition (WITNESSED): reuse_ratio %.4f x %.2f prompt tok/turn x %d turn(s) = %d reused token(s)\n",
					ratio, *r.PromptTokensPerTurn, r.MultiTurnTurns, r.MultiTurnReusedTokens)
			}
		}
	}
	if r.ContextExtensionTokens > 0 || r.ContextBudgetTokens > 0 || r.UsageRows > 0 {
		if r.ContextExtensionTokens == 0 {
			// Honest zero (#2711): a zero here means the formula's precondition — fak
			// compaction fired and shed in at least one ledgered session — is unmet.
			// Say so, with the corpus size, instead of a bare 0 (or no line at all)
			// that could read as a wiring gap or a silently-stale figure.
			fmt.Fprintf(&b, "  session extension: still zero — no WITNESSED compaction-shed tokens in any of the %d recorded usage row(s); the formula's precondition (fak compaction fired) is unmet, not a wiring gap\n", r.UsageRows)
		} else {
			fmt.Fprintf(&b, "  session extension: %d WITNESSED context token(s) shed", r.ContextExtensionTokens)
			if r.ContextBudgetTokens > 0 && r.EquivalentContextWindow != nil && r.ContextExtensionPct != nil {
				fmt.Fprintf(&b, " = %.2f%% of a %d-token budget (%.4f window-equivalent)",
					*r.ContextExtensionPct, r.ContextBudgetTokens, *r.EquivalentContextWindow)
			} else {
				fmt.Fprintf(&b, " (pass --context-budget-tokens to normalize into window-equivalent)")
			}
			fmt.Fprintf(&b, "; cited from %d ledgered shed session(s)\n", r.ShedRows)
		}
	}
	if r.DollarBlindRows > 0 {
		fmt.Fprintf(&b, "  pricing: %d savings row(s) dollar-blind; token evidence is counted, dollar fields are not\n", r.DollarBlindRows)
	}
	fmt.Fprintf(&b, "  provenance: %s\n", r.Provenance)
	return b.String()
}
