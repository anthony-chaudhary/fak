package cachevaluereport

// Track 2 of epic #1301 — the OBSERVED provider-$ cache economics — and the
// two-track P&L fold (#1304) that puts it SIDE BY SIDE with the WITNESSED kernel
// reuse (Track 1) without ever blending them.
//
// PROVENANCE FENCE (the whole reason the tracks never merge): Track 1 is
// WITNESSED — fak authored the cache bytes, so the reuse is byte-identical by
// construction. Track 2 is OBSERVED — the provider reported cache_read /
// cache_creation and fak only relays it; the $ NET is a cost PROJECTION priced
// from a caller-supplied base $/MTok, never a fak-WITNESSED claim. Mixing a
// projection into a witnessed number would launder the projection's uncertainty
// into the witness, so the fold keeps two separate accounts and emits a NET line
// that is explicitly labelled OBSERVED.
//
// The Track-2 ledger row mirrors the axes named by rung B (#1303): the four
// billable token axes the provider reports (input / cache_read / cache_creation /
// output), the compaction token-shed (`--compact-history-budget`, #745), and the
// $ duals priced from the model's base input/output $/MTok. This package defines
// the row, reader, and append helper; the CLI front doors wire live session summaries
// into those rows.

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

// SavingsLedgerSchema tags each durable OBSERVED-$ row (rung B, #1303). It is
// distinct from the kernel ledger schema (Track 1) so a reader can never confuse
// a WITNESSED row for an OBSERVED one.
const SavingsLedgerSchema = "fak-cache-savings-ledger/1"

// DefaultSavingsLedgerRel is the live Track-2 OBSERVED-$ ledger. Runtime rows
// stay under the gitignored .fak state root so guard/serve exits cannot dirty the
// shared tree. The tracked docs/nightrun/cache-savings.jsonl file is the last
// published historical snapshot; the two provenances still never share a row.
const DefaultSavingsLedgerRel = ".fak/nightrun/cache-savings.jsonl"

// providerCacheReadMultiplier is the price of a cached-prefix READ relative to base input.
// It READS the canonical cacheprice.ReadMultiplier (a tier-1 leaf this tier-2 package may
// import) rather than re-declaring the 0.1 literal, so the provider-rebate math here and the
// gateway/agent/resume pricing surfaces value an identical cached token identically by
// CONSTRUCTION (#2798) — not by a drift-pin test mirroring a bare literal.
const providerCacheReadMultiplier = cacheprice.ReadMultiplier

// compactionShedMarginalMultiplier is the marginal value of a compaction-shed token when the
// fire landed on a WARM prefix: a token the provider would have billed as a cache_read, not
// fresh input at 1.0x, is only worth its read-cost marginal when compaction drops it. Booking
// it at 1.0x — the pre-#2794 report convention — over-credited fak's compaction 10x on every
// warm fire and was the single source of the inflated fak_share (#2798). It READS the SAME
// canonical cacheprice.ReadMultiplier the fire gate is pinned to (agent.defaultCacheReadMult,
// consulted by agent.CacheBurstBreakEvenTurns / headBurstEconomics), so the report's shed
// marginal and the gate's readMult are one symbol — they agree by construction, not by a
// swept literal.
const compactionShedMarginalMultiplier = cacheprice.ReadMultiplier

// ValuationBasis names the price basis a saved-token / $ figure was booked at, so a
// number can never be read without knowing how it was priced — the exact gap that let
// 1.0x-on-warm slip through (#2796). Every fak $ figure carries one; a renderer refuses
// to print an unlabeled fak dollar.
type ValuationBasis string

const (
	// ValuationBasisFullInput prices a shed token at the full input rate (1.0x). Honest
	// ONLY for an observed-COLD fire — one the provider did not already have cached, so
	// dropping it avoids a fresh-input billing. #2794 keeps 1.0x for this case alone.
	ValuationBasisFullInput ValuationBasis = "FULL_INPUT"
	// ValuationBasisCacheReadMarginal prices a shed token at the cache-read marginal
	// (0.1x) — correct for a WARM fire, where the dropped token would have been billed
	// as a cache_read, not fresh input. This is compactionShedMarginalMultiplier.
	ValuationBasisCacheReadMarginal ValuationBasis = "CACHE_READ_MARGINAL"
	// ValuationBasisBlendedMarginal prices a shed lump PROPORTIONALLY: the warm portion
	// the provider evidenced as cache_reads (min(shed, cache_read)) at the 0.1x read
	// marginal, the cold remainder at 1.0x full input. It is the basis whenever a session's
	// shed straddles both — i.e. 0 < cache_read < shed — which the earlier binary warm/cold
	// flip could not express, mislabeling a mixed lump as wholly warm or wholly cold
	// (#2794/#2798). See cacheprice.ShedTokenEquiv.
	ValuationBasisBlendedMarginal ValuationBasis = "BLENDED_MARGINAL"
	// ValuationBasisObservedNet is the fully-netted provider-observed value (read rebate
	// minus write premium), the provider prompt-cache row's basis.
	ValuationBasisObservedNet ValuationBasis = "OBSERVED_NET"
)

const (
	// SavingsDollarStatusBlind marks rows whose token axes are observed but whose
	// dollar axes are intentionally unpriced because no trusted base price was configured.
	SavingsDollarStatusBlind = "dollar_blind"
	savingsDollarStatusMixed = "mixed"
)

// CacheCreationTierProvenanceGatewayAttributed marks a row's CacheCreationTokensUpgraded
// split as fak's own per-turn upgrade witness (#2179) rather than a provider-reported
// value — the Anthropic usage block never splits 5m vs 1h creation tokens itself.
const CacheCreationTierProvenanceGatewayAttributed = "gateway_attributed"

// SavingsPricing is the caller-supplied base price for the model in play. DollarBlind
// keeps zero-dollar rows explicit when the live session has provider counters but no
// trusted price table, so $0 cannot be read as a priced no-savings result.
type SavingsPricing struct {
	InputPerMTokUSD  float64
	OutputPerMTokUSD float64
	Source           string
	DollarBlind      bool
}

// SavingsObservation is the live-session input that becomes one or more durable
// Track-2 rows. The provider prompt-cache and fak compaction mechanisms are emitted
// as separate rows so owner/mechanism roll-ups never blend them.
type SavingsObservation struct {
	SessionType string
	Provider    string
	Context     string

	InputTokens          uint64
	CacheReadTokens      uint64
	CacheCreationTokens  uint64
	OutputTokens         uint64
	CompactionShedTokens uint64

	// CompactionCacheReadTokens is the OBSERVED provider cache_read_input_tokens seen
	// at compaction fires (gateway AdjudicationSummary.CompactionCacheReadTokens). It is
	// the warm witness cacheprice.ShedTokenEquiv prices the shed on: the warm PORTION,
	// min(shed, this), was already provider cache_reads and prices at the 0.1x cache-read
	// marginal, while the cold remainder keeps 1.0x fresh input. Zero means observed-cold
	// (or no witness) and the whole shed keeps the full-input basis; a value >= shed makes
	// it wholly warm; anything between is a BLENDED lump. Not a claim fak preserved the
	// cache (byte-identity is) — just the provider's relayed read count, used to price and
	// label (see shedValuationBasis) the shed honestly.
	CompactionCacheReadTokens uint64

	// CacheCreationTokensUpgraded is the subset of CacheCreationTokens written while
	// the managed-cache 1h TTL-upgrade rung (--managed-cache) was active for that
	// turn (#2179). GATEWAY_ATTRIBUTED, not provider-reported: the Anthropic usage
	// block never splits 5m vs 1h creation tokens, so a caller with no attribution
	// signal (e.g. a plain dev-session transcript with no fak gateway in the loop)
	// leaves this 0, and WritePremiumUSD falls back to pricing the whole total at
	// the 5m tier, byte-identical to before this field existed.
	CacheCreationTokensUpgraded uint64

	// CompactionFired / CompactionBailed / CompactionAnchorStarved are the WITNESSED health
	// counters from AdjudicationSummary. They let the durable record distinguish a session
	// where the lever FIRED but shed nothing (anchor-starved, #1407) from one where it was
	// idle or disabled -- the gap #2039 closes. CompactionBudget is the resident-token
	// threshold (0 = lever OFF); persisted so a reader can tell enabled-but-idle from disabled.
	CompactionFired         uint64
	CompactionBailed        uint64
	CompactionAnchorStarved uint64
	CompactionBudget        int

	// BillingMode is the seat posture resolved at write time (#3664) — see
	// NormalizeBillingMode for the closed set. A producer that cannot see the billing
	// class leaves it blank, which reads as unknown and folds NOTIONAL.
	BillingMode string

	Pricing SavingsPricing
}

// SavingsRow is one durable, append-only cache-economics row — Track 2 of epic #1301,
// the per-session shape rung B (#1303) persists. Provider prompt-cache token axes are
// OBSERVED/provider-relayed; fak compaction rows carry a WITNESSED fak-authored shed-token
// axis whose dollar value is still only a projection from supplied base rates.
//
// The token axes are the four the Anthropic usage block reports (mirroring
// gateway.CacheUsage). The $ axes are priced from the session's base input/output
// $/MTok (e.g. Opus 4.8 = {5,25}) so the row stays correct as prices change
// without re-pricing here. CompactionShedTokens is the prompt tokens the
// `--compact-history-budget` (#745) compaction dropped before they were ever sent
// — a real OBSERVED saving on the input axis, valued at the base input price.
type SavingsRow struct {
	Schema      string `json:"schema"`
	Date        string `json:"date"` // YYYY-MM-DD (UTC), the bucketing key
	SessionType string `json:"session_type,omitempty"`
	Provider    string `json:"provider,omitempty"`
	Mechanism   string `json:"mechanism,omitempty"`
	Context     string `json:"context,omitempty"`
	GeneratedAt string `json:"generated_at,omitempty"`

	// Token axes (OBSERVED, provider-reported).
	InputTokens         uint64 `json:"input_tokens"`
	CacheReadTokens     uint64 `json:"cache_read_tokens"`
	CacheCreationTokens uint64 `json:"cache_creation_tokens"`
	OutputTokens        uint64 `json:"output_tokens"`

	// CacheCreationTokensUpgraded is the subset of CacheCreationTokens attributed to
	// the managed-cache 1h TTL tier; CacheCreationTierProvenance names the source of
	// that attribution ("gateway_attributed" — fak's own per-turn upgrade witness,
	// since the provider wire never splits 5m vs 1h creation tokens, #2179). Both are
	// absent/zero when the observation carried no attribution, in which case
	// WritePremiumUSD prices the whole CacheCreationTokens total at the 5m tier.
	CacheCreationTokensUpgraded uint64 `json:"cache_creation_tokens_upgraded,omitempty"`
	CacheCreationTierProvenance string `json:"cache_creation_tier_provenance,omitempty"`

	// CompactionShedTokens is the input tokens `--compact-history-budget` (#745)
	// dropped before the turn was sent — a fak-authored token saving on the spend side.
	CompactionShedTokens uint64 `json:"compaction_shed_tokens,omitempty"`

	// CompactionCacheReadTokens is the OBSERVED provider cache_read_input_tokens at the
	// compaction fires this row rolls up — the warm witness #2794/#2798 prices on. It is
	// persisted so a reader (and InferValuationBasis on a legacy row) can re-derive the
	// three-way basis via shedValuationBasis: fully warm (>= shed → CACHE_READ_MARGINAL),
	// observed-cold (0 → FULL_INPUT), or a mixed lump (0 < it < shed → BLENDED_MARGINAL,
	// warm portion at 0.1x + cold remainder at 1.0x).
	CompactionCacheReadTokens uint64 `json:"compaction_cache_read_tokens,omitempty"`

	// CompactionFired / CompactionBailed / CompactionAnchorStarved are the WITNESSED health
	// counters persisted per #2039 so a fired-but-shed-zero session leaves a row, not silence.
	// CompactionBudget is the resident-token threshold the lever fires past (0 = OFF).
	CompactionFired         uint64 `json:"compaction_fired,omitempty"`
	CompactionBailed        uint64 `json:"compaction_bailed,omitempty"`
	CompactionAnchorStarved uint64 `json:"compaction_anchor_starved,omitempty"`
	CompactionBudget        int    `json:"compaction_budget,omitempty"`

	// Saved-token-equivalents (OBSERVED), the read rebate minus the write premium,
	// in input-token units — the same quantity vcachegov.TelemetrySavingsProof
	// reports. NetSavedTokenEquiv keeps its sign: a cold-write-only session reads
	// NEGATIVE until reads repay writes (#1303 says do not floor at zero).
	SavedTokenEquiv    float64 `json:"saved_token_equiv,omitempty"`
	NetSavedTokenEquiv float64 `json:"net_saved_token_equiv,omitempty"`

	// $ duals (OBSERVED projection), priced from the base rates below. RebateUSD is
	// the read rebate, WritePremiumUSD the cache-write premium paid, SpendUSD the
	// API dollars still incurred this session, CompactionSavedUSD the value of the
	// shed tokens. NetUSD = RebateUSD + CompactionSavedUSD − WritePremiumUSD −
	// SpendUSD, the single per-session NET the report folds.
	RebateUSD          float64 `json:"rebate_usd"`
	WritePremiumUSD    float64 `json:"write_premium_usd"`
	SpendUSD           float64 `json:"spend_usd"`
	CompactionSavedUSD float64 `json:"compaction_saved_usd,omitempty"`
	NetUSD             float64 `json:"net_usd"`

	InputPerMTokUSD  float64 `json:"input_per_mtok_usd,omitempty"`
	OutputPerMTokUSD float64 `json:"output_per_mtok_usd,omitempty"`
	PricingSource    string  `json:"pricing_source,omitempty"`
	DollarStatus     string  `json:"dollar_status,omitempty"`

	// BillingMode is WHICH seat the $ duals above were billed to, stamped at write from
	// the session's resolved credential class (#3664). PricingSource says what rate the
	// row was priced AT; this says whether that price was actually charged per token.
	// Absent on every row written before the field existed and on any producer that
	// cannot see its own billing — both read as unknown and fold NOTIONAL, never
	// real-dollar (NormalizeBillingMode / RealDollarBillingMode).
	BillingMode string `json:"billing_mode,omitempty"`

	// Fidelity is the mechanism's context-faithfulness class, stamped at write from
	// Fidelity(Mechanism) so a producer can never drift from the prose ("lossless"
	// for a byte-identical provider prompt-cache hit, "bounded" for bounded-lossy
	// compaction shedding). It is additive/omitempty: readers predating it ignore it,
	// and the audit fold derives it from Mechanism for rows written before this field.
	Fidelity string `json:"fidelity,omitempty"`

	// ValuationBasis names the price basis the row's fak $ figures were booked at
	// (#2796) — FULL_INPUT for an observed-cold compaction fire, CACHE_READ_MARGINAL
	// for a warm one (the 0.1x #2794 case), OBSERVED_NET for the provider prompt-cache
	// row. A fak dollar figure without a basis is refused at render (basisOrRefuse). It
	// is omitempty for back-compat: rows written before this field are basis-inferred by
	// InferValuationBasis from the mechanism at fold time.
	ValuationBasis ValuationBasis `json:"valuation_basis,omitempty"`
}

// InferValuationBasis derives the price basis for a row written before ValuationBasis
// existed, so the fold and the render-refusal never trip on a legacy ledger. Provider
// prompt-cache rows are OBSERVED_NET; a compaction row's warm/cold/blended basis is deferred
// to shedValuationBasis — the ONE place that decision is made (#2798) — so an inferred basis
// can never drift from the basis the row was actually priced at. Non-dollar rows and unknown
// mechanisms return empty (no fak $ to label).
func (r SavingsRow) InferValuationBasis() ValuationBasis {
	if r.ValuationBasis != "" {
		return r.ValuationBasis
	}
	switch {
	case r.Mechanism == "provider_prompt_cache":
		return ValuationBasisObservedNet
	case r.Mechanism == "compaction_shed":
		return shedValuationBasis(r.CompactionShedTokens, r.CompactionCacheReadTokens)
	default:
		return ""
	}
}

// NetUSDComputed re-derives the NET from the row's own component $ fields, so a
// reader can check the stored NetUSD against its parts. It is the byte-for-byte
// identity the recompute test asserts:
//
//	NET = read rebate + compaction saving − write premium − API spend still incurred
func (r SavingsRow) NetUSDComputed() float64 {
	return r.RebateUSD + r.CompactionSavedUSD - r.WritePremiumUSD - r.SpendUSD
}

// SavingsBucket is one period's OBSERVED-$ roll-up (Track 2). It keeps the four
// component dollar accounts separate (so the NET is always re-derivable) plus the
// running cumulative NET, which is what crosses break-even.
type SavingsBucket struct {
	Period    string `json:"period"` // ISO week, e.g. "2026-W26"
	Start     string `json:"start"`  // earliest row date in the bucket (YYYY-MM-DD)
	Provider  string `json:"provider"`
	Mechanism string `json:"mechanism"`
	Sessions  int    `json:"sessions"`

	InputTokens               uint64 `json:"input_tokens"`
	CacheReadTokens           uint64 `json:"cache_read_tokens"`
	CacheCreationTokens       uint64 `json:"cache_creation_tokens"`
	OutputTokens              uint64 `json:"output_tokens"`
	CompactionShedTokens      uint64 `json:"compaction_shed_tokens"`
	CompactionCacheReadTokens uint64 `json:"compaction_cache_read_tokens,omitempty"`

	// Compaction health (#2039): fired/bailed/anchor_starved let the report
	// distinguish a lever that FIRED from one that was idle. CompactionBudget is
	// the max resident-token threshold seen across the bucket's sessions.
	CompactionFired         uint64 `json:"compaction_fired"`
	CompactionBailed        uint64 `json:"compaction_bailed"`
	CompactionAnchorStarved uint64 `json:"compaction_anchor_starved"`
	CompactionBudget        int    `json:"compaction_budget"`

	SavedTokenEquiv    float64 `json:"saved_token_equiv"`
	NetSavedTokenEquiv float64 `json:"net_saved_token_equiv"`

	RebateUSD          float64 `json:"rebate_usd"`
	WritePremiumUSD    float64 `json:"write_premium_usd"`
	SpendUSD           float64 `json:"spend_usd"`
	CompactionSavedUSD float64 `json:"compaction_saved_usd"`

	// NetUSD is this period's net: rebate + compaction − writePremium − spend.
	NetUSD float64 `json:"net_usd"`
	// CumulativeNetUSD is the running total through this period — the line that
	// crosses break-even. BrokeEven is true once it first turns non-negative.
	CumulativeNetUSD float64 `json:"cumulative_net_usd"`
	BrokeEven        bool    `json:"broke_even"`

	DollarStatus        string `json:"dollar_status,omitempty"`
	DollarBlindSessions int    `json:"dollar_blind_sessions,omitempty"`
}

// OwnerAttributionBucket is the report-level owner split, in token-equivalent
// units. Provider prompt-cache savings stay separate from fak-authored savings;
// vDSO is currently counted as avoided calls because no token axis exists for it.
type OwnerAttributionBucket struct {
	Period                        string  `json:"period"`
	ProviderPromptCacheTokenEquiv float64 `json:"provider_prompt_cache_token_equiv"`
	FakAuthoredTokenEquiv         float64 `json:"fak_authored_token_equiv"`
	// FakSharePct is fak's share of the period's total cache-value
	// token-equivalents, in percent — the "what % of the cache value is fak's?"
	// headline, pre-divided so no reader has to derive it from the two columns.
	// It is present only when the period's total is positive (see
	// FakShareOfTotalPct), so an empty or upside-down period can never render
	// as a 0% or 100% claim. The share covers ROWS RECORDED IN THE LEDGERS —
	// it says nothing about traffic neither track observed.
	FakSharePct             *float64 `json:"fak_share_pct,omitempty"`
	FakKVPrefixReusedTokens uint64   `json:"fak_kv_prefix_reused_tokens"`
	FakCompactionShedTokens uint64   `json:"fak_compaction_shed_tokens"`
	FakVDSOAvoidedCalls     uint64   `json:"fak_vdso_avoided_calls"`
}

// ComponentHealth is the per-cache-plane status row in the roll-up. It answers the
// operational question "which component is working, missing, dollar-blind, or only
// indirectly evidenced?" without making the reader infer that from three ledgers.
type ComponentHealth struct {
	Plane      string `json:"plane"`
	Component  string `json:"component"`
	Owner      string `json:"owner"`
	Fidelity   string `json:"fidelity"` // lossless | lossy | recoverable | passive
	Evidence   string `json:"evidence"` // WITNESSED | OBSERVED
	Status     string `json:"status"`   // measured | stale | insufficient | missing | dollar_blind
	Reason     string `json:"reason"`
	NextAction string `json:"next_action,omitempty"`
}

// FakShareOfTotalPct is fak's share of this period's total cache-value
// token-equivalents, in percent. ok is false when the total is zero or negative
// (nothing recorded, or a provider write premium that outweighed every saving) —
// a period with no positive total has no meaningful share, and reporting 0%
// there would read as "fak contributed nothing to a real total" when there is
// no real total to speak of.
func (b OwnerAttributionBucket) FakShareOfTotalPct() (float64, bool) {
	total := b.ProviderPromptCacheTokenEquiv + b.FakAuthoredTokenEquiv
	if total <= 0 {
		return 0, false
	}
	return 100 * b.FakAuthoredTokenEquiv / total, true
}

// NewSavingsRows converts one live session observation into the durable Track-2 row shape.
// Provider prompt-cache and fak-authored compaction are split into separate mechanisms while
// preserving the same session/date/context labels. The dollar fields are projections from the
// caller-supplied base $/MTok; with zero pricing they remain zero rather than fabricating a
// provider-specific price.
func NewSavingsRows(obs SavingsObservation, now time.Time) []SavingsRow {
	now = now.UTC()
	base := SavingsRow{
		Schema:        SavingsLedgerSchema,
		Date:          now.Format("2006-01-02"),
		SessionType:   obs.SessionType,
		Context:       obs.Context,
		GeneratedAt:   now.Format(time.RFC3339),
		PricingSource: strings.TrimSpace(obs.Pricing.Source),
		DollarStatus:  savingsDollarStatus(obs.Pricing),
	}
	// Stamp the seat posture only when the producer actually named one (#3664). A blank
	// observation stays BLANK on the wire rather than being normalized to an explicit
	// "unknown": both fold notional, but the absence honestly says "nobody looked",
	// which is exactly what every pre-#3664 row says, and it keeps the emitted JSON of
	// a producer that never learned about billing byte-identical to before.
	if strings.TrimSpace(obs.BillingMode) != "" {
		base.BillingMode = NormalizeBillingMode(obs.BillingMode)
	}
	var rows []SavingsRow
	if obs.CacheReadTokens > 0 || obs.CacheCreationTokens > 0 {
		row := base
		row.Provider = strings.TrimSpace(obs.Provider)
		row.Mechanism = "provider_prompt_cache"
		row.Fidelity = Fidelity(row.Mechanism)
		row.InputTokens = obs.InputTokens
		row.CacheReadTokens = obs.CacheReadTokens
		row.CacheCreationTokens = obs.CacheCreationTokens
		row.OutputTokens = obs.OutputTokens
		row.CacheCreationTokensUpgraded = clampUpgraded(obs.CacheCreationTokens, obs.CacheCreationTokensUpgraded)
		if row.CacheCreationTokensUpgraded > 0 {
			row.CacheCreationTierProvenance = CacheCreationTierProvenanceGatewayAttributed
		}
		row.ValuationBasis = ValuationBasisObservedNet
		row.SavedTokenEquiv = providerSavedTokenEquiv(obs.CacheReadTokens, obs.CacheCreationTokens, row.CacheCreationTokensUpgraded)
		row.NetSavedTokenEquiv = row.SavedTokenEquiv
		row.InputPerMTokUSD = obs.Pricing.InputPerMTokUSD
		row.OutputPerMTokUSD = obs.Pricing.OutputPerMTokUSD
		row.RebateUSD = perMTok(obs.Pricing.InputPerMTokUSD, float64(obs.CacheReadTokens)*(1-providerCacheReadMultiplier))
		row.WritePremiumUSD = perMTok(obs.Pricing.InputPerMTokUSD, blendedCacheWriteTokenEquiv(obs.CacheCreationTokens, row.CacheCreationTokensUpgraded)-float64(obs.CacheCreationTokens))
		row.SpendUSD = providerSpendUSD(obs, row.CacheCreationTokensUpgraded)
		row.NetUSD = row.NetUSDComputed()
		normalizeSavingsDimensions(&row)
		rows = append(rows, row)
	}
	// Emit a compaction row when the lever FIRED, BAILED, or SHED -- not only when it
	// shed tokens. The anchor-starved case (#1407/#2039) fires the lever but sheds
	// zero, and without a row here the durable record is indistinguishable from a
	// session where compaction was idle or disabled. The health fields carry the
	// diagnostic; CompactionShedTokens/CompactionSavedUSD stay 0 when nothing was shed.
	if obs.CompactionShedTokens > 0 || obs.CompactionFired > 0 || obs.CompactionBailed > 0 || obs.CompactionAnchorStarved > 0 {
		row := base
		row.Provider = "fak"
		row.Mechanism = "compaction_shed"
		row.Fidelity = Fidelity(row.Mechanism)
		row.CompactionShedTokens = obs.CompactionShedTokens
		row.CompactionCacheReadTokens = obs.CompactionCacheReadTokens
		row.CompactionFired = obs.CompactionFired
		row.CompactionBailed = obs.CompactionBailed
		row.CompactionAnchorStarved = obs.CompactionAnchorStarved
		row.CompactionBudget = obs.CompactionBudget
		// Price the shed at its honest PROPORTIONAL warm/cold blend (#2794/#2798). The warm
		// portion — min(shed, CompactionCacheReadTokens), tokens the provider would have
		// billed as cache_reads — costs 0.1x (the SAME marginal the fire gate,
		// agent.CacheBurstBreakEvenTurns, values a dropped cached token at); the cold
		// remainder, shed beyond any witnessed warm prefix, keeps the 1.0x full-input basis.
		// cacheprice.ShedTokenEquiv is the one source; ValuationBasis records which side (or
		// BLENDED) the lump landed on, so the number is never read without its price basis.
		row.ValuationBasis = shedValuationBasis(obs.CompactionShedTokens, obs.CompactionCacheReadTokens)
		row.SavedTokenEquiv = cacheprice.ShedTokenEquiv(obs.CompactionShedTokens, obs.CompactionCacheReadTokens)
		row.NetSavedTokenEquiv = row.SavedTokenEquiv
		row.InputPerMTokUSD = obs.Pricing.InputPerMTokUSD
		row.OutputPerMTokUSD = obs.Pricing.OutputPerMTokUSD
		// Price the $ from the SAME marginal token-equiv (row.SavedTokenEquiv = shed×mult),
		// not from the raw shed count, so the $ and the token-equiv can never disagree on
		// the shed token's value — the single-source-of-truth invariant of #2798.
		row.CompactionSavedUSD = perMTok(obs.Pricing.InputPerMTokUSD, row.SavedTokenEquiv)
		row.NetUSD = row.NetUSDComputed()
		normalizeSavingsDimensions(&row)
		rows = append(rows, row)
	}
	return rows
}

// shedValuationBasis names the price basis for a `shed` lump valued against `cacheRead`
// (the OBSERVED provider cache_read at the fires) via cacheprice.ShedTokenEquiv — the label
// that travels with the number so it is never read without its basis. A lump the provider
// wholly evidenced as warm (cacheRead ≥ shed) is CACHE_READ_MARGINAL; a wholly-cold lump
// (cacheRead == 0) is FULL_INPUT; anything straddling both (0 < cacheRead < shed) is the
// honest BLENDED_MARGINAL. A shed of 0 has no fak $ to label, so its basis is cosmetic and
// reported as FULL_INPUT (the neutral 1.0x). This is the ONE place the basis is decided,
// paired with the ONE place the value is (cacheprice.ShedTokenEquiv), so basis and number
// cannot disagree (#2794/#2798).
func shedValuationBasis(shed, cacheRead uint64) ValuationBasis {
	switch {
	case shed == 0 || cacheRead == 0:
		return ValuationBasisFullInput
	case cacheRead >= shed:
		return ValuationBasisCacheReadMarginal
	default:
		return ValuationBasisBlendedMarginal
	}
}

// fakAuthoredTokenEquiv is the fak-authored token-equivalent a compaction row or bucket
// represents, and the ONE place that decision is made (#2798). It returns the priced
// NetSavedTokenEquiv when the row already carries one (every row emitted since #2794 does),
// and otherwise re-prices the raw shed at its honest witnessed marginal via the shared
// cacheprice.ShedTokenEquiv blend (warm portion→0.1x, cold remainder→1.0x) — never the raw
// 1.0x that over-credited fak's compaction 10x on warm fires pre-#2794, nor the aggregate-warm
// 0.1x that under-credited a cold-dominant lump after it. A legacy row persisted before the
// priced fields existed reaches the fallback. Sharing this keeps the fleet-benefit roll-up
// (fakTokenEqFromRow) and the owner-attribution fold from disagreeing on such a row, which
// they did while each carried its own copy of the fallback.
func fakAuthoredTokenEquiv(netSavedTokenEquiv float64, shed, compactionCacheRead uint64) float64 {
	if netSavedTokenEquiv != 0 {
		return netSavedTokenEquiv
	}
	return cacheprice.ShedTokenEquiv(shed, compactionCacheRead)
}

// providerSavedTokenEquiv is the read rebate plus the write axis's saved-token-equiv
// (baseline uncached cost minus what was actually billed, blended across the 1h/5m
// tiers per creationUpgraded — see blendedCacheWriteTokenEquiv). creationUpgraded=0
// reproduces the prior all-5m convention byte-for-byte.
func providerSavedTokenEquiv(read, creation, creationUpgraded uint64) float64 {
	return float64(read)*(1-providerCacheReadMultiplier) + float64(creation) - blendedCacheWriteTokenEquiv(creation, creationUpgraded)
}

// providerSpendUSD is the API dollars still incurred this session: the uncached
// input remainder, the cache-read axis at its 0.1x rebate, and the cache-creation
// axis blended across the 1h/5m tiers per creationUpgraded (#2179), plus output.
// creationUpgraded=0 reproduces the prior all-5m convention byte-for-byte.
func providerSpendUSD(obs SavingsObservation, creationUpgraded uint64) float64 {
	inTok := float64(obs.InputTokens) +
		float64(obs.CacheReadTokens)*providerCacheReadMultiplier +
		blendedCacheWriteTokenEquiv(obs.CacheCreationTokens, creationUpgraded)
	outTok := float64(obs.OutputTokens)
	return perMTok(obs.Pricing.InputPerMTokUSD, inTok) + perMTok(obs.Pricing.OutputPerMTokUSD, outTok)
}

// blendedCacheWriteTokenEquiv is the actual billed token-equivalent for `total`
// cache-creation tokens, split across the 1h and 5m write tiers: `upgraded` (clamped
// to `total`) prices at vcachegov.WriteMult1Hour, the remainder at
// vcachegov.WriteMult5Minutes — the same convention vcachegov.ProveTelemetrySavings
// applies when a caller supplies Ephemeral1hInputTokens. upgraded=0 reproduces the
// pre-#2179 flat 5m-tier pricing byte-for-byte.
func blendedCacheWriteTokenEquiv(total, upgraded uint64) float64 {
	upgraded = clampUpgraded(total, upgraded)
	remainder := total - upgraded
	return float64(upgraded)*vcachegov.WriteMult1Hour + float64(remainder)*vcachegov.WriteMult5Minutes
}

// clampUpgraded caps upgraded at total so an inconsistent (upgraded > total) pair
// from a caller can never inflate the priced total.
func clampUpgraded(total, upgraded uint64) uint64 {
	if upgraded > total {
		return total
	}
	return upgraded
}

func perMTok(price, tokens float64) float64 {
	return price * tokens / 1_000_000
}

func savingsDollarStatus(p SavingsPricing) string {
	if p.DollarBlind || (p.InputPerMTokUSD == 0 && p.OutputPerMTokUSD == 0) {
		return SavingsDollarStatusBlind
	}
	return ""
}

// ParseSavingsLedger reads OBSERVED-$ rows (Track 2) from JSONL content. Like the
// kernel-ledger parser it skips blank and unparseable lines rather than failing,
// so a partially-written ledger still folds.
func ParseSavingsLedger(content string) []SavingsRow {
	rows := jsonlledger.Parse(content, func(r SavingsRow) bool { return r.Date != "" })
	for i := range rows {
		normalizeSavingsDimensions(&rows[i])
	}
	return rows
}

// ReadSavingsLedgerFile reads the Track-2 ledger at path. A missing file folds to
// no rows (the honest empty-Track-2 case), not an error.
func ReadSavingsLedgerFile(path string) []SavingsRow {
	return ParseSavingsLedger(string(jsonlledger.ReadTail(path, jsonlledger.DefaultActiveBytes)))
}

// AppendSavingsLine renders the JSONL line for one Track-2 row (the durable shape
// rung B appends; provided here so the writer and reader share one encoding).
func AppendSavingsLine(row SavingsRow) (string, error) {
	if row.Schema == "" {
		row.Schema = SavingsLedgerSchema
	}
	normalizeSavingsDimensions(&row)
	b, err := json.Marshal(row)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// AppendSavings appends one durable Track-2 row to the configured JSONL ledger.
func AppendSavings(ledgerPath string, row SavingsRow) error {
	line, err := AppendSavingsLine(row)
	if err != nil {
		return err
	}
	return jsonlledger.AppendBounded(ledgerPath, []byte(line), jsonlledger.DefaultActiveBytes)
}

func normalizeSavingsDimensions(row *SavingsRow) {
	if strings.TrimSpace(row.Provider) == "" {
		row.Provider = "unknown_provider"
	}
	if strings.TrimSpace(row.Mechanism) == "" {
		switch {
		case row.CacheReadTokens > 0 || row.CacheCreationTokens > 0:
			row.Mechanism = "provider_prompt_cache"
		case row.CompactionShedTokens > 0 || row.CompactionFired > 0 || row.CompactionBailed > 0 || row.CompactionAnchorStarved > 0:
			row.Mechanism = "compaction_shed"
		default:
			row.Mechanism = "unknown_mechanism"
		}
	}
	if strings.TrimSpace(row.DollarStatus) == "" &&
		row.InputPerMTokUSD == 0 &&
		row.OutputPerMTokUSD == 0 &&
		row.RebateUSD == 0 &&
		row.WritePremiumUSD == 0 &&
		row.SpendUSD == 0 &&
		row.CompactionSavedUSD == 0 &&
		(row.CacheReadTokens > 0 || row.CacheCreationTokens > 0 || row.CompactionShedTokens > 0 || row.CompactionFired > 0 || row.CompactionBailed > 0 || row.CompactionAnchorStarved > 0) {
		row.DollarStatus = SavingsDollarStatusBlind
	}
}

// foldSavings buckets Track-2 rows by ISO week, provider, and mechanism, then computes each
// bucket's NET plus the running cumulative that crosses break-even. It is pure (no clock, no
// I/O): bucketing comes from each row's own Date. Rows with an unparseable date are skipped,
// mirroring the Track-1 fold.
func foldSavings(rows []SavingsRow) []SavingsBucket {
	type agg struct {
		b     SavingsBucket
		start time.Time
	}
	byPeriod := map[string]*agg{}
	for _, row := range rows {
		normalizeSavingsDimensions(&row)
		d, err := time.Parse("2006-01-02", row.Date)
		if err != nil {
			continue
		}
		period := isoWeek(d)
		key := period + "\x00" + row.Provider + "\x00" + row.Mechanism
		a := byPeriod[key]
		if a == nil {
			a = &agg{b: SavingsBucket{Period: period, Provider: row.Provider, Mechanism: row.Mechanism}, start: d}
			byPeriod[key] = a
		}
		if d.Before(a.start) {
			a.start = d
		}
		b := &a.b
		b.Sessions++
		b.InputTokens += row.InputTokens
		b.CacheReadTokens += row.CacheReadTokens
		b.CacheCreationTokens += row.CacheCreationTokens
		b.OutputTokens += row.OutputTokens
		b.CompactionShedTokens += row.CompactionShedTokens
		b.CompactionCacheReadTokens += row.CompactionCacheReadTokens
		b.CompactionFired += row.CompactionFired
		b.CompactionBailed += row.CompactionBailed
		b.CompactionAnchorStarved += row.CompactionAnchorStarved
		if row.CompactionBudget > b.CompactionBudget {
			b.CompactionBudget = row.CompactionBudget
		}
		b.SavedTokenEquiv += row.SavedTokenEquiv
		b.NetSavedTokenEquiv += row.NetSavedTokenEquiv
		b.RebateUSD += row.RebateUSD
		b.WritePremiumUSD += row.WritePremiumUSD
		b.SpendUSD += row.SpendUSD
		b.CompactionSavedUSD += row.CompactionSavedUSD
		if row.DollarStatus == SavingsDollarStatusBlind {
			b.DollarBlindSessions++
		}
	}

	keys := sortedPeriodKeys(byPeriod)

	buckets := make([]SavingsBucket, 0, len(keys))
	var cumulative float64
	brokeEven := false
	for _, k := range keys {
		a := byPeriod[k]
		b := a.b
		b.Start = a.start.Format("2006-01-02")
		if b.DollarBlindSessions > 0 {
			if b.DollarBlindSessions == b.Sessions {
				b.DollarStatus = SavingsDollarStatusBlind
			} else {
				b.DollarStatus = savingsDollarStatusMixed
			}
		}
		b.NetUSD = b.RebateUSD + b.CompactionSavedUSD - b.WritePremiumUSD - b.SpendUSD
		if b.DollarStatus != SavingsDollarStatusBlind {
			cumulative += b.NetUSD
		}
		b.CumulativeNetUSD = cumulative
		if b.DollarStatus != SavingsDollarStatusBlind && !brokeEven && cumulative >= 0 {
			brokeEven = true
		}
		b.BrokeEven = brokeEven
		buckets = append(buckets, b)
	}
	return buckets
}

// TwoTrackReport is the #1304 P&L: Track 1 (WITNESSED kernel reuse) and Track 2
// (OBSERVED $ economics) SIDE BY SIDE, never blended, with the explicit per-period
// NET and the running cumulative that crosses break-even. The two sub-reports keep
// their own provenance self-labels; this envelope only carries the headline and the
// honesty fence that the $ NET is an OBSERVED projection.
type TwoTrackReport struct {
	Schema      string `json:"schema"`
	GeneratedAt string `json:"generated_at"`
	Since       string `json:"since,omitempty"` // the --since floor applied, if any

	Track1 Report          `json:"track1_witnessed_kernel"`
	Track2 []SavingsBucket `json:"track2_observed_usd"`

	OwnerAttribution []OwnerAttributionBucket `json:"owner_attribution"`
	FleetBenefit     FleetBenefitReport       `json:"fleet_benefit"`
	ComponentHealth  []ComponentHealth        `json:"component_health,omitempty"`

	// DevSessionBenefit is Track 3 (see devsession.go): the same provider_prompt_cache
	// economics priced over real, un-proxied Claude Code session transcripts. Set by the
	// CLI layer after folding (it requires session discovery/analysis I/O this package
	// does not perform); nil when the caller did not fold dev sessions.
	DevSessionBenefit *DevSessionBenefitReport `json:"dev_session_benefit,omitempty"`

	// LatestNetUSD / CumulativeNetUSD are the most-recent period's net and the
	// running total through it — the P&L headline. BrokeEven is whether the running
	// total has crossed zero.
	LatestNetUSD     float64 `json:"latest_net_usd"`
	CumulativeNetUSD float64 `json:"cumulative_net_usd"`
	BrokeEven        bool    `json:"broke_even"`
	DollarBlindRows  int     `json:"dollar_blind_rows,omitempty"`

	// ProjectionFence is the constant honesty self-label: the $ NET is a COST
	// PROJECTION over OBSERVED quantities, never a fak-WITNESSED claim, and tracks
	// stay side by side, never blended.
	ProjectionFence string `json:"projection_fence"`

	OK         bool   `json:"ok"`
	Verdict    string `json:"verdict"` // MEASURED | INSUFFICIENT
	Finding    string `json:"finding"`
	NextAction string `json:"next_action"`
}

// projectionFence is the #1301 / #1304 honesty fence string.
const projectionFence = "cost projection over labelled sources (OBSERVED provider cache_read/cache_creation plus WITNESSED fak-authored token axes priced from supplied base $/MTok), never blended into one cache claim; Track 1 (WITNESSED) and Track 2 (OBSERVED/projected $) stay side by side"

// FoldTwoTrack is the #1304 two-track P&L fold: it folds the WITNESSED kernel rows
// (Track 1) and the OBSERVED-$ rows (Track 2) into one report that shows both
// tracks side by side with an explicit NET line per period and the running total
// crossing break-even. It is PURE — the only time input is `now` (used to stamp
// GeneratedAt and to delegate the Track-1 fold). The two folds run independently;
// nothing is averaged or summed across the provenance boundary.
func FoldTwoTrack(track1 []cachevalueledger.Row, track2 []SavingsRow, now time.Time) TwoTrackReport {
	return FoldTwoTrackWithUsage(track1, track2, nil, now, FleetBenefitOptions{})
}

// FoldTwoTrackWithUsage is FoldTwoTrack plus the durable gateway-usage ledger join. It keeps the
// original two-track P&L intact and adds the cumulative fleet-benefit section that answers
// "how much did guard/serve usage accrue and how much spend/context did cache work save?"
func FoldTwoTrackWithUsage(track1 []cachevalueledger.Row, track2 []SavingsRow, usage []gatewayusageledger.Row, now time.Time, opts FleetBenefitOptions) TwoTrackReport {
	t1 := Fold(track1, now)
	t2 := foldSavings(track2)
	fleet := FoldFleetBenefit(track1, track2, usage, opts)

	rep := TwoTrackReport{
		Schema:           Schema,
		GeneratedAt:      now.UTC().Format(time.RFC3339),
		Track1:           t1,
		Track2:           t2,
		OwnerAttribution: foldOwnerAttribution(t1.Buckets, t2),
		FleetBenefit:     fleet,
		ComponentHealth:  foldComponentHealth(t1, t2, fleet, now),
		ProjectionFence:  projectionFence,
		OK:               true,
		Verdict:          "INSUFFICIENT",
	}
	if n := len(t2); n > 0 {
		last := t2[n-1]
		rep.LatestNetUSD = last.NetUSD
		rep.CumulativeNetUSD = last.CumulativeNetUSD
		rep.BrokeEven = last.BrokeEven
	}
	for _, b := range t2 {
		rep.DollarBlindRows += b.DollarBlindSessions
	}

	t1Measured := t1.Verdict == "MEASURED"
	t2Measured := len(t2) > 0
	t2AllDollarBlind := t2Measured && allSavingsBucketsDollarBlind(t2)
	switch {
	case t1Measured && t2AllDollarBlind:
		rep.Verdict = "MEASURED"
		rep.Finding = fmt.Sprintf("Track 1 realized reuse %.3f (%s); Track 2 has token evidence but is dollar-blind (no price configured)",
			t1.LatestReuseRatio, t1.LatestTrend)
		rep.NextAction = "set FAK_CACHEVALUE_INPUT_PER_MTOK_USD / FAK_CACHEVALUE_OUTPUT_PER_MTOK_USD or use a known provider/context default before treating Track 2 as dollars"
	case t2AllDollarBlind:
		rep.Verdict = "MEASURED"
		rep.Finding = "Track 2 has token evidence but is dollar-blind (no price configured); Track 1 has no multi-turn reuse to trend yet"
		rep.NextAction = "set FAK_CACHEVALUE_INPUT_PER_MTOK_USD / FAK_CACHEVALUE_OUTPUT_PER_MTOK_USD or use a known provider/context default before treating Track 2 as dollars"
	case t1Measured && t2Measured:
		rep.Verdict = "MEASURED"
		rep.Finding = fmt.Sprintf("Track 1 realized reuse %.3f (%s); Track 2 net $%.4f this period, cumulative $%.4f (%s)",
			t1.LatestReuseRatio, t1.LatestTrend, rep.LatestNetUSD, rep.CumulativeNetUSD, breakEvenLabel(rep.BrokeEven))
		rep.NextAction = "post the two-track P&L on a cadence (epic #1301 rung E/F)"
	case t1Measured:
		rep.Verdict = "MEASURED"
		rep.Finding = fmt.Sprintf("Track 1 realized reuse %.3f (%s); Track 2 has no OBSERVED-$ rows yet (rung B not appending)",
			t1.LatestReuseRatio, t1.LatestTrend)
		rep.NextAction = "wire the OBSERVED-$ per-session append (epic #1301 rung B, #1303) to populate Track 2"
	case t2Measured:
		rep.Verdict = "MEASURED"
		rep.Finding = fmt.Sprintf("Track 2 net $%.4f this period, cumulative $%.4f (%s); Track 1 has no multi-turn reuse to trend yet",
			rep.LatestNetUSD, rep.CumulativeNetUSD, breakEvenLabel(rep.BrokeEven))
		rep.NextAction = "accumulate multi-turn guard/serve sessions to populate Track 1"
	default:
		rep.Finding = "no WITNESSED reuse and no OBSERVED-$ rows yet; nothing to fold"
		rep.NextAction = "accumulate guard/serve/run sessions into both ledgers, then re-roll"
	}
	return rep
}

// savingsFeedStaleAfterHours is the drain-lag threshold for the savings-feed
// freshness row: the feed lands daily (cachevalue-feed.yml), so two missed
// cadences reads as stale rather than one slow run.
const savingsFeedStaleAfterHours = 48.0

func foldComponentHealth(t1 Report, t2 []SavingsBucket, fleet FleetBenefitReport, now time.Time) []ComponentHealth {
	out := []ComponentHealth{{
		Plane:     "local_kv",
		Component: "kernel_prefix_reuse",
		Owner:     "fak",
		Fidelity:  "lossless",
		Evidence:  "WITNESSED",
		Status:    "insufficient",
		Reason:    t1.Finding,
	}}
	if t1.Verdict == "MEASURED" {
		out[0].Status = "measured"
	} else {
		out[0].NextAction = t1.NextAction
	}

	provider := ComponentHealth{
		Plane:      "provider_prompt_cache",
		Component:  "provider_prompt_cache",
		Owner:      "provider",
		Fidelity:   "lossless",
		Evidence:   "OBSERVED",
		Status:     "missing",
		Reason:     "no provider prompt-cache rows in Track 2",
		NextAction: "append provider cache_read/cache_creation evidence to the Track-2 savings ledger",
	}
	providerBuckets, providerBlind := 0, 0
	for _, b := range t2 {
		if b.Mechanism != "provider_prompt_cache" {
			continue
		}
		providerBuckets++
		if b.DollarStatus == SavingsDollarStatusBlind {
			providerBlind++
		}
	}
	if providerBuckets > 0 {
		provider.Status = "measured"
		provider.Reason = fmt.Sprintf("%d provider prompt-cache bucket(s) folded from Track 2", providerBuckets)
		provider.NextAction = ""
		if providerBlind == providerBuckets {
			provider.Status = "dollar_blind"
			provider.Reason = "provider cache token evidence exists, but no trusted price was configured"
			provider.NextAction = "configure a trusted base price before treating Track-2 provider rows as dollars"
		}
	}
	out = append(out, provider)

	compaction := ComponentHealth{
		Plane:      "context_compression",
		Component:  "compaction_shed",
		Owner:      "fak",
		Fidelity:   "lossy",
		Evidence:   "WITNESSED",
		Status:     "missing",
		Reason:     "no fak-authored compaction/context-shed tokens recorded",
		NextAction: "enable or exercise fak compaction/headroom paths and append savings rows",
	}
	if fleet.ContextExtensionTokens > 0 || hasCompactionBucket(t2) {
		compaction.Status = "measured"
		compaction.Reason = fmt.Sprintf("%d context-extension token(s) attributed to fak compaction", fleet.ContextExtensionTokens)
		compaction.NextAction = ""
	}
	out = append(out, compaction)

	usage := ComponentHealth{
		Plane:      "gateway_usage",
		Component:  "guard_serve_usage_ledger",
		Owner:      "fak",
		Fidelity:   "passive",
		Evidence:   "WITNESSED",
		Status:     "missing",
		Reason:     "gateway usage ledger has no rows in this report window",
		NextAction: "run guard/serve sessions that append gateway-usage rows",
	}
	if fleet.UsageRows > 0 {
		usage.Status = "measured"
		usage.Reason = fmt.Sprintf("%d gateway usage row(s), %d exit session(s)", fleet.UsageRows, fleet.ExitSessions)
		usage.NextAction = ""
	}
	out = append(out, usage)

	// Freshness/drain-lag self-meter (#3394): the completeness rows above say which
	// plane has evidence; this row says how OLD that evidence is, so a stale savings
	// dashboard flags its own staleness instead of silently reading current.
	freshness := ComponentHealth{
		Plane:      "savings_feed",
		Component:  "feed_freshness",
		Owner:      "fak",
		Fidelity:   "passive",
		Evidence:   "WITNESSED",
		Status:     "missing",
		Reason:     "no timestamped ledger rows to age the feed against",
		NextAction: "append savings rows carrying generated_at so drain-lag is measurable",
	}
	last := fleet.SavingsLastUTC
	if last.IsZero() {
		last = fleet.LastRowUTC
	}
	if !last.IsZero() {
		hoursBehind := now.UTC().Sub(last).Hours()
		if hoursBehind > savingsFeedStaleAfterHours {
			freshness.Status = "stale"
			freshness.Reason = fmt.Sprintf("last savings row is %.1fh behind now (threshold %.0fh)", hoursBehind, savingsFeedStaleAfterHours)
			freshness.NextAction = "re-run the savings feed or append fresh session rows before reading the dashboard as current"
		} else {
			freshness.Status = "measured"
			freshness.Reason = fmt.Sprintf("last savings row is %.1fh behind now (threshold %.0fh)", hoursBehind, savingsFeedStaleAfterHours)
			freshness.NextAction = ""
		}
	}
	out = append(out, freshness)
	return out
}

func hasCompactionBucket(buckets []SavingsBucket) bool {
	for _, b := range buckets {
		if b.CompactionShedTokens > 0 || b.Provider == "fak" || strings.HasPrefix(b.Mechanism, "compaction") {
			return true
		}
	}
	return false
}

func allSavingsBucketsDollarBlind(buckets []SavingsBucket) bool {
	if len(buckets) == 0 {
		return false
	}
	for _, b := range buckets {
		if b.DollarStatus != SavingsDollarStatusBlind {
			return false
		}
	}
	return true
}

func foldOwnerAttribution(track1 []Bucket, track2 []SavingsBucket) []OwnerAttributionBucket {
	byPeriod := map[string]*OwnerAttributionBucket{}
	ensure := func(period string) *OwnerAttributionBucket {
		b := byPeriod[period]
		if b == nil {
			b = &OwnerAttributionBucket{Period: period}
			byPeriod[period] = b
		}
		return b
	}
	for _, b := range track1 {
		if b.Period == "" {
			continue
		}
		dst := ensure(b.Period)
		dst.FakKVPrefixReusedTokens += b.ReusedTokens
		dst.FakAuthoredTokenEquiv += float64(b.ReusedTokens)
	}
	for _, b := range track2 {
		if b.Period == "" {
			continue
		}
		dst := ensure(b.Period)
		switch {
		case b.Provider == "fak" || strings.HasPrefix(b.Mechanism, "compaction"):
			dst.FakCompactionShedTokens += b.CompactionShedTokens
			// A legacy bucket with no priced token-equiv is re-priced at its honest
			// witnessed marginal (warm→0.1x, cold→1.0x) rather than the raw 1.0x that
			// inflated fak_share pre-#2794 — the same rule the fleet-benefit roll-up uses,
			// via the shared fakAuthoredTokenEquiv (#2798).
			dst.FakAuthoredTokenEquiv += fakAuthoredTokenEquiv(b.NetSavedTokenEquiv, b.CompactionShedTokens, b.CompactionCacheReadTokens)
		case b.Mechanism == "provider_prompt_cache":
			if b.NetSavedTokenEquiv != 0 {
				dst.ProviderPromptCacheTokenEquiv += b.NetSavedTokenEquiv
			} else {
				dst.ProviderPromptCacheTokenEquiv += b.SavedTokenEquiv
			}
		}
	}
	keys := sortedPeriodKeys(byPeriod)
	out := make([]OwnerAttributionBucket, 0, len(keys))
	for _, k := range keys {
		b := *byPeriod[k]
		if pct, ok := b.FakShareOfTotalPct(); ok {
			b.FakSharePct = &pct
		}
		out = append(out, b)
	}
	return out
}

// fakDollarBasisOrRefuse returns the price basis for a bucket's fak dollar figure and
// whether it may be printed (#2796). A bucket whose fak $ is zero has nothing to label
// (basis "", ok true). A fak-authored bucket with a nonzero CompactionSavedUSD must carry
// an inferable basis — warm (cache_read>0 → CACHE_READ_MARGINAL) or cold (FULL_INPUT); a
// provider bucket is OBSERVED_NET. ok is false only when a nonzero fak $ has no inferable
// basis at all, which is the refusal the renderer prints instead of a bare number.
func (b SavingsBucket) fakDollarBasisOrRefuse() (ValuationBasis, bool) {
	isFak := b.Provider == "fak" || strings.HasPrefix(b.Mechanism, "compaction")
	if !isFak {
		if b.Mechanism == "provider_prompt_cache" {
			return ValuationBasisObservedNet, true
		}
		return "", true
	}
	if b.CompactionSavedUSD == 0 {
		return "", true
	}
	// A nonzero fak $ must carry an inferable basis. Only the compaction_shed mechanism has
	// one (warm→marginal, cold→full-input, mixed→blended); any other fak-authored mechanism
	// carrying a dollar figure is unpriced-by-basis and must be refused rather than printed
	// bare (#2796).
	if b.Mechanism != "compaction_shed" {
		return "", false
	}
	return shedValuationBasis(b.CompactionShedTokens, b.CompactionCacheReadTokens), true
}

func breakEvenLabel(broke bool) string {
	if broke {
		return "broke even"
	}
	return "below break-even"
}

// RenderTwoTrack produces a compact, deterministic terminal P&L: Track 1 on top
// (delegating to the Track-1 Render), then Track 2's per-period NET table with the
// running cumulative and the explicit break-even crossing. The two tracks are
// printed under separate, provenance-labelled headers so a reader can never mistake
// the OBSERVED $ projection for the WITNESSED reuse.
func RenderTwoTrack(r TwoTrackReport) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "cache-value P&L (two tracks, side by side) — %s\n", r.Verdict)
	if r.Finding != "" {
		fmt.Fprintf(&sb, "  %s\n", r.Finding)
	}
	fmt.Fprintf(&sb, "  fence: %s\n\n", r.ProjectionFence)

	sb.WriteString(Render(r.Track1))

	fmt.Fprintf(&sb, "\nTrack 2 (OBSERVED $, provider-relayed cost projection)\n")
	if len(r.Track2) == 0 {
		fmt.Fprintf(&sb, "  no OBSERVED-$ rows yet (Track-2 ledger empty)\n")
	} else {
		if r.DollarBlindRows > 0 {
			fmt.Fprintf(&sb, "  pricing: %d row(s) dollar-blind (no price configured); zero dollar fields are placeholders, not priced savings\n",
				r.DollarBlindRows)
		}
		fmt.Fprintf(&sb, "  %-9s  %-16s  %-23s  %5s  %10s  %10s  %10s  %10s  %12s  %-13s  %s\n",
			"week", "provider", "mechanism", "sess", "rebate$", "compact$", "writeprem$", "spend$", "net$", "pricing", "cumulative$ (break-even)")
		for _, b := range r.Track2 {
			be := ""
			if b.BrokeEven {
				be = "  >= break-even"
			}
			pricing := b.DollarStatus
			if pricing == "" {
				pricing = "priced"
			}
			// A fak $ figure must never render without its price basis (#2796). Stamp the
			// basis into the pricing cell for fak buckets; refuse (mark, don't print a bare
			// number) when a nonzero fak $ has no inferable basis.
			if basis, ok := b.fakDollarBasisOrRefuse(); !ok {
				pricing = "REFUSED:no-basis"
			} else if basis != "" {
				pricing = pricing + ":" + string(basis)
			}
			fmt.Fprintf(&sb, "  %-9s  %-16s  %-23s  %5d  %10.4f  %10.4f  %10.4f  %10.4f  %12.4f  %-13s  %12.4f%s\n",
				b.Period, b.Provider, b.Mechanism, b.Sessions, b.RebateUSD, b.CompactionSavedUSD, b.WritePremiumUSD, b.SpendUSD,
				b.NetUSD, pricing, b.CumulativeNetUSD, be)
		}
	}
	// Compaction lever health (#2039): surface the fire/starve/shed trend so an inert
	// lever (fired>0, shed=0) is distinguishable from an idle one across the folded ledger.
	if compactionBuckets := compactionLeverBuckets(r.Track2); len(compactionBuckets) > 0 {
		fmt.Fprintf(&sb, "\nCompaction lever health (fire/starve/shed trend)\n")
		fmt.Fprintf(&sb, "  %-9s  %5s  %6s  %6s  %8s  %10s  %8s\n",
			"week", "sess", "fired", "bailed", "starved", "shed_tok", "budget")
		for _, b := range compactionBuckets {
			fmt.Fprintf(&sb, "  %-9s  %5d  %6d  %6d  %8d  %10d  %8d\n", b.leverCells()...)
		}
	}
	sb.WriteString(RenderComponentHealth(r.ComponentHealth))
	fmt.Fprintf(&sb, "\nOwner attribution (token-equiv; provider prompt-cache vs fak-authored)\n")
	if len(r.OwnerAttribution) == 0 {
		fmt.Fprintf(&sb, "  no owner-attribution rows yet\n")
	} else {
		fmt.Fprintf(&sb, "  fak_share = fak_teq / (provider_teq + fak_teq), over rows RECORDED in the two ledgers; \"-\" when the period total is not positive\n")
		fmt.Fprintf(&sb, "  %-9s  %13s  %10s  %10s  %10s  %11s  %s\n",
			"week", "provider_teq", "fak_teq", "fak_share", "kv_tok", "compact_tok", "vdso_calls")
		for _, b := range r.OwnerAttribution {
			share := "-"
			if pct, ok := b.FakShareOfTotalPct(); ok {
				share = fmt.Sprintf("%.4f%%", pct)
			}
			fmt.Fprintf(&sb, "  %-9s  %13.0f  %10.0f  %10s  %10d  %11d  %d\n",
				b.Period, b.ProviderPromptCacheTokenEquiv, b.FakAuthoredTokenEquiv, share,
				b.FakKVPrefixReusedTokens, b.FakCompactionShedTokens, b.FakVDSOAvoidedCalls)
		}
	}
	sb.WriteString(RenderFleetBenefit(r.FleetBenefit))
	if r.DevSessionBenefit != nil {
		sb.WriteString(RenderDevSessionBenefit(*r.DevSessionBenefit))
	}
	return sb.String()
}

// RenderComponentHealth renders the per-plane health rows in a compact table. It is
// separate from RenderFleetBenefit so callers can place it before or after owner
// attribution without recomputing the report.
func RenderComponentHealth(rows []ComponentHealth) string {
	if len(rows) == 0 {
		return ""
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "\nComponent health (cache planes and evidence)\n")
	fmt.Fprintf(&sb, "  %-23s %-24s %-8s %-11s %-9s %-13s %s\n",
		"plane", "component", "owner", "fidelity", "evidence", "status", "reason")
	for _, h := range rows {
		fmt.Fprintf(&sb, "  %-23s %-24s %-8s %-11s %-9s %-13s %s\n",
			h.Plane, h.Component, h.Owner, h.Fidelity, h.Evidence, h.Status, h.Reason)
	}
	return sb.String()
}
