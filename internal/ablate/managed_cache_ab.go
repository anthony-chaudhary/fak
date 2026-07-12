package ablate

// Managed-cache posture A/B (#3631): the WHOLE-posture ACTIVE vs PASSIVE end-to-end net-$
// A/B on ONE frozen replay. Where compaction_ab.go (#2805), anchor_ab.go (#2809) and
// ttl1h_idle_ab.go (#3629) each isolate ONE lever, this arm A/Bs the FULL managed-cache
// posture — anchor + 1h TTL upgrade + prune, all ON (ACTIVE) vs all OFF (PASSIVE) — over one
// frozen workload, and reports the OBSERVED net-$ delta (rebate vs write-premium) for the
// WHOLE posture, not per-lever. Per-lever attribution stays with the sibling arms (#3631
// out-of-scope); this arm answers the single program claim "does managed-cache ACTIVE save
// net $ over PASSIVE end to end?" that no arm had A/B'd whole.
//
// VIA THE TRACK-2 AUDIT FOLD. Each posture arm's OBSERVED provider counters are built into a
// cachevaluereport.SavingsObservation and folded through the SAME Track-2 audit
// (cachevaluereport.FoldAudit) the durable savings ledger reconciles — so this arm's per-arm
// NetUSD / rebate / write-premium cannot disagree with the report side by CONSTRUCTION (the
// reconciliation witness). The 1h upgrade rides the ACTIVE arm as its gateway-attributed
// CacheCreationTokensUpgraded split (priced at the 2.0x write tier); the PASSIVE arm writes
// all creation at the default 5m tier. This arm re-prices NOTHING itself — it reuses the one
// Track-2 pricing so the whole-posture number and the ledger number are one accounting.
//
// IDENTICAL-WORKLOAD GUARD. Both arms replay the SAME frozen workload, so one WorkloadHash
// binds them — the two-arm form of ablate.Report.Validate's single-workload-hash guard. A
// caller supplies that hash (the frozen trace body's sha256, as defer_ab.go does); matching
// the two arms to the same workload is the caller's contract, a mismatch invalidates the
// delta, not this arithmetic.
//
// HONEST NEGATIVE DELTA. ACTIVE bundles a 2.0x 1h write premium and a prune burst; on a
// workload with little downstream reuse those can outweigh the rebate, so ACTIVE can COST
// money vs PASSIVE. The report NEVER launders that into a win: NetDeltaUSD keeps its sign and
// Verdict states NET-NEGATIVE when ACTIVE loses (#3631 done condition: "a negative delta is
// surfaced honestly").
//
// Generation posture (gen/next, #2783): a deterministic ($0, no model, no GPU) valuation over
// caller-supplied observed per-arm counters; it does not itself run the replay or ask a
// provider for a hit. Promotion evidence = a live capture of one workload under both postures
// whose ACTIVE-minus-PASSIVE net-$ clears zero. Demotion / retirement evidence = an ACTIVE arm
// whose net-$ does NOT beat PASSIVE on real traffic — the posture is the pure loss the claim
// warned against. Invalidating assumption = the two arms must be the SAME frozen workload (the
// WorkloadHash binds them) and the base $/MTok must be trusted (a dollar-blind price yields no
// net-$ and is refused).

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// The two posture arms and the sweep-row id for the ACTIVE-vs-PASSIVE net-$ delta. ACTIVE is
// named first — it is the treatment (the managed-cache posture the claim credits) — so a
// positive delta reads as "ACTIVE saved this much end to end vs PASSIVE", mirroring the
// treatment-first convention of compaction_on_vs_off and anchor_head_vs_firstbp.
const (
	// ManagedCacheArmActive is the full managed-cache posture: anchor + 1h TTL upgrade + prune.
	ManagedCacheArmActive = "managed_cache_active"
	// ManagedCacheArmPassive is the do-nothing posture: default 5m tier, no anchor, no prune.
	ManagedCacheArmPassive = "managed_cache_passive"
	// ManagedCacheABArmID is the sweep-row id for the whole-posture ACTIVE-vs-PASSIVE net-$ delta.
	ManagedCacheABArmID = "managed_cache_active_vs_passive"
)

// ManagedCacheReplay is ONE frozen workload replayed under BOTH managed-cache postures. Only
// the four billable token axes of each arm's gateway.CacheUsage are read; the 1h/5m write-tier
// split comes from ActiveUpgradedCreationTokens (the gateway-attributed upgrade witness), NOT
// the CacheUsage.WriteTTL field, so a caller states the upgrade explicitly and this arm never
// infers a tier it did not observe.
type ManagedCacheReplay struct {
	// WorkloadHash is the identical-workload anchor both arms bind to (the frozen trace body's
	// sha256, as defer_ab.go computes). Required: a whole-posture delta over two DIFFERENT
	// workloads is not apples-to-apples.
	WorkloadHash string

	// Active is the OBSERVED provider cache usage when the FULL managed-cache posture
	// (anchor + 1h TTL upgrade + prune) is ACTIVE for the replay.
	Active gateway.CacheUsage
	// ActiveUpgradedCreationTokens is the subset of Active.CacheCreationTokens written at the
	// 1h tier by the upgrade rung (gateway-attributed) — priced at the 2.0x write tier. Clamped
	// to Active.CacheCreationTokens by the Track-2 row builder, so an over-count cannot inflate
	// the write premium.
	ActiveUpgradedCreationTokens int

	// Passive is the OBSERVED provider cache usage under the PASSIVE posture (default 5m tier,
	// no anchor, no prune) on the SAME frozen workload. Its creation is priced wholly at 5m.
	Passive gateway.CacheUsage
}

// ManagedCacheArm is one posture arm's net-$ read straight off the Track-2 audit fold's
// cumulative account, so the arm's dollars are the ledger's dollars by construction. NetUSD =
// RebateUSD − WritePremiumUSD − SpendUSD (the arm has no compaction rows); rebate − write-
// premium equals counterfactual − spend, the audit's core reconciliation identity.
type ManagedCacheArm struct {
	ArmID               string  `json:"arm_id"`
	NetUSD              float64 `json:"net_usd"`
	RebateUSD           float64 `json:"rebate_usd"`
	WritePremiumUSD     float64 `json:"write_premium_usd"`
	SpendUSD            float64 `json:"spend_usd"`
	CounterfactualUSD   float64 `json:"counterfactual_usd"`
	CacheReadTokens     uint64  `json:"cache_read_tokens"`
	CacheCreationTokens uint64  `json:"cache_creation_tokens"`
}

// ManagedCacheABReport is the two-arm artifact: the frozen workload's hash, the ACTIVE and
// PASSIVE arms with their net-$ (rebate vs write-premium), and the ACTIVE-minus-PASSIVE net-$
// delta for the whole posture. Its JSON() is the report-artifact witness; ActiveSavesNet /
// Verdict are the worded read of the #3631 done condition, negative delta and all.
type ManagedCacheABReport struct {
	ArmID         string          `json:"arm_id"`
	WorkloadHash  string          `json:"workload_hash"`
	Active        ManagedCacheArm `json:"active"`
	Passive       ManagedCacheArm `json:"passive"`
	NetDeltaUSD   float64         `json:"net_delta_usd"` // Active.NetUSD − Passive.NetUSD
	PricingSource string          `json:"pricing_source,omitempty"`
	Caveat        string          `json:"caveat,omitempty"`
}

// ActiveSavesNet is the mechanical form of the #3631 done condition: the full managed-cache
// posture is net-beneficial end to end. True iff ACTIVE's net-$ strictly beats PASSIVE's.
func (r ManagedCacheABReport) ActiveSavesNet() bool { return r.NetDeltaUSD > 0 }

// Verdict renders the #3631 acceptance IN WORDS — a one-line net-dollar verdict on whether the
// managed-cache posture is net-beneficial end to end, with the three outcomes exhaustive and
// never richer than the delta behind them:
//
//   - positive delta → ACTIVE IS net-beneficial vs PASSIVE (the claim holds on this workload).
//   - negative delta → ACTIVE is NET-NEGATIVE vs PASSIVE: the 1h write premium + prune burst
//     outweighed the rebate, so the posture cost more than doing nothing — surfaced with its
//     sign, never floored to a win.
//   - zero delta → net-NEUTRAL: no measurable end-to-end difference on this workload.
func (r ManagedCacheABReport) Verdict() string {
	label := "is net-NEUTRAL vs PASSIVE (no measurable end-to-end net-$ difference)"
	switch {
	case r.NetDeltaUSD > 0:
		label = "IS net-beneficial vs PASSIVE"
	case r.NetDeltaUSD < 0:
		label = "is NET-NEGATIVE vs PASSIVE (the posture cost more than PASSIVE on this workload)"
	}
	return fmt.Sprintf("managed-cache ACTIVE %s: net $%+.4f end-to-end (ACTIVE net $%+.4f − PASSIVE net $%+.4f) over workload %s",
		label, r.NetDeltaUSD, r.Active.NetUSD, r.Passive.NetUSD, shortHash(r.WorkloadHash))
}

// SweepRow renders the human one-liner the witness calls for: the two posture net-$ figures,
// their delta, and each arm's rebate/write-premium split so a reader sees WHY the delta landed.
func (r ManagedCacheABReport) SweepRow() string {
	return fmt.Sprintf("%s: ACTIVE net $%+.4f vs PASSIVE net $%+.4f → Δ $%+.4f (rebate/write-prem ACTIVE $%.4f/$%.4f, PASSIVE $%.4f/$%.4f) over workload %s",
		r.ArmID, r.Active.NetUSD, r.Passive.NetUSD, r.NetDeltaUSD,
		r.Active.RebateUSD, r.Active.WritePremiumUSD, r.Passive.RebateUSD, r.Passive.WritePremiumUSD, shortHash(r.WorkloadHash))
}

// JSON renders the report as canonical indented JSON terminated by a newline — the ablate
// report-artifact witness (the two posture arms and their net-$) #3631 names.
func (r ManagedCacheABReport) JSON() []byte {
	b, _ := json.MarshalIndent(r, "", "  ")
	return append(b, '\n')
}

// ManagedCacheABSweep folds ONE frozen workload's two posture arms into the whole-posture net-$
// A/B. It prices each arm through the Track-2 audit fold (cachevaluereport.FoldAudit) and
// differences the two cumulative NETs. It fails closed on:
//
//   - a missing WorkloadHash — the two arms would not be pinned to one frozen workload, so the
//     delta would not be apples-to-apples;
//   - a dollar-blind base price — the whole-posture delta cannot be a fabricated $0; a net-$
//     A/B needs a trusted $/MTok;
//   - both arms empty, or a single arm with no billable cache activity (input/output only is
//     not a cache posture measurement).
func ManagedCacheABSweep(replay ManagedCacheReplay, pricing cachevaluereport.SavingsPricing, now time.Time) (ManagedCacheABReport, error) {
	if strings.TrimSpace(replay.WorkloadHash) == "" {
		return ManagedCacheABReport{}, errors.New("ablate: managed-cache A/B needs a WorkloadHash binding both posture arms to one frozen workload")
	}
	if pricing.DollarBlind || (pricing.InputPerMTokUSD == 0 && pricing.OutputPerMTokUSD == 0) {
		return ManagedCacheABReport{}, errors.New("ablate: managed-cache A/B needs a trusted base $/MTok to report net-$ (got dollar-blind pricing); the whole-posture delta cannot be a fabricated $0")
	}
	if replay.Active == (gateway.CacheUsage{}) && replay.Passive == (gateway.CacheUsage{}) {
		return ManagedCacheABReport{}, errors.New("ablate: managed-cache A/B needs token activity on at least one posture arm")
	}

	active, err := managedCacheArm(ManagedCacheArmActive, replay.Active, replay.ActiveUpgradedCreationTokens, pricing, now)
	if err != nil {
		return ManagedCacheABReport{}, fmt.Errorf("ablate: managed-cache ACTIVE arm: %w", err)
	}
	passive, err := managedCacheArm(ManagedCacheArmPassive, replay.Passive, 0, pricing, now)
	if err != nil {
		return ManagedCacheABReport{}, fmt.Errorf("ablate: managed-cache PASSIVE arm: %w", err)
	}

	return ManagedCacheABReport{
		ArmID:         ManagedCacheABArmID,
		WorkloadHash:  replay.WorkloadHash,
		Active:        active,
		Passive:       passive,
		NetDeltaUSD:   active.NetUSD - passive.NetUSD,
		PricingSource: strings.TrimSpace(pricing.Source),
		Caveat:        managedCacheABCaveat(),
	}, nil
}

// managedCacheArm prices ONE posture arm's observed counters through the Track-2 audit fold:
// it builds a SavingsObservation, turns it into the durable Track-2 provider row, and reads the
// net-$ (and its rebate/write-premium/counterfactual components) off FoldAudit's cumulative
// account. upgradedCreation is the arm's 1h-tier creation subset (0 for PASSIVE). It fails
// closed when the arm carries no cache activity — an input/output-only arm is not a cache
// posture measurement, and a fabricated zero would read as a measured no-op.
func managedCacheArm(armID string, u gateway.CacheUsage, upgradedCreation int, pricing cachevaluereport.SavingsPricing, now time.Time) (ManagedCacheArm, error) {
	obs := cachevaluereport.SavingsObservation{
		SessionType:                 "ablate",
		Provider:                    "anthropic",
		Context:                     strings.TrimSpace(pricing.Source),
		InputTokens:                 clampNonNeg(u.InputTokens),
		CacheReadTokens:             clampNonNeg(u.CacheReadTokens),
		CacheCreationTokens:         clampNonNeg(u.CacheCreationTokens),
		OutputTokens:                clampNonNeg(u.OutputTokens),
		CacheCreationTokensUpgraded: clampNonNeg(upgradedCreation),
		Pricing:                     pricing,
	}
	rows := cachevaluereport.NewSavingsRows(obs, now)
	if len(rows) == 0 {
		return ManagedCacheArm{}, fmt.Errorf("%s has no billable cache activity to fold (input/output only is not a cache posture measurement)", armID)
	}
	c := cachevaluereport.FoldAudit(rows, now).Cumulative
	return ManagedCacheArm{
		ArmID:               armID,
		NetUSD:              c.NetUSD,
		RebateUSD:           c.RebateUSD,
		WritePremiumUSD:     c.WritePremiumUSD,
		SpendUSD:            c.SpendUSD,
		CounterfactualUSD:   c.CounterfactualUSD,
		CacheReadTokens:     c.CacheReadTokens,
		CacheCreationTokens: c.CacheCreationToks,
	}, nil
}

// clampNonNeg maps a signed token count to its unsigned Track-2 axis, flooring a nonsensical
// negative at 0 so a bad counter can never wrap into a huge unsigned value.
func clampNonNeg(n int) uint64 {
	if n < 0 {
		return 0
	}
	return uint64(n)
}

// managedCacheABCaveat names the observational limits so a reader never mistakes this projection
// for a live provider hit or over-reads the whole-posture delta as a per-lever attribution.
func managedCacheABCaveat() string {
	return "OBSERVED per-arm provider counters priced through the Track-2 audit fold (cachevaluereport.FoldAudit) from a supplied base $/MTok — a cost PROJECTION, not a fak-WITNESSED claim; the two arms bundle anchor + 1h TTL upgrade + prune (per-lever attribution is the sibling ablations' job) and must be the SAME frozen workload (the WorkloadHash binds them). A negative Δ is surfaced with its sign, never floored to a win."
}
