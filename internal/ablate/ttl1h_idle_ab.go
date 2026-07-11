package ablate

// 1h-TTL upgrade ON/OFF A/B over a >5m idle-gap session (#3629). Sibling to compaction_ab.go
// (#2805), context_extension_ab.go (#2808) and anchor_observed.go (#2809): where the frozen-trace
// sweep in ablate.go flips FeatureTTL1H as one env-gated arm among many, this file adds the FOCUSED
// two-arm rung the issue asks for — a session replayed with an INJECTED >5m idle gap under the 1h
// upgrade ON vs OFF, reporting the OBSERVED cache-read fraction on the post-idle turn for each arm.
//
// WHY THE IDLE GAP IS THE WHOLE POINT. The provider's default cache tier lives 5 MINUTES. A session
// that idles PAST 5m loses that tier, so its next turn re-enters the SAME prefix as a COLD cache
// WRITE (cache_creation, 1.25x/2.0x) instead of a warm READ (cache_read, 0.1x). The 1h upgrade
// (FAK_ABLATE_TTL_1H / Config.CacheTTL1H, wired at gateway.go maybeUpgradeAnthropicCacheTTL1H) exists
// for exactly this: extend the stable-head breakpoint's retention to one hour so a >5m idle survives
// and the re-entry lands on a READ. The 1h tier costs 2.0x to WRITE, so if the post-idle read does
// not actually land the upgrade is a pure loss — and nobody had A/B'd it on a deliberately-idle
// workload. This arm is that measurement: the two arms differ ONLY in the retention tier, and the
// report reads the realized cache-read fraction straight off each post-idle turn.
//
// OBSERVED, not modeled. Each arm's post-idle turn is the caller-supplied REAL capture
// (gateway.CacheUsage, the same accounting compaction_ab.go prices), exactly as
// ContextExtensionFire.WarmShedTokens is an observed warm witness. This core decides NO survival and
// prices NO tokens: it computes the cache-read fraction (cache_read / prompt tokens) the capture
// witnessed and differences the two arms. Whether the 1h prefix truly survived the gap is a fact of
// the capture, not an assumption of this arithmetic.
//
// IDENTICAL-WORKLOAD GUARD. Both arms replay the SAME session, so the post-idle turn's total prompt
// tokens (input + read + creation) must be identical across arms — only the read/creation SPLIT may
// differ (survived-warm vs re-primed-cold). A mismatch means the arms re-entered DIFFERENT prefixes,
// so the fractions are not apples-to-apples; this fails closed, the two-arm form of ablate.Report.
// Validate's single-workload-hash guard.
//
// Generation posture (gen/now, #2783): a deterministic ($0, no model, no GPU, no clock beyond the
// caller-injected gap) valuation over caller-supplied observed post-idle turns. It does not itself
// run the idle replay or ask a provider for a hit; the collect-and-render shell lives outside this
// pure core (mirroring the compaction_ab.go / anchor_observed.go pure-core convention), and the
// JSON() artifact feeds `cachevalue status`.
//   - Promotion evidence: a live capture of a real >5m-idle session under both TTL tiers whose ON-arm
//     post-idle read fraction materially clears the OFF arm (UpgradeRealizesRead) — the realized read
//     actually landing at the 1h tier, not merely the write premium being paid.
//   - Demotion / retirement evidence: an ON arm whose post-idle read fraction does NOT clear the OFF
//     arm on real >5m-idle traffic — the 2.0x write bought no surviving read, so the upgrade is the
//     pure loss the issue warns of and the lever should be retired on that workload.
//   - Invalidating assumption (named on the report Caveat): the injected gap must be strictly >5m
//     (else the 5m tier itself survives and there is nothing to A/B) and within the 1h tier's own
//     retention (a gap >=1h can expire the ON arm too — its own limit, not a regression); and the two
//     captures must come from the SAME re-entered prefix, which the identical-workload guard enforces.
//
// Out of scope (per #3629): redefining the 0.1x/1.25x/2.0x multipliers (owned by cacheprice) and the
// billing-mode split (#2813). This arm reads a fraction off observed turns; it prices nothing.

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// The two arm labels and the sweep-row id for the 1h-upgrade idle A/B.
const (
	// TTL1HIdleArmOn is the upgrade-ON arm: the stable-head breakpoint written under the 1h tier,
	// so a >5m (sub-1h) idle survives and the post-idle re-entry lands on a cache READ.
	TTL1HIdleArmOn = "ttl_1h_on"
	// TTL1HIdleArmOff is the upgrade-OFF arm: the default 5m tier, which expires across a >5m idle,
	// so the post-idle re-entry re-primes the prefix as a COLD cache write.
	TTL1HIdleArmOff = "ttl_1h_off"
	// TTL1HIdleABArmID is the sweep-row id for the post-idle cache-read-fraction ON-vs-OFF split.
	TTL1HIdleABArmID = "ttl_1h_on_vs_off_idle"
)

// MaterialReadFractionGap is the minimum ON-minus-OFF post-idle cache-read-fraction the report calls
// "materially higher" (the #3629 done condition). Set at 0.5: at least half the re-entered post-idle
// prompt lands as a warm READ on the 1h arm that the 5m arm pays as a COLD write. A realistic re-entry
// (a large surviving prefix over a small fresh user turn) clears this comfortably; the bar rejects a
// marginal split as not yet a realized-read win.
const MaterialReadFractionGap = 0.5

// idleFiveMinutes / idleOneHour are the two tier retention boundaries the injected gap is checked
// against: below 5m even the default tier survives (nothing to A/B); at/above 1h the upgraded tier
// can expire too (its own limit). They mirror gateway.CacheTTL5m / gateway.CacheTTL1h as durations.
const (
	idleFiveMinutes = 5 * time.Minute
	idleOneHour     = time.Hour
)

// idlePromptTokens is the total prompt-side billed tokens on a post-idle turn: uncached input plus
// the re-entered prefix, whether that prefix landed warm (cache_read) or cold (cache_creation). It is
// the identical-workload key (same session ⇒ same prompt tokens both arms) and the read-fraction
// denominator, matching gateway.CachePricing's promptTokens = input + read + creation.
func idlePromptTokens(u gateway.CacheUsage) int64 {
	return int64(u.InputTokens + u.CacheReadTokens + u.CacheCreationTokens)
}

// idleCacheReadFraction is the OBSERVED cache-read fraction on a post-idle turn: the prefix tokens the
// provider served WARM (cache_read, 0.1x) over all prompt tokens. 0 when the turn billed no prompt
// tokens (guarded away by the builder). This is the realized-read signal the 1h upgrade is bought for.
func idleCacheReadFraction(u gateway.CacheUsage) float64 {
	prompt := idlePromptTokens(u)
	if prompt <= 0 {
		return 0
	}
	return float64(u.CacheReadTokens) / float64(prompt)
}

// IdleArm is one arm of the split: the retention tier it ran under, the OBSERVED post-idle turn, and
// the cache-read fraction read off that turn. The Turn is the caller's real capture (this core adds
// no synthetic tokens); CacheReadFraction and PromptTokens are derived so a JSON consumer need not
// recompute them.
type IdleArm struct {
	ArmID             string             `json:"arm_id"`
	TTL               string             `json:"ttl"` // gateway.CacheTTL1h ("1h") for ON, gateway.CacheTTL5m ("5m") for OFF
	Turn              gateway.CacheUsage `json:"post_idle_turn"`
	CacheReadFraction float64            `json:"cache_read_fraction"`
	PromptTokens      int64              `json:"prompt_tokens"`
}

// TTL1HIdleABReport is the two-arm artifact: the injected idle gap, the ON (1h) and OFF (5m) arms with
// their observed post-idle cache-read fractions, and the fraction delta between them. Its JSON() is the
// report artifact witness; UpgradeRealizesRead is the mechanical form of the #3629 done condition.
type TTL1HIdleABReport struct {
	ArmID             string  `json:"arm_id"`
	IdleGapSeconds    float64 `json:"idle_gap_seconds"`
	On                IdleArm `json:"on"`
	Off               IdleArm `json:"off"`
	ReadFractionDelta float64 `json:"read_fraction_delta"` // On.CacheReadFraction − Off.CacheReadFraction
	PromptTokens      int64   `json:"prompt_tokens"`       // the identical post-idle workload both arms re-entered
	Caveat            string  `json:"caveat,omitempty"`
}

// TTL1HIdleABSweep folds the two OBSERVED post-idle turns into the two-arm report. onTurn is the
// upgrade-ON (1h) re-entry, offTurn the upgrade-OFF (5m) re-entry, both on the SAME session across the
// SAME injected idle gap. It fails closed on:
//
//   - an idle gap that is not strictly >5m — the premise is a gap that expires the default tier; a
//     ≤5m gap survives on both arms, so there is no realized-read difference to measure;
//   - a post-idle turn with no billable prompt tokens on either arm — that is not a re-entry, and a
//     fabricated zero-token fraction would read as a measured no-op;
//   - arms whose post-idle prompt tokens differ — they re-entered DIFFERENT prefixes, so the two
//     read fractions are not apples-to-apples (the identical-workload guard).
//
// A gap at/above the 1h tier's own retention is allowed but flagged on the Caveat: the ON arm may
// itself have expired, so a low ON read fraction there is the tier's limit, not a regression.
func TTL1HIdleABSweep(idleGap time.Duration, onTurn, offTurn gateway.CacheUsage) (TTL1HIdleABReport, error) {
	if idleGap <= idleFiveMinutes {
		return TTL1HIdleABReport{}, fmt.Errorf("ablate: 1h-TTL idle A/B needs an injected gap strictly >5m (got %s); a ≤5m gap does not expire the default tier, so there is nothing to A/B", idleGap)
	}
	onPrompt := idlePromptTokens(onTurn)
	offPrompt := idlePromptTokens(offTurn)
	if onPrompt <= 0 || offPrompt <= 0 {
		return TTL1HIdleABReport{}, errors.New("ablate: 1h-TTL idle A/B needs a post-idle turn with billable prompt tokens on BOTH arms — a zero-token turn is not a re-entry")
	}
	if onPrompt != offPrompt {
		return TTL1HIdleABReport{}, fmt.Errorf("ablate: refusing to compare arms that re-entered different post-idle prefixes (on=%d prompt tokens, off=%d); the 1h/5m split must be measured over the SAME session so only the read/creation split differs", onPrompt, offPrompt)
	}

	onArm := IdleArm{
		ArmID:             TTL1HIdleArmOn,
		TTL:               string(gateway.CacheTTL1h),
		Turn:              onTurn,
		CacheReadFraction: idleCacheReadFraction(onTurn),
		PromptTokens:      onPrompt,
	}
	offArm := IdleArm{
		ArmID:             TTL1HIdleArmOff,
		TTL:               string(gateway.CacheTTL5m),
		Turn:              offTurn,
		CacheReadFraction: idleCacheReadFraction(offTurn),
		PromptTokens:      offPrompt,
	}
	return TTL1HIdleABReport{
		ArmID:             TTL1HIdleABArmID,
		IdleGapSeconds:    idleGap.Seconds(),
		On:                onArm,
		Off:               offArm,
		ReadFractionDelta: onArm.CacheReadFraction - offArm.CacheReadFraction,
		PromptTokens:      onPrompt,
		Caveat:            idleABCaveat(idleGap),
	}, nil
}

// UpgradeRealizesRead is the mechanical form of the #3629 done condition: the post-idle cache-read
// fraction is MATERIALLY higher with the upgrade ON. True iff the ON-minus-OFF fraction delta clears
// MaterialReadFractionGap — i.e. the realized read actually landed at the 1h tier, not merely that the
// 2.0x write premium was paid. A report where the ON arm did not clear the bar honestly reads false:
// the upgrade bought no surviving read on this capture.
func (r TTL1HIdleABReport) UpgradeRealizesRead() bool {
	return r.ReadFractionDelta >= MaterialReadFractionGap
}

// SweepRow renders the human one-liner the witness calls for: the two post-idle read fractions, their
// delta, whether the upgrade realized the read, and the injected gap they were measured across.
func (r TTL1HIdleABReport) SweepRow() string {
	verdict := "NOT materially higher (upgrade did not realize the post-idle read)"
	if r.UpgradeRealizesRead() {
		verdict = "materially higher (upgrade realizes the post-idle read)"
	}
	return fmt.Sprintf("%s: post-idle cache-read fraction on=%.3f (1h) vs off=%.3f (5m), Δ=%+.3f — %s across a %s idle gap over %d prompt tokens",
		r.ArmID, r.On.CacheReadFraction, r.Off.CacheReadFraction, r.ReadFractionDelta, verdict,
		time.Duration(r.IdleGapSeconds*float64(time.Second)).Round(time.Second), r.PromptTokens)
}

// JSON renders the report as canonical indented JSON terminated by a newline — the report artifact
// witness (the two arms and their post-idle read fractions) that `cachevalue status` folds.
func (r TTL1HIdleABReport) JSON() []byte {
	b, _ := json.MarshalIndent(r, "", "  ")
	return append(b, '\n')
}

// idleABCaveat names the observational limits so a reader never mistakes this read for a live provider
// hit or over-reads a gap past the 1h tier's own retention.
func idleABCaveat(idleGap time.Duration) string {
	base := "OBSERVED read fractions come from the caller-captured post-idle turns on the SAME session (identical-workload guard enforced); this core prices no tokens and decides no survival — it reports the read/creation split the capture witnessed, not a live provider hit."
	if idleGap >= idleOneHour {
		return base + fmt.Sprintf(" NOTE: the injected gap (%s) meets or exceeds the 1h tier's own retention, so the ON arm's prefix may itself have expired — a low ON read fraction here is the tier's limit, not a regression.", idleGap.Round(time.Second))
	}
	return base
}
