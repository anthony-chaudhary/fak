// relayvscompaction.go is the relay-vs-auto-compaction cost/fidelity bench
// (issue #1906, rung J4 of the perpetual-sessions epic #1860). The spine
// (docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md) ASSERTS that a relay —
// one goal run as a sequence of bounded legs, each handing an O(1) pointer-only
// baton to a fresh leg — is both cheaper and more faithful than auto-compaction
// for a perpetual machine goal. This file MEASURES that assertion with numbers
// you can move and prove you moved.
//
// It replays the SAME hermetic goal (a backlog drain whose transcript does not
// fit in one window) through two context strategies:
//
//   - COMPACTION (the industry default: Anthropic ships it server-side, Claude
//     Code fires it near the limit). Context grows until it hits a reactive
//     trigger at ~95% of the window, then the transcript is summarized IN PLACE
//     and the leg continues. Rewriting the middle of the prefix busts the prompt
//     cache from the compaction point, and the summary — written under the same
//     context-rot pressure that triggered it — drops a fraction of the
//     load-bearing facts, which the strategy then trusts rather than re-deriving.
//   - RELAY (#1860). Context grows until a SOFT arm mark (~60% of the window),
//     then rotates at the next safe point: the leg externalizes every
//     load-bearing fact to the durable store (git / ledger / issues) through the
//     normal witnessed path, writes an O(1) baton, and a fresh leg takes over
//     seeded only by that baton. The transcript is discarded, not summarized, so
//     nothing load-bearing is lost and the prefix is never rewritten mid-stream.
//
// and reports the four metrics the issue names, over a DURATION SWEEP (the same
// goal at growing total-work sizes), so the headline invariant is witnessed and
// not merely asserted:
//
//   - peak context: the relay's peak resident tokens are INVARIANT to goal
//     duration (the "flat / O(1) peak" the operator asked about) and sit below
//     the context-rot zone; compaction's peak is pinned at the wall.
//   - cache-hit rate: the relay preserves the prompt cache (it never rewrites the
//     prefix mid-stream; a rotation resets an O(1) baton, not a near-wall prefix).
//   - token spend: cache-aware effective token cost across the whole goal.
//   - accuracy delta: fraction of load-bearing facts still available at the end.
//
// PROVENANCE (net-true doctrine, docs/standards/net-true-value.md): this is a
// hermetic ANALYTIC model, not a live provider run — which the issue explicitly
// permits ("a hermetic or replayed comparison is acceptable"). What it proves
// today is the MEASUREMENT and the sign of the comparison under representative,
// named constants; the invalidating assumptions are listed in the report so a
// live leg-record run can replace the model and either confirm or demote the
// claim. See the report's `assumptions` and `promotion` fields.
//
// Re-run: `go test ./internal/bench -run RelayVsCompaction` (the report is also
// regenerable into testdata/relayvscompaction_report.json with UPDATE_GOLDEN=1).
package bench

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// The context strategies the bench compares, as stable labels an OBSERVED leg
// record tags its run with. Relay and Compaction are the original two arms
// (#1906/#2707) carried on both the analytic AND the OBSERVED path; SlidingWindow
// and ContextEditing are the two additional SOTA competitor arms (#2709) modeled
// analytically so the sweep compares against MORE of the field than the two
// summarizer-vs-relay arms — the operator asked for a comparison against
// "builtin auto-compaction OR other SOTA methods" (plural).
const (
	StrategyRelay          = "relay"
	StrategyCompaction     = "compaction"
	StrategySlidingWindow  = "sliding_window"
	StrategyContextEditing = "context_editing"
)

// RVCModel is the hermetic, named constant set the comparison runs over. Every
// field is policy-visible data (DOS lesson #3: no magic number buried in code)
// so a reviewer can see exactly what economics the sign of the result rests on,
// and a live run can override any of it.
type RVCModel struct {
	// WindowCeiling is the hard context-window ceiling in tokens.
	WindowCeiling int `json:"window_ceiling_tokens"`
	// SystemPrefix is the stable warm prefix (system prompt + tool schemas) that
	// heads every turn and is cached across a leg.
	SystemPrefix int `json:"system_prefix_tokens"`
	// GrowthPerTurn is the tokens each work turn adds to the live context (tool
	// calls + results + reasoning).
	GrowthPerTurn int `json:"growth_per_turn_tokens"`
	// BatonTokens is the relay's O(1) carryover: objective pin + re-verifiable
	// cursor + pointers into the durable store. A fraction of one turn's growth.
	BatonTokens int `json:"baton_tokens"`
	// CompactionTriggerFrac is the fraction of the window at which auto-compaction
	// fires reactively (~0.95: it fires when the window is nearly full).
	CompactionTriggerFrac float64 `json:"compaction_trigger_frac"`
	// CompactionKeepFrac is the fraction of the transcript the in-place summary
	// retains (a compressed shadow of the discarded turns).
	CompactionKeepFrac float64 `json:"compaction_keep_frac"`
	// CompactionLossFrac is the fraction of the in-window load-bearing facts each
	// compaction drops under pressure — and, because compaction trusts its summary
	// rather than re-querying the durable store, does not recover.
	CompactionLossFrac float64 `json:"compaction_loss_frac"`
	// RelayArmFrac is the SOFT fraction of the window at which the relay arms and
	// rotates at the next safe point (~0.60: well below the context-rot zone).
	RelayArmFrac float64 `json:"relay_arm_frac"`
	// FactsPerTurn is the load-bearing facts (decisions, artifacts) a turn
	// produces — the denominator of the accuracy metric.
	FactsPerTurn int `json:"facts_per_turn"`
	// CacheReadMult / CacheWriteMult are the prompt-cache price multipliers
	// (Anthropic-style: a cache read is ~0.1x base, a cache write ~1.25x base).
	CacheReadMult  float64 `json:"cache_read_mult"`
	CacheWriteMult float64 `json:"cache_write_mult"`
	// SlidingWindowKeepFrac is the fraction of the window a sliding-window drop
	// (LangChain `trim_messages`) keeps as a trailing recent window, discarding
	// the OLDEST turns when the resident context exceeds it. It is the doc's
	// "closest analog to fak": a DROP (not a summary rewrite), so it is
	// cache-preserving — but LOSSY, because unlike the relay it does not
	// externalize the dropped turns to a durable store, so their load-bearing
	// facts are gone irrecoverably.
	SlidingWindowKeepFrac float64 `json:"sliding_window_keep_frac"`
	// ContextEditingTriggerFrac / ContextEditingKeepFrac model Anthropic API
	// context-editing (`clear_tool_uses`): when the window reaches the trigger it
	// clears old tool RESULTS server-side down to the keep floor. Docs:
	// "Invalidates cached prompt prefixes when content is cleared" — so a clear
	// BUSTS the cache (the post-clear prefix is re-sent uncached), but the cleared
	// content is RECOVERABLE via the paired memory tool, so no load-bearing fact
	// is lost (it trades cache cost for fidelity, the opposite bet from the
	// sliding window).
	ContextEditingTriggerFrac float64 `json:"context_editing_trigger_frac"`
	ContextEditingKeepFrac    float64 `json:"context_editing_keep_frac"`
}

// DefaultRVCModel is the representative constant set. The numbers are chosen so
// the goal never fits in one window (the only regime the relay is for) and the
// crossings are clean; the SIGN of the comparison does not depend on their exact
// values (see the report assumptions).
func DefaultRVCModel() RVCModel {
	return RVCModel{
		WindowCeiling:         200_000,
		SystemPrefix:          4_000,
		GrowthPerTurn:         6_000,
		BatonTokens:           2_000,
		CompactionTriggerFrac: 0.95,
		CompactionKeepFrac:    0.30,
		CompactionLossFrac:    0.15,
		RelayArmFrac:          0.60,
		FactsPerTurn:          3,
		CacheReadMult:         0.1,
		CacheWriteMult:        1.25,
		// The sliding window keeps the same ~60% trailing window the relay arms
		// at, so the two bounded strategies are compared at the SAME peak — the
		// honest apples-to-apples: they differ on fidelity/cost, not on peak.
		SlidingWindowKeepFrac: 0.60,
		// Context-editing fires earlier than compaction (it clears tool bloat
		// proactively) and clears down to a mid-window floor.
		ContextEditingTriggerFrac: 0.80,
		ContextEditingKeepFrac:    0.50,
	}
}

// DefaultRVCDurations is the sweep of total-work sizes. Each is long enough that
// the goal does not fit in one window (so compaction fires at least once), and
// the set grows so the relay's peak-context invariance is witnessed across an 8x
// range rather than at a single point. The set includes the operator-requested
// 100/200/300-turn horizons (issue #2707) alongside the original {40,80,160,320}
// so the same sweep both replicates the #1906 sign and feeds the explicit ≥50%
// billed-savings gate (RVCSavingsGateHorizons / computeSavingsGates).
func DefaultRVCDurations() []int { return []int{40, 80, 100, 160, 200, 300, 320} }

// RVCSavingsTarget is the operator-requested billed-token cost-reduction bar the
// long-horizon gate checks (issue #2707): the relay must bill at least this
// fraction fewer tokens than compaction, i.e.
// (compaction.BilledTokens - relay.BilledTokens) / compaction.BilledTokens >= 0.50.
const RVCSavingsTarget = 0.50

// RVCSavingsGateHorizons is the exact set of goal-turn horizons the ≥50% savings
// gate is evaluated at (issue #2707). They are a subset of DefaultRVCDurations so
// the arms already exist in the sweep; a live leg-record run that supplies these
// durations gets the same gate for free through the OBSERVED seam.
func RVCSavingsGateHorizons() []int { return []int{100, 200, 300} }

// RVCArm is one strategy's result on a goal of a given duration.
type RVCArm struct {
	Strategy string `json:"strategy"`
	// Rotations is the number of context-management events: relay leg rotations,
	// or compaction summarizations.
	Rotations int `json:"rotations"`
	// PeakContext is the peak resident tokens the window had to hold — THE
	// flat-context metric. For the relay this is invariant to GoalTurns.
	PeakContext int `json:"peak_context_tokens"`
	// PeakContextFrac is PeakContext / WindowCeiling (headroom: how deep into the
	// context-rot zone the strategy operates).
	PeakContextFrac float64 `json:"peak_context_frac"`
	// BilledTokens is the cache-aware effective token spend across the whole goal.
	BilledTokens int `json:"billed_tokens"`
	// CacheHitRate is cached-input / total-input across the run. NOTE: a strategy
	// that runs a LARGER resident prefix scores a marginally higher raw hit rate
	// (a fixed per-turn delta is a smaller fraction of a bigger prefix), so this
	// rate alone flatters compaction — the load-bearing cache metric is
	// CacheBustTokens below.
	CacheHitRate float64 `json:"cache_hit_rate"`
	// CacheBustTokens is the tokens force-rewritten (re-sent uncached at write
	// price) because a context-management event INVALIDATED the prompt-cache
	// prefix — compaction rewriting the middle of the prefix, or a relay leg
	// rotation. This is the spine's "compaction busts the cache" claim made
	// measurable: compaction re-pays a near-wall prefix per event, the relay an
	// O(1) baton. The initial cold-prefix load (common to both) is NOT counted.
	CacheBustTokens int `json:"cache_bust_tokens"`
	// LoadBearingFacts / FactsRetained / Accuracy quantify fidelity.
	LoadBearingFacts int     `json:"load_bearing_facts"`
	FactsRetained    int     `json:"facts_retained"`
	Accuracy         float64 `json:"accuracy"`
}

// RVCPoint is every modeled strategy at one duration. Relay and Compaction are
// always present (the original two arms, on both the analytic and OBSERVED path).
// SlidingWindow and ContextEditing are the two additional SOTA competitor arms
// (#2709); they are pointers with `omitempty` so the OBSERVED leg-record path —
// which only captures relay/compaction runs — omits them rather than emitting an
// empty, unwitnessed arm. The analytic sweep populates all four.
type RVCPoint struct {
	GoalTurns      int     `json:"goal_turns"`
	Relay          RVCArm  `json:"relay"`
	Compaction     RVCArm  `json:"compaction"`
	SlidingWindow  *RVCArm `json:"sliding_window,omitempty"`
	ContextEditing *RVCArm `json:"context_editing,omitempty"`
}

// RVCDelta is the relay's win over compaction, reported at the DEEPEST duration
// (where the strategies separate most) so the headline is the honest worst case
// for compaction, not a cherry-picked short goal.
type RVCDelta struct {
	AtGoalTurns          int     `json:"at_goal_turns"`
	PeakContextReduction int     `json:"peak_context_reduction_tokens"`
	PeakContextRatio     float64 `json:"peak_context_ratio_relay_over_compaction"`
	BilledTokensSaved    int     `json:"billed_tokens_saved"`
	BilledRatio          float64 `json:"billed_ratio_relay_over_compaction"`
	CacheBustSaved       int     `json:"cache_bust_tokens_saved"`
	CacheBustRatio       float64 `json:"cache_bust_ratio_relay_over_compaction"`
	CacheHitRateGain     float64 `json:"cache_hit_rate_gain"`
	AccuracyGain         float64 `json:"accuracy_gain"`
	Finding              string  `json:"finding"`
}

// RVCSavingsGate is the explicit long-horizon acceptance gate at one goal-turn
// horizon (issue #2707): does the relay bill at least RVCSavingsTarget (50%) fewer
// tokens than compaction at this horizon? It is computed, not asserted — the
// `savings_meets_50pct_target` bit is honest either way, and when it is false the
// shortfall names exactly how far under the bar the model landed so a live run
// (follow-on) can confirm or demote the number without a harness rewrite.
type RVCSavingsGate struct {
	GoalTurns              int `json:"goal_turns"`
	CompactionBilledTokens int `json:"compaction_billed_tokens"`
	RelayBilledTokens      int `json:"relay_billed_tokens"`
	// BilledSavingsFrac is (compaction - relay) / compaction — the fraction of
	// billed tokens the relay saves at this horizon.
	BilledSavingsFrac float64 `json:"billed_savings_frac"`
	// Target is the bar (RVCSavingsTarget) this horizon is graded against.
	Target float64 `json:"target"`
	// SavingsMeets50pctTarget is the done-condition verdict per horizon:
	// BilledSavingsFrac >= Target.
	SavingsMeets50pctTarget bool `json:"savings_meets_50pct_target"`
	// ShortfallFrac is how far below the bar the horizon landed (Target -
	// BilledSavingsFrac), 0 when the gate is met — the honest shortfall the done
	// condition requires be named on a false verdict.
	ShortfallFrac float64 `json:"shortfall_frac"`
}

// Verdicts the bench can reach.
const (
	// VerdictRelayWins: the relay is strictly lower-peak, cheaper, and at least as
	// faithful as compaction at every swept duration — the #1906 acceptance.
	VerdictRelayWins = "relay_lower_peak_cheaper_more_faithful"
	// VerdictNoAdvantage: the model did not separate the strategies (an honest
	// finding, not a failure — it would say the relay's complexity is unjustified
	// under these constants).
	VerdictNoAdvantage = "no_measurable_relay_advantage"
)

// RelayVsCompactionReport is the full comparison.
type RelayVsCompactionReport struct {
	Schema     string     `json:"schema"`
	Provenance Provenance `json:"provenance"`
	Model      RVCModel   `json:"model"`
	// Strategies is the ordered set of context-management arms modeled at EVERY
	// swept duration. The analytic path carries all four (relay, compaction,
	// sliding_window, context_editing — #2709 widened the field from two);
	// the OBSERVED leg-record path carries only the two it captured. A row that
	// cites this report as a MEASURED artifact should confirm len(Strategies) >= 4
	// at the requested horizons.
	Strategies []string   `json:"strategies"`
	Sweep      []RVCPoint `json:"sweep"`
	// FlatPeakContext is the done-condition witness: the relay's peak context is
	// IDENTICAL across every swept duration (bounded by the per-leg ceiling,
	// independent of goal duration — the O(1) invariant).
	FlatPeakContext  bool `json:"flat_peak_context"`
	RelayPeakContext int  `json:"relay_peak_context_tokens"`
	// SavingsTarget is the ≥50% bar the long-horizon gate grades against (#2707).
	SavingsTarget float64 `json:"savings_target"`
	// SavingsGates is the explicit acceptance gate at each of the 100/200/300-turn
	// horizons the operator asked for (#2707): the billed-savings fraction and a
	// per-horizon `savings_meets_50pct_target` verdict (true, or an honest false
	// with the shortfall named). Empty if the sweep supplied none of those horizons.
	SavingsGates []RVCSavingsGate `json:"savings_gates"`
	// AllHorizonsMeet50pct is true iff every evaluated horizon meets the target AND
	// all three requested horizons are present — a false-closed rollup that does not
	// claim the bar is cleared over a missing horizon.
	AllHorizonsMeet50pct bool     `json:"all_horizons_meet_50pct"`
	Delta                RVCDelta `json:"delta"`
	Verdict              string   `json:"verdict"`
	Assumptions          []string `json:"assumptions"`
	Promotion            string   `json:"promotion"`
	DemotionRetirement   string   `json:"demotion_or_retirement"`
	InvalidatingUnknown  string   `json:"invalidating_assumption"`
}

// simulateCompaction replays the goal under reactive in-place compaction.
func simulateCompaction(m RVCModel, goalTurns int) RVCArm {
	trigger := int(m.CompactionTriggerFrac * float64(m.WindowCeiling))
	resident := m.SystemPrefix
	cacheValidUpTo := 0
	peak := resident
	var billed float64
	var cacheHits, cacheTotal, cacheBust, bustPending int
	rotations := 0
	factsInWindow := 0 // facts still live in the (summarized) window
	factsRetained := 0
	factsTotal := 0

	for turn := 0; turn < goalTurns; turn++ {
		// Bill this turn's input = the resident context re-sent to the model.
		cached := min(cacheValidUpTo, resident)
		uncached := resident - cached
		if bustPending > 0 { // uncached this turn is a management-induced re-write
			cacheBust += min(bustPending, uncached)
			bustPending = 0
		}
		billed += float64(cached)*m.CacheReadMult + float64(uncached)*m.CacheWriteMult
		cacheHits += cached
		cacheTotal += resident
		cacheValidUpTo = resident // everything seen this turn is now cached

		// Do the work: produce load-bearing facts, grow the context.
		factsTotal += m.FactsPerTurn
		factsInWindow += m.FactsPerTurn
		factsRetained += m.FactsPerTurn
		resident += m.GrowthPerTurn
		if resident > peak {
			peak = resident
		}

		// Reactive compaction at ~95%: summarize the transcript in place.
		if resident >= trigger {
			rotations++
			transcript := resident - m.SystemPrefix
			kept := int(m.CompactionKeepFrac * float64(transcript))
			resident = m.SystemPrefix + kept
			cacheValidUpTo = 0     // prefix rewritten -> cache fully busted
			bustPending = resident // next turn re-sends the whole near-wall summary
			// Lossy under pressure: drop a fraction of the in-window facts, and
			// (doctrine) trust the summary rather than re-query the durable store,
			// so the dropped facts are lost to the run's reasoning.
			lost := int(m.CompactionLossFrac * float64(factsInWindow))
			factsRetained -= lost
			factsInWindow -= lost
		}
	}
	return finishArm(StrategyCompaction, m, rotations, peak, billed, cacheHits, cacheTotal, cacheBust, factsTotal, factsRetained)
}

// simulateRelay replays the SAME goal under bounded-leg rotation.
func simulateRelay(m RVCModel, goalTurns int) RVCArm {
	arm := int(m.RelayArmFrac * float64(m.WindowCeiling))
	resident := m.SystemPrefix + m.BatonTokens
	cacheValidUpTo := 0
	peak := resident
	var billed float64
	var cacheHits, cacheTotal, cacheBust, bustPending int
	rotations := 0
	factsTotal := 0

	for turn := 0; turn < goalTurns; turn++ {
		cached := min(cacheValidUpTo, resident)
		uncached := resident - cached
		if bustPending > 0 { // uncached this turn is a rotation-induced re-write
			cacheBust += min(bustPending, uncached)
			bustPending = 0
		}
		billed += float64(cached)*m.CacheReadMult + float64(uncached)*m.CacheWriteMult
		cacheHits += cached
		cacheTotal += resident
		cacheValidUpTo = resident

		factsTotal += m.FactsPerTurn // every fact externalized -> all retained
		resident += m.GrowthPerTurn
		if resident > peak {
			peak = resident
		}

		// Arm at the soft mark and rotate at the safe point (modeled at end of
		// turn). The fresh leg is seeded only by the O(1) baton.
		if resident >= arm {
			rotations++
			resident = m.SystemPrefix + m.BatonTokens
			cacheValidUpTo = 0     // new leg, fresh prefix — but O(1), not near-wall
			bustPending = resident // next leg re-sends only the O(1) baton prefix
		}
	}
	// Fidelity: the fail-closed externalize gate guarantees every load-bearing
	// fact is durable before rotation, and the baton is re-verified at read, so
	// nothing load-bearing is lost.
	return finishArm(StrategyRelay, m, rotations, peak, billed, cacheHits, cacheTotal, cacheBust, factsTotal, factsTotal)
}

// simulateSlidingWindow replays the SAME goal under a LangChain-style
// `trim_messages` sliding window: keep a trailing recent window up to
// SlidingWindowKeepFrac of the ceiling, dropping the OLDEST turns when the
// resident context exceeds it. The drop re-uses the still-cached system prefix +
// retained window verbatim (no summary rewrite), so it is CACHE-PRESERVING — the
// one competitor design, with the relay, that reasons about preservation. Its
// weakness is FIDELITY: the dropped turns are gone with no externalization to a
// durable store, so their load-bearing facts are lost irrecoverably (the relay
// externalizes before it discards). This is the honest apples-to-apples arm: it
// ties the relay on the bounded-peak / cache axes and separates only on accuracy.
func simulateSlidingWindow(m RVCModel, goalTurns int) RVCArm {
	cap := int(m.SlidingWindowKeepFrac * float64(m.WindowCeiling))
	resident := m.SystemPrefix
	cacheValidUpTo := 0
	peak := resident
	var billed float64
	var cacheHits, cacheTotal, cacheBust, bustPending int
	rotations := 0
	factsInWindow := 0
	factsRetained := 0
	factsTotal := 0

	for turn := 0; turn < goalTurns; turn++ {
		cached := min(cacheValidUpTo, resident)
		uncached := resident - cached
		if bustPending > 0 {
			cacheBust += min(bustPending, uncached)
			bustPending = 0
		}
		billed += float64(cached)*m.CacheReadMult + float64(uncached)*m.CacheWriteMult
		cacheHits += cached
		cacheTotal += resident
		cacheValidUpTo = resident

		factsTotal += m.FactsPerTurn
		factsInWindow += m.FactsPerTurn
		factsRetained += m.FactsPerTurn
		resident += m.GrowthPerTurn
		if resident > peak {
			peak = resident
		}

		// Trim: drop the oldest turns (after the stable system prefix) to bring the
		// resident context back under the trailing-window cap. It is a DROP, not a
		// summary rewrite: the system prefix stays cached and the retained window
		// was already cached, so no content is force-rewritten (cache-preserving —
		// bustPending stays 0). But the dropped turns are irrecoverable, so a
		// proportional slice of the still-live facts is lost with no way to re-derive
		// them (no durable externalize gate).
		if resident > cap {
			rotations++
			dropped := resident - cap
			transcript := resident - m.SystemPrefix
			lost := int(float64(dropped) / float64(transcript) * float64(factsInWindow))
			factsRetained -= lost
			factsInWindow -= lost
			resident = cap
			cacheValidUpTo = min(cacheValidUpTo, resident)
		}
	}
	return finishArm(StrategySlidingWindow, m, rotations, peak, billed, cacheHits, cacheTotal, cacheBust, factsTotal, factsRetained)
}

// simulateContextEditing replays the SAME goal under Anthropic API
// context-editing (`clear_tool_uses`): grow until the trigger, then clear old
// tool results down to the keep floor. Per the provider docs a clear
// "Invalidates cached prompt prefixes when content is cleared", so the post-clear
// prefix is re-sent UNCACHED next turn (a cache-bust cost); but the cleared
// content is RECOVERABLE via the paired memory tool, so every load-bearing fact
// is retained. It is the mirror image of the sliding window: it keeps fidelity
// (accuracy 1.0) and pays cache, where the sliding window keeps cache and pays
// fidelity — and the relay is the only arm that keeps both.
func simulateContextEditing(m RVCModel, goalTurns int) RVCArm {
	trigger := int(m.ContextEditingTriggerFrac * float64(m.WindowCeiling))
	keep := int(m.ContextEditingKeepFrac * float64(m.WindowCeiling))
	resident := m.SystemPrefix
	cacheValidUpTo := 0
	peak := resident
	var billed float64
	var cacheHits, cacheTotal, cacheBust, bustPending int
	rotations := 0
	factsTotal := 0

	for turn := 0; turn < goalTurns; turn++ {
		cached := min(cacheValidUpTo, resident)
		uncached := resident - cached
		if bustPending > 0 {
			cacheBust += min(bustPending, uncached)
			bustPending = 0
		}
		billed += float64(cached)*m.CacheReadMult + float64(uncached)*m.CacheWriteMult
		cacheHits += cached
		cacheTotal += resident
		cacheValidUpTo = resident

		factsTotal += m.FactsPerTurn
		resident += m.GrowthPerTurn
		if resident > peak {
			peak = resident
		}

		// Clear at the trigger: shrink to the keep floor and invalidate the cache
		// prefix from the clear point. The next turn re-sends the post-clear prefix
		// uncached (bustPending). Facts are preserved: the memory tool makes the
		// cleared content recoverable, so retained == produced.
		if resident >= trigger {
			rotations++
			resident = keep
			cacheValidUpTo = 0
			bustPending = resident
		}
	}
	return finishArm(StrategyContextEditing, m, rotations, peak, billed, cacheHits, cacheTotal, cacheBust, factsTotal, factsTotal)
}

func finishArm(strategy string, m RVCModel, rotations, peak int, billed float64, cacheHits, cacheTotal, cacheBust, factsTotal, factsRetained int) RVCArm {
	arm := RVCArm{
		Strategy:         strategy,
		Rotations:        rotations,
		PeakContext:      peak,
		PeakContextFrac:  round4(float64(peak) / float64(m.WindowCeiling)),
		BilledTokens:     int(billed),
		CacheBustTokens:  cacheBust,
		LoadBearingFacts: factsTotal,
		FactsRetained:    factsRetained,
	}
	if cacheTotal > 0 {
		arm.CacheHitRate = round4(float64(cacheHits) / float64(cacheTotal))
	}
	if factsTotal > 0 {
		arm.Accuracy = round4(float64(factsRetained) / float64(factsTotal))
	}
	return arm
}

// BuildRelayVsCompactionReport runs the default model over the default sweep.
func BuildRelayVsCompactionReport() RelayVsCompactionReport {
	return BuildRelayVsCompactionReportFor(DefaultRVCModel(), DefaultRVCDurations())
}

// BuildRelayVsCompactionReportFor folds an arbitrary model + duration sweep — the
// seam a live leg-record run feeds observed constants into.
func BuildRelayVsCompactionReportFor(m RVCModel, durations []int) RelayVsCompactionReport {
	sweep := make([]RVCPoint, 0, len(durations))
	for _, n := range durations {
		sw := simulateSlidingWindow(m, n)
		ce := simulateContextEditing(m, n)
		sweep = append(sweep, RVCPoint{
			GoalTurns:      n,
			Relay:          simulateRelay(m, n),
			Compaction:     simulateCompaction(m, n),
			SlidingWindow:  &sw,
			ContextEditing: &ce,
		})
	}
	return assembleRVCReport(m, sweep, simulatedRVCProvenance())
}

// simulatedRVCProvenance labels the hermetic analytic path.
func simulatedRVCProvenance() Provenance {
	return simulatedProvenance(
		"go test ./internal/bench -run RelayVsCompaction",
		"fak/internal/bench.BuildRelayVsCompactionReport",
		"Hermetic ANALYTIC model, not a live provider run (the issue permits a "+
			"hermetic/replayed comparison). It witnesses the MEASUREMENT and the sign "+
			"of the comparison under the named constants in `model`; a live relay-vs-"+
			"compaction leg-record run can feed the same report shape via "+
			"BuildRelayVsCompactionReportFromRecords and either confirm or demote the claim.",
	)
}

// assembleRVCReport folds an already-populated sweep (simulated OR observed) into
// the full relayvscompaction.v1 report: the flat-peak witness, the deepest-
// duration delta, the verdict, and the net-true prose. The caller supplies the
// provenance so the SAME assembly serves both the analytic path and the OBSERVED
// leg-record path (BuildRelayVsCompactionReportFromRecords).
func assembleRVCReport(m RVCModel, sweep []RVCPoint, provenance Provenance) RelayVsCompactionReport {
	// The flat-peak witness: the relay's peak context is identical at every
	// duration (independent of total work done). An empty sweep has no data point to
	// witness the invariant, so flat is false — a claim needs at least one duration.
	flat := len(sweep) > 0
	relayPeak := 0
	if len(sweep) > 0 {
		relayPeak = sweep[0].Relay.PeakContext
		for _, p := range sweep {
			if p.Relay.PeakContext != relayPeak {
				flat = false
			}
		}
	}

	// Delta at the deepest duration (worst case for compaction). The relay wins if
	// at EVERY duration it is lower-peak, cheaper (billed), busts less cache, and
	// is at least as faithful. Raw cache-hit RATE is reported but deliberately NOT
	// an acceptance gate — it flatters the larger-prefix strategy (see RVCArm). An
	// empty sweep has ZERO comparisons and so cannot witness a win: it fails closed
	// to VerdictNoAdvantage rather than defaulting to a claim over no data points
	// (the fail-closed contract BuildRelayVsCompactionReportFromRecords already
	// enforces on the OBSERVED path).
	verdict := VerdictNoAdvantage
	if len(sweep) > 0 {
		verdict = VerdictRelayWins
		for _, p := range sweep {
			if !(p.Relay.PeakContext < p.Compaction.PeakContext &&
				p.Relay.BilledTokens < p.Compaction.BilledTokens &&
				p.Relay.CacheBustTokens < p.Compaction.CacheBustTokens &&
				p.Relay.Accuracy >= p.Compaction.Accuracy) {
				verdict = VerdictNoAdvantage
				break
			}
		}
	}

	var delta RVCDelta
	if len(sweep) > 0 {
		d := sweep[len(sweep)-1]
		delta = RVCDelta{
			AtGoalTurns:          d.GoalTurns,
			PeakContextReduction: d.Compaction.PeakContext - d.Relay.PeakContext,
			PeakContextRatio:     safeRatio(d.Relay.PeakContext, d.Compaction.PeakContext),
			BilledTokensSaved:    d.Compaction.BilledTokens - d.Relay.BilledTokens,
			BilledRatio:          safeRatio(d.Relay.BilledTokens, d.Compaction.BilledTokens),
			CacheBustSaved:       d.Compaction.CacheBustTokens - d.Relay.CacheBustTokens,
			CacheBustRatio:       safeRatio(d.Relay.CacheBustTokens, d.Compaction.CacheBustTokens),
			CacheHitRateGain:     round4(d.Relay.CacheHitRate - d.Compaction.CacheHitRate),
			AccuracyGain:         round4(d.Relay.Accuracy - d.Compaction.Accuracy),
		}
		delta.Finding = rvcFinding(delta, d, relayPeak, flat)
	}

	gates := computeSavingsGates(sweep)
	// AllHorizonsMeet50pct fails closed: it is true only when all three requested
	// horizons are present AND each meets the bar. A partial sweep (a horizon
	// missing) cannot witness the long-horizon claim.
	allMeet := len(gates) == len(RVCSavingsGateHorizons())
	for _, g := range gates {
		if !g.SavingsMeets50pctTarget {
			allMeet = false
		}
	}

	return RelayVsCompactionReport{
		Schema:               "relayvscompaction.v1",
		Provenance:           provenance,
		Model:                m,
		Strategies:           rvcStrategies(sweep),
		Sweep:                sweep,
		FlatPeakContext:      flat,
		RelayPeakContext:     relayPeak,
		SavingsTarget:        RVCSavingsTarget,
		SavingsGates:         gates,
		AllHorizonsMeet50pct: allMeet,
		Delta:                delta,
		Verdict:              verdict,
		Assumptions: []string{
			"Compaction TRUSTS its in-place summary rather than re-querying the durable store; a compacting agent that also re-derived dropped facts from git on demand would narrow the accuracy gap toward the relay's. This is the load-bearing doctrinal assumption.",
			"The relay's externalize gate is fail-closed: every load-bearing fact is committed/ledgered/filed before rotation and the baton is re-verified at read, so accuracy is 1.0. A leg that rotated with load-bearing state still transcript-only (a gate bypass) would drop below 1.0.",
			"Cache multipliers (read 0.1x, write 1.25x) and the window/growth/baton constants are representative Anthropic-style figures, not a measured provider run; different economics shift the magnitudes but not the sign — the relay resets an O(1) prefix, compaction rewrites a near-wall one.",
			"The goal does not fit in one window (the only regime a relay is for); for a goal that fits in one leg, neither strategy fires and there is no difference to measure.",
		},
		Promotion:           "Replace the modeled constants with OBSERVED per-leg context/cache/token records from a live relay-vs-compaction run (the internal/sessionreset + session.Recontinue relay path vs a compaction baseline), feeding BuildRelayVsCompactionReportFromRecords (the loader shipped here, mirroring loopverify's OBSERVED ledger path). When the observed run reproduces the sign, the leaf promotes toward `now`.",
		DemotionRetirement:  "If a live run shows compaction matching the relay on BOTH cost and fidelity for a goal class (e.g. a compaction that re-queries git), the relay's complexity is not justified for that class and this bench demotes the claim rather than defending it.",
		InvalidatingUnknown: "The single assumption most likely to flip the result is #1: that compaction does not re-derive dropped facts from the durable store. If it does, the fidelity advantage collapses and only the peak-context and cache advantages remain.",
	}
}

// rvcStrategies is the ordered arm roster the sweep actually carries: relay and
// compaction always (the original two), then sliding_window and context_editing
// when the points populate them (the analytic path — #2709). It is derived from
// the sweep rather than asserted so the OBSERVED path (two arms) and the analytic
// path (four) each report exactly what they modeled — an honest arm count, never
// a fabricated fourth arm the records do not back.
func rvcStrategies(sweep []RVCPoint) []string {
	if len(sweep) == 0 {
		return nil
	}
	strategies := []string{StrategyRelay, StrategyCompaction}
	p := sweep[0]
	if p.SlidingWindow != nil {
		strategies = append(strategies, StrategySlidingWindow)
	}
	if p.ContextEditing != nil {
		strategies = append(strategies, StrategyContextEditing)
	}
	return strategies
}

// computeSavingsGates evaluates the explicit ≥50% billed-savings gate (issue
// #2707) at each requested horizon (RVCSavingsGateHorizons) that the sweep
// actually contains. It reads the SAME BilledTokens the arms already produce
// (analytic OR observed), so the OBSERVED leg-record path gets the gate for free.
// A horizon absent from the sweep is skipped rather than fabricated — the gate is
// only reported where both arms exist.
func computeSavingsGates(sweep []RVCPoint) []RVCSavingsGate {
	byDur := make(map[int]RVCPoint, len(sweep))
	for _, p := range sweep {
		byDur[p.GoalTurns] = p
	}
	horizons := RVCSavingsGateHorizons()
	gates := make([]RVCSavingsGate, 0, len(horizons))
	for _, h := range horizons {
		p, ok := byDur[h]
		if !ok {
			continue
		}
		comp := p.Compaction.BilledTokens
		relay := p.Relay.BilledTokens
		var frac float64
		if comp > 0 {
			frac = round4(float64(comp-relay) / float64(comp))
		}
		meets := frac >= RVCSavingsTarget
		shortfall := 0.0
		if !meets {
			shortfall = round4(RVCSavingsTarget - frac)
		}
		gates = append(gates, RVCSavingsGate{
			GoalTurns:               h,
			CompactionBilledTokens:  comp,
			RelayBilledTokens:       relay,
			BilledSavingsFrac:       frac,
			Target:                  RVCSavingsTarget,
			SavingsMeets50pctTarget: meets,
			ShortfallFrac:           shortfall,
		})
	}
	return gates
}

func safeRatio(num, den int) float64 {
	if den == 0 {
		return 0
	}
	return round4(float64(num) / float64(den))
}

func rvcFinding(d RVCDelta, p RVCPoint, relayPeak int, flat bool) string {
	flatNote := "the relay's peak context is FLAT across the sweep"
	if !flat {
		flatNote = "the relay's peak context is NOT flat across the sweep (model regime error)"
	}
	return fmt.Sprintf(
		"At %d turns, %s (%d tokens, %.0f%% of the window — out of the context-rot zone), "+
			"while compaction peaks at %d (%.0f%% — pinned at the wall). The relay bills %d effective "+
			"tokens vs compaction's %d (%.0f%% of the cost), force-rewrites only %d cache-busted tokens vs "+
			"compaction's %d (%.0f%% — it resets an O(1) baton, not a near-wall prefix), and retains %.0f%% of "+
			"load-bearing facts vs %.0f%%. Raw cache-hit rate is comparable (%.2f vs %.2f; compaction's is "+
			"marginally higher only because it runs a larger resident prefix). Peak, cost, cache-preservation, "+
			"and fidelity favor the relay; the comparison is a hermetic model, so the numbers are the SIGN, "+
			"not a provider-billed total.",
		p.GoalTurns, flatNote, relayPeak, p.Relay.PeakContextFrac*100,
		p.Compaction.PeakContext, p.Compaction.PeakContextFrac*100,
		p.Relay.BilledTokens, p.Compaction.BilledTokens, d.BilledRatio*100,
		p.Relay.CacheBustTokens, p.Compaction.CacheBustTokens, d.CacheBustRatio*100,
		p.Relay.Accuracy*100, p.Compaction.Accuracy*100,
		p.Relay.CacheHitRate, p.Compaction.CacheHitRate,
	)
}

// JSON renders the report as stable, indented JSON (deterministic: no clock, no
// map iteration), so it is a re-derivable witness.
func (r RelayVsCompactionReport) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// ── OBSERVED leg-record path ────────────────────────────────────────────────
//
// The types and loaders below are the promotion seam the report's `promotion`
// field names: they fold OBSERVED per-leg records from a live (or replayed)
// relay-vs-compaction run into the SAME relayvscompaction.v1 shape, mirroring
// loopverify's OBSERVED ledger path (BuildLoopVerifyReportFromLoop*). The
// distinction OBSERVED vs SIMULATED is the DERIVATION path — measured leg records
// vs the analytic model — not a claim of a provider-billed total; the report's
// provenance Note carries that honesty, exactly as loopverify's does.

// RVCTurnRecord is one OBSERVED turn of a leg run: the input tokens actually sent
// this turn split cached/uncached (so the same cache-aware billing the analytic
// arms use applies), the tokens force-rewritten by a context-management event
// this turn (a compaction summarize or a relay leg rotation; 0 on an ordinary
// turn), the resident context after the turn, and the load-bearing facts the turn
// produced plus how many survived to the run's end. It is the measured counterpart
// of one iteration of simulateRelay/simulateCompaction.
type RVCTurnRecord struct {
	CachedInputTokens   int `json:"cached_input_tokens"`
	UncachedInputTokens int `json:"uncached_input_tokens"`
	CacheBustTokens     int `json:"cache_bust_tokens"`
	ResidentTokens      int `json:"resident_tokens"`
	FactsProduced       int `json:"facts_produced"`
	FactsRetained       int `json:"facts_retained"`
}

// RVCRunRecord is one OBSERVED arm run: a single strategy replayed over a goal of
// a given duration, as an ordered sequence of turn records. A comparison needs a
// relay AND a compaction run at each duration (one arm proves nothing).
type RVCRunRecord struct {
	Strategy  string          `json:"strategy"` // StrategyRelay | StrategyCompaction
	GoalTurns int             `json:"goal_turns"`
	Turns     []RVCTurnRecord `json:"turns"`
}

// observedArm folds a run's OBSERVED turn records into an RVCArm using the SAME
// cache-aware billing the analytic arms use (a cache read at CacheReadMult, an
// uncached/re-written token at CacheWriteMult). Rotations are counted from turns
// that force-rewrote cache; peak is the max resident context observed.
func observedArm(m RVCModel, run RVCRunRecord) RVCArm {
	var billed float64
	var cacheHits, cacheTotal, cacheBust int
	peak, rotations := 0, 0
	factsTotal, factsRetained := 0, 0
	for _, t := range run.Turns {
		billed += float64(t.CachedInputTokens)*m.CacheReadMult + float64(t.UncachedInputTokens)*m.CacheWriteMult
		cacheHits += t.CachedInputTokens
		cacheTotal += t.CachedInputTokens + t.UncachedInputTokens
		cacheBust += t.CacheBustTokens
		if t.CacheBustTokens > 0 {
			rotations++
		}
		if t.ResidentTokens > peak {
			peak = t.ResidentTokens
		}
		factsTotal += t.FactsProduced
		factsRetained += t.FactsRetained
	}
	return finishArm(run.Strategy, m, rotations, peak, billed, cacheHits, cacheTotal, cacheBust, factsTotal, factsRetained)
}

// BuildRelayVsCompactionReportFromRecords folds OBSERVED relay/compaction leg
// records into the relayvscompaction.v1 report, mirroring loopverify's OBSERVED
// ledger path. It refuses rather than emit an OBSERVED report the records do not
// back: no records, a run with no turns, an unknown strategy label, a duplicate
// arm, or a duration missing one of the two arms each fail closed — OBSERVED
// provenance is only produced when both arms are actually present at every swept
// duration.
func BuildRelayVsCompactionReportFromRecords(m RVCModel, runs []RVCRunRecord, command string) (RelayVsCompactionReport, error) {
	if len(runs) == 0 {
		return RelayVsCompactionReport{}, errors.New("relayvscompaction: no observed leg records")
	}
	type armPair struct {
		relay, compaction *RVCRunRecord
	}
	byDur := map[int]*armPair{}
	var order []int
	for i := range runs {
		r := &runs[i]
		if len(r.Turns) == 0 {
			return RelayVsCompactionReport{}, fmt.Errorf("relayvscompaction: observed %s run at %d turns has no turn records", r.Strategy, r.GoalTurns)
		}
		p := byDur[r.GoalTurns]
		if p == nil {
			p = &armPair{}
			byDur[r.GoalTurns] = p
			order = append(order, r.GoalTurns)
		}
		switch r.Strategy {
		case StrategyRelay:
			if p.relay != nil {
				return RelayVsCompactionReport{}, fmt.Errorf("relayvscompaction: duplicate relay run at %d turns", r.GoalTurns)
			}
			p.relay = r
		case StrategyCompaction:
			if p.compaction != nil {
				return RelayVsCompactionReport{}, fmt.Errorf("relayvscompaction: duplicate compaction run at %d turns", r.GoalTurns)
			}
			p.compaction = r
		default:
			return RelayVsCompactionReport{}, fmt.Errorf("relayvscompaction: unknown strategy %q (want %q|%q)", r.Strategy, StrategyRelay, StrategyCompaction)
		}
	}
	sort.Ints(order)
	sweep := make([]RVCPoint, 0, len(order))
	for _, n := range order {
		p := byDur[n]
		switch {
		case p.relay == nil:
			return RelayVsCompactionReport{}, fmt.Errorf("relayvscompaction: duration %d turns is missing the relay arm", n)
		case p.compaction == nil:
			return RelayVsCompactionReport{}, fmt.Errorf("relayvscompaction: duration %d turns is missing the compaction arm", n)
		}
		sweep = append(sweep, RVCPoint{
			GoalTurns:  n,
			Relay:      observedArm(m, *p.relay),
			Compaction: observedArm(m, *p.compaction),
		})
	}
	if command == "" {
		command = "relayvscompaction observed leg records"
	}
	return assembleRVCReport(m, sweep, Provenance{
		Kind:        ProvenanceObserved,
		Command:     command,
		GeneratedBy: "fak/internal/bench.BuildRelayVsCompactionReportFromRecords",
		Note: "Report folded from OBSERVED per-leg records (billed cached/uncached input, " +
			"cache-bust tokens, resident context, load-bearing facts), NOT the analytic " +
			"model. The checked-in sample is a REPLAYED capture that witnesses the loader " +
			"and schema, not a live provider-billed run; a live relay-vs-compaction run " +
			"feeds the same shape to promote the claim toward gen/now.",
	}), nil
}

// BuildRelayVsCompactionReportFromLegLedger reads a JSON array of OBSERVED
// RVCRunRecord leg records from path and folds them into the report — the file
// counterpart of loopverify's BuildLoopVerifyReportFromLoopLedger. The command
// field is derived from the file's base name (no machine-absolute path) so the
// resulting artifact is deterministic across checkouts.
func BuildRelayVsCompactionReportFromLegLedger(m RVCModel, path string) (RelayVsCompactionReport, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return RelayVsCompactionReport{}, err
	}
	var runs []RVCRunRecord
	if err := json.Unmarshal(data, &runs); err != nil {
		return RelayVsCompactionReport{}, fmt.Errorf("relayvscompaction: parse leg ledger %s: %w", filepath.Base(path), err)
	}
	return BuildRelayVsCompactionReportFromRecords(m, runs, "replayed leg-record sample "+filepath.Base(path))
}
