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
	"fmt"
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
	}
}

// DefaultRVCDurations is the sweep of total-work sizes. Each is long enough that
// the goal does not fit in one window (so compaction fires at least once), and
// the set grows so the relay's peak-context invariance is witnessed across a 8x
// range rather than at a single point.
func DefaultRVCDurations() []int { return []int{40, 80, 160, 320} }

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

// RVCPoint is both strategies at one duration.
type RVCPoint struct {
	GoalTurns  int    `json:"goal_turns"`
	Relay      RVCArm `json:"relay"`
	Compaction RVCArm `json:"compaction"`
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
	Sweep      []RVCPoint `json:"sweep"`
	// FlatPeakContext is the done-condition witness: the relay's peak context is
	// IDENTICAL across every swept duration (bounded by the per-leg ceiling,
	// independent of goal duration — the O(1) invariant).
	FlatPeakContext     bool     `json:"flat_peak_context"`
	RelayPeakContext    int      `json:"relay_peak_context_tokens"`
	Delta               RVCDelta `json:"delta"`
	Verdict             string   `json:"verdict"`
	Assumptions         []string `json:"assumptions"`
	Promotion           string   `json:"promotion"`
	DemotionRetirement  string   `json:"demotion_or_retirement"`
	InvalidatingUnknown string   `json:"invalidating_assumption"`
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
	return finishArm("compaction", m, rotations, peak, billed, cacheHits, cacheTotal, cacheBust, factsTotal, factsRetained)
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
	return finishArm("relay", m, rotations, peak, billed, cacheHits, cacheTotal, cacheBust, factsTotal, factsTotal)
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
		sweep = append(sweep, RVCPoint{
			GoalTurns:  n,
			Relay:      simulateRelay(m, n),
			Compaction: simulateCompaction(m, n),
		})
	}

	// The flat-peak witness: the relay's peak context is identical at every
	// duration (independent of total work done).
	flat := true
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
	// an acceptance gate — it flatters the larger-prefix strategy (see RVCArm).
	verdict := VerdictRelayWins
	for _, p := range sweep {
		if !(p.Relay.PeakContext < p.Compaction.PeakContext &&
			p.Relay.BilledTokens < p.Compaction.BilledTokens &&
			p.Relay.CacheBustTokens < p.Compaction.CacheBustTokens &&
			p.Relay.Accuracy >= p.Compaction.Accuracy) {
			verdict = VerdictNoAdvantage
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

	return RelayVsCompactionReport{
		Schema: "relayvscompaction.v1",
		Provenance: Provenance{
			Kind:        ProvenanceSimulated,
			Command:     "go test ./internal/bench -run RelayVsCompaction",
			GeneratedBy: "fak/internal/bench.BuildRelayVsCompactionReport",
			Note: "Hermetic ANALYTIC model, not a live provider run (the issue permits a " +
				"hermetic/replayed comparison). It witnesses the MEASUREMENT and the sign " +
				"of the comparison under the named constants in `model`; a live relay-vs-" +
				"compaction leg-record run can feed the same report shape via " +
				"BuildRelayVsCompactionReportFor and either confirm or demote the claim.",
		},
		Model:            m,
		Sweep:            sweep,
		FlatPeakContext:  flat,
		RelayPeakContext: relayPeak,
		Delta:            delta,
		Verdict:          verdict,
		Assumptions: []string{
			"Compaction TRUSTS its in-place summary rather than re-querying the durable store; a compacting agent that also re-derived dropped facts from git on demand would narrow the accuracy gap toward the relay's. This is the load-bearing doctrinal assumption.",
			"The relay's externalize gate is fail-closed: every load-bearing fact is committed/ledgered/filed before rotation and the baton is re-verified at read, so accuracy is 1.0. A leg that rotated with load-bearing state still transcript-only (a gate bypass) would drop below 1.0.",
			"Cache multipliers (read 0.1x, write 1.25x) and the window/growth/baton constants are representative Anthropic-style figures, not a measured provider run; different economics shift the magnitudes but not the sign — the relay resets an O(1) prefix, compaction rewrites a near-wall one.",
			"The goal does not fit in one window (the only regime a relay is for); for a goal that fits in one leg, neither strategy fires and there is no difference to measure.",
		},
		Promotion: "Replace the modeled constants with OBSERVED per-turn context/cache/token records from a live relay-vs-compaction run (the internal/sessionreset + session.Recontinue relay path vs a compaction baseline), feeding BuildRelayVsCompactionReportFor, the way loopverify accepts an OBSERVED ledger. When the observed run reproduces the sign, the leaf promotes toward `now`.",
		DemotionRetirement: "If a live run shows compaction matching the relay on BOTH cost and fidelity for a goal class (e.g. a compaction that re-queries git), the relay's complexity is not justified for that class and this bench demotes the claim rather than defending it.",
		InvalidatingUnknown: "The single assumption most likely to flip the result is #1: that compaction does not re-derive dropped facts from the durable store. If it does, the fidelity advantage collapses and only the peak-context and cache advantages remain.",
	}
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
