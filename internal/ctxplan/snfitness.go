package ctxplan

// snfitness.go — issue #867, the RSI FITNESS AXIS over witnessed attention-S/N.
//
// #858 (outcome_attention.go) closed the planner's LEARNING half on a witnessed reward:
// Forecast.Learn / Weights.Learn now train on attention attribution (OutcomeFromAttention /
// LearnFromAttention), not lexical-overlap guesses. This file closes the other half the #867
// build names — the KEEP half: it turns the witnessed attention-S/N (signalnoise.go,
// SignalNoiseFromAttention) into the single scalar FITNESS an rsiloop candidate is scored on, so
// a candidate planner/forecast change is KEPT only if it raises witnessed attention-S/N across a
// session — gated, like every fak RSI keep, by the truth-clean floor the rsiloop harness applies
// on top (internal/rsiloop, Harness.Measure → shipgate.Evaluate). The reward is witnessed FROM
// THE MODEL, so the keep-bit closes on real evidence, not a self-report.
//
// SHADOW BY CONSTRUCTION. WitnessedSNFitness is a PURE offline replay: it scores a candidate
// forecast over a RECORDED session and changes no live plan. Adopting a kept forecast into the
// live planner is a separate flag flip — the #858 two-posture honesty split, and the gate the
// #860 epic holds until experiment 1 (#866) shows the reward correlates with exact-eviction
// leave-one-out ground truth. Building the shadow measurement is the step that PRECEDES that
// flip, not one that pre-empts it: nothing here is on the planning path.
//
// rsiloop stays subsystem-agnostic — its engine treats Candidate.Payload opaquely so the
// keep/revert logic is independent of WHAT is tuned — so the metric belongs here, in ctxplan,
// and a driver wires it in one line:
//
//	h := rsiloop.Harness{
//	    MetricName:  "attention_sn",
//	    LowerBetter: false, // higher witnessed S/N is better
//	    BaselineMetric: func() (float64, string, error) { return WitnessedSNFitness(baseline, sess), ref, nil },
//	    Candidates:  func() []rsiloop.Candidate { /* one per proposed Forecast (e.g. LearnFromAttention's output) */ },
//	    Measure: func(c rsiloop.Candidate) (rsiloop.Measurement, error) {
//	        f := c.Payload.(Forecast)
//	        return rsiloop.Measurement{Metric: WitnessedSNFitness(f, sess), SuiteGreen: green, TruthClean: clean}, nil
//	    },
//	}
//
// shipgate.Evaluate then KEEPS a candidate only when its Metric strictly beats the running
// baseline (a real, turn-over-turn S/N gain) AND the suite is green AND the worktree is
// truth-clean. The rsiloop Journal already tracks that metric against `main` over time and the
// breaker early-exits a bad streak — the multi-session S/N trend the #867 acceptance calls for.

// Turn is one turn of a RECORDED session — the unit WitnessedSNFitness replays a candidate
// forecast over. It carries the candidate Spans available that turn, the resident Budget, the
// WITNESSED per-span attention Attribution (rung 2, signalnoise.go), and the demand-page Faults
// (elided spans the turn paged back — the under-resident axis, read exactly as the boolean path
// reads it). It is a self-contained record a recorder (cmd/ctxplanbench, a session replay) fills
// and the fitness function reads; nothing here reaches into kvmmu or the live planner.
type Turn struct {
	Spans       []Span      `json:"spans"`
	Budget      Budget      `json:"budget"`
	Attribution Attribution `json:"attribution,omitempty"`
	Faults      []string    `json:"faults,omitempty"`
}

// WitnessedSNFitness is the RSI fitness of a candidate Forecast (which EMBEDS its Weights, so one
// value captures a whole planner/forecast change) over a recorded session: the MEAN, across the
// session's turns, of the turn's witnessed attention-S/N Ratio DISCOUNTED by its under-resident
// FaultRatio — a number in [0,1], higher is better. For each turn it replays the candidate's plan
// (PlanCells — the same pure planner the live path runs) and scores the resident view with
// SignalNoiseFromAttention over the turn's WITNESSED attention mass, so the fitness rewards a
// forecast that keeps the spans the model actually attended to and penalizes one that keeps idle
// noise.
//
// The fault discount is the anti-gaming guard signalnoise.go's two-axis design mandates: a high
// Ratio earned by ELIDING needed spans is not a real win — it just moves their cost onto the
// fault axis, where this discount cancels it (FaultRatio→1 ⇒ factor→0). At zero faults the factor
// is 1, so the fitness is EXACTLY the mean witnessed attention-S/N; the discount only bites when a
// candidate starved the window. That makes "raise S/N" non-gameable by dropping needed spans —
// the property the rsiloop KEEP gate needs to fire only on a REAL S/N gain.
//
// It is the scalar an rsiloop Measure returns (see the file header). Pure and deterministic — the
// same (forecast, session) yields the same fitness, so a replay is a gate (re-run, diff empty). An
// empty session returns 1.0, the empty-window fail-to-best convention SignalNoise.Ratio already
// takes (no turns, no noise — nothing to curate).
func WitnessedSNFitness(f Forecast, session []Turn) float64 {
	if len(session) == 0 {
		return 1.0
	}
	var sum float64
	for _, t := range session {
		p := PlanCells(t.Spans, f, t.Budget, nil)
		sn := SignalNoiseFromAttention(p, t.Attribution, Outcome{Faults: t.Faults})
		// Ratio is the resident signal share; (1-FaultRatio) is the under-resident discount.
		// Both are already in [0,1] (Ratio by construction, FaultRatio by definition), so the
		// product stays in [0,1] and reduces to the pure witnessed S/N when nothing faulted.
		sum += sn.Ratio() * (1 - sn.FaultRatio())
	}
	return sum / float64(len(session))
}
