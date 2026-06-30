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

// WitnessedSNScore is the structured form of WitnessedSNFitness. Fitness is the
// scalar RSI keep metric (higher is better); the remaining fields are the score
// surface operators and sibling controls need to understand WHY the scalar moved.
// A no-evidence session uses the same fail-to-best convention as SignalNoise.Ratio:
// Fitness=1, MeanRatio=1, MeanFaultRatio=0, Grade="lean".
type WitnessedSNScore struct {
	Fitness           float64 `json:"fitness"`
	MeanRatio         float64 `json:"mean_ratio"`
	MeanFaultRatio    float64 `json:"mean_fault_ratio"`
	Grade             string  `json:"grade"`
	Turns             int     `json:"turns"`
	ScoredTurns       int     `json:"scored_turns"`
	SignalTokens      int     `json:"signal_tokens"`
	NoiseTokens       int     `json:"noise_tokens"`
	UnaccountedTokens int     `json:"unaccounted_tokens,omitempty"`
	FaultTokens       int     `json:"fault_tokens"`
	ResidentTokens    int     `json:"resident_tokens"`
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
// same (forecast, session) yields the same fitness, so a replay is a gate (re-run, diff empty).
// Turns with no attention witness and no faults are neutral: they carry no S/N evidence, so they
// are excluded from the mean instead of being misread as all-noise. A fault-only turn still
// contributes the under-resident pressure (1-FaultRatio), because demand paging is itself a
// witness. An empty or fully unwitnessed/no-fault session returns 1.0, the empty-window
// fail-to-best convention SignalNoise.Ratio already takes (no evidence, no noise — nothing to
// curate).
func WitnessedSNFitness(f Forecast, session []Turn) float64 {
	return ScoreWitnessedSN(f, session).Fitness
}

// ScoreWitnessedSN computes the full witnessed S/N scorecard that WitnessedSNFitness
// projects down to one scalar. It is pure and deterministic; the scalar fitness it
// returns is exactly the historical WitnessedSNFitness definition:
//
//	attention turn: ratio * (1 - fault_ratio)
//	fault-only turn: 1 - fault_ratio
//	no-witness/no-fault turn: skipped as neutral evidence
//
// The MeanRatio/MeanFaultRatio fields follow the same evidence convention. A
// fault-only turn has ratio=1 for scoring purposes (ratio evidence is absent, not
// bad), while its fault pressure still contributes to MeanFaultRatio and can grade
// the score "starving".
func ScoreWitnessedSN(f Forecast, session []Turn) WitnessedSNScore {
	score := WitnessedSNScore{Turns: len(session), Fitness: 1.0, MeanRatio: 1.0, Grade: "lean"}
	if len(session) == 0 {
		return score
	}
	var sum float64
	var ratioSum float64
	var faultRatioSum float64
	scored := 0
	for _, t := range session {
		p := PlanCells(t.Spans, f, t.Budget, nil)
		sn := SignalNoiseFromAttention(p, t.Attribution, Outcome{Faults: t.Faults})
		if len(t.Attribution) == 0 {
			if len(t.Faults) == 0 {
				continue
			}
			ratio := 1.0
			faultRatio := sn.FaultRatio()
			sum += ratio * (1 - faultRatio)
			ratioSum += ratio
			faultRatioSum += faultRatio
			score.addSignalNoise(sn)
			scored++
			continue
		}
		// Ratio is the resident signal share; (1-FaultRatio) is the under-resident discount.
		// Both are already in [0,1] (Ratio by construction, FaultRatio by definition), so the
		// product stays in [0,1] and reduces to the pure witnessed S/N when nothing faulted.
		ratio := sn.Ratio()
		faultRatio := sn.FaultRatio()
		sum += ratio * (1 - faultRatio)
		ratioSum += ratio
		faultRatioSum += faultRatio
		score.addSignalNoise(sn)
		scored++
	}
	if scored == 0 {
		return score
	}
	score.ScoredTurns = scored
	score.Fitness = sum / float64(scored)
	score.MeanRatio = ratioSum / float64(scored)
	score.MeanFaultRatio = faultRatioSum / float64(scored)
	score.Grade = gradeWitnessedSNScore(score.MeanRatio, score.MeanFaultRatio)
	return score
}

func (s *WitnessedSNScore) addSignalNoise(sn SignalNoise) {
	s.SignalTokens += sn.SignalTokens
	s.NoiseTokens += sn.NoiseTokens
	s.UnaccountedTokens += sn.UnaccountedTokens
	s.FaultTokens += sn.FaultTokens
	s.ResidentTokens += sn.ResidentTokens
}

func gradeWitnessedSNScore(ratio, faultRatio float64) string {
	if faultRatio > 0.25 {
		return "starving"
	}
	if ratio < 0.5 {
		return "bloated"
	}
	if ratio >= 0.8 && faultRatio <= 0.1 {
		return "lean"
	}
	return "ok"
}
