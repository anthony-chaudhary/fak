package frontierswe

// This file is the C14 time-to-solution METRIC (epic #1706, issue #1720): the
// per-trial reduction of a graded trajectory to the two numbers the FrontierSWE
// completion-time thesis is scored on —
//
//   - wall-clock-to-`correctness`-1.0 — the wall-clock a trial spent to reach a
//     solved reward (correctness 1.0). FrontierSWE grades a trial once, at the end,
//     so a solved trial reached correctness 1.0 at its final turn: the metric is the
//     trial's total wall-clock. A trial that never scored correctness 1.0 has no
//     time-to-solution — it is CENSORED (Reached=false), never folded in as a 0.
//   - turns-to-first-correct — the same, in turns: the trial's turn count when it
//     solved, else censored.
//
// The load-bearing honesty boundary is Provenance. On a mocked run the trajectory's
// wall-clock is PROJECTED from the [agent] budget, not measured (see run.go's
// TTSTrace doc). This metric carries that flag through unchanged so the C12 compare
// can never dress a projected floor up as a measured win. C14 computes nothing about
// whether a trial solved — that is the C3 scorer's correctness, handed in — it only
// projects a graded trial onto its (reached, wall, turns, provenance) tuple.

// TTSMetricSchema is the versioned id stamped on a fak.frontierswe.tts-metric.v1
// per-trial metric, so a reduced trial is inspectable and machine-joinable by C12.
const TTSMetricSchema = "fak.frontierswe.tts-metric.v1"

// Provenance labels whether a TTS number was measured on a real trajectory or
// projected from the budget on a mocked run. The distinction is the whole honesty
// story: a projected ratio is a deterministic floor, never a completion-time win.
const (
	ProvenanceMeasured  = "measured"  // a real trajectory's wall-clock was observed
	ProvenanceProjected = "projected" // a mocked run: wall-clock derived from the budget
)

// TTSMetric is the C14 per-trial reduction: whether the trial reached a solved
// reward, and if so the wall-clock and turn count it took. Reached is false for a
// censored trial (correctness never hit 1.0); its wall/turns stay 0 and MUST NOT be
// read as a fast solution. Provenance says whether the wall-clock was measured.
type TTSMetric struct {
	Schema           string  `json:"schema"`
	Task             string  `json:"task"`
	TrialID          string  `json:"trial_id,omitempty"`
	Reached          bool    `json:"reached"`             // correctness hit 1.0 — a solution exists to time
	Correctness      float64 `json:"correctness"`         // the C3 correctness handed in
	WallSecToCorrect float64 `json:"wall_sec_to_correct"` // wall-clock to correctness 1.0 (0 when censored)
	TurnsToCorrect   int     `json:"turns_to_correct"`    // turns to first correct (0 when censored)
	Provenance       string  `json:"provenance"`          // measured | projected
}

// correctnessSolved is the C3 threshold at which a trial counts as a reached
// solution. FrontierSWE's implementation/ml_research families score correctness in
// [0,1]; a full solution is 1.0. The epsilon mirrors the parity gate's tolerance so
// the two gates agree on what "correct" means.
func correctnessSolved(correctness float64) bool {
	return correctness+DefaultScoreParityTol >= 1.0
}

// TTSMetricOf reduces one graded trial to its C14 metric. correctness is the C3
// number (from eval / ScoreRewardTrial); trace is the trial's per-turn TTS trace
// (run.go); mocked flags a projected-wall-clock run. A trial that solved
// (correctness 1.0) is timed at its full trajectory — total wall-clock and turn
// count — because the grade is known only at the trajectory's end. A censored trial
// reports Reached=false with zero wall/turns, never a fabricated fast solution.
func TTSMetricOf(task, trialID string, correctness float64, trace TTSTrace, mocked bool) TTSMetric {
	prov := ProvenanceMeasured
	if mocked {
		prov = ProvenanceProjected
	}
	m := TTSMetric{
		Schema:      TTSMetricSchema,
		Task:        task,
		TrialID:     trialID,
		Correctness: correctness,
		Provenance:  prov,
	}
	if correctnessSolved(correctness) {
		m.Reached = true
		m.WallSecToCorrect = trace.TotalWallSec
		m.TurnsToCorrect = trace.Turns
	}
	return m
}
