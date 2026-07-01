package frontierswe

import (
	"fmt"
	"math"
)

const (
	ScoreParitySchema       = "fak.frontierswe.score-parity.v1"
	ScoreParityFailedReason = "FRONTIERSWE_SCORE_PARITY_FAILED"
	DefaultScoreParityTol   = 1e-9
)

// TrialScore is one scored FrontierSWE trial. It is intentionally post-scorer:
// C3 owns reward.json -> correctness/speedup/leaderboard score; this gate owns
// the distribution predicate over raw vs fak trials.
type TrialScore struct {
	ID            string   `json:"id,omitempty"`
	Task          string   `json:"task"`
	Correctness   float64  `json:"correctness"`
	Speedup       *float64 `json:"speedup,omitempty"`
	Score         float64  `json:"score"`
	AntiCheatFlag bool     `json:"anti_cheat_flag,omitempty"`
}

type ScoreParityReport struct {
	Schema    string            `json:"schema"`
	Passed    bool              `json:"passed"`
	Reason    string            `json:"reason,omitempty"`
	Tolerance float64           `json:"tolerance"`
	Predicate []string          `json:"predicate"`
	Raw       ScoreDistribution `json:"raw"`
	Fak       ScoreDistribution `json:"fak"`
	Failures  []string          `json:"failures,omitempty"`
}

type ScoreDistribution struct {
	Trials          int      `json:"trials"`
	AvgScore        float64  `json:"avg_score"`
	BestScore       float64  `json:"best_score"`
	AvgCorrectness  float64  `json:"avg_correctness"`
	BestCorrectness float64  `json:"best_correctness"`
	CorrectCount    int      `json:"correct_count"`
	SpeedupTrials   int      `json:"speedup_trials"`
	AvgSpeedup      *float64 `json:"avg_speedup,omitempty"`
	BestSpeedup     *float64 `json:"best_speedup,omitempty"`
}

// ScoreRewardTrial is the C3 scorer plus the C11 trial wrapper.
func ScoreRewardTrial(id, task string, reward *Reward, antiCheatFlagged bool) TrialScore {
	correctness, speedup := ExtractScore(reward, task)
	return TrialScore{
		ID: id, Task: task, Correctness: correctness, Speedup: speedup,
		Score:         GatedScoreAntiCheat(correctness, speedup, task, antiCheatFlagged),
		AntiCheatFlag: antiCheatFlagged,
	}
}

// ScoreParity is the FrontierSWE integrity gate before any time-to-solution
// claim: fak must not regress the raw arm's distribution. The exact predicate is:
// fak AvgScore >= raw AvgScore, fak BestScore >= raw BestScore, fak CorrectCount
// >= raw CorrectCount, and when the raw arm has full-correct speedup trials, fak
// AvgSpeedup and BestSpeedup must not regress either.
func ScoreParity(raw, fak []TrialScore, tolerance float64) ScoreParityReport {
	if tolerance <= 0 {
		tolerance = DefaultScoreParityTol
	}
	r := ScoreParityReport{
		Schema:    ScoreParitySchema,
		Tolerance: tolerance,
		Raw:       summarizeScores(raw),
		Fak:       summarizeScores(fak),
		Predicate: []string{
			"fak.avg_score >= raw.avg_score",
			"fak.best_score >= raw.best_score",
			"fak.correct_count >= raw.correct_count",
			"if raw has full-correct speedup trials: fak.avg_speedup >= raw.avg_speedup and fak.best_speedup >= raw.best_speedup",
		},
	}
	r.Failures = scoreParityFailures(r.Raw, r.Fak, tolerance)
	r.Passed = len(r.Failures) == 0
	if !r.Passed {
		r.Reason = ScoreParityFailedReason
	}
	return r
}

func summarizeScores(trials []TrialScore) ScoreDistribution {
	d := ScoreDistribution{Trials: len(trials)}
	if len(trials) == 0 {
		return d
	}
	bestScore := math.Inf(-1)
	bestCorrectness := math.Inf(-1)
	var scoreSum, corrSum, speedSum float64
	var bestSpeed *float64
	for _, t := range trials {
		scoreSum += t.Score
		corrSum += t.Correctness
		if t.Score > bestScore {
			bestScore = t.Score
		}
		if t.Correctness > bestCorrectness {
			bestCorrectness = t.Correctness
		}
		if t.Correctness >= 1.0 {
			d.CorrectCount++
			if t.Speedup != nil {
				d.SpeedupTrials++
				speedSum += *t.Speedup
				if bestSpeed == nil || *t.Speedup > *bestSpeed {
					v := *t.Speedup
					bestSpeed = &v
				}
			}
		}
	}
	d.AvgScore = scoreSum / float64(len(trials))
	d.BestScore = bestScore
	d.AvgCorrectness = corrSum / float64(len(trials))
	d.BestCorrectness = bestCorrectness
	if d.SpeedupTrials > 0 {
		avg := speedSum / float64(d.SpeedupTrials)
		d.AvgSpeedup = &avg
		d.BestSpeedup = bestSpeed
	}
	return d
}

func scoreParityFailures(raw, fak ScoreDistribution, tol float64) []string {
	var failures []string
	if raw.Trials == 0 {
		failures = append(failures, "raw arm has no trials")
	}
	if fak.Trials == 0 {
		failures = append(failures, "fak arm has no trials")
	}
	if fak.AvgScore+tol < raw.AvgScore {
		failures = append(failures, fmt.Sprintf("avg_score regressed: fak %.6f < raw %.6f", fak.AvgScore, raw.AvgScore))
	}
	if fak.BestScore+tol < raw.BestScore {
		failures = append(failures, fmt.Sprintf("best_score regressed: fak %.6f < raw %.6f", fak.BestScore, raw.BestScore))
	}
	if fak.CorrectCount < raw.CorrectCount {
		failures = append(failures, fmt.Sprintf("correct_count regressed: fak %d < raw %d", fak.CorrectCount, raw.CorrectCount))
	}
	if raw.SpeedupTrials > 0 {
		switch {
		case fak.SpeedupTrials == 0:
			failures = append(failures, "speedup regressed: raw has full-correct speedup trials but fak has none")
		case ptrLess(fak.AvgSpeedup, raw.AvgSpeedup, tol):
			failures = append(failures, fmt.Sprintf("avg_speedup regressed: fak %.6f < raw %.6f", deref(fak.AvgSpeedup), deref(raw.AvgSpeedup)))
		case ptrLess(fak.BestSpeedup, raw.BestSpeedup, tol):
			failures = append(failures, fmt.Sprintf("best_speedup regressed: fak %.6f < raw %.6f", deref(fak.BestSpeedup), deref(raw.BestSpeedup)))
		}
	}
	return failures
}

func ptrLess(a, b *float64, tol float64) bool {
	if b == nil {
		return false
	}
	if a == nil {
		return true
	}
	return *a+tol < *b
}

func deref(p *float64) float64 {
	if p == nil {
		return 0
	}
	return *p
}
