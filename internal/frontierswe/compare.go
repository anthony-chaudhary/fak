package frontierswe

import (
	"fmt"
	"strings"
)

// This file is the C12 raw-vs-fak COMPARE (epic #1706, issue #1718): the surface
// that folds two graded arms — the raw harness and the same harness routed through
// fak — into the one governed table a FrontierSWE completion-time row is recorded
// from. It composes the two gates that already exist so the honest ordering the
// runbook (FRONTIERSWE-TTS-RUNBOOK.md §8) mandates is enforced in code, not prose:
//
//  1. C11 score-parity FIRST (score_parity.go). fak must not regress the raw arm's
//     score distribution. A faster arm that scored lower is a regression, not a win,
//     so if parity fails the compare STOPS at PARITY_FAILED and emits no ratio.
//  2. Only then the C14 time-to-solution metric (tts_metric.go) per trial, folded to
//     a per-arm mean over the SOLVED trials, and the ratio T_fak/T_raw.
//
// The second load-bearing rule is provenance. A ratio built from a mocked run's
// PROJECTED wall-clock is a deterministic floor, never a measured win — so the
// verdict vocabulary itself carries the distinction (MEASURED_WIN vs PROJECTED_WIN),
// and a projected comparison can never be misread as a witnessed completion-time
// number. The compare fabricates nothing: no solved trial in an arm ⇒ GATED, parity
// regressed ⇒ PARITY_FAILED, and a ratio is emitted only when both arms actually
// reached a solution under a passing parity gate.

// CompareSchema is the versioned id stamped on the fak.frontierswe.compare.v1
// report emitted by `fak frontierswe compare`.
const CompareSchema = "fak.frontierswe.compare.v1"

// Compare verdict vocabulary. The provenance is baked into the WIN/NO_WIN verdicts
// so a caller can never quote a projected floor as a measured win by reading only
// the verdict field.
const (
	VerdictParityFailed   = "PARITY_FAILED"   // fak regressed raw's score distribution; no TTS claim
	VerdictGated          = "GATED"           // parity ok but an arm solved 0 trials — no time-to-solution to compare
	VerdictMeasuredWin    = "MEASURED_WIN"    // parity ok, ratio < 1 from measured trajectories
	VerdictMeasuredNoWin  = "MEASURED_NO_WIN" // parity ok, measured ratio >= 1 (no win)
	VerdictProjectedWin   = "PROJECTED_WIN"   // parity ok, ratio < 1 but from projected (mocked) wall-clock — a FLOOR, not a measured win
	VerdictProjectedNoWin = "PROJECTED_NO_WIN"
)

// Per-arm provenance rollup over the solved trials.
const provenanceMixed = "mixed" // an arm mixes measured and projected solved trials
const provenanceNone = "none"   // an arm solved no trials, so it has no wall-clock provenance

// GradedTrial is one arm's trial as C12 consumes it: the C11 score (correctness /
// speedup / gated score) plus the C14 TTS trace the trial was driven over, and
// whether that trace's wall-clock was projected (mocked run) rather than measured.
type GradedTrial struct {
	Score  TrialScore `json:"score"`
	Trace  TTSTrace   `json:"trace"`
	Mocked bool       `json:"mocked"`
}

// ArmTTS is the per-arm aggregate: the C14 metric of every trial, and the mean /
// best wall-clock over the SOLVED trials only (a censored trial contributes no time).
type ArmTTS struct {
	Trials        int         `json:"trials"`
	ReachedTrials int         `json:"reached_trials"`  // trials that solved (correctness 1.0)
	MeanWallSec   float64     `json:"mean_wall_sec"`   // mean wall-clock over solved trials (0 if none)
	BestWallSec   float64     `json:"best_wall_sec"`   // min wall-clock over solved trials (0 if none)
	MeanTurns     float64     `json:"mean_turns"`      // mean turns-to-first-correct over solved trials
	MeanReuseRate float64     `json:"mean_reuse_rate"` // C8: mean realized cross-turn reuse rate over the arm's trials (whole-trajectory, not solved-only)
	Provenance    string      `json:"provenance"`      // measured | projected | mixed | none
	Metrics       []TTSMetric `json:"metrics"`         // per-trial C14 metrics, for inspection
}

// CompareReport is the fak.frontierswe.compare.v1 payload: the C11 parity gate, the
// per-arm C14 aggregates, and — only when parity passes and both arms solved — the
// TTS ratio and a provenance-tagged verdict. TTSRatio is nil whenever a ratio must
// not be claimed (parity failed, or an arm solved nothing).
type CompareReport struct {
	Schema     string            `json:"schema"`
	Task       string            `json:"task"`
	Parity     ScoreParityReport `json:"parity"`                   // C11 — evaluated FIRST (carries the C3 Avg/Best/correct distribution)
	Raw        ArmTTS            `json:"raw"`                      // C14 aggregate, raw arm
	Fak        ArmTTS            `json:"fak"`                      // C14 aggregate, fak arm
	TTSRatio   *float64          `json:"tts_ratio,omitempty"`      // T_fak/T_raw, only when parity ok AND both arms solved
	Provenance string            `json:"provenance"`               // the ratio's provenance: measured | projected (absent when no ratio)
	FloorRatio *float64          `json:"c4_floor_ratio,omitempty"` // C4 deterministic floor T_fak/T_raw at the fak arm's realized reuse — the projection the measured ratio is checked against (#1718)
	OverClaim  bool              `json:"over_claim,omitempty"`     // a MEASURED ratio below the C4 floor — physically suspicious, surfaced not hidden
	Verdict    string            `json:"verdict"`
	Headline   string            `json:"headline"`
}

// CompareArms folds a raw arm and a fak arm into the C12 report, enforcing the
// runbook ordering: parity gate first, TTS ratio second, and only from solved
// trials under a passing gate. tolerance is the C11 score tolerance (<=0 ⇒ default).
func CompareArms(task string, raw, fak []GradedTrial, tolerance float64) CompareReport {
	rep := CompareReport{
		Schema: CompareSchema,
		Task:   task,
		Parity: ScoreParity(scoresOf(raw), scoresOf(fak), tolerance),
		Raw:    aggregateArm(task, raw),
		Fak:    aggregateArm(task, fak),
	}

	// The C4 deterministic floor is a projection independent of the parity/gated
	// verdict, so it is computed once up front and always carried: T_fak/T_raw =
	// C(r)/A projected from the fak arm's budget geometry at its realized reuse rate.
	// It is the number the measured ratio is checked against (#1718); over-claim (a
	// MEASURED ratio below it) is flagged in the ratio branch below.
	rep.FloorRatio = c4FloorRatio(task, fak, rep.Fak.MeanReuseRate)

	// 1) Parity FIRST. A regressed score distribution stops the compare — there is
	// no time-to-solution win to read, only a regression to fix.
	if !rep.Parity.Passed {
		rep.Verdict = VerdictParityFailed
		rep.Headline = fmt.Sprintf("parity failed (%s): %s — no TTS ratio claimed",
			rep.Parity.Reason, strings.Join(rep.Parity.Failures, "; "))
		return rep
	}

	// 2) A ratio needs a solved trial to time in BOTH arms. Absent one, the compare
	// is honestly GATED — never a fabricated ratio.
	if rep.Raw.ReachedTrials == 0 || rep.Fak.ReachedTrials == 0 {
		rep.Verdict = VerdictGated
		rep.Headline = fmt.Sprintf("parity ok but no time-to-solution to compare (raw solved %d/%d, fak solved %d/%d)",
			rep.Raw.ReachedTrials, rep.Raw.Trials, rep.Fak.ReachedTrials, rep.Fak.Trials)
		return rep
	}

	// 3) Ratio from the per-arm mean wall-clock over solved trials.
	ratio := rep.Fak.MeanWallSec / rep.Raw.MeanWallSec
	rep.TTSRatio = &ratio
	rep.Provenance = comparisonProvenance(rep.Raw.Provenance, rep.Fak.Provenance)
	// Over-claim: a MEASURED ratio that dips below the deterministic C4 floor is
	// physically suspicious (prefix reuse alone cannot beat the floor), so surface it
	// rather than bank the bigger number. A PROJECTED ratio IS the floor's own family,
	// so it is never flagged — only a measurement can over-claim against the projection.
	if rep.Provenance == ProvenanceMeasured && rep.FloorRatio != nil && ratio+tolClamp(tolerance) < *rep.FloorRatio {
		rep.OverClaim = true
	}
	win := ratio+tolClamp(tolerance) < 1.0
	rep.Verdict = verdictFor(rep.Provenance, win)
	rep.Headline = compareHeadline(rep.Provenance, win, ratio, rep.Raw.MeanWallSec, rep.Fak.MeanWallSec)
	return rep
}

// c4FloorRatio projects the deterministic C4 time-to-solution floor T_fak/T_raw =
// C(r)/A for the arm's task, from the arm's budget geometry (the [agent] timeout_sec
// carried on its TTS trace) at realized reuse rate r. It is the projection the
// measured ratio is checked against (#1718), reusing the same C4 TTSModel the
// describe surface projects with. Returns nil when no trial carries a budget to
// project from — a floor is never fabricated from a missing budget.
func c4FloorRatio(task string, arm []GradedTrial, reuse float64) *float64 {
	for _, t := range arm {
		if t.Trace.BudgetSec > 0 {
			ft := &Task{Name: task}
			ft.Agent.TimeoutSec = float64(t.Trace.BudgetSec)
			floor := ProjectTTS(ft, reuse, nil).Arms.TTSRatio
			return &floor
		}
	}
	return nil
}

// scoresOf projects the arm's trials onto the C11 TrialScore slice the parity gate
// reads.
func scoresOf(arm []GradedTrial) []TrialScore {
	out := make([]TrialScore, len(arm))
	for i, t := range arm {
		out[i] = t.Score
	}
	return out
}

// aggregateArm computes the C14 per-trial metrics and the per-arm mean/best over the
// solved trials only. A censored trial (Reached=false) contributes no wall-clock, so
// the mean is a true mean-time-to-solution over the trials that actually solved.
func aggregateArm(task string, arm []GradedTrial) ArmTTS {
	a := ArmTTS{Trials: len(arm), Metrics: make([]TTSMetric, 0, len(arm))}
	var wallSum, turnSum, reuseSum float64
	best := 0.0
	sawMeasured, sawProjected := false, false
	for _, t := range arm {
		// C8 realized reuse is a whole-trajectory property (it does not depend on
		// whether the trial solved), so it is summed over every trial, not just solved.
		reuseSum += t.Trace.CacheSeries.RealizedReuseRate
		m := TTSMetricOf(task, t.Score.ID, t.Score.Correctness, t.Trace, t.Mocked)
		a.Metrics = append(a.Metrics, m)
		if !m.Reached {
			continue
		}
		a.ReachedTrials++
		wallSum += m.WallSecToCorrect
		turnSum += float64(m.TurnsToCorrect)
		if a.ReachedTrials == 1 || m.WallSecToCorrect < best {
			best = m.WallSecToCorrect
		}
		switch m.Provenance {
		case ProvenanceMeasured:
			sawMeasured = true
		case ProvenanceProjected:
			sawProjected = true
		}
	}
	if a.ReachedTrials > 0 {
		a.MeanWallSec = wallSum / float64(a.ReachedTrials)
		a.MeanTurns = turnSum / float64(a.ReachedTrials)
		a.BestWallSec = best
	}
	if len(arm) > 0 {
		a.MeanReuseRate = reuseSum / float64(len(arm))
	}
	a.Provenance = armProvenance(sawMeasured, sawProjected)
	return a
}

// armProvenance rolls a set of solved trials' provenance into one label: measured if
// every solved trial was measured, projected if every one was projected, mixed if
// both appear, and none when the arm solved nothing.
func armProvenance(sawMeasured, sawProjected bool) string {
	switch {
	case sawMeasured && sawProjected:
		return provenanceMixed
	case sawMeasured:
		return ProvenanceMeasured
	case sawProjected:
		return ProvenanceProjected
	default:
		return provenanceNone
	}
}

// comparisonProvenance is the provenance of the RATIO across both arms: measured
// only when both arms are wholly measured. Any projection or mix on either side
// makes the ratio a projected floor — the conservative label that keeps a mocked
// number from ever reading as a measured win.
func comparisonProvenance(raw, fak string) string {
	if raw == ProvenanceMeasured && fak == ProvenanceMeasured {
		return ProvenanceMeasured
	}
	return ProvenanceProjected
}

// verdictFor picks the provenance-tagged win/no-win verdict so the verdict field
// alone can never be misquoted: a projected floor is never a MEASURED_WIN.
func verdictFor(provenance string, win bool) string {
	if provenance == ProvenanceMeasured {
		if win {
			return VerdictMeasuredWin
		}
		return VerdictMeasuredNoWin
	}
	if win {
		return VerdictProjectedWin
	}
	return VerdictProjectedNoWin
}

func compareHeadline(provenance string, win bool, ratio, rawWall, fakWall float64) string {
	kind := "measured"
	if provenance != ProvenanceMeasured {
		kind = "PROJECTED floor (not a measured win)"
	}
	if win {
		return fmt.Sprintf("%s: fak reached the same reward in %.3fx the wall-clock (%.1fs vs %.1fs) — parity green",
			kind, ratio, fakWall, rawWall)
	}
	return fmt.Sprintf("%s: no win — fak wall-clock %.3fx raw (%.1fs vs %.1fs), parity green",
		kind, ratio, fakWall, rawWall)
}

// tolClamp mirrors ScoreParity's tolerance defaulting so the win test uses the same
// epsilon the parity gate did.
func tolClamp(tolerance float64) float64 {
	if tolerance <= 0 {
		return DefaultScoreParityTol
	}
	return tolerance
}

// RenderCompareMarkdown renders the C12 report as the raw-vs-fak table a
// FRONTIERSWE-RESULTS.md row is recorded from. Parity and provenance are stated
// before the ratio so the table reads in the honest order.
func RenderCompareMarkdown(r CompareReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# FrontierSWE raw-vs-fak — %s\n\n", r.Task)
	fmt.Fprintf(&b, "**Verdict:** %s\n\n", r.Verdict)
	fmt.Fprintf(&b, "%s\n\n", r.Headline)

	fmt.Fprintf(&b, "## Parity gate (C11 — evaluated first)\n\n")
	fmt.Fprintf(&b, "- passed: **%t**\n", r.Parity.Passed)
	if !r.Parity.Passed {
		for _, f := range r.Parity.Failures {
			fmt.Fprintf(&b, "  - %s\n", f)
		}
	}
	// One side-by-side row per arm carrying the four families the issue names: the C3
	// gated score (avg/best + correct X/N), the C14 measured time-to-solution (solved
	// trials only), and the C8 realized reuse rate — plus the provenance label.
	fmt.Fprintf(&b, "\n## Score, time-to-solution, and reuse (per arm)\n\n")
	fmt.Fprintf(&b, "| arm | avg score | best score | correct X/N | solved | mean wall (s) | mean turns | realized reuse (C8) | provenance |\n")
	fmt.Fprintf(&b, "|---|---|---|---|---|---|---|---|---|\n")
	writeArmRow(&b, "raw", r.Parity.Raw, r.Raw)
	writeArmRow(&b, "fak", r.Parity.Fak, r.Fak)
	b.WriteString("\n")
	if r.TTSRatio != nil {
		fmt.Fprintf(&b, "**TTS ratio (T_fak/T_raw):** %.4f  (%s)\n", *r.TTSRatio, r.Provenance)
	} else {
		fmt.Fprintf(&b, "**TTS ratio:** — (not claimed: %s)\n", r.Verdict)
	}
	// The projection-vs-measurement column: the C4 deterministic floor beside the
	// measured ratio, so an over-claim (a measurement beating the floor) is visible.
	if r.FloorRatio != nil {
		fmt.Fprintf(&b, "**C4 floor (projected T_fak/T_raw):** %.4f — the projection the measurement is checked against\n", *r.FloorRatio)
		if r.OverClaim {
			fmt.Fprintf(&b, "> **OVER-CLAIM:** the measured ratio is below the deterministic C4 floor — re-check the measurement.\n")
		}
	}
	return b.String()
}

func writeArmRow(b *strings.Builder, name string, s ScoreDistribution, a ArmTTS) {
	fmt.Fprintf(b, "| %s | %.4f | %.4f | %d/%d | %d | %.1f | %.1f | %.4f | %s |\n",
		name, s.AvgScore, s.BestScore, s.CorrectCount, s.Trials, a.ReachedTrials, a.MeanWallSec, a.MeanTurns, a.MeanReuseRate, a.Provenance)
}
