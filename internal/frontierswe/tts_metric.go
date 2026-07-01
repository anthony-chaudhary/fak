package frontierswe

// This file is the C14 measured time-to-solution recorder (epic #1706, #1720) —
// the MEASURED value number the whole epic exists to produce. Where geometry.go
// (C4) and tts_describe.go project a deterministic TTS floor from a task's budget,
// and cachewitness_series.go (C8) folds a live trial's /metrics scrapes into the
// realized reuse rate r, THIS file consumes a completed run's per-turn trace (C9)
// plus that C8 reuse series and turns them into the operationally decisive number
// FrontierSWE's own leaderboard does NOT report: how long — in wall-clock and in
// turns — each arm took to reach a given correctness milestone.
//
// The natural, FrontierSWE-native milestone is correctness == 1.0 (the gate that
// unlocks the speedup tier in the reference SCORING). For tasks that rarely reach
// 1.0, RecordTTS falls back to a configurable fixed-quality milestone (e.g.
// correctness >= 0.5) so a per-arm number still exists. Both are recorded so a
// reader always sees which milestone the wall-clock number was measured against.
//
// MEASURED vs PROJECTION (the honesty boundary this package states everywhere).
//   - MEASURED: every wall-clock and turn number here comes straight from the run
//     trace (C9) and the reuse series (C8). WallClockToMilestone and
//     TurnsToFirstCorrect are read off the trajectory, not modeled — they are the
//     measurement, not a floor.
//   - PROJECTION: ProjectedTTSRatio is the C4 floor re-evaluated at the arm's
//     REALIZED reuse rate (from C8), recorded BESIDE the measured cross-arm ratio
//     so any over-claim (the projection promising a bigger speedup than the run
//     actually delivered) is visible in one place — the epic's over-/under-claim
//     requirement.
//
// Distribution handling. One TTSArmTrace is one trajectory. FrontierSWE runs
// n_concurrent_trials of a task, so an arm has a DISTRIBUTION of wall-clock-to-
// milestone values: apply RecordTTS per trial and reduce the per-arm
// WallClockToMilestone across trials (min / median / max) at the call site — the
// milestone semantics here (FIRST turn whose correctness crosses the threshold,
// on a possibly non-monotonic correctness signal) are per-trajectory and compose
// under any such reduction. The single-trajectory recorder is kept pure so the
// cross-trial reduction stays an explicit, auditable fold on top, mirroring how
// FoldCacheWitness keeps the per-scrape fold separate from any higher rollup.

// TTSMetricSchema is the versioned schema id stamped on the emitted record, the
// fak.frontierswe.tts.v1 payload C12 (compare) and C15 (authority) consume.
const TTSMetricSchema = "fak.frontierswe.tts.v1"

// DefaultTTSMilestone is the FrontierSWE-native correctness milestone: 1.0, the
// gate that unlocks the speedup tier in the reference scorer. RecordTTS uses it
// when TTSConfig.Milestone is unset (<= 0).
const DefaultTTSMilestone = 1.0

// TTSTurn is one point of an arm's per-turn trajectory trace (the C9 run output).
// WallClockSec is the elapsed wall-clock since the arm started, taken at the point
// this turn's correctness was recorded (the reward.json timestamp for the trial's
// state as of this turn). Correctness is the reward-derived correctness in [0,1]
// (score.go's ExtractScore output) AS OF this turn. Turn is the trajectory turn
// ordinal, carried so TurnsToFirstCorrect reports the real turn index even when the
// trace only records the turns at which correctness changed.
type TTSTurn struct {
	Turn         int     `json:"turn"`
	WallClockSec float64 `json:"wall_clock_sec"`
	Correctness  float64 `json:"correctness"`
}

// TTSArmTrace is one arm's completed run: its label, its per-turn trace, and the
// optional C8 reuse series + C4 geometry used for the measured-vs-projected
// comparison. An arm is a labeled run of the SAME task — canonically "fak" (the
// fak-routed agent) and "baseline" (the raw harness) — distinguished by Role so the
// cross-arm ratio and the projection know which trajectory is fak's. Reuse and
// Geometry are optional: an arm with neither still gets its measured wall-clock and
// turn numbers; only the ProjectedTTSRatio needs them.
type TTSArmTrace struct {
	Arm      string              `json:"arm"`
	Role     string              `json:"role,omitempty"` // "fak" | "baseline" | "" — drives the cross-arm ratio + projection
	Turns    []TTSTurn           `json:"turns"`
	Reuse    *CacheWitnessSeries `json:"reuse,omitempty"`    // C8 realized reuse series for this arm (feeds ProjectedTTSRatio)
	Geometry *TaskGeometry       `json:"geometry,omitempty"` // C4 geometry the projection is re-evaluated on
}

// TTSArmMeasure is the measured value for one arm: when (in wall-clock and in
// turns) it first reached the milestone, plus the C4 projection at its realized
// reuse rate. WallClockToMilestone and TurnsToFirstCorrect are 0 when the arm never
// reached the milestone (Reached=false); a reader must gate on Reached, not read a
// zero as "instant".
type TTSArmMeasure struct {
	Arm  string `json:"arm"`
	Role string `json:"role,omitempty"`

	// Reached / MilestoneUsed / UsedFallback record WHICH milestone the wall-clock
	// number was measured against: the primary milestone when the arm reached it,
	// else the fallback milestone when one is configured and the arm reached that.
	Reached       bool    `json:"reached"`
	MilestoneUsed float64 `json:"milestone_used"`
	UsedFallback  bool    `json:"used_fallback"`

	// The MEASURED numbers, read straight off the trajectory.
	WallClockToMilestone float64 `json:"wall_clock_to_milestone_sec"` // elapsed wall-clock to the first milestone-crossing turn
	TurnsToFirstCorrect  int     `json:"turns_to_first_correct"`      // turn ordinal of that first crossing (0 if never)
	FinalCorrectness     float64 `json:"final_correctness"`           // correctness at the end of the trajectory
	TotalTurns           int     `json:"total_turns"`
	FinalWallClockSec    float64 `json:"final_wall_clock_sec"`

	// RealizedReuseRate is the C8 measured reuse rate for this arm (0 when no series
	// was supplied); ProjectedTTSRatio is the C4 floor re-evaluated at it (0 when no
	// geometry was supplied). The projection is the PROJECTION half of the honesty
	// boundary; everything above is MEASURED.
	RealizedReuseRate float64 `json:"realized_reuse_rate"`
	ProjectedTTSRatio float64 `json:"projected_tts_ratio"`
}

// TTSMetric is the emitted fak.frontierswe.tts.v1 record: the per-arm measured
// values plus the cross-arm measured TTS ratio recorded beside the C4 projection so
// over-claim is visible in one place.
type TTSMetric struct {
	Schema            string          `json:"schema"`
	Task              string          `json:"task"`
	Milestone         float64         `json:"milestone"`          // the primary correctness milestone (default 1.0)
	FallbackMilestone float64         `json:"fallback_milestone"` // the fixed-quality fallback (0 = disabled)
	Arms              []TTSArmMeasure `json:"arms"`

	// MeasuredTTSRatio is the realized T_fak / T_baseline wall-clock-to-milestone —
	// the headline number FrontierSWE's leaderboard does not report. It is set only
	// when both the fak and baseline arms reached the SAME milestone (Comparable);
	// otherwise it is 0 and Comparable is false (never a divide-by-zero, never a
	// ratio across two different milestones).
	MeasuredTTSRatio float64 `json:"measured_tts_ratio"`
	// ProjectedTTSRatio is the fak arm's C4 projection at its realized reuse rate —
	// the PROJECTION the measured ratio is checked against.
	ProjectedTTSRatio float64 `json:"projected_tts_ratio"`
	// OverClaim = MeasuredTTSRatio - ProjectedTTSRatio, defined only when Comparable
	// and a projection exists. Positive means the run took a LARGER fraction of the
	// baseline than the floor projected — the projection over-claimed the speedup;
	// negative means the run beat the floor. It makes over-/under-claim a single
	// signed number a downstream authority (C15) can gate on.
	OverClaim float64 `json:"over_claim"`
	// Comparable reports whether MeasuredTTSRatio is a valid cross-arm number: both
	// the fak and baseline arms reached the same milestone.
	Comparable bool `json:"comparable"`
}

// TTSConfig is the input to RecordTTS: the task name, the milestone policy, and the
// per-arm run traces. Milestone defaults to DefaultTTSMilestone (1.0) when unset;
// FallbackMilestone of 0 disables the fallback (an arm that never reaches the
// primary milestone is simply reported as not reached).
type TTSConfig struct {
	Task              string
	Milestone         float64
	FallbackMilestone float64
	Arms              []TTSArmTrace
}

// RecordTTS computes the measured time-to-solution metric for a task's arms. For
// each arm it finds the first turn whose correctness crosses the primary milestone
// (else the fallback milestone, when configured), reading the wall-clock and turn
// ordinal straight off the trajectory; it then re-evaluates the C4 TTS floor at the
// arm's realized reuse rate. Finally it records the cross-arm measured TTS ratio
// (fak vs baseline) beside the fak arm's projection so over-claim is visible.
// Deterministic and I/O-free — the whole record is a pure fold over the input
// traces, exactly like FoldCacheWitness.
func RecordTTS(cfg TTSConfig) TTSMetric {
	milestone := cfg.Milestone
	if milestone <= 0 {
		milestone = DefaultTTSMilestone
	}

	out := TTSMetric{
		Schema:            TTSMetricSchema,
		Task:              cfg.Task,
		Milestone:         milestone,
		FallbackMilestone: cfg.FallbackMilestone,
		Arms:              make([]TTSArmMeasure, 0, len(cfg.Arms)),
	}
	for _, arm := range cfg.Arms {
		out.Arms = append(out.Arms, measureArm(arm, milestone, cfg.FallbackMilestone))
	}

	// The cross-arm measured ratio: fak vs baseline, only when both reached the SAME
	// milestone and the baseline has a non-zero time (never a divide-by-zero, never a
	// ratio spanning two different milestones).
	fak := findRole(out.Arms, "fak")
	base := findRole(out.Arms, "baseline")
	if fak != nil {
		out.ProjectedTTSRatio = fak.ProjectedTTSRatio
	}
	if fak != nil && base != nil &&
		fak.Reached && base.Reached &&
		fak.MilestoneUsed == base.MilestoneUsed &&
		base.WallClockToMilestone > 0 {
		out.MeasuredTTSRatio = fak.WallClockToMilestone / base.WallClockToMilestone
		out.Comparable = true
		if out.ProjectedTTSRatio > 0 {
			out.OverClaim = out.MeasuredTTSRatio - out.ProjectedTTSRatio
		}
	}
	return out
}

// measureArm reduces one arm's trace to its measured milestone crossing plus the C4
// projection at its realized reuse rate. It scans the trace IN ORDER (the trace is
// time-ordered by the run, and re-sorting would hide a real out-of-order capture bug
// — the same discipline FoldCacheWitness follows): the first turn to cross the
// primary milestone wins; if none does and a fallback is configured, the first to
// cross the fallback wins with UsedFallback set.
func measureArm(t TTSArmTrace, milestone, fallback float64) TTSArmMeasure {
	m := TTSArmMeasure{
		Arm:           t.Arm,
		Role:          t.Role,
		MilestoneUsed: milestone,
		TotalTurns:    len(t.Turns),
	}
	if t.Reuse != nil {
		m.RealizedReuseRate = t.Reuse.RealizedReuseRate
	}
	if n := len(t.Turns); n > 0 {
		// The trajectory's end state: the last recorded turn (time-ordered input).
		m.FinalCorrectness = t.Turns[n-1].Correctness
		m.FinalWallClockSec = t.Turns[n-1].WallClockSec
	}

	if hit, ok := firstAtLeast(t.Turns, milestone); ok {
		m.Reached = true
		m.MilestoneUsed = milestone
		m.WallClockToMilestone = hit.WallClockSec
		m.TurnsToFirstCorrect = hit.Turn
	} else if fallback > 0 {
		if hit, ok := firstAtLeast(t.Turns, fallback); ok {
			m.Reached = true
			m.UsedFallback = true
			m.MilestoneUsed = fallback
			m.WallClockToMilestone = hit.WallClockSec
			m.TurnsToFirstCorrect = hit.Turn
		}
	}

	// The PROJECTION half: the C4 floor re-evaluated at the arm's realized reuse.
	if t.Geometry != nil {
		m.ProjectedTTSRatio = TTSRatio(*t.Geometry, m.RealizedReuseRate)
	}
	return m
}

// firstAtLeast returns the first turn (in trajectory order) whose correctness is at
// least threshold, and whether one exists.
func firstAtLeast(turns []TTSTurn, threshold float64) (TTSTurn, bool) {
	for _, t := range turns {
		if t.Correctness >= threshold {
			return t, true
		}
	}
	return TTSTurn{}, false
}

// findRole returns the first measured arm carrying the given role, or nil. Roles are
// expected to be unique ("fak", "baseline"); the first match wins deterministically.
func findRole(arms []TTSArmMeasure, role string) *TTSArmMeasure {
	for i := range arms {
		if arms[i].Role == role {
			return &arms[i]
		}
	}
	return nil
}
