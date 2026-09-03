package doomloop

import (
	"fmt"
	"sync"
)

// NudgeMessageFormat is the canonical template format for a predictive nudge steer packet.
const NudgeMessageFormat = "Your last %d attempts showed no forward progress. Step back and inspect the file or error diagnostics before retrying."

// PredictiveNudge carries the actionable steer packet emitted when consecutive
// flat turns indicate an early doom-loop pattern, intervening with positive
// steering before reaching a full doom-loop halt.
type PredictiveNudge struct {
	ActiveObjective      string `json:"active_objective"`
	ConsecutiveFlatTurns int    `json:"consecutive_flat_turns"`
	ThresholdK           int    `json:"threshold_k"`
	LastObservedFailure  string `json:"last_observed_failure"`
	NudgeMessage         string `json:"nudge_message"`
	Triggered            bool   `json:"triggered"`
}

// FullHaltReached reports whether consecutive flat turns have reached or exceeded
// the hard doom-loop halt threshold K.
func (pn PredictiveNudge) FullHaltReached() bool {
	return pn.ThresholdK > 0 && pn.ConsecutiveFlatTurns >= pn.ThresholdK
}

// PredictiveNudgeThreshold returns the threshold for triggering a predictive nudge:
// ceiling(thresholdK / 2), with a minimum of 2.
func PredictiveNudgeThreshold(thresholdK int) int {
	if thresholdK <= 0 {
		thresholdK = DefaultConfig().TripWindows
	}
	nudgeThreshold := (thresholdK + 1) / 2
	if nudgeThreshold < 2 {
		nudgeThreshold = 2
	}
	return nudgeThreshold
}

// FormatNudgeMessage formats a predictive nudge message for the given number of flat turns.
func FormatNudgeMessage(consecutiveFlatTurns int) string {
	return fmt.Sprintf(NudgeMessageFormat, consecutiveFlatTurns)
}

// EvaluatePredictiveNudge evaluates whether consecutive flat turns warrant emitting
// a predictive nudge steer packet before reaching full doom-loop halt.
//
// When consecutive flat turns reach K/2 (ceiling of K/2, minimum 2) and have not
// yet reached full doom-loop halt at K, Triggered is true and NudgeMessage is populated.
func EvaluatePredictiveNudge(activeObjective string, consecutiveFlatTurns int, thresholdK int, lastObservedFailure string) PredictiveNudge {
	if thresholdK <= 0 {
		thresholdK = DefaultConfig().TripWindows
	}
	if consecutiveFlatTurns < 0 {
		consecutiveFlatTurns = 0
	}
	nudgeThreshold := PredictiveNudgeThreshold(thresholdK)
	triggered := consecutiveFlatTurns >= nudgeThreshold
	if thresholdK > nudgeThreshold && consecutiveFlatTurns >= thresholdK {
		triggered = false
	}
	var msg string
	if triggered {
		msg = FormatNudgeMessage(consecutiveFlatTurns)
	}
	return PredictiveNudge{
		ActiveObjective:      activeObjective,
		ConsecutiveFlatTurns: consecutiveFlatTurns,
		ThresholdK:           thresholdK,
		LastObservedFailure:  lastObservedFailure,
		NudgeMessage:         msg,
		Triggered:            triggered,
	}
}

// PredictiveNudgeTracker maintains turn-by-turn state for an active objective
// and evaluates predictive nudges.
type PredictiveNudgeTracker struct {
	mu                   sync.Mutex
	activeObjective      string
	thresholdK           int
	consecutiveFlatTurns int
	lastObservedFailure  string
}

// NewPredictiveNudgeTracker creates a new tracker for the given objective and threshold K.
func NewPredictiveNudgeTracker(activeObjective string, thresholdK int) *PredictiveNudgeTracker {
	if thresholdK <= 0 {
		thresholdK = DefaultConfig().TripWindows
	}
	return &PredictiveNudgeTracker{
		activeObjective: activeObjective,
		thresholdK:      thresholdK,
	}
}

// SetObjective updates the active objective. If the objective changed,
// the consecutive flat turns counter and failure reason reset.
func (t *PredictiveNudgeTracker) SetObjective(objective string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.activeObjective != objective {
		t.activeObjective = objective
		t.consecutiveFlatTurns = 0
		t.lastObservedFailure = ""
	}
}

// RecordTurn records one turn's outcome: whether verified progress was made,
// and any observed failure diagnostic. Returns the current PredictiveNudge.
func (t *PredictiveNudgeTracker) RecordTurn(progressMade bool, failure string) PredictiveNudge {
	t.mu.Lock()
	defer t.mu.Unlock()
	if progressMade {
		t.consecutiveFlatTurns = 0
		t.lastObservedFailure = ""
	} else {
		t.consecutiveFlatTurns++
		if failure != "" {
			t.lastObservedFailure = failure
		}
	}
	return EvaluatePredictiveNudge(t.activeObjective, t.consecutiveFlatTurns, t.thresholdK, t.lastObservedFailure)
}

// RecordFlatTurn records a turn that made no forward progress.
func (t *PredictiveNudgeTracker) RecordFlatTurn(failure string) PredictiveNudge {
	return t.RecordTurn(false, failure)
}

// RecordProgress records a turn that advanced verified progress.
func (t *PredictiveNudgeTracker) RecordProgress() PredictiveNudge {
	return t.RecordTurn(true, "")
}

// Evaluate returns the current predictive nudge assessment without mutating state.
func (t *PredictiveNudgeTracker) Evaluate() PredictiveNudge {
	t.mu.Lock()
	defer t.mu.Unlock()
	return EvaluatePredictiveNudge(t.activeObjective, t.consecutiveFlatTurns, t.thresholdK, t.lastObservedFailure)
}

// Reset clears all counters and failure diagnostics.
func (t *PredictiveNudgeTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.consecutiveFlatTurns = 0
	t.lastObservedFailure = ""
}

// ConsecutiveFlatTurns returns the current count of consecutive flat turns.
func (t *PredictiveNudgeTracker) ConsecutiveFlatTurns() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.consecutiveFlatTurns
}

// EvaluateResult evaluates a doomloop.Result and config to determine if a predictive
// nudge should be emitted.
func EvaluateResult(activeObjective string, res Result, cfg Config, lastObservedFailure string) PredictiveNudge {
	cfg = cfg.withDefaults()
	return EvaluatePredictiveNudge(activeObjective, res.BurningFlatStreak, cfg.TripWindows, lastObservedFailure)
}

// EvaluateSamples folds ordered samples into a doomloop.Result and evaluates
// whether a predictive nudge is recommended.
func EvaluateSamples(activeObjective string, samples []Sample, cfg Config, lastObservedFailure string) PredictiveNudge {
	res := Classify(samples, cfg)
	cfg = cfg.withDefaults()
	return EvaluatePredictiveNudge(activeObjective, res.BurningFlatStreak, cfg.TripWindows, lastObservedFailure)
}

// PredictiveNudge evaluates whether this Result warrants a predictive nudge.
func (r Result) PredictiveNudge(activeObjective string, cfg Config, lastObservedFailure string) PredictiveNudge {
	return EvaluateResult(activeObjective, r, cfg, lastObservedFailure)
}
