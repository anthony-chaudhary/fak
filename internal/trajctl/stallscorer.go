package trajctl

import (
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

const (
	// ActivityDivergenceScorerMethod is the W2 method that detects the common
	// "busy but not moving" shape: sessionaudit activity is high while the W3
	// progress curve remains flat.
	ActivityDivergenceScorerMethod = "activity-progress-divergence"
	// ActivityDivergenceScorerVersion is this implementation's version.
	ActivityDivergenceScorerVersion = "1"

	defaultActivityDivergenceThreshold = 0.60
	activityDivergenceProgressEpsilon  = 0.01
)

// ActivityDivergenceScorer emits a W2 row when a session shows high activity
// while the objective's witnessed progress curve is flat. It is pure: callers
// inject sessionaudit.Session rows and prior ScoreRows through EvidenceWindow.
type ActivityDivergenceScorer struct {
	// Threshold is the minimum divergence score required to emit a row. Zero uses
	// the conservative default.
	Threshold float64
}

// Method implements Scorer.
func (ActivityDivergenceScorer) Method() string { return ActivityDivergenceScorerMethod }

// Version implements Scorer.
func (ActivityDivergenceScorer) Version() string { return ActivityDivergenceScorerVersion }

// Score returns one W2 divergence row per stalled session. It stays silent when
// there is no prior progress curve, when progress improved, or when activity is
// below threshold.
func (s ActivityDivergenceScorer) Score(obj Objective, win EvidenceWindow) []ScoreRow {
	if obj.Status != StatusActive && obj.Status != StatusPaused {
		return nil
	}
	delta, ok := commitProgressDelta(obj.ID, win.PriorScores)
	if !ok || delta > activityDivergenceProgressEpsilon {
		return nil
	}
	threshold := s.Threshold
	if threshold <= 0 {
		threshold = defaultActivityDivergenceThreshold
	}
	var out []ScoreRow
	for _, sess := range win.Sessions {
		activity := sessionActivityScore(sess)
		if activity < threshold {
			continue
		}
		ref := sess.Path
		if ref == "" {
			ref = sess.Session
		}
		if ref == "" {
			ref = "session"
		}
		detail := fmt.Sprintf("activity=%.2f flat_commit_delta=%.2f turns=%d tools=%d read_only=%.2f io=%.1f",
			activity, delta, sess.AssistantTurns, sess.NToolUse,
			floatPtrValue(sess.ReadOnlyFrac), floatPtrValue(sess.IORatio))
		out = append(out, ScoreRow{
			ObjectiveID: obj.ID,
			Value:       activity,
			Method:      ActivityDivergenceScorerMethod,
			Version:     ActivityDivergenceScorerVersion,
			Witness:     W2,
			Evidence:    []EvidenceRef{{Kind: "transcript-span", Ref: ref, Detail: detail}},
			UnixMillis:  win.UnixMillis,
			SessionID:   sess.Session,
		})
	}
	return out
}

func commitProgressDelta(objectiveID string, scores []ScoreRow) (float64, bool) {
	var prev, latest float64
	n := 0
	for _, row := range scores {
		if row.ObjectiveID != objectiveID || row.Method != CommitScorerMethod {
			continue
		}
		prev = latest
		latest = row.Value
		n++
	}
	if n < 2 {
		return 0, false
	}
	return latest - prev, true
}

func sessionActivityScore(s sessionaudit.Session) float64 {
	if s.Error != "" {
		return 0
	}
	toolScore := clamp01(float64(s.NToolUse) / 6)
	turnScore := clamp01(float64(s.AssistantTurns) / 4)
	readOnlyScore := floatPtrValue(s.ReadOnlyFrac)
	ioScore := clamp01(floatPtrValue(s.IORatio) / 100)
	dupScore := clamp01(float64(s.DupAssistantLines) / 2)
	interruptScore := clamp01(float64(s.Interrupted) / 2)
	sizeScore := clamp01(float64(s.ToolInputChars+s.ToolResultChars) / 4000)

	score := 0.40*toolScore + 0.20*turnScore + 0.15*readOnlyScore +
		0.10*ioScore + 0.10*sizeScore + 0.03*dupScore + 0.02*interruptScore
	return clamp01(score)
}

func floatPtrValue(v *float64) float64 {
	if v == nil || math.IsNaN(*v) || math.IsInf(*v, 0) {
		return 0
	}
	return *v
}

func clamp01(v float64) float64 {
	switch {
	case v < 0:
		return 0
	case v > 1:
		return 1
	default:
		return v
	}
}
