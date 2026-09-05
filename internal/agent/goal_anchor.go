package agent

import (
	"fmt"
	"strings"
	"time"
)

// GoalAnchor maintains an immutable primary objective pinned at the working
// context across error recovery turns to prevent goal drift (#11066).
type GoalAnchor struct {
	Objective         string
	PinnedAt          time.Time
	RecoveryTurnCount int
}

// NewGoalAnchor constructs an immutable GoalAnchor pinning the primary objective verbatim.
func NewGoalAnchor(objective string) *GoalAnchor {
	return &GoalAnchor{
		Objective:         objective,
		PinnedAt:          time.Now().UTC(),
		RecoveryTurnCount: 0,
	}
}

// FormatRecoveryReinforcement returns a formatted context reinforcement block
// pinning the original objective alongside the recovery guidance.
func (g *GoalAnchor) FormatRecoveryReinforcement(recoveryGuidance string) string {
	obj := ""
	if g != nil {
		obj = g.Objective
	}
	return fmt.Sprintf("[PRIMARY GOAL ANCHOR]: %s\n[RECOVERY GUIDANCE]: %s\n[REMINDER]: Maintain focus on the primary goal above while resolving this error. If blocked or complex, decompose into smaller sub-steps or delegate to subagents.", obj, recoveryGuidance)
}

// RecordRecoveryTurn increments the counter of recovery turns undergone.
func (g *GoalAnchor) RecordRecoveryTurn() {
	if g == nil {
		return
	}
	g.RecoveryTurnCount++
}

// ValidateTextContainsAnchor checks if the original goal remains present in text.
func (g *GoalAnchor) ValidateTextContainsAnchor(targetText string) bool {
	if g == nil || g.Objective == "" {
		return false
	}
	if strings.Contains(targetText, g.Objective) {
		return true
	}
	trimmed := strings.TrimSpace(g.Objective)
	if trimmed != "" && strings.Contains(targetText, trimmed) {
		return true
	}
	return false
}
