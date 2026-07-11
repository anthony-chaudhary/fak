package safesync

import (
	"math"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func ScoreApplyVelocity(info Assessment, elapsed, budget time.Duration, runErr error) PushVelocity {
	if elapsed < 0 {
		elapsed = 0
	}
	if budget < time.Millisecond {
		budget = time.Millisecond
	}
	v := PushVelocity{ElapsedMS: elapsed.Milliseconds(), BudgetMS: budget.Milliseconds(), BudgetRatio: scorecard.Round3(float64(elapsed) / float64(budget)), Grade: "UNSCORED"}
	if runErr != nil {
		v.Notes = []string{"unscored: sync apply ended with INTERNAL_ERROR"}
		return v
	}
	if !info.Applied {
		reason := strings.TrimSpace(info.Reason)
		if reason == "" {
			reason = info.State
		}
		if reason == "" {
			reason = "NO_APPLY_EFFECT"
		}
		v.Notes = []string{"unscored: sync apply produced no fast-forward effect (" + reason + ")"}
		return v
	}
	v.Qualified = true
	credit := 1.0
	if elapsed > budget {
		credit = float64(budget) / float64(elapsed)
	}
	score := int(math.Round(100 * credit))
	if score < 0 {
		score = 0
	}
	if score > 100 {
		score = 100
	}
	v.Score = &score
	v.Grade = scorecard.GradeStd(float64(score))
	v.Notes = []string{"qualified: sync apply fast-forwarded HEAD"}
	return v
}
