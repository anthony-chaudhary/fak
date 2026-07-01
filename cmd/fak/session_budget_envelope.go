package main

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// applyGuardSessionBudgetEnvelope seeds a guard session's managed-context envelope at
// launch (#1573). It mutates only the existing session table axes: Budget, Pace, and
// TimeBudget. Spend/throughput remain parsed inspectable contract fields until a live
// runtime consumer enforces them.
func applyGuardSessionBudgetEnvelope(tbl *session.Table, traceID string, env session.BudgetEnvelope, hasEnvelope bool, contextOverride *int, effectiveContext int, wallLimit time.Duration, now time.Time) {
	if tbl == nil || traceID == "" {
		return
	}
	if hasEnvelope || effectiveContext > 0 || contextOverride != nil {
		b := session.NewBudgetEnvelope().SessionBudget()
		if hasEnvelope {
			b = env.SessionBudget()
		}
		if contextOverride != nil {
			b.ContextTokensLeft = *contextOverride
		} else if effectiveContext > 0 {
			b.ContextTokensLeft = effectiveContext
		}
		tbl.SetBudget(traceID, b)
	}
	if hasEnvelope {
		p := env.SessionPace()
		if p.MaxTokensPerTurn > 0 || p.MinTurnGapMs > 0 {
			tbl.SetPace(traceID, p)
		}
	}
	if wallLimit > 0 {
		tbl.StartTimeBudget(traceID, wallLimit, now)
	}
}
