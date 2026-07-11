package main

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

// renderSyncPushVelocity is the one human projection of PushResult.Velocity;
// JSON encodes the same struct directly. A missing budget means the caller
// constructed a synthetic/legacy PushResult rather than running SafePush.
func renderSyncPushVelocity(w io.Writer, v safesync.PushVelocity) {
	if v.BudgetMS <= 0 {
		return
	}
	elapsed := time.Duration(v.ElapsedMS) * time.Millisecond
	budget := time.Duration(v.BudgetMS) * time.Millisecond
	if v.Qualified && v.Score != nil {
		fmt.Fprintf(w, "  velocity: %s / %s budget (ratio %.3f, %d/100 %s)\n",
			elapsed, budget, v.BudgetRatio, *v.Score, v.Grade)
		return
	}
	note := "safe push did not publish"
	if len(v.Notes) > 0 && strings.TrimSpace(v.Notes[0]) != "" {
		note = strings.TrimSpace(v.Notes[0])
	}
	fmt.Fprintf(w, "  velocity: %s / %s budget (ratio %.3f, UNSCORED — %s)\n",
		elapsed, budget, v.BudgetRatio, note)
}

func validatePushVelocityBudget(budget time.Duration) error {
	if budget < time.Millisecond {
		return fmt.Errorf("--budget must be at least 1ms for sync push")
	}
	return nil
}
