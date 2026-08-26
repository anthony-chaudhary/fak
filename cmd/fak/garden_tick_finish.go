package main

import (
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gardenbudget"
	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
)

type gardenTickFinisher struct {
	stdout, stderr   io.Writer
	asJSON           bool
	budget           int
	started          time.Time
	root, cursorPath string
	state            *gardenTickState
	resume           *gardenbudget.Cursor
}

func (finisher gardenTickFinisher) finish(status, reason string, complete bool, plan gardenbundle.TickPlan,
	collection gardenbundle.CollectProgress, action gardenbudget.Result, code int) int {
	stdout, stderr := finisher.stdout, finisher.stderr
	root, cursorPath := finisher.root, finisher.cursorPath
	tickStart, state, resume := finisher.started, finisher.state, finisher.resume
	asJSON, budget := finisher.asJSON, finisher.budget
	counts := state.Counts
	if asJSON {
		out := map[string]any{
			"schema":                "fak.garden-tick.v2",
			"status":                status,
			"reason":                reason,
			"complete":              complete,
			"workspace":             root,
			"commit":                gardenbundle.HeadCommit(root),
			"dry_run":               plan.DryRun,
			"acted":                 complete && !plan.DryRun && (plan.Acted() || counts.acted()),
			"reaped_leases":         counts.Reaped,
			"reaped_sessions":       counts.Sessions,
			"reaped_lock_files":     counts.LockFiles,
			"reaped_intents":        counts.Intents,
			"reaped_growth_logs":    counts.Collected,
			"folded_sentinel_lines": counts.Folded,
			"surfaced_runs":         counts.Surfaced,
			"budget_seconds":        budget,
			"elapsed_millis":        time.Since(tickStart).Milliseconds(),
			"progress": map[string]any{
				"stage":      resume.Stage,
				"next":       resume.Next,
				"cursor":     cursorPath,
				"ticks":      resume.Ticks,
				"collection": collection,
				"action":     action,
			},
			"plan": plan,
		}
		if rc := encodeJSONOrFail(stdout, stderr, out, "fak garden tick"); rc != 0 {
			return rc
		}
		return code
	}
	renderGardenTick(stdout, plan, counts.Reaped, counts.Sessions, counts.Surfaced,
		counts.LockFiles, counts.Collected, counts.Intents, counts.Folded)
	if !complete {
		fmt.Fprintf(stdout, "  -> %s: %s; resume stage=%s next=%q checkpoint=%s\n",
			status, reason, resume.Stage, resume.Next, cursorPath)
	}
	writeGardenTickBudgetLine(stdout, action, cursorPath)
	return code

}
