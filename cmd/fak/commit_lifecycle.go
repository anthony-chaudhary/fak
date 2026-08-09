package main

import (
	"context"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/commitlifecycle"
	"github.com/anthony-chaudhary/fak/internal/wiprecon"
)

// commitLifecycleQueue folds the existing checkpoint reconciler into the shared
// commit-to-ship vocabulary. It is deliberately read-only: the queue names the
// exact existing command which advances each row, but executes none of them.
func commitLifecycleQueue(ctx context.Context, repo string) ([]commitlifecycle.Row, error) {
	result, err := wipReconcileAt(ctx, repo, time.Now())
	if err != nil {
		return nil, err
	}
	decisions := result.Decisions
	rows := make([]commitlifecycle.Row, 0, len(decisions))
	for _, d := range decisions {
		facts := commitlifecycle.Facts{Checkpoint: d.Session}
		switch d.Action {
		case wiprecon.ActSkip:
			facts.CheckpointLive = true
		case wiprecon.ActReclaim:
			facts.CheckpointApply = true
		case wiprecon.ActQuarantine:
			rows = append(rows, commitlifecycle.Row{
				State: commitlifecycle.Unknown,
				Action: commitlifecycle.Action{
					NeedsOperator: true,
					Reason:        d.Reason,
				},
			})
			continue
		case wiprecon.ActDiscardWitnessed:
			// The checkpoint delta is already in HEAD. Remote ancestry still has
			// to witness SHIPPED, so expose the existing push edge rather than
			// calling the checkpoint itself terminal.
			rows = append(rows, commitlifecycle.Row{
				State:  commitlifecycle.LandedUnpushed,
				Action: commitlifecycle.Action{Tool: "git", Args: []string{"push"}},
			})
			continue
		default:
			rows = append(rows, commitlifecycle.Row{
				State:  commitlifecycle.Unknown,
				Action: commitlifecycle.Action{NeedsOperator: true, Reason: "unknown checkpoint decision " + string(d.Action)},
			})
			continue
		}
		rows = append(rows, commitlifecycle.Fold(facts))
	}
	return rows, nil
}

func commitLifecycleActionText(action commitlifecycle.Action) string {
	if action.NeedsOperator {
		return "operator: " + action.Reason
	}
	if action.Tool == "" {
		return "none"
	}
	return strings.Join(append([]string{action.Tool}, action.Args...), " ")
}
