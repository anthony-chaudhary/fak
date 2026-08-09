package main

import (
	"context"
	"errors"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/commitlifecycle"
	"github.com/anthony-chaudhary/fak/internal/wiprecon"
)

// commitLifecycleQueue folds the existing checkpoint reconciler into the shared
// commit-to-ship vocabulary. It is deliberately read-only: the queue names the
// exact existing command which advances each row, but executes none of them.
func commitLifecycleQueue(ctx context.Context, repo string) ([]commitlifecycle.Row, error) {
	ancestry, err := commitlifecycle.InspectAncestry(ctx, repo, commitLifecycleGit)
	if err != nil {
		return nil, err
	}
	ancestryRows := commitlifecycle.AncestryRows(ancestry)
	rows := []commitlifecycle.Row{}
	result, err := wipReconcileAt(ctx, repo, time.Now())
	if err != nil {
		return nil, err
	}
	decisions := result.Decisions
	rows = append(rows, make([]commitlifecycle.Row, 0, len(decisions))...)
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
				Action: commitlifecycle.Action{Tool: "fak", Args: []string{"sync", "push"}},
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
	rows = append(rows, ancestryRows...)
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

func commitLifecycleGit(ctx context.Context, repo string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = repo
	out, err := cmd.CombinedOutput()
	if err == nil {
		return string(out), 0, nil
	}
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		return string(out), exit.ExitCode(), nil
	}
	return string(out), -1, err
}
