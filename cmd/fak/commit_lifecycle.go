package main

import (
	"context"
	"errors"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/commitlifecycle"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/internal/wiprecon"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
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
	// The recovery worklist already resolved each RECLAIM row's adoption receipt, so the
	// lifecycle queue reads ownership from it rather than re-deriving it. That keeps ONE
	// answer to "who holds this checkpoint": a second derivation here could disagree with
	// the queue the operator is looking at, and the disagreement would be invisible.
	reclaim := make(map[string]wiprecon.ReclaimRow, len(result.Reclaim))
	for _, r := range result.Reclaim {
		reclaim[r.Session] = r
	}
	rows = append(rows, make([]commitlifecycle.Row, 0, len(decisions))...)
	for _, d := range decisions {
		facts := commitlifecycle.Facts{Checkpoint: d.Session}
		switch d.Action {
		case wiprecon.ActSkip:
			facts.CheckpointLive = true
		case wiprecon.ActReclaim:
			facts.CheckpointApply = true
			if row, ok := reclaim[d.Session]; ok {
				facts.CheckpointAdoptedBy = row.AdoptedBy
				facts.CheckpointAdoptMine = row.AdoptedMine
				facts.CheckpointAdoptExpired = row.AdoptExpired
			}
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
	workerRows, err := workerworktree.Inventory(repo, nil)
	if err != nil {
		return nil, err
	}
	for _, worker := range workerRows {
		switch worker.State {
		case "LAND_READY":
			if len(worker.LandArgv) < 2 || worker.LandArgv[0] != "fak" {
				rows = append(rows, commitlifecycle.Row{State: commitlifecycle.Unknown, Action: commitlifecycle.Action{NeedsOperator: true, Reason: "worker inventory returned invalid land argv"}})
				continue
			}
			rows = append(rows, commitlifecycle.Row{State: commitlifecycle.LandReady, Action: commitlifecycle.Action{Tool: worker.LandArgv[0], Args: append([]string(nil), worker.LandArgv[1:]...)}})
		case "NEEDS_OPERATOR":
			rows = append(rows, commitlifecycle.Row{State: commitlifecycle.Unknown, Action: commitlifecycle.Action{NeedsOperator: true, Reason: worker.Reason}})
		}
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
	windowgate.ConfigureBackgroundCommand(cmd)
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
