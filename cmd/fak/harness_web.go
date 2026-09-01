package main

import (
	"context"
	"encoding/json"
	"io"
	"os/exec"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessweb"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func runHarnessWeb(stdout, stderr io.Writer, argv []string) int {
	return runHarnessWebWithCancel(context.Background(), stdout, stderr, argv)
}

func runHarnessWebWithCancel(ctx context.Context, stdout, stderr io.Writer, argv []string) int {
	return harnessweb.RunWithLocalWork(ctx, stdout, stderr, argv, harnessLocalWorkSource{})
}

type harnessLocalWorkSource struct{}

func (harnessLocalWorkSource) LiveIntentKeys(ctx context.Context, root string, now time.Time) ([]string, error) {
	live, _, err := leaseref.NewInDir(root).LiveIntents(ctx, now)
	if err != nil {
		return nil, err
	}
	keys := make([]string, 0, len(live))
	for _, intent := range live {
		keys = append(keys, intent.Key)
	}
	return keys, nil
}

var harnessDOSLive = func(ctx context.Context, root string) ([]byte, error) {
	timeoutScope, cancel := context.WithTimeout(ctx, 1500*time.Millisecond)
	defer cancel()
	cmd := exec.CommandContext(timeoutScope, "dos", "--workspace", root, "lease-lane", "live")
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.Output()
}

func (harnessLocalWorkSource) LiveDOSLeases(ctx context.Context, root string, _ time.Time) ([]harnessweb.LocalDOSLease, error) {
	data, err := harnessDOSLive(ctx, root)
	if err != nil {
		return nil, err
	}
	var rows []struct {
		Lane   string `json:"lane"`
		LoopID string `json:"loop_id"`
	}
	if err := json.Unmarshal(data, &rows); err != nil {
		return nil, err
	}
	leases := make([]harnessweb.LocalDOSLease, 0, len(rows))
	for _, row := range rows {
		leases = append(leases, harnessweb.LocalDOSLease{Lane: row.Lane, LoopID: row.LoopID})
	}
	return leases, nil
}

func (harnessLocalWorkSource) WorkerWorktreeLifecycleInputs(_ context.Context, root string, _ time.Time) ([]workerworktree.StatusEvidence, error) {
	_, paths := workerworktree.Count(root, nil)
	rows := worktreeWorkerLifecycleInventory(root, paths, worktreeWorkerLifecycleProbes{})
	inputs := make([]workerworktree.StatusEvidence, 0, len(rows))
	for _, row := range rows {
		intent, err := workerworktree.LoadIntent(row.Path)
		issue := 0
		if err == nil {
			issue = intent.IssueNumber
		}
		inputs = append(inputs, workerworktree.StatusEvidence{
			IssueNumber: issue, Lane: row.Association.Lane, Session: row.Association.LeaseID,
			HeadSHA: row.HeadSHA, BaseSHA: row.BaseSHA,
			AssociationKnown: row.Association.State == worktreeEvidenceAssociated,
			OwnerLive:        row.Liveness.Owner == worktreeEvidenceLive, LeaseLive: row.Liveness.Lease == worktreeEvidenceLive,
			Dirty: row.Cleanliness.State == worktreeEvidenceDirty, CleanupReady: row.ReapReadiness.Reapable,
		})
	}
	return inputs, nil
}
