package harnessweb

import (
	"context"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// LocalDOSLease is the operator-facing identity of one live DOS lane lease.
type LocalDOSLease struct {
	Lane   string `json:"lane"`
	LoopID string `json:"loop_id,omitempty"`
}

// LocalWorkSource supplies bounded, live local-work identities to the harness UI.
type LocalWorkSource interface {
	LiveIntentKeys(context.Context, string, time.Time) ([]string, error)
	LiveDOSLeases(context.Context, string, time.Time) ([]LocalDOSLease, error)
}

// LocalWorkerWorktreeSource is an optional, narrow extension for authoritative
// worker-worktree evidence. Inputs contain no paths; classification remains
// centralized in workerworktree.ProjectLifecycle.
type LocalWorkerWorktreeSource interface {
	WorkerWorktreeLifecycleInputs(context.Context, string, time.Time) ([]workerworktree.StatusEvidence, error)
}

type localWorkSet struct {
	Active int      `json:"active"`
	IDs    []string `json:"ids"`
	Error  string   `json:"error,omitempty"`
}

type workerWorktreeStateCounts struct {
	Active             int `json:"active"`
	UnlandedChanges    int `json:"unlanded_changes"`
	LandedWitnessed    int `json:"landed_witnessed"`
	CleanupReady       int `json:"cleanup_ready"`
	AssociationUnknown int `json:"association_unknown"`
}

type workerWorktreeOverview struct {
	Total  int                               `json:"total"`
	States workerWorktreeStateCounts         `json:"states"`
	Items  []workerworktree.StatusProjection `json:"items"`
	Error  string                            `json:"error,omitempty"`
}

type localWorkOverview struct {
	IssueIntents    localWorkSet           `json:"issue_intents"`
	DOSLeases       localWorkSet           `json:"dos_leases"`
	WorkerWorktrees workerWorktreeOverview `json:"worker_worktrees"`
}

func readLocalWorkOverview(ctx context.Context, source LocalWorkSource, root string, now time.Time) localWorkOverview {
	result := localWorkOverview{
		IssueIntents:    localWorkSet{IDs: []string{}},
		DOSLeases:       localWorkSet{IDs: []string{}},
		WorkerWorktrees: workerWorktreeOverview{Items: []workerworktree.StatusProjection{}},
	}
	if source == nil || strings.TrimSpace(root) == "" {
		return result
	}
	intents, intentErr := source.LiveIntentKeys(ctx, root, now)
	leases, leaseErr := source.LiveDOSLeases(ctx, root, now)
	intentIDs := boundedLocalWorkIDs(intents)
	leaseKeys := make([]string, 0, len(leases))
	for _, lease := range leases {
		key := strings.TrimSpace(lease.Lane)
		if loop := strings.TrimSpace(lease.LoopID); loop != "" {
			key += " (" + loop + ")"
		}
		leaseKeys = append(leaseKeys, key)
	}
	leaseIDs := boundedLocalWorkIDs(leaseKeys)
	result.IssueIntents = localWorkSet{Active: len(intentIDs), IDs: intentIDs}
	result.DOSLeases = localWorkSet{Active: len(leaseIDs), IDs: leaseIDs}
	if intentErr != nil {
		result.IssueIntents.Error = "unavailable"
	}
	if leaseErr != nil {
		result.DOSLeases.Error = "unavailable"
	}
	if workerSource, ok := source.(LocalWorkerWorktreeSource); ok {
		inputs, err := workerSource.WorkerWorktreeLifecycleInputs(ctx, root, now)
		if err != nil {
			result.WorkerWorktrees.Error = "unavailable"
		} else {
			result.WorkerWorktrees = projectWorkerWorktrees(inputs)
		}
	}
	return result
}

func projectWorkerWorktrees(inputs []workerworktree.StatusEvidence) workerWorktreeOverview {
	out := workerWorktreeOverview{Items: make([]workerworktree.StatusProjection, 0, len(inputs))}
	for _, input := range inputs {
		item := workerworktree.ProjectStatus(input)
		out.Items = append(out.Items, item)
		switch item.State {
		case workerworktree.DisplayActive:
			out.States.Active++
		case workerworktree.DisplayUnlandedChanges:
			out.States.UnlandedChanges++
		case workerworktree.DisplayLandedWitnessed:
			out.States.LandedWitnessed++
		case workerworktree.DisplayCleanupReady:
			out.States.CleanupReady++
		case workerworktree.DisplayAssociationUnknown:
			out.States.AssociationUnknown++
		}
	}
	out.Total = len(out.Items)
	return out
}

func boundedLocalWorkIDs(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	ids := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if len(value) > maxLocalWorkIDBytes {
			value = value[:maxLocalWorkIDBytes]
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		ids = append(ids, value)
	}
	sort.Strings(ids)
	if len(ids) > maxLocalWorkIDs {
		ids = ids[:maxLocalWorkIDs]
	}
	return ids
}
