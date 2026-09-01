package main

import (
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// collectWorkerWorktreeStatuses projects the authoritative local lifecycle
// inventory into the bounded, path-free shape shared by operator surfaces.
// It deliberately supplies no landed witness: clean, dead, or reapable local
// state is never completion evidence.
func collectWorkerWorktreeStatuses(root string) []workerworktree.StatusProjection {
	root = worktreeWorkerRoot(root)
	census := workerworktree.CapacityCensusFor(root, nil)
	if !census.Known {
		return []workerworktree.StatusProjection{}
	}
	rows := worktreeWorkerLifecycleInventory(root, census.Paths, worktreeWorkerLifecycleProbes{})
	out := make([]workerworktree.StatusProjection, 0, len(rows))
	for _, row := range rows {
		out = append(out, workerworktree.ProjectStatus(workerworktree.StatusEvidence{
			IssueNumber:      row.Intent.IssueNumber,
			Lane:             row.Association.Lane,
			AssociationKnown: row.Association.State == worktreeEvidenceAssociated,
			OwnerLive:        row.Liveness.Owner == worktreeEvidenceLive,
			LeaseLive:        row.Liveness.Lease == worktreeEvidenceLive,
			Dirty:            row.Cleanliness.State == worktreeEvidenceDirty,
			HeadSHA:          row.HeadSHA,
			BaseSHA:          row.BaseSHA,
			CleanupReady:     row.ReapReadiness.Reapable,
		}))
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].IssueNumber != out[j].IssueNumber {
			return out[i].IssueNumber < out[j].IssueNumber
		}
		if out[i].Lane != out[j].Lane {
			return out[i].Lane < out[j].Lane
		}
		return out[i].State < out[j].State
	})
	return out
}

func workerWorktreeStatusLabel(row workerworktree.StatusProjection) string {
	parts := []string{string(row.State)}
	if row.IssueNumber > 0 {
		parts = append(parts, "#"+strconv.Itoa(row.IssueNumber))
	}
	if row.Lane != "" {
		parts = append(parts, row.Lane)
	}
	if row.Session != "" {
		parts = append(parts, row.Session)
	}
	if row.Commit != "" {
		parts = append(parts, row.Commit)
	}
	return strings.Join(parts, " · ")
}
