package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func TestWorkerWorktreeStatusProjectionIsPathFreeAndDoesNotInventCompletion(t *testing.T) {
	rows := []worktreeWorkerLifecycleRow{
		{
			Path:    "/Users/private/fak-worker-wt-workerworktree-secret",
			Intent:  worktreeWorkerIntent{Status: worktreeWorkerIntentPresent, IssueNumber: 10551},
			HeadSHA: "bbbb", BaseSHA: "aaaa",
			Association:   worktreeWorkerAssociation{State: worktreeEvidenceAssociated, Lane: "workerworktree", OwnerPID: 42, LeaseID: "lease"},
			Liveness:      worktreeWorkerLiveness{Owner: worktreeEvidenceDead, Lease: worktreeEvidenceReleased},
			Cleanliness:   worktreeWorkerCleanliness{State: worktreeEvidenceDirty, DirtyPaths: []string{"secret/file"}},
			Lifecycle:     worktreeLifecycleDirty,
			ReapReadiness: worktreeWorkerReapReadiness{Verdict: worktreeKeep},
		},
		{
			Path:    "/Users/private/cleanup",
			Intent:  worktreeWorkerIntent{Status: worktreeWorkerIntentPresent, IssueNumber: 10552},
			HeadSHA: "aaaa", BaseSHA: "aaaa",
			Association:   worktreeWorkerAssociation{State: worktreeEvidenceAssociated, Lane: "workerworktree", OwnerPID: 43, LeaseID: "lease2"},
			Liveness:      worktreeWorkerLiveness{Owner: worktreeEvidenceDead, Lease: worktreeEvidenceReleased},
			Cleanliness:   worktreeWorkerCleanliness{State: worktreeEvidenceClean},
			Lifecycle:     worktreeLifecycleReady,
			ReapReadiness: worktreeWorkerReapReadiness{Reapable: true, Verdict: worktreeReapable},
		},
	}
	got := make([]workerworktree.StatusProjection, 0, len(rows))
	for _, row := range rows {
		got = append(got, workerworktree.ProjectStatus(workerworktree.StatusEvidence{
			IssueNumber: row.Intent.IssueNumber, Lane: row.Association.Lane,
			AssociationKnown: row.Association.State == worktreeEvidenceAssociated,
			OwnerLive:        row.Liveness.Owner == worktreeEvidenceLive, LeaseLive: row.Liveness.Lease == worktreeEvidenceLive,
			Dirty: row.Cleanliness.State == worktreeEvidenceDirty, HeadSHA: row.HeadSHA, BaseSHA: row.BaseSHA,
			CleanupReady: row.ReapReadiness.Reapable,
		}))
	}
	if got[0].State != workerworktree.DisplayUnlandedChanges {
		t.Fatalf("dirty dead worktree state = %q", got[0].State)
	}
	if got[1].State != workerworktree.DisplayCleanupReady {
		t.Fatalf("clean dead worktree state = %q, want cleanup_ready", got[1].State)
	}
	payload, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(payload), "/Users/private") || strings.Contains(string(payload), "secret/file") {
		t.Fatalf("projection leaked local path: %s", payload)
	}
	if strings.Contains(string(payload), string(workerworktree.DisplayLandedWitnessed)) {
		t.Fatalf("local lifecycle invented landed completion: %s", payload)
	}
}
