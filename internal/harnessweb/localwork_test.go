package harnessweb

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

type localWorkTestSource struct {
	workerInputs []workerworktree.StatusEvidence
	workerErr    error
}

func (localWorkTestSource) LiveIntentKeys(context.Context, string, time.Time) ([]string, error) {
	return nil, nil
}
func (localWorkTestSource) LiveDOSLeases(context.Context, string, time.Time) ([]LocalDOSLease, error) {
	return nil, nil
}
func (s localWorkTestSource) WorkerWorktreeLifecycleInputs(context.Context, string, time.Time) ([]workerworktree.StatusEvidence, error) {
	return s.workerInputs, s.workerErr
}

type legacyLocalWorkTestSource struct{}

func (legacyLocalWorkTestSource) LiveIntentKeys(context.Context, string, time.Time) ([]string, error) {
	return nil, nil
}
func (legacyLocalWorkTestSource) LiveDOSLeases(context.Context, string, time.Time) ([]LocalDOSLease, error) {
	return nil, nil
}

func TestReadLocalWorkOverviewProjectsWorkerWorktreeEvidence(t *testing.T) {
	got := readLocalWorkOverview(context.Background(), localWorkTestSource{workerInputs: []workerworktree.StatusEvidence{
		{IssueNumber: 10551, Lane: "workerworktree", AssociationKnown: true, Dirty: true},
		{Session: "lease-clean", AssociationKnown: true, CleanupReady: true},
		{IssueNumber: 7, AssociationKnown: true, HeadSHA: "ABCDEF1", BaseSHA: "base", LandedWitnessed: true},
		{Lane: `C:\secret\worker`},
	}}, t.TempDir(), time.Now())
	if got.WorkerWorktrees.Total != 4 || got.WorkerWorktrees.States.UnlandedChanges != 1 || got.WorkerWorktrees.States.CleanupReady != 1 || got.WorkerWorktrees.States.LandedWitnessed != 1 || got.WorkerWorktrees.States.AssociationUnknown != 1 {
		t.Fatalf("worker worktrees = %#v", got.WorkerWorktrees)
	}
	if got.WorkerWorktrees.Items[1].Complete {
		t.Fatal("cleanup_ready must not be complete")
	}
	if !got.WorkerWorktrees.Items[2].Complete {
		t.Fatal("landed_witnessed must be complete")
	}
	b, _ := json.Marshal(got)
	if strings.Contains(string(b), `C:\secret`) {
		t.Fatalf("absolute path leaked: %s", b)
	}
}

func TestReadLocalWorkOverviewWorkerSourceOptionalAndErrorsScrubbed(t *testing.T) {
	legacy := readLocalWorkOverview(context.Background(), legacyLocalWorkTestSource{}, t.TempDir(), time.Now())
	b, _ := json.Marshal(legacy)
	if !strings.Contains(string(b), `"items":[]`) {
		t.Fatalf("items must be []: %s", b)
	}
	failed := readLocalWorkOverview(context.Background(), localWorkTestSource{workerErr: errors.New(`C:\secret\detail`)}, t.TempDir(), time.Now())
	if failed.WorkerWorktrees.Error != "unavailable" {
		t.Fatalf("error = %q", failed.WorkerWorktrees.Error)
	}
}
