package main

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// A pool past the advisory setpoint is inspected with one git status per retained
// worktree. On a loaded host those statuses do not return, so the advisory has to
// give up on its own budget instead of holding the prepare that already finished.
//
// fak-test:runtime fast est=1s lane=default
func TestWorktreeWorkerCapacityAdvisoryIsBoundedPastSetpoint(t *testing.T) {
	old := worktreeWorkerCapacityAdvisoryBudget
	worktreeWorkerCapacityAdvisoryBudget = 50 * time.Millisecond
	t.Cleanup(func() { worktreeWorkerCapacityAdvisoryBudget = old })

	paths := make([]string, workerworktree.AdvisoryCapacitySetpoint+1)
	for i := range paths {
		paths[i] = fmt.Sprintf("/wt/fak-worker-wt-lane-%03d", i)
	}
	var inspected atomic.Int64
	probes := worktreeWorkerLifecycleProbes{
		ReadOwner: func(string) (workerworktree.OwnerStamp, error) {
			return workerworktree.OwnerStamp{PID: 1, LeaseID: "lease"}, nil
		},
		ProcessAlive: func(int) (bool, error) { return false, nil },
		LeaseLive:    func(string) (bool, error) { return false, nil },
		InspectContext: func(ctx context.Context, _, _ string) (worktreeWorkerRevisionEvidence, error) {
			inspected.Add(1)
			// A wedged status: it ends only when cancelled. The fallback keeps an
			// unbounded caller finite so the regression reads as a failure, not a hang.
			select {
			case <-ctx.Done():
				return worktreeWorkerRevisionEvidence{}, ctx.Err()
			case <-time.After(3 * time.Second):
				return worktreeWorkerRevisionEvidence{}, nil
			}
		},
	}

	start := time.Now()
	advisory := worktreeWorkerCapacityAdvisoryWith("/repo", workerworktree.CapacityCensus{Known: true, Paths: paths}, len(paths), "", nil, probes)
	elapsed := time.Since(start)

	if inspected.Load() == 0 {
		t.Fatal("the pool was never inspected; the test no longer exercises the advisory inventory")
	}
	if elapsed > 2*time.Second {
		t.Fatalf("capacity advisory took %s inspecting %d worktrees, want it bounded by its %s budget", elapsed, len(paths), worktreeWorkerCapacityAdvisoryBudget)
	}
	if !advisory.Allowed || advisory.Status != workerworktree.CapacityJustificationNeeded {
		t.Fatalf("a truncated inspection must still fail open with the over-setpoint advisory: %+v", advisory)
	}
	if !strings.Contains(advisory.Message, "recommendations are incomplete") {
		t.Fatalf("a truncated inspection must be visible to the operator, got %q", advisory.Message)
	}
}
