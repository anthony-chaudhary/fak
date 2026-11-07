package main

import (
	"testing"
	"time"
)

// The prepare budget must stay a bounded default: contention surfaces as a
// typed PREPARE_TIMEOUT, never an unbounded stall (internal/workerworktree
// TestPrepareOwnedBoundedTimeoutEmitsNoOwnerReceipt covers the bounded-runner
// behavior at the layer the flag feeds). --prepare-budget raises the bound per
// invocation on slow hosts; it must never move the default.
func TestWorkerWorktreePrepareBudgetDefaultStaysBounded(t *testing.T) {
	if defaultWorkerPrepareBudget != 2*time.Minute {
		t.Fatalf("defaultWorkerPrepareBudget = %s, want 2m: the default must stay tight so a wedged prepare fails closed fast; slow hosts raise it per invocation via --prepare-budget", defaultWorkerPrepareBudget)
	}
}
