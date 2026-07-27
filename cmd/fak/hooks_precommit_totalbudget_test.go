package main

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// TestGateBudgetForClampsToTheSharedWallClock is the #5335 whole-hook bound. A per-gate budget
// alone does not bound the hook: every gate that starts gets a fresh budget, so the worst case
// is (gate count x per-gate budget). gateBudgetFor is what converts that sum into a ceiling —
// each gate may only spend what is LEFT of the total.
func TestGateBudgetForClampsToTheSharedWallClock(t *testing.T) {
	perGate := 60 * time.Second

	// Plenty of total left: the gate gets its full per-gate budget, unchanged.
	if got, ok := gateBudgetFor(perGate, 90*time.Second); !ok || got != perGate {
		t.Fatalf("gateBudgetFor(60s, 90s) = (%s, %v), want (60s, true)", got, ok)
	}
	// Less total left than a full gate budget: clamp to what remains, so the gate cannot
	// overrun the hook's ceiling.
	if got, ok := gateBudgetFor(perGate, 10*time.Second); !ok || got != 10*time.Second {
		t.Fatalf("gateBudgetFor(60s, 10s) = (%s, %v), want (10s, true)", got, ok)
	}
	// Total spent: the remaining gates are skipped (fail-open), never restarted on a fresh budget.
	for _, spent := range []time.Duration{0, -time.Second, -time.Hour} {
		if got, ok := gateBudgetFor(perGate, spent); ok {
			t.Fatalf("gateBudgetFor(60s, %s) = (%s, true), want ok=false so the rest are skipped", spent, got)
		}
	}
}

// TestTotalBudgetBoundsTheWholeGateLoopUnderTheToolTimeout is the arithmetic that makes #5335 a
// feedback loop rather than a mere stall: a committer that outruns its ~120s tool timeout is
// SIGTERM'd mid-index-write and leaves a fresh stale `.git/index.lock`, wedging the next
// committer. The per-gate budget alone admits a worst case far past that timeout; the total is
// what keeps the hook under it.
func TestTotalBudgetBoundsTheWholeGateLoopUnderTheToolTimeout(t *testing.T) {
	const toolTimeout = 120 * time.Second

	gates := len(hooks.PreCommitGates())
	if gates < 2 {
		t.Fatalf("expected the real gate set, got %d gates", gates)
	}

	// The bound this change adds must sit under the timeout that manufactures stale locks.
	if defaultTotalBudget >= toolTimeout {
		t.Fatalf("defaultTotalBudget = %s, must be < the %s tool timeout it exists to stay under", defaultTotalBudget, toolTimeout)
	}

	// Without the total, the per-gate budget alone would allow this — the hole being closed.
	if perGateOnlyWorstCase := time.Duration(gates) * defaultCheckBudget; perGateOnlyWorstCase <= toolTimeout {
		t.Fatalf("per-gate-only worst case is %s over %d gates, already under the %s timeout — "+
			"this test no longer witnesses the hole the total budget closes", perGateOnlyWorstCase, gates, toolTimeout)
	}

	// With the total, the loop's ceiling is the total no matter how many gates there are.
	remaining := defaultTotalBudget
	for i := 0; i < gates; i++ {
		b, ok := gateBudgetFor(defaultCheckBudget, remaining)
		if !ok {
			break
		}
		remaining -= b
	}
	if remaining < 0 {
		t.Fatalf("the gate loop overspent the total by %s; the clamp is not holding", -remaining)
	}
}

// TestResolveTotalBudgetHonorsPositiveOverrideOnly pins the override contract, mirroring the
// per-gate budget's: a positive millisecond value wins, and a non-positive or unparseable value
// is ignored so the bound can never be disabled into the wedge it prevents.
func TestResolveTotalBudgetHonorsPositiveOverrideOnly(t *testing.T) {
	t.Setenv("FAK_PRECOMMIT_TOTAL_BUDGET_MS", "1500")
	if got := resolveTotalBudget(); got != 1500*time.Millisecond {
		t.Fatalf("override = %s, want 1500ms", got)
	}
	for _, bad := range []string{"0", "-5", "abc", ""} {
		t.Setenv("FAK_PRECOMMIT_TOTAL_BUDGET_MS", bad)
		if got := resolveTotalBudget(); got != defaultTotalBudget {
			t.Fatalf("override %q = %s, want default %s", bad, got, defaultTotalBudget)
		}
	}
}

// TestBoundedFailOpenExitCodeIsDistinctFromCouldNotRun pins why the timeout needs its own exit
// code. Exit 2 tells the shell wrapper to run the Python checkers instead — right when `fak` is
// simply absent, wrong when git itself is wedged, because every Python checker shells out to
// that same git with no timeout of its own and would rebuild the wedge the bound just escaped.
func TestBoundedFailOpenExitCodeIsDistinctFromCouldNotRun(t *testing.T) {
	if exitBoundedFailOpen == 2 {
		t.Fatal("bounded fail-open must NOT reuse exit 2; that falls through to the unbounded Python gates")
	}
	if exitBoundedFailOpen == 1 {
		t.Fatal("bounded fail-open must NOT reuse exit 1; a bound may only skip work, never add a refusal")
	}
	if exitBoundedFailOpen == 0 {
		t.Fatal("bounded fail-open must be distinguishable from a clean run so the skip stays visible")
	}
}
