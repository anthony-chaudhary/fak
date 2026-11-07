package committedtree

import (
	"context"
	"testing"
	"time"
)

// The extraction budget must stay a bounded default: a wedged extraction fails
// closed fast as a typed timeout, never an unbounded stall. The env override
// raises the bound per invocation on loaded hosts; it must never move the
// default, and a malformed or non-positive override falls back to the default.
func TestExtractionBudgetDefaultStaysBounded(t *testing.T) {
	t.Setenv(extractionBudgetEnv, "")
	if got := extractionBudget(); got != defaultTimeout {
		t.Fatalf("extractionBudget() with no override = %s, want defaultTimeout=%s", got, defaultTimeout)
	}
	if defaultTimeout != 2*time.Minute {
		t.Fatalf("defaultTimeout = %s, want 2m: the default must stay tight; slow hosts raise it per invocation via %s", defaultTimeout, extractionBudgetEnv)
	}
}

func TestExtractionBudgetEnvOverride(t *testing.T) {
	t.Setenv(extractionBudgetEnv, "20m")
	if got := extractionBudget(); got != 20*time.Minute {
		t.Fatalf("extractionBudget() with override 20m = %s, want 20m", got)
	}
}

func TestExtractionBudgetEnvOverrideFallsBackOnGarbage(t *testing.T) {
	for _, bad := range []string{"abc", "0s", "-5m", "  "} {
		t.Setenv(extractionBudgetEnv, bad)
		if got := extractionBudget(); got != defaultTimeout {
			t.Fatalf("extractionBudget() with override %q = %s, want defaultTimeout=%s (garbage and non-positive overrides must fall back)", bad, got, defaultTimeout)
		}
	}
}

// The resolved budget rides the extraction context so timeout diagnostics
// report the budget that actually bounded the run, not the package default.
func TestExtractionStopErrorReportsResolvedBudget(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	ctx = context.WithValue(ctx, budgetKey{}, 20*time.Minute)
	defer cancel()
	<-ctx.Done()
	err := extractionStopError(ctx)
	if err == nil {
		t.Fatal("extractionStopError on a done context must return an error")
	}
	want := "committed tree extraction timed out after 20m0s"
	if got := err.Error(); got[:len(want)] != want {
		t.Fatalf("extractionStopError = %q, want prefix %q (must report the resolved budget)", got, want)
	}
}
