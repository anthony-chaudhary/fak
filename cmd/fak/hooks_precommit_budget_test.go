package main

import (
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// TestCheckWithinBudgetSkipsAHangingGate is the #5335 fail-open property: a gate whose Check
// blocks far past its budget does NOT wedge the caller. checkWithinBudget returns
// errCheckBudgetExceeded at the budget (not when the gate finally unblocks), so the pre-commit
// loop skips the gate instead of hanging every commit in the clone. The hung gate stands in for
// the lock / O(refs) lease-fold that #5335 observed blocking the hook at near-zero CPU.
func TestCheckWithinBudgetSkipsAHangingGate(t *testing.T) {
	release := make(chan struct{})
	defer close(release) // let the abandoned goroutine finish when the test ends
	hung := hooks.Gate{Name: "HANG_TEST", Check: func(_ *hooks.StagedDiff) ([]hooks.Finding, error) {
		<-release
		return nil, nil
	}}

	start := time.Now()
	findings, err := checkWithinBudget(hung, &hooks.StagedDiff{}, 50*time.Millisecond)
	elapsed := time.Since(start)

	if !errors.Is(err, errCheckBudgetExceeded) {
		t.Fatalf("hanging gate err = %v, want errCheckBudgetExceeded", err)
	}
	if findings != nil {
		t.Fatalf("a cut-off gate must yield no findings, got %v", findings)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("checkWithinBudget waited %s — it must return at the budget, not for the gate", elapsed)
	}
}

// TestCheckWithinBudgetReturnsFastGateVerbatim proves the bound is transparent to a normal gate:
// one that returns within budget yields its own findings AND error unchanged, so wrapping every
// gate in the budget never alters a gate that is behaving.
func TestCheckWithinBudgetReturnsFastGateVerbatim(t *testing.T) {
	wantErr := errors.New("gate ran to completion")
	wantFindings := []hooks.Finding{{Gate: "FAST_TEST", Detail: "a real finding"}}
	fast := hooks.Gate{Name: "FAST_TEST", Check: func(_ *hooks.StagedDiff) ([]hooks.Finding, error) {
		return wantFindings, wantErr
	}}

	got, err := checkWithinBudget(fast, &hooks.StagedDiff{}, time.Second)
	if !errors.Is(err, wantErr) {
		t.Fatalf("fast gate error not returned verbatim: got %v want %v", err, wantErr)
	}
	if len(got) != 1 || got[0].Gate != "FAST_TEST" {
		t.Fatalf("fast gate findings not returned verbatim: %v", got)
	}
}

// TestResolveCheckBudgetHonorsPositiveOverrideOnly pins the override contract: a positive
// millisecond value wins, and a non-positive or unparseable value is ignored so the bound can
// never be disabled into the wedge it prevents.
func TestResolveCheckBudgetHonorsPositiveOverrideOnly(t *testing.T) {
	t.Setenv("FAK_PRECOMMIT_CHECK_BUDGET_MS", "250")
	if got := resolveCheckBudget(); got != 250*time.Millisecond {
		t.Fatalf("override = %s, want 250ms", got)
	}
	for _, bad := range []string{"0", "-5", "abc", ""} {
		t.Setenv("FAK_PRECOMMIT_CHECK_BUDGET_MS", bad)
		if got := resolveCheckBudget(); got != defaultCheckBudget {
			t.Fatalf("override %q = %s, want default %s", bad, got, defaultCheckBudget)
		}
	}
}
