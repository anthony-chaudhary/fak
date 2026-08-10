package main

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuefanout"
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// TestSelfcheckPasses is the demo's own gate: the -selfcheck path must return 0 on
// the current tree. It is the unit-test twin of the headless smoke witness.
func TestSelfcheckPasses(t *testing.T) {
	if rc := selfcheck(); rc != 0 {
		t.Fatalf("selfcheck() = %d, want 0", rc)
	}
}

// TestSpineFirstRefusalEnforced pins the guard path the demo advertises: a spine-less
// input is a deliberate contract refusal that names the missing field.
func TestSpineFirstRefusalEnforced(t *testing.T) {
	msg, outcome := refusalNoSpine()
	if outcome != issuefanout.OutcomeRefused {
		t.Fatalf("outcome = %q, want %q", outcome, issuefanout.OutcomeRefused)
	}
	if !strings.Contains(msg, "spine_ref is required") {
		t.Fatalf("refusal %q does not name the missing spine_ref", msg)
	}
}

// TestPlanIsDeterministicAndDispatchable pins the fan-out path: a supplied spine builds
// a plan whose every candidate carries the leaf marker-key prefix and passes the issue
// contract as dispatchable, and the plan is byte-identical across two builds.
func TestPlanIsDeterministicAndDispatchable(t *testing.T) {
	plan, err := issuefanout.Build(demoSpine())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if len(plan.Candidates) < issuefanout.MinFanout {
		t.Fatalf("candidates = %d, below fan-out floor %d", len(plan.Candidates), issuefanout.MinFanout)
	}
	sum := 0
	for _, n := range plan.AreaCounts {
		sum += n
	}
	if sum != len(plan.Candidates) {
		t.Fatalf("area counts sum %d != candidate count %d", sum, len(plan.Candidates))
	}
	for _, c := range plan.Candidates {
		if !strings.HasPrefix(c.Key, "fanout-issuefanout-") {
			t.Errorf("candidate key %q lacks marker-key prefix", c.Key)
		}
		if got := issuepolicy.ReviewCandidate(c, issuepolicy.Options{}).Dispatchability; got != issuepolicy.Dispatchable {
			t.Errorf("candidate %s dispatchability = %q, want %q", c.Key, got, issuepolicy.Dispatchable)
		}
	}
	plan2, err := issuefanout.Build(demoSpine())
	if err != nil {
		t.Fatalf("second Build: %v", err)
	}
	if !reflect.DeepEqual(plan, plan2) {
		t.Fatal("plan is not deterministic across two builds")
	}
}
