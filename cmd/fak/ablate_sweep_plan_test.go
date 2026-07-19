package main

import (
	"strings"
	"testing"
)

// #2831 opened the closed KnownFeatures enum into a concept registry and added the
// bounded sweep planner (internal/ablate/registry.go, planner.go). The registry and
// planner contracts are unit-tested in internal/ablate; what had no witness is the CLI
// surface that selects a plan. These pin the fail-loud half of that surface.
//
// The stake is silent degradation, not a typo'd flag: --sweep-plan sits next to the
// legacy --sweep full-lattice path in runAblate, and both share the same configs slice.
// A plan token that is unknown (or accepted-but-unbuilt, like greedy) must exit 2 with a
// diagnostic. If it ever fell through to BuildSweep instead, the operator would get a
// rendered table for a sweep they never asked for and no signal that the plan they named
// was never run — the exact "fabricated pass" the ablation harness exists to prevent.
//
// These arms deliberately assert only the paths that short-circuit BEFORE any sweep
// executes: a main/pairwise happy path re-execs one child per arm (rung 2,
// SweepViaSubprocess over os.Executable()), which under `go test` would re-exec the test
// binary. The executed-plan contract stays covered by internal/ablate's unit tests.

func TestAblateSweepPlanUnknownFailsLoud(t *testing.T) {
	code, out, errb := runAB("--sweep-plan", "bogus")
	if code != 2 {
		t.Fatalf("exit=%d, want 2 (usage); stdout=%s stderr=%s", code, out, errb)
	}
	if !strings.Contains(errb, `unknown sweep plan "bogus"`) {
		t.Fatalf("stderr missing the unknown-plan diagnostic: %s", errb)
	}
	if strings.Contains(out, "deltas vs") {
		t.Fatalf("unknown plan rendered a sweep table instead of refusing:\n%s", out)
	}
}

// greedy is named in #2831's design prose but is not built: BuildSweepPlan refuses it
// rather than approximating it. Pin the refusal so it cannot quietly become a main-effect
// sweep wearing the greedy name — a bundle search that silently returns single-concept
// arms would misreport which bundle the operator is being told to ship.
func TestAblateSweepPlanGreedyRefusesUntilWired(t *testing.T) {
	code, out, errb := runAB("--sweep-plan", "greedy")
	if code != 2 {
		t.Fatalf("exit=%d, want 2 (usage); stdout=%s stderr=%s", code, out, errb)
	}
	if !strings.Contains(errb, "not yet wired") {
		t.Fatalf("stderr missing the greedy-unwired diagnostic: %s", errb)
	}
	if strings.Contains(out, "deltas vs") {
		t.Fatalf("greedy plan rendered a sweep table instead of refusing:\n%s", out)
	}
}
