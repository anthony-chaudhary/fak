package main

import (
	"bytes"
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchloop"
	"github.com/anthony-chaudhary/fak/internal/benchruns"
	"github.com/anthony-chaudhary/fak/internal/nightrun"
)

// exactLineageParts builds a launch-gate input whose single feasible datum is an
// auto-runnable task on box-a and whose catalog already holds a prior run at the
// SAME commit on that box — the exact-lineage catalog #4600's reuse gate skips
// (reuse_run) unless the force-rerun escape hatch is set.
func exactLineageParts() benchloop.Parts {
	const (
		fullCommit = "d7ed1f6afa901604aaaabbbbccccddddeeee0000"
		fleetRunID = "box-a-bench-fleet-d7ed1f6afa901604-20260715T054347Z"
	)
	now, _ := time.Parse(time.RFC3339, "2026-07-16T00:00:00Z")
	return benchloop.Parts{
		Root:   "/repo",
		Now:    now.UTC(),
		Commit: fullCommit,
		Catalog: benchruns.Catalog{Runs: []benchruns.Run{{
			"run_id": fleetRunID, "machine_id": "box-a", "timestamp": "2026-07-15T05:43:47Z",
		}}},
		Caps: nightrun.Capabilities{Box: "box-a", GPU: "none", Net: true, Creds: map[string]bool{}},
		Tasks: []nightrun.Task{
			{ID: "collect-me", Value: nightrun.ValueCoverage, Run: "echo 12 tok/s", Acceptance: "12 tok/s"},
		},
	}
}

// TestBenchLoopNoReuseFlagForcesCollectLocal is the #5086 front-door proof: the
// cmd-lane --no-reuse flag exports FAK_BENCH_NO_REUSE=1 before benchloop.Load, so
// the launch gate on an exact-lineage catalog flips from reuse_run (skip) to
// collect_local (force re-run).
func TestBenchLoopNoReuseFlagForcesCollectLocal(t *testing.T) {
	// Baseline: with the escape hatch clear, the exact-lineage catalog skips.
	t.Setenv(benchloop.NoReuseEnv, "")
	if got := benchloop.StatusFromParts(exactLineageParts()).NextAction.Kind; got != "reuse_run" {
		t.Fatalf("precondition: action = %q, want reuse_run on an exact-lineage catalog", got)
	}

	// Route status through runBenchLoop WITH --no-reuse: the dispatcher must accept
	// the flag and export the env before it reaches benchloop.Load.
	t.Setenv("FAK_OFFLINE", "1")
	root := t.TempDir()
	var out, errb bytes.Buffer
	if code := runBenchLoop(&out, &errb, []string{"status", "--no-reuse", "--workspace", root, "--now", "2026-07-16T00:00:00Z"}); code != 0 {
		t.Fatalf("status --no-reuse code=%d stderr=%s", code, errb.String())
	}
	if v := os.Getenv(benchloop.NoReuseEnv); noReuseFlagTruthy(v) == false {
		t.Fatalf("%s = %q after --no-reuse, want a truthy export", benchloop.NoReuseEnv, v)
	}

	// With the env now exported by the flag, the same exact-lineage catalog forces
	// a re-run rather than reusing the prior artifact.
	if got := benchloop.StatusFromParts(exactLineageParts()).NextAction.Kind; got != "collect_local" {
		t.Fatalf("--no-reuse: action = %q, want collect_local (force re-run)", got)
	}
}

// TestStripNoReuseFlag proves the escape hatch is pulled out of argv uniformly for
// status|next|run, in either order relative to the subcommand token, and that an
// explicit falsey value leaves the gate on.
func TestStripNoReuseFlag(t *testing.T) {
	for _, tc := range []struct {
		name      string
		argv      []string
		wantForce bool
		wantRest  []string
	}{
		{"status", []string{"status", "--no-reuse"}, true, []string{"status"}},
		{"next-json", []string{"next", "--no-reuse", "--json"}, true, []string{"next", "--json"}},
		{"run-apply", []string{"run", "--no-reuse", "--apply"}, true, []string{"run", "--apply"}},
		{"flag-first", []string{"--no-reuse", "status"}, true, []string{"status"}},
		{"single-dash", []string{"status", "-no-reuse"}, true, []string{"status"}},
		{"absent", []string{"status", "--json"}, false, []string{"status", "--json"}},
		{"explicit-false", []string{"status", "--no-reuse=false"}, false, []string{"status"}},
		{"explicit-true", []string{"status", "--no-reuse=1"}, true, []string{"status"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rest, force := stripNoReuseFlag(tc.argv)
			if force != tc.wantForce {
				t.Fatalf("force = %v, want %v", force, tc.wantForce)
			}
			if len(rest) != len(tc.wantRest) {
				t.Fatalf("rest = %v, want %v", rest, tc.wantRest)
			}
			for i := range rest {
				if rest[i] != tc.wantRest[i] {
					t.Fatalf("rest = %v, want %v", rest, tc.wantRest)
				}
			}
		})
	}
}
