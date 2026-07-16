package benchloop

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchruns"
	"github.com/anthony-chaudhary/fak/internal/nightrun"
)

// fullCommit is a deterministic 40-hex HEAD; fleetRunID embeds its 16-char prefix the
// way the fleet bench harness names a run directory (…-bench-fleet-<sha>-<utc>).
const (
	fullCommit = "d7ed1f6afa901604aaaabbbbccccddddeeee0000"
	otherHead  = "0000eeeeddddccccbbbbaaaa406109afa6f1ed7d"
	fleetRunID = "box-a-bench-fleet-d7ed1f6afa901604-20260715T054347Z"
)

// reuseParts builds a status input whose only feasible datum is an auto-runnable task
// on box-a, so chooseAction lands in the collect/reuse branch. commit is the current
// HEAD lineage; runs is the recorded catalog.
func reuseParts(commit string, runs []benchruns.Run) Parts {
	return Parts{
		Root:    "/repo",
		Now:     reuseTime("2026-07-16T00:00:00Z"),
		Commit:  commit,
		Catalog: benchruns.Catalog{Runs: runs},
		Caps:    nightrun.Capabilities{Box: "box-a", GPU: "none", Net: true, Creds: map[string]bool{}},
		Tasks: []nightrun.Task{
			{ID: "collect-me", Value: nightrun.ValueCoverage, Run: "echo 12 tok/s", Acceptance: "12 tok/s"},
		},
	}
}

func reuseTime(s string) time.Time {
	tm, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err)
	}
	return tm.UTC()
}

// TestReuseSkipsRedundantSameLineageRun is the #4600 failure-class proof: a prior run
// at the SAME commit on the SAME box makes the next run redundant, so the launch gate
// must emit reuse_run (skip) instead of collect_local (execute). Before the fix,
// chooseAction always returned collect_local here — the wasted re-run the issue names.
func TestReuseSkipsRedundantSameLineageRun(t *testing.T) {
	prior := benchruns.Run{
		"run_id": fleetRunID, "machine_id": "box-a", "model": "qwen2.5-3b", "precision": "q8",
		"timestamp": "2026-07-15T05:43:47Z", "path": "experiments/benchmark/runs/by-machine/box-a/" + fleetRunID,
	}
	rep := StatusFromParts(reuseParts(fullCommit, []benchruns.Run{prior}))

	if rep.Local.Next == nil || rep.Local.Next.ID != "collect-me" {
		t.Fatalf("precondition: next datum = %+v, want collect-me", rep.Local.Next)
	}
	if !rep.Reuse.Reuse || rep.Reuse.RunID != fleetRunID {
		t.Fatalf("reuse decision = %+v, want reuse of %s", rep.Reuse, fleetRunID)
	}
	if rep.NextAction.Kind != "reuse_run" {
		t.Fatalf("action = %+v, want reuse_run (skip the redundant re-run)", rep.NextAction)
	}
	if !strings.Contains(rep.NextAction.Command, fleetRunID) {
		t.Fatalf("reuse_run command must point at the prior artifact, got %q", rep.NextAction.Command)
	}
	if !strings.Contains(RenderStatus(rep), "reuse: prior run "+fleetRunID) {
		t.Fatalf("status render must surface the reuse; got:\n%s", RenderStatus(rep))
	}
}

// TestReuseRerunsOnCommitDrift proves the freshness predicate: a prior run built at a
// DIFFERENT commit is not reused — the wasted-rerun skip must not fire when the code
// under test changed (the conservative realization of DetectInvalidation).
func TestReuseRerunsOnCommitDrift(t *testing.T) {
	prior := benchruns.Run{
		"run_id": fleetRunID, "machine_id": "box-a", "timestamp": "2026-07-15T05:43:47Z",
	}
	rep := StatusFromParts(reuseParts(otherHead, []benchruns.Run{prior}))
	if rep.Reuse.Reuse {
		t.Fatalf("commit drift must not reuse; got %+v", rep.Reuse)
	}
	if rep.NextAction.Kind != "collect_local" {
		t.Fatalf("action = %+v, want collect_local on commit drift", rep.NextAction)
	}
}

// TestReuseRerunsOnMachineDrift proves the machine axis: a prior run for the same
// commit on a DIFFERENT box is not reusable here (that box's number is not this box's).
func TestReuseRerunsOnMachineDrift(t *testing.T) {
	prior := benchruns.Run{
		"run_id": fleetRunID, "machine_id": "box-z", "timestamp": "2026-07-15T05:43:47Z",
	}
	rep := StatusFromParts(reuseParts(fullCommit, []benchruns.Run{prior}))
	if rep.Reuse.Reuse {
		t.Fatalf("machine drift must not reuse; got %+v", rep.Reuse)
	}
	if rep.NextAction.Kind != "collect_local" {
		t.Fatalf("action = %+v, want collect_local on machine drift", rep.NextAction)
	}
}

// TestReuseHonorsForceRerunEscapeHatch proves the --no-reuse fence: with the escape
// hatch set, an exact-lineage match still executes (never silently skips).
func TestReuseHonorsForceRerunEscapeHatch(t *testing.T) {
	t.Setenv(NoReuseEnv, "1")
	prior := benchruns.Run{
		"run_id": fleetRunID, "machine_id": "box-a", "timestamp": "2026-07-15T05:43:47Z",
	}
	rep := StatusFromParts(reuseParts(fullCommit, []benchruns.Run{prior}))
	if rep.Reuse.Reuse {
		t.Fatalf("%s must force a re-run; got reuse %+v", NoReuseEnv, rep.Reuse)
	}
	if rep.NextAction.Kind != "collect_local" {
		t.Fatalf("action = %+v, want collect_local under force-rerun", rep.NextAction)
	}
}

// TestLineageReuseUnit exercises the pure predicate across every axis.
func TestLineageReuseUnit(t *testing.T) {
	runs := []benchruns.Run{
		{"run_id": "r-old", "machine_id": "box-a", "model": "m1", "precision": "q8", "git_commit": fullCommit, "timestamp": "2026-07-10T00:00:00Z"},
		{"run_id": "r-new", "machine_id": "box-a", "model": "m1", "precision": "q8", "git_commit": fullCommit, "timestamp": "2026-07-15T00:00:00Z"},
	}
	// commit+machine wildcard-config → reuse the freshest covering run.
	if d := LineageReuse(runs, LineageKey{Commit: fullCommit, Machine: "box-a"}); !d.Reuse || d.RunID != "r-new" {
		t.Fatalf("wildcard config: %+v, want reuse r-new", d)
	}
	// pinned model/precision that match → still reuse.
	if d := LineageReuse(runs, LineageKey{Commit: fullCommit, Machine: "box-a", Model: "m1", Precision: "q8"}); !d.Reuse {
		t.Fatalf("pinned matching config: %+v, want reuse", d)
	}
	// pinned precision that differs → no reuse (config drift).
	if d := LineageReuse(runs, LineageKey{Commit: fullCommit, Machine: "box-a", Precision: "q4"}); d.Reuse {
		t.Fatalf("precision drift: %+v, want no reuse", d)
	}
	// empty commit → hard no-reuse.
	if d := LineageReuse(runs, LineageKey{Machine: "box-a"}); d.Reuse {
		t.Fatalf("empty commit: %+v, want no reuse", d)
	}
}

// TestRunCommitAndMatch pins commit extraction and truncation-tolerant matching.
func TestRunCommitAndMatch(t *testing.T) {
	// explicit field wins.
	if got := runCommit(benchruns.Run{"git_commit": fullCommit, "run_id": "x"}); got != fullCommit {
		t.Fatalf("runCommit explicit = %q, want %q", got, fullCommit)
	}
	// fall back to the fleet-embedded 16-hex token, not the 8-digit date.
	if got := runCommit(benchruns.Run{"run_id": fleetRunID}); got != "d7ed1f6afa901604" {
		t.Fatalf("runCommit embedded = %q, want d7ed1f6afa901604", got)
	}
	// unknown/absent → empty (no false commit).
	if got := runCommit(benchruns.Run{"git_commit": "unknown", "run_id": "box-a-smollm2-20260618"}); got != "" {
		t.Fatalf("runCommit date-only = %q, want empty", got)
	}
	// 16-hex prefix matches the 40-hex HEAD; a short/empty token never matches.
	if !commitMatch("d7ed1f6afa901604", fullCommit) {
		t.Fatal("commitMatch prefix should hold")
	}
	if commitMatch("d7ed1f6a", fullCommit) {
		t.Fatal("commitMatch must reject an <12-char token")
	}
	if commitMatch("", fullCommit) {
		t.Fatal("commitMatch must reject empty")
	}
}
