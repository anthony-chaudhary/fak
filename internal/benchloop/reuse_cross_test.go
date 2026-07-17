package benchloop

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/benchruns"
)

// crossHooks builds the #5088 evidence hooks for tests: the prior run's artifact is
// a fixed identity recorded at fullCommit, and the diff window prevCommit..HEAD
// resolves to changed. Both hooks record what they were asked so a test can pin the
// window orientation.
func crossHooks(t *testing.T, changed []string) CrossCommit {
	t.Helper()
	return CrossCommit{
		Artifact: func(r benchruns.Run) (benchcli.BenchmarkArtifact, bool) {
			return benchcli.BenchmarkArtifact{
				Schema:    benchcli.BenchmarkArtifactSchema,
				RunID:     runString(r, "run_id"),
				FAKCommit: fullCommit,
				Harness:   benchcli.HarnessInfo{Name: "fakbench"},
				Model:     benchcli.ModelSnapshot{Name: "qwen2.5-3b", Precision: "q8"},
				Config:    benchcli.ConfigSnapshot{Hash: "cfg-1"},
			}, true
		},
		ChangedPaths: func(prevCommit, headCommit string) ([]string, bool) {
			if !strings.HasPrefix(fullCommit, prevCommit) {
				t.Fatalf("changed-paths window starts at %q, want the prior run commit %q", prevCommit, fullCommit)
			}
			if headCommit != otherHead {
				t.Fatalf("changed-paths window ends at %q, want HEAD %q", headCommit, otherHead)
			}
			return changed, true
		},
	}
}

// TestCrossCommitReuseSkipsWhenWindowIsNonBench is the #5088 done-condition's first
// half: a bench run recorded at commit A is REUSED at HEAD B when B..A touched only
// non-bench paths — the commit-exact rule alone would have wasted a re-run here.
func TestCrossCommitReuseSkipsWhenWindowIsNonBench(t *testing.T) {
	prior := benchruns.Run{
		"run_id": fleetRunID, "machine_id": "box-a", "timestamp": "2026-07-15T05:43:47Z",
		"path": "experiments/benchmark/runs/by-machine/box-a/" + fleetRunID,
	}
	p := reuseParts(otherHead, []benchruns.Run{prior})
	p.Cross = crossHooks(t, []string{"docs/notes/SOMETHING.md", "README.md"})
	rep := StatusFromParts(p)

	if !rep.Reuse.Reuse || rep.Reuse.RunID != fleetRunID {
		t.Fatalf("cross-commit reuse = %+v, want reuse of %s", rep.Reuse, fleetRunID)
	}
	if rep.NextAction.Kind != "reuse_run" {
		t.Fatalf("action = %+v, want reuse_run across a non-bench commit window", rep.NextAction)
	}
}

// TestCrossCommitRerunsOnInvalidatingPath is the second half: the same prior run is
// RE-RUN when the window touches a path DetectInvalidation classifies as
// invalidating (bench code under internal/model/).
func TestCrossCommitRerunsOnInvalidatingPath(t *testing.T) {
	prior := benchruns.Run{
		"run_id": fleetRunID, "machine_id": "box-a", "timestamp": "2026-07-15T05:43:47Z",
	}
	p := reuseParts(otherHead, []benchruns.Run{prior})
	p.Cross = crossHooks(t, []string{"internal/model/decode.go"})
	rep := StatusFromParts(p)

	if rep.Reuse.Reuse {
		t.Fatalf("invalidating window must not reuse; got %+v", rep.Reuse)
	}
	if rep.NextAction.Kind != "collect_local" {
		t.Fatalf("action = %+v, want collect_local when the window invalidates", rep.NextAction)
	}
	if !strings.Contains(rep.Reuse.Reason, "invalidated") {
		t.Fatalf("reason should surface the invalidation verdict; got %q", rep.Reuse.Reason)
	}
}

// TestCrossCommitFailsClosedWithoutEvidence pins the conservative fallbacks: an
// unresolvable diff window or an unreadable artifact degrades to the commit-exact
// no-reuse verdict rather than guessing.
func TestCrossCommitFailsClosedWithoutEvidence(t *testing.T) {
	runs := []benchruns.Run{{
		"run_id": fleetRunID, "machine_id": "box-a", "timestamp": "2026-07-15T05:43:47Z",
	}}
	key := LineageKey{Commit: otherHead, Machine: "box-a"}

	noWindow := crossHooks(t, nil)
	noWindow.ChangedPaths = func(_, _ string) ([]string, bool) { return nil, false }
	if d := LineageReuseAcross(runs, key, noWindow); d.Reuse {
		t.Fatalf("unresolvable window must not reuse; got %+v", d)
	}

	noArtifact := crossHooks(t, nil)
	noArtifact.Artifact = func(benchruns.Run) (benchcli.BenchmarkArtifact, bool) {
		return benchcli.BenchmarkArtifact{}, false
	}
	if d := LineageReuseAcross(runs, key, noArtifact); d.Reuse {
		t.Fatalf("unreadable artifact must not reuse; got %+v", d)
	}

	if d := LineageReuseAcross(runs, key, CrossCommit{}); d.Reuse {
		t.Fatalf("zero-value hooks must keep the commit-exact rule; got %+v", d)
	}
}

// TestCrossCommitModelDriftInvalidates pins the model rung: with a pinned Next
// identity whose model differs from the prior artifact's, the window is invalid
// even when no path-based rule fires.
func TestCrossCommitModelDriftInvalidates(t *testing.T) {
	runs := []benchruns.Run{{
		"run_id": fleetRunID, "machine_id": "box-a", "timestamp": "2026-07-15T05:43:47Z",
	}}
	cc := crossHooks(t, []string{"docs/notes/SOMETHING.md"})
	cc.Next = benchcli.BenchmarkArtifact{Model: benchcli.ModelSnapshot{Name: "other-model", Precision: "q4"}}
	d := LineageReuseAcross(runs, LineageKey{Commit: otherHead, Machine: "box-a"}, cc)
	if d.Reuse {
		t.Fatalf("model drift must not reuse; got %+v", d)
	}
}
