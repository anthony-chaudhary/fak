package frontierswe

import "testing"

// TestBuildRunDrivesTurnsAndTrace is the C9 acceptance witness at the pure layer:
// a run drives >=1 turn, stamps the fak.frontierswe.run.v1 meta, and folds a
// non-empty per-turn TTS trace whose C8 reuse series bit.
func TestBuildRunDrivesTurnsAndTrace(t *testing.T) {
	task := &Task{Name: "git-to-zig"}
	task.Agent.TimeoutSec = 72000.0
	task.Job.Artifacts = []string{"solution.patch", "agent.log", "reward.json"}

	r := BuildRun(RunConfig{Task: task, Turns: 3, PredsOnly: true})

	if r.Meta.Schema != RunSchema {
		t.Fatalf("meta schema = %q, want %q", r.Meta.Schema, RunSchema)
	}
	if r.Meta.Turns < 1 {
		t.Fatalf("run drove %d turns, want >=1", r.Meta.Turns)
	}
	if r.Meta.Turns != 3 {
		t.Fatalf("run drove %d turns, want 3", r.Meta.Turns)
	}
	if len(r.Trace.Points) != 3 {
		t.Fatalf("trace has %d points, want 3", len(r.Trace.Points))
	}
	if !r.Trace.CacheSeries.CacheBit {
		t.Fatalf("C8 reuse series did not bite: %+v", r.Trace.CacheSeries)
	}
	if r.Trace.CacheSeries.RealizedReuseRate <= 0 {
		t.Fatalf("realized reuse rate = %v, want > 0", r.Trace.CacheSeries.RealizedReuseRate)
	}
	if !r.Meta.PredsOnly || !r.Meta.Mocked {
		t.Fatalf("expected preds-only mocked run, got %+v", r.Meta)
	}
	// Cumulative wall-clock is monotone and never exceeds the [agent] budget.
	var prev float64
	for i, p := range r.Trace.Points {
		if p.CumWallSec < prev {
			t.Fatalf("point %d wall %.2f < prev %.2f (not monotone)", i, p.CumWallSec, prev)
		}
		prev = p.CumWallSec
	}
	if r.Trace.TotalWallSec > float64(r.Meta.BudgetSec) {
		t.Fatalf("total wall %.2f exceeds budget %d", r.Trace.TotalWallSec, r.Meta.BudgetSec)
	}
}

// TestBuildRunRespectsBudget caps the driven turns at the budget-projected
// trajectory length: asking for more turns than the 20h budget projects must not
// overrun the budget.
func TestBuildRunRespectsBudget(t *testing.T) {
	task := &Task{Name: "tiny"}
	task.Agent.TimeoutSec = 3600.0 // 1h -> ProjectedTurns == turnsPerHour (100)

	projected := ProjectedTurns(task.Agent.TimeoutSec)
	r := BuildRun(RunConfig{Task: task, Turns: projected + 5000})

	if r.Meta.Turns != projected {
		t.Fatalf("turns = %d, want capped at projected %d", r.Meta.Turns, projected)
	}
	if r.Trace.TotalWallSec > task.Agent.TimeoutSec+1e-6 {
		t.Fatalf("total wall %.2f exceeds 1h budget %.2f", r.Trace.TotalWallSec, task.Agent.TimeoutSec)
	}
}

// TestBuildRunCollectsJobArtifacts records the job.yaml artifact list and derives
// the submission target and files from it.
func TestBuildRunCollectsJobArtifacts(t *testing.T) {
	task := &Task{Name: "cranelift-codegen-opt"}
	task.Agent.TimeoutSec = 72000.0
	task.Job.Artifacts = []string{"/app/wasmtime", "/logs/agent", "/logs/verifier"}

	r := BuildRun(RunConfig{Task: task, Turns: 2})

	if len(r.Artifacts) != 3 {
		t.Fatalf("collected %d artifacts, want 3: %+v", len(r.Artifacts), r.Artifacts)
	}
	for _, a := range r.Artifacts {
		if !a.Collected {
			t.Errorf("artifact %q not collected", a.Name)
		}
	}
	if r.Submission.Target != "/app/cranelift-codegen-opt" {
		t.Fatalf("submission target = %q, want /app/cranelift-codegen-opt", r.Submission.Target)
	}
	if len(r.Submission.Files) != 3 {
		t.Fatalf("submission lists %d files, want 3", len(r.Submission.Files))
	}
}

// TestBuildRunArtifactFallback uses the canonical trio when job.yaml declares no
// artifact list, so the collection shape is always exercised.
func TestBuildRunArtifactFallback(t *testing.T) {
	task := &Task{Name: "no-job-yaml"}
	task.Agent.TimeoutSec = 72000.0

	r := BuildRun(RunConfig{Task: task, Turns: 1})
	if len(r.Artifacts) != 3 {
		t.Fatalf("fallback collected %d artifacts, want 3", len(r.Artifacts))
	}
}
