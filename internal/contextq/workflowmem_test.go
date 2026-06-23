package contextq

import (
	"context"
	"testing"
)

// TestWorkflowMemoryBenchReportsAndFailsClosed is the witness for acceptance #5
// of issue #437: a workflow-memory benchmark that REPORTS resident bytes, the
// page/view fault rate, source coverage, stale reuse, poison leakage, and task
// success — and whose two trust numbers (stale reuse, poison leakage) are zero
// because the substrate enforces it, not because the workload never tried.
func TestWorkflowMemoryBenchReportsAndFailsClosed(t *testing.T) {
	im := attachFixture(t)
	rep := RunWorkflowMemoryBench(context.Background(), im, BenchRequest{
		GoalQuery:   "refund fee account",
		PoisonQuery: "sealed trust violation secret exfil",
	})

	t.Logf("workflow-memory report: %+v", rep)

	if rep.Turns != 5 {
		t.Fatalf("expected 5 turns, got %d", rep.Turns)
	}

	// The six named metrics must be present and internally consistent.
	if rep.ResidentBytes <= 0 {
		t.Fatalf("resident bytes not reported: %d", rep.ResidentBytes)
	}
	if rep.BenignPages <= 0 {
		t.Fatalf("benign page universe not reported: %d", rep.BenignPages)
	}
	if rep.Materializations != rep.Hits+rep.Faults {
		t.Fatalf("materialization mix inconsistent: mat=%d hits=%d faults=%d",
			rep.Materializations, rep.Hits, rep.Faults)
	}
	if rep.FaultRate < 0 || rep.FaultRate > 1 {
		t.Fatalf("fault rate out of [0,1]: %f", rep.FaultRate)
	}
	if rep.SourceCoverage <= 0 || rep.SourceCoverage > 1 {
		t.Fatalf("source coverage out of (0,1]: %f", rep.SourceCoverage)
	}
	if rep.PagesCovered > rep.BenignPages {
		t.Fatalf("covered %d > benign %d", rep.PagesCovered, rep.BenignPages)
	}

	// The cache economics must actually have been exercised — otherwise the
	// fault-rate and stale-reuse numbers would be vacuous.
	if rep.Faults == 0 {
		t.Fatal("cold build should have faulted raw pages to build views")
	}
	if rep.Hits == 0 {
		t.Fatal("warm reuse should have produced cache HITs")
	}
	if rep.Recomputes == 0 {
		t.Fatal("the policy-drift turn should have RECOMPUTED stale views")
	}

	// Fail-closed witnesses: both must be zero, and the sealed page must actually
	// have been refused (so the zero is earned, not vacuous).
	if rep.StaleReuse != 0 {
		t.Fatalf("stale view served as HIT (must fail closed): %d", rep.StaleReuse)
	}
	if rep.PoisonLeakage != 0 {
		t.Fatalf("a quarantined page leaked into a view (must fail closed): %d", rep.PoisonLeakage)
	}
	if rep.SealedRefused == 0 {
		t.Fatal("expected the sealed/poison page to be refused — the leakage probe never fired")
	}

	if !rep.TaskSuccess {
		t.Fatal("goal query materialized no benign view: task failed")
	}
}

func BenchmarkWorkflowMemory(b *testing.B) {
	im := attachFixture(b)
	req := BenchRequest{
		GoalQuery:   "refund fee account",
		PoisonQuery: "sealed trust violation secret exfil",
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = RunWorkflowMemoryBench(ctx, im, req)
	}
}
