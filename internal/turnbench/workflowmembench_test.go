package turnbench

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cdb"
	"github.com/anthony-chaudhary/fak/internal/contextq"
)

// wfmemFixture is the relative path to the shared CDB test fixture used by the
// contextq package — the same image, resolved from internal/turnbench.
const wfmemFixture = "../../testdata/cdb/session.jsonl"

// attachWFMemImage ingests and attaches the shared CDB test fixture so the
// workflow-memory benchmark runs against the same image as contextq's own tests.
func attachWFMemImage(tb testing.TB) *cdb.Image {
	tb.Helper()
	ctx := context.Background()
	rec, st, err := cdb.IngestSession(ctx, wfmemFixture, "wfmem-bench")
	if err != nil {
		tb.Fatalf("wfmem: ingest session: %v", err)
	}
	if st.Pages == 0 {
		tb.Fatal("wfmem: ingest recorded no pages")
	}
	dir := tb.TempDir()
	if err := rec.Persist(dir); err != nil {
		tb.Fatalf("wfmem: persist: %v", err)
	}
	im, err := cdb.Attach(dir)
	if err != nil {
		tb.Fatalf("wfmem: attach: %v", err)
	}
	return im
}

// wfmemReq is the deterministic, model-free workflow-memory workload: a 5-turn
// sequence (cold-build -> warm-reuse -> policy-drift-recompute -> warm-reuse ->
// poison-probe) that exercises the cache-economics and fail-closed trust paths.
var wfmemReq = contextq.BenchRequest{
	GoalQuery:   "refund fee account",
	PoisonQuery: "sealed trust violation secret exfil",
}

// TestWorkflowMemoryBench_MetricsPresent is the correctness gate: it drives the
// five-turn workload once and verifies the three reported metrics are present and
// internally consistent, and that the two fail-closed counters (StaleReuse,
// PoisonLeakage) are zero because the substrate enforces it — not because the
// workload never tried.
func TestWorkflowMemoryBench_MetricsPresent(t *testing.T) {
	im := attachWFMemImage(t)
	rep := contextq.RunWorkflowMemoryBench(context.Background(), im, wfmemReq)

	// (a) resident bytes: the demand-pageable universe must be non-zero.
	if rep.ResidentBytes <= 0 {
		t.Fatalf("resident_bytes not reported: %d", rep.ResidentBytes)
	}

	// (b) fault rate: must be in [0,1] and agree with the materialization mix.
	if rep.FaultRate < 0 || rep.FaultRate > 1 {
		t.Fatalf("fault_rate out of [0,1]: %f", rep.FaultRate)
	}
	if rep.Materializations > 0 {
		want := float64(rep.Faults) / float64(rep.Materializations)
		if rep.FaultRate != want {
			t.Fatalf("fault_rate = %f, want %f (faults=%d materializations=%d)",
				rep.FaultRate, want, rep.Faults, rep.Materializations)
		}
	}
	// The cache economics must actually be exercised so the fault_rate is not a
	// vacuous zero (which would happen if no pages were ever faulted).
	if rep.Faults == 0 {
		t.Fatal("cold build must fault raw pages to build views (fault_rate must be exercised)")
	}
	if rep.Hits == 0 {
		t.Fatal("warm-reuse turn must produce at least one cache HIT")
	}
	if rep.Recomputes == 0 {
		t.Fatal("policy-drift turn must trigger at least one RECOMPUTE")
	}

	// (c) stale/poison replay: the poison probe must be refused, and neither stale
	// views nor quarantined pages must enter the materialized set.
	if rep.SealedRefused == 0 {
		t.Fatal("poison probe never fired — sealed page was not refused (leakage gate is vacuous)")
	}
	if rep.StaleReuse != 0 {
		t.Fatalf("stale_reuse must be 0 (fail-closed): got %d", rep.StaleReuse)
	}
	if rep.PoisonLeakage != 0 {
		t.Fatalf("poison_leakage must be 0 (fail-closed): got %d", rep.PoisonLeakage)
	}

	if !rep.TaskSuccess {
		t.Fatal("goal query materialized no benign view: task failed")
	}
}

// TestWorkflowMemoryBench_Deterministic verifies that the workload is
// deterministic: two runs against the same image and request produce byte-
// identical aggregate metrics. The workload is model-free (no RNG, no wall-
// clock), so every field of WorkflowMemoryReport must be identical across runs.
func TestWorkflowMemoryBench_Deterministic(t *testing.T) {
	im := attachWFMemImage(t)
	ctx := context.Background()
	r1 := contextq.RunWorkflowMemoryBench(ctx, im, wfmemReq)
	r2 := contextq.RunWorkflowMemoryBench(ctx, im, wfmemReq)
	if r1 != r2 {
		t.Fatalf("workflow-memory workload is not deterministic:\n r1=%+v\n r2=%+v", r1, r2)
	}
}

// BenchmarkWorkflowMemory is the go test -bench entry point for the
// workflow-memory workload. It drives the deterministic 5-turn sequence
// (cold-build -> warm-reuse -> policy-drift-recompute -> warm-reuse ->
// poison-probe) over the attached CDB image and surfaces three metrics via
// b.ReportMetric so they appear in go test -bench output:
//
//	resident-bytes  — demand-pageable bytes of the attached image's working set
//	fault-rate      — fraction of materializations that required a raw page fault
//	stale-reuse     — stale-view served as HIT without recompute (fail-closed: 0)
//
// The poison-leaks counter (also fail-closed: 0) is the witness for the
// sealed/poison replay path required by issue #515.
func BenchmarkWorkflowMemory(b *testing.B) {
	im := attachWFMemImage(b)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()

	var rep contextq.WorkflowMemoryReport
	for i := 0; i < b.N; i++ {
		rep = contextq.RunWorkflowMemoryBench(ctx, im, wfmemReq)
	}

	// Workload-characteristic metrics are fixed per image/request; every iteration
	// produces the same values, so reporting after the loop is correct and avoids
	// overwriting them N times.
	b.ReportMetric(float64(rep.ResidentBytes), "resident-bytes")
	b.ReportMetric(rep.FaultRate, "fault-rate")
	b.ReportMetric(float64(rep.StaleReuse), "stale-reuse")
	b.ReportMetric(float64(rep.PoisonLeakage), "poison-leaks")
}
