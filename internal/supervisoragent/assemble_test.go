package supervisoragent

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetmon"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// TestWorkerStateFold pins the fleetmon-class -> WorkerState fold across the FULL
// closed classification set, plus the fail-safe default: any unmapped class folds
// to WorkerBlocked (needs attention), never WorkerHealthy. This is the one place
// the contract collapses fleetmon's seven states to the five the supervisor reasons
// over, so it is pinned so the collapse cannot drift silently.
func TestWorkerStateFold(t *testing.T) {
	cases := map[fleetmon.Classification]WorkerState{
		fleetmon.ClassHealthy:         WorkerHealthy,
		fleetmon.ClassCompletedFinal:  WorkerDone,
		fleetmon.ClassDead:            WorkerDead,
		fleetmon.ClassStaleTranscript: WorkerStale,
		fleetmon.ClassStaleChild:      WorkerStale,
		fleetmon.ClassAuthRateBlocked: WorkerBlocked,
		fleetmon.ClassAttention:       WorkerBlocked,
	}
	for cls, want := range cases {
		if got := workerStateFromClass(cls); got != want {
			t.Errorf("workerStateFromClass(%q) = %q, want %q", cls, got, want)
		}
	}
	// An unknown / future-added class must fail safe to blocked, never healthy.
	if got := workerStateFromClass(fleetmon.Classification("some-future-class")); got != WorkerBlocked {
		t.Errorf("unknown class folded to %q, want %q (fail-safe: never healthy)", got, WorkerBlocked)
	}
}

// TestAssembleProjectsPresentSurfaces drives the assembler with all four surfaces
// present and asserts it projects each upstream surface onto the closed contract:
// the census rows fold their class token to a WorkerState and keep identity; the
// leaseref view maps lane/lane_kind/tree onto Lease; the already-typed liveness and
// escalation heads pass through unchanged. Nothing is absent.
func TestAssembleProjectsPresentSurfaces(t *testing.T) {
	src := Sources{
		Liveness: Seen(Liveness{RunID: "RID-1", Class: "moving", Commits: 4}),
		Workers: Seen([]WorkerCensus{
			{RunID: "RID-1", Issue: "4478", Lane: "gateway", Class: fleetmon.ClassHealthy},
			{RunID: "RID-2", Issue: "4479", Lane: "cmd", Class: fleetmon.ClassAuthRateBlocked},
		}),
		Escalations: Seen([]Escalation{{ID: "E-1", RunID: "RID-2", Issue: "4479", Class: "auth", Severity: "operator", ReasonCode: "BLOCKED_AUTH"}}),
		Leases: Seen([]leaseref.ArbiterLease{
			{Lane: "gateway", LaneKind: "cluster", Tree: []string{"internal/gateway/**"}},
		}),
	}

	in := Assemble(src)

	if in.AnyAbsent() {
		t.Fatalf("all four surfaces were present, but AnyAbsent() reports absent: %v", in.AbsentWitnesses())
	}

	wantWorkers := []WorkerVerdict{
		{RunID: "RID-1", Issue: "4478", Lane: "gateway", State: WorkerHealthy},
		{RunID: "RID-2", Issue: "4479", Lane: "cmd", State: WorkerBlocked},
	}
	if !reflect.DeepEqual(in.Workers.Value, wantWorkers) {
		t.Errorf("projected workers = %+v, want %+v", in.Workers.Value, wantWorkers)
	}

	wantLeases := []Lease{{Lane: "gateway", Kind: "cluster", Tree: []string{"internal/gateway/**"}}}
	if !reflect.DeepEqual(in.Leases.Value, wantLeases) {
		t.Errorf("projected leases = %+v, want %+v", in.Leases.Value, wantLeases)
	}

	// Liveness and the escalation heads pass through unchanged (already payload-free).
	if !reflect.DeepEqual(in.Liveness.Value, src.Liveness.Value) {
		t.Errorf("liveness head changed under projection: got %+v, want %+v", in.Liveness.Value, src.Liveness.Value)
	}
	if !reflect.DeepEqual(in.Escalations.Value, src.Escalations.Value) {
		t.Errorf("escalation heads changed under projection: got %+v, want %+v", in.Escalations.Value, src.Escalations.Value)
	}
}

// TestAssemblePreservesAbsence proves the assembler never manufactures a witness: a
// source surface left Absent projects to an absent contract witness (driving
// AnyAbsent / AbsentWitnesses), while a present-but-empty source stays present. This
// is the fence-#1 invariant carried through the projection, not just the type.
func TestAssemblePreservesAbsence(t *testing.T) {
	src := Sources{
		Liveness:    Seen(Liveness{RunID: "RID-1"}),
		Workers:     Seen([]WorkerCensus{}),            // present, empty: census read, no workers running
		Escalations: Absent[[]Escalation](),            // the escalation queue could not be read
		Leases:      Absent[[]leaseref.ArbiterLease](), // the lease store could not be read
	}

	in := Assemble(src)

	want := []string{"escalations", "leases"}
	if got := in.AbsentWitnesses(); !reflect.DeepEqual(got, want) {
		t.Errorf("AbsentWitnesses() after assemble = %v, want %v", got, want)
	}
	if !in.Workers.Present {
		t.Error("a present, empty census must stay present after assemble, not collapse to absent")
	}
	if in.Workers.Value == nil {
		t.Error("a present, empty census must project to a non-nil empty slice")
	}
}
