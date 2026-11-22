package modelengine

import (
	"errors"
	"testing"
)

func TestInferStatsPhaseSplit(t *testing.T) {
	reg := NewInferStatsRegistry()
	llm := reg.Model("llm")

	// UpdateSuccess splits compute at compute_input_end and compute_output_start.
	if err := llm.UpdateSuccess(2, InferStatsTimestamps{
		RequestStartNS: 100, QueueStartNS: 110, ComputeStartNS: 150,
		ComputeInputEndNS: 170, ComputeOutputStartNS: 270, ComputeEndNS: 280,
		RequestEndNS: 300,
	}); err != nil {
		t.Fatal(err)
	}
	if err := llm.UpdateSuccessWithDuration(1, 1000, 1000, 1005, 1100, 3, 60, 7); err != nil {
		t.Fatal(err)
	}
	// Cache miss: the request span grows by the insertion time.
	llm.UpdateSuccessCacheMiss(9)
	// Cache hit: queue ends at lookup; inference count is not bumped.
	if err := llm.UpdateSuccessCacheHit(2000, 2002, 2010, 2030, 15); err != nil {
		t.Fatal(err)
	}
	if err := llm.UpdateFailure(3000, 3040, InferStatsFailureCanceled); err != nil {
		t.Fatal(err)
	}
	if err := llm.UpdateFailure(4000, 4001, InferStatsFailureReason(42)); err != nil {
		t.Fatal(err)
	}
	if err := reg.Model("embed").UpdateFailure(10, 20, InferStatsFailureRejected); err != nil {
		t.Fatal(err)
	}

	snaps := reg.Snapshot()
	if len(snaps) != 2 || snaps[0].Model != "embed" || snaps[1].Model != "llm" {
		t.Fatalf("snapshot order = %+v", snaps)
	}
	got := snaps[1]
	want := InferStats{
		FailureCount:            2,
		FailureDurationNS:       41,
		SuccessCount:            3,
		RequestDurationNS:       200 + 100 + 9 + 30,
		QueueDurationNS:         40 + 5 + 8,
		ComputeInputDurationNS:  20 + 3,
		ComputeInferDurationNS:  100 + 60,
		ComputeOutputDurationNS: 10 + 7,
		CacheHitCount:           1,
		CacheHitDurationNS:      15,
		CacheMissCount:          1,
		CacheMissDurationNS:     9,
	}
	want.FailureByReason[InferStatsFailureCanceled] = 1
	want.FailureByReason[InferStatsFailureOther] = 1
	if got.Stats != want {
		t.Fatalf("llm stats\n got %+v\nwant %+v", got.Stats, want)
	}
	if got.InferenceCount != 3 {
		t.Fatalf("inference count = %d, want 3", got.InferenceCount)
	}
	if c := got.Stats.ComputeDurationNS(); c != 23+160+17 {
		t.Fatalf("compute = %d", c)
	}
	if e := snaps[0].Stats; e.FailureCount != 1 || e.FailureByReason[InferStatsFailureRejected] != 1 || e.SuccessCount != 0 {
		t.Fatalf("embed stats = %+v", e)
	}

	names := map[InferStatsFailureReason]string{
		InferStatsFailureRejected: "REJECTED", InferStatsFailureCanceled: "CANCELED",
		InferStatsFailureBackend: "BACKEND", InferStatsFailureOther: "OTHER", -1: "OTHER",
	}
	for r, n := range names {
		if r.String() != n {
			t.Fatalf("%d.String() = %q, want %q", int(r), r.String(), n)
		}
	}

	// Out-of-order stamps are refused and leave the aggregate untouched.
	err := llm.UpdateSuccess(1, InferStatsTimestamps{ComputeStartNS: 50, ComputeInputEndNS: 40})
	var oe *InferStatsOrderError
	if !errors.As(err, &oe) || oe.Earlier != "compute_start" {
		t.Fatalf("want order refusal, got %v", err)
	}
	if err := llm.UpdateFailure(9, 1, InferStatsFailureBackend); err == nil {
		t.Fatal("reversed failure span accepted")
	}
	if after := llm.Snapshot(); after.Stats != want || after.InferenceCount != 3 {
		t.Fatalf("refused samples mutated stats: %+v", after)
	}

	// Snapshots are copies.
	got.Stats.SuccessCount = 99
	if llm.Snapshot().Stats.SuccessCount != 3 {
		t.Fatal("snapshot aliases live state")
	}

	// Default-off: nil registry and recorder are no-ops.
	var none *InferStatsRegistry
	if m := none.Model("x"); m != nil {
		t.Fatal("nil registry returned a recorder")
	}
	var nilModel *ModelInferStats
	if err := nilModel.UpdateSuccess(1, InferStatsTimestamps{ComputeStartNS: 5}); err != nil {
		t.Fatal(err)
	}
	nilModel.UpdateSuccessCacheMiss(1)
	if none.Snapshot() != nil || nilModel.Snapshot() != (ModelInferStatsSnapshot{}) {
		t.Fatal("nil snapshot not empty")
	}
}
