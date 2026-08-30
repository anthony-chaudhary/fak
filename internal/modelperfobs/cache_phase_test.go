package modelperfobs

import (
	"testing"
	"time"
)

func TestCachePhaseLatencyReceiptBoundsLabelsAndReconcilesTotals(t *testing.T) {
	var recorder CachePhaseLatencyRecorder
	recorder.Observe(CachePipelinePhasePrefill, 2*time.Millisecond)
	recorder.Observe(CachePipelinePhaseDecode, 3*time.Millisecond)
	for i := 0; i < 100; i++ {
		recorder.Observe(CachePipelinePhase("adversarial-"+string(rune(i))), time.Millisecond)
	}

	receipt := recorder.Receipt()
	if got, want := len(receipt.Phases), 3; got != want {
		t.Fatalf("phase cardinality = %d, want %d", got, want)
	}
	wantPhases := []CachePipelinePhase{CachePipelinePhasePrefill, CachePipelinePhaseDecode, CachePipelinePhaseOther}
	var observations uint64
	var total time.Duration
	for i, bucket := range receipt.Phases {
		if bucket.Phase != wantPhases[i] {
			t.Fatalf("phase[%d] = %q, want %q", i, bucket.Phase, wantPhases[i])
		}
		observations += bucket.Observations
		total += bucket.Total
	}
	if got, want := receipt.Phases[2].Observations, uint64(100); got != want {
		t.Fatalf("other observations = %d, want %d", got, want)
	}
	if receipt.Observations != observations || receipt.Total != total {
		t.Fatalf("unlabeled totals (%d, %s) do not reconcile with phases (%d, %s)", receipt.Observations, receipt.Total, observations, total)
	}
	if got, want := receipt.Total, 105*time.Millisecond; got != want {
		t.Fatalf("total = %s, want %s", got, want)
	}
}

func TestCachePhaseLatencyRecorderClampsNegativeDuration(t *testing.T) {
	var recorder CachePhaseLatencyRecorder
	recorder.Observe(CachePipelinePhasePrefill, -time.Millisecond)
	receipt := recorder.Receipt()
	if receipt.Observations != 1 || receipt.Total != 0 {
		t.Fatalf("receipt = %+v, want one zero-duration observation", receipt)
	}
}

func BenchmarkCachePhaseLatencyRecorder(b *testing.B) {
	var recorder CachePhaseLatencyRecorder
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		recorder.Observe(CachePipelinePhaseDecode, time.Microsecond)
	}
}
