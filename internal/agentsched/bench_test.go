package agentsched

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// Invariant: agent scheduler enforces priority queue ordering and shedder admission fail-closed.

// TestBenchmarkSuiteSanity verifies that the priority queue and admission governor
// execute reliably within benchmark parameters.
func TestBenchmarkSuiteSanity(t *testing.T) {
	pq := NewPriorityQueue(128)
	if pq.Capacity() != 128 {
		t.Fatalf("expected capacity 128, got %d", pq.Capacity())
	}

	task := &Task{
		ID:       "sanity-0",
		Priority: abi.ThreadPriorityP1Interactive,
	}
	if err := pq.Enqueue(task); err != nil {
		t.Fatalf("unexpected enqueue error: %v", err)
	}

	dequeued, ok := pq.Dequeue(true)
	if !ok || dequeued.ID != "sanity-0" {
		t.Fatalf("expected task sanity-0, got %v", dequeued)
	}
}

// BenchmarkPriorityQueueEnqueueDequeue measures single-threaded FIFO enqueue and dequeue latency.
func BenchmarkPriorityQueueEnqueueDequeue(b *testing.B) {
	pq := NewPriorityQueue(512)
	task := &Task{
		ID:       "bench-task",
		Priority: abi.ThreadPriorityP1Interactive,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = pq.Enqueue(task)
		_, _ = pq.Dequeue(true)
	}
}

// BenchmarkPriorityQueueMultiTierEnqueueDequeue measures throughput across all 4 priority tiers.
func BenchmarkPriorityQueueMultiTierEnqueueDequeue(b *testing.B) {
	pq := NewPriorityQueue(512)
	tasks := []*Task{
		{ID: "p0", Priority: abi.ThreadPriorityP0System},
		{ID: "p1", Priority: abi.ThreadPriorityP1Interactive},
		{ID: "p2", Priority: abi.ThreadPriorityP2Batch},
		{ID: "p3", Priority: abi.ThreadPriorityP3Speculative},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, t := range tasks {
			_ = pq.Enqueue(t)
		}
		for range tasks {
			_, _ = pq.Dequeue(true)
		}
	}
}

// BenchmarkGovernorCheckAdmission evaluates the 4-gate admission verification pipeline.
func BenchmarkGovernorCheckAdmission(b *testing.B) {
	headroom := NewMemoryProviderHeadroom()
	headroom.SetBudget("bench-acc", 1000000)

	gov := NewGovernor(GovernorConfig{
		BaseConcurrency: 16,
		Headroom:        headroom,
		Taxonomy:        laneadmit.Taxonomy{},
	})

	task := &Task{
		ID:           "bench-task",
		Priority:     abi.ThreadPriorityP1Interactive,
		Lane:         "gateway",
		Tree:         []string{"internal/gateway/*"},
		AccountID:    "bench-acc",
		TokensNeeded: 100,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = gov.CheckAdmission(task)
	}
}

// BenchmarkGovernorTryAdmitAndRelease measures the complete admission, lease assignment, and release cycle.
func BenchmarkGovernorTryAdmitAndRelease(b *testing.B) {
	headroom := NewMemoryProviderHeadroom()
	headroom.SetBudget("bench-acc", 1000000)

	gov := NewGovernor(GovernorConfig{
		BaseConcurrency: 16,
		Headroom:        headroom,
	})

	task := &Task{
		ID:           "bench-task",
		Priority:     abi.ThreadPriorityP1Interactive,
		Lane:         "gateway",
		AccountID:    "bench-acc",
		TokensNeeded: 10,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = gov.Submit(task)
		admitted, _, _ := gov.TryAdmit()
		if admitted != nil {
			gov.Release(admitted)
		}
	}
}

// BenchmarkTelemetryUpdate evaluates host telemetry ingestion and dynamic thermal downscaling.
func BenchmarkTelemetryUpdate(b *testing.B) {
	gov := NewGovernor(GovernorConfig{
		BaseConcurrency: 16,
	})

	telemetries := []HostTelemetry{
		{CPUPct: 50.0, FreeRAMBytes: 8 * 1024 * 1024 * 1024, ObservedAt: time.Now()},
		{CPUPct: 78.0, FreeRAMBytes: 8 * 1024 * 1024 * 1024, ObservedAt: time.Now()},
		{CPUPct: 90.0, ThermalPressure: true, ObservedAt: time.Now()},
		{CPUPct: 40.0, FreeRAMBytes: 8 * 1024 * 1024 * 1024, ObservedAt: time.Now()},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		gov.UpdateTelemetry(telemetries[i%len(telemetries)])
	}
}
