package agentsched

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

func BenchmarkPriorityQueue_EnqueueDequeue(b *testing.B) {
	pq := NewPriorityQueue(10000)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		p := abi.ThreadPriority(i % 4)
		task := &Task{
			ID:       fmt.Sprintf("task-%d", i),
			Priority: p,
		}
		if err := pq.Enqueue(task); err != nil {
			b.Fatalf("Enqueue failed: %v", err)
		}
		if _, ok := pq.Dequeue(true); !ok {
			b.Fatalf("Dequeue failed")
		}
	}
}

func BenchmarkPriorityQueue_Peek(b *testing.B) {
	pq := NewPriorityQueue(1024)
	for i := 0; i < 100; i++ {
		_ = pq.Enqueue(&Task{
			ID:       fmt.Sprintf("task-%d", i),
			Priority: abi.ThreadPriority(i % 4),
		})
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if _, ok := pq.Peek(true); !ok {
			b.Fatalf("Peek failed")
		}
	}
}

func BenchmarkGovernor_CheckAdmission(b *testing.B) {
	tax := laneadmit.Taxonomy{
		Loaded: true,
		Trees: map[string][]string{
			"bench-lane": {"internal/pkg/**"},
		},
	}
	gov := NewGovernor(GovernorConfig{
		BaseConcurrency: 1000,
		QueueCapacity:   2000,
		Taxonomy:        tax,
	})

	task := &Task{
		ID:           "bench-task",
		Priority:     abi.ThreadPriorityP1Interactive,
		Lane:         "bench-lane",
		AccountID:    "acc-bench",
		TokensNeeded: 10,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		verdict := gov.CheckAdmission(task)
		if !verdict.Admitted {
			b.Fatalf("CheckAdmission failed: %s", verdict.Reason)
		}
	}
}

func BenchmarkGovernor_TryAdmitAndRelease(b *testing.B) {
	tax := laneadmit.Taxonomy{
		Loaded: true,
		Trees: map[string][]string{
			"bench-lane": {"internal/bench/**"},
		},
	}
	gov := NewGovernor(GovernorConfig{
		BaseConcurrency: 16,
		QueueCapacity:   512,
		Taxonomy:        tax,
	})

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		task := &Task{
			ID:           fmt.Sprintf("task-%d", i),
			Priority:     abi.ThreadPriorityP1Interactive,
			Lane:         "bench-lane",
			AccountID:    "acc-bench",
			TokensNeeded: 5,
		}
		if err := gov.Submit(task); err != nil {
			b.Fatalf("Submit failed: %v", err)
		}

		admitted, verdict, err := gov.TryAdmit()
		if err != nil || !verdict.Admitted || admitted == nil {
			b.Fatalf("TryAdmit failed: err=%v, verdict=%+v", err, verdict)
		}

		gov.Release(admitted)
	}
}
