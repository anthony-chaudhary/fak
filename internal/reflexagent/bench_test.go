package reflexagent

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentopt"
)

// Invariant: reflex profile executes atomic tasks under disjoint lane leases without multi-agent coordination.
// Invariant: sub-millisecond spawn overhead enables high-throughput leaf execution under tree-disjoint boundaries.

func TestReflexMicroAgentProfile_ContractEnforcement(t *testing.T) {
	arbiter := agentopt.NewConcurrencyClassArbiter(map[string]int{
		"leaf": 16,
	})
	profile := NewReflexMicroAgentProfile(arbiter)

	task := ReflexTask{
		ID:           "contract-test-1",
		Description:  "Validate contract execution invariant",
		LaneName:     "lane-contract",
		TreePatterns: []string{"internal/reflexagent/*"},
		ExecuteFn: func(ctx context.Context) (any, error) {
			return 42, nil
		},
	}

	res, err := profile.SpawnAndExecute(context.Background(), task)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if !res.Success {
		t.Fatalf("expected task success flag, got error: %s", res.Error)
	}
	if res.Output != 42 {
		t.Fatalf("expected output 42, got %v", res.Output)
	}
}

func BenchmarkReflexMicroAgentProfile_SpawnAndExecute(b *testing.B) {
	arbiter := agentopt.NewConcurrencyClassArbiter(map[string]int{
		"leaf": 64,
	})
	profile := NewReflexMicroAgentProfile(arbiter)
	ctx := context.Background()

	b.Run("NoOpTask", func(b *testing.B) {
		task := ReflexTask{
			ID:           "bench-noop",
			LaneName:     "lane-bench-noop",
			TreePatterns: []string{"bench/noop/*"},
			ExecuteFn: func(ctx context.Context) (any, error) {
				return nil, nil
			},
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			res, err := profile.SpawnAndExecute(ctx, task)
			if err != nil || !res.Success {
				b.Fatalf("task execution failed: %v", err)
			}
		}
	})

	b.Run("PayloadTask", func(b *testing.B) {
		task := ReflexTask{
			ID:           "bench-payload",
			LaneName:     "lane-bench-payload",
			TreePatterns: []string{"bench/payload/*"},
			ExecuteFn: func(ctx context.Context) (any, error) {
				return "admitted-payload-token", nil
			},
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			res, err := profile.SpawnAndExecute(ctx, task)
			if err != nil || !res.Success {
				b.Fatalf("task execution failed: %v", err)
			}
		}
	})
}

func BenchmarkReflexMicroAgentProfile_RunParallel(b *testing.B) {
	arbiter := agentopt.NewConcurrencyClassArbiter(map[string]int{
		"leaf": 128,
	})
	profile := NewReflexMicroAgentProfile(arbiter)
	ctx := context.Background()

	b.Run("4Lanes", func(b *testing.B) {
		tasks := make([]ReflexTask, 4)
		for j := 0; j < 4; j++ {
			tasks[j] = ReflexTask{
				ID:           fmt.Sprintf("bench-par-4-%d", j),
				LaneName:     fmt.Sprintf("lane-par-4-%d", j),
				TreePatterns: []string{fmt.Sprintf("bench/par4/%d/*", j)},
				ExecuteFn: func(ctx context.Context) (any, error) {
					return j, nil
				},
			}
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			results, err := profile.RunParallel(ctx, tasks)
			if err != nil {
				b.Fatalf("parallel execution failed: %v", err)
			}
			if len(results) != 4 {
				b.Fatalf("expected 4 results, got %d", len(results))
			}
		}
	})

	b.Run("8Lanes", func(b *testing.B) {
		tasks := make([]ReflexTask, 8)
		for j := 0; j < 8; j++ {
			tasks[j] = ReflexTask{
				ID:           fmt.Sprintf("bench-par-8-%d", j),
				LaneName:     fmt.Sprintf("lane-par-8-%d", j),
				TreePatterns: []string{fmt.Sprintf("bench/par8/%d/*", j)},
				ExecuteFn: func(ctx context.Context) (any, error) {
					return j, nil
				},
			}
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			results, err := profile.RunParallel(ctx, tasks)
			if err != nil {
				b.Fatalf("parallel execution failed: %v", err)
			}
			if len(results) != 8 {
				b.Fatalf("expected 8 results, got %d", len(results))
			}
		}
	})
}

func BenchmarkReflexMicroAgentProfile_CollisionContention(b *testing.B) {
	arbiter := agentopt.NewConcurrencyClassArbiter(map[string]int{
		"leaf": 16,
	})
	profile := NewReflexMicroAgentProfile(arbiter)
	ctx := context.Background()

	// Acquire a blocking lease on bench/shared/*
	blockTask := ReflexTask{
		ID:           "blocker",
		LaneName:     "lane-blocker",
		TreePatterns: []string{"bench/shared/*"},
	}
	workerID := "bench-blocker-worker"
	arbRes := arbiter.AcquireLease(agentopt.LaneLeaseRequest{
		LaneKind:     "leaf",
		LaneName:     blockTask.LaneName,
		TreePatterns: blockTask.TreePatterns,
		WorkerID:     workerID,
	})
	if !arbRes.Granted {
		b.Fatalf("failed to acquire initial lease: %s", arbRes.Reason)
	}
	defer arbiter.ReleaseLease(workerID, blockTask.LaneName)

	collidingTask := ReflexTask{
		ID:           "colliding",
		LaneName:     "lane-colliding",
		TreePatterns: []string{"bench/shared/sub/*"},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res, err := profile.SpawnAndExecute(ctx, collidingTask)
		if err == nil || res.Success {
			b.Fatalf("expected collision error, got success")
		}
	}
}
