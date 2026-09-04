package reflexagent

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentopt"
)

func BenchmarkSpawnAndExecute(b *testing.B) {
	arbiter := agentopt.NewConcurrencyClassArbiter(map[string]int{
		"leaf":    1024,
		"cluster": 64,
		"global":  1,
	})
	profile := NewReflexMicroAgentProfile(arbiter)
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		task := ReflexTask{
			ID:           fmt.Sprintf("bench-task-%d", i),
			LaneKind:     "leaf",
			LaneName:     fmt.Sprintf("bench-lane-%d", i),
			TreePatterns: []string{fmt.Sprintf("pkg/sub%d/*.go", i%64)},
			ExecuteFn: func(ctx context.Context) (any, error) {
				return 42, nil
			},
		}

		res, err := profile.SpawnAndExecute(ctx, task)
		if err != nil || !res.Success {
			b.Fatalf("unexpected failure: err=%v, res=%+v", err, res)
		}
	}
}

func BenchmarkRunParallel(b *testing.B) {
	arbiter := agentopt.NewConcurrencyClassArbiter(map[string]int{
		"leaf":    1024,
		"cluster": 64,
		"global":  1,
	})
	profile := NewReflexMicroAgentProfile(arbiter)
	ctx := context.Background()

	const batchSize = 8

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		tasks := make([]ReflexTask, batchSize)
		for j := 0; j < batchSize; j++ {
			tasks[j] = ReflexTask{
				ID:           fmt.Sprintf("bench-par-%d-%d", i, j),
				LaneKind:     "leaf",
				LaneName:     fmt.Sprintf("bench-lane-%d-%d", i, j),
				TreePatterns: []string{fmt.Sprintf("pkg/part%d/*.go", j)},
				ExecuteFn: func(ctx context.Context) (any, error) {
					return "done", nil
				},
			}
		}

		results, err := profile.RunParallel(ctx, tasks)
		if err != nil {
			b.Fatalf("RunParallel failed: %v", err)
		}
		if len(results) != batchSize {
			b.Fatalf("expected %d results, got %d", batchSize, len(results))
		}
	}
}

func BenchmarkLeaseAcquisitionAndRelease(b *testing.B) {
	arbiter := agentopt.NewConcurrencyClassArbiter(map[string]int{
		"leaf": 1024,
	})
	ctx := context.Background()
	profile := NewReflexMicroAgentProfile(arbiter)

	task := ReflexTask{
		ID:           "bench-lease-reuse",
		LaneKind:     "leaf",
		LaneName:     "static-lane",
		TreePatterns: []string{"static/*.go"},
		ExecuteFn: func(ctx context.Context) (any, error) {
			return nil, nil
		},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		res, err := profile.SpawnAndExecute(ctx, task)
		if err != nil || !res.Success {
			b.Fatalf("lease acquire/release cycle failed: %v", err)
		}
	}
}
