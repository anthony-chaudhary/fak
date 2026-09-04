package agentopt

import (
	"context"
	"sync"
	"testing"
)

func TestParallelToolCallBatching(t *testing.T) {
	ctx := context.Background()
	executor := NewBatchExecutor(4)

	t.Run("parallel speedup on independent reads", func(t *testing.T) {
		calls := []BatchCall{
			{ID: "read-1", Name: "Read", ReadPaths: []string{"a.go"}, ReadOnly: true},
			{ID: "read-2", Name: "Read", ReadPaths: []string{"b.go"}, ReadOnly: true},
			{ID: "read-3", Name: "Read", ReadPaths: []string{"c.go"}, ReadOnly: true},
		}

		var inFlight sync.WaitGroup
		inFlight.Add(len(calls))
		release := make(chan struct{})
		go func() {
			inFlight.Wait()
			close(release)
		}()

		results, err := executor.Execute(ctx, calls, func(ctx context.Context, call BatchCall) (string, error) {
			inFlight.Done()
			select {
			case <-release:
			case <-ctx.Done():
				return "", ctx.Err()
			}
			return "content:" + call.ID, nil
		})

		if err != nil {
			t.Fatalf("batch execution error: %v", err)
		}
		if len(results) != 3 {
			t.Fatalf("expected 3 results, got %d", len(results))
		}
		for _, r := range results {
			if r.Stage != 0 {
				t.Fatalf("expected all independent reads in stage 0, got stage %d", r.Stage)
			}
		}
	})

	t.Run("preserves ordering on write dependencies", func(t *testing.T) {
		calls := []BatchCall{
			{ID: "step-1-read", Name: "Read", ReadPaths: []string{"file.txt"}, ReadOnly: true},
			{ID: "step-2-write", Name: "Write", WritePaths: []string{"file.txt"}, ReadOnly: false},
			{ID: "step-3-read", Name: "Read", ReadPaths: []string{"file.txt"}, ReadOnly: true},
		}

		var (
			mu          sync.Mutex
			order       []string
			fileContent = "initial"
		)

		results, err := executor.Execute(ctx, calls, func(ctx context.Context, call BatchCall) (string, error) {
			mu.Lock()
			defer mu.Unlock()
			order = append(order, call.ID)
			if call.ID == "step-2-write" {
				fileContent = "updated"
			}
			return fileContent, nil
		})

		if err != nil {
			t.Fatalf("execution error: %v", err)
		}
		if len(order) != 3 || order[0] != "step-1-read" || order[1] != "step-2-write" || order[2] != "step-3-read" {
			t.Fatalf("execution order violated: %v", order)
		}
		if results[0].Output != "initial" || results[2].Output != "updated" {
			t.Fatalf("data dependency violated: res[0]=%s, res[2]=%s", results[0].Output, results[2].Output)
		}
		if results[0].Stage != 0 || results[1].Stage != 1 || results[2].Stage != 2 {
			t.Fatalf("stage partitioning incorrect: stages=[%d, %d, %d]", results[0].Stage, results[1].Stage, results[2].Stage)
		}
	})
}

func TestBatchPartitionStages(t *testing.T) {
	executor := NewBatchExecutor(2)

	calls := []BatchCall{
		{ID: "r1", ReadPaths: []string{"src/a.go"}, ReadOnly: true},
		{ID: "r2", ReadPaths: []string{"src/b.go"}, ReadOnly: true},
		{ID: "w1", WritePaths: []string{"src/a.go"}, ReadOnly: false},
		{ID: "r3", ReadPaths: []string{"src/b.go"}, ReadOnly: true},
	}

	stages := executor.PartitionStages(calls)
	// r1 and r2 are independent (Stage 0).
	// w1 writes src/a.go, so it depends on r1 (Stage 1).
	// r3 reads src/b.go (no conflict with w1), but depends on earlier ordering.
	if len(stages) < 2 {
		t.Fatalf("expected at least 2 stages, got %d", len(stages))
	}
}
