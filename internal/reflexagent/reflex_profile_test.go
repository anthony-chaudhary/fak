package reflexagent

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentopt"
)

func TestReflexMicroAgentProfile_SpawnFastPath(t *testing.T) {
	arbiter := agentopt.NewConcurrencyClassArbiter(map[string]int{
		"leaf": 10,
	})
	profile := NewReflexMicroAgentProfile(arbiter)

	task := ReflexTask{
		ID:           "task-1",
		Description:  "Fast read test",
		LaneName:     "lane-1",
		TreePatterns: []string{"internal/pkg1/*"},
		ExecuteFn: func(ctx context.Context) (any, error) {
			return "done", nil
		},
	}

	res, err := profile.SpawnAndExecute(context.Background(), task)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if !res.Success {
		t.Errorf("task failed: %v", res.Error)
	}
	if res.Output != "done" {
		t.Errorf("expected output done, got %v", res.Output)
	}
	// Verify fast spawn time (under 50ms)
	if res.SpawnTime > 50*time.Millisecond {
		t.Errorf("spawn time exceeded micro-agent threshold: %v", res.SpawnTime)
	}
}

func TestReflexMicroAgentProfile_ParallelDisjoint(t *testing.T) {
	arbiter := agentopt.NewConcurrencyClassArbiter(map[string]int{
		"leaf": 10,
	})
	profile := NewReflexMicroAgentProfile(arbiter)

	tasks := []ReflexTask{
		{
			ID:           "t1",
			LaneName:     "lane-pkg1",
			TreePatterns: []string{"internal/pkg1/*"},
			ExecuteFn: func(ctx context.Context) (any, error) {
				time.Sleep(10 * time.Millisecond)
				return 1, nil
			},
		},
		{
			ID:           "t2",
			LaneName:     "lane-pkg2",
			TreePatterns: []string{"internal/pkg2/*"},
			ExecuteFn: func(ctx context.Context) (any, error) {
				time.Sleep(10 * time.Millisecond)
				return 2, nil
			},
		},
		{
			ID:           "t3",
			LaneName:     "lane-pkg3",
			TreePatterns: []string{"internal/pkg3/*"},
			ExecuteFn: func(ctx context.Context) (any, error) {
				time.Sleep(10 * time.Millisecond)
				return 3, nil
			},
		},
	}

	results, err := profile.RunParallel(context.Background(), tasks)
	if err != nil {
		t.Fatalf("parallel execution error: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	for i, r := range results {
		if !r.Success {
			t.Errorf("task %d failed: %s", i, r.Error)
		}
	}
}

func TestReflexMicroAgentProfile_TreeCollisionRefusal(t *testing.T) {
	arbiter := agentopt.NewConcurrencyClassArbiter(map[string]int{
		"leaf": 10,
	})
	profile := NewReflexMicroAgentProfile(arbiter)

	// Hold lease in background
	blockStarted := make(chan struct{})
	blockDone := make(chan struct{})

	go func() {
		_, _ = profile.SpawnAndExecute(context.Background(), ReflexTask{
			ID:           "blocking-task",
			LaneName:     "lane-shared",
			TreePatterns: []string{"internal/shared/*"},
			ExecuteFn: func(ctx context.Context) (any, error) {
				close(blockStarted)
				<-blockDone
				return nil, nil
			},
		})
	}()

	<-blockStarted
	defer close(blockDone)

	// Attempt colliding task
	collidingTask := ReflexTask{
		ID:           "colliding-task",
		LaneName:     "lane-colliding",
		TreePatterns: []string{"internal/shared/sub/*"},
		ExecuteFn: func(ctx context.Context) (any, error) {
			return nil, nil
		},
	}

	res, err := profile.SpawnAndExecute(context.Background(), collidingTask)
	if err == nil {
		t.Fatalf("expected collision error, got nil")
	}
	if !strings.Contains(res.Error, "REFUSE_TREE_COLLISION") {
		t.Errorf("expected REFUSE_TREE_COLLISION, got %s", res.Error)
	}
}
