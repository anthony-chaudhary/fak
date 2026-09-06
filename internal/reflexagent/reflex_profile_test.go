package reflexagent

import (
	"context"
	"errors"
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

func TestReflexMicroAgentProfile_PanicRecovery(t *testing.T) {
	arbiter := agentopt.NewConcurrencyClassArbiter(map[string]int{
		"leaf": 10,
	})
	profile := NewReflexMicroAgentProfile(arbiter)

	// 1. Single SpawnAndExecute panic isolation
	task := ReflexTask{
		ID:           "panic-task-1",
		LaneName:     "lane-panic",
		TreePatterns: []string{"internal/panic1/*"},
		ExecuteFn: func(ctx context.Context) (any, error) {
			panic("simulated fatal worker crash")
		},
	}

	res, err := profile.SpawnAndExecute(context.Background(), task)
	if err == nil {
		t.Fatalf("expected error from panicking task, got nil")
	}
	if res == nil {
		t.Fatalf("expected non-nil TaskResult")
	}
	if res.Success {
		t.Errorf("expected res.Success to be false on panic")
	}
	if !strings.Contains(res.Error, "simulated fatal worker crash") {
		t.Errorf("expected res.Error to contain panic details, got: %q", res.Error)
	}

	// Verify lease was cleanly released in arbiter
	if active := arbiter.ActiveCount(""); active != 0 {
		t.Errorf("expected 0 active leases after panic recovery, got %d", active)
	}

	// Verify subsequent task can reacquire the same lane immediately
	followUp := ReflexTask{
		ID:           "followup-task",
		LaneName:     "lane-panic",
		TreePatterns: []string{"internal/panic1/*"},
		ExecuteFn: func(ctx context.Context) (any, error) {
			return "recovered", nil
		},
	}
	res2, err2 := profile.SpawnAndExecute(context.Background(), followUp)
	if err2 != nil || !res2.Success {
		t.Fatalf("expected follow-up task on same lane to succeed, got err=%v, res=%+v", err2, res2)
	}

	// 2. Parallel RunParallel panic isolation
	tasks := []ReflexTask{
		{
			ID:           "par-ok-1",
			LaneName:     "lane-par-1",
			TreePatterns: []string{"internal/par1/*"},
			ExecuteFn: func(ctx context.Context) (any, error) {
				return "ok1", nil
			},
		},
		{
			ID:           "par-panic-2",
			LaneName:     "lane-par-2",
			TreePatterns: []string{"internal/par2/*"},
			ExecuteFn: func(ctx context.Context) (any, error) {
				panic("parallel worker panic")
			},
		},
		{
			ID:           "par-ok-3",
			LaneName:     "lane-par-3",
			TreePatterns: []string{"internal/par3/*"},
			ExecuteFn: func(ctx context.Context) (any, error) {
				return "ok3", nil
			},
		},
	}

	results, parErr := profile.RunParallel(context.Background(), tasks)
	if parErr == nil {
		t.Errorf("expected RunParallel to return error when a task panics")
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	if results[0] == nil || !results[0].Success {
		t.Errorf("expected task 0 to succeed: %+v", results[0])
	}
	if results[1] == nil || results[1].Success {
		t.Errorf("expected task 1 to fail on panic")
	} else if !strings.Contains(results[1].Error, "parallel worker panic") {
		t.Errorf("expected task 1 error to contain panic details, got: %q", results[1].Error)
	}
	if results[2] == nil || !results[2].Success {
		t.Errorf("expected task 2 to succeed: %+v", results[2])
	}

	// Verify all leases are released
	if active := arbiter.ActiveCount(""); active != 0 {
		t.Errorf("expected 0 active leases after parallel execution, got %d", active)
	}
}

func TestReflexMicroAgentProfile_CancelledContext(t *testing.T) {
	arbiter := agentopt.NewConcurrencyClassArbiter(map[string]int{
		"leaf": 0, // Budget 0: if lease acquisition is attempted, it would be refused by budget
	})
	profile := NewReflexMicroAgentProfile(arbiter)

	// 1. Pre-cancelled context with SpawnAndExecute
	ctxCancel, cancel := context.WithCancel(context.Background())
	cancel()

	executed := false
	task := ReflexTask{
		ID:           "cancelled-task",
		LaneName:     "lane-cancelled",
		TreePatterns: []string{"internal/cancel/*"},
		ExecuteFn: func(ctx context.Context) (any, error) {
			executed = true
			return "should-not-run", nil
		},
	}

	res, err := profile.SpawnAndExecute(ctxCancel, task)
	if err == nil {
		t.Fatalf("expected error for cancelled context, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled error, got %v", err)
	}
	if executed {
		t.Errorf("task ExecuteFn should not have been executed")
	}
	if res == nil {
		t.Fatalf("expected non-nil TaskResult")
	}
	if res.Success {
		t.Errorf("expected res.Success to be false")
	}
	if arbiter.ActiveCount("") != 0 {
		t.Errorf("expected 0 active leases, got %d", arbiter.ActiveCount(""))
	}

	// 2. Pre-expired context (DeadlineExceeded)
	ctxDeadline, deadlineCancel := context.WithDeadline(context.Background(), time.Now().Add(-1*time.Minute))
	defer deadlineCancel()

	res2, err2 := profile.SpawnAndExecute(ctxDeadline, task)
	if err2 == nil {
		t.Fatalf("expected error for deadline exceeded context, got nil")
	}
	if !errors.Is(err2, context.DeadlineExceeded) {
		t.Errorf("expected context.DeadlineExceeded error, got %v", err2)
	}
	if res2.Success {
		t.Errorf("expected res2.Success to be false")
	}
	if arbiter.ActiveCount("") != 0 {
		t.Errorf("expected 0 active leases, got %d", arbiter.ActiveCount(""))
	}

	// 3. Parallel RunParallel with cancelled context
	tasks := []ReflexTask{task, task}
	results, parErr := profile.RunParallel(ctxCancel, tasks)
	if parErr == nil {
		t.Errorf("expected error from RunParallel with cancelled context")
	}
	if !errors.Is(parErr, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", parErr)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	for i, r := range results {
		if r == nil || r.Success {
			t.Errorf("task %d should have failed", i)
		}
	}
	if arbiter.ActiveCount("") != 0 {
		t.Errorf("expected 0 active leases, got %d", arbiter.ActiveCount(""))
	}
}

func TestReflexMicroAgentProfile_NilArbiter(t *testing.T) {
	var nilProfile *ReflexMicroAgentProfile
	task := ReflexTask{ID: "t1"}

	if _, err := nilProfile.SpawnAndExecute(context.Background(), task); err == nil {
		t.Fatal("expected error on nil profile, got nil")
	}
	if _, err := nilProfile.RunParallel(context.Background(), []ReflexTask{task}); err == nil {
		t.Fatal("expected error on nil profile RunParallel, got nil")
	}

	profileWithoutArbiter := &ReflexMicroAgentProfile{}
	if _, err := profileWithoutArbiter.SpawnAndExecute(context.Background(), task); err == nil {
		t.Fatal("expected error on profile with nil arbiter, got nil")
	}
	if _, err := profileWithoutArbiter.RunParallel(context.Background(), []ReflexTask{task}); err == nil {
		t.Fatal("expected error on profile with nil arbiter RunParallel, got nil")
	}
}
