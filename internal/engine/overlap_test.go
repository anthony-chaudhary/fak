package engine

import (
	"context"
	"errors"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestOverlapSchedule(t *testing.T) {
	t.Run("synchronous mode", func(t *testing.T) {
		ctx := context.Background()
		s := NewOverlapRunner[string](0)
		if d := s.Depth(); d != 0 {
			t.Fatalf("unexpected depth: %d", d)
		}

		var executionOrder []string
		task1 := InFlightTask[string]{
			ID: "sync-1",
			Execute: func(ctx context.Context) (string, error) {
				executionOrder = append(executionOrder, "sync-1")
				return "out-1", nil
			},
		}
		task2 := InFlightTask[string]{
			ID: "sync-2",
			Execute: func(ctx context.Context) (string, error) {
				executionOrder = append(executionOrder, "sync-2")
				return "out-2", nil
			},
		}

		res1, err := s.Submit(ctx, task1)
		if err != nil {
			t.Fatalf("submit task1 failed: %v", err)
		}
		if res1 == nil || res1.Value != "out-1" || !res1.Committed {
			t.Fatalf("unexpected task1 result: %+v", res1)
		}
		if s.InFlightCount() != 0 {
			t.Fatalf("expected 0 in-flight, got %d", s.InFlightCount())
		}

		res2, err := s.Submit(ctx, task2)
		if err != nil {
			t.Fatalf("submit task2 failed: %v", err)
		}
		if res2 == nil || res2.Value != "out-2" || !res2.Committed {
			t.Fatalf("unexpected task2 result: %+v", res2)
		}
		if s.InFlightCount() != 0 {
			t.Fatalf("expected 0 in-flight, got %d", s.InFlightCount())
		}

		if len(executionOrder) != 2 || executionOrder[0] != "sync-1" || executionOrder[1] != "sync-2" {
			t.Fatalf("unexpected sequential execution order: %v", executionOrder)
		}

		allResults := s.Results()
		if len(allResults) != 2 {
			t.Fatalf("expected 2 results, got %d", len(allResults))
		}
	})

	t.Run("overlap mode", func(t *testing.T) {
		ctx := context.Background()
		s := NewOverlapRunner[string](2)
		if d := s.Depth(); d != 2 {
			t.Fatalf("unexpected depth: %d", d)
		}

		const latency = 60 * time.Millisecond
		task1 := InFlightTask[string]{
			ID: "overlap-1",
			Execute: func(ctx context.Context) (string, error) {
				time.Sleep(latency)
				return "out-1", nil
			},
		}
		task2 := InFlightTask[string]{
			ID: "overlap-2",
			Execute: func(ctx context.Context) (string, error) {
				time.Sleep(latency)
				return "out-2", nil
			},
		}

		start := time.Now()
		_, err := s.Submit(ctx, task1)
		if err != nil {
			t.Fatalf("submit task1 failed: %v", err)
		}
		_, err = s.Submit(ctx, task2)
		if err != nil {
			t.Fatalf("submit task2 failed: %v", err)
		}

		activeInFlight := s.InFlightCount()
		if activeInFlight <= 1 {
			t.Fatalf("expected InFlightCount > 1 during overlap, got %d", activeInFlight)
		}

		drained, err := s.Drain(ctx)
		if err != nil {
			t.Fatalf("drain failed: %v", err)
		}
		elapsed := time.Since(start)

		if len(drained) != 2 {
			t.Fatalf("expected 2 drained results, got %d", len(drained))
		}
		vals := make(map[string]bool)
		for _, r := range drained {
			if !r.Committed {
				t.Fatalf("expected task %s committed", r.ID)
			}
			vals[r.Value] = true
		}
		if !vals["out-1"] || !vals["out-2"] {
			t.Fatalf("missing expected outputs in drained: %+v", drained)
		}

		sumLatencies := latency * 2
		if elapsed >= sumLatencies {
			t.Fatalf("expected elapsed time (%v) < sum of latencies (%v)", elapsed, sumLatencies)
		}
		if s.InFlightCount() != 0 {
			t.Fatalf("expected 0 in-flight after drain, got %d", s.InFlightCount())
		}
	})

	t.Run("dependency resolution", func(t *testing.T) {
		ctx := context.Background()
		s := NewOverlapRunner[string](2)

		var aCommitted atomic.Bool
		var bSawACommitted atomic.Bool

		taskA := InFlightTask[string]{
			ID: "A",
			Execute: func(ctx context.Context) (string, error) {
				time.Sleep(40 * time.Millisecond)
				aCommitted.Store(true)
				return "valA", nil
			},
		}

		taskB := InFlightTask[string]{
			ID:        "B",
			DependsOn: []string{"A"},
			Execute: func(ctx context.Context) (string, error) {
				if aCommitted.Load() {
					bSawACommitted.Store(true)
				}
				return "valB", nil
			},
		}

		_, err := s.Submit(ctx, taskA)
		if err != nil {
			t.Fatalf("submit taskA failed: %v", err)
		}

		// Submit taskB while taskA is actively running.
		_, err = s.Submit(ctx, taskB)
		if err != nil {
			t.Fatalf("submit taskB failed: %v", err)
		}

		drained, err := s.Drain(ctx)
		if err != nil {
			t.Fatalf("drain failed: %v", err)
		}
		if len(drained) != 2 {
			t.Fatalf("expected 2 drained results, got %d", len(drained))
		}
		if !bSawACommitted.Load() {
			t.Fatalf("taskB ran before taskA completed and committed")
		}
	})

	t.Run("drain flushes all tasks", func(t *testing.T) {
		ctx := context.Background()
		s := NewOverlapRunner[string](3)

		expectedErr := errors.New("simulated task error")
		tasks := []InFlightTask[string]{
			{
				ID: "flush-1",
				Execute: func(ctx context.Context) (string, error) {
					time.Sleep(20 * time.Millisecond)
					return "val-1", nil
				},
			},
			{
				ID: "flush-2",
				Execute: func(ctx context.Context) (string, error) {
					time.Sleep(20 * time.Millisecond)
					return "", expectedErr
				},
			},
			{
				ID: "flush-3",
				Execute: func(ctx context.Context) (string, error) {
					time.Sleep(20 * time.Millisecond)
					return "val-3", nil
				},
			},
		}

		for _, task := range tasks {
			_, err := s.Submit(ctx, task)
			if err != nil {
				t.Fatalf("submit failed: %v", err)
			}
		}

		drained, err := s.Drain(ctx)
		if err != nil {
			t.Fatalf("unexpected drain error: %v", err)
		}

		if len(drained) != 3 {
			t.Fatalf("expected 3 results from drain, got %d", len(drained))
		}
		if s.InFlightCount() != 0 {
			t.Fatalf("expected 0 in-flight after drain, got %d", s.InFlightCount())
		}

		byID := make(map[string]*OverlapResult[string])
		for _, r := range drained {
			byID[r.ID] = r
		}

		r1 := byID["flush-1"]
		if r1 == nil || r1.Value != "val-1" || r1.Err != nil || !r1.Committed {
			t.Fatalf("unexpected flush-1 outcome: %+v", r1)
		}
		r2 := byID["flush-2"]
		if r2 == nil || r2.Err == nil || r2.Err.Error() != expectedErr.Error() || !r2.Committed {
			t.Fatalf("unexpected flush-2 error propagation: %+v", r2)
		}
		r3 := byID["flush-3"]
		if r3 == nil || r3.Value != "val-3" || r3.Err != nil || !r3.Committed {
			t.Fatalf("unexpected flush-3 outcome: %+v", r3)
		}
	})

	t.Run("error handling", func(t *testing.T) {
		ctx := context.Background()
		s := NewOverlapRunner[int](0)

		expectedErr := errors.New("direct fail")
		res, err := s.Submit(ctx, InFlightTask[int]{
			ID: "err-task",
			Execute: func(ctx context.Context) (int, error) {
				return 0, expectedErr
			},
		})
		if res == nil {
			t.Fatalf("expected non-nil result")
		}
		if res.Err == nil || res.Err.Error() != expectedErr.Error() {
			t.Fatalf("expected res.Err to match, got %v", res.Err)
		}
		if !res.Committed {
			t.Fatalf("expected res.Committed to be true")
		}

		// Verify close behavior
		s.Close()
		_, err = s.Submit(ctx, InFlightTask[int]{
			ID: "after-close",
			Execute: func(ctx context.Context) (int, error) {
				return 1, nil
			},
		})
		if err == nil {
			t.Fatalf("expected error submitting after close")
		}
	})

	t.Run("buffer yield when full", func(t *testing.T) {
		ctx := context.Background()
		s := NewOverlapRunner[string](1)

		_, err := s.Submit(ctx, InFlightTask[string]{
			ID: "b-1",
			Execute: func(ctx context.Context) (string, error) {
				time.Sleep(10 * time.Millisecond)
				return "res-1", nil
			},
		})
		if err != nil {
			t.Fatalf("submit b-1 failed: %v", err)
		}

		// Wait for b-1 to finish and buffer its result.
		time.Sleep(30 * time.Millisecond)

		// Submitting b-2 when buffer has 1 item (>= depth 1) yields b-1's completed result.
		yielded, err := s.Submit(ctx, InFlightTask[string]{
			ID: "b-2",
			Execute: func(ctx context.Context) (string, error) {
				return "res-2", nil
			},
		})
		if err != nil {
			t.Fatalf("submit b-2 failed: %v", err)
		}
		if yielded == nil || yielded.ID != "b-1" || yielded.Value != "res-1" {
			t.Fatalf("expected b-1 yielded, got %+v", yielded)
		}

		drained, err := s.Drain(ctx)
		if err != nil {
			t.Fatalf("drain failed: %v", err)
		}
		if len(drained) != 1 || drained[0].ID != "b-2" {
			t.Fatalf("expected b-2 in drained, got %+v", drained)
		}
	})
}

func countDrainWaitGoroutines() int {
	buf := make([]byte, 256*1024)
	n := runtime.Stack(buf, true)
	stack := string(buf[:n])
	count := 0
	for _, line := range strings.Split(stack, "\n") {
		if strings.Contains(line, "Drain.func") {
			count++
		}
	}
	return count
}

func TestOverlapRunner_Drain_ContextCancellationLeaksNoGoroutines(t *testing.T) {
	t.Run("already-canceled context", func(t *testing.T) {
		s := NewOverlapRunner[string](2)
		defer s.Close()

		taskBlock := make(chan struct{})
		defer close(taskBlock)

		_, err := s.Submit(context.Background(), InFlightTask[string]{
			ID: "block-1",
			Execute: func(ctx context.Context) (string, error) {
				<-taskBlock
				return "ok", nil
			},
		})
		if err != nil {
			t.Fatalf("submit failed: %v", err)
		}

		for i := 0; i < 50; i++ {
			if s.InFlightCount() == 1 {
				break
			}
			time.Sleep(2 * time.Millisecond)
		}
		if s.InFlightCount() != 1 {
			t.Fatalf("expected 1 in-flight task, got %d", s.InFlightCount())
		}

		baseGoroutines := runtime.NumGoroutine()

		canceledCtx, cancel := context.WithCancel(context.Background())
		cancel()

		const iters = 10
		for i := 0; i < iters; i++ {
			_, err := s.Drain(canceledCtx)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("expected context.Canceled, got %v", err)
			}
		}

		time.Sleep(20 * time.Millisecond)
		if count := countDrainWaitGoroutines(); count > 0 {
			t.Fatalf("expected 0 drain wait goroutines, found %d", count)
		}
		if leaked := runtime.NumGoroutine() - baseGoroutines; leaked >= iters {
			t.Fatalf("goroutine leak: %d goroutines leaked across %d calls", leaked, iters)
		}
	})

	t.Run("timing-out context", func(t *testing.T) {
		s := NewOverlapRunner[string](2)
		defer s.Close()

		taskBlock := make(chan struct{})
		defer close(taskBlock)

		_, err := s.Submit(context.Background(), InFlightTask[string]{
			ID: "block-2",
			Execute: func(ctx context.Context) (string, error) {
				<-taskBlock
				return "ok", nil
			},
		})
		if err != nil {
			t.Fatalf("submit failed: %v", err)
		}

		for i := 0; i < 50; i++ {
			if s.InFlightCount() == 1 {
				break
			}
			time.Sleep(2 * time.Millisecond)
		}
		if s.InFlightCount() != 1 {
			t.Fatalf("expected 1 in-flight task, got %d", s.InFlightCount())
		}

		baseGoroutines := runtime.NumGoroutine()

		const iters = 5
		for i := 0; i < iters; i++ {
			timeoutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
			_, err := s.Drain(timeoutCtx)
			cancel()
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("expected context.DeadlineExceeded, got %v", err)
			}
		}

		time.Sleep(20 * time.Millisecond)
		if count := countDrainWaitGoroutines(); count > 0 {
			t.Fatalf("expected 0 drain wait goroutines, found %d", count)
		}
		if leaked := runtime.NumGoroutine() - baseGoroutines; leaked >= iters {
			t.Fatalf("goroutine leak: %d goroutines leaked across %d calls", leaked, iters)
		}
	})
}
