package agenttopo

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestReflexParallelDispatchDisjointLanes(t *testing.T) {
	dispatcher := NewReflexDispatcher()

	tasks := []ReflexTask{
		{
			ID:             "task-gw-1",
			Lane:           "gateway",
			TreePatterns:   []string{"internal/gateway/**"},
			WitnessCommand: "go test ./internal/gateway -run TestGateway",
			Description:    "gateway leaf optimization",
			ExecuteFn: func(ctx context.Context) (int, string, error) {
				time.Sleep(15 * time.Millisecond)
				return 0, "PASS\nok internal/gateway 0.015s", nil
			},
		},
		{
			ID:             "task-eng-1",
			Lane:           "engine",
			TreePatterns:   []string{"internal/engine/**"},
			WitnessCommand: "go test ./internal/engine -run TestEngine",
			Description:    "engine leaf optimization",
			ExecuteFn: func(ctx context.Context) (int, string, error) {
				time.Sleep(15 * time.Millisecond)
				return 0, "PASS\nok internal/engine 0.015s", nil
			},
		},
		{
			ID:             "task-doc-1",
			Lane:           "docs",
			TreePatterns:   []string{"docs/**"},
			WitnessCommand: "go test ./docs -run TestDocs",
			Description:    "docs leaf optimization",
			ExecuteFn: func(ctx context.Context) (int, string, error) {
				time.Sleep(15 * time.Millisecond)
				return 0, "PASS\nok docs 0.015s", nil
			},
		},
	}

	start := time.Now()
	receipts, err := dispatcher.DispatchParallel(context.Background(), tasks)
	totalElapsed := time.Since(start)

	if err != nil {
		t.Fatalf("DispatchParallel returned unexpected error: %v", err)
	}
	if len(receipts) != 3 {
		t.Fatalf("expected 3 receipts, got %d", len(receipts))
	}

	for i, r := range receipts {
		if r.ExitCode != 0 {
			t.Errorf("task %s exit code = %d, want 0", r.TaskID, r.ExitCode)
		}
		if r.Status != "COMPLETED" {
			t.Errorf("task %s status = %q, want COMPLETED", r.TaskID, r.Status)
		}
		if r.Lane != tasks[i].Lane {
			t.Errorf("task %s lane = %q, want %q", r.TaskID, r.Lane, tasks[i].Lane)
		}
		if r.WitnessCommand != tasks[i].WitnessCommand {
			t.Errorf("task %s witness command = %q, want %q", r.TaskID, r.WitnessCommand, tasks[i].WitnessCommand)
		}
		if r.ElapsedTime <= 0 {
			t.Errorf("task %s elapsed time should be > 0, got %v", r.TaskID, r.ElapsedTime)
		}
		if r.ElapsedTime > 500*time.Millisecond {
			t.Errorf("task %s elapsed time %v exceeded sub-second expectation", r.TaskID, r.ElapsedTime)
		}
	}

	// Because tasks ran in parallel, 3 * 15ms should complete in well under 45ms + overhead.
	if totalElapsed > 250*time.Millisecond {
		t.Errorf("parallel dispatch was not fast: took %v", totalElapsed)
	}

	if active := dispatcher.ActiveLeases(); len(active) != 0 {
		t.Errorf("active leases should be empty after dispatch, got %d", len(active))
	}

	completed := dispatcher.CompletedTasks()
	if len(completed) != 3 {
		t.Errorf("completed tasks = %d, want 3", len(completed))
	}
}

func TestReflexTreeCollisionRefusal(t *testing.T) {
	dispatcher := NewReflexDispatcher()

	holdStarted := make(chan struct{})
	holdRelease := make(chan struct{})
	var wg sync.WaitGroup

	// Task 1: Holds lease on internal/gateway/**
	task1 := ReflexTask{
		ID:             "task-holder",
		Lane:           "gateway",
		TreePatterns:   []string{"internal/gateway/**"},
		WitnessCommand: "go test ./internal/gateway -run TestHold",
		Description:    "Holding lease on gateway tree",
		ExecuteFn: func(ctx context.Context) (int, string, error) {
			close(holdStarted)
			<-holdRelease
			return 0, "done holding", nil
		},
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		_, _ = dispatcher.Dispatch(context.Background(), task1)
	}()

	// Wait until task1 has acquired the lease and started execution
	<-holdStarted

	// Task 2: Attempts to acquire overlapping tree internal/gateway/sub/**
	task2 := ReflexTask{
		ID:             "task-colliding",
		Lane:           "gateway/sub",
		TreePatterns:   []string{"internal/gateway/sub/**"},
		WitnessCommand: "go test ./internal/gateway/sub -run TestSub",
		Description:    "Attempt overlapping leaf task",
		ExecuteFn: func(ctx context.Context) (int, string, error) {
			return 0, "should not execute", nil
		},
	}

	// 1. Evaluate should detect collision without mutating state
	evalErr := dispatcher.Evaluate(task2)
	if evalErr == nil {
		t.Fatalf("expected Evaluate to return collision error, got nil")
	}
	if !errors.Is(evalErr, ErrTreeCollision) {
		t.Errorf("Evaluate error = %v, want ErrTreeCollision", evalErr)
	}
	if !strings.Contains(evalErr.Error(), RefuseTreeCollision) {
		t.Errorf("Evaluate error %q missing %q", evalErr.Error(), RefuseTreeCollision)
	}

	// 2. Dispatch should refuse execution and return refusal receipt
	receipt, dispErr := dispatcher.Dispatch(context.Background(), task2)
	if dispErr == nil {
		t.Fatalf("expected Dispatch to return collision error, got nil")
	}
	if !errors.Is(dispErr, ErrTreeCollision) {
		t.Errorf("Dispatch error = %v, want ErrTreeCollision", dispErr)
	}
	if receipt.Status != "REFUSED" {
		t.Errorf("receipt status = %q, want REFUSED", receipt.Status)
	}
	if receipt.ExitCode == 0 {
		t.Errorf("refusal receipt exit code should be non-zero, got %d", receipt.ExitCode)
	}
	if !strings.Contains(receipt.Error, RefuseTreeCollision) {
		t.Errorf("receipt error %q does not contain %s", receipt.Error, RefuseTreeCollision)
	}

	// 3. Same-lane conflict check: another task attempting same lane "gateway"
	task3 := ReflexTask{
		ID:             "task-same-lane",
		Lane:           "gateway",
		WitnessCommand: "go test ./internal/gateway -run TestSameLane",
	}
	sameLaneErr := dispatcher.Evaluate(task3)
	if sameLaneErr == nil || !errors.Is(sameLaneErr, ErrTreeCollision) {
		t.Errorf("same lane evaluate error = %v, want ErrTreeCollision", sameLaneErr)
	}

	// Release Task 1
	close(holdRelease)
	wg.Wait()

	// Once Task 1 has released its lease, task2 can be evaluated and dispatched cleanly
	if err := dispatcher.Evaluate(task2); err != nil {
		t.Fatalf("after lease release, Evaluate should succeed, got: %v", err)
	}
}

func TestReflexReceiptAndZeroCoordinatorPollution(t *testing.T) {
	dispatcher := NewReflexDispatcher(ReflexWorkerProfile{
		Name:               "quarantine-profile",
		MaxDuration:        800 * time.Millisecond,
		MaxTouchedFiles:    3,
		QuarantineFailures: true,
	})

	// 50KB catastrophic crash diagnostic traceback
	hugePanicTraceback := fmt.Sprintf(
		"panic: runtime error: invalid memory address or nil pointer dereference\n"+
			"[signal SIGSEGV: segmentation violation code=0x2 addr=0x0 pc=0x104b2a3c]\n"+
			"goroutine 42 [running]:\n"+
			"%s\n"+
			"FAIL	github.com/anthony-chaudhary/fak/internal/engine [crash]\n",
		strings.Repeat("internal/engine.KernelFault(0x0, 0xdeadbeef)\n\t/fak/internal/engine/core.go:120 +0x4b\n", 400),
	)

	tasks := []ReflexTask{
		{
			ID:             "TASK-CLEAN-01",
			Lane:           "gateway",
			WitnessCommand: "go test ./internal/gateway -run TestAST",
			Description:    "Deliver valid AST node parsing",
			ExecuteFn: func(ctx context.Context) (int, string, error) {
				return 0, "PASS\nok internal/gateway 0.02s", nil
			},
		},
		{
			ID:             "TASK-CLEAN-02",
			Lane:           "engine",
			WitnessCommand: "go test ./internal/engine -run TestBad",
			Description:    "Attempted dangerous change causing compilation crash",
			ExecuteFn: func(ctx context.Context) (int, string, error) {
				return 2, hugePanicTraceback, errors.New("catastrophic crash: SIGSEGV")
			},
		},
		{
			ID:             "TASK-CLEAN-03",
			Lane:           "docs",
			WitnessCommand: "go test ./docs -run TestToken",
			Description:    "Deliver documentation update",
			ExecuteFn: func(ctx context.Context) (int, string, error) {
				return 0, "PASS\nok docs 0.01s", nil
			},
		},
	}

	// Dispatch all three tasks
	receipts, _ := dispatcher.DispatchParallel(context.Background(), tasks)
	if len(receipts) != 3 {
		t.Fatalf("receipts count = %d, want 3", len(receipts))
	}

	// 1. Verify receipt generation for all tasks
	r1 := receipts[0]
	if r1.TaskID != "TASK-CLEAN-01" || r1.ExitCode != 0 || r1.Status != "COMPLETED" {
		t.Errorf("r1 receipt unexpected: %+v", r1)
	}

	r2 := receipts[1]
	if r2.TaskID != "TASK-CLEAN-02" || r2.ExitCode != 2 || r2.Status != "FAILED" {
		t.Errorf("r2 receipt unexpected: %+v", r2)
	}

	r3 := receipts[2]
	if r3.TaskID != "TASK-CLEAN-03" || r3.ExitCode != 0 || r3.Status != "COMPLETED" {
		t.Errorf("r3 receipt unexpected: %+v", r3)
	}

	// 2. Zero coordinator context pollution verification:
	// Ensure no message in coordinator log contains raw panic / SIGSEGV trace
	coordMsgs := dispatcher.CoordinatorMessages()
	for idx, msg := range coordMsgs {
		if strings.Contains(msg, "panic: runtime error") {
			t.Fatalf("coordinator message [%d] polluted: contains raw panic text", idx)
		}
		if strings.Contains(msg, "SIGSEGV") {
			t.Fatalf("coordinator message [%d] polluted: contains SIGSEGV text", idx)
		}
		if strings.Contains(msg, "goroutine 42") {
			t.Fatalf("coordinator message [%d] polluted: contains goroutine stack trace", idx)
		}
		if strings.Contains(msg, "KernelFault") {
			t.Fatalf("coordinator message [%d] polluted: contains internal kernel fault dump", idx)
		}
	}

	// 3. Quarantine check: with QuarantineFailures=true, failed task contributes 0 messages to coordinator log
	if len(coordMsgs) != 2 {
		t.Fatalf("coordinator messages count = %d, want exactly 2 (TASK-CLEAN-01 and TASK-CLEAN-03)", len(coordMsgs))
	}

	// 4. Verify context reduction ratio: raw output > 15KB quarantined vs compact receipts
	if quarantined := dispatcher.RawBytesQuarantined(); quarantined < 10000 {
		t.Errorf("raw bytes quarantined = %d, want >= 10000", quarantined)
	}
	if ratio := dispatcher.ContextReductionRatio(); ratio < 0.90 {
		t.Errorf("context reduction ratio = %f, want >= 0.90", ratio)
	}
}

func TestReflexWorkerProfileAndS0S1Validation(t *testing.T) {
	profile := DefaultReflexWorkerProfile()
	if !profile.SubSecond() {
		t.Errorf("default profile must be sub-second, got duration %v", profile.MaxDuration)
	}

	dispatcher := NewReflexDispatcher(profile)

	// Test S0/S1 validation: > 3 files rejected
	taskTooManyFiles := ReflexTask{
		ID:             "task-invalid-files",
		Lane:           "gateway",
		TouchedFiles:   []string{"a.go", "b.go", "c.go", "d.go"},
		WitnessCommand: "go test ./...",
	}
	if err := dispatcher.Evaluate(taskTooManyFiles); !errors.Is(err, ErrInvalidS0S1) {
		t.Errorf("expected ErrInvalidS0S1 for >3 files, got %v", err)
	}

	// Test S0/S1 validation: chained witness command rejected
	taskChainedCmd := ReflexTask{
		ID:             "task-chained-cmd",
		Lane:           "gateway",
		TouchedFiles:   []string{"a.go"},
		WitnessCommand: "go test ./... && echo done",
	}
	if err := dispatcher.Evaluate(taskChainedCmd); !errors.Is(err, ErrInvalidS0S1) {
		t.Errorf("expected ErrInvalidS0S1 for chained command, got %v", err)
	}

	// Test S0/S1 validation: empty witness command rejected
	taskEmptyCmd := ReflexTask{
		ID:             "task-empty-cmd",
		Lane:           "gateway",
		TouchedFiles:   []string{"a.go"},
		WitnessCommand: "   ",
	}
	if err := dispatcher.Evaluate(taskEmptyCmd); !errors.Is(err, ErrInvalidS0S1) {
		t.Errorf("expected ErrInvalidS0S1 for empty command, got %v", err)
	}

	// Test timeout enforcement: task running past profile timeout
	shortProfile := NewReflexWorkerProfile("short", 20*time.Millisecond)
	shortDispatcher := NewReflexDispatcher(shortProfile)

	hangingTask := ReflexTask{
		ID:             "task-hanging",
		Lane:           "gateway",
		WitnessCommand: "go test ./...",
		ExecuteFn: func(ctx context.Context) (int, string, error) {
			select {
			case <-time.After(200 * time.Millisecond):
				return 0, "late", nil
			case <-ctx.Done():
				return 1, "cancelled", ctx.Err()
			}
		},
	}

	receipt, err := shortDispatcher.Dispatch(context.Background(), hangingTask)
	if err == nil || !errors.Is(err, ErrReflexTimeout) {
		t.Errorf("expected ErrReflexTimeout, got error: %v", err)
	}
	if receipt.ExitCode != 124 {
		t.Errorf("expected exit code 124 on timeout, got %d", receipt.ExitCode)
	}
}
