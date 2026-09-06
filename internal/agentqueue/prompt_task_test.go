package agentqueue

import (
	"errors"
	"fmt"
	"sync"
	"testing"
)

// Test that submitting a prompt task succeeds and returns state "queued".
func TestSubmitPromptTask_Queued(t *testing.T) {
	mgr := NewPromptTaskManager(10)

	spec := PromptTaskSpec{
		ParentSessionID: "sess-user-1234",
		IdempotencyKey:  "req-001",
		Prompt:          "Analyze recent error logs for subsystem crash",
	}

	handle, err := mgr.Submit(spec)
	if err != nil {
		t.Fatalf("unexpected error submitting prompt task: %v", err)
	}

	if handle.State != PromptTaskQueued {
		t.Errorf("expected state %q, got %q", PromptTaskQueued, handle.State)
	}
	if handle.TaskID == "" {
		t.Errorf("expected non-empty TaskID")
	}
	if handle.ParentSessionID != spec.ParentSessionID {
		t.Errorf("expected ParentSessionID %q, got %q", spec.ParentSessionID, handle.ParentSessionID)
	}
	if handle.IdempotencyKey != spec.IdempotencyKey {
		t.Errorf("expected IdempotencyKey %q, got %q", spec.IdempotencyKey, handle.IdempotencyKey)
	}
	if handle.Prompt != spec.Prompt {
		t.Errorf("expected Prompt %q, got %q", spec.Prompt, handle.Prompt)
	}
	if handle.CreatedAt.IsZero() {
		t.Errorf("expected non-zero CreatedAt")
	}
	if handle.Result != "" {
		t.Errorf("expected empty initial Result, got %q", handle.Result)
	}

	// Verify task can be retrieved by TaskID
	queried, err := mgr.Get(handle.TaskID)
	if err != nil {
		t.Fatalf("failed to query task: %v", err)
	}
	if queried.TaskID != handle.TaskID {
		t.Errorf("queried task ID mismatch: expected %q, got %q", handle.TaskID, queried.TaskID)
	}
	if queried.State != PromptTaskQueued {
		t.Errorf("queried state mismatch: expected %q, got %q", PromptTaskQueued, queried.State)
	}

	if mgr.Count() != 1 {
		t.Errorf("expected total count 1, got %d", mgr.Count())
	}
	if mgr.BacklogCount() != 1 {
		t.Errorf("expected backlog count 1, got %d", mgr.BacklogCount())
	}
}

// Test that submitting with the same idempotency key returns the existing task handle without spawning duplicates.
func TestSubmitPromptTask_CallerScopedDeduplication(t *testing.T) {
	mgr := NewPromptTaskManager(10)

	spec1 := PromptTaskSpec{
		ParentSessionID: "sess-caller-a",
		IdempotencyKey:  "idem-key-100",
		Prompt:          "First prompt submission",
	}

	handle1, err := mgr.Submit(spec1)
	if err != nil {
		t.Fatalf("first submit failed: %v", err)
	}

	// Submit again with the EXACT same session and idempotency key, but different prompt text
	spec2 := PromptTaskSpec{
		ParentSessionID: "sess-caller-a",
		IdempotencyKey:  "idem-key-100",
		Prompt:          "Different prompt with same key",
	}

	handle2, err := mgr.Submit(spec2)
	if err != nil {
		t.Fatalf("second submit failed: %v", err)
	}

	// Must return existing task handle without creating a duplicate
	if handle2.TaskID != handle1.TaskID {
		t.Errorf("deduplication failed: expected TaskID %q, got %q", handle1.TaskID, handle2.TaskID)
	}
	if handle2.Prompt != "First prompt submission" {
		t.Errorf("expected original prompt preserved, got %q", handle2.Prompt)
	}
	if !handle2.CreatedAt.Equal(handle1.CreatedAt) {
		t.Errorf("expected original CreatedAt preserved")
	}
	if mgr.Count() != 1 {
		t.Errorf("expected 1 task in manager, found %d", mgr.Count())
	}

	// Submit with DIFFERENT ParentSessionID but SAME IdempotencyKey (caller-scoped deduplication)
	specCallerB := PromptTaskSpec{
		ParentSessionID: "sess-caller-b",
		IdempotencyKey:  "idem-key-100",
		Prompt:          "Caller B prompt with same key",
	}

	handleB, err := mgr.Submit(specCallerB)
	if err != nil {
		t.Fatalf("caller B submit failed: %v", err)
	}
	if handleB.TaskID == handle1.TaskID {
		t.Errorf("caller-scoped deduplication violated: caller B should get distinct TaskID")
	}
	if mgr.Count() != 2 {
		t.Errorf("expected 2 tasks in manager after distinct caller submit, found %d", mgr.Count())
	}

	// Global / unparented idempotency test
	specGlobal := PromptTaskSpec{
		IdempotencyKey: "global-idem-200",
		Prompt:         "Global task",
	}
	handleG1, err := mgr.Submit(specGlobal)
	if err != nil {
		t.Fatalf("global submit 1 failed: %v", err)
	}
	handleG2, err := mgr.Submit(specGlobal)
	if err != nil {
		t.Fatalf("global submit 2 failed: %v", err)
	}
	if handleG1.TaskID != handleG2.TaskID {
		t.Errorf("global deduplication failed: expected %q, got %q", handleG1.TaskID, handleG2.TaskID)
	}
}

// Test that queue backlog capacity limits are enforced (rejecting when capacity is full).
func TestSubmitPromptTask_QueueCapacity_Reject(t *testing.T) {
	capacity := 2
	mgr := NewPromptTaskManager(capacity)

	// Fill queue up to capacity
	task1, err := mgr.Submit(PromptTaskSpec{
		ParentSessionID: "sess-1",
		IdempotencyKey:  "k1",
		Prompt:          "Task 1",
	})
	if err != nil {
		t.Fatalf("task 1 submit failed: %v", err)
	}
	if task1.State != PromptTaskQueued {
		t.Errorf("expected task 1 state %q, got %q", PromptTaskQueued, task1.State)
	}

	task2, err := mgr.Submit(PromptTaskSpec{
		ParentSessionID: "sess-1",
		IdempotencyKey:  "k2",
		Prompt:          "Task 2",
	})
	if err != nil {
		t.Fatalf("task 2 submit failed: %v", err)
	}
	if task2.State != PromptTaskQueued {
		t.Errorf("expected task 2 state %q, got %q", PromptTaskQueued, task2.State)
	}

	if mgr.BacklogCount() != capacity {
		t.Fatalf("expected backlog count %d, got %d", capacity, mgr.BacklogCount())
	}

	// 3rd task exceeds capacity -> must be rejected with ErrQueueCapacityExceeded
	_, err = mgr.Submit(PromptTaskSpec{
		ParentSessionID: "sess-1",
		IdempotencyKey:  "k3",
		Prompt:          "Task 3 exceeding capacity",
	})
	if !errors.Is(err, ErrQueueCapacityExceeded) {
		t.Fatalf("expected ErrQueueCapacityExceeded, got: %v", err)
	}

	// Resubmitting task 1 with its idempotency key must succeed and return existing handle
	dedupTask1, err := mgr.Submit(PromptTaskSpec{
		ParentSessionID: "sess-1",
		IdempotencyKey:  "k1",
		Prompt:          "Task 1 duplicate",
	})
	if err != nil {
		t.Fatalf("resubmitting existing task should not be blocked by capacity: %v", err)
	}
	if dedupTask1.TaskID != task1.TaskID {
		t.Errorf("expected existing task ID %q, got %q", task1.TaskID, dedupTask1.TaskID)
	}
}

// Test that queue backlog capacity limits are enforced (holding when capacity is full).
func TestSubmitPromptTask_QueueCapacity_Hold(t *testing.T) {
	capacity := 2
	mgr := NewPromptTaskManagerWithHold(capacity)

	// Fill queue to capacity
	_, err := mgr.Submit(PromptTaskSpec{ParentSessionID: "sess-1", Prompt: "Task 1"})
	if err != nil {
		t.Fatalf("task 1 submit failed: %v", err)
	}
	_, err = mgr.Submit(PromptTaskSpec{ParentSessionID: "sess-1", Prompt: "Task 2"})
	if err != nil {
		t.Fatalf("task 2 submit failed: %v", err)
	}

	// 3rd task exceeds capacity -> held instead of rejected
	task3, err := mgr.Submit(PromptTaskSpec{ParentSessionID: "sess-1", Prompt: "Task 3"})
	if err != nil {
		t.Fatalf("unexpected error on hold-on-full: %v", err)
	}
	if task3.State != PromptTaskHeld {
		t.Fatalf("expected task 3 state %q, got %q", PromptTaskHeld, task3.State)
	}

	// Verify SubmitOrHold directly on standard manager
	standardMgr := NewPromptTaskManager(1)
	_, err = standardMgr.Submit(PromptTaskSpec{ParentSessionID: "s", Prompt: "Task A"})
	if err != nil {
		t.Fatalf("task A submit failed: %v", err)
	}
	heldTask, err := standardMgr.SubmitOrHold(PromptTaskSpec{ParentSessionID: "s", Prompt: "Task B"})
	if err != nil {
		t.Fatalf("SubmitOrHold failed: %v", err)
	}
	if heldTask.State != PromptTaskHeld {
		t.Fatalf("expected held state from SubmitOrHold, got %q", heldTask.State)
	}

	// Test ReleaseHeld: when capacity opens up, held tasks transition to queued
	heldList := mgr.ListByState(PromptTaskHeld)
	if len(heldList) != 1 {
		t.Fatalf("expected 1 held task, found %d", len(heldList))
	}

	// Complete first task to free up backlog space
	queuedTasks := mgr.ListByState(PromptTaskQueued)
	if len(queuedTasks) == 0 {
		t.Fatalf("no queued tasks found")
	}
	if err := mgr.Complete(queuedTasks[0].TaskID, "done"); err != nil {
		t.Fatalf("failed to complete task: %v", err)
	}

	promoted := mgr.ReleaseHeld()
	if promoted != 1 {
		t.Fatalf("expected 1 task promoted from held to queued, got %d", promoted)
	}

	promotedTask, err := mgr.Get(task3.TaskID)
	if err != nil {
		t.Fatalf("failed to get promoted task: %v", err)
	}
	if promotedTask.State != PromptTaskQueued {
		t.Errorf("expected promoted task state %q, got %q", PromptTaskQueued, promotedTask.State)
	}
}

// Test task lifecycle state transitions (queued -> running -> completed).
func TestSubmitPromptTask_LifecycleTransitions(t *testing.T) {
	mgr := NewPromptTaskManager(5)

	handle, err := mgr.Submit(PromptTaskSpec{
		ParentSessionID: "sess-lifecycle",
		IdempotencyKey:  "idem-life-1",
		Prompt:          "Run lifecycle validation",
	})
	if err != nil {
		t.Fatalf("submit failed: %v", err)
	}
	if handle.State != PromptTaskQueued {
		t.Fatalf("expected initial state %q, got %q", PromptTaskQueued, handle.State)
	}

	// 1. Transition: queued -> running
	if err := mgr.Start(handle.TaskID); err != nil {
		t.Fatalf("failed transition to running: %v", err)
	}

	status, err := mgr.GetStatus(handle.TaskID)
	if err != nil {
		t.Fatalf("failed to get status: %v", err)
	}
	if status != PromptTaskRunning {
		t.Errorf("expected state %q, got %q", PromptTaskRunning, status)
	}

	// 2. Transition: running -> completed
	resultData := `{"summary": "processed 42 records", "status": "ok"}`
	if err := mgr.Complete(handle.TaskID, resultData); err != nil {
		t.Fatalf("failed transition to completed: %v", err)
	}

	completedTask, err := mgr.Get(handle.TaskID)
	if err != nil {
		t.Fatalf("failed to get completed task: %v", err)
	}
	if completedTask.State != PromptTaskCompleted {
		t.Errorf("expected state %q, got %q", PromptTaskCompleted, completedTask.State)
	}
	if completedTask.Result != resultData {
		t.Errorf("expected result %q, got %q", resultData, completedTask.Result)
	}
	if !completedTask.IsTerminal() {
		t.Errorf("expected task to be terminal")
	}

	// 3. Verify terminal state protection: cannot transition out of completed
	err = mgr.UpdateState(handle.TaskID, PromptTaskRunning)
	if !errors.Is(err, ErrInvalidTransition) {
		t.Errorf("expected ErrInvalidTransition when moving from completed to running, got: %v", err)
	}
}

// Test failure lifecycle transition: queued -> running -> failed.
func TestSubmitPromptTask_FailureLifecycle(t *testing.T) {
	mgr := NewPromptTaskManager(5)

	handle, err := mgr.Submit(PromptTaskSpec{
		ParentSessionID: "sess-fail",
		Prompt:          "Task that will fail",
	})
	if err != nil {
		t.Fatalf("submit failed: %v", err)
	}

	if err := mgr.Start(handle.TaskID); err != nil {
		t.Fatalf("start failed: %v", err)
	}

	failMessage := "execution deadline exceeded after 30s"
	if err := mgr.Fail(handle.TaskID, failMessage); err != nil {
		t.Fatalf("fail transition failed: %v", err)
	}

	failedTask, err := mgr.Get(handle.TaskID)
	if err != nil {
		t.Fatalf("failed to get task: %v", err)
	}
	if failedTask.State != PromptTaskFailed {
		t.Errorf("expected state %q, got %q", PromptTaskFailed, failedTask.State)
	}
	if failedTask.Result != failMessage {
		t.Errorf("expected failure message %q, got %q", failMessage, failedTask.Result)
	}
	if !failedTask.IsTerminal() {
		t.Errorf("expected failed task to be terminal")
	}

	// Verify terminal state protection
	err = mgr.UpdateState(handle.TaskID, PromptTaskQueued)
	if !errors.Is(err, ErrInvalidTransition) {
		t.Errorf("expected ErrInvalidTransition when moving from failed to queued, got: %v", err)
	}
}

// Test querying methods and Store integration.
func TestSubmitPromptTask_QueryAndStore(t *testing.T) {
	store := FileStore("test.json")
	mgr := store.PromptTaskManager(10)

	spec := PromptTaskSpec{
		ParentSessionID: "sess-query",
		IdempotencyKey:  "key-query",
		Prompt:          "Query test prompt",
	}

	handle, err := mgr.Submit(spec)
	if err != nil {
		t.Fatalf("submit failed: %v", err)
	}

	// Lookup by idempotency key
	byKey, ok := mgr.GetByIdempotencyKey("sess-query", "key-query")
	if !ok {
		t.Fatalf("expected to find task by idempotency key")
	}
	if byKey.TaskID != handle.TaskID {
		t.Errorf("expected task ID %q, got %q", handle.TaskID, byKey.TaskID)
	}

	// Lookup by non-existent task ID
	_, err = mgr.Get("non-existent-task-id")
	if !errors.Is(err, ErrTaskNotFound) {
		t.Errorf("expected ErrTaskNotFound, got: %v", err)
	}

	// ListByParentSession
	sessionTasks := mgr.ListByParentSession("sess-query")
	if len(sessionTasks) != 1 {
		t.Errorf("expected 1 task for session, got %d", len(sessionTasks))
	}

	// SetResult without state change
	if err := mgr.SetResult(handle.TaskID, "interim progress"); err != nil {
		t.Fatalf("SetResult failed: %v", err)
	}
	res, err := mgr.GetResult(handle.TaskID)
	if err != nil {
		t.Fatalf("GetResult failed: %v", err)
	}
	if res != "interim progress" {
		t.Errorf("expected result %q, got %q", "interim progress", res)
	}
}

// Test concurrent submissions for race safety.
func TestSubmitPromptTask_ConcurrentSubmissions(t *testing.T) {
	mgr := NewPromptTaskManager(100)
	var wg sync.WaitGroup
	workers := 10
	submitsPerWorker := 10

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < submitsPerWorker; i++ {
				_, _ = mgr.Submit(PromptTaskSpec{
					ParentSessionID: fmt.Sprintf("session-%d", workerID),
					IdempotencyKey:  fmt.Sprintf("key-%d", i),
					Prompt:          fmt.Sprintf("Worker %d Task %d", workerID, i),
				})
			}
		}(w)
	}

	wg.Wait()
	if mgr.Count() != workers*submitsPerWorker {
		t.Errorf("expected %d tasks, got %d", workers*submitsPerWorker, mgr.Count())
	}
}
