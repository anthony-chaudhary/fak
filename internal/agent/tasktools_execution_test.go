package agent

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

// Cancellation is a request to stop; capacity is released only after the
// admitted runner actually returns. Exercise both admission and promotion.
func TestTaskRunnerCancellationRetainsExecutionCapacity(t *testing.T) {
	started := make(chan string, 4)
	cancelObserved := make(chan struct{})
	releases := map[string]chan struct{}{}
	for _, name := range []string{"held", "peer", "next", "last"} {
		releases[name] = make(chan struct{})
	}
	defer func() {
		for _, ch := range releases {
			select {
			case <-ch:
			default:
				close(ch)
			}
		}
	}()
	st := NewTaskStateWithRunner(func(ctx context.Context, req ChildTaskRunRequest) (any, error) {
		started <- req.Prompt
		if req.Prompt == "held" {
			<-ctx.Done()
			close(cancelObserved)
		}
		<-releases[req.Prompt]
		return req.Prompt, nil
	})
	defer st.Close()
	st.SetLimits(2, 2)
	spawn := func(name string) TaskSpawnReceipt {
		r, err := st.Spawn(TaskSpawnRequest{Prompt: name})
		if err != nil {
			t.Fatalf("spawn %s: %v", name, err)
		}
		return r
	}
	waitStart := func() string {
		select {
		case name := <-started:
			return name
		case <-time.After(time.Second):
			t.Fatal("runner did not start")
			return ""
		}
	}
	assertPending := func(id string) {
		for _, item := range st.GetTasks() {
			if item.ID == id {
				if item.State != TaskStatePending {
					t.Fatalf("runner %s admitted while canceled execution still occupies capacity: %s", id, item.State)
				}
				return
			}
		}
		t.Fatalf("task %s missing", id)
	}
	held, peer := spawn("held"), spawn("peer")
	waitStart()
	waitStart()
	if _, err := st.Cancel(TaskCancelRequest{TaskID: held.TaskID}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-cancelObserved:
	case <-time.After(time.Second):
		t.Fatal("cancellation not observed")
	}
	next, last := spawn("next"), spawn("last")
	assertPending(next.TaskID)
	assertPending(last.TaskID)
	close(releases["peer"])
	if _, err := st.Wait(context.Background(), TaskWaitRequest{TaskID: peer.TaskID, TimeoutMs: 1000}); err != nil {
		t.Fatal(err)
	}
	if name := waitStart(); name != "next" {
		t.Fatalf("promoted %s, want next", name)
	}
	assertPending(last.TaskID)
	close(releases["held"])
	if name := waitStart(); name != "last" {
		t.Fatalf("promoted %s, want last", name)
	}
}

func TestTaskRunnerPublishesResultAndError(t *testing.T) {
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)

	requests := make(chan ChildTaskRunRequest, 2)
	runner := func(_ context.Context, req ChildTaskRunRequest) (any, error) {
		requests <- req
		if req.Prompt == "fail" {
			return nil, errors.New("child failed")
		}
		return "completed: " + req.Prompt, nil
	}
	if _, err := ArmTaskToolsWithRunner(2, 1, runner); err != nil {
		t.Fatalf("ArmTaskToolsWithRunner: %v", err)
	}
	st := GetActiveTaskState()

	ok, err := st.Spawn(TaskSpawnRequest{
		Prompt:       "inspect repository",
		Description:  "source audit",
		SubagentType: "researcher",
		ReadOnly:     true,
	})
	if err != nil {
		t.Fatalf("spawn successful child: %v", err)
	}
	failed, err := st.Spawn(TaskSpawnRequest{Prompt: "fail"})
	if err != nil {
		t.Fatalf("spawn failing child: %v", err)
	}

	wait, err := st.Wait(context.Background(), TaskWaitRequest{
		TaskIDs:   []string{ok.TaskID, failed.TaskID},
		TimeoutMs: 1_000,
	})
	if err != nil {
		t.Fatalf("wait: %v", err)
	}
	if wait.Status != "completed" || wait.Completed != 1 || wait.Failed != 1 {
		t.Fatalf("terminal counts = %+v, want one completed and one failed", wait)
	}
	if got := wait.Tasks[ok.TaskID]; got == nil || got.State != TaskStateCompleted || got.Result != "completed: inspect repository" {
		t.Fatalf("successful child = %+v", got)
	}
	if got := wait.Tasks[failed.TaskID]; got == nil || got.State != TaskStateFailed || got.Error != "child failed" {
		t.Fatalf("failed child = %+v", got)
	}

	seen := map[string]ChildTaskRunRequest{}
	for range 2 {
		req := <-requests
		seen[req.Prompt] = req
	}
	if req := seen["inspect repository"]; req.TaskID != ok.TaskID || req.Description != "source audit" || req.SubagentType != "researcher" || !req.ReadOnly {
		t.Fatalf("runner request did not preserve spawn contract: %+v", req)
	}
}

func TestTaskRunnerIdempotencyExecutesOnce(t *testing.T) {
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)

	started := make(chan struct{})
	release := make(chan struct{})
	var calls atomic.Int32
	runner := func(ctx context.Context, _ ChildTaskRunRequest) (any, error) {
		calls.Add(1)
		close(started)
		select {
		case <-release:
			return "one result", nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if _, err := ArmTaskToolsWithRunner(1, 1, runner); err != nil {
		t.Fatalf("ArmTaskToolsWithRunner: %v", err)
	}
	st := GetActiveTaskState()
	first, err := st.Spawn(TaskSpawnRequest{Prompt: "same work", IdempotencyKey: "stable-key"})
	if err != nil {
		t.Fatalf("first spawn: %v", err)
	}
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("child runner did not start")
	}

	replay, err := st.Spawn(TaskSpawnRequest{Prompt: "same work", IdempotencyKey: "stable-key"})
	if err != nil {
		t.Fatalf("idempotent replay: %v", err)
	}
	if !replay.Idempotent || replay.TaskID != first.TaskID {
		t.Fatalf("replay = %+v, want idempotent handle %q", replay, first.TaskID)
	}
	close(release)
	if _, err := st.Wait(context.Background(), TaskWaitRequest{TaskID: first.TaskID, TimeoutMs: 1_000}); err != nil {
		t.Fatalf("wait: %v", err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("runner calls = %d, want exactly 1", got)
	}
}

func TestTaskRunnerBacklogPromotesWithinLimits(t *testing.T) {
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)

	started := make(chan string, 2)
	releases := map[string]chan struct{}{
		"first":  make(chan struct{}),
		"second": make(chan struct{}),
	}
	runner := func(ctx context.Context, req ChildTaskRunRequest) (any, error) {
		started <- req.Prompt
		select {
		case <-releases[req.Prompt]:
			return req.Prompt + " done", nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if _, err := ArmTaskToolsWithRunner(1, 1, runner); err != nil {
		t.Fatalf("ArmTaskToolsWithRunner: %v", err)
	}
	st := GetActiveTaskState()
	first, err := st.Spawn(TaskSpawnRequest{Prompt: "first"})
	if err != nil {
		t.Fatalf("spawn first: %v", err)
	}
	if got := <-started; got != "first" {
		t.Fatalf("first runner = %q", got)
	}
	second, err := st.Spawn(TaskSpawnRequest{Prompt: "second"})
	if err != nil {
		t.Fatalf("spawn queued second: %v", err)
	}
	if _, err := st.Spawn(TaskSpawnRequest{Prompt: "over capacity"}); err == nil {
		t.Fatal("third spawn succeeded despite maxActive=1 and maxBacklog=1")
	}
	status, err := st.Status(TaskStatusRequest{})
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if status.Active != 1 || status.Pending != 1 {
		t.Fatalf("status = %+v, want active=1 pending=1", status)
	}
	select {
	case got := <-started:
		t.Fatalf("pending child started before capacity freed: %q", got)
	default:
	}

	close(releases["first"])
	if _, err := st.Wait(context.Background(), TaskWaitRequest{TaskID: first.TaskID, TimeoutMs: 1_000}); err != nil {
		t.Fatalf("wait first: %v", err)
	}
	select {
	case got := <-started:
		if got != "second" {
			t.Fatalf("promoted runner = %q, want second", got)
		}
	case <-time.After(time.Second):
		t.Fatal("pending child was not promoted")
	}
	close(releases["second"])
	wait, err := st.Wait(context.Background(), TaskWaitRequest{TaskID: second.TaskID, TimeoutMs: 1_000})
	if err != nil || wait.Completed != 1 {
		t.Fatalf("wait second = %+v, %v", wait, err)
	}
}

func TestTaskCancelStopsRunnerAndRemainsTerminal(t *testing.T) {
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)

	started := make(chan struct{})
	cancelled := make(chan struct{})
	returned := make(chan struct{})
	runner := func(ctx context.Context, _ ChildTaskRunRequest) (any, error) {
		close(started)
		<-ctx.Done()
		close(cancelled)
		defer close(returned)
		return nil, errors.New("runner observed cancellation")
	}
	if _, err := ArmTaskToolsWithRunner(1, 0, runner); err != nil {
		t.Fatalf("ArmTaskToolsWithRunner: %v", err)
	}
	st := GetActiveTaskState()
	spawn, err := st.Spawn(TaskSpawnRequest{Prompt: "long task"})
	if err != nil {
		t.Fatalf("spawn: %v", err)
	}
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("child runner did not start")
	}
	if receipt, err := st.Cancel(TaskCancelRequest{TaskID: spawn.TaskID, Reason: "stop"}); err != nil || !receipt.Cancelled {
		t.Fatalf("cancel = %+v, %v", receipt, err)
	}
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("cancel did not reach child context")
	}
	<-returned

	deadline := time.Now().Add(25 * time.Millisecond)
	for {
		status, err := st.Status(TaskStatusRequest{TaskID: spawn.TaskID})
		if err != nil {
			t.Fatalf("status: %v", err)
		}
		if len(status.Tasks) == 1 && status.Tasks[0].State != TaskStateCancelled {
			t.Fatalf("late runner return overwrote cancellation: %+v", status.Tasks[0])
		}
		if time.Now().After(deadline) {
			break
		}
		time.Sleep(time.Millisecond)
	}
	status, err := st.Status(TaskStatusRequest{TaskID: spawn.TaskID})
	if err != nil || len(status.Tasks) != 1 || status.Tasks[0].State != TaskStateCancelled || status.Tasks[0].Error != "stop" {
		t.Fatalf("terminal cancelled status = %+v, %v", status, err)
	}
	if wait, err := st.Wait(context.Background(), TaskWaitRequest{TaskID: spawn.TaskID, TimeoutMs: 100}); err != nil || wait.Cancelled != 1 {
		t.Fatalf("wait after cancel = %+v, %v", wait, err)
	}
}

func TestTaskRunnerWaitHasFiniteDefaultAndHonorsParentDeadline(t *testing.T) {
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)
	runner := func(ctx context.Context, _ ChildTaskRunRequest) (any, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	}
	if _, err := ArmTaskToolsWithRunner(1, 1, runner); err != nil {
		t.Fatalf("ArmTaskToolsWithRunner: %v", err)
	}
	if DefaultTaskWaitTimeout <= 0 || DefaultTaskWaitTimeout > MaxTaskWaitTimeout {
		t.Fatalf("default wait timeout = %v, max = %v", DefaultTaskWaitTimeout, MaxTaskWaitTimeout)
	}
	st := GetActiveTaskState()
	spawn, err := st.Spawn(TaskSpawnRequest{Prompt: "blocked child"})
	if err != nil {
		t.Fatalf("spawn: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	startedAt := time.Now()
	wait, err := st.Wait(ctx, TaskWaitRequest{TaskID: spawn.TaskID})
	if err != nil {
		t.Fatalf("wait: %v", err)
	}
	if wait.Status != "cancelled" || time.Since(startedAt) > time.Second {
		t.Fatalf("wait = %+v after %v", wait, time.Since(startedAt))
	}
}

func TestDisarmTaskToolsCancelsRunningChild(t *testing.T) {
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)
	started := make(chan struct{})
	cancelled := make(chan struct{})
	runner := func(ctx context.Context, _ ChildTaskRunRequest) (any, error) {
		close(started)
		<-ctx.Done()
		close(cancelled)
		return nil, ctx.Err()
	}
	if _, err := ArmTaskToolsWithRunner(1, 1, runner); err != nil {
		t.Fatalf("ArmTaskToolsWithRunner: %v", err)
	}
	st := GetActiveTaskState()
	spawn, err := st.Spawn(TaskSpawnRequest{Prompt: "running child"})
	if err != nil {
		t.Fatalf("spawn: %v", err)
	}
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("child runner did not start")
	}
	DisarmTaskTools()
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("disarm did not cancel child context")
	}
	status, err := st.Status(TaskStatusRequest{TaskID: spawn.TaskID})
	if err != nil || len(status.Tasks) != 1 || status.Tasks[0].State != TaskStateCancelled {
		t.Fatalf("status after disarm = %+v, %v", status, err)
	}
}
