package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"
)

type scopedTaskWitnessPlanner struct {
	name               string
	step               int
	statusSeen         chan<- string
	returnAfterSpawn   <-chan struct{}
	cancelBeforeReturn bool
}

func (p *scopedTaskWitnessPlanner) Complete(ctx context.Context, messages []Message, tools []ToolDef, _ ...SampleOpt) (*Completion, error) {
	if p.step == 0 {
		names := map[string]bool{}
		for _, d := range tools {
			names[d.Function.Name] = true
		}
		for _, name := range []string{ToolTaskSpawn, ToolTaskWait, ToolTaskStatus, ToolTaskCancel} {
			if !names[name] {
				return nil, fmt.Errorf("parent catalog missing %s", name)
			}
		}
	}
	call := func(name string, args any) (*Completion, error) {
		raw, _ := json.Marshal(args)
		return &Completion{Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: fmt.Sprintf("%s-%d", p.name, p.step), Type: "function", Function: Func{Name: name, Arguments: string(raw)}}}}, FinishReason: "tool_calls"}, nil
	}
	latest := func() string {
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i].Role == RoleTool {
				return messages[i].Content
			}
		}
		return ""
	}
	stage := p.step
	p.step++
	switch stage {
	case 0:
		return call(ToolTaskSpawn, TaskSpawnRequest{TaskID: "shared-child", IdempotencyKey: "shared-idempotency-key", Prompt: p.name, ReadOnly: true})
	case 1:
		if p.returnAfterSpawn != nil {
			select {
			case <-p.returnAfterSpawn:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			if p.cancelBeforeReturn {
				return call(ToolTaskCancel, TaskCancelRequest{TaskID: "shared-child", Reason: "this parent ended"})
			}
			return &Completion{Message: Message{Role: RoleAssistant, Content: "parent ended"}, FinishReason: "stop"}, nil
		}
		return call(ToolTaskStatus, TaskStatusRequest{TaskID: "shared-child"})
	case 2:
		if p.cancelBeforeReturn {
			var cancelled TaskCancelReceipt
			if err := json.Unmarshal([]byte(latest()), &cancelled); err != nil {
				return nil, err
			}
			if !cancelled.Cancelled || cancelled.TaskID != "shared-child" {
				return nil, fmt.Errorf("wrong cancellation receipt: %+v", cancelled)
			}
			return &Completion{Message: Message{Role: RoleAssistant, Content: "parent ended"}, FinishReason: "stop"}, nil
		}
		var status TaskStatusReceipt
		if err := json.Unmarshal([]byte(latest()), &status); err != nil {
			return nil, err
		}
		if len(status.Tasks) != 1 || status.Tasks[0].Prompt != p.name || status.Tasks[0].State != TaskStateRunning {
			return nil, fmt.Errorf("%s saw another parent's or stale status: %+v", p.name, status)
		}
		p.statusSeen <- p.name
		all := true
		return call(ToolTaskWait, TaskWaitRequest{TaskID: "shared-child", WaitAll: &all, TimeoutMs: 5000})
	default:
		var result TaskWaitReceipt
		if err := json.Unmarshal([]byte(latest()), &result); err != nil {
			return nil, err
		}
		task := result.Tasks["shared-child"]
		if result.Completed != 1 || task == nil || task.State != TaskStateCompleted || task.Result != p.name {
			return nil, fmt.Errorf("%s saw wrong child result: %+v", p.name, result)
		}
		return &Completion{Message: Message{Role: RoleAssistant, Content: p.name}, FinishReason: "stop"}, nil
	}
}

type scopedTaskWitnessOutcome struct {
	name    string
	metrics ArmMetrics
	err     error
}

func waitScopedSignal[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case value := <-ch:
		return value
	case <-time.After(5 * time.Second):
		t.Fatal("scoped task witness timed out")
		var zero T
		return zero
	}
}

func TestRunScopedChildTasksKeepIdenticalIDsAndStatusIndependent(t *testing.T) {
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)
	if _, err := ArmTaskTools(); err != nil {
		t.Fatal(err)
	}
	legacy := GetActiveTaskState()
	if _, err := legacy.Spawn(TaskSpawnRequest{TaskID: "shared-child", IdempotencyKey: "shared-idempotency-key", Prompt: "legacy-global"}); err != nil {
		t.Fatal(err)
	}
	if err := legacy.CompleteTask("shared-child", "legacy-global", nil); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	started := make(chan string, 2)
	statusSeen := make(chan string, 2)
	out := make(chan scopedTaskWitnessOutcome, 2)
	release := make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	runner := func(ctx context.Context, req ChildTaskRunRequest) (any, error) {
		started <- req.Prompt
		select {
		case <-release:
			return req.Prompt, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	for _, name := range []string{"alpha", "beta"} {
		name := name
		go func() {
			m, err := RunArm(ctx, &scopedTaskWitnessPlanner{name: name, statusSeen: statusSeen}, name, true, 4, nil, WithChildTaskRunner(1, 1, runner))
			out <- scopedTaskWitnessOutcome{name, m, err}
		}()
	}
	first, second := waitScopedSignal(t, started), waitScopedSignal(t, started)
	if first == second {
		t.Fatalf("idempotency leaked between parents: %q twice", first)
	}
	waitScopedSignal(t, statusSeen)
	waitScopedSignal(t, statusSeen)
	releaseOnce.Do(func() { close(release) })
	for range 2 {
		result := waitScopedSignal(t, out)
		if result.err != nil || result.metrics.HitTurnCap || result.metrics.FinalAnswer != result.name {
			t.Errorf("%s completion: %+v, %v", result.name, result.metrics, result.err)
		}
		if result.metrics.ToolCalls != 3 {
			t.Errorf("%s executed %d tools, want spawn/status/wait", result.name, result.metrics.ToolCalls)
		}
	}
	if GetActiveTaskState() != legacy {
		t.Fatal("scoped RunArm replaced the legacy singleton")
	}
	tasks := legacy.GetTasks()
	if len(tasks) != 1 || tasks[0].Prompt != "legacy-global" || tasks[0].State != TaskStateCompleted {
		t.Fatalf("scoped cleanup mutated legacy state: %+v", tasks)
	}
}

func TestRunScopedChildCleanupJoinsOwnEffectsAndLeavesPeerRunning(t *testing.T) {
	for _, explicit := range []bool{false, true} {
		t.Run(map[bool]string{false: "parent_return", true: "task_cancel"}[explicit], func(t *testing.T) {
			runScopedCleanupWitness(t, explicit)
		})
	}
}

func runScopedCleanupWitness(t *testing.T, explicit bool) {
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	started := make(chan string, 2)
	statusSeen := make(chan string, 1)
	mayReturn := make(chan struct{})
	cancelSeen := make(chan struct{})
	finishOwned := make(chan struct{})
	finishPeer := make(chan struct{})
	var ownOnce, peerOnce sync.Once
	defer ownOnce.Do(func() { close(finishOwned) })
	defer peerOnce.Do(func() { close(finishPeer) })
	ended := make(chan scopedTaskWitnessOutcome, 1)
	peerEnded := make(chan scopedTaskWitnessOutcome, 1)
	runner := func(ctx context.Context, req ChildTaskRunRequest) (any, error) {
		started <- req.Prompt
		if req.Prompt == "ending-parent" {
			<-ctx.Done()
			close(cancelSeen)
			<-finishOwned
			return nil, ctx.Err()
		}
		select {
		case <-finishPeer:
			return req.Prompt, nil
		case <-ctx.Done():
			return nil, fmt.Errorf("peer child cancelled by another parent: %w", ctx.Err())
		}
	}
	go func() {
		m, e := RunArm(ctx, &scopedTaskWitnessPlanner{name: "ending-parent", returnAfterSpawn: mayReturn, cancelBeforeReturn: explicit}, "end after spawning", true, 3, nil, WithChildTaskRunner(1, 1, runner))
		ended <- scopedTaskWitnessOutcome{"ending-parent", m, e}
	}()
	go func() {
		m, e := RunArm(ctx, &scopedTaskWitnessPlanner{name: "peer", statusSeen: statusSeen}, "wait for own child", true, 4, nil, WithChildTaskRunner(1, 1, runner))
		peerEnded <- scopedTaskWitnessOutcome{"peer", m, e}
	}()
	waitScopedSignal(t, started)
	waitScopedSignal(t, started)
	waitScopedSignal(t, statusSeen)
	close(mayReturn)
	waitScopedSignal(t, cancelSeen)
	select {
	case result := <-ended:
		t.Fatalf("parent returned while its child still had effects: %+v", result)
	case <-time.After(50 * time.Millisecond):
	}
	select {
	case result := <-peerEnded:
		t.Fatalf("peer ended when another parent closed: %+v", result)
	default:
	}
	ownOnce.Do(func() { close(finishOwned) })
	if result := waitScopedSignal(t, ended); result.err != nil {
		t.Fatalf("ending parent: %v", result.err)
	}
	select {
	case result := <-peerEnded:
		t.Fatalf("peer was not kept independent: %+v", result)
	default:
	}
	peerOnce.Do(func() { close(finishPeer) })
	if result := waitScopedSignal(t, peerEnded); result.err != nil || result.metrics.FinalAnswer != "peer" {
		t.Fatalf("peer completion: %+v", result)
	}
}
func (p *scopedTaskWitnessPlanner) Model() string { return "scoped-task-witness" }
