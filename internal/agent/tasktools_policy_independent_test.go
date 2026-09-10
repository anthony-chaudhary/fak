package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type scopedPolicyIndependentPlanner struct {
	ready      chan struct{}
	release    <-chan struct{}
	step       int
	wantDenied bool
}

func (p *scopedPolicyIndependentPlanner) Model() string { return "scoped-policy-witness" }
func (p *scopedPolicyIndependentPlanner) Complete(ctx context.Context, messages []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	call := func(name string, args string) (*Completion, error) {
		return &Completion{Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: name, Function: Func{Name: name, Arguments: args}}}}, FinishReason: "tool_calls"}, nil
	}
	latest := ""
	for _, m := range messages {
		if m.Role == RoleTool {
			latest = m.Content
		}
	}
	step := p.step
	p.step++
	if step == 0 {
		close(p.ready)
		select {
		case <-p.release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return call(ToolTaskSpawn, `{"task_id":"same-task","idempotency_key":"same-key","prompt":"owned child"}`)
	}
	if p.wantDenied {
		var receipt ToolReceipt
		if err := json.Unmarshal([]byte(latest), &receipt); err != nil || receipt.Reason != "POLICY_BLOCK" {
			return nil, fmt.Errorf("explicit denied parent's spawn was not denied: %s (%v)", latest, err)
		}
		return &Completion{Message: Message{Content: "denied as required"}}, nil
	}
	if step == 1 {
		var receipt TaskSpawnReceipt
		if err := json.Unmarshal([]byte(latest), &receipt); err != nil || receipt.TaskID != "same-task" || receipt.Error != "" {
			return nil, fmt.Errorf("permissive parent's policy overwritten: %s (%v)", latest, err)
		}
		return call(ToolTaskWait, `{"task_id":"same-task","timeout_ms":1000}`)
	}
	var receipt TaskWaitReceipt
	if err := json.Unmarshal([]byte(latest), &receipt); err != nil || receipt.Completed != 1 {
		return nil, fmt.Errorf("allowed child did not complete: %s (%v)", latest, err)
	}
	return &Completion{Message: Message{Content: "allowed child completed"}}, nil
}

func TestRunScopedChildPoliciesCannotOverwritePeerAdmission(t *testing.T) {
	t.Cleanup(Configure)
	adjudicator.Default.SetPolicy(adjudicator.Policy{
		Posture: adjudicator.PostureFailClosed,
		Allow:   map[string]bool{"outside-run-sentinel": true},
	})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	releaseA, releaseB := make(chan struct{}), make(chan struct{})
	var onceA, onceB sync.Once
	openA := func() { onceA.Do(func() { close(releaseA) }) }
	openB := func() { onceB.Do(func() { close(releaseB) }) }
	defer openA()
	defer openB()
	a := &scopedPolicyIndependentPlanner{ready: make(chan struct{}), release: releaseA}
	b := &scopedPolicyIndependentPlanner{ready: make(chan struct{}), release: releaseB, wantDenied: true}
	policyA := adjudicator.Policy{Posture: adjudicator.PostureFailClosed, Allow: map[string]bool{ToolTaskSpawn: true, ToolTaskWait: true}}
	policyB := adjudicator.Policy{Posture: adjudicator.PostureFailClosed, Allow: map[string]bool{ToolTaskWait: true}, Deny: map[string]abi.ReasonCode{ToolTaskSpawn: abi.ReasonPolicyBlock}}
	var startedA, startedB atomic.Int32
	type result struct {
		metrics ArmMetrics
		err     error
	}
	doneA, doneB := make(chan result, 1), make(chan result, 1)
	go func() {
		m, e := RunArm(ctx, a, "allowed parent", true, 4, nil, WithPolicySnapshot(policyA), WithChildTaskRunner(1, 1, func(context.Context, ChildTaskRunRequest) (any, error) { startedA.Add(1); return "owned", nil }))
		doneA <- result{m, e}
	}()
	select {
	case <-a.ready:
	case <-ctx.Done():
		t.Fatal("allowed parent did not enter planner")
	}
	go func() {
		m, e := RunArm(ctx, b, "denied parent", true, 3, nil, WithPolicySnapshot(policyB), WithChildTaskRunner(1, 1, func(context.Context, ChildTaskRunRequest) (any, error) { startedB.Add(1); return "must not run", nil }))
		doneB <- result{m, e}
	}()
	select {
	case <-b.ready:
	case <-ctx.Done():
		t.Fatal("denied parent did not enter planner")
	}
	// B's startup policy installation is complete before A dispatches.
	openA()
	var ra result
	select {
	case ra = <-doneA:
	case <-ctx.Done():
		t.Fatal("allowed parent did not finish")
	}
	openB()
	var rb result
	select {
	case rb = <-doneB:
	case <-ctx.Done():
		t.Fatal("denied parent did not finish")
	}
	if ra.err != nil || ra.metrics.Denies != 0 || ra.metrics.FinalAnswer != "allowed child completed" || startedA.Load() != 1 {
		t.Errorf("allowed parent: metrics=%+v err=%v children=%d", ra.metrics, ra.err, startedA.Load())
	}
	if rb.err != nil || rb.metrics.Denies != 1 || rb.metrics.FinalAnswer != "denied as required" || startedB.Load() != 0 {
		t.Errorf("denied parent: metrics=%+v err=%v children=%d", rb.metrics, rb.err, startedB.Load())
	}
	global := adjudicator.Default.PolicySnapshot()
	if len(global.Allow) != 1 || !global.Allow["outside-run-sentinel"] {
		t.Errorf("scoped runs changed the process-global policy: allow=%v", global.Allow)
	}
}
