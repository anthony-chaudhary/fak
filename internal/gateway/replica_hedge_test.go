package gateway

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type hedgeTestPlanner struct {
	model   string
	result  *agent.Completion
	err     error
	delay   time.Duration
	ignore  bool
	started chan struct{}

	mu    sync.Mutex
	calls int
}

func (p *hedgeTestPlanner) Model() string { return p.model }
func (p *hedgeTestPlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.mu.Lock()
	p.calls++
	p.mu.Unlock()
	if p.started != nil {
		select {
		case p.started <- struct{}{}:
		default:
		}
	}
	if p.ignore {
		time.Sleep(p.delay)
	} else {
		select {
		case <-time.After(p.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return p.result, p.err
}
func (p *hedgeTestPlanner) callCount() int { p.mu.Lock(); defer p.mu.Unlock(); return p.calls }

func zeroTemperatureOpt() agent.SampleOpt {
	zero := 0.0
	return agent.WithTemperature(&zero)
}
func completion(content string) *agent.Completion {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: content}, Model: "same"}
}

func hedgeRouter(primary, alternate agent.Planner, policy *HedgePolicy) *ReplicaRouter {
	r, err := NewReplicaRouter("same", []PlannerReplica{{Name: "primary", Planner: primary}, {Name: "alternate", Planner: alternate}})
	if err != nil {
		panic(err)
	}
	r.Hedge = policy
	return r
}

func eligibleHedgePolicy(observe func(HedgeReceipt)) *HedgePolicy {
	return &HedgePolicy{Enabled: true, PolicyVersion: "test/v1", ProviderContract: "openai-buffered/v1", ReadOnly: true, Deterministic: true, Delay: 5 * time.Millisecond, Deadline: time.Second, DrainTimeout: 20 * time.Millisecond, DuplicateWorkBudget: 1, SpareCapacity: func() bool { return true }, Observe: observe}
}

func TestReplicaRouterHedgeFirstValidAndCancellationReceipt(t *testing.T) {
	primary := &hedgeTestPlanner{model: "same", result: completion("slow"), delay: 200 * time.Millisecond}
	alternate := &hedgeTestPlanner{model: "same", result: completion("fast"), delay: time.Millisecond}
	var receipt HedgeReceipt
	r := hedgeRouter(primary, alternate, eligibleHedgePolicy(func(got HedgeReceipt) { receipt = got }))

	got, err := r.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "read only"}}, nil, zeroTemperatureOpt())
	if err != nil {
		t.Fatal(err)
	}
	if got.Message.Content != "fast" {
		t.Fatalf("winner content = %q", got.Message.Content)
	}
	if receipt.Schema != hedgeReceiptSchema || receipt.RealizedMode != "hedged" || receipt.Winner != "hedge" || !receipt.FirstValid {
		t.Fatalf("receipt = %+v", receipt)
	}
	if !receipt.CancellationRequested || !receipt.LocalTransportAcknowledgement || receipt.DrainOutcome != "completed" || receipt.ProviderWorkAfterCancel != "unknown" || receipt.CancelledBilling != "unknown" {
		t.Fatalf("cancellation receipt = %+v", receipt)
	}
	if primary.callCount() != 1 || alternate.callCount() != 1 {
		t.Fatalf("calls primary=%d alternate=%d", primary.callCount(), alternate.callCount())
	}
}

func TestReplicaRouterHedgeRejectsMalformedFirst(t *testing.T) {
	primary := &hedgeTestPlanner{model: "same", result: completion(""), delay: 8 * time.Millisecond}
	alternate := &hedgeTestPlanner{model: "same", result: completion("valid"), delay: 12 * time.Millisecond}
	var receipt HedgeReceipt
	r := hedgeRouter(primary, alternate, eligibleHedgePolicy(func(got HedgeReceipt) { receipt = got }))
	got, err := r.Complete(context.Background(), nil, nil, zeroTemperatureOpt())
	if err != nil {
		t.Fatal(err)
	}
	if got.Message.Content != "valid" || receipt.Winner != "hedge" || receipt.Primary.TerminalClass != "invalid_completion" {
		t.Fatalf("got=%+v receipt=%+v", got, receipt)
	}
}

func TestReplicaRouterHedgeAppliesAcceptedOutputValidator(t *testing.T) {
	primary := &hedgeTestPlanner{model: "same", result: completion("schema-invalid"), delay: 8 * time.Millisecond}
	alternate := &hedgeTestPlanner{model: "same", result: completion("accepted"), delay: 20 * time.Millisecond}
	var receipt HedgeReceipt
	policy := eligibleHedgePolicy(func(got HedgeReceipt) { receipt = got })
	policy.Validate = func(c *agent.Completion) error {
		if c.Message.Content != "accepted" {
			return errors.New("schema-invalid")
		}
		return nil
	}

	got, err := hedgeRouter(primary, alternate, policy).Complete(context.Background(), nil, nil, zeroTemperatureOpt())
	if err != nil {
		t.Fatal(err)
	}
	if got.Message.Content != "accepted" || receipt.Winner != "hedge" || receipt.Primary.TerminalClass != "invalid_completion" {
		t.Fatalf("got=%+v receipt=%+v", got, receipt)
	}
}

func TestReplicaRouterHedgePrimaryBeforeDelayStaysSingleDispatch(t *testing.T) {
	primary := &hedgeTestPlanner{model: "same", result: completion("primary"), delay: time.Millisecond}
	alternate := &hedgeTestPlanner{model: "same", result: completion("alternate")}
	var receipt HedgeReceipt
	policy := eligibleHedgePolicy(func(got HedgeReceipt) { receipt = got })
	policy.Delay = 100 * time.Millisecond
	r := hedgeRouter(primary, alternate, policy)
	got, err := r.Complete(context.Background(), nil, nil, zeroTemperatureOpt())
	if err != nil {
		t.Fatal(err)
	}
	if got.Message.Content != "primary" || alternate.callCount() != 0 || receipt.AbstentionReason != "primary_completed_before_delay" || receipt.RealizedMode != "primary_only" {
		t.Fatalf("got=%+v calls=%d receipt=%+v", got, alternate.callCount(), receipt)
	}
}

func TestReplicaRouterHedgeIneligibleCallsStaySingleDispatch(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*HedgePolicy)
		tools  []agent.ToolDef
	}{
		{name: "disabled", mutate: func(p *HedgePolicy) { p.Enabled = false }},
		{name: "not read only", mutate: func(p *HedgePolicy) { p.ReadOnly = false }},
		{name: "not deterministic", mutate: func(p *HedgePolicy) { p.Deterministic = false }},
		{name: "tools", tools: []agent.ToolDef{{Type: "function"}}},
		{name: "budget", mutate: func(p *HedgePolicy) { p.DuplicateWorkBudget = 0 }},
		{name: "unknown provider contract", mutate: func(p *HedgePolicy) { p.ProviderContract = "" }},
		{name: "missing capacity gate", mutate: func(p *HedgePolicy) { p.SpareCapacity = nil }},
		{name: "deadline before delay", mutate: func(p *HedgePolicy) { p.Deadline = p.Delay }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			primary := &hedgeTestPlanner{model: "same", result: completion("primary")}
			alternate := &hedgeTestPlanner{model: "same", result: completion("alternate")}
			policy := eligibleHedgePolicy(nil)
			if tc.mutate != nil {
				tc.mutate(policy)
			}
			r := hedgeRouter(primary, alternate, policy)
			got, err := r.Complete(context.Background(), nil, tc.tools, zeroTemperatureOpt())
			if err != nil {
				t.Fatal(err)
			}
			if got.Message.Content != "primary" || primary.callCount() != 1 || alternate.callCount() != 0 {
				t.Fatalf("got=%+v calls=%d/%d", got, primary.callCount(), alternate.callCount())
			}
		})
	}
}

func TestReplicaRouterHedgeBoundedDrainDoesNotHoldWinner(t *testing.T) {
	primary := &hedgeTestPlanner{model: "same", result: completion("late"), delay: 200 * time.Millisecond, ignore: true}
	alternate := &hedgeTestPlanner{model: "same", result: completion("winner"), delay: time.Millisecond}
	var receipt HedgeReceipt
	policy := eligibleHedgePolicy(func(got HedgeReceipt) { receipt = got })
	policy.DrainTimeout = 10 * time.Millisecond
	started := time.Now()
	got, err := hedgeRouter(primary, alternate, policy).Complete(context.Background(), nil, nil, zeroTemperatureOpt())
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(started); elapsed > 100*time.Millisecond {
		t.Fatalf("winner held for %s", elapsed)
	}
	if got.Message.Content != "winner" || receipt.DrainOutcome != "timeout" || receipt.LoserResultChannelClosed {
		t.Fatalf("got=%+v receipt=%+v", got, receipt)
	}
}

func TestReplicaRouterHedgeBothInvalidReturnsTypedJoinedError(t *testing.T) {
	primary := &hedgeTestPlanner{model: "same", err: errors.New("primary failed"), delay: 8 * time.Millisecond}
	alternate := &hedgeTestPlanner{model: "same", result: completion(""), delay: 4 * time.Millisecond}
	_, err := hedgeRouter(primary, alternate, eligibleHedgePolicy(nil)).Complete(context.Background(), nil, nil, zeroTemperatureOpt())
	if err == nil || err.Error() != "hedged completion failed: primary=error; hedge=invalid_completion" {
		t.Fatalf("error = %v", err)
	}
}
