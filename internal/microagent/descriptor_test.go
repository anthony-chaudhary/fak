package microagent_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

type descriptorGateway struct {
	calls    int
	messages []agent.Message
	tools    []agent.ToolDef
}

func (g *descriptorGateway) Model() string { return "descriptor-fixture" }
func (g *descriptorGateway) Complete(_ context.Context, m []agent.Message, tools []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	g.calls++
	g.messages = append([]agent.Message(nil), m...)
	g.tools = append([]agent.ToolDef(nil), tools...)
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "DONE: alpha"}}, nil
}

func TestDescriptorRunsThroughExistingHostGateway(t *testing.T) {
	gw := &descriptorGateway{}
	h, err := microagent.NewHost(gw, microagent.Config{Workers: 1, Queue: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	d := microagent.Descriptor{Schema: microagent.DescriptorSchema, ID: "d1", BaseID: "base-v1", TaskDelta: "do alpha",
		Tools: []string{"read_record"}, Budget: microagent.DescriptorBudget{MaxTurns: 1, MaxOutputTokens: 8},
		Continuation: []agent.Message{{Role: agent.RoleAssistant, Content: "prior"}}, OutputContract: microagent.OutputContract{Kind: "exact", Expected: "DONE: alpha"}}
	a, err := microagent.SpawnDescriptor(h, d, []agent.Message{{Role: agent.RoleSystem, Content: "shared base"}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		t.Fatal(err)
	}
	if a.Result != "DONE: alpha" || gw.calls != 1 {
		t.Fatalf("result=%q calls=%d", a.Result, gw.calls)
	}
	if len(gw.messages) != 3 || gw.messages[0].Content != "shared base" || gw.messages[2].Content != "do alpha" {
		t.Fatalf("messages=%+v", gw.messages)
	}
	if len(gw.tools) != 1 || gw.tools[0].Function.Name != "read_record" {
		t.Fatalf("tools=%+v", gw.tools)
	}
}

func TestDescriptorRejectsAmbientOrUnboundedShape(t *testing.T) {
	d := microagent.Descriptor{Schema: microagent.DescriptorSchema, ID: "d1", BaseID: "b", TaskDelta: "x", Budget: microagent.DescriptorBudget{MaxTurns: 2}, OutputContract: microagent.OutputContract{Kind: "nonempty"}}
	if err := d.Validate(); err == nil {
		t.Fatal("expected bounded one-turn rejection")
	}
	d.Budget = microagent.DescriptorBudget{MaxTurns: 1, MaxOutputTokens: 8}
	d.Tools = []string{"read", "read"}
	if err := d.Validate(); err == nil {
		t.Fatal("expected duplicate capability rejection")
	}
}

type multiTurnGateway struct {
	calls  int
	traces []string
}

func (g *multiTurnGateway) Model() string { return "multi-turn-fixture" }
func (g *multiTurnGateway) Complete(ctx context.Context, m []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	g.calls++
	g.traces = append(g.traces, microagent.TraceFromContext(ctx))
	content := fmt.Sprintf("PROGRESS-%d", g.calls)
	if g.calls == 3 {
		content = "DONE"
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: content}}, nil
}

func TestDescriptorV2AccountsTurnsAndRoundTripsContinuation(t *testing.T) {
	gw := &multiTurnGateway{}
	d := microagent.Descriptor{Schema: microagent.DescriptorSchemaV2, ID: "d2", BaseID: "base-v1", TaskDelta: "advance",
		Budget: microagent.DescriptorBudget{MaxTurns: 3, MaxOutputTokens: 8}, ContinuationToken: "continuation-d2",
		OutputContract: microagent.OutputContract{Kind: "exact", Expected: "DONE"}}
	a := &microagent.DescriptorAgent{Descriptor: d, Base: []agent.Message{{Role: agent.RoleSystem, Content: "shared base"}}}
	if done, err := a.Step(context.Background(), gw); err != nil || done {
		t.Fatalf("turn1 done=%v err=%v", done, err)
	}
	if a.TurnsUsed != 1 || len(a.History) != 1 {
		t.Fatalf("turn1 state=%+v", a)
	}

	store, err := microagent.NewHibernationStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Park(d.ID, a); err != nil {
		t.Fatal(err)
	}
	restored := &microagent.DescriptorAgent{Descriptor: d, Base: append([]agent.Message(nil), a.Base...)}
	if err := store.Wake(d.ID, restored); err != nil {
		t.Fatal(err)
	}
	if restored.TurnsUsed != 1 || len(restored.History) != 1 {
		t.Fatalf("restored=%+v", restored)
	}
	if done, err := restored.Step(context.Background(), gw); err != nil || done {
		t.Fatalf("turn2 done=%v err=%v", done, err)
	}
	if done, err := restored.Step(context.Background(), gw); err != nil || !done {
		t.Fatalf("turn3 done=%v err=%v", done, err)
	}
	if restored.TurnsUsed != 3 || restored.Result != "DONE" {
		t.Fatalf("final=%+v", restored)
	}
	for _, trace := range gw.traces {
		if trace != d.ContinuationToken {
			t.Fatalf("trace=%q want=%q", trace, d.ContinuationToken)
		}
	}
}

func TestDescriptorV2RefusesOverBudgetAndIdentitySwap(t *testing.T) {
	d := microagent.Descriptor{Schema: microagent.DescriptorSchemaV2, ID: "d2", BaseID: "b", TaskDelta: "x", Budget: microagent.DescriptorBudget{MaxTurns: 2, MaxOutputTokens: 8}, ContinuationToken: "ct-d2", OutputContract: microagent.OutputContract{Kind: "exact", Expected: "never"}}
	a := &microagent.DescriptorAgent{Descriptor: d}
	gw := &multiTurnGateway{}
	if done, err := a.Step(context.Background(), gw); err != nil || done {
		t.Fatalf("step1 done=%v err=%v", done, err)
	}
	if _, err := a.Step(context.Background(), gw); err == nil {
		t.Fatal("expected final output-contract/budget refusal")
	}
	if _, err := a.Step(context.Background(), gw); err == nil {
		t.Fatal("expected exhausted budget refusal")
	}
	b, err := a.Freeze()
	if err != nil {
		t.Fatal(err)
	}
	other := &microagent.DescriptorAgent{Descriptor: d}
	other.Descriptor.ID = "other"
	if err := other.Thaw(b); err == nil {
		t.Fatal("expected identity mismatch")
	}
}
