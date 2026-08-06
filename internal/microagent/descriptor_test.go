package microagent_test

import (
	"context"
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
