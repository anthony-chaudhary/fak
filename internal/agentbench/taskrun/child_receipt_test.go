package taskrun

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/codetools"
)

type receiptReadPlanner struct{ calls int }

func (p *receiptReadPlanner) Model() string { return "receipt-roundtrip" }
func (p *receiptReadPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	args := fmt.Sprintf(`{"file_path":%q}`, `C:\Users\bench\AppData\Local\Temp\fak-agentbench-trial-0123456789\workspace\internal\retry\delay.go`)
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: fmt.Sprintf("inspecting step %d", p.calls), ToolCalls: []agent.ToolCall{{ID: fmt.Sprintf("call-%02d", p.calls), Type: "function", Function: agent.Func{Name: codetools.ToolRead, Arguments: args}}}}}, nil
}

func TestChildReceiptTwelveTurnTranscriptRoundTripsUnderStdoutCap(t *testing.T) {
	p := &recordingPlanner{inner: &receiptReadPlanner{}}
	msgs := []agent.Message{
		{Role: agent.RoleSystem, Content: strings.Repeat("You are a careful Go maintainer. ", 90)},
		{Role: agent.RoleUser, Content: strings.Repeat("Fix the retry delay so it is capped at 1000ms. ", 70)},
	}
	var requests [][]agent.Message
	for turn := 1; turn <= maxTaskTurns; turn++ {
		if turn == 7 {
			msgs = append(append([]agent.Message(nil), msgs[:2]...), msgs[len(msgs)-4:]...)
		}
		requests = append(requests, append([]agent.Message(nil), msgs...))
		c, err := p.Complete(context.Background(), msgs, nil)
		if err != nil {
			t.Fatalf("turn %d: %v", turn, err)
		}
		call := c.Message.ToolCalls[0]
		msgs = append(msgs, c.Message, agent.Message{Role: agent.RoleTool, ToolCallID: call.ID, Content: fmt.Sprintf(`{"content":%q}`, strings.Repeat(fmt.Sprintf("// line of delay.go read at turn %d\n", turn), 45))})
	}

	var out boundedWriter
	if err := json.NewEncoder(&out).Encode(ChildReceipt{Schema: "receipt-roundtrip", Model: p.Model(), PlannerCalls: len(p.turns), Turns: p.turns}); err != nil {
		t.Fatal(err)
	}
	wire := out.Bytes()
	var got ChildReceipt
	if err := json.NewDecoder(bytes.NewReader(wire)).Decode(&got); err != nil {
		t.Fatalf("12-turn receipt (%d bytes on stdout) does not decode: %v", len(wire), err)
	}
	if got.PlannerCalls != maxTaskTurns || len(got.Turns) != maxTaskTurns {
		t.Fatalf("planner_calls=%d turns=%d, want %d", got.PlannerCalls, len(got.Turns), maxTaskTurns)
	}
	for turn := 1; turn < maxTaskTurns; turn++ {
		if id := fmt.Sprintf("call-%02d", turn); !followingToolSucceeded(got.Turns, id) {
			t.Fatalf("tool result for %s lost in the decoded receipt", id)
		}
	}

	var shape struct {
		Turns []struct {
			Prefix   int             `json:"prefix_messages"`
			Messages []agent.Message `json:"messages"`
		} `json:"turns"`
	}
	if err := json.Unmarshal(wire, &shape); err != nil {
		t.Fatal(err)
	}
	var prev []agent.Message
	for i, turn := range shape.Turns {
		if turn.Prefix > len(prev) {
			t.Fatalf("turn %d prefix %d exceeds previous request length %d", i+1, turn.Prefix, len(prev))
		}
		full := append(append([]agent.Message(nil), prev[:turn.Prefix]...), turn.Messages...)
		if !reflect.DeepEqual(full, requests[i]) {
			t.Fatalf("turn %d request does not reconstruct from the receipt", i+1)
		}
		prev = full
	}
}
