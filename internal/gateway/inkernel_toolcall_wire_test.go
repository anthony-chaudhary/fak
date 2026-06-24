package gateway

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// inkernelLiftPlanner reproduces the in-kernel planner's post-decode behavior: it decodes
// a raw turn whose text carries a Hermes <tool_call> (Qwen2.5's native dialect), then runs
// the SAME lift the in-kernel forward now runs, so the structured ToolCalls reach the
// gateway. This is the seam that turns `fak serve --gguf` from a chat toy into a
// tool-calling agent backend.
type inkernelLiftPlanner struct{ rawText string }

func (inkernelLiftPlanner) Model() string { return "qwen2.5-7b-q8" }

func (p inkernelLiftPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	// Mirror InKernelPlanner.Complete's tail: lift the model's text-form <tool_call> into
	// structured ToolCalls via the exported LiftTextToolCalls (the same helper the
	// in-kernel path runs through normalizeCompletionToolCalls), and set the tool_calls
	// finish reason when a call was recovered.
	msg := agent.LiftTextToolCalls(agent.Message{Role: agent.RoleAssistant, Content: p.rawText})
	finish := "stop"
	if len(msg.ToolCalls) > 0 {
		finish = "tool_calls"
	}
	return &agent.Completion{
		Message:      msg,
		FinishReason: finish,
		Usage:        agent.Usage{PromptTokens: 9, CompletionTokens: 5, TotalTokens: 14},
	}, nil
}

// TestInKernelToolCallReachesAnthropicWire proves the end-to-end property the whole
// feature exists for: a model that emits a tool call as TEXT through fak's own in-kernel
// forward is lifted to a structured call, survives the kernel adjudicator, and is rendered
// as a tool_use block with stop_reason "tool_use" on the /v1/messages wire Claude Code
// drives. Before the fix this returned end_turn with the call stuck in content.
func TestInKernelToolCallReachesAnthropicWire(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = inkernelLiftPlanner{rawText: `<tool_call>{"name": "allow_a", "arguments": {"x": 1}}</tool_call>`}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := json.RawMessage(`{"model":"claude-opus-4-8","messages":[{"role":"user","content":"list the files"}],
		"tools":[{"name":"allow_a","input_schema":{"type":"object"}}]}`)
	var resp anthropicMessageResponse
	if code := postJSON(t, ts.URL+"/v1/messages", body, &resp); code != 200 {
		t.Fatalf("status = %d", code)
	}
	if resp.StopReason != "tool_use" {
		t.Fatalf("stop_reason = %q, want tool_use (the in-kernel text tool_call must lift to a structured call)", resp.StopReason)
	}
	var sawToolUse bool
	for _, b := range resp.Content {
		if b.Type == "tool_use" && b.Name == "allow_a" {
			sawToolUse = true
		}
	}
	if !sawToolUse {
		t.Fatalf("no allow_a tool_use block on the wire: %+v", resp.Content)
	}
}
