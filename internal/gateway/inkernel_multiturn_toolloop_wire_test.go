package gateway

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// multiTurnLiftPlanner is a scripted stand-in for InKernelPlanner that drives a
// multi-turn agentic tool loop the way Claude Code does. It decides its next move
// from the transcript the harness fed back, emits the call as text, then lifts it
// with the same agent.LiftTextToolCalls path the in-kernel forward runs through
// normalizeCompletionToolCalls. Only the model decode is scripted; inbound
// tool_result decode, kernel adjudication, text-to-structured lift, and Anthropic
// response rendering are production code.
//
// This witnesses the loop the single-call proof (commit 4aec27a) left open (#610).
// The live forward's model-competence half is host-gated by CPU prefill latency, so
// it is measured on a GPU/Metal decode lane; this test pins the round-trip wiring
// that a live forward must drive.
type multiTurnLiftPlanner struct {
	seenResults []int    // # tool_results visible to the planner on each turn it was asked
	seenContent []string // the last tool_response body the planner read on each turn
}

func (p *multiTurnLiftPlanner) Model() string { return "qwen2.5-7b-q8" }

func (p *multiTurnLiftPlanner) Complete(_ context.Context, m []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	// DecodeAnthropicMessagesRequest turned each inbound Anthropic tool_result block into a
	// RoleTool message; counting them is how the model "reads the result and calls the next
	// tool". Without the harness round-trip these counts would never advance past 0.
	results, lastContent := 0, ""
	for _, msg := range m {
		if msg.Role == agent.RoleTool {
			results++
			lastContent = msg.Content
		}
	}
	p.seenResults = append(p.seenResults, results)
	p.seenContent = append(p.seenContent, lastContent)

	var raw string
	switch results {
	case 0:
		// Turn 1: nothing run yet -> emit the FIRST tool call.
		raw = `<tool_call>{"name": "allow_list_files", "arguments": {"path": "."}}</tool_call>`
	case 1:
		// Turn 2: read the first result, emit the SECOND tool call.
		raw = `<tool_call>{"name": "allow_read_file", "arguments": {"path": "main.go"}}</tool_call>`
	default:
		// Turn 3: both results are in -> final plain-text answer, no tool call (end_turn).
		raw = "Both files are present; main.go is the entrypoint."
	}
	msg := agent.LiftTextToolCalls(agent.Message{Role: agent.RoleAssistant, Content: raw})
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

// TestInKernelMultiTurnToolLoopReachesAnthropicWire witnesses the full agentic loop on
// fak's own in-kernel tool-calling path over the /v1/messages wire Claude Code
// drives (#610): call -> tool_result -> second call -> tool_result -> final
// answer, every call adjudicated. Turn 2's call exists only because the planner
// read turn 1's round-tripped tool_result, and the loop terminates once both
// results are in.
func TestInKernelMultiTurnToolLoopReachesAnthropicWire(t *testing.T) {
	srv := newTestServer(t)
	pl := &multiTurnLiftPlanner{}
	srv.planner = pl
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// The two tools advertised to the model for the whole session (both "allow*" so the
	// kernel adjudicator admits them; the loop must survive adjudication, not bypass it).
	tools := []map[string]any{
		{"name": "allow_list_files", "input_schema": map[string]any{"type": "object"}},
		{"name": "allow_read_file", "input_schema": map[string]any{"type": "object"}},
	}

	// --- Turn 1: user asks; the in-kernel forward emits the FIRST tool call. ---
	msgs := []map[string]any{
		{"role": "user", "content": "list the files then read main.go"},
	}
	resp1 := postAnthropic(t, ts.URL, "inkernel-multiturn-610-t1", multiTurnBody(t, msgs, tools))
	id1, name1 := firstToolUse(t, resp1)
	if resp1.StopReason != "tool_use" || name1 != "allow_list_files" {
		t.Fatalf("turn 1: stop=%q firstToolUse=%q, want tool_use/allow_list_files", resp1.StopReason, name1)
	}
	assertAdjudicatedAllow(t, resp1, "allow_list_files")

	// Claude Code's harness ran the tool and returns its result, then re-asks with the
	// full transcript (the stateless /v1/messages contract).
	msgs = append(msgs,
		map[string]any{"role": "assistant", "content": []map[string]any{
			{"type": "tool_use", "id": id1, "name": name1, "input": map[string]any{"path": "."}}}},
		map[string]any{"role": "user", "content": []map[string]any{
			{"type": "tool_result", "tool_use_id": id1, "content": "main.go go.mod"}}},
	)

	// --- Turn 2: the model READS the first result and emits the SECOND tool call. ---
	resp2 := postAnthropic(t, ts.URL, "inkernel-multiturn-610-t2", multiTurnBody(t, msgs, tools))
	id2, name2 := firstToolUse(t, resp2)
	if resp2.StopReason != "tool_use" || name2 != "allow_read_file" {
		t.Fatalf("turn 2: stop=%q firstToolUse=%q, want tool_use/allow_read_file (the round-tripped result must drive the next call)", resp2.StopReason, name2)
	}
	assertAdjudicatedAllow(t, resp2, "allow_read_file")

	msgs = append(msgs,
		map[string]any{"role": "assistant", "content": []map[string]any{
			{"type": "tool_use", "id": id2, "name": name2, "input": map[string]any{"path": "main.go"}}}},
		map[string]any{"role": "user", "content": []map[string]any{
			{"type": "tool_result", "tool_use_id": id2, "content": "package main"}}},
	)

	// --- Turn 3: both results are in -> final answer, the loop terminates (end_turn). ---
	resp3 := postAnthropic(t, ts.URL, "inkernel-multiturn-610-t3", multiTurnBody(t, msgs, tools))
	if resp3.StopReason != "end_turn" {
		t.Fatalf("turn 3: stop=%q, want end_turn (the model should stop after reading both results)", resp3.StopReason)
	}
	for _, b := range resp3.Content {
		if b.Type == "tool_use" {
			t.Fatalf("turn 3: unexpected tool_use %q; the loop should have terminated", b.Name)
		}
	}

	// The round-trip WIRING witness: the planner standing in for the in-kernel forward saw
	// 0, then 1, then 2 tool_results. That proves the harness threaded each result back into
	// the model's next prompt. A broken round-trip would leave every count at 0 (and turn 2
	// would re-emit the first call forever).
	if got := pl.seenResults; len(got) != 3 || got[0] != 0 || got[1] != 1 || got[2] != 2 {
		t.Fatalf("planner saw tool_results %v across turns, want [0 1 2] (the multi-turn round-trip)", got)
	}
	if pl.seenContent[1] != "main.go go.mod" || pl.seenContent[2] != "package main" {
		t.Fatalf("planner read tool_result content %q / %q, want the round-tripped bodies", pl.seenContent[1], pl.seenContent[2])
	}
}

// multiTurnBody marshals a /v1/messages request body (max_tokens is required on the
// Anthropic wire, so a real client always sends one).
func multiTurnBody(t *testing.T, messages, tools []map[string]any) string {
	t.Helper()
	b, err := json.Marshal(map[string]any{
		"model":      "claude-opus-4-8",
		"max_tokens": 256,
		"messages":   messages,
		"tools":      tools,
	})
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}
	return string(b)
}

// firstToolUse returns the id+name of the first tool_use block, failing if none survived
// adjudication.
func firstToolUse(t *testing.T, resp anthropicMessageResponse) (id, name string) {
	t.Helper()
	for _, b := range resp.Content {
		if b.Type == "tool_use" {
			return b.ID, b.Name
		}
	}
	t.Fatalf("no tool_use block in response: %+v", resp.Content)
	return "", ""
}

// assertAdjudicatedAllow proves the named call went THROUGH the kernel adjudicator and was
// admitted (the "all adjudicated" half of the witness), not merely echoed to the wire.
func assertAdjudicatedAllow(t *testing.T, resp anthropicMessageResponse, tool string) {
	t.Helper()
	if resp.Fak == nil {
		t.Fatalf("missing fak adjudication extension for %q", tool)
	}
	for _, a := range resp.Fak.Adjudications {
		if a.Tool == tool {
			if a.Verdict.Kind != "ALLOW" || !a.Admitted {
				t.Fatalf("tool %q adjudication = %+v (admitted=%v), want ALLOW/admitted", tool, a.Verdict, a.Admitted)
			}
			return
		}
	}
	t.Fatalf("no adjudication recorded for %q: %+v", tool, resp.Fak.Adjudications)
}
