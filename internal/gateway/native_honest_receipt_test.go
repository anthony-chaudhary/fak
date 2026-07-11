package gateway

// native_honest_receipt_test.go — #2414 in the gateway: the OWNED-loop serve path hands
// the model TYPED tool_result receipts, not prose. TestDeniedCallReturnsStructuredError
// drives a policy-denied call through the real `serve --native` path and proves the NEXT
// planner input carries a structured error bound to the originating call id (with the
// prose adjudicationNote provably ABSENT); TestNoOpReportsSkipped proves the owned loop's
// not-sent no-op surfaces status=skipped at the RunArm seam the native serve path drives.

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// recordingNativePlanner returns a fixed per-turn script and records the messages it was
// handed each turn, so the test can inspect what the owned loop placed in the NEXT
// planner input (the tool_result receipt on the originating call id).
type recordingNativePlanner struct {
	turns []*agent.Completion
	seen  [][]agent.Message
	n     int
}

func (p *recordingNativePlanner) Model() string { return "recording-native" }

func (p *recordingNativePlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.seen = append(p.seen, append([]agent.Message(nil), messages...))
	c := p.turns[p.n]
	if p.n < len(p.turns)-1 {
		p.n++
	}
	return c, nil
}

func nativeCallTurn(id, tool, args string, promptLen int) *agent.Completion {
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: id, Type: "function", Function: agent.Func{Name: tool, Arguments: args}}}},
		FinishReason: "tool_calls",
		Usage:        agent.Usage{PromptTokens: promptLen, CompletionTokens: 2},
	}
}

func findNativeReceipt(t *testing.T, msgs []agent.Message, callID string) agent.ToolReceipt {
	t.Helper()
	for _, msg := range msgs {
		if msg.Role != agent.RoleTool || msg.ToolCallID != callID {
			continue
		}
		if !strings.HasPrefix(strings.TrimSpace(msg.Content), "{") {
			t.Fatalf("owned-loop tool_result on call %q is prose, not a typed receipt: %q", callID, msg.Content)
		}
		var rc agent.ToolReceipt
		if err := json.Unmarshal([]byte(msg.Content), &rc); err != nil {
			t.Fatalf("tool_result on call %q is not a typed receipt: %v; content=%q", callID, err, msg.Content)
		}
		return rc
	}
	t.Fatalf("no tool_result bound to originating call id %q in the next planner input; msgs=%+v", callID, msgs)
	return agent.ToolReceipt{}
}

func TestDeniedCallReturnsStructuredError(t *testing.T) {
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})

	planner := &recordingNativePlanner{turns: []*agent.Completion{
		nativeCallTurn("call_del", "delete_account", `{"user_id":"mia_li_3668"}`, 8),
		{Message: agent.Message{Role: agent.RoleAssistant, Content: "I cannot delete the account."}, FinishReason: "stop", Usage: agent.Usage{PromptTokens: 12, CompletionTokens: 4}},
	}}

	srv, err := New(Config{EngineID: "localtools", Model: "test-model", VDSO: true, Native: true, NativeMaxTurns: 8})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{
		"model":      "test-model",
		"max_tokens": 256,
		"messages":   []map[string]string{{"role": "user", "content": "Delete my account."}},
	})
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/messages: %v", err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", resp.StatusCode, raw)
	}

	// The owned loop denied the call (native_arm witnesses the kernel-mediated arm).
	var got anthropicMessageResponse
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode response: %v; body=%s", err, raw)
	}
	if got.Fak == nil || got.Fak.NativeArm == nil {
		t.Fatalf("response carried no fak.native_arm — not the owned loop; body=%s", raw)
	}
	if got.Fak.NativeArm.Denies == 0 {
		t.Fatalf("native_arm.denies = 0 — the owned loop did not deny delete_account")
	}

	// The NEXT planner input (turn 2) carries the structured error on the originating id.
	if len(planner.seen) < 2 {
		t.Fatalf("planner ran %d turns, want >= 2 (the deny must drive an adaptation turn)", len(planner.seen))
	}
	rc := findNativeReceipt(t, planner.seen[1], "call_del")
	if rc.Status != agent.ToolResultError {
		t.Fatalf("denied receipt status = %q, want %q", rc.Status, agent.ToolResultError)
	}
	if rc.Reason != "POLICY_BLOCK" {
		t.Fatalf("denied receipt reason = %q, want POLICY_BLOCK", rc.Reason)
	}
	if rc.Disposition == "" {
		t.Fatalf("denied receipt carried no disposition")
	}

	// The prose adjudicationNote is provably ABSENT on the native path: no message the
	// owned loop authored carries the "[fak]" splice the proxy path uses.
	for i, msg := range planner.seen[len(planner.seen)-1] {
		if strings.Contains(msg.Content, "[fak]") {
			t.Fatalf("native path message %d carried a prose adjudicationNote: %q", i, msg.Content)
		}
	}
	if strings.Contains(got.Fak.NativeArm.FinalAnswer, "[fak]") {
		t.Fatalf("native final answer carried a prose adjudicationNote: %q", got.Fak.NativeArm.FinalAnswer)
	}
}

// TestNoOpReportsSkipped drives the owned loop (the seam the native serve path runs) with
// the before-consumption write barrier armed: a write behind a squashed speculative read
// is never dispatched and must surface as a typed status=skipped receipt, not a feigned
// success. The native serve path does not wire a speculator, so the not-sent no-op is
// witnessed directly at agent.RunArm — the loop the gateway serves.
func TestNoOpReportsSkipped(t *testing.T) {
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})

	planner := &recordingNativePlanner{turns: []*agent.Completion{
		nativeCallTurn("call_read", "get_user_details", `{"user_id":"u1"}`, 8),
		nativeCallTurn("call_book", "book_flight", `{"flight_id":"f1"}`, 10),
		{Message: agent.Message{Role: agent.RoleAssistant, Content: "done"}, FinishReason: "stop", Usage: agent.Usage{PromptTokens: 12, CompletionTokens: 2}},
	}}
	spec := abi.NewSpeculator(0)
	spec.Learn(abi.SpecPattern{
		Signature:   "get_user_details",
		PredictTool: "search_direct_flight",
		SuccessProb: 1.0,
		Meta:        map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
		DeriveArgs: func([]*abi.Result) (abi.Ref, bool) {
			return abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"origin":"SFO"}`), Len: 16}, true
		},
	})

	m, err := agent.RunArm(context.Background(), planner, "book me a flight", true, 10, nil, agent.WithSpeculator(spec))
	if err != nil {
		t.Fatalf("RunArm (with speculator): %v", err)
	}
	if m.WritesBarred == 0 {
		t.Fatalf("the write behind the squashed speculation was not barred (WritesBarred=0)")
	}
	if len(planner.seen) < 3 {
		t.Fatalf("planner ran %d turns, want >= 3", len(planner.seen))
	}
	rc := findNativeReceipt(t, planner.seen[2], "call_book")
	if rc.Status != agent.ToolResultSkipped {
		t.Fatalf("barred (not-sent) receipt status = %q, want %q", rc.Status, agent.ToolResultSkipped)
	}
	if rc.Reason != "WRITE_BARRED" {
		t.Fatalf("skipped receipt reason = %q, want WRITE_BARRED", rc.Reason)
	}
}
