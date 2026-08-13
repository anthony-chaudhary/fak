package gateway

// native_wire_test.go — the acceptance witness for #6657: what the client actually sent
// survives the native wire seam. Before this, `serve --native` reduced messages[] to
// lastUserText(req.Messages) and dropped req.Tools on the floor, so the owned loop ran the
// airline fixture against a one-line reconstruction of the conversation. These tests drive
// the REAL seam (POST /v1/messages against a `Native: true` gateway) and capture what
// reached the planner at the agent.RunArm boundary — the ordered transcript and the
// caller-declared tool schemas — for both the buffered and the streamed handler.

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// nativeWireRecorder captures every (messages, tools) pair the owned loop lowers into the
// model call, then ends the turn with a final answer so the loop runs exactly once.
type nativeWireRecorder struct {
	mu       sync.Mutex
	messages [][]agent.Message
	tools    [][]agent.ToolDef
}

func (p *nativeWireRecorder) Model() string { return "record-model" }

func (p *nativeWireRecorder) record(messages []agent.Message, tools []agent.ToolDef) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.messages = append(p.messages, append([]agent.Message(nil), messages...))
	p.tools = append(p.tools, append([]agent.ToolDef(nil), tools...))
}

func (p *nativeWireRecorder) firstTurn(t *testing.T) ([]agent.Message, []agent.ToolDef) {
	t.Helper()
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.messages) == 0 {
		t.Fatalf("the planner was never called — the owned loop never drove a turn")
	}
	return p.messages[0], p.tools[0]
}

func (p *nativeWireRecorder) calls() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.messages)
}

func (p *nativeWireRecorder) Complete(_ context.Context, messages []agent.Message, tools []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.record(messages, tools)
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "done"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 7, CompletionTokens: 1},
	}, nil
}

func (p *nativeWireRecorder) StreamingSupported() bool { return true }

func (p *nativeWireRecorder) CompleteStream(_ context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.record(messages, tools)
	if sink != nil {
		if err := sink("done"); err != nil {
			return nil, err
		}
	}
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "done"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 7, CompletionTokens: 1},
	}, nil
}

// newNativeWireServer builds a `serve --native` gateway whose planner records the
// RunArm boundary, and returns the live test server plus that recorder.
func newNativeWireServer(t *testing.T) (*httptest.Server, *nativeWireRecorder) {
	t.Helper()
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})

	srv, err := New(Config{
		EngineID:       "localtools",
		Model:          "record-model",
		VDSO:           true,
		Native:         true,
		NativeMaxTurns: 4,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	rec := &nativeWireRecorder{}
	srv.planner = rec
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts, rec
}

func postNativeMessages(t *testing.T, ts *httptest.Server, body map[string]any) (int, []byte) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/messages", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", "native-wire-trace")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/messages: %v", err)
	}
	defer resp.Body.Close()
	got, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, got
}

// wireConversation is the three-turn transcript every history assertion below uses: a
// prior user turn, the assistant's reply, and the live user turn. Only the last one
// survived the old lastUserText() seam.
func wireConversation() []map[string]any {
	return []map[string]any{
		{"role": "user", "content": "what is the order status for A-1?"},
		{"role": "assistant", "content": "A-1 shipped on Tuesday."},
		{"role": "user", "content": "and A-2?"},
	}
}

// wireTools is one caller-declared tool with a real JSON Schema — the request-scoped
// catalog that must reach the loop instead of the airline fixture.
func wireTools() []map[string]any {
	return []map[string]any{{
		"name":        "lookup_order",
		"description": "look up an order by id",
		"input_schema": map[string]any{
			"type":       "object",
			"properties": map[string]any{"order_id": map[string]any{"type": "string"}},
			"required":   []string{"order_id"},
		},
	}}
}

// assertConversationSurvived proves the ordered transcript reached the model call: the
// loop's own system prompt leads, then the caller's turns in order with their roles and
// content intact.
func assertConversationSurvived(t *testing.T, got []agent.Message) {
	t.Helper()
	if len(got) == 0 || got[0].Role != agent.RoleSystem {
		t.Fatalf("first seeded message = %+v, want the loop's system prompt", got)
	}
	want := []agent.Message{
		{Role: agent.RoleUser, Content: "what is the order status for A-1?"},
		{Role: agent.RoleAssistant, Content: "A-1 shipped on Tuesday."},
		{Role: agent.RoleUser, Content: "and A-2?"},
	}
	rest := got[1:]
	if len(rest) != len(want) {
		t.Fatalf("seeded %d caller messages, want %d (the full ordered conversation, not lastUserText); got=%+v", len(rest), len(want), rest)
	}
	for i, w := range want {
		if rest[i].Role != w.Role || rest[i].Content != w.Content {
			t.Errorf("message[%d] = {role:%q content:%q}, want {role:%q content:%q}", i, rest[i].Role, rest[i].Content, w.Role, w.Content)
		}
	}
}

// assertRequestToolCatalog proves the caller's declaration — not the airline fixture —
// is what the loop advertised, schema included.
func assertRequestToolCatalog(t *testing.T, got []agent.ToolDef) {
	t.Helper()
	if len(got) != 1 {
		names := make([]string, 0, len(got))
		for _, td := range got {
			names = append(names, td.Function.Name)
		}
		t.Fatalf("loop advertised %d tools %v, want exactly the 1 caller-declared tool (the airline fixture must not be substituted)", len(got), names)
	}
	fn := got[0].Function
	if fn.Name != "lookup_order" {
		t.Fatalf("tool name = %q, want %q", fn.Name, "lookup_order")
	}
	if fn.Description != "look up an order by id" {
		t.Errorf("tool description = %q, want the caller's", fn.Description)
	}
	var schema struct {
		Type       string         `json:"type"`
		Properties map[string]any `json:"properties"`
		Required   []string       `json:"required"`
	}
	if err := json.Unmarshal(fn.Parameters, &schema); err != nil {
		t.Fatalf("tool parameters are not the caller's JSON Schema: %v (raw=%s)", err, fn.Parameters)
	}
	if schema.Type != "object" {
		t.Errorf("schema type = %q, want %q", schema.Type, "object")
	}
	if _, ok := schema.Properties["order_id"]; !ok {
		t.Errorf("schema lost the order_id property: %s", fn.Parameters)
	}
	if len(schema.Required) != 1 || schema.Required[0] != "order_id" {
		t.Errorf("schema required = %v, want [order_id]", schema.Required)
	}
}

// TestNativeServePreservesConversationAndRequestTools is the buffered-handler witness.
func TestNativeServePreservesConversationAndRequestTools(t *testing.T) {
	ts, rec := newNativeWireServer(t)
	status, body := postNativeMessages(t, ts, map[string]any{
		"model":      "record-model",
		"max_tokens": 64,
		"messages":   wireConversation(),
		"tools":      wireTools(),
	})
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", status, body)
	}
	msgs, tools := rec.firstTurn(t)
	assertConversationSurvived(t, msgs)
	assertRequestToolCatalog(t, tools)
}

// TestNativeServeStreamPreservesConversationAndRequestTools is the streamed twin: the
// two handlers must share one conversion, so the same evidence holds on the SSE path.
func TestNativeServeStreamPreservesConversationAndRequestTools(t *testing.T) {
	ts, rec := newNativeWireServer(t)
	status, body := postNativeMessages(t, ts, map[string]any{
		"model":      "record-model",
		"max_tokens": 64,
		"stream":     true,
		"messages":   wireConversation(),
		"tools":      wireTools(),
	})
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", status, body)
	}
	msgs, tools := rec.firstTurn(t)
	assertConversationSurvived(t, msgs)
	assertRequestToolCatalog(t, tools)
}

// TestNativeServeNoToolsKeepsKernelCatalog is the explicit compatibility witness: a
// request that declares NO tools still gets the kernel-owned catalog, so the historical
// no-tools native run is unchanged — while its conversation is now preserved.
func TestNativeServeNoToolsKeepsKernelCatalog(t *testing.T) {
	ts, rec := newNativeWireServer(t)
	status, body := postNativeMessages(t, ts, map[string]any{
		"model":      "record-model",
		"max_tokens": 64,
		"messages":   wireConversation(),
	})
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", status, body)
	}
	msgs, tools := rec.firstTurn(t)
	assertConversationSurvived(t, msgs)
	want := agent.ToolCatalog()
	if len(tools) != len(want) {
		t.Fatalf("no-tools request advertised %d tools, want the kernel catalog's %d", len(tools), len(want))
	}
	for i := range want {
		if tools[i].Function.Name != want[i].Function.Name {
			t.Fatalf("tool[%d] = %q, want the kernel catalog's %q", i, tools[i].Function.Name, want[i].Function.Name)
		}
	}
}

// TestNativeServeMalformedToolFailsClosed proves P3: a tool declaration the seam cannot
// honor is refused with a typed reason BEFORE the loop runs — never silently dropped and
// never quietly replaced by the fixture.
func TestNativeServeMalformedToolFailsClosed(t *testing.T) {
	cases := []struct {
		name   string
		tools  []map[string]any
		reason string
	}{
		{
			name:   "missing name",
			tools:  []map[string]any{{"description": "no name", "input_schema": map[string]any{"type": "object"}}},
			reason: nativeReasonToolNameMissing,
		},
		{
			name: "duplicate name",
			tools: []map[string]any{
				{"name": "dup", "input_schema": map[string]any{"type": "object"}},
				{"name": "dup", "input_schema": map[string]any{"type": "object"}},
			},
			reason: nativeReasonToolNameDuplicate,
		},
		{
			name:   "schema is not an object",
			tools:  []map[string]any{{"name": "bad_schema", "input_schema": []any{"not", "an", "object"}}},
			reason: nativeReasonToolSchemaInvalid,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ts, rec := newNativeWireServer(t)
			status, body := postNativeMessages(t, ts, map[string]any{
				"model":      "record-model",
				"max_tokens": 64,
				"messages":   wireConversation(),
				"tools":      tc.tools,
			})
			if status != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400 (fail closed); body=%s", status, body)
			}
			var got struct {
				Error struct {
					Code    string `json:"code"`
					Message string `json:"message"`
				} `json:"error"`
			}
			if err := json.Unmarshal(body, &got); err != nil {
				t.Fatalf("decode error body: %v; body=%s", err, body)
			}
			if got.Error.Code != tc.reason {
				t.Errorf("error code = %q, want the closed reason %q; body=%s", got.Error.Code, tc.reason, body)
			}
			if n := rec.calls(); n != 0 {
				t.Errorf("planner was called %d times on a refused declaration — the loop must never run", n)
			}
		})
	}
}

// TestNativeServeStreamMalformedToolFailsClosed proves the streamed handler refuses on
// the same conversion, with a real 400 body rather than an SSE error frame after a 200.
func TestNativeServeStreamMalformedToolFailsClosed(t *testing.T) {
	ts, rec := newNativeWireServer(t)
	status, body := postNativeMessages(t, ts, map[string]any{
		"model":      "record-model",
		"max_tokens": 64,
		"stream":     true,
		"messages":   wireConversation(),
		"tools":      []map[string]any{{"description": "no name"}},
	})
	if status != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (fail closed before the stream opens); body=%s", status, body)
	}
	if n := rec.calls(); n != 0 {
		t.Errorf("planner was called %d times on a refused declaration — the loop must never run", n)
	}
}
