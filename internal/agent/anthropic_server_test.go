package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// A Claude-Code-shaped request: system as a block array, a prior assistant turn
// with a tool_use, and a user turn carrying the matching tool_result.
const ccRequest = `{
  "model": "claude-opus-4-8",
  "max_tokens": 4096,
  "temperature": 1,
  "system": [{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral"}}],
  "tools": [{"name":"Bash","description":"run a command","input_schema":{"type":"object","properties":{"command":{"type":"string"}}}}],
  "messages": [
    {"role":"user","content":"list the files"},
    {"role":"assistant","content":[
      {"type":"text","text":"I'll list them."},
      {"type":"tool_use","id":"toolu_01ABC","name":"Bash","input":{"command":"ls"}}
    ]},
    {"role":"user","content":[
      {"type":"tool_result","tool_use_id":"toolu_01ABC","content":[{"type":"text","text":"a.go\nb.go"}]}
    ]}
  ],
  "stream": true
}`

func TestDecodeAnthropicMessagesRequest(t *testing.T) {
	req, err := DecodeAnthropicMessagesRequest([]byte(ccRequest))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if req.Model != "claude-opus-4-8" {
		t.Errorf("model = %q", req.Model)
	}
	if !req.Stream {
		t.Errorf("stream not parsed")
	}
	if req.MaxTokens != 4096 {
		t.Errorf("max_tokens = %d", req.MaxTokens)
	}
	if req.System != "You are a coding agent." {
		t.Errorf("system block array not folded: %q", req.System)
	}
	// system (prepended) + user + assistant + tool_result = 4 canonical messages.
	if len(req.Messages) != 4 {
		t.Fatalf("messages = %d, want 4: %+v", len(req.Messages), req.Messages)
	}
	if req.Messages[0].Role != RoleSystem || req.Messages[0].Content != "You are a coding agent." {
		t.Errorf("messages[0] not system: %+v", req.Messages[0])
	}
	if req.Messages[1].Role != RoleUser || req.Messages[1].Content != "list the files" {
		t.Errorf("messages[1] (string content) wrong: %+v", req.Messages[1])
	}
	asst := req.Messages[2]
	if asst.Role != RoleAssistant || asst.Content != "I'll list them." {
		t.Errorf("assistant text wrong: %+v", asst)
	}
	if len(asst.ToolCalls) != 1 || asst.ToolCalls[0].ID != "toolu_01ABC" || asst.ToolCalls[0].Function.Name != "Bash" {
		t.Fatalf("assistant tool_use not decoded (id must survive): %+v", asst.ToolCalls)
	}
	if !strings.Contains(asst.ToolCalls[0].Function.Arguments, `"command":"ls"`) {
		t.Errorf("tool_use input not kept as raw args: %q", asst.ToolCalls[0].Function.Arguments)
	}
	tr := req.Messages[3]
	if tr.Role != RoleTool || tr.ToolCallID != "toolu_01ABC" || tr.Content != "a.go\nb.go" {
		t.Errorf("tool_result not mapped to RoleTool keyed by id: %+v", tr)
	}
	if len(req.Tools) != 1 || req.Tools[0].Function.Name != "Bash" {
		t.Fatalf("tools not mapped: %+v", req.Tools)
	}
	if !strings.Contains(string(req.Tools[0].Function.Parameters), `"command"`) {
		t.Errorf("input_schema not passed through: %s", req.Tools[0].Function.Parameters)
	}
	// Raw must be the ORIGINAL inbound bytes, byte-for-byte. This is what byte-exact
	// Anthropic passthrough forwards upstream so the client's cache_control prefix
	// survives — any divergence here is a guaranteed prompt-cache miss.
	if string(req.Raw) != ccRequest {
		t.Errorf("Raw not retained verbatim:\n got %q\nwant %q", req.Raw, ccRequest)
	}
}

// TestDecodeAnthropicSamplingParams pins that the inbound /v1/messages decode
// carries the per-request sampling controls the Anthropic Messages API defines —
// in particular top_k, which the in-kernel sampler honors but the inbound wire
// silently dropped before it was parsed here. top_p is checked alongside it so the
// two pointer params stay symmetric; an omitted control must decode to nil (the
// no-op default, byte-for-byte the pre-seam draw).
func TestDecodeAnthropicSamplingParams(t *testing.T) {
	req, err := DecodeAnthropicMessagesRequest([]byte(
		`{"model":"m","max_tokens":64,"top_k":40,"top_p":0.9,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if req.TopK == nil {
		t.Fatalf("top_k not parsed off the inbound wire (silently dropped)")
	}
	if *req.TopK != 40 {
		t.Errorf("top_k = %d, want 40", *req.TopK)
	}
	if req.TopP == nil || *req.TopP != 0.9 {
		t.Errorf("top_p = %v, want 0.9", req.TopP)
	}
}

func TestDecodeAnthropicTopKOmittedIsNil(t *testing.T) {
	// An omitted top_k must leave TopK nil so the planner keeps the full distribution
	// (WithTopK(nil) is a no-op) — identical to the pre-seam path.
	req, err := DecodeAnthropicMessagesRequest([]byte(
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if req.TopK != nil {
		t.Errorf("omitted top_k must decode to nil, got %v", *req.TopK)
	}
}

func TestDecodeAnthropicSystemString(t *testing.T) {
	req, err := DecodeAnthropicMessagesRequest([]byte(`{"model":"m","system":"plain","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if req.System != "plain" {
		t.Errorf("string system = %q", req.System)
	}
	if len(req.Messages) != 2 || req.Messages[0].Role != RoleSystem {
		t.Errorf("system message not prepended: %+v", req.Messages)
	}
}

func TestDecodeAnthropicParallelToolResults(t *testing.T) {
	// A single user turn carrying two tool_result blocks must fan out into two
	// RoleTool messages (parallel tool calls).
	body := `{"model":"m","messages":[{"role":"user","content":[
		{"type":"tool_result","tool_use_id":"t1","content":"r1"},
		{"type":"tool_result","tool_use_id":"t2","content":"r2"}
	]}]}`
	req, err := DecodeAnthropicMessagesRequest([]byte(body))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(req.Messages) != 2 {
		t.Fatalf("want 2 tool messages, got %d: %+v", len(req.Messages), req.Messages)
	}
	if req.Messages[0].ToolCallID != "t1" || req.Messages[1].ToolCallID != "t2" {
		t.Errorf("tool_result ids/order wrong: %+v", req.Messages)
	}
}

func TestAnthropicResponseBlocks(t *testing.T) {
	m := Message{Role: RoleAssistant, Content: "done", ToolCalls: []ToolCall{
		{ID: "tu1", Function: Func{Name: "Read", Arguments: `{"path":"x"}`}},
		{ID: "tu2", Function: Func{Name: "Bad", Arguments: `not json`}},
	}}
	blocks := AnthropicResponseBlocks(m)
	if len(blocks) != 3 {
		t.Fatalf("want text + 2 tool_use blocks, got %d: %+v", len(blocks), blocks)
	}
	if blocks[0].Type != "text" || blocks[0].Text != "done" {
		t.Errorf("block0 not text: %+v", blocks[0])
	}
	if blocks[1].Type != "tool_use" || blocks[1].ID != "tu1" || string(blocks[1].Input) != `{"path":"x"}` {
		t.Errorf("block1 tool_use wrong: %+v", blocks[1])
	}
	// A non-object argument must still render a well-formed (empty) input object.
	if string(blocks[2].Input) != "{}" {
		t.Errorf("malformed args must normalize to {}: %q", blocks[2].Input)
	}
	// Each input must be valid JSON (Claude Code parses it).
	for _, b := range blocks[1:] {
		var v any
		if err := json.Unmarshal(b.Input, &v); err != nil {
			t.Errorf("tool_use input not valid JSON: %q", b.Input)
		}
	}
}

func TestAnthropicStopReason(t *testing.T) {
	cases := []struct {
		finish string
		tools  bool
		want   string
	}{
		{"stop", false, "end_turn"},
		{"tool_calls", true, "tool_use"},
		{"end_turn", false, "end_turn"},
		{"length", false, "max_tokens"},
		{"anything", true, "tool_use"}, // surviving tool always wins
	}
	for _, c := range cases {
		if got := AnthropicStopReason(c.finish, c.tools); got != c.want {
			t.Errorf("AnthropicStopReason(%q,%v) = %q, want %q", c.finish, c.tools, got, c.want)
		}
	}
}

func TestEstimateAnthropicTokens(t *testing.T) {
	req, _ := DecodeAnthropicMessagesRequest([]byte(ccRequest))
	if n := EstimateAnthropicTokens(req); n <= 0 {
		t.Errorf("token estimate must be positive, got %d", n)
	}
}
