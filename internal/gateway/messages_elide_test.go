package gateway

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// These tests exercise the GATING of the oversized-tool_result elision transform on the gateway
// (the cache-safety of the rewrite itself is proven in internal/agent/anthropic_elide_test.go):
//   - OFF (threshold 0) is identity.
//   - a non-passthrough wire is identity even with a threshold set (the body is rebuilt downstream).
//   - ON + Anthropic passthrough shrinks an oversized old result but keeps the prefix verbatim.

// elideWireBody is a /v1/messages body with a cached head breakpoint plus an oversized old
// tool_result in the un-cached, non-recent middle that a positive threshold will shrink.
func elideWireBody(t *testing.T) []byte {
	t.Helper()
	type obj = map[string]any
	cc := obj{"type": "ephemeral"}
	big := strings.Repeat("scrolled past file dump line.\n", 400) // ~12 KB, well over the test threshold
	raw, err := json.Marshal(obj{
		"model": "claude-sonnet-4-6", "max_tokens": 1024, "stream": true,
		"system": []obj{{"type": "text", "text": "policy", "cache_control": cc}},
		"messages": []obj{
			{"role": "user", "content": []obj{{"type": "text", "text": "cached head", "cache_control": cc}}},                                  // 0 breakpoint
			{"role": "assistant", "content": []obj{{"type": "text", "text": "a1"}}},                                                           // 1
			{"role": "user", "content": []obj{{"type": "tool_result", "tool_use_id": "t2", "content": []obj{{"type": "text", "text": big}}}}}, // 2 ELIGIBLE
			{"role": "assistant", "content": []obj{{"type": "text", "text": "a3"}}},                                                           // 3
			{"role": "user", "content": []obj{{"type": "text", "text": "u4"}}},                                                                // 4
			{"role": "assistant", "content": []obj{{"type": "text", "text": "a5"}}},                                                           // 5
			{"role": "user", "content": []obj{{"type": "text", "text": "u6"}}},                                                                // 6
		},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

func anthropicPassthroughElideServer(threshold int) *Server {
	return &Server{
		planner:          &agent.HTTPPlanner{Provider: agent.ProviderAnthropic},
		elideResultBytes: threshold,
		logf:             func(string, ...any) {},
	}
}

func TestMaybeElideOffIsIdentity(t *testing.T) {
	req, err := agent.DecodeAnthropicMessagesRequest(elideWireBody(t))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	if anthropicPassthroughElideServer(0).maybeElideAnthropicRaw(req) {
		t.Fatal("threshold 0 must not fire")
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatal("threshold 0 must leave req.Raw unchanged")
	}
}

func TestMaybeElideNonPassthroughIsIdentity(t *testing.T) {
	req, err := agent.DecodeAnthropicMessagesRequest(elideWireBody(t))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	s := &Server{planner: agent.NewMockPlanner("m"), elideResultBytes: 2048, logf: func(string, ...any) {}}
	if s.anthropicPassthrough() {
		t.Fatal("mock planner must NOT be an anthropic passthrough")
	}
	s.maybeElideAnthropicRaw(req)
	if !bytes.Equal(req.Raw, orig) {
		t.Fatal("non-passthrough wire must leave req.Raw unchanged")
	}
}

// TestMaybeElideMessagesLocalModelPath proves the decoded-path elision is enabled for a LOCAL
// model served by fak (GLM-5.2 / Qwen-3.6-27B via an OpenAI backend or the in-kernel engine):
// the non-passthrough wire shrinks an old oversized tool result, the Anthropic passthrough does
// NOT (it is handled on req.Raw), and threshold 0 is identity.
func TestMaybeElideMessagesLocalModelPath(t *testing.T) {
	bigOld := strings.Repeat("scrolled-past command output line. ", 200) // ~7 KB, old → eligible
	msgs := []agent.Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "q"},
		{Role: "tool", ToolCallID: "t2", Content: bigOld}, // idx 2, old → shrink on the local path
		{Role: "assistant", Content: "a"},
		{Role: "user", Content: "u"},
		{Role: "assistant", Content: "b"},
		{Role: "user", Content: "c"},
		{Role: "assistant", Content: "d"},
	}
	// Non-passthrough (a mock planner stands in for a LOCAL GLM/Qwen model): elision fires.
	local := &Server{planner: agent.NewMockPlanner("glm-5.2"), elideResultBytes: 2048, logf: func(string, ...any) {}}
	if local.anthropicPassthrough() {
		t.Fatal("mock planner must NOT be an anthropic passthrough")
	}
	if got := local.maybeElideMessages(msgs); len(got[2].Content) >= len(bigOld) {
		t.Errorf("local-model (GLM/Qwen) path did not shrink the old oversized tool result: len=%d", len(got[2].Content))
	}
	// Anthropic passthrough: a no-op here (handled on req.Raw by maybeElideAnthropicRaw).
	pass := &Server{planner: &agent.HTTPPlanner{Provider: agent.ProviderAnthropic}, elideResultBytes: 2048, logf: func(string, ...any) {}}
	if !pass.anthropicPassthrough() {
		t.Fatal("expected an anthropic passthrough server")
	}
	if got := pass.maybeElideMessages(msgs); got[2].Content != bigOld {
		t.Error("passthrough path must NOT elide req.Messages (the kernel adjudicates full history; req.Raw is elided instead)")
	}
	// OFF (threshold 0) is identity on the local path too.
	off := &Server{planner: agent.NewMockPlanner("m"), elideResultBytes: 0, logf: func(string, ...any) {}}
	if got := off.maybeElideMessages(msgs); got[2].Content != bigOld {
		t.Error("threshold 0 must leave messages unchanged")
	}
}

func TestMaybeElideOnShrinksKeepsPrefix(t *testing.T) {
	req, err := agent.DecodeAnthropicMessagesRequest(elideWireBody(t))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)

	// Prefix boundary: the fixture's first cache_control breakpoint is on message[0], so the
	// protected prefix ends at message[0]'s end.
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(orig, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	_, spans, ok := decodeArrayElementsFromTest(t, orig, obj["messages"])
	if !ok {
		t.Fatal("decodeArrayElements failed")
	}
	prefixEnd := spans[0].end

	if !anthropicPassthroughElideServer(2048).maybeElideAnthropicRaw(req) {
		t.Fatal("expected elision to FIRE on an oversized old result at threshold 2048")
	}
	if len(req.Raw) >= len(orig) {
		t.Fatalf("expected a shorter body, got %d >= %d", len(req.Raw), len(orig))
	}
	if !bytes.Equal(orig[:prefixEnd], req.Raw[:prefixEnd]) {
		t.Fatal("protected prefix bytes changed — cache hit would be lost")
	}
	if _, err := agent.DecodeAnthropicMessagesRequest(req.Raw); err != nil {
		t.Fatalf("rewritten body failed to re-decode: %v", err)
	}
}
