package gateway

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// These tests exercise the GATING and the restore round-trip of the read-lifecycle STALE elision on
// the gateway (the cache-safety of the rewrite itself is proven in
// internal/agent/anthropic_elide_stale_test.go):
//   - OFF (flag false) is identity.
//   - a non-passthrough wire is identity even with the flag on (the body is rebuilt downstream).
//   - ON + Anthropic passthrough replaces a superseded Read with a marker, keeps the prefix verbatim,
//     and stashes the original so fak_context_restore(id) returns it byte-for-byte.

// staleReadWireBody is a /v1/messages body with a cached-head breakpoint, a Read of x.go in the
// un-cached middle, and a LATER Edit of x.go that supersedes it. The read body is large so the
// marker is strictly shorter (the transform never grows the body).
func staleReadWireBody(t *testing.T) []byte {
	t.Helper()
	type obj = map[string]any
	cc := obj{"type": "ephemeral"}
	big := strings.Repeat("pre-edit body of x.go line.\n", 400) // ~11 KB
	raw, err := json.Marshal(obj{
		"model": "claude-sonnet-4-6", "max_tokens": 1024, "stream": true,
		"system": []obj{{"type": "text", "text": "policy", "cache_control": cc}},
		"messages": []obj{
			{"role": "user", "content": []obj{{"type": "text", "text": "cached head", "cache_control": cc}}},                                       // 0 breakpoint
			{"role": "assistant", "content": []obj{{"type": "tool_use", "id": "rX", "name": "Read", "input": obj{"file_path": "/repo/x.go"}}}},     // 1
			{"role": "user", "content": []obj{{"type": "tool_result", "tool_use_id": "rX", "content": []obj{{"type": "text", "text": big}}}}},      // 2 ELIGIBLE + STALE
			{"role": "assistant", "content": []obj{{"type": "tool_use", "id": "eX", "name": "Edit", "input": obj{"file_path": "/repo/x.go"}}}},     // 3 supersedes rX
			{"role": "user", "content": []obj{{"type": "tool_result", "tool_use_id": "eX", "content": []obj{{"type": "text", "text": "edited"}}}}}, // 4
			{"role": "assistant", "content": []obj{{"type": "text", "text": "a5"}}},                                                                // 5
			{"role": "user", "content": []obj{{"type": "text", "text": "u6"}}},                                                                     // 6
			{"role": "assistant", "content": []obj{{"type": "text", "text": "a7"}}},                                                                // 7
			{"role": "user", "content": []obj{{"type": "text", "text": "u8"}}},                                                                     // 8
		},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

func staleReadServer(on bool) *Server {
	return &Server{
		planner:         &agent.HTTPPlanner{Provider: agent.ProviderAnthropic},
		elideStaleReads: on,
		logf:            func(string, ...any) {},
		metrics:         newGatewayMetrics(time.Now()),
	}
}

func TestMaybeElideStaleReadsOffIsIdentity(t *testing.T) {
	req, err := agent.DecodeAnthropicMessagesRequest(staleReadWireBody(t))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	if staleReadServer(false).maybeElideStaleReads(req, "trace-1") {
		t.Fatal("flag off must not fire")
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatal("flag off must leave req.Raw unchanged")
	}
}

func TestMaybeElideStaleReadsNonPassthroughIsIdentity(t *testing.T) {
	req, err := agent.DecodeAnthropicMessagesRequest(staleReadWireBody(t))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	s := &Server{planner: agent.NewMockPlanner("m"), elideStaleReads: true, logf: func(string, ...any) {}}
	if s.anthropicPassthrough() {
		t.Fatal("mock planner must NOT be an anthropic passthrough")
	}
	if s.maybeElideStaleReads(req, "trace-1") {
		t.Fatal("non-passthrough wire must not fire")
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatal("non-passthrough wire must leave req.Raw unchanged")
	}
}

// TestMaybeElideStaleReadsRoundTrip is the end-to-end witness: ON + passthrough shrinks the
// superseded read to a marker, preserves the cached prefix, and stashes the original so a
// fak_context_restore(id) under the same trace pages back the exact pre-edit body.
func TestMaybeElideStaleReadsRoundTrip(t *testing.T) {
	req, err := agent.DecodeAnthropicMessagesRequest(staleReadWireBody(t))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)

	// Prefix boundary: the fixture's first cache_control breakpoint is message[0].
	var o map[string]json.RawMessage
	if err := json.Unmarshal(orig, &o); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	_, spans, ok := decodeArrayElementsFromTest(t, orig, o["messages"])
	if !ok {
		t.Fatal("decodeArrayElements failed")
	}
	prefixEnd := spans[0].end

	const trace = "trace-round-trip"
	s := staleReadServer(true)
	if !s.maybeElideStaleReads(req, trace) {
		t.Fatal("expected the superseded read to be elided")
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

	// The marker carries an id=<hash> handle; extract it and restore under the same trace.
	id := extractRestoreID(t, req.Raw)
	got, err := s.restoreContext("", ContextRestoreRequest{ID: id, TraceID: trace})
	if err != nil {
		t.Fatalf("restoreContext(%q) failed: %v", id, err)
	}
	big := strings.Repeat("pre-edit body of x.go line.\n", 400)
	if got.Bytes != big {
		t.Errorf("restored bytes are not the original pre-edit read body (len %d vs %d)", len(got.Bytes), len(big))
	}
}

// extractRestoreID pulls the id=<hex> handle out of the stale-read marker in the rewritten body.
func extractRestoreID(t *testing.T, raw []byte) string {
	t.Helper()
	const key = "id="
	i := bytes.Index(raw, []byte("superseded by a later in-session edit"))
	if i < 0 {
		t.Fatal("stale marker not found in rewritten body")
	}
	j := bytes.Index(raw[i:], []byte(key))
	if j < 0 {
		t.Fatal("id= handle not found in marker")
	}
	start := i + j + len(key)
	end := start
	for end < len(raw) && isHexByte(raw[end]) {
		end++
	}
	if end == start {
		t.Fatal("empty restore id")
	}
	return string(raw[start:end])
}

func isHexByte(b byte) bool {
	return (b >= '0' && b <= '9') || (b >= 'a' && b <= 'f') || (b >= 'A' && b <= 'F')
}
