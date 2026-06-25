package gateway

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// The INBOUND twin of #555: maybeCompactInboundTools prunes tool DEFINITIONS the floor can
// never admit from the Anthropic passthrough's tools[], keeping the cache_control prefix
// byte-identical so the upstream prompt-cache hit survives. These tests exercise the GATING
// + the prune (the cache-safety of the splice itself is proven in internal/promptmmu).

// inboundToolsBody is a realistic /v1/messages body whose tools[] has an early tool carrying
// the cache_control breakpoint (the cached prefix) followed by more tools AFTER it — some
// floor-allowed, some floor-denied — so a prune has something to remove strictly after the
// breakpoint without ever touching the cached head.
func inboundToolsBody(t *testing.T) []byte {
	t.Helper()
	type obj map[string]any
	schema := obj{"type": "object", "properties": obj{}}
	tools := []obj{
		// The cached head: a tool carrying the breakpoint. NEVER droppable (it anchors the prefix).
		{"name": "read_file", "description": strings.Repeat("read ", 20), "input_schema": schema,
			"cache_control": obj{"type": "ephemeral"}},
		// After the breakpoint: a floor-ALLOWED tool (kept) and floor-DENIED tools (dropped).
		{"name": "Bash", "description": strings.Repeat("shell ", 20), "input_schema": schema},
		{"name": "WebFetch", "description": strings.Repeat("fetch the web ", 20), "input_schema": schema},
		{"name": "DeleteEverything", "description": strings.Repeat("danger ", 20), "input_schema": schema},
	}
	raw, err := json.Marshal(obj{
		"model": "claude-sonnet-4-6", "max_tokens": 1024, "stream": true,
		"system": []obj{{"type": "text", "text": "policy"}},
		"tools":  tools,
		"messages": []obj{
			{"role": "user", "content": []obj{{"type": "text", "text": "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// floorAllowing returns a ToolFloorDenies predicate: a tool name is droppable (true) unless
// it is in the allowed set. read_file and Bash are reachable; WebFetch and DeleteEverything
// are floor-denied for every arg, so their definitions may be pruned.
func floorAllowing(allowed ...string) func(string) bool {
	set := map[string]bool{}
	for _, a := range allowed {
		set[a] = true
	}
	return func(name string) bool { return !set[name] }
}

func anthropicServerWithFloor(deny func(string) bool) *Server {
	return &Server{
		planner:         &agent.HTTPPlanner{Provider: agent.ProviderAnthropic},
		toolFloorDenies: deny,
		logf:            func(string, ...any) {},
	}
}

// TestInboundToolsNilPredicateIsIdentity: with no floor predicate the body is unchanged.
func TestInboundToolsNilPredicateIsIdentity(t *testing.T) {
	raw := inboundToolsBody(t)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	s := anthropicServerWithFloor(nil)
	if pruned := s.maybeCompactInboundTools(req); pruned != nil {
		t.Fatalf("nil predicate must prune nothing, got %v", pruned)
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatalf("nil predicate must leave req.Raw unchanged")
	}
}

// TestInboundToolsNonPassthroughIsIdentity: a real floor predicate but a non-Anthropic
// upstream → identity (the body is rebuilt from req downstream; touching req.Raw is pointless).
func TestInboundToolsNonPassthroughIsIdentity(t *testing.T) {
	raw := inboundToolsBody(t)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	s := &Server{
		planner:         agent.NewMockPlanner("m"),
		toolFloorDenies: floorAllowing("read_file", "Bash"),
		logf:            func(string, ...any) {},
	}
	if s.anthropicPassthrough() {
		t.Fatal("mock planner must NOT be an anthropic passthrough")
	}
	if pruned := s.maybeCompactInboundTools(req); pruned != nil {
		t.Fatalf("non-passthrough must prune nothing, got %v", pruned)
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatalf("non-passthrough wire must leave req.Raw unchanged")
	}
}

// TestInboundToolsPrunesDeniedKeepsPrefix: ON + Anthropic passthrough + floor-denied tools
// after the breakpoint → those definitions are pruned, the kept/cached prefix is byte-
// identical, the floor-allowed tool survives, and the body still decodes.
func TestInboundToolsPrunesDeniedKeepsPrefix(t *testing.T) {
	raw := inboundToolsBody(t)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)

	// The cached prefix ends at the close of the LAST tool bearing a cache_control breakpoint
	// (read_file, index 0). Re-derive that boundary with the same span primitive promptmmu uses.
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(orig, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	_, spans, ok := decodeArrayElementsFromTest(t, orig, obj["tools"])
	if !ok || len(spans) == 0 {
		t.Fatal("could not locate tools[] spans")
	}
	prefixEnd := spans[0].end // end of read_file (the breakpoint-bearing head)

	s := anthropicServerWithFloor(floorAllowing("read_file", "Bash"))
	pruned := s.maybeCompactInboundTools(req)

	// WebFetch and DeleteEverything (floor-denied, after the breakpoint) must be removed.
	wantPruned := map[string]bool{"WebFetch": true, "DeleteEverything": true}
	if len(pruned) != len(wantPruned) {
		t.Fatalf("expected %d pruned tools, got %v", len(wantPruned), pruned)
	}
	for _, p := range pruned {
		if !wantPruned[p] {
			t.Fatalf("pruned an unexpected tool %q (must keep floor-allowed + cached-prefix tools)", p)
		}
	}

	if bytes.Equal(req.Raw, orig) {
		t.Fatalf("expected a prune, got identity")
	}
	if len(req.Raw) >= len(orig) {
		t.Fatalf("pruned body must be shorter, got %d >= %d", len(req.Raw), len(orig))
	}
	// The cached prefix (everything through read_file) is byte-identical → upstream cache hit survives.
	if prefixEnd > len(req.Raw) || !bytes.Equal(orig[:prefixEnd], req.Raw[:prefixEnd]) {
		t.Fatalf("cache prefix bytes changed (prefixEnd=%d)", prefixEnd)
	}
	// The result still decodes, and the floor-allowed Bash survives while the denied ones are gone.
	out, err := agent.DecodeAnthropicMessagesRequest(req.Raw)
	if err != nil {
		t.Fatalf("pruned body failed to re-decode: %v", err)
	}
	names := map[string]bool{}
	for _, td := range out.Tools {
		names[td.Function.Name] = true
	}
	if !names["read_file"] || !names["Bash"] {
		t.Fatalf("floor-allowed/cached tools must survive; got %v", names)
	}
	if names["WebFetch"] || names["DeleteEverything"] {
		t.Fatalf("floor-denied tools must be gone; got %v", names)
	}
}

// TestInboundToolsAllAllowedIsIdentity: when every advertised tool is floor-allowed there is
// nothing to drop → identity (a named no-op, not a spurious rewrite).
func TestInboundToolsAllAllowedIsIdentity(t *testing.T) {
	raw := inboundToolsBody(t)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	s := anthropicServerWithFloor(floorAllowing("read_file", "Bash", "WebFetch", "DeleteEverything"))
	if pruned := s.maybeCompactInboundTools(req); pruned != nil {
		t.Fatalf("all-allowed floor must prune nothing, got %v", pruned)
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatalf("all-allowed floor must leave req.Raw unchanged")
	}
}
