package gateway

import (
	"context"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestAnthropicOversizeElideRestoresOriginalToolResult(t *testing.T) {
	const trace = "anthropic-oversize-roundtrip"
	const original = "scrolled past file dump line.\n"
	want := strings.Repeat(original, 400)
	req, err := agent.DecodeAnthropicMessagesRequest(elideWireBody(t))
	if err != nil {
		t.Fatalf("decode request: %v", err)
	}
	srv := anthropicPassthroughElideServer(2048)
	srv.SetDefaultTraceID(trace)
	if !srv.maybeElideAnthropicRaw(req) {
		t.Fatal("oversized tool result was not elided")
	}
	marker := string(req.Raw)
	idMatch := regexp.MustCompile(`fak_context_restore id=(?:sha256:)?([0-9a-f]{64})`).FindStringSubmatch(marker)
	if len(idMatch) != 2 {
		t.Fatalf("elision marker has no restore id: %q", marker)
	}
	if !strings.Contains(marker, `trace_id=\"`+trace+`\"`) {
		t.Fatalf("elision marker has no callable trace id: %q", marker)
	}
	restored, err := srv.restoreContext("", ContextRestoreRequest{ID: idMatch[1], TraceID: trace})
	if err != nil {
		t.Fatalf("restore original: %v", err)
	}
	if restored.Bytes != want {
		t.Fatalf("restored tool result differs: got %d bytes, want %d", len(restored.Bytes), len(want))
	}
}

func TestDecodedOversizeElideRestoresOriginalToolResult(t *testing.T) {
	want := strings.Repeat("older command output line.\n", 400)
	srv := newTestServer(t)
	trace := srv.traceFor("") // simulate a request without X-Trace-Id
	srv.SetDefaultTraceID(trace)
	srv.elideResultBytes = 2048
	messages := []agent.Message{
		{Role: agent.RoleUser, Content: "request"},
		{Role: agent.RoleTool, ToolCallID: "old", Content: want},
		{Role: agent.RoleAssistant, Content: "a"},
		{Role: agent.RoleUser, Content: "u"},
		{Role: agent.RoleAssistant, Content: "b"},
		{Role: agent.RoleUser, Content: "c"},
		{Role: agent.RoleAssistant, Content: "d"},
	}
	elided := srv.maybeElideMessagesWithContext(context.Background(), messages)
	marker := elided[1].Content
	idMatch := regexp.MustCompile(`fak_context_restore id=(?:sha256:)?([0-9a-f]{64})`).FindStringSubmatch(marker)
	if len(idMatch) != 2 {
		t.Fatalf("decoded elision marker has no restore id: %q", marker)
	}
	traceMatch := regexp.MustCompile(`trace_id=("(?:\\.|[^"\\])*")`).FindStringSubmatch(marker)
	if len(traceMatch) != 2 {
		t.Fatalf("decoded elision marker has no callable trace id: %q", marker)
	}
	traceFromMarker, err := strconv.Unquote(traceMatch[1])
	if err != nil || traceFromMarker != trace {
		t.Fatalf("trace id in marker=%q, want %q: %v", traceFromMarker, trace, err)
	}
	restored, err := srv.restoreContext("", ContextRestoreRequest{ID: idMatch[1], TraceID: traceFromMarker})
	if err != nil {
		t.Fatalf("restore original: %v", err)
	}
	if restored.Bytes != want {
		t.Fatalf("restored tool result differs: got %d bytes, want %d", len(restored.Bytes), len(want))
	}
}
