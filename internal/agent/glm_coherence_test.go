package agent

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// glm_coherence_test.go — witnesses for the message<->segment coherence bridge.

func TestSegmentsFromMessagesMapsRolesAndAttachesWitness(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "you are an agent with tools and rules here"},
		{Role: "user", Content: "read foo.go"},
		{Role: "tool", Content: "contents of foo.go", ToolCallID: "call_1"},
		{Role: "assistant", Content: "done"},
	}
	witnessOf := func(id string) string {
		if id == "call_1" {
			return "git_sha:foo.go"
		}
		return ""
	}
	segs := SegmentsFromMessages(msgs, witnessOf)
	if len(segs) != 4 {
		t.Fatalf("got %d segments, want 4", len(segs))
	}
	if segs[0].Kind != cachemeta.SegStable {
		t.Fatalf("system -> %q, want stable", segs[0].Kind)
	}
	if segs[1].Kind != cachemeta.SegMessage || segs[3].Kind != cachemeta.SegMessage {
		t.Fatalf("user/assistant should map to message")
	}
	if segs[2].Kind != cachemeta.SegToolResult {
		t.Fatalf("tool -> %q, want tool_result", segs[2].Kind)
	}
	if segs[2].Witness != "git_sha:foo.go" {
		t.Fatalf("tool segment witness = %q, want git_sha:foo.go", segs[2].Witness)
	}
	if segs[0].Witness != "" {
		t.Fatalf("non-tool segment should carry no witness, got %q", segs[0].Witness)
	}
	for i, s := range segs {
		if s.Tokens <= 0 {
			t.Fatalf("segment %d has non-positive token estimate %d", i, s.Tokens)
		}
	}
}

func TestSegmentsFromMessagesNilWitnessOfIsAnalysisOnly(t *testing.T) {
	segs := SegmentsFromMessages([]Message{{Role: "tool", Content: "x", ToolCallID: "c"}}, nil)
	if segs[0].Witness != "" {
		t.Fatalf("nil witnessOf must leave segments un-witnessed, got %q", segs[0].Witness)
	}
}

// End-to-end: the bridge feeds the segment-level live seam, and a revoked tool witness
// breaks the GLM prefix at that tool segment — the whole message->segment->revoke->shape
// flow, minus only the live request-path re-emit.
func TestSegmentsFromMessagesDriveSegmentWitnessedShaping(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "system prompt with enough bytes to register tokens"},
		{Role: "tool", Content: "doc1 body bytes here", ToolCallID: "call_doc1"},
	}
	witnessOf := func(id string) string {
		if id == "call_doc1" {
			return "blob:doc1"
		}
		return ""
	}
	segs := SegmentsFromMessages(msgs, witnessOf)
	// doc1's blob was revoked -> break at the tool segment.
	shaped, dir, _ := cachemeta.ShapeGLMTurnSegmentWitnessed(segs, func(w string) bool { return w == "blob:doc1" })
	if !dir.Break {
		t.Fatalf("revoked tool witness but no break")
	}
	// The shaped turn has the marker inserted ahead of the (stale) tool segment.
	if len(shaped) != len(segs)+1 {
		t.Fatalf("shaped len %d, want %d (marker inserted)", len(shaped), len(segs)+1)
	}
	if div := cachemeta.Diverge(segs, shaped); div.FirstDivergeSeg != 1 {
		t.Fatalf("shaped turn should break at the marker before the tool segment, got seg %d", div.FirstDivergeSeg)
	}
}

// TestHTTPPlannerCoherenceShaperHook proves the §A4 request-side seam: when set, the
// shaper transforms the outbound messages before send; nil leaves the path unchanged;
// the caller's slice is never mutated.
func TestHTTPPlannerCoherenceShaperHook(t *testing.T) {
	var gotBody []byte
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	defer ts.Close()

	planner, err := NewProviderHTTPPlanner("openai", ts.URL, "glm-5.2", "k")
	if err != nil {
		t.Fatal(err)
	}
	msgs := []Message{{Role: "system", Content: "sys"}, {Role: "user", Content: "hi"}}

	// nil shaper: the default path — the sentinel must NOT appear.
	if _, err := planner.Complete(context.Background(), msgs, nil); err != nil {
		t.Fatalf("complete (nil shaper): %v", err)
	}
	if bytes.Contains(gotBody, []byte("COHERENCE_SENTINEL")) {
		t.Fatalf("nil shaper must leave the request unchanged")
	}

	// set shaper: it inserts a break marker, which must reach the wire.
	planner.CoherenceShaper = func(in []Message) []Message {
		return append([]Message{{Role: "system", Content: "COHERENCE_SENTINEL"}}, in...)
	}
	if _, err := planner.Complete(context.Background(), msgs, nil); err != nil {
		t.Fatalf("complete (with shaper): %v", err)
	}
	if !bytes.Contains(gotBody, []byte("COHERENCE_SENTINEL")) {
		t.Fatalf("shaper output was not sent to the upstream: %s", gotBody)
	}
	if len(msgs) != 2 || msgs[0].Content != "sys" {
		t.Fatalf("caller's messages slice was mutated: %+v", msgs)
	}
}

// TestShapeMessagesEndToEnd is the full message-side §A4 path: a revoked tool witness
// inserts a break marker message ahead of the stale tool message; an un-revoked world
// leaves the messages unchanged. This is exactly what the loop installs as the planner
// CoherenceShaper.
func TestShapeMessagesEndToEnd(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "system prompt with enough bytes for tokens"},
		{Role: "tool", Content: "doc1 body", ToolCallID: "call_doc1"},
		{Role: "user", Content: "continue"},
	}
	witnessOf := func(id string) string {
		if id == "call_doc1" {
			return "blob:doc1"
		}
		return ""
	}
	// Nothing revoked: messages unchanged.
	if got := ShapeMessages(msgs, witnessOf, func(string) bool { return false }); len(got) != len(msgs) {
		t.Fatalf("no revocation should leave messages unchanged, got len %d", len(got))
	}
	// doc1 revoked: a marker message is inserted ahead of the tool message (index 1).
	got := ShapeMessages(msgs, witnessOf, func(w string) bool { return w == "blob:doc1" })
	if len(got) != len(msgs)+1 {
		t.Fatalf("revoked witness should insert a marker message, got len %d", len(got))
	}
	if got[1].Role != "system" || got[1].Content == "" {
		t.Fatalf("inserted marker message malformed: %+v", got[1])
	}
	// The marker sits BEFORE the original tool message, which is now at index 2.
	if got[2].ToolCallID != "call_doc1" {
		t.Fatalf("the stale tool message should follow the marker; got %+v", got[2])
	}
	// Caller slice not mutated.
	if len(msgs) != 3 {
		t.Fatalf("caller messages mutated")
	}
}
