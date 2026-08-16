package gateway

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// ctxspans_test.go — witnesses for the enumeration arm of the context API: a trace's stashed dropped
// spans list back in stash order as SAFE, byte-free handles; a sealed/tombstoned span is LISTED but not
// restorable and leaks no bytes; an empty trace is a valid empty answer; and the verb is callable
// end-to-end over MCP. Mirrors the digest→bytes stash the restore path round-trips, read-only.

// TestContextSpansEnumeratesInOrder: two stashed entries list back in the order they were stashed, each
// carrying its id/descriptor/bytes, all Restorable (neither gate flipped).
func TestContextSpansEnumeratesInOrder(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-spans"
	first := []byte(`{"role":"user","content":"rotate the auth tokens"}`)
	second := []byte(`{"role":"user","content":"then redeploy the gateway"}`)
	idA := ctxplan.Digest(first)
	idB := ctxplan.Digest(second)
	srv.stashRestore(trace, idA, "rotate the auth tokens", first)
	srv.stashRestore(trace, idB, "then redeploy the gateway", second)

	got, err := srv.contextSpans("", ContextSpansRequest{TraceID: trace})
	if err != nil {
		t.Fatalf("contextSpans err = %v, want nil (self-read)", err)
	}
	if got.Schema != ctxSpansSchema {
		t.Fatalf("schema = %q, want %q", got.Schema, ctxSpansSchema)
	}
	if got.TraceID != trace {
		t.Fatalf("trace = %q, want %q", got.TraceID, trace)
	}
	if got.Count != 2 || len(got.Spans) != 2 {
		t.Fatalf("count = %d / len = %d, want 2 spans", got.Count, len(got.Spans))
	}
	// Stash order preserved (oldest first).
	if got.Spans[0].ID != idA || got.Spans[1].ID != idB {
		t.Fatalf("ids out of order: %q, %q; want %q, %q", got.Spans[0].ID, got.Spans[1].ID, idA, idB)
	}
	if got.Spans[0].Descriptor != "rotate the auth tokens" {
		t.Fatalf("descriptor = %q, want the orientation excerpt", got.Spans[0].Descriptor)
	}
	if got.Spans[0].Bytes != int64(len(first)) {
		t.Fatalf("bytes = %d, want %d (the size proxy)", got.Spans[0].Bytes, len(first))
	}
	for i, sp := range got.Spans {
		if !sp.Restorable {
			t.Fatalf("span %d Restorable = false, want true (no gate flipped)", i)
		}
	}
}

// TestContextSpansListsSuppressedNotRestorable: a sealed entry and a tombstoned entry are LISTED (their
// suppression is legible) with Restorable=false, and NO stashed bytes appear anywhere in the marshaled
// result — enumeration surfaces what is recoverable, never the recoverable content.
func TestContextSpansListsSuppressedNotRestorable(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-spans-gate"
	sealedBytes := []byte(`{"role":"user","content":"SEALED-SECRET-PAYLOAD"}`)
	tombBytes := []byte(`{"role":"user","content":"TOMBSTONED-SECRET-PAYLOAD"}`)
	sealedID := ctxplan.Digest(sealedBytes)
	tombID := ctxplan.Digest(tombBytes)
	srv.stashRestore(trace, sealedID, "s", sealedBytes)
	srv.stashRestore(trace, tombID, "t", tombBytes)

	// Flip the gates directly on the stashed entries (the operator context-control action), exactly as
	// TestRestoreTrustGateRefuses does.
	srv.ctxRestoreMu.Lock()
	for i := range srv.ctxRestore[trace].entries {
		switch srv.ctxRestore[trace].entries[i].id {
		case sealedID:
			srv.ctxRestore[trace].entries[i].sealed = true
		case tombID:
			srv.ctxRestore[trace].entries[i].tombstoned = true
		}
	}
	srv.ctxRestoreMu.Unlock()

	got, err := srv.contextSpans("", ContextSpansRequest{TraceID: trace})
	if err != nil {
		t.Fatalf("contextSpans err = %v, want nil (self-read)", err)
	}
	if got.Count != 2 {
		t.Fatalf("count = %d, want 2 (suppressed spans are still LISTED)", got.Count)
	}
	for _, sp := range got.Spans {
		switch sp.ID {
		case sealedID:
			if !sp.Sealed || sp.Restorable {
				t.Fatalf("sealed span: Sealed=%v Restorable=%v, want true/false", sp.Sealed, sp.Restorable)
			}
		case tombID:
			if !sp.Tombstoned || sp.Restorable {
				t.Fatalf("tombstoned span: Tombstoned=%v Restorable=%v, want true/false", sp.Tombstoned, sp.Restorable)
			}
		default:
			t.Fatalf("unexpected span id %q", sp.ID)
		}
	}

	// No full bytes leak: the marshaled result must not carry either suppressed payload, and the row has
	// no bytes field at all (only a size). This is the read-only invariant — listing is not recovery.
	raw, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	blob := string(raw)
	for _, secret := range []string{"SEALED-SECRET-PAYLOAD", "TOMBSTONED-SECRET-PAYLOAD"} {
		if strings.Contains(blob, secret) {
			t.Fatalf("enumeration leaked span bytes %q: %s", secret, blob)
		}
	}
	// The CtxSpan shape has a "bytes" size field but never a bytes STRING of content; assert the size is
	// the length, not the payload.
	if got.Spans[0].Bytes == 0 {
		t.Fatalf("size proxy missing on span 0")
	}
}

// TestContextSpansEmptyTraceIsEmptyAnswer: an unknown/empty trace returns Count 0 and a non-nil empty
// Spans slice — enumerating a trace that dropped nothing is a valid answer, not an error (unlike
// restore-by-id, which misses).
func TestContextSpansEmptyTraceIsEmptyAnswer(t *testing.T) {
	srv := newTestServer(t)
	got, err := srv.contextSpans("", ContextSpansRequest{TraceID: "never-stashed"})
	if err != nil {
		t.Fatalf("contextSpans err = %v, want nil (self-read)", err)
	}
	if got.Count != 0 {
		t.Fatalf("count = %d, want 0 for an unknown trace", got.Count)
	}
	if got.Spans == nil {
		t.Fatal("Spans is nil, want a non-nil empty slice")
	}
	if len(got.Spans) != 0 {
		t.Fatalf("len(Spans) = %d, want 0", len(got.Spans))
	}
	if got.Schema != ctxSpansSchema {
		t.Fatalf("schema = %q, want %q even on an empty answer", got.Schema, ctxSpansSchema)
	}
}

// TestContextSpansOverMCP: the verb is listed in tools/list and callable end-to-end over the MCP
// dispatch, returning the stashed handles with the right schema.
func TestContextSpansOverMCP(t *testing.T) {
	srv := newTestServer(t)
	search, err := srv.toolsSearch(ToolsSearchRequest{Query: "context spans", DetailLevel: "name"})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, descriptor := range search.Tools {
		if descriptor["name"] == "fak_context_spans" {
			found = true
		}
	}
	if !found {
		t.Fatal("fak_tools_search missing fak_context_spans")
	}

	const trace = "t-mcp-spans"
	taskBytes := []byte(`{"role":"user","content":"the originating task"}`)
	id := ctxplan.Digest(taskBytes)
	srv.stashRestore(trace, id, "the originating task", taskBytes)

	got := callMCPTool[CtxSpansResult](t, srv, "fak_context_spans", map[string]any{"trace_id": trace})
	if got.Schema != ctxSpansSchema {
		t.Fatalf("MCP spans schema = %q, want %q", got.Schema, ctxSpansSchema)
	}
	if got.Count != 1 || len(got.Spans) != 1 {
		t.Fatalf("MCP spans count = %d, want 1", got.Count)
	}
	if got.Spans[0].ID != id || !got.Spans[0].Restorable {
		t.Fatalf("MCP span = id %q restorable %v, want %q / true", got.Spans[0].ID, got.Spans[0].Restorable, id)
	}
}
