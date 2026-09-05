package gateway

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// ctxrestore_test.go — witnesses for the restore-by-ID recovery edge: a stashed originating task
// pages back in by its content-address handle, the trust gate refuses a sealed/tombstoned span, and
// an unknown id is a miss. Round-trips the same digest→bytes contract the compaction path feeds.

// TestRestoreRoundTrips: stash a dropped originating task under its handle, then restore it — the
// bytes come back verbatim, the excerpt echoes, and the provenance is WITNESSED.
func TestRestoreRoundTrips(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-restore"
	taskBytes := []byte(`{"role":"user","content":"rotate the auth tokens"}`)
	id := ctxplan.Digest(taskBytes)

	srv.stashRestore(trace, id, "rotate the auth tokens", taskBytes)

	got, err := srv.restoreContext("", ContextRestoreRequest{ID: id, TraceID: trace})
	if err != nil {
		t.Fatalf("restore: %v", err)
	}
	if got.Bytes != string(taskBytes) {
		t.Fatalf("restored bytes = %q, want %q", got.Bytes, taskBytes)
	}
	if got.ID != id || got.TraceID != trace {
		t.Fatalf("restore identity = id %q trace %q, want %q / %q", got.ID, got.TraceID, id, trace)
	}
	if !strings.Contains(got.Excerpt, "rotate the auth tokens") {
		t.Fatalf("restore excerpt = %q, want the task text", got.Excerpt)
	}
	if got.Provenance != "WITNESSED" {
		t.Fatalf("restore provenance = %q, want WITNESSED", got.Provenance)
	}
}

// TestRestoreMissUnknownID: an id the trace never stashed is a MISS (not a refusal) — the caller can
// tell "never had it" from "the gate held".
func TestRestoreMissUnknownID(t *testing.T) {
	srv := newTestServer(t)
	srv.stashRestore("t-miss", ctxplan.Digest([]byte("a")), "a", []byte("a"))

	_, err := srv.restoreContext("", ContextRestoreRequest{ID: "deadbeef", TraceID: "t-miss"})
	if !errors.Is(err, ErrRestoreMiss) {
		t.Fatalf("unknown id err = %v, want ErrRestoreMiss", err)
	}
	// A trace that never stashed anything is likewise a miss, not a panic.
	if _, err := srv.restoreContext("", ContextRestoreRequest{ID: "x", TraceID: "never"}); !errors.Is(err, ErrRestoreMiss) {
		t.Fatalf("unknown trace err = %v, want ErrRestoreMiss", err)
	}
}

// TestRestoreTrustGateRefuses: a sealed or tombstoned entry is REFUSED (never resurrected), and the
// refusal carries the specific ctxplan sentinel so a caller can branch on which gate held. Restore
// must not defeat the suppression that dropped the span.
func TestRestoreTrustGateRefuses(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-gate"
	sealedID := ctxplan.Digest([]byte("sealed"))
	tombID := ctxplan.Digest([]byte("tomb"))
	srv.stashRestore(trace, sealedID, "s", []byte("sealed"))
	srv.stashRestore(trace, tombID, "t", []byte("tomb"))

	// Flip the gates directly on the stashed entries (the operator context-control action).
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

	_, err := srv.restoreContext("", ContextRestoreRequest{ID: sealedID, TraceID: trace})
	if !errors.Is(err, ErrRestoreRefused) || !errors.Is(err, ctxplan.ErrSealed) {
		t.Fatalf("sealed restore err = %v, want refused+ErrSealed", err)
	}
	_, err = srv.restoreContext("", ContextRestoreRequest{ID: tombID, TraceID: trace})
	if !errors.Is(err, ErrRestoreRefused) || !errors.Is(err, ctxplan.ErrTombstoned) {
		t.Fatalf("tombstoned restore err = %v, want refused+ErrTombstoned", err)
	}
}

// TestRestoreEmptyIDRejected: a call with no id is a plain argument error, not a miss.
func TestRestoreEmptyIDRejected(t *testing.T) {
	srv := newTestServer(t)
	if _, err := srv.restoreContext("", ContextRestoreRequest{ID: "  ", TraceID: "t"}); err == nil {
		t.Fatal("empty id must error")
	}
}

// TestStashRestoreSafeNoops: nil/empty inputs are safe no-ops (nothing callable to stash), so the
// compaction path can feed them unconditionally.
func TestStashRestoreSafeNoops(t *testing.T) {
	srv := newTestServer(t)
	srv.stashRestore("", "id", "x", []byte("x")) // empty trace
	srv.stashRestore("t", "", "x", []byte("x"))  // empty id
	srv.stashRestore("t", "id", "x", nil)        // empty bytes
	if len(srv.ctxRestore) != 0 {
		t.Fatalf("safe no-ops stashed something: %v", srv.ctxRestore)
	}
}

// TestStashRestoreCompressesLargePayload (#5164): a payload at/above the compression floor is held
// deflated in the stash (smaller resident bytes, compressed flag set) yet restores VERBATIM, and
// enumeration still reports the true (uncompressed) span size. A small text turn stays verbatim.
func TestStashRestoreCompressesLargePayload(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-compress"
	big := []byte(`{"role":"user","content":"` + strings.Repeat("base64ish-image-bytes ", 512) + `"}`)
	if len(big) < ctxRestoreCompressThreshold {
		t.Fatalf("test payload %dB under the compression floor %dB", len(big), ctxRestoreCompressThreshold)
	}
	bigID := ctxplan.Digest(big)
	small := []byte(`{"role":"user","content":"a text turn"}`)
	smallID := ctxplan.Digest(small)

	srv.stashRestore(trace, bigID, "an image turn", big)
	srv.stashRestore(trace, smallID, "a text turn", small)

	srv.ctxRestoreMu.Lock()
	for _, e := range srv.ctxRestore[trace].entries {
		switch e.id {
		case bigID:
			if !e.compressed || len(e.bytes) >= len(big) {
				t.Errorf("large payload resident form = %dB compressed=%v, want deflated < %dB", len(e.bytes), e.compressed, len(big))
			}
			if e.rawLen != len(big) {
				t.Errorf("large payload rawLen = %d, want %d", e.rawLen, len(big))
			}
		case smallID:
			if e.compressed || string(e.bytes) != string(small) {
				t.Errorf("small payload must stay verbatim (compressed=%v)", e.compressed)
			}
		}
	}
	srv.ctxRestoreMu.Unlock()

	// The restored bytes are verbatim — compression is a resident-form detail, never a lossy one.
	got, err := srv.restoreContext("", ContextRestoreRequest{ID: bigID, TraceID: trace})
	if err != nil {
		t.Fatalf("restore compressed entry: %v", err)
	}
	if got.Bytes != string(big) {
		t.Fatalf("restored bytes differ from the dropped turn (len %d vs %d)", len(got.Bytes), len(big))
	}

	// Enumeration reports the TRUE span size, not the deflated stored size.
	spans, err := srv.contextSpans("", ContextSpansRequest{TraceID: trace})
	if err != nil {
		t.Fatalf("spans: %v", err)
	}
	for _, sp := range spans.Spans {
		if sp.ID == bigID && sp.Bytes != int64(len(big)) {
			t.Fatalf("span size = %d, want the uncompressed %d", sp.Bytes, len(big))
		}
	}
}

// TestStashRestoreMediaCap (#5164): media-class (large) entries overflow against the smaller
// media-specific cap, oldest-media-out, while text entries are untouched — an image-heavy session
// cannot fill all eight flat slots with full-size blobs.
func TestStashRestoreMediaCap(t *testing.T) {
	// Witness the RAM cap in isolation: with the durable media CAS (#5163) on, an evicted media
	// entry is deliberately still restorable from disk — that contract has its own witness in
	// ctxrestore_cas_test.go. This test is about the in-memory stash bound.
	t.Setenv(ctxRestoreCASEnvDir, "off")
	srv := newTestServer(t)
	const trace = "t-mediacap"

	text := []byte(`{"role":"user","content":"a small text turn"}`)
	textID := ctxplan.Digest(text)
	srv.stashRestore(trace, textID, "text", text)

	mediaIDs := make([]string, 0, maxCtxRestoreMediaEntriesPerSession+1)
	for i := 0; i <= maxCtxRestoreMediaEntriesPerSession; i++ {
		blob := []byte(`{"role":"user","content":"` + strings.Repeat(string(rune('a'+i)), ctxRestoreMediaThreshold) + `"}`)
		id := ctxplan.Digest(blob)
		mediaIDs = append(mediaIDs, id)
		srv.stashRestore(trace, id, "an image turn", blob)
	}

	// The oldest media entry was reclaimed; the newer ones and the text entry survive.
	if _, err := srv.restoreContext("", ContextRestoreRequest{ID: mediaIDs[0], TraceID: trace}); !errors.Is(err, ErrRestoreMiss) {
		t.Fatalf("oldest media entry err = %v, want ErrRestoreMiss (reclaimed by the media cap)", err)
	}
	for _, id := range mediaIDs[1:] {
		if _, err := srv.restoreContext("", ContextRestoreRequest{ID: id, TraceID: trace}); err != nil {
			t.Fatalf("surviving media entry %s: %v", id[:8], err)
		}
	}
	if _, err := srv.restoreContext("", ContextRestoreRequest{ID: textID, TraceID: trace}); err != nil {
		t.Fatalf("text entry must be untouched by the media cap: %v", err)
	}

	srv.ctxRestoreMu.Lock()
	if n := countMediaEntries(srv.ctxRestore[trace].entries); n != maxCtxRestoreMediaEntriesPerSession {
		t.Errorf("resident media entries = %d, want %d", n, maxCtxRestoreMediaEntriesPerSession)
	}
	srv.ctxRestoreMu.Unlock()
}

// TestRestoreOverMCP proves the cold restore schema remains discoverable through
// fak_tools_search and callable end-to-end after default-on tool filtering.
func TestRestoreOverMCP(t *testing.T) {
	srv := newTestServer(t)
	search, err := srv.toolsSearch(ToolsSearchRequest{Query: "context restore", DetailLevel: "name"})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, descriptor := range search.Tools {
		if descriptor["name"] == "fak_context_restore" {
			found = true
		}
	}
	if !found {
		t.Fatal("fak_tools_search missing fak_context_restore")
	}

	const trace = "t-mcp-restore"
	taskBytes := []byte(`{"role":"user","content":"the originating task"}`)
	id := ctxplan.Digest(taskBytes)
	srv.stashRestore(trace, id, "the originating task", taskBytes)

	got := callMCPTool[CtxRestoreResult](t, srv, "fak_context_restore", map[string]any{"id": id, "trace_id": trace})
	if got.Bytes != string(taskBytes) {
		t.Fatalf("MCP restore bytes = %q, want %q", got.Bytes, taskBytes)
	}
	if got.Schema != ctxRestoreSchema {
		t.Fatalf("MCP restore schema = %q, want %q", got.Schema, ctxRestoreSchema)
	}
}

// TestRestoreDigestSchemeMatchesCtxplan: the handle the gateway keys on is the SAME 64-char sha256
// hex the ctxplan.Digest scheme produces — proving compaction handles and ctxplan handles share one
// address space. (agent.originatingTaskDigestID is unexported; ctxplan.Digest is the shared public
// scheme, and both sha256-hex the same bytes.)
func TestRestoreDigestSchemeMatchesCtxplan(t *testing.T) {
	b := []byte(`{"role":"user","content":"x"}`)
	if got := ctxplan.Digest(b); len(got) != 64 { //boundarylint:ignore CHANGE_DETECTOR_TEST sha256 hex is a fixed 64-char width
		t.Fatalf("ctxplan.Digest length = %d, want 64-hex", len(got))
	}
}

// TestMCPPagedResultBoundedRetrieval proves that paged MMU result refs exposed at the MCP boundary
// advertise an actionable retrieval call, accept explicit range/limit bounds across subsequent slices,
// return valid continuation handles, reconstruct byte-exact content, and refuse cross-principal
// or unknown ref disclosure (#11551).
func TestMCPPagedResultBoundedRetrieval(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	abi.RegisterResultAdmitter(10, ctxmmu.New())

	srv, err := New(Config{EngineID: "test", Model: "test-model"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// 1. Force paging over MCP loopback.
	// We send a tool result > 16 KiB (oversize threshold) via fak_admit under trace "alice-trace" with principal "alice".
	const line = "0123456789abcdefghij\n"         // 21 bytes
	originalContent := strings.Repeat(line, 1500) // 31,500 bytes (> 16 KiB)

	admitCall := map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params": map[string]any{
			"name": "fak_admit",
			"arguments": map[string]any{
				"tool":     "allow_large_reader",
				"result":   originalContent,
				"trace_id": "alice-trace",
			},
		},
	}
	admitBytes, err := json.Marshal(admitCall)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest(http.MethodPost, ts.URL+"/mcp", bytes.NewReader(admitBytes))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Fak-Principal", "alice")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /mcp: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("admit status = %d, want 200", resp.StatusCode)
	}

	var rpcResp rpcResponse
	if err := json.NewDecoder(resp.Body).Decode(&rpcResp); err != nil {
		t.Fatalf("decode rpc response: %v", err)
	}
	if rpcResp.Error != nil {
		t.Fatalf("unexpected RPC error: %+v", rpcResp.Error)
	}

	var sysResp SyscallResponse
	decodeMCPResult(t, rpcResp.Result, &sysResp)

	if sysResp.Verdict.Kind != "TRANSFORM" {
		t.Fatalf("verdict = %s, want TRANSFORM (oversize paged)", sysResp.Verdict.Kind)
	}
	if sysResp.Result == nil {
		t.Fatal("result envelope is nil")
	}

	var pagedStub map[string]any
	if err := json.Unmarshal([]byte(sysResp.Result.Content), &pagedStub); err != nil {
		t.Fatalf("unmarshal paged content: %v, raw=%s", err, sysResp.Result.Content)
	}
	if paged, _ := pagedStub["_paged"].(bool); !paged {
		t.Fatalf("expected _paged=true in %+v", pagedStub)
	}
	ref, _ := pagedStub["ref"].(string)
	if ref == "" {
		t.Fatalf("expected non-empty ref in %+v", pagedStub)
	}

	// Verify enriched retrieval call in paged response
	retrievalCall, ok := pagedStub["retrieval"].(map[string]any)
	if !ok {
		t.Fatalf("paged response missing advertised retrieval call: %+v", pagedStub)
	}
	toolName, _ := retrievalCall["name"].(string)
	if toolName == "" {
		toolName, _ = retrievalCall["tool"].(string)
	}
	if toolName != "fak_context_restore" {
		t.Fatalf("advertised retrieval tool = %q, want fak_context_restore", toolName)
	}
	retrievalArgs, ok := retrievalCall["arguments"].(map[string]any)
	if !ok {
		t.Fatalf("retrieval call missing arguments: %+v", retrievalCall)
	}
	if retrievalArgs["id"] != ref {
		t.Fatalf("retrieval id = %v, want %v", retrievalArgs["id"], ref)
	}

	// 2. Follow the advertised bounded retrieval for two slices.
	// Slice 1: call fak_context_restore with advertised args
	slice1ReqBody, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      2,
		"method":  "tools/call",
		"params": map[string]any{
			"name":      "fak_context_restore",
			"arguments": retrievalArgs,
		},
	})
	req1, _ := http.NewRequest(http.MethodPost, ts.URL+"/mcp", bytes.NewReader(slice1ReqBody))
	req1.Header.Set("Content-Type", "application/json")
	req1.Header.Set("X-Fak-Principal", "alice")

	resp1, err := http.DefaultClient.Do(req1)
	if err != nil {
		t.Fatalf("POST /mcp slice1: %v", err)
	}
	defer resp1.Body.Close()
	var rpcResp1 rpcResponse
	if err := json.NewDecoder(resp1.Body).Decode(&rpcResp1); err != nil {
		t.Fatalf("decode slice1 rpc response: %v", err)
	}
	if rpcResp1.Error != nil {
		t.Fatalf("slice1 RPC error: %+v", rpcResp1.Error)
	}

	var slice1 CtxRestoreResult
	decodeMCPResult(t, rpcResp1.Result, &slice1)

	// Check output bounds and continuation on slice 1
	limitVal := int(retrievalArgs["limit"].(float64))
	if len(slice1.Bytes) > limitVal {
		t.Fatalf("slice1 bytes len = %d, exceeds limit %d", len(slice1.Bytes), limitVal)
	}
	if len(slice1.Bytes) == 0 {
		t.Fatal("slice1 returned empty bytes")
	}
	if !slice1.HasMore {
		t.Fatal("slice1 expected has_more = true")
	}
	if slice1.Continuation == nil {
		t.Fatal("slice1 expected continuation != nil")
	}
	if slice1.TotalBytes != len(originalContent) {
		t.Fatalf("slice1 total_bytes = %d, want %d", slice1.TotalBytes, len(originalContent))
	}
	if slice1.NextOffset != len(slice1.Bytes) {
		t.Fatalf("slice1 next_offset = %d, want %d", slice1.NextOffset, len(slice1.Bytes))
	}

	// Slice 2: follow continuation
	slice2ReqBody, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      3,
		"method":  "tools/call",
		"params": map[string]any{
			"name":      slice1.Continuation.Name,
			"arguments": slice1.Continuation.Arguments,
		},
	})
	req2, _ := http.NewRequest(http.MethodPost, ts.URL+"/mcp", bytes.NewReader(slice2ReqBody))
	req2.Header.Set("Content-Type", "application/json")
	req2.Header.Set("X-Fak-Principal", "alice")

	resp2, err := http.DefaultClient.Do(req2)
	if err != nil {
		t.Fatalf("POST /mcp slice2: %v", err)
	}
	defer resp2.Body.Close()
	var rpcResp2 rpcResponse
	if err := json.NewDecoder(resp2.Body).Decode(&rpcResp2); err != nil {
		t.Fatalf("decode slice2 rpc response: %v", err)
	}
	if rpcResp2.Error != nil {
		t.Fatalf("slice2 RPC error: %+v", rpcResp2.Error)
	}

	var slice2 CtxRestoreResult
	decodeMCPResult(t, rpcResp2.Result, &slice2)

	// Check output bounds and continuation on slice 2
	if slice2.HasMore {
		t.Fatal("slice2 expected has_more = false")
	}
	if slice2.Continuation != nil {
		t.Fatalf("slice2 expected nil continuation, got %+v", slice2.Continuation)
	}
	if slice2.Offset != len(slice1.Bytes) {
		t.Fatalf("slice2 offset = %d, want %d", slice2.Offset, len(slice1.Bytes))
	}
	if slice2.TotalBytes != len(originalContent) {
		t.Fatalf("slice2 total_bytes = %d, want %d", slice2.TotalBytes, len(originalContent))
	}

	// Byte-compare reconstructed content
	reconstructed := slice1.Bytes + slice2.Bytes
	if reconstructed != originalContent {
		t.Fatalf("reconstructed content mismatch: len got %d, want %d", len(reconstructed), len(originalContent))
	}

	// 3. Confirm unknown ref cannot disclose content
	unknownReqBody, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      4,
		"method":  "tools/call",
		"params": map[string]any{
			"name": "fak_context_restore",
			"arguments": map[string]any{
				"id":       "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
				"trace_id": "alice-trace",
			},
		},
	})
	reqUnknown, _ := http.NewRequest(http.MethodPost, ts.URL+"/mcp", bytes.NewReader(unknownReqBody))
	reqUnknown.Header.Set("Content-Type", "application/json")
	reqUnknown.Header.Set("X-Fak-Principal", "alice")
	respUnknown, err := http.DefaultClient.Do(reqUnknown)
	if err != nil {
		t.Fatalf("POST /mcp unknown: %v", err)
	}
	defer respUnknown.Body.Close()
	var rpcRespUnknown rpcResponse
	_ = json.NewDecoder(respUnknown.Body).Decode(&rpcRespUnknown)
	if rpcRespUnknown.Error == nil {
		t.Fatal("unknown ref expected JSON-RPC error, got success")
	}

	// 4. Confirm cross-principal ref cannot disclose content
	// 4a. Cross-principal under alice's trace
	crossReqBody, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      5,
		"method":  "tools/call",
		"params": map[string]any{
			"name": "fak_context_restore",
			"arguments": map[string]any{
				"id":       ref,
				"trace_id": "alice-trace",
			},
		},
	})
	reqCross, _ := http.NewRequest(http.MethodPost, ts.URL+"/mcp", bytes.NewReader(crossReqBody))
	reqCross.Header.Set("Content-Type", "application/json")
	reqCross.Header.Set("X-Fak-Principal", "bob") // bob attempting to read alice's trace/ref
	respCross, err := http.DefaultClient.Do(reqCross)
	if err != nil {
		t.Fatalf("POST /mcp cross: %v", err)
	}
	defer respCross.Body.Close()
	var rpcRespCross rpcResponse
	_ = json.NewDecoder(respCross.Body).Decode(&rpcRespCross)
	if rpcRespCross.Error == nil {
		t.Fatal("cross-principal ref expected JSON-RPC error, got success")
	}
	if !strings.Contains(rpcRespCross.Error.Message, "READ_SCOPE_DENIED") && !strings.Contains(rpcRespCross.Error.Message, "refused") {
		t.Fatalf("cross-principal error message = %q, expected refusal/READ_SCOPE_DENIED", rpcRespCross.Error.Message)
	}

	// 4b. Cross-principal under bob's own trace
	crossBobTraceBody, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      6,
		"method":  "tools/call",
		"params": map[string]any{
			"name": "fak_context_restore",
			"arguments": map[string]any{
				"id":       ref,
				"trace_id": "bob-trace",
			},
		},
	})
	reqCrossBob, _ := http.NewRequest(http.MethodPost, ts.URL+"/mcp", bytes.NewReader(crossBobTraceBody))
	reqCrossBob.Header.Set("Content-Type", "application/json")
	reqCrossBob.Header.Set("X-Fak-Principal", "bob")
	respCrossBob, err := http.DefaultClient.Do(reqCrossBob)
	if err != nil {
		t.Fatalf("POST /mcp cross-bob: %v", err)
	}
	defer respCrossBob.Body.Close()
	var rpcRespCrossBob rpcResponse
	_ = json.NewDecoder(respCrossBob.Body).Decode(&rpcRespCrossBob)
	if rpcRespCrossBob.Error == nil {
		t.Fatal("cross-principal ref under own trace expected JSON-RPC error, got success")
	}
}

func decodeMCPResult(t *testing.T, res any, v any) {
	t.Helper()
	m, ok := res.(map[string]any)
	if !ok {
		t.Fatalf("mcp result is not an object: %T", res)
	}
	rawContent, ok := m["content"]
	if !ok {
		t.Fatalf("mcp result has no content: %+v", m)
	}
	var text string
	switch c := rawContent.(type) {
	case []any:
		if len(c) == 0 {
			t.Fatalf("mcp result content empty: %+v", m)
		}
		item, _ := c[0].(map[string]any)
		text, _ = item["text"].(string)
	case []map[string]any:
		if len(c) == 0 {
			t.Fatalf("mcp result content empty: %+v", m)
		}
		text, _ = c[0]["text"].(string)
	default:
		t.Fatalf("unexpected content type: %T", rawContent)
	}
	if err := json.Unmarshal([]byte(text), v); err != nil {
		t.Fatalf("decode mcp text %q: %v", text, err)
	}
}
