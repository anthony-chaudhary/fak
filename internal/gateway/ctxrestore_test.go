package gateway

import (
	"errors"
	"strings"
	"testing"

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
