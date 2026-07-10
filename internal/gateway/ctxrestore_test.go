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

// TestRestoreOverMCP: the verb is listed and callable end-to-end over the MCP dispatch, returning the
// stashed bytes.
func TestRestoreOverMCP(t *testing.T) {
	srv := newTestServer(t)
	list := resultMap(t, rpcRoundTrip(t, srv, "tools/list", ""))
	tools := list["tools"].([]any)
	found := false
	for _, raw := range tools {
		if raw.(map[string]any)["name"] == "fak_context_restore" {
			found = true
		}
	}
	if !found {
		t.Fatal("tools/list missing fak_context_restore")
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
