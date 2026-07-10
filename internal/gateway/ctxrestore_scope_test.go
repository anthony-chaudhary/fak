package gateway

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/sessionread"
	"github.com/anthony-chaudhary/fak/internal/sessionread/screen"
)

// ctxrestore_scope_test.go — red-team witnesses for the C1 read-scope floor (#4192): the guard's
// context recovery edge (fak_context_restore / fak_context_spans) is reachable unauthenticated by
// any loopback process, so a caller's principal must be checked against the trace's owner before a
// dropped originating task is paged back in. These tests attack from the position of a NON-owning
// principal and prove the floor holds — a cross-principal read is refused READ_SCOPE_DENIED with no
// bytes and no existence leak — while a legitimate self-read still round-trips. They pair with the
// outbound taint screen (screen.ScreenOutbound) so a suppressed span is withheld from the owner too.

// TestScopeFloorRefusesCrossPrincipalRestore is the core red-team case: principal "attacker" cannot
// page back a task owned by principal "owner". The refusal is the closed READ_SCOPE_DENIED token,
// and it fires BEFORE any source lookup so the attacker learns nothing about the stash — not even
// that it exists.
func TestScopeFloorRefusesCrossPrincipalRestore(t *testing.T) {
	srv := newTestServer(t)
	const (
		trace = "t-owned"
		owner = "owner"
	)
	taskBytes := []byte(`{"role":"user","content":"rotate the production secrets"}`)
	id := ctxplan.Digest(taskBytes)

	srv.bindTraceOwner(trace, owner)
	srv.stashRestore(trace, id, "rotate the production secrets", taskBytes)

	// The owner reads its own trace: byte-exact round-trip.
	got, err := srv.restoreContext(owner, ContextRestoreRequest{ID: id, TraceID: trace})
	if err != nil {
		t.Fatalf("owner self-read err = %v, want nil", err)
	}
	if got.Bytes != string(taskBytes) {
		t.Fatalf("owner self-read bytes = %q, want verbatim task", got.Bytes)
	}

	// A different principal is refused with the closed scope token, and no bytes leak.
	res, err := srv.restoreContext("attacker", ContextRestoreRequest{ID: id, TraceID: trace})
	if reason := screen.RefusalReason(err); reason != sessionread.ReasonReadScopeDenied {
		t.Fatalf("cross-principal restore refusal = %q, want %q", reason, sessionread.ReasonReadScopeDenied)
	}
	if res.Bytes != "" {
		t.Fatalf("cross-principal restore leaked bytes: %q", res.Bytes)
	}
}

// TestScopeFloorRefusesCrossPrincipalSpans proves enumeration is scoped too: the descriptors
// fak_context_spans surfaces ARE the other principal's task text (bounded), so a cross-principal
// enumeration is refused READ_SCOPE_DENIED rather than silently returning the descriptor rows.
func TestScopeFloorRefusesCrossPrincipalSpans(t *testing.T) {
	srv := newTestServer(t)
	const (
		trace = "t-owned-spans"
		owner = "owner"
	)
	taskBytes := []byte(`{"role":"user","content":"deploy the signing key"}`)
	id := ctxplan.Digest(taskBytes)
	srv.bindTraceOwner(trace, owner)
	srv.stashRestore(trace, id, "deploy the signing key", taskBytes)

	// The owner enumerates its own handles.
	got, err := srv.contextSpans(owner, ContextSpansRequest{TraceID: trace})
	if err != nil {
		t.Fatalf("owner enumeration err = %v, want nil", err)
	}
	if got.Count != 1 {
		t.Fatalf("owner enumeration count = %d, want 1", got.Count)
	}

	// A different principal is refused; no descriptor rows cross the boundary.
	res, err := srv.contextSpans("attacker", ContextSpansRequest{TraceID: trace})
	if reason := screen.RefusalReason(err); reason != sessionread.ReasonReadScopeDenied {
		t.Fatalf("cross-principal spans refusal = %q, want %q", reason, sessionread.ReasonReadScopeDenied)
	}
	if res.Count != 0 || len(res.Spans) != 0 {
		t.Fatalf("cross-principal spans leaked rows: %+v", res.Spans)
	}
}

// TestScopeFloorRefusesUnprincipaledReadOfOwnedTrace closes the other direction: a caller presenting
// NO principal ("") cannot read a trace an authenticated principal owns. The floor fails closed on
// the empty caller rather than treating "" as a wildcard.
func TestScopeFloorRefusesUnprincipaledReadOfOwnedTrace(t *testing.T) {
	srv := newTestServer(t)
	const (
		trace = "t-owned-2"
		owner = "tenant-a"
	)
	taskBytes := []byte(`{"role":"user","content":"exfiltrate nothing"}`)
	id := ctxplan.Digest(taskBytes)
	srv.bindTraceOwner(trace, owner)
	srv.stashRestore(trace, id, "orientation", taskBytes)

	_, err := srv.restoreContext("", ContextRestoreRequest{ID: id, TraceID: trace})
	if reason := screen.RefusalReason(err); reason != sessionread.ReasonReadScopeDenied {
		t.Fatalf("unprincipaled read of owned trace refusal = %q, want %q", reason, sessionread.ReasonReadScopeDenied)
	}
}

// TestDefaultTraceNotReadableCrossPrincipal is done-condition 3: on a guard with NO RequireKey
// configured (the common loopback), the default trace's dropped originating task is served with no
// principal (owner "") — a caller presenting no principal (the wrapped model itself) self-reads it,
// but a caller naming a DIFFERENT principal is refused. The floor makes the boundary exist even
// where identity is unauthenticated.
func TestDefaultTraceNotReadableCrossPrincipal(t *testing.T) {
	srv := newTestServer(t)
	const dflt = "guard-default"
	srv.SetDefaultTraceID(dflt)
	// The served turns carried no principal (no auth proxy): the default trace's owner is "".
	srv.bindTraceOwner(dflt, "")

	taskBytes := []byte(`{"role":"user","content":"the originating task"}`)
	id := ctxplan.Digest(taskBytes)
	srv.stashRestore(dflt, id, "the originating task", taskBytes)

	// The wrapped model restores its OWN dropped task with a bare (trace-omitted, principal-"") call.
	got, err := srv.restoreContext("", ContextRestoreRequest{ID: id})
	if err != nil {
		t.Fatalf("loopback self-read err = %v, want nil", err)
	}
	if got.Bytes != string(taskBytes) {
		t.Fatalf("loopback self-read bytes = %q, want verbatim task", got.Bytes)
	}

	// Another loopback process that names itself a principal cannot read the default trace.
	res, err := srv.restoreContext("mallory", ContextRestoreRequest{ID: id})
	if reason := screen.RefusalReason(err); reason != sessionread.ReasonReadScopeDenied {
		t.Fatalf("cross-principal default-trace read refusal = %q, want %q", reason, sessionread.ReasonReadScopeDenied)
	}
	if res.Bytes != "" {
		t.Fatalf("cross-principal default-trace read leaked bytes: %q", res.Bytes)
	}
}

// TestOutboundTaintScreenWithholdsSuppressedFromOwner is done-condition 2: the outbound taint screen
// is defense-in-depth WITH the scope floor — even the legitimate owner cannot page back a span the
// operator sealed or tombstoned. The refusal carries the closed READ_TAINT_WITHHELD token AND still
// satisfies the historical ErrRestoreRefused + ctxplan-sentinel contract, and no bytes cross.
func TestOutboundTaintScreenWithholdsSuppressedFromOwner(t *testing.T) {
	srv := newTestServer(t)
	const (
		trace = "t-taint"
		owner = "owner"
	)
	taskBytes := []byte(`{"role":"user","content":"quarantined content"}`)
	id := ctxplan.Digest(taskBytes)
	srv.bindTraceOwner(trace, owner)
	srv.stashRestore(trace, id, "quarantined content", taskBytes)

	// A clean span discloses byte-exact to the owner.
	clean, err := srv.restoreContext(owner, ContextRestoreRequest{ID: id, TraceID: trace})
	if err != nil {
		t.Fatalf("clean owner read err = %v, want nil", err)
	}
	if clean.Bytes != string(taskBytes) {
		t.Fatalf("clean span disclosure = %q, want byte-exact", clean.Bytes)
	}

	// Seal it: the outbound screen now withholds the bytes even from the owner.
	if n := srv.sealRestore(id); n != 1 {
		t.Fatalf("sealRestore flipped %d, want 1", n)
	}
	res, err := srv.restoreContext(owner, ContextRestoreRequest{ID: id, TraceID: trace})
	if reason := screen.RefusalReason(err); reason != sessionread.ReasonReadTaintWithheld {
		t.Fatalf("sealed read refusal token = %q, want %q", reason, sessionread.ReasonReadTaintWithheld)
	}
	if !errors.Is(err, ErrRestoreRefused) || !errors.Is(err, ctxplan.ErrSealed) {
		t.Fatalf("sealed read err = %v, want to still satisfy ErrRestoreRefused+ErrSealed", err)
	}
	if res.Bytes != "" {
		t.Fatalf("sealed read leaked bytes: %q", res.Bytes)
	}
}

// TestScopeFloorUsesTaintScreenBytesByteExact pins the byte-exactness half of done-condition 2 at
// the disclosure point: a clean span the owner reads is the SAME bytes the stash holds, unaltered by
// the outbound screen (the screen adds nothing and removes nothing from a span it clears).
func TestScopeFloorUsesTaintScreenBytesByteExact(t *testing.T) {
	srv := newTestServer(t)
	const (
		trace = "t-exact"
		owner = "owner"
	)
	// Include bytes that must survive verbatim (unicode, quotes, braces).
	taskBytes := []byte(`{"role":"user","content":"café: {\"k\":\"v\"} — do X"}`)
	id := ctxplan.Digest(taskBytes)
	srv.bindTraceOwner(trace, owner)
	srv.stashRestore(trace, id, "orientation", taskBytes)

	got, err := srv.restoreContext(owner, ContextRestoreRequest{ID: id, TraceID: trace})
	if err != nil {
		t.Fatalf("owner read err = %v, want nil", err)
	}
	if got.Bytes != string(taskBytes) {
		t.Fatalf("disclosed bytes = %q, want byte-exact %q", got.Bytes, taskBytes)
	}
}
