package gateway

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// ctxrestore_gate_test.go — witnesses for the PRODUCTION trust-propagation wire (#3063): an operator
// seal/tombstone, keyed by a span's content-address digest, flips the matching stashed restoreEntry so
// restoreContext REFUSES it and contextSpans lists it Restorable=false. These prove the WIRING (the
// operator action reaches the stash), not just the already-witnessed refusal side (ctxrestore_test.go
// flips the flags by hand). The load-bearing test drives the SHIPPED operator tombstone —
// Server.contextChange over a real persisted recall image — end to end.

// TestTombstoneRestoreWiredThroughContextChange is the shipped end-to-end witness: the operator
// context-control tombstone (Server.contextChange, over a REAL persisted recall image) suppresses the
// restore handle stashed under the SAME content-address. Before the action the handle restores; after
// it, restoreContext refuses with ErrRestoreRefused+ctxplan.ErrTombstoned and contextSpans reports the
// span Tombstoned, Restorable=false. This is the gap #3063 closed: nothing but a unit test used to set
// that flag.
func TestTombstoneRestoreWiredThroughContextChange(t *testing.T) {
	srv := newTestServer(t)
	ctx := context.Background()

	// A real persisted recall image; its page digest is the content-address the operator tombstones.
	dir, digest := writeRecallImage(t, "gateway-gate-e2e")

	// Stash a restore handle under that SAME content-address. In production the dropped span and the
	// recall page share one digest because a content-address is a pure function of the bytes; here we
	// bind them deliberately by keying the stash on the page digest, which is exactly the join the wire
	// resolves on.
	const trace = "t-gate-e2e"
	taskBytes := []byte(`{"role":"user","content":"the originating task"}`)
	srv.stashRestore(trace, digest, "the originating task", taskBytes)

	// Precondition: the handle restores before the operator acts (the wire, not a pre-flipped flag, is
	// what suppresses it).
	if _, err := srv.restoreContext(ContextRestoreRequest{ID: digest, TraceID: trace}); err != nil {
		t.Fatalf("precondition: handle should restore before the tombstone, got %v", err)
	}

	// The SHIPPED operator action.
	resp, err := srv.contextChange(ctx, ContextChangeRequest{
		ImageDir:    dir,
		Step:        0,
		Digest:      digest,
		Reason:      "operator suppressed this span",
		RequestedBy: "operator",
	})
	if err != nil {
		t.Fatalf("contextChange: %v", err)
	}
	if resp.Action != string(recall.ContextActionTombstone) || !resp.Applied || !resp.Tombstoned {
		t.Fatalf("contextChange not an applied tombstone: %+v", resp)
	}

	// The wire fired: the restore handle is now refused with the tombstone sentinel.
	_, err = srv.restoreContext(ContextRestoreRequest{ID: digest, TraceID: trace})
	if !errors.Is(err, ErrRestoreRefused) || !errors.Is(err, ctxplan.ErrTombstoned) {
		t.Fatalf("post-tombstone restore err = %v, want refused+ErrTombstoned", err)
	}

	// Cross-feature: fak_context_spans now lists the span as suppressed, not restorable.
	spans := srv.contextSpans(ContextSpansRequest{TraceID: trace})
	if spans.Count != 1 {
		t.Fatalf("spans count = %d, want 1 (the suppressed span is still LISTED)", spans.Count)
	}
	if sp := spans.Spans[0]; sp.ID != digest || !sp.Tombstoned || sp.Restorable {
		t.Fatalf("span = %+v, want id=%s Tombstoned=true Restorable=false", sp, digest)
	}
}

// TestTombstoneRestoreDirectMethod drives the exported wiring method directly: tombstoneRestore(digest)
// flips exactly the matching handle, reports 1, and the refusal + not-restorable projection follow —
// the same outcome the contextChange path produces, isolated from the recall-image plumbing.
func TestTombstoneRestoreDirectMethod(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-gate-direct"
	taskBytes := []byte(`{"role":"user","content":"rotate the auth tokens"}`)
	id := ctxplan.Digest(taskBytes)
	srv.stashRestore(trace, id, "rotate the auth tokens", taskBytes)

	if n := srv.tombstoneRestore(id); n != 1 {
		t.Fatalf("tombstoneRestore flipped %d, want 1", n)
	}
	_, err := srv.restoreContext(ContextRestoreRequest{ID: id, TraceID: trace})
	if !errors.Is(err, ErrRestoreRefused) || !errors.Is(err, ctxplan.ErrTombstoned) {
		t.Fatalf("restore err = %v, want refused+ErrTombstoned", err)
	}
	if sp := srv.contextSpans(ContextSpansRequest{TraceID: trace}).Spans[0]; !sp.Tombstoned || sp.Restorable {
		t.Fatalf("span Tombstoned=%v Restorable=%v, want true/false", sp.Tombstoned, sp.Restorable)
	}
}

// TestSealRestoreSuppressesHandle witnesses the SEAL counterpart: sealRestore(digest) flips the sealed
// gate, restoreContext refuses with ctxplan.ErrSealed, and contextSpans reports Sealed, Restorable=false.
// Seal has no shipped operator-by-digest action yet (see sealRestore's doc), so this drives the exposed
// method directly — the mechanism is fully wired and ready for that future operator seam.
func TestSealRestoreSuppressesHandle(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-gate-seal"
	taskBytes := []byte(`{"role":"user","content":"quarantine me"}`)
	id := ctxplan.Digest(taskBytes)
	srv.stashRestore(trace, id, "quarantine me", taskBytes)

	if n := srv.sealRestore(id); n != 1 {
		t.Fatalf("sealRestore flipped %d, want 1", n)
	}
	_, err := srv.restoreContext(ContextRestoreRequest{ID: id, TraceID: trace})
	if !errors.Is(err, ErrRestoreRefused) || !errors.Is(err, ctxplan.ErrSealed) {
		t.Fatalf("restore err = %v, want refused+ErrSealed", err)
	}
	if sp := srv.contextSpans(ContextSpansRequest{TraceID: trace}).Spans[0]; !sp.Sealed || sp.Restorable {
		t.Fatalf("span Sealed=%v Restorable=%v, want true/false", sp.Sealed, sp.Restorable)
	}
}

// TestGateRestoreDigestNotStashedIsNoop: a tombstone for a digest no trace stashed flips 0 and touches
// nothing — a valid no-op, not an error. An unrelated stashed handle stays restorable. A blank digest is
// likewise a 0 no-op.
func TestGateRestoreDigestNotStashedIsNoop(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-gate-noop"
	keepBytes := []byte(`{"role":"user","content":"keep me restorable"}`)
	keepID := ctxplan.Digest(keepBytes)
	srv.stashRestore(trace, keepID, "keep me", keepBytes)

	if n := srv.tombstoneRestore(ctxplan.Digest([]byte("never-stashed"))); n != 0 {
		t.Fatalf("tombstoneRestore of an unstashed digest flipped %d, want 0", n)
	}
	if n := srv.tombstoneRestore("   "); n != 0 {
		t.Fatalf("tombstoneRestore of a blank digest flipped %d, want 0", n)
	}
	// The unrelated handle is untouched.
	if _, err := srv.restoreContext(ContextRestoreRequest{ID: keepID, TraceID: trace}); err != nil {
		t.Fatalf("unrelated handle should still restore, got %v", err)
	}
}

// TestGateRestoreCrossTrace: a content-address is trace-agnostic, so the SAME digest stashed under two
// traces is suppressed under BOTH by ONE operator tombstone. This is the design invariant — suppressing
// a span must suppress it everywhere it is recoverable, or the handle leaks it back in under the trace
// the operator did not name.
func TestGateRestoreCrossTrace(t *testing.T) {
	srv := newTestServer(t)
	const traceA, traceB = "t-gate-A", "t-gate-B"
	shared := []byte(`{"role":"user","content":"the same dropped span, two traces"}`)
	id := ctxplan.Digest(shared)
	srv.stashRestore(traceA, id, "shared span", shared)
	srv.stashRestore(traceB, id, "shared span", shared)

	if n := srv.tombstoneRestore(id); n != 2 {
		t.Fatalf("cross-trace tombstone flipped %d handles, want 2 (one per trace)", n)
	}
	for _, trace := range []string{traceA, traceB} {
		_, err := srv.restoreContext(ContextRestoreRequest{ID: id, TraceID: trace})
		if !errors.Is(err, ErrRestoreRefused) || !errors.Is(err, ctxplan.ErrTombstoned) {
			t.Fatalf("trace %s restore err = %v, want refused+ErrTombstoned", trace, err)
		}
	}
}

// TestGateRestoreIdempotent: flipping an already-tombstoned handle keeps it set. A second tombstone of
// the same digest re-reports the same count and never un-suppresses the span.
func TestGateRestoreIdempotent(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-gate-idem"
	taskBytes := []byte(`{"role":"user","content":"suppress me twice"}`)
	id := ctxplan.Digest(taskBytes)
	srv.stashRestore(trace, id, "suppress me twice", taskBytes)

	if n := srv.tombstoneRestore(id); n != 1 {
		t.Fatalf("first tombstone flipped %d, want 1", n)
	}
	if n := srv.tombstoneRestore(id); n != 1 {
		t.Fatalf("second (idempotent) tombstone flipped %d, want 1 (still matched, still set)", n)
	}
	if _, err := srv.restoreContext(ContextRestoreRequest{ID: id, TraceID: trace}); !errors.Is(err, ctxplan.ErrTombstoned) {
		t.Fatalf("after a repeat tombstone restore err = %v, want still ErrTombstoned", err)
	}
}
