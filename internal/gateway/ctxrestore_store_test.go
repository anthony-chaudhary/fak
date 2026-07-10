package gateway

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// ctxrestore_store_test.go — witnesses for the generalized restore-by-digest routing (#3062): the
// source-agnostic ctxplan.Store adapter resolves a benign span by its digest, refuses a sealed one
// through the store's own gate, and misses on an unknown digest; and restoreContext routes a
// compaction-stash MISS to a real recall core image so a recall PAGE pages back in by its recall digest
// — refused once an operator tombstones it. Proves the id space is unified across sources under one gate.

// TestRestoreFromStoreResolvesByDigest: the generic adapter finds a span by content-address in ANY
// ctxplan.Store (here the canonical MemStore — interface-identical to recall.CtxStore and the ctxview
// planner store) and pages its verbatim bytes in.
func TestRestoreFromStoreResolvesByDigest(t *testing.T) {
	store := ctxplan.NewMemStore()
	body := []byte(`{"role":"user","content":"the elided ctxview span"}`)
	span := store.Add("user", ctxplan.DurabilitySession, body, false)

	sp, got, err := restoreFromStore(context.Background(), store, span.Digest)
	if err != nil {
		t.Fatalf("restoreFromStore: %v", err)
	}
	if string(got) != string(body) {
		t.Fatalf("restored bytes = %q, want %q", got, body)
	}
	if sp.Digest != span.Digest {
		t.Fatalf("matched span digest = %q, want %q", sp.Digest, span.Digest)
	}
}

// TestRestoreFromStoreRefusesSealed: a sealed span is REFUSED through the store's trust gate, surfaced
// as ErrRestoreRefused wrapping ctxplan.ErrSealed — the same refusal vocabulary the tombstone stash
// returns, so a caller's errors.Is branch is identical whichever source held the digest.
func TestRestoreFromStoreRefusesSealed(t *testing.T) {
	store := ctxplan.NewMemStore()
	span := store.Add("tool", ctxplan.DurabilityBounded, []byte("quarantined bytes"), true)

	_, _, err := restoreFromStore(context.Background(), store, span.Digest)
	if !errors.Is(err, ErrRestoreRefused) {
		t.Fatalf("sealed span: want ErrRestoreRefused, got %v", err)
	}
	if !errors.Is(err, ctxplan.ErrSealed) {
		t.Fatalf("sealed span: want the ctxplan.ErrSealed sentinel, got %v", err)
	}
}

// TestRestoreFromStoreMissUnknownDigest: a digest the store does not hold is a MISS, distinct from a
// refusal — the "never had it" signal restoreContext branches on to keep falling through sources.
func TestRestoreFromStoreMissUnknownDigest(t *testing.T) {
	store := ctxplan.NewMemStore()
	store.Add("user", ctxplan.DurabilitySession, []byte("present"), false)

	if _, _, err := restoreFromStore(context.Background(), store, ctxplan.Digest([]byte("absent"))); !errors.Is(err, ErrRestoreMiss) {
		t.Fatalf("unknown digest: want ErrRestoreMiss, got %v", err)
	}
	// A nil store or blank digest is a safe miss, never a panic.
	if _, _, err := restoreFromStore(context.Background(), nil, "x"); !errors.Is(err, ErrRestoreMiss) {
		t.Fatalf("nil store: want ErrRestoreMiss, got %v", err)
	}
	if _, _, err := restoreFromStore(context.Background(), store, "   "); !errors.Is(err, ErrRestoreMiss) {
		t.Fatalf("blank digest: want ErrRestoreMiss, got %v", err)
	}
}

// TestRestoreContextRoutesToRecallImage: end-to-end — a digest the compaction-tombstone stash never
// held pages back in from a persisted recall core image when the caller names it, proving restore is no
// longer scoped to the single compaction source. The bytes are the recall page's verbatim body.
func TestRestoreContextRoutesToRecallImage(t *testing.T) {
	srv := newTestServer(t)
	dir, digest := writeRecallImage(t, "sess-restore-recall")

	got, err := srv.restoreContext(ContextRestoreRequest{ID: digest, TraceID: "t-recall", ImageDir: dir})
	if err != nil {
		t.Fatalf("restore from recall image: %v", err)
	}
	if !strings.Contains(got.Bytes, "prefer short answers") {
		t.Fatalf("restored recall page bytes = %q, want the recorded body", got.Bytes)
	}
	if got.ID != digest || got.Provenance != "WITNESSED" {
		t.Fatalf("restore identity = id %q provenance %q, want %q / WITNESSED", got.ID, got.Provenance, digest)
	}
}

// TestRestoreContextRefusesTombstonedRecallPage: once an operator tombstones the recall page (context
// control), its restore handle is REFUSED through the image's own gate — restore recovers a dropped
// span, never a suppressed one — surfaced as ErrRestoreRefused wrapping ctxplan.ErrTombstoned.
func TestRestoreContextRefusesTombstonedRecallPage(t *testing.T) {
	srv := newTestServer(t)
	dir, digest := writeRecallImage(t, "sess-restore-recall-tomb")

	if _, err := srv.contextChange(context.Background(), ContextChangeRequest{ImageDir: dir, Action: "tombstone", Step: 0, Reason: "suppress the recalled page for this witness"}); err != nil {
		t.Fatalf("tombstone recall page: %v", err)
	}

	_, err := srv.restoreContext(ContextRestoreRequest{ID: digest, TraceID: "t-recall-tomb", ImageDir: dir})
	if !errors.Is(err, ErrRestoreRefused) {
		t.Fatalf("tombstoned recall page: want ErrRestoreRefused, got %v", err)
	}
	if !errors.Is(err, ctxplan.ErrTombstoned) {
		t.Fatalf("tombstoned recall page: want the ctxplan.ErrTombstoned sentinel, got %v", err)
	}
}

// TestRestoreContextStashHitBeatsImage: the compaction-tombstone stash is authoritative — a stash hit
// is returned even when the caller also names an image, so the fall-through never shadows the anchor
// source or double-resolves a digest both hold.
func TestRestoreContextStashHitBeatsImage(t *testing.T) {
	srv := newTestServer(t)
	dir, _ := writeRecallImage(t, "sess-restore-stash-priority")
	const trace = "t-stash-priority"
	taskBytes := []byte(`{"role":"user","content":"the stashed originating task"}`)
	id := ctxplan.Digest(taskBytes)
	srv.stashRestore(trace, id, "the stashed originating task", taskBytes)

	got, err := srv.restoreContext(ContextRestoreRequest{ID: id, TraceID: trace, ImageDir: dir})
	if err != nil {
		t.Fatalf("restore stash hit with image set: %v", err)
	}
	if got.Bytes != string(taskBytes) {
		t.Fatalf("restored bytes = %q, want the stashed task %q", got.Bytes, taskBytes)
	}
}
