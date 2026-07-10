package gateway

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// ctxrestore_store.go — the GENERALIZED arm of fak_context_restore (issue #3062): the routing/adapter
// layer that lets a restore-by-digest call resolve bytes from any content-addressed ctxplan.Store, not
// just the compaction-tombstone stash in ctxrestore.go. Slice 1 (ctxrestore.go) served exactly one
// dropped-span source — the originating task the Anthropic passthrough compaction tombstones — under a
// per-trace digest→bytes stash. But the handle is content-addressed with the SAME sha256-hex scheme as
// ctxplan.Digest / recall / blob / memq, so the id space is already unified: a compaction handle and a
// ctxplan handle are the same address for the same bytes. What was missing was routing a restore call
// to whichever mechanism actually holds the bytes for that digest — this file is that router.
//
// The design axis (Law A2 — every value carries its owner): the router is deliberately SOURCE-AGNOSTIC.
// It does not switch on "is this a ctxview elision vs a recall page vs a memq cell"; it resolves against
// the ctxplan.Store INTERFACE, and content-addressing makes the source irrelevant — a digest either
// names a span the store holds or it does not. That is the whole point of one shared CAS: you look up by
// address, you do not route by provenance. recall.CtxStore is already a ctxplan.Store (Spans carries the
// page Digest, Materialize routes through the recall trust gate), the ctxplan planner's MemStore is one,
// and any future memq→ctxplan.Store adapter is one — so all three of the issue's named sources plug into
// this single adapter the moment a Store handle for them is in hand, with zero new routing code.
//
// Trust gate: the router NEVER bypasses the store's own gate. It materializes through Store.Materialize,
// which refuses a sealed span with ctxplan.ErrSealed and a suppressed one with ctxplan.ErrTombstoned;
// this file maps those to the SAME ErrRestoreRefused-wrapping-the-sentinel shape restoreContext already
// returns for the stash, so a caller's errors.Is branch is identical whichever source served the digest.
// "One trust gate" is satisfied by honoring each store's gate, not by inventing a second one.

// restoreFromStore resolves a restore-by-digest call against any ctxplan.Store: it finds the span whose
// content-address Digest equals the requested digest, then pages its bytes in THROUGH the store's trust
// gate. A store that holds no span at that digest is a MISS (ErrRestoreMiss) — the same "never had it"
// answer the stash gives — while a sealed/tombstoned span is a REFUSAL wrapping the ctxplan sentinel, so
// the router's refusal vocabulary is byte-for-byte the stash's. It returns the matched Span (for its safe
// Descriptor) alongside the bytes so restoreContext can echo "what it is" exactly as the stash path does.
//
// A nil store or a blank digest is a safe MISS: a content-address that names nothing can resolve nothing.
// The digest is trimmed so an id carrying incidental whitespace still matches the exact-string address.
func restoreFromStore(ctx context.Context, store ctxplan.Store, digest string) (ctxplan.Span, []byte, error) {
	digest = strings.TrimSpace(digest)
	if store == nil || digest == "" {
		return ctxplan.Span{}, nil, ErrRestoreMiss
	}
	spans, err := store.Spans(ctx)
	if err != nil {
		return ctxplan.Span{}, nil, err
	}
	for _, sp := range spans {
		if sp.Digest != digest {
			continue
		}
		body, err := store.Materialize(ctx, sp.ID)
		if err != nil {
			switch {
			case errors.Is(err, ctxplan.ErrSealed):
				return sp, nil, errWrap(ErrRestoreRefused, ctxplan.ErrSealed)
			case errors.Is(err, ctxplan.ErrTombstoned):
				return sp, nil, errWrap(ErrRestoreRefused, ctxplan.ErrTombstoned)
			default:
				return sp, nil, err
			}
		}
		return sp, body, nil
	}
	return ctxplan.Span{}, nil, ErrRestoreMiss
}

// restoreFromImage resolves a restore-by-digest call against a persisted recall core image: it loads the
// image into a Session, wraps it as a recall.CtxStore (a ctxplan.Store whose Spans carry the recall page
// Digest and whose Materialize routes through the recall trust gate), and routes the digest through the
// source-agnostic restoreFromStore adapter. This is the concrete second source #3062 wires end to end:
// a recall PAGE addressed by its recall digest pages back in through fak_context_restore, under the recall
// image's own seal/tombstone gate — the same recall.Load pattern the shipped contextChange path uses to
// reach a core image by dir. A load error surfaces verbatim (the caller named a bad image); a digest the
// image does not hold is a plain miss.
func (s *Server) restoreFromImage(ctx context.Context, imageDir, digest string) (ctxplan.Span, []byte, error) {
	sess, err := recall.Load(imageDir)
	if err != nil {
		return ctxplan.Span{}, nil, fmt.Errorf("load core image: %w", err)
	}
	return restoreFromStore(ctx, recall.NewCtxStore(sess), digest)
}
