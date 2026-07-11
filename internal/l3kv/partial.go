package l3kv

// partial.go — PARTIAL prefix lookup (#3897). The durable tier's read path is
// deliberately all-or-nothing PER BLOCK: store.Get re-verifies each record's
// sha256 and refuses a corrupt/torn record with a typed FAULT, and a prefix
// reader built on it (warmresume.RestorePrefix) counts a bad block cold. That
// fail-closed guard is right — but at the PREFIX level it is lossy: a prefix
// that is 90% deliverable with one bad mid block reads as "recompute
// everything", because nothing tells the caller WHICH sub-ranges were still
// good. LookupPartial is the additive verb that closes that gap: it classifies
// an ordered prefix of blocks against the tier and returns the exact servable
// sub-ranges alongside the exact invalid sub-ranges, so a caller recomputes
// ONLY the invalid spans and serves the rest.
//
// Additive fence. Nothing on the existing read path changes: store.Get,
// RestoreSpan, and RestorePrefix keep their byte-identical fail-closed
// behavior (a torn record is still refused, never served). LookupPartial is a
// new read-side classification over the same per-block verdicts; it serves no
// bytes itself — the caller pages each servable block back through the
// existing integrity-verified Get/RestoreSpan.
//
// Follow-on (NOT here): wiring the enginecache fall-through to consume
// invalidRanges so the recompute really narrows to the disclaimed spans.

import (
	"context"
	"errors"
	"fmt"
)

// Range identifies one contiguous sub-range of a looked-up prefix in both of
// the package's addressing units: the block ids [FirstBlock, LastBlock]
// (inclusive indices into the ordered block list handed to LookupPartial) and
// the position offset span [From, From+Positions) those blocks cover within
// the prefix (the same offset/positions model Span and ManifestSpan use, with
// From counted from the start of the looked-up prefix).
type Range struct {
	FirstBlock int // block id of the first block in the run (index into the prefix's ordered blocks)
	LastBlock  int // block id of the last block in the run, inclusive
	From       int // starting position offset of the run within the prefix
	Positions  int // total positions the run covers
}

// errNilStore is the one hard misuse error: there is no tier to classify against.
var errNilStore = errors.New("l3kv: LookupPartial nil store")

// LookupPartial classifies the ordered prefix blocks against the durable tier
// and returns the exact servable and invalid sub-ranges, coalesced into
// maximal contiguous runs in prefix order. A block is SERVABLE when the tier
// holds it intact (Get returns found with the integrity check passing) and
// INVALID otherwise — a clean miss (never staged / reaped) and a refused
// torn/corrupt or faulted record both land in invalidRanges, because neither
// can be served. The two slices partition the prefix exactly: every block id
// and every position appears in precisely one range, so a caller recomputes
// ONLY invalidRanges and pages the servable blocks back through the existing
// integrity-verified read path.
//
// It never converts a whole prefix to a MISS on one bad block — that lossy
// downgrade is exactly what this verb exists to avoid — and it never serves a
// bad block: the per-block verdict is the store's own fail-closed Get.
//
// Errors are reserved for misuse and cancellation, not for bad blocks: a nil
// store and a negative block length are refused, and a context cancellation
// returns ctx's error rather than mislabeling the unread remainder invalid.
func LookupPartial(ctx context.Context, store Store, blocks []ManifestSpan) (servableRanges, invalidRanges []Range, err error) {
	if store == nil {
		return nil, nil, errNilStore
	}
	offset := 0
	for i, blk := range blocks {
		// Fail out on cancellation BEFORE classifying: a canceled Get would
		// error for every remaining block, and absorbing that would disclaim
		// spans the tier may well hold.
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		if blk.Positions < 0 {
			return nil, nil, fmt.Errorf("l3kv: LookupPartial block %d has negative positions (%d)", i, blk.Positions)
		}
		_, found, gerr := store.Get(ctx, blk.Digest)
		target := &invalidRanges
		if gerr == nil && found {
			target = &servableRanges
		}
		if n := len(*target); n > 0 && (*target)[n-1].LastBlock == i-1 {
			// Contiguous with the previous same-verdict run — extend it.
			(*target)[n-1].LastBlock = i
			(*target)[n-1].Positions += blk.Positions
		} else {
			*target = append(*target, Range{FirstBlock: i, LastBlock: i, From: offset, Positions: blk.Positions})
		}
		offset += blk.Positions
	}
	return servableRanges, invalidRanges, nil
}
