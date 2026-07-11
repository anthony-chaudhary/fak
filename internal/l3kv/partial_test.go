package l3kv

// partial_test.go — witnesses for LookupPartial (#3897): a torn block narrows
// the recompute to its exact sub-range instead of downgrading the whole prefix
// to a MISS, while the existing per-block fail-closed read path stays intact.

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// blockDigest returns a distinct valid span-digest-shaped key (64 hex chars)
// for block i of a test prefix.
func blockDigest(i int) string { return fmt.Sprintf("%064x", i+1) }

// putPrefix stages one block per length in lens into store, keyed
// blockDigest(0..n-1), and returns the matching ordered manifest blocks.
func putPrefix(t *testing.T, ctx context.Context, store Store, lens ...int) []ManifestSpan {
	t.Helper()
	blocks := make([]ManifestSpan, 0, len(lens))
	for i, n := range lens {
		if err := store.Put(ctx, blockDigest(i), spanBytes(n+1)); err != nil {
			t.Fatalf("Put block %d: %v", i, err)
		}
		blocks = append(blocks, ManifestSpan{Digest: blockDigest(i), Positions: n})
	}
	return blocks
}

// corruptRecord flips the last on-disk byte of the record under key — a payload
// byte past the header, so store.Get refuses it as an integrity FAULT.
func corruptRecord(t *testing.T, dir, key string) {
	t.Helper()
	p := filepath.Join(dir, key)
	raw, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read record %s: %v", key, err)
	}
	raw[len(raw)-1] ^= 0xFF
	if err := os.WriteFile(p, raw, 0o644); err != nil {
		t.Fatalf("rewrite record %s: %v", key, err)
	}
}

// TestLookupPartialMidCorruptBlock is the #3897 headline witness: a prefix with
// one corrupt MID block yields exactly two servable ranges bracketing exactly
// one invalid range — NOT a whole-prefix miss — and each range names the
// precise block-id and offset span, so a caller recomputes only the bad span.
func TestLookupPartialMidCorruptBlock(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := newDiskStore(dir)
	if err != nil {
		t.Fatalf("newDiskStore: %v", err)
	}
	blocks := putPrefix(t, ctx, store, 10, 20, 30, 40, 50)
	corruptRecord(t, dir, blockDigest(2))

	servable, invalid, err := LookupPartial(ctx, store, blocks)
	if err != nil {
		t.Fatalf("LookupPartial: %v", err)
	}
	wantServable := []Range{
		{FirstBlock: 0, LastBlock: 1, From: 0, Positions: 30},
		{FirstBlock: 3, LastBlock: 4, From: 60, Positions: 90},
	}
	wantInvalid := []Range{
		{FirstBlock: 2, LastBlock: 2, From: 30, Positions: 30},
	}
	if !reflect.DeepEqual(servable, wantServable) {
		t.Fatalf("servable = %+v, want %+v", servable, wantServable)
	}
	if !reflect.DeepEqual(invalid, wantInvalid) {
		t.Fatalf("invalid = %+v, want %+v", invalid, wantInvalid)
	}

	// Refute guard: the existing per-block torn-read path is untouched — the
	// corrupt block is still a refused FAULT on Get, never served, while its
	// intact neighbors still read back clean.
	if _, found, err := store.Get(ctx, blockDigest(2)); err == nil {
		t.Fatalf("corrupt block Get returned no error (found=%v) — integrity guard changed", found)
	}
	if _, found, err := store.Get(ctx, blockDigest(1)); err != nil || !found {
		t.Fatalf("intact block Get = (found=%v, err=%v), want clean hit", found, err)
	}
}

// TestLookupPartialAllGood proves the no-damage shape: every block intact
// yields ONE servable range spanning the whole prefix and zero invalid ranges.
func TestLookupPartialAllGood(t *testing.T) {
	ctx := context.Background()
	store := newMemStore()
	blocks := putPrefix(t, ctx, store, 5, 15, 25, 35)

	servable, invalid, err := LookupPartial(ctx, store, blocks)
	if err != nil {
		t.Fatalf("LookupPartial: %v", err)
	}
	wantServable := []Range{{FirstBlock: 0, LastBlock: 3, From: 0, Positions: 80}}
	if !reflect.DeepEqual(servable, wantServable) {
		t.Fatalf("servable = %+v, want %+v", servable, wantServable)
	}
	if len(invalid) != 0 {
		t.Fatalf("invalid = %+v, want none", invalid)
	}
}

// TestLookupPartialAllBad proves the total-loss shape: no block deliverable
// (never staged) yields zero servable ranges and ONE invalid range covering
// the whole prefix — a precise disclaimer, not a silent recompute.
func TestLookupPartialAllBad(t *testing.T) {
	ctx := context.Background()
	store := newMemStore() // nothing staged: every block is a clean miss
	blocks := []ManifestSpan{
		{Digest: blockDigest(0), Positions: 10},
		{Digest: blockDigest(1), Positions: 20},
		{Digest: blockDigest(2), Positions: 30},
	}

	servable, invalid, err := LookupPartial(ctx, store, blocks)
	if err != nil {
		t.Fatalf("LookupPartial: %v", err)
	}
	if len(servable) != 0 {
		t.Fatalf("servable = %+v, want none", servable)
	}
	wantInvalid := []Range{{FirstBlock: 0, LastBlock: 2, From: 0, Positions: 60}}
	if !reflect.DeepEqual(invalid, wantInvalid) {
		t.Fatalf("invalid = %+v, want %+v", invalid, wantInvalid)
	}
}

// TestLookupPartialLeadingAndTrailingBad pins the boundary handling: a bad
// FIRST block and a bad LAST block each split cleanly at the prefix edge, with
// offsets and block ids exact on both sides.
func TestLookupPartialLeadingAndTrailingBad(t *testing.T) {
	ctx := context.Background()

	t.Run("leading-bad", func(t *testing.T) {
		dir := t.TempDir()
		store, err := newDiskStore(dir)
		if err != nil {
			t.Fatalf("newDiskStore: %v", err)
		}
		blocks := putPrefix(t, ctx, store, 10, 20, 30)
		corruptRecord(t, dir, blockDigest(0))

		servable, invalid, err := LookupPartial(ctx, store, blocks)
		if err != nil {
			t.Fatalf("LookupPartial: %v", err)
		}
		wantInvalid := []Range{{FirstBlock: 0, LastBlock: 0, From: 0, Positions: 10}}
		wantServable := []Range{{FirstBlock: 1, LastBlock: 2, From: 10, Positions: 50}}
		if !reflect.DeepEqual(invalid, wantInvalid) {
			t.Fatalf("invalid = %+v, want %+v", invalid, wantInvalid)
		}
		if !reflect.DeepEqual(servable, wantServable) {
			t.Fatalf("servable = %+v, want %+v", servable, wantServable)
		}
	})

	t.Run("trailing-bad", func(t *testing.T) {
		dir := t.TempDir()
		store, err := newDiskStore(dir)
		if err != nil {
			t.Fatalf("newDiskStore: %v", err)
		}
		blocks := putPrefix(t, ctx, store, 10, 20, 30)
		corruptRecord(t, dir, blockDigest(2))

		servable, invalid, err := LookupPartial(ctx, store, blocks)
		if err != nil {
			t.Fatalf("LookupPartial: %v", err)
		}
		wantServable := []Range{{FirstBlock: 0, LastBlock: 1, From: 0, Positions: 30}}
		wantInvalid := []Range{{FirstBlock: 2, LastBlock: 2, From: 30, Positions: 30}}
		if !reflect.DeepEqual(servable, wantServable) {
			t.Fatalf("servable = %+v, want %+v", servable, wantServable)
		}
		if !reflect.DeepEqual(invalid, wantInvalid) {
			t.Fatalf("invalid = %+v, want %+v", invalid, wantInvalid)
		}
	})
}

// TestLookupPartialAlternatingRuns pins the coalescing rule: runs extend only
// across CONTIGUOUS same-verdict blocks, so good,bad,good,bad never merges
// across the gaps.
func TestLookupPartialAlternatingRuns(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := newDiskStore(dir)
	if err != nil {
		t.Fatalf("newDiskStore: %v", err)
	}
	blocks := putPrefix(t, ctx, store, 10, 20, 30, 40)
	corruptRecord(t, dir, blockDigest(1))
	corruptRecord(t, dir, blockDigest(3))

	servable, invalid, err := LookupPartial(ctx, store, blocks)
	if err != nil {
		t.Fatalf("LookupPartial: %v", err)
	}
	wantServable := []Range{
		{FirstBlock: 0, LastBlock: 0, From: 0, Positions: 10},
		{FirstBlock: 2, LastBlock: 2, From: 30, Positions: 30},
	}
	wantInvalid := []Range{
		{FirstBlock: 1, LastBlock: 1, From: 10, Positions: 20},
		{FirstBlock: 3, LastBlock: 3, From: 60, Positions: 40},
	}
	if !reflect.DeepEqual(servable, wantServable) {
		t.Fatalf("servable = %+v, want %+v", servable, wantServable)
	}
	if !reflect.DeepEqual(invalid, wantInvalid) {
		t.Fatalf("invalid = %+v, want %+v", invalid, wantInvalid)
	}
}

// TestLookupPartialMisuseAndEmpty pins the reserved error surface: a nil store
// and a negative block length are refused, and an empty prefix classifies to
// nothing on both sides without error.
func TestLookupPartialMisuseAndEmpty(t *testing.T) {
	ctx := context.Background()

	if _, _, err := LookupPartial(ctx, nil, nil); err == nil {
		t.Fatal("nil store: want error, got nil")
	}
	if _, _, err := LookupPartial(ctx, newMemStore(), []ManifestSpan{{Digest: blockDigest(0), Positions: -1}}); err == nil {
		t.Fatal("negative positions: want error, got nil")
	}

	servable, invalid, err := LookupPartial(ctx, newMemStore(), nil)
	if err != nil {
		t.Fatalf("empty prefix: %v", err)
	}
	if len(servable) != 0 || len(invalid) != 0 {
		t.Fatalf("empty prefix classified to servable=%+v invalid=%+v, want none", servable, invalid)
	}
}
