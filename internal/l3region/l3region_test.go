package l3region

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// payload builds a deterministic n-byte body that spans several PageBytes pages, so a
// round-trip exercises the multi-page chunk/mset/mget/reassemble path, not one page.
func payload(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte((i*31 + 7) & 0xFF)
	}
	return b
}

// TestRegionRoundTripsBitExact is Stage 1's headline: a multi-page region Put into the
// fake L3 and Resolved back is byte-identical (max|Δ|=0), and the Ref is a RefRegion
// whose Digest is the whole-region content address and whose Len is the payload length.
func TestRegionRoundTripsBitExact(t *testing.T) {
	ctx := context.Background()
	be := New(NewL3Store())
	want := payload(PageBytes*2 + 123) // 3 pages: two full + a partial tail

	ref, err := be.Put(ctx, want)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if ref.Kind != abi.RefRegion {
		t.Fatalf("Put returned RefKind %d, want RefRegion", ref.Kind)
	}
	if ref.Digest != digest(want) {
		t.Fatalf("Ref.Digest = %s, want content address %s", ref.Digest, digest(want))
	}
	if ref.Len != int64(len(want)) {
		t.Fatalf("Ref.Len = %d, want %d", ref.Len, len(want))
	}
	got, err := be.Resolve(ctx, ref)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("round-trip not bit-exact: max|Δ|!=0 (got %d bytes, want %d)", len(got), len(want))
	}
}

// TestRefResolvesToPageKeySet proves the second Stage-1 property: a Ref resolves to a
// SET of L3 page keys. The set has the expected page count, every key is resident in
// the store, each page's bytes content-address to its key, and concatenating the pages
// in order reproduces the payload.
func TestRefResolvesToPageKeySet(t *testing.T) {
	ctx := context.Background()
	store := NewL3Store()
	be := New(store)
	want := payload(PageBytes*3 + 1) // 4 pages

	ref, err := be.Put(ctx, want)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	keys, err := be.PageKeys(ref)
	if err != nil {
		t.Fatalf("PageKeys: %v", err)
	}
	wantPages := (len(want) + PageBytes - 1) / PageBytes
	if len(keys) != wantPages {
		t.Fatalf("page-key set has %d keys, want %d (ceil(%d/%d))", len(keys), wantPages, len(want), PageBytes)
	}
	pages, missing, ok := store.Mget(keys)
	if !ok {
		t.Fatalf("page key %s in the set is not resident in L3", missing)
	}
	var reassembled []byte
	for i, pb := range pages {
		if digest(pb) != keys[i] {
			t.Fatalf("page %d is not content-addressed by its key", i)
		}
		reassembled = append(reassembled, pb...)
	}
	if !bytes.Equal(reassembled, want) {
		t.Fatal("concatenating the page-key set's pages did not reproduce the region")
	}
}

// TestResolveVerifiesPages is the "verify, don't trust" thesis (G1): if the L3 store
// hands back a page that does NOT match its claimed key (a lying / corrupted connector),
// Resolve REFUSES with a verify error rather than returning the substituted bytes.
func TestResolveVerifiesPages(t *testing.T) {
	ctx := context.Background()
	store := NewL3Store()
	be := New(store)
	want := payload(PageBytes + 10) // 2 pages
	ref, err := be.Put(ctx, want)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	keys, err := be.PageKeys(ref)
	if err != nil {
		t.Fatalf("PageKeys: %v", err)
	}
	// Simulate a lying connector: mutate the resident bytes under a key in place, so the
	// key still "exists" but its bytes no longer hash to it.
	store.mu.Lock()
	store.pages[keys[0]][0] ^= 0xFF
	store.mu.Unlock()

	got, err := be.Resolve(ctx, ref)
	if err == nil {
		t.Fatalf("Resolve returned %d bytes for a corrupted page; it must refuse (verify, don't trust)", len(got))
	}
	if !strings.Contains(err.Error(), "verify failed") {
		t.Fatalf("Resolve error = %q, want a page/region verify failure", err)
	}
}

// TestEvictedPagesNoLongerResolve proves the Stage-2 substrate is real: invalidating a
// region's backing page keys (Mdel — the mechanism KVCache.Evict will drive) makes the
// Ref resolve to a typed MISS rather than a half-region or a stale read.
func TestEvictedPagesNoLongerResolve(t *testing.T) {
	ctx := context.Background()
	store := NewL3Store()
	be := New(store)
	// Two DISTINCT full pages. The deterministic payload pattern has period 256, which
	// divides PageBytes, so two equal-length full pages are byte-identical and content-
	// address to ONE key (the dedup TestIdenticalContentDedups relies on). Perturb the
	// head of page 2 so the region has two distinct page keys — then "every backing page
	// key was invalidated" is a real 2-of-2 removal, not a 1-key dedup coincidence.
	body := payload(PageBytes * 2)
	body[PageBytes] ^= 0xFF
	ref, err := be.Put(ctx, body)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	keys, err := be.PageKeys(ref)
	if err != nil {
		t.Fatalf("PageKeys: %v", err)
	}
	if n := store.Mdel(keys); n != len(keys) {
		t.Fatalf("Mdel removed %d pages, want %d", n, len(keys))
	}
	_, err = be.Resolve(ctx, ref)
	if !errors.Is(err, ErrMiss) {
		t.Fatalf("Resolve after invalidation = %v, want ErrMiss", err)
	}
}

// TestIdenticalContentDedups proves content addressing: two Puts of identical bytes
// produce the same Digest and the same page-key set (one set of L3 pages, not two).
func TestIdenticalContentDedups(t *testing.T) {
	ctx := context.Background()
	store := NewL3Store()
	be := New(store)
	body := payload(PageBytes + 64)

	r1, err := be.Put(ctx, body)
	if err != nil {
		t.Fatalf("Put#1: %v", err)
	}
	setsAfterFirst, _, _ := store.Stats()
	r2, err := be.Put(ctx, body)
	if err != nil {
		t.Fatalf("Put#2: %v", err)
	}
	if r1.Digest != r2.Digest {
		t.Fatalf("identical content got different digests: %s vs %s", r1.Digest, r2.Digest)
	}
	k1, _ := be.PageKeys(r1)
	k2, _ := be.PageKeys(r2)
	if strings.Join(k1, ",") != strings.Join(k2, ",") {
		t.Fatal("identical content resolved to different page-key sets")
	}
	// The second Put re-issues Mset for the same keys, but no NEW pages land (idempotent):
	// the resident set is unchanged, proving dedup at the L3 tier.
	if got := residentPages(store); got != len(k1) {
		t.Fatalf("L3 holds %d pages after two identical Puts, want %d (deduped)", got, len(k1))
	}
	_ = setsAfterFirst
}

// TestInlineAndEmpty covers the boundary refs: an inline Ref resolves to its own bytes,
// and an empty region round-trips to an empty body (zero pages).
func TestInlineAndEmpty(t *testing.T) {
	ctx := context.Background()
	be := New(NewL3Store())

	inlineBytes := []byte("inline tool args")
	inline := abi.Ref{Kind: abi.RefInline, Inline: inlineBytes, Len: int64(len(inlineBytes))}
	got, err := be.Resolve(ctx, inline)
	if err != nil || !bytes.Equal(got, inlineBytes) {
		t.Fatalf("inline Resolve = %q err=%v, want %q", got, err, inlineBytes)
	}

	ref, err := be.Put(ctx, nil)
	if err != nil {
		t.Fatalf("Put(empty): %v", err)
	}
	keys, err := be.PageKeys(ref)
	if err != nil {
		t.Fatalf("PageKeys(empty): %v", err)
	}
	if len(keys) != 0 {
		t.Fatalf("empty region has %d page keys, want 0", len(keys))
	}
	body, err := be.Resolve(ctx, ref)
	if err != nil {
		t.Fatalf("Resolve(empty): %v", err)
	}
	if len(body) != 0 {
		t.Fatalf("empty region resolved to %d bytes, want 0", len(body))
	}
}

// TestBackendSatisfiesFrozenSeam proves the backend attaches to the frozen ABI exactly
// like the other RegionBackends: Resolver() returns the backend and Caps() advertises
// the L3-tier capability Kernel.Negotiate intersects.
func TestBackendSatisfiesFrozenSeam(t *testing.T) {
	be := New(NewL3Store())
	if be.Resolver() != abi.Resolver(be) {
		t.Fatal("Resolver() must return the backend itself")
	}
	caps := be.Caps()
	if len(caps) != 1 || caps[0] != CapL3Tier {
		t.Fatalf("Caps() = %v, want [%s]", caps, CapL3Tier)
	}
	// A region this backend never minted is a typed miss, not a panic.
	if _, err := be.Resolve(context.Background(), abi.Ref{Kind: abi.RefRegion, Digest: "deadbeef"}); !errors.Is(err, ErrMiss) {
		t.Fatalf("Resolve of an unknown region = %v, want ErrMiss", err)
	}
}

// residentPages counts the distinct pages currently held by the fake L3 (white-box).
func residentPages(s *L3Store) int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.pages)
}
