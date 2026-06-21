package blob

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// makeBytes returns a deterministic byte slice of length n.
func makeBytes(n int, seed byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i)*31 + seed
	}
	return b
}

// TestPutSmallInlineRoundTrip covers unit 64: a payload <= InlineMax stays inline
// (RefInline, bytes carried on the Ref) and Resolve returns byte-identical bytes
// without touching the CAS.
func TestPutSmallInlineRoundTrip(t *testing.T) {
	ctx := context.Background()
	s := New()

	for _, n := range []int{0, 1, 100, InlineMax} {
		small := makeBytes(n, 7)
		r, err := s.Put(ctx, small)
		if err != nil {
			t.Fatalf("Put(len=%d): unexpected error %v", n, err)
		}
		if r.Kind != abi.RefInline {
			t.Fatalf("Put(len=%d): Kind=%d, want RefInline(%d)", n, r.Kind, abi.RefInline)
		}
		if r.Len != int64(n) {
			t.Fatalf("Put(len=%d): Ref.Len=%d, want %d", n, r.Len, n)
		}
		if r.Digest != Digest(small) {
			t.Fatalf("Put(len=%d): Ref.Digest=%q, want %q", n, r.Digest, Digest(small))
		}
		// Inline Refs default to the fail-closed provenance baseline.
		if r.Taint != abi.TaintTainted {
			t.Fatalf("Put(len=%d): Taint=%d, want TaintTainted", n, r.Taint)
		}
		if r.Scope != abi.ScopeAgent {
			t.Fatalf("Put(len=%d): Scope=%d, want ScopeAgent", n, r.Scope)
		}
		got, err := s.Resolve(ctx, r)
		if err != nil {
			t.Fatalf("Resolve(len=%d): unexpected error %v", n, err)
		}
		if !bytes.Equal(got, small) {
			t.Fatalf("Resolve(len=%d): bytes not identical", n)
		}
	}

	// An inline Put never touches the CAS, so stats stay at zero.
	puts, hits, _ := s.Stats()
	if puts != 0 || hits != 0 {
		t.Fatalf("inline Put touched CAS: puts=%d hits=%d, want 0,0", puts, hits)
	}
}

// TestPutLargeBlobRoundTrip covers unit 64: a payload > InlineMax is stored in the
// CAS (RefBlob, no inline bytes) and Resolve returns byte-identical bytes.
func TestPutLargeBlobRoundTrip(t *testing.T) {
	ctx := context.Background()
	s := New()

	large := makeBytes(InlineMax+1, 11)
	r, err := s.Put(ctx, large)
	if err != nil {
		t.Fatalf("Put(large): unexpected error %v", err)
	}
	if r.Kind != abi.RefBlob {
		t.Fatalf("Put(large): Kind=%d, want RefBlob(%d)", r.Kind, abi.RefBlob)
	}
	if r.Inline != nil {
		t.Fatalf("Put(large): Inline should be nil for a CAS-backed Ref, got %d bytes", len(r.Inline))
	}
	if r.Len != int64(len(large)) {
		t.Fatalf("Put(large): Ref.Len=%d, want %d", r.Len, len(large))
	}
	if r.Digest != Digest(large) {
		t.Fatalf("Put(large): Ref.Digest=%q, want %q", r.Digest, Digest(large))
	}

	got, err := s.Resolve(ctx, r)
	if err != nil {
		t.Fatalf("Resolve(large): unexpected error %v", err)
	}
	if !bytes.Equal(got, large) {
		t.Fatalf("Resolve(large): bytes not identical")
	}

	// One CAS put, no dedup hit, one resolve.
	puts, hits, resolves := s.Stats()
	if puts != 1 || hits != 0 || resolves != 1 {
		t.Fatalf("Stats after one large Put+Resolve = (%d,%d,%d), want (1,0,1)", puts, hits, resolves)
	}
}

// TestResolveIsACopy confirms Resolve hands back an independent copy, not aliasing
// the stored buffer (a mutation of the returned slice must not corrupt the store).
func TestResolveIsACopy(t *testing.T) {
	ctx := context.Background()
	s := New()
	large := makeBytes(InlineMax+50, 3)
	r, err := s.Put(ctx, large)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, err := s.Resolve(ctx, r)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	for i := range got {
		got[i] ^= 0xFF
	}
	again, err := s.Resolve(ctx, r)
	if err != nil {
		t.Fatalf("Resolve(again): %v", err)
	}
	if !bytes.Equal(again, large) {
		t.Fatalf("Resolve returned an aliased buffer: mutation leaked into the store")
	}
}

// TestPageOutPageInInlineRoundTrip covers unit 64: PageOut of an inline Ref moves
// its bytes into the CAS and returns a bytes-free handle (RefBlob); PageIn
// re-materializes a byte-identical inline Ref.
func TestPageOutPageInInlineRoundTrip(t *testing.T) {
	ctx := context.Background()
	s := New()

	// Start from an inline Ref (small payload).
	payload := makeBytes(64, 5)
	inline, err := s.Put(ctx, payload)
	if err != nil {
		t.Fatalf("Put(inline): %v", err)
	}
	if inline.Kind != abi.RefInline {
		t.Fatalf("precondition: want RefInline, got Kind=%d", inline.Kind)
	}
	inline.Taint = abi.TaintQuarantined // exercise provenance propagation
	inline.Scope = abi.ScopeFleet

	handle, err := s.PageOut(ctx, inline)
	if err != nil {
		t.Fatalf("PageOut(inline): %v", err)
	}
	if handle.Kind != abi.RefBlob {
		t.Fatalf("PageOut: Kind=%d, want RefBlob(%d)", handle.Kind, abi.RefBlob)
	}
	if handle.Inline != nil {
		t.Fatalf("PageOut: handle carries inline bytes, want none")
	}
	if handle.Digest != Digest(payload) {
		t.Fatalf("PageOut: Digest=%q, want %q", handle.Digest, Digest(payload))
	}
	if handle.Len != int64(len(payload)) {
		t.Fatalf("PageOut: Len=%d, want %d", handle.Len, len(payload))
	}
	if handle.Taint != abi.TaintQuarantined || handle.Scope != abi.ScopeFleet {
		t.Fatalf("PageOut: provenance not propagated: taint=%d scope=%d", handle.Taint, handle.Scope)
	}

	back, err := s.PageIn(ctx, handle)
	if err != nil {
		t.Fatalf("PageIn: %v", err)
	}
	if back.Kind != abi.RefInline {
		t.Fatalf("PageIn: Kind=%d, want RefInline(%d)", back.Kind, abi.RefInline)
	}
	if !bytes.Equal(back.Inline, payload) {
		t.Fatalf("PageIn: bytes not identical to original payload")
	}
	if back.Digest != handle.Digest {
		t.Fatalf("PageIn: Digest=%q, want %q", back.Digest, handle.Digest)
	}
	if back.Len != int64(len(payload)) {
		t.Fatalf("PageIn: Len=%d, want %d", back.Len, len(payload))
	}
	if back.Taint != abi.TaintQuarantined || back.Scope != abi.ScopeFleet {
		t.Fatalf("PageIn: provenance not propagated: taint=%d scope=%d", back.Taint, back.Scope)
	}

	// Full round trip: the re-materialized bytes resolve byte-identically too.
	got, err := s.Resolve(ctx, back)
	if err != nil {
		t.Fatalf("Resolve(pagedIn): %v", err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("Resolve(pagedIn): bytes not identical")
	}
}

// TestContentDedup covers the dedup unit: Putting byte-identical large payloads
// twice records a dedupHit and stores exactly one blob.
func TestContentDedup(t *testing.T) {
	ctx := context.Background()
	s := New()

	large := makeBytes(1024, 13)

	r1, err := s.Put(ctx, large)
	if err != nil {
		t.Fatalf("Put #1: %v", err)
	}
	puts, hits, _ := s.Stats()
	if puts != 1 || hits != 0 {
		t.Fatalf("after Put #1: puts=%d hits=%d, want 1,0", puts, hits)
	}

	// Use a distinct backing array with identical content to prove dedup is by
	// content, not by pointer.
	dup := append([]byte(nil), large...)
	r2, err := s.Put(ctx, dup)
	if err != nil {
		t.Fatalf("Put #2: %v", err)
	}
	puts, hits, _ = s.Stats()
	if puts != 2 {
		t.Fatalf("after Put #2: puts=%d, want 2", puts)
	}
	if hits != 1 {
		t.Fatalf("after Put #2: dedupHits=%d, want 1 (identical content must dedup)", hits)
	}
	if r1.Digest != r2.Digest {
		t.Fatalf("identical content produced different digests: %q vs %q", r1.Digest, r2.Digest)
	}

	// Exactly one blob is physically stored.
	s.mu.RLock()
	stored := len(s.blobs)
	s.mu.RUnlock()
	if stored != 1 {
		t.Fatalf("CAS stored %d blobs, want 1 (content dedup)", stored)
	}

	// Both Refs resolve to the same identical bytes.
	g1, err := s.Resolve(ctx, r1)
	if err != nil {
		t.Fatalf("Resolve(r1): %v", err)
	}
	g2, err := s.Resolve(ctx, r2)
	if err != nil {
		t.Fatalf("Resolve(r2): %v", err)
	}
	if !bytes.Equal(g1, large) || !bytes.Equal(g2, large) {
		t.Fatalf("dedup Resolve: bytes not identical to original")
	}
}

// TestLenBytesTrackResidentCAS covers the footprint taps: Len/Bytes track the
// distinct blobs actually resident (inline payloads never count; dedup of
// identical content never double-counts).
func TestLenBytesTrackResidentCAS(t *testing.T) {
	ctx := context.Background()
	s := New()
	if s.Len() != 0 || s.Bytes() != 0 {
		t.Fatalf("fresh store: Len=%d Bytes=%d, want 0,0", s.Len(), s.Bytes())
	}

	a := makeBytes(InlineMax+10, 1)
	b := makeBytes(InlineMax+20, 2)
	if _, err := s.Put(ctx, a); err != nil {
		t.Fatalf("Put(a): %v", err)
	}
	if _, err := s.Put(ctx, b); err != nil {
		t.Fatalf("Put(b): %v", err)
	}
	if s.Len() != 2 {
		t.Fatalf("Len=%d, want 2", s.Len())
	}
	if want := int64(len(a) + len(b)); s.Bytes() != want {
		t.Fatalf("Bytes=%d, want %d", s.Bytes(), want)
	}

	// An inline payload never touches the CAS footprint.
	if _, err := s.Put(ctx, makeBytes(InlineMax, 3)); err != nil {
		t.Fatalf("Put(inline): %v", err)
	}
	if s.Len() != 2 {
		t.Fatalf("inline Put changed CAS Len: %d, want 2", s.Len())
	}

	// Dedup of identical content must NOT double-count bytes.
	if _, err := s.Put(ctx, append([]byte(nil), a...)); err != nil {
		t.Fatalf("Put(dup a): %v", err)
	}
	if want := int64(len(a) + len(b)); s.Bytes() != want {
		t.Fatalf("dedup double-counted bytes: Bytes=%d, want %d", s.Bytes(), want)
	}

	// PageOut of new bytes also grows the footprint.
	if _, err := s.PageOut(ctx, abi.Ref{Kind: abi.RefInline, Inline: makeBytes(InlineMax+5, 4), Len: int64(InlineMax + 5)}); err != nil {
		t.Fatalf("PageOut: %v", err)
	}
	if s.Len() != 3 {
		t.Fatalf("after PageOut: Len=%d, want 3", s.Len())
	}
}

// TestResetReclaimsCAS covers the lifecycle reclaim hook: Reset drops every blob
// and frees the footprint; a previously-stored Ref no longer resolves afterward.
func TestResetReclaimsCAS(t *testing.T) {
	ctx := context.Background()
	s := New()
	r, err := s.Put(ctx, makeBytes(InlineMax+100, 9))
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if s.Len() == 0 {
		t.Fatalf("precondition: CAS empty right after a large Put")
	}
	s.Reset()
	if s.Len() != 0 || s.Bytes() != 0 {
		t.Fatalf("after Reset: Len=%d Bytes=%d, want 0,0", s.Len(), s.Bytes())
	}
	if _, err := s.Resolve(ctx, r); err == nil {
		t.Fatalf("Resolve after Reset: want unknown-digest error, got nil")
	}
}

// TestByteBoundEvictsUnpinnedNotPinned is the core safety proof: under a tight byte
// bound, a PINNED digest (a live holder will resolve it later) survives unbounded
// churn, while UNPINNED digests (transient, no live holder) are evicted.
func TestByteBoundEvictsUnpinnedNotPinned(t *testing.T) {
	ctx := context.Background()
	s := newStore(0) // start unbounded so we control when eviction begins
	put := func(seed byte) abi.Ref {
		r, err := s.Put(ctx, makeBytes(1000, seed))
		if err != nil {
			t.Fatalf("Put: %v", err)
		}
		return r
	}

	keep := put(1)
	s.Pin(keep.Digest) // a live holder will resolve this later
	drop := put(2)     // unpinned, oldest -> first to go

	// Bound to ~2 blobs: keep (pinned, 1000B) always resides, leaving room for ~1
	// rolling unpinned blob; each further insert evicts the oldest unpinned.
	s.SetMaxBytes(2500)
	for i := byte(3); i < 12; i++ {
		put(i)
	}

	if _, err := s.Resolve(ctx, keep); err != nil {
		t.Fatalf("PINNED digest was evicted under churn: %v", err)
	}
	if _, err := s.Resolve(ctx, drop); err == nil {
		t.Fatalf("UNPINNED oldest digest should have been evicted")
	}
	if s.Evicted() == 0 {
		t.Fatalf("expected the byte bound to have evicted unpinned digests")
	}
	if s.Bytes() > 2500 {
		t.Fatalf("resident footprint %d exceeds the bound 2500 (only pinned may overshoot)", s.Bytes())
	}

	// Once the holder unpins, the digest becomes evictable and the bound reclaims it.
	s.Unpin(keep.Digest)
	s.SetMaxBytes(1) // force-evict everything now unpinned
	if _, err := s.Resolve(ctx, keep); err == nil {
		t.Fatalf("after Unpin, the digest should be evictable")
	}
}

// TestPinIsRefcounted proves dedup safety: a digest pinned by N holders survives
// until the LAST unpin (content-addressed dedup means the vDSO and the MMU can share
// one digest).
func TestPinIsRefcounted(t *testing.T) {
	ctx := context.Background()
	s := newStore(0) // store first under no bound, then pin, then tighten
	r, err := s.Put(ctx, makeBytes(1000, 7))
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	s.Pin(r.Digest)
	s.Pin(r.Digest)  // two independent holders share this digest
	s.SetMaxBytes(1) // now any unpinned digest is over budget; pinned ones must survive

	if _, err := s.Resolve(ctx, r); err != nil {
		t.Fatalf("doubly-pinned digest evicted under maxBytes=1: %v", err)
	}
	s.Unpin(r.Digest) // one holder releases; the other still pins
	if _, err := s.Resolve(ctx, r); err != nil {
		t.Fatalf("digest evicted while one holder still pins it: %v", err)
	}
	s.Unpin(r.Digest) // last holder releases -> now evictable, and maxBytes=1 reclaims it
	if _, err := s.Resolve(ctx, r); err == nil {
		t.Fatalf("fully-unpinned digest should have been reclaimed by the bound")
	}
}

// TestUnboundedStoreNeverEvicts confirms the legacy default (maxBytes<=0) keeps the
// append-only behavior — no eviction, every digest resolvable.
func TestUnboundedStoreNeverEvicts(t *testing.T) {
	ctx := context.Background()
	s := newStore(0)
	var refs []abi.Ref
	for i := byte(0); i < 50; i++ {
		r, err := s.Put(ctx, makeBytes(1000, i))
		if err != nil {
			t.Fatalf("Put: %v", err)
		}
		refs = append(refs, r)
	}
	if s.Evicted() != 0 {
		t.Fatalf("unbounded store evicted %d (want 0)", s.Evicted())
	}
	for _, r := range refs {
		if _, err := s.Resolve(ctx, r); err != nil {
			t.Fatalf("unbounded store lost a digest: %v", err)
		}
	}
}

// TestResolveUnknownDigest documents the error path for a CAS-backed Ref whose
// digest was never stored.
func TestResolveUnknownDigest(t *testing.T) {
	ctx := context.Background()
	s := New()
	_, err := s.Resolve(ctx, abi.Ref{Kind: abi.RefBlob, Digest: "deadbeef"})
	if err == nil {
		t.Fatalf("Resolve(unknown blob): want error, got nil")
	}
	if !strings.Contains(err.Error(), "unknown digest") {
		t.Fatalf("Resolve(unknown blob): error=%q, want it to mention 'unknown digest'", err)
	}
}

// TestDefaultStoreViaABI exercises the process-wide Default store both directly and
// through the registered abi.Resolver seam that init() wired up, confirming a CAS
// round trip survives the ABI boundary.
func TestDefaultStoreViaABI(t *testing.T) {
	ctx := context.Background()

	// The blob backend's Default is the registered RegionBackend's resolver.
	res := abi.ActiveResolver()
	if res == nil {
		t.Fatalf("abi.ActiveResolver() is nil; blob.init() should register the backend")
	}

	// A unique large payload so this test is independent of other Default users.
	large := makeBytes(InlineMax+200, 99)

	// Put through the abi.Resolver interface; Resolve through the same.
	ref, err := res.Put(ctx, large)
	if err != nil {
		t.Fatalf("Default Put via ABI: %v", err)
	}
	if ref.Kind != abi.RefBlob {
		t.Fatalf("Default Put via ABI: Kind=%d, want RefBlob", ref.Kind)
	}
	got, err := res.Resolve(ctx, ref)
	if err != nil {
		t.Fatalf("Default Resolve via ABI: %v", err)
	}
	if !bytes.Equal(got, large) {
		t.Fatalf("Default Resolve via ABI: bytes not identical")
	}

	// The same digest is resolvable directly off Default (one shared CAS).
	direct, err := Default.Resolve(ctx, ref)
	if err != nil {
		t.Fatalf("Default.Resolve direct: %v", err)
	}
	if !bytes.Equal(direct, large) {
		t.Fatalf("Default.Resolve direct: bytes not identical")
	}

	// Re-Putting the identical payload through Default must dedup (one blob even
	// across the ABI seam). Measure the delta to stay independent of other tests.
	_, h0, _ := Default.Stats()
	if _, err := Default.Put(ctx, append([]byte(nil), large...)); err != nil {
		t.Fatalf("Default re-Put: %v", err)
	}
	_, h1, _ := Default.Stats()
	if h1 != h0+1 {
		t.Fatalf("Default dedupHits delta = %d, want 1", h1-h0)
	}

	// Default page-out backend is registered under "blob".
	pob, ok := abi.PageOut("blob")
	if !ok {
		t.Fatalf("abi.PageOut(\"blob\") not registered")
	}
	handle, err := pob.PageOut(ctx, abi.Ref{Kind: abi.RefInline, Inline: makeBytes(32, 1), Len: 32})
	if err != nil {
		t.Fatalf("Default PageOut via ABI: %v", err)
	}
	in, err := pob.PageIn(ctx, handle)
	if err != nil {
		t.Fatalf("Default PageIn via ABI: %v", err)
	}
	if !bytes.Equal(in.Inline, makeBytes(32, 1)) {
		t.Fatalf("Default PageOut/PageIn via ABI: bytes not identical")
	}
}
