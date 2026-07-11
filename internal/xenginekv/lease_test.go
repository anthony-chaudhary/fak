package xenginekv

import (
	"bytes"
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// putSpan allocates one span for a lease test and fails the test on error.
func putSpan(t *testing.T, a *Arena, b []byte) abi.Ref {
	t.Helper()
	r, err := a.Put(context.Background(), b)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	return r
}

// TestCanEvictTruthTable proves the two-axis rule (#3384) over the full
// {pinned, readers} grid: can_evict = !pinned && readers == 0, so ONLY {0,0} is
// evictable — the load-bearing conjunction LMCache's rule contributes.
func TestCanEvictTruthTable(t *testing.T) {
	cases := []struct {
		pins, readers int
		want          bool
	}{
		{0, 0, true},
		{1, 0, false},
		{0, 1, false},
		{1, 1, false},
	}
	for _, tc := range cases {
		a := NewArena(1024)
		r := putSpan(t, a, []byte("two-axis span"))
		for i := 0; i < tc.pins; i++ {
			if err := a.Pin(r); err != nil {
				t.Fatalf("{%d,%d}: Pin #%d: %v", tc.pins, tc.readers, i+1, err)
			}
		}
		for i := 0; i < tc.readers; i++ {
			if _, ok := a.AcquireReader(r); !ok {
				t.Fatalf("{%d,%d}: AcquireReader #%d refused a resident span", tc.pins, tc.readers, i+1)
			}
		}
		if got := a.CanEvict(r); got != tc.want {
			t.Errorf("CanEvict with pins=%d readers=%d = %v, want %v", tc.pins, tc.readers, got, tc.want)
		}
		if got := a.TryEvict(r); got != tc.want { // the gate agrees with the predicate
			t.Errorf("TryEvict with pins=%d readers=%d = %v, want %v", tc.pins, tc.readers, got, tc.want)
		}
	}
}

// TestReaderLeaseBlocksTryEvict proves the TRANSIENT axis end to end via
// ResolveLeased: while the lease is outstanding the view's bytes are safe from
// TryEvict (the read-after-evict race is closed), and releasing the lease makes the
// span evictable. The release handle is idempotent.
func TestReaderLeaseBlocksTryEvict(t *testing.T) {
	ctx := context.Background()
	a := NewArena(1024)
	body := []byte("bytes a reader is mid-read on")
	r := putSpan(t, a, body)

	view, release, err := a.ResolveLeased(ctx, r)
	if err != nil {
		t.Fatalf("ResolveLeased: %v", err)
	}
	if !bytes.Equal(view, body) {
		t.Fatalf("ResolveLeased = %q, want %q", view, body)
	}
	if &view[0] != &a.buf[int64(r.Handle)] { // still the zero-copy view, not a copy
		t.Fatalf("ResolveLeased returned a copy, not a view: &view[0]=%p, &buf[off]=%p", &view[0], &a.buf[int64(r.Handle)])
	}
	if a.TryEvict(r) {
		t.Fatal("TryEvict evicted a span with an outstanding reader lease")
	}
	if !bytes.Equal(view, body) { // the refused eviction did not zero the bytes
		t.Fatalf("refused TryEvict corrupted the leased view: %q, want %q", view, body)
	}
	release()
	if !a.TryEvict(r) {
		t.Fatal("TryEvict refused after the only reader lease was released")
	}
	if _, err := a.Resolve(ctx, r); err == nil {
		t.Fatal("Resolve after TryEvict succeeded; the evicted span must be unresolvable")
	}
	for i, b := range view { // TryEvict zeroes like Evict: the dangling view reads zeros
		if b != 0 {
			t.Fatalf("TryEvict did not clear the span: byte %d = %#x", i, b)
		}
	}
	release() // idempotent: a second release must not underflow or touch other state
}

// TestPinBlocksTryEvict proves the PERSISTENT axis: a pinned span refuses TryEvict
// even with zero readers, and Unpin to zero makes it evictable.
func TestPinBlocksTryEvict(t *testing.T) {
	a := NewArena(1024)
	r := putSpan(t, a, []byte("pinned span"))
	if err := a.Pin(r); err != nil {
		t.Fatalf("Pin: %v", err)
	}
	if a.CanEvict(r) {
		t.Fatal("CanEvict true for a pinned span")
	}
	if a.TryEvict(r) {
		t.Fatal("TryEvict evicted a pinned span")
	}
	if err := a.Unpin(r); err != nil {
		t.Fatalf("Unpin: %v", err)
	}
	if !a.CanEvict(r) {
		t.Fatal("CanEvict false after the last Unpin")
	}
	if !a.TryEvict(r) {
		t.Fatal("TryEvict refused an unpinned, reader-free span")
	}
}

// TestRefcountsNeedFullRelease proves both axes are COUNTS, not flags: two pins need
// two unpins, and two reader leases need both releases, before TryEvict admits.
func TestRefcountsNeedFullRelease(t *testing.T) {
	a := NewArena(1024)

	// Persistent axis: 2 pins, evictable only after the 2nd unpin.
	rp := putSpan(t, a, []byte("doubly pinned"))
	for i := 0; i < 2; i++ {
		if err := a.Pin(rp); err != nil {
			t.Fatalf("Pin #%d: %v", i+1, err)
		}
	}
	if err := a.Unpin(rp); err != nil {
		t.Fatalf("Unpin #1: %v", err)
	}
	if a.TryEvict(rp) {
		t.Fatal("TryEvict evicted with one of two pins still held")
	}
	if err := a.Unpin(rp); err != nil {
		t.Fatalf("Unpin #2: %v", err)
	}
	if !a.TryEvict(rp) {
		t.Fatal("TryEvict refused after both pins released")
	}

	// Transient axis: 2 leases, evictable only after both releases.
	rr := putSpan(t, a, []byte("doubly leased"))
	rel1, ok := a.AcquireReader(rr)
	if !ok {
		t.Fatal("AcquireReader #1 refused a resident span")
	}
	rel2, ok := a.AcquireReader(rr)
	if !ok {
		t.Fatal("AcquireReader #2 refused a resident span")
	}
	rel1()
	if a.TryEvict(rr) {
		t.Fatal("TryEvict evicted with one of two reader leases outstanding")
	}
	rel2()
	if !a.TryEvict(rr) {
		t.Fatal("TryEvict refused after both reader leases released")
	}
}

// TestUnpinBelowZeroGuarded proves the guard mirroring the blob store's: an Unpin
// with no matching Pin is refused with NO state change — the span stays exactly as
// evictable as it was, and a later balanced Pin/Unpin still works.
func TestUnpinBelowZeroGuarded(t *testing.T) {
	a := NewArena(1024)
	r := putSpan(t, a, []byte("never pinned"))
	if err := a.Unpin(r); err == nil {
		t.Fatal("Unpin of a never-pinned span succeeded; below-zero must be guarded")
	}
	if !a.CanEvict(r) { // the refused Unpin must not have corrupted the counters
		t.Fatal("CanEvict false after a guarded Unpin; the guard must not change state")
	}
	if err := a.Pin(r); err != nil {
		t.Fatalf("Pin after guarded Unpin: %v", err)
	}
	if err := a.Unpin(r); err != nil {
		t.Fatalf("balanced Unpin: %v", err)
	}
	if err := a.Unpin(r); err == nil {
		t.Fatal("Unpin below zero (after a balanced pair) succeeded; must be guarded")
	}
	if !a.TryEvict(r) {
		t.Fatal("TryEvict refused a span whose pins were fully released")
	}
}

// TestEvictQuarantineIgnoresLeases pins down the DELIBERATE asymmetry: the Evict
// quarantine still fires through a pin AND an outstanding reader lease (security
// trumps a lease; the dangling view reads zeros, as arena.go documents), and the
// stale release afterwards is a safe no-op.
func TestEvictQuarantineIgnoresLeases(t *testing.T) {
	ctx := context.Background()
	a := NewArena(1024)
	poison := []byte("poisoned span a reader is still holding")
	r := putSpan(t, a, poison)
	if err := a.Pin(r); err != nil {
		t.Fatalf("Pin: %v", err)
	}
	view, release, err := a.ResolveLeased(ctx, r)
	if err != nil {
		t.Fatalf("ResolveLeased: %v", err)
	}
	if err := a.Evict(r); err != nil {
		t.Fatalf("Evict through pin+lease: %v (the quarantine must stay unconditional)", err)
	}
	for i, b := range view {
		if b != 0 {
			t.Fatalf("Evict did not clear the leased span: byte %d = %#x", i, b)
		}
	}
	release() // span already unmapped: must be a no-op, not an underflow or panic
	if a.CanEvict(r) {
		t.Fatal("CanEvict true for an evicted handle; a non-resident span has nothing to evict")
	}
	if a.TryEvict(r) {
		t.Fatal("TryEvict evicted a non-resident handle")
	}
}

// TestLeaseEdgeRefs proves the non-happy shapes: non-resident and non-region Refs
// take no lease and no pin, and ResolveLeased's inline path returns an owned copy
// with a callable no-op release.
func TestLeaseEdgeRefs(t *testing.T) {
	ctx := context.Background()
	a := NewArena(1024)

	ghost := abi.Ref{Kind: abi.RefRegion, Handle: 512, Len: 8} // never allocated
	if _, ok := a.AcquireReader(ghost); ok {
		t.Fatal("AcquireReader granted a lease on a non-resident handle")
	}
	if err := a.Pin(ghost); err == nil {
		t.Fatal("Pin of a non-resident handle succeeded")
	}
	if _, _, err := a.ResolveLeased(ctx, ghost); err == nil {
		t.Fatal("ResolveLeased of a non-resident handle succeeded")
	}

	inline := abi.Ref{Kind: abi.RefInline, Inline: []byte("inline args"), Len: 11}
	v, release, err := a.ResolveLeased(ctx, inline)
	if err != nil {
		t.Fatalf("ResolveLeased(inline): %v", err)
	}
	if !bytes.Equal(v, inline.Inline) {
		t.Fatalf("ResolveLeased(inline) = %q, want %q", v, inline.Inline)
	}
	release() // no-op, must not panic
	if _, ok := a.AcquireReader(inline); ok {
		t.Fatal("AcquireReader granted a lease on an inline Ref (nothing in the arena to lease)")
	}
	if a.CanEvict(inline) || a.TryEvict(inline) {
		t.Fatal("CanEvict/TryEvict admitted a non-region Ref")
	}
}
