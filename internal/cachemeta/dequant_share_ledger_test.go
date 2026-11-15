package cachemeta

import (
	"errors"
	"sync"
	"testing"
)

// dequant_share_ledger_test.go — expand-once-share-many dequant ledger
// (sglang PD dequant-once-share, sgl-project/sglang PR #40421).
//
// The contract pinned here: N concurrent consumers of the same quantized KV
// slice at the receiving stage share ONE expansion (Expansions stays 1, every
// consumer reads the same buffer), the entry's reference count tracks the
// live consumers, the entry releases only when the LAST consumer releases,
// and a double-release is refused fail-closed.

// dequantShareTestSlice builds a quantized slice: 8 int8 codes, one positive
// per-block scale, so the expansion is codes * 0.5.
func dequantShareTestSlice(digest string) DequantShareSlice {
	return DequantShareSlice{
		Digest: digest,
		Codes:  []int8{2, -4, 8, 0, 1, -1, 3, 6},
		Scales: []float32{0.5},
	}
}

// TestDequantShareLedgerSingleExpansionForNConsumers is the expand-once
// witness: N consumers of the same digest all land on the one expansion.
func TestDequantShareLedgerSingleExpansionForNConsumers(t *testing.T) {
	const n = 8
	l := NewDequantShareLedger()
	slice := dequantShareTestSlice("kv-slice-0")

	consumers := make([]*DequantShareConsumer, 0, n)
	for i := 0; i < n; i++ {
		c, err := l.Acquire(slice)
		if err != nil {
			t.Fatalf("acquire %d: %v", i, err)
		}
		consumers = append(consumers, c)
	}

	if l.Expansions != 1 {
		t.Fatalf("expansions = %d, want 1 (expand once for %d consumers)", l.Expansions, n)
	}
	if l.Acquires != n {
		t.Fatalf("acquires = %d, want %d", l.Acquires, n)
	}

	// Every consumer must be handed the SAME expanded buffer, and the
	// expansion must be the real dequant (codes * 0.5), not a copy.
	want := []float32{1, -2, 4, 0, 0.5, -0.5, 1.5, 3}
	for i, c := range consumers {
		if &c.entry.Expanded[0] != &consumers[0].entry.Expanded[0] {
			t.Fatalf("consumer %d got a different buffer than consumer 0 (no sharing)", i)
		}
		if got := c.entry.Expanded; len(got) != len(want) {
			t.Fatalf("consumer %d expanded length = %d, want %d", i, len(got), len(want))
		}
		for j := range want {
			if got := c.entry.Expanded[j]; got != want[j] {
				t.Fatalf("consumer %d expanded[%d] = %v, want %v", i, j, got, want[j])
			}
		}
	}

	// The entry is live with all N references held: not released yet.
	e, ok := l.Entry("kv-slice-0")
	if !ok {
		t.Fatalf("entry for kv-slice-0 missing")
	}
	if e.RefCount != n {
		t.Fatalf("refcount = %d, want %d", e.RefCount, n)
	}
	if e.Released {
		t.Fatalf("entry released while %d consumers still hold it", n)
	}
	if l.Released != 0 {
		t.Fatalf("released entries = %d, want 0", l.Released)
	}
}

// TestDequantShareLedgerReferenceCountingAndLastUseRelease pins the
// share-many / last-use half: the refcount tracks live consumers down and the
// entry releases exactly when the final consumer releases.
func TestDequantShareLedgerReferenceCountingAndLastUseRelease(t *testing.T) {
	l := NewDequantShareLedger()
	slice := dequantShareTestSlice("kv-slice-1")

	const n = 4
	consumers := make([]*DequantShareConsumer, 0, n)
	for i := 0; i < n; i++ {
		c, err := l.Acquire(slice)
		if err != nil {
			t.Fatalf("acquire %d: %v", i, err)
		}
		consumers = append(consumers, c)
	}

	// Release n-1 consumers: the entry must stay live at each step.
	for i := 0; i < n-1; i++ {
		if err := l.Release(consumers[i]); err != nil {
			t.Fatalf("release %d: %v", i, err)
		}
		e, _ := l.Entry("kv-slice-1")
		want := n - 1 - i
		if e.RefCount != want {
			t.Fatalf("after release %d: refcount = %d, want %d", i, e.RefCount, want)
		}
		if e.Released {
			t.Fatalf("entry released with %d consumers still holding it", want)
		}
		if l.Released != 0 {
			t.Fatalf("released entries = %d, want 0 before last release", l.Released)
		}
	}

	// The LAST release flips the entry to released.
	if err := l.Release(consumers[n-1]); err != nil {
		t.Fatalf("final release: %v", err)
	}
	e, _ := l.Entry("kv-slice-1")
	if e.RefCount != 0 {
		t.Fatalf("refcount after last release = %d, want 0", e.RefCount)
	}
	if !e.Released {
		t.Fatalf("entry not released after last consumer released")
	}
	if l.Released != 1 {
		t.Fatalf("released entries = %d, want 1", l.Released)
	}
	if l.Releases != n {
		t.Fatalf("releases = %d, want %d", l.Releases, n)
	}
}

// TestDequantShareLedgerDoubleReleaseRefusedFailClosed pins the fail-closed
// half: releasing the same consumer twice is refused and leaves the ledger
// state untouched (no double-decrement, no premature release).
func TestDequantShareLedgerDoubleReleaseRefusedFailClosed(t *testing.T) {
	l := NewDequantShareLedger()
	slice := dequantShareTestSlice("kv-slice-2")

	a, err := l.Acquire(slice)
	if err != nil {
		t.Fatalf("acquire a: %v", err)
	}
	b, err := l.Acquire(slice)
	if err != nil {
		t.Fatalf("acquire b: %v", err)
	}

	if err := l.Release(a); err != nil {
		t.Fatalf("first release of a: %v", err)
	}
	// Second release of the SAME consumer: refused, state untouched.
	if err := l.Release(a); !errors.Is(err, ErrDequantShareDoubleRelease) {
		t.Fatalf("double release of a: err = %v, want ErrDequantShareDoubleRelease", err)
	}
	// Nil consumer is also refused fail-closed.
	if err := l.Release(nil); !errors.Is(err, ErrDequantShareDoubleRelease) {
		t.Fatalf("nil release: err = %v, want ErrDequantShareDoubleRelease", err)
	}

	// The entry must still be live at refcount 1 (consumer b holds it): the
	// refused double-release did not decrement a second time.
	e, _ := l.Entry("kv-slice-2")
	if e.RefCount != 1 {
		t.Fatalf("refcount after refused double-release = %d, want 1", e.RefCount)
	}
	if e.Released {
		t.Fatalf("entry released while consumer b still holds it")
	}
	if l.Releases != 1 {
		t.Fatalf("releases = %d, want 1 (refused releases must not count)", l.Releases)
	}

	// The surviving consumer still releases cleanly on its own.
	if err := l.Release(b); err != nil {
		t.Fatalf("release b: %v", err)
	}
	if !e.Released || e.RefCount != 0 {
		t.Fatalf("after b released: refcount=%d released=%v, want 0/true", e.RefCount, e.Released)
	}
}

// TestDequantShareLedgerDistinctDigestsExpandSeparately guards the share key:
// different digests are different quantized slices and each gets its own
// expansion (the ledger shares within a digest, not across digests).
func TestDequantShareLedgerDistinctDigestsExpandSeparately(t *testing.T) {
	l := NewDequantShareLedger()
	a, err := l.Acquire(dequantShareTestSlice("kv-slice-a"))
	if err != nil {
		t.Fatalf("acquire a: %v", err)
	}
	b, err := l.Acquire(dequantShareTestSlice("kv-slice-b"))
	if err != nil {
		t.Fatalf("acquire b: %v", err)
	}
	if l.Expansions != 2 {
		t.Fatalf("expansions = %d, want 2 (one per distinct digest)", l.Expansions)
	}
	if &a.entry.Expanded[0] == &b.entry.Expanded[0] {
		t.Fatalf("distinct digests must not share a buffer")
	}
	if err := l.Release(a); err != nil {
		t.Fatalf("release a: %v", err)
	}
	if err := l.Release(b); err != nil {
		t.Fatalf("release b: %v", err)
	}
	if l.Released != 2 {
		t.Fatalf("released entries = %d, want 2", l.Released)
	}
}

// TestDequantShareLedgerConcurrentConsumersOneExpansion is the same-pass
// concurrency witness: N goroutines racing to Acquire the same digest all
// land on the single expansion and the refcount ends at exactly N.
func TestDequantShareLedgerConcurrentConsumersOneExpansion(t *testing.T) {
	const n = 32
	l := NewDequantShareLedger()
	slice := dequantShareTestSlice("kv-slice-race")

	var mu sync.Mutex
	consumers := make([]*DequantShareConsumer, 0, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, err := l.Acquire(slice)
			if err != nil {
				t.Errorf("concurrent acquire: %v", err)
				return
			}
			mu.Lock()
			consumers = append(consumers, c)
			mu.Unlock()
		}()
	}
	wg.Wait()

	if l.Expansions != 1 {
		t.Fatalf("concurrent acquires performed %d expansions, want 1", l.Expansions)
	}
	if len(consumers) != n {
		t.Fatalf("got %d consumers, want %d", len(consumers), n)
	}
	e, _ := l.Entry("kv-slice-race")
	if e.RefCount != n {
		t.Fatalf("refcount = %d, want %d", e.RefCount, n)
	}

	// Release all concurrently; the entry must end released exactly once.
	var rwg sync.WaitGroup
	for _, c := range consumers {
		rwg.Add(1)
		go func(c *DequantShareConsumer) {
			defer rwg.Done()
			if err := l.Release(c); err != nil {
				t.Errorf("concurrent release: %v", err)
			}
		}(c)
	}
	rwg.Wait()
	if !e.Released || e.RefCount != 0 {
		t.Fatalf("after all releases: refcount=%d released=%v, want 0/true", e.RefCount, e.Released)
	}
	if l.Released != 1 {
		t.Fatalf("released entries = %d, want 1", l.Released)
	}
}

// TestDequantShareLedgerMultiBlockScales pins the per-block scale model: a
// slice of two blocks (4 codes each) with distinct scales expands each code
// with its own block's scale, and the whole slice still expands exactly once.
func TestDequantShareLedgerMultiBlockScales(t *testing.T) {
	l := NewDequantShareLedger()
	slice := DequantShareSlice{
		Digest: "kv-slice-multi",
		// 2 blocks of 2 codes: block0 scale 0.5, block1 scale 2.0
		Codes:  []int8{2, 4, 1, -3},
		Scales: []float32{0.5, 2.0},
	}
	c, err := l.Acquire(slice)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	want := []float32{1, 2, 2, -6} // 2*0.5, 4*0.5, 1*2.0, -3*2.0
	if got := c.entry.Expanded; len(got) != len(want) {
		t.Fatalf("expanded length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got := c.entry.Expanded[i]; got != want[i] {
			t.Fatalf("expanded[%d] = %v, want %v (per-block scale)", i, got, want[i])
		}
	}
	// A second consumer of the same digest shares the one expansion.
	c2, err := l.Acquire(slice)
	if err != nil {
		t.Fatalf("acquire 2: %v", err)
	}
	if l.Expansions != 1 {
		t.Fatalf("expansions = %d, want 1", l.Expansions)
	}
	if &c.entry.Expanded[0] != &c2.entry.Expanded[0] {
		t.Fatalf("second consumer did not share the expanded buffer")
	}
}

// TestDequantShareLedgerRefusesBadSlices pins the fail-closed expansion
// guards: an empty slice or a non-positive/misaligned scale is refused before
// any expansion is recorded.
func TestDequantShareLedgerRefusesBadSlices(t *testing.T) {
	for _, tc := range []struct {
		name  string
		slice DequantShareSlice
		want  error
	}{
		{name: "empty_codes", slice: DequantShareSlice{Digest: "e", Codes: nil, Scales: []float32{1}}, want: ErrDequantShareEmpty},
		{name: "no_scales", slice: DequantShareSlice{Digest: "n", Codes: []int8{1}, Scales: nil}, want: ErrDequantShareBadScale},
		{name: "zero_scale", slice: DequantShareSlice{Digest: "z", Codes: []int8{1}, Scales: []float32{0}}, want: ErrDequantShareBadScale},
		{name: "negative_scale", slice: DequantShareSlice{Digest: "neg", Codes: []int8{1}, Scales: []float32{-1}}, want: ErrDequantShareBadScale},
		{name: "misaligned_scales", slice: DequantShareSlice{Digest: "m", Codes: []int8{1, 2, 3}, Scales: []float32{0.5, 0.5}}, want: ErrDequantShareBadScale},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l := NewDequantShareLedger()
			if _, err := l.Acquire(tc.slice); !errors.Is(err, tc.want) {
				t.Fatalf("acquire: err = %v, want %v", err, tc.want)
			}
			if l.Expansions != 0 {
				t.Fatalf("refused slice recorded %d expansions, want 0", l.Expansions)
			}
		})
	}
}
