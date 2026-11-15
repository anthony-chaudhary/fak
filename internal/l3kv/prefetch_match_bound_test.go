package l3kv

import (
	"errors"
	"fmt"
	"testing"
)

func TestPrefetchMatchBound(t *testing.T) {
	t.Run("match of N keys prefetches exactly N pages", func(t *testing.T) {
		const n = 5
		keys := make([]PageKey, n)
		for i := range keys {
			keys[i] = PageKey(fmt.Sprintf("page-%d", i))
		}
		index := NewPrefixIndex(keys...)
		ledger := NewReservationLedger()

		plan, err := BoundPrefetch(index, ledger, "req-1", n, keys)
		if err != nil {
			t.Fatalf("BoundPrefetch: %v", err)
		}
		if plan.Matched != n {
			t.Fatalf("Matched = %d, want %d", plan.Matched, n)
		}
		if plan.Bounded != n {
			t.Fatalf("Bounded = %d, want %d", plan.Bounded, n)
		}
		if plan.Requested != n {
			t.Fatalf("Requested = %d, want %d", plan.Requested, n)
		}
	})

	t.Run("reservation ledger reflects exactly N", func(t *testing.T) {
		const n = 7
		keys := make([]PageKey, n)
		for i := range keys {
			keys[i] = PageKey(fmt.Sprintf("k%d", i))
		}
		index := NewPrefixIndex(keys...)
		ledger := NewReservationLedger()

		if _, err := BoundPrefetch(index, ledger, "req-ledger", n, keys); err != nil {
			t.Fatalf("BoundPrefetch: %v", err)
		}
		if got := ledger.Reserved(); got != n {
			t.Fatalf("ledger.Reserved() = %d, want %d", got, n)
		}
		if got := ledger.ReservedFor("req-ledger"); got != n {
			t.Fatalf("ledger.ReservedFor(req-ledger) = %d, want %d", got, n)
		}

		// A second request charges additively; the first is untouched.
		if _, err := BoundPrefetch(index, ledger, "req-other", n, keys); err != nil {
			t.Fatalf("BoundPrefetch (second request): %v", err)
		}
		if got := ledger.Reserved(); got != 2*n {
			t.Fatalf("ledger.Reserved() = %d, want %d", got, 2*n)
		}
		if got := ledger.ReservedFor("req-ledger"); got != n {
			t.Fatalf("ledger.ReservedFor(req-ledger) = %d after second charge, want %d", got, n)
		}

		// Settling the first prefetch releases exactly its clamped span.
		if got := ledger.Release("req-ledger"); got != n {
			t.Fatalf("Release(req-ledger) = %d, want %d", got, n)
		}
		if got := ledger.Reserved(); got != n {
			t.Fatalf("ledger.Reserved() after release = %d, want %d", got, n)
		}
		if got := ledger.ReservedFor("req-ledger"); got != 0 {
			t.Fatalf("ledger.ReservedFor(req-ledger) after release = %d, want 0", got)
		}
	})

	t.Run("longer requested span is clamped to the match", func(t *testing.T) {
		// Only 3 of the 10 requested pages are resident: the prefetch must
		// stop at the matched-key boundary and charge 3, not 10.
		keys := make([]PageKey, 10)
		for i := range keys {
			keys[i] = PageKey(fmt.Sprintf("p%d", i))
		}
		index := NewPrefixIndex(keys[:3]...)
		ledger := NewReservationLedger()

		plan, err := BoundPrefetch(index, ledger, "req-clamp", 10, keys)
		if err != nil {
			t.Fatalf("BoundPrefetch: %v", err)
		}
		if plan.Matched != 3 {
			t.Fatalf("Matched = %d, want 3", plan.Matched)
		}
		if plan.Bounded != 3 {
			t.Fatalf("Bounded = %d, want 3 (clamped from 10)", plan.Bounded)
		}
		if got := ledger.Reserved(); got != 3 {
			t.Fatalf("ledger.Reserved() = %d, want 3 (admission charged the clamped span)", got)
		}
		if got := ledger.ReservedFor("req-clamp"); got != 3 {
			t.Fatalf("ledger.ReservedFor(req-clamp) = %d, want 3", got)
		}
	})

	t.Run("shorter requested span is honored as-is", func(t *testing.T) {
		keys := make([]PageKey, 8)
		for i := range keys {
			keys[i] = PageKey(fmt.Sprintf("s%d", i))
		}
		index := NewPrefixIndex(keys...)
		ledger := NewReservationLedger()

		plan, err := BoundPrefetch(index, ledger, "req-short", 2, keys)
		if err != nil {
			t.Fatalf("BoundPrefetch: %v", err)
		}
		if plan.Matched != 8 {
			t.Fatalf("Matched = %d, want 8", plan.Matched)
		}
		if plan.Bounded != 2 {
			t.Fatalf("Bounded = %d, want 2 (requested below the match)", plan.Bounded)
		}
		if got := ledger.Reserved(); got != 2 {
			t.Fatalf("ledger.Reserved() = %d, want 2", got)
		}
	})

	t.Run("match stops at the first missing key", func(t *testing.T) {
		// p0, p1, p3, p4 are resident but p2 is not: the leading match is 2,
		// even though 4 keys are resident overall.
		keys := []PageKey{"p0", "p1", "p2", "p3", "p4"}
		index := NewPrefixIndex(PageKey("p0"), PageKey("p1"), PageKey("p3"), PageKey("p4"))
		ledger := NewReservationLedger()

		plan, err := BoundPrefetch(index, ledger, "req-gap", 5, keys)
		if err != nil {
			t.Fatalf("BoundPrefetch: %v", err)
		}
		if plan.Matched != 2 {
			t.Fatalf("Matched = %d, want 2 (match stops at the gap)", plan.Matched)
		}
		if plan.Bounded != 2 {
			t.Fatalf("Bounded = %d, want 2", plan.Bounded)
		}
		if got := ledger.Reserved(); got != 2 {
			t.Fatalf("ledger.Reserved() = %d, want 2", got)
		}
	})

	t.Run("zero match is refused fail-closed", func(t *testing.T) {
		keys := []PageKey{"a", "b", "c"}
		ledger := NewReservationLedger()

		// No resident pages at all.
		empty := NewPrefixIndex()
		if _, err := BoundPrefetch(empty, ledger, "req-none", 3, keys); !errors.Is(err, ErrPrefetchNoMatch) {
			t.Fatalf("err = %v, want ErrPrefetchNoMatch", err)
		}
		// Resident pages exist but none lead the request.
		off := NewPrefixIndex(PageKey("x"), PageKey("y"))
		if _, err := BoundPrefetch(off, ledger, "req-none", 3, keys); !errors.Is(err, ErrPrefetchNoMatch) {
			t.Fatalf("err = %v, want ErrPrefetchNoMatch", err)
		}
		if got := ledger.Reserved(); got != 0 {
			t.Fatalf("ledger.Reserved() = %d after refused prefetches, want 0", got)
		}
		if got := ledger.ReservedFor("req-none"); got != 0 {
			t.Fatalf("ledger.ReservedFor(req-none) = %d, want 0", got)
		}
	})

	t.Run("invalid requests are refused before accounting", func(t *testing.T) {
		keys := []PageKey{"a", "b"}
		index := NewPrefixIndex(keys...)
		ledger := NewReservationLedger()

		cases := []struct {
			name      string
			index     *PrefixIndex
			ledger    *ReservationLedger
			request   string
			requested int
			ks        []PageKey
		}{
			{"nil index", nil, ledger, "req", 2, keys},
			{"nil ledger", index, nil, "req", 2, keys},
			{"empty request", index, ledger, "", 2, keys},
			{"zero requested", index, ledger, "req", 0, keys},
			{"negative requested", index, ledger, "req", -1, keys},
			{"empty keys", index, ledger, "req", 2, nil},
		}
		for _, tc := range cases {
			if _, err := BoundPrefetch(tc.index, tc.ledger, tc.request, tc.requested, tc.ks); !errors.Is(err, ErrInvalidPrefetchSpan) {
				t.Fatalf("%s: err = %v, want ErrInvalidPrefetchSpan", tc.name, err)
			}
		}
		if got := ledger.Reserved(); got != 0 {
			t.Fatalf("ledger.Reserved() = %d after invalid requests, want 0", got)
		}
	})
}
