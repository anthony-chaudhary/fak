// This file adds the SECOND evictability axis to the arena (#3384): the rule
// can_evict = !pinned && readers == 0, a clean-room borrow of LMCache's two-axis
// evictability (lmcache/v1/memory_management.py — technique only, original Go).
//
// Before it, the arena had ONE axis — none, really: Evict was unconditional, and the
// zero-copy Resolve contract (arena.go) documented the read-after-evict race as the
// HOLDER's problem ("a holder that keeps bytes across such a boundary must copy them").
// The two axes close that race for capacity-driven eviction:
//
//   - PERSISTENT pin (Pin/Unpin): a refcount held for as long as a live holder will
//     resolve the span LATER — the same idiom as blob.Store's pins map, addressed by
//     region handle instead of digest.
//   - TRANSIENT reader lease (AcquireReader/ResolveLeased): a count of zero-copy views
//     outstanding RIGHT NOW; each lease returns a release handle that decrements it.
//
// TryEvict is the new CAPACITY-driven gate: it evicts only when CanEvict holds — both
// counters zero — under one lock acquisition (no check-then-evict gap). The existing
// Evict is DELIBERATELY left unconditional: it is the QUARANTINE primitive, and a span
// adjudicated as poisoned must be physically clearable even while a reader holds a view
// (the dangling view then reads zeros — the exact property TestEvictQuarantine asserts).
// Security trumps a lease; capacity respects it.
package xenginekv

import (
	"context"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Pin protects a span from capacity eviction (TryEvict) for as long as a live holder
// will resolve it later — the PERSISTENT axis. It is refcounted, mirroring
// blob.Store.Pin: a span shared by several holders stays protected until the LAST
// Unpin. Unlike the blob store (where pinning ahead of a Put is meaningful), a pin
// here requires a RESIDENT span: the bump allocator never reuses offsets, so there is
// nothing a pin on a non-resident handle could ever protect. Pin never blocks the
// Evict quarantine — only TryEvict consults it.
func (a *Arena) Pin(r abi.Ref) error {
	if r.Kind != abi.RefRegion {
		return fmt.Errorf("xenginekv: Pin needs a RefRegion handle, got RefKind %d", r.Kind)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	off := int64(r.Handle)
	s, ok := a.live[off]
	if !ok {
		return fmt.Errorf("xenginekv: region handle %d is not resident (evicted or never allocated)", r.Handle)
	}
	s.pins++
	a.live[off] = s
	return nil
}

// Unpin releases one Pin; when the last holder unpins, the span becomes evictable
// again (readers permitting). Unpinning below zero is GUARDED — refused with an error
// and no state change. The blob store absorbs the same imbalance silently; the arena
// surfaces it instead, because a mismatched pin ledger on a zero-copy region means a
// caller's lifetime logic is wrong and hiding that invites a use-after-evict.
func (a *Arena) Unpin(r abi.Ref) error {
	if r.Kind != abi.RefRegion {
		return fmt.Errorf("xenginekv: Unpin needs a RefRegion handle, got RefKind %d", r.Kind)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	off := int64(r.Handle)
	s, ok := a.live[off]
	if !ok {
		return fmt.Errorf("xenginekv: region handle %d is not resident (evicted or never allocated)", r.Handle)
	}
	if s.pins <= 0 {
		return fmt.Errorf("xenginekv: Unpin of unpinned region handle %d (pin refcount already zero)", r.Handle)
	}
	s.pins--
	a.live[off] = s
	return nil
}

// AcquireReader takes a TRANSIENT reader lease on a resident span: while the lease is
// outstanding, TryEvict defers (readers > 0). The returned release is idempotent —
// calling it more than once decrements exactly once — and must NOT be invoked while
// holding the arena's lock (it takes the lock itself). ok is false for a non-region
// Ref or a span that is not resident; no lease is taken and release is nil.
//
// A lease guards against CAPACITY eviction only. The Evict quarantine still fires
// through an outstanding lease (see the file comment); release after a force-Evict is
// a safe no-op, because the counters were unmapped with the span and the bump
// allocator never re-issues the offset to a different span.
func (a *Arena) AcquireReader(r abi.Ref) (release func(), ok bool) {
	if r.Kind != abi.RefRegion {
		return nil, false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	off := int64(r.Handle)
	s, live := a.live[off]
	if !live {
		return nil, false
	}
	s.readers++
	a.live[off] = s
	return a.releaseFunc(off), true
}

// releaseFunc builds the once-only decrement for ONE acquired lease on off. The
// sync.Once makes a double release harmless (one lease, one decrement), and the
// residency re-check makes a release AFTER a force-Evict a no-op rather than an
// underflow — the offset can never name a different span (no freelist).
func (a *Arena) releaseFunc(off int64) func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			a.mu.Lock()
			defer a.mu.Unlock()
			s, ok := a.live[off]
			if !ok {
				return // quarantined while leased: the counters left with the span
			}
			if s.readers > 0 {
				s.readers--
				a.live[off] = s
			}
		})
	}
}

// ResolveLeased is Resolve with the read-after-evict race closed for the caller: it
// returns the same zero-copy VIEW and, atomically under the same lock, a transient
// reader lease — so a TryEvict cannot zero the bytes between the resolve and the read.
// The caller MUST call release when done with the view; until then the span refuses
// capacity eviction. Resolve's existing contract (view valid only until Evict) is
// untouched — a caller that wants the guarantee opts into this method.
//
// An inline Ref returns its own bytes (an owned copy, as with Resolve) and a no-op
// release: there is nothing in the arena to lease.
func (a *Arena) ResolveLeased(ctx context.Context, r abi.Ref) (view []byte, release func(), err error) {
	switch r.Kind {
	case abi.RefInline:
		return append([]byte(nil), r.Inline...), func() {}, nil
	case abi.RefRegion:
		a.mu.Lock()
		defer a.mu.Unlock()
		off := int64(r.Handle)
		s, ok := a.live[off]
		if !ok {
			return nil, nil, fmt.Errorf("xenginekv: region handle %d is not resident (evicted or never allocated)", r.Handle)
		}
		s.readers++
		a.live[off] = s
		if s.n == 0 {
			return []byte{}, a.releaseFunc(off), nil
		}
		return a.buf[s.off : s.off+s.n : s.off+s.n], a.releaseFunc(off), nil
	default:
		return nil, nil, fmt.Errorf("xenginekv: unsupported RefKind %d (this backend issues RefRegion)", r.Kind)
	}
}

// CanEvict reports the two-axis evictability rule for a span: resident AND not pinned
// AND no reader lease outstanding — LMCache's can_evict, on the arena. False for a
// non-resident handle (there is nothing to evict) and for a non-region Ref. It is an
// advisory snapshot; TryEvict re-checks under its own write lock, so gating decisions
// never rest on this value alone.
func (a *Arena) CanEvict(r abi.Ref) bool {
	if r.Kind != abi.RefRegion {
		return false
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	s, ok := a.live[int64(r.Handle)]
	return ok && s.pins == 0 && s.readers == 0
}

// TryEvict is the CAPACITY-driven eviction gate: it unmaps and zeroes the span exactly
// as Evict does, but ONLY when CanEvict holds — the span is resident, unpinned, and has
// no outstanding reader lease. The check and the eviction happen under one write-lock
// acquisition, so a lease can never slip in between. Returns whether the span was
// evicted; false means refused/deferred (pinned, leased, or not resident) and the
// caller retries later or picks another victim. For the quarantine path — a poisoned
// span that must go NOW regardless of holders — use Evict.
func (a *Arena) TryEvict(r abi.Ref) (evicted bool) {
	if r.Kind != abi.RefRegion {
		return false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	off := int64(r.Handle)
	s, ok := a.live[off]
	if !ok || s.pins != 0 || s.readers != 0 {
		return false
	}
	for i := s.off; i < s.off+s.n; i++ {
		a.buf[i] = 0 // physically clear, mirroring Evict: a dangling view reads zeros
	}
	delete(a.live, off)
	return true
}
