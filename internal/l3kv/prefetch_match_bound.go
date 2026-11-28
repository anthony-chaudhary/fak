package l3kv

import (
	"errors"
	"sync"
)

// bound (sgl-project/sglang PR #48522,
// python/sglang/srt/mem_cache/common.py:1;python/sglang/srt/managers/scheduler.py:1
// external storage must stop at the matched-key boundary. The upstream bug
// let the prefetch reserve more pages than the longest-prefix match
// justified, and the scheduler charged its admission accounting against the
// unbounded span. The ported contract: the prefetch span is clamped to the
// longest-prefix-match length, and the reservation ledger charges exactly
// the clamped span.
//
// This is a pure host-side accounting contract: no GPU, no network, no
// external storage client. Page keys are opaque identifiers for pages
// resident in the L3 tier; the PrefixIndex answers "how many leading keys of
// this request are actually resident" and the ReservationLedger answers "how
// many pages are reserved for in-flight prefetches".

var (
	// ErrPrefetchNoMatch reports that the request's key sequence has no
	// resident leading page: the prefetch is refused fail-closed and nothing
	// is reserved.
	ErrPrefetchNoMatch = errors.New("l3kv: prefetch refused: no matched prefix")
	// ErrInvalidPrefetchSpan reports an invalid prefetch request (nil index
	// or ledger, empty request label, empty key sequence, or a non-positive
	// requested span): the call is refused before any accounting.
	ErrInvalidPrefetchSpan = errors.New("l3kv: prefetch refused: invalid span")
	// ErrInvalidReservation reports a ledger charge with a non-positive page
	// count or an empty request label.
	ErrInvalidReservation = errors.New("l3kv: reservation refused: invalid charge")
)

// PageKey identifies one KV page resident in the external L3 tier.
type PageKey string

// PrefixIndex is the set of page keys resident in the external tier. It
// answers the longest-prefix-match question that bounds a prefetch: given a
// request's ordered key sequence, how many LEADING keys are resident. The
// match stops at the first missing key — a resident page past the gap does
// not extend the match, because a prefetch must read a contiguous prefix.
type PrefixIndex struct {
	mu    sync.RWMutex
	pages map[PageKey]struct{}
}

// NewPrefixIndex builds an index over the resident page keys.
func NewPrefixIndex(keys ...PageKey) *PrefixIndex {
	p := &PrefixIndex{pages: make(map[PageKey]struct{}, len(keys))}
	for _, k := range keys {
		p.pages[k] = struct{}{}
	}
	return p
}

// LongestPrefixMatch returns the number of leading keys in keys that are
// resident. It stops at the first missing key; an empty sequence matches 0.
func (p *PrefixIndex) LongestPrefixMatch(keys []PageKey) int {
	if p == nil {
		return 0
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	n := 0
	for _, k := range keys {
		if _, ok := p.pages[k]; !ok {
			break
		}
		n++
	}
	return n
}

// ReservationLedger is the admission-accounting ledger for in-flight L3
// prefetch reservations. It tracks how many pages are reserved in total and
// per request, so the scheduler charges admission against the bounded span
// and releases it when the prefetch settles.
type ReservationLedger struct {
	mu       sync.Mutex
	reserved int
	entries  map[string]int
}

// NewReservationLedger builds an empty ledger.
func NewReservationLedger() *ReservationLedger {
	return &ReservationLedger{entries: make(map[string]int)}
}

// Reserve charges n pages to request. It is additive: a second Reserve for
// the same request adds to its charge.
func (l *ReservationLedger) Reserve(request string, n int) error {
	if l == nil || request == "" || n <= 0 {
		return ErrInvalidReservation
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.entries[request] += n
	l.reserved += n
	return nil
}

// Release returns request's whole charge to the free pool and reports the
// number of pages released. Releasing an unknown or empty request releases 0.
func (l *ReservationLedger) Release(request string) int {
	if l == nil || request == "" {
		return 0
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	n := l.entries[request]
	if n > 0 {
		delete(l.entries, request)
		l.reserved -= n
	}
	return n
}

// Reserved reports the total pages reserved across all requests.
func (l *ReservationLedger) Reserved() int {
	if l == nil {
		return 0
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.reserved
}

// ReservedFor reports the pages reserved for one request.
func (l *ReservationLedger) ReservedFor(request string) int {
	if l == nil || request == "" {
		return 0
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.entries[request]
}

// PrefetchPlan is the outcome of a bounded prefetch decision.
type PrefetchPlan struct {
	// Matched is the longest-prefix-match length over the request's keys.
	Matched int
	// Requested is the span the caller asked to prefetch.
	Requested int
	// Bounded is the clamped span actually prefetched and charged:
	// min(Requested, Matched).
	Bounded int
}

// BoundPrefetch clamps the requested prefetch span to the
// longest-prefix-match boundary and charges the reservation ledger exactly
// the clamped span.
//
// matched is the number of leading keys in keys that index holds; bounded is
// min(requested, matched). A longer requested span is clamped down to the
// match — the SGLang #48522 fix: prefetch must not reserve pages the match
// does not justify — and a shorter requested span is honored as-is. On a
// zero match the call is refused fail-closed with ErrPrefetchNoMatch and the
// ledger is untouched; an invalid request (nil index or ledger, empty
// request label, empty keys, requested <= 0) is refused with
// ErrInvalidPrefetchSpan.
func BoundPrefetch(index *PrefixIndex, ledger *ReservationLedger, request string, requested int, keys []PageKey) (PrefetchPlan, error) {
	if index == nil || ledger == nil || request == "" || requested <= 0 || len(keys) == 0 {
		return PrefetchPlan{}, ErrInvalidPrefetchSpan
	}
	matched := index.LongestPrefixMatch(keys)
	if matched == 0 {
		return PrefetchPlan{}, ErrPrefetchNoMatch
	}
	bounded := matched
	if requested < bounded {
		bounded = requested
	}
	if err := ledger.Reserve(request, bounded); err != nil {
		return PrefetchPlan{}, err
	}
	return PrefetchPlan{Matched: matched, Requested: requested, Bounded: bounded}, nil
}
