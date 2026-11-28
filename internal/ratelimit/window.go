// Sliding-window + token-bucket per-route caps, ported host-side from the
// NVIDIA/go-ratelimit study (INSPIRE-ONLY; pin
//
// WHAT WAS BORROWED. The upstream library pairs two classic cap shapes behind
// one admission call: a SLIDING WINDOW (at most N events in any rolling
// duration — a rate cap) and a TOKEN BUCKET (a spend cap: a finite token pool
// that refills at a fixed rate). This file ports that window/bucket math and
// its cap-refusal grammar to a native in-process store: no Redis, no network,
// stdlib only. The Redis backend stays out on purpose — a distributed-store
// claim would need a distributed witness, and this leaf is the host-side
// governor the per-route spend/token/rate limits consult first.
//
// THE CAP-REFUSAL GRAMMAR. Every refusal is a structured Refusal, not a bare
// bool: which cap fired (window_full / bucket_empty / invalid_cap), the cap's
// limit, the current usage, the requested amount, and an advisory RetryAfter
// the caller can surface as a back-off. Invalid caps fail CLOSED: a
// misconfigured dimension (a window with no duration, a bucket with no refill
// rate, a negative cost) refuses the call instead of silently admitting it.
package ratelimit

import (
	"sync"
	"time"
)

// Refusal cap names — the closed vocabulary of cap-refusal grammar.
const (
	// RefusalWindowFull: the sliding window already holds Limit admitted
	// events; the call would exceed the rate cap.
	RefusalWindowFull = "window_full"
	// RefusalBucketEmpty: the token bucket holds fewer tokens than the
	// request's cost; the call would exceed the spend cap.
	RefusalBucketEmpty = "bucket_empty"
	// RefusalInvalidCap: the cap set (or the request cost) is malformed; the
	// call is refused fail-closed.
	RefusalInvalidCap = "invalid_cap"
)

// WindowCap is a sliding-window rate cap: at most Limit admitted events in any
// rolling Window. A zero Limit leaves the rate dimension unlimited.
type WindowCap struct {
	Window time.Duration
	Limit  int
}

// BucketCap is a token-bucket spend cap: a bucket of Capacity tokens, refilled
// at Rate tokens per second. A zero Capacity leaves the spend dimension
// unlimited.
type BucketCap struct {
	Capacity int64
	Rate     int64 // tokens per second
}

// RouteCaps is the per-route cap set: a rate cap (Window) and/or a spend cap
// (Bucket). Each dimension is configured independently; a zero dimension is
// unlimited.
type RouteCaps struct {
	Window WindowCap
	Bucket BucketCap
}

// Valid reports whether the cap set is well-formed. A dimension is configured
// when its limit/capacity is positive; a configured dimension must also have a
// positive window/rate. The second return names the fault for forensics.
func (c RouteCaps) Valid() (bool, string) {
	if c.Window.Limit > 0 && c.Window.Window <= 0 {
		return false, "window cap requires a positive duration"
	}
	if c.Bucket.Capacity > 0 && c.Bucket.Rate <= 0 {
		return false, "bucket cap requires a positive refill rate"
	}
	return true, ""
}

// Refusal is the structured cap refusal: which cap fired, its limit, the
// current usage, the requested amount, and the advisory back-off.
type Refusal struct {
	Cap        string        // RefusalWindowFull | RefusalBucketEmpty | RefusalInvalidCap
	Limit      int64         // the cap's limit (0 for invalid_cap)
	Used       int64         // current usage against the cap
	Requested  int64         // the amount this call asked for
	RetryAfter time.Duration // advisory: earliest time the call could be retried
}

// Admission is the decision for one request against a route's caps. Admit=true
// carries an empty Refusal; Admit=false carries the refusal that fired.
type Admission struct {
	Admit   bool
	Refusal Refusal
}

// routeState is one route's live cap state: admitted event times (ascending,
// all inside the current window) and the bucket's token count.
type routeState struct {
	events []time.Time
	tokens int64
	last   time.Time
}

// Store is the native in-process per-route cap store. One mutex guards all
// route state; there is no external backend.
type Store struct {
	mu     sync.Mutex
	now    func() time.Time
	routes map[string]*routeState
}

// NewStore returns a store that decides at the wall clock.
func NewStore() *Store {
	return &Store{now: time.Now, routes: map[string]*routeState{}}
}

// NewStoreWithClock returns a store that decides at an injected clock (tests,
// fixtures, and any caller that wants deterministic time).
func NewStoreWithClock(now func() time.Time) *Store {
	s := NewStore()
	if now != nil {
		s.now = now
	}
	return s
}

// Admit decides one request of `cost` tokens against the route's caps at the
// store's current time.
func (s *Store) Admit(route string, caps RouteCaps, cost int64) Admission {
	return s.AdmitAt(route, caps, cost, s.now())
}

// AdmitAt is the pure decision at an explicit time. It checks the caps in
// order — window first (the cheap rate shed), then bucket (the spend check) —
// and consumes state only on admission: a refused call advances neither the
// window count nor the bucket, so probing an exhausted route is idempotent.
func (s *Store) AdmitAt(route string, caps RouteCaps, cost int64, at time.Time) Admission {
	s.mu.Lock()
	defer s.mu.Unlock()

	if ok, _ := caps.Valid(); !ok {
		return Admission{Refusal: Refusal{Cap: RefusalInvalidCap, Requested: cost}}
	}
	if cost < 0 {
		return Admission{Refusal: Refusal{Cap: RefusalInvalidCap, Requested: cost}}
	}

	st := s.routes[route]
	if st == nil {
		st = &routeState{tokens: caps.Bucket.Capacity, last: at}
		s.routes[route] = st
	}

	// Sliding window: prune events that have rolled out of (at-Window, at],
	// then check the count. A refused call records no event.
	if caps.Window.Limit > 0 {
		s.pruneLocked(st, caps.Window.Window, at)
		if int64(len(st.events)) >= int64(caps.Window.Limit) {
			oldest := st.events[0]
			// The oldest event rolls out of the window at oldest+Window; the
			// call can be retried at that moment.
			retry := caps.Window.Window - at.Sub(oldest)
			if retry < 0 {
				retry = 0
			}
			return Admission{Refusal: Refusal{
				Cap:        RefusalWindowFull,
				Limit:      int64(caps.Window.Limit),
				Used:       int64(len(st.events)),
				Requested:  1,
				RetryAfter: retry,
			}}
		}
	}

	// Token bucket: refill for elapsed time, then check the spend.
	if caps.Bucket.Capacity > 0 {
		s.refillLocked(st, caps.Bucket, at)
		if st.tokens < cost {
			deficit := cost - st.tokens
			// Ceiling of deficit/rate in whole seconds: the earliest moment
			// the bucket can hold the full cost.
			seconds := (deficit + caps.Bucket.Rate - 1) / caps.Bucket.Rate
			return Admission{Refusal: Refusal{
				Cap:        RefusalBucketEmpty,
				Limit:      caps.Bucket.Capacity,
				Used:       caps.Bucket.Capacity - st.tokens,
				Requested:  cost,
				RetryAfter: time.Duration(seconds) * time.Second,
			}}
		}
	}

	// Admit: consume both dimensions.
	if caps.Window.Limit > 0 {
		st.events = append(st.events, at)
	}
	if caps.Bucket.Capacity > 0 {
		st.tokens -= cost
	}
	return Admission{Admit: true}
}

// State reports the route's live cap state at time at: the admitted events
// still inside the window and the bucket's current token count. It refills and
// prunes first, so the numbers are what the next AdmitAt would see.
func (s *Store) State(route string, caps RouteCaps, at time.Time) (events int, tokens int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ok, _ := caps.Valid(); !ok {
		return 0, 0
	}
	st := s.routes[route]
	if st == nil {
		return 0, caps.Bucket.Capacity
	}
	if caps.Window.Limit > 0 {
		s.pruneLocked(st, caps.Window.Window, at)
	}
	if caps.Bucket.Capacity > 0 {
		s.refillLocked(st, caps.Bucket, at)
	}
	return len(st.events), st.tokens
}

// pruneLocked drops events at or before at-Window, leaving the half-open
// window (at-Window, at]. Caller holds s.mu.
func (s *Store) pruneLocked(st *routeState, window time.Duration, at time.Time) {
	cutoff := at.Add(-window)
	i := 0
	for i < len(st.events) && !st.events[i].After(cutoff) {
		i++
	}
	if i > 0 {
		st.events = append(st.events[:0], st.events[i:]...)
	}
}

// refillLocked tops the bucket up for the time elapsed since the last touch,
// capped at Capacity. A clock that moves backward (at before st.last) refills
// nothing — the bucket never gains tokens from skew. The gained amount is
// computed in float space so a long elapsed interval cannot overflow int64.
// Caller holds s.mu.
func (s *Store) refillLocked(st *routeState, b BucketCap, at time.Time) {
	if at.After(st.last) {
		gained := int64(float64(b.Rate) * at.Sub(st.last).Seconds())
		st.tokens += gained
		if st.tokens > b.Capacity {
			st.tokens = b.Capacity
		}
		st.last = at
	}
}
