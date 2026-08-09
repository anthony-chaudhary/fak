package gitbroker

import (
	"errors"
	"sync"
	"sync/atomic"
)

// SINGLE-FLIGHT IS NOT A CACHE, AND THAT IS THE WHOLE POINT (#5623, rung 4 of
// #5619).
//
// Eight resident sessions plus N python dispatchers ask the same question of the
// same tree at the same moment. Coalescing those concurrent identical queries
// into ONE execution collapses that fan-in with no invalidation reasoning at
// all: the answer is still computed fresh from git, it is just computed once and
// handed to everyone who was already waiting. Nothing is stored, nothing
// outlives the call, so there is nothing that can go stale between calls.
//
// THE STALENESS BOUND, STATED. A joiner observes a snapshot taken when the
// LEADER's execution began, not when the joiner arrived. So a joiner's answer is
// stale by at most the leader's execution time — one `git` invocation. A caller
// that ran its own git would have paid that same interval as latency instead; the
// difference is which end of the interval the snapshot comes from. That bound is
// acceptable for Class A (immutable — the bound is vacuous) and Class B
// (informational). It is NOT acceptable for Class C, which is why a decisional
// query never reaches this group at all: see Class.Decisional.
//
// A LATE ARRIVAL DOES NOT JOIN. The in-flight entry is removed from the map
// BEFORE the leader's result is published, so a caller arriving after the leader
// finished starts its own fresh execution rather than adopting a completed
// answer. Without that ordering the bound above would silently widen from "the
// leader's execution" to "the leader's execution plus however long the entry
// lingered", which is a cache with no key — exactly the thing this file is not.

// errFlightAbandoned is what joiners see if a leader's execution dies without
// publishing a result (a panic unwinding through Do). Joiners get an ERROR they
// can fall back on, never a zero value that would read as a legitimate answer —
// the stallscan rule: "the thing broke" must not be spelled like "the tree is
// clean".
var errFlightAbandoned = errors.New("gitbroker: in-flight query abandoned")

// flightGroup coalesces concurrent identical calls, keyed by a query string.
// The zero value is ready to use.
type flightGroup[T any] struct {
	mu sync.Mutex
	m  map[string]*flight[T]

	// coalesced counts callers that JOINED someone else's execution — i.e. the
	// git invocations single-flight removed. It is reported as its own Stats
	// field so single-flight's contribution stays attributable rather than
	// bundled into the cache's hit count (#5623's acceptance gate).
	coalesced atomic.Int64
}

type flight[T any] struct {
	wg  sync.WaitGroup
	val T
	err error
}

// Do runs fn for key, or waits on an execution already in flight for that key.
// shared reports whether this caller joined another's execution rather than
// running fn itself.
//
// The error comes LAST, unlike the well-known x/sync/singleflight spelling
// (v, err, shared). That ordering is a standing trap — `v, _, err :=` compiles
// and silently binds err to a bool — so this group follows the Go convention
// instead.
func (g *flightGroup[T]) Do(key string, fn func() (T, error)) (val T, shared bool, err error) {
	g.mu.Lock()
	if g.m == nil {
		g.m = make(map[string]*flight[T])
	}
	if inflight, ok := g.m[key]; ok {
		g.mu.Unlock()
		inflight.wg.Wait()
		g.coalesced.Add(1)
		return inflight.val, true, inflight.err
	}
	f := new(flight[T])
	// Pre-set the error so an execution that never publishes (panic) leaves
	// joiners with a failure rather than a convincing-looking zero value.
	f.err = errFlightAbandoned
	f.wg.Add(1)
	g.m[key] = f
	g.mu.Unlock()

	defer func() {
		// Delete BEFORE Done: a caller that arrives from here on must start its
		// own execution instead of adopting an answer that is already finished.
		g.mu.Lock()
		delete(g.m, key)
		g.mu.Unlock()
		f.wg.Done()
	}()

	v, ferr := fn()
	f.val, f.err = v, ferr
	return v, false, ferr
}

// Coalesced reports how many callers joined an execution instead of running
// their own — the single-flight saving, countable on its own.
func (g *flightGroup[T]) Coalesced() int64 { return g.coalesced.Load() }
