package compute

import "sync"

// GraphEvictGuard is a host-side pending-use guard for the CUDA graph cache,
// adapted from llama.cpp's CUDA graph-pool eviction fix
// (ggml/src/ggml-cuda/common.cuh @00f113eceb2758158582d56f98bf476b2fe3ed32,
// MIT; see THIRD_PARTY_NOTICES.md, "llama.cpp"). Upstream, a captured graph that
// had been queued for imminent replay could still be evicted when graph-pool
// memory pressure triggered a cleanup pass, forcing a costly re-capture stall
// mid-decode. The fix guards the eviction decision on pending use: a graph with
// an outstanding replay is not a valid eviction victim.
//
// This guard tracks in-flight replays per graph key and answers the eviction
// predicate for them. It is keyed by the same string keys as GraphCachePolicy,
// and EvictableVictim selects a victim from that policy's LRU order rather than
// standing up a parallel registry. The policy's own room-miss eviction does not
// consult the guard: an eviction pass that must spare queued graphs picks its
// victim through EvictableVictim. It is host-buildable (no CUDA build tag) and
// safe for concurrent use.
type GraphEvictGuard struct {
	mu       sync.Mutex
	inFlight map[string]int
}

// NewGraphEvictGuard returns an empty guard. Every key is evictable until it is
// queued for replay.
func NewGraphEvictGuard() *GraphEvictGuard {
	return &GraphEvictGuard{inFlight: make(map[string]int)}
}

// QueueReplay records that a replay of the graph at key has been queued and is
// now in flight. The graph is non-evictable until a matching CompleteReplay.
// Replays may overlap, so the guard counts them: the graph stays non-evictable
// until the last outstanding replay completes.
func (g *GraphEvictGuard) QueueReplay(key string) {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.inFlight == nil {
		g.inFlight = make(map[string]int)
	}
	g.inFlight[key]++
}

// CompleteReplay records that one queued replay of the graph at key has
// finished. When the last outstanding replay completes the graph returns to the
// eviction pool. Completing a key with no outstanding replay is a no-op, so a
// stray completion can never make a still-in-flight graph evictable.
func (g *GraphEvictGuard) CompleteReplay(key string) {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	n, ok := g.inFlight[key]
	if !ok {
		return
	}
	if n <= 1 {
		delete(g.inFlight, key)
		return
	}
	g.inFlight[key] = n - 1
}

// Evictable reports whether the graph at key may be evicted. It is false while
// any replay of that graph is outstanding, so an eviction pass under memory
// pressure cannot free a graph that is still in flight.
func (g *GraphEvictGuard) Evictable(key string) bool {
	if g == nil {
		return true
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.inFlight[key] == 0
}

// CanEvict is an alias for Evictable, matching the upstream eviction-decision
// spelling.
func (g *GraphEvictGuard) CanEvict(key string) bool { return g.Evictable(key) }

// InFlight reports how many replays of the graph at key are currently
// outstanding.
func (g *GraphEvictGuard) InFlight(key string) int {
	if g == nil {
		return 0
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.inFlight[key]
}

// Pending reports the total number of outstanding replays across every key.
func (g *GraphEvictGuard) Pending() int {
	if g == nil {
		return 0
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	total := 0
	for _, n := range g.inFlight {
		total += n
	}
	return total
}

// EvictableVictim returns the least-recently-used key in the policy's LRU order
// that is not currently queued for replay, and whether such a key exists. It
// returns ("", false) when every retained entry is in flight, so an eviction
// pass under pressure skips the queued entries instead of freeing a graph that
// is still in flight.
//
// It reads the LRU order under the policy's lock, so it is safe beside a
// concurrent Run; the guard's own lock is only ever taken inside the policy's,
// never the other way round.
func (g *GraphEvictGuard) EvictableVictim(p *GraphCachePolicy) (string, bool) {
	if p == nil {
		return "", false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, key := range p.order {
		if g.Evictable(key) {
			return key, true
		}
	}
	return "", false
}
