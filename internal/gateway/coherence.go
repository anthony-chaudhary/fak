package gateway

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// coherence.go — productionizes the vDSO coherence bus + revocation through `fak serve`,
// moving the finer eraser (vdso/scope.go) and the integrity eraser (vdso/revoke.go) off
// the turnbench harness and onto LIVE multi-agent gateway traffic. Three surfaces:
//
//   - Config.Invalidation sets the process-global vDSO granularity at New(), so a fleet
//     sharing one gateway runs namespace/resource scoped invalidation instead of the v0.1
//     global full-flush (one write no longer strands every other agent's warmed reads).
//   - the "what changed" FEED (fak_changes / GET /v1/fak/changes): a bounded, cursor-
//     drained stream of typed write Mutations + Revocations, so every agent can learn
//     precisely what an OTHER agent changed or refuted and re-plan / evict its own private
//     cache — the cross-agent coherence signal made first-class on the wire, not a blunt
//     "something, somewhere, changed."
//   - the refutation trigger (fak_revoke / POST /v1/fak/revoke): a caller that learns an
//     external world-state witness (a git commit / blob hash / lease epoch) is poisoned
//     revokes it; every pooled consumer is causally evicted, future re-admission under it
//     is refused, and the eviction is broadcast on the same feed.
//
// The feed observes the process-global vdso.Default — the SAME instance every kernel
// syscall drives through the registered fast path — so a write or refutation by ANY agent
// routed through this gateway is visible on it.

const defaultFeedCap = 1024

// CoherenceEvent is one wire-facing change-feed entry: a write Mutation or a Revocation,
// tagged by Kind and ordered by the vDSO's shared monotone Seq (the drain cursor).
type CoherenceEvent struct {
	Kind       string   `json:"kind"`              // "mutation" | "revocation"
	Seq        uint64   `json:"seq"`               // shared coherence-bus sequence (the cursor)
	Tool       string   `json:"tool,omitempty"`    // mutation: the write-shaped tool
	Tags       []string `json:"tags,omitempty"`    // mutation: the invalidation scope
	Witness    string   `json:"witness,omitempty"` // revocation: the refuted witness
	Evicted    int      `json:"evicted,omitempty"` // revocation: entries stranded
	WorldVer   uint64   `json:"world_ver"`         // consistency clock at the event
	TrustEpoch uint64   `json:"trust_epoch"`       // integrity clock at the event
	// principal is the isolation principal that produced a mutation (unexported: an
	// internal routing key, NOT wire-facing — emitting it would re-leak the very
	// tenant identity the scoped drain exists to hide). "" for a global/system write or
	// a revocation (integrity broadcast). drain() uses it to scope a tenant's feed to
	// its own mutations; see drain.
	principal string
}

// coherenceFeed is a bounded ring of CoherenceEvents filled by subscribing to the
// process-global vDSO's mutation + revocation buses. It is the server's sliding window
// onto the bus; a client drains it by cursor. Bounded so a never-draining client cannot
// grow it without limit — the oldest events fall off, and a client that lapses past the
// retained window sees a Seq gap and re-syncs to head.
type coherenceFeed struct {
	mu      sync.Mutex
	ring    []CoherenceEvent // chronological; oldest at front
	cap     int
	cancels []func()
}

// newCoherenceFeed subscribes to vdso.Default's mutation + revocation buses and records
// each event into a bounded ring. capacity<=0 uses defaultFeedCap.
func newCoherenceFeed(capacity int) *coherenceFeed {
	if capacity <= 0 {
		capacity = defaultFeedCap
	}
	f := &coherenceFeed{cap: capacity}
	f.cancels = append(f.cancels,
		vdso.Default.Subscribe(func(m vdso.Mutation) {
			f.add(CoherenceEvent{Kind: "mutation", Seq: m.Seq, Tool: m.Tool, Tags: m.Tags,
				WorldVer: m.WorldVer, TrustEpoch: vdso.Default.TrustEpoch(), principal: m.Principal})
		}),
		vdso.Default.SubscribeRevocations(func(rv vdso.Revocation) {
			f.add(CoherenceEvent{Kind: "revocation", Seq: rv.Seq, Witness: rv.Witness,
				Evicted: rv.Evicted, WorldVer: vdso.Default.WorldVersion(), TrustEpoch: rv.TrustEpoch})
		}),
	)
	return f
}

// appendRingCapped appends ev to a bounded ring buffer, dropping the oldest events
// so the slice never exceeds capacity. A non-positive capacity leaves the ring
// unbounded. Shared by the coherence and session-change feeds, which keep
// element-typed rings with the same trim policy.
func appendRingCapped[T any](ring []T, ev T, capacity int) []T {
	ring = append(ring, ev)
	if capacity > 0 && len(ring) > capacity {
		ring = ring[len(ring)-capacity:] // drop the oldest
	}
	return ring
}

func (f *coherenceFeed) add(ev CoherenceEvent) {
	f.mu.Lock()
	f.ring = appendRingCapped(f.ring, ev, f.cap)
	f.mu.Unlock()
}

// drain returns every retained event with Seq > sinceSeq (sinceSeq==0 => all retained)
// that is VISIBLE to the requesting principal, plus the highest Seq now known (the
// client's next cursor). Visibility (closing the cross-tenant metadata leak — a mutation's
// Tags name which entities another tenant changed):
//
//   - principal=="" — a single-tenant gateway, an admin drain, or the v0.1 caller that
//     names no principal: sees ALL events (unchanged behavior).
//   - principal==P — a tenant: sees only events it may learn of: its OWN mutations
//     (ev.principal==P) plus principal-less events (ev.principal==""), which are global
//     writes and REVOCATIONS — the latter are integrity broadcasts that must reach every
//     consumer for causal eviction, and carry only a content-hash witness.
//
// The cursor advances over ALL retained events (not just the visible ones), so a tenant's
// next-cursor stays monotone and it never re-scans another tenant's already-elapsed Seqs.
func (f *coherenceFeed) drain(principal string, sinceSeq uint64) ([]CoherenceEvent, uint64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]CoherenceEvent, 0, len(f.ring))
	cursor := sinceSeq
	for _, ev := range f.ring {
		if ev.Seq > sinceSeq && visibleTo(ev, principal) {
			out = append(out, ev)
		}
		if ev.Seq > cursor {
			cursor = ev.Seq
		}
	}
	return out, cursor
}

// visibleTo reports whether a draining principal may see an event. An empty drainer
// principal (single-tenant / admin) sees everything; a tenant sees principal-less events
// (global writes + revocations) and its own mutations, never a peer tenant's.
func visibleTo(ev CoherenceEvent, principal string) bool {
	return principal == "" || ev.principal == "" || ev.principal == principal
}

func (f *coherenceFeed) close() {
	f.mu.Lock()
	cancels := f.cancels
	f.cancels = nil
	f.mu.Unlock()
	for _, c := range cancels {
		c()
	}
}

// changes drains the change feed for events after the client's cursor that are visible
// to the requesting principal (a tenant sees only its own mutations + global broadcasts;
// an empty principal sees everything — single-tenant / admin).
func (s *Server) changes(principal string, sinceSeq uint64) ([]CoherenceEvent, uint64) {
	return s.feed.drain(principal, sinceSeq)
}

// revoke triggers a fleet-wide refutation of an external world-state witness on the
// process-global vDSO: every pooled entry admitted under it is causally evicted, future
// re-admission is refused, and the eviction is broadcast on the change feed. Returns the
// local eviction count and the post-bump integrity epoch.
func (s *Server) revoke(witness string) (evicted int, trustEpoch uint64) {
	evicted = vdso.Default.Revoke(witness)
	return evicted, vdso.Default.TrustEpoch()
}

// Close releases the gateway's coherence-bus subscriptions. The long-running CLI never
// needs it (the process owns the bus for its lifetime); tests that construct many
// Servers call it so subscriptions do not accumulate on the global vDSO.
func (s *Server) Close() {
	if s.feed != nil {
		s.feed.close()
	}
	// Detach the vDSO cache-event sink this server installed (the sink is a single
	// global slot, not a multi-subscriber bus), so a closed server's cache-stream
	// fold stops receiving events and tests that construct many Servers do not leave
	// a dangling sink on the process-global vDSO.
	if s.cacheStream != nil {
		vdso.Default.SetCacheEventSink(nil)
	}
}
