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
				WorldVer: m.WorldVer, TrustEpoch: vdso.Default.TrustEpoch()})
		}),
		vdso.Default.SubscribeRevocations(func(rv vdso.Revocation) {
			f.add(CoherenceEvent{Kind: "revocation", Seq: rv.Seq, Witness: rv.Witness,
				Evicted: rv.Evicted, WorldVer: vdso.Default.WorldVersion(), TrustEpoch: rv.TrustEpoch})
		}),
	)
	return f
}

func (f *coherenceFeed) add(ev CoherenceEvent) {
	f.mu.Lock()
	f.ring = append(f.ring, ev)
	if len(f.ring) > f.cap {
		f.ring = f.ring[len(f.ring)-f.cap:] // drop the oldest
	}
	f.mu.Unlock()
}

// drain returns every retained event with Seq > sinceSeq (sinceSeq==0 => all retained),
// plus the highest Seq now known (the client's next cursor).
func (f *coherenceFeed) drain(sinceSeq uint64) ([]CoherenceEvent, uint64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]CoherenceEvent, 0, len(f.ring))
	cursor := sinceSeq
	for _, ev := range f.ring {
		if ev.Seq > sinceSeq {
			out = append(out, ev)
		}
		if ev.Seq > cursor {
			cursor = ev.Seq
		}
	}
	return out, cursor
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

// changes drains the change feed for events after the client's cursor.
func (s *Server) changes(sinceSeq uint64) ([]CoherenceEvent, uint64) {
	return s.feed.drain(sinceSeq)
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
}
