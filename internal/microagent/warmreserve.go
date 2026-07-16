package microagent

import "sync"

// Warm-reserve: the Release-side half of the two-watermark warm band (#4035).
//
// The ResidentCap folds (WarmRefill / WarmPark in hibernate.go) decide HOW MANY agents to
// keep warm; they hold no agents themselves. WarmReserve is the missing MECHANISM: a
// bounded, per-id pool of still-LIVE Hibernable agents that Release parks warm — in RAM,
// never Frozen — instead of decommitting to disk. A same-id re-Admit within the band pops
// the live agent straight back with ZERO Thaw: no Freeze, no disk write, no restore
// round-trip, just a pointer hand-off. That is the documented cold-start tax
// (a full Freeze -> disk -> Thaw on the next wake — cmd/tokendemo/cold.go, the GLM-5.2
// cold-start note) removed on the warm path. It is the worker/agent-process twin of the
// shipped KV-layer prewarm (#810 / epics #809 / #1072).
//
// Composition (this type knows neither of the others):
//
//   - ResidentCap (hibernate.go) is the pure counter; its WarmPark(idle) fold says how many
//     idle residents to Reserve now (draining to low-water, keeping a warm reserve) and its
//     WarmRefill(parked) fold says how many parked agents to warm-wake.
//   - HibernationStore (hibernate.go) is the cold path: Park freezes to disk, Wake Thaws back.
//   - WarmReserve sits between them: a scheduler Reserves an idle live agent here up to cap N
//     on Release, and Takes it back on the next same-id Admit before ever touching the store.
//
// Two invariants keep it honest, both in the acceptance (#4035):
//
//   - Correctness never depends on a warm hit. A Take miss ALWAYS falls through to the cold
//     HibernationStore.Wake, so a miss costs exactly the status-quo cold start — never a lost
//     agent, never a wrong answer.
//   - Cap 0 disables it byte-identically. NewWarmReserve(0) makes Reserve always refuse and
//     Take always miss, so a store plus a cap-0 reserve behaves exactly like a plain store —
//     the "band off" posture.
//
// Honest tension (load-bearing, from the issue body): a warm agent holds RAM and goes stale.
// The reserve is therefore small (cap-bounded, set by the scheduler at or below the
// ResidentCap Limit / preflight effective cap) and sheddable — Drain (or a per-id Take that
// discards) lets a scheduler shed stale warm agents on a staleness horizon. A live wiring
// into the Host.run loop is the documented follow-on, as with the rest of this gen/future
// hibernation seam (see hibernate.go's package note).
//
// WarmReserve is safe for concurrent use.
type WarmReserve struct {
	mu    sync.Mutex
	limit int
	warm  map[string]Hibernable
}

// NewWarmReserve builds a warm reserve holding at most limit still-live agents. limit <= 0
// DISABLES the reserve — Reserve always refuses and Take always misses — which is the
// byte-identical "band off" posture (a store plus a cap-0 reserve equals a plain store).
func NewWarmReserve(limit int) *WarmReserve {
	return &WarmReserve{limit: limit, warm: map[string]Hibernable{}}
}

// Reserve parks a still-live agent warm under id, returning true only if it was held: there
// was room under the cap AND id was not already reserved AND h is non-nil. On false the caller
// decommits the agent the cold way (HibernationStore.Park) — so a warm miss is never lossy,
// it just costs the status-quo cold round-trip. A true Reserve must be matched by exactly one
// Take (or Drain, or Evict-via-Take) before the same id is Reserved again; the duplicate-id
// refusal enforces the one-warm-agent-per-id rule (an id maps to a single hibernated agent).
func (r *WarmReserve) Reserve(id string, h Hibernable) bool {
	if h == nil {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.limit <= 0 || len(r.warm) >= r.limit {
		return false
	}
	if _, dup := r.warm[id]; dup {
		return false
	}
	r.warm[id] = h
	return true
}

// Take pops the warm live agent for id — the ZERO-Thaw re-admit path. It returns (agent, true)
// on a warm hit and (nil, false) on a miss. A miss is the signal to fall through to the cold
// HibernationStore.Wake: the reserve is a fast path, never a correctness dependency. Taking an
// id frees its slot for the next Reserve. A scheduler that wants to shed a stale warm agent
// (rather than reuse it) Takes it and discards the returned agent.
func (r *WarmReserve) Take(id string) (Hibernable, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	h, ok := r.warm[id]
	if ok {
		delete(r.warm, id)
	}
	return h, ok
}

// Warm reports whether id currently has a live agent held warm.
func (r *WarmReserve) Warm(id string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	_, ok := r.warm[id]
	return ok
}

// Len reports how many live agents are held warm right now. It never exceeds Cap by
// construction (Reserve refuses past the cap) — the "never exceeding the preflight effective
// cap" half of the #4035 acceptance.
func (r *WarmReserve) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.warm)
}

// Cap reports the warm-reserve capacity; 0 means the reserve is disabled (band off).
func (r *WarmReserve) Cap() int { return r.limit }

// Drain empties the reserve, returning every held agent so the caller can cold-park or discard
// them — the shed path for the honest staleness tension (a warm agent holds RAM and goes
// stale) and for shutdown. After Drain the reserve is empty (Len 0). The returned slice is nil
// when the reserve was already empty.
func (r *WarmReserve) Drain() []Hibernable {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.warm) == 0 {
		return nil
	}
	out := make([]Hibernable, 0, len(r.warm))
	for id, h := range r.warm {
		out = append(out, h)
		delete(r.warm, id)
	}
	return out
}
