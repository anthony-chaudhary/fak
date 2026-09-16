package model

import "sync/atomic"

// metal_observability.go — the LIVE Metal/CPU routing tally behind `fak up` /healthz and the
// startup stamp (#12875). Before this, /healthz served a snapshot frozen once at model load
// (cmd/fak/up_native.go): `metal_live_q8_weights:0` and no fallback count, so an operator could
// not tell whether decode ran on Metal or silently fell back to CPU. This state is updated at
// the exact seams that already record routing (Session.recordMetalFallback) and residency
// promotion, and is read at request time.
//
// It is observability-only: no routing, kernel, quant-selection, or numerical path reads these
// counters. The hot-path cost is one atomic add per recorded fallback (already off the fast path
// for GPU-executed routes) plus one atomic add when residency changes.
//
// Why counters on *Model and not a PhaseProfiler: the profiler is opt-in and NOT goroutine-safe
// (profile.go), and `fak up` creates one Session per request (inkernel_planner.go), so a
// session-scoped counter would reset every turn. Model is the process-wide owner shared by every
// request session, so these tallies survive the whole serve.

// metalFallbackRoutes is the fixed size of the per-route counter vector. It is indexed by
// metalFallbackRouteIndex, which maps each MetalFallbackRoute to a stable slot. Keeping it a
// fixed array (not a map) means an increment is an atomic add with no allocation or lock.
const metalFallbackRoutes = 16

// metalLiveState is the lock-free live routing tally for one *Model.
type metalLiveState struct {
	// fallbacks counts, per MetalFallbackRoute slot, the promised routes that actually
	// executed CPU work. Only routes that CPU-executed are counted (the same predicate as
	// PhaseProfiler.MetalFallbackCount), so a route that merely returned control to the
	// caller's ordinary dispatcher is not misreported as a CPU fallback.
	fallbacks [metalFallbackRoutes]atomic.Int64
	// total counts every recorded CPU fallback across all routes.
	total atomic.Int64
	// q8Resident / q6kResident hold the most recent observed device-resident weight counts.
	// They are 0 until a residency promotion (or an explicit refresh) records them.
	q8Resident  atomic.Int64
	q6kResident atomic.Int64
	// residencyKnown is 1 once a residency refresh has run, so a reader can distinguish
	// "not yet observed" from a genuine 0-resident model.
	residencyKnown atomic.Bool
}

// metalLive returns (creating on first use) this model's live observability state. A nil model
// returns nil so every accessor is nil-safe.
func (m *Model) metalLive() *metalLiveState {
	if m == nil {
		return nil
	}
	// The field is written at most once under the same once-guard idiom other lazy model
	// state uses (see hasWeight/q8). Load-bearing correctness does not depend on the race:
	// a duplicate allocation would only mean a counter increment lands in a discarded state,
	// never a wrong value reported for the winning state. Callers reach this only from the
	// single-owner generation path plus request-time health reads, and Go's atomic ops on
	// [16]atomic.Int64 are pointer-stable.
	if m.metalObs == nil {
		m.metalObs = &metalLiveState{}
	}
	return m.metalObs
}

// recordMetalFallbackLive tallies one promised-route CPU fallback on the model. Slot is derived
// from the route; an unknown route still advances the total.
func (m *Model) recordMetalFallbackLive(route MetalFallbackRoute) {
	st := m.metalLive()
	if st == nil {
		return
	}
	if idx, ok := metalFallbackRouteIndex(route); ok {
		st.fallbacks[idx].Add(1)
	}
	st.total.Add(1)
}

// noteMetalResidency records the latest observed device-resident Q8/Q6_K weight counts.
func (m *Model) noteMetalResidency(q6k, q8 int) {
	st := m.metalLive()
	if st == nil {
		return
	}
	st.q6kResident.Store(int64(q6k))
	st.q8Resident.Store(int64(q8))
	st.residencyKnown.Store(true)
}

// MetalFallbackSnapshot is the live per-route CPU-fallback tally for this model. It is safe to
// call concurrently at request time (lock-free atomic reads) and returns whole-serve totals that
// survive per-request Session churn. Total is the scalar sum of the per-route counts.
type MetalFallbackSnapshot struct {
	Total    int            `json:"total"`
	ByRoute  map[string]int `json:"by_route,omitempty"`
	Observed bool           `json:"observed"`
}

// MetalFallbackSnapshot returns the live fallback tally. Observed is false when the model has
// never recorded a fallback AND never observed residency — i.e. the observer is untouched, which
// is the honest "no Metal decision has been taken through this model yet" state. Total is taken
// from the scalar counter (so a route with no dedicated slot is still counted) and therefore may
// exceed the sum of the per-route map, which is the truthful breakdown of the mapped routes only.
func (m *Model) MetalFallbackSnapshot() MetalFallbackSnapshot {
	st := m.metalLive()
	if st == nil {
		return MetalFallbackSnapshot{}
	}
	snap := MetalFallbackSnapshot{
		Total:    int(st.total.Load()),
		ByRoute:  map[string]int{},
		Observed: st.total.Load() != 0 || st.residencyKnown.Load(),
	}
	for i := 0; i < metalFallbackRoutes; i++ {
		if n := int(st.fallbacks[i].Load()); n != 0 {
			snap.ByRoute[metalFallbackRouteByIndex(i)] += n
		}
	}
	return snap
}

// MetalResidencySnapshot returns the most recently observed device-resident Q8/Q6_K weight
// counts and whether any observation has happened yet.
func (m *Model) MetalResidencySnapshot() (q6kResident, q8Resident int, known bool) {
	st := m.metalLive()
	if st == nil {
		return 0, 0, false
	}
	return int(st.q6kResident.Load()), int(st.q8Resident.Load()), st.residencyKnown.Load()
}

// RefreshMetalResidency re-samples the live device-resident weight counts from metalgemm and
// records them. It is the explicit "read live now" seam /healthz uses, so a lazy later upload is
// reflected instead of a frozen zero. The underlying metalgemm counters are O(N) scans over the
// resident registry (not atomic themselves), so this is intended for the low-rate health/stamp
// read path, never the per-token hot loop — callers must not invoke it per generated token.
func (m *Model) RefreshMetalResidency() (q6kResident, q8Resident int) {
	q6kResident, q8Resident = liveMetalWeightCounts()
	m.noteMetalResidency(q6kResident, q8Resident)
	return q6kResident, q8Resident
}
