package gateway

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Identity is the allocation itself: even known-zero physical attempts own
// distinct entries. This ledger is separate from decode occupancy and spending.
type observedPrefillAttempt struct{ tokens int }

// Nonzero-sized allocation gives each placement an identity that cannot wrap or
// alias a still-live bound context, even when the same worker is selected again.
type observedPrefillEpoch struct{ marker byte }

// Caller holds FleetMembership.mu. Sum from live entries instead of saturating
// a running total: overflow disappears correctly when an attempt completes.
func (w *memberWorker) statusWithObservedPrefillLocked() WorkerStatus {
	s := WorkerStatus{Incarnation: w.incarnation, Spec: w.spec, Health: w.health, Draining: w.draining, Inflight: w.inflight, BookedOutputBlocks: w.bookedOutputBlocks}
	s.ObservedPrefillAttempts = len(w.observedPrefills)
	max := int(^uint(0) >> 1)
	for attempt := range w.observedPrefills {
		if attempt.tokens > max-s.ObservedPrefillTokens {
			s.ObservedPrefillTokens = max
			s.ObservedPrefillOverflow = true
		} else {
			s.ObservedPrefillTokens += attempt.tokens
		}
	}
	return s
}

// startObservedPrefill books exact owned work only while the captured worker
// still owns this reservation. Each native retry gets a separate entry. Completion
// removes only that entry, never the current reservation's replacement attempt.
// Cancellation alone does not mean compute stopped: the native observer's unwind
// callback releases the entry. Release/Retarget continue to own decode occupancy.
func (r *fleetReservation) startObservedPrefill(worker *memberWorker, epoch *observedPrefillEpoch, prompt, cached int) func(bool) {
	if r == nil || r.fleet == nil || worker == nil || epoch == nil || prompt < 0 || cached < 0 || cached > prompt {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	m := r.fleet
	m.mu.Lock()
	defer m.mu.Unlock()
	if r.released || r.prefillEpoch != epoch || r.engine != EngineNative || r.worker != worker || m.workers[r.workerID] != worker {
		return nil
	}
	attempt := &observedPrefillAttempt{tokens: prompt - cached}
	if worker.observedPrefills == nil {
		worker.observedPrefills = make(map[*observedPrefillAttempt]struct{})
	}
	worker.observedPrefills[attempt] = struct{}{}
	return func(bool) {
		m.mu.Lock()
		delete(worker.observedPrefills, attempt)
		m.mu.Unlock()
	}
}

// withObservedPrefill is used at every physical buffered/streaming/hedged call.
// Only an actual native planner and live native membership reservation bind.
// Existing caller instrumentation is preserved by declining automatic binding.
// These partial observations MUST NOT be fed to the routing scorer as full load.
func (route reservedPlannerReplica) withObservedPrefill(ctx context.Context) context.Context {
	p, ok := route.replica.Planner.(*agent.InKernelPlanner)
	r := route.reservation
	if !ok || p == nil || r == nil || r.fleet == nil {
		return ctx
	}
	r.mu.Lock()
	worker, epoch := r.worker, r.prefillEpoch
	valid := !r.released && r.engine == EngineNative && r.workerID == route.replica.Name
	r.mu.Unlock()
	if !valid {
		return ctx
	}
	bound, _ := p.TryWithOwnedPrefillObservation(ctx, func(prompt, cached int) func(bool) {
		return r.startObservedPrefill(worker, epoch, prompt, cached)
	})
	return bound
}
