package gateway

// replica_ejection.go — escalating dispatch ejection and all-unhealthy fail-open for
// the fleet membership registry.
//
// Ejection: when dispatch failures (not probes) cross a healthy worker to unhealthy,
// the worker is ejected for a window that a successful readiness probe cannot cut
// short. The first ejection lasts ejectBase, each repeat adds another ejectBase, and
// the window is capped at ejectMax. A worker whose last window ended at least ejectMax
// ago starts over at the base window. Without this a replica that accepts OPTIONS but
// fails real requests flaps back in on every 5s probe tick.
//
// Fail-open: when every non-draining holder of a model is unhealthy, placement admits
// them all instead of returning ErrNoHealthyWorker — a degraded answer beats a
// guaranteed outage when health evidence is uniformly bad. Draining workers and
// never-probed (unknown) workers are not resurrected.
//
// Concepts (Apache-2.0, re-expressed, no code copied):
//   llm-d-inference-scheduler pkg/epp/framework/plugins/filter/endpointhealth
//   @329a52efb001905378a98b38435a69b4feab88c2 (escalating ejection; all-unhealthy keeps all)
//   sglang sgl-router/src/health/circuit_breaker.rs@9285994accb631f8e0b04ff47f620d01814aba21
//   (all breakers open fails open).
// The durations are seeds for auto-tune, not values tuned for fak traffic.

import "time"

const (
	// defaultEjectBase is the first dispatch-ejection window and the per-repeat increment.
	defaultEjectBase = 30 * time.Second
	// defaultEjectMax caps the ejection window and is the quiet period after which
	// the escalation resets.
	defaultEjectMax = 5 * time.Minute
)

// ejectedLocked reports whether w is inside a dispatch-ejection window. Caller holds
// the lock.
func (m *FleetMembership) ejectedLocked(w *memberWorker) bool {
	return !w.ejectedUntil.IsZero() && m.now().Before(w.ejectedUntil)
}

// ejectLocked opens (or escalates) w's ejection window. Caller holds the lock.
func (m *FleetMembership) ejectLocked(w *memberWorker) {
	now := m.now()
	if !w.ejectedUntil.IsZero() && now.Sub(w.ejectedUntil) >= m.ejectMax {
		w.ejections = 0
	}
	if limit := int(m.ejectMax/m.ejectBase) + 1; w.ejections < limit {
		w.ejections++
	}
	d := time.Duration(w.ejections) * m.ejectBase
	if d > m.ejectMax {
		d = m.ejectMax
	}
	w.ejectedUntil = now.Add(d)
}

// failOpenCandidate reports whether w may be admitted when no holder of the model is
// admissible: unhealthy on evidence, but not draining. Caller holds the lock.
func (w *memberWorker) failOpenCandidate() bool {
	return w.health == HealthUnhealthy && !w.draining
}
