package gateway

import (
	"sort"
	"strconv"
	"strings"
	"sync"
)

// FleetMembershipMetrics drains the live membership/health/drain/failover loop
// onto the gateway serving-metrics surface — the Prometheus text the /metrics
// handler serves — every line carrying a per-worker `worker` label so the
// operator and any autoscaler/planner can watch membership move.
//
// FleetMembership records each transition into an internal log (DrainEvents) and
// holds no metrics dependency of its own, by design: the registry stays testable
// and surface-agnostic, and THIS is the bridge that publishes that log. Ingest
// folds the drained transition log into cumulative per-(worker, kind) counters —
// a removed worker's counts persist, the way a counter must — and Render writes
// both those counters and the live per-worker health/drain/inflight gauges read
// from a Snapshot. The counter side is the transition history; the gauge side is
// the current fleet state.
type FleetMembershipMetrics struct {
	mu sync.Mutex
	// counts is the cumulative transition tally keyed by worker, kind, and (for a
	// health_changed transition only) the state entered. It is monotonic and
	// survives a worker's removal so the counter never goes backwards.
	counts map[membershipCounterKey]uint64
}

// membershipCounterKey identifies one transition counter line. health is the
// empty string for every kind except EventHealthChanged, where it is the state
// the worker entered (the axis an alert watches: entries into "unhealthy").
type membershipCounterKey struct {
	worker string
	kind   MembershipKind
	health WorkerHealth
}

// NewFleetMembershipMetrics builds an empty accumulator. Feed it the registry's
// transition log with Ingest on the metrics publish cadence, then Render it (or
// call Publish to do both) inside the serving-metrics handler.
func NewFleetMembershipMetrics() *FleetMembershipMetrics {
	return &FleetMembershipMetrics{counts: make(map[membershipCounterKey]uint64)}
}

// Ingest drains m's transition log and folds every event into the per-worker
// counters. It is the live wiring: call it on the metrics publish cadence so no
// transition is lost between scrapes (DrainEvents clears the registry log).
func (fm *FleetMembershipMetrics) Ingest(m *FleetMembership) {
	if fm == nil || m == nil {
		return
	}
	fm.IngestEvents(m.DrainEvents())
}

// IngestEvents folds an already-drained transition log into the counters. It is
// the testable core of Ingest, decoupled from the registry so a test can assert
// the exact per-worker-labeled lines a known sequence of transitions produces.
func (fm *FleetMembershipMetrics) IngestEvents(events []MembershipEvent) {
	if fm == nil || len(events) == 0 {
		return
	}
	fm.mu.Lock()
	defer fm.mu.Unlock()
	if fm.counts == nil {
		fm.counts = make(map[membershipCounterKey]uint64)
	}
	for _, ev := range events {
		key := membershipCounterKey{worker: ev.WorkerID, kind: ev.Kind}
		if ev.Kind == EventHealthChanged {
			key.health = ev.To
		}
		fm.counts[key]++
	}
}

// Render writes the accumulated transition counters and the live per-worker state
// gauges (read from snap) to b as Prometheus text, matching the gateway /metrics
// format. snap is the registry's current Snapshot; pass it so the gauges reflect
// the live fleet while the counters carry the full transition history (including
// workers already removed). Output is deterministic: counters sort by
// worker/kind/health, gauges follow snap's registration order.
func (fm *FleetMembershipMetrics) Render(b *strings.Builder, snap []WorkerStatus) {
	if fm == nil || b == nil {
		return
	}
	fm.renderTransitions(b)
	renderWorkerGauges(b, snap)
}

// Publish drains m's transitions into the counters and renders the full fleet
// membership block in one call — the convenience the serving-metrics handler uses.
func (fm *FleetMembershipMetrics) Publish(b *strings.Builder, m *FleetMembership) {
	if fm == nil || b == nil || m == nil {
		return
	}
	fm.Ingest(m)
	fm.Render(b, m.Snapshot())
}

func (fm *FleetMembershipMetrics) renderTransitions(b *strings.Builder) {
	fm.mu.Lock()
	keys := make([]membershipCounterKey, 0, len(fm.counts))
	for k := range fm.counts {
		keys = append(keys, k)
	}
	vals := make([]uint64, len(keys))
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].worker != keys[j].worker {
			return keys[i].worker < keys[j].worker
		}
		if keys[i].kind != keys[j].kind {
			return keys[i].kind < keys[j].kind
		}
		return keys[i].health < keys[j].health
	})
	for i, k := range keys {
		vals[i] = fm.counts[k]
	}
	fm.mu.Unlock()

	writeHelpType(b, "fak_gateway_fleet_membership_transitions_total",
		"Worker membership/health/drain/failover transitions the live fleet loop recorded since process start, by kind (and entered health state for a health_changed transition), per worker.",
		"counter")
	for i, k := range keys {
		if k.kind == EventHealthChanged {
			b.WriteString("fak_gateway_fleet_membership_transitions_total{worker=\"")
			b.WriteString(promQuote(k.worker))
			b.WriteString("\",kind=\"")
			b.WriteString(promQuote(string(k.kind)))
			b.WriteString("\",health=\"")
			b.WriteString(promQuote(string(k.health)))
			b.WriteString("\"} ")
		} else {
			b.WriteString("fak_gateway_fleet_membership_transitions_total{worker=\"")
			b.WriteString(promQuote(k.worker))
			b.WriteString("\",kind=\"")
			b.WriteString(promQuote(string(k.kind)))
			b.WriteString("\"} ")
		}
		writeUint(b, vals[i])
	}
}

// renderWorkerGauges writes the live per-worker state from a Snapshot: current
// health (one line for the state the worker is in), admissibility, draining, and
// in-flight count. A drained worker that has already been removed is absent from
// snap, so its gauges drop while its transition counters persist.
func renderWorkerGauges(b *strings.Builder, snap []WorkerStatus) {
	writeHelpType(b, "fak_gateway_fleet_worker_health",
		"Current per-worker health as the live fleet loop sees it (value 1 on the line for the worker's current state).",
		"gauge")
	for _, w := range snap {
		b.WriteString("fak_gateway_fleet_worker_health{worker=\"")
		b.WriteString(promQuote(w.Spec.ID))
		b.WriteString("\",role=\"")
		b.WriteString(promQuote(string(w.Spec.Role)))
		b.WriteString("\",engine=\"")
		b.WriteString(promQuote(string(w.Spec.Engine)))
		b.WriteString("\",health=\"")
		b.WriteString(promQuote(string(w.Health)))
		b.WriteString("\"} 1\n")
	}

	writeHelpType(b, "fak_gateway_fleet_worker_admissible",
		"Whether the router may place NEW work on the worker (1 when healthy and not draining, else 0).",
		"gauge")
	for _, w := range snap {
		b.WriteString("fak_gateway_fleet_worker_admissible{worker=\"")
		b.WriteString(promQuote(w.Spec.ID))
		b.WriteString("\"} ")
		writeBool(b, w.Health == HealthHealthy && !w.Draining)
	}

	writeHelpType(b, "fak_gateway_fleet_worker_draining",
		"Whether the worker is draining (1) — the router routes no new work to it while in-flight requests finish.",
		"gauge")
	for _, w := range snap {
		b.WriteString("fak_gateway_fleet_worker_draining{worker=\"")
		b.WriteString(promQuote(w.Spec.ID))
		b.WriteString("\"} ")
		writeBool(b, w.Draining)
	}

	writeHelpType(b, "fak_gateway_fleet_worker_inflight",
		"In-flight requests acquired against the worker (drain waits for this to reach zero before removal).",
		"gauge")
	for _, w := range snap {
		b.WriteString("fak_gateway_fleet_worker_inflight{worker=\"")
		b.WriteString(promQuote(w.Spec.ID))
		b.WriteString("\"} ")
		writeInt(b, w.Inflight)
	}
}

// SetFleetMembership attaches (or, with nil, detaches) the live membership loop whose
// transitions the gateway /metrics surface publishes (#42). It is settable after New so a
// host can build the fleet view once it knows its workers, mirroring SetKVResidencyReclaimer /
// SetModelLoadProfile, and it lazily mints the cumulative accumulator the scrape folds into so
// the transition counters survive across scrapes. A nil receiver is a no-op.
func (s *Server) SetFleetMembership(m *FleetMembership) {
	if s == nil {
		return
	}
	s.fleetMu.Lock()
	s.fleet = m
	if m != nil && s.fleetMetrics == nil {
		s.fleetMetrics = NewFleetMembershipMetrics()
	}
	s.fleetMu.Unlock()
}

// writeFleetMembershipMetrics publishes the live fleet membership family onto the gateway
// /metrics surface (#42): it drains the attached loop's transition log into the cumulative
// per-(worker,kind) counter and renders that counter plus the current per-worker state gauges.
// A Server with no fleet attached emits nothing (no phantom worker series) — the same
// host-injected, inert-by-default posture as the KV-residency / pressure-relief seams.
func (s *Server) writeFleetMembershipMetrics(b *strings.Builder) {
	if s == nil || b == nil {
		return
	}
	s.fleetMu.Lock()
	fleet := s.fleet
	fm := s.fleetMetrics
	s.fleetMu.Unlock()
	if fleet == nil || fm == nil {
		return
	}
	fm.Publish(b, fleet)
}

func writeUint(b *strings.Builder, n uint64) {
	b.WriteString(strconv.FormatUint(n, 10))
	b.WriteByte('\n')
}

func writeInt(b *strings.Builder, n int) {
	b.WriteString(strconv.Itoa(n))
	b.WriteByte('\n')
}

func writeBool(b *strings.Builder, v bool) {
	if v {
		b.WriteString("1\n")
		return
	}
	b.WriteString("0\n")
}
