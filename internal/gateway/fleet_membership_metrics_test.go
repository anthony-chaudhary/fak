package gateway

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// Acceptance (#42): membership/health/drain/failover transitions are emitted on
// the serving-metrics surface with a per-worker label. The registry records the
// transitions into its log; this asserts the metrics bridge renders them as
// per-worker-labeled Prometheus lines in the gateway /metrics format, and that a
// removed worker's transition COUNTERS persist while its live state GAUGES drop.
func TestFleetMembershipMetricsEmitsPerWorkerTransitions(t *testing.T) {
	sp := newScriptedProbe()
	m := NewFleetMembership(MembershipConfig{HealthyAfter: 1, UnhealthyAfter: 1, Probe: sp.probe})
	mustAdd(t, m, WorkerSpec{ID: "w1", Role: RolePrefill, Engine: EngineExternal, Endpoint: "h1:8000"})
	mustAdd(t, m, WorkerSpec{ID: "w2", Role: RoleDecode, Engine: EngineNative, Endpoint: "h2:8000"})

	m.ProbeOnce(context.Background()) // both unknown -> healthy
	sp.set("w1", false)
	m.ProbeOnce(context.Background())     // w1 healthy -> unhealthy
	if err := m.Drain("w1"); err != nil { // unhealthy + idle -> drain_started + removed
		t.Fatalf("Drain w1: %v", err)
	}

	fm := NewFleetMembershipMetrics()
	var b strings.Builder
	fm.Publish(&b, m) // drains the transition log and renders counters + live gauges
	out := b.String()

	// Per-worker transition counters — including w1, which was REMOVED: a counter
	// must not vanish when the worker leaves the live set.
	wantLines := []string{
		`fak_gateway_fleet_membership_transitions_total{worker="w1",kind="added"} 1`,
		`fak_gateway_fleet_membership_transitions_total{worker="w1",kind="health_changed",health="healthy"} 1`,
		`fak_gateway_fleet_membership_transitions_total{worker="w1",kind="health_changed",health="unhealthy"} 1`,
		`fak_gateway_fleet_membership_transitions_total{worker="w1",kind="drain_started"} 1`,
		`fak_gateway_fleet_membership_transitions_total{worker="w1",kind="removed"} 1`,
		`fak_gateway_fleet_membership_transitions_total{worker="w2",kind="added"} 1`,
		`fak_gateway_fleet_membership_transitions_total{worker="w2",kind="health_changed",health="healthy"} 1`,
		// Live state gauges carry the per-worker label and the spec identity; only w2
		// survives (w1 was drained out), proving the gauge tracks the live fleet.
		`fak_gateway_fleet_worker_health{worker="w2",role="decode",engine="native",health="healthy"} 1`,
		`fak_gateway_fleet_worker_admissible{worker="w2"} 1`,
		`fak_gateway_fleet_worker_draining{worker="w2"} 0`,
		`fak_gateway_fleet_worker_inflight{worker="w2"} 0`,
	}
	for _, want := range wantLines {
		if !strings.Contains(out, want+"\n") {
			t.Fatalf("metrics surface missing line %q\n--- got ---\n%s", want, out)
		}
	}

	// The removed worker w1 keeps its counters but drops out of the live gauges.
	if strings.Contains(out, `fak_gateway_fleet_worker_health{worker="w1"`) {
		t.Fatalf("removed worker w1 still has a live health gauge:\n%s", out)
	}

	// Prometheus hygiene: every metric family has exactly one HELP and one TYPE.
	for _, name := range []string{
		"fak_gateway_fleet_membership_transitions_total",
		"fak_gateway_fleet_worker_health",
		"fak_gateway_fleet_worker_admissible",
		"fak_gateway_fleet_worker_draining",
		"fak_gateway_fleet_worker_inflight",
	} {
		if got := strings.Count(out, "# HELP "+name+" "); got != 1 {
			t.Fatalf("metric %q HELP count = %d, want 1", name, got)
		}
		if got := strings.Count(out, "# TYPE "+name+" "); got != 1 {
			t.Fatalf("metric %q TYPE count = %d, want 1", name, got)
		}
	}
}

// Failover is the in-flight-loss transition; assert it reaches the surface labeled
// with the worker the request moved OFF of, and that the inflight/draining gauges
// reflect a live acquire + drain.
func TestFleetMembershipMetricsFailoverAndLiveGauges(t *testing.T) {
	sp := newScriptedProbe()
	m := NewFleetMembership(MembershipConfig{HealthyAfter: 1, UnhealthyAfter: 5, Probe: sp.probe})
	mustAdd(t, m, WorkerSpec{ID: "wa"})
	mustAdd(t, m, WorkerSpec{ID: "wb"})
	m.ProbeOnce(context.Background())

	// wa fails the send, the request fails over to wb -> a failover transition on wa.
	if _, err := m.Dispatch(context.Background(), func(_ context.Context, s WorkerSpec) error {
		if s.ID == "wa" {
			return errors.New("connection refused")
		}
		return nil
	}); err != nil {
		t.Fatalf("Dispatch: %v", err)
	}

	// A live request is in flight on wb, and wb is draining (in-flight not yet done).
	if err := m.Acquire("wb"); err != nil {
		t.Fatalf("Acquire wb: %v", err)
	}
	if err := m.Drain("wb"); err != nil {
		t.Fatalf("Drain wb: %v", err)
	}

	fm := NewFleetMembershipMetrics()
	var b strings.Builder
	fm.Publish(&b, m)
	out := b.String()

	for _, want := range []string{
		`fak_gateway_fleet_membership_transitions_total{worker="wa",kind="failover"} 1`,
		`fak_gateway_fleet_worker_draining{worker="wb"} 1`,
		`fak_gateway_fleet_worker_inflight{worker="wb"} 1`,
		`fak_gateway_fleet_worker_admissible{worker="wb"} 0`, // draining -> not admissible
	} {
		if !strings.Contains(out, want+"\n") {
			t.Fatalf("metrics surface missing line %q\n--- got ---\n%s", want, out)
		}
	}
}

// A counter must be monotonic across scrapes and must not double-count: Publish
// drains the registry log each time, so a second Publish with no new transitions
// leaves the counts unchanged, and a fresh batch adds on top.
func TestFleetMembershipMetricsCounterMonotonicNoDoubleCount(t *testing.T) {
	sp := newScriptedProbe()
	m := NewFleetMembership(MembershipConfig{HealthyAfter: 1, UnhealthyAfter: 1, Probe: sp.probe})
	mustAdd(t, m, WorkerSpec{ID: "w1"})
	m.ProbeOnce(context.Background()) // added + health_changed(healthy)

	fm := NewFleetMembershipMetrics()
	var first strings.Builder
	fm.Publish(&first, m)
	if want := `fak_gateway_fleet_membership_transitions_total{worker="w1",kind="added"} 1`; !strings.Contains(first.String(), want+"\n") {
		t.Fatalf("first scrape missing %q\n%s", want, first.String())
	}

	// No new transitions: the second scrape must still read 1, not 2 (Publish drained
	// the log, so nothing is re-counted).
	var second strings.Builder
	fm.Publish(&second, m)
	if want := `fak_gateway_fleet_membership_transitions_total{worker="w1",kind="added"} 1`; !strings.Contains(second.String(), want+"\n") {
		t.Fatalf("second scrape double-counted or lost the added counter\n%s", second.String())
	}

	// A new transition accumulates on top of the retained count.
	sp.set("w1", false)
	m.ProbeOnce(context.Background()) // health_changed(unhealthy)
	m.ProbeOnce(context.Background()) // no transition (already unhealthy)
	var third strings.Builder
	fm.Publish(&third, m)
	if want := `fak_gateway_fleet_membership_transitions_total{worker="w1",kind="health_changed",health="unhealthy"} 1`; !strings.Contains(third.String(), want+"\n") {
		t.Fatalf("third scrape missing the new unhealthy transition\n%s", third.String())
	}
	// The retained added counter is still exactly 1.
	if want := `fak_gateway_fleet_membership_transitions_total{worker="w1",kind="added"} 1`; !strings.Contains(third.String(), want+"\n") {
		t.Fatalf("retained added counter changed across scrapes\n%s", third.String())
	}
}

// Acceptance (#42): the transitions reach the REAL gateway serving-metrics surface
// (renderMetrics / the /metrics handler), not just the bridge in isolation. A Server
// emits no fleet family until a loop is attached (no phantom series); once attached,
// scraping /metrics drains the loop and renders the per-worker transition counters and
// live state gauges; detaching with nil stops new publishes.
func TestFleetMembershipMetricsRenderedOnServingSurface(t *testing.T) {
	srv := newTestServer(t)

	// No fleet attached -> the family is absent from the surface (host-injected, inert).
	if pre := srv.renderMetrics(); strings.Contains(pre, "fak_gateway_fleet_membership_transitions_total") {
		t.Fatalf("fleet family present before SetFleetMembership:\n%s", pre)
	}

	sp := newScriptedProbe()
	m := NewFleetMembership(MembershipConfig{HealthyAfter: 1, UnhealthyAfter: 1, Probe: sp.probe})
	mustAdd(t, m, WorkerSpec{ID: "w1", Role: RolePrefill, Engine: EngineExternal, Endpoint: "h1:8000"})
	m.ProbeOnce(context.Background()) // added + health_changed(healthy)
	srv.SetFleetMembership(m)

	out := srv.renderMetrics()
	for _, want := range []string{
		`fak_gateway_fleet_membership_transitions_total{worker="w1",kind="added"} 1`,
		`fak_gateway_fleet_membership_transitions_total{worker="w1",kind="health_changed",health="healthy"} 1`,
		`fak_gateway_fleet_worker_health{worker="w1",role="prefill",engine="external",health="healthy"} 1`,
		`fak_gateway_fleet_worker_admissible{worker="w1"} 1`,
	} {
		if !strings.Contains(out, want+"\n") {
			t.Fatalf("serving-metrics surface missing line %q\n--- got ---\n%s", want, out)
		}
	}

	// Detaching stops publishing the live family (the counters live on the bridge, which
	// the Server no longer reaches once fleet is nil).
	srv.SetFleetMembership(nil)
	if post := srv.renderMetrics(); strings.Contains(post, `fak_gateway_fleet_worker_health{worker="w1"`) {
		t.Fatalf("fleet gauges still present after detaching the loop:\n%s", post)
	}
}

// Nil/empty guards: the bridge never panics on a nil receiver, nil builder, or
// empty fleet, and renders the metric headers even with no workers.
func TestFleetMembershipMetricsNilAndEmptySafe(t *testing.T) {
	var nilFM *FleetMembershipMetrics
	nilFM.Ingest(nil)
	nilFM.IngestEvents(nil)
	nilFM.Render(nil, nil)
	nilFM.Publish(nil, nil)

	fm := NewFleetMembershipMetrics()
	m := NewFleetMembership(MembershipConfig{})
	var b strings.Builder
	fm.Publish(&b, m)
	if !strings.Contains(b.String(), "# TYPE fak_gateway_fleet_membership_transitions_total counter") {
		t.Fatalf("empty fleet did not render the transition counter header:\n%s", b.String())
	}
}
