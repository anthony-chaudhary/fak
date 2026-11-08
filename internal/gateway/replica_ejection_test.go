package gateway

import (
	"context"
	"errors"
	"io"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fakeClock is a manually advanced membership clock.
type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time          { return c.t }
func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

// ejectionFleet registers w1/w2 healthy under default hysteresis (HealthyAfter=1,
// UnhealthyAfter=2) and default ejection seeds (30s base, 5m cap) on a fake clock.
func ejectionFleet(t *testing.T) (*FleetMembership, *scriptedProbe, *fakeClock) {
	t.Helper()
	sp := newScriptedProbe()
	clk := &fakeClock{t: time.Unix(1_800_000_000, 0)}
	m := NewFleetMembership(MembershipConfig{Probe: sp.probe, Now: clk.now})
	mustAdd(t, m, WorkerSpec{ID: "w1"})
	mustAdd(t, m, WorkerSpec{ID: "w2"})
	m.ProbeOnce(context.Background())
	return m, sp, clk
}

func healthOf(t *testing.T, m *FleetMembership, id string) WorkerHealth {
	t.Helper()
	s, ok := fleetWorkerStatus(m, id)
	if !ok {
		t.Fatalf("%s not registered", id)
	}
	return s.Health
}

// ejectByDispatch drives UnhealthyAfter transport failures against id so it
// crosses to unhealthy, which opens an ejection window.
func ejectByDispatch(t *testing.T, m *FleetMembership, id string) {
	t.Helper()
	for i := 0; i < m.unhealthyAfter; i++ {
		m.markDispatchFailure(id, io.ErrUnexpectedEOF)
	}
	if h := healthOf(t, m, id); h != HealthUnhealthy {
		t.Fatalf("%s health after %d dispatch failures = %s, want unhealthy", id, m.unhealthyAfter, h)
	}
}

// ejectionWindow is how long a successful probe stays unable to re-admit id.
func ejectionWindow(m *FleetMembership, id string) time.Duration {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.workers[id].ejectedUntil.Sub(m.now())
}

// fak-test:runtime fast est=10ms lane=default
func TestReplicaEjectionNthDispatchFailureEjectsForBase(t *testing.T) {
	m, _, clk := ejectionFleet(t)
	ctx := context.Background()

	m.markDispatchFailure("w1", io.ErrUnexpectedEOF)
	if h := healthOf(t, m, "w1"); h != HealthHealthy {
		t.Fatalf("N-1 failures: w1 health = %s, want healthy", h)
	}
	m.markDispatchFailure("w1", io.ErrUnexpectedEOF)
	if h := healthOf(t, m, "w1"); h != HealthUnhealthy {
		t.Fatalf("Nth failure: w1 health = %s, want unhealthy", h)
	}
	if got := ejectionWindow(m, "w1"); got != defaultEjectBase {
		t.Fatalf("first ejection window = %s, want %s", got, defaultEjectBase)
	}

	// The 5s probe tick answers OK but cannot cut the window short.
	for elapsed := fleetHealthInterval; elapsed < defaultEjectBase; elapsed += fleetHealthInterval {
		clk.advance(fleetHealthInterval)
		m.ProbeOnce(ctx)
		if h := healthOf(t, m, "w1"); h != HealthUnhealthy {
			t.Fatalf("re-admitted %s into a %s ejection", elapsed, defaultEjectBase)
		}
	}
	clk.advance(fleetHealthInterval)
	m.ProbeOnce(ctx)
	if h := healthOf(t, m, "w1"); h != HealthHealthy {
		t.Fatalf("after the window: w1 health = %s, want healthy", h)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestReplicaEjectionRepeatEscalatesAndCaps(t *testing.T) {
	m, _, clk := ejectionFleet(t)
	ctx := context.Background()

	for n := 1; n <= 12; n++ {
		ejectByDispatch(t, m, "w1")
		want := time.Duration(n) * defaultEjectBase
		if want > defaultEjectMax {
			want = defaultEjectMax
		}
		got := ejectionWindow(m, "w1")
		if got != want {
			t.Fatalf("ejection %d window = %s, want %s", n, got, want)
		}
		clk.advance(got)
		m.ProbeOnce(ctx) // re-admit right at the window edge
		if h := healthOf(t, m, "w1"); h != HealthHealthy {
			t.Fatalf("ejection %d: not re-admitted at window end (health %s)", n, h)
		}
	}

	// A quiet period of at least the cap resets the escalation to the base window.
	clk.advance(defaultEjectMax)
	ejectByDispatch(t, m, "w1")
	if got := ejectionWindow(m, "w1"); got != defaultEjectBase {
		t.Fatalf("after a %s quiet period window = %s, want reset to %s", defaultEjectMax, got, defaultEjectBase)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestReplicaEjectionBusyAndProbeFailuresDoNotEject(t *testing.T) {
	m, sp, clk := ejectionFleet(t)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		m.markDispatchFailure("w1", &agent.UpstreamStatusError{Status: 503})
	}
	if h := healthOf(t, m, "w1"); h != HealthHealthy {
		t.Fatalf("busy 503s ejected w1 (health %s)", h)
	}

	// A probe-driven outage keeps the pre-existing contract: the next OK probe re-admits.
	sp.set("w2", false)
	m.ProbeOnce(ctx)
	m.ProbeOnce(ctx)
	if h := healthOf(t, m, "w2"); h != HealthUnhealthy {
		t.Fatalf("w2 health after failed probes = %s, want unhealthy", h)
	}
	if got := ejectionWindow(m, "w2"); got > 0 {
		t.Fatalf("probe-only outage opened a %s ejection window", got)
	}
	sp.set("w2", true)
	clk.advance(fleetHealthInterval)
	m.ProbeOnce(ctx)
	if h := healthOf(t, m, "w2"); h != HealthHealthy {
		t.Fatalf("w2 not re-admitted by the next OK probe (health %s)", h)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestFleetMembershipFailOpenWhenEveryHolderUnhealthy(t *testing.T) {
	m, _, _ := ejectionFleet(t)

	// One healthy peer left: no fail-open, placement stays on the healthy one.
	ejectByDispatch(t, m, "w1")
	cands, err := m.CandidatesForModel("")
	if err != nil || len(cands) != 1 || cands[0].ID != "w2" {
		t.Fatalf("one ejected: candidates = %v, %v; want [w2]", cands, err)
	}

	// Every holder unhealthy (both ejected): admit them all instead of an outage.
	ejectByDispatch(t, m, "w2")
	cands, err = m.CandidatesForModel("")
	if err != nil || len(cands) != 2 {
		t.Fatalf("all ejected: candidates = %v, %v; want both (fail-open)", cands, err)
	}
	if _, err := m.PickForModel(""); err != nil {
		t.Fatalf("all ejected: PickForModel err = %v, want fail-open pick", err)
	}
	r, err := m.reserveForModel("", "", nil, nil, func(s []WorkerStatus) (fleetReservationPickResult, bool) {
		return fleetReservationPickResult{workerID: s[0].Spec.ID}, len(s) == 2
	})
	if err != nil {
		t.Fatalf("all ejected: reserveForModel err = %v, want a fail-open booking", err)
	}
	r.Release()

	// Fail-open still tries each worker once and then returns the typed verdict.
	var tried []string
	_, err = m.Dispatch(context.Background(), func(_ context.Context, spec WorkerSpec) error {
		tried = append(tried, spec.ID)
		return io.ErrUnexpectedEOF
	})
	if len(tried) != 2 || !errors.Is(err, ErrNoHealthyWorker) {
		t.Fatalf("all failing: tried %v err %v; want both tried then ErrNoHealthyWorker", tried, err)
	}
	if spec, err := m.Dispatch(context.Background(), func(context.Context, WorkerSpec) error { return nil }); err != nil {
		t.Fatalf("fail-open dispatch err = %v (spec %q), want served", err, spec.ID)
	}
}

// Leaf 1 witness (oss-port-llmd-replica-fail-open-ejection-20261105): the
// escalating ejection state machine — N-1 consecutive failures do not eject, the
// Nth ejects for the base window, repeats escalate one step up to the cap,
// 500/504 responses never increment the failure streak, and an all-ejected pool
// still returns a replica (fail-open).
//
// fak-test:runtime fast est=10ms lane=default
func TestReplicaEjectionEscalation(t *testing.T) {
	m, _, clk := ejectionFleet(t)
	ctx := context.Background()
	n := m.unhealthyAfter // N: consecutive counted failures that cross to unhealthy

	// N-1 consecutive failures do not eject; the Nth ejects for the base window.
	for i := 0; i < n-1; i++ {
		m.markDispatchFailure("w1", io.ErrUnexpectedEOF)
	}
	if h := healthOf(t, m, "w1"); h != HealthHealthy {
		t.Fatalf("N-1 failures: w1 health = %s, want healthy", h)
	}
	if got := ejectionWindow(m, "w1"); got > 0 {
		t.Fatalf("N-1 failures opened a %s ejection window, want none", got)
	}
	m.markDispatchFailure("w1", io.ErrUnexpectedEOF)
	if h := healthOf(t, m, "w1"); h != HealthUnhealthy {
		t.Fatalf("Nth failure: w1 health = %s, want unhealthy", h)
	}
	if got := ejectionWindow(m, "w1"); got != defaultEjectBase {
		t.Fatalf("first ejection window = %s, want %s", got, defaultEjectBase)
	}

	// A repeat ejection escalates one step and caps at the maximum window.
	for step := 2; ; step++ {
		clk.advance(ejectionWindow(m, "w1")) // expire the current window exactly
		m.ProbeOnce(ctx)                     // re-admit right at the window edge
		if h := healthOf(t, m, "w1"); h != HealthHealthy {
			t.Fatalf("ejection %d: not re-admitted at window end (health %s)", step, h)
		}
		for i := 0; i < n; i++ {
			m.markDispatchFailure("w1", io.ErrUnexpectedEOF)
		}
		want := time.Duration(step) * defaultEjectBase
		if want > defaultEjectMax {
			want = defaultEjectMax
		}
		if got := ejectionWindow(m, "w1"); got != want {
			t.Fatalf("ejection %d window = %s, want %s", step, got, want)
		}
		if want == defaultEjectMax {
			break
		}
	}

	// A 500 or 504 response never increments the failure streak: request-specific
	// errors and slow-but-working timeouts are not evidence the replica is broken.
	for i := 0; i < 3*n; i++ {
		m.markDispatchFailure("w2", &agent.UpstreamStatusError{Status: 500})
		m.markDispatchFailure("w2", &agent.UpstreamStatusError{Status: 504})
	}
	if h := healthOf(t, m, "w2"); h != HealthHealthy {
		t.Fatalf("500/504 responses drove w2 to %s, want healthy", h)
	}
	m.mu.Lock()
	fs := m.workers["w2"].failStreak
	m.mu.Unlock()
	if fs != 0 {
		t.Fatalf("500/504 responses left failStreak = %d, want 0", fs)
	}
	if got := ejectionWindow(m, "w2"); got > 0 {
		t.Fatalf("500/504 responses opened a %s ejection window, want none", got)
	}

	// With every replica ejected, selection still returns a replica (fail-open).
	ejectByDispatch(t, m, "w2") // w1 is inside its capped window from above
	cands, err := m.CandidatesForModel("")
	if err != nil || len(cands) != 2 {
		t.Fatalf("all ejected: candidates = %v, %v; want both (fail-open)", cands, err)
	}
	if _, err := m.PickForModel(""); err != nil {
		t.Fatalf("all ejected: PickForModel err = %v, want fail-open pick", err)
	}
}

// Leaf 2 witness (oss-port-llmd-replica-fail-open-ejection-20261105): the state
// machine wired into the live fleet health loop — an ejected replica leaves
// rotation for its window and returns when it expires, an all-ejected fleet keeps
// returning replicas, and the flat unhealthyAfter hysteresis remains the fallback
// when no dispatch ejection engages.
//
// fak-test:runtime fast est=10ms lane=default
func TestReplicaEjectionMembershipWiring(t *testing.T) {
	m, sp, clk := ejectionFleet(t)
	ctx := context.Background()

	// An ejected replica leaves rotation for its window and returns when it
	// expires: live probe ticks keep answering OK but cannot pull it back in.
	ejectByDispatch(t, m, "w1")
	ticks := 3
	for i := 0; i < ticks; i++ {
		clk.advance(fleetHealthInterval)
		m.ProbeOnce(ctx)
		adm := m.Admissible()
		if len(adm) != 1 || adm[0].ID != "w2" {
			t.Fatalf("tick %d inside the window: admissible = %v, want [w2]", i+1, admIDs(adm))
		}
		spec, ok := m.Pick()
		if !ok || spec.ID != "w2" {
			t.Fatalf("tick %d inside the window: Pick = %q, ok=%v; want w2", i+1, spec.ID, ok)
		}
	}
	clk.advance(defaultEjectBase - time.Duration(ticks)*fleetHealthInterval) // land exactly on expiry
	m.ProbeOnce(ctx)
	if h := healthOf(t, m, "w1"); h != HealthHealthy {
		t.Fatalf("after window expiry: w1 health = %s, want healthy", h)
	}
	adm := m.Admissible()
	if len(adm) != 2 {
		t.Fatalf("after window expiry: admissible = %v, want both back in rotation", admIDs(adm))
	}

	// An all-ejected fleet keeps returning replicas (fail-open); rotation never
	// empties even though no replica is admissible.
	ejectByDispatch(t, m, "w1")
	ejectByDispatch(t, m, "w2")
	spec, ok := m.Pick()
	if !ok || spec.ID == "" {
		t.Fatalf("all ejected: Pick = %q, ok=%v; want a fail-open replica", spec.ID, ok)
	}
	r, err := m.reserveForModel("", "", nil, nil, func(s []WorkerStatus) (fleetReservationPickResult, bool) {
		return fleetReservationPickResult{workerID: s[0].Spec.ID}, len(s) == 2
	})
	if err != nil {
		t.Fatalf("all ejected: reserveForModel err = %v, want a fail-open booking", err)
	}
	r.Release()
	if served, err := m.Dispatch(ctx, func(context.Context, WorkerSpec) error { return nil }); err != nil || served.ID == "" {
		t.Fatalf("all ejected: Dispatch = %q, %v; want served fail-open", served.ID, err)
	}

	// The flat unhealthyAfter hysteresis remains the fallback when no dispatch
	// ejection engages: a probe-driven outage crosses on N failed probes, opens no
	// ejection window, and the next OK probe re-admits under healthyAfter.
	fresh := NewFleetMembership(MembershipConfig{Probe: sp.probe, Now: clk.now})
	mustAdd(t, fresh, WorkerSpec{ID: "p1"})
	sp.set("p1", true)
	fresh.ProbeOnce(ctx)
	sp.set("p1", false)
	for i := 0; i < fresh.unhealthyAfter-1; i++ {
		fresh.ProbeOnce(ctx)
	}
	if h := healthOf(t, fresh, "p1"); h != HealthHealthy {
		t.Fatalf("N-1 failed probes: p1 health = %s, want healthy", h)
	}
	fresh.ProbeOnce(ctx)
	if h := healthOf(t, fresh, "p1"); h != HealthUnhealthy {
		t.Fatalf("Nth failed probe: p1 health = %s, want unhealthy", h)
	}
	if got := ejectionWindow(fresh, "p1"); got > 0 {
		t.Fatalf("probe-only outage opened a %s ejection window, want none", got)
	}
	sp.set("p1", true)
	fresh.ProbeOnce(ctx)
	if h := healthOf(t, fresh, "p1"); h != HealthHealthy {
		t.Fatalf("next OK probe: p1 health = %s, want healthy (flat hysteresis fallback)", h)
	}
}

func admIDs(specs []WorkerSpec) []string {
	out := make([]string, 0, len(specs))
	for _, s := range specs {
		out = append(out, s.ID)
	}
	return out
}

// fak-test:runtime fast est=10ms lane=default
func TestFleetMembershipFailOpenSkipsDrainingAndUnprobed(t *testing.T) {
	m, sp, _ := ejectionFleet(t)
	if err := m.Acquire("w2"); err != nil { // keeps the drain from removing w2
		t.Fatalf("Acquire(w2): %v", err)
	}
	if err := m.Drain("w2"); err != nil {
		t.Fatalf("Drain(w2): %v", err)
	}
	sp.set("w1", false)
	sp.set("w2", false)
	m.ProbeOnce(context.Background())
	m.ProbeOnce(context.Background())

	cands, err := m.CandidatesForModel("")
	if err != nil || len(cands) != 1 || cands[0].ID != "w1" {
		t.Fatalf("candidates = %v, %v; want [w1] (draining w2 never resurrected)", cands, err)
	}

	fresh := NewFleetMembership(MembershipConfig{Probe: sp.probe})
	mustAdd(t, fresh, WorkerSpec{ID: "w3"}) // registered, never probed
	if _, err := fresh.PickForModel(""); !errors.Is(err, ErrNoHealthyWorker) {
		t.Fatalf("unprobed fleet: PickForModel err = %v, want ErrNoHealthyWorker", err)
	}
}
