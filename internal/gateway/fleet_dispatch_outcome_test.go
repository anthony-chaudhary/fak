package gateway

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func dispatchOutcomeFleet(t *testing.T) *FleetMembership {
	t.Helper()
	m := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 2,
		Probe:          func(context.Context, WorkerSpec) bool { return true },
	})
	mustAdd(t, m, WorkerSpec{ID: "w1"})
	mustAdd(t, m, WorkerSpec{ID: "w2"})
	m.ProbeOnce(context.Background())
	return m
}

// failFirstPick dispatches twice, failing whichever worker keyed placement tries
// first with cause; it returns that worker's id.
func failFirstPick(t *testing.T, m *FleetMembership, cause error) string {
	t.Helper()
	first, err := m.pickKeyedForModel("", nil, "")
	if err != nil {
		t.Fatalf("pickKeyedForModel: %v", err)
	}
	for i := 0; i < 2; i++ {
		if _, err := m.Dispatch(context.Background(), func(_ context.Context, spec WorkerSpec) error {
			if spec.ID == first.ID {
				return cause
			}
			return nil
		}); err != nil {
			t.Fatalf("Dispatch %d: %v", i, err)
		}
	}
	return first.ID
}

// fak-test:runtime fast est=10ms lane=default
func TestFleetMembershipBusyDispatchKeepsWorkerAdmissible(t *testing.T) {
	m := dispatchOutcomeFleet(t)
	busy := failFirstPick(t, m, &agent.UpstreamStatusError{Status: 503})
	if !admissibleIDs(m)[busy] {
		t.Fatalf("%s ejected after two neutral 503 dispatch failures", busy)
	}
	if status, ok := fleetWorkerStatus(m, busy); !ok || status.Health != HealthHealthy {
		t.Fatalf("%s health = %+v, want healthy", busy, status)
	}
	if !hasEvent(m.DrainEvents(), EventFailover, busy) {
		t.Fatalf("no failover event recorded for neutral failure on %s", busy)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestFleetMembershipRealDispatchFailuresStillEject(t *testing.T) {
	for _, tc := range []struct {
		name  string
		cause error
	}{
		{"bad_gateway", &agent.UpstreamStatusError{Status: 502}},
		{"net_error", &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("refused")}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := dispatchOutcomeFleet(t)
			failed := failFirstPick(t, m, tc.cause)
			if admissibleIDs(m)[failed] {
				t.Fatalf("%s still admissible after two failed dispatches", failed)
			}
			if status, ok := fleetWorkerStatus(m, failed); !ok || status.Health != HealthUnhealthy {
				t.Fatalf("%s health = %+v, want unhealthy", failed, status)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDispatchOutcomeNeutralClassifier(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		{"429", &agent.UpstreamStatusError{Status: 429}, true},
		{"500", &agent.UpstreamStatusError{Status: 500}, true},
		{"503", &agent.UpstreamStatusError{Status: 503}, true},
		{"529", &agent.UpstreamStatusError{Status: 529}, true},
		{"wrapped_503", fmt.Errorf("upstream: %w", &agent.UpstreamStatusError{Status: 503}), true},
		{"502", &agent.UpstreamStatusError{Status: 502}, false},
		// 504 is neutral per the llm-d pin (filter.go:46-53@329a52e): the proxy timed
		// out waiting on a slow-but-working replica, so it is excluded from the
		// failure count like 500.
		{"504", &agent.UpstreamStatusError{Status: 504}, true},
		{"net_error", &net.OpError{Op: "read", Net: "tcp", Err: errors.New("reset")}, false},
		{"eof", io.EOF, false},
		{"plain", errors.New("down"), false},
		{"nil", nil, false},
	} {
		if got := dispatchOutcomeNeutral(tc.err); got != tc.want {
			t.Errorf("dispatchOutcomeNeutral(%s) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestReplicaDispatchBusyReplicaFailsOverAndStaysAdmissible(t *testing.T) {
	m := nativeReservationFleet(t, "w1", "w2")
	busy := &reservationTestPlanner{worker: "w1", membership: m, err: &agent.UpstreamStatusError{Status: 503}}
	spare := &reservationTestPlanner{worker: "w2", membership: m}
	r := nativeReservationRouter(t, m, NewCacheAwarePolicy(nil, DefaultSkewThreshold()), busy, spare)

	for i := 0; i < 2; i++ {
		got, err := r.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "busy"}}, nil)
		if err != nil {
			t.Fatalf("Complete %d: %v", i, err)
		}
		if got.Message.Content != "w2" {
			t.Fatalf("Complete %d served by %q, want failover to w2", i, got.Message.Content)
		}
	}
	if busy.calls.Load() != 2 {
		t.Fatalf("busy replica calls = %d, want 2 (first pick both times)", busy.calls.Load())
	}
	if !admissibleIDs(m)["w1"] {
		t.Fatal("busy replica w1 ejected after two 503 failovers")
	}
	assertReservationFleetClean(t, m)
}
