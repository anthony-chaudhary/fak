package researcharm

// e2e_arm429_demo_test.go — the end-to-end demonstration of the reported bug and its fix,
// driven through the REAL coordinator. Before the fix, arm "gw-bench" at its ceiling
// refused every further request instantly with the prose 429. This drives the exact
// reported shape and asserts the fixed behavior: the surplus WAITS and is SERVED.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"testing"
	"time"
)

// TestE2EArm429ReportedScenarioQueuesAndServes replays the issue: "arm gw-bench at
// 16/16 active requests -> 429". With the queue armed, 40 concurrent gw-bench requests
// against a ceiling of 4 all RECEIVE A SLOT (serializing through the ceiling), and ZERO
// are refused — the 429 the issue reported is gone.
func TestE2EArm429ReportedScenarioQueuesAndServes(t *testing.T) {
	const (
		ceiling = 4
		burst   = 40
	)
	c := NewCoordinator(ceiling)
	c.SetQueueDepth(128)
	c.SetAdmitWait(10 * time.Second)

	type outcome struct {
		err  error
		wait time.Duration
	}
	results := make(chan outcome, burst)
	for i := 0; i < burst; i++ {
		go func() {
			start := time.Now()
			req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
			req.Header.Set("X-Fak-Project-Arm", "gw-bench")
			l, err := c.AdmitWithWait(context.Background(), req, "/v1/chat/completions", "gw-1", 10*time.Second)
			if l != nil {
				time.Sleep(time.Millisecond) // hold briefly so the queue actually forms
				l.Done(1, nil)
			}
			results <- outcome{err: err, wait: time.Since(start)}
		}()
	}

	served, refused := 0, 0
	var waits []time.Duration
	for i := 0; i < burst; i++ {
		r := <-results
		if r.err != nil {
			refused++
			var sat *SaturationError
			if !errors.As(r.err, &sat) {
				t.Errorf("refusal %d was not typed: %T %v", i, r.err, r.err)
			}
			continue
		}
		served++
		waits = append(waits, r.wait)
	}

	if refused != 0 {
		t.Fatalf("refused = %d of %d, want 0 — the reported instant-429 is back", refused, burst)
	}
	if served != burst {
		t.Fatalf("served = %d, want all %d", served, burst)
	}

	// The queue drained completely: no slot leaked.
	if q := c.QueuedRequests(); q != 0 {
		t.Fatalf("queue not drained: %d waiting", q)
	}
	for _, a := range c.Snapshot().Arms {
		if a.ActiveRequests != 0 {
			t.Fatalf("arm %s leaked %d slot(s)", a.ID, a.ActiveRequests)
		}
		if a.ShedsTotal != 0 {
			t.Fatalf("arm %s shed %d request(s) despite a bounded queue absorbing the burst", a.ID, a.ShedsTotal)
		}
		if a.AdmittedAfterWait == 0 {
			t.Fatalf("arm %s reports no admitted_after_wait — the burst never exercised the queue", a.ID)
		}
	}

	// Some requests genuinely queued (proving the burst exceeded the ceiling).
	queuedWaits := 0
	for _, w := range waits {
		if w > 5*time.Millisecond {
			queuedWaits++
		}
	}
	if queuedWaits == 0 {
		t.Skip("host too fast to observe queuing on this run; the zero-refusal assertion above is the contract")
	}
	t.Logf("served %d/%d through a ceiling of %d with %d requests measurably queued and ZERO refusals",
		served, burst, ceiling, queuedWaits)
}

// TestE2EArmShedWireShapeIsActionable demonstrates what a client SEES when a shed does
// happen: a closed reason token and the arm's counters, so it can branch and back off
// instead of parsing prose.
func TestE2EArmShedWireShapeIsActionable(t *testing.T) {
	c := NewCoordinator(1)
	c.SetAdmitWait(50 * time.Millisecond)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Project-Arm", "gw-bench")
	held, err := c.Admit(context.Background(), req, "/v1/chat/completions", "gw-held")
	if err != nil {
		t.Fatalf("hold: %v", err)
	}
	defer held.Done(0, nil)

	_, shed := c.AdmitWithWait(context.Background(), req, "/v1/chat/completions", "gw-2", 50*time.Millisecond)

	// Render exactly what the gateway would put on the wire (the same fields).
	var sat *SaturationError
	if !errors.As(shed, &sat) {
		t.Fatalf("want *SaturationError, got %T", shed)
	}
	payload := map[string]any{
		"error":  map[string]any{"code": "arm_concurrency_saturated", "message": shed.Error()},
		"reason": sat.Reason(),
		"arm_id": sat.ArmID,
		"active": sat.Active,
		"limit":  sat.Limit,
		"waited": sat.Waited.Round(time.Millisecond).String(),
	}
	blob, _ := json.Marshal(payload)
	t.Logf("shed wire payload: %s", blob)

	if sat.Reason() != "ARM_CONCURRENCY_SATURATED" {
		t.Fatalf("the shed must carry the closed reason, got %q", sat.Reason())
	}
	if sat.ArmID != "gw-bench" || sat.Active != 1 || sat.Limit != 1 {
		t.Fatalf("counters must identify the arm and its live load, got %+v", sat)
	}
	if sat.Waited <= 0 {
		t.Fatalf("the shed must report how long it waited, got %s", sat.Waited)
	}
}
