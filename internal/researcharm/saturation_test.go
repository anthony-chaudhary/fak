package researcharm

// saturation_test.go — the contract of the TYPED capacity refusal and the queue
// observability layered on AdmitWithWait (saturation.go): a shed must be a
// *SaturationError carrying the closed reason ARM_CONCURRENCY_SATURATED, the arm's
// counters, and a wait-derived Retry-After; the queue must be BOUNDED; and the counters
// must reconcile.

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func satReq(arm string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Project-Arm", arm)
	return req
}

func holdArm(t *testing.T, c *Coordinator, arm string) *RequestLease {
	t.Helper()
	l, err := c.Admit(context.Background(), satReq(arm), "/v1/chat/completions", "trace-"+arm)
	if err != nil {
		t.Fatalf("admit %s: %v", arm, err)
	}
	return l
}

// TestArmShedIsTypedWithClosedReason pins the headline contract: a wait that runs out
// returns a *SaturationError carrying the closed reason, the arm's counters, a measured
// wait, and a Retry-After — and errors.Is still reaches the historical sentinel.
func TestArmShedIsTypedWithClosedReason(t *testing.T) {
	c := NewCoordinator(1)
	held := holdArm(t, c, "gw-bench")
	defer held.Done(0, nil)

	_, err := c.AdmitWithWait(context.Background(), satReq("gw-bench"), "/v1/chat/completions", "t", 40*time.Millisecond)

	var sat *SaturationError
	if !errors.As(err, &sat) {
		t.Fatalf("want *SaturationError, got %T: %v", err, err)
	}
	if !errors.Is(err, ErrArmConcurrencyExceeded) {
		t.Fatalf("the historical sentinel must stay reachable: %v", err)
	}
	if sat.Reason() != ReasonArmConcurrencySaturated {
		t.Fatalf("reason = %q, want %q", sat.Reason(), ReasonArmConcurrencySaturated)
	}
	if sat.ArmID != "gw-bench" {
		t.Fatalf("ArmID = %q, want gw-bench", sat.ArmID)
	}
	if sat.Active != 1 || sat.Limit != 1 {
		t.Fatalf("counters = %d/%d, want 1/1", sat.Active, sat.Limit)
	}
	if sat.Waited <= 0 {
		t.Fatalf("Waited = %s, want the measured wait", sat.Waited)
	}
	if sat.RetryAfter < time.Second {
		t.Fatalf("RetryAfter = %s, want at least 1s (actionable)", sat.RetryAfter)
	}
	// The shed is booked against the arm's capacity counter.
	for _, a := range c.Snapshot().Arms {
		if a.ID == "gw-bench" {
			if a.ShedsTotal != 1 {
				t.Fatalf("sheds_total = %d, want 1", a.ShedsTotal)
			}
			if a.WaitedTotal != 1 {
				t.Fatalf("waited_total = %d, want 1 (the request DID queue)", a.WaitedTotal)
			}
			if a.AdmittedAfterWait != 0 {
				t.Fatalf("admitted_after_wait = %d, want 0", a.AdmittedAfterWait)
			}
			return
		}
	}
	t.Fatal("arm gw-bench missing from snapshot")
}

// TestArmQueueGrantsFreedSlotTyped pins the happy path end to end: a queued request is
// served the moment a slot frees, and the grant is booked as the measured wait.
func TestArmQueueGrantsFreedSlotTyped(t *testing.T) {
	c := NewCoordinator(1)
	c.SetAdmitWait(2 * time.Second)
	held := holdArm(t, c, "gw-bench")

	done := make(chan error, 1)
	go func() {
		l, err := c.AdmitWithWait(context.Background(), satReq("gw-bench"), "/v1/chat/completions", "t", 2*time.Second)
		if err == nil && l != nil {
			l.Done(0, nil)
		}
		done <- err
	}()
	waitForQueue(t, c, 1)
	held.Done(0, nil)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("queued request was refused instead of served: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("queued request was never served after a slot freed")
	}
	if q := c.QueuedRequests(); q != 0 {
		t.Fatalf("queue not drained, %d waiting", q)
	}
	for _, a := range c.Snapshot().Arms {
		if a.ID != "gw-bench" {
			continue
		}
		if a.AdmittedAfterWait != 1 {
			t.Fatalf("admitted_after_wait = %d, want 1", a.AdmittedAfterWait)
		}
		if a.ShedsTotal != 0 {
			t.Fatalf("sheds_total = %d, want 0 — nobody was refused", a.ShedsTotal)
		}
		if a.AvgQueueWait <= 0 {
			t.Fatalf("avg_queue_wait = %s beside %d admits — a real wait must be measured", a.AvgQueueWait, a.AdmittedAfterWait)
		}
		return
	}
	t.Fatal("arm gw-bench missing from snapshot")
}

// TestArmQueueBoundShedsImmediately pins the BOUND: past QueueDepth a new arrival is shed
// at once rather than joining an unbounded backlog, carrying the full-queue shape.
func TestArmQueueBoundShedsImmediately(t *testing.T) {
	c := NewCoordinator(1)
	c.SetQueueDepth(1)
	held := holdArm(t, c, "bounded")
	defer held.Done(0, nil)

	// First waiter occupies the single queue slot.
	go func() {
		_, _ = c.AdmitWithWait(context.Background(), satReq("bounded"), "/v1/chat/completions", "w1", 30*time.Second)
	}()
	waitForQueue(t, c, 1)

	// Second arrival: queue full → shed WITHOUT waiting the 30s.
	start := time.Now()
	_, err := c.AdmitWithWait(context.Background(), satReq("bounded"), "/v1/chat/completions", "w2", 30*time.Second)
	elapsed := time.Since(start)

	var sat *SaturationError
	if !errors.As(err, &sat) {
		t.Fatalf("want *SaturationError, got %v", err)
	}
	if !sat.Full {
		t.Fatalf("refusal should report the full queue, got %+v", sat)
	}
	if sat.QueueBound != 1 {
		t.Fatalf("QueueBound = %d, want 1", sat.QueueBound)
	}
	if elapsed > time.Second {
		t.Fatalf("queue-full shed took %s — it must not wait", elapsed)
	}
}

// TestArmQueueUnboundedByDefault pins the compatibility posture: with no bound set the
// queue admits waiters (0 = unbounded), so trunk's behavior is preserved.
func TestArmQueueUnboundedByDefault(t *testing.T) {
	c := NewCoordinator(1)
	if got := c.QueueDepth(); got != 0 {
		t.Fatalf("QueueDepth = %d, want 0 (unbounded) by default", got)
	}
	held := holdArm(t, c, "free")
	defer held.Done(0, nil)

	_ = held
	for i := 0; i < 3; i++ {
		go func() {
			_, _ = c.AdmitWithWait(context.Background(), satReq("free"), "/v1/chat/completions", "t", 300*time.Millisecond)
		}()
	}
	// All three must be allowed to WAIT (no bound), then shed by their own budget.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && c.QueuedRequests() != 3 {
		time.Sleep(time.Millisecond)
	}
	if q := c.QueuedRequests(); q != 3 {
		t.Fatalf("queued = %d, want 3 (an unbounded queue admits every waiter)", q)
	}
}

// TestArmShedReportsEffectiveLeaseLimit pins that a shed reports the EFFECTIVE
// (lease-overridable) ceiling, not the arm's own, so a refusal can never read
// self-contradictorily ("at 1/16 in flight" while refusing).
func TestArmShedReportsEffectiveLeaseLimit(t *testing.T) {
	c := NewCoordinator(16)
	if _, err := c.AcquireLease(LeaseRequest{ArmID: "pinned", Mode: LeaseModeShared, Concurrency: 1, TTL: time.Minute}); err != nil {
		t.Fatalf("acquire lease: %v", err)
	}
	held := holdArm(t, c, "pinned")
	defer held.Done(0, nil)

	_, err := c.AdmitWithWait(context.Background(), satReq("pinned"), "/v1/chat/completions", "t", 40*time.Millisecond)
	var sat *SaturationError
	if !errors.As(err, &sat) {
		t.Fatalf("want a capacity refusal, got %v", err)
	}
	if sat.Limit != 1 {
		t.Fatalf("Limit = %d, want the lease-effective 1 (not the arm's own 16)", sat.Limit)
	}
}

// TestArmWaitRecordedEvenSubMicrosecond pins the sentinel collision: a wait that actually
// happened must never read as "never measured" (0µs truncation would leave avg 0 beside a
// nonzero admitted_after_wait).
func TestArmWaitRecordedEvenSubMicrosecond(t *testing.T) {
	c := NewCoordinator(1)
	c.SetAdmitWait(2 * time.Second)
	for i := 0; i < 100; i++ {
		held := holdArm(t, c, "us")
		go func() {
			l, err := c.AdmitWithWait(context.Background(), satReq("us"), "/v1/chat/completions", "t", 2*time.Second)
			if err == nil && l != nil {
				l.Done(0, nil)
			}
		}()
		waitForQueue(t, c, 1)
		held.Done(0, nil)
		// Wait for the granted waiter to RELEASE its slot (not merely leave the queue):
		// it is granted while still counted active until its own Done lands.
		deadline := time.Now().Add(3 * time.Second)
		for time.Now().Before(deadline) {
			for _, a := range c.Snapshot().Arms {
				if a.ID == "us" && a.ActiveRequests == 0 && a.WaitingRequests == 0 {
					goto settled
				}
			}
			time.Sleep(time.Millisecond)
		}
		t.Fatalf("iteration %d: the granted waiter never released its slot", i)
	settled:
	}
	for _, a := range c.Snapshot().Arms {
		if a.ID != "us" {
			continue
		}
		if a.AdmittedAfterWait == 0 {
			t.Fatal("no admits were recorded; the probe proved nothing")
		}
		if a.AvgQueueWait <= 0 {
			t.Fatalf("avg_queue_wait = %s beside %d admits — a sub-µs wait was recorded as unmeasured", a.AvgQueueWait, a.AdmittedAfterWait)
		}
		return
	}
	t.Fatal("arm us missing")
}

// TestArmSaturationsReadout pins the metrics-facing readout: one row per arm with the
// queue counters, and an empty (nil) readout for a nil coordinator.
func TestArmSaturationsReadout(t *testing.T) {
	var nilC *Coordinator
	if got := nilC.Saturations(); got != nil {
		t.Fatalf("nil coordinator should report no saturations, got %v", got)
	}

	c := NewCoordinator(2)
	held := holdArm(t, c, "metrics-arm")
	defer held.Done(0, nil)

	sats := c.Saturations()
	if len(sats) != 1 {
		t.Fatalf("saturations = %d rows, want 1", len(sats))
	}
	s := sats[0]
	if s.ArmID != "metrics-arm" || s.Active != 1 || s.Limit != 2 || s.Queued != 0 {
		t.Fatalf("readout = %+v, want arm=metrics-arm active=1 limit=2 queued=0", s)
	}
}

// TestArmQueueWaitCountersReconcile pins the accounting identity the metrics document:
// waited = admitted_after_wait + sheds + (still queued), so an operator's ratios are
// meaningful rather than silently drifting.
func TestArmQueueWaitCountersReconcile(t *testing.T) {
	c := NewCoordinator(1)
	c.SetAdmitWait(60 * time.Millisecond)
	held := holdArm(t, c, "recon")

	// One waiter that WILL be served.
	served := make(chan error, 1)
	go func() {
		l, err := c.AdmitWithWait(context.Background(), satReq("recon"), "/v1/chat/completions", "t1", 2*time.Second)
		if err == nil && l != nil {
			l.Done(0, nil)
		}
		served <- err
	}()
	waitForQueue(t, c, 1)

	// One waiter that will TIME OUT (the holder stays held).
	if _, err := c.AdmitWithWait(context.Background(), satReq("recon"), "/v1/chat/completions", "t2", 60*time.Millisecond); err == nil {
		t.Fatal("second waiter should have been shed")
	}
	held.Done(0, nil)
	if err := <-served; err != nil {
		t.Fatalf("first waiter should have been served: %v", err)
	}

	for _, a := range c.Snapshot().Arms {
		if a.ID != "recon" {
			continue
		}
		if a.WaitedTotal != 2 {
			t.Fatalf("waited_total = %d, want 2", a.WaitedTotal)
		}
		if a.AdmittedAfterWait != 1 {
			t.Fatalf("admitted_after_wait = %d, want 1", a.AdmittedAfterWait)
		}
		if a.ShedsTotal != 1 {
			t.Fatalf("sheds_total = %d, want 1", a.ShedsTotal)
		}
		if a.WaitingRequests != 0 {
			t.Fatalf("waiting = %d, want 0", a.WaitingRequests)
		}
		if a.WaitedTotal != a.AdmittedAfterWait+a.ShedsTotal+int64(a.WaitingRequests) {
			t.Fatalf("counters do not reconcile: waited=%d admitted=%d sheds=%d waiting=%d",
				a.WaitedTotal, a.AdmittedAfterWait, a.ShedsTotal, a.WaitingRequests)
		}
		return
	}
	t.Fatal("arm recon missing")
}

// TestArmAdmitWaitClampAndQueueDepthClamp pins the envelopes.
func TestArmAdmitWaitClampAndQueueDepthClamp(t *testing.T) {
	c := NewCoordinator(1)
	c.SetAdmitWait(time.Hour)
	if got := c.AdmitWait(); got != maxArmAdmissionWait {
		t.Fatalf("AdmitWait = %s, want clamped to %s", got, maxArmAdmissionWait)
	}
	c.SetAdmitWait(-time.Second)
	if got := c.AdmitWait(); got != 0 {
		t.Fatalf("negative AdmitWait = %s, want 0", got)
	}
	c.SetQueueDepth(999999)
	if got := c.QueueDepth(); got != maxArmQueueDepth {
		t.Fatalf("QueueDepth = %d, want clamped to %d", got, maxArmQueueDepth)
	}
	c.SetQueueDepth(-5)
	if got := c.QueueDepth(); got != 0 {
		t.Fatalf("negative QueueDepth = %d, want 0", got)
	}
}

// TestParseArmQueueDepth pins the env parser: unparseable/negative leave it unbounded
// with a note; a valid value is taken; an over-large one is clamped.
func TestParseArmQueueDepth(t *testing.T) {
	env := func(kv map[string]string) func(string) string {
		return func(k string) string { return kv[k] }
	}
	if n, note := ParseArmQueueDepth(env(nil)); n != 0 || note != "" {
		t.Fatalf("unset = (%d,%q), want (0,\"\")", n, note)
	}
	if n, _ := ParseArmQueueDepth(env(map[string]string{envArmQueueDepth: "12"})); n != 12 {
		t.Fatalf("n = %d, want 12", n)
	}
	if n, note := ParseArmQueueDepth(env(map[string]string{envArmQueueDepth: "lots"})); n != 0 || note == "" {
		t.Fatalf("unparseable = (%d,%q), want (0, a note)", n, note)
	}
	if n, note := ParseArmQueueDepth(env(map[string]string{envArmQueueDepth: "-3"})); n != 0 || note == "" {
		t.Fatalf("negative = (%d,%q), want (0, a note)", n, note)
	}
	if n, _ := ParseArmQueueDepth(env(map[string]string{envArmQueueDepth: "999999"})); n != maxArmQueueDepth {
		t.Fatalf("over-large = %d, want clamped to %d", n, maxArmQueueDepth)
	}
}

// TestArmZeroWaitPreservesLegacyRefusal pins the byte-for-byte historical posture: with
// wait 0 the instant path reports the ORIGINAL sentence and no measured wait.
func TestArmZeroWaitPreservesLegacyRefusal(t *testing.T) {
	c := NewCoordinator(1)
	held := holdArm(t, c, "legacy")
	defer held.Done(0, nil)

	start := time.Now()
	_, err := c.AdmitWithWait(context.Background(), satReq("legacy"), "/v1/chat/completions", "t", 0)
	if time.Since(start) > 200*time.Millisecond {
		t.Fatal("the zero-wait path must not block")
	}
	var sat *SaturationError
	if !errors.As(err, &sat) {
		t.Fatalf("want *SaturationError, got %T", err)
	}
	if sat.Waited != 0 {
		t.Fatalf("Waited = %s, want 0 on the instant path", sat.Waited)
	}
	if sat.Reason() != ReasonArmConcurrencySaturated {
		t.Fatalf("reason = %q, want the closed token", sat.Reason())
	}
}

// waitForQueue polls the coordinator until n requests are waiting, so tests never sleep
// on a fixed delay.
func waitForQueue(t *testing.T, c *Coordinator, n int) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if c.QueuedRequests() >= n {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for %d queued request(s); have %d", n, c.QueuedRequests())
}
