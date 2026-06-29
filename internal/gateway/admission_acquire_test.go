package gateway

import (
	"context"
	"net/http"
	"testing"
	"time"
)

// TestAcquireLeaseBoundaryShedsAndDenies is the issue-#35 live-boundary witness for the
// synchronous gateway admission seam — Acquire plus the AdmissionError -> HTTP status
// mapping (admissionErrorStatus) that sits over the policy in admission.go. It proves,
// without the full HTTP handler, that a SATURATED gate sheds the next request as a typed
// 429-mapped error BEFORE it can reach the planner (the backpressure surface that replaces
// unbounded queueing), that a denying per-tenant trust verdict is a 403, and that releasing
// a running slot promotes the queued waiter — the no-starvation edge on the live path.
func TestAcquireLeaseBoundaryShedsAndDenies(t *testing.T) {
	ctl := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, MaxWaiting: 1, AgingRounds: 1})
	ctx := context.Background()

	// First request takes the only running slot (fast-path admit -> a live lease).
	lease, err := ctl.Acquire(ctx, SeqRequest{TraceID: "first", Tokens: 1})
	if err != nil || lease == nil {
		t.Fatalf("first Acquire = (%v, %v), want an admitted lease", lease, err)
	}

	// Second request has no headroom; it joins the waiting queue and blocks until promoted.
	type acq struct {
		lease *AdmissionLease
		err   error
	}
	promoted := make(chan acq, 1)
	go func() {
		l, e := ctl.Acquire(ctx, SeqRequest{TraceID: "second", Tokens: 1})
		promoted <- acq{l, e}
	}()
	if !awaitAdmissionWaiting(ctl, 1, 2*time.Second) {
		t.Fatalf("second Acquire never queued; stats=%+v", ctl.Stats())
	}

	// Third request: no headroom AND the waiting queue is at its bound -> a synchronous shed,
	// surfaced as the typed served-path 429 (the backpressure signal), never an unbounded wait.
	_, err = ctl.Acquire(ctx, SeqRequest{TraceID: "third", Tokens: 1})
	if status, code, _, ok := admissionErrorStatus(err); !ok || status != http.StatusTooManyRequests || code != "scheduler_overloaded" {
		t.Fatalf("shed mapping = (%d, %q, %v), want (429, scheduler_overloaded, true); err=%v", status, code, ok, err)
	}

	// A denying trust verdict rejects admission outright as a 403, independent of headroom.
	_, err = ctl.Acquire(ctx, SeqRequest{TraceID: "denied", Tokens: 1, Trust: AdmissionTrust{Deny: true, Reason: "tenant-suspended"}})
	if status, code, _, ok := admissionErrorStatus(err); !ok || status != http.StatusForbidden || code != "scheduler_admission_denied" {
		t.Fatalf("deny mapping = (%d, %q, %v), want (403, scheduler_admission_denied, true); err=%v", status, code, ok, err)
	}

	// Releasing the running slot frees budget and promotes the waiting request — the live
	// boundary's no-starvation edge: the blocked Acquire returns with its own lease.
	lease.Release()
	select {
	case got := <-promoted:
		if got.err != nil || got.lease == nil {
			t.Fatalf("promoted Acquire = (%v, %v), want a lease after release", got.lease, got.err)
		}
		got.lease.Release()
	case <-time.After(2 * time.Second):
		t.Fatalf("queued request never promoted after release; stats=%+v", ctl.Stats())
	}

	if st := ctl.Stats(); st.Running != 0 || st.Waiting != 0 || st.Shed != 1 || st.Denied != 1 || st.Admitted != 2 {
		t.Fatalf("final stats = %+v, want running=0 waiting=0 shed=1 denied=1 admitted=2", st)
	}
}

// TestBeginServedAdmissionSeam witnesses the Server-level admission seam: with a controller
// wired it admits an underloaded request and returns a releasable lease; with no controller
// attached it is inert (nil lease, nil error) and the historical request path is byte-for-byte
// unchanged. This is the seam the gateway request path acquires through before the planner runs.
func TestBeginServedAdmissionSeam(t *testing.T) {
	ctx := context.Background()
	turn := servedSessionTurn{traceID: "seam", state: SessionState{Priority: 0}, maxTokens: 1}

	wired := newTestServer(t)
	wired.SetAdmissionController(NewAdmissionController(DefaultAdmissionPolicy()))
	lease, err := wired.beginServedAdmission(ctx, turn, nil, nil, 1)
	if err != nil {
		t.Fatalf("wired beginServedAdmission err = %v, want admit on an idle controller", err)
	}
	if lease == nil {
		t.Fatal("wired beginServedAdmission lease = nil, want a live lease on an idle controller")
	}
	lease.Release() // frees the slot; idempotent

	inert := newTestServer(t)
	lease, err = inert.beginServedAdmission(ctx, turn, nil, nil, 1)
	if err != nil || lease != nil {
		t.Fatalf("inert beginServedAdmission = (%v, %v), want (nil, nil) with no controller attached", lease, err)
	}
	lease.Release() // a nil lease Release is a no-op
}

// awaitAdmissionWaiting polls the gate until its waiting-queue depth reaches n or the
// deadline passes, so the shed witness is deterministic without reaching into internals.
func awaitAdmissionWaiting(c *AdmissionController, n int, d time.Duration) bool {
	deadline := time.Now().Add(d)
	for {
		if c.Stats().Waiting == n {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(time.Millisecond)
	}
}
