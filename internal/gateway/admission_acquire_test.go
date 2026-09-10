package gateway

import (
	"context"
	"errors"
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

func TestAcquireCancellationReschedulesEligibleFollower(t *testing.T) {
	ctl := NewAdmissionController(AdmissionPolicy{
		MaxNumSeqs:  3,
		TokenBudget: 10,
		MaxWaiting:  2,
		AgingRounds: 1,
	})

	running, err := ctl.Acquire(context.Background(), SeqRequest{TraceID: "running", Tokens: 6})
	if err != nil || running == nil {
		t.Fatalf("running Acquire = (%v, %v), want lease", running, err)
	}
	defer running.Release()

	type acquireResult struct {
		lease *AdmissionLease
		err   error
	}
	headCtx, cancelHead := context.WithCancel(context.Background())
	defer cancelHead()
	headResult := make(chan acquireResult, 1)
	go func() {
		lease, acquireErr := ctl.Acquire(headCtx, SeqRequest{TraceID: "head", Tokens: 5})
		headResult <- acquireResult{lease: lease, err: acquireErr}
	}()
	if !awaitAdmissionWaiting(ctl, 1, 2*time.Second) {
		t.Fatalf("blocked head never queued; stats=%+v", ctl.Stats())
	}

	followerCtx, cancelFollower := context.WithCancel(context.Background())
	defer cancelFollower()
	followerResult := make(chan acquireResult, 1)
	go func() {
		lease, acquireErr := ctl.Acquire(followerCtx, SeqRequest{TraceID: "follower", Tokens: 4})
		followerResult <- acquireResult{lease: lease, err: acquireErr}
	}()
	if !awaitAdmissionWaiting(ctl, 2, 2*time.Second) {
		t.Fatalf("fitting follower never queued behind head; stats=%+v", ctl.Stats())
	}
	if stats := ctl.Stats(); stats.Running != 1 || stats.Waiting != 2 || stats.TokensInUse != 6 || stats.QueuedTokens != 9 {
		t.Fatalf("live-head HOL stats = %+v, want running=1 waiting=2 tokens=6 queued_tokens=9", stats)
	}

	// The older five-token request cannot fit in the four remaining tokens, so the
	// younger four-token request must not jump it while the head remains live. Once
	// the head disconnects, the fitting follower must be promoted by that cancellation;
	// this test intentionally does not call Schedule or release the running lease here.
	cancelHead()
	select {
	case got := <-headResult:
		if got.lease != nil || !errors.Is(got.err, context.Canceled) {
			t.Fatalf("canceled head Acquire = (%v, %v), want (nil, context.Canceled)", got.lease, got.err)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("blocked head did not observe cancellation; stats=%+v", ctl.Stats())
	}

	var follower *AdmissionLease
	select {
	case got := <-followerResult:
		if got.err != nil || got.lease == nil {
			t.Fatalf("follower Acquire = (%v, %v), want promoted lease", got.lease, got.err)
		}
		follower = got.lease
	case <-time.After(2 * time.Second):
		t.Fatalf("fitting follower remained blocked after head cancellation; stats=%+v", ctl.Stats())
	}

	if stats := ctl.Stats(); stats.Running != 2 || stats.Waiting != 0 || stats.TokensInUse != 10 || stats.QueuedTokens != 0 {
		t.Fatalf("post-cancel stats = %+v, want running=2 waiting=0 tokens=10 queued_tokens=0", stats)
	}
	follower.Release()
	follower.Release() // cleanup is idempotent and must not underflow the budget.
	running.Release()
	running.Release()
	if stats := ctl.Stats(); stats.Running != 0 || stats.Waiting != 0 || stats.TokensInUse != 0 || stats.QueuedTokens != 0 {
		t.Fatalf("final stats = %+v, want all live budgets released", stats)
	}
}

// TestBeginServedAdmissionSeam witnesses the Server-level admission seam: with a controller
// wired it admits an underloaded request and returns a releasable lease; with no controller
// attached it is inert (nil lease, nil error) and the historical request path is byte-for-byte
// unchanged. This is the seam the gateway request path acquires through before the planner runs.
func TestBeginServedAdmissionRejectsRequestLargerThanTokenBudget(t *testing.T) {
	policy := DefaultAdmissionPolicy()
	policy.TokenBudget = 32
	server := newTestServer(t)
	server.SetAdmissionController(NewAdmissionController(policy))

	// Claude CLI commonly asks for a much larger output envelope than a tiny direct
	// probe. A request that can never fit must be refused instead of queued forever.
	lease, err := server.beginServedAdmission(context.Background(), servedSessionTurn{traceID: "claude-envelope"}, nil, nil, 64)
	if lease != nil {
		t.Fatal("beginServedAdmission lease != nil, want refusal for an impossible request")
	}
	var admissionErr *AdmissionError
	if !errors.As(err, &admissionErr) {
		t.Fatalf("beginServedAdmission err = %v, want typed AdmissionError", err)
	}
	if admissionErr.Verdict != VerdictRefused || admissionErr.Reason != "request tokens 64 exceed scheduler token budget 32" {
		t.Fatalf("admission error = %+v, want exact impossible-envelope reason", admissionErr)
	}
	if stats := server.admissionCtl.Stats(); stats.Waiting != 0 || stats.Refused != 1 {
		t.Fatalf("admission stats = %+v, want waiting=0 refused=1", stats)
	}
}
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
