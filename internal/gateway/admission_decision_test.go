package gateway

import (
	"context"
	"strconv"
	"testing"
	"time"
)

// admission_decision_test.go — the #13120 witness suite for the durable per-request
// AdmissionDecisionReceipt. Authored independently of the implementation: every assertion
// reads LastAdmissionDecision back through the public API and checks the RECEIPT against the
// spec, never against an internal field the test happened to poke.

// fixedClock returns a controller whose injectable clock always reports at, so At is exact.
func fixedClock(t *testing.T, at time.Time) *AdmissionController {
	t.Helper()
	c := NewAdmissionController(DefaultAdmissionPolicy())
	c.SetClock(func() time.Time { return at })
	return c
}

// TestAdmissionReceiptAdmittedRoundTrip drives the happy path through the LIVE Acquire
// boundary (the gateway's actual entry point) and checks every receipt field lands.
func TestAdmissionReceiptAdmittedRoundTrip(t *testing.T) {
	at := time.Date(2026, 9, 15, 9, 30, 0, 0, time.UTC)
	c := fixedClock(t, at)

	lease, err := c.Acquire(context.Background(), SeqRequest{
		TraceID:   "req-A",
		SessionID: "sess-X",
		Tokens:    128,
		Priority:  7,
	})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer lease.Release()

	rec, ok := c.LastAdmissionDecision("req-A#1")
	if !ok {
		t.Fatalf("no receipt under req-A#1; suffix binding missing")
	}
	if rec.TraceID != "req-A#1" {
		t.Errorf("TraceID = %q, want req-A#1", rec.TraceID)
	}
	if rec.SessionID != "sess-X" {
		t.Errorf("SessionID = %q, want sess-X", rec.SessionID)
	}
	if rec.Tokens != 128 || rec.Priority != 7 {
		t.Errorf("Tokens/Priority = %d/%d, want 128/7", rec.Tokens, rec.Priority)
	}
	if rec.Verdict != VerdictAdmitted.String() || !rec.Admitted {
		t.Errorf("Verdict/Admitted = %q/%v, want admitted/true", rec.Verdict, rec.Admitted)
	}
	if !rec.At.Equal(at) {
		t.Errorf("At = %v, want injected %v", rec.At, at)
	}
	if len(rec.Reservations) != 1 {
		t.Fatalf("Reservations = %+v, want exactly one token scope", rec.Reservations)
	}
	if r := rec.Reservations[0]; r.Scope != "tokens" || r.Tokens != 128 {
		t.Errorf("reservation = %+v, want {tokens 128}", r)
	}
}

// TestAdmissionReceiptOfferRecordsAdmitted also pins the pure Offer path: Offer records the
// same admitted receipt but keyed on the caller's RAW trace id (Offer assigns no suffix).
func TestAdmissionReceiptOfferRecordsAdmitted(t *testing.T) {
	c := NewAdmissionController(DefaultAdmissionPolicy())
	if v := c.Offer(SeqRequest{TraceID: "bare", Tokens: 5}); v != VerdictAdmitted {
		t.Fatalf("Offer = %s, want admitted", v)
	}
	rec, ok := c.LastAdmissionDecision("bare")
	if !ok {
		t.Fatal("no receipt for the admitted Offer keyed on the raw trace id")
	}
	if rec.Verdict != "admitted" || !rec.Admitted || rec.Tokens != 5 {
		t.Errorf("receipt = %+v, want admitted tokens=5", rec)
	}
}

// TestAdmissionReceiptRefusedImpossibleEnvelope pins the refused verdict: a request whose
// footprint exceeds the entire budget can never be admitted by waiting, so it is refused
// (not queued/shed) and names the deciding budget axis.
func TestAdmissionReceiptRefusedImpossibleEnvelope(t *testing.T) {
	c := NewAdmissionController(AdmissionPolicy{TokenBudget: 64, MaxWaiting: 32, AgingRounds: 1})

	v := c.Offer(SeqRequest{TraceID: "impossible", Tokens: 65})
	if v != VerdictRefused {
		t.Fatalf("verdict = %s, want refused", v)
	}
	rec, ok := c.LastAdmissionDecision("impossible")
	if !ok {
		t.Fatal("no receipt for the refused request")
	}
	if rec.Verdict != "refused" || rec.Admitted {
		t.Errorf("Verdict/Admitted = %q/%v, want refused/false", rec.Verdict, rec.Admitted)
	}
	if rec.Budget != "tokens" {
		t.Errorf("Budget = %q, want the deciding axis tokens", rec.Budget)
	}
	if rec.Reason == "" {
		t.Error("Reason empty on a refusal; operator cannot learn why")
	}
}

// TestAdmissionReceiptDeniedTrustVerdict pins the governance denial: a denying trust verdict
// rejects regardless of free headroom and carries the tenant reason verbatim.
func TestAdmissionReceiptDeniedTrustVerdict(t *testing.T) {
	c := NewAdmissionController(DefaultAdmissionPolicy())

	v := c.Offer(SeqRequest{
		TraceID: "tenant-B",
		Tokens:  1,
		Trust:   AdmissionTrust{Deny: true, Reason: "SLA_EXCEEDED"},
	})
	if v != VerdictDenied {
		t.Fatalf("verdict = %s, want denied", v)
	}
	rec, ok := c.LastAdmissionDecision("tenant-B")
	if !ok {
		t.Fatal("no receipt for the denied request")
	}
	if rec.Verdict != "denied" || rec.Admitted {
		t.Errorf("Verdict/Admitted = %q/%v, want denied/false", rec.Verdict, rec.Admitted)
	}
	if rec.Reason != "SLA_EXCEEDED" {
		t.Errorf("Reason = %q, want SLA_EXCEEDED", rec.Reason)
	}
}

// TestAdmissionReceiptShedWhenWaitingAtBound pins the backpressure shed: once the running set
// and the waiting queue are both saturated the next arrival is shed, not queued unboundedly.
func TestAdmissionReceiptShedWhenWaitingAtBound(t *testing.T) {
	c := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, MaxWaiting: 2, AgingRounds: 1})

	c.Offer(SeqRequest{TraceID: "r0"}) // consumes the single running slot
	c.Offer(SeqRequest{TraceID: "w0"}) // waits
	c.Offer(SeqRequest{TraceID: "w1"}) // waits (queue now full)

	if v := c.Offer(SeqRequest{TraceID: "shed"}); v != VerdictShed {
		t.Fatalf("verdict = %s, want shed", v)
	}
	rec, ok := c.LastAdmissionDecision("shed")
	if !ok {
		t.Fatal("no receipt for the shed request")
	}
	if rec.Verdict != "shed" || rec.Admitted {
		t.Errorf("Verdict/Admitted = %q/%v, want shed/false", rec.Verdict, rec.Admitted)
	}
	if rec.Budget != "max_waiting" {
		t.Errorf("Budget = %q, want max_waiting", rec.Budget)
	}
	if rec.Reason == "" {
		t.Error("Reason empty on a shed; the 429 has no explanation")
	}
}

// TestAdmissionReceiptAcquireSuffixBindingDistinguishesRequests is the identity witness: two
// HTTP requests for the SAME served session acquire DISTINCT suffixed traces, each with its
// own receipt, so the record is per-request rather than per-session.
func TestAdmissionReceiptAcquireSuffixBindingDistinguishesRequests(t *testing.T) {
	c := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 8, MaxWaiting: 8, AgingRounds: 1})

	l1, err := c.Acquire(context.Background(), SeqRequest{TraceID: "same", SessionID: "same", Tokens: 11})
	if err != nil {
		t.Fatalf("Acquire #1: %v", err)
	}
	defer l1.Release()
	l2, err := c.Acquire(context.Background(), SeqRequest{TraceID: "same", SessionID: "same", Tokens: 22})
	if err != nil {
		t.Fatalf("Acquire #2: %v", err)
	}
	defer l2.Release()

	r1, ok1 := c.LastAdmissionDecision("same#1")
	r2, ok2 := c.LastAdmissionDecision("same#2")
	if !ok1 || !ok2 {
		t.Fatalf("expected receipts for same#1 and same#2; got ok1=%v ok2=%v", ok1, ok2)
	}
	if r1.TraceID != "same#1" || r2.TraceID != "same#2" {
		t.Errorf("trace ids = %q/%q, want same#1/same#2", r1.TraceID, r2.TraceID)
	}
	if r1.Tokens != 11 || r2.Tokens != 22 {
		t.Errorf("footprints = %d/%d, want 11/22 uncollapsed", r1.Tokens, r2.Tokens)
	}
	if r1.SessionID != "same" || r2.SessionID != "same" {
		t.Errorf("session grouping = %q/%q, want both same", r1.SessionID, r2.SessionID)
	}
}

// TestAdmissionReceiptQueuedThenPromotedRefresh drives the Acquire queued->promoted refresh:
// a request blocked behind a full running set is recorded Queued, and when the holder
// completes and the waiter is promoted, the SAME trace's receipt is refreshed to Admitted.
func TestAdmissionReceiptQueuedThenPromotedRefresh(t *testing.T) {
	// One running slot, budget for exactly one 10-token request at a time.
	c := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, TokenBudget: 10, MaxWaiting: 8, AgingRounds: 0})

	holder, err := c.Acquire(context.Background(), SeqRequest{TraceID: "holder", Tokens: 10})
	if err != nil {
		t.Fatalf("Acquire(holder): %v", err)
	}

	// The second Acquire must queue (no headroom) rather than return immediately.
	leaseCh := make(chan *AdmissionLease, 1)
	errCh := make(chan error, 1)
	go func() {
		l, err := c.Acquire(context.Background(), SeqRequest{TraceID: "waiter", Tokens: 10})
		if err != nil {
			errCh <- err
			return
		}
		leaseCh <- l
	}()

	// Wait until the waiter's receipt exists as queued (bounded poll, no fixed sleep).
	waitForVerdict(t, c, "waiter#2", "queued")

	// Free the slot; the promote promotes the waiter and refreshes its receipt.
	holder.Release()
	lease, err := waitForLease(t, leaseCh, errCh)
	if err != nil {
		t.Fatalf("promoted Acquire returned error: %v", err)
	}
	defer lease.Release()

	post, ok := c.LastAdmissionDecision("waiter#2")
	if !ok {
		t.Fatal("receipt for waiter#2 vanished after promotion")
	}
	if post.Verdict != "admitted" || !post.Admitted {
		t.Errorf("post-promotion receipt = %+v, want refreshed to admitted/true", post)
	}
}

// TestAdmissionReceiptContextCancelIsTerminal is the terminal-on-cancel witness: a waiter whose
// context is cancelled while still queued records a terminal Expired receipt naming the
// cancellation reason (there is no "cancelled" member in the closed verdict enum).
func TestAdmissionReceiptContextCancelIsTerminal(t *testing.T) {
	c := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, TokenBudget: 5, MaxWaiting: 8, AgingRounds: 0})

	if _, err := c.Acquire(context.Background(), SeqRequest{TraceID: "holder", Tokens: 5, DecodeTTL: time.Hour}); err != nil {
		t.Fatalf("Acquire(holder): %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	waitDone := make(chan error, 1)
	go func() {
		_, err := c.Acquire(ctx, SeqRequest{TraceID: "waiter", Tokens: 5, DecodeTTL: time.Hour})
		waitDone <- err
	}()

	waitForVerdict(t, c, "waiter#2", "queued")
	cancel()
	select {
	case err := <-waitDone:
		if err == nil {
			t.Fatal("cancelled Acquire returned a nil error")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("cancelled waiter did not unblock")
	}

	rec, ok := c.LastAdmissionDecision("waiter#2")
	if !ok {
		t.Fatal("no terminal receipt for the cancelled waiter (waiter#2)")
	}
	if rec.Verdict != VerdictExpired.String() || rec.Admitted {
		t.Errorf("Verdict/Admitted = %q/%v, want expired/false", rec.Verdict, rec.Admitted)
	}
	if rec.Reason != "context cancelled while waiting" {
		t.Errorf("Reason = %q, want the cancellation reason", rec.Reason)
	}
}

// TestAdmissionReceiptRetentionEvictsOldest is the bounded-growth witness: the retained ring
// caps at admissionDecisionHistory and drops oldest-first, so a long serve cannot grow it.
func TestAdmissionReceiptRetentionEvictsOldest(t *testing.T) {
	c := NewAdmissionController(AdmissionPolicy{
		TokenBudget: 1 << 20, MaxNumSeqs: 1 << 20, MaxWaiting: 1 << 20, AgingRounds: 1,
	})

	total := admissionDecisionHistory + 5
	for i := 0; i < total; i++ {
		id := "k" + strconv.Itoa(i)
		if v := c.Offer(SeqRequest{TraceID: id, Tokens: 1}); v != VerdictAdmitted {
			t.Fatalf("offer %s = %s, want admitted", id, v)
		}
	}

	if _, ok := c.LastAdmissionDecision("k" + strconv.Itoa(total-1)); !ok {
		t.Error("newest decision was not retained")
	}
	for i := 0; i < total-admissionDecisionHistory; i++ {
		if _, ok := c.LastAdmissionDecision("k" + strconv.Itoa(i)); ok {
			t.Errorf("evicted decision k%d still retained", i)
		}
	}

	c.mu.Lock()
	mapLen, orderLen := len(c.decisions), len(c.decisionOrder)
	c.mu.Unlock()
	if mapLen != admissionDecisionHistory || orderLen != admissionDecisionHistory {
		t.Errorf("retained %d map / %d order, want both %d", mapLen, orderLen, admissionDecisionHistory)
	}
}

// TestAdmissionReceiptUnknownAndNilSafety covers the miss path and nil-receiver safety on the
// read-back API: an unknown trace reports not-found (never a fabricated zero receipt), and a
// nil controller answers false rather than panicking.
func TestAdmissionReceiptUnknownAndNilSafety(t *testing.T) {
	c := NewAdmissionController(DefaultAdmissionPolicy())
	if rec, ok := c.LastAdmissionDecision("ghost"); ok {
		t.Errorf("unknown trace returned ok=true with %+v", rec)
	}

	var nilCtl *AdmissionController
	if _, ok := nilCtl.LastAdmissionDecision("anything"); ok {
		t.Error("nil controller returned a receipt")
	}
}

// waitForVerdict polls (bounded, 2s max) until traceID's retained receipt carries want.
func waitForVerdict(t *testing.T, c *AdmissionController, traceID, want string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if rec, ok := c.LastAdmissionDecision(traceID); ok && rec.Verdict == want {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("receipt %s never reached verdict %q", traceID, want)
}

// waitForLease waits (bounded, 3s max) for the promoted Acquire either to yield a lease or an
// error, so the test never blocks forever on a scheduling bug.
func waitForLease(t *testing.T, leaseCh chan *AdmissionLease, errCh chan error) (*AdmissionLease, error) {
	t.Helper()
	select {
	case l := <-leaseCh:
		return l, nil
	case err := <-errCh:
		return nil, err
	case <-time.After(3 * time.Second):
		t.Fatal("promoted Acquire did not return within 3s")
		return nil, nil
	}
}
