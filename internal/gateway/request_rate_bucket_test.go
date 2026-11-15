package gateway

// request_rate_bucket_test.go — witness for the idiomatic Go re-expression of
// TensorFold's per-principal request-rate token buckets (borrow-port part-out
// oss-port-tensorfold-http-rate-admission-20261103, ADAPT verdict): refill math,
// retry-after, cap refusal, and the beginServedAdmission wired refusal/refund matrix.
// Software evidence only — no throughput or hardware claim.

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// testRateClock is the injectable monotonic clock the boundary tests advance by hand.
type testRateClock struct{ t time.Time }

func (c *testRateClock) now() time.Time          { return c.t }
func (c *testRateClock) advance(d time.Duration) { c.t = c.t.Add(d) }

func TestRequestRateBucketRefusals(t *testing.T) {
	t.Run("refillMath", func(t *testing.T) {
		clk := &testRateClock{t: time.Unix(0, 0).UTC()}
		g := NewRequestRateGate(RateBucketPolicy{RequestsPerSecond: 4, RequestBurst: 2})
		g.SetClock(clk.now)
		// Fresh bucket starts full: capacity = burst × one second's credit = 2e9 ns-tokens.
		l, err := g.Take("p", 2) // debit the whole burst
		if err != nil || l == nil {
			t.Fatalf("full-burst Take = (%v, %v), want an admitted lease", l, err)
		}
		if st := g.BucketFor("p"); st.Credit != 0 || st.Capacity != 2*rateTokenNs {
			t.Fatalf("state after full-burst Take = %+v, want credit 0 / capacity 2e9", st)
		}
		// 250ms at 4 req/s earns 2.5e8 ns × 4 = exactly 1e9 ns-tokens.
		clk.advance(250 * time.Millisecond)
		if got := g.BucketFor("p").Credit; got != rateTokenNs {
			t.Fatalf("credit after 250ms = %d, want exactly %d — the top-up is elapsed × rate", got, int64(rateTokenNs))
		}
		l2, err := g.Take("p", 1) // covers the refilled 1e9 exactly
		if err != nil || l2 == nil {
			t.Fatalf("post-top-up Take = (%v, %v), want admit at exactly-covered credit", l2, err)
		}
		if got := g.BucketFor("p").Credit; got != 0 {
			t.Fatalf("credit after exact-cover Take = %d, want 0 — the debit is exact", got)
		}
		// Cap clamp: a huge clock jump tops up to exactly capacity, never beyond.
		clk.advance(1000 * time.Hour)
		if got := g.BucketFor("p").Credit; got != 2*rateTokenNs {
			t.Fatalf("credit after a huge jump = %d, want the capacity ceiling %d — credit never exceeds capacity", got, int64(2*rateTokenNs))
		}
		l.Release()
		l2.Release()
	})

	t.Run("retryAfter", func(t *testing.T) {
		clk := &testRateClock{t: time.Unix(0, 0).UTC()}
		g := NewRequestRateGate(RateBucketPolicy{RequestsPerSecond: 1, RequestBurst: 10})
		g.SetClock(clk.now)
		if _, err := g.Take("p", 10); err != nil {
			t.Fatalf("full-burst Take err = %v, want admit", err)
		}
		// Deficit 10e9 at 1 req/s → 10 whole seconds.
		_, err := g.Take("p", 10)
		if !errors.Is(err, ErrRequestRateLimited) || !errors.Is(err, ErrRequestRate) {
			t.Fatalf("exhausted Take err = %v, want errors.Is(ErrRequestRateLimited) under the parent sentinel", err)
		}
		if ra, ok := RetryAfterSeconds(err); !ok || ra != 10 {
			t.Fatalf("RetryAfterSeconds = (%d, %v), want (10, true) — exact against the injected clock", ra, ok)
		}
		// Deficit 1e9 → exactly 1 second (the floor).
		_, err = g.Take("p", 1)
		if !errors.Is(err, ErrRequestRateLimited) {
			t.Fatalf("partial Take err = %v, want the rate class", err)
		}
		if ra, _ := RetryAfterSeconds(err); ra != 1 {
			t.Fatalf("RetryAfterSeconds = %d, want 1 — the one-second floor", ra)
		}
		// A partial top-up changes the hint: after 5s the deficit is 5e9 → 5 seconds.
		clk.advance(5 * time.Second)
		_, err = g.Take("p", 10)
		if !errors.Is(err, ErrRequestRateLimited) {
			t.Fatalf("partially topped-up Take err = %v, want the rate class", err)
		}
		if ra, _ := RetryAfterSeconds(err); ra != 5 {
			t.Fatalf("RetryAfterSeconds = %d, want 5 — the hint tracks the live deficit", ra)
		}
		// And the same clock admits what the hint promised.
		clk.advance(5 * time.Second)
		if _, err := g.Take("p", 10); err != nil {
			t.Fatalf("post-hint Take err = %v, want admit — the hint is exact against the injected clock", err)
		}
	})

	t.Run("burstOverride", func(t *testing.T) {
		// An explicit burst beats the rate: capacity = burst × one second's credit.
		g := NewRequestRateGate(RateBucketPolicy{RequestsPerSecond: 1, RequestBurst: 5})
		if _, err := g.Take("p", 5); err != nil {
			t.Fatalf("fresh-burst Take err = %v, want the full burst admitted up front", err)
		}
		_, err := g.Take("p", 5)
		if !errors.Is(err, ErrRequestRateLimited) {
			t.Fatalf("over-burst Take err = %v, want the rate class", err)
		}
		if ra, _ := RetryAfterSeconds(err); ra != 5 {
			t.Fatalf("RetryAfterSeconds = %d, want 5 — deficit 5e9 at 1 req/s", ra)
		}
	})

	t.Run("capRefusal", func(t *testing.T) {
		clk := &testRateClock{t: time.Unix(0, 0).UTC()}
		g := NewRequestRateGate(RateBucketPolicy{RequestsPerSecond: 100, PrincipalSlots: 2})
		g.SetClock(clk.now)
		la, err := g.Take("A", 1)
		if err != nil || la == nil {
			t.Fatalf("principal A Take = (%v, %v), want admit", la, err)
		}
		lb, err := g.Take("B", 1)
		if err != nil || lb == nil {
			t.Fatalf("principal B Take = (%v, %v), want admit", lb, err)
		}
		_, err = g.Take("C", 1)
		if !errors.Is(err, ErrRequestRateCapacity) || !errors.Is(err, ErrRequestRate) {
			t.Fatalf("third distinct principal Take err = %v, want errors.Is(ErrRequestRateCapacity) under the parent sentinel", err)
		}
		if errors.Is(err, ErrRequestRateLimited) {
			t.Fatalf("capacity exhaustion misclassified as the rate class: %v", err)
		}
		if snap := g.Snapshot(); snap.RejectedCapacity != 1 || snap.Buckets != 2 {
			t.Fatalf("snapshot = %+v, want exactly 1 capacity refusal and 2 live buckets — the table did not grow past its bound", snap)
		}
		// Existing principals are unaffected by the bound.
		if _, err := g.Take("A", 1); err != nil {
			t.Fatalf("repeat principal A Take err = %v, want admit — an existing seat costs no new slot", err)
		}
		// Release everything, let the buckets top up, and the seat comes back.
		la.Release()
		lb.Release()
		clk.advance(time.Minute) // 100 req/s ≫ the 100-request capacity: both buckets full
		lc, err := g.Take("C", 1)
		if err != nil || lc == nil {
			t.Fatalf("post-sweep principal C Take = (%v, %v), want the swept idle-full seat reused", lc, err)
		}
		lc.Release()
	})

	t.Run("zeroRateDisables", func(t *testing.T) {
		g := NewRequestRateGate(RateBucketPolicy{}) // rate 0 = unlimited
		for i := 0; i < 200; i++ {
			l, err := g.Take(fmt.Sprintf("principal-%d", i%150), 1)
			if err != nil {
				t.Fatalf("hold %d Take err = %v, want admit — a disabled gate refuses nothing", i, err)
			}
			if l != nil {
				t.Fatalf("hold %d lease = %+v, want nil — a disabled gate accounts nothing", i, l)
			}
		}
		snap := g.Snapshot()
		if snap.Buckets != 0 || snap.Admitted != 0 || snap.RejectedRate+snap.RejectedCapacity != 0 {
			t.Fatalf("disabled-gate snapshot = %+v, want nothing counted across 200 unbounded takes", snap)
		}
		if st := g.BucketFor("principal-0"); st != (RateBucketState{}) {
			t.Fatalf("BucketFor on a disabled gate = %+v, want the zero state", st)
		}
		var nilLease *RateLease
		nilLease.Release() // must be a no-op, not a panic
		nilLease.Cancel()
	})

	t.Run("sentinelClasses", func(t *testing.T) {
		cases := []struct {
			err    error
			want   error
			notErr []error
		}{
			{err: &rateRefusal{cause: ErrRequestRateLimited}, want: ErrRequestRateLimited, notErr: []error{ErrRequestRateCapacity}},
			{err: &rateRefusal{cause: ErrRequestRateCapacity}, want: ErrRequestRateCapacity, notErr: []error{ErrRequestRateLimited}},
		}
		for i, tc := range cases {
			if !errors.Is(tc.err, tc.want) {
				t.Fatalf("case %d: errors.Is(%v, own class) = false, want true", i, tc.err)
			}
			if !errors.Is(tc.err, ErrRequestRate) {
				t.Fatalf("case %d: errors.Is(%v, ErrRequestRate) = false, want the parent sentinel", i, tc.err)
			}
			for _, nw := range tc.notErr {
				if errors.Is(tc.err, nw) {
					t.Fatalf("case %d: errors.Is(%v, %v) = true, want the classes to stay distinct", i, tc.err, nw)
				}
			}
		}
		if _, ok := RetryAfterSeconds(errors.New("unrelated")); ok {
			t.Fatalf("RetryAfterSeconds on an unrelated error = ok, want the hint only on rate refusals")
		}
	})

	t.Run("cancelOnce", func(t *testing.T) {
		clk := &testRateClock{t: time.Unix(0, 0).UTC()}
		g := NewRequestRateGate(RateBucketPolicy{RequestsPerSecond: 10, RequestBurst: 10})
		g.SetClock(clk.now)
		l, err := g.Take("p", 10)
		if err != nil || l == nil {
			t.Fatalf("full-burst Take = (%v, %v), want an admitted lease", l, err)
		}
		if got := g.BucketFor("p").Credit; got != 0 {
			t.Fatalf("credit after Take = %d, want 0", got)
		}
		l.Cancel()
		if got := g.BucketFor("p").Credit; got != 10*rateTokenNs {
			t.Fatalf("credit after Cancel = %d, want the full burst refunded", got)
		}
		l.Cancel() // second Cancel must not refund twice
		if got := g.BucketFor("p").Credit; got != 10*rateTokenNs {
			t.Fatalf("credit after double Cancel = %d, want the burst still exactly once — no double refund", got)
		}
		l.Release() // post-Cancel Release is a no-op, not a panic

		// The cancelled seat is free at the slot bound: a second principal can take it.
		g2 := NewRequestRateGate(RateBucketPolicy{RequestsPerSecond: 10, RequestBurst: 10, PrincipalSlots: 1})
		g2.SetClock(clk.now)
		l1, err := g2.Take("A", 10)
		if err != nil || l1 == nil {
			t.Fatalf("principal A Take = (%v, %v), want admit", l1, err)
		}
		l1.Cancel()
		l2, err := g2.Take("B", 10)
		if err != nil || l2 == nil {
			t.Fatalf("post-cancel principal B Take = (%v, %v), want the freed seat — cancel releases the seat", l2, err)
		}
		l2.Release()
	})

	t.Run("beginServedAdmission", func(t *testing.T) {
		turn := servedSessionTurn{traceID: "rate-wire", state: SessionState{Priority: 0}, maxTokens: 1}

		// Exhausted live bucket: typed shed naming the rate class with its retry-after,
		// rendered into the 429 body through admissionErrorStatus.
		srv := newTestServer(t)
		clk := &testRateClock{t: time.Unix(0, 0).UTC()}
		rg := NewRequestRateGate(RateBucketPolicy{RequestsPerSecond: 1, RequestBurst: 1})
		rg.SetClock(clk.now)
		SetRequestRateGate(rg)
		defer SetRequestRateGate(nil)
		lease, err := srv.beginServedAdmission(context.Background(), turn, nil, nil, 1)
		if err != nil || lease == nil {
			t.Fatalf("first beginServedAdmission = (%v, %v), want an admitted lease", lease, err)
		}
		if got := rg.BucketFor("").Credit; got != 0 {
			t.Fatalf("credit after one admitted request = %d, want 0 — exactly one token debited from the caller's bucket", got)
		}
		_, err = srv.beginServedAdmission(context.Background(), turn, nil, nil, 1)
		var ae *AdmissionError
		if !errors.As(err, &ae) || ae.Verdict != VerdictShed {
			t.Fatalf("saturated beginServedAdmission err = %v, want *AdmissionError VerdictShed", err)
		}
		if !strings.Contains(ae.Reason, "request rate limit") || !strings.Contains(ae.Reason, "retry after 1 second") {
			t.Fatalf("shed reason = %q, want the rate class named with its retry-after", ae.Reason)
		}
		status, code, _, ok := admissionErrorStatus(err)
		if !ok || status != http.StatusTooManyRequests || code != "scheduler_overloaded" {
			t.Fatalf("HTTP mapping = (%d, %q, %v), want (429, scheduler_overloaded, true)", status, code, ok)
		}
		// The admitted request's seat rides the lease-release path; the top-up then
		// admits the next request.
		lease.Release()
		clk.advance(time.Second)
		lease2, err := srv.beginServedAdmission(context.Background(), turn, nil, nil, 1)
		if err != nil || lease2 == nil {
			t.Fatalf("post-top-up beginServedAdmission = (%v, %v), want admit", lease2, err)
		}
		lease2.Release()

		// Later-stage refusal refunds the token: the shared provider window refuses after
		// the rate gate admitted, so the tenant's credit must come back (cancel semantics
		// — the provider was never called).
		srv2 := newTestServer(t)
		rg2 := NewRequestRateGate(RateBucketPolicy{RequestsPerSecond: 100, RequestBurst: 100})
		SetRequestRateGate(rg2)
		shared := NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 10}})
		srv2.SetTokenRateGate(shared)
		if _, err := shared.Admit(ratelimit.NewTokenUsage(8, 0, 0)); err != nil {
			t.Fatalf("direct shared-gate Admit err = %v, want the saturating reservation admitted", err)
		}
		msgs := []agent.Message{{Role: "user", Content: strings.Repeat("x", 196)}} // ~51-token estimate
		_, err = srv2.beginServedAdmission(context.Background(), turn, msgs, nil, 1)
		if !errors.As(err, &ae) || ae.Verdict != VerdictShed || !strings.Contains(ae.Reason, "provider token budget") {
			t.Fatalf("provider-saturated beginServedAdmission err = %v, want the shared window to name itself", err)
		}
		if got := rg2.BucketFor("").Credit; got != 100*rateTokenNs {
			t.Fatalf("credit after a later-stage refusal = %d, want the full burst — the token was refunded", got)
		}

		// Scheduler-slot refusal refunds too, and frees the seat at the slot bound: the
		// second principal must reach the scheduler, not the rate-capacity wall.
		srv3 := newTestServer(t)
		rg3 := NewRequestRateGate(RateBucketPolicy{RequestsPerSecond: 100, RequestBurst: 100, PrincipalSlots: 1})
		SetRequestRateGate(rg3)
		policy := DefaultAdmissionPolicy()
		policy.TokenBudget = 32
		srv3.SetAdmissionController(NewAdmissionController(policy))
		ctxA := WithPrincipal(context.Background(), "org-a")
		_, err = srv3.beginServedAdmission(ctxA, turn, nil, nil, 64)
		if !errors.As(err, &ae) || ae.Verdict != VerdictRefused {
			t.Fatalf("impossible-envelope beginServedAdmission err = %v, want the scheduler's typed refusal", err)
		}
		if got := rg3.BucketFor("org-a").Credit; got != 100*rateTokenNs {
			t.Fatalf("org-a credit after the scheduler refusal = %d, want the full burst — the token was refunded", got)
		}
		ctxB := WithPrincipal(context.Background(), "org-b")
		_, err = srv3.beginServedAdmission(ctxB, turn, nil, nil, 64)
		if !errors.As(err, &ae) || ae.Verdict != VerdictRefused {
			t.Fatalf("org-b beginServedAdmission err = %v, want the scheduler's refusal — the cancelled org-a seat was swept, not a rate-capacity refusal", err)
		}

		// No live gate: byte-for-byte historical.
		SetRequestRateGate(nil)
		plain := newTestServer(t)
		lease, err = plain.beginServedAdmission(context.Background(), turn, nil, nil, 1)
		if err != nil || lease != nil {
			t.Fatalf("unwired-gate beginServedAdmission = (%v, %v), want the historical inert (nil, nil)", lease, err)
		}
	})
}
