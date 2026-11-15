package gateway

// inflight_admission_test.go — witness for the idiomatic Go re-expression of
// TensorFold's opt-in in-flight request-count admission (borrow-port part-out
// oss-port-tensorfold-http-inflight-admission-20261103, ADAPT verdict): caps,
// exhaustion, release-once, and the beginServedAdmission refusal/release matrix.
// Software evidence only — no throughput or hardware claim.

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// fak-test:runtime fast est=1ms lane=default
func TestInflightAdmissionPolicyOwnership(t *testing.T) {
	t.Parallel()
	t.Run("deleting caller cap cannot bypass live lease", func(t *testing.T) {
		limits := map[string]int{"k": 1}
		g := NewInflightGate(InflightPolicy{PerKey: limits})
		first, err := g.Acquire("p", "k")
		if err != nil || first == nil {
			t.Fatalf("first acquire: lease=%v error=%v", first, err)
		}
		defer first.Release()
		delete(limits, "k")
		denied, err := g.Acquire("p", "k")
		if denied != nil || !errors.Is(err, ErrInflightKey) || !errors.Is(err, ErrInflight) {
			t.Fatalf("caller mutation bypassed cap: lease=%v error=%v", denied, err)
		}
		if got := g.Snapshot(); got.Admitted != 1 || got.RejectedKey != 1 || got.InFlight != 1 {
			t.Fatalf("refusal changed live accounting: %+v", got)
		}
		first.Release()
		next, err := g.Acquire("p", "k")
		if err != nil || next == nil {
			t.Fatalf("release did not restore capacity: lease=%v error=%v", next, err)
		}
		next.Release()
	})
	t.Run("adding caller cap cannot arm inert gate", func(t *testing.T) {
		limits := map[string]int{}
		g := NewInflightGate(InflightPolicy{PerKey: limits})
		limits["k"] = 1
		for i := 0; i < 2; i++ {
			lease, err := g.Acquire("p", "k")
			if err != nil || lease != nil {
				lease.Release()
				t.Fatalf("caller mutation armed inert gate: lease=%v error=%v", lease, err)
			}
		}
		if got := g.Snapshot(); got != (InflightSnapshot{}) {
			t.Fatalf("inert gate retained accounting: %+v", got)
		}
	})
}

func TestInflightAdmissionLeasesRefusals(t *testing.T) {
	t.Run("caps", func(t *testing.T) {
		g := NewInflightGate(InflightPolicy{MaxInflight: 2})
		l1, err := g.Acquire("p", "k")
		if err != nil || l1 == nil {
			t.Fatalf("first Acquire = (%v, %v), want an admitted lease", l1, err)
		}
		if _, err := g.Acquire("q", "k"); err != nil {
			t.Fatalf("second Acquire err = %v, want admit under the global cap of 2", err)
		}
		if snap := g.Snapshot(); snap.InFlight != 2 || snap.Admitted != 2 {
			t.Fatalf("snapshot = %+v, want 2 in flight / 2 admitted", snap)
		}
		_, err = g.Acquire("r", "k")
		if !errors.Is(err, ErrInflightGlobal) || !errors.Is(err, ErrInflight) {
			t.Fatalf("third Acquire err = %v, want errors.Is(ErrInflightGlobal) under the parent sentinel", err)
		}
		if errors.Is(err, ErrInflightKey) {
			t.Fatalf("global exhaustion misclassified as the key class: %v", err)
		}

		kg := NewInflightGate(InflightPolicy{PerKey: map[string]int{"k": 1}})
		if _, err := kg.Acquire("p", "k"); err != nil {
			t.Fatalf("capped-key Acquire err = %v, want admit at count 0 of cap 1", err)
		}
		_, err = kg.Acquire("p", "k")
		if !errors.Is(err, ErrInflightKey) || !errors.Is(err, ErrInflight) {
			t.Fatalf("repeat capped-key Acquire err = %v, want errors.Is(ErrInflightKey)", err)
		}
		if _, err := kg.Acquire("p", "other"); err != nil {
			t.Fatalf("uncapped-key Acquire err = %v, want admit — an absent PerKey entry counts nothing", err)
		}
		if snap := kg.Snapshot(); snap.InFlight != 2 || snap.RejectedKey != 1 {
			t.Fatalf("per-key snapshot = %+v, want 2 in flight and exactly 1 key-class refusal", snap)
		}
	})

	t.Run("exhaustion", func(t *testing.T) {
		g := NewInflightGate(InflightPolicy{MaxInflight: 8, PerKey: map[string]int{"k": 8}, PrincipalSlots: 2})
		la, err := g.Acquire("A", "k")
		if err != nil {
			t.Fatalf("principal A Acquire err = %v, want admit", err)
		}
		lb, err := g.Acquire("B", "k")
		if err != nil {
			t.Fatalf("principal B Acquire err = %v, want admit", err)
		}
		lb2, err := g.Acquire("B", "k")
		if err != nil || lb2 == nil {
			t.Fatalf("repeat principal B Acquire = (%v, %v), want admit — an existing seat costs no new slot", lb2, err)
		}
		_, err = g.Acquire("C", "k")
		if !errors.Is(err, ErrInflightCapacity) || !errors.Is(err, ErrInflight) {
			t.Fatalf("third distinct principal Acquire err = %v, want errors.Is(ErrInflightCapacity)", err)
		}
		if snap := g.Snapshot(); snap.RejectedCapacity != 1 || snap.InFlight != 3 {
			t.Fatalf("snapshot = %+v, want exactly 1 capacity refusal and 3 live holds", snap)
		}
		la.Release()
		lc, err := g.Acquire("C", "k")
		if err != nil {
			t.Fatalf("post-release principal C Acquire err = %v, want the freed seat reused immediately", err)
		}
		if snap := g.Snapshot(); snap.InFlight != 3 || snap.Principals != 2 {
			t.Fatalf("post-sweep snapshot = %+v, want 3 live holds across 2 distinct principals", snap)
		}
		lc.Release()
		lb.Release()
		lb2.Release()
		if snap := g.Snapshot(); snap.InFlight != 0 {
			t.Fatalf("final snapshot = %+v, want every hold released", snap)
		}
	})

	t.Run("releaseOnce", func(t *testing.T) {
		g := NewInflightGate(InflightPolicy{MaxInflight: 1})
		l1, err := g.Acquire("a", "k")
		if err != nil {
			t.Fatalf("Acquire err = %v, want admit", err)
		}
		if _, err := g.Acquire("b", "k"); !errors.Is(err, ErrInflightGlobal) {
			t.Fatalf("saturated Acquire err = %v, want the global class", err)
		}
		var nilLease *InflightLease
		nilLease.Release() // must be a no-op, not a panic
		if got := g.Snapshot().InFlight; got != 1 {
			t.Fatalf("inflight after a refused acquire = %d, want 1 — a refused acquire holds nothing", got)
		}
		l1.Release()
		if got := g.Snapshot().InFlight; got != 0 {
			t.Fatalf("inflight after Release = %d, want 0", got)
		}
		l1.Release() // second Release is a no-op
		if got := g.Snapshot().InFlight; got != 0 {
			t.Fatalf("inflight after double Release = %d, want 0 — exactly once, never a double decrement", got)
		}
		if _, err := g.Acquire("c", "k"); err != nil {
			t.Fatalf("post-release Acquire err = %v, want full recovery to zero", err)
		}
	})

	t.Run("zeroLimitsCountNothing", func(t *testing.T) {
		g := NewInflightGate(InflightPolicy{})
		for i := 0; i < 200; i++ {
			l, err := g.Acquire(fmt.Sprintf("principal-%d", i%150), fmt.Sprintf("key-%d", i%7))
			if err != nil {
				t.Fatalf("hold %d Acquire err = %v, want admit — zero limits refuse nothing", i, err)
			}
			if l != nil {
				t.Fatalf("hold %d lease = %+v, want nil — an inert gate reserves nothing", i, l)
			}
		}
		snap := g.Snapshot()
		if snap.InFlight != 0 || snap.Principals != 0 || snap.Admitted != 0 ||
			snap.RejectedGlobal+snap.RejectedKey+snap.RejectedCapacity != 0 {
			t.Fatalf("zero-policy snapshot = %+v, want nothing counted across 200 unbounded holds", snap)
		}
	})

	t.Run("sentinelClasses", func(t *testing.T) {
		cases := []struct {
			err    error
			want   error
			notErr []error
		}{
			{err: &inflightRefusal{cause: ErrInflightGlobal}, want: ErrInflightGlobal, notErr: []error{ErrInflightKey, ErrInflightCapacity}},
			{err: &inflightRefusal{cause: ErrInflightKey}, want: ErrInflightKey, notErr: []error{ErrInflightGlobal, ErrInflightCapacity}},
			{err: &inflightRefusal{cause: ErrInflightCapacity}, want: ErrInflightCapacity, notErr: []error{ErrInflightGlobal, ErrInflightKey}},
		}
		for i, tc := range cases {
			if !errors.Is(tc.err, tc.want) {
				t.Fatalf("case %d: errors.Is(%v, own class) = false, want true", i, tc.err)
			}
			if !errors.Is(tc.err, ErrInflight) {
				t.Fatalf("case %d: errors.Is(%v, ErrInflight) = false, want the parent sentinel", i, tc.err)
			}
			for _, nw := range tc.notErr {
				if errors.Is(tc.err, nw) {
					t.Fatalf("case %d: errors.Is(%v, %v) = true, want the classes to stay distinct", i, tc.err, nw)
				}
			}
		}
	})

	t.Run("beginServedAdmission", func(t *testing.T) {
		turn := servedSessionTurn{traceID: "inflight-wire", state: SessionState{Priority: 0}, maxTokens: 1}

		// Global saturation: typed shed naming the global class, 429 mapping, and the
		// hold riding the request's single lease-release path exactly once.
		srv := newTestServer(t)
		ig := NewInflightGate(InflightPolicy{MaxInflight: 1})
		SetInflightGate(ig)
		defer SetInflightGate(nil)
		lease, err := srv.beginServedAdmission(context.Background(), turn, nil, nil, 1)
		if err != nil || lease == nil {
			t.Fatalf("beginServedAdmission = (%v, %v), want an admitted lease", lease, err)
		}
		if got := ig.Snapshot().InFlight; got != 1 {
			t.Fatalf("gate in flight = %d, want the hold visible before any release", got)
		}
		_, err = srv.beginServedAdmission(context.Background(), turn, nil, nil, 1)
		var ae *AdmissionError
		if !errors.As(err, &ae) || ae.Verdict != VerdictShed {
			t.Fatalf("saturated beginServedAdmission err = %v, want *AdmissionError VerdictShed", err)
		}
		if !strings.HasPrefix(ae.Reason, "inflight global") {
			t.Fatalf("shed reason = %q, want it to name the fired inflight class (global)", ae.Reason)
		}
		status, code, _, ok := admissionErrorStatus(err)
		if !ok || status != http.StatusTooManyRequests || code != "scheduler_overloaded" {
			t.Fatalf("HTTP mapping = (%d, %q, %v), want (429, scheduler_overloaded, true)", status, code, ok)
		}
		lease.Release()
		if got := ig.Snapshot().InFlight; got != 0 {
			t.Fatalf("inflight after lease Release = %d, want 0 — the hold rides the lease-release path", got)
		}
		lease.Release()
		if got := ig.Snapshot().InFlight; got != 0 {
			t.Fatalf("inflight after double lease Release = %d, want 0 — no double decrement", got)
		}

		// Per-key cap through the authenticated principal: org-a sheds at its cap while
		// org-b is unaffected, and the shed names the key class.
		srv2 := newTestServer(t)
		ig2 := NewInflightGate(InflightPolicy{PerKey: map[string]int{"org-a": 1}})
		SetInflightGate(ig2)
		ctxA := WithPrincipal(context.Background(), "org-a")
		leaseA, err := srv2.beginServedAdmission(ctxA, turn, nil, nil, 1)
		if err != nil || leaseA == nil {
			t.Fatalf("org-a beginServedAdmission = (%v, %v), want an admitted lease", leaseA, err)
		}
		_, err = srv2.beginServedAdmission(ctxA, turn, nil, nil, 1)
		if !errors.As(err, &ae) || ae.Verdict != VerdictShed || !strings.HasPrefix(ae.Reason, "inflight per-key") {
			t.Fatalf("org-a saturated err = %v, want the per-key class named in the shed", err)
		}
		ctxB := WithPrincipal(context.Background(), "org-b")
		leaseB, err := srv2.beginServedAdmission(ctxB, turn, nil, nil, 1)
		if err != nil || leaseB == nil {
			t.Fatalf("org-b beginServedAdmission = (%v, %v), want admit — org-a's cap must not touch org-b", leaseB, err)
		}
		leaseA.Release()
		leaseB.Release()
		if got := ig2.Snapshot().InFlight; got != 0 {
			t.Fatalf("inflight after both releases = %d, want 0", got)
		}

		// Later-stage refusal frees the hold charging nothing: the shared provider
		// window refuses after the inflight gate admitted, so the hold must come back.
		srv3 := newTestServer(t)
		ig3 := NewInflightGate(InflightPolicy{MaxInflight: 4})
		SetInflightGate(ig3)
		shared := NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 10}})
		srv3.SetTokenRateGate(shared)
		if _, err := shared.Admit(ratelimit.NewTokenUsage(8, 0, 0)); err != nil {
			t.Fatalf("direct shared-gate Admit err = %v, want the saturating reservation admitted", err)
		}
		msgs := []agent.Message{{Role: "user", Content: strings.Repeat("x", 196)}} // ~51-token estimate
		_, err = srv3.beginServedAdmission(context.Background(), turn, msgs, nil, 1)
		if !errors.As(err, &ae) || ae.Verdict != VerdictShed || !strings.Contains(ae.Reason, "provider token budget") {
			t.Fatalf("provider-saturated beginServedAdmission err = %v, want the shared window to name itself", err)
		}
		if got := ig3.Snapshot().InFlight; got != 0 {
			t.Fatalf("inflight after a later-stage refusal = %d, want 0 — the hold was freed charging nothing", got)
		}

		// Scheduler-slot refusal also frees the hold: the controller refuses an
		// impossible envelope after the inflight gate admitted, so the hold must come back.
		srv4 := newTestServer(t)
		ig4 := NewInflightGate(InflightPolicy{MaxInflight: 4})
		SetInflightGate(ig4)
		policy := DefaultAdmissionPolicy()
		policy.TokenBudget = 32
		srv4.SetAdmissionController(NewAdmissionController(policy))
		_, err = srv4.beginServedAdmission(context.Background(), turn, nil, nil, 64)
		if !errors.As(err, &ae) || ae.Verdict != VerdictRefused {
			t.Fatalf("impossible-envelope beginServedAdmission err = %v, want the scheduler's typed refusal", err)
		}
		if got := ig4.Snapshot().InFlight; got != 0 {
			t.Fatalf("inflight after a scheduler refusal = %d, want 0 — the hold was freed charging nothing", got)
		}

		// No live gate: byte-for-byte historical.
		SetInflightGate(nil)
		plain := newTestServer(t)
		lease, err = plain.beginServedAdmission(context.Background(), turn, nil, nil, 1)
		if err != nil || lease != nil {
			t.Fatalf("unwired-gate beginServedAdmission = (%v, %v), want the historical inert (nil, nil)", lease, err)
		}
	})
}
