package model

import (
	"reflect"
	"testing"
)

// TestKeyedAcceptedTokensSlotInvariant proves the keyed accept/bonus/residual
// coin stream is a function of (requestKey, domain, step) ONLY -- never the
// physical batch slot. Two samplers with unseeded/nondeterministic randSource
// but the SAME request key must produce byte-identical DraftVerificationResult,
// which is what makes preemption/re-admission reproducible. A third sampler
// with a DIFFERENT key must differ, so the assertion is not vacuous.
func TestKeyedAcceptedTokensSlotInvariant(t *testing.T) {
	draftTokens := []int{1, 2, 3}
	pTargets := [][]float32{
		{0.1, 0.9, 0.0, 0.0}, // pos 0: token 1 alpha 1.0
		{0.0, 0.0, 0.9, 0.1}, // pos 1: token 2 alpha 1.0
		{0.0, 0.0, 0.0, 1.0}, // pos 2: token 3 alpha 1.0
		{0.5, 0.5, 0.0, 0.0}, // bonus distribution
	}
	pDrafts := [][]float32{
		{0.1, 0.9, 0.0, 0.0},
		{0.0, 0.0, 0.9, 0.1},
		{0.0, 0.0, 0.9, 0.1},
	}

	// Force a rejection with a non-degenerate residual at position 1 so the
	// keyed residual draw is exercised too.
	pTargetsReject := [][]float32{
		{0.1, 0.9, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0}, // token 2 has alpha 0.0 => always rejects
		{0.0, 0.0, 0.0, 1.0},
	}

	// Two distinct physical "slots": both armed with the same request key but
	// backed by the shared, unseeded randSource (nondeterministic if consulted).
	slotA := NewRejectionSampler(nil)
	slotA.SetRequestKey("req-abc")
	slotB := NewRejectionSampler(nil)
	slotB.SetRequestKey("req-abc")

	resA := slotA.VerifyDraftSequenceKeyed(draftTokens, pTargetsReject, pDrafts)
	resB := slotB.VerifyDraftSequenceKeyed(draftTokens, pTargetsReject, pDrafts)

	if !reflect.DeepEqual(resA, resB) {
		t.Fatalf("same request key across slots diverged:\n A=%+v\n B=%+v", resA, resB)
	}
	if resA.RejectedAt != 1 {
		t.Fatalf("RejectedAt = %d, want 1", resA.RejectedAt)
	}
	if resA.AcceptedCount != 1 {
		t.Fatalf("AcceptedCount = %d, want 1", resA.AcceptedCount)
	}

	// All-accepted path: proves the keyed bonus draw is also slot-invariant.
	allA := slotA.VerifyDraftSequenceKeyed(draftTokens, pTargets, pDrafts)
	allB := slotB.VerifyDraftSequenceKeyed(draftTokens, pTargets, pDrafts)
	if !reflect.DeepEqual(allA, allB) {
		t.Fatalf("all-accepted bonus draw diverged:\n A=%+v\n B=%+v", allA, allB)
	}
	if allA.RejectedAt != -1 || allA.AcceptedCount != 3 {
		t.Fatalf("all-accepted result = %+v, want 3 accepted / RejectedAt -1", allA)
	}

	// Non-vacuity: a different request key must diverge at the accept coin for
	// some step in a bounded stream.
	keyABC := HashRequestID("req-abc")
	keyXYZ := HashRequestID("req-xyz")
	diverged := false
	for step := 0; step < 64; step++ {
		if KeyedUniform(keyABC, DomainAcceptCoin, step) != KeyedUniform(keyXYZ, DomainAcceptCoin, step) {
			diverged = true
			break
		}
	}
	if !diverged {
		t.Fatal("different request keys produced an identical 64-step accept-coin stream")
	}

	// Unkeyed sampler must keep the legacy path: KeyedOn false and the keyed
	// entry point deferring to the legacy randSource path.
	unkeyed := NewRejectionSampler(func() float32 { return 0.0 })
	if unkeyed.KeyedOn() {
		t.Fatal("fresh sampler reported KeyedOn() == true")
	}
	if got := unkeyed.VerifyDraftSequenceKeyed(draftTokens, pTargets, pDrafts); got.AcceptedCount != 3 {
		t.Fatalf("unkeyed delegation accepted = %d, want 3", got.AcceptedCount)
	}
	unkeyed.SetRequestKey("")
	if unkeyed.KeyedOn() {
		t.Fatal("empty request id did not clear the key")
	}
}

// TestKeyedRequestsDoNotCorrelate proves two co-resident requests that share
// every distribution but differ only in request id draw decorrelated accept
// coins, while each individual stream stays uniform enough.
func TestKeyedRequestsDoNotCorrelate(t *testing.T) {
	const trials = 20000
	const alpha = 0.5

	key1 := HashRequestID("r1")
	key2 := HashRequestID("r2")
	if key1 == key2 {
		t.Fatal("distinct request ids folded to the same key")
	}

	diffSteps := 0
	var accepts1, accepts2 int
	for step := 0; step < trials; step++ {
		u1 := KeyedUniform(key1, DomainAcceptCoin, step)
		u2 := KeyedUniform(key2, DomainAcceptCoin, step)
		if u1 != u2 {
			diffSteps++
		}
		if u1 <= alpha {
			accepts1++
		}
		if u2 <= alpha {
			accepts2++
		}
	}

	// Exact stream inequality: not identical at step 0, and overwhelmingly
	// divergent across the whole stream.
	if KeyedUniform(key1, DomainAcceptCoin, 0) == KeyedUniform(key2, DomainAcceptCoin, 0) {
		t.Fatal("two request keys produced the same accept coin at step 0")
	}
	if diffSteps < trials/2 {
		t.Fatalf("only %d/%d steps differed between request keys, want > half", diffSteps, trials)
	}

	// Both streams remain individually uniform: with alpha=0.5 the accept rate
	// must be ~0.5 and the two rates must be close.
	rate1 := float64(accepts1) / float64(trials)
	rate2 := float64(accepts2) / float64(trials)
	if rate1 < 0.45 || rate1 > 0.55 {
		t.Fatalf("request r1 accept rate = %g, want within [0.45,0.55] at alpha=0.5", rate1)
	}
	if rate2 < 0.45 || rate2 > 0.55 {
		t.Fatalf("request r2 accept rate = %g, want within [0.45,0.55] at alpha=0.5", rate2)
	}
	if d := rate1 - rate2; d < -0.05 || d > 0.05 {
		t.Fatalf("accept-rate difference = %g, want |rate1-rate2| < 0.05", d)
	}
}

// TestKeyedUniformRangeAndDomains pins the two structural invariants the design
// review relied on: every draw lands in [0,1), domains never collide on the same
// (key, step), and the step=0 draw is not a no-op.
func TestKeyedUniformRangeAndDomains(t *testing.T) {
	key := HashRequestID("req-range")
	for step := 0; step < 1024; step++ {
		u := KeyedUniform(key, DomainAcceptCoin, step)
		if u < 0 || u >= 1.0 {
			t.Fatalf("KeyedUniform step %d = %g, want [0,1)", step, u)
		}
	}
	for step := 0; step < 64; step++ {
		a := KeyedUniform(key, DomainAcceptCoin, step)
		b := KeyedUniform(key, DomainBonusToken, step)
		c := KeyedUniform(key, DomainResidual, step)
		if a == b || b == c || a == c {
			t.Fatalf("domain collision at step %d: accept=%g bonus=%g residual=%g", step, a, b, c)
		}
	}
	if KeyedUniform(0, DomainAcceptCoin, 0) == 0 {
		t.Fatal("keyed draw at (0, accept, 0) collapsed to 0")
	}
	if HashRequestID("") != 0 {
		t.Fatal("HashRequestID(\"\") != 0")
	}
	if HashRequestID("r1") == HashRequestID("r2") {
		t.Fatal("HashRequestID collision on distinct ids")
	}
}

// TestKeyedSingleTokenResidualsDoNotCollide proves the single-token keyed
// residual draw is keyed on the rejected token identity, not a fixed step 0, so
// two distinct rejections under one request do not share a residual coin. It
// also re-proves slot-independence for the single-token path.
func TestKeyedSingleTokenResidualsDoNotCollide(t *testing.T) {
	// nextResidualCoin is exercised directly: it has no target/draft inputs, so
	// no fixture is needed beyond the armed key. Both tokens are rejected
	// positions under one request; their residual coins must differ.
	s := NewRejectionSampler(nil)
	s.SetRequestKey("req-resid")

	c0 := s.nextResidualCoin(0)
	c1 := s.nextResidualCoin(1)
	if c0 == c1 {
		t.Fatalf("residual coins for distinct rejected tokens collide: %g == %g", c0, c1)
	}

	// Slot-invariance of the single-token path: same key on a second sampler
	// yields the same residual coin stream.
	s2 := NewRejectionSampler(nil)
	s2.SetRequestKey("req-resid")
	if s2.nextResidualCoin(0) != c0 || s2.nextResidualCoin(1) != c1 {
		t.Fatal("single-token residual coin not reproducible across samplers with the same key")
	}
}
