package model

import (
	"reflect"
	"testing"
)

// keyedSampler is the per-request RNG-identity capability #13146 requires,
// expressed as an interface so this file COMPILES against the parent commit as
// well as the fix: every post-fix symbol is reached through a method on this
// interface rather than a direct call. On the parent the type assertion fails
// (red); at the fix it holds (green).
//
// Domains are passed as plain uint64 so this file never names the post-fix
// DomainTag type (which would break the parent build).
type keyedSampler interface {
	SetRequestKey(requestID string)
	KeyedOn() bool
	VerifyDraftSequenceKeyed(draftTokens []int, pTargets, pDrafts [][]float32) DraftVerificationResult
	VerifyTokenKeyedU(token int, pTarget, pDraft []float32, domain uint64, step int) TokenVerificationResult
	// KeyedStep returns the keyed draw at (domain, step) for the armed request.
	KeyedStepU(domain uint64, step int) (float32, bool)
}

// keyedRequest arms a request key through the interface.
func keyedRequest(t *testing.T, s *RejectionSampler, requestID string) keyedSampler {
	t.Helper()
	ks, ok := any(s).(keyedSampler)
	if !ok {
		t.Fatalf("RejectionSampler does not expose the per-request keyed coin; cannot arm request %q (#13146)", requestID)
	}
	ks.SetRequestKey(requestID)
	if !ks.KeyedOn() {
		t.Fatalf("SetRequestKey(%q) did not arm a keyed stream", requestID)
	}
	return ks
}

// Domain ordinals, mirrored locally so the test file does not reference the
// post-fix DomainTag constants directly.
const (
	domAccept uint64 = 0x1
	domBonus  uint64 = 0x2
	domResid  uint64 = 0x3
)

// The tests in this file are written to COMPILE against the parent commit as
// well as the fix, so the mandatory red-then-green symptom witness (#13146) can
// build the parent tree and observe the defect. The per-request keyed coin is
// reached through the keyedSampler interface assertion (declared in
// qwen_mtp_seed_test.go) rather than a direct call, so the parent tree still
// compiles and the capability tests fail at RUNTIME there instead of at build.

// TestKeyedAcceptedTokensSlotInvariant proves the keyed accept/bonus/residual
// coin stream is a function of the request identity only -- never the physical
// batch slot. Two samplers with unseeded/nondeterministic randSource but the SAME
// request key must produce byte-identical DraftVerificationResult, which is what
// makes preemption/re-admission reproducible. A different key must differ, so the
// assertion is not vacuous.
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
	pTargetsReject := [][]float32{
		{0.1, 0.9, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0}, // token 2 alpha 0.0 => always rejects
		{0.0, 0.0, 0.0, 1.0},
	}

	runKeyed := func(requestID string, pt [][]float32) DraftVerificationResult {
		s := NewRejectionSampler(nil) // unseeded: identical results prove the key, not luck
		ks := keyedRequest(t, s, requestID)
		return ks.VerifyDraftSequenceKeyed(draftTokens, pt, pDrafts)
	}

	resA := runKeyed("req-abc", pTargetsReject)
	resB := runKeyed("req-abc", pTargetsReject)
	if !reflect.DeepEqual(resA, resB) {
		t.Fatalf("same request key across slots diverged:\n A=%+v\n B=%+v", resA, resB)
	}
	if resA.RejectedAt != 1 || resA.AcceptedCount != 1 {
		t.Fatalf("reject fixture result = %+v, want AcceptedCount 1 / RejectedAt 1", resA)
	}

	// All-accepted path: the keyed bonus draw must also be slot-invariant.
	allA := runKeyed("req-abc", pTargets)
	allB := runKeyed("req-abc", pTargets)
	if !reflect.DeepEqual(allA, allB) {
		t.Fatalf("all-accepted bonus draw diverged:\n A=%+v\n B=%+v", allA, allB)
	}
	if allA.RejectedAt != -1 || allA.AcceptedCount != 3 {
		t.Fatalf("all-accepted result = %+v, want 3 accepted / RejectedAt -1", allA)
	}

	// Non-vacuity: a different request key must diverge at some bounded step.
	ksA := keyedRequest(t, NewRejectionSampler(nil), "req-abc")
	uA, okA := ksA.KeyedStepU(domAccept, 0)
	ksX := keyedRequest(t, NewRejectionSampler(nil), "req-xyz")
	uX, okX := ksX.KeyedStepU(domAccept, 0)
	if !okA || !okX {
		t.Fatal("armed sampler reported no keyed step")
	}
	if uA == uX {
		t.Fatal("two distinct request keys produced the same coin at step 0")
	}

	// An unarmed sampler must not report a keyed stream.
	if ks, ok := any(NewRejectionSampler(nil)).(keyedSampler); ok && ks.KeyedOn() {
		t.Fatal("fresh sampler reported KeyedOn() == true")
	}
	// Arming then clearing restores the legacy path.
	clr := keyedRequest(t, NewRejectionSampler(nil), "req-clear")
	if !clr.KeyedOn() {
		t.Fatal("armed sampler reported KeyedOn() == false")
	}
	clr.SetRequestKey("")
	if clr.KeyedOn() {
		t.Fatal("empty request id did not clear the key")
	}
}

// TestKeyedRequestsDoNotCorrelate proves two co-resident requests that share
// every distribution but differ only in request id draw decorrelated accept
// coins, while each individual stream stays uniform enough.
func TestKeyedRequestsDoNotCorrelate(t *testing.T) {
	const trials = 20000
	const alpha = 0.5

	draw := func(requestID string, step int) float32 {
		ks := keyedRequest(t, NewRejectionSampler(nil), requestID)
		u, ok := ks.KeyedStepU(domAccept, step)
		if !ok {
			t.Fatalf("sampler for %q reported no keyed step", requestID)
		}
		return u
	}

	if draw("r1", 0) == draw("r2", 0) {
		t.Fatal("two distinct request ids produced the same accept coin at step 0")
	}

	diffSteps := 0
	var accepts1, accepts2 int
	for step := 0; step < trials; step++ {
		u1 := draw("r1", step)
		u2 := draw("r2", step)
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
	if diffSteps < trials/2 {
		t.Fatalf("only %d/%d steps differed between request keys, want > half", diffSteps, trials)
	}

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

// TestKeyedUniformRangeAndDomains pins the structural invariants the design
// relies on: every draw lands in [0,1), and the three domains never collide on
// the same (request, step).
func TestKeyedUniformRangeAndDomains(t *testing.T) {
	for step := 0; step < 1024; step++ {
		ks := keyedRequest(t, NewRejectionSampler(nil), "req-range")
		u, ok := ks.KeyedStepU(domAccept, step)
		if !ok {
			t.Fatal("armed sampler reported no keyed step")
		}
		if u < 0 || u >= 1.0 {
			t.Fatalf("keyed draw step %d = %g, want [0,1)", step, u)
		}
	}
	for step := 0; step < 64; step++ {
		ks := keyedRequest(t, NewRejectionSampler(nil), "req-domain")
		a, _ := ks.KeyedStepU(domAccept, step)
		b, _ := ks.KeyedStepU(domBonus, step)
		c, _ := ks.KeyedStepU(domResid, step)
		if a == b || b == c || a == c {
			t.Fatalf("domain collision at step %d: accept=%g bonus=%g residual=%g", step, a, b, c)
		}
	}
}

// TestKeyedSingleTokenResidualsDoNotCollide proves the single-token keyed
// residual draw is keyed on the rejected token identity, not a fixed step 0, so
// two distinct rejections under one request do not share a residual coin. It
// also re-proves reproducibility for the single-token path.
func TestKeyedSingleTokenResidualsDoNotCollide(t *testing.T) {
	c0 := residualCoin(t, "req-resid", 0)
	c1 := residualCoin(t, "req-resid", 1)
	if c0 == c1 {
		t.Fatalf("residual replacement index for distinct rejected tokens collide: %d == %d", c0, c1)
	}
	if residualCoin(t, "req-resid", 0) != c0 {
		t.Fatal("single-token residual coin not reproducible across samplers with the same key")
	}
}

// residualCoin drives a forced rejection for the given draft token under the
// named request and returns the residual replacement draw's observable index.
func residualCoin(t *testing.T, requestID string, token int) int {
	t.Helper()
	ks := keyedRequest(t, NewRejectionSampler(nil), requestID)
	var target, draft []float32
	if token == 0 {
		target = []float32{0.0, 0.5, 0.5}
		draft = []float32{1.0, 0.0, 0.0}
	} else {
		target = []float32{0.5, 0.0, 0.5}
		draft = []float32{0.0, 1.0, 0.0}
	}
	return ks.VerifyTokenKeyedU(token, target, draft, domResid, token).ReplacementToken
}
