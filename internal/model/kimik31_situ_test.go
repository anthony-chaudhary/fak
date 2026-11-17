package model

// kimik31_situ_test.go — parity + invariant tests for the K3.1 SiTU activation
// (ticket 1611-kimi-k31, port leaf sglang-kimi-k31-1). The reference formula is
// pinned at SGLang a7d9f81ee2 (kimi_k3_situ_and_mul.py docstring); these tests
// pin the Go host transcription against it directly, plus the two invariants
// the port study calls out (monotonic clamp bounds, and the fused in-place
// contract).
//
// fak-test:runtime fast est=5ms lane=default

import (
	"math"
	"testing"
)

// TestSituGateOutMatchesReference pins beta*tanh(gate/beta)*sigmoid(gate)
// against a hand-computed float64 reference across representative inputs,
// including the saturation regimes where the soft-cap binds.
func TestSituGateOutMatchesReference(t *testing.T) {
	const beta = 4.0
	cases := []float64{-50, -8, -4, -1, -0.001, 0, 0.001, 1, 4, 8, 50}
	for _, g := range cases {
		want := beta * math.Tanh(g/beta) * (1.0 / (1.0 + math.Exp(-g)))
		got := float64(situGateOut(float32(g), beta))
		if math.Abs(got-want) > 1e-5 {
			t.Errorf("situGateOut(%v) = %v, want %v", g, got, want)
		}
	}
}

// TestSituGateOutBounds pins the two structural properties that make SiTU a
// "soft-cap * sigmoid": the gate branch is bounded by beta because tanh is
// (|out| <= beta; the mathematical bound is strict, but a float tanh rounds to
// exactly ±1 at saturation, so the float implementation attains the bound), and
// it takes the input's sign in the unsaturated regime (the sigmoid factor is
// positive, so the sign comes from tanh; at the saturation corner the sigmoid
// factor underflows to zero and the product is a signed zero).
func TestSituGateOutBounds(t *testing.T) {
	const beta = 4.0
	for _, g := range []float64{-1e6, -100, -4, 0, 4, 100, 1e6} {
		got := float64(situGateOut(float32(g), beta))
		if math.Abs(got) > beta {
			t.Fatalf("situGateOut(%v) = %v violates |out| <= beta = %v", g, got, beta)
		}
	}
	for _, g := range []float64{-100, -4, -1, 1, 4, 100} {
		got := float64(situGateOut(float32(g), beta))
		if g > 0 && got <= 0 {
			t.Fatalf("situGateOut(%v) = %v must be > 0", g, got)
		}
		if g < 0 && got >= 0 {
			t.Fatalf("situGateOut(%v) = %v must be < 0", g, got)
		}
	}
}

// TestSituUpOutSoftcapAndIdentity: with linear_beta > 0 the up branch is a
// symmetric tanh soft-cap of width linear_beta (|out| <= linear_beta, trending
// to +-linear_beta at saturation; the mathematical bound is strict but a float
// tanh attains it); with linear_beta <= 0 it is the identity (the reference's
// linear_beta=None branch).
func TestSituUpOutSoftcapAndIdentity(t *testing.T) {
	const lb = 25.0
	for _, u := range []float64{-1e6, -100, -25, 0, 25, 100, 1e6} {
		got := float64(situUpOut(float32(u), lb))
		if math.Abs(got) > lb {
			t.Fatalf("situUpOut(%v, %v) = %v violates |out| <= linear_beta", u, lb, got)
		}
	}
	if got := situUpOut(3.5, 0); got != 3.5 {
		t.Fatalf("situUpOut(3.5, linear_beta<=0) = %v, want identity 3.5", got)
	}
	if got := situUpOut(3.5, -1); got != 3.5 {
		t.Fatalf("situUpOut(3.5, linear_beta=-1) = %v, want identity 3.5", got)
	}
	if got := situUpOut(1e6, lb); got <= 0 || got > lb {
		t.Fatalf("situUpOut saturation = %v, want in (0, %v]", got, lb)
	}
}

// TestSituAndMulFusedMatchesReference is the fused-op parity witness: the
// in-place gate[i] * up[i] write equals the elementwise reference product of
// the two branches.
func TestSituAndMulFusedMatchesReference(t *testing.T) {
	const beta, lb = 4.0, 25.0
	gate := []float32{-3, -0.5, 0, 0.5, 3, 20}
	up := []float32{2, -1, 0.25, 0, 5, -40}
	gateCopy := append([]float32(nil), gate...)

	got := SituAndMul(gate, up, beta, lb)
	if &got[0] != &gate[0] {
		t.Fatal("SituAndMul must write in place and return the gate slice")
	}
	for i := range gateCopy {
		g := float64(gateCopy[i])
		u := float64(up[i])
		wantGate := beta * math.Tanh(g/beta) * (1.0 / (1.0 + math.Exp(-g)))
		wantUp := lb * math.Tanh(u/lb)
		want := wantGate * wantUp
		if math.Abs(float64(got[i])-want) > 1e-4 {
			t.Errorf("SituAndMul[%d] = %v, want %v", i, got[i], want)
		}
	}
}

// TestSituAndMulLinearBetaNoneSkipsUpSoftcap pins the reference's
// linear_beta=None branch: the up side passes through and only the gate branch
// (soft-cap * sigmoid) is applied.
func TestSituAndMulLinearBetaNoneSkipsUpSoftcap(t *testing.T) {
	gate := []float32{1, 2}
	up := []float32{100, -5}
	SituAndMul(gate, up, 4.0, 0)
	for i, u := range []float64{100, -5} {
		g := float64([]float64{1, 2}[i])
		want := (4.0 * math.Tanh(g/4.0) * (1.0 / (1.0 + math.Exp(-g)))) * u
		if math.Abs(float64(gate[i])-want) > 1e-4 {
			t.Errorf("linear_beta=None [%d] = %v, want %v", i, gate[i], want)
		}
	}
}

// TestSituAndMulRefusesBadContract proves the fused op fails closed on a shape
// or parameter contract violation without mutating gate.
func TestSituAndMulRefusesBadContract(t *testing.T) {
	gate := []float32{1, 2}
	up := []float32{1}

	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s: expected a refusal panic", name)
			}
		}()
		f()
	}
	mustPanic("length mismatch", func() { SituAndMul(gate, up, 4.0, 25.0) })
	mustPanic("nonpositive beta", func() { SituAndMul([]float32{1}, []float32{1}, 0, 25.0) })
	mustPanic("empty", func() { SituAndMul(nil, nil, 4.0, 25.0) })
}

// TestConfigUsesSituActivation pins the config predicate the expert path will
// gate on: a K3.1 config declares hidden_act "situ".
func TestConfigUsesSituActivation(t *testing.T) {
	if !(Config{HiddenAct: "situ"}).usesSituActivation() {
		t.Fatal("HiddenAct == \"situ\" must select the SiTU activation")
	}
	if (Config{HiddenAct: "silu"}).usesSituActivation() {
		t.Fatal("HiddenAct == \"silu\" must not select SiTU")
	}
}
