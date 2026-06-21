//go:build !(darwin && cgo && fakmetal)

// Witness tests for OPEN math-proof obligations on package metalgemm — the DEFAULT
// (stub, non-fakmetal) build. The build tag mirrors metalgemm_stub.go exactly, so these
// tests compile and run in precisely the build whose contract they assert and never
// collide with the fakmetal-gated tests in metalgemm_test.go.
//
// OPEN closed here:
//
//	[stub-metal-interface-parity] The stub (default) build and the metal build of package
//	metalgemm present the same exported interface, and the stub introduces no math
//	divergence — it does no GPU math but degrades cleanly to "unavailable"
//	(Available()=false, Compiled()=false, Prefill ok=false) so callers fall back to the
//	pure-Go CPU path.
//
// Method: fak/docs/proofs/00-METHOD.md. Deterministic, non-vacuous, stdlib-only.
package metalgemm

import (
	"math"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// Interface parity (compile-time): bind every exported symbol of the package to
// a value of its DOCUMENTED signature (the signatures the metal build in
// metalgemm.go declares). If the stub's exported surface drifts from that shared
// interface — a renamed func, a changed arity, a changed type — this file fails
// to COMPILE, which fails the whole package. So the assignments below are a
// machine-checked witness that the two builds present the same interface.
//
// These are package-level vars (not in a test body) so the check is total and
// not dependent on any test running.
// ---------------------------------------------------------------------------

var (
	_ func() bool                                                                            = Available
	_ func() bool                                                                            = Compiled
	_ func([]float32, int, int) *Weight                                                      = Upload
	_ func([]float32) int                                                                    = UploadVec
	_ func(int, int, int, int, int, int, float32, float32, bool)                             = FwdConfig
	_ func(int, int, int, int, int, int, int, int, int, int, int, int, int)                  = FwdLayer
	_ func(int)                                                                              = FwdFinalNorm
	_ func()                                                                                 = Reset
	_ func([]float32, int, int, int, int) ([]float32, []float32, []float32, []float32, bool) = Prefill

	// Methods on *Weight, plus the two exported fields Out, In.
	_ func(*Weight, []float32, int, []float32) = (*Weight).MatMul
	_ func(*Weight)                            = (*Weight).Free
	_ func(*Weight) int                        = (*Weight).ID
	_                                          = Weight{Out: 0, In: 0}
)

// TestStubInterfaceParity_CompilesAndScalarContract is the runtime companion to the
// package-level binding above. The bindings prove (at compile time) that the stub
// exposes the same exported interface as the metal build; this test asserts the stub's
// scalar contract — the exact "unavailable" sentinels every fallback site keys on.
func TestStubInterfaceParity_CompilesAndScalarContract(t *testing.T) {
	if Available() {
		t.Fatalf("stub Available() must be false (no GPU math in the default build); got true")
	}
	if Compiled() {
		t.Fatalf("stub Compiled() must be false (binary not built with -tags fakmetal); got true")
	}
	// Idempotent / deterministic: probing twice never changes the verdict.
	for i := 0; i < 3; i++ {
		if Available() || Compiled() {
			t.Fatalf("stub Available()/Compiled() must stay false across calls; flipped on call %d", i)
		}
	}
}

// TestStubUploadDegradesToUnavailable proves Upload never hands back a usable handle in
// the stub build — for every shape (including the empty / mismatched ones), it returns nil,
// the documented "no usable handle" signal that forces the caller onto the CPU path. A nil
// handle (not a panic, not a fake handle) is the clean-degradation contract.
func TestStubUploadDegradesToUnavailable(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	shapes := [][2]int{{1, 1}, {4, 8}, {32, 64}, {128, 256}, {0, 0}, {7, 3}}
	for _, s := range shapes {
		out, in := s[0], s[1]
		w := make([]float32, out*in)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		if got := Upload(w, out, in); got != nil {
			t.Fatalf("stub Upload(out=%d,in=%d) must return nil; got %#v", out, in, got)
		}
	}
}

// TestStubUploadVecDegradesToUnavailable proves UploadVec returns the invalid id -1 for
// every input in the stub build — the documented "no slot" sentinel.
func TestStubUploadVecDegradesToUnavailable(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range []int{0, 1, 4, 16, 4096} {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		if got := UploadVec(v); got != -1 {
			t.Fatalf("stub UploadVec(n=%d) must return -1; got %d", n, got)
		}
	}
}

// TestStubPrefillDegradesToFallback proves the central fallback contract that
// internal/model/metal_prefill.go keys on: Prefill returns ok=false and all-nil output
// slices, so `if !ok { return nil }` (then the CPU/hybrid path) is taken. This is the
// no-math-divergence guarantee — the stub produces NO hidden state, it declines, so the
// only state the model ever folds is the pure-Go CPU prefill's. Asserted over a spread of
// (P, nLayers, w, H), including degenerate shapes.
func TestStubPrefillDegradesToFallback(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	cases := [][4]int{
		{1, 1, 8, 16},
		{4, 2, 16, 64},
		{8, 4, 32, 128},
		{0, 0, 0, 0},     // degenerate
		{16, 0, 64, 256}, // zero layers
	}
	for _, c := range cases {
		P, nLayers, w, H := c[0], c[1], c[2], c[3]
		X := make([]float32, P*H)
		for i := range X {
			X[i] = float32(rng.NormFloat64()) * 5.0 // large magnitude — must still just decline
		}
		lastPre, kraw, kpost, v, ok := Prefill(X, P, nLayers, w, H)
		if ok {
			t.Fatalf("stub Prefill%v must return ok=false; got ok=true", c)
		}
		if lastPre != nil || kraw != nil || kpost != nil || v != nil {
			t.Fatalf("stub Prefill%v must return all-nil slices when ok=false; got lastPre=%v kraw=%v kpost=%v v=%v",
				c, lastPre != nil, kraw != nil, kpost != nil, v != nil)
		}
	}
}

// TestStubNoOpsAreTotalAndPure proves the remaining inert entry points are TOTAL
// (never panic) and PURE w.r.t. caller-visible state in the stub build: a *Weight from a
// stub Upload is nil, so the no-op methods must tolerate nil; MatMul must NOT write into
// the caller's output buffer (it does no math — leaving the buffer untouched is exactly
// "no math divergence"); FwdConfig/FwdLayer/FwdFinalNorm/Reset must be safe no-ops.
//
// We exercise the methods on a freshly constructed zero-value *Weight (the only handle a
// caller can ever obtain in this build is nil or a zero handle), assert ID() reports the
// invalid handle, and assert MatMul leaves a pre-filled sentinel output unchanged.
func TestStubNoOpsAreTotalAndPure(t *testing.T) {
	// Free on a nil weight must be a safe no-op (the stub's documented behavior).
	var nilW *Weight
	nilW.Free() // must not panic

	// A zero-value handle is what the stub semantically yields; ID() must be the invalid -1.
	w := &Weight{Out: 4, In: 8}
	if id := w.ID(); id != -1 {
		t.Fatalf("stub (*Weight).ID() must be -1 (invalid handle); got %d", id)
	}

	// MatMul must do NO math: a sentinel-filled output is returned byte-identical.
	const P = 3
	x := make([]float32, P*w.In)
	for i := range x {
		x[i] = float32(i) - 1.5
	}
	const sentinel = float32(123.456)
	y := make([]float32, P*w.Out)
	for i := range y {
		y[i] = sentinel
	}
	w.MatMul(x, P, y) // no-op in stub
	for i := range y {
		if y[i] != sentinel {
			t.Fatalf("stub (*Weight).MatMul must not write output (no GPU math); y[%d]=%v changed from sentinel %v",
				i, y[i], sentinel)
		}
		if math.IsNaN(float64(y[i])) || math.IsInf(float64(y[i]), 0) {
			t.Fatalf("stub MatMul produced non-finite y[%d]=%v", i, y[i])
		}
	}
	w.Free() // must not panic

	// The forward-config/registration entry points are no-ops; calling them with both
	// benign and adversarial ids must never panic.
	FwdConfig(0, 0, 0, 0, 0, 0, 0, 0, false)
	FwdConfig(32, 4096, 128, 32, 8, 11008, 1e-5, 1e6, true)
	FwdLayer(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1)
	FwdLayer(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
	FwdFinalNorm(-1)
	FwdFinalNorm(7)
	Reset()
	Reset() // idempotent: tearing down empty state twice is safe
}
