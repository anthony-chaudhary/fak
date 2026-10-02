package model

import (
	"fmt"
	"math"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model/ffn"
)

// v41_swiglu_parallel_test.go — INDEPENDENT adversarial witness for the
// DeepSeek-V4.1 routed-expert contraction `v41SwiGLU`.
//
// Contract under test (from v41_forward.go):
//
//	y = down( silu_or_gelu(clamp(w1 . x)) * clamp(w3 . x) )
//
// with w1,w3 [I,H] row-major, w2 [H,I] row-major, xn length H, y length H,
// clamps applied to both projections when cfg.SwigluLimit > 0, and activation
// chosen by cfg.ActGeluTanh / cfg.ActGeluErf (else SiLU). v41SwiGLU's output
// MUST be bit-for-bit identical (math.Float32bits) to this independently
// derived serial reference:
//
//	h1 := matRows(w1, xn, I, H); h3 := matRows(w3, xn, I, H)
//	clampSwiGLUProjections(h1, h3, float32(cfg.SwigluLimit))
//	y  := ffn.Gated(h1, h3, act(v,cfg), matRows(w2, activated, H, I))
//
// matRows, clampSwiGLUProjections, ffn.Gated, act, v41SwiGLU,
// v41SwiGLUDispatches, enableV41SwiGLUWitness, and parThreshold are all
// reachable from package `model`; this file uses only that surface.

// v41TestLCG is a deterministic float32 generator (no rng, no time) so every
// witness here is byte-reproducible across runs and hosts.
func v41TestLCG(n int, seed uint64) []float32 {
	v := make([]float32, n)
	s := seed
	for i := range v {
		s = s*6364136223846793005 + 1442695040888963407
		v[i] = float32(int64(s>>40))/float32(1<<23) - 0.5
	}
	return v
}

// v41ClampRef is the independent serial reference: it reassembles the contract
// from package primitives instead of calling v41SwiGLU.
func v41ClampRef(w1, w3, w2, xn []float32, I, H int, cfg Config) []float32 {
	h1 := matRows(w1, xn, I, H)
	h3 := matRows(w3, xn, I, H)
	clampSwiGLUProjections(h1, h3, float32(cfg.SwigluLimit))
	y, err := ffn.Gated(h1, h3, func(v float32) float32 { return act(v, cfg) }, func(activated []float32) ([]float32, error) {
		return matRows(w2, activated, H, I), nil
	})
	if err != nil {
		panic(err)
	}
	return y
}

// v41NoClampRef is the reference with the clamp deliberately omitted. It exists
// only for the guarded RED probe showing the bit-identity witness discriminates
// an unclamped body.
func v41NoClampRef(w1, w3, w2, xn []float32, I, H int, cfg Config) []float32 {
	h1 := matRows(w1, xn, I, H)
	h3 := matRows(w3, xn, I, H)
	y, err := ffn.Gated(h1, h3, func(v float32) float32 { return act(v, cfg) }, func(activated []float32) ([]float32, error) {
		return matRows(w2, activated, H, I), nil
	})
	if err != nil {
		panic(err)
	}
	return y
}

// v41AssertBitsEqual compares EVERY element bit-for-bit and fails with the
// first mismatching index, value, and bits.
func v41AssertBitsEqual(t *testing.T, tag string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: len(want)=%d len(got)=%d", tag, len(want), len(got))
	}
	for i := range want {
		wb, gb := math.Float32bits(want[i]), math.Float32bits(got[i])
		if wb != gb {
			t.Fatalf("%s: element %d: reference %v (bits %#08x) != under-test %v (bits %#08x)",
				tag, i, want[i], wb, got[i], gb)
		}
	}
}

// TestV41SwiGLUIndependentBitIdentity is the load-bearing witness: across
// shapes {1,1},{7,3},{192,576},{576,576} (below and above parThreshold) x
// activations {SiLU, ActGeluTanh, ActGeluErf} x SwigluLimit {0, 2.5}, every
// output element must be bit-identical to the serial reference computed from
// FRESH, independent input copies.
//
// fak-test:runtime fast est=250ms lane=short
func TestV41SwiGLUIndependentBitIdentity(t *testing.T) {
	shapes := []struct{ I, H int }{
		{1, 1},     // tiny: far below parThreshold
		{7, 3},     // small: below parThreshold
		{192, 576}, // 110592 work units: above parThreshold (1<<16)
		{576, 576}, // 331776 work units: above parThreshold
	}
	policies := []struct {
		name string
		cfg  Config
	}{
		{"silu", Config{}},
		{"gelu_tanh", Config{ActGeluTanh: true}},
		{"gelu_erf", Config{ActGeluErf: true}},
	}
	limits := []float64{0, 2.5}

	// Both kernels of the fak#13294 seam are covered: the package-level serial
	// reference (v41SwiGLU, which every adapter witness compares against) and the
	// bulk-prefill row-parallel twin (v41SwiGLUParallel) the MoE contraction now
	// calls for a multi-token panel.
	impls := []struct {
		name string
		run  func(w1, w3, w2, xn []float32, I, H int, cfg Config) []float32
	}{
		{"serial", v41SwiGLU},
		{"parallel", v41SwiGLUParallel},
	}

	for _, sh := range shapes {
		for _, pol := range policies {
			for _, lim := range limits {
				I, H := sh.I, sh.H
				cfg := pol.cfg
				cfg.SwigluLimit = lim
				tag := fmt.Sprintf("%s/lim=%v I=%d H=%d", pol.name, lim, I, H)

				// Two wholly independent input sets: one for the reference, one
				// for the under-test call. The reference cannot observe or share
				// the under-test buffer.
				refW1 := v41TestLCG(I*H, uint64(I*131+H+1))
				refW3 := v41TestLCG(I*H, uint64(I*977+H*7+2))
				refW2 := v41TestLCG(H*I, uint64(H*53+I*3+3))
				refX := v41TestLCG(H, 101)

				want := v41ClampRef(refW1, refW3, refW2, refX, I, H, cfg)
				if len(want) != H {
					t.Fatalf("%s: reference len=%d, want H=%d", tag, len(want), H)
				}

				for _, impl := range impls {
					gotW1 := v41TestLCG(I*H, uint64(I*131+H+1))
					gotW3 := v41TestLCG(I*H, uint64(I*977+H*7+2))
					gotW2 := v41TestLCG(H*I, uint64(H*53+I*3+3))
					gotX := v41TestLCG(H, 101)
					got := impl.run(gotW1, gotW3, gotW2, gotX, I, H, cfg)
					if len(got) != H {
						t.Fatalf("%s/%s: len(got)=%d, want H=%d", impl.name, tag, len(got), H)
					}
					v41AssertBitsEqual(t, impl.name+"/"+tag, want, got)
				}
			}
		}
	}
}

// TestV41SwiGLUIndependentClamp pins the asymmetric clamp semantics: the gate
// is upper-clamped only; the up branch is clamped on BOTH bounds. Weights of
// magnitude 1e6 make an unclamped path produce visibly different bits.
//
// fak-test:runtime fast est=40ms lane=short
func TestV41SwiGLUIndependentClamp(t *testing.T) {
	const L = 2.0
	cfg := Config{SwigluLimit: L}
	// I=H=1: raw gate = w1[0]*x, raw up = w3[0]*x, y = w2[0]*act(gate)*up.
	cases := []struct {
		name       string
		w1, w3, x  float32
		expGate    float32 // gate after clamp
		expUp      float32 // up after clamp
		clampBinds bool
	}{
		{"gate_upper", 1e6, 1.0, 1.0, L, 1.0, true},         // +1e6 -> +L
		{"up_lower", 1.0, -1e6, 1.0, 1.0, -L, true},         // -1e6 -> -L
		{"up_upper", 1.0, 1e6, 1.0, 1.0, L, true},           // +1e6 -> +L
		{"gate_no_lower", -1e6, 1.0, 1.0, -1e6, 1.0, false}, // lower bound NOT applied
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			w1 := []float32{c.w1}
			w3 := []float32{c.w3}
			w2 := []float32{1.0}
			xn := []float32{c.x}

			got := v41SwiGLUParallel(append([]float32{}, w1...), append([]float32{}, w3...),
				append([]float32{}, w2...), append([]float32{}, xn...), 1, 1, cfg)

			// Independently compute the explicitly-clamped expected output.
			exp := matRows([]float32{1.0}, []float32{act(c.expGate, cfg) * c.expUp}, 1, 1)
			v41AssertBitsEqual(t, "clamp/"+c.name, exp, got)

			if c.clampBinds {
				unclamped := act(c.w1*c.x, cfg) * (c.w3 * c.x)
				if math.Float32bits(unclamped) == math.Float32bits(got[0]) {
					t.Fatalf("clamp/%s: clamped and unclamped outputs share bits %#08x; clamp is unobservable",
						c.name, math.Float32bits(got[0]))
				}
			}
		})
	}
}

// TestV41SwiGLUIndependentMutationAndIdempotence asserts the reference is built
// from fresh, non-aliasing copies (so the test cannot pass by comparing a
// buffer with itself) and that two fresh-input calls return identical bits:
// v41SwiGLU mutates only its internal gate scratch, never the caller's weights.
//
// fak-test:runtime fast est=120ms lane=short
func TestV41SwiGLUIndependentMutationAndIdempotence(t *testing.T) {
	const I, H = 192, 576 // above parThreshold: exercises the parallel path
	cfg := Config{SwigluLimit: 1.5}

	w1a := v41TestLCG(I*H, 7001)
	w3a := v41TestLCG(I*H, 7002)
	w2a := v41TestLCG(H*I, 7003)
	xa := v41TestLCG(H, 7004)

	w1b := append([]float32{}, w1a...)
	w3b := append([]float32{}, w3a...)
	w2b := append([]float32{}, w2a...)
	xb := append([]float32{}, xa...)

	first := v41SwiGLUParallel(w1a, w3a, w2a, xa, I, H, cfg)
	second := v41SwiGLUParallel(w1b, w3b, w2b, xb, I, H, cfg)
	v41AssertBitsEqual(t, "idempotence/fresh-inputs", first, second)

	if len(first) > 0 && len(second) > 0 && &first[0] == &second[0] {
		t.Fatal("first and second outputs alias one backing array; idempotence is vacuous")
	}

	// A third, independently generated reference must match both.
	refW1 := append([]float32{}, w1a...)
	refW3 := append([]float32{}, w3a...)
	refW2 := append([]float32{}, w2a...)
	refX := append([]float32{}, xa...)
	ref := v41ClampRef(refW1, refW3, refW2, refX, I, H, cfg)
	v41AssertBitsEqual(t, "reference/fresh-inputs", ref, first)
	if len(ref) > 0 && &ref[0] == &first[0] {
		t.Fatal("reference and under-test share a backing array; bit-identity is vacuous")
	}
}

// TestV41SwiGLUIndependentShapeRobustness asserts len(y)==H for every covered
// shape; a wrong length must fail rather than silently truncate.
//
// fak-test:runtime fast est=120ms lane=short
func TestV41SwiGLUIndependentShapeRobustness(t *testing.T) {
	shapes := []struct{ I, H int }{{1, 1}, {7, 3}, {192, 576}, {576, 576}}
	impls := []struct {
		name string
		run  func(w1, w3, w2, xn []float32, I, H int, cfg Config) []float32
	}{
		{"serial", v41SwiGLU},
		{"parallel", v41SwiGLUParallel},
	}
	for _, sh := range shapes {
		I, H := sh.I, sh.H
		for _, impl := range impls {
			w1 := v41TestLCG(I*H, 81)
			w3 := v41TestLCG(I*H, 82)
			w2 := v41TestLCG(H*I, 83)
			xn := v41TestLCG(H, 84)
			got := impl.run(w1, w3, w2, xn, I, H, Config{})
			if len(got) != H {
				t.Fatalf("%s I=%d H=%d: len(y)=%d, want H=%d", impl.name, I, H, len(got), H)
			}
		}
	}
}

// TestV41SwiGLUIndependentDegenerate pins the degenerate shapes. I==0 yields
// empty gate/up, which ffn.Gated rejects, and v41SwiGLU panics on that error;
// H==0 yields an empty down projection and a length-0 result. recover() asserts
// the panic instead of crashing the suite.
//
// fak-test:runtime fast est=20ms lane=short
func TestV41SwiGLUIndependentDegenerate(t *testing.T) {
	t.Run("I_zero_panics", func(t *testing.T) {
		for name, run := range map[string]func() []float32{
			"serial": func() []float32 {
				return v41SwiGLU([]float32{}, []float32{}, []float32{}, []float32{1, 2, 3}, 0, 3, Config{})
			},
			"parallel": func() []float32 {
				return v41SwiGLUParallel([]float32{}, []float32{}, []float32{}, []float32{1, 2, 3}, 0, 3, Config{})
			},
		} {
			func() {
				defer func() {
					if recover() == nil {
						t.Errorf("%s I==0: expected panic from ffn.Gated empty-projection rejection", name)
					}
				}()
				run()
			}()
		}
	})

	t.Run("H_zero_returns_empty", func(t *testing.T) {
		for name, run := range map[string]func() []float32{
			"serial": func() []float32 { return v41SwiGLU([]float32{}, []float32{}, []float32{}, []float32{}, 3, 0, Config{}) },
			"parallel": func() []float32 {
				return v41SwiGLUParallel([]float32{}, []float32{}, []float32{}, []float32{}, 3, 0, Config{})
			},
		} {
			if y := run(); len(y) != 0 {
				t.Errorf("%s H==0: len(y)=%d, want 0", name, len(y))
			}
		}
	})
}

// TestV41SwiGLUIndependentWitnessCounter probes the package-level dispatch
// witness. Witness OFF must leave the counter at zero (production posture);
// witness ON for a contraction at/above parThreshold must advance it by one.
// It restores the OFF posture on exit so order does not leak.
//
// fak-test:runtime fast est=120ms lane=short
func TestV41SwiGLUIndependentWitnessCounter(t *testing.T) {
	const I, H = 576, 576
	if I*H < parThreshold {
		t.Fatalf("shape %dx%d is below parThreshold=%d; dispatch arm would be vacuous", I, H, parThreshold)
	}
	defer func() { v41SwiGLUWitnessOn = false }()

	w1 := v41TestLCG(I*H, 11)
	w3 := v41TestLCG(I*H, 22)
	w2 := v41TestLCG(H*I, 33)
	xn := v41TestLCG(H, 44)

	// Arm 1: witness OFF must stay at zero.
	v41SwiGLUWitnessOn = false
	atomic.StoreInt64(&v41SwiGLUParallelCalls, 0)
	v41SwiGLU(w1, w3, w2, xn, I, H, Config{})
	if n := v41SwiGLUDispatches(); n != 0 {
		t.Fatalf("witness off: dispatches=%d, want 0", n)
	}

	// Arm 2: witness ON must observe exactly one large contraction.
	enableV41SwiGLUWitness()
	v41SwiGLUParallel(w1, w3, w2, xn, I, H, Config{})
	if n := v41SwiGLUDispatches(); n != 1 {
		t.Fatalf("witness on: dispatches=%d, want 1 (large contraction not counted)", n)
	}

	// Arm 3: a below-threshold contraction must NOT advance the counter.
	enableV41SwiGLUWitness()
	v41SwiGLUParallel([]float32{1}, []float32{1}, []float32{1}, []float32{1}, 1, 1, Config{})
	if n := v41SwiGLUDispatches(); n != 0 {
		t.Fatalf("witness on/below threshold: dispatches=%d, want 0", n)
	}
}

// TestV41SwiGLUIndependentRedProbe shows the bit-identity witness can FAIL: a
// deliberately wrong (unclamped) reference must NOT match v41SwiGLU when a
// clamp binds. The check is guarded so the suite stays green.
//
// fak-test:runtime fast est=40ms lane=short
func TestV41SwiGLUIndependentRedProbe(t *testing.T) {
	cfg := Config{SwigluLimit: 0.5}
	I, H := 4, 4
	// Same-sign large-magnitude weights so each raw projection sums to a large
	// nonzero value (a mixed-sign row would cancel to 0 and never bind the clamp).
	w1 := []float32{1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6}
	w3 := []float32{1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6}
	w2 := v41TestLCG(H*I, 9090)
	xn := []float32{1, 1, 1, 1}

	got := v41SwiGLUParallel(append([]float32{}, w1...), append([]float32{}, w3...),
		append([]float32{}, w2...), append([]float32{}, xn...), I, H, cfg)
	good := v41ClampRef(w1, w3, w2, xn, I, H, cfg)
	wrong := v41NoClampRef(w1, w3, w2, xn, I, H, cfg)

	// Positive control: the correct reference matches.
	v41AssertBitsEqual(t, "red-probe/correct-reference", good, got)

	// Negative control: the unclamped reference must differ somewhere.
	same := true
	for i := range wrong {
		if math.Float32bits(wrong[i]) != math.Float32bits(got[i]) {
			same = false
			break
		}
	}
	if same {
		t.Fatal("RED probe failed to discriminate: unclamped reference matched the clamped output; " +
			"the clamp assertion is not load-bearing")
	}
}
