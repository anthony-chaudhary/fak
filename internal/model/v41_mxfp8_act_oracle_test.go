package model

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// Unwired host emulation of DeepInfer's MXFP8 routed-expert GEMM inputs: the
// bf16 FFN-norm input and the weighted SwiGLU down operand are cast to e4m3
// with one UE8M0 scale per 32 values, while the router input and the GEMM
// outputs stay bf16. It reuses fak's existing activation reference and
// measures expert-output sensitivity on a synthetic fixture only.

var errV41MXFP8Act = errors.New("model: v41 MXFP8 activation oracle refused")

func v41MXFP8ActRoundTrip(x []float32) ([]float32, error) {
	if len(x) == 0 || len(x)%32 != 0 {
		return nil, fmt.Errorf("%w: width %d", errV41MXFP8Act, len(x))
	}
	for i, v := range x {
		if !finite32(v) {
			return nil, fmt.Errorf("%w: non-finite value %d", errV41MXFP8Act, i)
		}
	}
	out, err := v41DSparkActFP8(x)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", errV41MXFP8Act, err)
	}
	return out, nil
}

type v41MXFP8ActExpert struct {
	w1, w3, w2, input []float32
	I, H              int
	cfg               Config
	weight            float32
}

// run mirrors the host routed-expert graph with optional casts at the two GEMM
// inputs. DeepInfer fuses SwiGLU, the route weight and the down-input cast, so
// the F32 weighted value is cast directly; an intermediate bf16 rounding there
// is not observable upstream.
func (e v41MXFP8ActExpert) run(castInput, castDown bool) ([]float32, error) {
	x := make([]float32, e.H)
	for i, v := range e.input {
		x[i] = v41RoundBF16(v)
	}
	var err error
	if castInput {
		if x, err = v41MXFP8ActRoundTrip(x); err != nil {
			return nil, err
		}
	}
	gate, up := matRows(e.w1, x, e.I, e.H), matRows(e.w3, x, e.I, e.H)
	for i := range gate {
		gate[i], up[i] = v41RoundBF16(gate[i]), v41RoundBF16(up[i])
	}
	clampSwiGLUProjections(gate, up, float32(e.cfg.SwigluLimit))
	h := make([]float32, e.I)
	for i := range h {
		fused := silu(gate[i]) * up[i]
		h[i] = float32(fused * e.weight)
		if !castDown {
			h[i] = v41RoundBF16(h[i])
		}
	}
	if castDown {
		if h, err = v41MXFP8ActRoundTrip(h); err != nil {
			return nil, err
		}
	}
	y := matRows(e.w2, h, e.H, e.I)
	for i := range y {
		y[i] = v41RoundBF16(y[i])
	}
	return y, nil
}

func newV41MXFP8ActExpert(seed uint64, I, H int) v41MXFP8ActExpert {
	rng := rand.New(rand.NewPCG(seed, 12640))
	vec := func(n int, scale float64) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = v41RoundBF16(float32(rng.NormFloat64() * scale))
		}
		return out
	}
	e := v41MXFP8ActExpert{
		w1: vec(I*H, 1/math.Sqrt(float64(H))), w3: vec(I*H, 1/math.Sqrt(float64(H))), w2: vec(H*I, 1/math.Sqrt(float64(I))),
		input: vec(H, 1), I: I, H: H, weight: .501953125,
		cfg: Config{HiddenSize: H, MoEIntermediateSize: I, HeadDim: v41KVLoraRank, SwigluLimit: 2},
	}
	for _, i := range []int{3, 17, 40} {
		if i < H {
			e.input[i] *= 24
		}
	}
	return e
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=5ms lane=default
func TestV41MXFP8ActRoundTripGolden(t *testing.T) {
	t.Parallel()
	tile := func(head ...float32) []float32 { return v41KVQuantOracleTile(32, head...) }
	for _, tc := range []struct {
		name     string
		in, want []float32
	}{
		{"grid_unchanged", tile(448, -448, 1, 0x1p-6, 0x1p-9), tile(448, -448, 1, 0x1p-6, 0x1p-9)},
		{"scale_exp_one", tile(896, 448, -0x1p-8), tile(896, 448, -0x1p-8)},
		{"outlier_flush", tile(7168, 0x1p-7, 0x1p-5), tile(7168, 0, 0x1p-5)},
		{"subnormal_ties", tile(448, 0x1p-10, 3*0x1p-10, -3*0x1p-10), tile(448, 0, 0x1p-8, -0x1p-8)},
		{"amax_floor", tile(v41RoundBF16(1e-6), v41RoundBF16(1e-9)), tile(0x1p-20, 0x1p-30)},
		{"caller_bf16_preround", tile(448, v41RoundBF16(1.0625001)), tile(448, 1)},
		{"no_internal_bf16", tile(448, 1.0625001), tile(448, 1.125)},
	} {
		got, err := v41MXFP8ActRoundTrip(tc.in)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		v41IndexMXFP4AssertBits(t, got, tc.want)
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=5ms lane=default
func TestV41MXFP8ActRoundTripRefusesMalformed(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		x    []float32
	}{
		{"empty", nil},
		{"width 33", make([]float32, 33)},
		{"nan", v41KVQuantOracleTile(32, float32(math.NaN()))},
		{"positive inf", v41KVQuantOracleTile(64, 0, float32(math.Inf(1)))},
		{"negative inf", v41KVQuantOracleTile(32, float32(math.Inf(-1)))},
	} {
		if _, err := v41MXFP8ActRoundTrip(tc.x); !errors.Is(err, errV41MXFP8Act) {
			t.Fatalf("%s: err=%v", tc.name, err)
		}
	}
	for _, tc := range []struct {
		I, H             int
		castIn, castDown bool
	}{{32, 48, true, false}, {48, 32, false, true}} {
		e := newV41MXFP8ActExpert(9, tc.I, tc.H)
		if _, err := e.run(tc.castIn, tc.castDown); !errors.Is(err, errV41MXFP8Act) {
			t.Fatalf("I=%d H=%d: err=%v", tc.I, tc.H, err)
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=5ms lane=default
func TestV41MXFP8ActRoundTripErrorBound(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewPCG(13, 3019))
	x := make([]float32, 4096)
	for i := range x {
		v := rng.NormFloat64() * math.Exp(3*rng.NormFloat64())
		if i/32 == 5 {
			v *= 1e-7
		}
		x[i] = v41RoundBF16(float32(v))
	}
	once, err := v41MXFP8ActRoundTrip(x)
	if err != nil {
		t.Fatal(err)
	}
	twice, err := v41MXFP8ActRoundTrip(once)
	if err != nil {
		t.Fatal(err)
	}
	v41IndexMXFP4AssertBits(t, twice, once)
	for g := 0; g < len(x); g += 32 {
		var amax float32
		for _, v := range x[g : g+32] {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		e := v41KVQuantOracleFP8DivExp(amax)
		for i := g; i < g+32; i++ {
			bound := math.Max(math.Ldexp(math.Abs(float64(x[i])), -4), math.Ldexp(1, e-10))
			if diff := math.Abs(float64(once[i]) - float64(x[i])); diff > bound {
				t.Fatalf("value %d: |%g-%g|=%g exceeds %g", i, once[i], x[i], diff, bound)
			}
			if scaled := math.Abs(math.Ldexp(float64(once[i]), -e)); scaled > 448 {
				t.Fatalf("value %d scaled magnitude %g exceeds e4m3 max", i, scaled)
			}
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=5ms lane=default
func TestV41MXFP8ActExpertSensitivity(t *testing.T) {
	t.Parallel()
	e := newV41MXFP8ActExpert(21, 32, 64)
	x := make([]float32, e.H)
	for i, v := range e.input {
		x[i] = v41RoundBF16(v)
	}
	gate, up := matRows(e.w1, x, e.I, e.H), matRows(e.w3, x, e.I, e.H)
	clampedGate, clampedUp := 0, 0
	for i := range gate {
		if v41RoundBF16(gate[i]) > float32(e.cfg.SwigluLimit) {
			clampedGate++
		}
		if math.Abs(float64(v41RoundBF16(up[i]))) > e.cfg.SwigluLimit {
			clampedUp++
		}
	}
	if clampedGate == 0 || clampedUp == 0 || clampedGate == e.I {
		t.Fatalf("SwiGLU clamp coverage gate=%d up=%d of %d", clampedGate, clampedUp, e.I)
	}

	base, err := e.run(false, false)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Cfg: e.cfg}
	production, err := m.v41FullRoutedExpert(0, "ffn.experts.0", e.input, e.weight, e.cfg, nil, nil,
		func() ([]float32, []float32, []float32, error) { return e.w1, e.w3, e.w2, nil }, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	v41IndexMXFP4AssertBits(t, production, base)

	for _, arm := range []struct {
		name             string
		castIn, castDown bool
	}{{"gate_up_input", true, false}, {"down_operand", false, true}, {"both", true, true}} {
		got, err := e.run(arm.castIn, arm.castDown)
		if err != nil {
			t.Fatal(err)
		}
		var num, den, maxAbs float64
		for i := range base {
			d := float64(got[i]) - float64(base[i])
			num += d * d
			den += float64(base[i]) * float64(base[i])
			maxAbs = math.Max(maxAbs, math.Abs(d))
		}
		rel := math.Sqrt(num / den)
		if !(rel > 0 && rel <= 0x1p-3) {
			t.Fatalf("%s: rel_l2=%g outside (0, 2^-3]", arm.name, rel)
		}
		t.Logf("SW-VERIFIED synthetic fixture v41_mxfp8_expert_rel_l2 %s=%.5f max_abs=%.5g", arm.name, rel, maxAbs)
	}
}
