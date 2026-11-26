//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"math/rand"
	"slices"
	"testing"
	"time"
)

// The key-split attention matches the single-pass kernel at the shipped
// geometry, short and long context, including a per-row window cut.
//
// fak-test:runtime integration est=5s lane=default
func TestInk2DraftAttendSplitMatchesSingle(t *testing.T) {
	d := ink2TestDevice(t)
	st := d.dfState()
	defer func() { st.split = 0 }()
	rng := rand.New(rand.NewSource(19))
	const rows, heads, kvHeads, dim, window, cap = 8, 32, 8, 128, 2048, 2048 + 16
	kvw := kvHeads * dim
	qv, kbv, vbv := ink2ConvTestRand(rng, rows*heads*dim, 1), ink2ConvTestRand(rng, rows*kvw, 1), ink2ConvTestRand(rng, rows*kvw, 1)
	kcv, vcv := ink2ConvTestRand(rng, cap*kvw, 1), ink2ConvTestRand(rng, cap*kvw, 1)
	q, kb, vb := ink2ConvTestPut(t, d, 0, qv), ink2ConvTestPut(t, d, 0, kbv), ink2ConvTestPut(t, d, 0, vbv)
	kc, vc := ink2ConvTestPut(t, d, 0, kcv), ink2ConvTestPut(t, d, 0, vcv)
	single, split := ink2TestAlloc(t, d, rows*heads*dim), ink2TestAlloc(t, d, rows*heads*dim)
	for _, ctx := range []int{64, 160, 512, 2048, 2100, 4000} {
		posDev := ink2TestAttendUpload(t, d, ink2TestAttendMeta(ctx))
		run := func(mode int8, out Ink2Ref) error {
			st.split = mode
			return d.DraftAttend(q, kb, vb, kc, vc, out, posDev, rows, heads, kvHeads, dim, window, cap)
		}
		tSingle := ink2DFTestGPU(t, d, func() error { return run(-1, single) })
		tSplit := ink2DFTestGPU(t, d, func() error { return run(1, split) })
		want, got := ink2ConvTestRead(t, d, single, rows*heads*dim), ink2ConvTestRead(t, d, split, rows*heads*dim)
		_, mag := ink2DFTestAttendHost(qv, kbv, vbv, kcv, vcv, ctx, rows, heads, kvHeads, dim, window, cap)
		// Relative to the magnitude of the value terms each output sums.
		var worst float64
		for i := range want {
			worst = max(worst, math.Abs(float64(got[i]-want[i]))/mag[i])
		}
		if worst > 1e-5 {
			t.Errorf("context %d: worst difference %.3g of the term magnitude, want <= 1e-5", ctx, worst)
		}
		t.Logf("context %d: single %v, split %v, worst difference %.3g of the term magnitude", ctx, tSingle, tSplit, worst)
	}
}

// ink2DFTestQ8Host quantizes a held BF16 weight as ink2_df_q8_quantize does
// and returns the dequantized values.
func ink2DFTestQ8Host(w []float32, rows, cols int) []float32 {
	out := make([]float32, len(w))
	for r := range rows {
		var amax float32
		for _, v := range w[r*cols : (r+1)*cols] {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		var inv float32
		if amax > 0 {
			inv = 127 / amax
		}
		scale := amax / 127
		for c := range cols {
			qv := math.Max(-127, math.Min(127, math.RoundToEven(float64(w[r*cols+c]*inv))))
			out[r*cols+c] = float32(qv) * scale
		}
	}
	return out
}

// The int8 draft linears compute with the per-row quantized weight, and the
// fused ones write what the separate int8 launches write.
//
// fak-test:runtime integration est=3s lane=default
func TestInk2DraftQ8LinearsMatchQuantizedHost(t *testing.T) {
	d := ink2TestDevice(t)
	st := d.dfState()
	st.q8 = 1
	defer func() { st.q8 = 0 }()
	rng := rand.New(rand.NewSource(23))
	const cols = 512
	for _, tokens := range []int{1, 3, 8} {
		xv := ink2ConvTestRand(rng, tokens*cols, 1)
		x := ink2ConvTestPut(t, d, 0, xv)
		shapes := []int{70, 16, 33}
		var ws []Ink2Weight
		var sep, fused []Ink2Ref
		for _, r := range shapes {
			w, held := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, r*cols, .05), r, cols, 0)
			ws = append(ws, w)
			a, b := ink2TestAlloc(t, d, tokens*r), ink2TestAlloc(t, d, tokens*r)
			if err := d.DraftLinear([]Ink2Ref{x}, w, a, tokens); err != nil {
				t.Fatal(err)
			}
			want, mag := ink2DFTestLinearHost([][]float32{xv}, ink2DFTestQ8Host(held, r, cols), r, cols, tokens)
			got := ink2ConvTestRead(t, d, a, tokens*r)
			for i := range want {
				if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-5*mag[i]+1e-7 {
					t.Fatalf("%d tokens, %d rows: element %d = %v, want %v (terms %.3g)", tokens, r, i, got[i], want[i], mag[i])
				}
			}
			sep, fused = append(sep, a), append(fused, b)
		}
		if err := d.DraftLinearMulti([]Ink2Ref{x}, ws, fused, tokens); err != nil {
			t.Fatal(err)
		}
		for i := range ws {
			if !slices.Equal(ink2ConvTestRead(t, d, fused[i], tokens*shapes[i]), ink2ConvTestRead(t, d, sep[i], tokens*shapes[i])) {
				t.Fatalf("%d tokens: int8 multi output %d differs from int8 DraftLinear", tokens, i)
			}
		}
		up, _ := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, shapes[0]*cols, .05), shapes[0], cols, 0)
		u, g := ink2TestAlloc(t, d, tokens*shapes[0]), ink2TestAlloc(t, d, tokens*shapes[0])
		if err := d.DraftLinear([]Ink2Ref{x}, up, u, tokens); err != nil {
			t.Fatal(err)
		}
		if err := d.DraftSiluMul(sep[0], u, sep[0], tokens*shapes[0]); err != nil {
			t.Fatal(err)
		}
		if err := d.DraftLinearSiluMul(x, ws[0], up, g, Ink2Ref{}, tokens); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(ink2ConvTestRead(t, d, g, tokens*shapes[0]), ink2ConvTestRead(t, d, sep[0], tokens*shapes[0])) {
			t.Fatalf("%d tokens: int8 gated linear differs from int8 DraftLinear and DraftSiluMul", tokens)
		}
	}
}

// TestInk2DraftQ8GPUTime times the int8 and BF16 draft linears at the shipped
// fc, q/k/v and gate/up shapes for a block of 8 rows.
//
// fak-test:runtime integration est=10s lane=default
func TestInk2DraftQ8GPUTime(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates draft-sized weights")
	}
	d := ink2TestDevice(t)
	st := d.dfState()
	defer func() { st.q8 = 0 }()
	rng := rand.New(rand.NewSource(29))
	const rows, hidden, inter, qw, kvw = 8, 2304, 6912, 4096, 1024
	taps := make([]Ink2Ref, 4)
	for i := range taps {
		taps[i] = ink2ConvTestPut(t, d, 0, ink2ConvTestRand(rng, rows*hidden, 1))
	}
	weight := func(r, c int) Ink2Weight {
		w, _ := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, r*c, .05), r, c, 0)
		return w
	}
	fc, wq, wk, wv, gate, up := weight(hidden, 4*hidden), weight(qw, hidden), weight(kvw, hidden), weight(kvw, hidden), weight(inter, hidden), weight(inter, hidden)
	fused, yq, yk, yv, g := ink2TestAlloc(t, d, rows*hidden), ink2TestAlloc(t, d, rows*qw), ink2TestAlloc(t, d, rows*kvw), ink2TestAlloc(t, d, rows*kvw), ink2TestAlloc(t, d, rows*inter)
	// Alternate the modes so shared-GPU load hits both alike; keep each best.
	best := map[int8]*[3]time.Duration{-1: {}, 1: {}}
	for round := range 4 {
		for _, mode := range []int8{-1, 1} {
			st.q8 = mode
			times := [3]time.Duration{
				ink2DFTestGPU(t, d, func() error { return d.DraftLinear(taps, fc, fused, rows) }),
				ink2DFTestGPU(t, d, func() error {
					return d.DraftLinearMulti(taps[:1], []Ink2Weight{wq, wk, wv}, []Ink2Ref{yq, yk, yv}, rows)
				}),
				ink2DFTestGPU(t, d, func() error { return d.DraftLinearSiluMul(taps[0], gate, up, g, Ink2Ref{}, rows) }),
			}
			for i, v := range times {
				if round == 0 || v < best[mode][i] {
					best[mode][i] = v
				}
			}
		}
	}
	for _, mode := range []int8{-1, 1} {
		b := best[mode]
		t.Logf("int8 %v: fc %dx%d %v, qkv %v, gate+up silu %v", mode > 0, hidden, 4*hidden, b[0], b[1], b[2])
	}
}
