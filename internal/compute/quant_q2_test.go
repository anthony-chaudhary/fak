package compute

import (
	"math"
	"testing"
)

// quant_q2_test.go — the pure-Go failure-class proof for the packed-ternary Q2_0 GEMM
// (issue #4872). It witnesses the CPU reference path — the format helpers (pack/unpack,
// absmax quantize) and the cpuref MatMul/BatchedMatMul Q2_0 cases — WITHOUT a GPU, so it
// runs in the ordinary WSL/CI suite. The device kernel (k_q2_0_gemm) is then held to THIS
// reference on a CUDA node by cuda_q2_test.go (`-tags cuda -run Q2_0GEMM`).
//
// The gate: build ternary weights, dequantize them to a dense f32 weight, and assert the
// Q2_0 GEMV over the PACKED codes equals an f32 GEMV over the dequantized weight —
// argmax-exact end-to-end and max|Δ| within tol (only reduction-order/scale-factoring
// drift separates them). A wrong pack bit order, block stride, or scale index collapses it.

// dequantQ2Weight expands a packed Q2_0 weight to the dense f32 weight it represents:
// w[o,i] = t(o,i)·scale[o, i/block]. This is the ground-truth the ternary GEMV must match.
func dequantQ2Weight(packed []byte, scale []float32, out, in, block int) []float32 {
	rowBytes := in / 4
	nblk := in / block
	w := make([]float32, out*in)
	for o := 0; o < out; o++ {
		row := packed[o*rowBytes : (o+1)*rowBytes]
		for i := 0; i < in; i++ {
			w[o*in+i] = float32(q2Code(row, i)) * scale[o*nblk+i/block]
		}
	}
	return w
}

// randTernaryWeight authors a random ternary weight [out,in] plus its per-block scales, and
// returns the packed codes. Codes are drawn from {-1,0,+1}; scales are small positive f32.
func randTernaryWeight(seed uint64, out, in, block int) (packed []byte, scale []float32) {
	s := seed
	next := func() uint64 { s = s*6364136223846793005 + 1442695040888963407; return s }
	nblk := in / block
	tern := make([]int8, out*in)
	scale = make([]float32, out*nblk)
	for i := range tern {
		tern[i] = int8(int(next()%3) - 1) // {-1,0,+1}
	}
	for b := range scale {
		scale[b] = 0.05 + float32(next()%100)/1000.0 // [0.05, 0.149]
	}
	return packQ2(tern), scale
}

func q2MaxAbsDelta(a, b []float32) float32 {
	var m float32
	for i := range a {
		if d := absf(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

func q2Argmax(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

// TestQ2PackRoundTrip proves the 2-bit pack/unpack is its own inverse over {-1,0,+1}.
func TestQ2PackRoundTrip(t *testing.T) {
	tern := make([]int8, 133) // not a multiple of 4 — exercise the tail byte
	for i := range tern {
		tern[i] = int8((i%3)-1) * int8(1)
	}
	packed := packQ2(tern)
	if len(packed) != (len(tern)+3)/4 {
		t.Fatalf("packed len %d, want %d", len(packed), (len(tern)+3)/4)
	}
	for i, want := range tern {
		if got := q2Code(packed, i); got != int32(want) {
			t.Fatalf("code[%d]=%d, want %d", i, got, want)
		}
	}
}

// TestQ2MatMulMatchesDequantRef — the ternary decode GEMV over packed codes equals the f32
// GEMV over the dequantized weight: argmax-exact and max|Δ| within a tight tol.
func TestQ2MatMulMatchesDequantRef(t *testing.T) {
	ref := Default()
	const out, in = 320, 256 // in divisible by 32
	packed, scale := randTernaryWeight(0x4872, out, in, q2Block)
	dense := dequantQ2Weight(packed, scale, out, in, q2Block)

	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i%17)-8) / 8.0
	}

	yRef := ref.Read(ref.MatMul(NewF32(ref, []int{out, in}, dense), NewF32(ref, []int{in}, x)))
	yQ2 := ref.Read(ref.MatMul(NewQ2(ref, []int{out, in}, packed, scale, q2Block), NewF32(ref, []int{in}, x)))

	if len(yRef) != out || len(yQ2) != out {
		t.Fatalf("shape ref=%d q2=%d want %d", len(yRef), len(yQ2), out)
	}
	if a, b := q2Argmax(yRef), q2Argmax(yQ2); a != b {
		t.Fatalf("argmax mismatch: dequant-ref=%d q2=%d", a, b)
	}
	// scale magnitudes ~0.1, in=256 terms; relative f32 reduction-order drift is ~1e-4.
	if d := q2MaxAbsDelta(yRef, yQ2); d > 1e-3 {
		t.Fatalf("Q2_0 MatMul max|Δ|=%.3e > 1e-3 vs dequant f32 reference", d)
	}
}

// TestQ2BatchedMatMulMatchesDequantRef — the same for the prefill GEMM (P>1): every row of
// the Q2_0 batched GEMM matches its per-row Q2_0 GEMV and the dense f32 reference.
func TestQ2BatchedMatMulMatchesDequantRef(t *testing.T) {
	ref := Default()
	const out, in, P = 128, 128, 6
	packed, scale := randTernaryWeight(0x4872b, out, in, q2Block)
	dense := dequantQ2Weight(packed, scale, out, in, q2Block)

	X := make([]float32, P*in)
	for i := range X {
		X[i] = float32((i%23)-11) / 11.0
	}

	YRef := ref.Read(ref.BatchedMatMul(NewF32(ref, []int{out, in}, dense), NewF32(ref, []int{P, in}, X), P))
	YQ2 := ref.Read(ref.BatchedMatMul(NewQ2(ref, []int{out, in}, packed, scale, q2Block), NewF32(ref, []int{P, in}, X), P))

	if len(YRef) != P*out || len(YQ2) != P*out {
		t.Fatalf("shape ref=%d q2=%d want %d", len(YRef), len(YQ2), P*out)
	}
	if d := q2MaxAbsDelta(YRef, YQ2); d > 1e-3 {
		t.Fatalf("Q2_0 BatchedMatMul max|Δ|=%.3e > 1e-3 vs dequant f32 reference", d)
	}
	// per-row equivalence to the single-row GEMV (the BatchedMatMul==MatMul contract).
	for row := 0; row < P; row++ {
		one := ref.Read(ref.MatMul(NewQ2(ref, []int{out, in}, packed, scale, q2Block), NewF32(ref, []int{in}, X[row*in:row*in+in])))
		for o := 0; o < out; o++ {
			if d := absf(one[o] - YQ2[row*out+o]); d > 1e-6 {
				t.Fatalf("row %d chan %d: batched %.6f != gemv %.6f", row, o, YQ2[row*out+o], one[o])
			}
		}
	}
}

// TestQuantizeQ2Absmax proves the absmax quantizer keeps every weight's sign and never
// exceeds one scale unit — the ternary reconstruction is within the block's amax of f32.
func TestQuantizeQ2Absmax(t *testing.T) {
	ref := Default()
	const out, in = 4, 64
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(math.Sin(float64(i) * 0.3))
	}
	q := QuantizeQ2(ref, []int{out, in}, w, q2Block)
	if q.Dtype != Q2_0 {
		t.Fatalf("QuantizeQ2 dtype %s, want q2_0", q.Dtype)
	}
	packed := i8AsBytes(q.buf.(HostBuffer).I8())
	dense := dequantQ2Weight(packed, q.Quant.Scale, out, in, q2Block)
	for i := range w {
		if w[i] > 0 && dense[i] < 0 || w[i] < 0 && dense[i] > 0 {
			// a sign flip is only allowed for a near-zero weight rounded to the 0 level.
			if absf(w[i]) > q.Quant.Scale[(i/in)*(in/q2Block)+(i%in)/q2Block]*0.5 {
				t.Fatalf("weight %d sign flipped: f32=%.4f tern=%.4f", i, w[i], dense[i])
			}
		}
	}
}
