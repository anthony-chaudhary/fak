//go:build amd64

package model

import (
	"math"
	"testing"
)

// TestQGemm8AsmMatchesScalar pins the AVX-512 register-blocked tile kernel (qgemm8tile512,
// driven by qGemm8) to its scalar reference (qGemm8scalar with lanes=16) BIT-FOR-BIT. The
// tile asm uses no FMA and reduces its 16 lanes in the same pairwise tree qgemm8cell emits,
// so every output must agree to the last bit. Shapes include non-multiples of the 5×4 tile
// (odd out, odd P) to exercise the scalar remainder paths the dispatcher falls back to.
//
// Unlike qdot8's bit-identity (which also ties the AVX2 tier in), this is a prefill-GEMM
// contract: the Q8 path's real correctness gate is argmax-exact + logit cosine vs f32
// (TestQuantMatchesF32Logits), since quantization is lossy. This test is the anti-asm-bug
// anchor — a wrong stride, lane, or reduction breaks it immediately.
func TestQGemm8AsmMatchesScalar(t *testing.T) {
	if !detectAVX512() {
		t.Skip("AVX-512 not available — tile kernel not exercised on this host")
	}
	if qtier != tierAVX512 {
		t.Skipf("qtier=%d (not AVX-512); qGemm8 would not use the tile asm", qtier)
	}
	type shape struct{ out, in, P int }
	shapes := []shape{
		{4, 32, 4},      // single tile, single block
		{8, 64, 8},      // 2x2 tiles, 2 blocks
		{64, 576, 16},   // clean: real q/o proj shape, multi tile-col
		{192, 576, 13},  // P not a multiple of 4 (token remainder)
		{6, 64, 7},      // out and P both non-multiples (row + token remainder)
		{1536, 576, 12}, // gate/up shape
		{576, 1536, 9},  // down shape (nblk=48), odd P
	}
	for _, s := range shapes {
		w := mkVec(s.out*s.in, uint64(s.out*s.in*2654435761+s.P*40503+1))
		qt := quantizeQ8(w, s.out, s.in)
		X := mkVec(s.P*s.in, uint64(s.P*s.in*2246822519+s.out*15485863+7))
		qp := quantizeBatchPanel(X, s.P, s.in)

		got := qGemm8(qt, qp)            // asm tile (+ scalar-ref remainder)
		want := qGemm8scalar(qt, qp, 16) // all-scalar reference, matching lane count
		if len(got) != len(want) {
			t.Fatalf("out=%d in=%d P=%d: len %d != %d", s.out, s.in, s.P, len(got), len(want))
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				tok, o := i/s.out, i%s.out
				t.Fatalf("out=%d in=%d P=%d: Y[tok=%d,o=%d]=%v (bits %#x) != scalar %v (bits %#x) — NOT bit-identical",
					s.out, s.in, s.P, tok, o, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}
}

func TestQGemm8IntoManyMatchesSeparate(t *testing.T) {
	in, P := 64, 7
	X := mkVec(P*in, 12345)
	qp := quantizeBatchPanel(X, P, in)
	targets := []struct {
		out  int
		seed uint64
	}{
		{8, 101},
		{12, 202},
		{6, 303},
	}
	gotTargets := make([]qgemm8Target, len(targets))
	want := make([][]float32, len(targets))
	for i, tg := range targets {
		qt := quantizeQ8(mkVec(tg.out*in, tg.seed), tg.out, in)
		got := make([]float32, P*tg.out)
		exp := make([]float32, P*tg.out)
		qGemm8Into(qt, qp, exp)
		gotTargets[i] = qgemm8Target{qt: qt, Y: got}
		want[i] = exp
	}

	qGemm8IntoMany(qp, gotTargets...)
	for i, tg := range gotTargets {
		for j, exp := range want[i] {
			if math.Float32bits(tg.Y[j]) != math.Float32bits(exp) {
				tok, o := j/targets[i].out, j%targets[i].out
				t.Fatalf("target %d tok=%d o=%d: got %v bits %#x, want %v bits %#x",
					i, tok, o, tg.Y[j], math.Float32bits(tg.Y[j]), exp, math.Float32bits(exp))
			}
		}
	}
}
