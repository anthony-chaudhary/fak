package model

import (
	"math"
	"testing"
)

// TestQuantizeRowAsmMatchesScalar pins quantizeRowQ8 (the AVX-512 kernel when present) to the
// scalar reference quantizeRowQ8scalar BIT-FOR-BIT: identical codes and identical scale bits.
// The activation quantizer feeds the GEMM, so any divergence would shift the Q8 logits; this
// keeps the kernel from spending the correctness budget the FAK_QPROFILE optimization was not
// allowed to touch. Inputs deliberately include zero blocks (d==0), denormals (d underflows
// to 0), all-negative blocks, large dynamic range, and values landing exactly on the
// round-half boundary (x*inv == k+0.5), which exercises every branch of q8round.
func TestQuantizeRowAsmMatchesScalar(t *testing.T) {
	rows := [][]float32{
		mkVec(32, 12345),    // 1 block, generic
		mkVec(64, 67890),    // 2 blocks
		mkVec(576, 314159),  // q/o/gate inner dim (18 blocks)
		mkVec(1536, 271828), // down inner dim (48 blocks)
		make([]float32, 96), // 3 all-zero blocks (d==0 path)
		negRow(64, 999),     // all-negative
		denormRow(32),       // denormal -> d underflows to 0
		halfBoundaryRow(),   // values that hit x*inv == k+0.5 exactly
		mixedRow(),          // one zero block among nonzero blocks
	}
	for ri, x := range rows {
		nblk := len(x) / qBlk
		qa := make([]int8, len(x))
		da := make([]float32, nblk)
		qs := make([]int8, len(x))
		ds := make([]float32, nblk)
		quantizeRowQ8(x, qa, da, nblk)       // dispatched (asm on AVX-512)
		quantizeRowQ8scalar(x, qs, ds, nblk) // reference
		for i := range qa {
			if qa[i] != qs[i] {
				t.Fatalf("row %d code[%d]=%d != scalar %d (block %d, lane %d)", ri, i, qa[i], qs[i], i/qBlk, i%qBlk)
			}
		}
		for b := range da {
			if math.Float32bits(da[b]) != math.Float32bits(ds[b]) {
				t.Fatalf("row %d scale[%d]=%v (bits %#x) != scalar %v (bits %#x)", ri, b, da[b], math.Float32bits(da[b]), ds[b], math.Float32bits(ds[b]))
			}
		}
	}
}

// TestQuantizeVecQ8MatchesScalar pins the decode activation quantizer to the same scalar
// reference as the prefill panel quantizer. A future arm64 NEON implementation should only
// change quantizeRowQ8's dispatch target; this gate keeps the decode-facing q8Vec bit-exact.
func TestQuantizeVecQ8MatchesScalar(t *testing.T) {
	rows := [][]float32{
		mkVec(32, 1212),
		mkVec(576, 3434),
		mkVec(1536, 5656),
		make([]float32, 64),
		negRow(96, 7878),
		denormRow(32),
		halfBoundaryRow(),
		mixedRow(),
	}
	for ri, x := range rows {
		got := quantizeVecQ8(x)
		wantQ := make([]int8, len(x))
		wantD := make([]float32, len(x)/qBlk)
		quantizeRowQ8scalar(x, wantQ, wantD, len(wantD))
		if got.nblk != len(wantD) {
			t.Fatalf("row %d nblk=%d != scalar %d", ri, got.nblk, len(wantD))
		}
		for i := range got.q {
			if got.q[i] != wantQ[i] {
				t.Fatalf("row %d code[%d]=%d != scalar %d (block %d, lane %d)", ri, i, got.q[i], wantQ[i], i/qBlk, i%qBlk)
			}
		}
		for b := range got.d {
			if math.Float32bits(got.d[b]) != math.Float32bits(wantD[b]) {
				t.Fatalf("row %d scale[%d]=%v (bits %#x) != scalar %v (bits %#x)", ri, b, got.d[b], math.Float32bits(got.d[b]), wantD[b], math.Float32bits(wantD[b]))
			}
		}
	}
}

func TestQuantizeVecQ8IntoReuseMatchesScalarAndClearsZeroBlocks(t *testing.T) {
	rows := [][]float32{
		mkVec(1536, 9191), // grow to max decode scratch size first
		make([]float32, 1536),
		mkVec(576, 9292),
		make([]float32, 576),
		mixedRow(),
	}
	var scratch q8Vec
	for ri, x := range rows {
		got := quantizeVecQ8Into(&scratch, x)
		wantQ := make([]int8, len(x))
		wantD := make([]float32, len(x)/qBlk)
		quantizeRowQ8scalar(x, wantQ, wantD, len(wantD))
		if got.nblk != len(wantD) || len(got.q) != len(wantQ) || len(got.d) != len(wantD) {
			t.Fatalf("row %d shape q=%d d=%d nblk=%d, want q=%d d=%d nblk=%d",
				ri, len(got.q), len(got.d), got.nblk, len(wantQ), len(wantD), len(wantD))
		}
		for i := range got.q {
			if got.q[i] != wantQ[i] {
				t.Fatalf("row %d code[%d]=%d != scalar %d (block %d, lane %d)", ri, i, got.q[i], wantQ[i], i/qBlk, i%qBlk)
			}
		}
		for b := range got.d {
			if math.Float32bits(got.d[b]) != math.Float32bits(wantD[b]) {
				t.Fatalf("row %d scale[%d]=%v (bits %#x) != scalar %v (bits %#x)", ri, b, got.d[b], math.Float32bits(got.d[b]), wantD[b], math.Float32bits(wantD[b]))
			}
		}
	}
}

func negRow(n int, seed uint64) []float32 {
	v := mkVec(n, seed)
	for i := range v {
		v[i] = -float32(math.Abs(float64(v[i]))) - 0.01
	}
	return v
}

func denormRow(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = math.Float32frombits(uint32(i%7 + 1)) // tiny denormals; amax/127 underflows to 0
	}
	return v
}

// halfBoundaryRow builds a block whose max is 127*step so inv is a clean power-of-two-ish
// scale and several entries land exactly on k+0.5 / -(k+0.5) after x*inv.
func halfBoundaryRow() []float32 {
	v := make([]float32, 32)
	// amax = 127 so d = 1, inv = 1; then any half-integer value hits the round-half branch.
	v[0] = 127
	for i := 1; i < 32; i++ {
		v[i] = float32(i) - 64.5 // ..., -63.5, -62.5, ... exact half-integers
	}
	return v
}

func mixedRow() []float32 {
	v := mkVec(96, 4242)
	for i := 32; i < 64; i++ { // middle block all zero
		v[i] = 0
	}
	return v
}
