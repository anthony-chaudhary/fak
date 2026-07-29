package model

import (
	"encoding/binary"
	"math"
	"testing"
)

// quant_fold_witness_test.go — number-pinning witnesses for the shared helpers the
// duplication-dedup pass introduced in this package. Each test recomputes the folded
// path's result from an INDEPENDENT formulation (not by calling the helper the fold now
// shares), so it fails if a fold moved a single code, scale or index rather than merely
// moving statements.

// TestQuantizeQ8MatchesIndependentBlockReference pins quantizeQ8's Q8_0 output — every
// code and every block scale — against a reference written from the format definition
// (amax over the 32-wide block, d = amax/127, code = round-half-away(x * 1/d), all-zero
// codes when d == 0). quantizeQ8's per-block body now delegates to quantizeRowQ8scalar
// instead of carrying its own copy of that math; this test would fail on any drift
// between the two, including the d == 0 block where the old body skipped the write and
// the shared kernel writes explicit zeros.
//
// The rounding is expressed as math.Round over the float32 product, which is the
// formulation q8round documents itself equivalent to on |x| <= 127 — deliberately NOT
// q8round's own truncate-then-inspect-the-fraction trick, so the two are independent.
func TestQuantizeQ8MatchesIndependentBlockReference(t *testing.T) {
	const out, in = 3, 4 * qBlk
	w := make([]float32, out*in)
	for o := 0; o < out; o++ {
		for i := 0; i < in; i++ {
			blk := i / qBlk
			switch {
			case blk == 1 && o == 0:
				// an all-zero block: d == 0, codes must be zero
			case blk == 2:
				// exact ties at the code boundary: amax/127 * k + half a step
				w[o*in+i] = float32(o+1) * 0.5 * float32((i%qBlk)-16)
			default:
				w[o*in+i] = float32((i*37+o*11)%251-125) * 0.017
			}
		}
	}

	qt := quantizeQ8(w, out, in)
	if qt.out != out || qt.in != in || qt.nblk != in/qBlk {
		t.Fatalf("quantizeQ8 shape = (out %d, in %d, nblk %d), want (%d, %d, %d)",
			qt.out, qt.in, qt.nblk, out, in, in/qBlk)
	}
	if len(qt.q) != out*in || len(qt.d) != out*qt.nblk {
		t.Fatalf("quantizeQ8 storage = (codes %d, scales %d), want (%d, %d)",
			len(qt.q), len(qt.d), out*in, out*qt.nblk)
	}

	nonZeroBlocks := 0
	for o := 0; o < out; o++ {
		for b := 0; b < qt.nblk; b++ {
			blk := w[o*in+b*qBlk : o*in+(b+1)*qBlk]
			var amax float32
			for _, v := range blk {
				if v < 0 {
					v = -v
				}
				if v > amax {
					amax = v
				}
			}
			wantD := amax / 127
			if got := qt.d[o*qt.nblk+b]; math.Float32bits(got) != math.Float32bits(wantD) {
				t.Fatalf("row %d block %d: scale = %v (bits %#x), want %v (bits %#x)",
					o, b, got, math.Float32bits(got), wantD, math.Float32bits(wantD))
			}
			if wantD == 0 {
				for i := 0; i < qBlk; i++ {
					if got := qt.q[o*in+b*qBlk+i]; got != 0 {
						t.Fatalf("row %d block %d elem %d: zero block encoded code %d, want 0", o, b, i, got)
					}
				}
				continue
			}
			nonZeroBlocks++
			inv := float32(1.0) / wantD
			for i := 0; i < qBlk; i++ {
				prod := blk[i] * inv
				r := math.Round(float64(prod))
				if r > 127 {
					r = 127
				}
				if r < -127 {
					r = -127
				}
				if got, want := qt.q[o*in+b*qBlk+i], int8(r); got != want {
					t.Fatalf("row %d block %d elem %d: code = %d, want %d (x=%v d=%v)",
						o, b, i, got, want, blk[i], wantD)
				}
			}
		}
	}
	if nonZeroBlocks == 0 {
		t.Fatal("witness is vacuous: no non-zero block was quantized")
	}
}

// TestNewQ8TensorAllocatesZeroedStorage pins the allocation contract the Q4_K GEMM
// extraction path depends on: nblk is taken as given (the extraction re-blocks a Q4_K
// super-block into eight sub-blocks, so nblk != in/qBlk there), and both the codes and
// the scales start at zero — which is what a d == 0 block encodes, so a producer may
// skip writing one.
func TestNewQ8TensorAllocatesZeroedStorage(t *testing.T) {
	const out, in, nblk = 2, 96, 24 // nblk deliberately != in/qBlk (= 3)
	qt := newQ8Tensor(out, in, nblk)
	if qt.out != out || qt.in != in || qt.nblk != nblk {
		t.Fatalf("newQ8Tensor dims = (%d, %d, %d), want (%d, %d, %d)", qt.out, qt.in, qt.nblk, out, in, nblk)
	}
	if len(qt.q) != out*in || len(qt.d) != out*nblk {
		t.Fatalf("newQ8Tensor storage = (codes %d, scales %d), want (%d, %d)",
			len(qt.q), len(qt.d), out*in, out*nblk)
	}
	for i, c := range qt.q {
		if c != 0 {
			t.Fatalf("code %d = %d, want 0", i, c)
		}
	}
	for i, d := range qt.d {
		if d != 0 {
			t.Fatalf("scale %d = %v, want 0", i, d)
		}
	}
	if qt.accelF32 != nil {
		t.Fatalf("accelF32 = %v, want nil until q8PrepareAccelWeight runs", qt.accelF32)
	}
}

// TestDecodeGroupIdxWordsSignExtendsPerWidth pins the g_idx word decode after the width
// test was hoisted out of the element loop: a 4-byte width must read signed 32-bit words
// and an 8-byte width signed 64-bit ones, over the SAME byte buffer shapes GPTQ ships for
// I32/U32 and I64/U64. Negative words are included because the decode sign-extends, so a
// width mix-up would surface as a garbage index rather than a length error.
func TestDecodeGroupIdxWordsSignExtendsPerWidth(t *testing.T) {
	want := []int{0, 1, 7, -1, -7, 2147483647}

	b32 := make([]byte, len(want)*4)
	for i, v := range want {
		binary.LittleEndian.PutUint32(b32[i*4:], uint32(int32(v)))
	}
	got32 := make([]int, len(want))
	decodeGroupIdxWords(got32, b32, 4)
	for i := range want {
		if got32[i] != want[i] {
			t.Fatalf("width 4: out[%d] = %d, want %d", i, got32[i], want[i])
		}
	}

	b64 := make([]byte, len(want)*8)
	for i, v := range want {
		binary.LittleEndian.PutUint64(b64[i*8:], uint64(int64(v)))
	}
	got64 := make([]int, len(want))
	decodeGroupIdxWords(got64, b64, 8)
	for i := range want {
		if got64[i] != want[i] {
			t.Fatalf("width 8: out[%d] = %d, want %d", i, got64[i], want[i])
		}
	}

	// The two widths must not be interchangeable: reading the 64-bit buffer as 32-bit
	// words would interleave the high halves, so a lost width test cannot pass silently.
	strided := make([]int, len(want))
	decodeGroupIdxWords(strided, b64, 4)
	same := true
	for i := range want {
		if strided[i] != want[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("witness is vacuous: width 4 and width 8 decoded the 64-bit buffer identically")
	}
}
