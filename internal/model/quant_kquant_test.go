package model

import (
	"encoding/binary"
	"testing"
)

// lcgBytes fills b with a deterministic LCG byte stream (no math/rand dependency / seed flake).
func lcgBytes(b []byte, seed uint64) {
	s := seed
	for i := range b {
		s = s*6364136223846793005 + 1442695040888963407
		b[i] = byte(s >> 33)
	}
}

// f16One is the IEEE-754 half-precision encoding of 1.0 (0x3C00), written little-endian into a
// super-block's scale field so random code bytes decode to FINITE values (a random f16 d could be
// inf/NaN and break the bit-exact compare).
const f16One = 0x3C00

// refKQuantMatRows is the reference resident-k-quant GEMV: dequant each super-block via the SAME
// per-block routine, materialize the full f32 row, then the SAME fixed-order 4-accumulator dot.
// kQuantMatRows must equal this BIT-FOR-BIT — it proves the streaming GEMV wrapper (and its row
// parallelism) is arithmetically identical to a dequant-then-dot.
func refKQuantMatRows(qt *kQuantTensor, x []float32) []float32 {
	bb := qt.kind.blockBytes()
	blockWeights := qt.kind.blockWeights()
	y := make([]float32, qt.out)
	buf := make([]float32, blockWeights)
	for o := 0; o < qt.out; o++ {
		row := qt.raw[o*qt.rowBytes():]
		var acc float32
		for b := 0; b < qt.nblk; b++ {
			kQuantDequantSuperBlock(buf, row[b*bb:(b+1)*bb], qt.kind)
			xs := x[b*blockWeights:]
			var s0, s1, s2, s3 float32
			for i := 0; i < blockWeights; i += 4 {
				s0 += buf[i] * xs[i]
				s1 += buf[i+1] * xs[i+1]
				s2 += buf[i+2] * xs[i+2]
				s3 += buf[i+3] * xs[i+3]
			}
			acc += (s0 + s1) + (s2 + s3)
		}
		y[o] = acc
	}
	return y
}

func TestKQuantMatRowsMatchesDequantRef(t *testing.T) {
	// This asserts kQuantMatRows is BIT-IDENTICAL to dequant-then-dot, which only the f32 GEMV is —
	// the int8 Q5_K path is approximate (activation quantization). Pin the f32 path so a production
	// FAK_KQ_INT8=1 in the env does not flip kQuantMatRows to int8 and muddy this bit-identity check
	// (TestQ5KInt8MatchesF32 covers the int8 path under its own cosine gate).
	setKQuantSDOTForTest(false)
	t.Cleanup(func() { kQuantSDOTForce = 0 })
	const (
		out = 9   // odd, to exercise the parallel row split's tail
		in  = 512 // 2 super-blocks per row
	)
	for _, tc := range []struct {
		name string
		kind kQuantKind
	}{{"Q5_K", kindQ5K}, {"Q6_K", kindQ6K}, {"IQ3_XXS", kindIQ3XXS}, {"IQ4_XS", kindIQ4XS}, {"Q8_0", kindQ8_0}} {
		t.Run(tc.name, func(t *testing.T) {
			nblk := in / tc.kind.blockWeights()
			bb := tc.kind.blockBytes()
			raw := make([]byte, out*nblk*bb)
			lcgBytes(raw, 0x9e3779b97f4a7c15)
			pinResidentQuantScales(raw, out, nblk, tc.kind)
			qt := quantizeKQuantFromRaw(raw, out, in, tc.kind)
			x := make([]float32, in)
			for i := range x {
				x[i] = float32((i*7)%23) - 11
			}
			got := kQuantMatRows(qt, x)
			want := refKQuantMatRows(qt, x)
			if len(got) != out {
				t.Fatalf("len(got)=%d, want %d", len(got), out)
			}
			for o := 0; o < out; o++ {
				if got[o] != want[o] {
					t.Fatalf("row %d: kQuantMatRows=%v, ref=%v (GEMV not bit-identical to dequant-then-dot)", o, got[o], want[o])
				}
			}
		})
	}
}

func pinResidentQuantScales(raw []byte, out, nblk int, kind kQuantKind) {
	bb := kind.blockBytes()
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := raw[(o*nblk+b)*bb:]
			switch kind {
			case kindQ6K:
				binary.LittleEndian.PutUint16(blk[q6kBlockBytes-2:], f16One)
			default:
				binary.LittleEndian.PutUint16(blk[0:], f16One)
				if kind == kindQ5K {
					binary.LittleEndian.PutUint16(blk[2:], 0) // min = 0
				}
			}
		}
	}
}

// TestKQuantDequantGolden breaks the port-circularity with HAND-DERIVED goldens: a super-block
// constructed with chosen d/min/scales/codes must dequantize to the value the GGML k-quant
// formula predicts by hand. A wrong port (mis-packed scales, wrong code assembly) fails here.
func TestKQuantDequantGolden(t *testing.T) {
	// Q6_K: d=2.0, all 16 int8 scales=1, all ql/qh=0 -> code q = 0 assembled, minus 32 = -32;
	// weight = d * scale * q = 2 * 1 * (-32) = -64 for all 256.
	t.Run("Q6_K", func(t *testing.T) {
		blk := make([]byte, q6kBlockBytes)
		// ql[0:128]=0, qh[128:192]=0 already zero.
		for i := 0; i < qkK/16; i++ { // 16 scales
			blk[qkK/2+qkK/4+i] = 1
		}
		binary.LittleEndian.PutUint16(blk[q6kBlockBytes-2:], f16Two())
		dst := make([]float32, qkK)
		q6kDequantSuperBlock(dst, blk)
		for i := 0; i < qkK; i++ {
			if dst[i] != -64 {
				t.Fatalf("Q6_K[%d]=%v, want -64 (d=2, scale=1, code=-32)", i, dst[i])
			}
		}
	})
	// Q5_K: d=2.0, min=0, scales packed so every (scale,min)=(1,0), qh=0, ql=0x33
	// -> low/high nibble code=3, +hi(0); weight = d1*(3) - m1 = 2*1*3 - 0 = 6 for all 256.
	t.Run("Q5_K", func(t *testing.T) {
		blk := make([]byte, q5kBlockBytes)
		binary.LittleEndian.PutUint16(blk[0:], f16Two()) // d=2
		binary.LittleEndian.PutUint16(blk[2:], 0)        // min=0
		// scales (12 B) so GetScaleMinK4(0..7) -> (1,0): see quant_kquant.go packing.
		sc := blk[4 : 4+12]
		sc[0], sc[1], sc[2], sc[3] = 1, 1, 1, 1
		sc[4], sc[5], sc[6], sc[7] = 0, 0, 0, 0
		sc[8], sc[9], sc[10], sc[11] = 1, 1, 1, 1
		// qh (32 B) = 0; ql (128 B) = 0x33 (both nibbles = 3).
		ql := blk[4+12+qkK/8 : q5kBlockBytes]
		for i := range ql {
			ql[i] = 0x33
		}
		dst := make([]float32, qkK)
		q5kDequantSuperBlock(dst, blk)
		for i := 0; i < qkK; i++ {
			if dst[i] != 6 {
				t.Fatalf("Q5_K[%d]=%v, want 6 (d=2, scale=1, min=0, code=3)", i, dst[i])
			}
		}
	})
	t.Run("Q8_0", func(t *testing.T) {
		blk := make([]byte, q8_0BlockBytes)
		binary.LittleEndian.PutUint16(blk[0:], f16Two()) // d=2
		for i := 0; i < q8_0BlockWeights; i++ {
			blk[2+i] = byte(int8(i%5) - 2)
		}
		dst := make([]float32, q8_0BlockWeights)
		q8_0DequantBlock(dst, blk)
		for i := 0; i < q8_0BlockWeights; i++ {
			want := float32(int8(i%5)-2) * 2
			if dst[i] != want {
				t.Fatalf("Q8_0[%d]=%v, want %v (d=2, signed code)", i, dst[i], want)
			}
		}
	})
	t.Run("IQ4_XS", func(t *testing.T) {
		blk := make([]byte, iq4xsBlockBytes)
		binary.LittleEndian.PutUint16(blk[0:], f16One) // d=1
		binary.LittleEndian.PutUint16(blk[2:], 0xAAAA) // high scale bits = 2
		for ib := 0; ib < qkK/32; ib++ {
			blk[4+ib/2] |= 1 << (4 * uint(ib%2)) // low scale bits = 1
			for j := 0; j < 16; j++ {
				blk[4+qkK/64+ib*16+j] = 0x98 // low code 8, high code 9
			}
		}
		dst := make([]float32, qkK)
		iq4xsDequantSuperBlock(dst, blk)
		for ib := 0; ib < qkK/32; ib++ {
			off := ib * 32
			for j := 0; j < 16; j++ {
				if dst[off+j] != 1 {
					t.Fatalf("IQ4_XS[%d]=%v, want 1 (ls=33, code=8)", off+j, dst[off+j])
				}
				if dst[off+j+16] != 13 {
					t.Fatalf("IQ4_XS[%d]=%v, want 13 (ls=33, code=9)", off+j+16, dst[off+j+16])
				}
			}
		}
	})
	t.Run("IQ3_XXS", func(t *testing.T) {
		blk := make([]byte, iq3xxsBlockBytes)
		binary.LittleEndian.PutUint16(blk[0:], f16One) // d=1; zero aux => db=0.25
		dst := make([]float32, qkK)
		iq3xxsDequantSuperBlock(dst, blk)
		for i, v := range dst {
			if v != 1 {
				t.Fatalf("IQ3_XXS[%d]=%v, want 1 (grid[0]=4, positive signs, db=0.25)", i, v)
			}
		}
	})
}

// f16Two returns the IEEE-754 half encoding of 2.0 (0x4000).
func f16Two() uint16 { return 0x4000 }

// TestResidentMatRowsDispatchesKQuant proves the host expert decode seam routes a resident
// k-quant weight to kQuantMatRows (the dispatch the GLM CPU-offloaded experts take), and that
// AddResidentQ6K/Q5K store under an eligible name.
func TestResidentMatRowsDispatchesKQuant(t *testing.T) {
	const out, in = 4, 256
	name := "model.layers.3.mlp.experts.1.down_proj.weight"
	bb := kindQ6K.blockBytes()
	raw := make([]byte, out*(in/qkK)*bb)
	lcgBytes(raw, 0xdeadbeef)
	for o := 0; o < out; o++ {
		binary.LittleEndian.PutUint16(raw[o*bb+q6kBlockBytes-2:], f16One)
	}

	// AddResidentQ6K stores under the eligible name.
	b := NewQuantBuilder(Config{}, false)
	if err := b.AddResidentQ6K(name, []int{out, in}, raw); err != nil {
		t.Fatalf("AddResidentQ6K: %v", err)
	}
	if b.m.kqw[name] == nil {
		t.Fatalf("AddResidentQ6K did not store %q in kqw", name)
	}

	// residentMatRows dispatches the resident k-quant weight to kQuantMatRows.
	m := &Model{kqw: map[string]*kQuantTensor{name: quantizeKQuantFromRaw(raw, out, in, kindQ6K)}}
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(i%5) - 2
	}
	got := m.residentMatRows(name, x, out, in)
	want := kQuantMatRows(m.kqw[name], x)
	for o := 0; o < out; o++ {
		if got[o] != want[o] {
			t.Fatalf("row %d: residentMatRows=%v, kQuantMatRows=%v (dispatch mismatch)", o, got[o], want[o])
		}
	}
	if !m.hasKQuant(name) || m.KQuantCount() != 1 {
		t.Fatalf("hasKQuant=%v count=%d, want true/1", m.hasKQuant(name), m.KQuantCount())
	}
}

func TestResidentMatRowsDispatchesIQAndQ8RawExperts(t *testing.T) {
	const out, in = 4, 256
	name := "model.layers.3.mlp.experts.1.gate_proj.weight"
	for _, tc := range []struct {
		name string
		kind kQuantKind
		add  func(*QuantBuilder, string, []int, []byte) error
	}{
		{"IQ3_XXS", kindIQ3XXS, (*QuantBuilder).AddResidentIQ3XXS},
		{"IQ4_XS", kindIQ4XS, (*QuantBuilder).AddResidentIQ4XS},
		{"Q8_0", kindQ8_0, (*QuantBuilder).AddResidentQ8_0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			nblk := in / tc.kind.blockWeights()
			raw := make([]byte, out*nblk*tc.kind.blockBytes())
			lcgBytes(raw, 0xbead1234)
			pinResidentQuantScales(raw, out, nblk, tc.kind)

			b := NewQuantBuilder(Config{}, false)
			if err := tc.add(b, name, []int{out, in}, raw); err != nil {
				t.Fatalf("AddResident%s: %v", tc.name, err)
			}
			if b.m.kqw[name] == nil || b.m.kqw[name].kind != tc.kind {
				t.Fatalf("AddResident%s did not store %q as %s", tc.name, name, tc.kind)
			}

			m := &Model{kqw: map[string]*kQuantTensor{name: quantizeKQuantFromRaw(raw, out, in, tc.kind)}}
			x := make([]float32, in)
			for i := range x {
				x[i] = float32(i%7) - 3
			}
			got := m.residentMatRows(name, x, out, in)
			want := kQuantMatRows(m.kqw[name], x)
			for o := 0; o < out; o++ {
				if got[o] != want[o] {
					t.Fatalf("row %d: residentMatRows=%v, kQuantMatRows=%v (dispatch mismatch)", o, got[o], want[o])
				}
			}
		})
	}
}
