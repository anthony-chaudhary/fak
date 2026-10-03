package model

import (
	"encoding/binary"
	"math"
	"testing"
)

// quant_kquant_int8_q2k_test.go is the manifest-declared parity witness for the Q2_K int8 GEMV
// (internal/model/quant_kquant_int8_q2k.go). It is authored against the SPECIFICATION, not the
// implementation: the oracle is the f32 kQuantMatRows / kQuantMatRowsRange path (byte-layout
// authority = q2kDequantSuperBlockScalar), and the candidate is the int8 path under the SDOT gate.
//
// Q2_K is AFFINE with 16-wide sub-blocks: w[j] = d*sc_s*q2[j] - min*m_s, q8 activation
// x[j] = dx_b*qx[j] per 32-wide block. The int8 path is therefore APPROXIMATE (it adds activation
// quantization) so the f32 x-oracle gap is dominated by quantization noise, not kernel error. Two
// independent checks separate those: (a) cosine + the canonical per-row relative bound, and (b) a
// float64 EXACT oracle — dot(dequant(w), dequant_activation) — which the kernel must match to a
// small absolute error. (b) is the invariant that actually fails on a swapped half / wrong
// sub-block map; (a) alone is cancellation-sensitive (even the shipped Q5_K/Q6_K metric exceeds
// 0.02 on some seeds).

// q2kTestTensor builds a resident Q2_K tensor over randomized LCG code/scale bytes, forcing d = 1.0
// (f16One) and a caller-supplied finite min so no Inf/NaN can appear.
func q2kTestTensor(out, in int, seed uint64, minBits uint16) *kQuantTensor {
	nblk := in / qkK
	bb := kindQ2K.blockBytes()
	raw := make([]byte, out*nblk*bb)
	lcgBytes(raw, seed)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := raw[(o*nblk+b)*bb:]
			binary.LittleEndian.PutUint16(blk[80:], f16One)  // d = 1.0
			binary.LittleEndian.PutUint16(blk[82:], minBits) // finite min
		}
	}
	return quantizeKQuantFromRaw(raw, out, in, kindQ2K)
}

// q2kTestActivation returns a non-trivial activation; variant 0 is the established deterministic
// pattern, variant 1 is a seeded pseudo-random variant.
func q2kTestActivation(in int, seed uint64, variant int) []float32 {
	x := make([]float32, in)
	if variant == 0 {
		for i := range x {
			x[i] = float32((i*7)%23) - 11
		}
		return x
	}
	b := make([]byte, in)
	lcgBytes(b, seed^0xabcdef0123456789)
	for i := range x {
		x[i] = float32(int(b[i])%23) - 11
	}
	return x
}

// q2kCosineMaxRel reports cosine similarity and the max per-row relative error (denominator floored
// at 1) — the established Q5_K/Q6_K parity metric.
func q2kCosineMaxRel(got, want []float32) (cos, maxRel float64) {
	var dot, ng, nw float64
	for o := range want {
		dot += float64(got[o]) * float64(want[o])
		ng += float64(got[o]) * float64(got[o])
		nw += float64(want[o]) * float64(want[o])
		den := math.Abs(float64(want[o]))
		if den < 1 {
			den = 1
		}
		if rel := math.Abs(float64(got[o]-want[o])) / den; rel > maxRel {
			maxRel = rel
		}
	}
	return dot / (math.Sqrt(ng)*math.Sqrt(nw) + 1e-12), maxRel
}

// q2kExactOracleErr returns max_o |got[o] - exact_oracle[o]| / scale, where the exact oracle is
// dot(dequant(w_o), dequant_activation) accumulated in float64 and scale = max_o |f32 oracle|. A
// wrong 2-bit decode, swapped 16-lane half, or off-by-one scale index moves this far above 0.
func q2kExactOracleErr(qt *kQuantTensor, qv q8Vec, got, want []float32) float64 {
	scale := 1.0
	for _, v := range want {
		if a := math.Abs(float64(v)); a > scale {
			scale = a
		}
	}
	buf := make([]float32, qkK)
	rowBytes := qt.rowBytes()
	var worst float64
	for o := 0; o < qt.out; o++ {
		row := qt.raw[o*rowBytes : (o+1)*rowBytes]
		var acc float64
		for b := 0; b < qt.nblk; b++ {
			q2kDequantSuperBlockScalar(buf, row[b*q2kBlockBytes:(b+1)*q2kBlockBytes])
			for i := 0; i < qkK; i++ {
				bx := b*8 + i/32
				acc += float64(buf[i]) * float64(qv.d[bx]) * float64(qv.q[b*qkK+i])
			}
		}
		if e := math.Abs(float64(got[o])-acc) / scale; e > worst {
			worst = e
		}
	}
	return worst
}

func q2kSameBits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// TestQ2KQ8KInt8MatchesF32 is the canonical oracle-parity test: with the finite canonical fixture the
// int8 path (gate ON) must track the f32 path (gate OFF) within the established cosine / relative
// bounds, match an independent float64 exact oracle, and the raw range kernel must track the f32
// range over the same rows.
func TestQ2KQ8KInt8MatchesF32(t *testing.T) {
	const (
		out = 9   // odd, to exercise the parallel row split's tail
		in  = 512 // 2 super-blocks per row
	)
	qt := q2kTestTensor(out, in, 0x123456789abcdef0, 0x3000) // finite min = 0.125
	x := q2kTestActivation(in, 0, 0)

	setKQuantSDOTForTest(false)
	want := kQuantMatRows(qt, x)

	setKQuantSDOTForTest(true)
	t.Cleanup(func() { kQuantSDOTForce = 0 })
	got := kQuantMatRows(qt, x)

	cos, maxRel := q2kCosineMaxRel(got, want)
	if cos < 0.9999 {
		t.Fatalf("int8 Q2_K vs f32 cosine %.6f < 0.9999 (got=%v want=%v)", cos, got, want)
	}
	if maxRel > 0.02 {
		t.Fatalf("int8 Q2_K vs f32 max rel err %.4f > 0.02", maxRel)
	}
	qv := quantizeVecQ8(x)
	if e := q2kExactOracleErr(qt, qv, got, want); e > 1e-4 {
		t.Fatalf("int8 Q2_K vs exact f64 oracle normalized abs err %.6g > 1e-4", e)
	}

	// Direct raw-range kernel vs the f32 range over the same rows.
	direct := make([]float32, qt.out)
	q2kMatRowsRangeInt8Raw(qt.raw, qt, qv, direct, 0, qt.out)
	ref := make([]float32, qt.out)
	kQuantMatRowsRange(qt, x, ref, 0, qt.out)
	cos2, maxRel2 := q2kCosineMaxRel(direct, ref)
	if cos2 < 0.9999 || maxRel2 > 0.02 {
		t.Fatalf("q2kMatRowsRangeInt8Raw vs f32 range cosine %.6f maxRel %.4f", cos2, maxRel2)
	}
	t.Logf("Q2_K int8 vs f32: cosine=%.8f maxRel=%.5f (raw-range cos=%.8f)", cos, maxRel, cos2)
}

// TestQ2KInt8RandomizedSeeds is the adversarial breadth test: over six distinct randomized weight
// seeds and two activation variants, the kernel must track the f32 oracle and match the exact f64
// oracle. A swapped 16-lane half, a wrong scale index, or a wrong sub-block->activation-block map
// would blow the exact-oracle bound.
func TestQ2KInt8RandomizedSeeds(t *testing.T) {
	const (
		out = 9
		in  = 512
	)
	seeds := []uint64{
		0x123456789abcdef0, 0x0fedcba987654321, 0xdeadbeefcafef00d,
		0x0123456789abcdef, 0xfedcba9876543210, 0xa5a5a5a55a5a5a5a,
	}
	for si, seed := range seeds {
		for variant := 0; variant < 2; variant++ {
			t.Run(seedName(si, variant), func(t *testing.T) {
				qt := q2kTestTensor(out, in, seed, 0x3000)
				x := q2kTestActivation(in, seed, variant)

				setKQuantSDOTForTest(false)
				want := kQuantMatRows(qt, x)

				setKQuantSDOTForTest(true)
				t.Cleanup(func() { kQuantSDOTForce = 0 })
				got := kQuantMatRows(qt, x)

				cos, _ := q2kCosineMaxRel(got, want)
				if cos < 0.9999 {
					t.Fatalf("seed %#x variant %d: cosine %.6f < 0.9999", seed, variant, cos)
				}
				qv := quantizeVecQ8(x)
				if e := q2kExactOracleErr(qt, qv, got, want); e > 1e-4 {
					t.Fatalf("seed %#x variant %d: exact-oracle normalized abs err %.6g > 1e-4", seed, variant, e)
				}
			})
		}
	}
}

func seedName(si, variant int) string {
	const hexd = "0123456789abcdef"
	return string([]byte{'s', hexd[si%16], '/', 'v', byte('0' + variant)})
}

// q2kSetCode packs a 2-bit code into the Q2_K q-region for sub-block s, lane l, following the
// dequant byte layout: within a 128-weight chunk, four columns j = 0..3 with shift 2*j; column j's
// low 16 lanes live at q[chunk*32 + l] and the high 16 at q[chunk*32 + 16 + l].
func q2kSetCode(q []byte, s, l int, code byte) {
	chunk := s / 8
	sl := s % 8
	j := sl / 2
	half := sl % 2
	idx := chunk*32 + half*16 + l
	shift := uint(2 * j)
	q[idx] = (q[idx] &^ (3 << shift)) | ((code & 3) << shift)
}

// TestQ2KInt8SubBlockAlignment is the adversarial sub-block fixture: all 16 per-sub-block scales are
// distinct, the low-16 and high-16 code halves differ, and adjacent sub-blocks that share one
// 32-wide activation block carry different codes. A swapped half, an off-by-one scale index, or a
// wrong sub-block->activation-block map all move the int8 result away from the f32 oracle.
func TestQ2KInt8SubBlockAlignment(t *testing.T) {
	const (
		out = 4
		in  = 256 // one super-block per row: 16 sub-blocks, 8 activation blocks
	)
	bb := kindQ2K.blockBytes()
	raw := make([]byte, out*bb)
	for o := 0; o < out; o++ {
		blk := raw[o*bb:]
		scales := blk[:16]
		for s := 0; s < 16; s++ {
			// low nibble = s (all 16 distinct), high nibble = 15-s (all distinct).
			scales[s] = byte((15-s)<<4) | byte(s)
		}
		q := blk[16:80]
		for i := range q {
			q[i] = 0
		}
		for s := 0; s < 16; s++ {
			for l := 0; l < 16; l++ {
				if s%2 == 0 {
					q2kSetCode(q, s, l, byte((s+l)&3)) // low half
				} else {
					q2kSetCode(q, s, l, byte((s+l+1)&3)) // high half differs
				}
			}
		}
		binary.LittleEndian.PutUint16(blk[80:], f16One) // d = 1.0
		binary.LittleEndian.PutUint16(blk[82:], 0x3800) // min = 0.5 (f16)
	}
	qt := quantizeKQuantFromRaw(raw, out, in, kindQ2K)

	// Per-32-block growing amplitude makes a wrong activation-block map visibly wrong.
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(1+i/32) * (float32((i*7)%23) - 11)
	}

	setKQuantSDOTForTest(false)
	want := kQuantMatRows(qt, x)

	setKQuantSDOTForTest(true)
	t.Cleanup(func() { kQuantSDOTForce = 0 })
	got := kQuantMatRows(qt, x)

	cos, maxRel := q2kCosineMaxRel(got, want)
	if cos < 0.9999 {
		t.Fatalf("Q2_K sub-block alignment: cosine %.6f < 0.9999 (got=%v want=%v)", cos, got, want)
	}
	if maxRel > 0.02 {
		t.Fatalf("Q2_K sub-block alignment: max rel err %.4f > 0.02 (got=%v want=%v)", maxRel, got, want)
	}
	qv := quantizeVecQ8(x)
	if e := q2kExactOracleErr(qt, qv, got, want); e > 1e-4 {
		t.Fatalf("Q2_K sub-block alignment: exact-oracle normalized abs err %.6g > 1e-4", e)
	}
	t.Logf("Q2_K sub-block alignment: cosine=%.8f maxRel=%.5f", cos, maxRel)
}

// TestQ2KInt8Zero pins the zero contract: an all-zero activation yields exactly zero (no NaN), and
// an all-zero code region with min = 0 also yields exactly zero. The f32 oracle must agree.
func TestQ2KInt8Zero(t *testing.T) {
	const (
		out = 6
		in  = 512
	)

	t.Run("zero-activation", func(t *testing.T) {
		qt := q2kTestTensor(out, in, 0x51ee7, 0x3800)
		x := make([]float32, in)

		setKQuantSDOTForTest(false)
		want := kQuantMatRows(qt, x)

		setKQuantSDOTForTest(true)
		t.Cleanup(func() { kQuantSDOTForce = 0 })
		got := kQuantMatRows(qt, x)

		for o := 0; o < out; o++ {
			if math.IsNaN(float64(got[o])) || math.IsInf(float64(got[o]), 0) {
				t.Fatalf("row %d: non-finite %v", o, got[o])
			}
			if got[o] != 0 {
				t.Fatalf("row %d: zero activation gave %v, want exactly 0", o, got[o])
			}
			if want[o] != 0 {
				t.Fatalf("oracle row %d: zero activation gave %v, want exactly 0", o, want[o])
			}
		}
	})

	t.Run("zero-codes-min-zero", func(t *testing.T) {
		qt := q2kTestTensor(out, in, 0x0c0de5, 0) // min = 0
		// Zero the 64-byte q region of every super-block so every 2-bit code is 0.
		nblk := in / qkK
		bb := kindQ2K.blockBytes()
		for o := 0; o < out; o++ {
			for b := 0; b < nblk; b++ {
				blk := qt.raw[(o*nblk+b)*bb:]
				for i := 16; i < 80; i++ {
					blk[i] = 0
				}
			}
		}
		x := q2kTestActivation(in, 0x1234, 0)

		setKQuantSDOTForTest(false)
		want := kQuantMatRows(qt, x)

		setKQuantSDOTForTest(true)
		t.Cleanup(func() { kQuantSDOTForce = 0 })
		got := kQuantMatRows(qt, x)

		for o := 0; o < out; o++ {
			if math.IsNaN(float64(got[o])) || math.IsInf(float64(got[o]), 0) {
				t.Fatalf("row %d: non-finite %v", o, got[o])
			}
			if got[o] != 0 {
				t.Fatalf("row %d: zero codes + min=0 gave %v, want exactly 0", o, got[o])
			}
			if want[o] != 0 {
				t.Fatalf("oracle row %d: zero codes + min=0 gave %v, want exactly 0", o, want[o])
			}
		}
	})
}

// TestQ2KInt8Finite asserts every int8 output over randomized finite blocks/activation is finite and
// tracks the f32 oracle and the exact float64 oracle.
func TestQ2KInt8Finite(t *testing.T) {
	const (
		out = 11
		in  = 1024 // 4 super-blocks
	)
	qt := q2kTestTensor(out, in, 0xf1e17ef00d, 0x3400) // min = 0.25
	x := q2kTestActivation(in, 0xa5a5a5a5, 1)

	setKQuantSDOTForTest(false)
	want := kQuantMatRows(qt, x)

	setKQuantSDOTForTest(true)
	t.Cleanup(func() { kQuantSDOTForce = 0 })
	got := kQuantMatRows(qt, x)

	for o := 0; o < out; o++ {
		if math.IsNaN(float64(got[o])) || math.IsInf(float64(got[o]), 0) {
			t.Fatalf("row %d: non-finite int8 output %v", o, got[o])
		}
		if math.IsNaN(float64(want[o])) || math.IsInf(float64(want[o]), 0) {
			t.Fatalf("row %d: non-finite f32 output %v", o, want[o])
		}
	}
	cos, _ := q2kCosineMaxRel(got, want)
	if cos < 0.9999 {
		t.Fatalf("finite: cosine %.6f < 0.9999", cos)
	}
	qv := quantizeVecQ8(x)
	if e := q2kExactOracleErr(qt, qv, got, want); e > 1e-4 {
		t.Fatalf("finite: exact-oracle normalized abs err %.6g > 1e-4", e)
	}
}

// TestQ2KInt8ReachableDispatch proves the gate actually routes kindQ2K to the int8 kernel: gate-ON
// kQuantMatRows equals a direct q2kMatRowsRangeInt8Raw computation bit-for-bit, and gate-OFF equals
// the f32 path bit-for-bit. It also drives the two multi-row callers (kQuantBatchRows and
// kQuantMatRowsSubset) and checks each reaches the Q2_K path.
func TestQ2KInt8ReachableDispatch(t *testing.T) {
	const (
		out = 7
		in  = 512
	)
	qt := q2kTestTensor(out, in, 0xd157a7c4, 0x3000)
	x := q2kTestActivation(in, 0xbeef, 0)
	subset := []int{0, 2, 3, out - 1}

	// Gate ON: kQuantMatRows must be the int8 kernel.
	setKQuantSDOTForTest(true)
	t.Cleanup(func() { kQuantSDOTForce = 0 })
	gotOn := kQuantMatRows(qt, x)
	qv := quantizeVecQ8(x)
	direct := make([]float32, qt.out)
	q2kMatRowsRangeInt8Raw(qt.raw, qt, qv, direct, 0, qt.out)
	if !q2kSameBits(gotOn, direct) {
		t.Fatalf("gate-ON kQuantMatRows not bit-identical to q2kMatRowsRangeInt8Raw: got=%v direct=%v", gotOn, direct)
	}

	// Batch caller (one tensor, rowsPer == out, per-expert activation) reaches the same kernel.
	yb := [][]float32{make([]float32, qt.out)}
	kQuantBatchRows([]*kQuantTensor{qt}, [][]float32{x}, qt.out, yb)
	if !q2kSameBits(yb[0], gotOn) {
		t.Fatalf("kQuantBatchRows (gate ON) not bit-identical to kQuantMatRows: got=%v want=%v", yb[0], gotOn)
	}

	// Subset caller reaches the same kernel for each requested row.
	gOn := kQuantMatRowsSubset(qt, x, subset)
	for i, tok := range subset {
		if math.Float32bits(gOn[i]) != math.Float32bits(gotOn[tok]) {
			t.Fatalf("kQuantMatRowsSubset (gate ON) row %d (tok %d) = %v, want %v", i, tok, gOn[i], gotOn[tok])
		}
	}

	// Gate OFF: kQuantMatRows must stay the byte-identical f32 path.
	setKQuantSDOTForTest(false)
	gotOff := kQuantMatRows(qt, x)
	ref := make([]float32, qt.out)
	kQuantMatRowsRange(qt, x, ref, 0, qt.out)
	if !q2kSameBits(gotOff, ref) {
		t.Fatalf("gate-OFF kQuantMatRows not bit-identical to f32 kQuantMatRowsRange: got=%v ref=%v", gotOff, ref)
	}

	yb2 := [][]float32{make([]float32, qt.out)}
	kQuantBatchRows([]*kQuantTensor{qt}, [][]float32{x}, qt.out, yb2)
	if !q2kSameBits(yb2[0], gotOff) {
		t.Fatalf("kQuantBatchRows (gate OFF) not bit-identical to f32 kQuantMatRows: got=%v want=%v", yb2[0], gotOff)
	}
	gOff := kQuantMatRowsSubset(qt, x, subset)
	for i, tok := range subset {
		if math.Float32bits(gOff[i]) != math.Float32bits(gotOff[tok]) {
			t.Fatalf("kQuantMatRowsSubset (gate OFF) row %d (tok %d) = %v, want %v", i, tok, gOff[i], gotOff[tok])
		}
	}

	// Oracle cross-check: gate-ON vs gate-OFF within the approximate tolerance.
	cos, maxRel := q2kCosineMaxRel(gotOn, gotOff)
	if cos < 0.9999 || maxRel > 0.02 {
		t.Fatalf("gate-ON vs gate-OFF: cosine %.6f maxRel %.4f", cos, maxRel)
	}
}

// TestQ2KInt8GateContract pins the Q2_K-specific gate contract: kindQ2K is enabled by the gate, it
// tracks kQuantSDOTDefault when the force is 0, and the test force overrides it both ways.
func TestQ2KInt8GateContract(t *testing.T) {
	restore := kQuantSDOTForce
	t.Cleanup(func() { kQuantSDOTForce = restore })

	kQuantSDOTForce = 0
	if got := kQuantSDOTEnabled(kindQ2K); got != kQuantSDOTDefault {
		t.Fatalf("kQuantSDOTEnabled(kindQ2K)=%v must track the FAK_KQ_INT8 env default %v", got, kQuantSDOTDefault)
	}

	setKQuantSDOTForTest(true)
	if !kQuantSDOTEnabled(kindQ2K) {
		t.Fatal("kQuantSDOTEnabled(kindQ2K) must be true when forced on")
	}

	setKQuantSDOTForTest(false)
	if kQuantSDOTEnabled(kindQ2K) {
		t.Fatal("kQuantSDOTEnabled(kindQ2K) must be false when forced off")
	}

	kQuantSDOTForce = 0
}
