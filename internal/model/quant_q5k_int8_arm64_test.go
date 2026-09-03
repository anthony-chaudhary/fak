//go:build arm64 && !(fakaccel && darwin && cgo)

package model

import (
	"encoding/binary"
	"math/rand"
	"testing"
)

// TestQ5KReduceAsmMatchesScalar pins the arm64 NEON SDOT Q5_K reduction kernel to the scalar
// reference on the integer reductions it owns (IS = Σ q5*qx, SS = Σ qx), bit-for-bit — the arm64
// counterpart of the amd64 AVX2/VNNI test. The float combine is shared Go, so once the int32
// reductions match, the full asm-path dot equals the scalar-int8-path dot exactly. The Q5_K 5th bit
// is reassembled from qh with a per-chunk right shift and mask; any mismatch indicates a bug in
// 5th-bit extraction, nibble unpacking, SDOT accumulation, or register clobber. SDOT and the
// ones-vector sum are associative with no overflow on these ranges, so any lane order yields the
// same int32. Skips on a CPU without FEAT_DotProd.
func TestQ5KReduceAsmMatchesScalar(t *testing.T) {
	if !detectDotProd() {
		t.Skip("FEAT_DotProd (asimddp) not available — q5k asm inactive")
	}

	t.Run("deterministic_lcg", func(t *testing.T) {
		const out, in = 9, 768 // 3 super-blocks/row, odd rows
		nblk := in / qkK
		bb := kindQ5K.blockBytes()
		raw := make([]byte, out*nblk*bb)
		lcgBytes(raw, 0xC5C5A5A5DEADBEEF) // varied bytes exercise all 8 qh bit positions
		for o := 0; o < out; o++ {
			for b := 0; b < nblk; b++ {
				blk := raw[(o*nblk+b)*bb:]
				binary.LittleEndian.PutUint16(blk[0:], f16One) // d=1.0
				binary.LittleEndian.PutUint16(blk[2:], 0)      // min=0.0
			}
		}
		qt := quantizeKQuantFromRaw(raw, out, in, kindQ5K)

		x := make([]float32, in)
		for i := range x {
			x[i] = float32((i*7)%23) - 11
		}
		qv := quantizeVecQ8(x)

		isAsm := make([]int32, nblk*8)
		ssAsm := make([]int32, nblk*8)
		isSc := make([]int32, nblk*8)
		ssSc := make([]int32, nblk*8)
		rowBytes := qt.rowBytes()
		for o := 0; o < out; o++ {
			row := qt.raw[o*rowBytes : (o+1)*rowBytes]
			q5kReduceRowAsm(&row[0], nblk, &qv.q[0], &isAsm[0], &ssAsm[0])
			q5kReduceRowScalar(row, nblk, qv.q, isSc, ssSc)
			for i := range isAsm {
				if isAsm[i] != isSc[i] {
					t.Fatalf("row %d IS[%d]: asm=%d scalar=%d (q5/qh-bit/nibble mismatch)", o, i, isAsm[i], isSc[i])
				}
				if ssAsm[i] != ssSc[i] {
					t.Fatalf("row %d SS[%d]: asm=%d scalar=%d (activation-sum mismatch)", o, i, ssAsm[i], ssSc[i])
				}
			}
		}
		t.Logf("q5k NEON SDOT reduce bit-identical to scalar across %d rows x %d sub-blocks (neonDot=%v)", out, nblk*8, neonDot)
	})

	t.Run("random_inputs", func(t *testing.T) {
		rng := rand.New(rand.NewSource(42))
		for trial := 0; trial < 10; trial++ {
			out := 1 + rng.Intn(16)
			nblk := 1 + rng.Intn(8) // 1..8 super-blocks
			in := nblk * qkK
			bb := kindQ5K.blockBytes()
			raw := make([]byte, out*nblk*bb)
			for i := range raw {
				raw[i] = byte(rng.Intn(256))
			}

			qx := make([]int8, in)
			for i := range qx {
				qx[i] = int8(rng.Intn(255) - 127)
			}

			isAsm := make([]int32, nblk*8)
			ssAsm := make([]int32, nblk*8)
			isSc := make([]int32, nblk*8)
			ssSc := make([]int32, nblk*8)

			for o := 0; o < out; o++ {
				row := raw[o*nblk*bb : (o+1)*nblk*bb]
				q5kReduceRowAsm(&row[0], nblk, &qx[0], &isAsm[0], &ssAsm[0])
				q5kReduceRowScalar(row, nblk, qx, isSc, ssSc)
				for i := range isAsm {
					if isAsm[i] != isSc[i] {
						t.Fatalf("trial %d row %d IS[%d]: asm=%d scalar=%d", trial, o, i, isAsm[i], isSc[i])
					}
					if ssAsm[i] != ssSc[i] {
						t.Fatalf("trial %d row %d SS[%d]: asm=%d scalar=%d", trial, o, i, ssAsm[i], ssSc[i])
					}
				}
			}
		}
	})

	t.Run("extreme_values", func(t *testing.T) {
		nblk := 4
		in := nblk * qkK
		bb := kindQ5K.blockBytes()

		// Case 1: all 0xFF weights (q5 = 31 everywhere) with max positive activations (+127)
		raw := make([]byte, nblk*bb)
		for i := range raw {
			raw[i] = 0xFF
		}
		qx := make([]int8, in)
		for i := range qx {
			qx[i] = 127
		}
		isAsm := make([]int32, nblk*8)
		ssAsm := make([]int32, nblk*8)
		isSc := make([]int32, nblk*8)
		ssSc := make([]int32, nblk*8)

		q5kReduceRowAsm(&raw[0], nblk, &qx[0], &isAsm[0], &ssAsm[0])
		q5kReduceRowScalar(raw, nblk, qx, isSc, ssSc)
		for i := range isAsm {
			if isAsm[i] != isSc[i] {
				t.Fatalf("max-pos IS[%d]: asm=%d scalar=%d", i, isAsm[i], isSc[i])
			}
			if ssAsm[i] != ssSc[i] {
				t.Fatalf("max-pos SS[%d]: asm=%d scalar=%d", i, ssAsm[i], ssSc[i])
			}
		}

		// Case 2: all 0xFF weights with max negative activations (-128)
		for i := range qx {
			qx[i] = -128
		}
		q5kReduceRowAsm(&raw[0], nblk, &qx[0], &isAsm[0], &ssAsm[0])
		q5kReduceRowScalar(raw, nblk, qx, isSc, ssSc)
		for i := range isAsm {
			if isAsm[i] != isSc[i] {
				t.Fatalf("max-neg IS[%d]: asm=%d scalar=%d", i, isAsm[i], isSc[i])
			}
			if ssAsm[i] != ssSc[i] {
				t.Fatalf("max-neg SS[%d]: asm=%d scalar=%d", i, ssAsm[i], ssSc[i])
			}
		}

		// Case 3: all zero weights and activations
		for i := range raw {
			raw[i] = 0
		}
		for i := range qx {
			qx[i] = 0
		}
		q5kReduceRowAsm(&raw[0], nblk, &qx[0], &isAsm[0], &ssAsm[0])
		q5kReduceRowScalar(raw, nblk, qx, isSc, ssSc)
		for i := range isAsm {
			if isAsm[i] != 0 || isSc[i] != 0 {
				t.Fatalf("zero IS[%d]: asm=%d scalar=%d", i, isAsm[i], isSc[i])
			}
			if ssAsm[i] != 0 || ssSc[i] != 0 {
				t.Fatalf("zero SS[%d]: asm=%d scalar=%d", i, ssAsm[i], ssSc[i])
			}
		}
	})

	t.Run("zero_blocks", func(t *testing.T) {
		// Calling with nblk=0 should cleanly return without touching any memory
		var dummyRow byte
		var dummyQx int8
		var dummyIS, dummySS int32
		q5kReduceRowAsm(&dummyRow, 0, &dummyQx, &dummyIS, &dummySS)
	})
}

// TestQ5KReduceRowDispatch verifies that q5kReduceRow dispatches to the NEON asm path when
// neonDot is true, producing identical results to q5kReduceRowScalar.
func TestQ5KReduceRowDispatch(t *testing.T) {
	const out, in = 4, 512
	nblk := in / qkK
	bb := kindQ5K.blockBytes()
	raw := make([]byte, out*nblk*bb)
	lcgBytes(raw, 0x123456789abcdef0)

	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*11)%31) - 15
	}
	qv := quantizeVecQ8(x)

	isDisp := make([]int32, nblk*8)
	ssDisp := make([]int32, nblk*8)
	isSc := make([]int32, nblk*8)
	ssSc := make([]int32, nblk*8)

	for o := 0; o < out; o++ {
		row := raw[o*nblk*bb : (o+1)*nblk*bb]
		q5kReduceRow(row, nblk, qv.q, isDisp, ssDisp)
		q5kReduceRowScalar(row, nblk, qv.q, isSc, ssSc)
		for i := range isDisp {
			if isDisp[i] != isSc[i] {
				t.Fatalf("row %d IS[%d]: dispatch=%d scalar=%d", o, i, isDisp[i], isSc[i])
			}
			if ssDisp[i] != ssSc[i] {
				t.Fatalf("row %d SS[%d]: dispatch=%d scalar=%d", o, i, ssDisp[i], ssSc[i])
			}
		}
	}
}

// TestQ5KMatRowsInt8EndToEnd checks that q5kMatRowsRangeInt8 produces bit-identical float results
// when using the asm reducer versus the scalar reducer.
func TestQ5KMatRowsInt8EndToEnd(t *testing.T) {
	const out, in = 8, 512
	nblk := in / qkK
	bb := kindQ5K.blockBytes()
	raw := make([]byte, out*nblk*bb)
	lcgBytes(raw, 0xfeedbeefcafebabe)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := raw[(o*nblk+b)*bb:]
			binary.LittleEndian.PutUint16(blk[0:], f16One)
			binary.LittleEndian.PutUint16(blk[2:], 0)
		}
	}
	qt := quantizeKQuantFromRaw(raw, out, in, kindQ5K)

	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i % 17) - 8)
	}
	qv := quantizeVecQ8(x)

	yAsm := make([]float32, out)
	q5kMatRowsRangeInt8(qt, qv, yAsm, 0, out)

	// Compute scalar reference directly
	ySc := make([]float32, out)
	IS := make([]int32, nblk*8)
	SS := make([]int32, nblk*8)
	rowBytes := qt.rowBytes()
	for o := 0; o < out; o++ {
		row := qt.raw[o*rowBytes : (o+1)*rowBytes]
		q5kReduceRowScalar(row, nblk, qv.q, IS, SS)
		ySc[o] = kQuantCombineRow(row, nblk, qv.d, IS, SS)
	}

	for o := 0; o < out; o++ {
		if yAsm[o] != ySc[o] {
			t.Fatalf("row %d: yAsm=%f ySc=%f (diff %e)", o, yAsm[o], ySc[o], yAsm[o]-ySc[o])
		}
	}
}

func BenchmarkQ5KReduceRowAsm(b *testing.B) {
	if !detectDotProd() {
		b.Skip("FEAT_DotProd not available")
	}
	const in = 6144
	nblk := in / qkK
	row := make([]byte, nblk*kindQ5K.blockBytes())
	lcgBytes(row, 0x12345678)
	qx := make([]int8, in)
	for i := range qx {
		qx[i] = int8(i % 127)
	}
	IS := make([]int32, nblk*8)
	SS := make([]int32, nblk*8)

	b.SetBytes(int64(nblk * kindQ5K.blockBytes()))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q5kReduceRowAsm(&row[0], nblk, &qx[0], &IS[0], &SS[0])
	}
}

func BenchmarkQ5KReduceRowScalar(b *testing.B) {
	const in = 6144
	nblk := in / qkK
	row := make([]byte, nblk*kindQ5K.blockBytes())
	lcgBytes(row, 0x12345678)
	qx := make([]int8, in)
	for i := range qx {
		qx[i] = int8(i % 127)
	}
	IS := make([]int32, nblk*8)
	SS := make([]int32, nblk*8)

	b.SetBytes(int64(nblk * kindQ5K.blockBytes()))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q5kReduceRowScalar(row, nblk, qx, IS, SS)
	}
}
