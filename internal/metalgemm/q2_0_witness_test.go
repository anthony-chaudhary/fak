//go:build !(darwin && arm64 && cgo)

// Witness tests for the ternary (Q2_0) math obligations of package metalgemm — the stub build
// (non-Apple-Silicon or cgo-disabled). The build tag mirrors metalgemm_stub.go exactly, so these
// run in every build that CANNOT execute the Metal kernel, and never collide with the
// Apple-Silicon parity tests in q2_0_test.go.
//
// Why this file exists (issue #4873). The done-condition witness — "Metal Q2_0 GEMV/GEMVBatch runs
// on Apple Silicon and matches CPU-ref ternary GEMM within tolerance" — needs an Apple host to
// execute, and no node in docs/fleet-compute-nodes.md is Apple Silicon. That gates the DEVICE run,
// not the MATH. The Metal kernel (q2_0.m) is a transliteration of one specific reference: the
// packing convention, the ternary code set, and the GEMV contraction in q2_0_ref_test.go. This file
// pins that reference executably in every build, so the half of the claim that does not need a GPU
// is proven here and the Apple-gated half is a transliteration check against a fixed target rather
// than against an unstated one. Both name Q2_0, so `go test ./internal/metalgemm -run Q2_0`
// witnesses the reference here and the device parity on an Apple host.
//
// Deterministic, non-vacuous, stdlib-only.
package metalgemm

import (
	"math"
	"math/rand"
	"testing"
)

// TestQ2_0PackingBitLayoutIsFourCodesPerByteLowFirst pins the wire format the in-shader unpack in
// q2_0.m hard-codes: four 2-bit codes per byte, LOW CODE FIRST, code c meaning d*(c-2). The kernel
// reads `(bits >> 2*j) & 0x3` and subtracts 2; if this layout ever changed, that shader would
// silently read transposed weights. Asserted against a hand-built byte, not a round trip, so the
// test cannot agree with a self-consistently wrong packer.
func TestQ2_0PackingBitLayoutIsFourCodesPerByteLowFirst(t *testing.T) {
	// Codes 0,1,2,3 in slots j=0..3 -> bits 0b11_10_01_00 = 0xE4 (low code first).
	const want = byte(0xE4)
	got := byte(0)
	for j, c := range []byte{0, 1, 2, 3} {
		got |= c << (2 * uint(j))
	}
	if got != want {
		t.Fatalf("low-code-first packing of codes {0,1,2,3} = %#02x, want %#02x", got, want)
	}

	// The dequant of that byte at d=1 must be the codes minus the offset 2, in slot order.
	dst := make([]float32, Q2_0BlockWeights)
	q := make([]byte, Q2_0BlockBytes)
	q[0] = want
	q2_0DequantBlock(dst, 1.0, q)
	for j, want := range []float32{-2, -1, 0, 1} {
		if dst[j] != want {
			t.Fatalf("dequant slot %d of byte %#02x at d=1 = %v, want %v (code-2 offset)", j, q[0], dst[j], want)
		}
	}
	// Every remaining slot comes from a zero byte -> code 0 -> value -2*d. This pins that the
	// offset is applied unconditionally (a zero byte is NOT a zero weight; only a zero SCALE is).
	for i := 4; i < Q2_0BlockWeights; i++ {
		if dst[i] != -2 {
			t.Fatalf("dequant slot %d of a zero byte at d=1 = %v, want -2 (offset applied unconditionally)", i, dst[i])
		}
	}

	// The scale multiplies the offset code: at d=3, code 3 -> 3*(3-2) = 3.
	q2_0DequantBlock(dst, 3.0, q)
	if dst[3] != 3 {
		t.Fatalf("dequant of code 3 at d=3 = %v, want 3 (d*(c-2))", dst[3])
	}
}

// TestQ2_0QuantizeEmitsTernaryCodeSet proves the headline of the ticket — that this really is a
// TERNARY path. With d = amax, every value in a non-zero block satisfies |v/d| <= 1, so
// round(v/d) lands in {-1,0,+1} and the emitted code c = round+2 lands in {1,2,3}. The 4th code
// (c=0, the -2d slot) is therefore UNREACHABLE, and the live code set is exactly {-1,0,+1}·d.
// If the scale formula ever drifted (say d = amax/2), codes would escape that set and the "ternary"
// claim — and the 1.71-bpw footprint argument resting on it — would be false.
func TestQ2_0QuantizeEmitsTernaryCodeSet(t *testing.T) {
	rng := rand.New(rand.NewSource(4873))
	dst := make([]byte, Q2_0BlockBytes)
	seen := map[int]bool{}
	for trial := 0; trial < 400; trial++ {
		src := make([]float32, Q2_0BlockWeights)
		nonZero := false
		for i := range src {
			// Spread of magnitudes and signs, occasionally exactly zero.
			if rng.Intn(8) == 0 {
				src[i] = 0
				continue
			}
			src[i] = float32(rng.NormFloat64()) * float32(1+rng.Intn(100))
			nonZero = true
		}
		if !nonZero {
			continue
		}
		d := q2_0QuantizeBlock(dst, src)
		if d <= 0 {
			t.Fatalf("trial %d: non-zero block got scale d=%v, want > 0", trial, d)
		}
		for _, b := range dst {
			for j := 0; j < 4; j++ {
				c := int((b >> (2 * uint(j))) & 0x3)
				seen[c] = true
				if c == 0 {
					t.Fatalf("trial %d: quantizer emitted code 0 (the -2d slot); with d=amax the code "+
						"set must be ternary {1,2,3} <-> {-1,0,+1}", trial)
				}
			}
		}
	}
	// Non-vacuous: all three ternary codes must actually occur across the corpus, or the assertion
	// above would pass trivially on a degenerate quantizer that only ever emitted code 2 (zero).
	for _, c := range []int{1, 2, 3} {
		if !seen[c] {
			t.Fatalf("ternary code %d never emitted across the corpus — the code-set assertion is vacuous", c)
		}
	}
}

// TestQ2_0ZeroBlockDequantizesToExactlyZero pins the one carve-out to the code-set rule above: an
// all-zero block short-circuits to d=0 with all-zero code bytes, so its codes ARE 0 (the otherwise
// unreachable slot). That is harmless precisely because d=0 makes every reconstructed weight
// 0*(0-2) = 0. The kernel relies on this: it applies the -2 offset unconditionally and lets the
// scale zero the block out.
func TestQ2_0ZeroBlockDequantizesToExactlyZero(t *testing.T) {
	dst := make([]byte, Q2_0BlockBytes)
	src := make([]float32, Q2_0BlockWeights) // all zero
	d := q2_0QuantizeBlock(dst, src)
	if d != 0 {
		t.Fatalf("zero block scale = %v, want exactly 0", d)
	}
	for i, b := range dst {
		if b != 0 {
			t.Fatalf("zero block code byte %d = %#02x, want 0", i, b)
		}
	}
	got := make([]float32, Q2_0BlockWeights)
	q2_0DequantBlock(got, d, dst)
	for i, v := range got {
		if v != 0 {
			t.Fatalf("zero block reconstructed weight %d = %v, want exactly 0", i, v)
		}
	}
}

// TestQ2_0RoundTripErrorBoundedByOneQuantum pins the honest cost of 2 bits: reconstruction error is
// bounded by half a quantum (d/2, since codes step by d), and the +/-amax peaks come back EXACTLY.
// This is the lossy-path discipline internal/model/quant_q2.go carries, restated for the payload
// this package uploads — it bounds what "within tolerance" is allowed to mean downstream.
func TestQ2_0RoundTripErrorBoundedByOneQuantum(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	dst := make([]byte, Q2_0BlockBytes)
	got := make([]float32, Q2_0BlockWeights)
	for trial := 0; trial < 200; trial++ {
		src := make([]float32, Q2_0BlockWeights)
		for i := range src {
			src[i] = float32(rng.NormFloat64())
		}
		// The block's TRUE peak magnitude is the scale the quantizer must choose. (Deriving it
		// rather than forcing one matters: a planted "peak" is not necessarily the max — a random
		// draw can exceed it, which is exactly what a forced-peak version of this test tripped on.)
		var amax float32
		for _, v := range src {
			a := v
			if a < 0 {
				a = -a
			}
			if a > amax {
				amax = a
			}
		}

		d := q2_0QuantizeBlock(dst, src)
		if d != amax {
			t.Fatalf("trial %d: scale d=%v, want the block's amax=%v", trial, d, amax)
		}
		q2_0DequantBlock(got, d, dst)

		peakExact := false
		for i := range src {
			err := math.Abs(float64(got[i] - src[i]))
			// Codes step by d, so round-to-nearest cannot err by more than half a quantum.
			if err > float64(d)/2+1e-5 {
				t.Fatalf("trial %d: weight %d round-tripped %v -> %v, error %v exceeds half a quantum %v",
					trial, i, src[i], got[i], err, float64(d)/2)
			}
			// An element AT the peak magnitude maps to code ±1 and must come back EXACTLY.
			if float32(math.Abs(float64(src[i]))) == amax {
				if got[i] != src[i] {
					t.Fatalf("trial %d: the peak %v did not reconstruct exactly; got %v (d=%v)",
						trial, src[i], got[i], d)
				}
				peakExact = true
			}
		}
		if !peakExact {
			t.Fatalf("trial %d: no element sat at the peak magnitude — the exact-peak check is vacuous", trial)
		}
	}
}

// TestQ2_0RefGEMVMatchesDenseDequantized is the load-bearing obligation: the reference GEMV must be
// a faithful contraction of the dequantized weights. It walks the blocked/packed layout, so a
// stride, block-offset, or row-major error would make it disagree with the dense matvec over the
// SAME reconstructed matrix. Both accumulate float32 in identical k order, so the agreement here is
// BIT-EXACT — no tolerance to hide a mistake in. This is what makes the Apple-side parity check
// meaningful: it compares Metal against a reference that is itself pinned to the dense truth.
func TestQ2_0RefGEMVMatchesDenseDequantized(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	shapes := [][2]int{{1, 32}, {4, 64}, {8, 128}, {33, 32}, {7, 256}, {64, 96}}
	for _, s := range shapes {
		out, in := s[0], s[1]
		w := make([]float32, out*in)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		x := make([]float32, in)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		codes, scales := q2_0Quantize(w, out, in)
		if len(codes) != out*(in/Q2_0BlockWeights)*Q2_0BlockBytes {
			t.Fatalf("shape %v: packed codes = %d bytes, want %d", s, len(codes), out*(in/Q2_0BlockWeights)*Q2_0BlockBytes)
		}

		dense := q2_0Dequantize(codes, scales, out, in)
		wantY := make([]float32, out)
		for o := 0; o < out; o++ {
			var acc float32
			for k := 0; k < in; k++ {
				acc += dense[o*in+k] * x[k]
			}
			wantY[o] = acc
		}

		gotY := q2_0RefGEMV(codes, scales, x, out, in)
		for o := 0; o < out; o++ {
			if gotY[o] != wantY[o] {
				t.Fatalf("shape %v row %d: ref GEMV = %v, dense dequantized matvec = %v (must be bit-exact)",
					s, o, gotY[o], wantY[o])
			}
			if math.IsNaN(float64(gotY[o])) || math.IsInf(float64(gotY[o]), 0) {
				t.Fatalf("shape %v row %d: ref GEMV produced non-finite %v", s, o, gotY[o])
			}
		}
	}
}

// TestQ2_0RefGEMVBatchMatchesPerRowGEMV pins the batch contract the Metal GEMVBatch must honor:
// n batched GEMVs of one weight are exactly n independent GEMVs, concatenated row-major. Batching
// is a submission-cost optimization and must be numerically invisible — if the Metal kernel ever
// mixed activation rows (a wrong per-dispatch X offset, the easiest bug in mg_q2_0_gemv_batch),
// this is the contract it would violate.
func TestQ2_0RefGEMVBatchMatchesPerRowGEMV(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	const out, in, n = 12, 64, 5
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	codes, scales := q2_0Quantize(w, out, in)

	Xcat := make([]float32, n*in)
	for i := range Xcat {
		Xcat[i] = float32(rng.NormFloat64())
	}
	gotCat := q2_0RefGEMVBatch(codes, scales, Xcat, n, out, in)
	for i := 0; i < n; i++ {
		want := q2_0RefGEMV(codes, scales, Xcat[i*in:(i+1)*in], out, in)
		for o := 0; o < out; o++ {
			if gotCat[i*out+o] != want[o] {
				t.Fatalf("batch row %d col %d = %v, want the standalone GEMV's %v",
					i, o, gotCat[i*out+o], want[o])
			}
		}
	}
	// Non-vacuous: distinct activation rows must give distinct results, or a kernel that ignored
	// the row offset entirely would still pass the equality above.
	distinct := false
	for o := 0; o < out; o++ {
		if gotCat[o] != gotCat[out+o] {
			distinct = true
			break
		}
	}
	if !distinct {
		t.Fatalf("batch rows 0 and 1 are identical — the per-row check is vacuous")
	}
}

// TestQ2_0ResidentFootprintIsThreeEighthsOfAByte pins the memory argument that motivates this whole
// path: 8 code bytes + one f32 scale per 32 weights = 0.375 B/weight. That is what puts a 27B model
// (~10 GB here) inside an Apple unified pool that f16 (~54 GB) and q4_k_m (~16 GB) overflow. The
// constants are the ones q2_0.go exports and q2_0.m's pointer arithmetic assumes.
func TestQ2_0ResidentFootprintIsThreeEighthsOfAByte(t *testing.T) {
	if Q2_0BlockWeights != 32 {
		t.Fatalf("Q2_0BlockWeights = %d, want 32", Q2_0BlockWeights)
	}
	if Q2_0BlockBytes != 8 {
		t.Fatalf("Q2_0BlockBytes = %d, want 8 (32 codes x 2 bits)", Q2_0BlockBytes)
	}
	if Q2_0BlockBytes*8 != Q2_0BlockWeights*2 {
		t.Fatalf("packed bits %d != %d codes x 2 bits", Q2_0BlockBytes*8, Q2_0BlockWeights*2)
	}
	bytesPerWeight := float64(Q2_0BlockBytes+4) / float64(Q2_0BlockWeights)
	if math.Abs(bytesPerWeight-0.375) > 1e-9 {
		t.Fatalf("resident footprint = %v B/weight, want 0.375", bytesPerWeight)
	}
	// The packed payload a caller must hand UploadQ2_0 has exactly that size, and Q2_0PayloadBytes
	// reports it without needing a Metal device.
	const out, in = 16, 128
	codes, scales := q2_0Quantize(make([]float32, out*in), out, in)
	want := int(bytesPerWeight * float64(out*in))
	if got := len(codes) + 4*len(scales); got != want {
		t.Fatalf("payload for [%d,%d] = %d B, want %d B (0.375 B/weight)", out, in, got, want)
	}
	if got := Q2_0PayloadBytes(out, in); got != want {
		t.Fatalf("Q2_0PayloadBytes(%d,%d) = %d, want %d", out, in, got, want)
	}
	// It refuses exactly the shapes UploadQ2_0 refuses, so a caller budgeting residency and a
	// caller uploading agree on what is representable.
	for _, bad := range [][2]int{{16, 48}, {0, 128}, {16, 0}, {-1, 128}} {
		if got := Q2_0PayloadBytes(bad[0], bad[1]); got != 0 {
			t.Fatalf("Q2_0PayloadBytes(%d,%d) = %d, want 0 (unrepresentable shape)", bad[0], bad[1], got)
		}
	}
}

// TestQ2_0QuantizeRejectsUnalignedReductionDim pins the guard that keeps a non-multiple-of-32
// reduction dim from silently dropping the tail — the same fail-loudly discipline
// internal/model.quantizeQ2 takes, and the precondition UploadQ2_0 enforces before it ever reaches
// the shader (whose pointer arithmetic assumes in = nblk*32).
func TestQ2_0QuantizeRejectsUnalignedReductionDim(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("q2_0Quantize with in=48 (not a multiple of 32) must panic, not silently truncate")
		}
	}()
	q2_0Quantize(make([]float32, 2*48), 2, 48)
}
