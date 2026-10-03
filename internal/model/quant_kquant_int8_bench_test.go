package model

import (
	"encoding/binary"
	"math"
	"testing"
)

// BenchmarkQ5K{F32,Int8}GEMV A/B the two Q5_K decode GEMV paths on a GLM-5.2-shaped expert row set
// (out=2048 expert-intermediate, in=6144 hidden). The f32 path dequants 256 f32 per super-block per
// row; the int8 path quantizes the activation once and does the compact integer reduction. Run both
// to read the per-call ns and the int8 speedup. Single-worker (parThreshold/parFor folded out) so the
// bench measures the kernel, not the scheduler.
func benchKQuantFixture(b *testing.B, out, in int) (*kQuantTensor, []float32) {
	b.Helper()
	nblk := in / qkK
	bb := kindQ5K.blockBytes()
	raw := make([]byte, out*nblk*bb)
	lcgBytes(raw, 0xfeedface12345678)
	for o := 0; o < out; o++ {
		for bk := 0; bk < nblk; bk++ {
			blk := raw[(o*nblk+bk)*bb:]
			binary.LittleEndian.PutUint16(blk[0:], f16One) // d
			binary.LittleEndian.PutUint16(blk[2:], 0)      // min
		}
	}
	qt := quantizeKQuantFromRaw(raw, out, in, kindQ5K)
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*13)%29) - 14
	}
	return qt, x
}

func BenchmarkQ5KF32GEMV(b *testing.B) {
	qt, x := benchKQuantFixture(b, 2048, 6144)
	y := make([]float32, qt.out)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kQuantMatRowsRange(qt, x, y, 0, qt.out)
	}
}

func BenchmarkQ5KInt8GEMV(b *testing.B) {
	qt, x := benchKQuantFixture(b, 2048, 6144)
	y := make([]float32, qt.out)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qv := quantizeVecQ8(x)
		q5kMatRowsRangeInt8(qt, qv, y, 0, qt.out)
	}
}

// BenchmarkQ2K{F32,Int8Kernel,Int8EndToEnd}GEMV are HOST-ONLY microbenchmark arms for the Q2_K
// decode GEMV. They are NOT Strix hardware evidence and justify no performance claim: they run on
// whatever host compiled the package and only A/B the arithmetic paths under a matched fixture.
//
// The two MATCHED arms drive the SAME production caller kQuantMatRowsIntoWorkers with the SAME
// captured kQuantDecodeWorkers() budget: BenchmarkQ2KF32GEMV is the production gate-OFF f32
// reference and BenchmarkQ2KInt8EndToEndGEMV is the production gate-ON int8 arm (which charges
// quantizeVecQ8 on every timed iteration). BenchmarkQ2KInt8KernelGEMV is DIAGNOSTIC-ONLY: it calls
// the raw kernel q2kMatRowsRangeInt8 with an activation pre-quantized once outside the timer, so
// it measures the compact integer reduction in isolation and is NOT the matched production
// comparison.
//
// Arming the int8 path uses setKQuantSDOTForTest only — kQuantSDOTDefault and the FAK_KQ_INT8 env
// var are never touched, so the production default-off posture is preserved. Each matched arm
// proves its production-caller seam bit-exact with q2kSameBits before timing (caller reachability
// under the gate; byte-identical f32 fallback with the gate off), computes an independent expected
// oracle outside the timer, zeroes the output immediately before the timed loop, and after timing
// bit-compares the timed output to that oracle. A no-op or wrong-output timed loop therefore fails
// loudly instead of silently reporting a near-zero number; every arm also asserts a finite,
// non-degenerate (non-all-zero) result. All setup, oracle computation, and validation stay outside
// the timed region — only the for i := 0; i < b.N; i++ { ... } loop is timed.
//
// Shape provenance (REAL, not synthetic): the pinned artifact unsloth/Qwen3.8-27B-GGUF revision
// 4ca720788d1e01f1bff70c033e0d0028fd02e502, file Qwen3.8-27B-UD-Q2_K_XL.gguf, has
// HiddenSize=5120, IntermediateSize=17408, NumLayers=64, VocabSize=248320, dense (NumExperts=0),
// and 16 real Q2_K tensors. The most frequent Q2_K tensor is blk.<n>.attn_q.weight, GGUF dims
// [5120, 12288] (reduction in=5120, rows out=12288), 7 of the 16 tensors. All three arms pin that
// dominant class: out=12288, in=5120 (both multiples of qkK=256).
const (
	benchQ2KOut = 12288
	benchQ2KIn  = 5120
)

// benchQ2KFixture builds a resident Q2_K tensor and a deterministic non-trivial activation at the
// pinned real shape. Random LCG bytes give non-zero 2-bit codes/scales; pinResidentQuantScales then
// forces d = 1.0 (f16One) and min = 0 per super-block so every output is finite (no Inf/NaN) and
// the bit-exact guards below are meaningful.
func benchQ2KFixture(b *testing.B, out, in int) (*kQuantTensor, []float32) {
	b.Helper()
	raw := make([]byte, out*(in/qkK)*q2kBlockBytes)
	lcgBytes(raw, 0xfeedface12345678)
	pinResidentQuantScales(raw, out, in/qkK, kindQ2K)
	qt := quantizeKQuantFromRaw(raw, out, in, kindQ2K)
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*13)%29) - 14
	}
	// Fixture-level non-vacuity: reject a degenerate activation or all-zero quantized weight.
	// This runs once, before timing, and makes a silently-degenerate fixture fail loudly.
	activationNonZero := false
	for _, v := range x {
		if v != 0 {
			activationNonZero = true
			break
		}
	}
	if !activationNonZero {
		b.Fatal("Q2_K fixture activation is all-zero; bench would silently no-op")
	}
	weightNonZero := false
	for _, by := range raw {
		if by != 0 {
			weightNonZero = true
			break
		}
	}
	if !weightNonZero {
		b.Fatal("Q2_K fixture quantized weight is all-zero; bench would silently no-op")
	}
	return qt, x
}

// BenchmarkQ2KF32GEMV is the MATCHED F32 reference: the production caller kQuantMatRowsIntoWorkers
// with the int8 gate forced OFF, so it measures the byte-identical f32 fallback of the same code
// path the end-to-end arm drives. It shares the SAME captured kQuantDecodeWorkers() budget.
func BenchmarkQ2KF32GEMV(b *testing.B) {
	// save+restore the exact prior force so no arm can leak the gate for later benchmarks.
	prev := kQuantSDOTForce
	b.Cleanup(func() { kQuantSDOTForce = prev })
	setKQuantSDOTForTest(false)
	qt, x := benchQ2KFixture(b, benchQ2KOut, benchQ2KIn)
	budget := kQuantDecodeWorkers() // captured ONCE, shared by the timed path and the oracle
	// Pre-timing oracle: the production gate-OFF caller must produce finite, non-all-zero output.
	want := make([]float32, qt.out)
	kQuantMatRowsIntoWorkers(qt, x, want, budget)
	nonZero := false
	for _, v := range want {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			b.Fatalf("Q2_K f32 fixture produced non-finite output %v", v)
		}
		if v != 0 {
			nonZero = true
		}
	}
	if !nonZero {
		b.Fatal("Q2_K f32 fixture produced all-zero output; bench would silently no-op")
	}
	// Poison y before the timed loop so the post-timing oracle proves the loop actually wrote.
	y := make([]float32, qt.out)
	for i := range y {
		y[i] = 0
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kQuantMatRowsIntoWorkers(qt, x, y, budget)
	}
	b.StopTimer()
	// Post-timing EXACT oracle: the timed production gate-OFF path must have written the
	// byte-identical f32 result computed outside the timer.
	if !q2kSameBits(y, want) {
		b.Fatal("Q2_K f32 timed production path != pre-timing f32 oracle")
	}
	nonZero = false
	for _, v := range y {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			b.Fatalf("Q2_K f32 timed loop produced non-finite output %v", v)
		}
		if v != 0 {
			nonZero = true
		}
	}
	if !nonZero {
		b.Fatal("Q2_K f32 timed loop wrote no output; bench would silently no-op")
	}
}

// BenchmarkQ2KInt8KernelGEMV is DIAGNOSTIC-ONLY: it calls the raw kernel q2kMatRowsRangeInt8 with an
// activation pre-quantized ONCE outside the timed loop, measuring the compact integer reduction in
// isolation. It is NOT the matched production comparison — the matched int8 arm is
// BenchmarkQ2KInt8EndToEndGEMV, which drives the production caller and charges quantizeVecQ8 per
// iteration. Reported numbers here are a lower-bound kernel-only diagnostic, not a claim.
func BenchmarkQ2KInt8KernelGEMV(b *testing.B) {
	// save+restore the exact prior force so no arm can leak the gate for later benchmarks.
	prev := kQuantSDOTForce
	b.Cleanup(func() { kQuantSDOTForce = prev })
	setKQuantSDOTForTest(true)
	qt, x := benchQ2KFixture(b, benchQ2KOut, benchQ2KIn)
	qv := quantizeVecQ8(x) // pre-quantized ONCE, outside the timed loop: kernel-only diagnostic
	// Pre-timing oracle: the raw kernel must produce finite, non-all-zero output.
	want := make([]float32, qt.out)
	q2kMatRowsRangeInt8(qt, qv, want, 0, qt.out)
	nonZero := false
	for _, v := range want {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			b.Fatalf("Q2_K int8 kernel fixture produced non-finite output %v", v)
		}
		if v != 0 {
			nonZero = true
		}
	}
	if !nonZero {
		b.Fatal("Q2_K int8 kernel fixture produced all-zero output; bench would silently no-op")
	}
	// Poison y before the timed loop so the post-timing oracle proves the loop actually wrote.
	y := make([]float32, qt.out)
	for i := range y {
		y[i] = 0
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q2kMatRowsRangeInt8(qt, qv, y, 0, qt.out)
	}
	b.StopTimer()
	// Post-timing EXACT oracle: bit-compare the timed output to the independently computed kernel
	// result; a no-op or zero-writing timed loop fails here.
	if !q2kSameBits(y, want) {
		b.Fatal("Q2_K int8 kernel timed loop != pre-timing kernel oracle")
	}
	nonZero = false
	for _, v := range y {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			b.Fatalf("Q2_K int8 kernel timed loop produced non-finite output %v", v)
		}
		if v != 0 {
			nonZero = true
		}
	}
	if !nonZero {
		b.Fatal("Q2_K int8 kernel timed loop wrote no output; bench would silently no-op")
	}
}

// BenchmarkQ2KInt8EndToEndGEMV is the MATCHED int8 arm: it drives the PRODUCTION caller
// kQuantMatRowsIntoWorkers with the gate ON and shares the SAME captured kQuantDecodeWorkers()
// budget as BenchmarkQ2KF32GEMV, charging quantizeVecQ8 (activation quantization) on every timed
// iteration. Reachability and the byte-identical gate-OFF fallback are proven before timing.
func BenchmarkQ2KInt8EndToEndGEMV(b *testing.B) {
	// save+restore the exact prior force so no arm can leak the gate for later benchmarks.
	// Registered BEFORE the reachability assert so a b.Fatal there cannot leak the gate.
	prev := kQuantSDOTForce
	b.Cleanup(func() { kQuantSDOTForce = prev })
	qt, x := benchQ2KFixture(b, benchQ2KOut, benchQ2KIn)
	budget := kQuantDecodeWorkers() // captured ONCE, shared by both matched arms

	// (1) Caller reachability under the gate: the production caller must route kindQ2K to the
	// int8 kernel, bit-identical to a direct q2kMatRowsRangeInt8Raw over the quantized activation.
	setKQuantSDOTForTest(true)
	gotOn := make([]float32, qt.out)
	kQuantMatRowsIntoWorkers(qt, x, gotOn, budget)
	direct := make([]float32, qt.out)
	q2kMatRowsRangeInt8Raw(qt.raw, qt, quantizeVecQ8(x), direct, 0, qt.out)
	if !q2kSameBits(gotOn, direct) {
		b.Fatal("Q2_K caller reachability: production caller (gate ON) != direct int8 kernel")
	}
	// Non-vacuous guard: bit-equality alone would pass on a degenerate (all-zero/zero==zero)
	// fixture, so assert the shared result is finite and has at least one non-zero element.
	nonZero := false
	for _, v := range gotOn {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			b.Fatalf("Q2_K end-to-end fixture produced non-finite output %v", v)
		}
		if v != 0 {
			nonZero = true
		}
	}
	if !nonZero {
		b.Fatal("Q2_K end-to-end fixture produced all-zero output; bench would silently no-op")
	}

	// (2) Default-off invariant: with the gate off the production caller must fall back to the
	// byte-identical f32 path (never silently int8).
	setKQuantSDOTForTest(false)
	gotOff := make([]float32, qt.out)
	kQuantMatRowsIntoWorkers(qt, x, gotOff, budget)
	ref := make([]float32, qt.out)
	kQuantMatRowsRange(qt, x, ref, 0, qt.out)
	if !q2kSameBits(gotOff, ref) {
		b.Fatal("Q2_K default-off invariant: gate OFF caller != f32 reference")
	}

	// (3) Arm under test: gate ON, production caller, charges quantizeVecQ8 per iteration.
	setKQuantSDOTForTest(true)
	// Poison y before the timed loop so the post-timing oracle proves the loop actually wrote.
	y := make([]float32, qt.out)
	for i := range y {
		y[i] = 0
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kQuantMatRowsIntoWorkers(qt, x, y, budget)
	}
	b.StopTimer()
	// Post-timing EXACT oracle: the timed production gate-ON path charges quantizeVecQ8 and must
	// have written the exact int8-kernel result computed outside the timer.
	if !q2kSameBits(y, direct) {
		b.Fatal("Q2_K end-to-end timed production path != pre-timing int8 oracle")
	}
	nonZero = false
	for _, v := range y {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			b.Fatalf("Q2_K end-to-end timed loop produced non-finite output %v", v)
		}
		if v != 0 {
			nonZero = true
		}
	}
	if !nonZero {
		b.Fatal("Q2_K end-to-end timed loop wrote no output; bench would silently no-op")
	}
}
