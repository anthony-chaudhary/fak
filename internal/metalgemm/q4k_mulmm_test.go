//go:build darwin && arm64 && cgo

package metalgemm

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"testing"
)

// Tolerances for the fak#13692 kernel_mul_mm port against the independent CPU oracle.
//
// The control arms (scalar Q4_K, naive Q6_K) compute in f32 end to end, so they keep the strict
// bounds every pre-#13692 Q4_K parity witness uses: cosine >= 0.999999 and maxRel <= 5e-3.
//
// mul_mm keeps f32 activations and f32 accumulators but rounds every dequantized weight to half
// before the simdgroup MMA. Round-to-nearest half has unit roundoff u = 2^-11 ~= 4.9e-4, so each
// weight carries an independent relative error of at most u (uniform, std u/sqrt(3) ~= 2.8e-4).
// For an output y = sum_i w_i*x_i the induced error is sum_i dw_i*x_i, whose standard deviation is
// ~2.8e-4 * ||w o x||_2, while |y| itself is ~||w o x||_2 for these mixed-sign activations. The
// maxRel metric (q4kTestCosineMaxRel) only scores outputs with |y| >= rms(y), so a typical scored
// element sits near 2.8e-4 relative error and the largest of up to ~2e5 scored elements (P=2048)
// is ~5 sigma, about 1.5e-3. That leaves the strict 5e-3 bound only ~3x headroom, so mul_mm gets an
// explicit 1e-2 maxRel (~35 sigma of the half-rounding model: a real indexing or scale bug lands at
// O(1) relative error, three orders of magnitude above it). Cosine follows the same model:
// 1 - cos ~= (rel)^2/2 ~= 4e-8, so 0.99999 is a 250x margin and still rejects any mis-tiled or
// mis-scaled block (a single wrong 16-weight chunk in a 512-wide row already costs > 1e-3).
const (
	kquantStrictMinCosine = 0.999999
	kquantStrictMaxRel    = 5e-3
	kquantMulMMMinCosine  = 0.99999
	kquantMulMMMaxRel     = 1e-2
)

const q6kTestBlockBytes = 210

// q6kIndependentReference is an independent CPU transcription of the 210-byte GGUF Q6_K block
// (ql[128] low nibbles, qh[64] high 2-bit pairs, scales[16] int8, d f16) as ggml's
// dequantize_row_q6_K defines it. It never calls a Metal kernel, so it proves both Metal Q6_K arms
// against the format rather than against each other.
func q6kIndependentReference(raw []byte, out, in int, x []float32) []float32 {
	nblk := in / 256
	y := make([]float32, out)
	w := make([]float32, 256)
	for o := 0; o < out; o++ {
		var acc float64
		for b := 0; b < nblk; b++ {
			base := (o*nblk + b) * q6kTestBlockBytes
			blk := raw[base : base+q6kTestBlockBytes]
			ql := blk[0:128]
			qh := blk[128:192]
			sc := blk[192:208]
			d := math.Float32frombits(q4kTestF16Bits(binary.LittleEndian.Uint16(blk[208:210])))
			for n := 0; n < 2; n++ { // two 128-weight halves
				qlh, qhh, sch := ql[64*n:], qh[32*n:], sc[8*n:]
				for l := 0; l < 32; l++ {
					is := l / 16
					q1 := int((qlh[l]&0x0f)|((qhh[l]>>0)&3)<<4) - 32
					q2 := int((qlh[l+32]&0x0f)|((qhh[l]>>2)&3)<<4) - 32
					q3 := int((qlh[l]>>4)|((qhh[l]>>4)&3)<<4) - 32
					q4 := int((qlh[l+32]>>4)|((qhh[l]>>6)&3)<<4) - 32
					w[128*n+l+0] = d * float32(int8(sch[is+0])) * float32(q1)
					w[128*n+l+32] = d * float32(int8(sch[is+2])) * float32(q2)
					w[128*n+l+64] = d * float32(int8(sch[is+4])) * float32(q3)
					w[128*n+l+96] = d * float32(int8(sch[is+6])) * float32(q4)
				}
			}
			xb := x[b*256 : (b+1)*256]
			for i := 0; i < 256; i++ {
				acc += float64(w[i]) * float64(xb[i])
			}
		}
		y[o] = float32(acc)
	}
	return y
}

// kquantTestPanel is the deterministic mixed-sign activation panel every mul_mm witness uses.
func kquantTestPanel(P, in int) []float32 {
	x := make([]float32, P*in)
	for i := range x {
		x[i] = float32((i*37)%251-125) / 127
	}
	return x
}

func kquantReferencePanel(ref func([]float32) []float32, x []float32, P, in, out int) []float32 {
	y := make([]float32, P*out)
	for p := 0; p < P; p++ {
		copy(y[p*out:(p+1)*out], ref(x[p*in:(p+1)*in]))
	}
	return y
}

// TestKQuantMulMMMatchesIndependentCPUOracle is the fak#13692 parity witness, modelled on
// TestQ4KM5CooperativeSMEMCandidateMatchesIndependentCPUOracle: Q4_K and Q6_K mul_mm against an
// independent CPU decode of the GGUF block layouts at P in {8,32,128,512,2048}, plus P=1, a P that
// leaves a partial 32-token tile (45 = 32+13), and row counts that leave a partial 64-row tile
// (96 = 64+32, and 40 < 64). The f32 control arm runs beside each candidate and is held to its
// strict pre-#13692 bound.
func TestKQuantMulMMMatchesIndependentCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	if bits := kquantMulMMBits(); bits&3 != 3 {
		t.Fatalf("mul_mm pipelines unavailable on a Metal host: ready bits=%b", bits)
	}
	defer ResetQ4K()

	const in = 512
	type shape struct{ out, P int }
	var shapes []shape
	for _, P := range []int{8, 32, 128, 512, 2048} {
		shapes = append(shapes, shape{96, P})
	}
	shapes = append(shapes, shape{96, 1}, shape{96, 45}, shape{40, 45}, shape{40, 1}, shape{128, 64})

	q4Weights := map[int]*Q4KWeight{}
	q4Raw := map[int][]byte{}
	q6Weights := map[int]*Q6KWeight{}
	q6Raw := map[int][]byte{}
	for _, s := range shapes {
		if _, ok := q4Weights[s.out]; ok {
			continue
		}
		q4Raw[s.out] = q4kTestRaw(s.out, in, 0x13692+uint64(s.out))
		q4Weights[s.out] = UploadQ4K(q4Raw[s.out], s.out, in)
		if q4Weights[s.out] == nil {
			t.Fatalf("UploadQ4K(%d,%d) returned nil", s.out, in)
		}
		defer q4Weights[s.out].Release()
		q6Raw[s.out] = q6kTestRaw(s.out, in, 0x6692+uint64(s.out))
		q6Weights[s.out] = UploadQ6K(q6Raw[s.out], s.out, in)
		if q6Weights[s.out] == nil {
			t.Fatalf("UploadQ6K(%d,%d) returned nil", s.out, in)
		}
		defer q6Weights[s.out].Release()
	}

	for _, s := range shapes {
		out, P := s.out, s.P
		t.Run(fmt.Sprintf("out%d_P%d", out, P), func(t *testing.T) {
			x := kquantTestPanel(P, in)

			// Q4_K: scalar control vs mul_mm vs the independent CPU oracle.
			raw4 := q4Raw[out]
			ref4 := kquantReferencePanel(func(xr []float32) []float32 { return q4kVectorizedReference(raw4, out, in, xr) }, x, P, in, out)
			control4 := make([]float32, P*out)
			if id := q4Weights[out].GEMMWithEventsMode(x, P, control4, nil, Q4KGEMMModeScalar); id != (Q4KGEMMIdentity{Requested: Q4KGEMMExecutedScalar, Executed: Q4KGEMMExecutedScalar}) {
				t.Fatalf("Q4_K scalar identity=%+v", id)
			}
			mulmm4 := make([]float32, P*out)
			wantMulMM := Q4KGEMMIdentity{Requested: Q4KGEMMExecutedMulMM, Executed: Q4KGEMMExecutedMulMM}
			if id := q4Weights[out].GEMMWithEventsMode(x, P, mulmm4, nil, Q4KGEMMModeMulMM); id != wantMulMM {
				t.Fatalf("Q4_K mul_mm identity=%+v want %+v", id, wantMulMM)
			}
			if ms := LastGEMMGPUMs(); !(ms > 0) || math.IsInf(ms, 0) {
				t.Fatalf("Q4_K mul_mm published a non-positive on-GPU window %g", ms)
			}
			kquantAssertParity(t, "Q4_K scalar control vs CPU oracle", ref4, control4, kquantStrictMinCosine, kquantStrictMaxRel)
			kquantAssertParity(t, "Q4_K mul_mm vs CPU oracle", ref4, mulmm4, kquantMulMMMinCosine, kquantMulMMMaxRel)
			kquantAssertParity(t, "Q4_K mul_mm vs scalar control", control4, mulmm4, kquantMulMMMinCosine, kquantMulMMMaxRel)

			// Q6_K: naive control vs mul_mm vs the independent CPU oracle.
			raw6 := q6Raw[out]
			ref6 := kquantReferencePanel(func(xr []float32) []float32 { return q6kIndependentReference(raw6, out, in, xr) }, x, P, in, out)
			control6 := make([]float32, P*out)
			if got := q6Weights[out].GEMMWithEventsMode(x, P, control6, nil, Q4KGEMMModeScalar); got != Q4KGEMMExecutedScalar {
				t.Fatalf("Q6_K naive executed=%v want %v", got, Q4KGEMMExecutedScalar)
			}
			mulmm6 := make([]float32, P*out)
			if got := q6Weights[out].GEMMWithEventsMode(x, P, mulmm6, nil, Q4KGEMMModeMulMM); got != Q4KGEMMExecutedMulMM {
				t.Fatalf("Q6_K mul_mm executed=%v want %v", got, Q4KGEMMExecutedMulMM)
			}
			if ms := LastGEMMGPUMs(); !(ms > 0) || math.IsInf(ms, 0) {
				t.Fatalf("Q6_K mul_mm published a non-positive on-GPU window %g", ms)
			}
			kquantAssertParity(t, "Q6_K naive control vs CPU oracle", ref6, control6, kquantStrictMinCosine, kquantStrictMaxRel)
			kquantAssertParity(t, "Q6_K mul_mm vs CPU oracle", ref6, mulmm6, kquantMulMMMinCosine, kquantMulMMMaxRel)
			kquantAssertParity(t, "Q6_K mul_mm vs naive control", control6, mulmm6, kquantMulMMMinCosine, kquantMulMMMaxRel)
		})
	}
}

func kquantAssertParity(t *testing.T, name string, want, got []float32, minCos, maxRel float64) {
	t.Helper()
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s: non-finite output[%d]=%g", name, i, v)
		}
	}
	cosine, rel := q4kTestCosineMaxRel(want, got)
	t.Logf("%s: cosine=%.9f maxRel=%.3e", name, cosine, rel)
	if cosine < minCos || rel > maxRel {
		t.Fatalf("%s: cosine=%.9f (min %g) maxRel=%.3e (max %g)", name, cosine, minCos, rel, maxRel)
	}
}

// TestKQuantMulMMUnavailableFailsClosed pins the deterministic fail-closed witnesses: a requested
// but unavailable mul_mm kernel reports the typed NotExecuted identity, records no command buffer,
// leaves LastGEMMGPUMs alone, and never touches the caller's output buffer.
func TestKQuantMulMMUnavailableFailsClosed(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	const out, in, P = 96, 512, 45
	x := kquantTestPanel(P, in)

	q4 := UploadQ4K(q4kTestRaw(out, in, 0x13692), out, in)
	if q4 == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer q4.Release()
	q6 := UploadQ6K(q6kTestRaw(out, in, 0x6692), out, in)
	if q6 == nil {
		t.Fatal("UploadQ6K returned nil")
	}
	defer q6.Release()

	marked := func() []float32 {
		y := make([]float32, P*out)
		for i := range y {
			y[i] = 13692
		}
		return y
	}

	// Q4_K.
	y4 := marked()
	before4 := slices.Clone(y4)
	gpuBefore := lastGEMMGPUMs.Load()
	obs4 := NewExecutionObservation(ExecutionQ4KGEMM)
	id := q4.GEMMWithEventsMode(x, P, y4, obs4, Q4KGEMMModeMulMMUnavailable)
	if id != (Q4KGEMMIdentity{Requested: Q4KGEMMExecutedMulMM, Executed: Q4KGEMMNotExecuted}) {
		t.Fatalf("Q4_K unavailable identity=%+v", id)
	}
	if want := Q4KGEMMIdentityForMode(P, Q4KGEMMModeMulMMUnavailable, Q4KGEMMNotExecuted); id != want {
		t.Fatalf("Q4_K unavailable identity=%+v, Q4KGEMMIdentityForMode=%+v", id, want)
	}
	if !slices.Equal(y4, before4) {
		t.Fatal("Q4_K unavailable mul_mm changed the output instead of failing closed")
	}
	if snap, err := obs4.Snapshot(); err != nil || len(snap.Events) != 0 {
		t.Fatalf("Q4_K unavailable mul_mm recorded events=%+v err=%v, want none", snap.Events, err)
	}
	if lastGEMMGPUMs.Load() != gpuBefore {
		t.Fatal("Q4_K unavailable mul_mm updated LastGEMMGPUMs")
	}

	// Q6_K.
	y6 := marked()
	before6 := slices.Clone(y6)
	obs6 := NewExecutionObservation(ExecutionQ6KGEMM)
	if got := q6.GEMMWithEventsMode(x, P, y6, obs6, Q4KGEMMModeMulMMUnavailable); got != Q4KGEMMNotExecuted {
		t.Fatalf("Q6_K unavailable executed=%v want %v", got, Q4KGEMMNotExecuted)
	}
	if !slices.Equal(y6, before6) {
		t.Fatal("Q6_K unavailable mul_mm changed the output instead of failing closed")
	}
	if snap, err := obs6.Snapshot(); err != nil || len(snap.Events) != 0 {
		t.Fatalf("Q6_K unavailable mul_mm recorded events=%+v err=%v, want none", snap.Events, err)
	}
	if lastGEMMGPUMs.Load() != gpuBefore {
		t.Fatal("Q6_K unavailable mul_mm updated LastGEMMGPUMs")
	}
	// An undefined mode is not a fail-closed witness: the unified Q6_K contract runs the naive
	// q6k_gemm for every mode other than mul_mm / small-P, reported as executed scalar.
	yUndef := make([]float32, P*out)
	if got := q6.GEMMWithEventsMode(x, P, yUndef, nil, Q4KGEMMMode(7)); got != Q4KGEMMExecutedScalar {
		t.Fatalf("Q6_K undefined mode executed=%v, want naive (scalar)", got)
	}
	// A short output or nil weight never executes.
	if got := q6.GEMMWithEventsMode(x, P, y6[:len(y6)-1], nil, Q4KGEMMModeMulMM); got != Q4KGEMMNotExecuted {
		t.Fatalf("Q6_K short output executed=%v", got)
	}
	var nilW *Q6KWeight
	if got := nilW.GEMMWithEventsMode(x, P, y6, nil, Q4KGEMMModeMulMM); got != Q4KGEMMNotExecuted {
		t.Fatalf("nil Q6_K weight executed=%v", got)
	}
}

// TestKQuantMulMMProductionRoutingContract pins the production selector after fak#13692 on top of
// the fak#13694 small-P GEMV: mul_mm for Q4_K and Q6_K at P >= the floor (8 for both), the small-P
// GEMV for 2..7 below it, scalar/naive at P=1; and SetGEMMUseMulMM(false) as the kill switch that
// restores the pre-#13692 selector (small-P for 2..20, then MM32/M5, then scalar/naive).
// Global switches are saved and restored.
func TestKQuantMulMMProductionRoutingContract(t *testing.T) {
	if Q4KMulMMMinPrompt != 8 || Q6KMulMMMinPrompt != 8 {
		t.Fatalf("receipted floors changed: Q4_K=%d Q6_K=%d, want 8 and 8", Q4KMulMMMinPrompt, Q6KMulMMMinPrompt)
	}
	// The process default is ON (no test in this package may leak a disabled switch).
	if !GEMMUseMulMM() {
		t.Fatal("GEMMUseMulMM()=false at test entry; the fak#13692 default is ON")
	}
	priorMulMM := GEMMUseMulMM()
	defer SetGEMMUseMulMM(priorMulMM)
	priorSmallP := q4kUseSmallP.Swap(true)
	defer q4kUseSmallP.Store(priorSmallP)
	priorMM := q4kUseMM.Swap(false)
	defer q4kUseMM.Store(priorMM)
	priorM5 := q4kUseM5.Swap(false)
	defer q4kUseM5.Store(priorM5)

	bits := kquantMulMMBits()
	if Available() && bits&3 != 3 {
		t.Fatalf("mul_mm pipelines unavailable on a Metal host: ready bits=%b", bits)
	}
	ready := map[string]bool{"q4_k": bits&1 != 0, "q6_k": bits&2 != 0}

	// want is the independent expectation of the selector order for one format.
	want := func(format string, P int, mulMM bool) Q4KGEMMMode {
		switch {
		case mulMM && ready[format] && P >= 8:
			return Q4KGEMMModeMulMM
		case P >= 2 && P <= 20:
			return Q4KGEMMModeSmallPGEMV
		default:
			return Q4KGEMMModeScalar
		}
	}
	prompts := []int{1, 2, 7, 8, 9, 16, 20, 21, 31, 32, 33, 64, 128, 512, 2048}
	for _, mulMM := range []bool{true, false} {
		SetGEMMUseMulMM(mulMM)
		if GEMMUseMulMM() != mulMM {
			t.Fatalf("SetGEMMUseMulMM(%t) left GEMMUseMulMM()=%t", mulMM, GEMMUseMulMM())
		}
		for _, P := range prompts {
			w4 := want("q4_k", P, mulMM)
			if got := Q4KGEMMModeForPrompt(P); got != w4 {
				t.Fatalf("mulMM=%t Q4KGEMMModeForPrompt(%d)=%v want %v", mulMM, P, got, w4)
			}
			wantReq := Q4KGEMMExecutedScalar
			switch w4 {
			case Q4KGEMMModeMulMM:
				wantReq = Q4KGEMMExecutedMulMM
			case Q4KGEMMModeSmallPGEMV:
				wantReq = Q4KGEMMExecutedSmallPGEMV
			}
			if got := Q4KGEMMRequestedExecution(P); got != wantReq {
				t.Fatalf("mulMM=%t Q4KGEMMRequestedExecution(%d)=%v want %v", mulMM, P, got, wantReq)
			}
			if w6, got := want("q6_k", P, mulMM), Q6KGEMMModeForPrompt(P); got != w6 {
				t.Fatalf("mulMM=%t Q6KGEMMModeForPrompt(%d)=%v want %v", mulMM, P, got, w6)
			}
		}
	}

	// With mul_mm on, it outranks every older opt-in at and above the floor.
	SetGEMMUseMulMM(true)
	q4kUseMM.Store(true)
	q4kUseM5.Store(true)
	for _, P := range []int{8, 20, 32, 64, 128} {
		if ready["q4_k"] && Q4KGEMMModeForPrompt(P) != Q4KGEMMModeMulMM {
			t.Fatalf("P=%d with MM32/M5/small-P opt-ins on did not keep the mul_mm default", P)
		}
	}
	q4kUseMM.Store(false)
	q4kUseM5.Store(false)

	// Kill switch: the pre-#13692 opt-ins become reachable again exactly as before.
	SetGEMMUseMulMM(false)
	q4kUseSmallP.Store(false)
	for _, P := range prompts {
		if got := Q4KGEMMModeForPrompt(P); got != Q4KGEMMModeScalar {
			t.Fatalf("kill switch + small-P off: Q4KGEMMModeForPrompt(%d)=%v want scalar", P, got)
		}
		if got := Q6KGEMMModeForPrompt(P); got != Q4KGEMMModeScalar {
			t.Fatalf("kill switch + small-P off: Q6KGEMMModeForPrompt(%d)=%v want naive (scalar)", P, got)
		}
	}
	q4kUseMM.Store(true)
	if Available() && Q4KGEMMModeForPrompt(32) != Q4KGEMMModeMM32 {
		t.Fatalf("kill switch + FAK_Q4K_MM: P=32 selected %v want MM32", Q4KGEMMModeForPrompt(32))
	}
	q4kUseMM.Store(false)
	q4kUseM5.Store(true)
	for _, P := range []int{64, 128, 256} {
		w := Q4KGEMMModeScalar
		if Q4KM5CrossoverAdmits(P) {
			w = Q4KGEMMModeM5CooperativeSMEM
		}
		if got := Q4KGEMMModeForPrompt(P); got != w {
			t.Fatalf("kill switch + M5: P=%d selected %v want %v", P, got, w)
		}
	}
	q4kUseM5.Store(false)
	q4kUseSmallP.Store(true)

	if !Available() {
		return
	}
	// Live production seams: the default paths execute the selected kernel.
	defer ResetQ4K()
	const out, in = 96, 512
	q4 := UploadQ4K(q4kTestRaw(out, in, 0x13692), out, in)
	if q4 == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer q4.Release()
	q6 := UploadQ6K(q6kTestRaw(out, in, 0x6692), out, in)
	if q6 == nil {
		t.Fatal("UploadQ6K returned nil")
	}
	defer q6.Release()
	for _, tc := range []struct {
		mulMM bool
		P     int
		want  Q4KGEMMExecution
	}{
		{true, 1, Q4KGEMMExecutedScalar},
		{true, 7, Q4KGEMMExecutedSmallPGEMV},
		{true, 8, Q4KGEMMExecutedMulMM},
		{true, 20, Q4KGEMMExecutedMulMM},
		{true, 32, Q4KGEMMExecutedMulMM},
		{false, 8, Q4KGEMMExecutedSmallPGEMV},
		{false, 20, Q4KGEMMExecutedSmallPGEMV},
		{false, 32, Q4KGEMMExecutedScalar},
	} {
		SetGEMMUseMulMM(tc.mulMM)
		x := kquantTestPanel(tc.P, in)
		y := make([]float32, tc.P*out)
		if id := q4.GEMMWithEvents(x, tc.P, y, nil); id != (Q4KGEMMIdentity{Requested: tc.want, Executed: tc.want}) {
			t.Fatalf("mulMM=%t P=%d default Q4_K identity=%+v want %v", tc.mulMM, tc.P, id, tc.want)
		}
		if got := q6.GEMMWithEventsMode(x, tc.P, y, nil, Q6KGEMMModeForPrompt(tc.P)); got != tc.want {
			t.Fatalf("mulMM=%t P=%d default Q6_K executed=%v want %v", tc.mulMM, tc.P, got, tc.want)
		}
	}
}

// TestKQuantMulMMEncodesIntoGraphCommandBuffer proves the mul_mm kernels join a graph-owned
// command buffer: a chained EncodeQ4K -> EncodeQ4KFrom plus EncodeQ6K / EncodeQ6KFrom in ONE
// ProjectionGraph with the mul_mm modes, committed once, matches the standalone mul_mm results
// bit for bit (same kernel, same inputs), and the graph refuses mode changes after the first encode.
func TestKQuantMulMMEncodesIntoGraphCommandBuffer(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	const in, mid, out = 512, 256, 96
	wA := UploadQ4K(q4kTestRaw(mid, in, 0xA692), mid, in)
	wB := UploadQ4K(q4kTestRaw(out, mid, 0xB692), out, mid)
	w6 := UploadQ6K(q6kTestRaw(out, in, 0xC692), out, in)
	w6b := UploadQ6K(q6kTestRaw(out, mid, 0xD692), out, mid)
	if wA == nil || wB == nil || w6 == nil || w6b == nil {
		t.Fatal("weight upload returned nil")
	}
	defer wA.Release()
	defer wB.Release()
	defer w6.Release()
	defer w6b.Release()

	for _, P := range []int{45, 128} {
		t.Run(fmt.Sprintf("P%d", P), func(t *testing.T) {
			x := kquantTestPanel(P, in)

			// Standalone mul_mm chain.
			ya := make([]float32, P*mid)
			mulmm := Q4KGEMMIdentity{Requested: Q4KGEMMExecutedMulMM, Executed: Q4KGEMMExecutedMulMM}
			if id := wA.GEMMWithEventsMode(x, P, ya, nil, Q4KGEMMModeMulMM); id != mulmm {
				t.Fatalf("standalone A identity=%+v", id)
			}
			yb := make([]float32, P*out)
			if id := wB.GEMMWithEventsMode(ya, P, yb, nil, Q4KGEMMModeMulMM); id != mulmm {
				t.Fatalf("standalone B identity=%+v", id)
			}
			y6 := make([]float32, P*out)
			if got := w6.GEMMWithEventsMode(x, P, y6, nil, Q4KGEMMModeMulMM); got != Q4KGEMMExecutedMulMM {
				t.Fatalf("standalone Q6_K executed=%v", got)
			}
			y6b := make([]float32, P*out)
			if got := w6b.GEMMWithEventsMode(ya, P, y6b, nil, Q4KGEMMModeMulMM); got != Q4KGEMMExecutedMulMM {
				t.Fatalf("standalone Q6_K-from executed=%v", got)
			}

			g, err := BeginProjectionGraph(x, nil, nil, P, in)
			if err != nil {
				t.Fatal(err)
			}
			defer g.Free()
			if g.Q4KGEMMMode() != Q4KGEMMModeScalar || g.Q6KGEMMMode() != Q4KGEMMModeScalar {
				t.Fatalf("fresh graph modes=%v/%v, want scalar/naive", g.Q4KGEMMMode(), g.Q6KGEMMMode())
			}
			if g.SetQ6KGEMMMode(Q4KGEMMModeMulMMUnavailable) || g.Q6KGEMMMode() != Q4KGEMMModeScalar {
				t.Fatal("graph accepted the unavailable Q6_K mode or changed state")
			}
			// Q6_K accepts only scalar or mul_mm on its own; small-P only rides a small-P Q4_K graph.
			for _, m := range []Q4KGEMMMode{Q4KGEMMModeSmallPGEMV, Q4KGEMMModeM5CooperativeSMEM, Q4KGEMMModeMM32} {
				if g.SetQ6KGEMMMode(m) || g.Q6KGEMMMode() != Q4KGEMMModeScalar {
					t.Fatalf("graph accepted Q6_K mode %v without a small-P Q4_K graph, or changed state", m)
				}
			}
			if g.SetQ4KGEMMMode(Q4KGEMMModeMulMMUnavailable) || g.Q4KGEMMMode() != Q4KGEMMModeScalar {
				t.Fatal("graph accepted the unavailable Q4_K mode or changed state")
			}
			if !g.SetQ4KGEMMMode(Q4KGEMMModeMulMM) || g.Q4KGEMMMode() != Q4KGEMMModeMulMM {
				t.Fatalf("graph refused Q4_K mul_mm: mode=%v", g.Q4KGEMMMode())
			}
			if !g.SetQ6KGEMMMode(Q4KGEMMModeMulMM) || g.Q6KGEMMMode() != Q4KGEMMModeMulMM {
				t.Fatalf("graph refused Q6_K mul_mm: mode=%v", g.Q6KGEMMMode())
			}
			ra, err := g.EncodeQ4K(wA)
			if err != nil {
				t.Fatalf("EncodeQ4K: %v", err)
			}
			// Mode changes are refused once anything is encoded, and the modes stay put.
			if g.SetQ4KGEMMMode(Q4KGEMMModeScalar) || g.Q4KGEMMMode() != Q4KGEMMModeMulMM {
				t.Fatal("graph accepted a Q4_K mode change after encode")
			}
			if g.SetQ6KGEMMMode(Q4KGEMMModeScalar) || g.Q6KGEMMMode() != Q4KGEMMModeMulMM {
				t.Fatal("graph accepted a Q6_K mode change after encode")
			}
			rb, err := g.EncodeQ4KFrom(wB, ra)
			if err != nil {
				t.Fatalf("EncodeQ4KFrom: %v", err)
			}
			r6, err := g.EncodeQ6K(w6)
			if err != nil {
				t.Fatalf("EncodeQ6K: %v", err)
			}
			r6b, err := g.EncodeQ6KFrom(w6b, ra)
			if err != nil {
				t.Fatalf("EncodeQ6KFrom: %v", err)
			}
			got, _, err := g.FinishRead(ra, rb, r6, r6b)
			if err != nil {
				t.Fatalf("FinishRead: %v", err)
			}
			for i, pair := range []struct {
				name       string
				want, have []float32
			}{
				{"EncodeQ4K", ya, got[0]},
				{"EncodeQ4KFrom", yb, got[1]},
				{"EncodeQ6K", y6, got[2]},
				{"EncodeQ6KFrom", y6b, got[3]},
			} {
				if len(pair.want) != len(pair.have) {
					t.Fatalf("%s: len=%d want %d", pair.name, len(pair.have), len(pair.want))
				}
				for j := range pair.want {
					if math.Float32bits(pair.want[j]) != math.Float32bits(pair.have[j]) {
						t.Fatalf("result %d %s[%d]=%g, standalone mul_mm %g", i, pair.name, j, pair.have[j], pair.want[j])
					}
				}
			}
		})
	}
}
