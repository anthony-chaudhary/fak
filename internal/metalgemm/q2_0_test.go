//go:build darwin && arm64 && cgo

// Apple-Silicon parity tests for the Metal ternary (Q2_0) GEMV — the done-condition witness for
// issue #4873: "Metal Q2_0 GEMV/GEMVBatch runs on Apple Silicon and matches CPU-ref ternary GEMM
// within tolerance." Run with `go test ./internal/metalgemm -run Q2_0` on an Apple host.
//
// The CPU-ref is q2_0_ref_test.go (shared with the stub build, which pins the reference's own math
// obligations in q2_0_witness_test.go — including that the reference is a bit-exact contraction of
// the dense dequantized weights). So a pass here means the shader agrees with a reference that is
// itself pinned to the dense truth, not merely with a second copy of the same assumption.
//
// Tolerance: the kernel factors the block scale d out of the block sum and reduces across 32 lanes
// with simd_sum, while the reference dequantizes to d*(c-2) first and accumulates sequentially. The
// codes are exact small integers, so the only divergence is float association order — a relative
// tolerance well under the 2-bit quantization error the path already carries.
package metalgemm

import (
	"math"
	"math/rand"
	"testing"
)

// q2_0CloseEnough reports whether got and want agree to within a mixed relative/absolute
// tolerance. Reductions of ~in terms in float32 diverge by association order alone, so the bar is
// scaled to the magnitude of the result rather than a bare epsilon.
func q2_0CloseEnough(got, want float32) bool {
	d := math.Abs(float64(got - want))
	if d <= 1e-4 {
		return true
	}
	return d <= 1e-4*math.Max(1, math.Abs(float64(want)))
}

// q2_0RandomWeight builds a random [out,in] f32 matrix, packs it ternary, and returns the payload
// alongside the reference-visible pair.
func q2_0RandomWeight(rng *rand.Rand, out, in int) (codes []byte, scales []float32) {
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	return q2_0Quantize(w, out, in)
}

// TestMetalQ2_0GemvMatchesCPU is the issue #4873 done-condition witness for the decode GEMV: the
// resident ternary weight runs on the GPU with in-shader 2-bit unpack and reproduces the CPU-ref
// ternary GEMM within tolerance, across a spread of shapes (including nblk < 32 and nblk > 32, the
// two sides of the 32-lane block-striding loop in the shader).
func TestMetalQ2_0GemvMatchesCPU(t *testing.T) {
	if !Available() {
		t.Skip("no Metal device (issue #4873 witness requires an Apple-Silicon host)")
	}
	defer ResetQ2_0()
	rng := rand.New(rand.NewSource(4873))

	// in/32 = nblk: 1, 2, 4, 32, 40 — striding the lanes under-, exactly-, and over-subscribed.
	shapes := [][2]int{{1, 32}, {8, 64}, {17, 128}, {64, 1024}, {5, 1280}, {128, 256}}
	for _, s := range shapes {
		out, in := s[0], s[1]
		codes, scales := q2_0RandomWeight(rng, out, in)
		w := UploadQ2_0(codes, scales, out, in)
		if w == nil {
			t.Fatalf("shape [%d,%d]: UploadQ2_0 returned nil on an available device", out, in)
		}
		if w.Out != out || w.In != in || w.Nblk != in/Q2_0BlockWeights {
			t.Fatalf("shape [%d,%d]: handle dims = (Out=%d,In=%d,Nblk=%d)", out, in, w.Out, w.In, w.Nblk)
		}

		x := make([]float32, in)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		want := q2_0RefGEMV(codes, scales, x, out, in)
		got := make([]float32, out)
		w.GEMV(x, got)

		for o := 0; o < out; o++ {
			if !q2_0CloseEnough(got[o], want[o]) {
				t.Fatalf("shape [%d,%d] row %d: Metal GEMV = %v, CPU-ref = %v (delta %v)",
					out, in, o, got[o], want[o], math.Abs(float64(got[o]-want[o])))
			}
			if math.IsNaN(float64(got[o])) || math.IsInf(float64(got[o]), 0) {
				t.Fatalf("shape [%d,%d] row %d: Metal GEMV produced non-finite %v", out, in, o, got[o])
			}
		}
		// Non-vacuous: an all-zero result would satisfy the comparison only if the reference were
		// also all-zero, which random ternary weights never are.
		nonZero := false
		for _, v := range want {
			if v != 0 {
				nonZero = true
				break
			}
		}
		if !nonZero {
			t.Fatalf("shape [%d,%d]: reference is all-zero — the parity check is vacuous", out, in)
		}
	}
}

// TestMetalQ2_0GemvBatchMatchesCPU is the issue #4873 done-condition witness for GEMVBatch: n
// activation rows through ONE command buffer must equal n independent reference GEMVs. The failure
// this pins is the per-dispatch X/Y offset arithmetic in mg_q2_0_gemv_batch — a mixed-up row would
// still produce plausible finite numbers.
func TestMetalQ2_0GemvBatchMatchesCPU(t *testing.T) {
	if !Available() {
		t.Skip("no Metal device (issue #4873 witness requires an Apple-Silicon host)")
	}
	defer ResetQ2_0()
	rng := rand.New(rand.NewSource(99))

	const out, in = 24, 128
	codes, scales := q2_0RandomWeight(rng, out, in)
	w := UploadQ2_0(codes, scales, out, in)
	if w == nil {
		t.Fatalf("UploadQ2_0 returned nil on an available device")
	}
	for _, n := range []int{1, 2, 7, 16} {
		Xcat := make([]float32, n*in)
		for i := range Xcat {
			Xcat[i] = float32(rng.NormFloat64())
		}
		want := q2_0RefGEMVBatch(codes, scales, Xcat, n, out, in)
		got := make([]float32, n*out)
		w.GEMVBatch(Xcat, n, got)

		for i := 0; i < n; i++ {
			for o := 0; o < out; o++ {
				g, wv := got[i*out+o], want[i*out+o]
				if !q2_0CloseEnough(g, wv) {
					t.Fatalf("n=%d batch row %d col %d: Metal = %v, CPU-ref = %v (delta %v)",
						n, i, o, g, wv, math.Abs(float64(g-wv)))
				}
			}
		}
		// The batch must agree with the standalone GEMV too — same weight, same activation row.
		single := make([]float32, out)
		w.GEMV(Xcat[:in], single)
		for o := 0; o < out; o++ {
			if !q2_0CloseEnough(single[o], got[o]) {
				t.Fatalf("n=%d: standalone GEMV row %d = %v, batch row 0 = %v", n, o, single[o], got[o])
			}
		}
	}
}

// TestMetalQ2_0GemvGroupMatchesCPU pins the decode-group primitive: n DIFFERENT ternary weights
// sharing ONE activation in a single command buffer (the q/k/v, gate/up pattern) must each equal
// their standalone reference GEMV, sliced at the right offsets.
func TestMetalQ2_0GemvGroupMatchesCPU(t *testing.T) {
	if !Available() {
		t.Skip("no Metal device (issue #4873 witness requires an Apple-Silicon host)")
	}
	defer ResetQ2_0()
	rng := rand.New(rand.NewSource(1234))

	const in = 96
	outs := []int{8, 20, 3}
	ws := make([]*Q2_0Weight, len(outs))
	allCodes := make([][]byte, len(outs))
	allScales := make([][]float32, len(outs))
	for i, out := range outs {
		codes, scales := q2_0RandomWeight(rng, out, in)
		allCodes[i], allScales[i] = codes, scales
		ws[i] = UploadQ2_0(codes, scales, out, in)
		if ws[i] == nil {
			t.Fatalf("weight %d: UploadQ2_0 returned nil on an available device", i)
		}
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	got := GEMVGroupQ2_0(ws, x)
	if len(got) != len(outs) {
		t.Fatalf("GEMVGroupQ2_0 returned %d result slices, want %d", len(got), len(outs))
	}
	for i, out := range outs {
		if len(got[i]) != out {
			t.Fatalf("group result %d has length %d, want %d", i, len(got[i]), out)
		}
		want := q2_0RefGEMV(allCodes[i], allScales[i], x, out, in)
		for o := 0; o < out; o++ {
			if !q2_0CloseEnough(got[i][o], want[o]) {
				t.Fatalf("group weight %d row %d: Metal = %v, CPU-ref = %v", i, o, got[i][o], want[o])
			}
		}
	}

	// A mismatched In must be declined (nil), so the caller falls back to per-weight GEMV.
	odd, _ := q2_0RandomWeight(rng, 4, in+Q2_0BlockWeights)
	oddScales := make([]float32, 4*(in+Q2_0BlockWeights)/Q2_0BlockWeights)
	if w := UploadQ2_0(odd, oddScales, 4, in+Q2_0BlockWeights); w != nil {
		if res := GEMVGroupQ2_0([]*Q2_0Weight{ws[0], w}, x); res != nil {
			t.Fatalf("GEMVGroupQ2_0 with mismatched In must return nil; got %d slices", len(res))
		}
	}
}

// TestMetalQ2_0UploadRejectsBadShapes pins UploadQ2_0's precondition guard: the shader's pointer
// arithmetic assumes in = nblk*32 and a full-length payload, so an unaligned reduction dim or a
// short slice must be refused with nil (the documented "no usable handle" signal) rather than
// reaching the GPU and reading out of bounds.
func TestMetalQ2_0UploadRejectsBadShapes(t *testing.T) {
	if !Available() {
		t.Skip("no Metal device (issue #4873 witness requires an Apple-Silicon host)")
	}
	defer ResetQ2_0()

	const out, in = 4, 64
	nblk := in / Q2_0BlockWeights
	codes := make([]byte, out*nblk*Q2_0BlockBytes)
	scales := make([]float32, out*nblk)

	if w := UploadQ2_0(codes, scales, out, 48); w != nil {
		t.Fatalf("UploadQ2_0 with in=48 (not a multiple of 32) must return nil")
	}
	if w := UploadQ2_0(codes, scales, 0, in); w != nil {
		t.Fatalf("UploadQ2_0 with out=0 must return nil")
	}
	if w := UploadQ2_0(codes[:len(codes)-1], scales, out, in); w != nil {
		t.Fatalf("UploadQ2_0 with a short codes slice must return nil")
	}
	if w := UploadQ2_0(codes, scales[:len(scales)-1], out, in); w != nil {
		t.Fatalf("UploadQ2_0 with a short scales slice must return nil")
	}
	// The well-formed payload still uploads — so the rejections above are guards, not a broken path.
	if w := UploadQ2_0(codes, scales, out, in); w == nil {
		t.Fatalf("UploadQ2_0 with a well-formed payload must return a handle")
	}
}

func q2UploadTest(t *testing.T, out, in int, seed int64) (*Q2_0Weight, []byte, []float32) {
	t.Helper()
	codes, scales := q2_0RandomWeight(rand.New(rand.NewSource(seed)), out, in)
	w := UploadQ2_0(codes, scales, out, in)
	if w == nil {
		t.Fatal("UploadQ2_0 returned nil for valid payload")
	}
	return w, codes, scales
}
func q2RequireClose(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d != %d", len(got), len(want))
	}
	for i := range got {
		if !q2_0CloseEnough(got[i], want[i]) {
			t.Fatalf("[%d] got %g want %g", i, got[i], want[i])
		}
	}
}

func TestQ2_0GEMMParity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	ResetQ2_0()
	defer ResetQ2_0()
	const in, out, p = 64, 5, 3
	w, codes, scales := q2UploadTest(t, out, in, 41)
	X := make([]float32, p*in)
	for i := range X {
		X[i] = float32((i%13)-6) / 11
	}
	got := make([]float32, p*out)
	w.GEMM(X, p, got)
	for tok := 0; tok < p; tok++ {
		q2RequireClose(t, got[tok*out:(tok+1)*out], q2_0RefGEMV(codes, scales, X[tok*in:(tok+1)*in], out, in))
	}
}

func TestFusedMLPQ2_0Parity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	ResetQ2_0()
	defer ResetQ2_0()
	gate, gc, gs := q2UploadTest(t, 64, 64, 51)
	up, uc, us := q2UploadTest(t, 64, 64, 52)
	down, dc, ds := q2UploadTest(t, 7, 64, 53)
	x := make([]float32, 64)
	for i := range x {
		x[i] = float32((i%9)-4) / 7
	}
	g := q2_0RefGEMV(gc, gs, x, 64, 64)
	u := q2_0RefGEMV(uc, us, x, 64, 64)
	inter := make([]float32, 64)
	for i := range inter {
		inter[i] = (g[i] / (1 + float32(math.Exp(float64(-g[i]))))) * u[i]
	}
	want := q2_0RefGEMV(dc, ds, inter, 7, 64)
	got := make([]float32, 7)
	if !FusedMLPQ2_0(gate, up, down, x, got) {
		t.Fatal("FusedMLPQ2_0 refused valid shapes")
	}
	q2RequireClose(t, got, want)
}
