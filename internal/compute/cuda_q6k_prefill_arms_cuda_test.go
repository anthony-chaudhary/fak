//go:build cuda && cgo

package compute

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"
)

// Qwen2.5-7B Q6_K projection shapes (hidden 3584, ffn 18944, kv 512): attn_v [512,3584] and
// ffn_down [3584,18944]. 129 rows take the out-of-bounds row fallback (129 % 128 != 0).
var q6kMMQParityShapes = []struct{ out, in int }{{512, 3584}, {3584, 18944}, {129, 3584}}

var q6kMMQParityPanels = []int{cudaQ6KMMQMinCols, 64, 512, 513}

// randQ6KBytes authors random valid Q6_K super-blocks: every ql/qh/scale byte is a legal code,
// so only the f16 d at byte 208 needs a bounded value.
func randQ6KBytes(g *lcg, out, in int) []byte {
	const block = 210
	raw := make([]byte, out*(in/256)*block)
	for b := 0; b < len(raw)/block; b++ {
		base := b * block
		for i := 0; i < 208; i++ {
			raw[base+i] = byte(g.u32())
		}
		binary.LittleEndian.PutUint16(raw[base+208:], f32ToF16Bits(0.0005+absf(g.f())*0.001))
	}
	return raw
}

func dequantQ6KWeight(raw []byte, out, in int) []float32 {
	w := make([]float32, out*in)
	for b := 0; b < out*(in/256); b++ {
		dequantQ6K(w[b*256:(b+1)*256], raw[b*210:(b+1)*210])
	}
	return w
}

func cudaQ6KMMQOrSkip(t *testing.T) *cudaBackend {
	t.Helper()
	cb := cudaOrSkip(t)
	if err := CheckCUDAQ6KPrefillArm(CUDAQ6KPrefillMMQ, cudaBackendSM(cb)); err != nil {
		t.Skipf("Q6_K mmq arm refused on %s: %v", cb.Tier(), err)
	}
	before := CUDAQ6KPrefillArmInUse()
	t.Cleanup(func() {
		if err := SetCUDAQ6KPrefillArm(before); err != nil {
			t.Errorf("restore Q6_K prefill arm %q: %v", before, err)
		}
	})
	return cb
}

// runQ6KPrefillArm runs one Q6_K BatchedMatMul on arm and proves which kernel produced it.
func runQ6KPrefillArm(t *testing.T, cb *cudaBackend, w Tensor, X []float32, in, P int, arm CUDAQ6KPrefillArm) []float32 {
	t.Helper()
	if err := SetCUDAQ6KPrefillArm(arm); err != nil {
		t.Fatalf("SetCUDAQ6KPrefillArm(%s): %v", arm, err)
	}
	x := mkResident(cb, []int{P, in}, X)
	defer cb.Free(x)
	_, before := cb.Q56KMMQLaunchCounts()
	y := cb.BatchedMatMul(w, x, P)
	defer cb.Free(y)
	got := cb.Read(y) // synchronizes: a submission count alone cannot pass
	_, after := cb.Q56KMMQLaunchCounts()
	want := uint64(0)
	if cudaQ6KPrefillMMQEligible(arm, Q6_K, P, false) || CUDAQ56KMMQQualificationEnabled() && P >= 128 {
		want = 1
	}
	if after-before != want {
		t.Fatalf("arm=%s P=%d ran %d Q6_K MMQ launches, want %d", arm, P, after-before, want)
	}
	return got
}

// TestCUDAQ6KPrefillArmMMQMatchesDequant holds the default mmq arm to the legacy dequant panel
// (the FAK_CUDA_Q6K_PREFILL=dequant path) on Qwen2.5-7B shapes, with the same gate as
// TestCUDAQ4KPrefillArmMMQMatchesRef for cosine and argmax, with a wider maxAbs bound (see q6kMMQRefRelMax).
//
// fak-test:runtime slow est=60s
func TestCUDAQ6KPrefillArmMMQMatchesDequant(t *testing.T) {
	cb := cudaQ6KMMQOrSkip(t)
	for _, shape := range q6kMMQParityShapes {
		var seed lcg = 0x5136
		g := &seed
		raw := randQ6KBytes(g, shape.out, shape.in)
		w := cb.Upload(NewQ6K(cb, []int{shape.out, shape.in}, raw), Q6_K)
		for _, P := range q6kMMQParityPanels {
			t.Run(fmt.Sprintf("%dx%d/P%d", shape.out, shape.in, P), func(t *testing.T) {
				X := rscale(g, P*shape.in, 1.0)
				want := runQ6KPrefillArm(t, cb, w, X, shape.in, P, CUDAQ6KPrefillDequant)
				got := runQ6KPrefillArm(t, cb, w, X, shape.in, P, CUDAQ6KPrefillMMQ)
				cos, maxAbs, scale := checkQ6KMMQAgainstRef(t, want, got)
				t.Logf("Q6_K mmq vs dequant %dx%d P=%d cosine=%.8f maxAbs=%.3e of max |ref| %.3e (tier=%s)",
					shape.out, shape.in, P, cos, maxAbs, scale, cb.Tier())
			})
		}
		cb.Free(w)
	}
}

// TestCUDAQ6KPrefillArmMMQMatchesCPURef checks the mmq arm against the cpuref f32 GEMM over the
// dequant of the same bytes, on a shape small enough for the host reference.
//
// fak-test:runtime slow est=20s
func TestCUDAQ6KPrefillArmMMQMatchesCPURef(t *testing.T) {
	cb := cudaQ6KMMQOrSkip(t)
	ref := Default()
	const out, in = 129, 3584
	var seed lcg = 0x5137
	g := &seed
	raw := randQ6KBytes(g, out, in)
	wRef := mkResident(ref, []int{out, in}, dequantQ6KWeight(raw, out, in))
	w := cb.Upload(NewQ6K(cb, []int{out, in}, raw), Q6_K)
	defer cb.Free(w)
	for _, P := range []int{cudaQ6KMMQMinCols, 64, 129} {
		t.Run(fmt.Sprintf("%dx%d/P%d", out, in, P), func(t *testing.T) {
			X := rscale(g, P*in, 1.0)
			want := ref.Read(ref.BatchedMatMul(wRef, mkResident(ref, []int{P, in}, X), P))
			got := runQ6KPrefillArm(t, cb, w, X, in, P, CUDAQ6KPrefillMMQ)
			cos, maxAbs, scale := checkQ6KMMQAgainstRef(t, want, got)
			t.Logf("Q6_K mmq vs cpuref %dx%d P=%d cosine=%.8f maxAbs=%.3e of max |ref| %.3e (tier=%s)",
				out, in, P, cos, maxAbs, scale, cb.Tier())
		})
	}
}

// TestCUDAQ6KPrefillArmRouting witnesses the selection: an unset env resolves to the arch
// default, a panel below cudaQ6KMMQMinCols never launches MMQ, and the dequant off-switch
// launches none at any P (runQ6KPrefillArm asserts the launch deltas).
//
// fak-test:runtime slow est=10s
func TestCUDAQ6KPrefillArmRouting(t *testing.T) {
	cb := cudaQ6KMMQOrSkip(t)
	if os.Getenv(CUDAQ6KPrefillEnv) == "" {
		if got, want := CUDAQ6KPrefillArmInUse(), DefaultCUDAQ6KPrefillArm(cudaBackendSM(cb)); got != want {
			t.Fatalf("unset %s resolved to %q on %s, want %q", CUDAQ6KPrefillEnv, got, cb.Tier(), want)
		}
	}
	const out, in = 512, 3584
	var seed lcg = 0x5138
	g := &seed
	w := cb.Upload(NewQ6K(cb, []int{out, in}, randQ6KBytes(g, out, in)), Q6_K)
	defer cb.Free(w)
	for _, P := range []int{1, 3, 4, cudaQ6KMMQMinCols - 1} {
		runQ6KPrefillArm(t, cb, w, rscale(g, P*in, 1.0), in, P, CUDAQ6KPrefillMMQ)
	}
	for _, P := range []int{cudaQ6KMMQMinCols, 64, 512} {
		runQ6KPrefillArm(t, cb, w, rscale(g, P*in, 1.0), in, P, CUDAQ6KPrefillDequant)
	}
}

// q6kMMQRefRelMax bounds the q8_1 activation rounding drift on fully random Q6_K bytes (random
// signed int8 sub-block scales), which measures ~4.5e-3 of max |ref| on sm_120 at cosine 0.99999.
// Kernel arithmetic is pinned exactly against the int8 model by TestCUDAQ56KMMQ; this gate only
// bounds the expected quantization error, so it is wider than the Q4_K bound.
const q6kMMQRefRelMax = 1e-2

func checkQ6KMMQAgainstRef(t *testing.T, want, got []float32) (cos, maxAbs, scale float64) {
	t.Helper()
	for _, v := range want {
		scale = math.Max(scale, math.Abs(float64(v)))
	}
	cos, maxAbs = cosine(want, got), maxAbsDelta(want, got)
	limit := q6kMMQRefRelMax * scale
	if cos < q4kMMQRefCosineMin || maxAbs > limit || argmaxF32(got) != argmaxF32(want) {
		t.Fatalf("mmq Q6_K vs f32 ref: cosine=%.8f (min %.4f) maxAbs=%.3e (limit %.3e = %.0e of max |ref| %.3e) argmax=%d want=%d",
			cos, q4kMMQRefCosineMin, maxAbs, limit, q6kMMQRefRelMax, scale, argmaxF32(got), argmaxF32(want))
	}
	return cos, maxAbs, scale
}
