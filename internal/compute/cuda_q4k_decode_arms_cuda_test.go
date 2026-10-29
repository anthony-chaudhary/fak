//go:build cuda && cgo

package compute

import (
	"fmt"
	"math"
	"testing"
)

// q4kMMVQParityShapes are the mmvq decode parity shapes. 321 rows is odd, so the two-rows-per-
// block P=2/P=3 kernels take the clamped last-row read; 1024x5120 is a Qwen hidden-width K.
var q4kMMVQParityShapes = []struct{ out, in int }{{321, 256}, {1024, 5120}}

// TestCUDAQ4KDecodeArmMMVQMatchesRef is the mmvq decode arm's keep/remove gate (fak#8635): the
// strict Q4_K gate the f32 GEMV and the mmq prefill arm pass (cosine >= cudaQ4KCosineMin,
// maxAbs <= 0.02, exact argmax) against the cpuref f32 GEMM over the dequant of the same bytes,
// for MatMul (P=1) and BatchedMatMul (P=2, 3).
//
// fak-test:runtime slow est=10s
func TestCUDAQ4KDecodeArmMMVQMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	ref := Default()
	for _, shape := range q4kMMVQParityShapes {
		var seed lcg = 0x4d56
		g := &seed
		raw := randQ4KBytes(g, shape.out, shape.in)
		w := mkResident(ref, []int{shape.out, shape.in}, dequantQ4KWeight(raw, shape.out, shape.in))
		wQ4K := cb.Upload(newQ4KHost(cb, []int{shape.out, shape.in}, raw), Q4_K)
		for _, P := range []int{1, 2, 3} {
			t.Run(fmt.Sprintf("%dx%d/P%d", shape.out, shape.in, P), func(t *testing.T) {
				X := rscale(g, P*shape.in, 1.0)
				want := ref.Read(ref.BatchedMatMul(w, mkResident(ref, []int{P, shape.in}, X), P))
				got := runQ4KMMVQ(t, cb, wQ4K, X, shape.in, P)
				cos, maxAbs := cosine(want, got), maxAbsDelta(want, got)
				if cos < cudaQ4KCosineMin || maxAbs > 0.02 {
					t.Fatalf("mmvq Q4_K decode: cosine=%.8f maxAbs=%.3e (gate cosine>=%.4f maxAbs<=0.02)", cos, maxAbs, cudaQ4KCosineMin)
				}
				for r := 0; r < P; r++ {
					if a, b := argmaxF32(got[r*shape.out:(r+1)*shape.out]), argmaxF32(want[r*shape.out:(r+1)*shape.out]); a != b {
						t.Fatalf("mmvq Q4_K decode row %d argmax mismatch: got=%d want=%d", r, a, b)
					}
				}
				t.Logf("mmvq Q4_K decode %dx%d P=%d cosine=%.8f maxAbs=%.3e argmax-exact (tier=%s)",
					shape.out, shape.in, P, cos, maxAbs, cb.Tier())
			})
		}
		cb.Free(wQ4K)
		ref.Free(w)
	}
}

// TestCUDAQ4KDecodeArmMMVQMatchesInt8Model checks the port against a CPU replay of the MMVQ
// arithmetic (q4kMMVQInt8Model). It does not decide keep or remove; it tells a port defect apart
// from the q8_1 activation error the strict gate above measures.
//
// fak-test:runtime slow est=10s
func TestCUDAQ4KDecodeArmMMVQMatchesInt8Model(t *testing.T) {
	cb := cudaOrSkip(t)
	for _, shape := range q4kMMVQParityShapes {
		var seed lcg = 0x4d56
		g := &seed
		raw := randQ4KBytes(g, shape.out, shape.in)
		wQ4K := cb.Upload(newQ4KHost(cb, []int{shape.out, shape.in}, raw), Q4_K)
		for _, P := range []int{1, 2, 3} {
			t.Run(fmt.Sprintf("%dx%d/P%d", shape.out, shape.in, P), func(t *testing.T) {
				X := rscale(g, P*shape.in, 1.0)
				model := q4kMMVQInt8Model(raw, X, shape.out, shape.in, P)
				got := runQ4KMMVQ(t, cb, wQ4K, X, shape.in, P)
				scale := 0.0
				for _, v := range model {
					scale = math.Max(scale, math.Abs(float64(v)))
				}
				if d, limit := maxAbsDelta(model, got), 2e-5*scale; d > limit {
					t.Fatalf("mmvq Q4_K decode vs int8 model: maxAbs=%.3e > %.3e (2e-5 of max |y| %.3e)", d, limit, scale)
				}
			})
		}
		cb.Free(wQ4K)
	}
}

// runQ4KMMVQ runs one P < 4 Q4_K GEMV on the mmvq arm (MatMul for P=1, BatchedMatMul otherwise)
// and proves the MMVQ kernel ran exactly once.
func runQ4KMMVQ(t *testing.T, cb *cudaBackend, wQ4K Tensor, X []float32, in, P int) []float32 {
	t.Helper()
	before := CUDAQ4KDecodeArmInUse()
	if err := SetCUDAQ4KDecodeArm(CUDAQ4KDecodeMMVQ); err != nil {
		t.Fatalf("SetCUDAQ4KDecodeArm(mmvq): %v", err)
	}
	defer func() {
		if err := SetCUDAQ4KDecodeArm(before); err != nil {
			t.Errorf("restore decode arm %q: %v", before, err)
		}
	}()
	launches := cb.q4kMMVQLaunchesForTest()
	var y Tensor
	if P == 1 {
		y = cb.MatMul(wQ4K, mkResident(cb, []int{in}, X))
	} else {
		y = cb.BatchedMatMul(wQ4K, mkResident(cb, []int{P, in}, X), P)
	}
	got := cb.Read(y)
	if n := cb.q4kMMVQLaunchesForTest() - launches; n != 1 {
		t.Fatalf("mmvq launches = %d, want 1 (the arm did not run the MMVQ kernel)", n)
	}
	return got
}

// q4kMMVQInt8Model replays the mmvq arm on the CPU: the activation is quantized to int8 per 32
// values with d = amax/127 and q = roundf(x/d), d read back through its f16 copy (quantize_q8_1,
// block_q8_1.ds); each 32-value weight-code dot and code sum is an exact integer; the super-block
// epilogue is d*Σ d8*dot*sc - dmin*Σ d8*Σq8*m (vec_dot_q4_K_q8_1_impl_vmmq).
func q4kMMVQInt8Model(raw []byte, X []float32, out, in, P int) []float32 {
	groups := in / 32
	qx := make([]int32, P*in)
	d8 := make([]float64, P*groups)
	for t := 0; t < P; t++ {
		for gi := 0; gi < groups; gi++ {
			xs := X[t*in+gi*32 : t*in+gi*32+32]
			var amax float32
			for _, v := range xs {
				amax = float32(math.Max(float64(amax), math.Abs(float64(v))))
			}
			if amax == 0 {
				continue
			}
			d := amax / 127
			for l, v := range xs {
				qx[t*in+gi*32+l] = int32(math.Round(float64(v / d)))
			}
			d8[t*groups+gi] = float64(roundF16RNE(d))
		}
	}
	y := make([]float32, P*out)
	for t := 0; t < P; t++ {
		for o := 0; o < out; o++ {
			acc := 0.0
			for sb := 0; sb < in/256; sb++ {
				base := (o*(in/256) + sb) * blockQ4KBytesTest
				d := float64(f16BitsToF32(uint16(raw[base]) | uint16(raw[base+1])<<8))
				dmin := float64(f16BitsToF32(uint16(raw[base+2]) | uint16(raw[base+3])<<8))
				scales, q := raw[base+4:base+16], raw[base+16:base+blockQ4KBytesTest]
				sumD, sumM := 0.0, 0.0
				for s := 0; s < 8; s++ {
					sc, m := getScaleMinK4Test(s, scales)
					gi := sb*8 + s
					var dot, sum8 int32
					for l := 0; l < 32; l++ {
						code := q[(s/2)*32+l]
						if s%2 == 1 {
							code >>= 4
						}
						xq := qx[t*in+gi*32+l]
						dot += int32(code&0x0f) * xq
						sum8 += xq
					}
					sumD += d8[t*groups+gi] * float64(dot*int32(sc))
					sumM += d8[t*groups+gi] * float64(sum8*int32(m))
				}
				acc += d*sumD - dmin*sumM
			}
			y[t*out+o] = float32(acc)
		}
	}
	return y
}
