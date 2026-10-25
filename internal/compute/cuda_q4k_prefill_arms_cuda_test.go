//go:build cuda && cgo

package compute

import (
	"fmt"
	"math"
	"testing"
)

// q4kMMQParityShapes are the P-04 parity shapes. 320 rows take the out-of-bounds row fallback
// (320 % 128 != 0); 1024 rows do not.
var q4kMMQParityShapes = []struct{ out, in int }{{320, 256}, {1024, 5120}}

var q4kMMQParityPanels = []int{128, 512}

// TestCUDAQ4KPrefillArmMMQMatchesRef is the mmq arm's keep/remove gate (P-04, fak#13983). It
// drives the arm through SetCUDAQ4KPrefillArm and holds it to the strict large-panel Q4_K gate
// the compensated route passes (TestCUDAQ4KLargePanelCompensatedFP16MatchesRef): cosine >=
// cudaQ4KCosineMin, maxAbs <= 0.02 and exact argmax against the cpuref f32 GEMM over the dequant
// of the same bytes.
//
// fak-test:runtime slow est=20s
func TestCUDAQ4KPrefillArmMMQMatchesRef(t *testing.T) {
	cb := cudaQ4KMMQOrSkip(t)
	ref := Default()
	for _, shape := range q4kMMQParityShapes {
		var seed lcg = 0x4b4d
		g := &seed
		raw := randQ4KBytes(g, shape.out, shape.in)
		w := mkResident(ref, []int{shape.out, shape.in}, dequantQ4KWeight(raw, shape.out, shape.in))
		wQ4K := cb.Upload(newQ4KHost(cb, []int{shape.out, shape.in}, raw), Q4_K)
		for _, P := range q4kMMQParityPanels {
			t.Run(fmt.Sprintf("%dx%d/P%d", shape.out, shape.in, P), func(t *testing.T) {
				X := rscale(g, P*shape.in, 1.0)
				want := ref.Read(ref.BatchedMatMul(w, mkResident(ref, []int{P, shape.in}, X), P))
				got := runQ4KMMQPanel(t, cb, wQ4K, X, shape.in, P)
				cos, maxAbs := cosine(want, got), maxAbsDelta(want, got)
				if cos < cudaQ4KCosineMin || maxAbs > 0.02 {
					t.Fatalf("mmq Q4_K panel: cosine=%.8f maxAbs=%.3e (gate cosine>=%.4f maxAbs<=0.02)", cos, maxAbs, cudaQ4KCosineMin)
				}
				if argmaxF32(got) != argmaxF32(want) {
					t.Fatalf("mmq Q4_K panel argmax mismatch: got=%d want=%d", argmaxF32(got), argmaxF32(want))
				}
				t.Logf("mmq Q4_K panel %dx%d P=%d cosine=%.8f maxAbs=%.3e argmax-exact (tier=%s)",
					shape.out, shape.in, P, cos, maxAbs, cb.Tier())
			})
		}
	}
}

// TestCUDAQ4KPrefillArmMMQMatchesInt8Model checks the port against a CPU replay of the MMQ
// arithmetic (q4kMMQInt8Model). It does not decide keep or remove; it tells a port defect apart
// from the int8 activation error the strict gate above measures.
//
// fak-test:runtime slow est=30s
func TestCUDAQ4KPrefillArmMMQMatchesInt8Model(t *testing.T) {
	cb := cudaQ4KMMQOrSkip(t)
	for _, shape := range q4kMMQParityShapes {
		var seed lcg = 0x4b4d
		g := &seed
		raw := randQ4KBytes(g, shape.out, shape.in)
		wQ4K := cb.Upload(newQ4KHost(cb, []int{shape.out, shape.in}, raw), Q4_K)
		for _, P := range q4kMMQParityPanels {
			t.Run(fmt.Sprintf("%dx%d/P%d", shape.out, shape.in, P), func(t *testing.T) {
				X := rscale(g, P*shape.in, 1.0)
				model := q4kMMQInt8Model(raw, X, shape.out, shape.in, P)
				got := runQ4KMMQPanel(t, cb, wQ4K, X, shape.in, P)
				scale := 0.0
				for _, v := range model {
					scale = math.Max(scale, math.Abs(float64(v)))
				}
				if d, limit := maxAbsDelta(model, got), 2e-5*scale; d > limit {
					t.Fatalf("mmq Q4_K panel vs int8 model: maxAbs=%.3e > %.3e (2e-5 of max |y| %.3e)", d, limit, scale)
				}
				t.Logf("mmq Q4_K panel %dx%d P=%d matches the int8 model: maxAbs=%.3e of max |y| %.3e",
					shape.out, shape.in, P, maxAbsDelta(model, got), scale)
			})
		}
	}
}

func cudaQ4KMMQOrSkip(t *testing.T) *cudaBackend {
	t.Helper()
	cb := cudaOrSkip(t)
	if err := CheckCUDAQ4KPrefillArm(CUDAQ4KPrefillMMQ, cudaBackendSM(cb)); err != nil {
		t.Skipf("mmq arm refused on %s: %v", cb.Tier(), err)
	}
	before, err := CUDAQ4KPrefillArmInUse()
	if err != nil {
		t.Fatalf("%s refused at init: %v", CUDAQ4KPrefillEnv, err)
	}
	t.Cleanup(func() {
		if err := SetCUDAQ4KPrefillArm(before); err != nil {
			t.Errorf("restore arm %q: %v", before, err)
		}
	})
	return cb
}

// runQ4KMMQPanel runs one P >= 128 Q4_K panel on the mmq arm and proves the MMQ kernel ran.
func runQ4KMMQPanel(t *testing.T, cb *cudaBackend, wQ4K Tensor, X []float32, in, P int) []float32 {
	t.Helper()
	if err := SetCUDAQ4KPrefillArm(CUDAQ4KPrefillMMQ); err != nil {
		t.Fatalf("SetCUDAQ4KPrefillArm(mmq): %v", err)
	}
	defer func() {
		if err := SetCUDAQ4KPrefillArm(CUDAQ4KPrefillCompensated); err != nil {
			t.Errorf("SetCUDAQ4KPrefillArm(compensated): %v", err)
		}
	}()
	launches := cb.q4kMMQLaunchesForTest()
	got := cb.Read(cb.BatchedMatMul(wQ4K, mkResident(cb, []int{P, in}, X), P))
	if n := cb.q4kMMQLaunchesForTest() - launches; n != 1 {
		t.Fatalf("mmq launches = %d, want 1 (the arm did not run the MMQ kernel)", n)
	}
	return got
}

// q4kMMQInt8Model replays the mmq arm on the CPU: the activation is quantized to int8 per 32
// values with an f16 scale and an f16 sum of the unquantized values (quantize_mmq_q8_1, DS4); the
// weight scale products d*sc and -dmin*m are rounded to f16 (load_tiles_q4_K); each 32-value dot
// is an exact integer; the epilogue is that of vec_dot_q8_1_q8_1_mma.
func q4kMMQInt8Model(raw []byte, X []float32, out, in, P int) []float32 {
	groups := in / 32
	nib := make([]int32, out*in)
	wScale := make([]float64, out*groups)
	wMin := make([]float64, out*groups)
	for o := 0; o < out; o++ {
		for sb := 0; sb < in/256; sb++ {
			base := (o*(in/256) + sb) * blockQ4KBytesTest
			d := f16BitsToF32(uint16(raw[base]) | uint16(raw[base+1])<<8)
			dmin := f16BitsToF32(uint16(raw[base+2]) | uint16(raw[base+3])<<8)
			scales, q := raw[base+4:base+16], raw[base+16:base+blockQ4KBytesTest]
			for s := 0; s < 8; s++ {
				sc, m := getScaleMinK4Test(s, scales)
				gi := o*groups + sb*8 + s
				wScale[gi] = float64(roundF16RNE(d * float32(sc)))
				wMin[gi] = float64(roundF16RNE(-dmin * float32(m)))
				for l := 0; l < 32; l++ {
					code := q[(s/2)*32+l]
					if s%2 == 1 {
						code >>= 4
					}
					nib[o*in+sb*256+s*32+l] = int32(code & 0x0f)
				}
			}
		}
	}
	qx := make([]int32, P*in)
	xScale := make([]float64, P*groups)
	xSum := make([]float64, P*groups)
	for t := 0; t < P; t++ {
		for gi := 0; gi < groups; gi++ {
			xs := X[t*in+gi*32 : t*in+gi*32+32]
			var amax float32
			var part [8]float32 // one 4-value sum per quantize thread, reduced in its xor-butterfly order
			for l, v := range xs {
				amax = float32(math.Max(float64(amax), math.Abs(float64(v))))
				part[l/4] += v
			}
			sum := ((part[0] + part[4]) + (part[2] + part[6])) + ((part[1] + part[5]) + (part[3] + part[7]))
			if amax == 0 {
				continue
			}
			dInv := float32(127) / amax
			for l, v := range xs {
				qx[t*in+gi*32+l] = int32(math.Round(float64(v * dInv)))
			}
			xScale[t*groups+gi] = float64(roundF16RNE(float32(1) / dInv))
			xSum[t*groups+gi] = float64(roundF16RNE(sum))
		}
	}
	y := make([]float32, P*out)
	for t := 0; t < P; t++ {
		for o := 0; o < out; o++ {
			acc := 0.0
			for gi := 0; gi < groups; gi++ {
				var dot int32
				for l := 0; l < 32; l++ {
					dot += nib[o*in+gi*32+l] * qx[t*in+gi*32+l]
				}
				acc += wScale[o*groups+gi]*xScale[t*groups+gi]*float64(dot) + wMin[o*groups+gi]*xSum[t*groups+gi]
			}
			y[t*out+o] = float32(acc)
		}
	}
	return y
}

// roundF16RNE rounds f to the nearest f16 value, ties to even, as __float2half and half2
// multiplication do.
func roundF16RNE(f float32) float32 {
	if f == 0 {
		return 0
	}
	e := math.Ilogb(float64(f))
	if e < -14 {
		e = -14
	}
	ulp := math.Ldexp(1, e-10)
	return float32(math.RoundToEven(float64(f)/ulp) * ulp)
}
