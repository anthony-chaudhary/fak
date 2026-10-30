//go:build cuda && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
	"time"
)

// cuda_q3k_device_test.go — on-device parity and GB/s for the resident Q3_K CUDA MMVQ
// (cuda/q3k_mmvq.cuh). The reference is the CPU backend's Q3_K MatMul (q3kRowDot) over the
// same super-block bytes. FAK_CUDA_REQUIRE_DEVICE=1 turns a missing device into a failure.

// q3kDeviceShapes: 321 rows is odd (two-rows-per-block clamp); 5120x2304 and 2304x5120 are
// the DS V4.1 routed-expert down / up geometries.
var q3kDeviceShapes = []struct{ out, in int }{{321, 256}, {321, 2304}, {5120, 2304}, {2304, 5120}}

// q3kDeviceCases: the mmvq arm serves P <= 3 only; the f32 arm covers every P.
func q3kDeviceCases() []struct {
	P   int
	arm CUDAQ4KDecodeArm
} {
	var cases []struct {
		P   int
		arm CUDAQ4KDecodeArm
	}
	for _, P := range []int{1, 2, 3, 8, 128} {
		cases = append(cases, struct {
			P   int
			arm CUDAQ4KDecodeArm
		}{P, CUDAQ4KDecodeF32})
		if P <= 3 {
			cases = append(cases, struct {
				P   int
				arm CUDAQ4KDecodeArm
			}{P, CUDAQ4KDecodeMMVQ})
		}
	}
	return cases
}

func q3kCUDAOrFail(tb testing.TB) *cudaBackend {
	tb.Helper()
	cb, ok := Pick("cuda").(*cudaBackend)
	if !ok {
		if os.Getenv("FAK_CUDA_REQUIRE_DEVICE") == "1" {
			tb.Fatal("FAK_CUDA_REQUIRE_DEVICE=1 but the cuda backend is not registered")
		}
		tb.Skip("cuda backend not registered (no reachable CUDA device)")
	}
	return cb
}

// TestCUDAQ3KMatchesCPUOracle gates device Q3_K MatMul (P=1) and BatchedMatMul (P=2,3,8,128)
// against the CPU oracle on both decode arms: cosine >= cudaQ3KCosineMin, exact per-row argmax,
// maxAbs within 1e-4 of max |y| (f32 panel, P=128 dequant + compensated fp16 GEMM) or 1e-2
// (mmvq, q8_1 activations). On the mmvq arm (P <= 3) it also checks the q8_1 integer replay to
// 2e-5 of max |y|, which isolates a port defect from activation quantization, and that exactly
// one MMVQ kernel ran; every other case must run none.
//
// fak-test:runtime slow est=15s
func TestCUDAQ3KMatchesCPUOracle(t *testing.T) {
	cb := q3kCUDAOrFail(t)
	ref := Default()
	if !cb.SupportsQ3KMatMul() || !BackendSupportsDeviceWeightDtype(cb, Q3_K) {
		t.Fatal("cuda backend does not advertise Q3_K MatMul")
	}
	for si, sh := range q3kDeviceShapes {
		raw := q3kTestRaw(sh.out, sh.in, uint64(0x3a3+si))
		devW := cb.Upload(NewQ3K(cb, []int{sh.out, sh.in}, raw), Q3_K)
		refW := NewQ3K(ref, []int{sh.out, sh.in}, raw)
		rowBytes := (sh.in / q3kSuper) * q3kSuperBlock
		for _, c := range q3kDeviceCases() {
			P, arm := c.P, c.arm
			if P == 128 && sh.in > 2304 {
				continue
			}
			t.Run(fmt.Sprintf("%dx%d/P%d/%s", sh.out, sh.in, P, arm), func(t *testing.T) {
				before := CUDAQ4KDecodeArmInUse()
				if err := SetCUDAQ4KDecodeArm(arm); err != nil {
					t.Fatalf("SetCUDAQ4KDecodeArm(%s): %v", arm, err)
				}
				defer func() { _ = SetCUDAQ4KDecodeArm(before) }()
				mmvq := arm == CUDAQ4KDecodeMMVQ
				X := q3kTestX(P, sh.in, uint64(P*31+si))
				var want []float32
				if P == 1 {
					want = ref.Read(ref.MatMul(refW, NewF32(ref, []int{sh.in}, X)))
				} else {
					want = ref.Read(ref.BatchedMatMul(refW, NewF32(ref, []int{P, sh.in}, X), P))
				}
				launches := cb.q3kMMVQLaunchesForTest()
				var y Tensor
				if P == 1 {
					devX := mkResident(cb, []int{sh.in}, X)
					defer cb.Free(devX)
					y = cb.MatMul(devW, devX)
				} else {
					devX := mkResident(cb, []int{P, sh.in}, X)
					defer cb.Free(devX)
					y = cb.BatchedMatMul(devW, devX, P)
				}
				got := cb.Read(y)
				cb.Free(y)
				wantLaunches := uint64(0)
				if mmvq {
					wantLaunches = 1
				}
				if n := cb.q3kMMVQLaunchesForTest() - launches; n != wantLaunches {
					t.Fatalf("Q3_K MMVQ launches = %d, want %d", n, wantLaunches)
				}
				if len(got) != len(want) {
					t.Fatalf("len %d != %d", len(got), len(want))
				}
				scale := 0.0
				for _, v := range want {
					scale = math.Max(scale, math.Abs(float64(v)))
				}
				rel := 1e-4
				if mmvq {
					rel = 1e-2
				}
				cos, maxAbs := cosine(want, got), maxAbsDelta(want, got)
				if cos < cudaQ3KCosineMin || maxAbs > rel*scale {
					t.Fatalf("cosine=%.8f maxAbs=%.3e of max |y| %.3e (gate cosine>=%.4f maxAbs<=%.0e*max|y|)",
						cos, maxAbs, scale, cudaQ3KCosineMin, rel)
				}
				for r := 0; r < P; r++ {
					if a, b := argmaxF32(got[r*sh.out:(r+1)*sh.out]), argmaxF32(want[r*sh.out:(r+1)*sh.out]); a != b {
						t.Fatalf("row %d argmax mismatch: got=%d want=%d", r, a, b)
					}
				}
				replay := -1.0
				if mmvq {
					modelMax := 0.0
					for r := 0; r < P; r++ {
						qs, ds := q3kQ8_1(X[r*sh.in:(r+1)*sh.in], roundF16RNE)
						for o := 0; o < sh.out; o++ {
							m := q3kMMVQRowModel(raw[o*rowBytes:(o+1)*rowBytes], qs, ds)
							modelMax = math.Max(modelMax, math.Abs(m-float64(got[r*sh.out+o])))
						}
					}
					if modelMax > 2e-5*scale {
						t.Fatalf("vs q8_1 replay: maxAbs=%.3e > 2e-5 of max |y| %.3e", modelMax, scale)
					}
					replay = modelMax
				}
				t.Logf("Q3_K %s %dx%d P=%d cosine=%.8f maxAbs=%.3e max|y|=%.3e replayMaxAbs=%.3e argmax-exact (tier=%s)",
					arm, sh.out, sh.in, P, cos, maxAbs, scale, replay, cb.Tier())
			})
		}
		cb.Free(devW)
	}
}

// BenchmarkCUDAQ3K reports P=1 weight-stream GB/s for Q3_K (f32 panel and mmvq arms) at the DS
// V4.1 routed-expert shapes, with Q2_K (k_q2k_gemv) and Q4_K (f32 and mmvq arms) at the same
// shape as comparators. One Read after b.N launches drains the stream.
func BenchmarkCUDAQ3K(b *testing.B) {
	cb := q3kCUDAOrFail(b)
	shapes := []struct{ out, in int }{{5120, 2304}, {2304, 5120}}
	kinds := []struct {
		name  string
		dt    Dtype
		arm   CUDAQ4KDecodeArm
		block int
	}{
		{"q3_k-f32", Q3_K, CUDAQ4KDecodeF32, q3kSuperBlock},
		{"q3_k-mmvq", Q3_K, CUDAQ4KDecodeMMVQ, q3kSuperBlock},
		{"q2_k", Q2_K, "", q2kSuperBlock},
		{"q4_k-f32", Q4_K, CUDAQ4KDecodeF32, 144},
		{"q4_k-mmvq", Q4_K, CUDAQ4KDecodeMMVQ, 144},
	}
	for _, sh := range shapes {
		for _, k := range kinds {
			b.Run(k.name+"/"+strconv.Itoa(sh.out)+"x"+strconv.Itoa(sh.in), func(b *testing.B) {
				if k.arm != "" {
					before := CUDAQ4KDecodeArmInUse()
					if err := SetCUDAQ4KDecodeArm(k.arm); err != nil {
						b.Fatalf("SetCUDAQ4KDecodeArm(%s): %v", k.arm, err)
					}
					defer func() { _ = SetCUDAQ4KDecodeArm(before) }()
				}
				var raw []byte
				var host Tensor
				switch k.dt {
				case Q3_K:
					raw = q3kTestRaw(sh.out, sh.in, 0xbe3c)
					host = NewQ3K(cb, []int{sh.out, sh.in}, raw)
				case Q2_K:
					raw = iqGEMVTestRaw(sh.out, sh.in, k.block, 0xbe7c4)
					host = NewQ2K(cb, []int{sh.out, sh.in}, raw)
				default:
					var seed lcg = 0x4d56
					raw = randQ4KBytes(&seed, sh.out, sh.in)
					host = newQ4KHost(cb, []int{sh.out, sh.in}, raw)
				}
				devW := cb.Upload(host, k.dt)
				defer cb.Free(devW)
				devX := mkResident(cb, []int{sh.in}, q3kTestX(1, sh.in, 7))
				defer cb.Free(devX)
				cb.Free(cb.MatMul(devW, devX)) // warm
				b.SetBytes(int64(len(raw)))
				b.ResetTimer()
				start := time.Now()
				var y Tensor
				for i := 0; i < b.N; i++ {
					if i > 0 {
						cb.Free(y)
					}
					y = cb.MatMul(devW, devX)
				}
				_ = cb.Read(y)
				el := time.Since(start)
				b.StopTimer()
				cb.Free(y)
				b.ReportMetric(float64(len(raw))*float64(b.N)/el.Seconds()/1e9, "weightGB/s")
			})
		}
	}
}
