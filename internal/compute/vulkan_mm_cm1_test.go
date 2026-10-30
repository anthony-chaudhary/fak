//go:build vulkan && (windows || linux) && cgo

package compute

// fak-test:runtime slow est=7s

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"
)

// mmCM1RoundF16 rounds to the nearest binary16 value (ties to even), the conversion the
// GEMM applies to both dequantized weights and activations before the fp16 multiply.
func mmCM1RoundF16(f float32) float32 {
	a := math.Abs(float64(f))
	if a == 0 || math.IsInf(a, 0) || math.IsNaN(a) {
		return f
	}
	ulp := 0x1p-24
	if a >= 0x1p-14 {
		_, exp := math.Frexp(a)
		ulp = math.Ldexp(1, exp-11)
	}
	return float32(math.RoundToEven(float64(f)/ulp) * ulp)
}

func mmCM1Weight(t *testing.T, dt Dtype, seed int64, out, in int) (Tensor, []float32) {
	t.Helper()
	switch dt {
	case Q2_K:
		rng := rand.New(rand.NewSource(seed))
		raw := make([]byte, out*(in/q2kSuper)*q2kSuperBlock)
		rng.Read(raw)
		deq := make([]float32, out*in)
		for b := 0; b < len(raw)/q2kSuperBlock; b++ {
			blk := raw[b*q2kSuperBlock : (b+1)*q2kSuperBlock]
			binaryPutFloat16(blk[80:82], float32(0.005+rng.Float64()*0.02))
			binaryPutFloat16(blk[82:84], float32(0.002+rng.Float64()*0.01))
			q2kDequantSuperBlock(deq[b*q2kSuper:(b+1)*q2kSuper], blk)
		}
		return NewQ2K(Default(), []int{out, in}, raw), deq
	case Q4_K:
		rng := rand.New(rand.NewSource(seed))
		raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
		deq := make([]float32, out*in)
		for b := 0; b < len(raw)/q4kSuperBlock; b++ {
			blk := raw[b*q4kSuperBlock : (b+1)*q4kSuperBlock]
			randQ4KBlockC(rng, blk)
			q4kDequantBlock(deq[b*q4kSuper:(b+1)*q4kSuper], blk)
		}
		return NewQ4K(Default(), []int{out, in}, raw), deq
	default:
		if isRawIQ(dt) {
			return rawIQParityWeight(dt, seed, out, in)
		}
		t.Fatalf("no mm_cm1 fixture for %v", dt)
	}
	return Tensor{}, nil
}

func mmCM1Oracle(deq, x []float32, out, in, tokens int, roundF16 bool) []float32 {
	w, xs := deq, x
	if roundF16 {
		w, xs = make([]float32, len(deq)), make([]float32, len(x))
		for i, v := range deq {
			w[i] = mmCM1RoundF16(v)
		}
		for i, v := range x {
			xs[i] = mmCM1RoundF16(v)
		}
	}
	y := make([]float32, tokens*out)
	for p := 0; p < tokens; p++ {
		xr := xs[p*in : (p+1)*in]
		for o := 0; o < out; o++ {
			wr := w[o*in : (o+1)*in]
			var s float64
			for i := range wr {
				s += float64(wr[i]) * float64(xr[i])
			}
			y[p*out+o] = float32(s)
		}
	}
	return y
}

// mmCM1SequenceMatMul runs the Qwen3.5 sequence-prefill matmul switch, the production
// caller of the route, and reads the [tokens, out] panel back.
func mmCM1SequenceMatMul(v *vulkanBackend, w, x Tensor, tokens int) []float32 {
	vulkanMu.Lock()
	y := v.qwen35VulkanSequenceMatMulLocked(w, x, tokens)
	vulkanMu.Unlock()
	return copyRead(v, y)
}

func mmCM1Dtypes() []Dtype {
	dts := make([]Dtype, 0, len(vulkanMMCM1Formats))
	for dt := range vulkanMMCM1Formats {
		dts = append(dts, dt)
	}
	sort.Slice(dts, func(i, j int) bool { return dts[i] < dts[j] })
	return dts
}

func requireMMCM1(t *testing.T, dt Dtype) {
	t.Helper()
	if err := vulkanMMCM1DeviceStatus(dt); err != nil {
		if os.Getenv("FAK_VULKAN_REQUIRE_MM_CM1") == "1" {
			t.Fatalf("required mm_cm1 %v kernel unavailable: %v", dt, err)
		}
		t.Skipf("mm_cm1 %v kernel unavailable: %v", dt, err)
	}
}

// TestVulkanMMCM1QuantMatchesCPU proves each GEMM variant, reached through the sequence
// matmul switch, against the CPU dequant of the same verbatim block bytes: tightly against an
// fp16-rounded-input oracle (only reduction order differs) and loosely against the exact f32
// oracle. Shapes cover full tiles, partial row/token tiles and an output width that forces
// the staged (unaligned) store.
// fak-test:runtime slow est=5s
func TestVulkanMMCM1QuantMatchesCPU(t *testing.T) {
	v := vk(t)
	prev := vulkanMMCM1SetConfig(VulkanMMCM1Config{Enabled: true, MinTokens: 8})
	t.Cleanup(func() { vulkanMMCM1SetConfig(prev) })
	for _, dt := range mmCM1Dtypes() {
		t.Run(dt.String(), func(t *testing.T) {
			requireMMCM1(t, dt)
			for _, sh := range []struct{ out, in int }{{256, 512}, {200, 1024}, {37, 256}} {
				seed := int64(0x05f0 + sh.out*31 + sh.in + int(dt)*7919)
				hw, deq := mmCM1Weight(t, dt, seed, sh.out, sh.in)
				dw := v.Upload(hw, dt)
				if dw.Dtype != dt {
					v.Free(dw)
					t.Skipf("native %v upload unavailable (uploaded as %v)", dt, dw.Dtype)
				}
				for _, tokens := range []int{9, 64, 512} {
					t.Run(fmt.Sprintf("out%d_in%d_p%d", sh.out, sh.in, tokens), func(t *testing.T) {
						s := lcg(uint64(seed) + uint64(tokens))
						x := randVec(&s, tokens*sh.in)
						dx := v.Upload(NewF32(Default(), []int{tokens, sh.in}, x), F32)
						defer func() { v.Free(dx); v.Recycle() }()
						before := VulkanMMCM1Snapshot()
						got := mmCM1SequenceMatMul(v, dw, dx, tokens)
						after := VulkanMMCM1Snapshot()
						if after.GEMM != before.GEMM+1 || after.Unavailable != before.Unavailable {
							t.Fatalf("route counters before %+v after %+v: GEMM did not serve the call", before, after)
						}
						rawIQAssertClose(t, "GEMM vs fp16-input oracle", mmCM1Oracle(deq, x, sh.out, sh.in, tokens, true), got, 5e-4)
						rawIQAssertClose(t, "GEMM vs exact oracle", mmCM1Oracle(deq, x, sh.out, sh.in, tokens, false), got, 8e-3)
					})
				}
				v.Free(dw)
				v.Recycle()
			}
		})
	}
}

// TestVulkanMMCM1SequenceRouteMatchesMatvec runs the Qwen3.5 sequence-prefill matmul switch
// with the GEMM on and off over the same operands and holds both routes to the CPU oracle.
// fak-test:runtime slow est=2s
func TestVulkanMMCM1SequenceRouteMatchesMatvec(t *testing.T) {
	v := vk(t)
	const out, in, tokens = 192, 768, 33
	for _, dt := range mmCM1Dtypes() {
		t.Run(dt.String(), func(t *testing.T) {
			requireMMCM1(t, dt)
			hw, deq := mmCM1Weight(t, dt, 0x5e0+int64(dt), out, in)
			dw := v.Upload(hw, dt)
			if dw.Dtype != dt {
				v.Free(dw)
				t.Skipf("native %v upload unavailable (uploaded as %v)", dt, dw.Dtype)
			}
			s := lcg(0x5e1)
			x := randVec(&s, tokens*in)
			dx := v.Upload(NewF32(Default(), []int{tokens, in}, x), F32)
			defer func() { v.Free(dw); v.Free(dx); v.Recycle() }()
			run := func(enabled bool) []float32 {
				prev := vulkanMMCM1SetConfig(VulkanMMCM1Config{Enabled: enabled, MinTokens: 8})
				defer vulkanMMCM1SetConfig(prev)
				return mmCM1SequenceMatMul(v, dw, dx, tokens)
			}
			before := VulkanMMCM1Snapshot()
			gemm := run(true)
			matvec := run(false)
			after := VulkanMMCM1Snapshot()
			if after.GEMM != before.GEMM+1 || after.Disabled != before.Disabled+1 {
				t.Fatalf("route counters before %+v after %+v: want one GEMM and one disabled fallback", before, after)
			}
			want := mmCM1Oracle(deq, x, out, in, tokens, false)
			rawIQAssertClose(t, "GEMM route vs exact oracle", want, gemm, 8e-3)
			rawIQAssertClose(t, "matvec route vs exact oracle", want, matvec, 1e-4)
		})
	}
}

// TestVulkanMMCM1Q2KShortPanelsMatchCPU holds the threshold experiment to the
// issue's exact short-panel set. P=1 must retain the decode matvec; P>1 must
// reach the GEMM when the experimental threshold is one.
// fak-test:runtime slow est=2s
func TestVulkanMMCM1Q2KShortPanelsMatchCPU(t *testing.T) {
	v := vk(t)
	requireMMCM1(t, Q2_K)
	prev := vulkanMMCM1SetConfig(VulkanMMCM1Config{Enabled: true, MinTokens: 1})
	t.Cleanup(func() { vulkanMMCM1SetConfig(prev) })

	const out, in = 256, 512
	hw, deq := mmCM1Weight(t, Q2_K, 0x12001, out, in)
	dw := v.Upload(hw, Q2_K)
	defer func() { v.Free(dw); v.Recycle() }()
	if dw.Dtype != Q2_K {
		t.Fatalf("required native Q2_K upload returned %v", dw.Dtype)
	}

	for _, tokens := range []int{1, 3, 4, 7} {
		t.Run(fmt.Sprintf("p%d", tokens), func(t *testing.T) {
			s := lcg(0x12001 + uint64(tokens))
			x := randVec(&s, tokens*in)
			dx := v.Upload(NewF32(Default(), []int{tokens, in}, x), F32)
			defer v.Free(dx)

			before := VulkanMMCM1Snapshot()
			got := mmCM1SequenceMatMul(v, dw, dx, tokens)
			after := VulkanMMCM1Snapshot()
			if tokens == 1 {
				if after.BelowThreshold != before.BelowThreshold+1 || after.GEMM != before.GEMM {
					t.Fatalf("route counters before %+v after %+v: P=1 did not retain matvec", before, after)
				}
				rawIQAssertClose(t, "matvec vs exact oracle", mmCM1Oracle(deq, x, out, in, tokens, false), got, 1e-4)
				return
			}
			if after.GEMM != before.GEMM+1 || after.BelowThreshold != before.BelowThreshold {
				t.Fatalf("route counters before %+v after %+v: P=%d did not use GEMM", before, after, tokens)
			}
			rawIQAssertClose(t, "GEMM vs fp16-input oracle", mmCM1Oracle(deq, x, out, in, tokens, true), got, 5e-4)
			rawIQAssertClose(t, "GEMM vs exact oracle", mmCM1Oracle(deq, x, out, in, tokens, false), got, 8e-3)
		})
	}
}

// BenchmarkVulkanMMCM1Q2KShortPanels measures both threshold arms from one
// binary. Set FAK_VULKAN_MM_CM1_MIN_TOKENS to 8 for the retained baseline or 1
// for the candidate; the first untimed dispatch verifies the selected route.
func BenchmarkVulkanMMCM1Q2KShortPanels(b *testing.B) {
	v, ok := Lookup("vulkan")
	if !ok {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			b.Fatal("required Vulkan device is not registered")
		}
		b.Skip("Vulkan backend unavailable")
	}
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" && !strings.Contains(strings.ToLower(v.Tier()), strings.ToLower(expected)) {
		b.Fatalf("device %q does not match required %q", v.Tier(), expected)
	}
	if err := vulkanMMCM1DeviceStatus(Q2_K); err != nil {
		if os.Getenv("FAK_VULKAN_REQUIRE_MM_CM1") == "1" {
			b.Fatalf("required mm_cm1 Q2_K kernel unavailable: %v", err)
		}
		b.Skipf("mm_cm1 Q2_K kernel unavailable: %v", err)
	}
	const out, in = 5120, 5120
	hw := q2FusedWeight(Q2_K, out, in)
	dw := v.Upload(hw, Q2_K)
	defer v.Free(dw)
	if dw.Dtype != Q2_K {
		b.Fatalf("required native Q2_K upload returned %v", dw.Dtype)
	}
	cfg := vulkanMMCM1CurrentConfig()
	if !cfg.Enabled {
		b.Fatal("mm_cm1 must be enabled for the threshold comparison")
	}
	if os.Getenv("FAK_V41_Q2K_TOKEN_PANEL") != "0" {
		b.Fatal("set FAK_V41_Q2K_TOKEN_PANEL=0 to isolate the mm_cm1 threshold arms")
	}

	for _, tokens := range []int{1, 3, 4, 7} {
		b.Run(fmt.Sprintf("threshold%d_p%d", cfg.MinTokens, tokens), func(b *testing.B) {
			s := lcg(0x12001 + uint64(tokens))
			x := randVec(&s, tokens*in)
			dx := v.Upload(NewF32(Default(), []int{tokens, in}, x), F32)
			defer v.Free(dx)
			run := func() {
				var dy Tensor
				if tokens == 1 {
					dy = v.MatMul(dw, dx)
				} else {
					dy = v.BatchedMatMul(dw, dx, tokens)
				}
				v.Free(dy)
			}

			before := VulkanMMCM1Snapshot()
			run()
			after := VulkanMMCM1Snapshot()
			if tokens == 1 {
				if after != before {
					b.Fatalf("route counters before %+v after %+v: P=1 must bypass mm_cm1", before, after)
				}
				b.ReportMetric(float64(tokens), "tokens/op")
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					run()
				}
				return
			}
			wantGEMM := tokens > cfg.MinTokens
			if wantGEMM && after.GEMM != before.GEMM+1 {
				b.Fatalf("route counters before %+v after %+v: P=%d threshold=%d did not use GEMM", before, after, tokens, cfg.MinTokens)
			}
			if !wantGEMM && after.BelowThreshold != before.BelowThreshold+1 {
				b.Fatalf("route counters before %+v after %+v: P=%d threshold=%d did not retain matvec", before, after, tokens, cfg.MinTokens)
			}

			b.ReportMetric(float64(tokens), "tokens/op")
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				run()
			}
		})
	}
}

// fak-test:runtime slow est=15s
// TestVulkanMMCM1LargePromptMatchesCPU checks each GEMM variant at production prefill
// geometry (a 5120x5120 projection at P=512 and an ffn_down-width 17408 reduction at a
// P=777 token tail) against the CPU dequant on a fixed sample of output elements, so the
// oracle stays bounded while every row tile, token tile and K block is exercised.
func TestVulkanMMCM1LargePromptMatchesCPU(t *testing.T) {
	v := vk(t)
	prev := vulkanMMCM1SetConfig(VulkanMMCM1Config{Enabled: true, MinTokens: 8})
	t.Cleanup(func() { vulkanMMCM1SetConfig(prev) })
	for _, dt := range mmCM1Dtypes() {
		t.Run(dt.String(), func(t *testing.T) {
			requireMMCM1(t, dt)
			for _, sh := range []struct{ out, in, tokens int }{{5120, 5120, 512}, {640, 17408, 777}} {
				t.Run(fmt.Sprintf("out%d_in%d_p%d", sh.out, sh.in, sh.tokens), func(t *testing.T) {
					seed := int64(0x1a5e + sh.out + sh.in + int(dt)*7919)
					hw, deq := mmCM1Weight(t, dt, seed, sh.out, sh.in)
					dw := v.Upload(hw, dt)
					if dw.Dtype != dt {
						v.Free(dw)
						t.Skipf("native %v upload unavailable (uploaded as %v)", dt, dw.Dtype)
					}
					s := lcg(uint64(seed))
					x := randVec(&s, sh.tokens*sh.in)
					dx := v.Upload(NewF32(Default(), []int{sh.tokens, sh.in}, x), F32)
					defer func() { v.Free(dw); v.Free(dx); v.Recycle() }()
					before := VulkanMMCM1Snapshot()
					got := mmCM1SequenceMatMul(v, dw, dx, sh.tokens)
					if after := VulkanMMCM1Snapshot(); after.GEMM != before.GEMM+1 {
						t.Fatalf("route counters before %+v after %+v: GEMM did not serve the call", before, after)
					}
					rng := rand.New(rand.NewSource(seed))
					const samples = 2048
					want, gotS := make([]float32, samples), make([]float32, samples)
					for i := 0; i < samples; i++ {
						tok, row := rng.Intn(sh.tokens), rng.Intn(sh.out)
						if i < 4 {
							tok, row = []int{0, sh.tokens - 1, 0, sh.tokens - 1}[i], []int{0, 0, sh.out - 1, sh.out - 1}[i]
						}
						wr, xr := deq[row*sh.in:(row+1)*sh.in], x[tok*sh.in:(tok+1)*sh.in]
						var acc float64
						for k := range wr {
							acc += float64(mmCM1RoundF16(wr[k])) * float64(mmCM1RoundF16(xr[k]))
						}
						want[i], gotS[i] = float32(acc), got[tok*sh.out+row]
					}
					rawIQAssertClose(t, "GEMM vs fp16-input oracle (sampled)", want, gotS, 5e-4)
				})
			}
		})
	}
}
