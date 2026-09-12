//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// Keep the primitive at the same error ceiling as the full-model promotion
// contract. The exact-FP32 path must clear this before a model run.
const q4KTiledF32MaxNormalizedResidual = 1e-4

func q4KTiledF32ExplicitlyEnabled() bool {
	return os.Getenv("FAK_VULKAN_Q4K_TILED_FP32") == "candidate"
}

func q4KTiledF32Device(tb testing.TB) *vulkanBackend {
	tb.Helper()
	backend, ok := Lookup("vulkan")
	if !ok {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" || q4KTiledF32ExplicitlyEnabled() {
			tb.Fatal("required Vulkan Q4_K exact-FP32 tiled device is not registered")
		}
		tb.Skip("Vulkan backend unavailable")
	}
	v, ok := backend.(*vulkanBackend)
	if !ok {
		tb.Fatalf("Vulkan backend has unexpected type %T", backend)
	}
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" && !strings.Contains(strings.ToLower(v.Tier()), strings.ToLower(expected)) {
		tb.Fatalf("device %q does not match required %q", v.Tier(), expected)
	}
	if !v.VulkanQ4KTiledF32Available() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" || q4KTiledF32ExplicitlyEnabled() {
			tb.Fatalf("Vulkan device %q lacks required native Q4_K exact-FP32 tiled capability", v.Tier())
		}
		tb.Skipf("Vulkan device %q lacks native Q4_K exact-FP32 tiled capability", v.Tier())
	}
	tb.Logf("Vulkan device: %s", v.Tier())
	return v
}

func q4KTiledF32Fixture(out, in, tokens, seed int) (Tensor, Tensor) {
	raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
	for block := range len(raw) / q4kSuperBlock {
		fillCraftedQ4KBlock(raw[block*q4kSuperBlock:(block+1)*q4kSuperBlock], block+seed)
	}
	x := make([]float32, tokens*in)
	for token := range tokens {
		for column := range in {
			phase := float64((token+1)*(column+3)+seed*11) * 0.013
			x[token*in+column] = 0.35 + 0.3*float32(math.Sin(phase)) + 0.1*float32(math.Cos(phase*0.37))
		}
	}
	return NewQ4K(Default(), []int{out, in}, raw), NewF32(Default(), []int{tokens, in}, x)
}

func checkQ4KTiledF32Parity(t *testing.T, got, want []float32, tokens, out int) {
	t.Helper()
	if len(got) != len(want) || len(got) != tokens*out {
		t.Fatalf("output length = %d, CPU reference = %d, want %d", len(got), len(want), tokens*out)
	}
	var squaredError, squaredReference float64
	for token := range tokens {
		gotRow := got[token*out : (token+1)*out]
		wantRow := want[token*out : (token+1)*out]
		for row, value := range gotRow {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("token %d output %d is non-finite: %v", token, row, value)
			}
			reference := wantRow[row]
			if math.IsNaN(float64(reference)) || math.IsInf(float64(reference), 0) {
				t.Fatalf("token %d CPU reference %d is non-finite: %v", token, row, reference)
			}
			difference := float64(value - reference)
			squaredError += difference * difference
			squaredReference += float64(reference) * float64(reference)
		}
		if gotArgmax, wantArgmax := argmaxF32(gotRow), argmaxF32(wantRow); gotArgmax != wantArgmax {
			t.Fatalf("token %d argmax = %d, CPU reference = %d", token, gotArgmax, wantArgmax)
		}
		cosine := cosineC(gotRow, wantRow)
		if math.IsNaN(cosine) || math.IsInf(cosine, 0) || cosine < 0.99999 {
			t.Fatalf("token %d cosine %.8f is non-finite or < 0.99999", token, cosine)
		}
	}
	if squaredReference == 0 || math.IsNaN(squaredReference) || math.IsInf(squaredReference, 0) ||
		math.IsNaN(squaredError) || math.IsInf(squaredError, 0) {
		t.Fatalf("invalid residual inputs: squared_error=%g squared_reference=%g", squaredError, squaredReference)
	}
	// The candidate must use enough operand precision to satisfy the same 1e-4
	// ceiling as whole-model qualification. Cosine and argmax alone can hide
	// systematic projection error.
	normalizedResidual := math.Sqrt(squaredError / squaredReference)
	if math.IsNaN(normalizedResidual) || math.IsInf(normalizedResidual, 0) || normalizedResidual > q4KTiledF32MaxNormalizedResidual {
		t.Fatalf("normalized residual %.8g is non-finite or > %.8g", normalizedResidual, q4KTiledF32MaxNormalizedResidual)
	}
	t.Logf("Q4_K tiled FP32 parity: tokens=%d out=%d samples=%d cosine>=0.99999 normalized_residual=%.8g", tokens, out, len(got), normalizedResidual)
}

func runQ4KTiledF32CandidateParity(t *testing.T, v *vulkanBackend, hostWeight, hostInput Tensor, tokens, out int) {
	t.Helper()
	want := Default().Read(Default().BatchedMatMul(hostWeight, hostInput, tokens))
	deviceWeight := v.Upload(hostWeight, Q4_K)
	defer v.Free(deviceWeight)
	deviceInput := v.Upload(hostInput, F32)
	defer v.Free(deviceInput)
	before := v.VulkanDebugQ4KTiledF32Dispatches()
	deviceOutput := v.BatchedMatMul(deviceWeight, deviceInput, tokens)
	defer v.Free(deviceOutput)
	got := v.Read(deviceOutput)
	delta := v.VulkanDebugQ4KTiledF32Dispatches() - before
	if delta == 0 {
		t.Fatal("candidate overflow fallback produced no exact-FP32 tiled dispatch")
	}
	t.Logf("Q4_K tiled FP32 route: tokens=%d out=%d dispatch_delta=%d", tokens, out, delta)
	checkQ4KTiledF32Parity(t, got, want, tokens, out)
}

func q4KTiledF32ResidualAndBitMismatches(got, want []float32) (float64, int) {
	var squaredError, squaredReference float64
	bitMismatches := 0
	for i, value := range got {
		reference := want[i]
		difference := float64(value - reference)
		squaredError += difference * difference
		squaredReference += float64(reference) * float64(reference)
		if math.Float32bits(value) != math.Float32bits(reference) {
			bitMismatches++
		}
	}
	return math.Sqrt(squaredError / squaredReference), bitMismatches
}

func runQ4KTiledF32ActualReductionDiagnostic(t *testing.T, v *vulkanBackend, hostWeight, hostInput Tensor, tokens, out int) {
	t.Helper()
	want := Default().Read(Default().BatchedMatMul(hostWeight, hostInput, tokens))
	deviceWeight := v.Upload(hostWeight, Q4_K)
	defer v.Free(deviceWeight)
	deviceInput := v.Upload(hostInput, F32)
	defer v.Free(deviceInput)

	run := func(arm string) ([]float32, uint64) {
		t.Helper()
		t.Setenv("FAK_VULKAN_Q4K_ARM", arm)
		before := v.VulkanDebugQ4KTiledF32Dispatches()
		deviceOutput := v.BatchedMatMul(deviceWeight, deviceInput, tokens)
		defer v.Free(deviceOutput)
		return v.Read(deviceOutput), v.VulkanDebugQ4KTiledF32Dispatches() - before
	}

	scalar, scalarDispatches := run("scalar")
	candidate, candidateDispatches := run("")
	if scalarDispatches != 0 {
		t.Fatalf("forced GPU scalar arm selected %d tiled FP32 dispatches", scalarDispatches)
	}
	if candidateDispatches == 0 {
		t.Fatal("actual-reduction candidate produced no tiled FP32 dispatch")
	}
	candidateCPUResidual, candidateCPUBits := q4KTiledF32ResidualAndBitMismatches(candidate, want)
	scalarCPUResidual, scalarCPUBits := q4KTiledF32ResidualAndBitMismatches(scalar, want)
	candidateScalarResidual, candidateScalarBits := q4KTiledF32ResidualAndBitMismatches(candidate, scalar)
	t.Logf("Q4_K actual-reduction diagnostic: tokens=%d out=%d in=%d candidate_dispatches=%d candidate_cpu_residual=%.8g candidate_cpu_bit_mismatches=%d scalar_cpu_residual=%.8g scalar_cpu_bit_mismatches=%d candidate_scalar_residual=%.8g candidate_scalar_bit_mismatches=%d",
		tokens, out, hostInput.Shape[len(hostInput.Shape)-1], candidateDispatches,
		candidateCPUResidual, candidateCPUBits, scalarCPUResidual, scalarCPUBits,
		candidateScalarResidual, candidateScalarBits)
	if candidateScalarBits != 0 || candidateScalarResidual != 0 {
		t.Fatalf("actual-reduction tiled FP32 result differs from GPU scalar: residual=%.8g bit_mismatches=%d", candidateScalarResidual, candidateScalarBits)
	}
	checkQ4KTiledF32Parity(t, scalar, want, tokens, out)
	checkQ4KTiledF32Parity(t, candidate, want, tokens, out)
}

func TestVulkanQ4KTiledF32PrefillMatchesCPUReference(t *testing.T) {
	v := q4KTiledF32Device(t)
	t.Setenv("FAK_VULKAN_Q4K_TILED_FP32", "candidate")
	t.Setenv("FAK_VULKAN_Q4K_ARM", "")

	tests := []struct {
		name            string
		tokens, out, in int
		wantCandidate   bool
	}{
		{name: "decode_p1_odd_output", tokens: 1, out: 67, in: 256},
		{name: "prefill_p4", tokens: 4, out: 64, in: 256, wantCandidate: true},
		{name: "prefill_p16_odd_output", tokens: 16, out: 67, in: 512, wantCandidate: true},
		{name: "prefill_p64", tokens: 64, out: 64, in: 512, wantCandidate: true},
		{name: "prefill_p512_odd_output", tokens: 512, out: 67, in: 512, wantCandidate: true},
	}
	for index, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			hostWeight, hostInput := q4KTiledF32Fixture(test.out, test.in, test.tokens, 1709+index*97)
			want := Default().Read(Default().BatchedMatMul(hostWeight, hostInput, test.tokens))
			deviceWeight := v.Upload(hostWeight, Q4_K)
			defer v.Free(deviceWeight)
			deviceInput := v.Upload(hostInput, F32)
			defer v.Free(deviceInput)

			before := v.VulkanDebugQ4KTiledF32Dispatches()
			deviceOutput := v.BatchedMatMul(deviceWeight, deviceInput, test.tokens)
			defer v.Free(deviceOutput)
			got := v.Read(deviceOutput)
			delta := v.VulkanDebugQ4KTiledF32Dispatches() - before
			if test.wantCandidate && delta == 0 {
				t.Fatal("candidate was requested for multi-token prefill but no exact-FP32 tiled dispatch was observed")
			}
			if !test.wantCandidate && delta != 0 {
				t.Fatalf("single-token decode selected exact-FP32 tiled path (%d dispatches)", delta)
			}
			t.Logf("Q4_K tiled FP32 route: tokens=%d out=%d dispatch_delta=%d", test.tokens, test.out, delta)
			checkQ4KTiledF32Parity(t, got, want, test.tokens, test.out)
		})
	}

	t.Run("finite_dequant_exceeds_fp16", func(t *testing.T) {
		const tokens, out, in = 4, 67, 256
		raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
		for block := range len(raw) / q4kSuperBlock {
			packed := raw[block*q4kSuperBlock : (block+1)*q4kSuperBlock]
			fillCraftedQ4KBlock(packed, block+4703)
			binary.LittleEndian.PutUint16(packed[0:2], Float32ToFloat16Bits(65504))
			binary.LittleEndian.PutUint16(packed[2:4], 0)
		}
		x := make([]float32, tokens*in)
		for index := range x {
			x[index] = 0.001 + 0.0002*float32(math.Sin(float64(index+1)*0.17))
		}
		hostWeight := NewQ4K(Default(), []int{out, in}, raw)
		hostInput := NewF32(Default(), []int{tokens, in}, x)
		runQ4KTiledF32CandidateParity(t, v, hostWeight, hostInput, tokens, out)
	})

	t.Run("finite_activation_exceeds_fp16", func(t *testing.T) {
		const tokens, out, in = 4, 67, 256
		hostWeight, _ := q4KTiledF32Fixture(out, in, tokens, 5107)
		x := make([]float32, tokens*in)
		for token := range tokens {
			for column := range in {
				x[token*in+column] = 0.2 + 0.05*float32(math.Cos(float64(token*in+column+1)*0.11))
			}
			x[token*in+(token*47)%in] = 70000 + float32(token*1000)
		}
		hostInput := NewF32(Default(), []int{tokens, in}, x)
		runQ4KTiledF32CandidateParity(t, v, hostWeight, hostInput, tokens, out)
	})

	t.Run("precision_stress", func(t *testing.T) {
		const tokens, out, in = 16, 67, 512
		hostWeight, _ := q4KTiledF32Fixture(out, in, tokens, 6101)
		x := make([]float32, tokens*in)
		for token := range tokens {
			for column := range in {
				// Signed, non-smooth magnitudes prevent first-order FP16 operand
				// error from averaging away as it does in the smooth fixture.
				hash := uint32((token+1)*2654435761) ^ uint32((column+11)*2246822519)
				sign := float32(1)
				if hash&1 != 0 {
					sign = -1
				}
				magnitude := float32(0.03125) + float32((hash>>8)&0xffff)/65535*31.5
				x[token*in+column] = sign * magnitude
			}
		}
		runQ4KTiledF32CandidateParity(t, v, hostWeight, NewF32(Default(), []int{tokens, in}, x), tokens, out)
	})

	// The short-K fixtures above catch operand splitting defects, but the dense
	// Qwen3.8 projections reduce over 5120 or 17408 inputs. Keep output small so
	// these required-device diagnostics isolate reduction width and cancellation
	// without turning the primitive test into another whole-model run.
	for _, test := range []struct {
		name         string
		in           int
		cancellation bool
	}{
		{name: "actual_hidden_reduction_smooth", in: 5120},
		{name: "actual_ffn_reduction_signed_cancellation", in: 17408, cancellation: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			const tokens, out = 32, 129
			hostWeight, hostInput := q4KTiledF32Fixture(out, test.in, tokens, 7103+test.in)
			if test.cancellation {
				x := make([]float32, tokens*test.in)
				for token := range tokens {
					for column := range test.in {
						hash := uint32((token+3)*2654435761) ^ uint32((column+17)*2246822519)
						sign := float32(1)
						if hash&1 != 0 {
							sign = -1
						}
						magnitude := float32(0.03125) + float32((hash>>8)&0xffff)/65535*31.5
						x[token*test.in+column] = sign * magnitude
					}
				}
				hostInput = NewF32(Default(), []int{tokens, test.in}, x)
			}
			runQ4KTiledF32ActualReductionDiagnostic(t, v, hostWeight, hostInput, tokens, out)
		})
	}

	t.Run("scalar_override_wins", func(t *testing.T) {
		t.Setenv("FAK_VULKAN_Q4K_ARM", "scalar")
		const tokens, out, in = 4, 67, 512
		hostWeight, hostInput := q4KTiledF32Fixture(out, in, tokens, 2903)
		want := Default().Read(Default().BatchedMatMul(hostWeight, hostInput, tokens))
		deviceWeight := v.Upload(hostWeight, Q4_K)
		defer v.Free(deviceWeight)
		deviceInput := v.Upload(hostInput, F32)
		defer v.Free(deviceInput)
		before := v.VulkanDebugQ4KTiledF32Dispatches()
		deviceOutput := v.BatchedMatMul(deviceWeight, deviceInput, tokens)
		defer v.Free(deviceOutput)
		got := v.Read(deviceOutput)
		if delta := v.VulkanDebugQ4KTiledF32Dispatches() - before; delta != 0 {
			t.Fatalf("scalar override selected exact-FP32 tiled path (%d dispatches)", delta)
		}
		checkQ4KTiledF32Parity(t, got, want, tokens, out)
	})
}

func TestVulkanQ4KTiledF32DefaultRemainsScalar(t *testing.T) {
	v := q4KTiledF32Device(t)
	oldTiled, hadTiled := os.LookupEnv("FAK_VULKAN_Q4K_TILED_FP32")
	oldArm, hadArm := os.LookupEnv("FAK_VULKAN_Q4K_ARM")
	t.Cleanup(func() {
		if hadTiled {
			_ = os.Setenv("FAK_VULKAN_Q4K_TILED_FP32", oldTiled)
		} else {
			_ = os.Unsetenv("FAK_VULKAN_Q4K_TILED_FP32")
		}
		if hadArm {
			_ = os.Setenv("FAK_VULKAN_Q4K_ARM", oldArm)
		} else {
			_ = os.Unsetenv("FAK_VULKAN_Q4K_ARM")
		}
	})

	run := func(name string, tokens int, tiledEnv, armEnv *string, wantDispatch bool) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			if tiledEnv == nil {
				_ = os.Unsetenv("FAK_VULKAN_Q4K_TILED_FP32")
			} else {
				_ = os.Setenv("FAK_VULKAN_Q4K_TILED_FP32", *tiledEnv)
			}
			if armEnv == nil {
				_ = os.Unsetenv("FAK_VULKAN_Q4K_ARM")
			} else {
				_ = os.Setenv("FAK_VULKAN_Q4K_ARM", *armEnv)
			}
			const out, in = 67, 512
			hostWeight, hostInput := q4KTiledF32Fixture(out, in, tokens, 8209+tokens)
			want := Default().Read(Default().BatchedMatMul(hostWeight, hostInput, tokens))
			deviceWeight := v.Upload(hostWeight, Q4_K)
			defer v.Free(deviceWeight)
			deviceInput := v.Upload(hostInput, F32)
			defer v.Free(deviceInput)
			before := v.VulkanDebugQ4KTiledF32Dispatches()
			deviceOutput := v.BatchedMatMul(deviceWeight, deviceInput, tokens)
			defer v.Free(deviceOutput)
			got := v.Read(deviceOutput)
			delta := v.VulkanDebugQ4KTiledF32Dispatches() - before
			if wantDispatch && delta == 0 {
				t.Fatal("expected exact-FP32 tiled dispatch, observed none")
			}
			if !wantDispatch && delta != 0 {
				t.Fatalf("expected scalar route, observed %d exact-FP32 tiled dispatches", delta)
			}
			checkQ4KTiledF32Parity(t, got, want, tokens, out)
		})
	}
	value := func(s string) *string { return &s }
	run("unset_p32_scalar", 32, nil, nil, false)
	run("empty_p32_scalar", 32, value(""), nil, false)
	run("unset_p16_scalar", 16, nil, nil, false)
	run("unset_p1_scalar", 1, nil, nil, false)
	run("candidate_p16", 16, value("candidate"), nil, true)
	run("candidate_to_unset_p16_does_not_stick", 16, nil, nil, false)
	run("off_p32_disables", 32, value("off"), nil, false)
	run("disable_to_empty_p32_retains_scalar", 32, value(""), nil, false)
	run("zero_p32_disables", 32, value("0"), nil, false)
	run("scalar_override_wins", 32, value("candidate"), value("scalar"), false)
	run("zero_arm_override_wins", 32, value("candidate"), value("0"), false)
}

func BenchmarkVulkanQ4KTiledF32Prefill(b *testing.B) {
	v := q4KTiledF32Device(b)
	oldTiled, hadTiled := os.LookupEnv("FAK_VULKAN_Q4K_TILED_FP32")
	oldArm, hadArm := os.LookupEnv("FAK_VULKAN_Q4K_ARM")
	b.Cleanup(func() {
		if hadTiled {
			_ = os.Setenv("FAK_VULKAN_Q4K_TILED_FP32", oldTiled)
		} else {
			_ = os.Unsetenv("FAK_VULKAN_Q4K_TILED_FP32")
		}
		if hadArm {
			_ = os.Setenv("FAK_VULKAN_Q4K_ARM", oldArm)
		} else {
			_ = os.Unsetenv("FAK_VULKAN_Q4K_ARM")
		}
	})
	_ = os.Setenv("FAK_VULKAN_Q4K_TILED_FP32", "candidate")

	for _, tokens := range []int{8, 32, 128} {
		b.Run(fmt.Sprintf("P%d", tokens), func(b *testing.B) {
			const out, in = 4096, 5120
			hostWeight, hostInput := q4KTiledF32Fixture(out, in, tokens, 3907+tokens)
			deviceWeight := v.Upload(hostWeight, Q4_K)
			defer v.Free(deviceWeight)
			deviceInput := v.Upload(hostInput, F32)
			defer v.Free(deviceInput)

			const samplesPerIteration = 6
			run := func(arm string) (int64, uint64) {
				b.Helper()
				if err := os.Setenv("FAK_VULKAN_Q4K_ARM", arm); err != nil {
					b.Fatalf("set Q4_K arm: %v", err)
				}
				before := v.VulkanDebugQ4KTiledF32Dispatches()
				start := time.Now()
				output := v.BatchedMatMul(deviceWeight, deviceInput, tokens)
				values := v.Read(output)
				v.Free(output)
				elapsed := time.Since(start).Nanoseconds()
				for index, value := range values {
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
						b.Fatalf("arm=%q output %d is non-finite: %v", arm, index, value)
					}
				}
				delta := v.VulkanDebugQ4KTiledF32Dispatches() - before
				if arm == "scalar" && delta != 0 {
					b.Fatalf("scalar benchmark selected exact-FP32 tiled path (%d dispatches)", delta)
				}
				if arm == "" && delta == 0 {
					b.Fatal("candidate benchmark produced no exact-FP32 tiled dispatch")
				}
				return elapsed, delta
			}

			// One synchronized warmup per arm precedes the measured paired samples.
			_, _ = run("scalar")
			_, _ = run("")
			var scalarNS, candidateNS []int64
			var scalarDispatches, candidateDispatches []uint64
			b.ResetTimer()
			for range b.N {
				for sample := range samplesPerIteration {
					if sample%2 == 0 {
						ns, dispatches := run("scalar")
						scalarNS = append(scalarNS, ns)
						scalarDispatches = append(scalarDispatches, dispatches)
						ns, dispatches = run("")
						candidateNS = append(candidateNS, ns)
						candidateDispatches = append(candidateDispatches, dispatches)
						continue
					}
					ns, dispatches := run("")
					candidateNS = append(candidateNS, ns)
					candidateDispatches = append(candidateDispatches, dispatches)
					ns, dispatches = run("scalar")
					scalarNS = append(scalarNS, ns)
					scalarDispatches = append(scalarDispatches, dispatches)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(len(scalarNS)), "samples/arm")
			b.Logf("Q4_K prefill primitive receipt: tokens=%d out=%d in=%d timing_scope=MatMul+Read+Free scalar_ns=%v candidate_ns=%v scalar_dispatch_deltas=%v candidate_dispatch_deltas=%v",
				tokens, out, in, scalarNS, candidateNS, scalarDispatches, candidateDispatches)
		})
	}
}
