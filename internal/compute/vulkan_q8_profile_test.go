//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"
)

// TestVulkanQ8CooperativeDecode checks partial output tiles and input windows.
// Opt-in FFN shapes measure a synthetic resident primitive, not model tokens/s.
func TestVulkanQ8CooperativeDecode(t *testing.T) {
	backend, available := Lookup("vulkan")
	if !available {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required Vulkan device is not registered")
		}
		t.Skip("Vulkan device is not registered")
	}
	v := backend.(*vulkanBackend)
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" && !strings.Contains(strings.ToLower(v.Tier()), strings.ToLower(expected)) {
		t.Fatalf("device %q does not match required %q", v.Tier(), expected)
	}
	if !v.haveQ8 {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required device lacks Q8 support")
		}
		t.Skip("device lacks Q8 support")
	}
	shapes := [][3]int{{37, 64, 1}, {259, 2080, 1}, {259, 2080, 3}}
	if os.Getenv("FAK_VULKAN_Q8_PROFILE") == "1" {
		shapes = append(shapes, [3]int{17408, 5120, 1}, [3]int{5120, 17408, 1}, [3]int{17408, 5120, 4})
	}
	if os.Getenv("FAK_VULKAN_Q8_PREFILL_PROFILE") == "1" {
		shapes = append(shapes, [3]int{17408, 5120, 16}, [3]int{5120, 17408, 16}, [3]int{17408, 5120, 64}, [3]int{5120, 17408, 64})
	}
	for _, shape := range shapes {
		t.Run(fmt.Sprintf("%dx%d_p%d", shape[0], shape[1], shape[2]), func(t *testing.T) {
			out, in, batch := shape[0], shape[1], shape[2]
			rng := rand.New(rand.NewSource(11963))
			// Quantize one row at a time to preserve the exact seeded fixture
			// without retaining a full FP32 copy beside the resident Q8 matrix.
			codes, scales := make([]int8, out*in), make([]float32, out*(in/32))
			rowWeights, input := make([]float32, in), make([]float32, batch*in)
			for row := 0; row < out; row++ {
				for i := range rowWeights {
					rowWeights[i] = (rng.Float32()*2 - 1) * .1
				}
				rowCodes, rowScales := quantizeVecQ8(rowWeights, 32)
				copy(codes[row*in:], rowCodes)
				copy(scales[row*(in/32):], rowScales)
			}
			for i := range input {
				input[i] = rng.Float32()*2 - 1
			}
			c := cpu()
			hw := NewQ8(c, []int{out, in}, codes, scales, 32)
			hx := NewF32(c, []int{batch, in}, input)
			defer c.Free(hw)
			defer c.Free(hx)
			hy := c.BatchedMatMul(hw, hx, batch)
			want := c.Read(hy)
			defer c.Free(hy)
			w, x := v.Upload(hw, Q8_0), v.Upload(hx, F32)
			defer v.Free(w)
			defer v.Free(x)
			var samples []int64
			worstRelativeL2, worstMaxAbs := 0.0, 0.0
			for step := -1; step < 6; step++ {
				if step == -1 {
					v.BeginBatch()
				}
				started := time.Now()
				y := v.BatchedMatMul(w, x, batch)
				if step == -1 {
					v.FlushBatch()
				}
				got := v.Read(y)
				elapsed := time.Since(started).Nanoseconds()
				v.Free(y)
				if len(got) != len(want) {
					t.Fatalf("output elements=%d, want %d", len(got), len(want))
				}
				var squaredError, squaredReference, maxDifference float64
				for i, value := range got {
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
						t.Fatalf("non-finite output at %d", i)
					}
					delta := float64(value) - float64(want[i])
					squaredError += delta * delta
					squaredReference += float64(want[i]) * float64(want[i])
					maxDifference = math.Max(maxDifference, math.Abs(delta))
				}
				relativeL2 := math.Sqrt(squaredError / squaredReference)
				if !(squaredReference > 0) || math.IsNaN(relativeL2) || relativeL2 > 1e-4 || maxDifference > 1e-3 || argmaxF32(got) != argmaxF32(want) {
					t.Fatalf("step %d: relative L2=%g max abs=%g argmax=%d want=%d", step, relativeL2, maxDifference, argmaxF32(got), argmaxF32(want))
				}
				for token := 0; token < batch; token++ {
					lo, hi := token*out, (token+1)*out
					if argmaxF32(got[lo:hi]) != argmaxF32(want[lo:hi]) {
						t.Fatalf("step %d token %d: argmax mismatch", step, token)
					}
				}
				worstRelativeL2 = math.Max(worstRelativeL2, relativeL2)
				worstMaxAbs = math.Max(worstMaxAbs, maxDifference)
				if step > 0 {
					samples = append(samples, elapsed)
				}
			}
			encoded, err := json.Marshal(map[string]any{
				"schema": "fak.vulkan.q8-primitive-profile.v1", "engine": "fak-native-compute",
				"scope": "synthetic resident Q8 matrix multiplication; no model execution", "device": v.Tier(),
				"shape": [2]int{out, in}, "batch": batch, "fixture_seed": 11963, "warmup_samples": 1, "batched_validation_samples": 1,
				"dispatch_and_output_read_ns": samples, "max_relative_l2": worstRelativeL2,
				"max_abs_error": worstMaxAbs, "argmax_exact": true, "per_token_argmax_exact": true,
				"timing_note": "wall time includes output allocation and synchronized read; excludes upload, validation and free",
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Log(string(encoded))
		})
	}
}
