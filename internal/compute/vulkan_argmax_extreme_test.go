//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// These fixtures have independently specified first-max indices. They do not
// use another Vulkan argmax path as an oracle. NaN and empty input are outside
// this finite-input regression contract.
func vulkanExtremeArgmaxValues(n, winner, tied int) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = -3e30
	}
	values[winner] = -2e30
	if tied >= 0 {
		values[tied] = -2e30
	}
	return values
}

// Estimate only; the device test has not been timed.
// fak-test:runtime integration est=1s lane=optin
func TestVulkanArgmaxExtremeFiniteIdentity(t *testing.T) {
	v := vk(t)
	for _, tc := range []struct {
		name            string
		n, winner, tied int
	}{
		{"two_values", 2, 1, -1},
		{"padded_last_lane", 257, 256, -1},
		{"strided_last_lane", 513, 512, -1},
		{"cross_lane_tie", 257, 17, 256},
		{"cross_group_tie", 513, 256, 512},
	} {
		t.Run(tc.name, func(t *testing.T) {
			x := v.Upload(NewF32(Default(), []int{tc.n}, vulkanExtremeArgmaxValues(tc.n, tc.winner, tc.tied)), F32)
			defer v.Free(x)
			if got := v.Argmax(x); got != tc.winner {
				t.Fatalf("argmax n=%d got=%d want independent first maximum%d", tc.n, got, tc.winner)
			}
		})
	}
	for _, n := range []int{1, 257, 513} {
		values := make([]float32, n)
		for i := range values {
			values[i] = -math.MaxFloat32
		}
		x := v.Upload(NewF32(Default(), []int{n}, values), F32)
		got := v.Argmax(x)
		v.Free(x)
		if got != 0 {
			t.Fatalf("all lowest finite values n=%d chose%d, want first index0", n, got)
		}
	}
}

// Estimate only; the device test has not been timed.
// fak-test:runtime integration est=2s lane=optin
func TestVulkanFusedArgmaxExtremeFiniteIdentity(t *testing.T) {
	v := vk(t)
	for _, tc := range []struct {
		name                  string
		out, in, winner, tied int
	}{
		{"single_group", 2, 1, 1, -1},
		{"single_group_wide_staging", 2, 1025, 1, -1},
		{"blocks_final_tail", 257, 1, 256, -1},
		{"three_blocks_final_tail", 513, 1, 512, -1},
		{"blocks_first_tie", 513, 1, 256, 512},
	} {
		t.Run(tc.name, func(t *testing.T) {
			logits := vulkanExtremeArgmaxValues(tc.out, tc.winner, tc.tied)
			weights := make([]float32, tc.out*tc.in)
			for row, value := range logits {
				weights[row*tc.in] = value
			}
			input := make([]float32, tc.in)
			input[0] = 1
			w := v.Upload(NewF32(Default(), []int{tc.out, tc.in}, weights), F32)
			defer v.Free(w)
			x := v.Upload(NewF32(Default(), []int{tc.in}, input), F32)
			defer v.Free(x)
			if got := v.MatMulArgmax(w, x); got != tc.winner {
				t.Fatalf("fused argmax got%d want independent first maximum%d", got, tc.winner)
			}
			if tc.in == 1 {
				// mean(x*x)=1, eps=3 gives an exact positive scale0.5. All
				// products remain finite and ordering/ties are unchanged.
				norm := v.Upload(NewF32(Default(), []int{1}, []float32{1}), F32)
				defer v.Free(norm)
				if got := v.RMSNormMatMulArgmax(w, x, norm, 3); got != tc.winner {
					t.Fatalf("norm fused argmax got%d want independent first maximum%d", got, tc.winner)
				}
			}
		})
	}
}

// Estimate only; this source check has not been timed.
// fak-test:runtime fast est=1ms lane=default
func TestVulkanArgmaxIdentitySourceContract(t *testing.T) {
	for _, name := range []string{"argmax.comp", "argmax_pairs.comp", "matmul_argmax.comp", "matmul_argmax_blocks.comp", "rmsnorm_matmul_argmax_blocks.comp"} {
		raw, err := os.ReadFile(filepath.Join("shaders", name))
		if err != nil {
			t.Fatal(err)
		}
		source := string(raw)
		if strings.Count(source, "uintBitsToFloat(0xff800000u)") != 1 || strings.Contains(source, "-1e30") {
			t.Errorf("%s lacks exact negative-infinity max identity", name)
		}
	}
}
