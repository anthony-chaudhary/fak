//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// vulkan_q3k_test.go is the device-side witness for fak#13677: the pinned
// DeepSeek-V4.1-Flash Q2_K stores its routed-expert DOWN projection as Q3_K, so the
// streamed-expert route stays device-only only once Q3_K has a real Vulkan GEMV.
// This mirrors vulkan_q5k_test.go: device execution parity against an independent
// F32 dequant oracle, plus a static source contract that ties the shader, the C++
// shim and the Go seam together even on a host with no reachable Vulkan device.

// TestVulkanQ3KMatMulStrix verifies that Vulkan consumes GGUF Q3_K bytes in place.
// A 110-byte super-block is only two-byte aligned, so the single-block case is
// deliberately 110 bytes (2 mod 4) and the host must still pad the resident
// allocation to four bytes. The wider cases cover alternating 110-byte block
// alignment, the V4.1 hidden/FFN reduction sizes, output workgroup tails, and
// P=1/2/4 dispatch.
// fak-test:runtime integration est=3s lane=optin
func TestVulkanQ3KMatMulStrix(t *testing.T) {
	if runVulkanProfileFixture(t) {
		return
	}
	v := vk(t)
	capability, ok := any(v).(interface{ SupportsQ3KMatMul() bool })
	if !ok || !capability.SupportsQ3KMatMul() {
		t.Fatal("registered Vulkan backend does not expose the native q3k_matmul pipeline")
	}
	window, available, err := BeginBackendExecutionObservation(v)
	if err != nil || !available {
		t.Fatalf("begin Vulkan Q3_K execution observation: available=%t err=%v", available, err)
	}
	defer func() {
		observation, endErr := window.End()
		if endErr != nil {
			t.Errorf("end Vulkan Q3_K execution observation: %v", endErr)
			return
		}
		if !observation.TransferCountersObserved || observation.Counters.ComputeDispatches == 0 {
			t.Errorf("Q3_K test did not prove observed Vulkan dispatch: %+v", observation)
		}
	}()

	for _, tc := range []struct {
		name       string
		out, in, p int
	}{
		{name: "single_superblock", out: 1, in: 256, p: 1},
		{name: "hidden_decode_tail", out: 67, in: 5120, p: 1},
		{name: "hidden_panel_tail", out: 33, in: 5120, p: 2},
		{name: "ffn_panel_tail", out: 17, in: 17408, p: 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := q3kStrixFixture(tc.out, tc.in)

			x := q5KStrixInput(tc.p, tc.in)
			want := q3kDequantF32Oracle(tc.out, tc.in, tc.p, raw, x)

			hostWeight := NewQ3K(Default(), []int{tc.out, tc.in}, raw)
			weight := v.Upload(hostWeight, Q3_K)
			defer v.Free(weight)
			if weight.Dtype != Q3_K {
				t.Fatalf("resident weight dtype=%s, want packed Q3_K", weight.Dtype)
			}
			resident, ok := weight.buf.(*vulkanBuf)
			if !ok || resident.ptr == nil {
				t.Fatal("Q3_K weight did not become a Vulkan resident buffer")
			}
			wantResidentBytes := (len(raw) + 3) &^ 3
			if resident.n != wantResidentBytes {
				t.Fatalf("resident Q3_K bytes=%d, want four-byte padded packed size %d (raw=%d)", resident.n, wantResidentBytes, len(raw))
			}
			if resident.n >= tc.out*tc.in*F32.Bytes() {
				t.Fatalf("resident Q3_K bytes=%d indicate F32 materialization (F32=%d)", resident.n, tc.out*tc.in*F32.Bytes())
			}

			shape := []int{tc.in}
			if tc.p > 1 {
				shape = []int{tc.p, tc.in}
			}
			dx := v.UploadClass(NewF32(Default(), shape, x), F32, MemoryActivation, "Q3_K test input")
			defer v.Free(dx)
			var output Tensor
			if tc.p == 1 {
				output = v.MatMul(weight, dx)
			} else {
				output = v.BatchedMatMul(weight, dx, tc.p)
			}
			defer v.Free(output)
			got := v.Read(output)
			if len(got) != tc.p*tc.out {
				t.Fatalf("output length=%d, want %d", len(got), tc.p*tc.out)
			}

			const minCosine = 0.99999
			if cosine := cosineC(got, want); math.IsNaN(cosine) || cosine < minCosine {
				t.Fatalf("packed Q3_K cosine=%.9f, want >= %.5f", cosine, minCosine)
			}
			maxDelta, maxReference := q5KMaxAbsDelta(got, want)
			const maxRelativeDelta = 2e-5
			if maxDelta > maxRelativeDelta*math.Max(1, maxReference) {
				t.Fatalf("packed Q3_K max-abs delta=%g exceeds relative gate %g (reference max=%g)", maxDelta, maxRelativeDelta, maxReference)
			}
			for token := 0; token < tc.p; token++ {
				start := token * tc.out
				if gotArgmax, wantArgmax := argmaxF32(got[start:start+tc.out]), argmaxF32(want[start:start+tc.out]); gotArgmax != wantArgmax {
					t.Fatalf("token %d argmax=%d, CPU oracle=%d", token, gotArgmax, wantArgmax)
				}
			}
		})
	}
}

// TestVulkanQ3KSourceContract pins the wiring a device-less host cannot execute: the
// q3k_matmul shader's packed-layout descriptors, the C++ shim's optional loader and
// entry point, and the Go seam's byte-size pad plus the Q3_K MatMul/BatchedMatMul
// cases. A future edit that drops any leg fails here without needing a GPU.
func TestVulkanQ3KSourceContract(t *testing.T) {
	// The build inventory is delegated to the versioned current registry; its
	// source-file placement is not the module inclusion contract.
	if !slices.Contains(computebuild.CurrentVulkanShaderRegistryV5(), "q3k_matmul") {
		t.Fatal("current Vulkan build registry is missing q3k_matmul")
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("os.Getwd failed: %v", err)
	}
	repoRoot := findRepoRootForTest(t, wd)
	checks := []struct {
		path    string
		clauses []string
	}{
		{
			path: filepath.Join(repoRoot, "internal", "compute", "shaders", "q3k_matmul.comp"),
			clauses: []string{
				"#version 450",
				"layout(std430, set = 0, binding = 0) readonly buffer Q3K",
				"layout(std430, set = 0, binding = 1) readonly buffer X",
				"layout(std430, set = 0, binding = 2) writeonly buffer Y",
				"110u",
			},
		},
		{
			path: filepath.Join(repoRoot, "internal", "compute", "vulkan_shim.cpp"),
			clauses: []string{
				`buildKernel(g_kern[K_Q3K_MATMUL], P("q3k_matmul.spv")`,
				"fvk_have_q3k_matmul",
				"fvk_q3k_matmul_f32",
			},
		},
		{
			path: filepath.Join(repoRoot, "internal", "compute", "vulkan.go"),
			clauses: []string{
				"C.fvk_q3k_matmul_f32",
				"case Q3_K:",
				"vulkanQ3KByteSizes",
			},
		},
	}
	for _, check := range checks {
		if _, err := os.Stat(check.path); err != nil {
			t.Fatalf("read %s: %v", check.path, err)
		}
		raw, err := os.ReadFile(check.path)
		if err != nil {
			t.Fatalf("read %s: %v", check.path, err)
		}
		for _, clause := range check.clauses {
			if !strings.Contains(string(raw), clause) {
				t.Errorf("%s missing Q3_K contract clause %q", filepath.Base(check.path), clause)
			}
		}
	}
}

// q3kStrixFixture produces deterministic finite Q3_K super-blocks: modest f16
// super-scales (2^-6 / 2^-7, alternating per block so adjacent 110-byte blocks are
// distinguishable) keep decoded weights small, and the hmask/low/scales bytes are
// exercised unconditionally.
func q3kStrixFixture(out, in int) []byte {
	blocksPerRow := in / q3kSuper
	raw := make([]byte, out*blocksPerRow*q3kSuperBlock)
	for block := 0; block < out*blocksPerRow; block++ {
		base := block * q3kSuperBlock
		d := uint16(0x2800) // 2^-6
		if block&1 != 0 {
			d = 0x2c00 // 2^-4
		}
		// 32 hmask bytes, 64 low-code bytes and 12 packed sub-scale bytes.
		for i := 0; i < q3kSuperBlock-2; i++ {
			raw[base+i] = byte((block*29 + i*37 + 11) & 0xff)
		}
		binary.LittleEndian.PutUint16(raw[base+q3kSuperBlock-2:], d)
	}
	return raw
}

// q3kDequantF32Oracle dequantizes the packed fixture with the compute reference Q3_K
// decoder (independent of the shader) and evaluates the same GEMM on the F32
// reference backend.
func q3kDequantF32Oracle(out, in, tokens int, raw []byte, x []float32) []float32 {
	weights := make([]float32, out*in)
	DequantQ3K(weights, raw)
	ref := Default()
	weight := NewF32(ref, []int{out, in}, weights)
	inputShape := []int{in}
	if tokens > 1 {
		inputShape = []int{tokens, in}
	}
	input := ref.Upload(NewF32(ref, inputShape, x), F32)
	if tokens == 1 {
		return ref.Read(ref.MatMul(weight, input))
	}
	return ref.Read(ref.BatchedMatMul(weight, input, tokens))
}
