package compute

import (
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestQwen35VLLMFusedQKSourceContract(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	kernel := read("cuda/qwen35_qknorm_rope_cache_vllm.cuh")
	for _, required := range []string{
		"SPDX-License-Identifier: Apache-2.0",
		"Copyright contributors to the vLLM project",
		"fa3b9d1377219247060d3c8b49eae199c9e7e154",
		"csrc/fused_qknorm_rope_cache_kernel.cu",
		"__fmaf_rn(value, value, sum_squares)",
		"__shfl_xor_sync(subwarp_mask, sum_squares, offset, kWarpSize)",
		"if (is_k) KRawCache[cache_offset + dim] = normalized_head[dim]",
		"FAK_VLLM_QK_LAUNCH(192, true)",
		"cudaStreamSynchronize(g_stream)",
		"g_cuda_cleanup_status = (int)drained",
	} {
		if !strings.Contains(kernel, required) {
			t.Errorf("missing donor/ownership contract %q", required)
		}
	}
	if !strings.Contains(read("cuda_kernels.cu"), "#include \"cuda/qwen35_qknorm_rope_cache_vllm.cuh\"") {
		t.Fatal("candidate is not built into the native CUDA archive")
	}
	wrapper := read("cuda_qwen35_qknorm_rope_cache_vllm.go")
	if !strings.Contains(wrapper, "os.Getenv(\"FAK_CUDA_QWEN35_VLLM_QK\") == \"1\"") ||
		!strings.Contains(wrapper, "func (k *cudaKV) AppendQKNormRoPE(") {
		t.Fatal("ordinary KV capability or explicit candidate gate is missing")
	}
	if os.Getenv("FAK_CUDA_GDN_REQUIRED") == "1" {
		if be, ok := Lookup("cuda"); !ok || be == nil || be.Name() != "cuda" {
			t.Fatal("required physical witness cannot pass in a non-CUDA build")
		}
	}
}
