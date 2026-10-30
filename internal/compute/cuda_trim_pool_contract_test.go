package compute

import (
	"strings"
	"testing"
)

// Source contract only; no native cleanup failure or CUDA execution is exercised.
// fak-test:runtime fast est=10ms lane=default
func TestCUDATrimPoolCheckedLifetimeContract(t *testing.T) {
	memory := readCUDASource(t, "cuda/runtime_memory.cuh")
	header := readCUDASource(t, "cuda_backend.h")
	goSource := readCUDASource(t, "cuda.go")
	ordered := func(source string, fragments ...string) {
		t.Helper()
		offset := 0
		for _, fragment := range fragments {
			next := strings.Index(source[offset:], fragment)
			if next < 0 {
				t.Fatalf("missing or reordered trim contract: %q", fragment)
			}
			offset += next + len(fragment)
		}
	}
	trim := cudaFunctionBody(t, memory, "fcuda_trim_pool_large")
	ordered(trim, "if (v41_candidates_stream_uncertain()) return;", "g_pool.begin()",
		"it->first > max_keep_bytes", "while (!bucket.empty())", "void *p = bucket.back();",
		"cudaError_t status = cudaFree(p);", "g_cuda_cleanup_status = (int)status;", "return;",
		"fak_alloc_untrack(p);", "bucket.pop_back();", "g_pool.erase(it)")
	if !strings.Contains(trim, "if (status != cudaSuccess) {\n          g_cuda_cleanup_status = (int)status;\n          CK(status);\n          return;\n        }\n        fak_alloc_untrack(p);") {
		t.Fatal("failed trim must retain its pointer and ledger before returning")
	}
	if !strings.Contains(header, "void fcuda_trim_pool_large(size_t max_keep_bytes);") ||
		!strings.Contains(header, "int fcuda_trim_pool_large_checked(size_t max_keep_bytes);") {
		t.Fatal("checked trim must preserve the legacy void ABI")
	}
	ordered(cudaFunctionBody(t, memory, "fcuda_trim_pool_large_checked"), "fcuda_trim_pool_large(max_keep_bytes);", "return g_cuda_cleanup_status;", "cudaErrorNotReady")
	ordered(cudaFunctionBody(t, goSource, "TrimLarge"), "maxKeepBytes = 0", "cudaMu.Lock()", "defer cudaMu.Unlock()",
		"C.fcuda_trim_pool_large_checked", "var cause error", "c.faultLatch.Observe(DeviceFaultExecution", "fault != nil", "cause = fault", "panic(&CUDAOpError{", "Code: status", "Err: cause")
	if strings.Contains(goSource, "C.fcuda_trim_pool_large(") {
		t.Fatal("Go trim must consume checked native status")
	}
}
