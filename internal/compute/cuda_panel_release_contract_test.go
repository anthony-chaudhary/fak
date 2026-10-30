package compute

import (
	"strings"
	"testing"
)

// Source contract only: this does not qualify native execution or fault recovery.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAPanelReleaseCheckedLifetimeContract(t *testing.T) {
	memory := readCUDASource(t, "cuda/runtime_memory.cuh")
	services := readCUDASource(t, "cuda/runtime_services.cuh")
	root := readCUDASource(t, "cuda_kernels.cu")
	goSource := readCUDASource(t, "cuda_prompt_panel.go")
	header := readCUDASource(t, "cuda_backend.h")
	ordered := func(source string, fragments ...string) {
		t.Helper()
		previous := 0
		for _, fragment := range fragments {
			next := strings.Index(source[previous:], fragment)
			if next < 0 {
				t.Fatalf("missing or reordered cleanup contract: %q", fragment)
			}
			previous += next + len(fragment)
		}
	}
	ordered(root, "static int g_cuda_cleanup_status = 0;", "static bool v41_candidates_stream_uncertain(", "if (g_cuda_cleanup_status != 0) return true;")
	if strings.Count(root+memory+services, "g_cuda_cleanup_status = 0;") != 1 {
		t.Fatal("cleanup uncertainty must not reset after initialization")
	}
	release := cudaFunctionBody(t, memory, "fcuda_panel_release")
	ordered(release, "if (v41_candidates_stream_uncertain()) return g_panel_live.size();",
		"cudaError_t status = cudaStreamSynchronize(g_stream);",
		"g_cuda_cleanup_status = (int)status;", "return g_panel_live.size();",
		"for (auto it = g_panel_pool.begin()", "status = cudaFree(p);",
		"fak_alloc_untrack(p);", "bucket.pop_back();", "g_panel_cached_bytes -= it->first;", "g_panel_pool.erase(it)")
	if !strings.Contains(release, "if (status != cudaSuccess) {\n        g_cuda_cleanup_status = (int)status;\n        CK(status);\n        return g_panel_live.size();\n      }\n      fak_alloc_untrack(p);") {
		t.Fatal("failed free must retain its entry and ledger before returning")
	}
	for _, check := range []struct{ source, name, firstEffect string }{
		{memory, "fcuda_malloc_tagged", "g_pool.find(bytes)"},
		{memory, "fcuda_malloc_managed_tagged", "cudaMallocManaged("},
		{memory, "fcuda_malloc_panel_tagged", "g_panel_pool.find(bytes)"},
		{services, "fcuda_malloc_on_tagged", "fcuda_device_scope scope(device)"},
		{memory, "fcuda_graph_prewarm", "g_pool.size()"},
		{memory, "fcuda_free", "q4_coeff_cache_evict(d)"},
		{memory, "fcuda_discard_unsubmitted", "cudaStreamSynchronize(g_stream)"},
		{services, "fcuda_free_on", "fcuda_device_scope scope(device)"},
	} {
		ordered(cudaFunctionBody(t, check.source, check.name), "if (g_cuda_cleanup_status != 0) return", check.firstEffect)
	}
	ordered(cudaFunctionBody(t, memory, "fcuda_trim_pool_large"), "v41_candidates_stream_uncertain()", "g_pool.begin()")
	if !strings.Contains(header, "int fcuda_panel_release_checked(size_t *live);") ||
		!strings.Contains(header, "size_t fcuda_panel_release(void);") {
		t.Fatal("checked cleanup must preserve the count-only ABI")
	}
	ordered(cudaFunctionBody(t, memory, "fcuda_panel_release_checked"), "if (!live)", "*live = fcuda_panel_release();", "return g_cuda_cleanup_status;", "cudaErrorNotReady")
	ordered(cudaFunctionBody(t, goSource, "releasePromptPanelLocked"), "C.fcuda_panel_release_checked(&live)", "c.promptPanelLastLive = int64(live)", "c.faultLatch.Observe(DeviceFaultExecution", "cudaPromptPanelFail(", "return int64(live)")
	if strings.Contains(goSource, "C.fcuda_panel_release()") || strings.Count(goSource, "c.releasePromptPanelLocked()") != 2 {
		t.Fatal("both Go panel cleanup callers must consume checked native status")
	}
}
