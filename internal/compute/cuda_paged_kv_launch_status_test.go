package compute

import (
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestCUDAPagedWriteLaunchStatusSourceContract(t *testing.T) {
	b, err := os.ReadFile("cuda/paged_attention_vllm.cuh")
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	start := strings.Index(s, `extern "C" int fcuda_paged_kv_write_pages_f32(`)
	end := strings.Index(s, `extern "C" int fcuda_paged_kv_gather_f32(`)
	if start < 0 || end <= start {
		t.Fatal("multi-page adapter source missing")
	}
	body := s[start:end]
	capturesBeforeEnd := func(src string) bool {
		normal := strings.Join(strings.Fields(src), " ")
		return strings.Count(normal, "cudaGetLastError()") == 1 &&
			strings.Contains(normal, `cudaError_t launch_status = cudaSuccess; FCUDA_TS("fak_paged_kv_write_pages", tokens,`) &&
			strings.Contains(normal, `head_size, block_size)); launch_status = cudaGetLastError()); return launch_status == cudaSuccess ? 0 : 4;`)
	}
	if !capturesBeforeEnd(body) {
		t.Fatal("launch status must be captured inside FCUDA_TS, before profiler end can change last-error")
	}
	for _, mutation := range []string{
		strings.Replace(body, "launch_status = cudaGetLastError());", ");\n  launch_status = cudaGetLastError();", 1),
		strings.Replace(body, "return launch_status == cudaSuccess", "return cudaGetLastError() == cudaSuccess", 1),
	} {
		if capturesBeforeEnd(mutation) {
			t.Fatal("source contract accepted a late/overwritten launch-status mutation")
		}
	}
}
