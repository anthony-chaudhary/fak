package compute

import (
	"os"
	"strings"
	"testing"
)

// Binding for the actual C++ host control's dependency-order gate. This does not
// execute CUDA math; the retained C++ control uses the real profiler end/helper.
// fak-test:runtime fast est=1ms lane=default
func TestCUDAPagedV2ProducerStatusSourceContract(t *testing.T) {
	b, err := os.ReadFile("cuda/paged_attention_vllm.cuh")
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	start := strings.Index(s, "  auto kern2 = paged_attention_v2_kernel")
	if start < 0 {
		t.Fatal("v2 producer source missing")
	}
	end := strings.Index(s[start:], "  const int reduce_shared_mem_size")
	if end <= 0 {
		t.Fatal("reduction preparation boundary missing")
	}
	body := s[start : start+end]
	valid := func(src string) bool {
		// Comments do not affect the actual statement order.
		var code []string
		for _, line := range strings.Split(src, "\n") {
			if i := strings.Index(line, "//"); i >= 0 {
				line = line[:i]
			}
			code = append(code, line)
		}
		normal := strings.Join(strings.Fields(strings.Join(code, "\n")), " ")
		return strings.Contains(normal, `cudaError_t producer_status = cudaSuccess; FCUDA_TS("fak_paged_attention_v2", num_heads,`) &&
			strings.HasSuffix(normal, `kv_head_stride)); producer_status = cudaGetLastError()); if (producer_status != cudaSuccess) return 4;`)
	}
	if !valid(body) {
		t.Fatal("producer failure must return after profiler cleanup but before reduction preparation")
	}
	for _, mutant := range []string{
		strings.Replace(body, "if (producer_status != cudaSuccess) return 4;", "", 1),
		strings.Replace(body, "producer_status = cudaGetLastError());", "producer_status = cudaGetLastError(); if (producer_status != cudaSuccess) return 4;);", 1),
	} {
		if valid(mutant) {
			t.Fatal("source binding accepted absent gate or early return that bypasses profiler end")
		}
	}
}
