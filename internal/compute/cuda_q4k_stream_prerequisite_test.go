package compute

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// This is a source-bound integration regression, not a CUDA/device witness.
func TestCUDAQ4KCheckedStreamPrerequisites(t *testing.T) {
	read := func(name string) string {
		t.Helper()
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	kernel := read("cuda_kernels.cu")
	for _, fragment := range []string{
		"#if defined(FAK_CUDA_FLASHMLA_DECODE) || defined(FAK_CUDA_V41_EMBED_GATHER) || defined(FAK_CUDA_V41_INDEX_SELECT)",
		"#define FAK_CUDA_V41_CHECKED_STREAM 1",
		"static std::atomic<bool> g_v41_checked_stream_uncertain{false};",
		"if (g_v41_checked_stream_uncertain.load()) return true;",
		"extern \"C\" int fak_v41_checked_stream_terminal(void)",
		"return v41_candidates_stream_uncertain() ? 1 : 0;",
	} {
		if !strings.Contains(kernel, fragment) {
			t.Fatalf("missing kernel safety seam: %s", fragment)
		}
	}
	last := -1
	for _, name := range []string{"cuda/model_candidates.cuh", "cuda/v41_checked_stream.cuh", "cuda/q4k_shared_input.cuh", "cuda/q4k_shared_input_qualification.cuh"} {
		i := strings.Index(kernel, "#include \""+name+"\"")
		if i <= last {
			t.Fatalf("missing or out-of-order include %s", name)
		}
		last = i
	}
	// Each checked path must refuse before any allocation/reuse or enqueue.
	guards := map[string][]string{
		"cuda/runtime_memory.cuh":   {"fcuda_malloc_tagged", "fcuda_malloc_managed_tagged", "fcuda_free", "fcuda_discard_unsubmitted", "fcuda_malloc_panel_tagged", "fcuda_graph_prewarm"},
		"cuda/runtime_services.cuh": {"fcuda_malloc_on_tagged", "fcuda_free_on"},
		"cuda/peer_handoff.cuh":     {"fcuda_handoff_alloc", "fcuda_handoff_free", "fcuda_handoff_pinned_alloc", "fcuda_handoff_pinned_free"},
		"cuda/attention.cuh":        {"fcuda_attention_f32"},
		"cuda/fattn_mma_hd256.cuh":  {"fak_qwen38_fattn_mma_hd256_launch"},
		"cuda/q3k_mmvq.cuh":         {"fcuda_q3k_matmul_f32"},
		"cuda/q4k_mmvq.cuh":         {"fcuda_q4k_mmvq_matmul"},
		"cuda/quant_matmul.cuh":     {"q8_quant_rows"},
	}
	for path, names := range guards {
		s := read(path)
		for _, name := range names {
			re := regexp.MustCompile(`(?m)^[^\n]*\b` + regexp.QuoteMeta(name) + `\([^;{}]*\)\s*\{`)
			loc := re.FindStringIndex(s)
			if loc == nil {
				t.Fatalf("missing function %s in %s", name, path)
			}
			head := strings.TrimSpace(s[loc[1]:])
			if strings.HasPrefix(head, "if (!d) return 0;") {
				head = strings.TrimSpace(strings.TrimPrefix(head, "if (!d) return 0;"))
			}
			if strings.HasPrefix(head, "if (!h) return 0;") {
				head = strings.TrimSpace(strings.TrimPrefix(head, "if (!h) return 0;"))
			}
			if !strings.HasPrefix(head, "#ifdef FAK_CUDA_V41_CHECKED_STREAM\n") {
				t.Fatalf("guard not first: %s", name)
			}
			guard := strings.SplitN(head, "#endif", 2)[0]
			if !strings.Contains(guard, "if (v41_candidates_stream_uncertain()) return") {
				t.Fatalf("missing terminal admission: %s", name)
			}
		}
	}
	state := read("cuda_backend_state.go")
	if !strings.Contains(state, "func (c *cudaBackend) cudaBufForSubmit(t Tensor) *cudaBuf {\n\tcudaV41CheckedSubmit()") {
		t.Fatal("Go submission must check shared terminal state first")
	}
	on := read("cuda_v41_checked_stream.go")
	if !strings.HasPrefix(on, "//go:build cuda && cgo\n") || !strings.Contains(on, "if C.fak_v41_checked_stream_terminal() != 0") || !strings.Contains(on, "panic(&BackendError{") {
		t.Fatal("every CUDA profile must link the checked Go terminal bridge")
	}
	if _, err := os.Stat("cuda_v41_checked_stream_off.go"); !os.IsNotExist(err) {
		t.Fatal("profile-dependent no-op bridge must not survive")
	}
	if !strings.Contains(read("cuda_backend.h"), "int fak_v41_checked_stream_terminal(void);") {
		t.Fatal("native archive must expose the terminal bridge ABI")
	}
	// Do not regress current public property/driver APIs while porting guards.
	memory := read("cuda/runtime_memory.cuh")
	for _, name := range []string{"fcuda_device_l2_cache_size", "fcuda_device_warp_size", "fcuda_driver_get_version"} {
		if !strings.Contains(memory, name+"(") {
			t.Fatalf("lost public device API %s", name)
		}
	}
}
