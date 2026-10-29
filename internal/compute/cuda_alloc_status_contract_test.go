package compute

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

var (
	cudaRawMallocRE     = regexp.MustCompile(`\bcudaMalloc(Managed)?\s*\(`)
	cudaStringLiteralRE = regexp.MustCompile(`"(?:[^"\\]|\\.)*"`)
)

// Raw allocations that deliberately leave a miss pending: the caller has no status return, so
// it skips the launch and lets the next status reader refuse the request typed.
var cudaAllocPendingSites = map[string]int{
	"attention.cuh":  1, // fcuda_attention_f32 scratch (void ABI)
	"spec_graph.cuh": 1, // ensure_argmax_idx (int token-id ABI)
}

// Files outside this contract until their owner lands; each needs a named follow-on.
var cudaAllocContractDeferred = map[string]string{
	// Rewritten by lane cuda-kv-sizing-footprint-20261029 (Q4 coefficient ring arena); its
	// g_q8_* scratch and arena allocations join the contract once that rewrite lands.
	"quant_matmul.cuh": "cuda-kv-sizing-footprint-20261029",
}

// TestCUDARawAllocationsConsumeCapacityMiss pins that every raw cudaMalloc/cudaMallocManaged
// in the CUDA fragments goes through fcuda_alloc_status, so one capacity miss cannot leave a
// stale "out of memory" that a later CK or GDN/KV preflight turns into a context poison
// (the raider h2h transformer_ops.cuh:169 signature).
// fak-test:runtime fast est=50ms lane=default
func TestCUDARawAllocationsConsumeCapacityMiss(t *testing.T) {
	files, err := filepath.Glob(filepath.Join("cuda", "*.cuh"))
	if err != nil || len(files) == 0 {
		t.Fatalf("glob CUDA fragments: %v (n=%d)", err, len(files))
	}
	sort.Strings(files)
	pending := map[string]int{}
	var unwrapped []string
	for _, path := range files {
		name := filepath.Base(path)
		if _, ok := cudaAllocContractDeferred[name]; ok {
			continue
		}
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(b), "\n") {
			code := line
			if j := strings.Index(code, "//"); j >= 0 {
				code = code[:j]
			}
			code = cudaStringLiteralRE.ReplaceAllString(code, `""`)
			for _, loc := range cudaRawMallocRE.FindAllStringIndex(code, -1) {
				if strings.HasSuffix(strings.TrimRight(code[:loc[0]], " "), "fcuda_alloc_status(") {
					continue
				}
				if cudaAllocPendingSites[name] > 0 {
					pending[name]++
					continue
				}
				unwrapped = append(unwrapped, name+":"+strconv.Itoa(i+1))
			}
		}
	}
	if len(unwrapped) > 0 {
		t.Fatalf("raw CUDA allocations must be wrapped in fcuda_alloc_status: %v", unwrapped)
	}
	for name, want := range cudaAllocPendingSites {
		if pending[name] != want {
			t.Fatalf("%s pending-status allocations = %d, want %d", name, pending[name], want)
		}
	}
}

// TestCUDAPendingAllocationSitesNeverLaunchOnNull pins that the two pending-status sites
// refuse before their launch instead of handing a null buffer to a kernel.
// fak-test:runtime fast est=20ms lane=default
func TestCUDAPendingAllocationSitesNeverLaunchOnNull(t *testing.T) {
	read := func(name string) string {
		t.Helper()
		b, err := os.ReadFile(filepath.Join("cuda", name))
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	attention := read("attention.cuh")
	start := strings.Index(attention, `extern "C" void fcuda_attention_f32(`)
	launch := strings.Index(attention, `k_attention<<<`)
	if start < 0 || launch < start {
		t.Fatal("fcuda_attention_f32 launch not found")
	}
	if body := attention[start:launch]; !strings.Contains(body, "!g_attn_scratch) {") ||
		!strings.Contains(body, "return;") || !strings.Contains(body, "g_attn_scratch_cap = 0;") {
		t.Fatal("fcuda_attention_f32 must reset its scratch and return before launching on a miss")
	}
	graph := read("spec_graph.cuh")
	start = strings.Index(graph, `extern "C" int fcuda_argmax_f32(`)
	launch = strings.Index(graph[max(start, 0):], `k_argmax<<<`)
	if start < 0 || launch < 0 || !strings.Contains(graph[start:start+launch], "if (!ensure_argmax_idx()) return -1;") {
		t.Fatal("fcuda_argmax_f32 must refuse before launching k_argmax when the index alloc misses")
	}
}
