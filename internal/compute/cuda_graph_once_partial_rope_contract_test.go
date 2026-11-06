package compute

import (
	"context"
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

var graphOncePartialRoPEHost = flag.Bool("fak-graph-once-partial-rope-host", false, "run source-bound partial-RoPE native admission/recording control witness; not CUDA execution")

// Native-source assertions are not a dynamic-position or GPU correctness receipt.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAGraphOncePartialRoPESourceContract(t *testing.T) {
	transformer := readCUDASource(t, "cuda/transformer_ops.cuh")
	graph := readCUDASource(t, "cuda/graph_capture_once.cuh")
	kernel := cudaFunctionBody(t, transformer, "k_partial_rope_copy")
	for _, required := range []string{
		"template <bool device_position = false>\n__global__ void k_partial_rope_copy",
		"double theta, const int *position = nullptr)",
	} {
		if !strings.Contains(transformer, required) {
			t.Fatalf("partial-RoPE template missing %q", required)
		}
	}
	for _, required := range []string{
		"if (device_position) pos = position[0]", "size_t i", "size_t n", "size_t base",
		"if (d >= rotaryDim) { out[i] = in[i]; return; }",
		"int half = rotaryDim / 2", "pow(theta, -(2.0 * j) / rotaryDim)",
		"cos((double)pos * freq)", "sin((double)pos * freq)",
		"in[base + j]", "in[base + j + half]",
		"a * cs - b * sn : b * cs + a * sn",
	} {
		if !strings.Contains(kernel, required) {
			t.Fatalf("partial-RoPE arithmetic/source contract missing %q", required)
		}
	}
	eager := cudaFunctionBody(t, transformer, "fcuda_partial_rope_qk_f32")
	if strings.Count(eager, "k_partial_rope_copy<false><<<") != 2 || strings.Contains(eager, "s->position") {
		t.Fatal("ordinary eager/recapture path stopped using the host position")
	}
	body := cudaFunctionBody(t, graph, "fak_graph_once_partial_rope")
	previous := -1
	for _, fragment := range []string{
		"graph_once_capture_admit(s)", "v41_candidates_stream_uncertain()",
		"graph_once_partial_rope_inputs(s, q, k, qout, kout", "cudaPeekAtLastError()",
		"unsigned qblocks = qn / 256 + (qn % 256 != 0)",
		"k_partial_rope_copy<true><<<qblocks", "cudaGetLastError()",
		"k_partial_rope_copy<true><<<kblocks",
	} {
		i := strings.Index(body, fragment)
		if i <= previous {
			t.Fatalf("missing/reordered preflight or record stage %q", fragment)
		}
		previous = i
	}
	if strings.Count(body, ",theta,s->position)") != 2 || strings.Count(body, "cudaGetLastError()") != 2 {
		t.Fatal("both Q/K must use the same retained owner scalar and checked record status")
	}
	for _, forbidden := range []string{"cudaMalloc", "cudaFree", "cudaMemcpy", "cudaStreamSynchronize", "s->next++", "s->next =", "graph_once_set_position"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("record-only partial RoPE adds an effect %q", forbidden)
		}
	}
	closure := readCUDASource(t, "cuda_kernels.cu")
	if !strings.Contains(closure, "#ifdef FAK_CUDA_CAPTURE_ONCE_QUALIFICATION\n#include \"cuda/graph_capture_once.cuh\"\n#endif") ||
		strings.Index(closure, "cuda/transformer_ops.cuh") > strings.Index(closure, "cuda/graph_capture_once.cuh") {
		t.Fatal("template must precede the qualification-only capture wrapper")
	}
}

// The optional host harness extracts actual native preflight/wrapper bodies. Its
// FCUDA_TS seam records launches; it does not execute kernels or certify CUDA.
// fak-test:runtime medium est=1s lane=optin
func TestCUDAGraphOncePartialRoPEHostAdmission(t *testing.T) {
	if !*graphOncePartialRoPEHost {
		t.Skip("explicit host witness not selected; native/GPU evidence remains absent")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("requested native admission witness requires a host C++ compiler")
	}
	graph := readCUDASource(t, "cuda/graph_capture_once.cuh")
	definition := func(prefix, name string) string {
		t.Helper()
		start := strings.Index(graph, prefix+name+"(")
		if start < 0 {
			t.Fatalf("missing native definition %s", name)
		}
		open := strings.Index(graph[start:], "{")
		if open < 0 {
			t.Fatalf("missing native body %s", name)
		}
		return graph[start:start+open] + cudaFunctionBody(t, graph, name) + "\n"
	}
	program := graphOncePartialRoPEHarness +
		definition("static int ", "graph_once_capture_admit") +
		definition("static bool ", "graph_once_partial_rope_span") +
		definition("static bool ", "graph_once_partial_rope_inputs") +
		definition("extern \"C\" int ", "fak_graph_once_partial_rope") + graphOncePartialRoPEMain
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "partial-rope.cpp"), filepath.Join(dir, "partial-rope")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile native admission witness: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("native admission witness: %v\n%s", err, out)
	}
}

const graphOncePartialRoPEHarness = `
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>
#include <initializer_list>
#include <unordered_map>
#include <vector>
using std::isfinite;
struct fak_graph_once { int *position; bool terminal; };
static fak_graph_once state;
static fak_graph_once *g_graph_once_capture;
static bool g_capture_open, uncertain;
static int launches, failed_launch, pending_error, error_reads;
static constexpr int cudaSuccess = 0;
static std::unordered_map<void *, size_t> g_live;
static std::vector<unsigned> grids;
static bool v41_candidates_stream_uncertain() { return uncertain; }
static int cudaPeekAtLastError() { return pending_error; }
static int cudaGetLastError() { ++error_reads; return failed_launch == launches ? 7 : 0; }
#define FCUDA_TS(label, grid, ...) do { ++launches; grids.push_back((unsigned)(grid)); } while (false)
`

const graphOncePartialRoPEMain = `
static float *q, *k, *qo, *ko;
static void reset() {
 q = (float *)(uintptr_t)0x10000; k = (float *)(uintptr_t)0x20000;
 qo = (float *)(uintptr_t)0x30000; ko = (float *)(uintptr_t)0x40000;
 state = {(int *)(uintptr_t)0x90000, false};
 g_graph_once_capture = &state; g_capture_open = true; uncertain = false;
 launches = failed_launch = pending_error = error_reads = 0;
 grids.clear(); g_live.clear();
 for (float *p : {q,k,qo,ko}) g_live[p] = 1u << 20;
}
static int call(int qh=4, int kh=2, int hd=65, int rd=64, double theta=10000) {
 return fak_graph_once_partial_rope(&state,q,k,qo,ko,qh,kh,hd,rd,theta);
}
static void refused() { assert(call() == 1 && state.terminal && launches == 0 && error_reads == 0); }
int main() {
 reset(); assert(call() == 0 && !state.terminal && launches == 2 && error_reads == 2);
 assert(grids[0] == 2 && grids[1] == 1);
 reset(); assert(call(4,2,256,256) == 0 && grids[0] == 4 && grids[1] == 2);
 struct Shape { int qh,kh,hd,rd; };
 for (auto x : {Shape{0,2,65,64}, Shape{-1,2,65,64}, Shape{4,0,65,64},
                Shape{4,2,0,64}, Shape{3,2,65,64}, Shape{4,2,65,0},
                Shape{4,2,65,63}, Shape{4,2,65,66}, Shape{INT_MAX,1,2,2}}) {
  reset(); assert(call(x.qh,x.kh,x.hd,x.rd) == 1 && state.terminal && launches == 0);
 }
 for (double theta : {0.0,-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
  reset(); assert(call(4,2,65,64,theta) == 1 && state.terminal && launches == 0);
 }
 reset(); q=nullptr; refused();
 reset(); k=nullptr; refused();
 reset(); qo=nullptr; refused();
 reset(); ko=nullptr; refused();
 reset(); state.position=nullptr; refused();
 reset(); g_live.erase(k); refused();
 reset(); g_live[qo]=4*65*sizeof(float)-1; refused();
 reset(); qo=q; refused();
 reset(); qo=(float *)((uintptr_t)q+sizeof(float)); g_live[qo]=1u<<20; refused();
 reset(); ko=qo; refused();
 reset(); qo=(float *)state.position; g_live[qo]=1u<<20; refused();
 reset(); qo=(float *)(UINTPTR_MAX-3); g_live[qo]=1u<<20; refused();
 reset(); qo=(float *)(uintptr_t)0x30001; g_live[qo]=1u<<20; refused();
 reset(); k=q; assert(call()==0 && launches==2); // Read/read alias is legal.
 reset(); uncertain=true; refused();
 reset(); pending_error=7; assert(call()==3 && state.terminal && launches==0 && error_reads==0);
 reset(); g_capture_open=false; assert(call()==1 && !state.terminal && launches==0);
 reset(); fak_graph_once other=state;
 assert(fak_graph_once_partial_rope(&other,q,k,qo,ko,4,2,65,64,10000)==1 && !other.terminal && !state.terminal && launches==0);
 for (int stage : {1,2}) {
  reset(); failed_launch=stage;
  assert(call()==3 && state.terminal && launches==stage && error_reads==stage);
 }
 // Pure literal boundary check: no huge allocation and no CUDA launch.
 unsigned n=INT_MAX, blocks=n/256+(n%256!=0);
 assert(blocks==8388608 && (uint64_t)(blocks-1)*256+255==INT_MAX);
 return 0;
}
`
