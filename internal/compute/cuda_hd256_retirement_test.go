package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// The actual staging launcher and actual fallback caller run with mocked CUDA
// APIs and kernel execution. This is ownership/control-flow proof, not GPU proof.
// fak-test:runtime medium est=2s lane=default
func TestCUDAHD256RetirementFailureIsSticky(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	// Preserve the whole native launcher and its real static ownership state.
	owner := read("cuda/fattn_mma_hd256.cuh")
	start := strings.Index(owner, "static half2 *g_qwen38_fattn_kv16 = nullptr;")
	if start < 0 {
		t.Fatal("actual hd256 owner missing")
	}
	caller := read("cuda/attention.cuh")
	startCaller := strings.Index(caller, "extern \"C\" int fak_qwen35_causal_attention_panel_f32(")
	if startCaller < 0 {
		t.Fatal("actual fallback caller missing")
	}
	endCaller := strings.Index(caller[startCaller:], "\n// Prompt-panel causal attention")
	if endCaller < 0 {
		t.Fatal("actual fallback caller boundary missing")
	}
	// The wrapper now calls the partition selector and a shared SIMT helper.
	// Keep both real bodies so fallback remains an actual-source witness.
	var splitConstants strings.Builder
	for _, name := range []string{"QWEN38_SPLITKV_PREFIX_THRESHOLD", "QWEN38_SPLITKV_MAX_PARTITIONS"} {
		found := false
		for _, line := range strings.Split(caller, "\n") {
			if strings.HasPrefix(line, "#define "+name+" ") {
				splitConstants.WriteString(line + "\n")
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("actual split-KV constant %s missing", name)
		}
	}
	partition := qwen38SourceFunction(t, caller, "static int qwen38_splitkv_partitions(")
	simt := qwen38SourceFunction(t, caller, "static int qwen38_causal_attention_panel_launch_simt(")
	program := hd256RetirementHarness + splitConstants.String() + owner[start:] + partition + "\n" + simt + "\n" + caller[startCaller:startCaller+endCaller] + hd256RetirementMain
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "retirement.cpp"), filepath.Join(dir, "retirement")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-Wno-unused-variable", "-Wno-unused-parameter", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile actual owner/caller: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("actual retirement/fallback control: %v\n%s", err, out)
	}
}

const hd256RetirementHarness = `
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
using std::isfinite;
using cudaError_t = int;
struct half2 { float x, y; };
struct dim3 { dim3(unsigned, unsigned=1, unsigned=1) {} };
constexpr int cudaSuccess=0, cudaErrorMemoryAllocation=2, cudaDevAttrComputeCapabilityMajor=1;
constexpr int QWEN38_PROMPT_HEAD_DIM=256, FLASH_THREADS=128;
constexpr int FAK_ALLOC_ATTENTION_SCRATCH=1, FAK_ALLOC_DEVICE_LOCAL=2;
#define FAK_ALLOC_SITE(x) x
namespace fak_fattn_mma {
constexpr int nbatch_fa=32, DKQ_QWEN38=256, ncols1_qwen38=8, ncols2_qwen38=8;
constexpr int nbatch_K2=128, nbatch_V2=128, nthreads=128, warp_size=32, nbatch_combine=128;
}
static half2 old_storage[32768], new_storage[32768];
static int g_cuda_cleanup_status=0, free_status=0, alloc_status=0;
static int frees=0, allocs=0, tracks=0, untracks=0, stages=0, mma=0, simt=0, queries=0;
static bool g_capture_open=false, old_tracked=true;
static int cudaGetDevice(int *p) { ++queries; *p=0; return 0; }
static int cudaDeviceGetAttribute(int *p, int, int) { *p=8; return 0; }
static int cudaGetLastError() { return 0; }
static int cudaFree(void *p) { assert(p==old_storage); assert(old_tracked); ++frees; return free_status; }
static int cudaMalloc(half2 **p, size_t) { ++allocs; if (!alloc_status) *p=new_storage; return alloc_status; }
// This witness uses prefix=0: split-KV allocation must never replace its SIMT fallback.
static int cudaMalloc(float **, size_t) { assert(false && "unexpected split-KV allocation"); return cudaErrorMemoryAllocation; }
static const char *cudaGetErrorString(int) { return "injected"; }
static int fcuda_alloc_status(int status) { return status; }
static void fak_alloc_untrack(void *p) { assert(p==old_storage && old_tracked); old_tracked=false; ++untracks; }
static void fak_alloc_track(void *p, int, int, const char *, size_t) { assert(p==new_storage); ++tracks; }
static bool qwen38_prefill_attention_wants_mma() { return true; }
static void launch(const char *name) {
 if (!strcmp(name,"k_qwen38_fattn_stage_kv_f16")) ++stages;
 else if (!strcmp(name,"k_qwen38_fattn_mma_hd256")) ++mma;
 else if (!strcmp(name,"k_qwen38_causal_attention_panel_hd256")) ++simt;
 else assert(false);
}
// Drop only kernel execution syntax; actual host status and routing remain.
#define FCUDA_TS(name, grid, ...) launch(name)
`

const hd256RetirementMain = `
static void reset() {
 g_cuda_cleanup_status=free_status=alloc_status=0;
 frees=allocs=tracks=untracks=stages=mma=simt=queries=0;
 g_capture_open=false; old_tracked=true;
 g_qwen38_fattn_kv16=old_storage; g_qwen38_fattn_kv16_bytes=1;
 g_qwen38_fattn_mma_launches=0;
}
static int ownerCall() {
 float x=1;
 return fak_qwen38_fattn_mma_hd256_launch(&x,&x,&x,&x,1,0,24,4,1.f);
}
static int call() {
 float x=1;
 assert(qwen38_splitkv_partitions(0,1)==1);
 return fak_qwen35_causal_attention_panel_f32(&x,&x,&x,&x,1,0,24,4,256,1.f);
}
int main() {
 reset(); free_status=17;
 assert(call()==52017);
 assert(frees==1 && !allocs && !untracks && !tracks && !stages && !mma && !simt);
 assert(g_qwen38_fattn_kv16==old_storage && g_qwen38_fattn_kv16_bytes==1 && old_tracked);
 assert(g_cuda_cleanup_status==17 && g_qwen38_fattn_mma_launches==0);
 // A transient API return becoming success cannot permit a second free or reuse.
 free_status=0; int prior_queries=queries;
 assert(call()==52017 && frees==1 && queries==prior_queries && !simt);
 // The wrapper also has a terminal guard now; exercise the owner directly so
 // its own sticky retirement protection cannot pass vacuously.
 assert(ownerCall()==52017 && frees==1 && queries==prior_queries && !simt);
 g_qwen38_fattn_kv16_bytes=1000000;
 assert(call()==52017 && !stages && !mma && !simt);
 assert(ownerCall()==52017 && frees==1 && queries==prior_queries && !stages && !mma && !simt);
 reset(); g_cuda_cleanup_status=23;
 assert(call()==52023 && !frees && !allocs && !queries && !simt);
 assert(ownerCall()==52023 && !frees && !allocs && !queries && !simt);
 reset();
 assert(call()==0 && frees==1 && untracks==1 && allocs==1 && tracks==1);
 assert(stages==1 && mma==1 && !simt && g_qwen38_fattn_mma_launches==1);
 assert(g_qwen38_fattn_kv16==new_storage && g_qwen38_fattn_kv16_bytes>1);
 reset(); alloc_status=2;
 assert(call()==0 && frees==1 && untracks==1 && allocs==1 && !tracks);
 assert(!stages && !mma && simt==1 && g_qwen38_fattn_mma_launches==0);
 assert(g_qwen38_fattn_kv16==nullptr && g_qwen38_fattn_kv16_bytes==0);
 reset(); g_capture_open=true;
 assert(call()==0 && !frees && !allocs && !untracks && simt==1);
 assert(g_qwen38_fattn_kv16==old_storage && old_tracked);
}
`
