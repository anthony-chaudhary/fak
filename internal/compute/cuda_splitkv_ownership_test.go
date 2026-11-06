package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This exercises the actual host ownership/status code with mocked CUDA APIs.
// It does not qualify CUDA syntax, device execution, kernel math, or async faults.
// fak-test:runtime medium est=2s lane=default
// fak-test:justify Compiles an extracted host caller twice, with checked-stream admission on and off.
func TestCUDASplitKVScratchOwnershipAndStatus(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable; split-KV ownership witness not run")
	}
	source := qwen38PromptAttentionSource(t, "cuda/attention.cuh")
	status := qwen38PromptAttentionSource(t, "cuda/alloc_status.h")
	var constants strings.Builder
	for _, name := range []string{"QWEN38_PROMPT_HEAD_DIM", "QWEN38_SPLITKV_PREFIX_THRESHOLD", "QWEN38_SPLITKV_MAX_PARTITIONS"} {
		found := false
		for _, line := range strings.Split(source, "\n") {
			if strings.HasPrefix(line, "#define "+name+" ") {
				constants.WriteString(line + "\n")
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing production constant %s", name)
		}
	}
	actual := constants.String() +
		qwen38SourceFunction(t, status, "static inline cudaError_t fcuda_alloc_status(") + "\n" +
		qwen38SourceFunction(t, source, "static int qwen38_splitkv_partitions(") + "\n" +
		qwen38SourceFunction(t, source, "static int qwen38_causal_attention_panel_launch_simt(") + "\n" +
		qwen38SourceFunction(t, source, "extern \"C\" int fak_qwen35_causal_attention_panel_f32(") + "\n"
	for _, checked := range []bool{false, true} {
		name, define := "plain", ""
		if checked {
			name, define = "checked", "#define FAK_CUDA_V41_CHECKED_STREAM 1\n"
		}
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			cpp, binary := filepath.Join(dir, "splitkv.cpp"), filepath.Join(dir, "splitkv.exe")
			if err := os.WriteFile(cpp, []byte(define+splitKVOwnershipHarness+actual+splitKVOwnershipMain), 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
			defer cancel()
			if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-Wno-unused-variable", "-Wno-unused-function", "-Wno-unused-parameter", cpp, "-o", binary).CombinedOutput(); err != nil {
				t.Fatalf("compile actual split-KV caller: %v\n%s", err, out)
			}
			cases := []string{"oom", "fatal-alloc", "null-alloc", "retire", "growth", "capture", "terminal"}
			if checked {
				cases = append(cases, "checked")
			}
			for _, scenario := range cases {
				t.Run(scenario, func(t *testing.T) {
					// A separate process resets the production function's static scratch.
					run, stop := context.WithTimeout(t.Context(), 5*time.Second)
					defer stop()
					if out, err := exec.CommandContext(run, binary, scenario).CombinedOutput(); err != nil {
						t.Fatalf("actual split-KV %s: %v\n%s", scenario, err, out)
					}
				})
			}
		})
	}
}

const splitKVOwnershipHarness = `
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
using std::isfinite;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorMemoryAllocation=2, cudaErrorNotReady=34;
constexpr int FLASH_THREADS=128, FAK_ALLOC_ATTENTION_SCRATCH=1, FAK_ALLOC_DEVICE_LOCAL=2;
#define FAK_ALLOC_SITE(x) x
static int g_cuda_cleanup_status=0, pending=0, alloc_status=0, free_status=0;
static int allocs=0, frees=0, tracks=0, untracks=0, reads=0, partial=0, merge=0, simt=0, mma=0, selections=0;
static bool g_capture_open=false, uncertain=false, want_mma=false, null_alloc=false;
static float storage[4];
static void *tracked=nullptr, *freed=nullptr, *allocated=nullptr;
static bool v41_candidates_stream_uncertain() { return uncertain; }
static int cudaGetLastError() { ++reads; int e=pending; pending=0; return e; }
static const char *cudaGetErrorString(int) { return "injected"; }
static int cudaMalloc(float **p, size_t) {
 ++allocs; assert(!tracked);
 if (alloc_status) { pending=alloc_status; *p=nullptr; return alloc_status; }
 *p=null_alloc ? nullptr : &storage[allocs % 4]; allocated=*p; return 0;
}
static int cudaFree(void *p) {
 ++frees; assert(p==tracked && tracked);
 if (!free_status) freed=p;
 return free_status;
}
static void fak_alloc_untrack(void *p) {
 assert(p==tracked && p==freed); ++untracks; tracked=nullptr; freed=nullptr;
}
static void fak_alloc_track(void *p, int, int, const char *, size_t) {
 assert(p && !tracked && p==allocated); ++tracks; tracked=p;
}
static bool qwen38_prefill_attention_wants_mma() { ++selections; return want_mma; }
static int fak_qwen38_fattn_mma_hd256_launch(const float *, const float *, const float *, float *, int, int, int, int, float) { ++mma; return -2; }
static void launch(const char *name) {
 if (!strcmp(name,"k_qwen38_causal_attention_panel_hd256")) ++simt;
 else if (!strcmp(name,"k_qwen38_causal_attention_panel_hd256_splitkv_partial")) { assert(tracked); ++partial; }
 else if (!strcmp(name,"k_qwen38_splitkv_lse_merge_hd256")) { assert(tracked); ++merge; }
 else assert(false);
}
// Only device launch syntax is replaced; host allocation, status and routing are real.
#define FCUDA_TS(name, grid, ...) launch(name)
`

const splitKVOwnershipMain = `
static int call(int tokens=1, int prefix=1024) {
 float x=1;
 return fak_qwen35_causal_attention_panel_f32(&x,&x,&x,&x,tokens,prefix,24,4,256,1.f);
}
static void no_effects() {
 assert(!allocs && !frees && !tracks && !untracks && !reads && !partial && !merge && !simt && !mma && !selections);
}
int main(int argc, char **argv) {
 assert(argc==2);
 const char *s=argv[1];
 if (!strcmp(s,"oom") || !strcmp(s,"fatal-alloc") || !strcmp(s,"null-alloc")) {
  null_alloc=!strcmp(s,"null-alloc");
  alloc_status=null_alloc ? 0 : (!strcmp(s,"oom") ? 2 : 700);
  assert(call()==50000+alloc_status);
  assert(allocs==1 && !tracked && !tracks && !partial && !merge && !simt && !frees);
  assert(pending==(alloc_status==700 ? 700 : 0));
  if (alloc_status==700) return 0; // Sticky execution faults must not be consumed as capacity misses.
  alloc_status=0; null_alloc=false;
  assert(call()==0 && allocs==2 && tracks==1 && partial==1 && merge==1);
 } else if (!strcmp(s,"retire")) {
  assert(call()==0); void *old=tracked;
  free_status=17;
  assert(call(2)==52017 && frees==1 && !untracks && allocs==1 && tracks==1 && tracked==old);
  assert(g_cuda_cleanup_status==17 && partial==1 && merge==1 && !simt);
  int prior_reads=reads, prior_selections=selections;
  free_status=0; want_mma=true;
  assert(call(2)==52017); // Never retry the uncertain retirement.
  assert(call(1,0)==52017); // Short SIMT path cannot bypass the terminal latch.
  g_capture_open=true;
  assert(call()==52017); // Already-fitting scratch cannot be reused either.
  assert(frees==1 && !untracks && allocs==1 && tracks==1 && tracked==old);
  assert(reads==prior_reads && selections==prior_selections && !mma && !simt && partial==1 && merge==1);
 } else if (!strcmp(s,"growth")) {
  assert(call()==0); void *old=tracked;
  assert(call(2)==0 && frees==1 && untracks==1 && allocs==2 && tracks==2 && tracked!=old);
  assert(partial==2 && merge==2 && !simt && !g_cuda_cleanup_status);
 } else if (!strcmp(s,"capture")) {
  g_capture_open=true;
  assert(call()==0 && simt==1 && !allocs && !frees && !tracks);
  g_capture_open=false; assert(call()==0);
  g_capture_open=true;
  assert(call()==0 && partial==2 && merge==2);
  assert(call(2)==0 && simt==2 && allocs==1 && !frees && tracks==1);
 } else if (!strcmp(s,"terminal")) {
  g_cuda_cleanup_status=23;
  assert(call()==52023); no_effects();
 } else if (!strcmp(s,"checked")) {
  uncertain=true;
  assert(call()==52034); no_effects();
 } else assert(false);
}
`
