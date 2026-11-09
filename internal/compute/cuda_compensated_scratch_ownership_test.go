package compute

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Compile exact production allocator/owner/caller blocks against fake CUDA
// allocation and stream callbacks. This is not a CUDA build or device witness.
// fak-test:runtime medium est=1s lane=default
func TestCUDACompensatedScratchFailureOwnership(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	read := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	between := func(source, first, last string) string {
		t.Helper()
		begin := strings.Index(source, first)
		if begin < 0 {
			t.Fatalf("production boundary missing: %s", first)
		}
		end := strings.Index(source[begin:], last)
		if end < 0 {
			t.Fatalf("production boundary missing: %s", last)
		}
		return source[begin : begin+end]
	}
	function := func(source, signature string) string {
		t.Helper()
		begin := strings.Index(source, signature)
		if begin < 0 {
			t.Fatalf("production function missing: %s", signature)
		}
		open := strings.Index(source[begin:], "{")
		if open < 0 {
			t.Fatal("production function has no body")
		}
		depth := 0
		for i := begin + open; i < len(source); i++ {
			switch source[i] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return source[begin : i+1]
				}
			}
		}
		t.Fatal("production function body is incomplete")
		return ""
	}
	gdn, quant := read("cuda/qwen35_gdn.cuh"), read("cuda/quant_matmul.cuh")
	iq4xs := read("cuda/iq4xs_mmvq.cuh")
	q3k := read("cuda/q3k_mmvq.cuh")
	memory, header := read("cuda/runtime_memory.cuh"), read("cuda_backend.h")
	loan := strings.Contains(gdn, "struct SequenceSplitOwner")
	// Compile the actual ledger, including its pointer table and atomic counters.
	// Only the header-only pragma is omitted in this generated translation unit.
	program := strings.TrimPrefix(read("cuda/alloc_ledger.h"), "#pragma once\n") + "\n" + cudaCompensatedScratchHarness
	program += "\n" + function(read("cuda_kernels.cu"), "static bool v41_candidates_stream_uncertain(void)") + "\n"
	// Keep the allocator's real pool accounting and pending-membership dependency.
	program += "\n" + between(memory, "static size_t g_pool_bytes", "static size_t fcuda_pool_default_cap")
	program += "\n" + read("cuda/v41_compressor_pending.cuh")
	for _, signature := range []string{
		"static bool fcuda_pool_may_release(void)",
		"static void fcuda_pool_evict_to_cap(void)",
		"static void fcuda_pool_put(void *d, size_t bytes)",
	} {
		program += "\n" + function(memory, signature) + "\n"
	}
	// Current allocation entrypoints consume capacity misses through this exact
	// production helper. Keep older pre-helper baselines executable as well.
	if strings.Contains(memory, "fcuda_alloc_status(") {
		program += "\n" + function(read("cuda/alloc_status.h"), "static inline cudaError_t fcuda_alloc_status(") + "\n"
	}
	if loan {
		program += "\n#define TEST_SEQUENCE_LOAN 1\nextern \"C\" {\n" +
			between(header, "enum {\n  FCUDA_SEQUENCE_SPLIT_CONTRACT", "/* End sequence-owned compensated split scratch ABI. */") + "\n}\n"
	}
	for _, signature := range []string{
		"extern \"C\" void *fcuda_malloc_tagged(size_t bytes, int cls, int site)",
		"extern \"C\" void *fcuda_malloc(size_t bytes)",
		"extern \"C\" void fcuda_free(void *d)",
		"extern \"C\" int fcuda_discard_unsubmitted(void *d, size_t expected_bytes)",
		"extern \"C\" void fcuda_graph_prewarm(int extra)",
	} {
		program += "\n" + function(memory, signature) + "\n"
	}
	program += between(gdn, "#define FCUDA_SCRATCH_ALLOC_FAILED ", "// Same three ordered GEMMs")
	// Execute the actual helper, including status capture and timing-event exit.
	// The old void helper remains extractable so the new failure witness is RED
	// on swallowed statuses rather than merely a missing-signature assertion.
	gemmSignature := "static int fp16_compensated_gemm("
	if !strings.Contains(gdn, gemmSignature) {
		gemmSignature = "static void fp16_compensated_gemm("
	}
	program += "\n" + function(gdn, gemmSignature) + "\n"
	for i, tc := range []struct{ source, signature, branch, sizeDecl string }{
		{gdn, "static int fp16_compensated_matmul(", "", "size_t wn ="},
		{gdn, "static int q8_dequant_sgemm(", "", "size_t wn ="},
		{quant, "extern \"C\" int fcuda_q4k_matmul_f32(", "if (P >= 128)", "size_t wn ="},
		{quant, "extern \"C\" int fcuda_q2k_matmul_f32(", "if (P >= 128)", "size_t wn ="},
		{iq4xs, "extern \"C\" int fcuda_iq4xs_matmul_f32(", "if (P >= FCUDA_IQ4XS_DEQUANT_MIN_P)", "const size_t wn ="},
		{q3k, "extern \"C\" int fcuda_q3k_matmul_f32(", "if (valid && P >= 128)", "const size_t wn ="},
	} {
		body := function(tc.source, tc.signature)
		if tc.branch != "" {
			body = function(body, tc.branch)
		}
		block := between(body, tc.sizeDecl, "FCUDA_TS(")
		// Only the two kernel launches are represented by fake pending reads.
		// The production GEMM helper executes against fake cuBLAS below. Never
		// silently omit ownership effects in the replaced kernel-launch span.
		launches := between(body, "FCUDA_TS(", "fp16_compensated_gemm(")
		for _, forbidden := range []string{"malloc", "Malloc", "free", "Free", "discard", "release", "g_pool", "g_live"} {
			if strings.Contains(launches, forbidden) {
				t.Fatalf("unexpected ownership effect in production launch span %s: %s", tc.signature, forbidden)
			}
		}
		gemm := strings.Index(body, "fp16_compensated_gemm(")
		semicolon := strings.Index(body[gemm:], ";")
		if semicolon < 0 {
			t.Fatalf("production GEMM statement is incomplete: %s", tc.signature)
		}
		// Execute the exact GEMM statement (including its result binding) and
		// post-GEMM suffix through the containing branch close. Cleanup is the
		// production release, not a modeled call.
		statementStart := strings.LastIndex(body[:gemm], "\n") + 1
		statement := body[statementStart : gemm+semicolon+1]
		release := body[gemm+semicolon+1 : len(body)-1]
		program += fmt.Sprintf("\nstatic int route%d(int out=320, int in=256, int tokens=128) {\n", i) +
			"const int rows=out, cols=in, P=tokens;\nfloat output=0; float *Y=&output, *dY=&output;\n" + block +
			"\n++launches; ++pending_reads;\n" + statement + release + "\n}\n"
	}
	if loan {
		program += "\n" + function(gdn, "static cudaError_t qwen35_gdn_fence(")
		program += "\n" + function(gdn, "static int qwen35_gdn_drain_after_error(")
		program += "\n" + function(read("cuda/attention.cuh"), "extern \"C\" int fak_qwen35_sequence_sync(")
		// Copy the actual whole-GDN final fence too, not a modeled notifier.
		lastFence := between(function(gdn, "extern \"C\" int fcuda_qwen35_gdn_sequence_f32("),
			"cudaError_t sync = cudaStreamSynchronize(g_stream);", "g_qwen35_gdn_operations")
		program += "\nstatic int gdn_final_fence() {\n" + lastFence + "\nreturn 0;\n}\n"
	}
	program += cudaCompensatedScratchMain
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "scratch.cpp"), filepath.Join(dir, "scratch")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-Wno-unused-variable", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile production scratch ownership blocks: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("production scratch ownership: %v\n%s", err, out)
	}
	// These tagged Go adapter checks are source contracts only, not execution.
	mapping := read("cuda_prompt_panel.go")
	for _, want := range []string{
		"cudaCompensatedGEMMFailedBase = 93000",
		"rc > cudaCompensatedGEMMFailedBase && rc < cudaCompensatedGEMMFailedBase+1000",
		`Site: "gemm-cublas", Msg: "compensated cuBLAS GEMM submission failed", Code: rc, Class: MemoryActivation`,
		"cudaScratchRetireFailedBase = 83000",
		"rc > cudaScratchRetireFailedBase && rc < cudaScratchRetireFailedBase+1000",
		`Site: "gemm-scratch-retirement", Msg: "scratch retirement failed after allocation miss", Code: rc - cudaScratchRetireFailedBase, Class: MemoryScratchpad`,
	} {
		if !strings.Contains(mapping, want) {
			t.Fatalf("production scratch-retirement decoder missing %q", want)
		}
	}
	for _, want := range []string{
		"status > cudaScratchRetireFailedBase && status < cudaScratchRetireFailedBase+1000",
		`kernelErr.Stage = "scratch-retirement"`,
		"kernelErr.Code = status - cudaScratchRetireFailedBase",
	} {
		if !strings.Contains(read("cuda_specialized.go"), want) {
			t.Fatalf("production GDN scratch-retirement decoder missing %q", want)
		}
	}
	if loan {
		sequence := function(read("cuda_qwen35_sequence.go"), "func (c *cudaBackend) Qwen35SequencePrefill(")
		allRows := function(sequence, "\tif req.NeedAllLogits {")
		end := strings.Index(allRows, "if splitErr := finishSplit(); splitErr != nil {")
		head := strings.Index(allRows, "logitsRows, logitsRowsBuf, err = c.qwen35SequenceMatMulLocked(req.Output, finalPanel, tokens, \"output-all-head\")")
		if end < 0 || head <= end {
			t.Fatal("all-row head must follow checked layer-loan retirement")
		}
		failure := function(allRows, "if splitErr := finishSplit(); splitErr != nil")
		for _, want := range []string{
			"for i := transientStart; i < len(c.transient); i++ {",
			"atomic.StoreUint32(&c.transient[i].invalid, 1)",
			"c.qwen35SequencePoisonStates(req)",
			"result = Qwen35SequencePrefillResult{}",
			"return result, splitErr",
		} {
			if !strings.Contains(failure, want) {
				t.Fatalf("early layer-loan retirement failure handler missing %q", want)
			}
		}
		// The raw-hidden panel is the final residual owner. Only the
		// unrequested branch may release it before the final output fence.
		release := between(sequence, "\tif rawHiddenBuf == nil {", "\n\tvar logits Tensor")
		unrequested := function(release, "if rawHiddenBuf == nil {")
		requested := between(release, "} else {", "\n\t}")
		if strings.Count(release, "c.releaseTransientBuffers(") != 2 ||
			!strings.Contains(unrequested, "c.releaseTransientBuffers([]*cudaBuf{xBuf, finalPanelBuf})") ||
			!strings.Contains(requested, "c.releaseTransientBuffers([]*cudaBuf{finalPanelBuf})") ||
			strings.Contains(requested, "xBuf") || strings.Contains(requested, "rawHiddenBuf") {
			t.Fatal("early release must retain requested raw-hidden rows and preserve unrequested owner order")
		}
		for _, want := range []string{
			"if splitToken != 0 {\n\t\tfinalFenceErr = finishSplit()\n\t} else {\n\t\tfinalFenceErr = c.qwen35SequenceSyncLocked()\n\t}",
			"atomic.StoreUint32(&logitsRowsBuf.invalid, 1)",
			"LogitsRows: logitsRows",
			"c.qwen35SequenceReleaseSinceLocked(transientStart, lastBuf, logitsBuf, logitsRowsBuf, rawHiddenBuf)",
		} {
			if !strings.Contains(sequence, want) {
				t.Fatalf("all-row output fence/publication contract missing %q", want)
			}
		}
		for _, want := range []string{
			"if status.phase != C.FCUDA_SEQUENCE_SPLIT_CONTRACT &&",
			"(status.phase != C.FCUDA_SEQUENCE_SPLIT_ALLOCATE || status.allocation_was_oom == 0)",
			`c.faultLatch.Observe(DeviceFaultExecution, "qwen35-sequence-split-begin", int(status.code))`,
			`c.faultLatch.Observe(DeviceFaultExecution, "qwen35-sequence-split-end", int(status.code))`,
			"Code: int(status.allocation_code), Class: MemoryScratchpad",
			"cause = errors.Join(allocation, allocationStatus, cause)",
			"case C.FCUDA_SEQUENCE_SPLIT_CAPTURE_QUERY:",
			"splitToken = 0 // one retirement attempt, including panic/Goexit unwind",
			"err = errors.Join(err, cleanupErr)",
			"finalFenceErr = finishSplit()",
		} {
			if !strings.Contains(read("cuda_qwen35_sequence.go"), want) {
				t.Fatalf("production sequence loan exit contract missing %q", want)
			}
		}
	}
}

const cudaCompensatedScratchHarness = `
#include <cstddef>
#include <cstdio>
#include <unordered_map>
#include <utility>
#include <vector>
using __half = unsigned short;
using cublasStatus_t = int;
static constexpr int CUBLAS_STATUS_SUCCESS=0, CUBLAS_OP_T=1, CUBLAS_OP_N=0;
static constexpr int CUDA_R_16F=2, CUDA_R_32F=3, CUBLAS_COMPUTE_32F=4, CUBLAS_GEMM_DEFAULT_TENSOR_OP=5;
static int g_blas=0, cublas_calls=0, cublas_fail_at=0, cublas_error=13;
static constexpr int QK_K=256;
static unsigned long long g_iq4xs_dequant_gemm_launches=0;
static int timing_begins=0, timing_ends=0;
static std::vector<const void *> cublas_weights, cublas_inputs;
static std::vector<float> cublas_betas;
#define FCUDA_TS(name, grid, ...) do { ++timing_begins; __VA_ARGS__; ++timing_ends; } while (0)
static cublasStatus_t cublasGemmEx(int, int, int, int, int, int, const float *,
    const void *weight, int, int, const void *input, int, int, const float *beta,
    void *, int, int, int, int) {
  ++cublas_calls;
  cublas_weights.push_back(weight); cublas_inputs.push_back(input); cublas_betas.push_back(*beta);
  return cublas_calls==cublas_fail_at ? cublas_error : CUBLAS_STATUS_SUCCESS;
}
using cudaError_t = int;
using cudaStreamCaptureStatus = int;
static constexpr int cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorMemoryAllocation=2, cudaErrorNotReady=34;
static constexpr int cudaStreamCaptureStatusNone=0, cudaStreamCaptureStatusActive=1, cudaStreamCaptureStatusInvalidated=2;
static bool g_capture_open=false;
static bool g_v41_router_uncertain=false;
static bool g_v41_mmid_uncertain=false;
static int g_cuda_cleanup_status=0;
static cudaStreamCaptureStatus runtime_capture=cudaStreamCaptureStatusNone;
static int capture_queries=0, capture_query_error=0;
static int g_stream=0, attempts=0, alloc_fail_at=0, free_attempts=0, free_fail_at=0;
static int alloc_error=cudaErrorMemoryAllocation;
static int last_error=0, get_last_error_calls=0;
static int sync_calls=0, sync_fail_at=0, retired=0, launches=0, pending_reads=0;
static unsigned char storage[256];
static size_t next_slot=0;
static std::unordered_map<void *, size_t> physical, g_live, g_managed_live, g_panel_live;
static std::vector<std::pair<void *, size_t>> allocated;
static std::unordered_map<size_t, std::vector<void *>> g_pool, g_panel_pool;
static size_t g_panel_live_bytes=0, g_panel_cached_bytes=0;
static const char *cudaGetErrorString(int) { return "injected CUDA fault"; }
static int cudaGetLastError() {
  ++get_last_error_calls;
  const int result=last_error;
  last_error=0;
  return result;
}
static int cudaMalloc(void **ptr, size_t bytes) {
  ++attempts;
  if (attempts==alloc_fail_at) {
    *ptr=nullptr;
    if (alloc_error!=cudaSuccess) last_error=alloc_error;
    return alloc_error;
  }
  if (next_slot>=sizeof(storage)) return last_error=cudaErrorMemoryAllocation;
  *ptr=&storage[next_slot++];
  physical[*ptr]=bytes;
  allocated.push_back({*ptr,bytes});
  return 0;
}
static int cudaFree(void *ptr) {
  ++free_attempts;
  if (free_attempts==free_fail_at) return 700;
  if (pending_reads || physical.erase(ptr)!=1) return cudaErrorInvalidValue;
  ++retired;
  return 0;
}
static int cudaStreamSynchronize(int) {
  ++sync_calls;
  if (sync_calls==sync_fail_at) return 700;
  pending_reads=0;
  return 0;
}
[[maybe_unused]] static int cudaStreamIsCapturing(int, cudaStreamCaptureStatus *capture) {
  ++capture_queries;
  if (capture_query_error) return capture_query_error;
  *capture=runtime_capture;
  return 0;
}
static void q4_coeff_cache_evict(const void *) {}
#define CK(value) do { (void)(value); } while (0)
static size_t pooled() {
  size_t count=0;
  for (const auto &bucket : g_pool) count+=bucket.second.size();
  return count;
}
`

const cudaCompensatedScratchMain = `
#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "ownership line=%d allocs=%d frees=%d retired=%d pooled=%zu live=%zu sync=%d launches=%d last_error=%d last_reads=%d\n", \
    __LINE__, attempts, free_attempts, retired, pooled(), g_live.size(), sync_calls, launches, last_error, get_last_error_calls); return 1; \
} } while (0)
static void reset_context() {
  // Reset only the fake device context; drain the real ledger through its API.
  while (!g_alloc_tags.empty()) fak_alloc_untrack(g_alloc_tags.begin()->first);
  g_alloc_untracked_frees.store(0);
  // Fake process teardown between cases; production has no quarantine reset API.
  attempts=alloc_fail_at=free_attempts=free_fail_at=sync_calls=sync_fail_at=retired=launches=pending_reads=0;
  next_slot=0; g_capture_open=false;
  g_v41_router_uncertain=false; g_v41_mmid_uncertain=false; g_cuda_cleanup_status=0;
  runtime_capture=cudaStreamCaptureStatusNone; capture_queries=capture_query_error=0;
  alloc_error=cudaErrorMemoryAllocation;
  last_error=get_last_error_calls=0;
  cublas_calls=cublas_fail_at=timing_begins=timing_ends=0; cublas_error=13;
  g_iq4xs_dequant_gemm_launches=0;
  cublas_weights.clear(); cublas_inputs.clear(); cublas_betas.clear();
  physical.clear(); g_live.clear(); g_managed_live.clear(); g_panel_live.clear();
  allocated.clear();
  g_pool.clear(); g_panel_pool.clear();
  g_pool_bytes=g_pool_peak_bytes=g_pool_clock=g_pool_hits=g_pool_misses=g_pool_evictions=0;
  g_pool_cap_bytes=(size_t)512 << 20; g_pool_graph_deep=false; g_pool_touch.clear();
  g_panel_live_bytes=g_panel_cached_bytes=0;
  g_scratch_alloc_test_fault=0;
#ifdef TEST_SEQUENCE_LOAN
  g_sequence_split={}; g_sequence_split_generation=0;
#endif
}
#ifdef TEST_SEQUENCE_LOAN
static bool private_ledger_matches() {
  int64_t snapshot[FAK_ALLOC_CELLS*5]={}, unknown=0;
  if (fcuda_alloc_ledger_read(snapshot,FAK_ALLOC_CELLS,&unknown)!=FAK_ALLOC_CELLS || unknown) return false;
  int64_t bytes=0, count=0;
  for (int i=0; i<4; ++i) if (g_sequence_split.ptr[i]) {
    bytes+=(int64_t)g_sequence_split.bytes[i]; ++count;
    auto tag=g_alloc_tags.find(g_sequence_split.ptr[i]);
    if (tag==g_alloc_tags.end() || tag->second.cls!=FAK_ALLOC_PREFILL_SCRATCH ||
        tag->second.kind!=FAK_ALLOC_DEVICE_LOCAL || tag->second.bytes!=g_sequence_split.bytes[i]) return false;
  }
  const int cell=(FAK_ALLOC_PREFILL_SCRATCH*FAK_ALLOC_KIND_COUNT+FAK_ALLOC_DEVICE_LOCAL)*5;
  if (snapshot[cell]!=bytes || snapshot[cell+2]!=count) {
    std::fprintf(stderr,"private ledger mismatch: live=%lld count=%lld expected_bytes=%lld expected_count=%lld\n",
        (long long)snapshot[cell],(long long)snapshot[cell+2],(long long)bytes,(long long)count);
    return false;
  }
  return true;
}
#endif
int main() {
  using Route=int (*)(int,int,int);
  const Route routes[]={route0, route1, route2, route3, route4, route5};
#ifdef TEST_SEQUENCE_LOAN
  // GDN owns ordinary transient buffers even when no split loan is active.
  // Exercise the real fence/drain/final-fence and allocator-free blocks: an
  // unsuccessful completion must retain these owners, not just private loans.
  for (int site=0; site<3; ++site) {
    reset_context();
    void *owner=fcuda_malloc(1024);
    CHECK(owner && g_live.size()==1);
    ++pending_reads;
    sync_fail_at=1;
    if (site==0) CHECK(qwen35_gdn_fence()==700);
    if (site==1) CHECK(qwen35_gdn_drain_after_error(20001)==70700);
    if (site==2) CHECK(gdn_final_fence()==70700);
    fcuda_free(owner);
    CHECK(g_live.count(owner)==1 && pooled()==0 && free_attempts==0);
    CHECK(g_cuda_cleanup_status==700 && v41_candidates_stream_uncertain());
    CHECK(sync_calls==1 && pending_reads==1 && physical.count(owner)==1);
    // A later successful fence cannot rehabilitate the retained storage.
    sync_fail_at=0;
    CHECK(qwen35_gdn_fence()==0 && gdn_final_fence()==0);
    fcuda_free(owner);
    CHECK(g_cuda_cleanup_status==700 && g_live.count(owner)==1 && pooled()==0);
  }
  for (int site=0; site<2; ++site) {
    reset_context();
    void *owner=fcuda_malloc(1024);
    CHECK(owner);
    g_cuda_cleanup_status=701; // an earlier family already quarantined the stream
    sync_fail_at=1;
    if (site==0) CHECK(qwen35_gdn_fence()==700);
    if (site==1) CHECK(gdn_final_fence()==70700);
    fcuda_free(owner);
    CHECK(g_cuda_cleanup_status==701 && g_live.count(owner)==1 && pooled()==0);
  }
  // Capture remains a no-sync success; a genuine successful drain keeps the
  // original launch error and still allows ordinary pool retirement.
  reset_context();
  void *owner=fcuda_malloc(1024);
  CHECK(owner);
  ++pending_reads; g_capture_open=true;
  CHECK(qwen35_gdn_fence()==0 && sync_calls==0 && pending_reads==1);
  CHECK(g_cuda_cleanup_status==0 && !v41_candidates_stream_uncertain());
  g_capture_open=false;
  CHECK(qwen35_gdn_drain_after_error(20001)==20001 && sync_calls==1 && pending_reads==0);
  fcuda_free(owner);
  CHECK(g_cuda_cleanup_status==0 && g_live.empty() && pooled()==1);
#endif
  // The shared allocator consumes a real capacity miss exactly once, while
  // leaving a non-OOM execution fault pending for the next status reader.
  reset_context(); alloc_fail_at=1;
  CHECK(fcuda_malloc(1024)==nullptr);
  CHECK(last_error==0 && get_last_error_calls==1 && g_live.empty());
  reset_context(); alloc_fail_at=1; alloc_error=700;
  CHECK(fcuda_malloc(1024)==nullptr);
  CHECK(last_error==700 && get_last_error_calls==0 && g_live.empty());
  // Synchronous cuBLAS errors cannot become scratch-release success. Test the
  // exact helper and all six actual caller suffixes; these are synthetic
  // statuses, not a CUDA numerical, device, or asynchronous-error witness.
  for (Route route : routes) {
    for (int fail=1; fail<=3; ++fail) {
      reset_context(); cublas_fail_at=fail;
      CHECK(route(320,256,128)==93000+cublas_error);
      CHECK(cublas_calls==fail && timing_begins==fail && timing_ends==fail);
      CHECK(g_iq4xs_dequant_gemm_launches==(route==route4 ? 1u : 0u));
      CHECK(attempts==4 && pooled()==4 && g_live.empty() && retired==0);
    }
    reset_context();
    CHECK(route(320,256,128)==0);
    CHECK(cublas_calls==3 && timing_begins==3 && timing_ends==3);
    CHECK(g_iq4xs_dequant_gemm_launches==(route==route4 ? 1u : 0u));
    CHECK(cublas_weights[0]==allocated[0].first && cublas_weights[1]==allocated[1].first && cublas_weights[2]==allocated[0].first);
    CHECK(cublas_inputs[0]==allocated[2].first && cublas_inputs[1]==allocated[2].first && cublas_inputs[2]==allocated[3].first);
    CHECK(cublas_betas[0]==0.0f && cublas_betas[1]==1.0f && cublas_betas[2]==1.0f);
  }
  // IQ4_XS enters this same production branch at 32 rows, not 128. Its
  // existing counter is an attempted-route witness even when cuBLAS refuses.
  for (int fail=1; fail<=3; ++fail) {
    reset_context(); cublas_fail_at=fail;
    CHECK(route4(320,256,32)==93000+cublas_error);
    CHECK(cublas_calls==fail && g_iq4xs_dequant_gemm_launches==1);
    CHECK(pooled()==4 && g_live.empty() && retired==0);
  }
  // Preserve the original no-scope allocation-failure and success contracts.
  for (Route route : routes) {
    for (int miss=1; miss<=4; ++miss) {
      reset_context(); alloc_fail_at=miss;
      CHECK(route(320,256,128)==FCUDA_SPLIT_SCRATCH_ALLOC_FAILED);
      CHECK(attempts==miss && retired==miss-1 && g_live.empty() && pooled()==0 && launches==0);
    }
    reset_context();
    CHECK(route(320,256,128)==0);
    CHECK(attempts==4 && retired==0 && pooled()==4 && g_live.empty() && launches==1);
    reset_context(); alloc_fail_at=4; free_fail_at=1;
    CHECK(route(320,256,128)==FCUDA_SCRATCH_RETIRE_FAILED_BASE+700);
    CHECK(free_attempts==3 && retired==2 && g_live.size()==1 && pooled()==0 && launches==0);
  }
  reset_context();
  void *prior_foreign=fcuda_malloc(1024); fcuda_free(prior_foreign);
  CHECK(route0(320,256,128)==0 && pending_reads==1 && attempts==5);
  // The next operation borrows both W halves, then misses its larger X.
  // Exact unpublished retirement must drain their prior submitted reads.
  alloc_fail_at=6;
  CHECK(route0(320,256,256)==FCUDA_SPLIT_SCRATCH_ALLOC_FAILED);
  CHECK(attempts==6 && launches==1 && sync_calls==2 && pending_reads==0);
  CHECK(free_attempts==2 && retired==2 && g_live.empty() && pooled()==3 && physical.size()==3);
  CHECK(physical.count(allocated[1].first)==0 && physical.count(allocated[2].first)==0);
  CHECK(physical.at(prior_foreign)==1024);
  CHECK(physical.at(allocated[3].first)==65536 && physical.at(allocated[4].first)==65536);
#ifndef TEST_SEQUENCE_LOAN
  reset_context();
  CHECK(route0(320,256,128)==0 && route1(640,256,128)==0);
  std::fprintf(stderr, "sequence loan missing: allocs=%d pooled=%zu live=%zu\n", attempts, pooled(), g_live.size());
  return 1;
#else
  const size_t wbytes=640*256*sizeof(__half), xbytes=128*512*sizeof(__half);
  unsigned long long token=0;
  fcuda_sequence_split_status status{};
  // A shared terminal entry cannot begin a fresh loan or retire an active
  // one, even when a later stream fence would succeed. Exercise both the
  // precise cleanup status and the peer-family uncertainty fallback.
  for (int stop=0; stop<2; ++stop) {
    const int stopped=stop==0 ? 719 : cudaErrorNotReady;
    reset_context();
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
    CHECK(route0(320,256,128)==0 && pending_reads==1);
    const auto owners=allocated;
    if (stop==0) g_cuda_cleanup_status=719;
    else g_v41_router_uncertain=true;
    CHECK(fcuda_sequence_split_end(token,&status)==stopped);
    CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_QUARANTINE && status.code==stopped);
    CHECK(g_sequence_split.phase==SequenceSplitPhase::Quarantined);
    CHECK(attempts==4 && capture_queries==1 && sync_calls==0 && free_attempts==0);
    CHECK(g_live.size()==4 && physical.size()==4 && pooled()==0 && pending_reads==1);
    for (int i=0; i<4; ++i) {
      CHECK(g_sequence_split.ptr[i]==owners[i].first && g_sequence_split.bytes[i]==owners[i].second);
    }
    CHECK(private_ledger_matches());
    CHECK(g_cuda_cleanup_status==(stop==0 ? 719 : 0));
    // A later flag change cannot rehabilitate the quarantined loan.
    g_cuda_cleanup_status=0; g_v41_router_uncertain=false;
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==stopped);
    CHECK(token==0 && status.phase==FCUDA_SEQUENCE_SPLIT_QUARANTINE && status.code==stopped);
    CHECK(attempts==4 && capture_queries==1 && sync_calls==0 && free_attempts==0);
    CHECK(private_ledger_matches());

    reset_context();
    if (stop==0) g_cuda_cleanup_status=719;
    else g_v41_router_uncertain=true;
    token=123;
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==stopped);
    CHECK(token==0 && status.phase==FCUDA_SEQUENCE_SPLIT_QUARANTINE && status.code==stopped);
    CHECK(attempts==0 && capture_queries==0 && sync_calls==0 && free_attempts==0);
    CHECK(g_live.empty() && physical.empty() && private_ledger_matches());
  }
  reset_context();
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
  CHECK(route0(320,256,128)==0);
  sync_fail_at=1;
  CHECK(fak_qwen35_sequence_sync()==60700);
  g_cuda_cleanup_status=719;
  CHECK(fcuda_sequence_split_end(token,&status)==700);
  CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_QUARANTINE && status.code==700);
  CHECK(g_sequence_split.failed_fence==700 && g_cuda_cleanup_status==719);
  CHECK(sync_calls==1 && capture_queries==1 && free_attempts==0 && pending_reads==1);
  CHECK(g_live.size()==4 && physical.size()==4 && private_ledger_matches());
  for (Route route : routes) {
    for (int fail=1; fail<=3; ++fail) {
      reset_context();
      CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
      cublas_fail_at=fail;
      CHECK(route(320,256,128)==93000+cublas_error);
      CHECK(cublas_calls==fail && timing_begins==fail && timing_ends==fail);
      CHECK(g_iq4xs_dequant_gemm_launches==(route==route4 ? 1u : 0u));
      CHECK(!g_sequence_split.borrowed && g_live.size()==4 && pooled()==0 && retired==0);
      CHECK(private_ledger_matches());
      CHECK(fcuda_sequence_split_end(token,&status)==0);
      CHECK(g_live.empty() && physical.empty() && retired==4 && private_ledger_matches());
    }
  }
  reset_context();
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
  CHECK(private_ledger_matches());
  CHECK(fcuda_sequence_split_end(token,&status)==0);
  CHECK(private_ledger_matches());
  for (int miss=1; miss<=4; ++miss) {
    reset_context(); alloc_fail_at=miss;
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==cudaErrorMemoryAllocation);
    CHECK(token==0 && status.phase==FCUDA_SEQUENCE_SPLIT_ALLOCATE && status.allocation_was_oom);
    CHECK(attempts==miss && free_attempts==miss-1 && retired==miss-1);
    CHECK(last_error==0 && get_last_error_calls==1);
    CHECK(allocated.size()==(size_t)miss-1);
    for (int i=0; i<miss-1; ++i) CHECK(allocated[i].second==(i<2 ? wbytes : xbytes));
    CHECK(g_live.empty() && physical.empty() && g_pool.empty() && sync_calls==0);
    CHECK(private_ledger_matches());
  }
  // Synthetic misses never consume an unrelated sticky error.
  for (int miss=1; miss<=4; ++miss) {
    reset_context(); g_scratch_alloc_test_fault=miss; last_error=719;
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==cudaErrorMemoryAllocation);
    CHECK(token==0 && attempts==miss-1 && free_attempts==miss-1 && retired==miss-1);
    CHECK(last_error==719 && get_last_error_calls==0);
    CHECK(g_live.empty() && physical.empty() && g_pool.empty());
  }
  reset_context(); alloc_fail_at=1; alloc_error=cudaSuccess; last_error=719;
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==cudaErrorMemoryAllocation);
  CHECK(token==0 && attempts==1 && free_attempts==0 && g_live.empty());
  CHECK(last_error==719 && get_last_error_calls==0); // success+null is synthetic too
  for (int fail=1; fail<=3; ++fail) {
    reset_context(); alloc_fail_at=4; free_fail_at=fail;
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==700);
    CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_ROLLBACK && status.allocation_code==cudaErrorMemoryAllocation);
    CHECK(get_last_error_calls==1 && status.code==700);
    CHECK(token==0 && free_attempts==3 && retired==2 && g_live.size()==1 && physical.size()==1);
    const auto held=allocated[fail-1];
    CHECK(g_sequence_split.ptr[fail-1]==held.first && g_sequence_split.bytes[fail-1]==held.second);
    CHECK(g_live.at(held.first)==held.second && physical.at(held.first)==held.second);
    for (int i=0; i<3; ++i) if (i!=fail-1) CHECK(g_sequence_split.ptr[i]==nullptr);
    CHECK(g_sequence_split.phase==SequenceSplitPhase::Quarantined);
    CHECK(private_ledger_matches());
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==700);
    CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_QUARANTINE && attempts==4);
  }
  reset_context(); alloc_fail_at=2; alloc_error=700;
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==700);
  CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_ALLOCATE && !status.allocation_was_oom);
  CHECK(last_error==700 && get_last_error_calls==0);
  CHECK(g_live.empty() && physical.empty() && free_attempts==1);
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==700 && attempts==2);
  CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_QUARANTINE);

  reset_context(); alloc_fail_at=4; alloc_error=719; free_fail_at=2;
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==700);
  CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_ROLLBACK && status.allocation_code==719 && status.code==700);
  CHECK(!status.allocation_was_oom && free_attempts==3 && retired==2 && g_live.size()==1);

  reset_context(); g_capture_open=true;
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==cudaErrorInvalidValue && attempts==0);
  CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_CONTRACT && capture_queries==0);
  for (int capture : {cudaStreamCaptureStatusActive, cudaStreamCaptureStatusInvalidated}) {
    reset_context(); runtime_capture=capture; // independent of bookkeeping flag
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==cudaErrorInvalidValue);
    CHECK(token==0 && status.phase==FCUDA_SEQUENCE_SPLIT_CONTRACT && attempts==0 && capture_queries==1);
    CHECK(g_sequence_split.phase==SequenceSplitPhase::Idle && g_sequence_split_generation==0);
    CHECK(runtime_capture==capture && sync_calls==0 && free_attempts==0 && g_live.empty());
    runtime_capture=cudaStreamCaptureStatusNone;
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
    CHECK(fcuda_sequence_split_end(token,&status)==0 && retired==4);
  }
  reset_context(); capture_query_error=701;
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==701);
  CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_CAPTURE_QUERY && token==0 && attempts==0);
  CHECK(g_sequence_split.phase==SequenceSplitPhase::Quarantined && sync_calls==0 && free_attempts==0);
  capture_query_error=0;
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==701);
  CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_QUARANTINE && capture_queries==1 && attempts==0);
  reset_context();
  CHECK(fcuda_sequence_split_begin((size_t)-1,xbytes,&token,&status)==cudaErrorInvalidValue && attempts==0);
  g_sequence_split_generation=(unsigned long long)-1;
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==cudaErrorInvalidValue && attempts==0);

  reset_context();
  // A foreign general-pool bucket is preserved, not swept or borrowed.
  void *foreign=fcuda_malloc(1024); fcuda_free(foreign);
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
  CHECK(allocated.size()==5);
  for (int i=0; i<4; ++i) {
    const auto owner=allocated[i+1];
    CHECK(owner.second==(i<2 ? wbytes : xbytes));
    CHECK(g_sequence_split.ptr[i]==owner.first && g_sequence_split.bytes[i]==owner.second);
    CHECK(g_live.at(owner.first)==owner.second && physical.at(owner.first)==owner.second);
    for (int j=0; j<i; ++j) CHECK(owner.first!=allocated[j+1].first);
  }
  const unsigned long long first_token=token;
  unsigned long long rejected=0;
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&rejected,&status)==cudaErrorInvalidValue);
  CHECK(rejected==0 && g_sequence_split.token==first_token && attempts==5);
  CHECK(route0(320,256,128)==0 && route1(640,256,128)==0);
  CHECK(route2(256,512,128)==0 && route3(512,256,128)==0);
  CHECK(attempts==5 && free_attempts==0 && pooled()==1 && g_live.size()==4);
  CHECK(route0(641,256,128)==-1 && attempts==5 && launches==4);
  FP16CompensatedScratch lease;
  CHECK(fp16_compensated_scratch(320*256,128*256,&lease)==0);
  for (int i=0; i<4; ++i) CHECK(lease.ptr[i]==allocated[i+1].first);
  FP16CompensatedScratch other;
  CHECK(fp16_compensated_scratch(320*256,128*256,&other)==-1);
  auto corrupt=lease; corrupt.ptr[0]=(__half *)foreign;
  CHECK(fp16_compensated_release(&corrupt)==-1);
  corrupt=lease; ++corrupt.loan;
  CHECK(fp16_compensated_release(&corrupt)==-1);
  corrupt=lease; corrupt.loan=0;
  CHECK(fp16_compensated_release(&corrupt)==-1 && free_attempts==0 && pooled()==1);
  CHECK(fp16_compensated_release(&lease)==0 && fp16_compensated_release(&lease)==-1);
  CHECK(fcuda_sequence_split_end(token+1,&status)==cudaErrorInvalidValue && sync_calls==0);
  CHECK(fcuda_sequence_split_end(token,&status)==0);
  CHECK(sync_calls==1 && free_attempts==4 && retired==4 && g_live.empty() && physical.size()==1);
  CHECK(fcuda_sequence_split_end(token,&status)==cudaErrorInvalidValue && free_attempts==4);
  fcuda_graph_prewarm(3);
  CHECK(g_pool.size()==1 && pooled()==3 && attempts==7); // no private size keys
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0 && token!=first_token);
  CHECK(fcuda_sequence_split_end(first_token,&status)==cudaErrorInvalidValue);
  CHECK(fcuda_sequence_split_end(token,&status)==0);

  // Retire layer owners before an oversized ordinary all-row output head.
  reset_context();
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
  CHECK(route0(320,256,128)==0 && pending_reads==1);
  CHECK(fcuda_sequence_split_end(token,&status)==0);
  CHECK(g_sequence_split.phase==SequenceSplitPhase::Idle && g_sequence_split.token==0);
  CHECK(sync_calls==1 && retired==4 && free_attempts==4 && physical.empty() && g_live.empty());
  FP16CompensatedScratch head;
  CHECK(fp16_compensated_scratch(4096*512,128*512,&head)==0);
  CHECK(head.loan==0 && head.acquired && attempts==8 && allocated.size()==8);
  for (int i=0; i<4; ++i) {
    CHECK(head.ptr[i]==allocated[i+4].first);
    CHECK(physical.at(head.ptr[i])==(i<2 ? 4096*512*sizeof(__half) : xbytes));
  }
  ++launches; ++pending_reads; // output kernel/GEMM boundary remains simulated
  CHECK(fp16_compensated_release(&head)==0 && !head.acquired && pooled()==4);
  CHECK(sync_calls==1 && pending_reads==1 && retired==4 && free_attempts==4);
  CHECK(fak_qwen35_sequence_sync()==0 && sync_calls==2 && pending_reads==0);
  CHECK(g_sequence_split.phase==SequenceSplitPhase::Idle && free_attempts==4);

  for (int fail=1; fail<=4; ++fail) {
    reset_context();
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
    CHECK(route0(320,256,128)==0);
    free_fail_at=fail;
    CHECK(fcuda_sequence_split_end(token,&status)==700 && status.phase==FCUDA_SEQUENCE_SPLIT_FREE);
    CHECK(sync_calls==1 && free_attempts==4 && retired==3 && g_live.size()==1 && pooled()==0);
    const auto held=allocated[fail-1];
    CHECK(physical.size()==1 && g_live.at(held.first)==held.second && physical.at(held.first)==held.second);
    CHECK(private_ledger_matches());
    for (int i=0; i<4; ++i) CHECK(g_sequence_split.ptr[i]==(i==fail-1 ? held.first : nullptr));
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==700 && attempts==4);
  }
  // End-fence failure and both earlier GDN/shared-sync failure encodings.
  for (int site=0; site<4; ++site) {
    reset_context();
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
    CHECK(route0(320,256,128)==0);
    sync_fail_at=1;
    if (site==1) CHECK(qwen35_gdn_drain_after_error(20001)==70700);
    if (site==2) CHECK(gdn_final_fence()==70700);
    if (site==3) CHECK(fak_qwen35_sequence_sync()==60700);
    const int phase=(site==1 || site==2) ? FCUDA_SEQUENCE_SPLIT_QUARANTINE : FCUDA_SEQUENCE_SPLIT_FENCE;
    CHECK(fcuda_sequence_split_end(token,&status)==700 && status.phase==phase);
    CHECK(sync_calls==1 && free_attempts==0 && g_live.size()==4 && physical.size()==4 && pooled()==0);
    CHECK(private_ledger_matches());
    CHECK(route0(4096,512,128)==-1 && attempts==4 && launches==1);
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==700 && attempts==4);
  }
  reset_context();
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
  CHECK(route0(320,256,128)==0);
  sync_fail_at=1;
  CHECK(fak_qwen35_sequence_sync()==60700);
  sync_fail_at=0;
  CHECK(fak_qwen35_sequence_sync()==0 && sync_calls==2);
  CHECK(fcuda_sequence_split_end(token,&status)==700 && status.phase==FCUDA_SEQUENCE_SPLIT_FENCE);
  CHECK(sync_calls==2 && free_attempts==0 && g_live.size()==4 && physical.size()==4);
  reset_context();
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
  // A caller interrupted after borrow/submission still has one checked exit.
  FP16CompensatedScratch interrupted;
  CHECK(fp16_compensated_scratch(320*256,128*256,&interrupted)==0);
  ++pending_reads;
  CHECK(fcuda_sequence_split_end(token,&status)==0 && sync_calls==1 && retired==4);
  CHECK(fp16_compensated_release(&interrupted)==-1 && free_attempts==4);

  reset_context();
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
  g_capture_open=true;
  CHECK(route0(320,256,128)==-1 && launches==0);
  CHECK(fcuda_sequence_split_end(token,&status)==cudaErrorInvalidValue);
  CHECK(g_capture_open && sync_calls==0 && free_attempts==0 && g_live.size()==4);
  g_capture_open=false;
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==cudaErrorInvalidValue);
  CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_QUARANTINE && attempts==4);

  for (int capture : {cudaStreamCaptureStatusActive, cudaStreamCaptureStatusInvalidated}) {
    reset_context();
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
    CHECK(route0(320,256,128)==0);
    runtime_capture=capture;
    CHECK(fcuda_sequence_split_end(token,&status)==cudaErrorInvalidValue);
    CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_CONTRACT && capture_queries==2);
    CHECK(runtime_capture==capture && sync_calls==0 && free_attempts==0 && g_live.size()==4 && pending_reads==1);
    runtime_capture=cudaStreamCaptureStatusNone;
    CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==cudaErrorInvalidValue);
    CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_QUARANTINE && capture_queries==2 && attempts==4);
  }
  reset_context();
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
  CHECK(route0(320,256,128)==0);
  capture_query_error=701;
  CHECK(fcuda_sequence_split_end(token,&status)==701);
  CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_CAPTURE_QUERY && capture_queries==2);
  CHECK(sync_calls==0 && free_attempts==0 && g_live.size()==4 && physical.size()==4 && pending_reads==1);
  capture_query_error=0;
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==701);
  CHECK(status.phase==FCUDA_SEQUENCE_SPLIT_QUARANTINE && capture_queries==2 && attempts==4);

  reset_context(); sync_fail_at=1;
  CHECK(fak_qwen35_sequence_sync()==60700); // no-loan semantics unchanged
  sync_fail_at=0;
  CHECK(fcuda_sequence_split_begin(wbytes,xbytes,&token,&status)==0);
  CHECK(fcuda_sequence_split_end(token,&status)==0 && retired==4);
  return 0;
#endif
}
`
