// cuda_kernels.cu — the CUDA C++ hardware seam behind the typed compute.Backend.
//
// Compiled offline by nvcc into a static lib (libfakcuda.a) that the cgo wrapper
// (cuda.go, //go:build cuda) links. Every op here is f32: this first device backend is
// an *Approx* peer of the cpuref *Reference* — held to the argmax-exact + logit-cosine
// gate, NOT to max|Δ|=0. cuBLAS SGEMM (a different reduction order than the model's fdot
// tree) is what makes that distinction real and honest. Quantized weight dtypes are not
// implemented on device yet; the Go MatMul refuses them with a clear message so the
// boundary stays explicit (see the deferred-work GitHub issues).
//
// DIRECTION.md: this is a sanctioned hardware seam in a statically-typed compiled
// language, OFF the request path, behind a re-validated typed boundary (the flat C ABI
// in cuda_backend.h). The default `go build` excludes it; only `-tags cuda` links it.

#include "cuda_backend.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>  /* __half + __float2half — the fp16 compute path (#484) */
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <vector>

static cublasHandle_t g_blas = nullptr;
static cublasLtHandle_t g_cublas_lt = nullptr; // shared by explicit FP8 and NVFP4 arms
static void *g_blas_workspace = nullptr; // explicit cuBLAS workspace so it never cudaMalloc's mid-capture (#969)
static bool g_capture_open = false;      // true between fcuda_graph_begin and end/abort: arms the #969 dalloc guard
static int g_graph_fence_test_fault = 0; // test hook: the next graph fence reports a replayed fault
// All device work runs on g_stream (a blocking stream, so a synchronous cudaMemcpy on the
// legacy default stream still fences it). One stream is what makes the whole per-token op
// sequence CAPTURABLE into a CUDA graph — the only way to collapse ~600 WSL CUDA calls/
// token (proven floor ~12 tok/s) down to one graph launch.
static cudaStream_t g_stream = 0;

#define CK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
  fprintf(stderr, "fak-cuda: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); } } while (0)

// Caching device allocator. cudaMalloc/cudaFree are slow (~0.5 ms each on WSL and they
// implicitly serialize the device), and the forward loop allocates ~hundreds of small
// output buffers per token. A size-bucketed free list recycles freed buffers so a steady-
// state decode pays ~zero allocation cost after warm-up — the same arena trick llama.cpp's
// graph allocator uses, kept minimal. Single-threaded by the Go-side cudaMu mutex.
static std::unordered_map<size_t, std::vector<void *>> g_pool; // free buffers, by exact byte size
static std::unordered_map<void *, size_t> g_live;              // live ptr -> its byte size
static std::unordered_map<void *, size_t> g_managed_live;      // live cudaMallocManaged ptr -> byte size
// Prompt-panel scratch (model prompt panel, cuda_prompt_panel.go). Panel transients scale with
// the chunk's row count, so they get their own exact-size free list that fcuda_panel_release
// empties when the panel ends. They never enter g_pool, where fcuda_graph_prewarm would deepen
// every panel-sized bucket to its spare count before the next captured decode.
static std::unordered_map<size_t, std::vector<void *>> g_panel_pool;
static std::unordered_map<void *, size_t> g_panel_live;
// Panel bytes held by callers and cached in g_panel_pool, and the scope's peak of their sum.
static size_t g_panel_live_bytes = 0, g_panel_cached_bytes = 0, g_panel_peak_bytes = 0;

#ifdef FAK_CUDA_V41_BF16_PUBLISH
// A failed drain is not permission to recycle memory. At most one uncertain
// publication can be retained: the candidate closes permanently until process
// teardown. Preserve its source as well as both operation-owned allocations.
static const void *g_v41_bf16_quarantine[3] = {nullptr, nullptr, nullptr};
static bool g_v41_bf16_uncertain = false;
#endif

#ifdef FAK_CUDA_V41_TAIL_ROPE
// One terminal uncertain QK operation retains at most five exact allocation
// bases. Rows/readback retain fewer. Never overwrite or reset this quarantine.
static const void *g_v41_rope_quarantine[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
static bool g_v41_rope_uncertain = false;
static uint64_t g_v41_rope_operations = 0;
static uint64_t g_v41_rope_launches = 0;
#ifdef FAK_CUDA_V41_TAIL_ROPE_TEST
static int g_v41_rope_test_fault = 0;
static uint64_t g_v41_rope_test_free_attempts = 0;
#endif
#endif

#ifdef FAK_CUDA_V41_INDEXER
// Terminal uncertain score operation retains its three inputs and one output.
static const void *g_v41_indexer_quarantine[4] = {nullptr, nullptr, nullptr, nullptr};
static bool g_v41_indexer_uncertain = false;
static uint64_t g_v41_indexer_completed = 0;
#endif

#if defined(FAK_CUDA_V41_BF16_PUBLISH) || defined(FAK_CUDA_V41_TAIL_ROPE) || defined(FAK_CUDA_V41_INDEXER)
// Both candidates share g_stream. A successful peer call cannot rehabilitate
// an uncertain operation, even when the Go backend has no fault latch.
static bool v41_candidates_stream_uncertain(void) {
#ifdef FAK_CUDA_V41_BF16_PUBLISH
  if (g_v41_bf16_uncertain) return true;
#endif
#ifdef FAK_CUDA_V41_TAIL_ROPE
  if (g_v41_rope_uncertain) return true;
#endif
#ifdef FAK_CUDA_V41_INDEXER
  if (g_v41_indexer_uncertain) return true;
#endif
  return false;
}
#endif

// Host-transfer witnesses (#482/#4738): cumulative bytes in each direction. Every successful
// h2d/d2h copy adds to its own counter so a whole-operation test can prove zero traffic in both
// directions after initial upload. Monotonic; tests reset them around each measured step.
static size_t g_host_bytes = 0;
static size_t g_h2d_bytes = 0;
// Qwen3.8 PLE hash dispatch witness. This counts successful kernel enqueues,
// not preparation copies or result reads; Go samples it around Dispatch.
static size_t g_qwen4exp_ple_hash_launches = 0;

// Whole-operation Qwen3.5/3.6 GDN witness (#4725/#4738). This counts confirmed
// completed operations, not enqueues; the Go fixture reads it together with both
// transfer counters to prove one real recurrent operation ran without host staging.
static size_t g_qwen35_gdn_operations = 0;
// One-shot deterministic failure injection used only through the unexported Go test helper.
// Stages 2..6 model a launch check failure; stage 7 models final async execution failure.
static int g_qwen35_gdn_test_fault_stage = 0;

// Implementation fragments remain in this single translation unit so they share the stream,
// allocators, library handles, capture state, and counters. Preserve this dependency order.
#include "cuda/kernel_profile.cuh"
#include "cuda/runtime_memory.cuh"
#include "cuda/runtime_services.cuh"
#include "cuda/model_candidates.cuh"
#include "cuda_v41_indexer.cu"
#include "cuda/qwen35_gdn.cuh"
#include "cuda/q4k_mmq.cuh"
#include "cuda/quant_matmul.cuh"
#include "cuda/transformer_ops.cuh"
#include "cuda/qwen35_attention_postprocess.cuh"
#include "cuda/qwen35_residual_norm.cuh"
#include "cuda/fattn_mma_hd256.cuh"
#include "cuda/attention.cuh"
#include "cuda/kv_precision.cuh"
#include "cuda/spec_graph.cuh"
#include "cuda/weight_formats.cuh"
#include "cuda/q4k_fp8lt.cuh"
#include "cuda/mxfp8_cast.cuh"
#include "cuda/peer_handoff.cuh"
