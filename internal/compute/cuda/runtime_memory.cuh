#include "alloc_status.h"
extern "C" int fcuda_init(char *name, int namelen, int *sm, size_t *total_mem) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 1;
  if (cudaSetDevice(0) != cudaSuccess) return 2;
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 3;
  if (name && namelen > 0) { strncpy(name, p.name, namelen - 1); name[namelen - 1] = 0; }
  if (sm) *sm = p.major * 10 + p.minor;
  if (total_mem) *total_mem = p.totalGlobalMem;
  if (!g_blas) { if (cublasCreate(&g_blas) != CUBLAS_STATUS_SUCCESS) return 4; }
  cublasSetPointerMode(g_blas, CUBLAS_POINTER_MODE_HOST);
  if (!g_stream) CK(cudaStreamCreate(&g_stream));
  cublasSetStream(g_blas, g_stream);
  // Pre-allocate an explicit cuBLAS workspace (#969). With NO user workspace, cuBLAS lazily
  // cudaMalloc's its own internal workspace the first time it sees a given GEMM problem — and
  // CUDA forbids cudaMalloc while g_stream is mid stream-capture. Because the decode GEMMs are
  // captured into the CUDA-graph (cuda_graph_test.go / the HAL fast path), that lazy alloc fired
  // INSIDE capture on a fresh A100 (sm_80, CUDA 12.8/13.0) and crashed graph replay with
  // "cudaMalloc(256 bytes) ... operation not permitted when stream is capturing". Handing cuBLAS
  // a user-owned workspace makes it use that buffer exclusively and never allocate during capture.
  // 32 MiB is NVIDIA's recommended size for Hopper and is safe (a superset) on Ampere/Ada too.
  if (!g_blas_workspace) {
    const size_t ws = (size_t)32 * 1024 * 1024;
    if (fcuda_alloc_status(cudaMalloc(&g_blas_workspace, ws)) == cudaSuccess) {
      fak_alloc_track(g_blas_workspace, FAK_ALLOC_WORKSPACE_CUBLAS, FAK_ALLOC_DEVICE_LOCAL,
                      FAK_ALLOC_SITE("cublas-workspace"), ws);
      cublasSetWorkspace(g_blas, g_blas_workspace, ws);
    }
  }
  return 0;
}

// fcuda_set_tf32 toggles the cuBLAS math mode for the f32 SGEMM path (Lever 4 of the
// H100-KERNEL-5X-ROADMAP). ON => CUBLAS_TF32_TENSOR_OP_MATH: the f32 GEMMs (the prefill
// projections — fcuda_matmul_f32 / cublasSgemm above) run on Hopper/Ampere TENSOR CORES at
// TF32 input precision (10-bit mantissa) with F32 accumulation, instead of the FP32 CUDA
// cores. That is a large prefill speedup (the compute-bound phase) at a small, DISCLOSED
// precision cost — TF32 keeps the f32 exponent, so the dynamic range is unchanged and only
// the mantissa narrows. OFF (default) => CUBLAS_DEFAULT_MATH: the pedantic FP32-core path the
// recorded device-vs-cpuref cosine floors were witnessed against, so an existing run's numerics
// never change unless TF32 is explicitly requested. The F16 HGEMM peers (fcuda_matmul_f16) are
// unaffected — they already select a tensor-op algorithm explicitly via cublasGemmEx. Idempotent
// and safe before fcuda_init (no handle yet => nothing to set; the Go init applies it after
// fcuda_init creates g_blas).
extern "C" void fcuda_set_tf32(int on) {
  if (!g_blas) return;
  cublasSetMathMode(g_blas, on ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
}

extern "C" int fcuda_mem_info(size_t *free_mem, size_t *total_mem) {
  size_t free_b = 0;
  size_t total_b = 0;
  cudaError_t e = cudaMemGetInfo(&free_b, &total_b);
  if (e != cudaSuccess) return (int)e;
  if (free_mem) *free_mem = free_b;
  if (total_mem) *total_mem = total_b;
  return 0;
}

extern "C" void *fcuda_malloc_tagged(size_t bytes, int cls, int site) {
  if (g_cuda_cleanup_status != 0) return nullptr;
  if (bytes == 0) bytes = 1;
  auto it = g_pool.find(bytes);
  void *d = nullptr;
  if (it != g_pool.end() && !it->second.empty()) {
    d = it->second.back();
    it->second.pop_back();
    fak_alloc_retag(d, cls, site);
  } else {
    if (g_capture_open) {
      // #969: a device alloc on the captured decode path is the exact bug this guard exists to
      // kill — CUDA forbids cudaMalloc between BeginCapture and EndCapture, so it would crash the
      // whole serve with the opaque "operation not permitted while the stream is capturing". Every
      // graphed-decode scratch (the cuBLAS workspace, the g_q8_* activation-quant buffers, the
      // argmax index) MUST be pre-warmed before fcuda_graph_begin. Fail LOUD naming the size so the
      // offending un-pre-warmed allocation is pinpointed instead of surfacing as a generic capture
      // error. Returning nullptr routes the Go side to its typed DeviceAllocError panic.
      fprintf(stderr, "fak-cuda: BUG #969: fcuda_malloc(%zu) on a pool miss while a CUDA graph capture is OPEN — pre-warm this scratch before BeginCapture\n", bytes);
      return nullptr;
    }
    cudaError_t _e = fcuda_alloc_status(cudaMalloc(&d, bytes));
    if (_e != cudaSuccess) {
      // Report the TRUE reason instead of letting CK swallow it and the Go caller panic with no
      // cause: an out-of-memory says "out of memory", a context poisoned by a prior asynchronous
      // kernel/launch fault says e.g. "an illegal memory access was encountered". Returning nullptr
      // keeps a genuine OOM loud (dalloc still panics) — this EXPOSES the error, it does not mask it.
      fprintf(stderr, "fak-cuda: cudaMalloc(%zu bytes) failed: %s\n", bytes, cudaGetErrorString(_e));
      return nullptr;
    }
    fak_alloc_track(d, cls, FAK_ALLOC_DEVICE_LOCAL, site, bytes);
  }
  g_live[d] = bytes;
  return d;
}
extern "C" void *fcuda_malloc(size_t bytes) {
  return fcuda_malloc_tagged(bytes, FAK_ALLOC_OTHER, FAK_ALLOC_SITE("fcuda_malloc"));
}
extern "C" void *fcuda_malloc_managed_tagged(size_t bytes, int cls, int site) {
  if (g_cuda_cleanup_status != 0) return nullptr;
  if (bytes == 0) bytes = 1;
  void *d = nullptr;
  cudaError_t _e = fcuda_alloc_status(cudaMallocManaged(&d, bytes, cudaMemAttachGlobal));
  if (_e != cudaSuccess) {
    fprintf(stderr, "fak-cuda: cudaMallocManaged(%zu bytes) failed: %s\n", bytes, cudaGetErrorString(_e));
    return nullptr;
  }
  fak_alloc_track(d, cls, FAK_ALLOC_MANAGED, site, bytes);
  g_managed_live[d] = bytes;
  return d;
}
extern "C" void *fcuda_malloc_managed(size_t bytes) {
  return fcuda_malloc_managed_tagged(bytes, FAK_ALLOC_OTHER, FAK_ALLOC_SITE("fcuda_malloc_managed"));
}
static void q4_coeff_cache_evict(const void *q);

extern "C" void fcuda_free(void *d) {
  if (!d) return;
  if (g_cuda_cleanup_status != 0) return;
  if (g_v41_router_uncertain) return;
#ifdef FAK_CUDA_V41_BF16_PUBLISH
  // Publication drains prior stream work too; its three operands cannot name
  // every allocation still in use after an unsuccessful drain.
  if (g_v41_bf16_uncertain) return;
#endif
#ifdef FAK_CUDA_V41_COMPRESSOR
  // A failed drain can cover earlier no-emit ring stores, not just the last
  // call's pointers. Keep every allocation out of reuse while it is uncertain.
  if (g_v41_compressor_uncertain) return;
#endif
#ifdef FAK_CUDA_V41_ROUTED_SWIGLU
  for (const void *held : g_v41_routed_swiglu_quarantine)
    if (d == held) return;
#endif
#ifdef FAK_CUDA_V41_ENGRAM_GATE
  for (const void *held : g_v41_engram_quarantine)
    if (d == held) return;
#endif
#ifdef FAK_CUDA_V41_SHARED_ATTENTION
  for (const void *held : g_v41_sa_quarantine)
    if (d == held) return;
#endif
#ifdef FAK_CUDA_V41_COMPRESSOR_POOL
  for (const void *held : g_v41_pool_quarantine)
    if (d == held) return;
#endif
#ifdef FAK_CUDA_DEEPGEMM_WKV
  for (const void *held : g_deepgemm_wkv_quarantine) {
    if (d == held) return;
  }
#endif
#ifdef FAK_CUDA_FLASHMLA_PREFILL
  for (const void *held : g_flashmla_quarantine) {
    if (d == held) return;
  }
#endif
#ifdef FAK_CUDA_V41_MHC
  for (const void *held : g_v41_mhc_quarantine) {
    if (d == held) return;
  }
#endif
#ifdef FAK_CUDA_V41_INDEXER
  for (const void *held : g_v41_indexer_quarantine) {
    if (d == held) return;
  }
#endif
#ifdef FAK_CUDA_V41_BF16_PUBLISH
  for (const void *held : g_v41_bf16_quarantine) {
    if (d == held) return;
  }
#endif
#ifdef FAK_CUDA_V41_TAIL_ROPE
  for (const void *held : g_v41_rope_quarantine) {
    if (d == held) return;
  }
#endif
  q4_coeff_cache_evict(d);
  auto mit = g_managed_live.find(d);
  if (mit != g_managed_live.end()) {
    g_managed_live.erase(mit);
#if defined(FAK_CUDA_V41_TAIL_ROPE) && defined(FAK_CUDA_V41_TAIL_ROPE_TEST)
    ++g_v41_rope_test_free_attempts;
#endif
    fak_alloc_untrack(d);
    cudaFree(d);
    return;
  }
  auto pit = g_panel_live.find(d);
  if (pit != g_panel_live.end()) {
    fak_alloc_retag(d, FAK_ALLOC_IDLE_POOL, FAK_ALLOC_SITE("panel-pool"));
    g_panel_pool[pit->second].push_back(d); // panel scratch stays out of g_pool
    g_panel_live_bytes -= pit->second;
    g_panel_cached_bytes += pit->second;
    g_panel_live.erase(pit);
    return;
  }
  auto it = g_live.find(d);
  if (it != g_live.end()) {
    fak_alloc_retag(d, FAK_ALLOC_IDLE_POOL, FAK_ALLOC_SITE("device-pool"));
    g_pool[it->second].push_back(d); // return to the pool for reuse, don't cudaFree
    g_live.erase(it);
  } else {
#if defined(FAK_CUDA_V41_TAIL_ROPE) && defined(FAK_CUDA_V41_TAIL_ROPE_TEST)
    ++g_v41_rope_test_free_attempts;
#endif
    fak_alloc_untrack(d);
    cudaFree(d);
  }
}

// Only for a caller-owned, unpublished reservation. Never sweep a pool or
// release a pointer with an unknown owner. A failed drain/free leaves the live
// record intact and unavailable to pool reuse; the caller must surface the fault.
extern "C" int fcuda_discard_unsubmitted(void *d, size_t expected_bytes) {
  if (!d || g_capture_open) return (int)cudaErrorInvalidValue;
  if (g_cuda_cleanup_status != 0) return g_cuda_cleanup_status;
  if (g_v41_router_uncertain) return (int)cudaErrorInvalidValue;
#ifdef FAK_CUDA_V41_BF16_PUBLISH
  if (g_v41_bf16_uncertain) return (int)cudaErrorInvalidValue;
#endif
#ifdef FAK_CUDA_V41_COMPRESSOR
  if (g_v41_compressor_uncertain) return (int)cudaErrorInvalidValue;
#endif
#ifdef FAK_CUDA_V41_ROUTED_SWIGLU
  for (const void *held : g_v41_routed_swiglu_quarantine)
    if (d == held) return (int)cudaErrorInvalidValue;
#endif
#ifdef FAK_CUDA_V41_ENGRAM_GATE
  for (const void *held : g_v41_engram_quarantine)
    if (d == held) return (int)cudaErrorInvalidValue;
#endif
#ifdef FAK_CUDA_V41_SHARED_ATTENTION
  for (const void *held : g_v41_sa_quarantine)
    if (d == held) return (int)cudaErrorInvalidValue;
#endif
#ifdef FAK_CUDA_V41_COMPRESSOR_POOL
  for (const void *held : g_v41_pool_quarantine)
    if (d == held) return (int)cudaErrorInvalidValue;
#endif
#ifdef FAK_CUDA_DEEPGEMM_WKV
  for (const void *held : g_deepgemm_wkv_quarantine)
    if (d == held) return (int)cudaErrorInvalidValue;
#endif
#ifdef FAK_CUDA_FLASHMLA_PREFILL
  for (const void *held : g_flashmla_quarantine)
    if (d == held) return (int)cudaErrorInvalidValue;
#endif
  auto it = g_live.find(d);
  if (it == g_live.end() || it->second != expected_bytes)
    return (int)cudaErrorInvalidValue;
#ifdef FAK_CUDA_V41_MHC
  for (const void *held : g_v41_mhc_quarantine)
    if (d == held) return (int)cudaErrorInvalidValue;
#endif
#ifdef FAK_CUDA_V41_INDEXER
  for (const void *held : g_v41_indexer_quarantine)
    if (d == held) return (int)cudaErrorInvalidValue;
#endif
#ifdef FAK_CUDA_V41_BF16_PUBLISH
  for (const void *held : g_v41_bf16_quarantine)
    if (d == held) return (int)cudaErrorInvalidValue;
#endif
#ifdef FAK_CUDA_V41_TAIL_ROPE
  for (const void *held : g_v41_rope_quarantine)
    if (d == held) return (int)cudaErrorInvalidValue;
#endif
  // A pool-borrowed reservation can still follow its previous owner's queued
  // work on g_stream, despite this reservation not yet having submitted work.
  cudaError_t status = cudaStreamSynchronize(g_stream);
  if (status != cudaSuccess) return (int)status;
  status = cudaFree(d);
  if (status != cudaSuccess) return (int)status;
  fak_alloc_untrack(d);
  g_live.erase(it);
  return 0;
}

extern "C" size_t fcuda_live_allocations(void) {
  return g_live.size() + g_managed_live.size() + g_panel_live.size();
}

// fcuda_malloc_panel serves prompt-panel scratch from g_panel_pool, never managed memory.
// nullptr means the device allocation failed or a graph capture is open.
extern "C" void *fcuda_malloc_panel_tagged(size_t bytes, int cls, int site) {
  if (g_cuda_cleanup_status != 0) return nullptr;
  if (bytes == 0) bytes = 1;
  if (g_capture_open) return nullptr;
  void *d = nullptr;
  auto it = g_panel_pool.find(bytes);
  if (it != g_panel_pool.end() && !it->second.empty()) {
    d = it->second.back();
    it->second.pop_back();
    g_panel_cached_bytes -= bytes;
    fak_alloc_retag(d, cls, site);
  } else {
    cudaError_t e = fcuda_alloc_status(cudaMalloc(&d, bytes));
    if (e != cudaSuccess || !d) {
      fprintf(stderr, "fak-cuda: prompt panel cudaMalloc(%zu bytes) failed: %s\n", bytes, cudaGetErrorString(e));
      return nullptr;
    }
    fak_alloc_track(d, cls, FAK_ALLOC_DEVICE_LOCAL, site, bytes);
  }
  g_panel_live[d] = bytes;
  g_panel_live_bytes += bytes;
  if (g_panel_live_bytes + g_panel_cached_bytes > g_panel_peak_bytes)
    g_panel_peak_bytes = g_panel_live_bytes + g_panel_cached_bytes;
  return d;
}
extern "C" void *fcuda_malloc_panel(size_t bytes) {
  return fcuda_malloc_panel_tagged(bytes, FAK_ALLOC_PREFILL_SCRATCH, FAK_ALLOC_SITE("fcuda_malloc_panel"));
}

// fcuda_panel_peak_reset restarts the panel peak at the bytes held now (a new panel scope).
extern "C" void fcuda_panel_peak_reset(void) {
  g_panel_peak_bytes = g_panel_live_bytes + g_panel_cached_bytes;
}

// fcuda_panel_release drains g_stream, then frees every pooled panel buffer. It returns how
// many panel buffers callers still hold; those return to g_panel_pool when freed and leave at
// the next release. A failed fence/free retains the remaining cache and its
// ledger records; the checked entry point reports the sticky native error.
extern "C" size_t fcuda_panel_release(void) {
  if (v41_candidates_stream_uncertain()) return g_panel_live.size();
  cudaError_t status = cudaStreamSynchronize(g_stream);
  if (status != cudaSuccess) {
    g_cuda_cleanup_status = (int)status;
    CK(status);
    return g_panel_live.size();
  }
  for (auto it = g_panel_pool.begin(); it != g_panel_pool.end(); ) {
    auto &bucket = it->second;
    while (!bucket.empty()) {
      void *p = bucket.back();
      status = cudaFree(p);
      if (status != cudaSuccess) {
        g_cuda_cleanup_status = (int)status;
        CK(status);
        return g_panel_live.size();
      }
      fak_alloc_untrack(p);
      bucket.pop_back();
      g_panel_cached_bytes -= it->first;
    }
    it = g_panel_pool.erase(it);
  }
  return g_panel_live.size();
}

extern "C" int fcuda_panel_release_checked(size_t *live) {
  if (!live) return (int)cudaErrorInvalidValue;
  *live = fcuda_panel_release();
  if (g_cuda_cleanup_status != 0) return g_cuda_cleanup_status;
  return v41_candidates_stream_uncertain() ? (int)cudaErrorNotReady : 0;
}

// fcuda_pool_stats reports the cached (free) bytes and non-empty bucket count of g_pool, its
// largest such bucket size, and the panel pool's cached bytes and live buffers. Diagnostics.
extern "C" void fcuda_pool_stats(size_t *pooledBytes, size_t *buckets, size_t *largestBucket,
                                 size_t *panelCachedBytes, size_t *panelLive, size_t *panelPeakBytes) {
  size_t bytes = 0, count = 0, largest = 0, panel = 0;
  for (auto &kv : g_pool) {
    if (kv.second.empty()) continue;
    bytes += kv.first * kv.second.size();
    count++;
    if (kv.first > largest) largest = kv.first;
  }
  for (auto &kv : g_panel_pool) panel += kv.first * kv.second.size();
  if (pooledBytes) *pooledBytes = bytes;
  if (buckets) *buckets = count;
  if (largestBucket) *largestBucket = largest;
  if (panelCachedBytes) *panelCachedBytes = panel;
  if (panelLive) *panelLive = g_panel_live.size();
  if (panelPeakBytes) *panelPeakBytes = g_panel_peak_bytes;
}

// fcuda_graph_prewarm deepens every pool bucket by `extra` spare buffers (#969). The warm
// forward before capture pools ONE set of each transient size, but a single captured decodeChain
// holds several same-size transients live AT ONCE (e.g. the per-layer RMSNorm outputs), so the
// pool can drain mid-forward and the next same-size devTr misses -> cudaMalloc, which is illegal
// while g_stream is capturing. Called once right before fcuda_graph_begin (outside capture), this
// guarantees headroom so the captured forward is served entirely from the free list. Idempotent
// and cheap: it only tops each bucket up to `extra` spares (a few hundred bytes each).
extern "C" void fcuda_graph_prewarm(int extra) {
  if (g_cuda_cleanup_status != 0) return;
  if (extra <= 0) return;
  // Snapshot the sizes first: pushing into g_pool while iterating it would invalidate iterators.
  std::vector<size_t> sizes;
  sizes.reserve(g_pool.size());
  for (auto &kv : g_pool) sizes.push_back(kv.first);
  for (size_t bytes : sizes) {
    auto &bucket = g_pool[bytes];
    while ((int)bucket.size() < extra) {
      void *d = nullptr;
      cudaError_t e = fcuda_alloc_status(cudaMalloc(&d, bytes));
      if (e != cudaSuccess || !d) {
        break; // best-effort; a real OOM still fails loud later
      }
      fak_alloc_track(d, FAK_ALLOC_GRAPH_POOL, FAK_ALLOC_DEVICE_LOCAL, FAK_ALLOC_SITE("graph-prewarm"), bytes);
      bucket.push_back(d);
    }
  }
}

extern "C" void fcuda_trim_pool_large(size_t max_keep_bytes) {
  if (v41_candidates_stream_uncertain()) return;
  for (auto it = g_pool.begin(); it != g_pool.end(); ) {
    if (it->first > max_keep_bytes) {
      auto &bucket = it->second;
      while (!bucket.empty()) {
        void *p = bucket.back();
        cudaError_t status = cudaFree(p);
        if (status != cudaSuccess) {
          g_cuda_cleanup_status = (int)status;
          CK(status);
          return;
        }
        fak_alloc_untrack(p);
        bucket.pop_back();
      }
      it = g_pool.erase(it);
    } else {
      ++it;
    }
  }
}
extern "C" int fcuda_trim_pool_large_checked(size_t max_keep_bytes) {
  fcuda_trim_pool_large(max_keep_bytes);
  if (g_cuda_cleanup_status != 0) return g_cuda_cleanup_status;
  return v41_candidates_stream_uncertain() ? (int)cudaErrorNotReady : 0;
}
extern "C" void fcuda_h2d(void *d, const void *h, size_t n) {
  fcuda_ts_untimed_transfer();
  cudaError_t e = cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
  if (e == cudaSuccess) g_h2d_bytes += n;
  else fprintf(stderr, "fak-cuda: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
}
extern "C" int fcuda_d2h_checked(void *h, const void *d, size_t n) {
  fcuda_ts_untimed_transfer();
  cudaError_t e = cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    fprintf(stderr, "fak-cuda: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (int)e + 1000;
  }
  g_host_bytes += n;
  return 0;
}
extern "C" void fcuda_d2h(void *h, const void *d, size_t n) {
  (void)fcuda_d2h_checked(h, d, n);
}
// Device-to-device copies stay on the default stream but are ASYNC w.r.t. the host: a
// synchronous cudaMemcpy fences the whole device, and RoPE + every KV append issues one,
// so a 30-layer decode paid ~150 full device syncs per token (catastrophic on WSL, where a
// sync is ~1-2 ms). Stream ordering still serializes them against the kernels correctly;
// the only host fence we keep is the final logits d2h in Read.
extern "C" void fcuda_d2d(void *dst, const void *src, size_t n) {
  FCUDA_TS("d2d_copy", dim3(0, 0), CK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, g_stream)));
}
extern "C" void fcuda_sync(void) { CK(cudaDeviceSynchronize()); }
