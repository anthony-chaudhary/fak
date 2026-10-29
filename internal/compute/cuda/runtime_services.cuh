#ifndef FAK_CUDA_DEVICE_SCOPE_HOST_TEST
// Algorithm adapted from FlashML-org/FreeToken
// python/freetoken/kernel/triton/ple_hash.py:31-83 at
// de77d16a5ac36494d85964f846b50ad36f7acc21 (Apache-2.0).
//
// One block owns one flattened token and one thread owns one PLE head. The
// request-local coordinate is recovered from cuSeqLens inside this launch, so
// neither a packed [B,context+length] window nor a separate token-index kernel
// is needed. Every potentially invalid address is formed only inside its
// proven-valid branch; masked invalid pointer formation is deliberately absent.
__global__ void k_qwen4exp_ple_hash_i64(
    const int64_t *input_ids, const int64_t *ngram_context,
    const int32_t *cu_seqlens, const int64_t *multipliers,
    const int64_t *vocab_sizes, const int64_t *offsets, int64_t *rows,
    int tokens, int requests, int context_len, int ngram_size,
    int heads_per_ngram, int num_heads, int64_t boundary_token) {
  const int token = (int)blockIdx.x;
  if (token >= tokens) return;

  __shared__ int request_shared;
  __shared__ int local_shared;
  if (threadIdx.x == 0) {
    int lo = 0;
    int hi = requests;
    while (lo + 1 < hi) {
      const int mid = lo + (hi - lo) / 2;
      if (cu_seqlens[mid] <= token) lo = mid;
      else hi = mid;
    }
    request_shared = lo;
    local_shared = token - cu_seqlens[lo];
  }
  __syncthreads();

  const int head = (int)threadIdx.x;
  if (head >= num_heads) return;
  const int order = 2 + head / heads_per_ngram;
  const int request = request_shared;
  const int local = local_shared;

  // Unsigned products give the required, defined modulo-2^64 wrap. NVIDIA
  // CUDA uses two's-complement int64_t, so the final cast interprets the exact
  // same bits as the signed Torch/Go oracle before remainder.
  uint64_t mixed = (uint64_t)input_ids[token] * (uint64_t)multipliers[0];
  bool valid = true;
  for (int shift = 1; shift < order; ++shift) {
    const int column = context_len + local - shift;
    int64_t raw = boundary_token;
    if (column >= context_len) {
      // column >= context_len iff local >= shift, hence token-shift is still
      // inside this request and is a valid input_ids address.
      raw = input_ids[token - shift];
    } else if (column >= 0) {
      raw = ngram_context[(size_t)request * (size_t)context_len + (size_t)column];
    }
    valid = valid && column >= 0 && raw != boundary_token;
    const int64_t selected = valid ? raw : boundary_token;
    mixed ^= (uint64_t)selected * (uint64_t)multipliers[shift];
  }

  const int64_t vocab = vocab_sizes[head];
  int64_t rem = (int64_t)mixed % vocab;
  if (rem < 0) rem += vocab; // floor remainder for a positive divisor
  rows[(size_t)token * (size_t)num_heads + (size_t)head] = rem + offsets[head];
}

extern "C" int fcuda_qwen4exp_ple_hash_i64(
    const int64_t *input_ids, const int64_t *ngram_context,
    const int32_t *cu_seqlens, const int64_t *multipliers,
    const int64_t *vocab_sizes, const int64_t *offsets, int64_t *rows,
    int tokens, int requests, int context_len, int ngram_size,
    int heads_per_ngram, int num_heads, int64_t boundary_token) {
  if (!input_ids || !ngram_context || !cu_seqlens || !multipliers ||
      !vocab_sizes || !offsets || !rows || tokens <= 0 || requests <= 0 ||
      context_len != ngram_size - 1 || ngram_size < 2 ||
      ngram_size - 1 > 1024 || heads_per_ngram <= 0 ||
      heads_per_ngram > 1024 / (ngram_size - 1) ||
      num_heads != heads_per_ngram * (ngram_size - 1) ||
      num_heads > 1024) return -1;
  int threads = 32;
  while (threads < num_heads) threads <<= 1;
  (void)cudaGetLastError(); // discard stale status before attributing this launch
  FCUDA_TS("k_qwen4exp_ple_hash_i64", tokens, k_qwen4exp_ple_hash_i64<<<tokens, threads, 0, g_stream>>>(
      input_ids, ngram_context, cu_seqlens, multipliers, vocab_sizes, offsets,
      rows, tokens, requests, context_len, ngram_size, heads_per_ngram,
      num_heads, boundary_token));
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) return (int)launched;
  ++g_qwen4exp_ple_hash_launches;
  return 0;
}

extern "C" int fcuda_qwen4exp_ple_hash_sync(void) {
  return (int)cudaStreamSynchronize(g_stream);
}
extern "C" size_t fcuda_qwen4exp_ple_hash_launches(void) {
  return g_qwen4exp_ple_hash_launches;
}

static cudaEvent_t g_trace_start;
extern "C" double fcuda_event_elapsed_ms_start(void) {
    CK(cudaEventCreate(&g_trace_start));
    CK(cudaEventRecord(g_trace_start, g_stream));
    return 0.0;
}
extern "C" double fcuda_event_elapsed_ms_end(void) {
    cudaEvent_t end;
    CK(cudaEventCreate(&end));
    CK(cudaEventRecord(end, g_stream));
    CK(cudaEventSynchronize(end));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, g_trace_start, end));
    CK(cudaEventDestroy(end));
    CK(cudaEventDestroy(g_trace_start));
    return (double)ms;
}

#endif

extern "C" int fcuda_set_device(int device) {
  cudaError_t e = cudaSetDevice(device);
  if (e != cudaSuccess) {
    fprintf(stderr, "fak-cuda: cudaSetDevice(%d) failed: %s\n", device, cudaGetErrorString(e));
    return (int)e + 1000;
  }
  return 0;
}

#include "device_scope.h"
#include "alloc_status.h"

extern "C" void *fcuda_malloc_on(int device, size_t bytes) {
  if (bytes == 0) bytes = 1;
  fcuda_device_scope scope(device);
  if (scope.status() != 0) return nullptr;
  void *d = nullptr;
  cudaError_t e = fcuda_alloc_status(cudaMalloc(&d, bytes));
  if (e != cudaSuccess) {
    fprintf(stderr, "fak-cuda: cudaMalloc(%zu bytes) on device %d failed: %s\n", bytes, device, cudaGetErrorString(e));
    return nullptr;
  }
  return d;
}

extern "C" void fcuda_free_on(int device, void *d) {
  if (!d) return;
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
  fcuda_device_scope scope(device);
  if (scope.status() != 0) return;
  q4_coeff_cache_evict(d);
#if defined(FAK_CUDA_V41_TAIL_ROPE) && defined(FAK_CUDA_V41_TAIL_ROPE_TEST)
  ++g_v41_rope_test_free_attempts;
#endif
  CK(cudaFree(d));
}

extern "C" void fcuda_h2d_on(int device, void *d, const void *h, size_t n) {
  fcuda_device_scope scope(device);
  if (scope.status() != 0) return;
  fcuda_ts_untimed_transfer();
  cudaError_t e = cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
  if (e == cudaSuccess) g_h2d_bytes += n;
  else fprintf(stderr, "fak-cuda: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
}

extern "C" int fcuda_d2h_on_checked(int device, void *h, const void *d, size_t n) {
  fcuda_device_scope scope(device);
  if (scope.status() != 0) return scope.status();
  fcuda_ts_untimed_transfer();
  cudaError_t e = cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    fprintf(stderr, "fak-cuda: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (int)e + 1000;
  }
  g_host_bytes += n;
  return 0;
}

extern "C" void fcuda_d2h_on(int device, void *h, const void *d, size_t n) {
  (void)fcuda_d2h_on_checked(device, h, d, n);
}

extern "C" void fcuda_d2d_on(int device, void *dst, const void *src, size_t n) {
  fcuda_device_scope scope(device);
  if (scope.status() != 0) return;
  fcuda_ts_untimed_transfer(); // a peer device's copy: outside the single-device event ring
  // The process stream belongs to device 0. Use this device's legacy stream,
  // then fence explicitly: even a synchronous D2D memcpy need not block host.
  cudaError_t e = cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, cudaStreamLegacy);
  if (e == cudaSuccess) e = cudaStreamSynchronize(cudaStreamLegacy);
  if (e != cudaSuccess)
    fprintf(stderr, "fak-cuda: device %d copy/fence failed: %s\n", device, cudaGetErrorString(e));
}

// host-transfer witness accessors (#482/#4738): see g_host_bytes/g_h2d_bytes above.
extern "C" size_t fcuda_hostxfer_bytes(void) { return g_host_bytes; }
extern "C" void fcuda_hostxfer_reset(void) { g_host_bytes = 0; }
extern "C" size_t fcuda_h2dxfer_bytes(void) { return g_h2d_bytes; }
extern "C" void fcuda_h2dxfer_reset(void) { g_h2d_bytes = 0; }
