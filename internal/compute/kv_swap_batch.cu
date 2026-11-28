// Batched KV block transport between a mapped pinned host slab and device
// memory: many small blocks move in one persistent-CTA launch driven by
// descriptor arrays instead of one cudaMemcpyAsync per block. Meant to be
// included by cuda_kernels.cu (one-line #include, wiring owned by the consumer
// packet) so it shares g_stream, the pooled allocator and the transfer
// witnesses of the single libfakcuda object.
//
// vllm/v1/kv_offload/cpu/swap_blocks_triton.py at pin
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// This file is a modified derivative: the Triton _swap_blocks_kernel is
// re-expressed as a CUDA kernel for fak's flat C ABI. Divergences, all
// deliberate:
//   - Triton's masked WORDS_PER_CHUNK vector becomes the CTA's threads sweeping
//     each chunk; the job stride (job = pid; job += num_progs) is unchanged.
//   - Upstream moves size // 8 words and drops any remainder; here the
//     size % 8 trailing bytes are copied too, so a block of any length equals a
//     plain memcpy.
//   - The THRESHOLD_BYTES crossover, which upstream leaves to its caller, is
//     enforced next to MIN_N: such batches are refused with status -4 and the
//     caller takes fcuda_kv_swap_dma.
//   - The word loads need naturally aligned endpoints on device; a batch with
//     a misaligned non-empty endpoint is sent to DMA instead.
//   - The kernel dereferences host addresses directly, so the pinned slab must
//     be mapped with its device pointer equal to its host pointer (UVA); that
//     is checked per call rather than assumed.
//
// Contract (mirrored by the Go reference in kv_swap_batch.go):
//   - src_addrs[j], dst_addrs[j], sizes[j] describe job j: copy sizes[j] bytes.
//   - Program pid of num_progs = min(KSB_NUM_SMS, n) owns jobs pid, pid +
//     num_progs, ...; every job is owned by exactly one program.
//   - Destinations must not overlap each other or any source: jobs run
//     concurrently, so overlap would make the result order-dependent. The host
//     descriptor builder refuses such batches before they reach this file.
// Statuses: 0 ok, -1 null pointer or refused geometry, -2 graph capture open,
// -3 device allocation failure, -4 batch belongs on the DMA path (too few
// descriptors, a block above the threshold, an endpoint off an 8-byte boundary,
// or a slab that is not UVA-mapped),
// positive cudaError_t for copy/launch/execution.

#include <limits.h>

#define KSB_NUM_SMS 12
#define KSB_THRESHOLD_BYTES (28 * 1024)
#define KSB_MIN_N 16
#define KSB_THREADS 256

// One CTA per program. Each CTA walks its strided jobs; inside a job its
// threads split every words_per_chunk window, then the first size % 8 threads
// move the byte tail.
__global__ void k_ksb_swap_blocks(
    const int64_t *__restrict__ src_addrs,
    const int64_t *__restrict__ dst_addrs,
    const int64_t *__restrict__ sizes,
    int64_t n_jobs,
    int64_t words_per_chunk) {
  const int64_t num_progs = (int64_t)gridDim.x;
  const int64_t tid = (int64_t)threadIdx.x;
  const int64_t nthreads = (int64_t)blockDim.x;
  for (int64_t job = (int64_t)blockIdx.x; job < n_jobs; job += num_progs) {
    const int64_t *src = (const int64_t *)(uintptr_t)__ldg(src_addrs + job);
    int64_t *dst = (int64_t *)(uintptr_t)__ldg(dst_addrs + job);
    const int64_t size = __ldg(sizes + job);
    const int64_t words = size / 8;
    for (int64_t start = 0; start < words; start += words_per_chunk) {
      int64_t end = start + words_per_chunk;
      if (end > words) end = words;
      for (int64_t idx = start + tid; idx < end; idx += nthreads) dst[idx] = src[idx];
    }
    const int64_t tail = size - words * 8;
    if (tid < tail) {
      const unsigned char *sb = (const unsigned char *)src + words * 8;
      unsigned char *db = (unsigned char *)dst + words * 8;
      db[tid] = sb[tid];
    }
  }
}

static size_t g_ksb_launches = 0;
static size_t g_ksb_dma_copies = 0;

// ksb_validate checks the descriptor arrays and reports the largest block and
// whether every non-empty endpoint is 8-byte aligned; false means status -1.
static bool ksb_validate(const int64_t *src_addrs, const int64_t *dst_addrs, const int64_t *sizes,
                         int64_t n, int64_t *max_size, bool *aligned) {
  if (n < 0 || n > INT32_MAX) return false;
  if (n > 0 && (!src_addrs || !dst_addrs || !sizes)) return false;
  int64_t m = 0;
  bool a = true;
  for (int64_t j = 0; j < n; ++j) {
    if (sizes[j] < 0) return false;
    if (sizes[j] > 0 && (src_addrs[j] == 0 || dst_addrs[j] == 0)) return false;
    if (sizes[j] > 0 && ((src_addrs[j] | dst_addrs[j]) & 7) != 0) a = false;
    if (sizes[j] > m) m = sizes[j];
  }
  *max_size = m;
  *aligned = a;
  return true;
}

// 1 when the batch belongs on the kernel path, 0 when it belongs on DMA.
extern "C" int fcuda_kv_swap_batch_route(const int64_t *sizes, int64_t n) {
  if (!sizes || n < KSB_MIN_N) return 0;
  for (int64_t j = 0; j < n; ++j) {
    if (sizes[j] > KSB_THRESHOLD_BYTES) return 0;
  }
  return 1;
}

// One synchronous batched move. host_slab/host_slab_bytes name the pinned slab
// every host-side address falls in; device-side addresses are ordinary device
// pointers. bytes_per_chunk must be a positive multiple of 8.
extern "C" int fcuda_kv_swap_batch(
    const int64_t *src_addrs, const int64_t *dst_addrs, const int64_t *sizes, int64_t n,
    int64_t bytes_per_chunk, const void *host_slab, size_t host_slab_bytes) {
  int64_t max_size = 0;
  bool aligned = true;
  if (!ksb_validate(src_addrs, dst_addrs, sizes, n, &max_size, &aligned)) return -1;
  if (bytes_per_chunk < 8 || bytes_per_chunk % 8 != 0) return -1;
  if (!host_slab || host_slab_bytes == 0) return -1;
  if (n < KSB_MIN_N || max_size > KSB_THRESHOLD_BYTES || !aligned) return -4;
  if (g_capture_open) return -2;
  (void)cudaGetLastError(); // discard stale status before attributing this call
  void *mapped = NULL;
  if (cudaHostGetDevicePointer(&mapped, (void *)host_slab, 0) != cudaSuccess || mapped != host_slab) {
    (void)cudaGetLastError();
    return -4;
  }
  const size_t desc_bytes = (size_t)n * sizeof(int64_t);
  void *dSrc = fcuda_malloc(desc_bytes);
  void *dDst = fcuda_malloc(desc_bytes);
  void *dSizes = fcuda_malloc(desc_bytes);
  int status = 0;
  cudaError_t e = cudaSuccess;
  if (!dSrc || !dDst || !dSizes) {
    fcuda_free(dSrc); fcuda_free(dDst); fcuda_free(dSizes);
    return -3;
  }
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dSrc, src_addrs, desc_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += desc_bytes;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dDst, dst_addrs, desc_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += desc_bytes;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dSizes, sizes, desc_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += desc_bytes;
  {
    const unsigned int grid = (unsigned int)(n < KSB_NUM_SMS ? n : KSB_NUM_SMS);
    FCUDA_TS("k_ksb_swap_blocks", grid,
             k_ksb_swap_blocks<<<grid, KSB_THREADS, 0, g_stream>>>(
                 (const int64_t *)dSrc, (const int64_t *)dDst, (const int64_t *)dSizes,
                 n, bytes_per_chunk / 8));
    if ((e = cudaGetLastError()) != cudaSuccess) { status = (int)e; goto done; }
  }
  if ((e = cudaStreamSynchronize(g_stream)) != cudaSuccess) { status = (int)e; goto done; }
  ++g_ksb_launches;
done:
  fcuda_free(dSrc);
  fcuda_free(dDst);
  fcuda_free(dSizes);
  return status;
}

// The DMA path: one async copy per block on g_stream, in descriptor order,
// fenced once. Direction is inferred per block from the unified address space.
extern "C" int fcuda_kv_swap_dma(
    const int64_t *src_addrs, const int64_t *dst_addrs, const int64_t *sizes, int64_t n) {
  int64_t max_size = 0;
  bool aligned = true;
  if (!ksb_validate(src_addrs, dst_addrs, sizes, n, &max_size, &aligned)) return -1;
  if (g_capture_open) return -2;
  (void)cudaGetLastError();
  cudaError_t e = cudaSuccess;
  for (int64_t j = 0; j < n; ++j) {
    if (sizes[j] == 0) continue;
    fcuda_ts_untimed_transfer();
    e = cudaMemcpyAsync((void *)(uintptr_t)dst_addrs[j], (const void *)(uintptr_t)src_addrs[j],
                        (size_t)sizes[j], cudaMemcpyDefault, g_stream);
    if (e != cudaSuccess) break;
    ++g_ksb_dma_copies;
  }
  const cudaError_t fence = cudaStreamSynchronize(g_stream);
  if (e != cudaSuccess) return (int)e;
  return fence == cudaSuccess ? 0 : (int)fence;
}

// Witnesses: cumulative successful kernel launches and DMA block copies.
extern "C" size_t fcuda_kv_swap_batch_launches(void) {
  return g_ksb_launches;
}

extern "C" size_t fcuda_kv_swap_dma_copies(void) {
  return g_ksb_dma_copies;
}
