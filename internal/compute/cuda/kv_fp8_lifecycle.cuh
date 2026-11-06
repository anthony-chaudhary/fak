// SPDX-License-Identifier: Apache-2.0
// FP8 eager lifetime adapter. Original Fak code; numerical donor lineage is in
// kv_fp8.cuh. Include AFTER runtime_memory.cuh and kv_precision.cuh, which define
// ordinary allocation maps, ledger functions and all retained-pointer guards.
// This proposal is intentionally unlinked until its Go/owner/ingress gates land.
#ifndef FAK_CUDA_KV_FP8_LIFECYCLE_CUH
#define FAK_CUDA_KV_FP8_LIFECYCLE_CUH
#include "../cuda_kv_fp8_lifecycle.h"
#include "kv_fp8.cuh"
#include <new>
#include <stdlib.h>
#include <string.h>

namespace fak_kv_fp8_lifecycle {
using result = fcuda_kv_fp8_lifecycle_result;
static result refusal(int status, int phase) {
  result r = {}; r.status = status; r.phase = phase; return r;
}
static result failed(int status, int phase) {
  if (!status) status = FCUDA_KV_FP8_RECEIPT_FAILED;
  if (!g_cuda_cleanup_status) g_cuda_cleanup_status = status;
  result r = refusal(status, phase); r.terminal = 1; return r;
}
static result admit(int phase) {
  if (v41_candidates_stream_uncertain())
    return failed(g_cuda_cleanup_status ? g_cuda_cleanup_status : (int)cudaErrorNotReady,phase);
  if (g_capture_open) return refusal((int)cudaErrorStreamCaptureUnsupported,phase);
  int device = -1;
  cudaError_t e = cudaGetDevice(&device);
  if (e != cudaSuccess) return failed((int)e,phase);
  if (device != 0) return refusal((int)cudaErrorInvalidDevice,phase);
  e = cudaPeekAtLastError();
  if (e != cudaSuccess) return failed((int)e,phase); // never clear a prior error
  return refusal(0,phase);
}
static const char *site_name(int role) {
  return role == FCUDA_KV_FP8_OWNER_PLANE ? "kv-fp8-owner" :
         role == FCUDA_KV_FP8_OWNER_WORKSPACE ? "kv-fp8-workspace" : nullptr;
}
static int allocation_class(int role) {
  return role == FCUDA_KV_FP8_OWNER_PLANE ? FAK_ALLOC_KV_CACHE : FAK_ALLOC_ACTIVATIONS_SCRATCH;
}
static bool live(const void *p, size_t bytes, int role) {
  const char *site = site_name(role);
  uintptr_t begin = (uintptr_t)p;
  if (!site || !p || !bytes || bytes > UINTPTR_MAX-begin) return false;
  auto found = g_live.find((void *)p);
  if (found == g_live.end() || found->second != bytes ||
      g_managed_live.find((void *)p) != g_managed_live.end() ||
      g_panel_live.find((void *)p) != g_panel_live.end() ||
      kv_snapshot_release_refused((void *)p)) return false;
  fak_alloc_tags_guard guard;
  auto tag = g_alloc_tags.find(p);
  if (tag == g_alloc_tags.end()) return false;
  const int sites = g_alloc_site_n.load(std::memory_order_relaxed);
  const fak_alloc_tag &t = tag->second;
  return t.bytes == bytes && t.cls == allocation_class(role) && t.kind == FAK_ALLOC_DEVICE_LOCAL &&
      t.site > FAK_ALLOC_SITE_OVERFLOW && t.site < sites && t.site < FAK_ALLOC_SITE_CAP &&
      strcmp(g_alloc_sites[t.site].name,site) == 0;
}

// C-owned status-copy target survives every uncertain copy/fence/free. The list
// is protected by cudaMu and never drained/retried after a terminal outcome.
struct readback {
  void *host;
  const void *device;
  readback *next;
};
static readback *readbacks = nullptr;
static result stop(result r, int phase) {
  return r.terminal ? r : failed(r.status,phase);
}
} // namespace fak_kv_fp8_lifecycle

extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_lifecycle_admit(void) {
  return fak_kv_fp8_lifecycle::admit(FCUDA_KV_FP8_PHASE_ADMIT);
}
extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_lifecycle_failed(int status, int phase) {
  return fak_kv_fp8_lifecycle::failed(status,phase);
}
extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_allocate_owned(
    void **out, size_t bytes, int role) {
  using namespace fak_kv_fp8_lifecycle;
  result r = admit(FCUDA_KV_FP8_PHASE_ALLOCATE);
  if (r.status) return r;
  const char *name = site_name(role);
  if (!out || *out || !bytes || !name) return refusal((int)cudaErrorInvalidValue,r.phase);
  int site;
  try { site = fak_alloc_site_intern(name); }
  catch (...) { return failed(FCUDA_KV_FP8_BOOKKEEPING_FAILED,r.phase); }
  if (site <= FAK_ALLOC_SITE_OVERFLOW) return refusal((int)cudaErrorInvalidValue,r.phase);
  // Write the caller's already-rooted allocation slot before ledger operations.
  // No managed fallback or idle-pool reuse is permitted in this baseline.
  cudaError_t e = fcuda_alloc_status(cudaMalloc(out,bytes));
  if (e != cudaSuccess) {
    if (e == cudaErrorMemoryAllocation && !*out) return refusal((int)e,r.phase);
    return failed((int)e,r.phase);
  }
  try {
    g_live[*out] = bytes;
    fak_alloc_track(*out,allocation_class(role),FAK_ALLOC_DEVICE_LOCAL,site,bytes);
  } catch (...) {
    // Pointer is still returned/rooted. Never free or pretend it was unallocated
    // after partial bookkeeping; the backend closes and retains it.
    return failed(FCUDA_KV_FP8_BOOKKEEPING_FAILED,r.phase);
  }
  return r;
}
extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_begin_owned(int *status) {
  using namespace fak_kv_fp8_lifecycle;
  result r = admit(FCUDA_KV_FP8_PHASE_BEGIN);
  if (r.status) return stop(r,r.phase);
  if (!live(status,sizeof(int),FCUDA_KV_FP8_OWNER_WORKSPACE))
    return failed((int)cudaErrorInvalidValue,r.phase);
  int code = fcuda_kv_fp8_begin_async(status);
  return code ? failed(code,r.phase) : r;
}
extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_copy_owned(
    void *dst, size_t dst_bytes, size_t dst_offset,
    const void *src, size_t src_bytes, size_t src_offset, size_t bytes) {
  using namespace fak_kv_fp8_lifecycle;
  result r = admit(FCUDA_KV_FP8_PHASE_COPY);
  if (r.status) return stop(r,r.phase);
  if (!bytes || !live(dst,dst_bytes,FCUDA_KV_FP8_OWNER_PLANE) ||
      !live(src,src_bytes,FCUDA_KV_FP8_OWNER_PLANE) ||
      dst == src || dst_offset > dst_bytes || bytes > dst_bytes-dst_offset ||
      src_offset > src_bytes || bytes > src_bytes-src_offset)
    return failed((int)cudaErrorInvalidValue,r.phase);
  uintptr_t d = (uintptr_t)dst + dst_offset, s = (uintptr_t)src + src_offset;
  if (d < s+bytes && s < d+bytes) return failed((int)cudaErrorInvalidValue,r.phase);
  cudaError_t e = cudaMemcpyAsync((void *)d,(const void *)s,bytes,cudaMemcpyDeviceToDevice,g_stream);
  return e == cudaSuccess ? r : failed((int)e,r.phase);
}
extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_finish_owned(
    const int *status, int read_numerical) {
  using namespace fak_kv_fp8_lifecycle;
  result r = admit(FCUDA_KV_FP8_PHASE_FENCE);
  if (r.status) return stop(r,r.phase);
  if ((read_numerical != 0 && read_numerical != 1) ||
      (read_numerical && !live(status,sizeof(int),FCUDA_KV_FP8_OWNER_WORKSPACE)) ||
      (!read_numerical && status))
    return failed((int)cudaErrorInvalidValue,r.phase);
  readback *holder = nullptr;
  if (read_numerical) {
    holder = (readback *)calloc(1,sizeof(readback));
    if (!holder) return failed((int)cudaErrorMemoryAllocation,FCUDA_KV_FP8_PHASE_HOST_ALLOCATE);
    holder->device = status;
    holder->next = readbacks; readbacks = holder; // root BEFORE any asynchronous reference
    cudaError_t e = cudaHostAlloc(&holder->host,sizeof(int),cudaHostAllocDefault);
    if (e != cudaSuccess) return failed((int)e,FCUDA_KV_FP8_PHASE_HOST_ALLOCATE);
    try {
      fak_alloc_track(holder->host,FAK_ALLOC_ACTIVATIONS_SCRATCH,FAK_ALLOC_PINNED_HOST,
                      FAK_ALLOC_SITE("kv-fp8-status-readback"),sizeof(int));
    } catch (...) { return failed(FCUDA_KV_FP8_BOOKKEEPING_FAILED,FCUDA_KV_FP8_PHASE_HOST_ALLOCATE); }
    fcuda_ts_untimed_transfer();
    e = cudaMemcpyAsync(holder->host,status,sizeof(int),cudaMemcpyDeviceToHost,g_stream);
    if (e != cudaSuccess) return failed((int)e,FCUDA_KV_FP8_PHASE_STATUS_COPY);
  }
  cudaError_t e = cudaStreamSynchronize(g_stream);
  if (e != cudaSuccess) return failed((int)e,FCUDA_KV_FP8_PHASE_FENCE);
  r.completed = 1; r.numerical_checked = 1;
  if (holder) {
    r.numerical = (uint32_t)*(const int *)holder->host;
    g_host_bytes += sizeof(int); // an actual successful status transfer, not zero-D2H
    e = cudaFreeHost(holder->host);
    if (e != cudaSuccess) {
      result failure = failed((int)e,FCUDA_KV_FP8_PHASE_HOST_RETIRE);
      failure.completed = r.completed; failure.numerical_checked = r.numerical_checked;
      failure.numerical = r.numerical; return failure;
    }
    const bool tracked = fak_alloc_untrack(holder->host);
    holder->host = nullptr;
    if (!tracked) {
      result failure = failed(FCUDA_KV_FP8_RECEIPT_FAILED,FCUDA_KV_FP8_PHASE_HOST_RETIRE);
      failure.completed = r.completed; failure.numerical_checked = r.numerical_checked;
      failure.numerical = r.numerical; return failure;
    }
    if (readbacks != holder) return failed(FCUDA_KV_FP8_RECEIPT_FAILED,FCUDA_KV_FP8_PHASE_HOST_RETIRE);
    readbacks = holder->next;
    free(holder);
  }
  return r;
}
extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_release_owned(
    void *expected, size_t bytes) {
  using namespace fak_kv_fp8_lifecycle;
  result r = admit(FCUDA_KV_FP8_PHASE_RETIRE);
  if (r.status) return stop(r,r.phase);
  if (!live(expected,bytes,FCUDA_KV_FP8_OWNER_PLANE) &&
      !live(expected,bytes,FCUDA_KV_FP8_OWNER_WORKSPACE))
    return failed((int)cudaErrorInvalidValue,r.phase);
  // Producer readiness alone does not fence later attention reads. This eager
  // baseline fences every retirement; a batched retirement ticket is later work.
  cudaError_t e = cudaStreamSynchronize(g_stream);
  if (e != cudaSuccess) return failed((int)e,FCUDA_KV_FP8_PHASE_FENCE);
  e = cudaFree(expected);
  if (e != cudaSuccess) return failed((int)e,r.phase);
  r.retired = 1; // publish disposition BEFORE any bookkeeping can fail
  try {
    const bool tracked = fak_alloc_untrack(expected);
    g_live.erase(expected);
    if (!tracked) {
      result failure = failed(FCUDA_KV_FP8_RECEIPT_FAILED,r.phase);
      failure.retired = 1; return failure;
    }
  } catch (...) {
    result failure = failed(FCUDA_KV_FP8_BOOKKEEPING_FAILED,r.phase);
    failure.retired = 1; return failure;
  }
  return r;
}
#endif
