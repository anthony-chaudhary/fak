#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Shih-Ying Yeh (aka KohakuBlueLeaf, KBlueLeaf).
// Copyright 2026 Netra Systems, Inc.
//
// Modified C++ translation of KohakuBlueleaf/Tokenomos at
// eeb47ee09a3c5e66411e90313b017c211d6a3fbb (Apache-2.0). The complete Apache
// License 2.0 is retained in the repository root LICENSE.
// Local behavior: fixed two-arena native owner, caller-supplied device
// destinations, explicit same-stream seal before returning to Go, checked
// terminal retirement, allocation-ledger accounting and fail-closed
// capture/device admission.
//
// This is one process-owned pool, serialized by the existing cudaMu. Normal
// lifetime matches g_stream (the CUDA backend has no session-owned shutdown).
// There is no destructor/atexit CUDA call. At most 128 KiB pinned host storage,
// two CUDA events and fixed metadata are retained once, not once per model.
// Every put copies borrowed host bytes before return; every arena reuse waits
// for its event. A cancelled caller cannot abandon an unsealed borrowed source.
// Faults drain the owning stream before release. An uncertain drain/cleanup
// retains remaining owners and closes the existing device allocator's latch.
// A terminal pool is never reset or reallocated in this process.

#include <cstddef>
#include <cstring>
#include "alloc_ledger.h"

struct fak_tokenomos_staging_pool {
  static constexpr size_t capacity = 64 * 1024;
  static constexpr size_t alignment = 64;
  struct arena {
    void *host = nullptr;
    cudaEvent_t done = nullptr;
    bool pending = false;
  } arenas[2];
  cudaStream_t stream = nullptr;
  size_t at = 0;
  unsigned current = 0, next = 0;
  bool ready = false, open = false, dirty = false;
  bool terminal = false, drained = false, retained = false;
  cudaError_t first_error = cudaSuccess, cleanup_error = cudaSuccess;

  void retain(cudaError_t error) {
    retained = true;
    cleanup_error = error;
    if (g_cuda_cleanup_status == 0) g_cuda_cleanup_status = (int)error;
  }

  // Only a successful stream drain authorizes either event or host release.
  // Do not retry a failed CUDA destroy/free: its ownership outcome is uncertain.
  cudaError_t fail(cudaError_t error) {
    if (terminal) return first_error;
    terminal = true;
    first_error = error == cudaSuccess ? cudaErrorUnknown : error;
    cudaError_t fence = cudaStreamSynchronize(stream ? stream : g_stream);
    if (fence != cudaSuccess) {
      retain(fence);
      return first_error;
    }
    drained = true;
    open = dirty = false;
    for (arena &a : arenas) {
      a.pending = false;
      if (a.done) {
        cudaError_t release = cudaEventDestroy(a.done);
        if (release != cudaSuccess) {
          retain(release);
          return first_error;
        }
        a.done = nullptr;
      }
      if (a.host) {
        cudaError_t release = cudaFreeHost(a.host);
        if (release != cudaSuccess) {
          retain(release);
          return first_error;
        }
        fak_alloc_untrack(a.host);
        a.host = nullptr;
      }
    }
    stream = nullptr;
    return first_error;
  }

  cudaError_t init(cudaStream_t owner) {
    if (terminal) return first_error;
    if (ready) return stream == owner ? cudaSuccess : cudaErrorInvalidValue;
    if (!owner) return cudaErrorInvalidValue;
    stream = owner;
    for (arena &a : arenas) {
      cudaError_t error = cudaHostAlloc(&a.host, capacity, cudaHostAllocDefault);
      if (error != cudaSuccess) return fail(error);
      fak_alloc_track(a.host, FAK_ALLOC_ACTIVATIONS_SCRATCH, FAK_ALLOC_PINNED_HOST,
                      FAK_ALLOC_SITE("tokenomos-id-staging"), capacity);
      error = cudaEventCreateWithFlags(&a.done, cudaEventDisableTiming);
      if (error != cudaSuccess) return fail(error);
    }
    ready = true;
    return cudaSuccess;
  }

  cudaError_t seal() {
    if (terminal) return first_error;
    if (!open) return cudaSuccess;
    if (dirty) {
      arena &a = arenas[current];
      cudaError_t error = cudaEventRecord(a.done, stream);
      if (error != cudaSuccess) return fail(error);
      a.pending = true;
    }
    open = dirty = false;
    return cudaSuccess;
  }

  cudaError_t begin(cudaStream_t owner) {
    cudaError_t error = init(owner);
    if (error != cudaSuccess) return error;
    // Preserve the donor's implicit previous-batch seal too. Real gather calls
    // additionally seal immediately so cancellation needs no later begin call.
    error = seal();
    if (error != cudaSuccess) return error;
    arena &a = arenas[next];
    if (a.pending) {
      error = cudaEventSynchronize(a.done);
      if (error != cudaSuccess) return fail(error);
      a.pending = false;
    }
    current = next;
    next = (next + 1) % 2;
    at = 0;
    open = true;
    return cudaSuccess;
  }

  cudaError_t put(void *device, const void *host, size_t bytes, void (*before_copy)() = nullptr) {
    if (terminal) return first_error;
    if (!open || !device || !host || bytes == 0 || bytes > capacity - at)
      return cudaErrorInvalidValue;
    auto *source = static_cast<unsigned char *>(arenas[current].host) + at;
    std::memcpy(source, host, bytes);
    // The caller owns callback serialization; generic pools default to no accounting.
    if (before_copy) before_copy();
    cudaError_t error = cudaMemcpyAsync(device, source, bytes, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) return fail(error);
    // at and capacity are multiples of alignment; checked room above makes
    // both this rounding and the next subtraction overflow-safe.
    at += ((bytes + alignment - 1) / alignment) * alignment;
    dirty = true;
    return cudaSuccess;
  }
};

static fak_tokenomos_staging_pool g_tokenomos_staging;

static int fak_tokenomos_staging_error(cudaError_t error) {
  return 20000 + (int)g_tokenomos_staging.fail(error);
}

// Refusal never falls through to an unrecorded alternative after admission.
// Device zero and g_stream are the current native singleton's actual contract.
static int fak_tokenomos_staging_admit() {
  if (g_tokenomos_staging.terminal) return 20000 + (int)g_tokenomos_staging.first_error;
  if (!g_blas || !g_stream || g_capture_open || v41_candidates_stream_uncertain()) return -3;
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return fak_tokenomos_staging_error(error);
  if (device != 0) return -3;
  cudaStreamCaptureStatus capture;
  error = cudaStreamIsCapturing(g_stream, &capture);
  if (error != cudaSuccess) return fak_tokenomos_staging_error(error);
  if (capture != cudaStreamCaptureStatusNone) return -3;
  error = cudaPeekAtLastError();
  return error == cudaSuccess ? 0 : fak_tokenomos_staging_error(error);
}

static int fak_tokenomos_staging_copy(void *device, const void *host, size_t bytes, void (*before_copy)() = nullptr) {
  int admitted = fak_tokenomos_staging_admit();
  if (admitted != 0) return admitted;
  if (!device || !host || bytes == 0) return -1;
  if (bytes > fak_tokenomos_staging_pool::capacity) {
    // Oversize is decided before arena allocation/mutation. Keep the existing
    // synchronous route; this is not recovery from a selected-path failure.
    if (before_copy) before_copy();
    cudaError_t error = cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
    return error == cudaSuccess ? 0 : fak_tokenomos_staging_error(error);
  }
  cudaError_t error = g_tokenomos_staging.begin(g_stream);
  if (error == cudaSuccess) error = g_tokenomos_staging.put(device, host, bytes, before_copy);
  if (error == cudaSuccess) error = g_tokenomos_staging.seal();
  return error == cudaSuccess ? 0 : fak_tokenomos_staging_error(error);
}
