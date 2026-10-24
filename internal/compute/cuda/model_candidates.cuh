// Private whole-sequence bindings. They deliberately do not extend cuda_backend.h:
// the public flat ABI remains the stable primitive surface, while cuda_kernels.go
// owns these cgo-local entry points and the complete resident-sequence contract.
__global__ void k_qwen35_embedding_gather(const float *embedding, const int *ids,
                                           float *out, int tokens, int hidden) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)tokens * hidden;
  if (i >= n) return;
  int token = (int)(i / hidden);
  int col = (int)(i - (size_t)token * hidden);
  out[i] = embedding[(size_t)ids[token] * hidden + col];
}

extern "C" int fak_qwen35_embedding_gather_f32(
    const float *dEmbedding, const int *hIDs, int *dIDs, float *dOut,
    int tokens, int hidden) {
  if (!dEmbedding || !hIDs || !dIDs || !dOut || tokens <= 0 || hidden <= 0) return -1;
  cudaGetLastError();
  size_t idsBytes = (size_t)tokens * sizeof(int);
  cudaError_t copy = cudaMemcpy(dIDs, hIDs, idsBytes, cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) return 10000 + (int)copy;
  g_h2d_bytes += idsBytes;
  size_t n = (size_t)tokens * hidden;
  k_qwen35_embedding_gather<<<(n + 255) / 256, 256, 0, g_stream>>>(
      dEmbedding, dIDs, dOut, tokens, hidden);
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 20000 + (int)launch;
}

extern "C" int fak_qwen35_pointer_is_device(const void *pointer) {
  if (!pointer) return 0;
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return attributes.type == cudaMemoryTypeDevice ? 1 : 0;
}

#ifdef FAK_CUDA_V41_BF16_PUBLISH
// Candidate-only ordinary V4.1 BF16 publication. No floating-point operation is
// permitted here: preserving signed zero, subnormals and exact RNE ties is the
// contract, independently of compiler fast-math flags or device FTZ settings.
static_assert(sizeof(uint32_t) == 4 && sizeof(unsigned int) == 4,
              "V4.1 BF16 status requires four 32-bit words");

static bool v41_bf16_publish_site(uint32_t site) {
  return site == 1u || site == 2u || (site >= 5u && site <= 27u);
}

static bool v41_bf16_publish_owned(const void *pointer, size_t bytes) {
  auto found = g_live.find(const_cast<void *>(pointer));
  if (!pointer || found == g_live.end() || found->second < bytes) return false;
  cudaPointerAttributes attributes;
  cudaError_t error = cudaPointerGetAttributes(&attributes, pointer);
  return error == cudaSuccess && attributes.type == cudaMemoryTypeDevice && attributes.device == 0;
}

extern "C" int fak_v41_bf16_publish_state(void) {
  if (!g_blas || !g_stream || g_capture_open || g_v41_bf16_uncertain) return -3;
  if (v41_candidates_stream_uncertain()) return -3;
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return 10000 + (int)error;
  if (device != 0) return -3;
  cudaStreamCaptureStatus capture;
  error = cudaStreamIsCapturing(g_stream, &capture);
  if (error != cudaSuccess) return 10000 + (int)error;
  if (capture != cudaStreamCaptureStatusNone) return -3;
  error = cudaPeekAtLastError();
  return error == cudaSuccess ? 0 : 10000 + (int)error;
}

extern "C" int fak_v41_bf16_publish_input(const uint32_t *dInput, int count) {
  if (count <= 0 || (size_t)count > SIZE_MAX / sizeof(uint32_t)) return -1;
  return v41_bf16_publish_owned(dInput, (size_t)count * sizeof(uint32_t)) ? 0 : -2;
}

__global__ void k_v41_bf16_publish_status(uint32_t *status, uint32_t count, uint32_t site) {
  status[0] = 0x42463143u;
  status[1] = site;
  // status[2] retains the fresh all-ones success sentinel until atomicMin.
  status[3] = count;
}

__global__ void k_v41_bf16_publish(const uint32_t *input, uint32_t *output,
                                    uint32_t *status, uint32_t count) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count; i += blockDim.x * gridDim.x) {
    const uint32_t u = input[i];
    const uint32_t rounded = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    if ((u & 0x7f800000u) == 0x7f800000u ||
        (rounded & 0x7f800000u) == 0x7f800000u) {
      atomicMin((unsigned int *)&status[2], (unsigned int)i);
    } else {
      output[i] = rounded;
    }
  }
}

static int v41_bf16_publish_drain(int primary, const uint32_t *input,
                                  uint32_t *output, uint32_t *status) {
  if (cudaStreamSynchronize(g_stream) != cudaSuccess) {
    g_v41_bf16_quarantine[0] = input;
    g_v41_bf16_quarantine[1] = output;
    g_v41_bf16_quarantine[2] = status;
    g_v41_bf16_uncertain = true;
  }
  return primary;
}

extern "C" int fak_v41_bf16_publish_f32(
    const uint32_t *dInput, uint32_t *dOutput, uint32_t *dStatus,
    void *hStatus, int count, uint32_t site) {
  if (!hStatus || !v41_bf16_publish_site(site) || count <= 0 ||
      (size_t)count > SIZE_MAX / sizeof(uint32_t)) return -1;
  if (dInput == dOutput || dInput == dStatus || dOutput == dStatus) return -2;
  if (!v41_bf16_publish_owned(dInput, (size_t)count * sizeof(uint32_t)) ||
      !v41_bf16_publish_owned(dOutput, (size_t)count * sizeof(uint32_t)) ||
      !v41_bf16_publish_owned(dStatus, 16)) return -2;
  int state = fak_v41_bf16_publish_state();
  if (state != 0) return state;

  // Freshness is proved by checked initialization on every call, followed by
  // FIFO header/rounding launches and a checked fence. A cached success record
  // or a status decoder on its own cannot establish this property.
  cudaError_t error = cudaMemsetAsync(dStatus, 0xff, 16, g_stream);
  if (error != cudaSuccess)
    return v41_bf16_publish_drain(20000 + (int)error, dInput, dOutput, dStatus);
  k_v41_bf16_publish_status<<<1, 1, 0, g_stream>>>(dStatus, (uint32_t)count, site);
  error = cudaGetLastError();
  if (error != cudaSuccess)
    return v41_bf16_publish_drain(30000 + (int)error, dInput, dOutput, dStatus);
  unsigned int blocks = ((unsigned int)count + 255u) / 256u;
  if (blocks > 65535u) blocks = 65535u;
  k_v41_bf16_publish<<<blocks, 256, 0, g_stream>>>(dInput, dOutput, dStatus, (uint32_t)count);
  error = cudaGetLastError();
  if (error != cudaSuccess)
    return v41_bf16_publish_drain(40000 + (int)error, dInput, dOutput, dStatus);
  error = cudaStreamSynchronize(g_stream);
  if (error != cudaSuccess)
    return v41_bf16_publish_drain(50000 + (int)error, dInput, dOutput, dStatus);
  // Synchronous D2H after the successful fence cannot retain the caller's Go
  // status array. Only the 16-byte envelope is transferred, even on arithmetic
  // failure; output publication is decided by the Go decoder after this call.
  error = cudaMemcpy(hStatus, dStatus, 16, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
    return v41_bf16_publish_drain(60000 + (int)error, dInput, dOutput, dStatus);
  g_host_bytes += 16;
  return 0;
}
#endif

#ifdef FAK_CUDA_V41_TAIL_ROPE
// Ordinary candidate only. Exact arithmetic qualification remains false until
// the actual module, compiler FTZ settings and selected device are witnessed.
static_assert(sizeof(uint32_t) == 4, "V4.1 tail RoPE requires binary32 words");

extern "C" int fak_v41_tail_rope_state(void) {
  if (!g_blas || !g_stream || g_capture_open || v41_candidates_stream_uncertain()) return -3;
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) return 10000 + (int)error;
  int device = -1;
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) return 10000 + (int)error;
  if (device != 0) return -3;
  cudaStreamCaptureStatus capture;
  error = cudaStreamIsCapturing(g_stream, &capture);
  if (error != cudaSuccess) return 10000 + (int)error;
  if (capture != cudaStreamCaptureStatusNone) return -3;
  // This bounded candidate cannot quarantine pointers from arbitrary previous
  // submissions. Admit only an idle stream, never drain unknown work as if it
  // belonged to this operation. cudaMu excludes concurrent backend enqueues.
  error = cudaStreamQuery(g_stream);
  if (error == cudaErrorNotReady) return -3;
  return error == cudaSuccess ? 0 : 10000 + (int)error;
}

// Exact allocation-base membership plus runtime device identity; no unowned
// subview, forged capacity, managed allocation or foreign device is admitted.
static int v41_tail_rope_owned(const void *pointer, uint64_t count) {
  if (!pointer || count == 0 || count > UINT32_MAX || count > SIZE_MAX / sizeof(uint32_t)) return -2;
  auto found = g_live.find(const_cast<void *>(pointer));
  if (found == g_live.end() || found->second < (size_t)count * sizeof(uint32_t)) return -2;
  cudaPointerAttributes attributes;
  cudaError_t error = cudaPointerGetAttributes(&attributes, pointer);
  if (error != cudaSuccess) return 11000 + (int)error;
  return attributes.type == cudaMemoryTypeDevice && attributes.device == 0 ? 0 : -2;
}

extern "C" int fak_v41_tail_rope_input(const uint32_t *dInput, uint64_t count) {
  int state = fak_v41_tail_rope_state();
  return state == 0 ? v41_tail_rope_owned(dInput, count) : state;
}

static bool v41_tail_rope_geometry(int rows, int headDim, int rotaryDim, uint64_t *qWords) {
  if (rows <= 0 || headDim <= 0 || rotaryDim <= 0 || rotaryDim > headDim || rotaryDim % 2 != 0) return false;
  *qWords = (uint64_t)rows * (uint64_t)headDim;
  // Retain the common Go validator's combined-index domain even for Rows.
  return *qWords + (uint64_t)headDim <= UINT32_MAX &&
         *qWords <= SIZE_MAX / sizeof(uint32_t) &&
         (uint64_t)headDim <= SIZE_MAX / sizeof(uint32_t);
}

__global__ void k_v41_tail_rope(const uint32_t *q, const uint32_t *kv,
    const uint32_t *table, uint32_t *qOut, uint32_t *kvOut,
    uint64_t qWords, uint64_t totalWords, uint32_t headDim, uint32_t rotaryDim) {
  const uint32_t tail = headDim - rotaryDim;
  // The admitted count can approach UINT32_MAX: widening the stride and its
  // addition prevents wraparound even on the final out-of-range iteration.
  for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < totalWords; i += (uint64_t)blockDim.x * gridDim.x) {
    const bool isQ = i < qWords;
    const uint64_t local = isQ ? i : i - qWords;
    const uint32_t lane = (uint32_t)(local % headDim);
    const uint32_t *input = isQ ? q : kv;
    uint32_t *output = isQ ? qOut : kvOut;
    if (lane < tail) {
      output[local] = input[local]; // Untouched prefix is a uint32 copy.
    } else if ((lane - tail) % 2u == 0u) {
      const uint32_t j = lane - tail;
      const float a = __uint_as_float(input[local]);
      const float b = __uint_as_float(input[local + 1u]);
      const float s = __uint_as_float(table[j]);
      const float c = __uint_as_float(table[j + 1u]);
      const float ac = __fmul_rn(a, c);
      const float bs = __fmul_rn(b, s);
      const float bc = __fmul_rn(b, c);
      const float as = __fmul_rn(a, s);
      output[local] = __float_as_uint(__fsub_rn(ac, bs));
      output[local + 1u] = __float_as_uint(__fadd_rn(bc, as));
    }
  }
}

static int v41_tail_rope_drain(int primary, const uint32_t *q, const uint32_t *kv,
    const uint32_t *table, uint32_t *qOut, uint32_t *kvOut, bool forceUncertain) {
  cudaError_t drained = cudaStreamSynchronize(g_stream);
  if (drained != cudaSuccess || forceUncertain) {
    // Fixed slots are sufficient for QK; nulls/aliases need no additional slot.
    // State admission forbids a second operation from replacing these bases.
    g_v41_rope_quarantine[0] = q;
    g_v41_rope_quarantine[1] = kv;
    g_v41_rope_quarantine[2] = table;
    g_v41_rope_quarantine[3] = qOut;
    g_v41_rope_quarantine[4] = kvOut;
    g_v41_rope_uncertain = true;
  }
  return primary;
}

static int v41_tail_rope_dispatch(const uint32_t *dQ, const uint32_t *dKV,
    const uint32_t *dTable, uint32_t *dQOut, uint32_t *dKVOut,
    int rows, int headDim, int rotaryDim, bool paired) {
  uint64_t qWords = 0;
  if (!v41_tail_rope_geometry(rows, headDim, rotaryDim, &qWords)) return -1;
  int state = fak_v41_tail_rope_state();
  if (state != 0) return state;
  if (!dQOut || (paired && (!dKVOut || dQOut == dKVOut))) return -2;
  if (dQOut == dQ || dQOut == dTable || (paired && dQOut == dKV) ||
      (paired && (dKVOut == dQ || dKVOut == dKV || dKVOut == dTable))) return -2;
  const void *pointers[5] = {dQ, dKV, dTable, dQOut, dKVOut};
  const uint64_t counts[5] = {qWords, (uint64_t)headDim, (uint64_t)rotaryDim, qWords, (uint64_t)headDim};
  for (int i = 0; i < 5; ++i) {
    if (!paired && (i == 1 || i == 4)) continue;
    int owned = v41_tail_rope_owned(pointers[i], counts[i]);
    if (owned != 0) return owned;
  }
  int fault = 0;
#ifdef FAK_CUDA_V41_TAIL_ROPE_TEST
  if (g_v41_rope_test_fault != 4) {
    fault = g_v41_rope_test_fault;
    g_v41_rope_test_fault = 0;
  }
#endif
  if (fault == 1) return -9; // Proven pre-enqueue refusal.
  const uint64_t totalWords = qWords + (paired ? (uint64_t)headDim : 0u);
  unsigned int blocks = (unsigned int)((totalWords + 255u) / 256u);
  if (blocks > 65535u) blocks = 65535u;
  k_v41_tail_rope<<<blocks, 256, 0, g_stream>>>(dQ, dKV, dTable, dQOut, dKVOut,
      qWords, totalWords, (uint32_t)headDim, (uint32_t)rotaryDim);
  // State admission inspected, without consuming, the old error. This get is
  // now attributable to this launch, not a way to erase an unknown prior error.
  cudaError_t error = cudaGetLastError();
  if (error == cudaSuccess) ++g_v41_rope_launches;
  if (error == cudaSuccess && (fault == 2 || fault == 5)) error = cudaErrorLaunchFailure;
  if (error != cudaSuccess)
    return v41_tail_rope_drain(20000 + (int)error, dQ, dKV, dTable, dQOut, dKVOut, fault == 5);
  error = cudaStreamSynchronize(g_stream);
  if (error == cudaSuccess && fault == 3) error = cudaErrorUnknown;
  if (error != cudaSuccess)
    return v41_tail_rope_drain(30000 + (int)error, dQ, dKV, dTable, dQOut, dKVOut, false);
  ++g_v41_rope_operations;
  return 0;
}

extern "C" int fak_v41_tail_rope_qk(const uint32_t *dQ, const uint32_t *dKV,
    const uint32_t *dTable, uint32_t *dQOut, uint32_t *dKVOut,
    int rows, int headDim, int rotaryDim) {
  return v41_tail_rope_dispatch(dQ, dKV, dTable, dQOut, dKVOut, rows, headDim, rotaryDim, true);
}

extern "C" int fak_v41_tail_rope_rows(const uint32_t *dQ, const uint32_t *dTable,
    uint32_t *dQOut, int rows, int headDim, int rotaryDim) {
  return v41_tail_rope_dispatch(dQ, nullptr, dTable, dQOut, nullptr, rows, headDim, rotaryDim, false);
}

extern "C" int fak_v41_tail_rope_readback(void *hOutput, const uint32_t *dInput, uint64_t count) {
  if (!hOutput) return -1;
  int state = fak_v41_tail_rope_input(dInput, count);
  if (state != 0) return state;
  cudaError_t error = cudaStreamSynchronize(g_stream);
  if (error != cudaSuccess)
    return v41_tail_rope_drain(40000 + (int)error, dInput, nullptr, nullptr, nullptr, nullptr, false);
#ifdef FAK_CUDA_V41_TAIL_ROPE_TEST
  if (g_v41_rope_test_fault == 4) {
    g_v41_rope_test_fault = 0;
    return 50000 + (int)cudaErrorUnknown;
  }
#endif
  // Synchronous D2H never retains the caller's Go slice across this ABI call.
  error = cudaMemcpy(hOutput, dInput, (size_t)count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
    return v41_tail_rope_drain(50000 + (int)error, dInput, nullptr, nullptr, nullptr, nullptr, false);
  g_host_bytes += (size_t)count * sizeof(uint32_t);
  return 0;
}

extern "C" uint64_t fak_v41_tail_rope_counter(int kind) {
  switch (kind) {
    case 0: return g_v41_rope_operations;
    case 1: return g_v41_rope_launches;
    case 2: return g_h2d_bytes;
    case 3: return g_host_bytes;
    case 4: {
      uint64_t count = 0;
      for (int i = 0; i < 5; ++i) {
        if (!g_v41_rope_quarantine[i]) continue;
        bool duplicate = false;
        for (int j = 0; j < i; ++j) duplicate |= g_v41_rope_quarantine[i] == g_v41_rope_quarantine[j];
        if (!duplicate) ++count;
      }
      return count;
    }
    case 5: return v41_candidates_stream_uncertain() ? 1 : 0;
    case 6: return g_live.size();
    default: return UINT64_MAX;
  }
}

extern "C" int fak_v41_tail_rope_test_fault(int stage) {
#ifdef FAK_CUDA_V41_TAIL_ROPE_TEST
  if (stage < 0 || stage > 5 || v41_candidates_stream_uncertain()) return -1;
  g_v41_rope_test_fault = stage;
  return 0;
#else
  return -8;
#endif
}

extern "C" int fak_v41_tail_rope_test_free(int onDevice) {
#ifdef FAK_CUDA_V41_TAIL_ROPE_TEST
  // Probe all held exact bases, including withheld outputs; never arbitrary
  // pointers. Each must stay live after the pool and direct CUDA-free routes.
  if (!g_v41_rope_uncertain) return -1;
  for (const void *pointer : g_v41_rope_quarantine) {
    if (!pointer) continue;
    void *base = const_cast<void *>(pointer);
    if (g_live.find(base) == g_live.end()) return -1;
    const uint64_t attempts = g_v41_rope_test_free_attempts;
    if (onDevice) fcuda_free_on(0, base);
    else fcuda_free(base);
    // free_on deliberately does not edit g_live. Membership alone cannot
    // distinguish its guard from a real direct free followed by a stale map.
    if (g_v41_rope_test_free_attempts != attempts) return -3;
    if (g_live.find(base) == g_live.end()) return -2;
  }
  return 0;
#else
  return -8;
#endif
}

extern "C" int fak_v41_tail_rope_bf16_state(void) {
#ifdef FAK_CUDA_V41_BF16_PUBLISH
  return fak_v41_bf16_publish_state();
#else
  return -8;
#endif
}
#endif
