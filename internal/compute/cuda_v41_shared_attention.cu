// Included by cuda_kernels.cu, not a separate translation unit. Ref #13993.
//
// FlashMLA-class SIMT V4.1 shared-latent sink attention (one decode position):
// a split-KV online-softmax partial kernel, then an N-way base-2 LSE combine
// that folds the sink last. Numerics follow v41SparseMLADecodeReference
// (internal/compute/v41_sparse_mla_decode.go), never the serial three-pass
// contract. The combine stage is adapted from vLLM's merge_attn_states kernel
// (Apache-2.0, see third_party/vllm-merge-attn-states/NOTICE), generalized
// from two states to N splits and rewritten to the reference's exp2 order.
// Portable SIMT (sm80..sm120); no tensor-core or sm100-only instruction.
#ifdef FAK_CUDA_V41_SHARED_ATTENTION

#define V41SA_THREADS 256
#define V41SA_WARPS (V41SA_THREADS / 32)
#define V41SA_COMBINE_THREADS 128
#define V41SA_MAX_SPLITS 64

// Status word layout matches decodeV41SharedAttentionStatus: stage, slot+1,
// element+1, offending binary32 bits. The first claim per head wins; which of
// several concurrent faults wins is unspecified (this class is not BitExact).
__device__ static void v41sa_fault(unsigned *status, int head, unsigned stage,
    unsigned slot1, unsigned elem1, float v) {
  unsigned *w = status + (size_t)head * 4u;
  if (atomicCAS(&w[0], 0u, 0xffffffffu) == 0u) {
    w[1] = slot1;
    w[2] = elem1;
    w[3] = __float_as_uint(v);
    __threadfence();
    atomicExch(&w[0], stage);
  }
}

__device__ static inline bool v41sa_finite(float v) {
  return (__float_as_uint(v) & 0x7f800000u) != 0x7f800000u;
}

// grid = (splits, head groups). Dynamic shared memory holds the group's q
// rows, one KV tile, the group's value accumulators and the tile's scores.
// Each KV row is loaded once per tile for every head in the group.
__global__ void k_v41_shared_attention_split(const float *__restrict__ q,
    const float *__restrict__ kv, float *__restrict__ part_out,
    float *__restrict__ part_lse, unsigned *status, int rows, int heads,
    int dim, int hpb, int tile, int tiles, int splits, float scale2) {
  extern __shared__ float smem[];
  float *sq = smem;                       // [hpb][dim]
  float *skv = sq + (size_t)hpb * dim;    // [tile][dim]
  float *sacc = skv + (size_t)tile * dim; // [hpb][dim]
  float *ss = sacc + (size_t)hpb * dim;   // [hpb][tile]
  __shared__ float s_run[8], s_sum[8], s_alpha[8];
  const int split = blockIdx.x;
  const int h0 = blockIdx.y * hpb;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int lo = (int)((long long)split * tiles / splits * tile);
  const int hi = (int)min((long long)(split + 1) * tiles / splits * tile, (long long)rows);
  for (int i = tid; i < hpb * dim; i += V41SA_THREADS) {
    const int h = h0 + i / dim;
    sq[i] = h < heads ? q[(size_t)h * dim + i % dim] : 0.f;
    sacc[i] = 0.f;
  }
  if (tid < hpb) {
    s_run[tid] = -INFINITY;
    s_sum[tid] = 0.f;
  }
  for (int t = lo; t < hi; t += tile) {
    const int n = min(tile, hi - t);
    __syncthreads(); // previous tile's accumulate finished with skv/ss
    for (int i = tid; i < n * dim; i += V41SA_THREADS)
      skv[i] = kv[(size_t)t * dim + i];
    __syncthreads();
    for (int pair = warp; pair < hpb * n; pair += V41SA_WARPS) {
      const int hh = pair / n, r = pair % n;
      float dot = 0.f;
      for (int d = lane; d < dim; d += 32)
        dot = __fadd_rn(dot, __fmul_rn(sq[hh * dim + d], skv[r * dim + d]));
      for (int off = 16; off > 0; off >>= 1)
        dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffffu, dot, off));
      if (lane == 0) {
        const float v = __fmul_rn(dot, scale2);
        ss[hh * tile + r] = v;
        if (!v41sa_finite(v) && h0 + hh < heads)
          v41sa_fault(status, h0 + hh, 1u, (unsigned)(t + r) + 1u, 0u, v);
      }
    }
    __syncthreads();
    if (tid < hpb) {
      const float running = s_run[tid];
      float top = running;
      for (int r = 0; r < n; ++r) top = fmaxf(top, ss[tid * tile + r]);
      const float alpha = running == -INFINITY ? 0.f : exp2f(__fsub_rn(running, top));
      float sum = __fmul_rn(s_sum[tid], alpha);
      for (int r = 0; r < n; ++r) {
        const float p = exp2f(__fsub_rn(ss[tid * tile + r], top));
        ss[tid * tile + r] = p;
        sum = __fadd_rn(sum, p);
      }
      s_run[tid] = top;
      s_sum[tid] = sum;
      s_alpha[tid] = alpha;
      if (!v41sa_finite(sum) && h0 + tid < heads)
        v41sa_fault(status, h0 + tid, 3u, 0u, 0u, sum);
    }
    __syncthreads();
    for (int i = tid; i < hpb * dim; i += V41SA_THREADS) {
      const int hh = i / dim, d = i % dim;
      float a = __fmul_rn(sacc[i], s_alpha[hh]);
      for (int r = 0; r < n; ++r)
        a = __fadd_rn(a, __fmul_rn(ss[hh * tile + r], skv[r * dim + d]));
      sacc[i] = a;
    }
  }
  __syncthreads();
  for (int i = tid; i < hpb * dim; i += V41SA_THREADS) {
    const int hh = i / dim, h = h0 + hh, d = i % dim;
    if (h >= heads) continue;
    // sum >= 1: the maximum score contributes exp2(0).
    const float v = __fdiv_rn(sacc[i], s_sum[hh]);
    if (!v41sa_finite(v)) v41sa_fault(status, h, 6u, 0u, (unsigned)d + 1u, v);
    part_out[((size_t)h * splits + split) * dim + d] = v;
  }
  if (tid < hpb && h0 + tid < heads)
    part_lse[(size_t)(h0 + tid) * splits + split] = __fadd_rn(s_run[tid], log2f(s_sum[tid]));
}

// grid = heads. N-way generalization of vLLM merge_attn_states: the merged LSE
// is top + log2(sum exp2(lse_s - top)); each partial is weighted by
// exp2(lse_s - lse), and the sink rescales by 1/(1+exp2(sink*log2e - lse)).
__global__ void k_v41_shared_attention_combine(const float *__restrict__ part_out,
    const float *__restrict__ part_lse, const float *__restrict__ sink,
    float *__restrict__ out, unsigned *status, int heads, int dim, int splits,
    int has_sink) {
  const int h = blockIdx.x;
  const float *lse_h = part_lse + (size_t)h * splits;
  float top = lse_h[0];
  for (int s = 1; s < splits; ++s) top = fmaxf(top, lse_h[s]);
  float sum = 0.f;
  for (int s = 0; s < splits; ++s) sum = __fadd_rn(sum, exp2f(__fsub_rn(lse_h[s], top)));
  const float lse = __fadd_rn(top, log2f(sum));
  float factor = 1.f;
  if (has_sink) {
    const float gap = __fsub_rn(__fmul_rn(sink[h], 1.4426950408889634f), lse);
    factor = __frcp_rn(__fadd_rn(1.f, exp2f(gap)));
  }
  if (threadIdx.x == 0) {
    if (!v41sa_finite(lse)) v41sa_fault(status, h, 3u, 0u, 0u, lse);
    else if (!v41sa_finite(factor)) v41sa_fault(status, h, 3u, 0u, 0u, factor);
  }
  for (int d = threadIdx.x; d < dim; d += V41SA_COMBINE_THREADS) {
    float o = 0.f;
    for (int s = 0; s < splits; ++s) {
      const float w = exp2f(__fsub_rn(lse_h[s], lse));
      o = __fadd_rn(o, __fmul_rn(w, part_out[((size_t)h * splits + s) * dim + d]));
    }
    if (has_sink) o = __fmul_rn(o, factor);
    if (!v41sa_finite(o)) v41sa_fault(status, h, 6u, 0u, (unsigned)d + 1u, o);
    out[(size_t)h * dim + d] = o;
  }
}

// Grow-only native scratch: partial outputs/LSEs and the device status words.
static float *g_v41_sa_part = nullptr;
static size_t g_v41_sa_part_bytes = 0;
static unsigned *g_v41_sa_status = nullptr;
static size_t g_v41_sa_status_bytes = 0;
static cudaEvent_t g_v41_sa_ev[2] = {nullptr, nullptr};

extern "C" int fak_v41_shared_attention_state(void) {
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
  return 0;
}

static int v41_sa_owned(const void *p, uint64_t count) {
  if (!p || count == 0 || count > INT_MAX || count > SIZE_MAX / sizeof(float)) return -2;
  auto found = g_live.find(const_cast<void *>(p));
  if (found == g_live.end() || found->second < (size_t)count * sizeof(float)) return -2;
  cudaPointerAttributes attributes;
  cudaError_t error = cudaPointerGetAttributes(&attributes, p);
  if (error != cudaSuccess) return 11000 + (int)error;
  return attributes.type == cudaMemoryTypeDevice && attributes.device == 0 ? 0 : -2;
}

// Chooses heads-per-block and tile rows that fit the opt-in shared budget, then
// the split count. force_splits > 0 pins the split count (tests, sweeps).
static int v41_sa_plan(int rows, int heads, int dim, int force_splits,
    int *hpb_out, int *tile_out, int *splits_out, size_t *smem_out) {
  int device = 0, optin = 0, sms = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return -3;
  if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess) return -3;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) return -3;
  const size_t budget = (size_t)optin - 3 * 8 * sizeof(float) - 256;
  int hpb = heads < 8 ? heads : 8;
  for (; hpb >= 1; hpb = hpb == 1 ? 0 : hpb / 2) {
    for (int tile = 16; tile >= 1; tile /= 2) {
      const size_t floats = (size_t)2 * hpb * dim + (size_t)tile * dim + (size_t)hpb * tile;
      if (floats * sizeof(float) <= budget) {
        const int tiles = (rows + tile - 1) / tile;
        const int groups = (heads + hpb - 1) / hpb;
        int splits = force_splits > 0 ? force_splits : (2 * sms + groups - 1) / groups;
        if (splits > V41SA_MAX_SPLITS) splits = V41SA_MAX_SPLITS;
        if (splits > tiles) splits = tiles;
        if (splits < 1) splits = 1;
        *hpb_out = hpb;
        *tile_out = tile;
        *splits_out = splits;
        *smem_out = floats * sizeof(float);
        return 0;
      }
    }
  }
  return -1; // headDim too wide for this candidate's shared-memory envelope
}

static int v41_sa_grow(void **p, size_t *have, size_t need) {
  if (*have >= need) return 0;
  void *next = nullptr;
  cudaError_t error = fcuda_alloc_status(cudaMalloc(&next, need));
  if (error != cudaSuccess) return -1;
  fak_alloc_track(next, FAK_ALLOC_ACTIVATIONS_SCRATCH, FAK_ALLOC_DEVICE_LOCAL,
                  FAK_ALLOC_SITE("v41-shared-attention-scratch"), need);
  if (*p) {
    fak_alloc_untrack(*p);
    cudaFree(*p);
  }
  *p = next;
  *have = need;
  return 0;
}

static int v41_sa_drain(int status, const float *q, const float *kv,
    const float *sink, float *out) {
  cudaError_t drained = cudaStreamSynchronize(g_stream);
  if (drained != cudaSuccess) {
    // Admission closes permanently; scratch and caller buffers are never reused.
    g_v41_sa_quarantine[0] = q;
    g_v41_sa_quarantine[1] = kv;
    g_v41_sa_quarantine[2] = sink;
    g_v41_sa_quarantine[3] = out;
    g_v41_sa_uncertain = true;
    return 50000 + (int)drained;
  }
  return status;
}

extern "C" int fak_v41_shared_attention_f32(const float *q, const float *kv,
    const float *sink, float *out, unsigned *host_status, int heads, int dim,
    int rows, int mode, int has_sink, float scale, int force_splits,
    int *used_tile, int *used_splits, float *kernel_ms) {
  (void)mode; // Plain/Compressed differ only for all-sentinel scores (see Go doc).
  if (rows <= 0 || heads <= 0 || dim <= 0 || !host_status) return -1;
  const uint64_t qCount = (uint64_t)heads * dim, kCount = (uint64_t)rows * dim;
  if (qCount > INT_MAX || kCount > INT_MAX) return -1;
  int state = fak_v41_shared_attention_state();
  if (state != 0) return state;
  if (out == q || out == kv || out == sink) return -2;
  const void *pointers[] = {q, kv, sink, out};
  const uint64_t counts[] = {qCount, kCount, has_sink ? (uint64_t)heads : 1u, qCount};
  for (int i = 0; i < 4; ++i) {
    int owned = v41_sa_owned(pointers[i], counts[i]);
    if (owned != 0) return owned;
  }
  int hpb = 0, tile = 0, splits = 0;
  size_t smem = 0;
  if (v41_sa_plan(rows, heads, dim, force_splits, &hpb, &tile, &splits, &smem) != 0) return 2;
  const int tiles = (rows + tile - 1) / tile;
  const size_t partBytes = (size_t)heads * splits * ((size_t)dim + 1) * sizeof(float);
  const size_t statusBytes = (size_t)heads * 4 * sizeof(unsigned);
  if (v41_sa_grow((void **)&g_v41_sa_part, &g_v41_sa_part_bytes, partBytes) != 0 ||
      v41_sa_grow((void **)&g_v41_sa_status, &g_v41_sa_status_bytes, statusBytes) != 0)
    return -1;
  if (!g_v41_sa_ev[0] && (cudaEventCreate(&g_v41_sa_ev[0]) != cudaSuccess ||
                          cudaEventCreate(&g_v41_sa_ev[1]) != cudaSuccess))
    return -3;
  cudaError_t error = cudaFuncSetAttribute(k_v41_shared_attention_split,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (error != cudaSuccess) return 30000 + (int)error;
  float *partOut = g_v41_sa_part;
  float *partLSE = g_v41_sa_part + (size_t)heads * splits * dim;
  const float scale2 = scale * 1.4426950408889634f; // one rounded product, as the reference
  FCUDA_TS("fill", 1, error = cudaMemsetAsync(g_v41_sa_status, 0, statusBytes, g_stream));
  if (error != cudaSuccess) return v41_sa_drain(30000 + (int)error, q, kv, sink, out);
  cudaEventRecord(g_v41_sa_ev[0], g_stream);
  const dim3 grid((unsigned)splits, (unsigned)((heads + hpb - 1) / hpb));
  FCUDA_TS("k_v41_shared_attention_split", grid,
    k_v41_shared_attention_split<<<grid, V41SA_THREADS, smem, g_stream>>>(q, kv, partOut,
      partLSE, g_v41_sa_status, rows, heads, dim, hpb, tile, tiles, splits, scale2));
  error = cudaGetLastError();
  if (error != cudaSuccess) return v41_sa_drain(30000 + (int)error, q, kv, sink, out);
  FCUDA_TS("k_v41_shared_attention_combine", heads,
    k_v41_shared_attention_combine<<<heads, V41SA_COMBINE_THREADS, 0, g_stream>>>(partOut,
      partLSE, sink, out, g_v41_sa_status, heads, dim, splits, has_sink));
  error = cudaGetLastError();
  if (error != cudaSuccess) return v41_sa_drain(30000 + (int)error, q, kv, sink, out);
  cudaEventRecord(g_v41_sa_ev[1], g_stream);
  error = cudaStreamSynchronize(g_stream);
  if (error != cudaSuccess) return v41_sa_drain(40000 + (int)error, q, kv, sink, out);
  // The shared checked D2H ABI owns copy completion and byte accounting.
  const int copied = fcuda_d2h_checked(host_status, g_v41_sa_status, statusBytes);
  if (copied != 0) return v41_sa_drain(59000 + copied, q, kv, sink, out);
  if (kernel_ms) {
    float ms = 0.f;
    if (cudaEventElapsedTime(&ms, g_v41_sa_ev[0], g_v41_sa_ev[1]) == cudaSuccess) *kernel_ms = ms;
  }
  if (used_tile) *used_tile = tile;
  if (used_splits) *used_splits = splits;
  ++g_v41_sa_completed;
  return 0;
}

extern "C" uint64_t fak_v41_shared_attention_completed(void) { return g_v41_sa_completed; }
#endif
