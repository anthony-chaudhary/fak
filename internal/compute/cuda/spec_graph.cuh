// ---- Split-KV Multi-Query Speculative Verify Attention (#11100) -------------------
// When speculative decoding verifies qLen draft tokens (e.g. 4, 8, 16), FA2 assigns only
// one thread block per query head, leaving GPU SMs underutilized.
// k_spec_verify_attention tiles queries across BLOCK_M with KV sequence splitting across
// NUM_SEGMENTS thread blocks, producing partial online-softmax statistics.
// k_spec_verify_combine then merges these partial reductions across segments.
#define SPEC_VERIFY_BLOCK_M 4
#define SPEC_VERIFY_NUM_SEGMENTS 8
#define BLOCK_M SPEC_VERIFY_BLOCK_M
#define NUM_SEGMENTS SPEC_VERIFY_NUM_SEGMENTS

__global__ void k_spec_verify_attention(
    const float *Q, const float *K, const float *V,
    float *partial_out, float *partial_max, float *partial_sum,
    int qLen, int kvLen, int nH, int nHkv, int hd, float scale) {
  int m_block = blockIdx.x;
  int h = blockIdx.y;
  int seg_idx = blockIdx.z;

  if (h >= nH || seg_idx >= NUM_SEGMENTS) return;

  int grp = nH / nHkv;
  int kvh = h / grp;
  int w = nHkv * hd;
  int tid = threadIdx.x;

  extern __shared__ float smem[];
  float *qs = smem;
  float *red = smem + hd;

  int seg_len = (kvLen + NUM_SEGMENTS - 1) / NUM_SEGMENTS;
  int kv_start = seg_idx * seg_len;
  int kv_end = kv_start + seg_len;
  if (kv_end > kvLen) kv_end = kvLen;

  int prefix = kvLen - qLen;

  for (int m = 0; m < BLOCK_M; m++) {
    int qi = m_block * BLOCK_M + m;
    if (qi >= qLen) break;

    const float *qh = Q + ((size_t)qi * nH + h) * hd;
    for (int d = tid; d < hd; d += blockDim.x) {
      qs[d] = qh[d];
    }
    __syncthreads();

    // A finite sentinel is incorrect for valid scores below that sentinel: the
    // first online-softmax update would retain an empty normalization state.
    float m_stat = -INFINITY;
    float l_stat = 0.f;
    float acc[FLASH_ACC_MAX];
#pragma unroll
    for (int k = 0; k < FLASH_ACC_MAX; k++) acc[k] = 0.f;

    int max_attend = prefix + qi + 1;
    int cur_end = kv_end < max_attend ? kv_end : max_attend;

    if (kv_start < cur_end) {
      for (int j = kv_start; j < cur_end; j++) {
        const float *kj = K + (size_t)j * w + (size_t)kvh * hd;
        float partial = 0.f;
        for (int d = tid; d < hd; d += blockDim.x) {
          partial += qs[d] * kj[d];
        }
        red[tid] = partial;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
          if (tid < s) red[tid] += red[tid + s];
          __syncthreads();
        }
        float sc = red[0] * scale;
        __syncthreads();

        float mNew = fmaxf(m_stat, sc);
        float corr = expf(m_stat - mNew);
        float p = expf(sc - mNew);
        l_stat = l_stat * corr + p;

        const float *vj = V + (size_t)j * w + (size_t)kvh * hd;
        int k = 0;
        for (int d = tid; d < hd; d += blockDim.x, k++) {
          acc[k] = acc[k] * corr + p * vj[d];
        }
        m_stat = mNew;
      }
    }

    float invL = (l_stat > 0.f) ? (1.f / l_stat) : 0.f;
    size_t out_base = (((size_t)qi * nH + h) * NUM_SEGMENTS + seg_idx) * hd;
    int k = 0;
    for (int d = tid; d < hd; d += blockDim.x, k++) {
      partial_out[out_base + d] = acc[k] * invL;
    }

    if (tid == 0) {
      size_t stat_idx = ((size_t)qi * nH + h) * NUM_SEGMENTS + seg_idx;
      partial_max[stat_idx] = m_stat;
      partial_sum[stat_idx] = l_stat;
    }
    __syncthreads();
  }
}

__global__ void k_spec_verify_combine(
    const float *partial_out, const float *partial_max, const float *partial_sum,
    float *Out, int qLen, int nH, int hd) {
  int m_block = blockIdx.x;
  int h = blockIdx.y;
  if (h >= nH) return;

  int tid = threadIdx.x;
  __shared__ float s_max[NUM_SEGMENTS];
  __shared__ float s_sum[NUM_SEGMENTS];
  __shared__ float s_weights[NUM_SEGMENTS];
  __shared__ float s_global_sum;

  for (int m = 0; m < BLOCK_M; m++) {
    int qi = m_block * BLOCK_M + m;
    if (qi >= qLen) break;

    size_t stat_base = ((size_t)qi * nH + h) * NUM_SEGMENTS;
    if (tid < NUM_SEGMENTS) {
      s_max[tid] = partial_max[stat_base + tid];
      s_sum[tid] = partial_sum[stat_base + tid];
    }
    __syncthreads();

    if (tid == 0) {
      float g_max = -INFINITY;
      for (int s = 0; s < NUM_SEGMENTS; s++) {
        if (s_sum[s] > 0.f && s_max[s] > g_max) {
          g_max = s_max[s];
        }
      }
      float total_w = 0.f;
      for (int s = 0; s < NUM_SEGMENTS; s++) {
        if (s_sum[s] > 0.f) {
          float w = s_sum[s] * expf(s_max[s] - g_max);
          s_weights[s] = w;
          total_w += w;
        } else {
          s_weights[s] = 0.f;
        }
      }
      s_global_sum = total_w;
    }
    __syncthreads();

    float inv_w = (s_global_sum > 0.f) ? (1.f / s_global_sum) : 0.f;
    for (int d = tid; d < hd; d += blockDim.x) {
      float acc = 0.f;
      for (int s = 0; s < NUM_SEGMENTS; s++) {
        if (s_weights[s] > 0.f) {
          size_t p_offset = (((size_t)qi * nH + h) * NUM_SEGMENTS + s) * hd + d;
          acc += s_weights[s] * partial_out[p_offset];
        }
      }
      Out[((size_t)qi * nH + h) * hd + d] = acc * inv_w;
    }
    __syncthreads();
  }
}

extern "C" int fcuda_spec_verify_attention_f32(
    const float *dQ, const float *dK, const float *dV, float *dOut,
    int qLen, int kvLen, int nH, int nKV, int hd, float scale) {
  if (!dQ || !dK || !dV || !dOut) return -1;
  if (qLen <= 0 || kvLen <= 0 || kvLen < qLen) return -1;
  if (nH <= 0 || nKV <= 0 || (nH % nKV) != 0) return -1;
  if (hd <= 0 || hd > FLASH_ACC_MAX * FLASH_THREADS) return -1;
  if (scale <= 0.f || !isfinite(scale)) {
    scale = 1.0f / sqrtf((float)hd);
  }

  size_t partial_elements = (size_t)qLen * nH * NUM_SEGMENTS * hd;
  size_t stats_elements = (size_t)qLen * nH * NUM_SEGMENTS;
  size_t total_scratch_bytes = (partial_elements + 2 * stats_elements) * sizeof(float);

  float *scratch = (float *)fcuda_malloc(total_scratch_bytes);
  if (!scratch) return -2;

  float *d_partial_out = scratch;
  float *d_partial_max = scratch + partial_elements;
  float *d_partial_sum = d_partial_max + stats_elements;

  dim3 grid_attn((qLen + BLOCK_M - 1) / BLOCK_M, nH, NUM_SEGMENTS);
  dim3 block_attn(FLASH_THREADS);
  size_t shmem_attn = ((size_t)hd + FLASH_THREADS) * sizeof(float);

  FCUDA_TS("k_spec_verify_attention", grid_attn, k_spec_verify_attention<<<grid_attn, block_attn, shmem_attn, g_stream>>>(
      dQ, dK, dV, d_partial_out, d_partial_max, d_partial_sum,
      qLen, kvLen, nH, nKV, hd, scale));

  dim3 grid_combine((qLen + BLOCK_M - 1) / BLOCK_M, nH);
  dim3 block_combine(FLASH_THREADS);

  FCUDA_TS("k_spec_verify_combine", grid_combine, k_spec_verify_combine<<<grid_combine, block_combine, 0, g_stream>>>(
      d_partial_out, d_partial_max, d_partial_sum,
      dOut, qLen, nH, hd));

  fcuda_free(scratch);
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? 0 : (int)err;
}

// Dedicated K<=32 branch-speculative attention primitive. Kept as an included
// translation-unit fragment because libfakcuda is intentionally built from the
// single cuda_kernels.cu object on every supported host.
#include "../tree_attention.cu"

// Routed-expert FP8 block-32 / MXFP4 E8M0 GEMV (RoutedExpertDeviceKernel), included for
// the same single-object reason.
#include "../routed_expert_gemv.cu"

// ---- GLM-MoE-DSA sparse attention over the host-selected key set ------------------
// model.glmDsaAttendCached's inner loop on the device. GLM-5.2's attention is SPARSE: a learned
// indexer picks the top-k keys a query attends, and the softmax(scale·q·k)·ΣwV runs over only
// that selected set. The selection (the f64 index scores + top-k) is computed HOST-side and the
// selected K/V rows are gathered contiguous, so this kernel attends exactly the same keys the
// host loop would — its only divergence is the f32 reduction order (Approx, cudaDsaSparseAttnCosineMin).
// Two things differ from k_flash_attention: (1) it streams the nSel GATHERED selected rows (per
// position the gather laid all nH heads contiguous: head h at i*nH*kd + h*kd for K, i*nH*vd + h*vd
// for V), not a contiguous causal window; (2) MLA's key width (kd = qkNope+qkRope) and value width
// (vd) DIFFER, so it carries both instead of one hd. Same online-softmax form (running max/sum/acc),
// so no scores[nSel] row is ever materialized. One block per query head; FLASH_THREADS threads split
// the dims; the only scratch is per-block shared memory (the query row + a reduction row).
__global__ void k_dsa_sparse_attend(const float *Q, const float *selK, const float *selV, float *Out,
                                    int nSel, int nH, int kd, int vd, float scale) {
  int h = blockIdx.x;
  if (h >= nH) return;
  const float *qh = Q + (size_t)h * kd;
  extern __shared__ float smem[];
  float *qs = smem;            // [kd]            : query row cached once in shared
  float *red = smem + kd;      // [FLASH_THREADS] : dot-product reduction staging
  int tid = threadIdx.x;
  for (int d = tid; d < kd; d += FLASH_THREADS) qs[d] = qh[d];
  __syncthreads();
  // online-softmax running state; acc[k] owns value dim d = tid + k*FLASH_THREADS, m/l replicated.
  float m = -INFINITY, l = 0.f;
  float acc[FLASH_ACC_MAX];
#pragma unroll
  for (int k = 0; k < FLASH_ACC_MAX; k++) acc[k] = 0.f;
  for (int i = 0; i < nSel; i++) {
    const float *ki = selK + (size_t)i * nH * kd + (size_t)h * kd;
    float partial = 0.f;
    for (int d = tid; d < kd; d += FLASH_THREADS) partial += qs[d] * ki[d];
    red[tid] = partial;
    __syncthreads();
    for (int s = FLASH_THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float sc = red[0] * scale;        // every thread reads the reduced dot
    __syncthreads();                  // WAR: finish all red[0] reads before the next key reuses red
    float mNew = fmaxf(m, sc);
    float corr = expf(m - mNew);      // 0 on the first key (m = -inf): clears the empty acc
    float p = expf(sc - mNew);
    l = l * corr + p;
    const float *vi = selV + (size_t)i * nH * vd + (size_t)h * vd;
    int k = 0;
    for (int d = tid; d < vd; d += FLASH_THREADS, k++) acc[k] = acc[k] * corr + p * vi[d];
    m = mNew;
  }
  float invL = l > 0.f ? 1.f / l : 0.f;
  int k = 0;
  for (int d = tid; d < vd; d += FLASH_THREADS, k++) Out[(size_t)h * vd + d] = acc[k] * invL;
}
// fcuda_dsa_sparse_attend_f32 launches the sparse-attend kernel: one block per query head,
// FLASH_THREADS threads, dynamic shared memory for the query row + the reduction row (sized on
// the KEY width kd). No per-call global scratch (the online form needs no nSel-sized buffer).
extern "C" void fcuda_dsa_sparse_attend_f32(const float *dQ, const float *dSelK, const float *dSelV,
                                            float *dOut, int nSel, int nH, int kd, int vd, float scale) {
  size_t shmem = ((size_t)kd + FLASH_THREADS) * sizeof(float);
  FCUDA_TS("k_dsa_sparse_attend", nH, k_dsa_sparse_attend<<<nH, FLASH_THREADS, shmem, g_stream>>>(dQ, dSelK, dSelV, dOut, nSel, nH, kd, vd, scale));
}

// ---- GLM-MoE-DSA learned-indexer score + top-k selection ---------------------------
// The LAST GLM-5.2 compute that was host-resident even after the dense projections and the
// sparse-attention compute moved to the kernel: the learned indexer that picks WHICH keys a query
// attends. For each cached key k (position k, valid iff k<=queryPos) and the one query, score(k) =
// Σ_h weights[h]·relu(scale·dot(indexQ_h, indexK_k)). The per-head dot is accumulated in DOUBLE so
// the device score equals the host f64 score bit-closely — the indexer drives a DISCRETE top-k, so
// it must be reduction-FAITHFUL (selection-stable), not just cosine-close like the f32 GEMM lanes. A
// single flipped selection would diverge the forward far past any cosine floor, so f64 here is what
// lets the device attend the SAME keys the host would and keeps the downstream witness argmax-exact.
//
// k_dsa_index_score: one block per key, blockDim threads stride the (head,dim) work; the per-key
// score lands in dScores[k]. Masked keys (k>queryPos) get -inf so the top-k never picks them.
__global__ void k_dsa_index_score(const float *indexQ, const float *indexK, const float *weights,
                                  double *dScores, int nKeys, int nH, int indexDim, int queryPos,
                                  float scale) {
  int k = blockIdx.x;
  if (k >= nKeys) return;
  if (k > queryPos) { if (threadIdx.x == 0) dScores[k] = -1e300; return; }
  const float *key = indexK + (size_t)k * indexDim;
  __shared__ double red[256];
  double acc = 0.0;
  // Each thread sums a slice of the (head,dim) grid: head h, dim d, contributing
  //   weights[h]·relu(scale·Σ_d q·k) — but relu is per-HEAD, so accumulate per-head dots first.
  // With nH and indexDim both small, one thread owns whole heads (stride blockDim over heads).
  for (int h = threadIdx.x; h < nH; h += blockDim.x) {
    const float *qh = indexQ + (size_t)h * indexDim;
    double hd = 0.0;
    for (int d = 0; d < indexDim; d++) hd += (double)qh[d] * (double)key[d];
    double hs = hd * (double)scale;
    if (hs < 0.0) hs = 0.0;
    acc += (double)weights[h] * hs;
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) dScores[k] = red[0];
}

// k_dsa_index_topk: single block. Repeats a max-reduction topK times, each pass picking the highest
// remaining score (ties by LOWER position — the dsaTopKIndices order) and masking it out for the
// next pass. nKeys is small (one decode step's causal window) and topK tiny, so the O(topK·nKeys)
// host-equivalent selection is cheap on one block. Writes the selected positions to dSel[0..ret-1].
__global__ void k_dsa_index_topk(const double *dScores, int nKeys, int queryPos, int topK,
                                 int *dSel, int *dCount) {
  __shared__ double vbest[256];
  __shared__ int ibest[256];
  __shared__ char taken[4096]; // == DSA_TOPK_MAX_KEYS; the host wrapper declines nKeys past it, so
                               // every key the top-k sees here is maskable (no un-masked re-select tail).
  int nValid = (queryPos + 1 < nKeys) ? queryPos + 1 : nKeys;
  for (int i = threadIdx.x; i < nKeys && i < 4096; i += blockDim.x) taken[i] = 0;
  __syncthreads();
  int picked = 0;
  int want = topK < nValid ? topK : nValid;
  for (int pass = 0; pass < want; pass++) {
    double bv = -1e300; int bi = -1;
    for (int i = threadIdx.x; i < nValid; i += blockDim.x) {
      if (i < 4096 && taken[i]) continue;
      double v = dScores[i];
      if (bi < 0 || v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    vbest[threadIdx.x] = bv; ibest[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        double ov = vbest[threadIdx.x + s]; int oi = ibest[threadIdx.x + s];
        int cur = ibest[threadIdx.x];
        if (oi >= 0 && (cur < 0 || ov > vbest[threadIdx.x] || (ov == vbest[threadIdx.x] && oi < cur))) {
          vbest[threadIdx.x] = ov; ibest[threadIdx.x] = oi;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      int sel = ibest[0];
      dSel[picked] = sel;
      if (sel >= 0 && sel < 4096) taken[sel] = 1;
    }
    __syncthreads();
    picked++;
  }
  if (threadIdx.x == 0) *dCount = picked;
}

// DSA_TOPK_MAX_KEYS is the causal-window cap k_dsa_index_topk's shared `taken[]` mask can cover. A
// decode step's nKeys is the causal window (one position's history), well under this for every context
// the device path serves today; but to keep the boundary HONEST rather than silently degrading, the
// host wrapper refuses (sentinel -1 → caller falls back to the host f64 loop) when nKeys exceeds it,
// instead of running a topk whose un-maskable tail could re-select a position. Must match the literal
// shared-array size in k_dsa_index_topk.
#define DSA_TOPK_MAX_KEYS 4096

// fcuda_dsa_index_select_f32 scores all keys (k_dsa_index_score, f64-accumulated) then selects the
// top-k positions (k_dsa_index_topk) on the device, copying back only the small index list. The
// f64 score scratch is allocated internally (sized [nKeys] doubles) so the caller never sees the
// double dtype. Returns the number of positions written into host outIdx (= min(topK, #valid keys)),
// or -1 to DECLINE (nKeys past the shared-mem top-k cap) so the caller keeps the host selection.
extern "C" int fcuda_dsa_index_select_f32(const float *dIndexQ, const float *dIndexK,
                                          const float *dWeights, int nKeys, int nH,
                                          int indexDim, int queryPos, int topK, float scale,
                                          int *outIdx) {
  if (nKeys <= 0 || topK <= 0) return 0;
  if (nKeys > DSA_TOPK_MAX_KEYS) return -1; // window past the maskable top-k tail: decline, host falls back
  int threads = 256;
  double *dScores = (double *)fcuda_malloc(sizeof(double) * (size_t)nKeys);
  FCUDA_TS("k_dsa_index_score", nKeys, k_dsa_index_score<<<nKeys, threads, 0, g_stream>>>(dIndexQ, dIndexK, dWeights,
                                                     dScores, nKeys, nH, indexDim,
                                                     queryPos, scale));
  int *dSel = (int *)fcuda_malloc(sizeof(int) * (size_t)topK);
  int *dCount = (int *)fcuda_malloc(sizeof(int));
  FCUDA_TS("k_dsa_index_topk", 1, k_dsa_index_topk<<<1, threads, 0, g_stream>>>(dScores, nKeys, queryPos, topK, dSel, dCount));
  int hCount = 0;
  fcuda_ts_untimed_transfer();
  CK(cudaMemcpy(&hCount, dCount, sizeof(int), cudaMemcpyDeviceToHost));
  if (hCount < 0) hCount = 0;
  if (hCount > topK) hCount = topK;
  if (hCount > 0) {
    fcuda_ts_untimed_transfer();
    CK(cudaMemcpy(outIdx, dSel, sizeof(int) * (size_t)hCount, cudaMemcpyDeviceToHost));
  }
  g_host_bytes += sizeof(int) * (size_t)(hCount + 1); // only the index list crosses host-ward
  fcuda_free(dScores);
  fcuda_free(dSel);
  fcuda_free(dCount);
  return hCount;
}

// ---- argmax (first index of the max value — cpuref tie-break) -------------------
__global__ void k_argmax(const float *L, int n, int *outIdx) {
  __shared__ float vbest[256];
  __shared__ int   ibest[256];
  float bv = -INFINITY; int bi = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = L[i];
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  vbest[threadIdx.x] = bv; ibest[threadIdx.x] = bi; __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      float ov = vbest[threadIdx.x + s]; int oi = ibest[threadIdx.x + s];
      if (ov > vbest[threadIdx.x] || (ov == vbest[threadIdx.x] && oi < ibest[threadIdx.x])) {
        vbest[threadIdx.x] = ov; ibest[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *outIdx = ibest[0];
}
// ---- CUDA graph capture/replay (collapse ~600 calls/token -> 1 launch) -----------
// The decode op-graph has IDENTICAL topology every token (same ops); only kernel/memcpy
// PARAMS change (the RoPE/attention position, nPos, the KV-append destination offset). So
// we instantiate ONCE and, every later token, cudaGraphExecUpdate the kept exec with the
// freshly-captured graph — patching just those params, NOT recompiling. That removes the
// per-token instantiate cost that made naive per-token capture a no-win.
static cudaGraphExec_t g_exec = nullptr;

extern "C" void fcuda_graph_reset(void) {
  if (g_exec) { cudaGraphExecDestroy(g_exec); g_exec = nullptr; }
}

// fcuda_graph_abort ends an open capture and throws the captured graph away WITHOUT launching
// it — the recovery half of fcuda_graph_begin for the case where the Go side unwound (panicked)
// mid-capture. Without this the stream stays in capture mode and every subsequent op fails with
// "operation not permitted while the stream is capturing", cascading the whole serve. Clearing
// any sticky error keeps the context usable for the next request. Best-effort: a failed
// EndCapture still leaves cudaGetLastError cleared so the next op is not poisoned by it.
extern "C" void fcuda_graph_abort(void) {
  cudaGraph_t graph = nullptr;
  g_capture_open = false; // #969: capture torn down; the dalloc guard is disarmed again
  cudaStreamEndCapture(g_stream, &graph);
  if (graph) cudaGraphDestroy(graph);
  cudaGetLastError(); // swallow the capture-state error so the next op starts clean
}

extern "C" int fcuda_graph_begin(void) {
  // Global mode: capture every op submitted to g_stream regardless of thread (the Go
  // caller LockOSThread-pins the token, and cudaMu serializes, so nothing else submits).
  if (cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeGlobal) != cudaSuccess) return 1;
  g_capture_open = true; // #969: arm the dalloc guard for the captured region
  return 0;
}
extern "C" int fcuda_graph_end_launch(void) {
  cudaGraph_t graph = nullptr;
  g_capture_open = false; // #969: capture closed; device allocation is permitted again
  cudaError_t e = cudaStreamEndCapture(g_stream, &graph);
  if (e != cudaSuccess || !graph) { fprintf(stderr, "fak-cuda: EndCapture: %s\n", cudaGetErrorString(e)); return FCUDA_GRAPH_END_CAPTURE_FAILED; }
  if (g_exec == nullptr) {
    e = cudaGraphInstantiate(&g_exec, graph, 0);
    if (e != cudaSuccess) { fprintf(stderr, "fak-cuda: Instantiate: %s\n", cudaGetErrorString(e)); cudaGraphDestroy(graph); g_exec = nullptr; return FCUDA_GRAPH_INSTANTIATE_FAILED; }
  } else {
    cudaGraphExecUpdateResultInfo info;
    e = cudaGraphExecUpdate(g_exec, graph, &info);
    if (e != cudaSuccess) {
      // topology drifted (shouldn't, but be safe): re-instantiate from scratch.
      static int warned = 0;
      if (warned < 3) { fprintf(stderr, "fak-cuda: ExecUpdate failed: %s (result=%d) -> re-instantiate\n", cudaGetErrorString(e), (int)info.result); warned++; }
      cudaGetLastError();
      cudaGraphExecDestroy(g_exec);
      e = cudaGraphInstantiate(&g_exec, graph, 0);
      if (e != cudaSuccess) { fprintf(stderr, "fak-cuda: Re-instantiate: %s\n", cudaGetErrorString(e)); cudaGraphDestroy(graph); g_exec = nullptr; return FCUDA_GRAPH_INSTANTIATE_FAILED; }
    }
  }
  int rc = 0;
  e = cudaGraphLaunch(g_exec, g_stream);
  if (e != cudaSuccess) { fprintf(stderr, "fak-cuda: Launch: %s\n", cudaGetErrorString(e)); rc = FCUDA_GRAPH_LAUNCH_FAILED; }
  // The fence is where an asynchronous fault inside the replayed graph appears; return it
  // instead of only logging, so GraphEndLaunch raises it at this token, not a later op.
  e = cudaStreamSynchronize(g_stream);
  if (g_graph_fence_test_fault) {
    g_graph_fence_test_fault = 0;
    e = cudaErrorLaunchFailure;
  }
  if (e != cudaSuccess) {
    fprintf(stderr, "fak-cuda: graph fence: %s\n", cudaGetErrorString(e));
    // A failed fence leaves execution uncertain even when the launch also failed.
    rc = FCUDA_GRAPH_FENCE_FAILED;
  }
  cudaGraphDestroy(graph); // keep g_exec for the next token's ExecUpdate
  return rc;
}

// Persistent argmax index buffer (#969), grown ONCE and reused, exactly like g_attn_scratch
// above. The previous code did fcuda_malloc(sizeof(int)) + fcuda_free per call; the pooled
// allocator misses (-> cudaMalloc) on the first argmax of any size class, and CUDA forbids
// cudaMalloc while g_stream is mid-capture ("operation not permitted when stream is capturing").
// Since argmax is the final op of the captured decode graph, that lazy alloc crashed graph
// capture/replay. A single persistent device int touched on every token never allocates inside
// capture. ensure_argmax_idx() runs on the FIRST (uncaptured) fcuda_argmax_f32 call — the same
// pre-warm role the explicit cuBLAS workspace prealloc in fcuda_init plays for the GEMM scratch
// and the uncaptured halLogitsWarm step (internal/model/hal.go) plays for every pooled op output —
// so by the time the first captured token runs, this buffer is already resident. There is no
// separate fcuda_graph_warm() entry point; warming is exactly these three pre-capture allocations,
// and the #969 g_capture_open guard in fcuda_malloc fails loud if any scratch was missed.
static int *g_argmax_idx = nullptr;
static void ensure_argmax_idx(void) {
  if (!g_argmax_idx) CK(cudaMalloc(&g_argmax_idx, sizeof(int)));
}

extern "C" int fcuda_argmax_f32(const float *dLogits, int n) {
  int hIdx = 0;
  ensure_argmax_idx(); // no-op after the first call; never allocates inside a capture
  FCUDA_TS("k_argmax", 1, k_argmax<<<1, 256, 0, g_stream>>>(dLogits, n, g_argmax_idx));
  fcuda_ts_untimed_transfer();
  CK(cudaMemcpy(&hIdx, g_argmax_idx, sizeof(int), cudaMemcpyDeviceToHost));
  g_host_bytes += sizeof(int); // only the token id crosses host-ward — the #482 witness
  return hIdx;
}
