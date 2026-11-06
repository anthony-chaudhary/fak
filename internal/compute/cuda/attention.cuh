// ---- Decode attention: NAIVE one-block-per-head (the #486 baseline) --------------
// q[nH*hd]; K,V [nPos, nKV*hd]; out[nH*hd]; scores scratch [nH*nPos]. This is the
// original correct-but-naive kernel (commit 54f8b58): it materializes a FULL scores[nPos]
// row per head in GLOBAL memory (g_attn_scratch) and makes four passes over it (raw score
// write, max, exp-in-place, weighted-V read). The flash kernel below replaces it on the
// live Attention path; this one is RETAINED only as the fused-vs-naive microbench baseline
// (#486) — fcuda_attention_f32 keeps it reachable from the Go side's attentionNaive.
__global__ void k_attention(const float *Q, const float *K, const float *V, float *Out,
                            float *Scores, int nPos, int nH, int nKV, int hd, float scale) {
  int h = blockIdx.x;
  if (h >= nH) return;
  int grp = nH / nKV;
  int kvh = h / grp;
  int w = nKV * hd;
  const float *qh = Q + (size_t)h * hd;
  float *sc = Scores + (size_t)h * nPos;
  __shared__ float red[128];
  // phase 1: raw scores = scale * dot(qh, K[j, kvh])
  for (int j = threadIdx.x; j < nPos; j += blockDim.x) {
    const float *kh = K + (size_t)j * w + kvh * hd;
    float d = 0.f;
    for (int e = 0; e < hd; e++) d += qh[e] * kh[e];
    sc[j] = d * scale;
  }
  __syncthreads();
  // phase 2: block-reduce max
  float lm = -INFINITY;
  for (int j = threadIdx.x; j < nPos; j += blockDim.x) lm = fmaxf(lm, sc[j]);
  red[threadIdx.x] = lm; __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
  float mx = red[0];
  __syncthreads();
  // phase 3: exp in place + block-reduce sum
  float ls = 0.f;
  for (int j = threadIdx.x; j < nPos; j += blockDim.x) { float e = expf(sc[j] - mx); sc[j] = e; ls += e; }
  red[threadIdx.x] = ls; __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
  float sum = red[0];
  __syncthreads();
  // phase 4: out[d] = Σ_j (sc[j]/sum) * V[j, kvh, d]
  for (int d = threadIdx.x; d < hd; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < nPos; j++) acc += sc[j] * V[(size_t)j * w + kvh * hd + d];
    Out[(size_t)h * hd + d] = acc / sum;
  }
}
// Persistent attention scratch, sized to nH*maxPos and grown ONCE (outside any graph
// capture, since maxPos is fixed from the first call). A per-call pooled scratch would
// change size as nPos grows each token -> a cudaMalloc mid-capture (illegal). Reused across
// layers/tokens; each k_attention writes-then-reads it within one call, and calls are
// serialized on g_stream, so sharing is safe.
static float *g_attn_scratch = nullptr;
static int g_attn_scratch_cap = 0; // floats
extern "C" void fcuda_attention_f32(const float *dQ, const float *dK, const float *dV, float *dOut,
                                    int nPos, int maxPos, int nH, int nKV, int hd, float scale) {
#ifdef FAK_CUDA_V41_CHECKED_STREAM
  if (v41_candidates_stream_uncertain()) return;
#endif
  int need = nH * (maxPos > nPos ? maxPos : nPos);
  if (need > g_attn_scratch_cap) {
    if (g_attn_scratch) {
      fak_alloc_untrack(g_attn_scratch);
      cudaFree(g_attn_scratch);
    }
    g_attn_scratch = nullptr;
    g_attn_scratch_cap = 0;
    // Pending-status site: on a miss, never launch onto a null scratch (an illegal access would
    // poison the context for good). The allocation status stays this thread's last error so the
    // next status reader refuses this request typed instead of trusting unwritten output.
    cudaError_t e = cudaMalloc(&g_attn_scratch, (size_t)need * sizeof(float));
    if (e != cudaSuccess || !g_attn_scratch) {
      fprintf(stderr, "fak-cuda: attention scratch cudaMalloc(%zu bytes) failed: %s\n",
              (size_t)need * sizeof(float), cudaGetErrorString(e));
      g_attn_scratch = nullptr;
      return;
    }
    fak_alloc_track(g_attn_scratch, FAK_ALLOC_ATTENTION_SCRATCH, FAK_ALLOC_DEVICE_LOCAL,
                    FAK_ALLOC_SITE("attention-f32-scratch"), (size_t)need * sizeof(float));
    g_attn_scratch_cap = need;
  }
  FCUDA_TS("k_attention", nH, k_attention<<<nH, 128, 0, g_stream>>>(dQ, dK, dV, dOut, g_attn_scratch, nPos, nH, nKV, hd, scale));
}

// ---- Flash / online-softmax decode attention (#486) ------------------------------
// The fused replacement for k_attention. One block per query head; the FLASH_THREADS
// threads of a block SPLIT the head dimension. The KV window is streamed in-place with a
// RUNNING (max m, sum l, output accumulator acc) — the FlashAttention online-softmax — so:
//   • NO scores[nPos] buffer is ever materialized (the naive kernel's g_attn_scratch is
//     gone on this path); the only scratch is per-block SHARED memory (the query row +
//     a reduction staging row), allocated by the launch and reused for every key, every
//     layer and every token — there is NO per-call global allocation at all.
//   • one streaming pass over K and V replaces the naive kernel's four passes over a
//     global scores row, cutting the HBM traffic the decode attention pays.
// causal/grp/scale are consumed as kernel PARAMS (grp = nH/nKV selects the KV head; the
// cache holds exactly the attendable keys, so causality is by construction; scale folds
// into the score). The math is the cpuref softmax(scale·q·k)·V reordered into the online
// form — only the f32 reduction order differs, which is what keeps the lane Approx (the
// recorded cudaFlashAttnCosineMin floor), not bit-identity.
//
// Online-softmax recurrence, per key j (every thread runs it identically off the reduced
// score, so m and l stay replicated and consistent across the block):
//   s      = scale · dot(q, K_j)
//   m'     = max(m, s)
//   corr   = exp(m − m')           // rescales the running sum/acc onto the new max
//   p      = exp(s − m')
//   l      = l·corr + p
//   acc[d] = acc[d]·corr + p·V_j[d]   // each thread owns the dims d it strides
// and out[d] = acc[d]/l after the window. Bit-faithful to softmax then ΣwV; no full row.
#define FLASH_THREADS 128
// FLASH_ACC_MAX bounds the head dims one thread carries = ceil(hd / FLASH_THREADS). 8 covers
// hd ≤ 1024 (every real attention head dim is ≤ 256), so the per-thread acc lives in
// registers/local memory, never a global scores row.
#define FLASH_ACC_MAX 8
template <bool device_position = false>
__global__ void k_flash_attention(const float *Q, const float *K, const float *V, float *Out,
                                  int nPos, int nH, int nKV, int hd, float scale, const int *position = nullptr) {
  if (device_position) nPos = position[1];
  int h = blockIdx.x;
  if (h >= nH) return;
  int grp = nH / nKV;       // query heads per KV head (GQA/MQA group)
  int kvh = h / grp;        // the KV head this query head reads
  int w = nKV * hd;         // per-position stride in K/V
  const float *qh = Q + (size_t)h * hd;
  extern __shared__ float smem[];
  float *qs = smem;                 // [hd]            : the query row, cached once in shared
  float *red = smem + hd;           // [FLASH_THREADS] : dot-product reduction staging
  int tid = threadIdx.x;
  // cache the query row in shared memory (threads stride hd).
  for (int d = tid; d < hd; d += FLASH_THREADS) qs[d] = qh[d];
  __syncthreads();
  // online-softmax running state. acc[k] is this thread's accumulator for owned dim
  // d = tid + k*FLASH_THREADS; m and l are replicated across the block.
  float m = -INFINITY, l = 0.f;
  float acc[FLASH_ACC_MAX];
#pragma unroll
  for (int k = 0; k < FLASH_ACC_MAX; k++) acc[k] = 0.f;
  for (int j = 0; j < nPos; j++) {
    const float *kj = K + (size_t)j * w + (size_t)kvh * hd;
    // partial dot over this thread's strided dims, then a block reduction -> full score.
    float partial = 0.f;
    for (int d = tid; d < hd; d += FLASH_THREADS) partial += qs[d] * kj[d];
    red[tid] = partial;
    __syncthreads();
    for (int s = FLASH_THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float sc = red[0] * scale;        // every thread reads the reduced dot
    __syncthreads();                  // WAR: finish all red[0] reads before next key reuses red
    float mNew = fmaxf(m, sc);
    float corr = expf(m - mNew);      // 0 on the first key (m = -inf): clears the empty acc
    float p = expf(sc - mNew);        // expf (not __expf) so the only divergence from the
    l = l * corr + p;                 // reference is f32 reduction order, not a faster-exp ulp

    const float *vj = V + (size_t)j * w + (size_t)kvh * hd;
    int k = 0;
    for (int d = tid; d < hd; d += FLASH_THREADS, k++) acc[k] = acc[k] * corr + p * vj[d];
    m = mNew;
  }
  float invL = l > 0.f ? 1.f / l : 0.f;
  int k = 0;
  for (int d = tid; d < hd; d += FLASH_THREADS, k++) Out[(size_t)h * hd + d] = acc[k] * invL;
}
// fcuda_flash_attention_f32 launches the flash kernel: one block per query head, FLASH_THREADS
// threads, and just enough dynamic shared memory for the query row + the reduction row. Unlike
// the naive entrypoint there is NO g_attn_scratch — the online form needs no nPos-sized buffer,
// so nothing is allocated or grown per call. maxPos is accepted for a signature parallel to the
// naive baseline (the microbench calls both) but is unused here.
extern "C" void fcuda_flash_attention_f32(const float *dQ, const float *dK, const float *dV, float *dOut,
                                          int nPos, int maxPos, int nH, int nKV, int hd, float scale) {
  (void)maxPos;
  size_t shmem = ((size_t)hd + FLASH_THREADS) * sizeof(float);
  FCUDA_TS("k_flash_attention", nH, k_flash_attention<false><<<nH, FLASH_THREADS, shmem, g_stream>>>(dQ, dK, dV, dOut, nPos, nH, nKV, hd, scale));
}

// Prompt-panel counterpart of k_flash_attention. K/V already contain the
// complete appended panel; query row t is restricted to prefix+t+1, preserving
// causality without materializing a scores panel or replaying rows from Go. The
// first visible value seeds (m,l,acc) exactly; later values use the online-softmax
// recurrence, avoiding any finite sentinel assumption about the score range.
__global__ void k_qwen35_causal_attention_panel(
    const float *Q, const float *K, const float *V, float *Out,
    int tokens, int prefix, int nH, int nKV, int hd, float scale) {
  int flat = blockIdx.x;
  int h = flat % nH, token = flat / nH;
  if (h >= nH || token >= tokens) return;
  int tid = threadIdx.x, kvh = h / (nH / nKV), width = nKV * hd;
  int nPos = prefix + token + 1;
  const float *qh = Q + (size_t)token * nH * hd + (size_t)h * hd;
  extern __shared__ float smem[];
  float *qs = smem;
  float *red = smem + hd;
  for (int d = tid; d < hd; d += FLASH_THREADS) qs[d] = qh[d];
  __syncthreads();
  float m = 0.f, l = 0.f;
  float acc[FLASH_ACC_MAX];
#pragma unroll
  for (int k = 0; k < FLASH_ACC_MAX; k++) acc[k] = 0.f;
  for (int j = 0; j < nPos; j++) {
    const float *kj = K + (size_t)j * width + (size_t)kvh * hd;
    float partial = 0.f;
    for (int d = tid; d < hd; d += FLASH_THREADS) partial += qs[d] * kj[d];
    red[tid] = partial;
    __syncthreads();
    for (int s = FLASH_THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float score = red[0] * scale;
    __syncthreads();
    const float *vj = V + (size_t)j * width + (size_t)kvh * hd;
    if (j == 0) {
      m = score;
      l = 1.f;
      int k = 0;
      for (int d = tid; d < hd; d += FLASH_THREADS, k++) acc[k] = vj[d];
      continue;
    }
    float nextM = fmaxf(m, score);
    float correction = expf(m - nextM), probability = expf(score - nextM);
    l = l * correction + probability;
    int k = 0;
    for (int d = tid; d < hd; d += FLASH_THREADS, k++) {
      acc[k] = acc[k] * correction + probability * vj[d];
    }
    m = nextM;
  }
  float invL = l > 0.f ? 1.f / l : 0.f;
  int k = 0;
  for (int d = tid; d < hd; d += FLASH_THREADS, k++) {
    Out[(size_t)token * nH * hd + (size_t)h * hd + d] = acc[k] * invL;
  }
}

// Qwen3.8 27B uses a 256-wide attention head. Keep that production geometry on
// a named source path so a source/ABI audit can prove it does not accidentally
// regress to the legacy warp-only body that returned without writing hd > 32.
// The recurrence deliberately matches k_qwen35_causal_attention_panel above:
// one 128-thread block owns a row, each thread owns two dimensions, and shared
// memory holds the query plus the block-reduction lane.
#define QWEN38_PROMPT_HEAD_DIM 256
__global__ void k_qwen38_causal_attention_panel_hd256(
    const float *Q, const float *K, const float *V, float *Out,
    int tokens, int prefix, int nH, int nKV, float scale) {
  int flat = blockIdx.x;
  int h = flat % nH, token = flat / nH;
  if (h >= nH || token >= tokens) return;
  int tid = threadIdx.x, kvh = h / (nH / nKV);
  int width = nKV * QWEN38_PROMPT_HEAD_DIM;
  int nPos = prefix + token + 1;
  const float *qh = Q + (size_t)token * nH * QWEN38_PROMPT_HEAD_DIM +
                    (size_t)h * QWEN38_PROMPT_HEAD_DIM;
  extern __shared__ float smem[];
  float *qs = smem;
  float *red = smem + QWEN38_PROMPT_HEAD_DIM;
  for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS) qs[d] = qh[d];
  __syncthreads();
  float m = 0.f, l = 0.f;
  float acc[QWEN38_PROMPT_HEAD_DIM / FLASH_THREADS];
#pragma unroll
  for (int k = 0; k < QWEN38_PROMPT_HEAD_DIM / FLASH_THREADS; k++) acc[k] = 0.f;
  for (int j = 0; j < nPos; j++) {
    const float *kj = K + (size_t)j * width +
                      (size_t)kvh * QWEN38_PROMPT_HEAD_DIM;
    float partial = 0.f;
    for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS) {
      partial += qs[d] * kj[d];
    }
    red[tid] = partial;
    __syncthreads();
    for (int s = FLASH_THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float score = red[0] * scale;
    __syncthreads();
    const float *vj = V + (size_t)j * width +
                      (size_t)kvh * QWEN38_PROMPT_HEAD_DIM;
    if (j == 0) {
      m = score;
      l = 1.f;
      int k = 0;
      for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS, k++) acc[k] = vj[d];
      continue;
    }
    float nextM = fmaxf(m, score);
    float correction = expf(m - nextM), probability = expf(score - nextM);
    l = l * correction + probability;
    int k = 0;
    for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS, k++) {
      acc[k] = acc[k] * correction + probability * vj[d];
    }
    m = nextM;
  }
  float invL = l > 0.f ? 1.f / l : 0.f;
  int k = 0;
  for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS, k++) {
    Out[(size_t)token * nH * QWEN38_PROMPT_HEAD_DIM +
        (size_t)h * QWEN38_PROMPT_HEAD_DIM + d] = acc[k] * invL;
  }
}

// ---- Split-KV chunked-prefill panel (vLLM merge_attn_states borrow) --------------
// ADAPT of vllm-project/vllm csrc/attention/merge_attn_states.cu:18 at
// ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0; notice at
// third_party/vllm-merge-attn-states/NOTICE): rows whose prefix exceeds
// QWEN38_SPLITKV_PREFIX_THRESHOLD split their causal KV range into at most
// QWEN38_SPLITKV_MAX_PARTITIONS contiguous partitions, one CTA per
// (row, head, partition). Each CTA runs k_qwen38_causal_attention_panel_hd256's
// online-softmax recurrence over its own key range and writes the UNNORMALIZED
// partial (m, l, O); the merge kernel folds the per-row partials by LSE —
// M = max m_p, L = sum l_p exp(m_p - M), O = sum O_p exp(m_p - M), out = O/L —
// the same log-sum-exp combine upstream does with LSE = m + ln(l). Re-expressed
// for Fak's f32 hd256 panel: no upstream code blocks are copied, the two-state
// prefix/suffix form is generalized to N partitions, and an empty partition
// (j0 == j1) writes the sentinel partial m = -INFINITY, l = 0, O = 0 that the
// merge drops exactly (weight exp(-INFINITY - M) = 0). Short prefixes keep the
// single-pass panel bitwise: qwen38_splitkv_partitions returns 1 below the
// threshold and the launcher takes the unchanged path. Complementary to the P-05
// MMA verdict (fattn_mma_hd256.cuh): when the tensor-core arm is selected and
// available it still owns the panel; split-KV parallelizes only the SIMT arm.
#define QWEN38_SPLITKV_PREFIX_THRESHOLD 1024
#define QWEN38_SPLITKV_MAX_PARTITIONS 4
static int qwen38_splitkv_partitions(int prefix, int tokens) {
  (void)tokens;
  if (prefix < QWEN38_SPLITKV_PREFIX_THRESHOLD) return 1;
  int p = 1 + prefix / QWEN38_SPLITKV_PREFIX_THRESHOLD;
  if (p > QWEN38_SPLITKV_MAX_PARTITIONS) p = QWEN38_SPLITKV_MAX_PARTITIONS;
  return p;
}
// Partial kernel: same geometry and recurrence as k_qwen38_causal_attention_panel_hd256,
// restricted to the row's partition [j0, j1) (floor-divided boundaries, so unaligned
// splits and empty leading partitions fall out of the integer arithmetic). The first key
// of the partition seeds (m, l, acc) exactly like the panel's j == 0 branch; the slot
// layout is [m, l, O[QWEN38_PROMPT_HEAD_DIM]] per (rowHead, partition).
__global__ void k_qwen38_causal_attention_panel_hd256_splitkv_partial(
    const float *Q, const float *K, const float *V, float *Partials,
    int tokens, int prefix, int nH, int nKV, float scale, int nsplits) {
  int rowHead = blockIdx.x % (tokens * nH);
  int partition = blockIdx.x / (tokens * nH);
  int h = rowHead % nH, token = rowHead / nH;
  if (h >= nH || token >= tokens || partition >= nsplits) return;
  int tid = threadIdx.x, kvh = h / (nH / nKV);
  int width = nKV * QWEN38_PROMPT_HEAD_DIM;
  int nPos = prefix + token + 1;
  int j0 = (int)((long long)partition * nPos / nsplits);
  int j1 = (int)((long long)(partition + 1) * nPos / nsplits);
  const float *qh = Q + (size_t)token * nH * QWEN38_PROMPT_HEAD_DIM +
                    (size_t)h * QWEN38_PROMPT_HEAD_DIM;
  extern __shared__ float smem[];
  float *qs = smem;
  float *red = smem + QWEN38_PROMPT_HEAD_DIM;
  for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS) qs[d] = qh[d];
  __syncthreads();
  float *slot = Partials + ((size_t)rowHead * nsplits + partition) *
                         (QWEN38_PROMPT_HEAD_DIM + 2);
  if (j0 >= j1) {
    // Empty partition: the sentinel partial the merge drops exactly.
    if (tid == 0) {
      slot[0] = -INFINITY;
      slot[1] = 0.f;
    }
    for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS) slot[2 + d] = 0.f;
    return;
  }
  float m = 0.f, l = 0.f;
  float acc[QWEN38_PROMPT_HEAD_DIM / FLASH_THREADS];
#pragma unroll
  for (int k = 0; k < QWEN38_PROMPT_HEAD_DIM / FLASH_THREADS; k++) acc[k] = 0.f;
  for (int j = j0; j < j1; j++) {
    const float *kj = K + (size_t)j * width +
                      (size_t)kvh * QWEN38_PROMPT_HEAD_DIM;
    float partial = 0.f;
    for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS) {
      partial += qs[d] * kj[d];
    }
    red[tid] = partial;
    __syncthreads();
    for (int s = FLASH_THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float score = red[0] * scale;
    __syncthreads();
    const float *vj = V + (size_t)j * width +
                      (size_t)kvh * QWEN38_PROMPT_HEAD_DIM;
    if (j == j0) {
      m = score;
      l = 1.f;
      int k = 0;
      for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS, k++) acc[k] = vj[d];
      continue;
    }
    float nextM = fmaxf(m, score);
    float correction = expf(m - nextM), probability = expf(score - nextM);
    l = l * correction + probability;
    int k = 0;
    for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS, k++) {
      acc[k] = acc[k] * correction + probability * vj[d];
    }
    m = nextM;
  }
  if (tid == 0) {
    slot[0] = m;
    slot[1] = l;
  }
  int k = 0;
  for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS, k++) slot[2 + d] = acc[k];
}
// Merge kernel: one block per (row, head); every thread strides the head dims exactly as
// the panel kernels do. The LSE fold rescales each partial onto the row max M and divides
// by the merged denominator; an all-empty row (M == -INFINITY, unreachable for a valid
// causal row) writes zeros instead of dividing by zero.
__global__ void k_qwen38_splitkv_lse_merge_hd256(
    const float *Partials, float *Out, int tokens, int nH, int nsplits) {
  int flat = blockIdx.x;
  int h = flat % nH, token = flat / nH;
  if (h >= nH || token >= tokens) return;
  int tid = threadIdx.x;
  const float *base = Partials + (size_t)flat * nsplits * (QWEN38_PROMPT_HEAD_DIM + 2);
  float mMax = -INFINITY;
  for (int p = 0; p < nsplits; p++) {
    mMax = fmaxf(mMax, base[p * (QWEN38_PROMPT_HEAD_DIM + 2)]);
  }
  float *outRow = Out + (size_t)token * nH * QWEN38_PROMPT_HEAD_DIM +
                   (size_t)h * QWEN38_PROMPT_HEAD_DIM;
  if (mMax == -INFINITY) {
    for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS) outRow[d] = 0.f;
    return;
  }
  float lSum = 0.f;
  float acc[QWEN38_PROMPT_HEAD_DIM / FLASH_THREADS];
#pragma unroll
  for (int k = 0; k < QWEN38_PROMPT_HEAD_DIM / FLASH_THREADS; k++) acc[k] = 0.f;
  for (int p = 0; p < nsplits; p++) {
    const float *slot = base + p * (QWEN38_PROMPT_HEAD_DIM + 2);
    float weight = expf(slot[0] - mMax);
    lSum += slot[1] * weight;
    int k = 0;
    for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS, k++) {
      acc[k] += slot[2 + d] * weight;
    }
  }
  float invL = lSum > 0.f ? 1.f / lSum : 0.f;
  int k = 0;
  for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS, k++) {
    outRow[d] = acc[k] * invL;
  }
}

// FAK_CUDA_QWEN38_PREFILL_ATTN selects the Qwen3.8 prompt-attention kernel: "simt" forces
// the SIMT kernel above (the parity oracle and rollback), "mma" the tensor-core port in
// fattn_mma_hd256.cuh on sm_80+. Unset keeps SIMT until the P-05 hardware A/B records a
// keep verdict; that receipt flips QWEN38_PREFILL_ATTN_MMA_DEFAULT to 1.
#define QWEN38_PREFILL_ATTN_MMA_DEFAULT 0
static bool qwen38_prefill_attention_wants_mma(void) {
  const char *mode = getenv("FAK_CUDA_QWEN38_PREFILL_ATTN");
  if (mode && strcmp(mode, "simt") == 0) return false;
  if (mode && strcmp(mode, "mma") == 0) return true;
  return QWEN38_PREFILL_ATTN_MMA_DEFAULT != 0;
}

// The unchanged single-pass SIMT panel launch: short prefixes, and any call
// that must stay allocation-free (scratch growth is illegal mid-capture).
static int qwen38_causal_attention_panel_launch_simt(
    const float *dQ, const float *dK, const float *dV, float *dOut,
    int tokens, int prefix, int nH, int nKV, float scale) {
  size_t shmem = ((size_t)QWEN38_PROMPT_HEAD_DIM + FLASH_THREADS) * sizeof(float);
  FCUDA_TS("k_qwen38_causal_attention_panel_hd256", tokens * nH, k_qwen38_causal_attention_panel_hd256<<<tokens * nH, FLASH_THREADS, shmem, g_stream>>>(
      dQ, dK, dV, dOut, tokens, prefix, nH, nKV, scale));
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 50000 + (int)launch;
}

extern "C" int fak_qwen35_causal_attention_panel_f32(
    const float *dQ, const float *dK, const float *dV, float *dOut,
    int tokens, int prefix, int nH, int nKV, int hd, float scale) {
  // A failed retirement has unknown completion: retain ownership and never retry or reuse.
  if (g_cuda_cleanup_status != 0) return 52000 + g_cuda_cleanup_status;
#ifdef FAK_CUDA_V41_CHECKED_STREAM
  if (v41_candidates_stream_uncertain()) return 52000 + (int)cudaErrorNotReady;
#endif
  // Any causal panel is admitted: the kernel bounds row t by prefix+t+1 itself,
  // so only the Qwen3.8 head geometry stays exact (matching the Go validator).
  if (!dQ || !dK || !dV || !dOut ||
      tokens <= 0 || prefix < 0 || nH != 24 || nKV != 4 ||
      hd != QWEN38_PROMPT_HEAD_DIM ||
      tokens > INT_MAX / nH || prefix > INT_MAX - tokens ||
      !isfinite(scale) || scale <= 0.f) return -1;
  // width is an int inside the kernel and the Go sequence cache uses int-sized
  // element offsets at its C ABI. Refuse every product that cannot be represented
  // before launch, rather than allowing an overflowed address calculation to no-op
  // or write an unrelated allocation.
  if (nKV > INT_MAX / hd || nH > INT_MAX / hd) return -1;
  int kvWidth = nKV * hd, qWidth = nH * hd;
  int positions = prefix + tokens;
  if (positions > INT_MAX / kvWidth || tokens > INT_MAX / qWidth) return -1;
  int nsplits = qwen38_splitkv_partitions(prefix, tokens);
  if (nsplits > 1 && tokens > INT_MAX / nH / nsplits) return -1;
  // Preserve the first observed stream/runtime fault before selector fallback.
  if (g_cuda_cleanup_status != 0) return 52000 + g_cuda_cleanup_status;
#ifdef FAK_CUDA_V41_CHECKED_STREAM
  if (v41_candidates_stream_uncertain()) return 52000 + (int)cudaErrorNotReady;
#endif
  cudaError_t prior = cudaGetLastError();
  if (prior != cudaSuccess) {
    if (g_cuda_cleanup_status == 0) g_cuda_cleanup_status = (int)prior;
    return 52000 + g_cuda_cleanup_status;
  }
  if (qwen38_prefill_attention_wants_mma()) {
    // -2 means below sm_80, capture growth, or a known null-allocation OOM.
    // No operand work was submitted, so the SIMT kernel may run instead.
    int mma = fak_qwen38_fattn_mma_hd256_launch(dQ, dK, dV, dOut, tokens, prefix, nH, nKV, scale);
    if (mma != -2) return mma;
  }
  if (nsplits <= 1) {
    return qwen38_causal_attention_panel_launch_simt(dQ, dK, dV, dOut, tokens, prefix, nH, nKV, scale);
  }
  // Split-KV SIMT panel: one CTA per (row, head, partition) writes its unnormalized
  // (m, l, O) partial into grow-only scratch; the LSE merge kernel folds them per row.
  static float *g_splitkv_scratch = nullptr;
  static size_t g_splitkv_scratch_cap = 0; // floats
  // The preceding int geometry check bounds slots, but bytes also have to fit
  // size_t before any retirement/allocation (including a 32-bit host build).
  const size_t slots = (size_t)tokens * nH * nsplits;
  const size_t slotWidth = QWEN38_PROMPT_HEAD_DIM + 2;
  if (slots > (size_t)-1 / slotWidth / sizeof(float)) return -1;
  const size_t need = slots * slotWidth;
  auto splitFailed = [](cudaError_t status) {
    if (status == cudaSuccess) status = cudaErrorInvalidValue;
    if (g_cuda_cleanup_status == 0) g_cuda_cleanup_status = (int)status;
    return 50000 + g_cuda_cleanup_status;
  };
  bool splitReady = need <= g_splitkv_scratch_cap;
  if (!splitReady && !g_capture_open) {
    if (g_splitkv_scratch) {
      float *retiring = g_splitkv_scratch;
      cudaError_t retired = cudaFree(retiring);
      if (retired != cudaSuccess) {
        if (g_cuda_cleanup_status == 0) g_cuda_cleanup_status = (int)retired;
        return 52000 + g_cuda_cleanup_status;
      }
      // The physical free is final: bookkeeping refusal cannot retry it.
      g_splitkv_scratch = nullptr;
      g_splitkv_scratch_cap = 0;
      if (!fak_alloc_untrack(retiring)) return splitFailed(cudaErrorInvalidValue);
    }
    g_splitkv_scratch = nullptr;
    g_splitkv_scratch_cap = 0;
    cudaError_t e = fcuda_alloc_status(cudaMalloc(&g_splitkv_scratch, need * sizeof(float)));
    if (e != cudaSuccess) {
      fprintf(stderr, "fak-cuda: splitkv scratch cudaMalloc(%zu bytes) failed: %s\n",
              need * sizeof(float), cudaGetErrorString(e));
      // Preserve a pointer returned with failure. Only null OOM is a known
      // capacity refusal that leaves future calls eligible to retry.
      if (e == cudaErrorMemoryAllocation && !g_splitkv_scratch) return 50000 + (int)e;
      return splitFailed(e);
    }
    if (!g_splitkv_scratch) return splitFailed(cudaErrorInvalidValue);
    g_splitkv_scratch_cap = need;
    try {
      fak_alloc_track(g_splitkv_scratch, FAK_ALLOC_ATTENTION_SCRATCH, FAK_ALLOC_DEVICE_LOCAL,
                      FAK_ALLOC_SITE("qwen38-splitkv-scratch"), need * sizeof(float));
    } catch (...) {
      // Retain device ownership/extent if host ledger insertion failed.
      return splitFailed(cudaErrorMemoryAllocation);
    }
    splitReady = true;
  }
  if (!splitReady) {
    // Scratch growth is illegal mid-capture: keep the bitwise single-pass panel.
    return qwen38_causal_attention_panel_launch_simt(dQ, dK, dV, dOut, tokens, prefix, nH, nKV, scale);
  }
  size_t shmem = ((size_t)QWEN38_PROMPT_HEAD_DIM + FLASH_THREADS) * sizeof(float);
  FCUDA_TS("k_qwen38_causal_attention_panel_hd256_splitkv_partial", tokens * nH * nsplits,
           k_qwen38_causal_attention_panel_hd256_splitkv_partial<<<tokens * nH * nsplits, FLASH_THREADS, shmem, g_stream>>>(
               dQ, dK, dV, g_splitkv_scratch, tokens, prefix, nH, nKV, scale, nsplits));
  cudaError_t partialLaunch = cudaGetLastError();
  if (partialLaunch != cudaSuccess) return splitFailed(partialLaunch);
  if (g_cuda_cleanup_status != 0) return 50000 + g_cuda_cleanup_status;
#ifdef FAK_CUDA_V41_CHECKED_STREAM
  if (v41_candidates_stream_uncertain()) return splitFailed(cudaErrorNotReady);
#endif
  FCUDA_TS("k_qwen38_splitkv_lse_merge_hd256", tokens * nH,
           k_qwen38_splitkv_lse_merge_hd256<<<tokens * nH, FLASH_THREADS, 0, g_stream>>>(
               g_splitkv_scratch, dOut, tokens, nH, nsplits));
  cudaError_t mergeLaunch = cudaGetLastError();
  if (mergeLaunch != cudaSuccess) return splitFailed(mergeLaunch);
  if (g_cuda_cleanup_status != 0) return 50000 + g_cuda_cleanup_status;
#ifdef FAK_CUDA_V41_CHECKED_STREAM
  if (v41_candidates_stream_uncertain()) return splitFailed(cudaErrorNotReady);
#endif
  return mergeLaunch == cudaSuccess ? 0 : 50000 + (int)mergeLaunch;
}

// Prompt-panel causal attention for the dense HAL prefill: one block per (row, head), each
// running k_flash_attention's body over its causal window [0, prefix+t], so a row is bitwise
// the decode Attention over prefix+t+1 keys, non-finite scores included.
__global__ void k_prompt_causal_attention_panel(
    const float *Q, const float *K, const float *V, float *Out,
    int tokens, int prefix, int nH, int nKV, int hd, float scale) {
  int flat = blockIdx.x;
  int h = flat % nH, token = flat / nH;
  if (h >= nH || token >= tokens) return;
  int grp = nH / nKV;
  int kvh = h / grp;
  int w = nKV * hd;
  int nPos = prefix + token + 1;
  const float *qh = Q + (size_t)token * nH * hd + (size_t)h * hd;
  extern __shared__ float smem[];
  float *qs = smem;
  float *red = smem + hd;
  int tid = threadIdx.x;
  for (int d = tid; d < hd; d += FLASH_THREADS) qs[d] = qh[d];
  __syncthreads();
  float m = -INFINITY, l = 0.f;
  float acc[FLASH_ACC_MAX];
#pragma unroll
  for (int k = 0; k < FLASH_ACC_MAX; k++) acc[k] = 0.f;
  for (int j = 0; j < nPos; j++) {
    const float *kj = K + (size_t)j * w + (size_t)kvh * hd;
    float partial = 0.f;
    for (int d = tid; d < hd; d += FLASH_THREADS) partial += qs[d] * kj[d];
    red[tid] = partial;
    __syncthreads();
    for (int s = FLASH_THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float sc = red[0] * scale;
    __syncthreads();
    float mNew = fmaxf(m, sc);
    float corr = expf(m - mNew);
    float p = expf(sc - mNew);
    l = l * corr + p;
    const float *vj = V + (size_t)j * w + (size_t)kvh * hd;
    int k = 0;
    for (int d = tid; d < hd; d += FLASH_THREADS, k++) acc[k] = acc[k] * corr + p * vj[d];
    m = mNew;
  }
  float invL = l > 0.f ? 1.f / l : 0.f;
  int k = 0;
  for (int d = tid; d < hd; d += FLASH_THREADS, k++) {
    Out[(size_t)token * nH * hd + (size_t)h * hd + d] = acc[k] * invL;
  }
}

// Launches k_prompt_causal_attention_panel for any GQA geometry with
// hd <= FLASH_THREADS*FLASH_ACC_MAX. K/V hold prefix+tokens positions.
extern "C" int fcuda_causal_attention_panel_f32(
    const float *dQ, const float *dK, const float *dV, float *dOut,
    int tokens, int prefix, int nH, int nKV, int hd, float scale) {
  if (!dQ || !dK || !dV || !dOut || tokens <= 0 || prefix < 0 || nH <= 0 ||
      nKV <= 0 || nH % nKV != 0 || hd <= 0 || hd > FLASH_THREADS * FLASH_ACC_MAX ||
      !isfinite(scale) || scale <= 0.f) return -1;
  if (tokens > INT_MAX / nH || prefix > INT_MAX - tokens ||
      nKV > INT_MAX / hd || nH > INT_MAX / hd) return -1;
  int kvWidth = nKV * hd, qWidth = nH * hd;
  if (prefix + tokens > INT_MAX / kvWidth || tokens > INT_MAX / qWidth) return -1;
  // Preserve the first observed stream/runtime fault before selector fallback.
  if (g_cuda_cleanup_status != 0) return 52000 + g_cuda_cleanup_status;
#ifdef FAK_CUDA_V41_CHECKED_STREAM
  if (v41_candidates_stream_uncertain()) return 52000 + (int)cudaErrorNotReady;
#endif
  cudaError_t prior = cudaGetLastError();
  if (prior != cudaSuccess) {
    if (g_cuda_cleanup_status == 0) g_cuda_cleanup_status = (int)prior;
    return 52000 + g_cuda_cleanup_status;
  }
  // The tensor-core port is instantiated only for the Qwen3.8 hd256 geometry, and its f16
  // K/V staging pays off at prompt length: route P>=128 panels through it when
  // FAK_CUDA_QWEN38_PREFILL_ATTN selects mma. Only -2 permits fallback before
  // operand work; runtime faults return positively and close shared admission.
  if (nH == 24 && nKV == 4 && hd == QWEN38_PROMPT_HEAD_DIM && prefix + tokens >= 128 &&
      qwen38_prefill_attention_wants_mma()) {
    int mma = fak_qwen38_fattn_mma_hd256_launch(dQ, dK, dV, dOut, tokens, prefix, nH, nKV, scale);
    if (mma != -2) return mma;
  }
  size_t shmem = ((size_t)hd + FLASH_THREADS) * sizeof(float);
  FCUDA_TS("k_prompt_causal_attention_panel", tokens * nH, k_prompt_causal_attention_panel<<<tokens * nH, FLASH_THREADS, shmem, g_stream>>>(
      dQ, dK, dV, dOut, tokens, prefix, nH, nKV, hd, scale));
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 71000 + (int)launch;
}

extern "C" int fak_qwen35_sequence_sync(void) {
  cudaError_t completed = cudaStreamSynchronize(g_stream);
  sequence_split_note_failed_fence(completed);
  return completed == cudaSuccess ? 0 : 60000 + (int)completed;
}
