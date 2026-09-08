// Included by cuda_kernels.cu so this kernel shares the backend stream while
// retaining a focused source seam. The 32 compact mask rows travel in CUDA's
// launch-parameter space, not a global-memory mask allocation.

#define TREE_ATTN_THREADS 128
#define TREE_ATTN_ACC_MAX 8
#define TREE_ATTN_MAX_CANDIDATES 32

struct fak_tree_mask32 {
  uint32_t rows[TREE_ATTN_MAX_CANDIDATES];
};

// For query qi, the allowed set is the committed prefix plus exactly the bits
// in masks.rows[qi]. With topological parent indices, those bits are self and
// the repeated parent chain, so siblings, descendants, and other branches are
// excluded. The online recurrence maintains
//   m=max(score), l=sum(exp(score-m)), acc=sum(exp(score-m)*V)
// over precisely that set; acc/l is therefore the batch softmax result in exact
// arithmetic. Device exp/reduction order makes this an approximate f32 peer.
__global__ void k_tree_verify_attention(
    const float *Q, const float *K, const float *V, float *Out,
    fak_tree_mask32 masks,
    int qLen, int kvLen, int nH, int nKV, int hd, float scale) {
  int qi = blockIdx.x;
  int h = blockIdx.y;
  int tid = threadIdx.x;
  if (qi >= qLen || h >= nH) return;

  int group = nH / nKV;
  int kvh = h / group;
  int kvWidth = nKV * hd;
  int prefix = kvLen - qLen;
  uint32_t rowMask = masks.rows[qi];

  extern __shared__ float smem[];
  float *qs = smem;
  float *red = smem + hd;
  const float *qh = Q + ((size_t)qi * nH + h) * hd;
  for (int dim = tid; dim < hd; dim += TREE_ATTN_THREADS) qs[dim] = qh[dim];
  __syncthreads();

  float m = -INFINITY;
  float l = 0.f;
  float acc[TREE_ATTN_ACC_MAX];
#pragma unroll
  for (int slot = 0; slot < TREE_ATTN_ACC_MAX; slot++) acc[slot] = 0.f;

  for (int j = 0; j < kvLen; j++) {
    int candidate = j - prefix;
    bool allowed = j < prefix ||
        (candidate >= 0 && candidate < qLen &&
         (rowMask & (uint32_t(1) << candidate)) != 0);
    if (!allowed) continue;

    const float *kj = K + (size_t)j * kvWidth + (size_t)kvh * hd;
    float partial = 0.f;
    for (int dim = tid; dim < hd; dim += TREE_ATTN_THREADS) partial += qs[dim] * kj[dim];
    red[tid] = partial;
    __syncthreads();
    for (int stride = TREE_ATTN_THREADS / 2; stride > 0; stride >>= 1) {
      if (tid < stride) red[tid] += red[tid + stride];
      __syncthreads();
    }

    float score = red[0] * scale;
    __syncthreads();
    float nextM = fmaxf(m, score);
    float correction = expf(m - nextM);
    float probability = expf(score - nextM);
    l = l * correction + probability;

    const float *vj = V + (size_t)j * kvWidth + (size_t)kvh * hd;
    int slot = 0;
    for (int dim = tid; dim < hd; dim += TREE_ATTN_THREADS, slot++) {
      acc[slot] = acc[slot] * correction + probability * vj[dim];
    }
    m = nextM;
  }

  float invL = 1.f / l;
  int slot = 0;
  float *oh = Out + ((size_t)qi * nH + h) * hd;
  for (int dim = tid; dim < hd; dim += TREE_ATTN_THREADS, slot++) oh[dim] = acc[slot] * invL;
}

extern "C" int fcuda_tree_verify_attention_f32(
    const float *dQ, const float *dK, const float *dV, float *dOut,
    const uint32_t *hMaskRows,
    int qLen, int kvLen, int nH, int nKV, int hd, float scale) {
  if (!dQ || !dK || !dV || !dOut || !hMaskRows) return -1;
  if (qLen <= 0 || qLen > TREE_ATTN_MAX_CANDIDATES || kvLen < qLen) return -1;
  if (nH <= 0 || nKV <= 0 || nH % nKV != 0) return -1;
  if (hd <= 0 || hd > TREE_ATTN_THREADS * TREE_ATTN_ACC_MAX) return -1;
  if (!(scale > 0.f) || !isfinite(scale)) return -1;

  uint32_t validBits = qLen == 32 ? UINT32_MAX : ((uint32_t(1) << qLen) - 1);
  fak_tree_mask32 masks = {};
  for (int q = 0; q < qLen; q++) {
    uint32_t bits = hMaskRows[q];
    uint32_t past = q == 31 ? 0 : (validBits & ~((uint32_t(1) << (q + 1)) - 1));
    if ((bits & ~validBits) != 0 || (bits & (uint32_t(1) << q)) == 0 || (bits & past) != 0) return -1;
    masks.rows[q] = bits;
  }

  cudaGetLastError();
  dim3 grid(qLen, nH);
  size_t shmem = ((size_t)hd + TREE_ATTN_THREADS) * sizeof(float);
  k_tree_verify_attention<<<grid, TREE_ATTN_THREADS, shmem, g_stream>>>(
      dQ, dK, dV, dOut, masks, qLen, kvLen, nH, nKV, hd, scale);
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 70000 + (int)launch;
}
