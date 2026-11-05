// sampling_topk.cu — the device sampling-truncation kernel family for
// wide-vocabulary rows: the CUDA half of the samplingTruncationCluster path
// declared at internal/agent/inkernel_sampling.go. One CTA per row (the
// borrowed persistent-CTA scheduler shape generalizes a fixed grid to
// round-robin row assignment; the single-CTA-per-row form is what the
// filtered/unified upstream kernels use for the decode regime this seam
// serves).
//
// ADAPTED from vLLM (Apache-2.0; vllm-project/vllm at
// ae065918640a198a56f6f17eb45e251c3229fe41):
//   csrc/persistent_topk.cuh:868  persistent_topk_kernel — persistent-CTA row
//       scheduler; coarse-histogram + byte-refinement threshold selection
//       over monotone keys; warp-aggregated emission.
//   csrc/persistent_topk.cuh:1295 FilteredTopKUnifiedKernel — single-CTA-per-row
//       filtered top-k with a bounded shared stash and overflow fallback.
//   csrc/sampled_topk.cuh:144     sampled_topk_kernel — sample-estimated cutoff,
//       survivor compaction, exact full-row fallback.
//   csrc/sampler.cu:16            apply_repetition_penalties_kernel — tiled
//       row-parallel pre-truncation penalty pass.
// No upstream source is copied; the kernels re-express the borrowed shapes in
// fak's wide-vocab sampling contract. Deliberate deviations from upstream:
//   * ties at the selection cutoff resolve by ASCENDING token index (upstream
//     leaves them to atomic order); the host probSorter tieByIndex contract
//     requires the deterministic order.
//   * the repetition penalty is the additive OpenAI frequency/presence form
//     the host chain applies (upstream applies a multiplicative GPT-2 form).
// UNWIRED: this fragment is not yet #included into cuda_kernels.cu and its ABI
// is not yet declared in cuda_backend.h; the successor wiring packet lands
// both. Until then the host applyCPUSamplingTruncation chain remains the
// correctness oracle and fallback. The host face (sampling_topk.go) is the
// exact float64 twin of these kernels' contract; device results are an Approx
// peer held to the argmax-exact + mass-gate tolerance, not max|Δ|=0.

#define SAMPLING_TOPK_THREADS 1024
#define SAMPLING_TOPK_BINS 256

// Per-row status codes written to dStatus[row]: 0 ok, 1 non-finite row,
// 2 zero post-truncation mass. The host face maps both refusals onto its
// ErrNonFiniteLogits-equivalent sentinel.
#define SAMPLING_STATUS_OK 0
#define SAMPLING_STATUS_NON_FINITE 1
#define SAMPLING_STATUS_ZERO_MASS 2

// Monotone f32 -> u32 key: unsigned key order equals numeric value order
// (the standard sign-flip radix transform). NaN is out of contract — the host
// face rejects non-finite rows before launch; the kernels still refuse a row
// that carries one (status 1) rather than mis-selecting.
__device__ __forceinline__ unsigned int sampling_key(float x) {
  const unsigned int bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ bool sampling_nonfinite(float x) {
  return !(x == x) || x == INFINITY || x == -INFINITY;
}

// Whole-block sum of an int, via warp shuffles + one shared word per warp.
// scratch must be disjoint from any other concurrently used shared storage.
__device__ __forceinline__ int sampling_block_sum_int(int v, int* scratch) {
  const unsigned mask = 0xffffffffu;
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) scratch[wid] = v;
  __syncthreads();
  const int nw = SAMPLING_TOPK_THREADS / 32;
  int total = (threadIdx.x < nw) ? scratch[threadIdx.x] : 0;
  for (int off = 16; off > 0; off >>= 1) total += __shfl_down_sync(mask, total, off);
  if (threadIdx.x == 0) scratch[0] = total;
  __syncthreads();
  return scratch[0];
}

// Whole-block sum of a float, same shape.
__device__ __forceinline__ float sampling_block_sum_f32(float v, float* scratch) {
  const unsigned mask = 0xffffffffu;
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) scratch[wid] = v;
  __syncthreads();
  const int nw = SAMPLING_TOPK_THREADS / 32;
  float total = (threadIdx.x < nw) ? scratch[threadIdx.x] : 0.f;
  for (int off = 16; off > 0; off >>= 1) total += __shfl_down_sync(mask, total, off);
  if (threadIdx.x == 0) scratch[0] = total;
  __syncthreads();
  return scratch[0];
}

// ============================================================================
// Stage 1: repetition penalties (borrowed shape: csrc/sampler.cu:16).
// One block per (row, vocab tile); threads stride the tile. Additive OpenAI
// frequency/presence form computed in double to track the host formula:
//   logit[t] -= freq*count[t] + pres      (count[t] > 0)
// counts shorter than vocab treat uncovered tokens as count 0.
// ============================================================================
__global__ void __launch_bounds__(SAMPLING_TOPK_THREADS)
    k_sampling_repetition_penalty_f32(
        float* __restrict__ logits,         // [rows, vocab], in place
        const int32_t* __restrict__ counts, // [countsLen] per-token counts
        int countsLen, int rows, int vocab, double freq, double pres) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tileStart = blockIdx.y * blockDim.x;
  const int tileEnd = min(tileStart + (int)blockDim.x, vocab);
  float* lg = logits + (size_t)row * vocab;
  for (int t = tileStart + threadIdx.x; t < tileEnd; t += blockDim.x) {
    if (t < countsLen && counts[t] > 0) {
      lg[t] -= (float)(freq * (double)counts[t] + pres);
    }
  }
}

// ============================================================================
// Stage 2: top-k truncation (borrowed shape: persistent_topk.cuh:868 coarse
// histogram + byte refinement; FilteredTopKUnifiedKernel single-CTA emission).
// One CTA per row. Descend the 32-bit monotone key MSB-first, four bytes,
// isolating the exact cutoff key c with #{key > c} < k <= #{key >= c}; keep
// everything strictly above c and fill the remaining slots from the equal-key
// group in ASCENDING index order (the deterministic tie-break). Excluded
// weights are zeroed in place; the surviving f32 mass lands in dMass[row].
// ============================================================================
struct SamplingTopKSmem {
  int hist[2][SAMPLING_TOPK_BINS];  // ping-pong count histograms
  int scan[SAMPLING_TOPK_THREADS];  // chunked exclusive-scan scratch
  float massScan[SAMPLING_TOPK_THREADS];
  int scalars[8];                   // broadcasts: above, need, boundary, suf, flags
};

__device__ __forceinline__ unsigned int sampling_cutoff_key_f32(
    const float* row, int n, int k, int* histA, int* histB) {
  unsigned int prefix = 0;
  int offset = 0; // items strictly above the current prefix level
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = threadIdx.x; b < SAMPLING_TOPK_BINS; b += SAMPLING_TOPK_THREADS) histA[b] = 0;
    __syncthreads();
    // fixedMask covers exactly the bytes fixed so far (above the current one);
    // the first round fixes nothing, so every item matches. Shifting a 32-bit
    // value by 32 is UB, hence the explicit guard.
    const unsigned int fixedMask = (shift + 8 >= 32) ? 0u : (0xFFFFFFFFu << (shift + 8));
    for (int i = threadIdx.x; i < n; i += SAMPLING_TOPK_THREADS) {
      const unsigned int key = sampling_key(row[i]);
      if ((key & fixedMask) == (prefix & fixedMask)) {
        atomicAdd(&histA[(key >> shift) & 0xFF], 1);
      }
    }
    __syncthreads();
    // Suffix counts: histB[b] = #{matching items with bin > b}.
    for (int b = threadIdx.x; b < SAMPLING_TOPK_BINS; b += SAMPLING_TOPK_THREADS) histB[b] = histA[b];
    __syncthreads();
    for (int stride = 1; stride < SAMPLING_TOPK_BINS; stride <<= 1) {
      if (threadIdx.x < SAMPLING_TOPK_BINS - stride) histB[threadIdx.x] += histB[threadIdx.x + stride];
      __syncthreads();
    }
    const int rem = k - offset;
    if (threadIdx.x == 0) {
      unsigned int boundary = 0;
      int sufAtBoundary = 0;
      for (int b = SAMPLING_TOPK_BINS - 1; b >= 0; b--) {
        if (histB[b] < rem && histB[b] + histA[b] >= rem) {
          boundary = (unsigned int)b;
          sufAtBoundary = histB[b];
          break;
        }
      }
      histB[0] = (int)boundary;
      histB[1] = sufAtBoundary;
    }
    __syncthreads();
    offset += histB[1];
    prefix |= (unsigned int)histB[0] << shift;
  }
  return prefix;
}

__global__ void __launch_bounds__(SAMPLING_TOPK_THREADS)
    k_sampling_topk_truncate_f32(
        float* __restrict__ probs, int* __restrict__ status,
        float* __restrict__ massOut, int n, int k) {
  extern __shared__ char smem_raw[];
  SamplingTopKSmem& s = *reinterpret_cast<SamplingTopKSmem*>(smem_raw);
  const int row = blockIdx.x;

  // Pass 1: classify against nothing yet — just refuse non-finite rows early.
  int bad = 0;
  for (int i = threadIdx.x; i < n; i += SAMPLING_TOPK_THREADS) {
    if (sampling_nonfinite(probs[i])) bad = 1;
  }
  s.scalars[0] = sampling_block_sum_int(bad, s.scan);
  __syncthreads();
  if (s.scalars[0] != 0) {
    if (threadIdx.x == 0) status[row] = SAMPLING_STATUS_NON_FINITE;
    return;
  }

  const unsigned int cutoff = sampling_cutoff_key_f32(probs, n, k, s.hist[0], s.hist[1]);

  // Pass 2: count items strictly above the cutoff.
  int aboveLocal = 0;
  for (int i = threadIdx.x; i < n; i += SAMPLING_TOPK_THREADS) {
    if (sampling_key(probs[i]) > cutoff) aboveLocal++;
  }
  const int above = sampling_block_sum_int(aboveLocal, s.scan);
  __syncthreads();
  // The cutoff invariant gives 0 <= k - above <= #{ties}; the scan below can
  // keep at most one slot per tie, so no further clamp is needed.
  const int need = k - above;

  // Pass 3: chunked exclusive scan over the tie indicator, writing back in the
  // same pass so no per-element rank storage is needed. A tie at ascending
  // index rank r is kept iff r < need. The running chunk base lives in shared
  // memory because every lane needs the updated value on the next chunk.
  float massLocal = 0.f;
  s.scalars[2] = 0;
  __syncthreads();
  for (int chunkStart = 0; chunkStart < n; chunkStart += SAMPLING_TOPK_THREADS) {
    const int i = chunkStart + threadIdx.x;
    const bool inRow = i < n;
    const unsigned int key = inRow ? sampling_key(probs[i]) : 0u;
    const int val = inRow && key == cutoff ? 1 : 0;
    s.scan[threadIdx.x] = val;
    __syncthreads();
    for (int stride = 1; stride < SAMPLING_TOPK_THREADS; stride <<= 1) {
      const int j = threadIdx.x - stride;
      if (j >= 0) s.scan[threadIdx.x] += s.scan[j];
      __syncthreads();
    }
    const int excl = s.scan[threadIdx.x] - val;
    if (inRow) {
      const bool keep = key > cutoff || (key == cutoff && excl + s.scalars[2] < need);
      if (!keep) probs[i] = 0.f;
      massLocal += keep ? probs[i] : 0.f;
    }
    __syncthreads();
    if (threadIdx.x == 0) s.scalars[2] += s.scan[SAMPLING_TOPK_THREADS - 1];
    __syncthreads();
  }
  const float mass = sampling_block_sum_f32(massLocal, s.massScan);
  __syncthreads();
  if (threadIdx.x == 0) {
    massOut[row] = mass;
    status[row] = (mass > 0.f) ? SAMPLING_STATUS_OK : SAMPLING_STATUS_ZERO_MASS;
  }
}

// ============================================================================
// Stage 3: top-p (nucleus) truncation. One CTA per row. The boundary value is
// found by descending the monotone key MSB-first over BIN MASSES (the borrowed
// threshold shape applied to mass instead of counts): keep the smallest prefix
// of the (value desc, index asc) total order whose mass reaches topP times the
// incoming post-top-k mass. Ties at the cutoff fill in ascending index order
// until the running mass reaches the target. Bin masses use float atomics, so
// this stage is an Approx peer of the host walk (the exact float64 contract
// lives in sampling_topk.go); the wiring gate enforces mass agreement.
// dMass holds the incoming mass per row on entry and the surviving mass on
// exit.
// ============================================================================
struct SamplingNucleusSmem {
  float binMass[SAMPLING_TOPK_BINS];    // current round's bin masses
  float massSuffix[SAMPLING_TOPK_BINS];  // suffix copy: mass of bins > b
  float massScan[SAMPLING_TOPK_THREADS]; // chunked exclusive-scan scratch
  float chunkBase;                       // running tie-mass base across chunks
  float sufBroadcast;                    // thread-0 broadcast of the boundary suffix
  int scalars[4];                        // [0] bad flag, [1] boundary bin
};

__global__ void __launch_bounds__(SAMPLING_TOPK_THREADS)
    k_sampling_nucleus_f32(
        float* __restrict__ probs, int* __restrict__ status,
        float* __restrict__ massInOut, int n, double topP) {
  extern __shared__ char smem_raw[];
  SamplingNucleusSmem& s = *reinterpret_cast<SamplingNucleusSmem*>(smem_raw);
  const int row = blockIdx.x;

  int bad = 0;
  for (int i = threadIdx.x; i < n; i += SAMPLING_TOPK_THREADS) {
    if (sampling_nonfinite(probs[i])) bad = 1;
  }
  s.scalars[0] = sampling_block_sum_int(bad, s.massScan + SAMPLING_TOPK_THREADS / 2);
  __syncthreads();
  if (s.scalars[0] != 0) {
    if (threadIdx.x == 0) status[row] = SAMPLING_STATUS_NON_FINITE;
    return;
  }

  const float target = (float)(topP * (double)massInOut[row]);

  // Byte-refinement descent over bin masses. offsetMass carries the mass of
  // items strictly above the current prefix level; rem is what still has to be
  // crossed inside the matching set.
  unsigned int prefix = 0;
  float offsetMass = 0.f;
  bool keepAll = false;
  for (int shift = 24; shift >= 0 && !keepAll; shift -= 8) {
    for (int b = threadIdx.x; b < SAMPLING_TOPK_BINS; b += SAMPLING_TOPK_THREADS) {
      s.binMass[b] = 0.f;
      s.massSuffix[b] = 0.f;
    }
    __syncthreads();
    // fixedMask covers exactly the bytes fixed so far (see the top-k kernel).
    const unsigned int fixedMask = (shift + 8 >= 32) ? 0u : (0xFFFFFFFFu << (shift + 8));
    for (int i = threadIdx.x; i < n; i += SAMPLING_TOPK_THREADS) {
      const unsigned int key = sampling_key(probs[i]);
      if ((key & fixedMask) == (prefix & fixedMask)) {
        atomicAdd(&s.binMass[(key >> shift) & 0xFF], probs[i]);
      }
    }
    __syncthreads();
    // Suffix masses: massSuffix[b] = mass of matching items with bin > b.
    for (int b = threadIdx.x; b < SAMPLING_TOPK_BINS; b += SAMPLING_TOPK_THREADS) {
      s.massSuffix[b] = s.binMass[b];
    }
    __syncthreads();
    for (int stride = 1; stride < SAMPLING_TOPK_BINS; stride <<= 1) {
      if (threadIdx.x < SAMPLING_TOPK_BINS - stride) {
        s.massSuffix[threadIdx.x] += s.massSuffix[threadIdx.x + stride];
      }
      __syncthreads();
    }
    const float rem = target - offsetMass;
    if (threadIdx.x == 0) {
      int boundary = -1;
      for (int b = SAMPLING_TOPK_BINS - 1; b >= 0; b--) {
        if (s.massSuffix[b] < rem && s.massSuffix[b] + s.binMass[b] >= rem) {
          boundary = b;
          s.sufBroadcast = s.massSuffix[b];
          break;
        }
      }
      s.scalars[1] = boundary;
    }
    __syncthreads();
    if (s.scalars[1] < 0) {
      // No crossing inside the matching set: the target is not reached before
      // the end of the row — keep everything (the host walk does the same).
      keepAll = true;
      break;
    }
    offsetMass += s.sufBroadcast;
    prefix |= (unsigned int)s.scalars[1] << shift;
  }
  if (keepAll) {
    if (threadIdx.x == 0) status[row] = SAMPLING_STATUS_OK;
    return;
  }
  const unsigned int cutoff = prefix;

  // Mass of items strictly above the cutoff.
  float aboveLocal = 0.f;
  for (int i = threadIdx.x; i < n; i += SAMPLING_TOPK_THREADS) {
    if (sampling_key(probs[i]) > cutoff) aboveLocal += probs[i];
  }
  const float massAbove = sampling_block_sum_f32(aboveLocal, s.massScan);
  __syncthreads();
  const float tieTarget = target - massAbove;

  // Chunked exclusive scan over tie WEIGHTS; a tie is kept iff its cumulative
  // mass before it stays below tieTarget (the host walk's stop rule).
  float massLocal = 0.f;
  s.chunkBase = 0.f;
  __syncthreads();
  for (int chunkStart = 0; chunkStart < n; chunkStart += SAMPLING_TOPK_THREADS) {
    const int i = chunkStart + threadIdx.x;
    const bool inRow = i < n;
    const unsigned int key = inRow ? sampling_key(probs[i]) : 0u;
    const float val = inRow && key == cutoff ? probs[i] : 0.f;
    s.massScan[threadIdx.x] = val;
    __syncthreads();
    for (int stride = 1; stride < SAMPLING_TOPK_THREADS; stride <<= 1) {
      const int j = threadIdx.x - stride;
      if (j >= 0) s.massScan[threadIdx.x] += s.massScan[j];
      __syncthreads();
    }
    const float excl = s.massScan[threadIdx.x] - val;
    if (inRow) {
      const bool keep = key > cutoff || (key == cutoff && excl + s.chunkBase < tieTarget);
      if (!keep) probs[i] = 0.f;
      massLocal += keep ? probs[i] : 0.f;
    }
    __syncthreads();
    if (threadIdx.x == 0) s.chunkBase += s.massScan[SAMPLING_TOPK_THREADS - 1];
    __syncthreads();
  }
  const float mass = sampling_block_sum_f32(massLocal, s.massScan);
  __syncthreads();
  if (threadIdx.x == 0) {
    massInOut[row] = mass;
    status[row] = (mass > 0.f) ? SAMPLING_STATUS_OK : SAMPLING_STATUS_ZERO_MASS;
  }
}

// ============================================================================
// Stage 4: sampled-token selection (borrowed shape: csrc/sampled_topk.cuh:144
// exact CDF selection). One CTA per row. A chunked block prefix-sum builds the
// strict CDF; the coin target is located by chunk, then a binary search finds
// the first position whose inclusive cumulative mass exceeds coin*mass — the
// host's strict boundary rule (zero-mass tokens never selected, even at coin
// zero; a coin exactly on a boundary belongs to the next interval). An
// exhausted row falls back to the last nonzero token, else index n-1.
// ============================================================================
struct SamplingDrawSmem {
  float cdf[SAMPLING_TOPK_THREADS];  // inclusive CDF of the active chunk
  float chunkEnd[1024];              // inclusive CDF at each chunk end
  int scalars[4];                    // [1] winner chunk
};

__device__ __forceinline__ void sampling_chunk_scan(float* buf, const float* src,
                                                    int n, int chunkStart) {
  const int i = chunkStart + threadIdx.x;
  buf[threadIdx.x] = (i < n) ? src[i] : 0.f;
  __syncthreads();
  for (int stride = 1; stride < SAMPLING_TOPK_THREADS; stride <<= 1) {
    const int j = threadIdx.x - stride;
    if (j >= 0) buf[threadIdx.x] += buf[j];
    __syncthreads();
  }
}

__global__ void __launch_bounds__(SAMPLING_TOPK_THREADS)
    k_sampling_draw_f32(
        const float* __restrict__ probs, const float* __restrict__ coins,
        int* __restrict__ tokenOut, int n) {
  extern __shared__ char smem_raw[];
  SamplingDrawSmem& s = *reinterpret_cast<SamplingDrawSmem*>(smem_raw);
  const int row = blockIdx.x;
  const float* p = probs + (size_t)row * n;

  int nChunks = (n + SAMPLING_TOPK_THREADS - 1) / SAMPLING_TOPK_THREADS;
  float base = 0.f;
  for (int c = 0; c < nChunks; c++) {
    sampling_chunk_scan(s.cdf, p, n, c * SAMPLING_TOPK_THREADS);
    if (threadIdx.x == 0) s.chunkEnd[c] = base + s.cdf[SAMPLING_TOPK_THREADS - 1];
    __syncthreads();
    base = s.chunkEnd[c];
  }
  const float total = base;
  const float goal = coins[row] * total;

  int winChunk = -1;
  if (threadIdx.x == 0) {
    for (int c = 0; c < nChunks; c++) {
      if (s.chunkEnd[c] > goal) {
        winChunk = c;
        break;
      }
    }
    s.scalars[1] = winChunk;
  }
  __syncthreads();
  winChunk = s.scalars[1];

  int token = -1;
  if (winChunk >= 0) {
    const int i0 = winChunk * SAMPLING_TOPK_THREADS;
    sampling_chunk_scan(s.cdf, p, n, i0);
    if (threadIdx.x == 0) {
      // First position whose inclusive cumulative mass exceeds the goal.
      int lo = 0, hi = SAMPLING_TOPK_THREADS - 1;
      while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (base + s.cdf[mid] > goal) hi = mid;
        else lo = mid + 1;
      }
      token = i0 + lo;
    }
  } else {
    // Coin at/above the full mass: fall back to the last nonzero token.
    if (threadIdx.x == 0) {
      for (int i = n - 1; i >= 0; i--) {
        if (p[i] > 0.f) {
          token = i;
          break;
        }
      }
      if (token < 0) token = n - 1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) tokenOut[row] = token;
}

// ============================================================================
// Host ABI (flat C, backend-stream launches). rc: 0 ok, -1 invalid argument,
// 70000+cudaError launch failure. Per-row refusals land in dStatus.
// ============================================================================
extern "C" int fcuda_sampling_repetition_penalty_f32(
    float* dLogits, const int32_t* dCounts, int countsLen,
    int rows, int vocab, double freq, double pres) {
  if (!dLogits || rows <= 0 || vocab <= 0 || countsLen < 0 ||
      (countsLen > 0 && !dCounts)) {
    return -1;
  }
  if (!(freq >= 0.0) || !(pres >= 0.0) || !isfinite(freq) || !isfinite(pres)) return -1;
  cudaGetLastError();
  dim3 grid(rows, (vocab + SAMPLING_TOPK_THREADS - 1) / SAMPLING_TOPK_THREADS);
  FCUDA_TS("k_sampling_repetition_penalty_f32", grid,
           k_sampling_repetition_penalty_f32<<<grid, SAMPLING_TOPK_THREADS, 0, g_stream>>>(
               dLogits, dCounts, countsLen, rows, vocab, freq, pres));
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 70000 + (int)launch;
}

extern "C" int fcuda_sampling_topk_truncate_f32(
    float* dProbs, int* dStatus, float* dMass, int rows, int vocab, int topK) {
  if (!dProbs || !dStatus || !dMass || rows <= 0 || vocab <= 0 ||
      topK <= 0 || topK >= vocab) {
    return -1;
  }
  const size_t smem = sizeof(SamplingTopKSmem);
  cudaGetLastError();
  if (cudaFuncSetAttribute(k_sampling_topk_truncate_f32,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem) !=
      cudaSuccess) {
    return -1;
  }
  dim3 grid(rows);
  FCUDA_TS("k_sampling_topk_truncate_f32", grid,
           k_sampling_topk_truncate_f32<<<grid, SAMPLING_TOPK_THREADS, smem, g_stream>>>(
               dProbs, dStatus, dMass, vocab, topK));
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 70000 + (int)launch;
}

extern "C" int fcuda_sampling_nucleus_f32(
    float* dProbs, int* dStatus, float* dMass, int rows, int vocab, double topP) {
  if (!dProbs || !dStatus || !dMass || rows <= 0 || vocab <= 0 ||
      !(topP > 0.0) || !(topP < 1.0) || !isfinite(topP)) {
    return -1;
  }
  const size_t smem = sizeof(SamplingNucleusSmem);
  cudaGetLastError();
  if (cudaFuncSetAttribute(k_sampling_nucleus_f32,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem) !=
      cudaSuccess) {
    return -1;
  }
  dim3 grid(rows);
  FCUDA_TS("k_sampling_nucleus_f32", grid,
           k_sampling_nucleus_f32<<<grid, SAMPLING_TOPK_THREADS, smem, g_stream>>>(
               dProbs, dStatus, dMass, vocab, topP));
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 70000 + (int)launch;
}

extern "C" int fcuda_sampling_draw_f32(
    const float* dProbs, const float* dCoins, int* dToken, int rows, int vocab) {
  if (!dProbs || !dCoins || !dToken || rows <= 0 || vocab <= 0 ||
      vocab > (1 << 20)) {
    return -1;
  }
  const size_t smem = sizeof(SamplingDrawSmem);
  cudaGetLastError();
  if (cudaFuncSetAttribute(k_sampling_draw_f32,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem) !=
      cudaSuccess) {
    return -1;
  }
  dim3 grid(rows);
  FCUDA_TS("k_sampling_draw_f32", grid,
           k_sampling_draw_f32<<<grid, SAMPLING_TOPK_THREADS, smem, g_stream>>>(
               dProbs, dCoins, dToken, vocab));
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 70000 + (int)launch;
}
