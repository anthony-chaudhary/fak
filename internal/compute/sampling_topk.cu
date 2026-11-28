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
// UNWIRED: this approximate float32 structural family is not #included into
// cuda_kernels.cu and its ABI is not declared in cuda_backend.h. The served
// applyCPUSamplingTruncation chain remains the oracle/fallback; sampling_topk.go
// is an independent float64 host reference, not an exact twin of this fragment.
// Activation under #14198 requires exact per-stage rounding, signed-zero/tie
// order, ascending-token-ID mass/CDF and request-local 53-bit seeded parity,
// plus transactional refusal, lifetime/completion and served transfer evidence.
// Argmax agreement or a mass tolerance cannot substitute for those gates.

// This remains an UNWIRED approximate f32 primitive family. Structural repairs
// below do not establish the float64 / request-local 53-bit seeded law required
// by #14198. Do not enable resident served sampling with this fragment.
#include <limits.h>
#include <float.h>

#define SAMPLING_TOPK_THREADS 1024
#define SAMPLING_TOPK_BINS 256

// Per-row status codes written to dStatus[row]: 0 ok, 1 invalid probability
// (non-finite or negative), 2 invalid post-truncation mass. The host face maps both refusals onto its
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
    for (int b = threadIdx.x; b < SAMPLING_TOPK_BINS; b += SAMPLING_TOPK_THREADS) {
      histB[b] = (b + 1 < SAMPLING_TOPK_BINS) ? histA[b + 1] : 0;
    }
    __syncthreads();
    for (int stride = 1; stride < SAMPLING_TOPK_BINS; stride <<= 1) {
      const int add = (threadIdx.x < SAMPLING_TOPK_BINS - stride)
          ? histB[threadIdx.x + stride] : 0;
      // Every lane reads the previous round before any lane writes this round.
      __syncthreads();
      if (threadIdx.x < SAMPLING_TOPK_BINS) histB[threadIdx.x] += add;
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
  float* p = probs + (size_t)row * n;

  // Pass 1: classify against nothing yet — just refuse non-finite rows early.
  int bad = 0;
  for (int i = threadIdx.x; i < n; i += SAMPLING_TOPK_THREADS) {
    if (sampling_nonfinite(p[i]) || p[i] < 0.f) bad = 1;
  }
  const int badCount = sampling_block_sum_int(bad, s.scan);
  if (threadIdx.x == 0) s.scalars[0] = badCount;
  __syncthreads();
  if (s.scalars[0] != 0) {
    if (threadIdx.x == 0) status[row] = SAMPLING_STATUS_NON_FINITE;
    return;
  }

  const unsigned int cutoff = sampling_cutoff_key_f32(p, n, k, s.hist[0], s.hist[1]);

  // Pass 2: count items strictly above the cutoff.
  int aboveLocal = 0;
  for (int i = threadIdx.x; i < n; i += SAMPLING_TOPK_THREADS) {
    if (sampling_key(p[i]) > cutoff) aboveLocal++;
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
  if (threadIdx.x == 0) s.scalars[2] = 0;
  __syncthreads();
  for (int chunkStart = 0; chunkStart < n; chunkStart += SAMPLING_TOPK_THREADS) {
    const int i = chunkStart + threadIdx.x;
    const bool inRow = i < n;
    const unsigned int key = inRow ? sampling_key(p[i]) : 0u;
    const int val = inRow && key == cutoff ? 1 : 0;
    s.scan[threadIdx.x] = val;
    __syncthreads();
    for (int stride = 1; stride < SAMPLING_TOPK_THREADS; stride <<= 1) {
      const int j = threadIdx.x - stride;
      const int add = (j >= 0) ? s.scan[j] : 0;
      __syncthreads();
      s.scan[threadIdx.x] += add;
      __syncthreads();
    }
    const int excl = s.scan[threadIdx.x] - val;
    if (inRow) {
      const bool keep = key > cutoff || (key == cutoff && excl + s.scalars[2] < need);
      if (!keep) p[i] = 0.f;
      massLocal += keep ? p[i] : 0.f;
    }
    __syncthreads();
    if (threadIdx.x == 0) s.scalars[2] += s.scan[SAMPLING_TOPK_THREADS - 1];
    __syncthreads();
  }
  const float mass = sampling_block_sum_f32(massLocal, s.massScan);
  __syncthreads();
  if (threadIdx.x == 0) {
    massOut[row] = mass;
    status[row] = (mass > 0.f && !sampling_nonfinite(mass))
        ? SAMPLING_STATUS_OK : SAMPLING_STATUS_ZERO_MASS;
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
  int badScratch[SAMPLING_TOPK_THREADS / 32]; // typed integer reduction storage
  int scalars[4];                        // [0] bad flag, [1] boundary bin
};

__global__ void __launch_bounds__(SAMPLING_TOPK_THREADS)
    k_sampling_nucleus_f32(
        float* __restrict__ probs, int* __restrict__ status,
        float* __restrict__ massInOut, int n, double topP) {
  extern __shared__ char smem_raw[];
  SamplingNucleusSmem& s = *reinterpret_cast<SamplingNucleusSmem*>(smem_raw);
  const int row = blockIdx.x;
  float* p = probs + (size_t)row * n;

  int bad = 0;
  for (int i = threadIdx.x; i < n; i += SAMPLING_TOPK_THREADS) {
    if (sampling_nonfinite(p[i]) || p[i] < 0.f) bad = 1;
  }
  const int badCount = sampling_block_sum_int(bad, s.badScratch);
  if (threadIdx.x == 0) s.scalars[0] = badCount;
  __syncthreads();
  if (s.scalars[0] != 0) {
    if (threadIdx.x == 0) status[row] = SAMPLING_STATUS_NON_FINITE;
    return;
  }

  const float incomingMass = massInOut[row];
  if (!(incomingMass > 0.f) || sampling_nonfinite(incomingMass)) {
    if (threadIdx.x == 0) status[row] = SAMPLING_STATUS_ZERO_MASS;
    return;
  }
  // Keep the narrowest positive nucleus nonempty even if the product underflows.
  // DBL_MIN is below every positive f32 weight, so this floor only selects rank 0.
  const double product = topP * (double)incomingMass;
  const double target = product > 0.0 ? product : DBL_MIN;

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
      const unsigned int key = sampling_key(p[i]);
      if ((key & fixedMask) == (prefix & fixedMask)) {
        atomicAdd(&s.binMass[(key >> shift) & 0xFF], p[i]);
      }
    }
    __syncthreads();
    // Suffix masses: massSuffix[b] = mass of matching items with bin > b.
    for (int b = threadIdx.x; b < SAMPLING_TOPK_BINS; b += SAMPLING_TOPK_THREADS) {
      s.massSuffix[b] = (b + 1 < SAMPLING_TOPK_BINS) ? s.binMass[b + 1] : 0.f;
    }
    __syncthreads();
    for (int stride = 1; stride < SAMPLING_TOPK_BINS; stride <<= 1) {
      const float add = (threadIdx.x < SAMPLING_TOPK_BINS - stride)
          ? s.massSuffix[threadIdx.x + stride] : 0.f;
      __syncthreads();
      if (threadIdx.x < SAMPLING_TOPK_BINS) s.massSuffix[threadIdx.x] += add;
      __syncthreads();
    }
    const double rem = target - (double)offsetMass;
    if (threadIdx.x == 0) {
      int boundary = -1;
      for (int b = SAMPLING_TOPK_BINS - 1; b >= 0; b--) {
        if ((double)s.massSuffix[b] < rem &&
            (double)s.massSuffix[b] + (double)s.binMass[b] >= rem) {
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
    if (sampling_key(p[i]) > cutoff) aboveLocal += p[i];
  }
  const float massAbove = sampling_block_sum_f32(aboveLocal, s.massScan);
  __syncthreads();
  const double tieTarget = target - (double)massAbove;

  // Chunked exclusive scan over tie WEIGHTS; a tie is kept iff its cumulative
  // mass before it stays below tieTarget (the host walk's stop rule).
  float massLocal = 0.f;
  if (threadIdx.x == 0) s.chunkBase = 0.f;
  __syncthreads();
  for (int chunkStart = 0; chunkStart < n; chunkStart += SAMPLING_TOPK_THREADS) {
    const int i = chunkStart + threadIdx.x;
    const bool inRow = i < n;
    const unsigned int key = inRow ? sampling_key(p[i]) : 0u;
    const float val = inRow && key == cutoff ? p[i] : 0.f;
    s.massScan[threadIdx.x] = val;
    __syncthreads();
    for (int stride = 1; stride < SAMPLING_TOPK_THREADS; stride <<= 1) {
      const int j = threadIdx.x - stride;
      const float add = (j >= 0) ? s.massScan[j] : 0.f;
      __syncthreads();
      s.massScan[threadIdx.x] += add;
      __syncthreads();
    }
    const float excl = s.massScan[threadIdx.x] - val;
    if (inRow) {
      const bool keep = key > cutoff || (key == cutoff && (double)excl + (double)s.chunkBase < tieTarget);
      if (!keep) p[i] = 0.f;
      massLocal += keep ? p[i] : 0.f;
    }
    __syncthreads();
    if (threadIdx.x == 0) s.chunkBase += s.massScan[SAMPLING_TOPK_THREADS - 1];
    __syncthreads();
  }
  const float mass = sampling_block_sum_f32(massLocal, s.massScan);
  __syncthreads();
  if (threadIdx.x == 0) {
    massInOut[row] = mass;
    status[row] = (mass > 0.f && !sampling_nonfinite(mass))
        ? SAMPLING_STATUS_OK : SAMPLING_STATUS_ZERO_MASS;
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
    const float add = (j >= 0) ? buf[j] : 0.f;
    __syncthreads();
    buf[threadIdx.x] += add;
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
      const float before = (winChunk == 0) ? 0.f : s.chunkEnd[winChunk - 1];
      int lo = 0, hi = min(SAMPLING_TOPK_THREADS, n - i0) - 1;
      while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (before + s.cdf[mid] > goal) hi = mid;
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
  if (!dLogits || rows <= 0 || vocab <= 0 ||
      vocab > INT_MAX - (SAMPLING_TOPK_THREADS - 1) || countsLen < 0 ||
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
      vocab > INT_MAX - (SAMPLING_TOPK_THREADS - 1) || topK <= 0 || topK >= vocab) {
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
      vocab > INT_MAX - (SAMPLING_TOPK_THREADS - 1) || !(topP > 0.0) || !(topP < 1.0) || !isfinite(topP)) {
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

// ============================================================================
// UNWIRED bit-plane certified LM-head argmax family.
//
// ADAPTED from magnitudedev/magnitude (Apache-2.0):
//   inference/engine/kernels/kernels/readout.seismic:278@8d7dbcc1ba
//   inference/engine/kernels/kernels/cuda/lib/projection/planes.cuh:64@814f72a
//
// Clean-room CUDA C++ expression of the top-bit-plane certified readout:
//   1) compute activation summary statistics (sum, positive sum, negative sum, L1 norm);
//   2) read ONLY the top `topBits` MSB bit-planes with shared-memory activation tile
//      reuse (planes.cuh:64 staged-activation inner loop) and compute a guaranteed
//      row interval [low_r, high_r] bounding the exact F32 dot product;
//   3) certify the argmax margin (bestLow = max_r low_r) and collect ONLY rows with
//      high_r > bestLow (or high_r == bestLow with r < bestLowIdx) as contenders;
//   4) if zero other contenders remain, exit early with bestLowIdx (zero refinement);
//      otherwise refine ONLY contending rows against exact F32 weights.
//
// Status: [SW-VERIFIED] source contract only — intentionally UNWIRED from
// cuda_kernels.cu and cuda_backend.h (token_time: unmeasured / HW-qualification-pending).
// ============================================================================
#define BITPLANE_HEAD_THREADS 256
#define BITPLANE_HEAD_TILE_K 256
#define BITPLANE_STATUS_OK 0
#define BITPLANE_STATUS_NON_FINITE 1

struct BitplaneWorkspaceHeader {
  double sumX;
  double posSumX;
  double negSumX;
  double absSumX;
  double maxAbsX;
  int status;
  int bestLowIdx;
  int refineCount;
  int reserved;
};

__device__ __forceinline__ double bitplane_block_sum_f64(double v, double* scratch) {
  const unsigned mask = 0xffffffffu;
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) scratch[wid] = v;
  __syncthreads();
  const int nw = BITPLANE_HEAD_THREADS / 32;
  double total = (threadIdx.x < nw) ? scratch[threadIdx.x] : 0.0;
  for (int off = 16; off > 0; off >>= 1) total += __shfl_down_sync(mask, total, off);
  if (threadIdx.x == 0) scratch[0] = total;
  __syncthreads();
  return scratch[0];
}

__device__ __forceinline__ double bitplane_block_max_f64(double v, double* scratch) {
  const unsigned mask = 0xffffffffu;
  for (int off = 16; off > 0; off >>= 1) {
    const double other = __shfl_down_sync(mask, v, off);
    if (other > v) v = other;
  }
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) scratch[wid] = v;
  __syncthreads();
  const int nw = BITPLANE_HEAD_THREADS / 32;
  double best = (threadIdx.x < nw) ? scratch[threadIdx.x] : 0.0;
  for (int off = 16; off > 0; off >>= 1) {
    const double other = __shfl_down_sync(mask, best, off);
    if (other > best) best = other;
  }
  if (threadIdx.x == 0) scratch[0] = best;
  __syncthreads();
  return scratch[0];
}

__device__ __forceinline__ int bitplane_block_any_int(int flag, int* scratch) {
  const unsigned mask = 0xffffffffu;
  for (int off = 16; off > 0; off >>= 1) flag |= __shfl_down_sync(mask, flag, off);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) scratch[wid] = flag;
  __syncthreads();
  const int nw = BITPLANE_HEAD_THREADS / 32;
  int any = (threadIdx.x < nw) ? scratch[threadIdx.x] : 0;
  for (int off = 16; off > 0; off >>= 1) any |= __shfl_down_sync(mask, any, off);
  if (threadIdx.x == 0) scratch[0] = any;
  __syncthreads();
  return scratch[0];
}

__device__ __forceinline__ double* bitplane_ws_bounds(void* ws) {
  return reinterpret_cast<double*>(reinterpret_cast<char*>(ws) + 64);
}

__device__ __forceinline__ int* bitplane_ws_contenders(void* ws, int vocab) {
  return reinterpret_cast<int*>(bitplane_ws_bounds(ws) + 2 * (size_t)vocab);
}

__device__ __forceinline__ float* bitplane_ws_scores(void* ws, int vocab) {
  return reinterpret_cast<float*>(bitplane_ws_contenders(ws, vocab) + (size_t)vocab);
}

__global__ void __launch_bounds__(BITPLANE_HEAD_THREADS)
    k_bitplane_x_stats(const float* __restrict__ x, int hiddenDim, void* __restrict__ ws) {
  __shared__ double red[BITPLANE_HEAD_THREADS / 32];
  __shared__ int badRed[BITPLANE_HEAD_THREADS / 32];
  double sumLocal = 0.0, posLocal = 0.0, negLocal = 0.0, absLocal = 0.0, maxLocal = 0.0;
  int bad = 0;
  for (int k = threadIdx.x; k < hiddenDim; k += BITPLANE_HEAD_THREADS) {
    const float xk = x[k];
    if (sampling_nonfinite(xk)) {
      bad = 1;
      continue;
    }
    const double dx = (double)xk;
    sumLocal += dx;
    if (dx > 0.0) posLocal += dx;
    else if (dx < 0.0) negLocal += dx;
    const double ax = fabs(dx);
    absLocal += ax;
    if (ax > maxLocal) maxLocal = ax;
  }
  const int anyBad = bitplane_block_any_int(bad, badRed);
  const double sumAll = bitplane_block_sum_f64(sumLocal, red);
  const double posAll = bitplane_block_sum_f64(posLocal, red);
  const double negAll = bitplane_block_sum_f64(negLocal, red);
  const double absAll = bitplane_block_sum_f64(absLocal, red);
  const double maxAll = bitplane_block_max_f64(maxLocal, red);
  if (threadIdx.x == 0) {
    BitplaneWorkspaceHeader* hdr = reinterpret_cast<BitplaneWorkspaceHeader*>(ws);
    hdr->sumX = sumAll;
    hdr->posSumX = posAll;
    hdr->negSumX = negAll;
    hdr->absSumX = absAll;
    hdr->maxAbsX = maxAll;
    hdr->status = anyBad ? BITPLANE_STATUS_NON_FINITE : BITPLANE_STATUS_OK;
    hdr->bestLowIdx = 0;
    hdr->refineCount = 0;
    hdr->reserved = 0;
  }
}

__global__ void __launch_bounds__(BITPLANE_HEAD_THREADS)
    k_bitplane_bounds_stage(
        const uint32_t* __restrict__ dPlanes,
        const float* __restrict__ dRowMin,
        const float* __restrict__ dRowScale,
        const float* __restrict__ dRowEps,
        const float* __restrict__ dRowMaxAbs,
        const float* __restrict__ dX,
        int vocab, int hiddenDim, int totalBits, int topBits,
        void* __restrict__ ws) {
  const BitplaneWorkspaceHeader* hdr = reinterpret_cast<const BitplaneWorkspaceHeader*>(ws);
  if (hdr->status != BITPLANE_STATUS_OK) return;
  const int row = blockIdx.x;
  if (row >= vocab) return;

  __shared__ double smemX[BITPLANE_HEAD_TILE_K];
  __shared__ double red[BITPLANE_HEAD_THREADS / 32];

  const int wordsPerRow = (hiddenDim + 31) / 32;
  const int remBits = totalBits - topBits;
  double topCodeDotLocal = 0.0;

  // Staged activation reuse: each tile stages BITPLANE_HEAD_TILE_K elements of x
  // into shared memory once and reuses them across all top bit-plane words.
  for (int t0 = 0; t0 < hiddenDim; t0 += BITPLANE_HEAD_TILE_K) {
    const int tileLen = min(BITPLANE_HEAD_TILE_K, hiddenDim - t0);
    if (threadIdx.x < tileLen) {
      smemX[threadIdx.x] = (double)dX[t0 + threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.x < tileLen) {
      const int k = t0 + threadIdx.x;
      const int wordIdx = k >> 5;
      const int bitIdx = k & 31;
      uint32_t topCode = 0u;
      for (int p = 0; p < topBits; ++p) {
        const size_t planeOffset = ((size_t)row * (size_t)totalBits + (size_t)p) * (size_t)wordsPerRow;
        const uint32_t word = dPlanes[planeOffset + (size_t)wordIdx];
        topCode = (topCode << 1) | ((word >> bitIdx) & 1u);
      }
      const uint32_t shiftedCode = topCode << remBits;
      topCodeDotLocal += (double)shiftedCode * smemX[threadIdx.x];
    }
    __syncthreads();
  }

  const double topCodeDot = bitplane_block_sum_f64(topCodeDotLocal, red);
  if (threadIdx.x != 0) return;

  double* bounds = bitplane_ws_bounds(ws);
  const float minF = dRowMin[row];
  const float scaleF = dRowScale[row];
  const float epsF = dRowEps[row];
  const float maxAbsF = dRowMaxAbs[row];
  if (sampling_nonfinite(minF) || sampling_nonfinite(scaleF) ||
      sampling_nonfinite(epsF) || sampling_nonfinite(maxAbsF) ||
      scaleF < 0.0f || epsF < 0.0f || maxAbsF < 0.0f) {
    bounds[2 * (size_t)row] = -INFINITY;
    bounds[2 * (size_t)row + 1] = INFINITY;
    return;
  }

  const double maxAbsW = (double)maxAbsF;
  if (maxAbsW * hdr->maxAbsX * (double)hiddenDim >= (double)FLT_MAX * 0.25) {
    bounds[2 * (size_t)row] = -INFINITY;
    bounds[2 * (size_t)row + 1] = INFINITY;
    return;
  }

  const double minVal = (double)minF;
  const double scale = (double)scaleF;
  const double eps = (double)epsF;
  const double baseDot = minVal * hdr->sumX + scale * topCodeDot;
  const uint32_t tailMask = (remBits == 0) ? 0u : ((1u << remBits) - 1u);
  const double tailMin = scale * (double)tailMask * hdr->negSumX;
  const double tailMax = scale * (double)tailMask * hdr->posSumX;

  const double n = (double)hiddenDim + 2.0;
  const double u32 = ldexp(1.0, -24);
  const double gamma32 = (n * u32 < 0.5) ? (n * u32) / (1.0 - n * u32) : 1.0;
  const double f32Slack = gamma32 * maxAbsW * hdr->absSumX + n * ldexp(1.0, -149);
  const uint32_t maxCode = (1u << totalBits) - 1u;
  const double f64Slack = ((double)hiddenDim + 8.0) * ldexp(1.0, -52) *
      ((fabs(minVal) + scale * (double)maxCode) * hdr->absSumX + 1.0);
  const double pad = eps * hdr->absSumX + f32Slack + f64Slack;

  bounds[2 * (size_t)row] = baseDot + tailMin - pad;
  bounds[2 * (size_t)row + 1] = baseDot + tailMax + pad;
}

__global__ void __launch_bounds__(BITPLANE_HEAD_THREADS)
    k_bitplane_certify_contenders(
        int vocab, int* __restrict__ dTokenOut, int* __restrict__ dRefinedOut,
        void* __restrict__ ws) {
  BitplaneWorkspaceHeader* hdr = reinterpret_cast<BitplaneWorkspaceHeader*>(ws);
  if (hdr->status != BITPLANE_STATUS_OK) {
    if (threadIdx.x == 0) {
      dTokenOut[0] = -1;
      dRefinedOut[0] = 0;
    }
    return;
  }
  const double* bounds = bitplane_ws_bounds(ws);
  int* contenders = bitplane_ws_contenders(ws, vocab);

  __shared__ double sLow[BITPLANE_HEAD_THREADS];
  __shared__ int sIdx[BITPLANE_HEAD_THREADS];
  __shared__ int sScan[BITPLANE_HEAD_THREADS];
  __shared__ int sBase;

  double localLow = -INFINITY;
  int localIdx = -1;
  for (int r = threadIdx.x; r < vocab; r += BITPLANE_HEAD_THREADS) {
    const double lo = bounds[2 * (size_t)r];
    if (lo != lo) {
      localLow = INFINITY;
      localIdx = -2;
      break;
    }
    if (localIdx == -1 || lo > localLow || (lo == localLow && r < localIdx)) {
      localLow = lo;
      localIdx = r;
    }
  }
  sLow[threadIdx.x] = localLow;
  sIdx[threadIdx.x] = localIdx;
  __syncthreads();

  for (int stride = BITPLANE_HEAD_THREADS / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const int iA = sIdx[threadIdx.x];
      const int iB = sIdx[threadIdx.x + stride];
      const double lA = sLow[threadIdx.x];
      const double lB = sLow[threadIdx.x + stride];
      if (iB == -2 || (iA != -2 && iB >= 0 && (iA < 0 || lB > lA || (lB == lA && iB < iA)))) {
        sLow[threadIdx.x] = lB;
        sIdx[threadIdx.x] = iB;
      }
    }
    __syncthreads();
  }

  const double bestLow = sLow[0];
  const int bestLowIdx = sIdx[0];
  if (bestLowIdx < 0) {
    if (threadIdx.x == 0) {
      hdr->status = BITPLANE_STATUS_NON_FINITE;
      hdr->refineCount = 0;
      dTokenOut[0] = -1;
      dRefinedOut[0] = 0;
    }
    return;
  }

  if (threadIdx.x == 0) {
    hdr->bestLowIdx = bestLowIdx;
    sBase = 0;
  }
  __syncthreads();

  for (int chunkStart = 0; chunkStart < vocab; chunkStart += BITPLANE_HEAD_THREADS) {
    const int r = chunkStart + threadIdx.x;
    const bool inRow = r < vocab;
    const double hi = inRow ? bounds[2 * (size_t)r + 1] : -INFINITY;
    const int keep = (inRow && r != bestLowIdx &&
        (hi != hi || hi > bestLow || (hi == bestLow && r < bestLowIdx))) ? 1 : 0;
    sScan[threadIdx.x] = keep;
    __syncthreads();
    for (int stride = 1; stride < BITPLANE_HEAD_THREADS; stride <<= 1) {
      const int j = threadIdx.x - stride;
      const int add = (j >= 0) ? sScan[j] : 0;
      __syncthreads();
      sScan[threadIdx.x] += add;
      __syncthreads();
    }
    if (keep) {
      const int excl = sScan[threadIdx.x] - 1;
      contenders[sBase + excl] = r;
    }
    __syncthreads();
    if (threadIdx.x == 0) sBase += sScan[BITPLANE_HEAD_THREADS - 1];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const int otherCount = sBase;
    if (otherCount == 0) {
      hdr->refineCount = 0;
      dTokenOut[0] = bestLowIdx;
      dRefinedOut[0] = 0;
    } else {
      contenders[otherCount] = bestLowIdx;
      hdr->refineCount = otherCount + 1;
      dRefinedOut[0] = otherCount + 1;
    }
  }
}

__global__ void __launch_bounds__(BITPLANE_HEAD_THREADS)
    k_bitplane_refine_contenders(
        const float* __restrict__ dWeightsF32, const float* __restrict__ dX,
        int vocab, int hiddenDim, void* __restrict__ ws) {
  const BitplaneWorkspaceHeader* hdr = reinterpret_cast<const BitplaneWorkspaceHeader*>(ws);
  if (hdr->status != BITPLANE_STATUS_OK || hdr->refineCount <= 0) return;

  const int* contenders = bitplane_ws_contenders(ws, vocab);
  float* scores = bitplane_ws_scores(ws, vocab);
  __shared__ float sSum[BITPLANE_HEAD_THREADS];
  __shared__ int sBad[BITPLANE_HEAD_THREADS / 32];

  for (int c = blockIdx.x; c < hdr->refineCount; c += gridDim.x) {
    const int row = contenders[c];
    const float* wRow = dWeightsF32 + (size_t)row * (size_t)hiddenDim;
    float localSum = 0.0f;
    int bad = 0;
    for (int k = threadIdx.x; k < hiddenDim; k += BITPLANE_HEAD_THREADS) {
      const float wk = wRow[k];
      if (sampling_nonfinite(wk)) {
        bad = 1;
      } else {
        localSum += wk * dX[k];
      }
    }
    const int anyBad = bitplane_block_any_int(bad, sBad);
    sSum[threadIdx.x] = localSum;
    __syncthreads();
    for (int stride = BITPLANE_HEAD_THREADS / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) sSum[threadIdx.x] += sSum[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      scores[c] = anyBad ? NAN : sSum[0];
    }
    __syncthreads();
  }
}

__global__ void __launch_bounds__(BITPLANE_HEAD_THREADS)
    k_bitplane_pick_winner(
        int vocab, int* __restrict__ dTokenOut, int* __restrict__ dRefinedOut,
        void* __restrict__ ws) {
  BitplaneWorkspaceHeader* hdr = reinterpret_cast<BitplaneWorkspaceHeader*>(ws);
  if (hdr->status != BITPLANE_STATUS_OK || hdr->refineCount <= 0) return;

  const int count = hdr->refineCount;
  const int* contenders = bitplane_ws_contenders(ws, vocab);
  const float* scores = bitplane_ws_scores(ws, vocab);

  __shared__ float sVal[BITPLANE_HEAD_THREADS];
  __shared__ int sRow[BITPLANE_HEAD_THREADS];
  __shared__ int sBad[BITPLANE_HEAD_THREADS / 32];

  float bestVal = -INFINITY;
  int bestRow = -1;
  int bad = 0;
  for (int c = threadIdx.x; c < count; c += BITPLANE_HEAD_THREADS) {
    const float sc = scores[c];
    const int row = contenders[c];
    if (sampling_nonfinite(sc)) {
      bad = 1;
      continue;
    }
    if (bestRow < 0 || sc > bestVal || (sc == bestVal && row < bestRow)) {
      bestVal = sc;
      bestRow = row;
    }
  }
  const int anyBad = bitplane_block_any_int(bad, sBad);
  sVal[threadIdx.x] = bestVal;
  sRow[threadIdx.x] = bestRow;
  __syncthreads();

  for (int stride = BITPLANE_HEAD_THREADS / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const int rA = sRow[threadIdx.x];
      const int rB = sRow[threadIdx.x + stride];
      const float vA = sVal[threadIdx.x];
      const float vB = sVal[threadIdx.x + stride];
      if (rB >= 0 && (rA < 0 || vB > vA || (vB == vA && rB < rA))) {
        sVal[threadIdx.x] = vB;
        sRow[threadIdx.x] = rB;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (anyBad || sRow[0] < 0) {
      hdr->status = BITPLANE_STATUS_NON_FINITE;
      dTokenOut[0] = -1;
      dRefinedOut[0] = count;
    } else {
      dTokenOut[0] = sRow[0];
      dRefinedOut[0] = count;
    }
  }
}

extern "C" size_t fcuda_bitplane_certified_argmax_workspace_bytes(int vocab) {
  const size_t perRow = 2 * sizeof(double) + sizeof(int) + sizeof(float);
  if (vocab <= 0 || (size_t)vocab > (SIZE_MAX - 64) / perRow) return 0;
  return 64 + (size_t)vocab * perRow;
}

extern "C" int fcuda_bitplane_certified_argmax(
    const uint32_t* dPlanes, const float* dRowMin, const float* dRowScale,
    const float* dRowEps, const float* dRowMaxAbs, const float* dWeightsF32,
    const float* dX, void* dWorkspace, size_t workspaceBytes,
    int* dTokenOut, int* dRefinedOut,
    int vocab, int hiddenDim, int totalBits, int topBits) {
  if (!dPlanes || !dRowMin || !dRowScale || !dRowEps || !dRowMaxAbs ||
      !dWeightsF32 || !dX || !dWorkspace || !dTokenOut || !dRefinedOut ||
      vocab <= 0 || hiddenDim <= 0 || totalBits <= 0 || totalBits > 16 ||
      topBits <= 0 || topBits > totalBits || hiddenDim > INT_MAX / vocab ||
      ((uintptr_t)dWorkspace % alignof(double)) != 0) {
    return -1;
  }
  const size_t needed = fcuda_bitplane_certified_argmax_workspace_bytes(vocab);
  if (needed == 0 || workspaceBytes < needed) return -1;

  cudaGetLastError();

  const float* pX = dX;
  int pHidden = hiddenDim;
  void* pWs = dWorkspace;
  void* argsStats[] = {&pX, &pHidden, &pWs};
  cudaError_t launch = cudaSuccess;
  FCUDA_TS("k_bitplane_x_stats", dim3(1),
           launch = cudaLaunchKernel((const void*)k_bitplane_x_stats,
                                     dim3(1), dim3(BITPLANE_HEAD_THREADS),
                                     argsStats, 0, g_stream));
  if (launch != cudaSuccess) return 70000 + (int)launch;

  const uint32_t* pPlanes = dPlanes;
  const float* pMin = dRowMin;
  const float* pScale = dRowScale;
  const float* pEps = dRowEps;
  const float* pMaxAbs = dRowMaxAbs;
  int pVocab = vocab;
  int pTotalBits = totalBits;
  int pTopBits = topBits;
  void* argsBounds[] = {&pPlanes, &pMin, &pScale, &pEps, &pMaxAbs, &pX,
                        &pVocab, &pHidden, &pTotalBits, &pTopBits, &pWs};
  FCUDA_TS("k_bitplane_bounds_stage", dim3(vocab),
           launch = cudaLaunchKernel((const void*)k_bitplane_bounds_stage,
                                     dim3(vocab), dim3(BITPLANE_HEAD_THREADS),
                                     argsBounds, 0, g_stream));
  if (launch != cudaSuccess) return 70000 + (int)launch;

  int* pTok = dTokenOut;
  int* pRef = dRefinedOut;
  void* argsCert[] = {&pVocab, &pTok, &pRef, &pWs};
  FCUDA_TS("k_bitplane_certify_contenders", dim3(1),
           launch = cudaLaunchKernel((const void*)k_bitplane_certify_contenders,
                                     dim3(1), dim3(BITPLANE_HEAD_THREADS),
                                     argsCert, 0, g_stream));
  if (launch != cudaSuccess) return 70000 + (int)launch;

  const float* pW = dWeightsF32;
  const int refineBlocks = vocab < 256 ? vocab : 256;
  void* argsRefine[] = {&pW, &pX, &pVocab, &pHidden, &pWs};
  FCUDA_TS("k_bitplane_refine_contenders", dim3(refineBlocks),
           launch = cudaLaunchKernel((const void*)k_bitplane_refine_contenders,
                                     dim3(refineBlocks), dim3(BITPLANE_HEAD_THREADS),
                                     argsRefine, 0, g_stream));
  if (launch != cudaSuccess) return 70000 + (int)launch;

  void* argsPick[] = {&pVocab, &pTok, &pRef, &pWs};
  FCUDA_TS("k_bitplane_pick_winner", dim3(1),
           launch = cudaLaunchKernel((const void*)k_bitplane_pick_winner,
                                     dim3(1), dim3(BITPLANE_HEAD_THREADS),
                                     argsPick, 0, g_stream));
  return launch == cudaSuccess ? 0 : 70000 + (int)launch;
}
