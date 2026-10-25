// Routed-expert FP8 block-32 / MXFP4 E8M0 GEMV kernels (RoutedExpertDeviceKernel).
// Included by cuda_kernels.cu so the kernels share g_stream, the pooled allocator and
// the transfer witnesses of the single libfakcuda object.
//
// Contract: bit-identical to the Go oracles GemvFP8Block32E8M0 / GemvMXFP4E8M0 for every
// non-NaN output (a NaN output is NaN on both sides; IEEE leaves its payload unspecified).
//   - One thread owns one output row and accumulates it serially in K order, starting
//     from +0, exactly as the oracle's `acc += float32(w * x[i])` loop.
//   - Every multiply and add is inline PTX mul.rn.f32 / add.rn.f32: the explicit .rn
//     rounding forbids FMA contraction and the absent .ftz keeps IEEE subnormals no
//     matter what -ftz / --use_fast_math the translation unit is compiled with.
//   - E4M3, E2M1 and E8M0 are decoded by exact bit construction (E8M0 byte 0 is the
//     subnormal 2^-127, 0x00400000). The host refuses E8M0 0xff and E4M3 NaN bytes
//     before launch, so the kernel never sees them.
// Layout: a block owns 32 output rows (one FP8 tile height). Per K chunk its 256 threads
// stage a 32-row x 128-byte packed tile (128 FP8 or 256 MXFP4 K values) into shared memory
// with 16-byte coalesced loads, plus the x slice and decoded scales; the next chunk is
// prefetched into registers while warp 0 accumulates the current one from shared memory.
// SIMT only, so one source serves sm_80..sm_120.

#include <chrono>

#define REX_ROWS 32
#define REX_THREADS 256
#define REX_TILE_BYTES 128
#define REX_TILE_WORDS (REX_TILE_BYTES / 4)
#define REX_STRIDE_WORDS (REX_TILE_WORDS + 1) // +1 word: row-per-thread smem reads are conflict-free
#define REX_FP8_CHUNK_K REX_TILE_BYTES        // one byte per K value
#define REX_MX_CHUNK_K (2 * REX_TILE_BYTES)   // two nibbles per byte
#define REX_MX_GROUPS (REX_MX_CHUNK_K / 32)   // E8M0 scale groups per row per chunk
#define REX_MX_SCALE_STRIDE (REX_MX_GROUPS + 1)

#define REX_FORMAT_FP8_BLOCK32 0
#define REX_FORMAT_MXFP4 1

static __device__ __forceinline__ float rex_mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

static __device__ __forceinline__ float rex_add_rn(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// 2^(b-127) for b in [0,254]; byte 0 is the f32 subnormal 2^-127.
static __device__ __forceinline__ float rex_e8m0(uint32_t b) {
  return __uint_as_float(b ? (b << 23) : 0x00400000u);
}

// float8_e4m3fn (bias 7, no Inf); NaN bytes are refused on the host.
static __device__ __forceinline__ float rex_e4m3(uint32_t b) {
  const uint32_t sign = (b & 0x80u) << 24;
  const uint32_t e = (b >> 3) & 0x0fu;
  const uint32_t m = b & 0x07u;
  uint32_t mag;
  if (e != 0u) {
    mag = ((e + 120u) << 23) | (m << 20);
  } else {
    // m * 2^-9 is a normal f32 for m in 1..7 and exact in any rounding mode.
    mag = __float_as_uint(rex_mul_rn((float)m, 0.001953125f));
  }
  return __uint_as_float(sign | mag);
}

// OCP E2M1 (1 sign, 2 exponent bias 1, 1 mantissa): {0,.5,1,1.5,2,3,4,6} and negatives,
// nibble 8 is -0 like the oracle's table.
static __device__ __forceinline__ float rex_e2m1(uint32_t nib) {
  const uint32_t sign = (nib & 0x8u) << 28;
  const uint32_t e = (nib >> 1) & 0x3u;
  const uint32_t m = nib & 0x1u;
  const uint32_t mag = e ? (((e + 126u) << 23) | (m << 22)) : (m ? 0x3f000000u : 0u);
  return __uint_as_float(sign | mag);
}

// Thread t owns tile row t/8, bytes [(t%8)*16, +16) of a 32-row x 128-byte tile whose
// rows start at W + (row0+r)*rowBytes + b0. Bytes past nb or rows past `rows` read as 0
// and no address outside the matrix is formed. The 16-byte vector path needs a 16-byte
// row pitch: the pooled cudaMalloc base is 256-byte aligned and b0 is a multiple of 128.
static __device__ __forceinline__ void rex_load_tile(
    const uint8_t *__restrict__ W, size_t rowBytes, size_t row0, int rows,
    size_t b0, int nb, uint32_t v[4]) {
  const int tid = (int)threadIdx.x;
  const int r = tid >> 3;
  const int c0 = (tid & 7) << 4;
  v[0] = v[1] = v[2] = v[3] = 0u;
  if (r >= rows || c0 >= nb) return;
  const uint8_t *p = W + (row0 + (size_t)r) * rowBytes + b0 + (size_t)c0;
  if (c0 + 16 <= nb && (rowBytes & 15u) == 0u) {
    const uint4 q = __ldg(reinterpret_cast<const uint4 *>(p));
    v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
    return;
  }
  const int n = min(16, nb - c0);
  for (int j = 0; j < n; ++j) v[j >> 2] |= (uint32_t)__ldg(p + j) << ((j & 3) << 3);
}

static __device__ __forceinline__ void rex_store_tile(uint32_t *sW, const uint32_t v[4]) {
  const int tid = (int)threadIdx.x;
  uint32_t *dst = sW + (tid >> 3) * REX_STRIDE_WORDS + ((tid & 7) << 2);
  dst[0] = v[0]; dst[1] = v[1]; dst[2] = v[2]; dst[3] = v[3];
}

__global__ void __launch_bounds__(REX_THREADS) k_rex_gemv_fp8_block32_e8m0(
    const uint8_t *__restrict__ W, const uint8_t *__restrict__ S,
    const float *__restrict__ X, float *__restrict__ Y, long long O, long long I) {
  __shared__ uint32_t sW[REX_ROWS * REX_STRIDE_WORDS];
  __shared__ float sX[REX_FP8_CHUNK_K];
  __shared__ float sS[REX_FP8_CHUNK_K / 32];
  const int tid = (int)threadIdx.x;
  const size_t rowTile = (size_t)blockIdx.x;
  const size_t row0 = rowTile * REX_ROWS;
  const int rows = (int)min((long long)REX_ROWS, O - (long long)row0);
  const long long sI = (I + 31) / 32;
  const long long chunks = (I + REX_FP8_CHUNK_K - 1) / REX_FP8_CHUNK_K;
  const uint8_t *scaleRow = S + rowTile * (size_t)sI;

  uint32_t v[4];
  float xv = 0.0f;
  uint32_t sb = 127u;
  rex_load_tile(W, (size_t)I, row0, rows, 0, (int)min((long long)REX_FP8_CHUNK_K, I), v);
  if (tid < REX_FP8_CHUNK_K && tid < I) xv = __ldg(X + tid);
  if (tid < REX_FP8_CHUNK_K / 32 && tid < sI) sb = __ldg(scaleRow + tid);

  float acc = 0.0f;
  for (long long ch = 0; ch < chunks; ++ch) {
    const long long k0 = ch * REX_FP8_CHUNK_K;
    const int n = (int)min((long long)REX_FP8_CHUNK_K, I - k0);
    if (ch > 0) __syncthreads(); // warp 0 finished reading the previous chunk
    rex_store_tile(sW, v);
    if (tid < REX_FP8_CHUNK_K) sX[tid] = xv;
    if (tid < REX_FP8_CHUNK_K / 32) sS[tid] = rex_e8m0(sb);
    __syncthreads();
    if (ch + 1 < chunks) {
      const long long k1 = k0 + REX_FP8_CHUNK_K;
      rex_load_tile(W, (size_t)I, row0, rows, (size_t)k1, (int)min((long long)REX_FP8_CHUNK_K, I - k1), v);
      xv = 0.0f;
      if (tid < REX_FP8_CHUNK_K && k1 + tid < I) xv = __ldg(X + k1 + tid);
      sb = 127u;
      const long long j = k1 / 32 + tid;
      if (tid < REX_FP8_CHUNK_K / 32 && j < sI) sb = __ldg(scaleRow + j);
    }
    if (tid < rows) {
      const uint32_t *rowW = sW + tid * REX_STRIDE_WORDS;
      const int words = n >> 2;
      for (int wi = 0; wi < words; ++wi) {
        const uint32_t word = rowW[wi];
        const float s = sS[wi >> 3];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float w = rex_mul_rn(rex_e4m3((word >> (q << 3)) & 0xffu), s);
          acc = rex_add_rn(acc, rex_mul_rn(w, sX[(wi << 2) + q]));
        }
      }
      for (int c = words << 2; c < n; ++c) {
        const uint32_t b = (rowW[c >> 2] >> ((c & 3) << 3)) & 0xffu;
        const float w = rex_mul_rn(rex_e4m3(b), sS[c >> 5]);
        acc = rex_add_rn(acc, rex_mul_rn(w, sX[c]));
      }
    }
  }
  if (tid < rows) Y[row0 + (size_t)tid] = acc;
}

__global__ void __launch_bounds__(REX_THREADS) k_rex_gemv_mxfp4_e8m0(
    const uint8_t *__restrict__ W, const uint8_t *__restrict__ S,
    const float *__restrict__ X, float *__restrict__ Y, long long O, long long I) {
  __shared__ uint32_t sW[REX_ROWS * REX_STRIDE_WORDS];
  __shared__ float sX[REX_MX_CHUNK_K];
  __shared__ float sS[REX_ROWS * REX_MX_SCALE_STRIDE];
  const int tid = (int)threadIdx.x;
  const size_t row0 = (size_t)blockIdx.x * REX_ROWS;
  const int rows = (int)min((long long)REX_ROWS, O - (long long)row0);
  const long long rowBytes = I / 2;
  const long long scaleCols = (I + 31) / 32;
  const long long chunks = (I + REX_MX_CHUNK_K - 1) / REX_MX_CHUNK_K;
  // Scale staging: thread t owns row t/8, group t%8 of the chunk.
  const int sr = tid >> 3;
  const int sg = tid & 7;

  uint32_t v[4];
  float xv = 0.0f;
  uint32_t sb = 127u;
  rex_load_tile(W, (size_t)rowBytes, row0, rows, 0, (int)min((long long)REX_TILE_BYTES, rowBytes), v);
  if (tid < I) xv = __ldg(X + tid);
  if (sr < rows && sg < scaleCols) sb = __ldg(S + (row0 + (size_t)sr) * (size_t)scaleCols + (size_t)sg);

  float acc = 0.0f;
  for (long long ch = 0; ch < chunks; ++ch) {
    const long long b0 = ch * REX_TILE_BYTES;
    const int nb = (int)min((long long)REX_TILE_BYTES, rowBytes - b0);
    if (ch > 0) __syncthreads();
    rex_store_tile(sW, v);
    sX[tid] = xv;
    sS[sr * REX_MX_SCALE_STRIDE + sg] = rex_e8m0(sb);
    __syncthreads();
    if (ch + 1 < chunks) {
      const long long b1 = b0 + REX_TILE_BYTES;
      const long long k1 = 2 * b1;
      rex_load_tile(W, (size_t)rowBytes, row0, rows, (size_t)b1, (int)min((long long)REX_TILE_BYTES, rowBytes - b1), v);
      xv = 0.0f;
      if (k1 + tid < I) xv = __ldg(X + k1 + tid);
      sb = 127u;
      const long long j = k1 / 32 + sg;
      if (sr < rows && j < scaleCols) sb = __ldg(S + (row0 + (size_t)sr) * (size_t)scaleCols + (size_t)j);
    }
    if (tid < rows) {
      const uint32_t *rowW = sW + tid * REX_STRIDE_WORDS;
      const float *rowS = sS + tid * REX_MX_SCALE_STRIDE;
      const int words = nb >> 2;
      for (int wi = 0; wi < words; ++wi) {
        const uint32_t word = rowW[wi];
        const float s = rowS[wi >> 2]; // 8 K values per word, 32 per scale group
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          // byte j of a row holds K=2j in its low nibble and K=2j+1 in its high nibble,
          // so K value c of the word sits at bit 4*(c%8).
          const float w = rex_mul_rn(rex_e2m1((word >> (q << 2)) & 0xfu), s);
          acc = rex_add_rn(acc, rex_mul_rn(w, sX[(wi << 3) + q]));
        }
      }
      for (int c = words << 3; c < 2 * nb; ++c) {
        const uint32_t nib = (rowW[c >> 3] >> ((c & 7) << 2)) & 0xfu;
        const float w = rex_mul_rn(rex_e2m1(nib), rowS[c >> 5]);
        acc = rex_add_rn(acc, rex_mul_rn(w, sX[c]));
      }
    }
  }
  if (tid < rows) Y[row0 + (size_t)tid] = acc;
}

static bool rex_mul_size(size_t a, size_t b, size_t *out) {
  if (a != 0 && b > SIZE_MAX / a) return false;
  *out = a * b;
  return true;
}

static double rex_now_ms(void) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

static size_t g_rex_launches = 0;

// One-shot deterministic fault seam, armed only through the unexported Go test helper
// routedExpertInjectFaultForTest: ALLOC fails the next call's device allocation (-3);
// LAUNCH rejects its launch with cudaErrorInvalidConfiguration before any kernel runs.
#define REX_TEST_FAULT_ALLOC 1
#define REX_TEST_FAULT_LAUNCH 2
static int g_rex_test_fault = 0;

static bool rex_take_test_fault(int kind) {
  if (g_rex_test_fault != kind) return false;
  g_rex_test_fault = 0;
  return true;
}

// rex_geom is the validated byte geometry of one [O,I] routed-expert GEMV.
struct rex_geom {
  size_t weight, scale, x, y, blocks;
};

// rex_geometry validates format and shape and sizes every operand; false means status -1.
static bool rex_geometry(int format, int64_t O, int64_t I, rex_geom *g) {
  if (O <= 0 || I <= 0) return false;
  if (format != REX_FORMAT_FP8_BLOCK32 && format != REX_FORMAT_MXFP4) return false;
  const size_t o = (size_t)O, i = (size_t)I;
  const size_t scaleCols = (i + 31) / 32;
  if (format == REX_FORMAT_FP8_BLOCK32) {
    if (!rex_mul_size(o, i, &g->weight) || !rex_mul_size((o + 31) / 32, scaleCols, &g->scale)) return false;
  } else {
    if (i % 2 != 0) return false;
    if (!rex_mul_size(o, i / 2, &g->weight) || !rex_mul_size(o, scaleCols, &g->scale)) return false;
  }
  if (!rex_mul_size(i, sizeof(float), &g->x) || !rex_mul_size(o, sizeof(float), &g->y)) return false;
  g->blocks = (o + REX_ROWS - 1) / REX_ROWS;
  return g->blocks <= (size_t)INT_MAX;
}

// rex_run is the one synchronous GEMV both ABIs share: (upload the packed operands,) upload
// x, launch, fence, read y back. Host weight/scales non-null is the per-call path: they are
// copied into pooled buffers freed on return. Null means resW/resS already hold the resident
// packed bytes and only x and y move. Statuses as fcuda_routed_expert_gemv_e8m0.
static int rex_run(int format, const rex_geom &g, const uint8_t *weight, const uint8_t *scales,
                   const void *resW, const void *resS, const float *x, float *y,
                   int64_t O, int64_t I, double *timing_ms) {
  const bool upload = weight != nullptr;
  void *dW = upload ? fcuda_malloc(g.weight) : nullptr;
  void *dS = upload ? fcuda_malloc(g.scale) : nullptr;
  void *dX = fcuda_malloc(g.x);
  void *dY = fcuda_malloc(g.y);
  const void *kW = upload ? dW : resW;
  const void *kS = upload ? dS : resS;
  int status = 0;
  double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
  float kernelMs = 0.0f;
  static cudaEvent_t ev[2] = {nullptr, nullptr};
  cudaError_t e = cudaSuccess;
  if (!kW || !kS || !dX || !dY || rex_take_test_fault(REX_TEST_FAULT_ALLOC)) { status = -3; goto done; }
  (void)cudaGetLastError(); // discard stale status before attributing this call
  if (timing_ms) {
    for (int k = 0; k < 2; ++k) {
      if (!ev[k] && (e = cudaEventCreate(&ev[k])) != cudaSuccess) { status = (int)e; goto done; }
    }
    t0 = rex_now_ms();
  }
  if (upload) {
    fcuda_ts_untimed_transfer();
    if ((e = cudaMemcpy(dW, weight, g.weight, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
    g_h2d_bytes += g.weight;
    fcuda_ts_untimed_transfer();
    if ((e = cudaMemcpy(dS, scales, g.scale, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
    g_h2d_bytes += g.scale;
  }
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dX, x, g.x, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += g.x;
  if (timing_ms) {
    if ((e = cudaStreamSynchronize(0)) != cudaSuccess) { status = (int)e; goto done; }
    t1 = rex_now_ms();
    if ((e = cudaEventRecord(ev[0], g_stream)) != cudaSuccess) { status = (int)e; goto done; }
  }
  if (rex_take_test_fault(REX_TEST_FAULT_LAUNCH)) { status = (int)cudaErrorInvalidConfiguration; goto done; }
  if (format == REX_FORMAT_FP8_BLOCK32) {
    FCUDA_TS("k_rex_gemv_fp8_block32_e8m0", (unsigned int)g.blocks, k_rex_gemv_fp8_block32_e8m0<<<(unsigned int)g.blocks, REX_THREADS, 0, g_stream>>>(
        (const uint8_t *)kW, (const uint8_t *)kS, (const float *)dX, (float *)dY, (long long)O, (long long)I));
  } else {
    FCUDA_TS("k_rex_gemv_mxfp4_e8m0", (unsigned int)g.blocks, k_rex_gemv_mxfp4_e8m0<<<(unsigned int)g.blocks, REX_THREADS, 0, g_stream>>>(
        (const uint8_t *)kW, (const uint8_t *)kS, (const float *)dX, (float *)dY, (long long)O, (long long)I));
  }
  if ((e = cudaGetLastError()) != cudaSuccess) { status = (int)e; goto done; }
  if (timing_ms && (e = cudaEventRecord(ev[1], g_stream)) != cudaSuccess) { status = (int)e; goto done; }
  if ((e = cudaStreamSynchronize(g_stream)) != cudaSuccess) { status = (int)e; goto done; }
  ++g_rex_launches;
  if (timing_ms) {
    if ((e = cudaEventElapsedTime(&kernelMs, ev[0], ev[1])) != cudaSuccess) { status = (int)e; goto done; }
    t2 = rex_now_ms();
  }
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(y, dY, g.y, cudaMemcpyDeviceToHost)) != cudaSuccess) { status = (int)e; goto done; }
  g_host_bytes += g.y;
  if (timing_ms) {
    t3 = rex_now_ms();
    timing_ms[0] = t1 - t0;
    timing_ms[1] = (double)kernelMs;
    timing_ms[2] = t3 - t2;
  }
done:
  fcuda_free(dW);
  fcuda_free(dS);
  fcuda_free(dX);
  fcuda_free(dY);
  return status;
}

// One synchronous routed-expert GEMV: upload the PACKED weight/scale bytes and x, launch,
// fence, read y back. Returns 0 on success; -1 invalid geometry/pointers; -2 refused while a
// CUDA graph capture is open; -3 device allocation failure; a positive cudaError_t for any
// copy, launch or execution fault. y is meaningful only when 0 is returned. timing_ms
// (optional, 3 entries) receives host-wall H2D ms, device-event kernel ms, host-wall D2H ms.
extern "C" int fcuda_routed_expert_gemv_e8m0(
    int format, const uint8_t *weight, size_t weight_bytes,
    const uint8_t *scales, size_t scale_bytes, const float *x, float *y,
    int64_t O, int64_t I, double *timing_ms) {
  if (!weight || !scales || !x || !y) return -1;
  rex_geom g;
  if (!rex_geometry(format, O, I, &g)) return -1;
  if (weight_bytes != g.weight || scale_bytes != g.scale) return -1;
  if (g_capture_open) return -2;
  return rex_run(format, g, weight, scales, nullptr, nullptr, x, y, O, I, timing_ms);
}

extern "C" size_t fcuda_routed_expert_gemv_launches(void) {
  return g_rex_launches;
}

// Resident routed-expert weights: pooled device copies of one projection's packed weight and
// scale bytes, named by an id that is never reused, so a stale or released id is refused (-1)
// rather than aliasing a later upload. Callers serialize every entry point (Go holds cudaMu).
struct rex_resident {
  int format;
  int64_t O, I;
  void *dW, *dS;
};
static std::unordered_map<uint64_t, rex_resident> g_rex_resident;
static uint64_t g_rex_resident_next = 0;
static size_t g_rex_resident_uploads = 0;

// Upload one projection's PACKED bytes once and return its id in *handle. Returns 0, -1
// (geometry/pointers), -2 (graph capture open), -3 (allocation) or a positive cudaError_t;
// on any non-zero status nothing stays allocated and *handle is 0.
extern "C" int fcuda_routed_expert_resident_upload(
    int format, const uint8_t *weight, size_t weight_bytes,
    const uint8_t *scales, size_t scale_bytes, int64_t O, int64_t I, uint64_t *handle) {
  if (!handle) return -1;
  *handle = 0;
  if (!weight || !scales) return -1;
  rex_geom g;
  if (!rex_geometry(format, O, I, &g)) return -1;
  if (weight_bytes != g.weight || scale_bytes != g.scale) return -1;
  if (g_capture_open) return -2;
  void *dW = fcuda_malloc(weight_bytes);
  void *dS = fcuda_malloc(scale_bytes);
  cudaError_t e = cudaSuccess;
  if (!dW || !dS || rex_take_test_fault(REX_TEST_FAULT_ALLOC)) {
    fcuda_free(dW);
    fcuda_free(dS);
    return -3;
  }
  (void)cudaGetLastError();
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dW, weight, weight_bytes, cudaMemcpyHostToDevice)) == cudaSuccess) {
    g_h2d_bytes += weight_bytes;
    fcuda_ts_untimed_transfer();
    if ((e = cudaMemcpy(dS, scales, scale_bytes, cudaMemcpyHostToDevice)) == cudaSuccess) g_h2d_bytes += scale_bytes;
  }
  if (e != cudaSuccess) {
    fcuda_free(dW);
    fcuda_free(dS);
    return (int)e;
  }
  *handle = ++g_rex_resident_next;
  g_rex_resident[*handle] = rex_resident{format, O, I, dW, dS};
  ++g_rex_resident_uploads;
  return 0;
}

// GEMV over a resident id: only x and y cross the bus. format, O and I must match the
// upload. Statuses as fcuda_routed_expert_gemv_e8m0, plus -1 for an unknown id.
extern "C" int fcuda_routed_expert_resident_gemv_e8m0(
    uint64_t handle, int format, const float *x, float *y, int64_t O, int64_t I, double *timing_ms) {
  if (!x || !y) return -1;
  auto it = g_rex_resident.find(handle);
  if (it == g_rex_resident.end()) return -1;
  const rex_resident &r = it->second;
  if (r.format != format || r.O != O || r.I != I) return -1;
  rex_geom g;
  if (!rex_geometry(format, O, I, &g)) return -1;
  if (g_capture_open) return -2;
  return rex_run(format, g, nullptr, nullptr, r.dW, r.dS, x, y, O, I, timing_ms);
}

// Return a resident id's buffers to the pool. Returns 0, -1 (unknown id) or -2 (graph capture
// open: the id stays resident and usable).
extern "C" int fcuda_routed_expert_resident_release(uint64_t handle) {
  auto it = g_rex_resident.find(handle);
  if (it == g_rex_resident.end()) return -1;
  if (g_capture_open) return -2;
  fcuda_free(it->second.dW);
  fcuda_free(it->second.dS);
  g_rex_resident.erase(it);
  return 0;
}

// Witnesses: cumulative successful resident uploads, and ids currently resident.
extern "C" size_t fcuda_routed_expert_resident_uploads(void) {
  return g_rex_resident_uploads;
}

extern "C" size_t fcuda_routed_expert_resident_live(void) {
  return g_rex_resident.size();
}

extern "C" void fcuda_routed_expert_gemv_test_fault(int kind) {
  g_rex_test_fault = kind;
}
