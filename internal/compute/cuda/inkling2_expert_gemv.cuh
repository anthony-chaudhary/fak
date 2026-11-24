// inkling2_expert_gemv.cuh: bandwidth-shaped NVFP4 expert GEMV for the resident
// inkling2_loop decode. Included by inkling2_loop.cuh after k_ink2_expert_nvfp4,
// whose arithmetic it keeps bit for bit: lane l still owns 16-value blocks
// l, l+32, ... of every row, each block is the same 16-step fmaf chain scaled
// once, and the lane sums meet in the same shuffle tree. Only the schedule moves:
//   - the grid is sized to about 4 blocks per SM and every warp walks rows with a
//     grid stride, so the old 576-block grid's second, nearly empty wave is gone;
//   - a row's KB packed words and scale bytes are all issued before its math, the
//     next row's loads are issued before the current row's reduction, and the
//     first row's loads overlap the activation staging;
//   - the staged activation is swizzled per 16-value block so the float4 reads of
//     32 lanes (block stride 64 B) hit distinct shared-memory bank groups.
// FAK_INKLING2_NVFP4_GEMV=0 keeps the reference kernel.

#ifndef INK2_NV2_WARPS
#define INK2_NV2_WARPS 8
#endif
#ifndef INK2_NV2_ROWS
#define INK2_NV2_ROWS 1
#endif
#define INK2_NV2_MAX_KB 8
#ifndef INK2_NV2_BLOCKS_PER_SM
#define INK2_NV2_BLOCKS_PER_SM 4
#endif

// Physical float4 slot of activation float4 j (0..3) of 16-value block b.
static __device__ __forceinline__ int ink2_nv2_slot(int b, int j) { return b * 4 + ((j + (b >> 1)) & 3); }

template <int KB, int R>
__global__ void __launch_bounds__(INK2_NV2_WARPS * 32)
    k_ink2_expert_nvfp4_v2(const uint8_t *__restrict__ Packed, const uint8_t *__restrict__ Scales,
                           const int *ids, const float *X, int xStride, float *Y, int rows, int in,
                           const float *escale, int situW, double beta, double lb) {
  extern __shared__ float4 xs4[];
  __shared__ float lut[16];
  const int slot = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int nb = in >> 4;
  const size_t prow = (size_t)(in >> 1);
  const size_t ebase = (size_t)ids[slot] * rows;
  int o0 = (blockIdx.x * INK2_NV2_WARPS + (threadIdx.x >> 5)) * R;
  const int ostride = gridDim.x * INK2_NV2_WARPS * R;

  uint2 pk[KB][R];
  uint32_t sb[KB][R];
  auto load = [&](int base) {
#pragma unroll
    for (int k = 0; k < KB; ++k) {
      const int b = lane + 32 * k;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const size_t row = ebase + (size_t)(base + r);
        const bool ok = b < nb && base + r < rows;
        pk[k][r] = ok ? __ldcs((const uint2 *)(Packed + row * prow + (size_t)b * 8)) : make_uint2(0u, 0u);
        sb[k][r] = ok ? (uint32_t)__ldcs(Scales + row * (size_t)nb + b) : 0u;
      }
    }
  };
  if (o0 < rows) load(o0);

  const float *x = X + (size_t)slot * xStride;
  float *xsf = (float *)xs4;
  for (int i = threadIdx.x; i < in; i += blockDim.x) {
    const int b = i >> 4;
    xsf[ink2_nv2_slot(b, (i >> 2) & 3) * 4 + (i & 3)] = nv4_bf16_rne(ink2_stage(x, i, situW, beta, lb));
  }
  if (threadIdx.x < 16) lut[threadIdx.x] = nv4_e2m1(threadIdx.x);
  __syncthreads();

  const float es = escale ? escale[ids[slot]] : 0.f;
  while (o0 < rows) {
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0.f;
#pragma unroll
    for (int k = 0; k < KB; ++k) {
      const int b = lane + 32 * k;
      if (b < nb) {
        float xb[16];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float4 v = xs4[ink2_nv2_slot(b, j)];
          xb[4 * j] = v.x;
          xb[4 * j + 1] = v.y;
          xb[4 * j + 2] = v.z;
          xb[4 * j + 3] = v.w;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
          float part = 0.f;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const uint32_t word = j < 4 ? pk[k][r].x : pk[k][r].y;
            const uint32_t byte = (word >> (8 * (j & 3))) & 0xffu;
            part = fmaf(lut[byte & 0x0fu], xb[2 * j], part);
            part = fmaf(lut[byte >> 4], xb[2 * j + 1], part);
          }
          sum[r] = fmaf(part, nv4_e4m3fn(sb[k][r]), sum[r]);
        }
      }
    }
    const int cur = o0;
    o0 += ostride;
    if (o0 < rows) load(o0);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float v = sum[r];
      for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
      const int o = cur + r;
      if (lane == 0 && o < rows) Y[(size_t)slot * rows + o] = escale ? __fmul_rn(v, es) : v;
    }
  }
}

// Blocks per routed slot so the whole k-slot grid is about one resident wave.
static int ink2_nv2_blocks_per_slot(int rows, int k) {
  static int sms = 0;
  if (sms == 0) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0) sms = 84;
  }
  const int groups = (rows + INK2_NV2_ROWS - 1) / INK2_NV2_ROWS;
  const int needed = (groups + INK2_NV2_WARPS - 1) / INK2_NV2_WARPS;
  int g = (sms * INK2_NV2_BLOCKS_PER_SM + k - 1) / k;
  if (g > needed) g = needed;
  return g < 1 ? 1 : g;
}

static int g_ink2_nv2_mode = -1;

static bool ink2_nv2_enabled(void) {
  if (g_ink2_nv2_mode < 0) {
    const char *v = getenv("FAK_INKLING2_NVFP4_GEMV");
    g_ink2_nv2_mode = (v && v[0] == '0' && v[1] == 0) ? 0 : 1;
  }
  return g_ink2_nv2_mode == 1;
}

// Test hook: 1 selects the v2 kernel, 0 the reference; returns the previous mode.
extern "C" int fcuda_ink2_nvfp4_gemv_select(int mode) {
  const int prev = ink2_nv2_enabled() ? 1 : 0;
  g_ink2_nv2_mode = mode ? 1 : 0;
  return prev;
}

// Launches the v2 kernel; returns false (nothing launched) when the shape needs the
// reference kernel.
static bool ink2_expert_nvfp4_v2(const uint8_t *W, const uint8_t *scales, const int *ids, const float *X,
                                 int xStride, float *Y, int k, int rows, int in, const float *escale, int situW,
                                 double beta, double lb) {
  const int kb = ((in >> 4) + 31) / 32;
  if (!ink2_nv2_enabled() || kb < 1 || kb > INK2_NV2_MAX_KB) return false;
  const dim3 grid(ink2_nv2_blocks_per_slot(rows, k), k);
  const size_t smem = (size_t)in * sizeof(float);
#define INK2_NV2_CASE(KB)                                                                                     \
  case KB:                                                                                                    \
    FCUDA_TS("k_ink2_expert_nvfp4_v2", grid,                                                                  \
             k_ink2_expert_nvfp4_v2<KB, INK2_NV2_ROWS><<<grid, INK2_NV2_WARPS * 32, smem, g_stream>>>(        \
                 W, scales, ids, X, xStride, Y, rows, in, escale, situW, beta, lb));                          \
    break;
  switch (kb) {
    INK2_NV2_CASE(1)
    INK2_NV2_CASE(2)
    INK2_NV2_CASE(3)
    INK2_NV2_CASE(4)
    INK2_NV2_CASE(5)
    INK2_NV2_CASE(6)
    INK2_NV2_CASE(7)
    INK2_NV2_CASE(8)
  }
#undef INK2_NV2_CASE
  return true;
}

