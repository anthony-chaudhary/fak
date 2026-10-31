// ---- Q4_K q8_1 DP4A matrix-vector product for decode (fak#8635) -------------------------
// Adapted from llama.cpp ggml/src/ggml-cuda/mmvq.cu (mul_mat_vec_q, calc_nwarps,
// calc_rows_per_block), vecdotq.cuh (vec_dot_q4_K_q8_1, vec_dot_q4_K_q8_1_impl_vmmq) and
// quantize.cu (quantize_q8_1) at
// 0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd (MIT).
// Copyright (c) 2023-2026 The ggml authors. The MIT permission notice is reproduced in
// THIRD_PARTY_NOTICES.md ("llama.cpp Q4_K MMVQ decode GEMV").
//
// The default Q4_K instantiation uses the NVIDIA GENERIC parameter table (Ampere and newer,
// so sm_80/sm_86/sm_89/sm_90/sm_100/sm_120 all take it): 4 warps for 1..4 activation columns,
// one output row per CUDA block at one column and two rows otherwise, vdr 2. Other changes for
// Fak:
//   * No fusion (gate, bias, NVFP4 scale), MoE ids, channel or sample broadcast: one resident
//     Q4_K weight [out, in] times one f32 activation [P, in], P in 1..3, dst [P, out].
//   * With two rows per CUDA block the second row's weight read is clamped to the last row, so
//     an odd out never reads past the resident weight (upstream relies on row padding); its
//     result is still not written.
//   * The activation row is not padded to MATRIX_ROW_PADDING: Fak's Q4_K rows are a whole
//     number of 256-value super-blocks, so every q8_1 block the dot reads exists.
//   * The q8_1 scratch is a persistent grown-once buffer, like the Q8_0 GEMV scratch (#969),
//     because decode is captured into a CUDA graph; ggml_cuda_pdl_* is dropped.

namespace fak_q4k_mmvq {

constexpr int warp_size = 32;
constexpr int QK_K = 256;
constexpr int QK8_1 = 32;
constexpr int QR4_K = 2;
constexpr int QI4_K = QK_K / (4 * QR4_K);
constexpr int QI8_1 = QK8_1 / 4;
constexpr int VDR_Q4_K_Q8_1_MMVQ = 2;
constexpr int max_cols = 3;

struct block_q4_K {
    half2   dm;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144, "Q4_K super-block is 144 bytes");

struct block_q8_1 {
    half2  ds;          // d, sum of the 32 unquantized values
    int8_t qs[QK8_1];
};
static_assert(sizeof(block_q8_1) == 36, "q8_1 block is 36 bytes");

template <int width>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset, width);
    return x;
}

template <int width>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    return x;
}

// quantize.cu quantize_q8_1: one thread per value, a 32-lane group per q8_1 block.
static __global__ void __launch_bounds__(256, 1)
quantize_q8_1(const float *__restrict__ x, block_q8_1 *__restrict__ y, int ne0) {
    const int i0 = blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= ne0) return;
    const int i1 = blockIdx.y;
    const int64_t i_cont = (int64_t)i1 * ne0 + i0;
    const int64_t ib = i_cont / QK8_1;
    const int iqs = (int)(i_cont % QK8_1);
    const float xi = x[(size_t)i1 * ne0 + i0];
    float amax = fabsf(xi);
    float sum = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum = warp_reduce_sum<QK8_1>(sum);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    y[ib].qs[iqs] = q;
    if (iqs > 0) return;
    y[ib].ds = make_half2(d, sum);
}

// vecdotq.cuh vec_dot_q4_K_q8_1_impl_vmmq (contiguous v/x values).
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int *__restrict__ v, const int *__restrict__ u, const uint8_t *__restrict__ sc,
    const uint8_t *__restrict__ m, const half2 &dm4, const float *__restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i + 0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

// vecdotq.cuh vec_dot_q4_K_q8_1.
static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1, const int &kbx, const int &iqs) {
    const block_q4_K *bq4_K = (const block_q4_K *)vbq + kbx;
    int v[2];
    int u[2 * QR4_K];
    float d8[QR4_K];
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int *q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];
    const uint16_t *scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t *sc = (const uint8_t *)aux;
    const uint8_t *m = sc + 2;
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}


// Additional Q5_K/Q6_K dots from vecdotq.cuh at the same immutable llama.cpp
// pin above (MIT, Copyright (c) 2023-2026 The ggml authors; full license in
// THIRD_PARTY_NOTICES.md). Modified only for local intrinsics/load names.
// Selection is qualification-only: q8_1 changes F32-activation precision.
#ifdef FAK_CUDA_Q56K_MMVQ_QUALIFICATION
constexpr int QR5_K = 2, QR6_K = 2, QI6_K = 32;
struct block_q5_K { half2 dm; uint8_t scales[12], qh[32], qs[128]; };
struct block_q6_K { uint8_t ql[128], qh[64]; int8_t scales[16]; half d; };
static_assert(sizeof(block_q5_K) == 176, "canonical Q5_K");
static_assert(sizeof(block_q6_K) == 210, "canonical Q6_K");
// A Q6_K block is only two-byte aligned. Preserve the donor's two-word load,
// never turn its 210-byte stride into an unaligned int dereference.
static __device__ __forceinline__ int q56_get_int_b2(const void *p, int i) {
    const uint16_t *v = (const uint16_t *)p;
    return (int)((uint32_t)v[2*i] | ((uint32_t)v[2*i+1] << 16));
}
static __device__ __forceinline__ int q56_get_int_b4(const void *p, int i) {
    return ((const int *)p)[i];
}
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = __dp4a(v0i, u[2*i+0], __dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = __dp4a(0x01010101, u[2*i+0], __dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = q56_get_int_b2(bq6_K->ql, iqs);
    const int vh = q56_get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = q56_get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}
struct dot_q5 {
    static constexpr int qi = 32, vdr = 2;
    static __device__ __forceinline__ float run(const void *w, const block_q8_1 *x, int b, int q) {
        return vec_dot_q5_K_q8_1(w, x, b, q);
    }
};
struct dot_q6 {
    static constexpr int qi = 32, vdr = 1;
    static __device__ __forceinline__ float run(const void *w, const block_q8_1 *x, int b, int q) {
        return vec_dot_q6_K_q8_1(w, x, b, q);
    }
};
#endif
struct dot_q4 {
    static constexpr int qi = QI4_K, vdr = VDR_Q4_K_Q8_1_MMVQ;
    static __device__ __forceinline__ float run(const void *w, const block_q8_1 *x, int b, int q) {
        return vec_dot_q4_K_q8_1(w, x, b, q);
    }
};

// mmvq.cu calc_nwarps / calc_rows_per_block, MMVQ_PARAMETERS_GENERIC rows for ncols_dst 1..4.
static constexpr __host__ __device__ int calc_nwarps(int ncols_dst) { return ncols_dst <= 4 ? 4 : 2; }
static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst) { return ncols_dst == 1 ? 1 : 2; }

// mmvq.cu mul_mat_vec_q, Q4_K, no fusion, one channel and sample.
template <int ncols_dst, typename Dot = dot_q4>
__launch_bounds__(calc_nwarps(ncols_dst) * warp_size, 1)
static __global__ void mul_mat_vec_q(const void *__restrict__ vx, const block_q8_1 *__restrict__ y,
                                     float *__restrict__ dst, const int ncols_x, const int nrows_x) {
    constexpr int qk = QK_K;
    constexpr int qi = Dot::qi;
    constexpr int vdr = Dot::vdr;
    constexpr int nwarps = calc_nwarps(ncols_dst);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst);

    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block * blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const int stride_row_x = blocks_per_row_x;
    const int stride_col_y = ncols_x / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (tid % (qi / vdr));
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                const int row = min(row0 + i, nrows_x - 1);
                tmp[j][i] += Dot::run(vx, &y[j * stride_col_y + kby], row * stride_row_x + kbx, kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j)
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    dst += row0;
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps - 1; ++l) tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if (threadIdx.x == i && (rows_per_cuda_block == 1 || row0 + i < nrows_x)) dst[j * nrows_x + i] = tmp[j][i];
        }
    }
}

} // namespace fak_q4k_mmvq

// Persistent q8_1 activation scratch for the decode GEMV, grown once (see the #969 note on the
// Q8_0 GEMV scratch: a fresh pooled size class mid graph-capture broke replay).
static fak_q4k_mmvq::block_q8_1 *g_q4k_mmvq_y = nullptr;
static size_t g_q4k_mmvq_y_cap = 0; // bytes
static unsigned long long g_q4k_mmvq_launches = 0;

extern "C" unsigned long long fcuda_q4k_mmvq_launch_count(void) { return g_q4k_mmvq_launches; }

// fcuda_q4k_mmvq_matmul is the mmvq decode arm of fcuda_q4k_matmul_f32 (cuda_backend.h):
// Y[P,out] = X[P,in] @ W[out,in]^T for P in 1..3. It quantizes X to q8_1 per 32 values on
// device and runs llama.cpp's DP4A Q4_K dot. Returns 0, FCUDA_SCRATCH_ALLOC_FAILED, or
// FCUDA_Q4K_MMVQ_REFUSED for a shape this arm does not take (nothing launched on either).
#define FCUDA_Q4K_MMVQ_REFUSED 80006
static int fcuda_q4k_mmvq_matmul(const uint8_t *dQ4K, const float *dX, float *dY, int out, int in, int P) {
#ifdef FAK_CUDA_V41_CHECKED_STREAM
  if (v41_candidates_stream_uncertain()) return FCUDA_Q4K_MMVQ_REFUSED;
#endif
    using namespace fak_q4k_mmvq;
    if (!dQ4K || !dX || !dY || out <= 0 || in <= 0 || in % QK_K != 0 || P < 1 || P > max_cols)
        return FCUDA_Q4K_MMVQ_REFUSED;
    size_t need = (size_t)P * (in / QK8_1) * sizeof(block_q8_1);
    if (need > g_q4k_mmvq_y_cap) {
        if (g_q4k_mmvq_y) {
            fak_alloc_untrack(g_q4k_mmvq_y);
            cudaFree(g_q4k_mmvq_y);
        }
        g_q4k_mmvq_y = nullptr;
        g_q4k_mmvq_y_cap = 0;
        if (fcuda_alloc_status(cudaMalloc(&g_q4k_mmvq_y, need)) != cudaSuccess || !g_q4k_mmvq_y) {
            g_q4k_mmvq_y = nullptr;
            return FCUDA_SCRATCH_ALLOC_FAILED;
        }
        fak_alloc_track(g_q4k_mmvq_y, FAK_ALLOC_ACTIVATIONS_SCRATCH, FAK_ALLOC_DEVICE_LOCAL,
                        FAK_ALLOC_SITE("q4k-mmvq-q8_1-activations"), need);
        g_q4k_mmvq_y_cap = need;
    }
    dim3 qgrid((in + 255) / 256, P);
    FCUDA_TS("quantize_q8_1", qgrid, quantize_q8_1<<<qgrid, 256, 0, g_stream>>>(dX, g_q4k_mmvq_y, in));
    switch (P) {
    case 1: {
        dim3 grid((out + calc_rows_per_block(1) - 1) / calc_rows_per_block(1)), block(warp_size, calc_nwarps(1));
        FCUDA_TS("k_q4k_mmvq_gemv", grid, mul_mat_vec_q<1><<<grid, block, 0, g_stream>>>(dQ4K, g_q4k_mmvq_y, dY, in, out));
        break;
    }
    case 2: {
        dim3 grid((out + calc_rows_per_block(2) - 1) / calc_rows_per_block(2)), block(warp_size, calc_nwarps(2));
        FCUDA_TS("k_q4k_mmvq_gemv", grid, mul_mat_vec_q<2><<<grid, block, 0, g_stream>>>(dQ4K, g_q4k_mmvq_y, dY, in, out));
        break;
    }
    default: {
        dim3 grid((out + calc_rows_per_block(3) - 1) / calc_rows_per_block(3)), block(warp_size, calc_nwarps(3));
        FCUDA_TS("k_q4k_mmvq_gemv", grid, mul_mat_vec_q<3><<<grid, block, 0, g_stream>>>(dQ4K, g_q4k_mmvq_y, dY, in, out));
        break;
    }
    }
    ++g_q4k_mmvq_launches;
    return 0;
}
