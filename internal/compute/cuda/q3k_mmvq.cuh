// ---- Q3_K q8_1 DP4A matrix-vector product for decode + Q3_K dequant ----------------------
// Adapted from llama.cpp ggml/src/ggml-cuda/vecdotq.cuh (get_int_b2, get_int_b4,
// VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1_impl_mmvq, vec_dot_q3_K_q8_1), mmvq.cu
// (mul_mat_vec_q, Q3_K table entries), convert.cu (dequantize_block_q3_K and its 64-thread
// launcher) and ggml-common.h (block_q3_K) at
// 571d0d540df04f25298d0e159e520d9fc62ed121 (MIT).
// Copyright (c) 2023-2026 The ggml authors. The MIT permission notice is reproduced in
// THIRD_PARTY_NOTICES.md ("llama.cpp Q3_K MMVQ decode GEMV").
//
// Only the Q3_K instantiation on the NVIDIA GENERIC parameter table is kept (Ampere and newer,
// so sm_80/sm_86/sm_89/sm_90/sm_100/sm_120 all take it): 4 warps for 1..4 activation columns,
// one output row per CUDA block at one column and two rows otherwise, vdr 1. Other changes for
// Fak:
//   * The q8_1 activation block, quantize_q8_1, warp_reduce_sum and the GENERIC nwarps /
//     rows-per-block helpers are the ones already ported in cuda/q4k_mmvq.cuh (same upstream
//     code); this file reuses them instead of porting a second copy.
//   * No fusion (gate, bias), MoE ids, channel or sample broadcast: one resident Q3_K weight
//     [out, in] times one f32 activation [P, in], dst [P, out]. MMVQ serves P in 1..3 when the
//     k-quant mmvq decode arm is selected (FAK_CUDA_Q4K_DECODE=mmvq, shared with Q4_K); otherwise
//     P < 128 takes the f32 dequant-fused panel in cuda/q3k_matmul.cuh. P >= 128 dequantizes to a
//     transient f32 image (dequantize_block_q3_K) and runs Fak's compensated fp16 GEMM.
//   * With two rows per CUDA block the second row's weight read is clamped to the last row, so
//     an odd out never reads past the resident weight (upstream relies on row padding); its
//     result is still not written.
//   * The activation row is not padded to MATRIX_ROW_PADDING: Fak's Q3_K rows are a whole
//     number of 256-value super-blocks, so every q8_1 block the dot reads exists.
//   * The q8_1 scratch is a persistent grown-once buffer, like the Q4_K mmvq scratch, because
//     decode is captured into a CUDA graph; ggml_cuda_pdl_* is dropped.

namespace fak_q3k_mmvq {

using fak_q4k_mmvq::block_q8_1;
using fak_q4k_mmvq::calc_nwarps;
using fak_q4k_mmvq::calc_rows_per_block;
using fak_q4k_mmvq::warp_reduce_sum;

constexpr int warp_size = 32;
constexpr int QK_K = 256;
constexpr int QK8_1 = 32;
constexpr int QR3_K = 4;
constexpr int QI3_K = QK_K / (4 * QR3_K);
constexpr int QI8_1 = QK8_1 / 4;
constexpr int VDR_Q3_K_Q8_1_MMVQ = 1;
constexpr int max_cols = 3;

// ggml-common.h block_q3_K.
struct block_q3_K {
    uint8_t hmask[QK_K / 8]; // quants - high bit
    uint8_t qs[QK_K / 4];    // quants - low 2 bits
    uint8_t scales[12];      // scales, quantized with 6 bits
    half    d;               // super-block scale
};
static_assert(sizeof(block_q3_K) == sizeof(half) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

// vecdotq.cuh get_int_b2 / get_int_b4.
static __device__ __forceinline__ int get_int_b2(const void *x, const int &i32) {
    const uint16_t *x16 = (const uint16_t *)x; // assume at least 2 byte alignment
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void *x, const int &i32) {
    return ((const int *)x)[i32]; // assume at least 4 byte alignment
}

// vecdotq.cuh vec_dot_q3_K_q8_1_impl_mmvq (contiguous v/x values).
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int &vl, const int &vh, const int *__restrict__ u, const uint8_t *__restrict__ scales,
    const int &scale_offset, const float &d3, const float *__restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2 * i;

        const int isc_low = isc % (QK_K / 32);
        const int sc_shift_low = 4 * (isc / (QK_K / 32));
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K / 64);
        const int sc_shift_high = 2 * (isc / (QK_K / 64));
        const int sc_high = ((scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2 * i)) & 0x03030303;

        const int vih = ((vh >> i) << 2) & 0x04040404;

        const int vi = __vsubss4(vil, vih);

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d3 * sumf;
}

// vecdotq.cuh vec_dot_q3_K_q8_1.
static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1, const int &kbx, const int &iqs) {

    const block_q3_K *bq3_K = (const block_q3_K *)vbq + kbx;

    const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

    const float d = __half2float(bq3_K->d);

    const int vl = get_int_b2(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K / 2)) >> bq8_offset;

    int u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i] = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

// mmvq.cu mul_mat_vec_q, Q3_K, no fusion, one channel and sample.
template <int ncols_dst>
__launch_bounds__(calc_nwarps(ncols_dst) * warp_size, 1)
static __global__ void mul_mat_vec_q(const void *__restrict__ vx, const block_q8_1 *__restrict__ y,
                                     float *__restrict__ dst, const int ncols_x, const int nrows_x) {
    constexpr int qk = QK_K;
    constexpr int qi = QI3_K;
    constexpr int vdr = VDR_Q3_K_Q8_1_MMVQ;
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
                tmp[j][i] += vec_dot_q3_K_q8_1(vx, &y[j * stride_col_y + kby], row * stride_row_x + kbx, kqs);
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

// convert.cu dequantize_block_q3_K (f32 destination), one CUDA block of 64 threads per
// 256-value super-block.
static __global__ void dequantize_block_q3_K(const void *__restrict__ vx, float *__restrict__ yy) {

    const int64_t i = blockIdx.x;
    const block_q3_K *x = (const block_q3_K *)vx;

    const int64_t r = threadIdx.x / 4;
    const int64_t tid = r / 2;
    const int64_t is0 = r % 2;
    const int64_t l0 = 16 * is0 + 4 * (threadIdx.x % 4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4 * n;

    uint8_t m = 1 << (4 * n + j);
    int64_t is = 8 * n + 2 * j + is0;
    int shift = 2 * j;

    int8_t us = is < 4 ? (x[i].scales[is - 0] & 0xF) | (((x[i].scales[is + 8] >> 0) & 3) << 4) :
                is < 8 ? (x[i].scales[is - 0] & 0xF) | (((x[i].scales[is + 4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is - 8] >> 4) | (((x[i].scales[is + 0] >> 4) & 3) << 4) :
                          (x[i].scales[is - 8] >> 4) | (((x[i].scales[is - 4] >> 6) & 3) << 4);
    float d_all = __half2float(x[i].d);
    float dl = d_all * (us - 32);

    float *y = yy + i * QK_K + 128 * n + 32 * j;
    const uint8_t *q = x[i].qs + 32 * n;
    const uint8_t *hm = x[i].hmask;

    for (int l = l0; l < l0 + 4; ++l) {
        y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
    }
}

} // namespace fak_q3k_mmvq

// Persistent q8_1 activation scratch for the Q3_K GEMV, grown once (see the Q4_K mmvq note).
static fak_q4k_mmvq::block_q8_1 *g_q3k_mmvq_y = nullptr;
static size_t g_q3k_mmvq_y_cap = 0; // bytes
static unsigned long long g_q3k_mmvq_launches = 0;

extern "C" unsigned long long fcuda_q3k_mmvq_launch_count(void) { return g_q3k_mmvq_launches; }

static int fcuda_q3k_mmvq_launch(const uint8_t *dQ3K, const fak_q4k_mmvq::block_q8_1 *y, float *dY,
                                  int out, int in, int cols) {
    using namespace fak_q3k_mmvq;
    switch (cols) {
    case 1: {
        dim3 grid((out + calc_rows_per_block(1) - 1) / calc_rows_per_block(1)), block(warp_size, calc_nwarps(1));
        FCUDA_TS("k_q3k_mmvq_gemv", grid, fak_q3k_mmvq::mul_mat_vec_q<1><<<grid, block, 0, g_stream>>>(dQ3K, y, dY, in, out));
        break;
    }
    case 2: {
        dim3 grid((out + calc_rows_per_block(2) - 1) / calc_rows_per_block(2)), block(warp_size, calc_nwarps(2));
        FCUDA_TS("k_q3k_mmvq_gemv", grid, fak_q3k_mmvq::mul_mat_vec_q<2><<<grid, block, 0, g_stream>>>(dQ3K, y, dY, in, out));
        break;
    }
    default: {
        dim3 grid((out + calc_rows_per_block(3) - 1) / calc_rows_per_block(3)), block(warp_size, calc_nwarps(3));
        FCUDA_TS("k_q3k_mmvq_gemv", grid, fak_q3k_mmvq::mul_mat_vec_q<3><<<grid, block, 0, g_stream>>>(dQ3K, y, dY, in, out));
        break;
    }
    }
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) return (int)launch_status;
    ++g_q3k_mmvq_launches;
    return 0;
}

// fcuda_q3k_matmul_f32 (cuda_backend.h): Y[P,out] = X[P,in] @ W[out,in]^T over resident Q3_K.
extern "C" int fcuda_q3k_matmul_f32(const uint8_t *dQ3K, const float *dX, float *dY, int out, int in, int P) {
    using namespace fak_q3k_mmvq;
    const bool valid = dQ3K && dX && dY && out > 0 && in > 0 && in % QK_K == 0 && P > 0;
    if (valid && P >= 128) {
        const size_t wn = (size_t)out * in;
        float *w = (float *)scratch_malloc(wn * sizeof(float));
        if (!w) return FCUDA_SCRATCH_ALLOC_FAILED;
        const int nb = (int)(wn / QK_K);
        FCUDA_TS("k_q3k_dequant", nb, dequantize_block_q3_K<<<nb, 64, 0, g_stream>>>(dQ3K, w));
        const int status = fp16_compensated_matmul(w, dX, dY, out, in, P);
        fcuda_free(w);
        return status;
    }
    if (!valid || P > max_cols || g_q4k_decode_arm != 1) return fcuda_q3k_panel_f32(dQ3K, dX, dY, out, in, P);
    const size_t need = (size_t)P * (in / QK8_1) * sizeof(block_q8_1);
    if (need > g_q3k_mmvq_y_cap) {
        if (g_q3k_mmvq_y) {
            fak_alloc_untrack(g_q3k_mmvq_y);
            cudaFree(g_q3k_mmvq_y);
        }
        g_q3k_mmvq_y = nullptr;
        g_q3k_mmvq_y_cap = 0;
        if (fcuda_alloc_status(cudaMalloc(&g_q3k_mmvq_y, need)) != cudaSuccess || !g_q3k_mmvq_y) {
            g_q3k_mmvq_y = nullptr;
            return FCUDA_SCRATCH_ALLOC_FAILED;
        }
        fak_alloc_track(g_q3k_mmvq_y, FAK_ALLOC_ACTIVATIONS_SCRATCH, FAK_ALLOC_DEVICE_LOCAL,
                        FAK_ALLOC_SITE("q3k-mmvq-q8_1-activations"), need);
        g_q3k_mmvq_y_cap = need;
    }
    dim3 qgrid((in + 255) / 256, P);
    FCUDA_TS("quantize_q8_1", qgrid, fak_q4k_mmvq::quantize_q8_1<<<qgrid, 256, 0, g_stream>>>(dX, g_q3k_mmvq_y, in));
    // Do not enqueue a dependent dot product after a rejected quantizer launch.
    // These are immediate launch checks, not a stream completion fence.
    const cudaError_t quantize_status = cudaGetLastError();
    if (quantize_status != cudaSuccess) return (int)quantize_status;
    return fcuda_q3k_mmvq_launch(dQ3K, g_q3k_mmvq_y, dY, out, in, P);
}
