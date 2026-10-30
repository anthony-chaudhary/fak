// ---- IQ4_XS resident GEMV (f32 and q8_1 DP4A MMVQ) + dequant-split prefill GEMM -----------
// Adapted from llama.cpp at 571d0d540df04f25298d0e159e520d9fc62ed121 (MIT):
//   * ggml/src/ggml-common.h block_iq4_xs and kvalues_iq4nl (fcuda_kvalues_iq4nl, iq_gemv.cuh);
//   * ggml/src/ggml-cuda/convert.cu dequantize_block_iq4_xs (the per-thread (ib, il) split used
//     by k_iq4xs_gemv and k_iq4xs_dequant_split below);
//   * ggml/src/ggml-cuda/vecdotq.cuh get_int_from_table_16 (the portable char4 form, not the
//     __byte_perm form), VDR_IQ4_XS_Q8_1_MMVQ and vec_dot_iq4_xs_q8_1;
//   * ggml/src/ggml-cuda/mmvq.cu mul_mat_vec_q for GGML_TYPE_IQ4_XS on the NVIDIA GENERIC table.
// Copyright (c) 2023-2026 The ggml authors. The MIT permission notice is reproduced in
// THIRD_PARTY_NOTICES.md ("llama.cpp IQ4_XS resident GEMV").
//
// Changes for Fak (the same reductions as cuda/q3k_mmvq.cuh):
//   * One resident IQ4_XS weight [out, in] (verbatim 136-byte 256-weight super-blocks, row-major)
//     times one f32 activation [P, in], dst [P, out]. No fusion, MoE ids or broadcast.
//   * The q8_1 block, quantize_q8_1, warp_reduce_sum and the GENERIC nwarps / rows-per-block
//     helpers are the ones already ported in cuda/q4k_mmvq.cuh; the q8_1 scratch is a persistent
//     grown-once buffer because decode is captured into a CUDA graph.
//   * The second row of a two-row CUDA block is clamped to the last row, so an odd out never
//     reads past the resident weight; its result is not written.
//   * Prefill (P >= FCUDA_IQ4XS_DEQUANT_MIN_P) decodes straight into the fp16 hi/lo halves the
//     compensated tensor-core GEMM consumes (no f32 weight image), like k_q8_dequant_split.

namespace fak_iq4xs {

using fak_q4k_mmvq::block_q8_1;
using fak_q4k_mmvq::calc_nwarps;
using fak_q4k_mmvq::calc_rows_per_block;
using fak_q4k_mmvq::warp_reduce_sum;

constexpr int warp_size = 32;
constexpr int QK_K = 256;
constexpr int QK8_1 = 32;
constexpr int QR4_XS = 2;
constexpr int QI4_XS = QK_K / (4 * QR4_XS);
constexpr int VDR_IQ4_XS_Q8_1_MMVQ = 4;
constexpr int max_cols = 3;

// ggml-common.h block_iq4_xs.
struct block_iq4_xs {
    half     d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K / 64];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(block_iq4_xs) == 136, "IQ4_XS super-block is 136 bytes");

static __device__ __forceinline__ int sub_scale(const block_iq4_xs *b, int ib) {
    return ((((b->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((b->scales_h >> 2 * ib) & 3) << 4)) - 32);
}

// vecdotq.cuh get_int_from_table_16, portable form.
static __device__ __forceinline__ int2 get_int_from_table_16(const int &q4, const signed char *table) {
    const int q0_32 = (q4 >> 0) & 0x0F0F0F0F;
    const int8_t *q0_8 = (const int8_t *)&q0_32;
    const char4 val0_8 = make_char4(table[q0_8[0]], table[q0_8[1]], table[q0_8[2]], table[q0_8[3]]);

    const int q1_32 = (q4 >> 4) & 0x0F0F0F0F;
    const int8_t *q1_8 = (const int8_t *)&q1_32;
    const char4 val1_8 = make_char4(table[q1_8[0]], table[q1_8[1]], table[q1_8[2]], table[q1_8[3]]);

    return make_int2(*((const int *)&val0_8), *((const int *)&val1_8));
}

// vecdotq.cuh vec_dot_iq4_xs_q8_1.
static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1, const int &kbx, const int &iqs) {

    const block_iq4_xs *bq4 = (const block_iq4_xs *)vbq + kbx;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = ((const int *)bq4->qs)[iqs + j];
        const int2 v = get_int_from_table_16(aux_q4, fcuda_kvalues_iq4nl);

        const int u0 = ((const int *)bq8_1[iqs / 4].qs)[j + 0];
        const int u1 = ((const int *)bq8_1[iqs / 4].qs)[j + 4];

        sumi = __dp4a(v.x, u0, sumi);
        sumi = __dp4a(v.y, u1, sumi);
    }

    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;

    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs / 4].ds);
    return d * sumi;
}

// mmvq.cu mul_mat_vec_q, IQ4_XS, no fusion, one channel and sample.
template <int ncols_dst>
__launch_bounds__(calc_nwarps(ncols_dst) * warp_size, 1)
static __global__ void mul_mat_vec_q(const void *__restrict__ vx, const block_q8_1 *__restrict__ y,
                                     float *__restrict__ dst, const int ncols_x, const int nrows_x) {
    constexpr int qk = QK_K;
    constexpr int qi = QI4_XS;
    constexpr int vdr = VDR_IQ4_XS_Q8_1_MMVQ;
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
                tmp[j][i] += vec_dot_iq4_xs_q8_1(vx, &y[j * stride_col_y + kby], row * stride_row_x + kbx, kqs);
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

// k_iq4xs_gemv: f32-activation warp-per-row GEMV (the k_iq4nl_gemv layout). Lane
// (ib = lane>>2, il = lane&3) decodes sub-block ib's bytes 16*ib + 4*il .. +3, i.e. weights
// 32*ib + 4*il + j (low nibble) and +16 (high nibble), exactly as dequantize_block_iq4_xs.
static __global__ void k_iq4xs_gemv(const unsigned char *W, const float *X, float *Y, int out, int in, int P) {
    const int lane = threadIdx.x & 31;
    const int o = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int t = blockIdx.y;
    if (o >= out || t >= P) return;
    const int ib = lane >> 2, il = lane & 3;
    const int nsb = in / QK_K;
    const block_iq4_xs *wrow = (const block_iq4_xs *)W + (size_t)o * nsb;
    const float *xrow = X + (size_t)t * in + 32 * ib + 4 * il;
    float sum = 0.f;
    for (int sb = 0; sb < nsb; ++sb) {
        const block_iq4_xs *blk = wrow + sb;
        const unsigned char *q4 = blk->qs + 16 * ib + 4 * il;
        const float d = __half2float(blk->d) * (float)sub_scale(blk, ib);
        const float *xs = xrow + (size_t)sb * QK_K;
        const float4 lo = *(const float4 *)xs;
        const float4 hi = *(const float4 *)(xs + 16);
        const float xl[4] = {lo.x, lo.y, lo.z, lo.w};
        const float xh[4] = {hi.x, hi.y, hi.z, hi.w};
        float part = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            part = fmaf((float)fcuda_kvalues_iq4nl[q4[j] & 0xf], xl[j], part);
            part = fmaf((float)fcuda_kvalues_iq4nl[q4[j] >> 4], xh[j], part);
        }
        sum = fmaf(d, part, sum);
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) Y[(size_t)t * out + o] = sum;
}

// k_iq4xs_dequant_split: dequantize_block_iq4_xs (32 threads per super-block) writing the fp16
// hi/lo halves of each weight directly, with the explicit rounding k_q8_dequant_split uses.
static __global__ void k_iq4xs_dequant_split(const unsigned char *W, __half *hi, __half *lo) {
    const size_t i = blockIdx.x;
    const block_iq4_xs *x = (const block_iq4_xs *)W + i;
    const int il = threadIdx.x / 8; // 0..3
    const int ib = threadIdx.x % 8; // 0..7
    const size_t base = i * QK_K + 32 * ib + 4 * il;
    const unsigned char *q4 = x->qs + 16 * ib + 4 * il;
    const float d = __fmul_rn(__half2float(x->d), (float)sub_scale(x, ib));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float v0 = __fmul_rn(d, (float)fcuda_kvalues_iq4nl[q4[j] & 0xf]);
        const float v1 = __fmul_rn(d, (float)fcuda_kvalues_iq4nl[q4[j] >> 4]);
        const __half h0 = __float2half_rn(v0), h1 = __float2half_rn(v1);
        hi[base + j] = h0;
        lo[base + j] = __float2half_rn(__fsub_rn(v0, __half2float(h0)));
        hi[base + j + 16] = h1;
        lo[base + j + 16] = __float2half_rn(__fsub_rn(v1, __half2float(h1)));
    }
}

} // namespace fak_iq4xs

// FCUDA_IQ4XS_DEQUANT_MIN_P is the first panel width that decodes the weight once into fp16
// halves and runs the compensated tensor-core GEMM; narrower panels re-read the packed weight
// per token through the GEMV.
#define FCUDA_IQ4XS_DEQUANT_MIN_P 32

static fak_q4k_mmvq::block_q8_1 *g_iq4xs_mmvq_y = nullptr;
static size_t g_iq4xs_mmvq_y_cap = 0; // bytes
static unsigned long long g_iq4xs_mmvq_launches = 0;
static unsigned long long g_iq4xs_dequant_gemm_launches = 0;

extern "C" unsigned long long fcuda_iq4xs_mmvq_launch_count(void) { return g_iq4xs_mmvq_launches; }
extern "C" unsigned long long fcuda_iq4xs_dequant_gemm_launch_count(void) { return g_iq4xs_dequant_gemm_launches; }

static void fcuda_iq4xs_mmvq_launch(const uint8_t *dW, const fak_q4k_mmvq::block_q8_1 *y, float *dY,
                                    int out, int in, int cols) {
    using namespace fak_iq4xs;
    switch (cols) {
    case 1: {
        dim3 grid((out + calc_rows_per_block(1) - 1) / calc_rows_per_block(1)), block(warp_size, calc_nwarps(1));
        FCUDA_TS("k_iq4xs_mmvq_gemv", grid, fak_iq4xs::mul_mat_vec_q<1><<<grid, block, 0, g_stream>>>(dW, y, dY, in, out));
        break;
    }
    case 2: {
        dim3 grid((out + calc_rows_per_block(2) - 1) / calc_rows_per_block(2)), block(warp_size, calc_nwarps(2));
        FCUDA_TS("k_iq4xs_mmvq_gemv", grid, fak_iq4xs::mul_mat_vec_q<2><<<grid, block, 0, g_stream>>>(dW, y, dY, in, out));
        break;
    }
    default: {
        dim3 grid((out + calc_rows_per_block(3) - 1) / calc_rows_per_block(3)), block(warp_size, calc_nwarps(3));
        FCUDA_TS("k_iq4xs_mmvq_gemv", grid, fak_iq4xs::mul_mat_vec_q<3><<<grid, block, 0, g_stream>>>(dW, y, dY, in, out));
        break;
    }
    }
    ++g_iq4xs_mmvq_launches;
}

// fcuda_iq4xs_matmul_f32 (cuda_backend.h): Y[P,out] = X[P,in] @ W[out,in]^T over resident IQ4_XS.
extern "C" int fcuda_iq4xs_matmul_f32(const uint8_t *dW, const float *dX, float *dY, int out, int in, int P) {
    using namespace fak_iq4xs;
    if (!dW || !dX || !dY || out <= 0 || in <= 0 || in % QK_K != 0 || P <= 0) return (int)cudaErrorInvalidValue;
    if (P >= FCUDA_IQ4XS_DEQUANT_MIN_P) {
        const size_t wn = (size_t)out * in, xn = (size_t)P * in;
        FP16CompensatedScratch scratch;
        const int scratch_status = fp16_compensated_scratch(wn, xn, &scratch);
        if (scratch_status != 0) return scratch_status;
        __half *wHi = scratch.ptr[0], *wLo = scratch.ptr[1], *xHi = scratch.ptr[2], *xLo = scratch.ptr[3];
        const int nb = (int)(wn / QK_K);
        FCUDA_TS("k_iq4xs_dequant_split", nb, k_iq4xs_dequant_split<<<nb, 32, 0, g_stream>>>(dW, wHi, wLo));
        FCUDA_TS("k_fp16_split", (xn + 255) / 256, k_fp16_split<<<(xn + 255) / 256, 256, 0, g_stream>>>(dX, xHi, xLo, xn));
        fp16_compensated_gemm(wHi, wLo, xHi, xLo, dY, out, in, P);
        ++g_iq4xs_dequant_gemm_launches;
        return fp16_compensated_release(&scratch);
    }
    if (P > max_cols || g_q4k_decode_arm != 1) {
        dim3 grid((out + 7) / 8, P);
        FCUDA_TS("k_iq4xs_gemv", grid, k_iq4xs_gemv<<<grid, 256, 0, g_stream>>>(dW, dX, dY, out, in, P));
        return 0;
    }
    const size_t need = (size_t)P * (in / QK8_1) * sizeof(block_q8_1);
    if (need > g_iq4xs_mmvq_y_cap) {
        if (g_iq4xs_mmvq_y) {
            fak_alloc_untrack(g_iq4xs_mmvq_y);
            cudaFree(g_iq4xs_mmvq_y);
        }
        g_iq4xs_mmvq_y = nullptr;
        g_iq4xs_mmvq_y_cap = 0;
        if (fcuda_alloc_status(cudaMalloc(&g_iq4xs_mmvq_y, need)) != cudaSuccess || !g_iq4xs_mmvq_y) {
            g_iq4xs_mmvq_y = nullptr;
            return FCUDA_SCRATCH_ALLOC_FAILED;
        }
        fak_alloc_track(g_iq4xs_mmvq_y, FAK_ALLOC_ACTIVATIONS_SCRATCH, FAK_ALLOC_DEVICE_LOCAL,
                        FAK_ALLOC_SITE("iq4xs-mmvq-q8_1-activations"), need);
        g_iq4xs_mmvq_y_cap = need;
    }
    dim3 qgrid((in + 255) / 256, P);
    FCUDA_TS("quantize_q8_1", qgrid, fak_q4k_mmvq::quantize_q8_1<<<qgrid, 256, 0, g_stream>>>(dX, g_iq4xs_mmvq_y, in));
    fcuda_iq4xs_mmvq_launch(dW, g_iq4xs_mmvq_y, dY, out, in, P);
    return 0;
}
