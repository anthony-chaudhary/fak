// PRIVATE, UNQUALIFIED: Q2_K prompt-only MMQ; no default dispatch change.
// Q2 loader/dot/D2S6 quantizer/config direct-adapted from ggml-org/llama.cpp
// 571d0d540df04f25298d0e159e520d9fc62ed121, ggml/src/ggml-cuda/
// mmq-load-tiles.cuh, mmq-vec-dot.cuh, quantize.cu, mmq.cuh,
// mmq-config-ampere.cuh, mma.cuh, vecdotq.cuh; block ABI from ggml-common.h.
// Launch/writeback/stream-k scaffolding reused from Fak q4k_mmq.cuh, whose
// upstream pin is 0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd.
// Copyright (c) 2023-2026 The ggml authors. MIT text follows.
/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


*/
#pragma once
#if defined(FAK_CUDA_Q2K_MMQ_PREFILL)
#ifndef FAK_CUDA_V41_CHECKED_STREAM
#error "Q2 MMQ candidate requires shared checked-stream poison support"
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define FAK_Q2K_MMQ_AVAILABLE 1
#endif

namespace fak_q2k_mmq {

constexpr int warp_size = 32;

// common.cuh / ggml-common.h
constexpr int QK_K = 256;
constexpr int QK8_1 = 32;
constexpr int QR2_K = 4;
constexpr int QI8_1 = QK8_1 / 4;
constexpr int QI2_K = QK_K / (4*QR2_K);
constexpr int VDR_Q2_K_Q8_1_MMQ = 4;
constexpr int MATRIX_ROW_PADDING = 512;
#define FAK_Q2_MMQ_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

struct block_q2_K {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    half2 dm;
};
static_assert(sizeof(block_q2_K) == 84, "Q2_K super-block is 84 bytes");

static uint3 init_fastdiv_values(uint64_t d_64) {
    uint32_t d = (uint32_t) d_64;
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }
    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}

// vecdotq.cuh
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

// mmq.cuh
constexpr int MMQ_ITER_K = 256;
constexpr int QK8_1_MMQ = 4*QK8_1;

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // D2S6: two FP16 scales per 64 values and six FP16 sums per 16 values.
    half d2s6[8];
    int8_t qs[QK8_1_MMQ];
};
static_assert(sizeof(block_q8_1_mmq) == QK8_1_MMQ + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");

constexpr int MMQ_TILE_NE_K = 32;
constexpr int MMQ_TILE_Y_K = MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1;

// ggml_cuda_mmq_get_config_ampere(GGML_TYPE_Q2_K, J, fallback) and its derived values.
constexpr int nthreads = 256;
constexpr int occupancy = 1;
constexpr int I = 128;
constexpr int nwarps = nthreads / warp_size;
constexpr int qk = QK_K;
constexpr int ITER_K = MMQ_ITER_K;
constexpr int blocks_per_iter = ITER_K / qk;
constexpr int sram_stride = 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K + 4; // GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K
static_assert(sram_stride % 8 == 4, "Wrong padding.");

static constexpr __host__ __device__ bool config_has_J(int J, bool fallback) {
    return fallback
        ? (J == 8 || J == 16 || J == 32 || J == 64 || J == 128)
        : (J == 8 || J == 16 || J == 24 || J == 32 || J == 40 || J == 48 ||
           J == 64 || J == 80 || J == 96 || J == 112 || J == 128);
}

static constexpr __host__ __device__ int rows_per_warp_for(int J) {
    return J >= 48 && J % 16 == 0 ? 32 : 16;
}

// mma.cuh: the NVIDIA DATA_LAYOUT_I_MAJOR int tiles this instantiation uses.
template <int I_, int J_>
struct tile {
    static constexpr int I  = I_;
    static constexpr int J  = J_;
    static constexpr int ne = I * J / 32;
    int x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        if constexpr (I == 8 && (J == 4 || J == 8)) {
            return threadIdx.x / 4;
        } else if constexpr (I == 16 && J == 8) {
            return ((l / 2) * 8) + (threadIdx.x / 4);
        } else {
            static_assert(I == -1, "unsupported tile");
            return -1;
        }
    }

    static __device__ __forceinline__ int get_j(const int l) {
        if constexpr (I == 8 && J == 4) {
            return threadIdx.x % 4;
        } else if constexpr (I == 8 && J == 8) {
            return (l * 4) + (threadIdx.x % 4);
        } else if constexpr (I == 16 && J == 8) {
            return ((threadIdx.x % 4) * 2) + (l % 2);
        } else {
            static_assert(I == -1, "unsupported tile");
            return -1;
        }
    }
};

template <int I_, int J_>
static __device__ __forceinline__ void load_generic(tile<I_, J_> & t, const int * __restrict__ xs0, const int stride) {
#pragma unroll
    for (int l = 0; l < t.ne; ++l) {
        t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
    }
}

static __device__ __forceinline__ void load_ldmatrix(tile<16, 8> & t, const int * __restrict__ xs0, const int stride) {
#ifdef FAK_Q2K_MMQ_AVAILABLE
    int * xi = (int * ) t.x;
    const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(xs));
#else
    (void) t; (void) xs0; (void) stride;
    __trap();
#endif // FAK_Q2K_MMQ_AVAILABLE
}

static __device__ __forceinline__ void mma(tile<16, 8> & D, const tile<16, 4> & A, const tile<8, 4> & B) {
#ifdef FAK_Q2K_MMQ_AVAILABLE
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
#else
    (void) D; (void) A; (void) B;
    __trap();
#endif
}

// mmq-load-tiles.cuh: NVIDIA MMA branch.
template <int J, bool fallback> static __device__ __forceinline__ void load_tiles_q2_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR2_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = (const block_q2_K *) x + kbx0 + i*stride;

        const int x_ql_0 = get_int_b2(bxi->qs, kqsx);

#pragma unroll
        for (int l = 0; l < QR2_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_qs_k = (x_ql_0 >> (2*l)) & 0x03030303;

            x_qs[i*sram_stride           + k] = x_qs_k;
        }

        const int sc_m = bxi->scales[kqsx];
        const half2 x_dm_ik = __hmul2(bxi->dm, __floats2half2_rn(sc_m & 0x0F, sc_m >> 4));

        x_dm[i*sram_stride         + kqsx] = x_dm_ik;
    }
}


// mmq-vec-dot.cuh: NVIDIA MMA branch.
template <int J, bool fallback> static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {


    typedef tile<16, 4> tile_A;
    typedef tile<16, 8> tile_A_8;
    typedef tile< 8, 4> tile_B;
    typedef tile<16, 8> tile_C;

    constexpr int rows_per_warp = rows_per_warp_for(J);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A  A[ntx][8];
    float  dA[ntx][tile_C::ne/2][8];
    float  mA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(((tile_A_8 *) A[n])[k01/QI8_1], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }
    }

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1/2) {
                const int k0 = k00 + k01;

                const float2 dm = __half22float2(x_dm[i*sram_stride + k0/(QI8_1/2)]);

                dA[n][l][k01/(QI8_1/2)] = dm.x;
                mA[n][l][k01/(QI8_1/2)] = dm.y;
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
        float2 dB[tile_C::ne/2];

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int j = j0 + tile_C::get_j(l);

            dB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B B[2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);

            tile_C Cm[2];
            if (k01 >= MMQ_TILE_NE_K * 3/4) {
                tile_A A1;
                A1.x[0] = 0x01010101;
                A1.x[1] = 0x01010101;
                mma(Cm[0], A1, B[0]);
                mma(Cm[1], A1, B[1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C Cd[2];

                mma(Cd[0], A[n][k01/4 + 0], B[0]);
                mma(Cd[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    float tmp = Cd[0].x[l]*dA[n][l/2][k01/4 + 0] + Cd[1].x[l]*dA[n][l/2][k01/4 + 1];
                    if (k01 >= MMQ_TILE_NE_K * 3/4) {
                        tmp -= Cm[0].x[l]*mA[n][l/2][k01/4 + 0] + Cm[1].x[l]*mA[n][l/2][k01/4 + 1];
                    }
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp*(k01 < MMQ_TILE_NE_K/2 ? dB[l%2].x : dB[l%2].y);
                }
            }
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K * 3/4; k01 += QI8_1) {
            float2 sB[tile_C::ne/2];

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                sB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= mA[n][l/2][k01/4 + 0]*sB[l%2].x;
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= mA[n][l/2][k01/4 + 1]*sB[l%2].y;
                }
            }
        }
    }
}

// mmq.cuh
template <int J, bool fallback>
static __device__ __forceinline__ void write_back_mma(
        const float * __restrict__ sum, float * __restrict__ dst, const int stride, const int i_max, const int j_max) {
    typedef tile<16,  8> tile_C;

    constexpr int rows_per_warp = rows_per_warp_for(J);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_C::I);

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n*tile_C::I + tile_C::get_i(l);

                if (fallback && i > i_max) {
                    continue;
                }

                dst[j*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
            }
        }
    }
}

template <int J, bool fallback, bool fixup>
static __device__ __forceinline__ void mul_mat_q_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q;
    int * tile_x = tile_y + FAK_Q2_MMQ_PAD(J*MMQ_TILE_Y_K, nwarps*warp_size);

    constexpr int ne_block = QK8_1_MMQ;

    float sum[J*I / (nwarps*warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles_q2_K<J, fallback>(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot_q2_K_q8_1_mma<J, fallback>(tile_x, tile_y, sum, 0);

        __syncthreads();

        {
            const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot_q2_K_q8_1_mma<J, fallback>(tile_x, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

    if (fixup) {
        write_back_mma<J, fallback>(sum, tmp_fixup + blockIdx.x*(J*I), I, I, J);
    } else {
        write_back_mma<J, fallback>(sum, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598

template <int J, bool fallback>
__launch_bounds__(nthreads, occupancy)
static __global__ void mul_mat_q(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y,
        const int stride_col_dst, const uint3 ntx) {
#ifdef FAK_Q2K_MMQ_AVAILABLE
    const uint32_t nty = (nrows_x + I - 1) / I; // Number of tiles y

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc      = int64_t(blockIdx.x)    *(ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc_stop = int64_t(blockIdx.x + 1)*(ntx.z*nty*blocks_per_ne00.z) / gridDim.x;

    kbc      -= fastmodulo(kbc,      blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= fastmodulo(kbc_stop, blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = fastmodulo(kbc, blocks_per_ne00);
    int kb0_stop  = min(blocks_per_ne00.z, uint32_t(kb0_start + kbc_stop - kbc));
    while (kbc < kbc_stop && kb0_stop == int(blocks_per_ne00.z)) {
        const uint2 tmp2 = fast_div_modulo(fastdiv(kbc, blocks_per_ne00), ntx);
        const int jt = tmp2.y;
        const int it = tmp2.x;

        const int offset_y   = jt*J*(sizeof(block_q8_1_mmq)/sizeof(int));
        const int offset_dst = jt*J*stride_col_dst + it*I;

        const int tile_x_max_i = nrows_x   - it*I - 1;
        const int tile_y_max_j = ncols_dst - jt*J - 1;

        const int offset_x = it*I*stride_row_x;

        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        mul_mat_q_process_tile<J, fallback, fixup>
            (x, offset_x, y + offset_y, dst + offset_dst, tmp_fixup,
             stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);

        kbc += blocks_per_ne00.z;
        kbc -= fastmodulo(kbc, blocks_per_ne00);

        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const uint2 tmp2 = fast_div_modulo(fastdiv(kbc, blocks_per_ne00), ntx);
    const int jt = tmp2.y;
    const int it = tmp2.x;

    const int offset_y   = jt*J*(sizeof(block_q8_1_mmq)/sizeof(int));
    const int offset_dst = jt*J*stride_col_dst + it*I;

    const int tile_x_max_i = nrows_x   - it*I - 1;
    const int tile_y_max_j = ncols_dst - jt*J - 1;

    const int offset_x = it*I*stride_row_x;

    constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    mul_mat_q_process_tile<J, fallback, fixup>
        (x, offset_x, y + offset_y, dst + offset_dst, tmp_fixup,
         stride_row_x, ncols_y, stride_col_dst,
         tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
#else
    (void) x; (void) y; (void) dst; (void) tmp_fixup; (void) blocks_per_ne00; (void) nrows_x;
    (void) ncols_dst; (void) stride_row_x; (void) ncols_y; (void) stride_col_dst; (void) ntx;
    __trap();
#endif // FAK_Q2K_MMQ_AVAILABLE
}

template <int J, bool fallback>
__launch_bounds__(nthreads/2, 1)
static __global__ void mul_mat_q_stream_k_fixup(
        float * __restrict__ dst, const float * __restrict__ tmp_last_tile, const uint3 blocks_per_ne00,
        const int nrows_x, const int ncols_dst, const int stride_col_dst, const uint3 ntx) {
    constexpr int nwarps_fixup = (nthreads / 2) / warp_size;

    float sum[J / nwarps_fixup] = {0.0f};
    const int i = blockIdx.y*warp_size + threadIdx.x;

    const int nty = (nrows_x + I - 1) / I;

    const int bidx0 = blockIdx.x;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc0      = int64_t(blockIdx.x)    *(ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc0_stop = int64_t(blockIdx.x + 1)*(ntx.z*nty*blocks_per_ne00.z) / gridDim.x;

    kbc0      -= fastmodulo(kbc0,      blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= fastmodulo(kbc0_stop, blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, blocks_per_ne00) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, blocks_per_ne00) == fastdiv(kbc0_stop, blocks_per_ne00) && fastmodulo(kbc0_stop, blocks_per_ne00) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    bool any_fixup = false;

    // Iterate over previous blocks and sum up partial sums written to fixup buffer.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        int kbc = int64_t(bidx)*(ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
        kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps_fixup) {
            const int j = j0 + threadIdx.y;

            sum[j0/nwarps_fixup] += tmp_last_tile[bidx*(J*I) + j*I + i];
        }

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, blocks_per_ne00) == 0 || fastdiv(kbc, blocks_per_ne00) < fastdiv(kbc0, blocks_per_ne00)) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) {
        return;
    }

    const uint2 tmp2 = fast_div_modulo(fastdiv(kbc0, blocks_per_ne00), ntx);
    const int jt = tmp2.y;
    const int it = tmp2.x;

    const int offset_dst = jt*J*stride_col_dst + it*I;
    dst += offset_dst;

    const int i_max = nrows_x   - it*I - 1;
    const int j_max = ncols_dst - jt*J - 1;
    if (fallback && i > i_max) {
        return;
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps_fixup) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

        dst[j*stride_col_dst + i] += sum[j0/nwarps_fixup];
    }
}

// quantize.cu: quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, scatter=false> for one channel.
constexpr int CUDA_QUANTIZE_BLOCK_SIZE_MMQ = 128;
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

static __global__ void quantize_mmq_q8_1_d2s6(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1) {

    constexpr int vals_per_scale = 64;
    constexpr int vals_per_sum = 16;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i00 = i0;
    const int64_t base_idx = blockIdx.x*s01;

    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t k_block = i0 / QK8_1_MMQ; // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ; // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, warp_size));
    }

    float sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
    for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, warp_size);
    }

    // Defined zero/padding conversion; avoid float NaN to integer conversion.
    const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = amax == 0.0f ? 0.0f : 1.0f / d_inv;

    const int64_t ib = k_block*ne1 + blockIdx.x;

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (iqs % 16 == 0 && iqs < 96) {
        y[ib].d2s6[2 + iqs/16] = sum;
        if (iqs % 64 == 0) {
            y[ib].d2s6[iqs/64] = d;
        }
    }
}

// Per-device launch facts the host side of mmq.cuh reads from ggml_cuda_info().
struct device_info {
    bool known;
    int cc;
    int compiled_arch; // ggml_cuda_highest_compiled_arch: the virtual arch of the loaded MMQ image
    int nsm;
    size_t smpbo;
    bool smem_raised[2][129];
};
static device_info g_device_info[16];

static size_t nbytes_shared(int J) {
    const size_t nbs_x = (size_t) I * sram_stride * 4;
    const size_t nbs_y = (size_t) J * sizeof(block_q8_1_mmq);
    return nbs_x + FAK_Q2_MMQ_PAD(nbs_y, nthreads*sizeof(int));
}

template <int J, bool fallback>
static int launch_mul_mat_q(const char * x, const int * y, float * dst, int nrows_x, int ncols_x,
                            int ncols_dst, device_info & info, cudaStream_t stream) {
    const int nbytes = (int) nbytes_shared(J);
    if (!info.smem_raised[fallback][J]) {
        cudaError_t attr_status = cudaFuncSetAttribute(mul_mat_q<J, fallback>, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes);
        if (attr_status != cudaSuccess) return v41_checked_stream_error(51000, attr_status);
        info.smem_raised[fallback][J] = true;
    }

    const dim3 block_dims(warp_size, nwarps, 1);
    const int nty = (nrows_x   + I - 1) / I;
    const int ntx = (ncols_dst + J - 1) / J;

    const uint3 blocks_per_ne00_fd = init_fastdiv_values(ncols_x / qk);
    const uint3 ntx_fd             = init_fastdiv_values(ntx);

    // For the stream-k kernel it is possible to run it with tiling by setting the number of CUDA blocks equal to the number of tiles.
    // This is worthwhile if the efficiency of tiling is high and skipping the fixup kernel is more important.
    const int ntiles_dst = ntx * nty;
    const int64_t tiles_nwaves = ((int64_t)ntiles_dst + info.nsm - 1) / info.nsm;
    const int tiles_efficiency_percent = (int)(100LL * ntiles_dst / (info.nsm*tiles_nwaves));
    const dim3 block_nums_stream_k(tiles_efficiency_percent >= 90 ? ntiles_dst : info.nsm, 1, 1);

    if ((int64_t) ntiles_dst * blocks_per_ne00_fd.z >= (1 << 30)) { // kbc would overflow.
        return (int)cudaErrorInvalidValue;
    }

    const bool fixup_needed = ntiles_dst % block_nums_stream_k.x != 0;

    float * tmp_fixup = nullptr;
    if (fixup_needed) {
        tmp_fixup = (float *) scratch_malloc((size_t) block_nums_stream_k.x * J*I * sizeof(float));
        if (!tmp_fixup) {
            // Quantization is already queued. A null scratch allocation may
            // report an asynchronous CUDA fault, not just OOM: prove completion
            // before the caller can retire/recycle the activation buffer.
            const cudaError_t completion_status = cudaStreamSynchronize(stream);
            if (completion_status != cudaSuccess) {
                (void) v41_checked_stream_error(56000, completion_status);
            }
            // Preserve the primary allocation error even if drain also failed.
            // On failed drain the shared poison makes caller fcuda_free a no-op.
            return FCUDA_SCRATCH_ALLOC_FAILED;
        }
    }

    const dim3 block_nums_fixup(block_nums_stream_k.x, I/warp_size, 1);
    const dim3 block_dims_fixup(block_dims.x, block_dims.y/2, block_dims.z);

    const int stride_row_x = ncols_x / qk;
    FCUDA_TS("q2_mul_mat_q", block_nums_stream_k, mul_mat_q<J, fallback><<<block_nums_stream_k, block_dims, nbytes, stream>>>
        (x, y, dst, tmp_fixup, blocks_per_ne00_fd, nrows_x, ncols_dst, stride_row_x, ncols_dst, nrows_x, ntx_fd));

    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        const int status = v41_checked_stream_error(52000, launch_status);
        // Poison is global; fcuda_free must not recycle either in-flight scratch.
        if (tmp_fixup) fcuda_free(tmp_fixup);
        return status;
    }
    if (fixup_needed) {
        FCUDA_TS("q2_mul_mat_q_stream_k_fixup", block_nums_fixup, mul_mat_q_stream_k_fixup<J, fallback><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
            (dst, tmp_fixup, blocks_per_ne00_fd, nrows_x, ncols_dst, nrows_x, ntx_fd));
        launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            const int status = v41_checked_stream_error(53000, launch_status);
            fcuda_free(tmp_fixup);
            return status;
        }
        fcuda_free(tmp_fixup);
    }
    return 0;
}

template <bool fallback>
static int mul_mat_q_switch_J(const char * x, const int * y, float * dst, int nrows_x, int ncols_x,
                              int ncols_dst, device_info & info, cudaStream_t stream) {
    int J_best        = 0;
    int ntiles_J_best = INT_MAX;

    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        if (!config_has_J(J, fallback)) {
            continue;
        }

        if (nbytes_shared(J) > info.smpbo) {
            continue;
        }

        const int ntiles_x = (ncols_dst + J - 1) / J;

        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }

    switch (J_best) {
        case   8: return launch_mul_mat_q<  8, fallback>(x, y, dst, nrows_x, ncols_x, ncols_dst, info, stream);
        case  16: return launch_mul_mat_q< 16, fallback>(x, y, dst, nrows_x, ncols_x, ncols_dst, info, stream);
        case  24: return launch_mul_mat_q< 24, fallback>(x, y, dst, nrows_x, ncols_x, ncols_dst, info, stream);
        case  32: return launch_mul_mat_q< 32, fallback>(x, y, dst, nrows_x, ncols_x, ncols_dst, info, stream);
        case  40: return launch_mul_mat_q< 40, fallback>(x, y, dst, nrows_x, ncols_x, ncols_dst, info, stream);
        case  48: return launch_mul_mat_q< 48, fallback>(x, y, dst, nrows_x, ncols_x, ncols_dst, info, stream);
        case  64: return launch_mul_mat_q< 64, fallback>(x, y, dst, nrows_x, ncols_x, ncols_dst, info, stream);
        case  80: return launch_mul_mat_q< 80, fallback>(x, y, dst, nrows_x, ncols_x, ncols_dst, info, stream);
        case  96: return launch_mul_mat_q< 96, fallback>(x, y, dst, nrows_x, ncols_x, ncols_dst, info, stream);
        case 112: return launch_mul_mat_q<112, fallback>(x, y, dst, nrows_x, ncols_x, ncols_dst, info, stream);
        case 128: return launch_mul_mat_q<128, fallback>(x, y, dst, nrows_x, ncols_x, ncols_dst, info, stream);
        default:  return (int)cudaErrorInvalidConfiguration;
    }
}

} // namespace fak_q2k_mmq

// Successful MMQ launches, read by the device parity witness to prove the mmq arm (not the
// compensated route) produced the output it checks.
static unsigned long long g_q2k_mmq_launches = 0;

extern "C" unsigned long long fcuda_q2k_mmq_launch_count(void) {
    return g_q2k_mmq_launches;
}

// fcuda_q2k_mmq_matmul is the mmq arm of fcuda_q2k_matmul_f32 (cuda_backend.h): it quantizes
// X to q8_1 in the MMQ layout and runs the Q2_K MMQ kernel straight off the resident weight.
// Returns 0 after checked enqueue (not GPU completion), allocation/native error, or
// FCUDA_Q2K_MMQ_REFUSED only for pre-enqueue admission. Launch failures poison
// the shared stream and preserve allocations; they must never trigger fallback.
#define FCUDA_Q2K_MMQ_REFUSED 80035
// Internal only: existing locked Q2 submission paths own W/X provenance.
// Callers must preserve owned same-runtime W/X, fresh nonalias Y and locking;
// pool reuse uses g_stream. Revalidate all Q2 callers before activation.
// This is not a raw-pointer public ABI and must not be called on imported streams.
static int fcuda_q2k_mmq_matmul(const uint8_t *dQ2K, const float *dX, float *dY, int out, int in, int P) {
  using namespace fak_q2k_mmq;
  // Only a pre-enqueue refusal may fall back. Runtime/launch failures propagate.
  if (v41_candidates_stream_uncertain()) return 90003;
  if (g_capture_open || !g_stream || !dQ2K || !dX || !dY ||
      out <= 0 || in <= 0 || P < 128 || in % QK_K != 0 ||
      ((uintptr_t)dQ2K & 3) != 0 || ((uintptr_t)dX & 15) != 0 ||
      ((uintptr_t)dY & 3) != 0 || out > INT_MAX-127 || P > INT_MAX-127 ||
      (int64_t)P*out >= INT_MAX || (int64_t)out*(in/QK_K) >= INT_MAX ||
      ((int64_t)in + MATRIX_ROW_PADDING-1)/MATRIX_ROW_PADDING > 65535) {
    return FCUDA_Q2K_MMQ_REFUSED;
  }
  cudaError_t pending = cudaPeekAtLastError();
  if (pending != cudaSuccess) return v41_checked_stream_error(50000, pending);
  int device = 0;
  cudaError_t device_status = cudaGetDevice(&device);
  if (device_status != cudaSuccess) return v41_checked_stream_error(50000, device_status);
  if (device < 0 || device >= 16) return FCUDA_Q2K_MMQ_REFUSED;
  device_info &info = g_device_info[device];
  if (!info.known) {
    int major = 0, minor = 0, nsm = 0, smpbo = 0;
    cudaError_t query_status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (query_status == cudaSuccess) query_status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (query_status == cudaSuccess) query_status = cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, device);
    if (query_status == cudaSuccess) query_status = cudaDeviceGetAttribute(&smpbo, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (query_status != cudaSuccess) return v41_checked_stream_error(50000, query_status);
    cudaFuncAttributes fa;
    query_status = cudaFuncGetAttributes(&fa, mul_mat_q<128, false>);
    if (query_status != cudaSuccess) return v41_checked_stream_error(50000, query_status);
    info.cc = 10*major + minor;
    info.compiled_arch = fa.ptxVersion;
    info.nsm = nsm;
    info.smpbo = (size_t) smpbo;
    info.known = true;
  }
  if (info.cc < 80 || info.compiled_arch < 80 || info.nsm <= 0 || info.smpbo < nbytes_shared(8)) return FCUDA_Q2K_MMQ_REFUSED;

  // ggml_cuda_mul_mat_q: the q8_1 activation holds every row padded to MATRIX_ROW_PADDING, plus
  // one J_max of blocks so the last y tile may read past the final token.
  const int64_t in_padded = FAK_Q2_MMQ_PAD((int64_t) in, MATRIX_ROW_PADDING);
  // Kernels index the packed activation in int words, including the tail read.
  if ((int64_t)P * (in_padded/QK8_1_MMQ) * (sizeof(block_q8_1_mmq)/sizeof(int)) +
      128*(sizeof(block_q8_1_mmq)/sizeof(int)) >= INT_MAX) return FCUDA_Q2K_MMQ_REFUSED;
  const size_t y_bytes = (size_t) P * in_padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ +
                         (size_t) 128 * sizeof(block_q8_1_mmq);
  void *y = scratch_malloc(y_bytes);
  if (!y) return FCUDA_SCRATCH_ALLOC_FAILED;

  const int64_t block_num_y = (in_padded + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
  const dim3 num_blocks(P, block_num_y, 1);
  const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
  // Defined extra token-tail blocks for the final shared-memory tile.
  cudaError_t init_status = cudaMemsetAsync((char *)y + y_bytes - 128*sizeof(block_q8_1_mmq), 0,
                                          128*sizeof(block_q8_1_mmq), g_stream);
  if (init_status != cudaSuccess) {
    const int status = v41_checked_stream_error(54000, init_status);
    fcuda_free(y);
    return status;
  }
  FCUDA_TS("quantize_mmq_q8_1_d2s6", num_blocks, quantize_mmq_q8_1_d2s6<<<num_blocks, block_size, 0, g_stream>>>(dX, y, in, in, in_padded, P));

  cudaError_t quant_status = cudaGetLastError();
  if (quant_status != cudaSuccess) {
    const int status = v41_checked_stream_error(55000, quant_status);
    fcuda_free(y);
    return status;
  }
  const bool fallback = out % 128 != 0;
  int status = fallback
      ? mul_mat_q_switch_J<true>((const char *) dQ2K, (const int *) y, dY, out, in, P, info, g_stream)
      : mul_mat_q_switch_J<false>((const char *) dQ2K, (const int *) y, dY, out, in, P, info, g_stream);
  fcuda_free(y);
  if (status == 0) {
    g_q2k_mmq_launches++;
    return 0;
  }
  return status; // Never retry/fall back after enqueue or mask the native failure.
}

#endif // FAK_CUDA_Q2K_MMQ_PREFILL
