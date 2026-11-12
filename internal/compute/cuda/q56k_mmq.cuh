// ---- Retained Q5_K/Q6_K prompt-panel MMQ family (qualification only) ----
// Adapted from llama.cpp at 0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd (MIT).
// Copyright (c) 2023-2026 The ggml authors. Full permission notice is in
// THIRD_PARTY_NOTICES.md ("llama.cpp").
//
// This is the Ampere integer-MMA path, also usable by later compiled devices.
// Q5_K uses DS4 (half scale and sum); Q6_K uses D4 (float scale), signed
// six-bit quants and separate signed 16-value subscales. Neither widens weights.
// It deliberately quantizes F32 activations to temporary Q8_1; parity and
// end-to-end cost must be qualified, not inferred from source or launch count.
// The old void ABI, P<128 panel math and independent P1..3 MMVQ remain unchanged.
// One explicit owner holds activation and optional Stream-K fixup buffers.
// Successful calls enqueue on g_stream; first-error status is captured inside
// each profiling scope. Failed drains latch shared AND private-loan ownership.
// No managed spill, global persistent workspace, extra success fence or retry.

extern "C" int fcuda_q56k_mmq_enabled(void) {
#ifdef FAK_CUDA_Q56K_MMQ_QUALIFICATION
    return 1;
#else
    return 0;
#endif
}
static unsigned long long g_q56k_mmq_launches[2] = {};
extern "C" unsigned long long fcuda_q56k_mmq_launch_count(int kind) {
    return kind == 5 || kind == 6 ? g_q56k_mmq_launches[kind-5] : 0;
}

#ifdef FAK_CUDA_Q56K_MMQ_QUALIFICATION
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define FAK_Q56K_MMQ_AVAILABLE 1
#endif

namespace fak_q56k_mmq {

constexpr int warp_size = 32;

constexpr int QK_K = 256;
constexpr int QK8_1 = 32;
constexpr int QR5_K = 2;
constexpr int QR6_K = 2;
constexpr int QI5_K = QK_K / (4*QR5_K);
constexpr int QI6_K = QK_K / (4*QR6_K);
constexpr int QI8_1 = QK8_1 / 4;
constexpr int MATRIX_ROW_PADDING = 512;
#define FAK_Q56K_MMQ_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

struct block_q5_K {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
};
struct block_q6_K {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t scales[QK_K/16];
    half d;
};
static_assert(sizeof(block_q5_K) == 176, "Q5_K canonical packed extent");
static_assert(sizeof(block_q6_K) == 210, "Q6_K canonical packed extent");

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

// Q6_K blocks are only two-byte aligned.
static __device__ __forceinline__ int get_int_b2(const void *x, const int &i32) {
    const uint16_t *x16 = (const uint16_t *)x;
    return (int)x16[2*i32] | ((int)x16[2*i32+1] << 16);
}
static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

constexpr int MMQ_ITER_K = 256;
constexpr int QK8_1_MMQ = 4*QK8_1;

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // Both layouts use the same 16-byte metadata prefix.
    union { half2 ds4[4]; float d4[4]; };
    int8_t qs[QK8_1_MMQ];
};
static_assert(sizeof(block_q8_1_mmq) == QK8_1_MMQ + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");

constexpr int MMQ_TILE_NE_K = 32;
constexpr int MMQ_TILE_Y_K = MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1;

// ggml_cuda_mmq_get_config_ampere(GGML_TYPE_Q5_K/GGML_TYPE_Q6_K, J, fallback) and its derived values.
constexpr int nthreads = 256;
constexpr int occupancy = 1;
constexpr int I = 128;
constexpr int nwarps = nthreads / warp_size;
constexpr int qk = QK_K;
constexpr int ITER_K = MMQ_ITER_K;
constexpr int blocks_per_iter = ITER_K / qk;
static_assert(blocks_per_iter == 1, "Q5_K/Q6_K work cap requires one quant block per K iteration.");
constexpr int sram_stride = 2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_1 + 4; // GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1
static_assert(sram_stride == 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI6_K + MMQ_TILE_NE_K/8 + 7, "Q6_K SRAM stride");
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

// The NVIDIA DATA_LAYOUT_I_MAJOR int tiles this instantiation uses.
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
        if constexpr (I == 8 && (J == 4 || J == 8)) {
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
#ifdef FAK_Q56K_MMQ_AVAILABLE
    int * xi = (int * ) t.x;
    const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(xs));
#else
    (void) t; (void) xs0; (void) stride;
    __trap();
#endif // FAK_Q56K_MMQ_AVAILABLE
}

static __device__ __forceinline__ void mma(tile<16, 8> & D, const tile<16, 8> & A, const tile<8, 8> & B) {
#ifdef FAK_Q56K_MMQ_AVAILABLE
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
#else
    (void) D; (void) A; (void) B;
    __trap();
#endif // FAK_Q56K_MMQ_AVAILABLE
}

static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}


static __device__ __forceinline__ void load_ldmatrix(tile<16, 4> &t, const int *xs0, const int stride) {
#ifdef FAK_Q56K_MMQ_AVAILABLE
    const int *xs = xs0 + (threadIdx.x % 16)*stride;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(t.x[0]), "=r"(t.x[1]) : "l"(xs));
#else
    (void)t; (void)xs0; (void)stride; __trap();
#endif
}
static __device__ __forceinline__ void mma(tile<16, 8> &D, const tile<16, 4> &A, const tile<8, 4> &B) {
#ifdef FAK_Q56K_MMQ_AVAILABLE
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
#else
    (void)D; (void)A; (void)B; __trap();
#endif
}

template <int J, bool fallback> static __device__ __forceinline__ void load_tiles_q5_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;
        const int ky = QR5_K*txi;

        const int ql = get_int_b4(bxi->qs, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b4(bxi->qh, txi % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (txi / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (txi / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + QI5_K/4;

        x_qs[i*sram_stride + kq0] = ql0 | qh0;
        x_qs[i*sram_stride + kq1] = ql1 | qh1;
    }

    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % I;
        {
            if (fallback) {
                i = min(i, i_max);
            }

            const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;

            const int * scales = (const int *) bxi->scales;
            const int ksc = threadIdx.x % 2;

            const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;

            const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
            for (int l = 0; l < int(sizeof(int)); ++l) {
                x_dm[i*sram_stride + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
            }
        }
    }
}

template <int J, bool fallback> static __device__ __forceinline__ void load_tiles_q6_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    int   * x_sc = (int   *) (x_df + MMQ_TILE_NE_K/QI6_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR6_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;

        const int ql = get_int_b2(bxi->ql, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b2(bxi->qh, (QI6_K/4) * (txi / (QI6_K/2)) + txi % (QI6_K/4));
        const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 =  (qh >> ((txi & 0x08) >> 2))       & 0x30303030;

        const int kq0 = 2*txi - txi % (QI6_K/2) + 0;
        const int kq1 = 2*txi - txi % (QI6_K/2) + QI6_K/2;

        x_qs[i*sram_stride + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*sram_stride + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;

        x_df[i*sram_stride]                     = bxi->d;
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + (threadIdx.x % (MMQ_TILE_NE_K/8)) / 4;

        x_sc[i*sram_stride + threadIdx.x%4] = get_int_b2(bxi->scales, threadIdx.x % (MMQ_TILE_NE_K/8));
    }
}

template <int J, bool fallback> static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16,  8> tile_A;
    typedef tile< 8,  8> tile_B;
    typedef tile<16,  8> tile_C;

    constexpr int rows_per_warp = rows_per_warp_for(J);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    tile_A   A[ntx][MMQ_TILE_NE_K/QI8_1];
    float2 dmA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_1];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_1], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                const int k0 = k00 + k01;

                dmA[n][l][k01/QI8_1] = __half22float2(x_dm[i*sram_stride + k0/QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B   B;
            float2 dsB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;
                }
            }
        }
    }
}

template <int J, bool fallback> static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {

    typedef tile<16, 4> tile_A;
    typedef tile< 8, 4> tile_B;
    typedef tile<16, 8> tile_C;
    constexpr int rows_per_warp = rows_per_warp_for(J);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A   A[ntx][8];
    int    scA[ntx][tile_C::ne/2][8];
    float   dA[ntx][tile_C::ne/2];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/4 + 0], x_qs + (i0 + n*tile_A::I)*sram_stride + (k0 + 0),         sram_stride);
            load_ldmatrix(A[n][k01/4 + 1], x_qs + (i0 + n*tile_A::I)*sram_stride + (k0 + tile_A::J), sram_stride);
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 16) {
            const int k0 = k00 + k01;

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

                const int      sc_packed = x_sc[i*sram_stride + k0/16];
                const int8_t * sc        = (const int8_t *) &sc_packed;

#pragma unroll
                for (int ksc = 0; ksc < sizeof(int); ++ksc) {
                    scA[n][l][k01/4 + ksc] = sc[ksc];
                }
            }
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

            dA[n][l] = x_df[i*sram_stride];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
        float tmp[ntx][tile_C::ne] = {{0.0f}};

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + 0         + k01, MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + tile_B::J + k01, MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2];
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    tmp[n][l] += (C[0].x[l]*scA[n][l/2][k01/4 + 0] + C[1].x[l]*scA[n][l/2][k01/4 + 1])*dB[l%2];
                }
            }
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp[n][l]*dA[n][l/2];
            }
        }
    }
}

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

template <int kind, int J, bool fallback, bool fixup>
static __device__ __forceinline__ void mul_mat_q_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q;
    int * tile_x = tile_y + FAK_Q56K_MMQ_PAD(J*MMQ_TILE_Y_K, nwarps*warp_size);

    constexpr int ne_block = QK8_1_MMQ;

    float sum[J*I / (nwarps*warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        if constexpr (kind == 5) load_tiles_q5_K<J, fallback>(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        else load_tiles_q6_K<J, fallback>(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        if constexpr (kind == 5) vec_dot_q8_1_q8_1_mma<J, fallback>(tile_x, tile_y, sum, 0);
        else vec_dot_q6_K_q8_1_mma<J, fallback>(tile_x, tile_y, sum, 0);

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

        if constexpr (kind == 5) vec_dot_q8_1_q8_1_mma<J, fallback>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        else vec_dot_q6_K_q8_1_mma<J, fallback>(tile_x, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

    if (fixup) {
        write_back_mma<J, fallback>(sum, tmp_fixup + blockIdx.x*(J*I), I, I, J);
    } else {
        write_back_mma<J, fallback>(sum, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598

template <int kind, int J, bool fallback>
__launch_bounds__(nthreads, occupancy)
static __global__ void mul_mat_q(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y,
        const int stride_col_dst, const uint3 ntx) {
#ifdef FAK_Q56K_MMQ_AVAILABLE
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
        mul_mat_q_process_tile<kind, J, fallback, fixup>
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
    mul_mat_q_process_tile<kind, J, fallback, fixup>
        (x, offset_x, y + offset_y, dst + offset_dst, tmp_fixup,
         stride_row_x, ncols_y, stride_col_dst,
         tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
#else
    (void) x; (void) y; (void) dst; (void) tmp_fixup; (void) blocks_per_ne00; (void) nrows_x;
    (void) ncols_dst; (void) stride_row_x; (void) ncols_y; (void) stride_col_dst; (void) ntx;
    __trap();
#endif // FAK_Q56K_MMQ_AVAILABLE
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

// quantize.cu: ordinary one-channel DS4/D4 layouts, no expert scatter.
constexpr int CUDA_QUANTIZE_BLOCK_SIZE_MMQ = 128;
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

template <int kind>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1) {

    constexpr int vals_per_scale = 32;
    constexpr int vals_per_sum   = 32;

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

    float sum = 0.0f;
    if constexpr (kind == 5) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, warp_size);
    }
    // Explicit zero block avoids 0*Inf conversion; nonzero arithmetic is donor-exact.
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

    if (iqs % 32 == 0) {
        if constexpr (kind == 5) y[ib].ds4[iqs/32] = make_half2(d, sum);
        else y[ib].d4[iqs/32] = d;
    }
}

struct device_info {
    bool known;
    int cc, compiled_arch, nsm, grid_x, grid_y;
    size_t smpbo;
    bool smem_raised[2][129];
};
static device_info g_device_info[2]; // device 0, one entry per packed format

static size_t nbytes_shared(int J) {
    return (size_t)I*sram_stride*sizeof(int) +
           FAK_Q56K_MMQ_PAD((size_t)J*sizeof(block_q8_1_mmq), (size_t)nthreads*sizeof(int));
}

struct launch_plan {
    int J, blocks;
    bool fallback, fixup;
    size_t activation_bytes, fixup_bytes;
};

static int plan_geometry(int out, int in, int p, device_info &info, launch_plan &plan) {
    if (out <= 0 || in <= 0 || p < 128 || in % QK_K ||
        (int64_t)out*in >= INT_MAX || (int64_t)p*in >= INT_MAX ||
        (int64_t)p*out >= INT_MAX || info.nsm <= 0) return -75701;
    const int64_t padded = FAK_Q56K_MMQ_PAD((int64_t)in, MATRIX_ROW_PADDING);
    const int64_t qblocks = padded / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    if (p > info.grid_x || qblocks > info.grid_y) return -75701;
    plan = {};
    plan.fallback = out % I != 0;
    int best_tiles = INT_MAX;
    for (int J = 8; J <= 128 && best_tiles > 1; J += 8) {
        if (!config_has_J(J, plan.fallback) || nbytes_shared(J) > info.smpbo) continue;
        const int tiles = p/J + (p%J != 0);
        if (tiles < best_tiles) { plan.J = J; best_tiles = tiles; }
    }
    if (!plan.J) return -75705;
    const int64_t ntiles = (int64_t)best_tiles*(out/I + (out%I != 0));
    const int64_t work = ntiles*(in/QK_K);
    if (work <= 0 || work >= (1LL<<30)) return -75701;
    const int64_t waves = (ntiles + info.nsm - 1)/info.nsm;
    const int efficiency = (int)(100*ntiles/(info.nsm*waves));
    // The same useful-work cap as the retained Q4_K Stream-K producer.
    plan.blocks = efficiency >= 90 ? (int)ntiles : (int)(work < info.nsm ? work : info.nsm);
    plan.fixup = ntiles % plan.blocks != 0;
    const uint64_t ybytes = (uint64_t)p*padded/QK8_1_MMQ*sizeof(block_q8_1_mmq) +
                            (uint64_t)128*sizeof(block_q8_1_mmq);
    const uint64_t fbytes = plan.fixup ? (uint64_t)plan.blocks*plan.J*I*sizeof(float) : 0;
    if (ybytes > SIZE_MAX || fbytes > SIZE_MAX || ybytes > SIZE_MAX-fbytes) return -75701;
    plan.activation_bytes = (size_t)ybytes;
    plan.fixup_bytes = (size_t)fbytes;
    return 0;
}

template <int kind>
static int device_state(device_info *&out_info) {
    if (!g_stream || v41_candidates_stream_uncertain()) return -75704;
    if (g_capture_open || g_q4k_qkv_async_pending != 0) return -75702;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaError_t e = cudaStreamIsCapturing(g_stream, &capture);
    if (e != cudaSuccess) return (int)e;
    if (capture != cudaStreamCaptureStatusNone) return -75702;
    int device = -1;
    e = cudaGetDevice(&device);
    if (e != cudaSuccess) return (int)e;
    if (device != 0) return -75705;
    device_info &info = g_device_info[kind-5];
    if (!info.known) {
        cudaDeviceProp prop{};
        e = cudaGetDeviceProperties(&prop, device);
        if (e != cudaSuccess) return (int)e;
        cudaFuncAttributes fa{};
        e = cudaFuncGetAttributes(&fa, mul_mat_q<kind, 128, false>);
        if (e != cudaSuccess) return (int)e;
        info.cc = 10*prop.major + prop.minor;
        info.compiled_arch = fa.ptxVersion;
        info.nsm = prop.multiProcessorCount;
        info.grid_x = prop.maxGridSize[0];
        info.grid_y = prop.maxGridSize[1];
        info.smpbo = prop.sharedMemPerBlockOptin;
        info.known = true;
    }
    if (info.cc < 80 || info.compiled_arch < 80) return -75705;
    out_info = &info;
    return (int)cudaPeekAtLastError(); // never clear a peer launch refusal
}

static int extent(const void *p, size_t n, size_t align) {
    if (!p || (uintptr_t)p % align || n > UINTPTR_MAX-(uintptr_t)p || v41c_pending_contains(p))
        return -75703;
    auto live = g_live.find((void *)p);
    auto panel = g_panel_live.find((void *)p);
    if ((live == g_live.end() || live->second < n) &&
        (panel == g_panel_live.end() || panel->second < n)) return -75703;
    cudaPointerAttributes attr{};
    cudaError_t e = cudaPointerGetAttributes(&attr, p);
    if (e != cudaSuccess) return (int)e;
    return attr.type == cudaMemoryTypeDevice && attr.device == 0 ? 0 : -75703;
}
static bool overlap(const void *a, size_t an, const void *b, size_t bn) {
    return (uintptr_t)a < (uintptr_t)b+bn && (uintptr_t)b < (uintptr_t)a+an;
}

template <int kind>
static int preflight(const uint8_t *w, const float *x, int out, int in, int p,
                     device_info *&info, launch_plan &plan) {
    int rc = device_state<kind>(info);
    if (rc) return rc;
    rc = plan_geometry(out, in, p, *info, plan);
    if (rc) return rc;
    const size_t wb = (size_t)out*(in/QK_K)*(kind == 5 ? 176 : 210);
    rc = extent(w, wb, kind == 5 ? 4 : 2);
    if (rc || !x) return rc; // weight-only whole-operation admission
    const size_t xb = (size_t)p*in*sizeof(float);
    rc = extent(x, xb, 16);
    if (rc) return rc;
    return overlap(w, wb, x, xb) ? -75703 : 0;
}

// A failed drain must not be laundered by later split-end or allocator activity.
// Keep the first failure; all uncertain allocations remain in native live maps.
static int retire(void *activation, void *fixup, int first) {
    if (first) {
        const cudaError_t drained = cudaStreamSynchronize(g_stream);
        sequence_split_note_failed_fence(drained);
        if (drained != cudaSuccess && !g_cuda_cleanup_status)
            g_cuda_cleanup_status = (int)drained;
    }
    if (g_cuda_cleanup_status)
        sequence_split_note_failed_fence((cudaError_t)g_cuda_cleanup_status);
    fcuda_free(fixup);
    fcuda_free(activation);
    if (g_cuda_cleanup_status)
        sequence_split_note_failed_fence((cudaError_t)g_cuda_cleanup_status);
    return first ? first : g_cuda_cleanup_status;
}
static int allocation_failure() {
    const int status = (int)cudaPeekAtLastError();
    return status ? status : FCUDA_SCRATCH_ALLOC_FAILED;
}

template <int kind, int J, bool fallback>
static int run_variant(const uint8_t *w, const float *x, float *y, int out, int in, int p,
                       device_info &info, const launch_plan &plan) {
    static_assert(kind == 5 || kind == 6, "packed family only");
    const int shared_bytes = (int)nbytes_shared(J);
    if (!info.smem_raised[fallback][J]) {
        const cudaError_t e = cudaFuncSetAttribute(mul_mat_q<kind,J,fallback>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
        if (e != cudaSuccess) return (int)e;
        info.smem_raised[fallback][J] = true;
    }
    // Both owned allocations precede the first enqueue. Neither uses managed memory.
    void *activation = fcuda_malloc_tagged(plan.activation_bytes, FAK_ALLOC_PREFILL_SCRATCH,
                                           FAK_ALLOC_SITE("q56k-mmq-q8_1"));
    if (!activation) return allocation_failure();
    float *fixup = nullptr;
    if (plan.fixup) {
        fixup = (float *)fcuda_malloc_tagged(plan.fixup_bytes, FAK_ALLOC_PREFILL_SCRATCH,
                                            FAK_ALLOC_SITE("q56k-mmq-stream-k"));
        if (!fixup) return retire(activation, nullptr, allocation_failure());
    }
    // The final shared Y tile may over-read into the J_max tail. Initialize it
    // before quantization; no pool bytes or prior request metadata are consumed.
    int rc;
    FCUDA_TS("fill", dim3(0, 0), rc = (int)cudaMemsetAsync(activation, 0, plan.activation_bytes, g_stream));
    if (rc) return retire(activation, fixup, rc);
    const int64_t padded = FAK_Q56K_MMQ_PAD((int64_t)in, MATRIX_ROW_PADDING);
    const dim3 qgrid(p, padded/(4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ), 1);
    FCUDA_TS("q56k_mmq_quantize", qgrid, {
        quantize_mmq_q8_1<kind><<<qgrid, CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 0, g_stream>>>
            (x, activation, in, in, padded, p);
        rc = (int)cudaGetLastError();
    });
    if (rc) return retire(activation, fixup, rc);
    const dim3 block(warp_size, nwarps, 1), grid(plan.blocks, 1, 1);
    const uint3 kb = init_fastdiv_values(in/QK_K);
    const uint3 nx = init_fastdiv_values(p/J + (p%J != 0));
    FCUDA_TS("q56k_mmq_mul_mat_q", grid, {
        mul_mat_q<kind,J,fallback><<<grid, block, shared_bytes, g_stream>>>
            ((const char *)w, (const int *)activation, y, fixup, kb, out, p,
             in/QK_K, p, out, nx);
        rc = (int)cudaGetLastError();
    });
    if (rc) return retire(activation, fixup, rc);
    if (plan.fixup) {
        const dim3 fgrid(plan.blocks, I/warp_size, 1), fblock(warp_size, nwarps/2, 1);
        FCUDA_TS("q56k_mmq_mul_mat_q_stream_k_fixup", fgrid, {
            mul_mat_q_stream_k_fixup<J,fallback><<<fgrid, fblock, 0, g_stream>>>
                (y, fixup, kb, out, p, out, nx);
            rc = (int)cudaGetLastError();
        });
    }
    rc = retire(activation, fixup, rc);
    if (!rc) ++g_q56k_mmq_launches[kind-5]; // submissions, not completion or speed
    return rc;
}

template <int kind, bool fallback>
static int dispatch(const uint8_t *w, const float *x, float *y, int out, int in, int p,
                    device_info &info, const launch_plan &plan) {
#define FAK_Q56K_MMQ_CASE(J) case J: return run_variant<kind,J,fallback>(w,x,y,out,in,p,info,plan)
    switch (plan.J) {
        FAK_Q56K_MMQ_CASE(8); FAK_Q56K_MMQ_CASE(16); FAK_Q56K_MMQ_CASE(24);
        FAK_Q56K_MMQ_CASE(32); FAK_Q56K_MMQ_CASE(40); FAK_Q56K_MMQ_CASE(48);
        FAK_Q56K_MMQ_CASE(64); FAK_Q56K_MMQ_CASE(80); FAK_Q56K_MMQ_CASE(96);
        FAK_Q56K_MMQ_CASE(112); FAK_Q56K_MMQ_CASE(128);
        default: return -75705;
    }
#undef FAK_Q56K_MMQ_CASE
}
template <int kind>
static int run(const uint8_t *w, const float *x, float *y, int out, int in, int p) {
    if (!x) return -75703;
    device_info *info = nullptr;
    launch_plan plan{};
    int rc = preflight<kind>(w,x,out,in,p,info,plan);
    if (rc) return rc;
    const size_t yb = (size_t)p*out*sizeof(float), xb = (size_t)p*in*sizeof(float);
    const size_t wb = (size_t)out*(in/QK_K)*(kind == 5 ? 176 : 210);
    rc = extent(y,yb,alignof(float));
    if (rc) return rc;
    if (overlap(y,yb,x,xb) || overlap(y,yb,w,wb)) return -75703;
    return plan.fallback ? dispatch<kind,true>(w,x,y,out,in,p,*info,plan)
                         : dispatch<kind,false>(w,x,y,out,in,p,*info,plan);
}
} // namespace fak_q56k_mmq
#endif // FAK_CUDA_Q56K_MMQ_QUALIFICATION

extern "C" int fcuda_q56k_mmq_preflight(const uint8_t *w, const float *x,
                                        int out, int in, int p, int kind) {
#ifdef FAK_CUDA_Q56K_MMQ_QUALIFICATION
    if (p < 128) return 0; // this opt-in does not change decode or short-panel admission
    fak_q56k_mmq::device_info *info = nullptr;
    fak_q56k_mmq::launch_plan plan{};
    if (kind == 5) return fak_q56k_mmq::preflight<5>(w,x,out,in,p,info,plan);
    if (kind == 6) return fak_q56k_mmq::preflight<6>(w,x,out,in,p,info,plan);
    return -75701;
#else
    (void)w; (void)x; (void)out; (void)in; (void)p; (void)kind;
    return 0;
#endif
}

// Explicit ordinary-projection ABI: never replace the legacy checked functions
// also consumed by packed GDN's independent native projection/fence contract.
extern "C" int fcuda_q56k_mmq_matmul_checked_f32(const uint8_t *w, const float *x,
                                                float *y, int out, int in, int p, int kind) {
#ifdef FAK_CUDA_Q56K_MMQ_QUALIFICATION
    if (kind == 5) return fak_q56k_mmq::run<5>(w,x,y,out,in,p);
    if (kind == 6) return fak_q56k_mmq::run<6>(w,x,y,out,in,p);
    return -75701;
#else
    (void)w; (void)x; (void)y; (void)out; (void)in; (void)p; (void)kind;
    return -75705;
#endif
}

// A failed ordinary Read is already a completion observation. Preserve that
// exact status in both owner domains without retrying or adding a new fence.
extern "C" void fcuda_q56k_mmq_retirement_failed(int first) {
    if (!first) return;
    sequence_split_note_failed_fence((cudaError_t)first);
    if (!g_cuda_cleanup_status) g_cuda_cleanup_status = first;
}
