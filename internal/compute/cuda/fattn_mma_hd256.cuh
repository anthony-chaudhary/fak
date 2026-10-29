// ---- Qwen3.8 hd256 tensor-core prompt attention (P-05, fak#13984) ------------------
// Adapted from llama.cpp ggml/src/ggml-cuda/fattn-mma-f16.cuh, mma.cuh and cp-async.cuh at
// 0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd (MIT).
// Copyright (c) 2023-2026 The ggml authors. The MIT permission notice is reproduced in
// THIRD_PARTY_NOTICES.md ("llama.cpp fattn-mma-f16 hd256 prompt attention").
//
// One upstream instantiation is kept: DKQ = DV = 256, ncols1 = 8, ncols2 = 8 (GQA-packed:
// Qwen3.8 has 6 query heads per KV head, so 6 of the 8 packed columns do work), the Ampere
// config row (256, 256, 64): 128 threads, occupancy 2, nbatch_fa 32, nbatch_K2/V2/combine 128,
// two cp.async stages, Q kept in registers. The function bodies below are the upstream
// NVIDIA (TURING_MMA_AVAILABLE) code paths with every C++17 constexpr-if resolved for that one
// instantiation, because this translation unit compiles as C++14 (the deltas are recorded in
// testdata/qwen38_kernel_provenance.json). Other changes for Fak:
//   * The ggml mask tensor is replaced by the implicit causal bound key <= prefix + row, and
//     KV tiles that no row of a query tile can see are skipped (upstream KV_max).
//   * Fak keeps f32 K/V. The launcher stages them once per call into an f16 copy padded with
//     zeros to a multiple of nbatch_fa, so upstream's padded-K, cp.async loader runs unchanged.
//   * No ALiBi, logit softcap, sinks, V-is-K view, stream-k split or fixup buffer: one CUDA
//     block owns one (query tile, KV head) pair.
//   * Q and the output stay f32 in Fak's [tokens, heads*256] row-major layout, which matches
//     ggml's Q/dst strides. As upstream, KQ and the softmax statistics accumulate in f32 and
//     V*softmax(KQ) accumulates in f16 (mma .f16.f16.f16.f16).

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define FAK_FATTN_MMA_AVAILABLE 1
#endif

namespace fak_fattn_mma {

constexpr int warp_size = 32;

// fattn-common.cuh
#define FAK_FATTN_SOFTMAX_FTZ_THRESHOLD -20.0f
#define FAK_FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

// ggml_cuda_fattn_mma_get_config_ampere(256, 256, 64)
constexpr int DKQ_QWEN38 = 256;
constexpr int ncols1_qwen38 = 8;
constexpr int ncols2_qwen38 = 8;
constexpr int nthreads = 128;
constexpr int occupancy = 2;
constexpr int nbatch_fa = 32;
constexpr int nbatch_K2 = 128;
constexpr int nbatch_V2 = 128;
constexpr int nbatch_combine = 128;
constexpr int nstages = 2;
constexpr bool Q_in_reg = true;

// mma.cuh: the NVIDIA DATA_LAYOUT_I_MAJOR tiles this instantiation uses.
template <int I_, int J_, typename T>
struct tile {};

template <int I_, int J_>
struct tile<I_, J_, float> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32;
    float x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        if (I == 8 && J == 4) {
            return threadIdx.x / 4;
        } else if (I == 8 && J == 8) {
            return threadIdx.x / 4;
        } else if (I == 16 && J == 8) {
            return ((l / 2) * 8) + (threadIdx.x / 4);
        } else if (I == 16 && J == 16) {
            return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
        } else {
            return -1;
        }
    }

    static __device__ __forceinline__ int get_j(const int l) {
        if (I == 8 && J == 4) {
            return threadIdx.x % 4;
        } else if (I == 8 && J == 8) {
            return (l * 4) + (threadIdx.x % 4);
        } else if (I == 16 && J == 8) {
            return ((threadIdx.x % 4) * 2) + (l % 2);
        } else if (I == 16 && J == 16) {
            return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
        } else {
            return -1;
        }
    }
};

template <int I_, int J_>
struct tile<I_, J_, half2> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / warp_size;
    half2 x[ne] = {{0.0f, 0.0f}};

    static __device__ __forceinline__ int get_i(const int l) {
        if (I == 8 && J == 8) {
            return threadIdx.x / 4;
        } else if (I == 16 && J == 4) {
            return (l * 8) + (threadIdx.x / 4);
        } else if (I == 16 && J == 8) {
            return ((l % 2) * 8) + (threadIdx.x / 4);
        } else if (I == 32 && J == 8) {
            return ((l / 4) * 16) + ((l % 2) * 8) + (threadIdx.x / 4);
        } else {
            return -1;
        }
    }

    static __device__ __forceinline__ int get_j(const int l) {
        if (I == 8 && J == 8) {
            return (l * 4) + (threadIdx.x % 4);
        } else if (I == 16 && J == 4) {
            return threadIdx.x % 4;
        } else if (I == 16 && J == 8) {
            return ((l / 2) * 4) + (threadIdx.x % 4);
        } else if (I == 32 && J == 8) {
            return ((l & 2) * 2) + (threadIdx.x % 4);
        } else {
            return -1;
        }
    }
};

template <int I, int J>
static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
    tile<I, J/2, half2> ret;
#pragma unroll
    for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
        ret.x[l0/2] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
    }
    return ret;
}

template <typename T>
static __device__ __forceinline__ void load_ldmatrix(
        tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
#ifdef FAK_FATTN_MMA_AVAILABLE
    int * xi = (int * ) t.x;
    const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(xs));
#else
    (void) t; (void) xs0; (void) stride;
#endif // FAK_FATTN_MMA_AVAILABLE
}

template <int I, typename T>
static __device__ __forceinline__ void load_ldmatrix_trans(
        tile<I, 8, T> & t, const T * __restrict__ xs0, const int stride) {
#ifdef FAK_FATTN_MMA_AVAILABLE
    static_assert(I == 16, "bad tile width");
    int * xi = (int *) t.x;
    const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
        : "l"(xs));
#else
    (void) t; (void) xs0; (void) stride;
#endif // FAK_FATTN_MMA_AVAILABLE
}

static __device__ __forceinline__ void mma(
        tile<16, 8, half2> & D, const tile<16, 8, half2> & A, const tile<16, 8, half2> & B) {
#ifdef FAK_FATTN_MMA_AVAILABLE
    const int * Axi = (const int *) A.x;
    const int * Bxi = (const int *) B.x;
    int       * Dxi = (int       *) D.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(Dxi[0]), "+r"(Dxi[1])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
#else
    (void) D; (void) A; (void) B;
#endif // FAK_FATTN_MMA_AVAILABLE
}

static __device__ __forceinline__ void mma(
        tile<16, 16, float> & D, const tile<16, 8, half2> & A, const tile<16, 8, half2> & B) {
#ifdef FAK_FATTN_MMA_AVAILABLE
    const int * Axi = (const int *) A.x;
    const int * Bxi = (const int *) B.x;
    int       * Dxi = (int       *) D.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(Dxi[4]), "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
#else
    (void) D; (void) A; (void) B;
#endif // FAK_FATTN_MMA_AVAILABLE
}

// cp-async.cuh
static __device__ __forceinline__ unsigned int cvta_generic_to_shared(void * generic_ptr) {
#ifdef FAK_FATTN_MMA_AVAILABLE
    return (unsigned int) __cvta_generic_to_shared(generic_ptr);
#else
    (void) generic_ptr;
    return 0;
#endif // FAK_FATTN_MMA_AVAILABLE
}

template <int preload>
static __device__ __forceinline__ void cp_async_cg_16(const unsigned int dst, const void * src) {
    static_assert(preload == 64, "bad preload");
#ifdef FAK_FATTN_MMA_AVAILABLE
    asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;"
        : : "r"(dst), "l"(src));
#else
    (void) dst; (void) src;
#endif // FAK_FATTN_MMA_AVAILABLE
}

static __device__ __forceinline__ void cp_async_wait_all() {
#ifdef FAK_FATTN_MMA_AVAILABLE
    asm volatile("cp.async.wait_all;");
#endif // FAK_FATTN_MMA_AVAILABLE
}

// common.cuh ggml_cuda_unroll
template <int n>
struct unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        unroll<n - 1>{}(f, args...);
    }
};

template <>
struct unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

// flash_attn_ext_f16_load_tile, use_cp_async = true (oob_check = false: K/V are padded).
template<int stride_tile, int nwarps, int nbatch_fa>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV) {
    // K/V data is loaded with decreasing granularity for D for better memory bandwidth.
    // The minimum granularity is 16 bytes.
    constexpr int h2_per_chunk = 16/sizeof(half2);
    const int chunks_per_row = D2 / h2_per_chunk;
    constexpr int preload = 64;

    const unsigned int tile_KV_32 = cvta_generic_to_shared(tile_KV);

    auto load = [&] (const int n) {
        const int stride_k = warp_size >> n;
        const int k0_start = stride_k == warp_size ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
        const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
        const int stride_i = warp_size / stride_k;

        if (k0_start == k0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                break;
            }

#pragma unroll
            for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                cp_async_cg_16<preload>(tile_KV_32 + i*(stride_tile*sizeof(half2)) + k*16, KV + i*stride_KV + k*h2_per_chunk);
            }
        }
    };
    // 1: max 32*16=512 bytes, 256 half
    // 2: max 16*16=256 bytes, 128 half
    // 3: max  8*16=128 bytes,  64 half
    // 4: max  4*16= 64 bytes,  32 half
    // 5: max  2*16= 32 bytes,  16 half
    // 6: max  1*16= 16 bytes,   8 half
    unroll<6>{}(load);
}

// flash_attn_ext_f16_iter with nstages = 2, Q_in_reg, cols_per_warp = 16 (Turing+ mma),
// no softcap and no V-is-K view; the mask add is the implicit causal bound.
template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool last_iter,
    typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const int stride_K,
        const int stride_V,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        T_B_KQ       * const __restrict__ Q_B,
        T_C_VKQ      * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int kb0,
        const int causal_row0) {
#ifdef FAK_FATTN_MMA_AVAILABLE
    constexpr int  ncols           = ncols1 * ncols2;
    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = 2;
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    static_assert(cols_per_warp == 16, "only the wide (16-column) KQ tile is ported");

    constexpr int stride_tile_K = nbatch_K2 + 4;
    constexpr int stride_tile_V = nbatch_V2 + 4;

    const int k_VKQ_0 = kb0 * nbatch_fa;
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];

    static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi stage loading");
    cp_async_wait_all();
    __syncthreads();
    flash_attn_ext_f16_load_tile<stride_tile_V, nwarps, nbatch_fa>
        (V_h2 + int64_t(k_VKQ_0)*stride_V, tile_V, nbatch_V2, stride_V);

#pragma unroll
    for (int k0_start = (DKQ/2-1) - (DKQ/2-1) % nbatch_K2; k0_start >= 0; k0_start -= nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < DKQ/2 ? k0_start + nbatch_K2 : DKQ/2;

        // Calculate tile of KQ:
#pragma unroll
        for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
            const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
#pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                T_A_KQ K_A;
                load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);
                // Wide version of KQ_C is column-major, swap A and B for CUDA.
                mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[k_KQ_0/T_A_KQ::J], K_A);
            }
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    // Fak: implicit causal mask in place of the ggml mask tile. Key k_VKQ_0 + i is visible to
    // query row causal_row0 + j (prefix + jt*ncols1 + j) when i <= row.
#pragma unroll
    for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
        const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::J;
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            const int i = k_VKQ_0 + i0 + T_C_KQ::get_j(l);
            const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l)) / ncols2;

            if (i > causal_row0 + j) {
                KQ_C[i00/(np*T_C_KQ::J)].x[l] = -INFINITY;
            }
        }
    }

    // Calculate softmax for each KQ column using the current max. value.
    // The divisor is stored in KQ_rowsum and will be applied at the end.
    static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            // Turing + Volta:
            const int KQ_idx = (l/2) % 2;
            KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FAK_FATTN_KQ_MAX_OFFSET);
        }
    }

#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        // Values per KQ column are spread across 4 threads:
        constexpr int offset_first = 2;
        constexpr int offset_last  = 1;
#pragma unroll
        for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
            KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
        }
    }

    static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            // Turing + Volta:
            const int KQ_idx = (l/2) % 2;
            KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[KQ_idx]);
            KQ_rowsum_add[KQ_idx] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= FAK_FATTN_SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                    VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                }
            }
        }
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
        B[k] = get_half2(KQ_C[k]);
    }

    // Preload K tile for next iteration:
    cp_async_wait_all();
    __syncthreads();
    if (!last_iter) {
        flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa>
            (K_h2 + int64_t(k_VKQ_0 + nbatch_fa)*stride_K, tile_K, nbatch_K2, stride_K);
    }

    // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
#pragma unroll
    for (int i0_start = 0; i0_start < DV; i0_start += 2*nbatch_V2) {
        static_assert(DV % (2*nbatch_V2) == 0, "bad loop size");
        const int i0_stop = i0_start + 2*nbatch_V2;
        const half2 * tile_V_i = tile_V;

#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += T_A_VKQ::I) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

                T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
                load_ldmatrix_trans(A, tile_V_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
                // Wide version of VKQ_C is column-major, swap A and B for CUDA.
                mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/(np*T_A_VKQ::J)], A);
            }
        }
    }
#else
    (void) K_h2; (void) V_h2; (void) stride_K; (void) stride_V; (void) tile_K; (void) tile_V;
    (void) Q_B; (void) VKQ_C; (void) KQ_max; (void) KQ_rowsum; (void) kb0; (void) causal_row0;
#endif // FAK_FATTN_MMA_AVAILABLE
}

// mma_tile_sizes<DV, ncols> for TURING_MMA_AVAILABLE, ncols != 8.
struct mma_tile_sizes_wide {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};

// flash_attn_ext_f16_process_tile for needs_fixup = is_fixup = false, no sinks.
template<int DKQ, int DV, int ncols1, int ncols2, int nwarps>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        float2       * const __restrict__ dstk,
        const float scale,
        const int ne01,
        const int ne02,
        const int gqa_ratio,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int jt,
        const int zt_gqa,
        const int kb0_start,
        const int kb0_stop,
        const int causal_row0) {
#ifdef FAK_FATTN_MMA_AVAILABLE
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int ncols = ncols1 * ncols2;
    using     T_A_KQ    = mma_tile_sizes_wide::T_A_KQ;
    using     T_B_KQ    = mma_tile_sizes_wide::T_B_KQ;
    using     T_C_KQ    = mma_tile_sizes_wide::T_C_KQ;
    using     T_A_VKQ   = mma_tile_sizes_wide::T_A_VKQ;
    using     T_B_VKQ   = mma_tile_sizes_wide::T_B_VKQ;
    using     T_C_VKQ   = mma_tile_sizes_wide::T_C_VKQ;

    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = 2;
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    static_assert(np == 1, "the cross-warp VKQ combine is not ported");

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    constexpr int stride_tile_Q = DKQ/2     + 4;
    constexpr int stride_tile_K = nbatch_K2 + 4;

    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = tile_Q;
    half2 * tile_V    = tile_K + nbatch_fa * stride_tile_K;

    T_B_KQ    Q_B[DKQ/(2*T_B_KQ::J)];
    T_C_VKQ VKQ_C[DV/(2*T_C_VKQ::J)];

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        const int stride_k  = warp_size >> n;
        const int k0_start  = stride_k == warp_size ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
        const int stride_jc = warp_size / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if ((ncols1 == 1 || jt*ncols1 + j < ne01) && (ncols2 == 1 || zt_gqa*ncols2 + c < gqa_ratio)) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
        }
    }

    __syncthreads();

    int kb0 = kb0_start;

    // Preload K data for first iteration when using cp_async with multiple stages:
    flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa>
        (K_h2 + int64_t(kb0)*nbatch_fa*stride_K, tile_K, nbatch_K2, stride_K);

    // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
    for (; kb0 < kb0_stop-1; ++kb0) {
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, false,
             T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (K_h2, V_h2, stride_K, stride_V, tile_K, tile_V, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, kb0, causal_row0);
    }
    flash_attn_ext_f16_iter
        <DKQ, DV, ncols1, ncols2, nwarps, true,
         T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
        (K_h2, V_h2, stride_K, stride_V, tile_K, tile_V, Q_B, VKQ_C,
         KQ_max, KQ_rowsum, kb0, causal_row0);

    // With multi-stage loading there is no __syncthreads at the end of the iter,
    //     there can be a race condition on shared memory access for combining/writing back results.
    if (nwarps*cols_per_warp > nbatch_fa) {
        __syncthreads();
    }

    // Finally, sum up partial KQ rowsums.
    {
        // The partial sums are spread across 8/4 threads.
        constexpr int offset_first = 2;
        constexpr int offset_last  = 1;
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, warp_size);
            }
        }
    }

    // Write VKQ accumulators to shared memory in column-major format (np == 1), then do
    // one large write to VRAM.
    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");

    {
        // jc_cwm = jc combine write meta
        // KQ_cmr = KQ combine max rowsum
        // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
        const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;

        if (thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        {
            const int j0 = threadIdx.y*cols_per_warp;
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    const int j = j0 + T_C_VKQ::get_i(l);
                    const int k = k1 + T_C_VKQ::get_j(l);

                    tile_Q[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
                }
            }
        }

        __syncthreads();

        {
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int stride_k  = warp_size >> n;
                const int k0_start  = stride_k == warp_size ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
                const int stride_jc = warp_size / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j_dst = jc_dst / ncols2;
                    const int c_dst = jc_dst % ncols2;

                    if ((ncols1 > 1 && jt*ncols1 + j_dst >= ne01) || (ncols2 > 1 && zt_gqa*ncols2 + c_dst >= gqa_ratio)) {
                        continue;
                    }

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = __half22float2(tile_Q[jc_tile_K * tile_stride + k]);

                        const float KQ_rowsum_j = meta_j[1];
                        dstk_val.x /= KQ_rowsum_j;
                        dstk_val.y /= KQ_rowsum_j;

                        dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                    }
                }
            }
        }
    }
#else
    (void) Q_f2; (void) K_h2; (void) V_h2; (void) dstk; (void) scale; (void) ne01; (void) ne02;
    (void) gqa_ratio; (void) stride_Q1; (void) stride_Q2; (void) stride_K; (void) stride_V;
    (void) jt; (void) zt_gqa; (void) kb0_start; (void) kb0_stop; (void) causal_row0;
#endif // FAK_FATTN_MMA_AVAILABLE
}

// Fak entry (replaces flash_attn_ext_f16's stream-k loop): blockIdx.x is the query tile,
// heaviest causal tiles first, blockIdx.y the KV head. Q/dst are f32 [tokens, nH*256];
// K16/V16 are the staged f16 copies [padded positions, nKV*256].
__launch_bounds__(nthreads, occupancy)
static __global__ void k_qwen38_fattn_mma_hd256(
        const float * __restrict__ Q, const half2 * __restrict__ K16, const half2 * __restrict__ V16,
        float * __restrict__ dst, const int tokens, const int prefix, const int nH, const int nKV,
        const float scale) {
#ifdef FAK_FATTN_MMA_AVAILABLE
    constexpr int DKQ = DKQ_QWEN38;
    constexpr int DV = DKQ_QWEN38;
    constexpr int ncols1 = ncols1_qwen38;
    constexpr int ncols2 = ncols2_qwen38;
    constexpr int nwarps = nthreads / warp_size;

    const int jt = gridDim.x - 1 - blockIdx.x;
    const int z_KV = blockIdx.y;
    const int gqa_ratio = nH / nKV;
    const int zt_gqa = 0;
    const int zt_Q = z_KV*gqa_ratio;

    const float2 * Q_f2 = (const float2 *) (Q + (size_t) zt_Q*DKQ);
    const half2 * K_h2 = K16 + (size_t) z_KV*(DKQ/2);
    const half2 * V_h2 = V16 + (size_t) z_KV*(DV/2);
    float2 * dstk = ((float2 *) dst) + (size_t) zt_Q*(DV/2);

    const int stride_Q1 = nH*DKQ/2;
    const int stride_Q2 = DKQ/2;
    const int stride_K = nKV*DKQ/2;
    const int stride_V = nKV*DV/2;

    const int last_row = min((jt + 1)*ncols1, tokens);
    const int kb0_stop = (prefix + last_row + nbatch_fa - 1) / nbatch_fa;

    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps>
        (Q_f2, K_h2, V_h2, dstk, scale, tokens, nH, gqa_ratio,
         stride_Q1, stride_Q2, stride_K, stride_V, jt, zt_gqa, 0, kb0_stop, prefix + jt*ncols1);
#else
    (void) Q; (void) K16; (void) V16; (void) dst; (void) tokens; (void) prefix; (void) nH; (void) nKV; (void) scale;
    __trap();
#endif // FAK_FATTN_MMA_AVAILABLE
}

// Stages f32 K/V rows [0, positions) as f16 and zero-fills rows [positions, padded).
static __global__ void k_qwen38_fattn_stage_kv_f16(
        const float2 * __restrict__ K, const float2 * __restrict__ V,
        half2 * __restrict__ K16, half2 * __restrict__ V16, const size_t live2, const size_t total2) {
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < total2; i += (size_t) gridDim.x*blockDim.x) {
        if (i < live2) {
            const float2 k = K[i], v = V[i];
            K16[i] = make_half2(k.x, k.y);
            V16[i] = make_half2(v.x, v.y);
        } else {
            K16[i] = make_half2(0.0f, 0.0f);
            V16[i] = make_half2(0.0f, 0.0f);
        }
    }
}

} // namespace fak_fattn_mma

// Grow-only f16 K/V staging buffer for the mma prompt-attention path, sized once per
// largest prompt. Growth is refused while a graph capture is open (cudaMalloc is illegal
// there), and the caller then keeps the SIMT kernel.
static half2 *g_qwen38_fattn_kv16 = nullptr;
static size_t g_qwen38_fattn_kv16_bytes = 0;
// Successful mma launches, read by the device parity witness to prove the tensor-core path
// (not the SIMT fallback) produced the output it checks.
static unsigned long long g_qwen38_fattn_mma_launches = 0;

extern "C" unsigned long long fak_qwen38_fattn_mma_launch_count(void) {
    return g_qwen38_fattn_mma_launches;
}

// Returns 0 on launch, a positive 52000+ code on a CUDA error, or -2 when the mma path is
// unavailable for this call (device below sm_80, or staging growth during capture) so the
// caller can fall back to the SIMT kernel before any effect.
static int fak_qwen38_fattn_mma_hd256_launch(
        const float *dQ, const float *dK, const float *dV, float *dOut,
        int tokens, int prefix, int nH, int nKV, float scale) {
    int device = 0, major = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
        cudaGetLastError();
        return -2;
    }
    if (major < 8) return -2;
    const int nbatch_fa = fak_fattn_mma::nbatch_fa;
    const int positions = prefix + tokens;
    const size_t padded = ((size_t) positions + nbatch_fa - 1) / nbatch_fa * nbatch_fa;
    const size_t kvWidth = (size_t) nKV * fak_fattn_mma::DKQ_QWEN38;
    const size_t total2 = padded * kvWidth / 2;
    const size_t need = 2 * total2 * sizeof(half2);
    if (need > g_qwen38_fattn_kv16_bytes) {
        if (g_capture_open) return -2;
        if (g_qwen38_fattn_kv16) {
            fak_alloc_untrack(g_qwen38_fattn_kv16);
            cudaFree(g_qwen38_fattn_kv16);
        }
        g_qwen38_fattn_kv16 = nullptr;
        g_qwen38_fattn_kv16_bytes = 0;
        cudaError_t grown = fcuda_alloc_status(cudaMalloc(&g_qwen38_fattn_kv16, need));
        if (grown != cudaSuccess) {
            g_qwen38_fattn_kv16 = nullptr;
            cudaGetLastError();
            return -2;
        }
        fak_alloc_track(g_qwen38_fattn_kv16, FAK_ALLOC_ATTENTION_SCRATCH, FAK_ALLOC_DEVICE_LOCAL,
                        FAK_ALLOC_SITE("qwen38-fattn-kv16"), need);
        g_qwen38_fattn_kv16_bytes = need;
    }
    half2 *K16 = g_qwen38_fattn_kv16, *V16 = g_qwen38_fattn_kv16 + total2;
    const size_t live2 = (size_t) positions * kvWidth / 2;
    const int stageThreads = 256;
    const size_t stageBlocks = (total2 + stageThreads - 1) / stageThreads;
    FCUDA_TS("k_qwen38_fattn_stage_kv_f16", (unsigned) (stageBlocks < 65535 ? stageBlocks : 65535), fak_fattn_mma::k_qwen38_fattn_stage_kv_f16<<<(unsigned) (stageBlocks < 65535 ? stageBlocks : 65535), stageThreads, 0, g_stream>>>(
        (const float2 *) dK, (const float2 *) dV, K16, V16, live2, total2));
    cudaError_t staged = cudaGetLastError();
    if (staged != cudaSuccess) return 52000 + (int) staged;

    const int ncols1 = fak_fattn_mma::ncols1_qwen38;
    const int stride_tile = fak_fattn_mma::DKQ_QWEN38/2 + 4;
    // max(nbytes_shared_combine, nbytes_shared_Q, nbytes_shared_KV_2stage) for Q_in_reg; no mask tile.
    const size_t shmem_Q = (size_t) ncols1 * fak_fattn_mma::ncols2_qwen38 * stride_tile * sizeof(half2);
    const size_t shmem_KV = (size_t) nbatch_fa * (fak_fattn_mma::nbatch_K2 + 4 + fak_fattn_mma::nbatch_V2 + 4) * sizeof(half2);
    const size_t shmem_combine = (size_t) (fak_fattn_mma::nthreads / fak_fattn_mma::warp_size) * 16 * (fak_fattn_mma::nbatch_combine + 4) * sizeof(half2);
    size_t shmem = shmem_Q > shmem_KV ? shmem_Q : shmem_KV;
    if (shmem_combine > shmem) shmem = shmem_combine;
    dim3 grid((unsigned) ((tokens + ncols1 - 1) / ncols1), (unsigned) nKV, 1);
    FCUDA_TS("k_qwen38_fattn_mma_hd256", grid, fak_fattn_mma::k_qwen38_fattn_mma_hd256<<<grid, dim3(fak_fattn_mma::warp_size, fak_fattn_mma::nthreads / fak_fattn_mma::warp_size, 1), shmem, g_stream>>>(
        dQ, K16, V16, dOut, tokens, prefix, nH, nKV, scale));
    cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return 52000 + (int) launch;
    g_qwen38_fattn_mma_launches++;
    return 0;
}
