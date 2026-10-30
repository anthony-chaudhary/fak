// Included by cuda_kernels.cu after cuda/q3k_mmvq.cuh, not a separate translation unit.
//
// DeepSeek-V4.1 routed experts on CUDA: the id GEMV over every pick of a layer,
// the fused gate/up GEMV with the clamped SwiGLU in its epilogue, and the down
// GEMV with the checked BF16 publication and the routed sum in its epilogue.
//
// Ported from llama.cpp ggml/src/ggml-cuda at 571d0d540df04f25298d0e159e520d9fc62ed121
// (MIT, Copyright (c) 2023-2026 The ggml authors; see THIRD_PARTY_NOTICES.md,
// "llama.cpp MUL_MAT_ID MMVQ routed-expert GEMV"):
//   * mmvq.cu:480 mul_mat_vec_q with ids (channel_x = ids[channel_dst], grid Y is
//     the selected expert) and its fused gate + GLU epilogue (fusion.gate,
//     tmp_gate, tmp_shared_gate), at the NVIDIA GENERIC table, ncols_dst = 1
//     (4 warps, one row per CUDA block, so the clamp of the shared rows path is
//     not needed);
//   * vecdotq.cuh:364 vec_dot_q2_K_q8_1_impl_mmvq and :819 vec_dot_q2_K_q8_1,
//     ggml-common.h block_q2_K. Q3_K reuses fak_q3k_mmvq::vec_dot_q3_K_q8_1 and
//     the q8_1 block, quantize_q8_1 and warp_reduce_sum of cuda/q4k_mmvq.cuh
//     (ported from the same upstream), not a second copy.
// The clamped SwiGLU is antirez/ds4 ds4_cuda.cu:6188 swiglu_kernel at
// 80ebbc396aee40eedc1d829222f3362d10fa4c6c (MIT, The ds4.c authors): g = min(g, c),
// u = clamp(u, +-c), silu(g) * u * weight. Its lane is routed_swiglu_lane in
// cuda_v41_routed_swiglu.cu (shared, so both CUDA routed paths publish the same
// bits), which adds the V4.1 BF16 publication and fault stages.
//
// Changed for Fak:
//   * ids is the V41RoutedIDWords image: [0,P) slot in the bound slot table,
//     [P,2P) route weight bits, [2P,3P) routed-sum order. The slot table is up
//     to eight separate ring-page pointers passed by value, as llama_moe_cache
//     remaps ids through its per-layer slot map (llama.cpp PR #29887), instead
//     of one 3-D expert bank with stride_channel_x.
//   * The GLU epilogue is the V4.1 checked clamp-SwiGLU (routed_swiglu_lane),
//     not ggml_cuda_op_silu_single; faults go to the pick's own 16-byte record
//     (magic, count, lowest activation lane, lowest down-operand lane).
//   * The down kernel loops the picks in ids sum order inside one CUDA block
//     per row and folds the checked BF16 publication (record: magic, site,
//     lowest lane, count) and the ordered F32 sum into the epilogue, as
//     v41_routed_down_sum.comp does on Vulkan.
//   * Activations are quantized to q8_1 into a transient scratch per call;
//     no bias, NVFP4 scale, sample broadcast or ggml_cuda_pdl_*.

namespace fak_v41_mmid {

using fak_q4k_mmvq::block_q8_1;
using fak_q4k_mmvq::warp_reduce_sum;

constexpr int warp_size = 32;
constexpr int QK_K = 256;
constexpr int QK8_1 = 32;
constexpr int QI8_1 = QK8_1 / 4;
constexpr int QR2_K = 4;
constexpr int QI2_K = QK_K / (4 * QR2_K);
constexpr int VDR_Q2_K_Q8_1_MMVQ = 1;
constexpr int max_slots = 8;
// MMVQ_PARAMETERS_GENERIC at ncols_dst = 1.
constexpr int nwarps = 4;
constexpr int rows_per_cuda_block = 1;

enum { type_q2_K = 0, type_q3_K = 1 };

struct slot_table {
    const void *w[max_slots];
};

// ggml-common.h block_q2_K.
struct block_q2_K {
    uint8_t scales[QK_K / 16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K / 4];      // quants
    half2   dm;                // super-block scale for quantized scales / mins
};
static_assert(sizeof(block_q2_K) == 2 * sizeof(half) + QK_K / 16 + QK_K / 4, "wrong q2_K block size/padding");

// vecdotq.cuh vec_dot_q2_K_q8_1_impl_mmvq (contiguous v/x values).
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int &v, const int *__restrict__ u, const uint8_t *__restrict__ scales,
    const half2 &dm2, const float *__restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2 * i];

        const int vi = (v >> (2 * i)) & 0x03030303;

        sumf_d += d8[i] * (__dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m << 8;
        m |= m << 16;
        sumf_m += d8[i] * __dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x * sumf_d - dm2f.y * sumf_m;
}

// vecdotq.cuh vec_dot_q2_K_q8_1.
static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1, const int &kbx, const int &iqs) {

    const block_q2_K *bq2_K = (const block_q2_K *)vbq + kbx;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

    const uint8_t *scales = bq2_K->scales + scale_offset;

    const int v = fak_q3k_mmvq::get_int_b4(bq2_K->qs, iqs);
    int u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        u[i] = fak_q3k_mmvq::get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

// mmvq.cu get_vec_dot_q_cuda / ggml_cuda_type_traits for the two served types.
// Both have qi = 16 and vdr = 1, so one block geometry serves both.
template <int type>
static __device__ __forceinline__ float vec_dot(const void *vx, const block_q8_1 *y, const int &kbx, const int &iqs) {
    if constexpr (type == type_q2_K) {
        return vec_dot_q2_K_q8_1(vx, y, kbx, iqs);
    } else {
        return fak_q3k_mmvq::vec_dot_q3_K_q8_1(vx, y, kbx, iqs);
    }
}
constexpr int qi = 16;
constexpr int vdr = 1;
static_assert(QI2_K == qi && fak_q3k_mmvq::QI3_K == qi, "both routed types take qi 16");
static_assert(VDR_Q2_K_Q8_1_MMVQ == vdr && fak_q3k_mmvq::VDR_Q3_K_Q8_1_MMVQ == vdr, "both routed types take vdr 1");

static __device__ __forceinline__ const void *slot_weight(const slot_table &t, unsigned slot) {
    return t.w[slot < (unsigned)max_slots ? slot : 0];
}

// mmvq.cu mul_mat_vec_q at ncols_dst = 1 with ids. Grid (nrows_x, picks).
// Without fusion dst[pick*nrows_x + row] = W[slot(pick)] . y_pick. With fusion
// vx is the up bank, vgate the gate bank, and dst is the checked SwiGLU mid.
template <int type, bool has_fusion>
__launch_bounds__(nwarps *warp_size, 1)
static __global__ void mul_mat_vec_q_id(
        const slot_table vx_slots, const slot_table vgate_slots, const block_q8_1 *__restrict__ vy,
        const unsigned *__restrict__ ids, float *__restrict__ dst, unsigned *__restrict__ status,
        const int ncols_x, const int nrows_x, const int stride_channel_y, const int picks, const float limit) {
    constexpr int qk = QK_K;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block * blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const int stride_row_x = blocks_per_row_x;

    const unsigned channel_dst = blockIdx.y;
    const unsigned channel_x = ids[channel_dst];
    const unsigned channel_y = channel_dst;

    const void *vx = slot_weight(vx_slots, channel_x);
    [[maybe_unused]] const void *vgate = has_fusion ? slot_weight(vgate_slots, channel_x) : nullptr;

    if constexpr (has_fusion) {
        if (blockIdx.x == 0 && tid == 0) {
            status[4 * channel_dst + 0] = 0x52535731u;
            status[4 * channel_dst + 1] = (unsigned)nrows_x;
        }
    }

    float tmp = 0.0f;
    [[maybe_unused]] float tmp_gate = 0.0f;

    const block_q8_1 *y = vy + channel_y * stride_channel_y;
    const int kbx_offset = row0 * stride_row_x;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi / vdr));

        tmp += vec_dot<type>(vx, &y[kby], kbx_offset + kbx, kqs);
        if constexpr (has_fusion) {
            tmp_gate += vec_dot<type>(vgate, &y[kby], kbx_offset + kbx, kqs);
        }
    }

    __shared__ float tmp_shared[nwarps - 1][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[has_fusion ? nwarps - 1 : 1][warp_size];

    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
        if constexpr (has_fusion) {
            tmp_shared_gate[threadIdx.y - 1][threadIdx.x] = tmp_gate;
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int l = 0; l < nwarps - 1; ++l) {
        tmp += tmp_shared[l][threadIdx.x];
        if constexpr (has_fusion) {
            tmp_gate += tmp_shared_gate[l][threadIdx.x];
        }
    }
    tmp = warp_reduce_sum<warp_size>(tmp);
    if constexpr (has_fusion) {
        tmp_gate = warp_reduce_sum<warp_size>(tmp_gate);
    }

    if (threadIdx.x == 0) {
        const size_t out = (size_t)channel_dst * nrows_x + row0;
        if constexpr (has_fusion) {
            const float weight = __uint_as_float(ids[picks + channel_dst]);
            unsigned bits;
            const unsigned stage = routed_swiglu_lane(tmp_gate, tmp, limit, weight, &bits);
            if (stage) atomicMin(status + 4 * channel_dst + 1 + stage, (unsigned)row0);
            dst[out] = __uint_as_float(bits);
        } else {
            dst[out] = tmp;
        }
    }
}

// The down projection of every pick, its checked BF16 publication and the
// routed sum. Grid (nrows_x). Each CUDA block owns one output row and visits
// the picks in ids sum order (ids[2P + j]), so the sum is one sequential F32 add
// per pick over a +0 accumulator. y is the q8_1 image of the [P, ncols_x] mid.
template <int type>
__launch_bounds__(nwarps *warp_size, 1)
static __global__ void mul_mat_vec_q_down_sum(
        const slot_table vx_slots, const block_q8_1 *__restrict__ vy, const unsigned *__restrict__ ids,
        float *__restrict__ dst, unsigned *__restrict__ status, const int ncols_x, const int nrows_x,
        const int picks, const unsigned site) {
    constexpr int qk = QK_K;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int row0 = blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const int stride_channel_y = ncols_x / QK8_1;

    if (row0 == 0 && tid == 0) {
        for (int p = 0; p < picks; ++p) {
            status[4 * p + 0] = 0x42463143u;
            status[4 * p + 1] = site;
            status[4 * p + 3] = (unsigned)nrows_x;
        }
    }

    __shared__ float tmp_shared[nwarps - 1][warp_size];
    float acc = 0.0f;
    for (int j = 0; j < picks; ++j) {
        const unsigned p = ids[2 * picks + j];
        const void *vx = slot_weight(vx_slots, ids[p]);
        const block_q8_1 *y = vy + p * stride_channel_y;
        float tmp = 0.0f;
        for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk / QK8_1);
            const int kqs = vdr * (tid % (qi / vdr));
            tmp += vec_dot<type>(vx, &y[kby], row0 * blocks_per_row_x + kbx, kqs);
        }
        if (threadIdx.y > 0) tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
        __syncthreads();
        if (threadIdx.y == 0) {
#pragma unroll
            for (int l = 0; l < nwarps - 1; ++l) tmp += tmp_shared[l][threadIdx.x];
            tmp = warp_reduce_sum<warp_size>(tmp);
            if (threadIdx.x == 0) {
                const unsigned bits = __float_as_uint(tmp);
                const unsigned rounded = routed_swiglu_bf16(bits);
                if (routed_swiglu_nonfinite(bits) || routed_swiglu_nonfinite(rounded))
                    atomicMin(status + 4 * p + 2, (unsigned)row0);
                acc = __fadd_rn(acc, __uint_as_float(rounded));
            }
        }
        __syncthreads();
    }
    if (tid == 0) dst[row0] = acc;
}

// v41_routed_swiglu_id.comp over every pick: grid (x, picks), weight from ids.
static __global__ void routed_swiglu_id(const float *__restrict__ gate, const float *__restrict__ up,
        float *__restrict__ out, const unsigned *__restrict__ ids, unsigned *__restrict__ status,
        const int count, const int picks, const float limit) {
    const unsigned pick = blockIdx.y;
    const unsigned first = blockIdx.x * blockDim.x + threadIdx.x;
    if (first == 0) {
        status[4 * pick + 0] = 0x52535731u;
        status[4 * pick + 1] = (unsigned)count;
    }
    const float weight = __uint_as_float(ids[picks + pick]);
    const size_t base = (size_t)pick * count;
    for (unsigned i = first; i < (unsigned)count; i += gridDim.x * blockDim.x) {
        unsigned bits;
        const unsigned stage = routed_swiglu_lane(gate[base + i], up[base + i], limit, weight, &bits);
        if (stage) atomicMin(status + 4 * pick + 1 + stage, i);
        out[base + i] = __uint_as_float(bits);
    }
}

// v41_routed_publish_sum.comp: checked BF16 publication of every pick's down
// row (one record per pick) and their sum in ids order.
static __global__ void routed_publish_down_sum(const float *__restrict__ down, float *__restrict__ sum,
        const unsigned *__restrict__ ids, unsigned *__restrict__ status, const int count, const int picks,
        const unsigned site) {
    const unsigned first = blockIdx.x * blockDim.x + threadIdx.x;
    if (first == 0) {
        for (int p = 0; p < picks; ++p) {
            status[4 * p + 0] = 0x42463143u;
            status[4 * p + 1] = site;
            status[4 * p + 3] = (unsigned)count;
        }
    }
    for (unsigned i = first; i < (unsigned)count; i += gridDim.x * blockDim.x) {
        for (int p = 0; p < picks; ++p) {
            const unsigned bits = __float_as_uint(down[(size_t)p * count + i]);
            const unsigned rounded = routed_swiglu_bf16(bits);
            if (routed_swiglu_nonfinite(bits) || routed_swiglu_nonfinite(rounded)) atomicMin(status + 4 * p + 2, i);
        }
        float acc = 0.0f;
        for (int j = 0; j < picks; ++j) {
            const unsigned p = ids[2 * picks + j];
            acc = __fadd_rn(acc, __uint_as_float(routed_swiglu_bf16(__float_as_uint(down[(size_t)p * count + i]))));
        }
        sum[i] = acc;
    }
}

} // namespace fak_v41_mmid

static bool g_v41_mmid_uncertain = false;
static unsigned long long g_v41_mmid_launches = 0;

extern "C" unsigned long long fak_v41_mmid_launch_count(void) { return g_v41_mmid_launches; }

extern "C" int fak_v41_mmid_state(void) {
    if (!g_stream || g_capture_open || g_v41_mmid_uncertain) return -3;
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) return 10000 + (int)error;
    int device = -1;
    error = cudaGetDevice(&device);
    if (error != cudaSuccess) return 10000 + (int)error;
    if (device != 0) return -3;
    cudaStreamCaptureStatus capture;
    error = cudaStreamIsCapturing(g_stream, &capture);
    if (error != cudaSuccess) return 10000 + (int)error;
    return capture == cudaStreamCaptureStatusNone ? 0 : -3;
}

// Live, pooled, device-0 storage of at least bytes.
static int v41_mmid_owned(const void *p, uint64_t bytes) {
    if (!p || bytes == 0) return -2;
    auto found = g_live.find(const_cast<void *>(p));
    if (found == g_live.end() || found->second < (size_t)bytes) return -2;
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, p);
    if (error != cudaSuccess) return 11000 + (int)error;
    return attributes.type == cudaMemoryTypeDevice && attributes.device == 0 ? 0 : -2;
}

static int v41_mmid_slots(const void *const *src, int nslots, fak_v41_mmid::slot_table *dst) {
    if (!src || nslots <= 0 || nslots > fak_v41_mmid::max_slots) return 2;
    for (int i = 0; i < fak_v41_mmid::max_slots; ++i) dst->w[i] = src[i < nslots ? i : 0];
    for (int i = 0; i < nslots; ++i)
        if (!src[i]) return 2;
    return 0;
}

// An enqueue failure leaves earlier work unknown: drain, and close the
// candidate if even the drain fails.
static int v41_mmid_drain(int status) {
    cudaError_t drained = cudaStreamSynchronize(g_stream);
    if (drained != cudaSuccess) {
        g_v41_mmid_uncertain = true;
        return 50000 + (int)drained;
    }
    return status;
}

static int v41_mmid_launched(void) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return v41_mmid_drain(30000 + (int)error);
    ++g_v41_mmid_launches;
    return 0;
}

// Quantizes rows x [rows, in] into a fresh q8_1 scratch the caller frees after
// enqueueing its consumer (the pool is stream ordered).
static int v41_mmid_quantize(const float *x, int in, int rows, fak_q4k_mmvq::block_q8_1 **out) {
    const size_t need = (size_t)rows * (in / fak_v41_mmid::QK8_1) * sizeof(fak_q4k_mmvq::block_q8_1);
    auto *y = (fak_q4k_mmvq::block_q8_1 *)scratch_malloc(need);
    if (!y) return FCUDA_SCRATCH_ALLOC_FAILED;
    dim3 qgrid((in + 255) / 256, rows);
    FCUDA_TS("quantize_q8_1", qgrid, fak_q4k_mmvq::quantize_q8_1<<<qgrid, 256, 0, g_stream>>>(x, y, in));
    int launched = v41_mmid_launched();
    if (launched != 0) {
        fcuda_free(y);
        return launched;
    }
    *out = y;
    return 0;
}

static int v41_mmid_geometry(int dtype, int rows, int in, int picks) {
    if ((dtype != fak_v41_mmid::type_q2_K && dtype != fak_v41_mmid::type_q3_K) || rows <= 0 || in <= 0 ||
        in % fak_v41_mmid::QK_K != 0 || picks <= 0 || picks > fak_v41_mmid::max_slots || rows > 65535 * 1024)
        return 2;
    return 0;
}

// dst[P*rows] = W[slot(p)] . x_p; x_stride 0 shares one x row.
extern "C" int fak_v41_mmid_gemv(int dtype, const void *const *slots, int nslots, const float *x, int x_stride,
                                 const unsigned *ids, float *dst, int rows, int in, int picks) {
    using namespace fak_v41_mmid;
    if (v41_mmid_geometry(dtype, rows, in, picks) || (x_stride != 0 && x_stride != in)) return 2;
    slot_table table;
    if (v41_mmid_slots(slots, nslots, &table)) return 2;
    int state = fak_v41_mmid_state();
    if (state != 0) return state;
    int owned = v41_mmid_owned(x, (uint64_t)(x_stride ? picks : 1) * in * 4u);
    if (owned == 0) owned = v41_mmid_owned(ids, (uint64_t)picks * 12u);
    if (owned == 0) owned = v41_mmid_owned(dst, (uint64_t)picks * rows * 4u);
    if (owned != 0) return owned;
    block_q8_1 *y = nullptr;
    int q = v41_mmid_quantize(x, in, x_stride ? picks : 1, &y);
    if (q != 0) return q;
    const int stride_channel_y = x_stride ? in / QK8_1 : 0;
    const dim3 grid(rows, picks), block(warp_size, nwarps);
    if (dtype == type_q2_K) {
        FCUDA_TS("mul_mat_vec_q_id", grid, (mul_mat_vec_q_id<type_q2_K, false><<<grid, block, 0, g_stream>>>(
            table, table, y, ids, dst, nullptr, in, rows, stride_channel_y, picks, 0.0f)));
    } else {
        FCUDA_TS("mul_mat_vec_q_id", grid, (mul_mat_vec_q_id<type_q3_K, false><<<grid, block, 0, g_stream>>>(
            table, table, y, ids, dst, nullptr, in, rows, stride_channel_y, picks, 0.0f)));
    }
    int launched = v41_mmid_launched();
    fcuda_free(y);
    return launched;
}

// mid[P*rows] = checked SwiGLU(gate[slot(p)] . x_p, up[slot(p)] . x_p) * w_p;
// status holds P routed-SwiGLU records.
extern "C" int fak_v41_mmid_gate_up_swiglu(int dtype, const void *const *gates, const void *const *ups, int nslots,
                                           const float *x, int x_stride, const unsigned *ids, float *mid,
                                           unsigned *status, int rows, int in, int picks, float limit) {
    using namespace fak_v41_mmid;
    if (v41_mmid_geometry(dtype, rows, in, picks) || (x_stride != 0 && x_stride != in) || !isfinite(limit)) return 2;
    slot_table gate_table, up_table;
    if (v41_mmid_slots(gates, nslots, &gate_table) || v41_mmid_slots(ups, nslots, &up_table)) return 2;
    int state = fak_v41_mmid_state();
    if (state != 0) return state;
    int owned = v41_mmid_owned(x, (uint64_t)(x_stride ? picks : 1) * in * 4u);
    if (owned == 0) owned = v41_mmid_owned(ids, (uint64_t)picks * 12u);
    if (owned == 0) owned = v41_mmid_owned(mid, (uint64_t)picks * rows * 4u);
    if (owned == 0) owned = v41_mmid_owned(status, (uint64_t)picks * 16u);
    if (owned != 0) return owned;
    cudaError_t e = cudaSuccess;
    FCUDA_TS("fill", dim3(0, 0), e = cudaMemsetAsync(status, 0xff, (size_t)picks * 16u, g_stream));
    if (e != cudaSuccess) return v41_mmid_drain(20000 + (int)e);
    block_q8_1 *y = nullptr;
    int q = v41_mmid_quantize(x, in, x_stride ? picks : 1, &y);
    if (q != 0) return q;
    const int stride_channel_y = x_stride ? in / QK8_1 : 0;
    const dim3 grid(rows, picks), block(warp_size, nwarps);
    if (dtype == type_q2_K) {
        FCUDA_TS("mul_mat_vec_q_id", grid, (mul_mat_vec_q_id<type_q2_K, true><<<grid, block, 0, g_stream>>>(
            up_table, gate_table, y, ids, mid, status, in, rows, stride_channel_y, picks, limit)));
    } else {
        FCUDA_TS("mul_mat_vec_q_id", grid, (mul_mat_vec_q_id<type_q3_K, true><<<grid, block, 0, g_stream>>>(
            up_table, gate_table, y, ids, mid, status, in, rows, stride_channel_y, picks, limit)));
    }
    int launched = v41_mmid_launched();
    fcuda_free(y);
    return launched;
}

// sum[rows] = sum over j of bf16(down[slot(p_j)] . mid_{p_j}), p_j = ids[2P+j];
// status holds P checked-BF16 records.
extern "C" int fak_v41_mmid_down_sum(int dtype, const void *const *downs, int nslots, const float *mid,
                                     const unsigned *ids, float *sum, unsigned *status, int rows, int in,
                                     int picks, unsigned site) {
    using namespace fak_v41_mmid;
    if (v41_mmid_geometry(dtype, rows, in, picks)) return 2;
    slot_table table;
    if (v41_mmid_slots(downs, nslots, &table)) return 2;
    int state = fak_v41_mmid_state();
    if (state != 0) return state;
    int owned = v41_mmid_owned(mid, (uint64_t)picks * in * 4u);
    if (owned == 0) owned = v41_mmid_owned(ids, (uint64_t)picks * 12u);
    if (owned == 0) owned = v41_mmid_owned(sum, (uint64_t)rows * 4u);
    if (owned == 0) owned = v41_mmid_owned(status, (uint64_t)picks * 16u);
    if (owned != 0) return owned;
    cudaError_t e = cudaSuccess;
    FCUDA_TS("fill", dim3(0, 0), e = cudaMemsetAsync(status, 0xff, (size_t)picks * 16u, g_stream));
    if (e != cudaSuccess) return v41_mmid_drain(20000 + (int)e);
    block_q8_1 *y = nullptr;
    int q = v41_mmid_quantize(mid, in, picks, &y);
    if (q != 0) return q;
    const dim3 grid(rows), block(warp_size, nwarps);
    if (dtype == type_q2_K) {
        FCUDA_TS("mul_mat_vec_q_down_sum", grid, (mul_mat_vec_q_down_sum<type_q2_K><<<grid, block, 0, g_stream>>>(
            table, y, ids, sum, status, in, rows, picks, site)));
    } else {
        FCUDA_TS("mul_mat_vec_q_down_sum", grid, (mul_mat_vec_q_down_sum<type_q3_K><<<grid, block, 0, g_stream>>>(
            table, y, ids, sum, status, in, rows, picks, site)));
    }
    int launched = v41_mmid_launched();
    fcuda_free(y);
    return launched;
}

extern "C" int fak_v41_mmid_swiglu_id(const float *gate, const float *up, float *out, const unsigned *ids,
                                      unsigned *status, int count, int picks, float limit) {
    if (count <= 0 || picks <= 0 || picks > fak_v41_mmid::max_slots || !isfinite(limit)) return 2;
    int state = fak_v41_mmid_state();
    if (state != 0) return state;
    const uint64_t bytes = (uint64_t)picks * count * 4u;
    int owned = v41_mmid_owned(gate, bytes);
    if (owned == 0) owned = v41_mmid_owned(up, bytes);
    if (owned == 0) owned = v41_mmid_owned(out, bytes);
    if (owned == 0) owned = v41_mmid_owned(ids, (uint64_t)picks * 12u);
    if (owned == 0) owned = v41_mmid_owned(status, (uint64_t)picks * 16u);
    if (owned != 0) return owned;
    cudaError_t e = cudaSuccess;
    FCUDA_TS("fill", dim3(0, 0), e = cudaMemsetAsync(status, 0xff, (size_t)picks * 16u, g_stream));
    if (e != cudaSuccess) return v41_mmid_drain(20000 + (int)e);
    unsigned blocks = ((unsigned)count + 255u) / 256u;
    if (blocks > 65535u) blocks = 65535u;
    const dim3 grid(blocks, picks);
    FCUDA_TS("routed_swiglu_id", grid, (fak_v41_mmid::routed_swiglu_id<<<grid, 256, 0, g_stream>>>(
        gate, up, out, ids, status, count, picks, limit)));
    return v41_mmid_launched();
}

extern "C" int fak_v41_mmid_publish_sum(const float *down, float *sum, const unsigned *ids, unsigned *status,
                                        int count, int picks, unsigned site) {
    if (count <= 0 || picks <= 0 || picks > fak_v41_mmid::max_slots) return 2;
    int state = fak_v41_mmid_state();
    if (state != 0) return state;
    int owned = v41_mmid_owned(down, (uint64_t)picks * count * 4u);
    if (owned == 0) owned = v41_mmid_owned(sum, (uint64_t)count * 4u);
    if (owned == 0) owned = v41_mmid_owned(ids, (uint64_t)picks * 12u);
    if (owned == 0) owned = v41_mmid_owned(status, (uint64_t)picks * 16u);
    if (owned != 0) return owned;
    cudaError_t e = cudaSuccess;
    FCUDA_TS("fill", dim3(0, 0), e = cudaMemsetAsync(status, 0xff, (size_t)picks * 16u, g_stream));
    if (e != cudaSuccess) return v41_mmid_drain(20000 + (int)e);
    unsigned blocks = ((unsigned)count + 255u) / 256u;
    if (blocks > 65535u) blocks = 65535u;
    FCUDA_TS("routed_publish_down_sum", blocks, (fak_v41_mmid::routed_publish_down_sum<<<blocks, 256, 0, g_stream>>>(
        down, sum, ids, status, count, picks, site)));
    return v41_mmid_launched();
}

// Checked completion, then one copy of n status bytes. A failed drain closes
// the candidate: the deferred kernels may still own their outputs.
extern "C" int fak_v41_mmid_read(void *host, const void *dev, size_t n) {
    if (!host || !dev || n == 0) return 2;
    cudaError_t e = cudaStreamSynchronize(g_stream);
    if (e != cudaSuccess) {
        g_v41_mmid_uncertain = true;
        return 40000 + (int)e;
    }
    int copied = fcuda_d2h_checked(host, dev, n);
    return copied == 0 ? 0 : 59000 + copied;
}
