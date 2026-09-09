//go:build darwin && arm64 && cgo

// qwen35_decode.m — Wide-M speculative verification kernels on Metal 4.
//
// Implements:
//   1. Multi-token recurrent GDN state step across M speculative tokens (M=2..4)
//      with bit-exact numeric parity against serial single-token execution.
//   2. Tail-causal wide-M SDPA verification tile kernel on Metal 4.
//   3. Fused single-command-buffer verification dispatch chaining:
//      Wide-M Q4_K GEMM -> Batched Recurrent GDN -> Tail-causal SDPA
//      with zero host-GPU round-trip synchronizations.

#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern id<MTLDevice>       gDev;
extern id<MTLCommandQueue> gQueue;
int mg_init(void);

// Forward declarations of external Q4_K helpers in q4k.m
int mg_q4k_gemv_wide_m_encode(void* cb_ptr, int wid, void* x_buf_ptr, void* y_buf_ptr, int m);

typedef struct SDPANAXConstants {
    uint32_t gqa_factor;
    uint32_t draft_len;
    uint32_t M;
    uint32_t head_dim;
    uint32_t prefix_len;
    uint32_t total_kv;
    float scale;
    uint32_t tile_n;
    uint32_t order;
    uint32_t has_tree_mask;
    uint32_t tree_mask[32];
} SDPANAXConstants;

#define QWEN35_MAX_STATES 256

typedef struct {
    id<MTLBuffer> conv;      // [(convKernel - 1) * convDim * sizeof(float)]
    id<MTLBuffer> recurrent; // [nV * kHd * vHd * sizeof(float)]
    int nK;
    int nV;
    int kHd;
    int vHd;
    int convKernel;
    int inUse;
} Qwen35DecodeState;

static Qwen35DecodeState gQwen35States[QWEN35_MAX_STATES];
static int gQwen35StateCount = 0;

static id<MTLComputePipelineState> psoGDNConvWideM = nil;
static id<MTLComputePipelineState> psoGDNQKNormWideM = nil;
static id<MTLComputePipelineState> psoGDNRecurrentWideM = nil;
static id<MTLComputePipelineState> psoSDPATailCausalTile = nil;
static int gQwen35PipelinesReady = 0;

#define NAX_TILE_N_MAX 32
#define NAX_MAX_HEAD_DIM 128

typedef struct SDPANAXThreadgroupStorage {
    float k_tile[NAX_TILE_N_MAX * NAX_MAX_HEAD_DIM];
    float v_transposed[NAX_MAX_HEAD_DIM * NAX_TILE_N_MAX];
} SDPANAXThreadgroupStorage;

static NSString *kQwen35DecodeMSL = @R"MSL(
#include <metal_stdlib>
using namespace metal;

inline float gdn_silu(float x) { return x / (1.0f + exp(-x)); }
inline float gdn_softplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

// ==============================================================================
// Batched Recurrent GDN Step Shaders
// ==============================================================================

// gdn_conv_wide_m: Advances the 1D convolution window across M sequential or tree-structured tokens.
// Each channel runs across token 0..tokens-1 in order, maintaining exact state.
kernel void gdn_conv_wide_m(
    device const float *mixed [[buffer(0)]],
    device const float *convW [[buffer(1)]],
    device float *convState   [[buffer(2)]],
    device float *convOut     [[buffer(3)]],
    constant int& tokens      [[buffer(4)]],
    constant int& convDim     [[buffer(5)]],
    constant int& kernelSize  [[buffer(6)]],
    device const int *parents [[buffer(7)]],
    uint channel [[thread_position_in_grid]]
) {
    if (channel >= (uint)convDim) return;
    float window[4];
    float saved_windows[32][3];
    int kMinus1 = min(kernelSize - 1, 3);
    for (int j = 0; j < kMinus1; ++j) {
        window[j] = convState[(long)j * convDim + channel];
    }
    for (int token = 0; token < tokens; ++token) {
        int p = parents ? parents[token] : (token - 1);
        if (p >= 0 && p < tokens && p < 32) {
            for (int j = 0; j < kMinus1; ++j) {
                window[j] = saved_windows[p][j];
            }
        } else if (p < 0 && token > 0) {
            for (int j = 0; j < kMinus1; ++j) {
                window[j] = convState[(long)j * convDim + channel];
            }
        }
        float acc = 0.0f;
        int wb = (int)channel * kernelSize;
        for (int j = 0; j < kMinus1; ++j) acc += convW[wb + j] * window[j];
        float current = mixed[(long)token * convDim + channel];
        acc += convW[wb + kernelSize - 1] * current;
        convOut[(long)token * convDim + channel] = gdn_silu(acc);
        for (int j = 0; j < kMinus1 - 1; ++j) window[j] = window[j + 1];
        if (kMinus1 > 0) window[kMinus1 - 1] = current;
        if (token < 32) {
            for (int j = 0; j < kMinus1; ++j) {
                saved_windows[token][j] = window[j];
            }
        }
    }
    for (int j = 0; j < kMinus1; ++j) {
        convState[(long)j * convDim + channel] = window[j];
    }
}

// gdn_qk_norm_wide_m: Evaluates L2-normalization for Q and K heads across M tokens.
kernel void gdn_qk_norm_wide_m(
    device const float *convOut [[buffer(0)]],
    device float *qNorm         [[buffer(1)]],
    device float *kNorm         [[buffer(2)]],
    constant int& tokens        [[buffer(3)]],
    constant int& convDim       [[buffer(4)]],
    constant int& nK            [[buffer(5)]],
    constant int& kHd           [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 groupSize [[threads_per_threadgroup]]
) {
    int head = (int)group.x, token = (int)group.y;
    uint lanes = groupSize.x;
    if (head >= nK || token >= tokens) return;
    threadgroup float qss[256];
    threadgroup float kss[256];
    long row = (long)token * convDim;
    float q = lane < (uint)kHd ? convOut[row + (long)head * kHd + lane] : 0.0f;
    float k = lane < (uint)kHd ? convOut[row + (long)nK * kHd + (long)head * kHd + lane] : 0.0f;
    qss[lane] = q * q;
    kss[lane] = k * k;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) { qss[lane] += qss[lane + offset]; kss[lane] += kss[lane + offset]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < (uint)kHd) {
        float qinv = 1.0f / sqrt(qss[0] + 1.0e-6f);
        float kinv = 1.0f / sqrt(kss[0] + 1.0e-6f);
        float scale = 1.0f / sqrt((float)kHd);
        long dst = ((long)token * nK + head) * kHd + lane;
        qNorm[dst] = q * qinv * scale;
        kNorm[dst] = k * kinv;
    }
}

// gdn_recurrent_wide_m: Evaluates recurrent state update across M tokens in registers and tree-branching buffers.
// When parents are provided, branches read their true parent's state and record their own state.
kernel void gdn_recurrent_wide_m(
    device const float *convOut [[buffer(0)]],
    device const float *qNorm   [[buffer(1)]],
    device const float *kNorm   [[buffer(2)]],
    device const float *z       [[buffer(3)]],
    device const float *b       [[buffer(4)]],
    device const float *a       [[buffer(5)]],
    device const float *aLog    [[buffer(6)]],
    device const float *dtBias  [[buffer(7)]],
    device const float *norm    [[buffer(8)]],
    device float *state         [[buffer(9)]],
    device float *core          [[buffer(10)]],
    constant int& tokens        [[buffer(11)]],
    constant int& convDim       [[buffer(12)]],
    constant int& nK            [[buffer(13)]],
    constant int& nV            [[buffer(14)]],
    constant int& kHd           [[buffer(15)]],
    constant int& vHd           [[buffer(16)]],
    constant float& eps         [[buffer(17)]],
    device const int *parents   [[buffer(18)]],
    device float *treeState     [[buffer(19)]],
    constant uint32_t& parentMask [[buffer(20)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]
) {
    if (head >= (uint)nV) return;
    int repeat = nV / nK;
    int keyHead = (int)head / repeat;
    int keyDim = nK * kHd;
    float4 localState[16];
    int nChunks = kHd / 4;
    if (nChunks > 16) nChunks = 16;

    if (lane < (uint)vHd) {
        for (int i = 0; i < nChunks; ++i) {
            localState[i] = float4(
                state[((long)head * kHd + i * 4 + 0) * vHd + lane],
                state[((long)head * kHd + i * 4 + 1) * vHd + lane],
                state[((long)head * kHd + i * 4 + 2) * vHd + lane],
                state[((long)head * kHd + i * 4 + 3) * vHd + lane]
            );
        }
    }
    threadgroup float tg_sq[8];
    uint simd_id = lane / 32;
    uint simd_lane = lane % 32;
    float neg_exp_aLog = -exp(aLog[head]);
    float head_dtBias = dtBias[head];

    int last_token = -1;
    for (int token = 0; token < tokens; ++token) {
        int p = parents ? parents[token] : (token - 1);
        if (lane < (uint)vHd) {
            if (p != last_token) {
                if (p < 0) {
                    for (int i = 0; i < nChunks; ++i) {
                        localState[i] = float4(
                            state[((long)head * kHd + i * 4 + 0) * vHd + lane],
                            state[((long)head * kHd + i * 4 + 1) * vHd + lane],
                            state[((long)head * kHd + i * 4 + 2) * vHd + lane],
                            state[((long)head * kHd + i * 4 + 3) * vHd + lane]
                        );
                    }
                } else if (treeState != nullptr) {
                    long pOffset = (((long)p * nV + (long)head) * kHd) * vHd + lane;
                    for (int i = 0; i < nChunks; ++i) {
                        localState[i] = float4(
                            treeState[pOffset + (long)(i * 4 + 0) * vHd],
                            treeState[pOffset + (long)(i * 4 + 1) * vHd],
                            treeState[pOffset + (long)(i * 4 + 2) * vHd],
                            treeState[pOffset + (long)(i * 4 + 3) * vHd]
                        );
                    }
                }
            }
        }
        last_token = token;

        // Register-level SIMD broadcast of K and Q vector chunks and scalar factors
        device const float4 *qRow4 = (device const float4 *)(qNorm + ((long)token * nK + keyHead) * kHd);
        device const float4 *kRow4 = (device const float4 *)(kNorm + ((long)token * nK + keyHead) * kHd);

        float4 my_k = (simd_lane < (uint)nChunks) ? kRow4[simd_lane] : float4(0.0f);
        float4 my_q = (simd_lane < (uint)nChunks) ? qRow4[simd_lane] : float4(0.0f);

        float beta_0 = (simd_lane == 0) ? (1.0f / (1.0f + exp(-b[(long)token * nV + head]))) : 0.0f;
        float decay_0 = (simd_lane == 0) ? exp(neg_exp_aLog * gdn_softplus(a[(long)token * nV + head] + head_dtBias)) : 0.0f;
        float beta = simd_broadcast(beta_0, 0);
        float decay = simd_broadcast(decay_0, 0);

        float readout = 0.0f;
        if (lane < (uint)vHd) {
            float4 kvmem4 = float4(0.0f);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (i < nChunks) {
                    float4 val = localState[i] * decay;
                    localState[i] = val;
                    float4 k_val = simd_broadcast(my_k, (uint16_t)i);
                    kvmem4 += val * k_val;
                }
            }
            float kvmem = kvmem4.x + kvmem4.y + kvmem4.z + kvmem4.w;
            long valueIndex = (long)token * convDim + 2L * keyDim + (long)head * vHd + lane;
            float delta = (convOut[valueIndex] - kvmem) * beta;
            long tokOffset = (((long)token * nV + (long)head) * kHd) * vHd + lane;
            float4 readout4 = float4(0.0f);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (i < nChunks) {
                    float4 k_val = simd_broadcast(my_k, (uint16_t)i);
                    float4 q_val = simd_broadcast(my_q, (uint16_t)i);
                    float4 val = localState[i] + k_val * delta;
                    localState[i] = val;
                    readout4 += val * q_val;
                    if (treeState != nullptr) {
                        treeState[tokOffset + (long)(i * 4 + 0) * vHd] = val.x;
                        treeState[tokOffset + (long)(i * 4 + 1) * vHd] = val.y;
                        treeState[tokOffset + (long)(i * 4 + 2) * vHd] = val.z;
                        treeState[tokOffset + (long)(i * 4 + 3) * vHd] = val.w;
                    }
                }
            }
            readout = readout4.x + readout4.y + readout4.z + readout4.w;
        }

        float sq = (lane < (uint)vHd) ? readout * readout : 0.0f;
        float simd_sq = simd_sum(sq);
        if (simd_lane == 0) {
            tg_sq[simd_id] = simd_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float total_sq = tg_sq[0] + ((lanes > 32) ? tg_sq[1] : 0.0f);

        if (lane < (uint)vHd) {
            float inv = rsqrt(total_sq / (float)vHd + eps);
            long vd = (long)head * vHd + lane;
            core[(long)token * nV * vHd + vd] = norm[lane] * readout * inv * gdn_silu(z[(long)token * nV * vHd + vd]);
        }
    }
    if (lane < (uint)vHd) {
        for (int i = 0; i < nChunks; ++i) {
            state[((long)head * kHd + i * 4 + 0) * vHd + lane] = localState[i].x;
            state[((long)head * kHd + i * 4 + 1) * vHd + lane] = localState[i].y;
            state[((long)head * kHd + i * 4 + 2) * vHd + lane] = localState[i].z;
            state[((long)head * kHd + i * 4 + 3) * vHd + lane] = localState[i].w;
        }
    }
}

// ==============================================================================
// Tail-Causal Wide-M SDPA Verification Tile Kernel
// ==============================================================================

#define NAX_TILE_N_MAX 32
#define NAX_MAX_HEAD_DIM 128

struct SDPANAXConstantsMSL {
    uint gqa_factor;   // Number of query heads sharing a KV head
    uint draft_len;    // Number of speculative draft tokens (e.g. 2..32)
    uint M;            // Total query rows: M = gqa_factor * draft_len
    uint head_dim;     // Head dimension D (e.g. 64 or 128)
    uint prefix_len;   // Sequence prefix length
    uint total_kv;     // Total KV tokens = prefix_len + draft_len
    float scale;       // Attention scale factor 1.0f / sqrt(head_dim)
    uint tile_n;       // KV tile size Bc (e.g. 32)
    uint order;        // 0: HeadMajor, 1: TokenMajor
    uint has_tree_mask;// 1 if tree_mask is active
    uint tree_mask[32];// bitmask per draft token
};

struct SDPANAXThreadgroupStorageMSL {
    float k_tile[NAX_TILE_N_MAX * NAX_MAX_HEAD_DIM];
    float v_transposed[NAX_MAX_HEAD_DIM * NAX_TILE_N_MAX];
};

inline uint nax_draft_token_index(uint m, constant SDPANAXConstantsMSL& c) {
    if (c.order == 1) {
        return c.gqa_factor > 0 ? (m / c.gqa_factor) : 0;
    }
    return c.draft_len > 0 ? (m % c.draft_len) : 0;
}

inline uint nax_max_causal_key(uint m, constant SDPANAXConstantsMSL& c) {
    return c.prefix_len + nax_draft_token_index(m, c);
}

inline bool nax_can_attend(uint m, uint global_key_pos, constant SDPANAXConstantsMSL& c) {
    if (global_key_pos < c.prefix_len) {
        return true;
    }
    uint t = nax_draft_token_index(m, c);
    uint draft_k = global_key_pos - c.prefix_len;
    if (draft_k > t) return false;
    if (c.has_tree_mask) {
        return (c.tree_mask[t] & (1u << draft_k)) != 0;
    }
    return true;
}

kernel void sdpa_nax_tail_causal_tile(
    device const float* Q [[buffer(0)]],            // [M, HeadDim]
    device const float* K [[buffer(1)]],            // [TotalKV, HeadDim]
    device const float* V [[buffer(2)]],            // [TotalKV, HeadDim]
    device float* Out [[buffer(3)]],                // [M, HeadDim]
    device float* LSE [[buffer(4)]],                // [M]
    constant SDPANAXConstantsMSL& c [[buffer(5)]],
    threadgroup SDPANAXThreadgroupStorageMSL& tg_mem [[threadgroup(0)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    if (c.M == 0 || c.head_dim == 0 || c.total_kv == 0) return;
    uint m = tg_idx;
    if (m >= c.M) return;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    // Preload Q values for row m in registers (eliminating global Q memory reads inside tile/k loop)
    float q_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint q_idx = 0;
    for (uint d = simd_lane; d < c.head_dim; d += 32) {
        q_reg[q_idx++] = Q[m * c.head_dim + d];
    }

    uint max_k_for_row = nax_max_causal_key(m, c);
    uint num_tiles = (c.total_kv + c.tile_n - 1) / c.tile_n;

    for (uint tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint j_start = tile_idx * c.tile_n;
        if (j_start > max_k_for_row) {
            break; // Future tiles are strictly causal and cannot be attended
        }
        uint j_end = min(j_start + c.tile_n, c.total_kv);
        uint tile_len = j_end - j_start;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (c.head_dim == 64) {
            device const float4* K4 = (device const float4*)(K + j_start * 64);
            threadgroup float4* k_tile4 = (threadgroup float4*)tg_mem.k_tile;
            uint total_f4 = tile_len * 16;
            for (uint idx4 = tid; idx4 < total_f4; idx4 += tg_size) {
                k_tile4[idx4] = K4[idx4];
            }
            device const float4* V4 = (device const float4*)(V + j_start * 64);
            for (uint idx4 = tid; idx4 < total_f4; idx4 += tg_size) {
                uint k_row = idx4 >> 4;
                uint k_col4 = (idx4 & 15) << 2;
                float4 v_val = V4[idx4];
                tg_mem.v_transposed[(k_col4 + 0) * c.tile_n + k_row] = v_val.x;
                tg_mem.v_transposed[(k_col4 + 1) * c.tile_n + k_row] = v_val.y;
                tg_mem.v_transposed[(k_col4 + 2) * c.tile_n + k_row] = v_val.z;
                tg_mem.v_transposed[(k_col4 + 3) * c.tile_n + k_row] = v_val.w;
            }
        } else {
            uint total_elements = tile_len * c.head_dim;
            for (uint idx = tid; idx < total_elements; idx += tg_size) {
                uint k_row = idx / c.head_dim;
                uint k_col = idx % c.head_dim;
                uint global_k_pos = j_start + k_row;
                tg_mem.k_tile[k_row * c.head_dim + k_col] = K[global_k_pos * c.head_dim + k_col];
                tg_mem.v_transposed[k_col * c.tile_n + k_row] = V[global_k_pos * c.head_dim + k_col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < tile_len; ++k) {
            uint global_key_pos = j_start + k;
            float score = -INFINITY;
            if (nax_can_attend(m, global_key_pos, c)) {
                float partial_qk = 0.0f;
                if (c.head_dim == 64) {
                    partial_qk = q_reg[0] * tg_mem.k_tile[k * 64 + simd_lane] +
                                 q_reg[1] * tg_mem.k_tile[k * 64 + simd_lane + 32];
                } else {
                    uint d_idx = 0;
                    for (uint d = simd_lane; d < c.head_dim; d += 32) {
                        partial_qk += q_reg[d_idx++] * tg_mem.k_tile[k * c.head_dim + d];
                    }
                }
                score = simd_sum(partial_qk) * c.scale;
            }

            float m_new = max(m_prev, score);
            float alpha = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_new);
            float p = (score == -INFINITY) ? 0.0f : exp(score - m_new);
            l_prev = alpha * l_prev + p;

            if (c.head_dim == 64) {
                float v_val0 = tg_mem.v_transposed[simd_lane * c.tile_n + k];
                float v_val1 = tg_mem.v_transposed[(simd_lane + 32) * c.tile_n + k];
                acc[0] = alpha * acc[0] + p * v_val0;
                acc[1] = alpha * acc[1] + p * v_val1;
            } else {
                uint d_idx = 0;
                for (uint d = simd_lane; d < c.head_dim; d += 32) {
                    float v_val = tg_mem.v_transposed[d * c.tile_n + k];
                    acc[d_idx] = alpha * acc[d_idx] + p * v_val;
                    d_idx++;
                }
            }
            m_prev = m_new;
        }
    }

    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    uint d_idx = 0;
    for (uint d = simd_lane; d < c.head_dim; d += 32) {
        Out[m * c.head_dim + d] = acc[d_idx++] * inv_l;
    }
    if (simd_lane == 0) {
        LSE[m] = (l_prev > 0.0f) ? (m_prev + log(l_prev)) : -INFINITY;
    }
}
)MSL";

static int qwen35_init_pipelines(void) {
    if (gQwen35PipelinesReady) return 1;
    if (!mg_init() || gDev == nil) return 0;
    @synchronized(gDev) {
        if (gQwen35PipelinesReady) return 1;
        NSError *err = nil;
        id<MTLLibrary> lib = [gDev newLibraryWithSource:kQwen35DecodeMSL options:nil error:&err];
        if (!lib) {
            NSLog(@"qwen35_decode: MSL compilation failed: %@", err);
            return 0;
        }
        MTLComputePipelineDescriptor *desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.supportIndirectCommandBuffers = YES;

        desc.computeFunction = [lib newFunctionWithName:@"gdn_conv_wide_m"];
        psoGDNConvWideM = [gDev newComputePipelineStateWithDescriptor:desc options:0 reflection:nil error:&err];
        if (!psoGDNConvWideM) {
            NSLog(@"qwen35_decode: psoGDNConvWideM build failed: %@", err);
            return 0;
        }

        desc.computeFunction = [lib newFunctionWithName:@"gdn_qk_norm_wide_m"];
        psoGDNQKNormWideM = [gDev newComputePipelineStateWithDescriptor:desc options:0 reflection:nil error:&err];
        if (!psoGDNQKNormWideM) {
            NSLog(@"qwen35_decode: psoGDNQKNormWideM build failed: %@", err);
            return 0;
        }

        desc.computeFunction = [lib newFunctionWithName:@"gdn_recurrent_wide_m"];
        psoGDNRecurrentWideM = [gDev newComputePipelineStateWithDescriptor:desc options:0 reflection:nil error:&err];
        if (!psoGDNRecurrentWideM) {
            NSLog(@"qwen35_decode: psoGDNRecurrentWideM build failed: %@", err);
            return 0;
        }

        desc.computeFunction = [lib newFunctionWithName:@"sdpa_nax_tail_causal_tile"];
        psoSDPATailCausalTile = [gDev newComputePipelineStateWithDescriptor:desc options:0 reflection:nil error:&err];
        if (!psoSDPATailCausalTile) {
            NSLog(@"qwen35_decode: psoSDPATailCausalTile build failed: %@", err);
            return 0;
        }
        gQwen35PipelinesReady = 1;
        return 1;
    }
}

typedef struct {
    id<MTLBuffer> mixedBuf;
    id<MTLBuffer> zBuf;
    id<MTLBuffer> bBuf;
    id<MTLBuffer> aBuf;
    id<MTLBuffer> convWBuf;
    id<MTLBuffer> aLogBuf;
    id<MTLBuffer> dtBiasBuf;
    id<MTLBuffer> normBuf;
    id<MTLBuffer> coreOutBuf;
    id<MTLBuffer> parentsBuf;
    id<MTLBuffer> treeStateBuf;
    id<MTLBuffer> convOut;
    id<MTLBuffer> qNorm;
    id<MTLBuffer> kNorm;
    id<MTLBuffer> qBuf;
    id<MTLBuffer> kBuf;
    id<MTLBuffer> vBuf;
    id<MTLBuffer> outBuf;
    id<MTLBuffer> lseBuf;
    id<MTLBuffer> constBuf;
    id<MTLBuffer> gemmX;
    id<MTLBuffer> gemmY;
    id<MTLIndirectCommandBuffer> icb;
    int icbSlotCount;
    int lastTokens;
    int lastSdpaM;
    int lastConvDim;
    int lastNK;
    int lastNV;
    int lastKHd;
    int lastVHd;
    int lastConvKernel;
    int lastGdnHandle;
    id<MTLBuffer> lastEffectiveMixed;
} WideMVerifyScratch;

static WideMVerifyScratch gVerifyScratch;

static id<MTLBuffer> verify_ensure_buffer(id<MTLBuffer> buf, size_t needed, MTLResourceOptions options) {
    if (buf != nil && [buf length] >= needed) {
        return buf;
    }
    size_t allocSize = needed < 262144 ? 262144 : (needed < 4194304 ? 4194304 : needed);
    return [gDev newBufferWithLength:allocSize options:options];
}

// ==============================================================================
// State Lifecycle APIs
// ==============================================================================

int mg_qwen35_decode_create(int nK, int nV, int kHd, int vHd, int convKernel) {
    if (!qwen35_init_pipelines()) return -1;
    int convDim = 2 * (nK * kHd) + (nV * vHd);
    size_t convBytes = (size_t)(convKernel - 1) * convDim * sizeof(float);
    size_t recurrentBytes = (size_t)nV * kHd * vHd * sizeof(float);

    id<MTLBuffer> convBuf = [gDev newBufferWithLength:convBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> recurrentBuf = [gDev newBufferWithLength:recurrentBytes options:MTLResourceStorageModeShared];
    if (!convBuf || !recurrentBuf) return -1;
    memset([convBuf contents], 0, convBytes);
    memset([recurrentBuf contents], 0, recurrentBytes);

    @synchronized(gDev) {
        int handle = -1;
        for (int i = 0; i < gQwen35StateCount; ++i) {
            if (!gQwen35States[i].inUse) {
                handle = i;
                break;
            }
        }
        if (handle == -1) {
            if (gQwen35StateCount >= QWEN35_MAX_STATES) return -1;
            handle = gQwen35StateCount++;
        }
        gQwen35States[handle].conv = convBuf;
        gQwen35States[handle].recurrent = recurrentBuf;
        gQwen35States[handle].nK = nK;
        gQwen35States[handle].nV = nV;
        gQwen35States[handle].kHd = kHd;
        gQwen35States[handle].vHd = vHd;
        gQwen35States[handle].convKernel = convKernel;
        gQwen35States[handle].inUse = 1;
        return handle;
    }
}

void mg_qwen35_decode_reset(int handle) {
    if (handle < 0 || handle >= gQwen35StateCount) return;
    @synchronized(gDev) {
        if (!gQwen35States[handle].inUse) return;
        memset([gQwen35States[handle].conv contents], 0, [gQwen35States[handle].conv length]);
        memset([gQwen35States[handle].recurrent contents], 0, [gQwen35States[handle].recurrent length]);
    }
}

void mg_qwen35_decode_release(int handle) {
    if (handle < 0 || handle >= gQwen35StateCount) return;
    @synchronized(gDev) {
        if (!gQwen35States[handle].inUse) return;
        gQwen35States[handle].conv = nil;
        gQwen35States[handle].recurrent = nil;
        gQwen35States[handle].inUse = 0;
    }
}

int mg_qwen35_decode_get_state(int handle, float* conv_out, float* recurrent_out) {
    if (handle < 0 || handle >= gQwen35StateCount) return 0;
    @synchronized(gDev) {
        if (!gQwen35States[handle].inUse) return 0;
        if (conv_out) {
            memcpy(conv_out, [gQwen35States[handle].conv contents], [gQwen35States[handle].conv length]);
        }
        if (recurrent_out) {
            memcpy(recurrent_out, [gQwen35States[handle].recurrent contents], [gQwen35States[handle].recurrent length]);
        }
        return 1;
    }
}

int mg_qwen35_decode_set_state(int handle, const float* conv_in, const float* recurrent_in) {
    if (handle < 0 || handle >= gQwen35StateCount) return 0;
    @synchronized(gDev) {
        if (!gQwen35States[handle].inUse) return 0;
        if (conv_in) {
            memcpy([gQwen35States[handle].conv contents], conv_in, [gQwen35States[handle].conv length]);
        }
        if (recurrent_in) {
            memcpy([gQwen35States[handle].recurrent contents], recurrent_in, [gQwen35States[handle].recurrent length]);
        }
        return 1;
    }
}

// ==============================================================================
// Recurrent GDN Step Execution
// ==============================================================================

static int mg_qwen35_gdn_encode_into(
    id<MTLCommandBuffer> cb,
    id<MTLBuffer> convStateBuf,
    id<MTLBuffer> recStateBuf,
    id<MTLBuffer> mixedBuf,
    id<MTLBuffer> zBuf,
    id<MTLBuffer> bBuf,
    id<MTLBuffer> aBuf,
    id<MTLBuffer> convWBuf,
    id<MTLBuffer> aLogBuf,
    id<MTLBuffer> dtBiasBuf,
    id<MTLBuffer> normBuf,
    id<MTLBuffer> coreOutBuf,
    int tokens, int nK, int nV, int kHd, int vHd, int convKernel, float eps,
    id<MTLBuffer> parentsBuf,
    id<MTLBuffer> treeStateBuf
) {
    int keyDim = nK * kHd;
    int valueDim = nV * vHd;
    int convDim = 2 * keyDim + valueDim;

    id<MTLBuffer> convOut = [gDev newBufferWithLength:(size_t)tokens * convDim * sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> qNorm = [gDev newBufferWithLength:(size_t)tokens * keyDim * sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> kNorm = [gDev newBufferWithLength:(size_t)tokens * keyDim * sizeof(float) options:MTLResourceStorageModePrivate];
    if (!convOut || !qNorm || !kNorm) return 0;

    id<MTLBuffer> activeParentsBuf = parentsBuf;
    if (!activeParentsBuf) {
        int linearParents[32];
        linearParents[0] = -1;
        for (int i = 1; i < 32; ++i) linearParents[i] = i - 1;
        activeParentsBuf = [gDev newBufferWithBytes:linearParents length:(size_t)tokens * sizeof(int) options:MTLResourceStorageModeShared];
        if (!activeParentsBuf) return 0;
    }

    id<MTLBuffer> activeTreeStateBuf = treeStateBuf;
    if (!activeTreeStateBuf) {
        activeTreeStateBuf = [gDev newBufferWithLength:(size_t)tokens * nV * kHd * vHd * sizeof(float) options:MTLResourceStorageModePrivate];
        if (!activeTreeStateBuf) return 0;
    }

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) return 0;

    // 1. Convolution
    [enc setComputePipelineState:psoGDNConvWideM];
    [enc setBuffer:mixedBuf offset:0 atIndex:0];
    [enc setBuffer:convWBuf offset:0 atIndex:1];
    [enc setBuffer:convStateBuf offset:0 atIndex:2];
    [enc setBuffer:convOut offset:0 atIndex:3];
    [enc setBytes:&tokens length:sizeof(tokens) atIndex:4];
    [enc setBytes:&convDim length:sizeof(convDim) atIndex:5];
    [enc setBytes:&convKernel length:sizeof(convKernel) atIndex:6];
    [enc setBuffer:activeParentsBuf offset:0 atIndex:7];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)convDim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // 2. QK Normalization
    int qThreads = kHd <= 32 ? 32 : (kHd <= 64 ? 64 : 128);
    [enc setComputePipelineState:psoGDNQKNormWideM];
    [enc setBuffer:convOut offset:0 atIndex:0];
    [enc setBuffer:qNorm offset:0 atIndex:1];
    [enc setBuffer:kNorm offset:0 atIndex:2];
    [enc setBytes:&tokens length:sizeof(tokens) atIndex:3];
    [enc setBytes:&convDim length:sizeof(convDim) atIndex:4];
    [enc setBytes:&nK length:sizeof(nK) atIndex:5];
    [enc setBytes:&kHd length:sizeof(kHd) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nK, (NSUInteger)tokens, 1)
        threadsPerThreadgroup:MTLSizeMake((NSUInteger)qThreads, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // 3. Recurrent Delta Update
    int vThreads = vHd <= 32 ? 32 : (vHd <= 64 ? 64 : 128);
    uint32_t parentMask = 0;
    [enc setComputePipelineState:psoGDNRecurrentWideM];
    [enc setBuffer:convOut offset:0 atIndex:0];
    [enc setBuffer:qNorm offset:0 atIndex:1];
    [enc setBuffer:kNorm offset:0 atIndex:2];
    [enc setBuffer:zBuf offset:0 atIndex:3];
    [enc setBuffer:bBuf offset:0 atIndex:4];
    [enc setBuffer:aBuf offset:0 atIndex:5];
    [enc setBuffer:aLogBuf offset:0 atIndex:6];
    [enc setBuffer:dtBiasBuf offset:0 atIndex:7];
    [enc setBuffer:normBuf offset:0 atIndex:8];
    [enc setBuffer:recStateBuf offset:0 atIndex:9];
    [enc setBuffer:coreOutBuf offset:0 atIndex:10];
    [enc setBytes:&tokens length:sizeof(tokens) atIndex:11];
    [enc setBytes:&convDim length:sizeof(convDim) atIndex:12];
    [enc setBytes:&nK length:sizeof(nK) atIndex:13];
    [enc setBytes:&nV length:sizeof(nV) atIndex:14];
    [enc setBytes:&kHd length:sizeof(kHd) atIndex:15];
    [enc setBytes:&vHd length:sizeof(vHd) atIndex:16];
    [enc setBytes:&eps length:sizeof(eps) atIndex:17];
    [enc setBuffer:activeParentsBuf offset:0 atIndex:18];
    [enc setBuffer:activeTreeStateBuf offset:0 atIndex:19];
    [enc setBytes:&parentMask length:sizeof(parentMask) atIndex:20];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nV, 1, 1)
        threadsPerThreadgroup:MTLSizeMake((NSUInteger)vThreads, 1, 1)];
    [enc endEncoding];
    return 1;
}

int mg_qwen35_decode_step_wide_m(
    int handle,
    const float* mixed, const float* z, const float* b, const float* a,
    const float* convW, const float* aLog, const float* dtBias, const float* norm,
    float* core_out,
    int tokens, int nK, int nV, int kHd, int vHd, int convKernel, float eps,
    const int* parents
) {
    if (!qwen35_init_pipelines()) return 0;
    if (handle < 0 || handle >= gQwen35StateCount || tokens <= 0 || eps <= 0.0f) return 0;
    if (!mixed || !z || !b || !a || !convW || !aLog || !dtBias || !norm || !core_out) return 0;

    Qwen35DecodeState state;
    @synchronized(gDev) {
        if (!gQwen35States[handle].inUse) return 0;
        state = gQwen35States[handle];
    }
    if (state.nK != nK || state.nV != nV || state.kHd != kHd || state.vHd != vHd || state.convKernel != convKernel) {
        return 0;
    }

    int convDim = 2 * (nK * kHd) + (nV * vHd);
    int valueDim = nV * vHd;

    @autoreleasepool {
        id<MTLBuffer> mixedBuf = [gDev newBufferWithBytes:mixed length:(size_t)tokens * convDim * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> zBuf = [gDev newBufferWithBytes:z length:(size_t)tokens * valueDim * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [gDev newBufferWithBytes:b length:(size_t)tokens * nV * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> aBuf = [gDev newBufferWithBytes:a length:(size_t)tokens * nV * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> convWBuf = [gDev newBufferWithBytes:convW length:(size_t)convDim * convKernel * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> aLogBuf = [gDev newBufferWithBytes:aLog length:(size_t)nV * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> dtBiasBuf = [gDev newBufferWithBytes:dtBias length:(size_t)nV * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> normBuf = [gDev newBufferWithBytes:norm length:(size_t)vHd * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> coreOutBuf = [gDev newBufferWithLength:(size_t)tokens * valueDim * sizeof(float) options:MTLResourceStorageModeShared];

        if (!mixedBuf || !zBuf || !bBuf || !aBuf || !convWBuf || !aLogBuf || !dtBiasBuf || !normBuf || !coreOutBuf) {
            return 0;
        }

        id<MTLBuffer> parentsBuf = nil;
        id<MTLBuffer> treeStateBuf = nil;
        if (parents != NULL) {
            parentsBuf = [gDev newBufferWithBytes:parents length:(size_t)tokens * sizeof(int) options:MTLResourceStorageModeShared];
            treeStateBuf = [gDev newBufferWithLength:(size_t)tokens * nV * kHd * vHd * sizeof(float) options:MTLResourceStorageModePrivate];
            if (!parentsBuf || !treeStateBuf) return 0;
        }

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        if (!cb) return 0;

        if (!mg_qwen35_gdn_encode_into(cb, state.conv, state.recurrent, mixedBuf, zBuf, bBuf, aBuf,
                                      convWBuf, aLogBuf, dtBiasBuf, normBuf, coreOutBuf,
                                      tokens, nK, nV, kHd, vHd, convKernel, eps,
                                      parentsBuf, treeStateBuf)) {
            return 0;
        }

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return 0;

        memcpy(core_out, [coreOutBuf contents], (size_t)tokens * valueDim * sizeof(float));
        return 1;
    }
}

int mg_qwen35_decode_step(
    int handle,
    const float* mixed, const float* z, const float* b, const float* a,
    const float* convW, const float* aLog, const float* dtBias, const float* norm,
    float* core_out,
    int nK, int nV, int kHd, int vHd, int convKernel, float eps
) {
    return mg_qwen35_decode_step_wide_m(handle, mixed, z, b, a, convW, aLog, dtBias, norm,
                                        core_out, 1, nK, nV, kHd, vHd, convKernel, eps, NULL);
}

// ==============================================================================
// Tail-Causal SDPA Dispatch
// ==============================================================================

int mg_sdpa_nax_tail_causal_tile_run(
    const float* Q, const float* K, const float* V,
    float* Out, float* LSE,
    int gqaFactor, int draftLen, int M, int headDim,
    int prefixLen, int totalKV, float scale, int tileN, int order,
    int hasTreeMask, const uint32_t* treeMask
) {
    if (!qwen35_init_pipelines()) return 0;
    if (!Q || !K || !V || !Out || !LSE || M <= 0 || headDim <= 0 || totalKV <= 0) return 0;

    @autoreleasepool {
        size_t qBytes = (size_t)M * headDim * sizeof(float);
        size_t kvBytes = (size_t)totalKV * headDim * sizeof(float);
        size_t lseBytes = (size_t)M * sizeof(float);

        id<MTLBuffer> qBuf = [gDev newBufferWithBytes:Q length:qBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> kBuf = [gDev newBufferWithBytes:K length:kvBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> vBuf = [gDev newBufferWithBytes:V length:kvBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = [gDev newBufferWithLength:qBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> lseBuf = [gDev newBufferWithLength:lseBytes options:MTLResourceStorageModeShared];

        if (!qBuf || !kBuf || !vBuf || !outBuf || !lseBuf) return 0;

        SDPANAXConstants constants;
        memset(&constants, 0, sizeof(constants));
        constants.gqa_factor = (uint32_t)gqaFactor;
        constants.draft_len = (uint32_t)draftLen;
        constants.M = (uint32_t)M;
        constants.head_dim = (uint32_t)headDim;
        constants.prefix_len = (uint32_t)prefixLen;
        constants.total_kv = (uint32_t)totalKV;
        constants.scale = scale;
        constants.tile_n = (uint32_t)tileN;
        constants.order = (uint32_t)order;
        constants.has_tree_mask = (uint32_t)hasTreeMask;
        if (hasTreeMask && treeMask) {
            uint32_t copyCount = draftLen > 32 ? 32 : (uint32_t)draftLen;
            memcpy(constants.tree_mask, treeMask, copyCount * sizeof(uint32_t));
        }

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return 0;

        uint threads = 32;
        uint num_tgs = (uint)M;

        [enc setComputePipelineState:psoSDPATailCausalTile];
        [enc setBuffer:qBuf offset:0 atIndex:0];
        [enc setBuffer:kBuf offset:0 atIndex:1];
        [enc setBuffer:vBuf offset:0 atIndex:2];
        [enc setBuffer:outBuf offset:0 atIndex:3];
        [enc setBuffer:lseBuf offset:0 atIndex:4];
        [enc setBytes:&constants length:sizeof(constants) atIndex:5];
        [enc setThreadgroupMemoryLength:sizeof(SDPANAXThreadgroupStorage) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)threads, 1, 1)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return 0;

        memcpy(Out, [outBuf contents], qBytes);
        memcpy(LSE, [lseBuf contents], lseBytes);
        return 1;
    }
}

// ==============================================================================
// Integrated Verification Sequence in a Single Command Buffer
// ==============================================================================
// Evaluates:
//   1. Wide-M Q4_K GEMM
//   2. Batched recurrent GDN state step
//   3. Tail-causal wide-M SDPA tile
// inside ONE command buffer with zero host synchronization round-trips.

int mg_metal_wide_m_verify_step(
    int q4_wid,
    const float* draft_input, // [M, In]
    float* gemm_out,          // [M, Out]
    int gdn_handle,
    const float* z, const float* b, const float* a,
    const float* convW, const float* aLog, const float* dtBias, const float* norm,
    float* gdn_core_out,
    int nK, int nV, int kHd, int vHd, int convKernel, float eps,
    const int* parents,
    const float* Q, const float* K, const float* V,
    float* sdpa_out, float* lse_out,
    int gqaFactor, int draftLen, int sdpaM, int headDim,
    int prefixLen, int totalKV, float sdpaScale, int tileN, int order,
    int hasTreeMask, const uint32_t* treeMask
) {
    if (!qwen35_init_pipelines()) return 0;
    if (draftLen < 2 || draftLen > 32) return 0;

    Qwen35DecodeState gdnState;
    @synchronized(gDev) {
        if (gdn_handle < 0 || gdn_handle >= gQwen35StateCount || !gQwen35States[gdn_handle].inUse) {
            return 0;
        }
        gdnState = gQwen35States[gdn_handle];
    }

    int tokens = draftLen;
    int keyDim = nK * kHd;
    int valueDim = nV * vHd;
    int convDim = 2 * keyDim + valueDim;
    size_t qBytes = (size_t)sdpaM * headDim * sizeof(float);
    size_t kvBytes = (size_t)totalKV * headDim * sizeof(float);
    size_t lseBytes = (size_t)sdpaM * sizeof(float);
    size_t gemmYBytes = (size_t)tokens * convDim * sizeof(float);

    @autoreleasepool {
        // Ensure reusable persistent scratch buffers to eliminate host heap/VRAM allocation overhead
        if (gVerifyScratch.mixedBuf == nil || [gVerifyScratch.mixedBuf length] < (size_t)tokens * convDim * sizeof(float) ||
            [gVerifyScratch.treeStateBuf length] < (size_t)tokens * nV * kHd * vHd * sizeof(float)) {
            @synchronized(gDev) {
                if (q4_wid >= 0) {
                    gVerifyScratch.gemmX = verify_ensure_buffer(gVerifyScratch.gemmX, (size_t)tokens * convDim * sizeof(float), MTLResourceStorageModeShared);
                    gVerifyScratch.gemmY = verify_ensure_buffer(gVerifyScratch.gemmY, (size_t)tokens * convDim * sizeof(float), MTLResourceStorageModeShared);
                }
                gVerifyScratch.mixedBuf = verify_ensure_buffer(gVerifyScratch.mixedBuf, (size_t)tokens * convDim * sizeof(float), MTLResourceStorageModeShared);
                gVerifyScratch.zBuf = verify_ensure_buffer(gVerifyScratch.zBuf, (size_t)tokens * valueDim * sizeof(float), MTLResourceStorageModeShared);
                gVerifyScratch.bBuf = verify_ensure_buffer(gVerifyScratch.bBuf, (size_t)tokens * nV * sizeof(float), MTLResourceStorageModeShared);
                gVerifyScratch.aBuf = verify_ensure_buffer(gVerifyScratch.aBuf, (size_t)tokens * nV * sizeof(float), MTLResourceStorageModeShared);
                gVerifyScratch.convWBuf = verify_ensure_buffer(gVerifyScratch.convWBuf, (size_t)convDim * convKernel * sizeof(float), MTLResourceStorageModeShared);
                gVerifyScratch.aLogBuf = verify_ensure_buffer(gVerifyScratch.aLogBuf, (size_t)nV * sizeof(float), MTLResourceStorageModeShared);
                gVerifyScratch.dtBiasBuf = verify_ensure_buffer(gVerifyScratch.dtBiasBuf, (size_t)nV * sizeof(float), MTLResourceStorageModeShared);
                gVerifyScratch.normBuf = verify_ensure_buffer(gVerifyScratch.normBuf, (size_t)vHd * sizeof(float), MTLResourceStorageModeShared);
                gVerifyScratch.coreOutBuf = verify_ensure_buffer(gVerifyScratch.coreOutBuf, (size_t)tokens * valueDim * sizeof(float), MTLResourceStorageModeShared);
                gVerifyScratch.parentsBuf = verify_ensure_buffer(gVerifyScratch.parentsBuf, (size_t)tokens * sizeof(int), MTLResourceStorageModeShared);
                gVerifyScratch.treeStateBuf = verify_ensure_buffer(gVerifyScratch.treeStateBuf, (size_t)tokens * nV * kHd * vHd * sizeof(float), MTLResourceStorageModePrivate);
                gVerifyScratch.convOut = verify_ensure_buffer(gVerifyScratch.convOut, (size_t)tokens * convDim * sizeof(float), MTLResourceStorageModePrivate);
                gVerifyScratch.qNorm = verify_ensure_buffer(gVerifyScratch.qNorm, (size_t)tokens * keyDim * sizeof(float), MTLResourceStorageModePrivate);
                gVerifyScratch.kNorm = verify_ensure_buffer(gVerifyScratch.kNorm, (size_t)tokens * keyDim * sizeof(float), MTLResourceStorageModePrivate);
                gVerifyScratch.qBuf = verify_ensure_buffer(gVerifyScratch.qBuf, qBytes, MTLResourceStorageModeShared);
                gVerifyScratch.kBuf = verify_ensure_buffer(gVerifyScratch.kBuf, kvBytes, MTLResourceStorageModeShared);
                gVerifyScratch.vBuf = verify_ensure_buffer(gVerifyScratch.vBuf, kvBytes, MTLResourceStorageModeShared);
                gVerifyScratch.outBuf = verify_ensure_buffer(gVerifyScratch.outBuf, qBytes, MTLResourceStorageModeShared);
                gVerifyScratch.lseBuf = verify_ensure_buffer(gVerifyScratch.lseBuf, lseBytes, MTLResourceStorageModeShared);
                gVerifyScratch.constBuf = verify_ensure_buffer(gVerifyScratch.constBuf, 4096, MTLResourceStorageModeShared);
            }
        }

        if (!gVerifyScratch.mixedBuf || !gVerifyScratch.zBuf || !gVerifyScratch.bBuf || !gVerifyScratch.aBuf ||
            !gVerifyScratch.convWBuf || !gVerifyScratch.aLogBuf || !gVerifyScratch.dtBiasBuf || !gVerifyScratch.normBuf ||
            !gVerifyScratch.coreOutBuf || !gVerifyScratch.parentsBuf || !gVerifyScratch.treeStateBuf ||
            !gVerifyScratch.convOut || !gVerifyScratch.qNorm || !gVerifyScratch.kNorm ||
            !gVerifyScratch.qBuf || !gVerifyScratch.kBuf || !gVerifyScratch.vBuf ||
            !gVerifyScratch.outBuf || !gVerifyScratch.lseBuf || !gVerifyScratch.constBuf) {
            return 0;
        }

        // Copy input slices into shared scratch buffers
        if (q4_wid >= 0 && draft_input) {
            memcpy([gVerifyScratch.gemmX contents], draft_input, (size_t)tokens * convDim * sizeof(float));
        } else if (draft_input) {
            memcpy([gVerifyScratch.mixedBuf contents], draft_input, (size_t)tokens * convDim * sizeof(float));
        }
        memcpy([gVerifyScratch.zBuf contents], z, (size_t)tokens * valueDim * sizeof(float));
        memcpy([gVerifyScratch.bBuf contents], b, (size_t)tokens * nV * sizeof(float));
        memcpy([gVerifyScratch.aBuf contents], a, (size_t)tokens * nV * sizeof(float));
        memcpy([gVerifyScratch.convWBuf contents], convW, (size_t)convDim * convKernel * sizeof(float));
        memcpy([gVerifyScratch.aLogBuf contents], aLog, (size_t)nV * sizeof(float));
        memcpy([gVerifyScratch.dtBiasBuf contents], dtBias, (size_t)nV * sizeof(float));
        memcpy([gVerifyScratch.normBuf contents], norm, (size_t)vHd * sizeof(float));
        if (parents != NULL) {
            memcpy([gVerifyScratch.parentsBuf contents], parents, (size_t)tokens * sizeof(int));
        } else {
            int linearParents[32];
            linearParents[0] = -1;
            for (int i = 1; i < 32; ++i) linearParents[i] = i - 1;
            memcpy([gVerifyScratch.parentsBuf contents], linearParents, (size_t)tokens * sizeof(int));
        }
        memcpy([gVerifyScratch.qBuf contents], Q, qBytes);
        memcpy([gVerifyScratch.kBuf contents], K, kvBytes);
        memcpy([gVerifyScratch.vBuf contents], V, kvBytes);

        // Populate constant buffer (256-byte aligned offsets for universal Apple Silicon compliance)
        SDPANAXConstants constants;
        memset(&constants, 0, sizeof(constants));
        constants.gqa_factor = (uint32_t)gqaFactor;
        constants.draft_len = (uint32_t)draftLen;
        constants.M = (uint32_t)sdpaM;
        constants.head_dim = (uint32_t)headDim;
        constants.prefix_len = (uint32_t)prefixLen;
        constants.total_kv = (uint32_t)totalKV;
        constants.scale = sdpaScale;
        constants.tile_n = (uint32_t)tileN;
        constants.order = (uint32_t)order;
        constants.has_tree_mask = (uint32_t)hasTreeMask;
        if (hasTreeMask && treeMask) {
            uint32_t copyCount = draftLen > 32 ? 32 : (uint32_t)draftLen;
            memcpy(constants.tree_mask, treeMask, copyCount * sizeof(uint32_t));
        }

        uint8_t *cPtr = (uint8_t *)[gVerifyScratch.constBuf contents];
        *((int *)(cPtr + 0)) = tokens;
        *((int *)(cPtr + 256)) = convDim;
        *((int *)(cPtr + 512)) = convKernel;
        *((int *)(cPtr + 768)) = nK;
        *((int *)(cPtr + 1024)) = nV;
        *((int *)(cPtr + 1280)) = kHd;
        *((int *)(cPtr + 1536)) = vHd;
        *((float *)(cPtr + 1792)) = eps;
        uint32_t parentMask = 0;
        if (parents != NULL) {
            for (int t = 0; t < tokens; ++t) {
                if (parents[t] >= 0 && parents[t] < 32) {
                    parentMask |= (1u << parents[t]);
                }
            }
        }
        *((uint32_t *)(cPtr + 1808)) = parentMask;
        memcpy(cPtr + 2048, &constants, sizeof(constants));

        id<MTLCommandBuffer> cb = [gQueue commandBufferWithUnretainedReferences];
        if (!cb) cb = [gQueue commandBuffer];
        if (!cb) return 0;

        if (q4_wid >= 0) {
            if (!mg_q4k_gemv_wide_m_encode((__bridge void*)cb, q4_wid, (__bridge void*)gVerifyScratch.gemmX, (__bridge void*)gVerifyScratch.gemmY, tokens)) {
                return 0;
            }
        }
        id<MTLBuffer> effectiveMixed = (q4_wid >= 0) ? gVerifyScratch.gemmY : gVerifyScratch.mixedBuf;

        // Check ICB readiness and support
        int useICB = 0;
        if (gVerifyScratch.icb == nil) {
            @synchronized(gDev) {
                if (gVerifyScratch.icb == nil) {
                    MTLIndirectCommandBufferDescriptor *icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
                    icbDesc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
                    icbDesc.inheritBuffers = NO;
                    icbDesc.inheritPipelineState = NO;
                    icbDesc.maxKernelBufferBindCount = 24;
                    gVerifyScratch.icb = [gDev newIndirectCommandBufferWithDescriptor:icbDesc maxCommandCount:4 options:MTLResourceStorageModeShared];
                }
            }
        }
        if (gVerifyScratch.icb != nil && q4_wid < 0) {
            useICB = 1;
        }

        int qThreads = kHd <= 32 ? 32 : (kHd <= 64 ? 64 : 128);
        int vThreads = vHd <= 32 ? 32 : (vHd <= 64 ? 64 : 128);

        if (useICB) {
            int icbNeedsRecord = (gVerifyScratch.lastTokens != tokens ||
                                  gVerifyScratch.lastSdpaM != sdpaM ||
                                  gVerifyScratch.lastConvDim != convDim ||
                                  gVerifyScratch.lastNK != nK ||
                                  gVerifyScratch.lastNV != nV ||
                                  gVerifyScratch.lastKHd != kHd ||
                                  gVerifyScratch.lastVHd != vHd ||
                                  gVerifyScratch.lastConvKernel != convKernel ||
                                  gVerifyScratch.lastGdnHandle != gdn_handle ||
                                  gVerifyScratch.lastEffectiveMixed != effectiveMixed);

            if (icbNeedsRecord) {
                int slot = 0;

                // 1. SDPA Tail Causal Tile (dispatches concurrently with GDN pipeline - zero data hazard)
                {
                    id<MTLIndirectComputeCommand> cmd = [gVerifyScratch.icb indirectComputeCommandAtIndex:(NSUInteger)slot++];
                    [cmd setComputePipelineState:psoSDPATailCausalTile];
                    [cmd setKernelBuffer:gVerifyScratch.qBuf offset:0 atIndex:0];
                    [cmd setKernelBuffer:gVerifyScratch.kBuf offset:0 atIndex:1];
                    [cmd setKernelBuffer:gVerifyScratch.vBuf offset:0 atIndex:2];
                    [cmd setKernelBuffer:gVerifyScratch.outBuf offset:0 atIndex:3];
                    [cmd setKernelBuffer:gVerifyScratch.lseBuf offset:0 atIndex:4];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:2048 atIndex:5];
                    [cmd setThreadgroupMemoryLength:sizeof(SDPANAXThreadgroupStorage) atIndex:0];
                    [cmd concurrentDispatchThreadgroups:MTLSizeMake((NSUInteger)sdpaM, 1, 1)
                                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                }

                // 2. GDN Conv (executes concurrently with SDPA)
                {
                    id<MTLIndirectComputeCommand> cmd = [gVerifyScratch.icb indirectComputeCommandAtIndex:(NSUInteger)slot++];
                    [cmd setComputePipelineState:psoGDNConvWideM];
                    [cmd setKernelBuffer:effectiveMixed offset:0 atIndex:0];
                    [cmd setKernelBuffer:gVerifyScratch.convWBuf offset:0 atIndex:1];
                    [cmd setKernelBuffer:gdnState.conv offset:0 atIndex:2];
                    [cmd setKernelBuffer:gVerifyScratch.convOut offset:0 atIndex:3];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:0 atIndex:4];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:256 atIndex:5];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:512 atIndex:6];
                    [cmd setKernelBuffer:gVerifyScratch.parentsBuf offset:0 atIndex:7];
                    [cmd concurrentDispatchThreadgroups:MTLSizeMake(((NSUInteger)convDim + 255) / 256, 1, 1)
                                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }

                // 3. GDN QK Norm (barrier waits for Conv)
                {
                    id<MTLIndirectComputeCommand> cmd = [gVerifyScratch.icb indirectComputeCommandAtIndex:(NSUInteger)slot++];
                    [cmd setBarrier];
                    [cmd setComputePipelineState:psoGDNQKNormWideM];
                    [cmd setKernelBuffer:gVerifyScratch.convOut offset:0 atIndex:0];
                    [cmd setKernelBuffer:gVerifyScratch.qNorm offset:0 atIndex:1];
                    [cmd setKernelBuffer:gVerifyScratch.kNorm offset:0 atIndex:2];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:0 atIndex:3];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:256 atIndex:4];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:768 atIndex:5];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:1280 atIndex:6];
                    [cmd concurrentDispatchThreadgroups:MTLSizeMake((NSUInteger)nK, (NSUInteger)tokens, 1)
                                  threadsPerThreadgroup:MTLSizeMake((NSUInteger)qThreads, 1, 1)];
                }

                // 4. GDN Recurrent (barrier waits for QK Norm)
                {
                    id<MTLIndirectComputeCommand> cmd = [gVerifyScratch.icb indirectComputeCommandAtIndex:(NSUInteger)slot++];
                    [cmd setBarrier];
                    [cmd setComputePipelineState:psoGDNRecurrentWideM];
                    [cmd setKernelBuffer:gVerifyScratch.convOut offset:0 atIndex:0];
                    [cmd setKernelBuffer:gVerifyScratch.qNorm offset:0 atIndex:1];
                    [cmd setKernelBuffer:gVerifyScratch.kNorm offset:0 atIndex:2];
                    [cmd setKernelBuffer:gVerifyScratch.zBuf offset:0 atIndex:3];
                    [cmd setKernelBuffer:gVerifyScratch.bBuf offset:0 atIndex:4];
                    [cmd setKernelBuffer:gVerifyScratch.aBuf offset:0 atIndex:5];
                    [cmd setKernelBuffer:gVerifyScratch.aLogBuf offset:0 atIndex:6];
                    [cmd setKernelBuffer:gVerifyScratch.dtBiasBuf offset:0 atIndex:7];
                    [cmd setKernelBuffer:gVerifyScratch.normBuf offset:0 atIndex:8];
                    [cmd setKernelBuffer:gdnState.recurrent offset:0 atIndex:9];
                    [cmd setKernelBuffer:gVerifyScratch.coreOutBuf offset:0 atIndex:10];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:0 atIndex:11];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:256 atIndex:12];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:768 atIndex:13];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:1024 atIndex:14];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:1280 atIndex:15];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:1536 atIndex:16];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:1792 atIndex:17];
                    [cmd setKernelBuffer:gVerifyScratch.parentsBuf offset:0 atIndex:18];
                    [cmd setKernelBuffer:gVerifyScratch.treeStateBuf offset:0 atIndex:19];
                    [cmd setKernelBuffer:gVerifyScratch.constBuf offset:1808 atIndex:20];
                    [cmd concurrentDispatchThreadgroups:MTLSizeMake((NSUInteger)nV, 1, 1)
                                  threadsPerThreadgroup:MTLSizeMake((NSUInteger)vThreads, 1, 1)];
                }

                gVerifyScratch.icbSlotCount = slot;
                gVerifyScratch.lastTokens = tokens;
                gVerifyScratch.lastSdpaM = sdpaM;
                gVerifyScratch.lastConvDim = convDim;
                gVerifyScratch.lastNK = nK;
                gVerifyScratch.lastNV = nV;
                gVerifyScratch.lastKHd = kHd;
                gVerifyScratch.lastVHd = vHd;
                gVerifyScratch.lastConvKernel = convKernel;
                gVerifyScratch.lastGdnHandle = gdn_handle;
                gVerifyScratch.lastEffectiveMixed = effectiveMixed;
            }

            id<MTLResource> resList[22];
            int rCount = 0;
            resList[rCount++] = effectiveMixed;
            resList[rCount++] = gVerifyScratch.convWBuf;
            resList[rCount++] = gdnState.conv;
            resList[rCount++] = gVerifyScratch.convOut;
            resList[rCount++] = gVerifyScratch.constBuf;
            resList[rCount++] = gVerifyScratch.parentsBuf;
            resList[rCount++] = gVerifyScratch.qNorm;
            resList[rCount++] = gVerifyScratch.kNorm;
            resList[rCount++] = gVerifyScratch.zBuf;
            resList[rCount++] = gVerifyScratch.bBuf;
            resList[rCount++] = gVerifyScratch.aBuf;
            resList[rCount++] = gVerifyScratch.aLogBuf;
            resList[rCount++] = gVerifyScratch.dtBiasBuf;
            resList[rCount++] = gVerifyScratch.normBuf;
            resList[rCount++] = gdnState.recurrent;
            resList[rCount++] = gVerifyScratch.coreOutBuf;
            resList[rCount++] = gVerifyScratch.treeStateBuf;
            resList[rCount++] = gVerifyScratch.qBuf;
            resList[rCount++] = gVerifyScratch.kBuf;
            resList[rCount++] = gVerifyScratch.vBuf;
            resList[rCount++] = gVerifyScratch.outBuf;
            resList[rCount++] = gVerifyScratch.lseBuf;

            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (!enc) return 0;
            [enc useResources:resList count:(NSUInteger)rCount usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            [enc executeCommandsInBuffer:gVerifyScratch.icb withRange:NSMakeRange(0, (NSUInteger)gVerifyScratch.icbSlotCount)];
            [enc endEncoding];
        } else {
            // Direct consolidated execution inside a single compute command encoder
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (!enc) return 0;

            // 1. Convolution
            [enc setComputePipelineState:psoGDNConvWideM];
            [enc setBuffer:effectiveMixed offset:0 atIndex:0];
            [enc setBuffer:gVerifyScratch.convWBuf offset:0 atIndex:1];
            [enc setBuffer:gdnState.conv offset:0 atIndex:2];
            [enc setBuffer:gVerifyScratch.convOut offset:0 atIndex:3];
            [enc setBytes:&tokens length:sizeof(tokens) atIndex:4];
            [enc setBytes:&convDim length:sizeof(convDim) atIndex:5];
            [enc setBytes:&convKernel length:sizeof(convKernel) atIndex:6];
            [enc setBuffer:gVerifyScratch.parentsBuf offset:0 atIndex:7];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)convDim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 2. QK Normalization
            [enc setComputePipelineState:psoGDNQKNormWideM];
            [enc setBuffer:gVerifyScratch.convOut offset:0 atIndex:0];
            [enc setBuffer:gVerifyScratch.qNorm offset:0 atIndex:1];
            [enc setBuffer:gVerifyScratch.kNorm offset:0 atIndex:2];
            [enc setBytes:&tokens length:sizeof(tokens) atIndex:3];
            [enc setBytes:&convDim length:sizeof(convDim) atIndex:4];
            [enc setBytes:&nK length:sizeof(nK) atIndex:5];
            [enc setBytes:&kHd length:sizeof(kHd) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nK, (NSUInteger)tokens, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)qThreads, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 3. Recurrent Delta Update
            [enc setComputePipelineState:psoGDNRecurrentWideM];
            [enc setBuffer:gVerifyScratch.convOut offset:0 atIndex:0];
            [enc setBuffer:gVerifyScratch.qNorm offset:0 atIndex:1];
            [enc setBuffer:gVerifyScratch.kNorm offset:0 atIndex:2];
            [enc setBuffer:gVerifyScratch.zBuf offset:0 atIndex:3];
            [enc setBuffer:gVerifyScratch.bBuf offset:0 atIndex:4];
            [enc setBuffer:gVerifyScratch.aBuf offset:0 atIndex:5];
            [enc setBuffer:gVerifyScratch.aLogBuf offset:0 atIndex:6];
            [enc setBuffer:gVerifyScratch.dtBiasBuf offset:0 atIndex:7];
            [enc setBuffer:gVerifyScratch.normBuf offset:0 atIndex:8];
            [enc setBuffer:gdnState.recurrent offset:0 atIndex:9];
            [enc setBuffer:gVerifyScratch.coreOutBuf offset:0 atIndex:10];
            [enc setBytes:&tokens length:sizeof(tokens) atIndex:11];
            [enc setBytes:&convDim length:sizeof(convDim) atIndex:12];
            [enc setBytes:&nK length:sizeof(nK) atIndex:13];
            [enc setBytes:&nV length:sizeof(nV) atIndex:14];
            [enc setBytes:&kHd length:sizeof(kHd) atIndex:15];
            [enc setBytes:&vHd length:sizeof(vHd) atIndex:16];
            [enc setBytes:&eps length:sizeof(eps) atIndex:17];
            [enc setBuffer:gVerifyScratch.parentsBuf offset:0 atIndex:18];
            [enc setBuffer:gVerifyScratch.treeStateBuf offset:0 atIndex:19];
            [enc setBytes:&parentMask length:sizeof(parentMask) atIndex:20];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nV, 1, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)vThreads, 1, 1)];

            // 4. Tail-causal SDPA (dispatches concurrently with GDN Recurrent - zero data hazard)
            [enc setComputePipelineState:psoSDPATailCausalTile];
            [enc setBuffer:gVerifyScratch.qBuf offset:0 atIndex:0];
            [enc setBuffer:gVerifyScratch.kBuf offset:0 atIndex:1];
            [enc setBuffer:gVerifyScratch.vBuf offset:0 atIndex:2];
            [enc setBuffer:gVerifyScratch.outBuf offset:0 atIndex:3];
            [enc setBuffer:gVerifyScratch.lseBuf offset:0 atIndex:4];
            [enc setBytes:&constants length:sizeof(constants) atIndex:5];
            [enc setThreadgroupMemoryLength:sizeof(SDPANAXThreadgroupStorage) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)sdpaM, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return 0;

        // Copy readbacks into caller-provided destination buffers
        if (gemm_out && q4_wid >= 0) {
            memcpy(gemm_out, [gVerifyScratch.gemmY contents], gemmYBytes);
        }
        if (gdn_core_out) {
            memcpy(gdn_core_out, [gVerifyScratch.coreOutBuf contents], (size_t)tokens * valueDim * sizeof(float));
        }
        if (sdpa_out) {
            memcpy(sdpa_out, [gVerifyScratch.outBuf contents], qBytes);
        }
        if (lse_out) {
            memcpy(lse_out, [gVerifyScratch.lseBuf contents], lseBytes);
        }
        return useICB ? 2 : 1;
    }
}
