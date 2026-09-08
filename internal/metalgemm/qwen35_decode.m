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

// gdn_conv_wide_m: Advances the 1D convolution window across M sequential tokens.
// Each channel runs across token 0..tokens-1 in order, maintaining exact state.
kernel void gdn_conv_wide_m(
    device const float *mixed [[buffer(0)]],
    device const float *convW [[buffer(1)]],
    device float *convState   [[buffer(2)]],
    device float *convOut     [[buffer(3)]],
    constant int& tokens      [[buffer(4)]],
    constant int& convDim     [[buffer(5)]],
    constant int& kernelSize  [[buffer(6)]],
    uint channel [[thread_position_in_grid]]
) {
    if (channel >= (uint)convDim) return;
    float window[7];
    for (int j = 0; j < kernelSize - 1; ++j) {
        window[j] = convState[(long)j * convDim + channel];
    }
    for (int token = 0; token < tokens; ++token) {
        float acc = 0.0f;
        int wb = (int)channel * kernelSize;
        for (int j = 0; j < kernelSize - 1; ++j) acc += convW[wb + j] * window[j];
        float current = mixed[(long)token * convDim + channel];
        acc += convW[wb + kernelSize - 1] * current;
        convOut[(long)token * convDim + channel] = gdn_silu(acc);
        for (int j = 0; j < kernelSize - 2; ++j) window[j] = window[j + 1];
        if (kernelSize > 1) window[kernelSize - 2] = current;
    }
    for (int j = 0; j < kernelSize - 1; ++j) {
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

// gdn_recurrent_wide_m: Evaluates recurrent state update across M tokens in registers.
// The recurrent state is maintained across token 0..tokens-1 sequentially, ensuring
// bit-exact agreement with serial execution.
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
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]
) {
    if (head >= (uint)nV) return;
    int repeat = nV / nK;
    int keyHead = (int)head / repeat;
    int keyDim = nK * kHd;
    float localState[128];
    if (lane < (uint)vHd) {
        for (int i = 0; i < 128; ++i) {
            if (i < kHd) localState[i] = state[((long)head * kHd + i) * vHd + lane];
        }
    }
    threadgroup float squares[256];
    for (int token = 0; token < tokens; ++token) {
        device const float *qRow = qNorm + ((long)token * nK + keyHead) * kHd;
        device const float *kRow = kNorm + ((long)token * nK + keyHead) * kHd;
        float readout = 0.0f;
        if (lane < (uint)vHd) {
            float beta = 1.0f / (1.0f + exp(-b[(long)token * nV + head]));
            float decay = exp(-exp(aLog[head]) * gdn_softplus(a[(long)token * nV + head] + dtBias[head]));
            float kvmem = 0.0f;
            for (int i = 0; i < 128; ++i) {
                if (i < kHd) {
                    float value = localState[i] * decay;
                    localState[i] = value;
                    kvmem += value * kRow[i];
                }
            }
            long valueIndex = (long)token * convDim + 2L * keyDim + (long)head * vHd + lane;
            float delta = (convOut[valueIndex] - kvmem) * beta;
            for (int i = 0; i < 128; ++i) {
                if (i < kHd) {
                    float value = localState[i] + kRow[i] * delta;
                    localState[i] = value;
                    readout += value * qRow[i];
                }
            }
        }
        squares[lane] = lane < (uint)vHd ? readout * readout : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = lanes >> 1; offset > 0; offset >>= 1) {
            if (lane < offset) squares[lane] += squares[lane + offset];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane < (uint)vHd) {
            float inv = 1.0f / sqrt(squares[0] / (float)vHd + eps);
            long vd = (long)head * vHd + lane;
            core[(long)token * nV * vHd + vd] = norm[lane] * readout * inv * gdn_silu(z[(long)token * nV * vHd + vd]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < (uint)vHd) {
        for (int i = 0; i < 128; ++i) {
            if (i < kHd) state[((long)head * kHd + i) * vHd + lane] = localState[i];
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
    uint draft_len;    // Number of speculative draft tokens (e.g. 2..4)
    uint M;            // Total query rows: M = gqa_factor * draft_len
    uint head_dim;     // Head dimension D (e.g. 64 or 128)
    uint prefix_len;   // Sequence prefix length
    uint total_kv;     // Total KV tokens = prefix_len + draft_len
    float scale;       // Attention scale factor 1.0f / sqrt(head_dim)
    uint tile_n;       // KV tile size Bc (e.g. 32)
    uint order;        // 0: HeadMajor, 1: TokenMajor
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

    uint max_k_for_row = nax_max_causal_key(m, c);
    uint num_tiles = (c.total_kv + c.tile_n - 1) / c.tile_n;

    for (uint tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint j_start = tile_idx * c.tile_n;
        uint j_end = min(j_start + c.tile_n, c.total_kv);
        uint tile_len = j_end - j_start;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint total_elements = tile_len * c.head_dim;
        for (uint idx = tid; idx < total_elements; idx += tg_size) {
            uint k_row = idx / c.head_dim;
            uint k_col = idx % c.head_dim;
            uint global_k_pos = j_start + k_row;
            tg_mem.k_tile[k_row * c.head_dim + k_col] = K[global_k_pos * c.head_dim + k_col];
            tg_mem.v_transposed[k_col * c.tile_n + k_row] = V[global_k_pos * c.head_dim + k_col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (j_start > max_k_for_row) {
            continue;
        }

        for (uint k = 0; k < tile_len; ++k) {
            uint global_key_pos = j_start + k;
            float score = -INFINITY;
            if (global_key_pos <= max_k_for_row) {
                float partial_qk = 0.0f;
                for (uint d = simd_lane; d < c.head_dim; d += 32) {
                    partial_qk += Q[m * c.head_dim + d] * tg_mem.k_tile[k * c.head_dim + d];
                }
                score = simd_sum(partial_qk) * c.scale;
            }

            float m_new = max(m_prev, score);
            float alpha = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_new);
            float p = (score == -INFINITY) ? 0.0f : exp(score - m_new);
            l_prev = alpha * l_prev + p;

            uint d_idx = 0;
            for (uint d = simd_lane; d < c.head_dim; d += 32) {
                float v_val = tg_mem.v_transposed[d * c.tile_n + k];
                acc[d_idx] = alpha * acc[d_idx] + p * v_val;
                d_idx++;
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
        psoGDNConvWideM = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gdn_conv_wide_m"] error:&err];
        psoGDNQKNormWideM = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gdn_qk_norm_wide_m"] error:&err];
        psoGDNRecurrentWideM = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gdn_recurrent_wide_m"] error:&err];
        psoSDPATailCausalTile = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"sdpa_nax_tail_causal_tile"] error:&err];
        if (!psoGDNConvWideM || !psoGDNQKNormWideM || !psoGDNRecurrentWideM || !psoSDPATailCausalTile) {
            NSLog(@"qwen35_decode: pipeline creation failed: %@", err);
            return 0;
        }
        gQwen35PipelinesReady = 1;
        return 1;
    }
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
    int tokens, int nK, int nV, int kHd, int vHd, int convKernel, float eps
) {
    int keyDim = nK * kHd;
    int valueDim = nV * vHd;
    int convDim = 2 * keyDim + valueDim;

    id<MTLBuffer> convOut = [gDev newBufferWithLength:(size_t)tokens * convDim * sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> qNorm = [gDev newBufferWithLength:(size_t)tokens * keyDim * sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> kNorm = [gDev newBufferWithLength:(size_t)tokens * keyDim * sizeof(float) options:MTLResourceStorageModePrivate];
    if (!convOut || !qNorm || !kNorm) return 0;

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
    int tokens, int nK, int nV, int kHd, int vHd, int convKernel, float eps
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

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        if (!cb) return 0;

        if (!mg_qwen35_gdn_encode_into(cb, state.conv, state.recurrent, mixedBuf, zBuf, bBuf, aBuf,
                                      convWBuf, aLogBuf, dtBiasBuf, normBuf, coreOutBuf,
                                      tokens, nK, nV, kHd, vHd, convKernel, eps)) {
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
                                        core_out, 1, nK, nV, kHd, vHd, convKernel, eps);
}

// ==============================================================================
// Tail-Causal SDPA Dispatch
// ==============================================================================

int mg_sdpa_nax_tail_causal_tile_run(
    const float* Q, const float* K, const float* V,
    float* Out, float* LSE,
    int gqaFactor, int draftLen, int M, int headDim,
    int prefixLen, int totalKV, float scale, int tileN, int order
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
        constants.gqa_factor = (uint32_t)gqaFactor;
        constants.draft_len = (uint32_t)draftLen;
        constants.M = (uint32_t)M;
        constants.head_dim = (uint32_t)headDim;
        constants.prefix_len = (uint32_t)prefixLen;
        constants.total_kv = (uint32_t)totalKV;
        constants.scale = scale;
        constants.tile_n = (uint32_t)tileN;
        constants.order = (uint32_t)order;

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
    const float* Q, const float* K, const float* V,
    float* sdpa_out, float* lse_out,
    int gqaFactor, int draftLen, int sdpaM, int headDim,
    int prefixLen, int totalKV, float sdpaScale, int tileN, int order
) {
    if (!qwen35_init_pipelines()) return 0;
    if (draftLen < 2 || draftLen > 8) return 0;

    Qwen35DecodeState gdnState;
    @synchronized(gDev) {
        if (gdn_handle < 0 || gdn_handle >= gQwen35StateCount || !gQwen35States[gdn_handle].inUse) {
            return 0;
        }
        gdnState = gQwen35States[gdn_handle];
    }

    int tokens = draftLen;
    int convDim = 2 * (nK * kHd) + (nV * vHd);
    int valueDim = nV * vHd;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        if (!cb) return 0;

        // 1. Stage inputs
        // A. Q4_K GEMM
        id<MTLBuffer> gemmX = nil;
        id<MTLBuffer> gemmY = nil;
        size_t gemmYBytes = 0;
        if (q4_wid >= 0 && draft_input && gemm_out) {
            size_t inBytes = (size_t)tokens * convDim * sizeof(float);
            gemmYBytes = (size_t)tokens * convDim * sizeof(float);
            gemmX = [gDev newBufferWithBytes:draft_input length:inBytes options:MTLResourceStorageModeShared];
            gemmY = [gDev newBufferWithLength:gemmYBytes options:MTLResourceStorageModeShared];
            if (!gemmX || !gemmY) return 0;
            if (!mg_q4k_gemv_wide_m_encode((__bridge void*)cb, q4_wid, (__bridge void*)gemmX, (__bridge void*)gemmY, tokens)) {
                return 0;
            }
        }

        // B. Recurrent GDN step
        id<MTLBuffer> mixedBuf = gemmY ? gemmY : [gDev newBufferWithBytes:draft_input length:(size_t)tokens * convDim * sizeof(float) options:MTLResourceStorageModeShared];
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

        if (!mg_qwen35_gdn_encode_into(cb, gdnState.conv, gdnState.recurrent, mixedBuf, zBuf, bBuf, aBuf,
                                      convWBuf, aLogBuf, dtBiasBuf, normBuf, coreOutBuf,
                                      tokens, nK, nV, kHd, vHd, convKernel, eps)) {
            return 0;
        }

        // C. Tail-causal SDPA
        size_t qBytes = (size_t)sdpaM * headDim * sizeof(float);
        size_t kvBytes = (size_t)totalKV * headDim * sizeof(float);
        size_t lseBytes = (size_t)sdpaM * sizeof(float);

        id<MTLBuffer> qBuf = [gDev newBufferWithBytes:Q length:qBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> kBuf = [gDev newBufferWithBytes:K length:kvBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> vBuf = [gDev newBufferWithBytes:V length:kvBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = [gDev newBufferWithLength:qBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> lseBuf = [gDev newBufferWithLength:lseBytes options:MTLResourceStorageModeShared];

        if (!qBuf || !kBuf || !vBuf || !outBuf || !lseBuf) return 0;

        SDPANAXConstants constants;
        constants.gqa_factor = (uint32_t)gqaFactor;
        constants.draft_len = (uint32_t)draftLen;
        constants.M = (uint32_t)sdpaM;
        constants.head_dim = (uint32_t)headDim;
        constants.prefix_len = (uint32_t)prefixLen;
        constants.total_kv = (uint32_t)totalKV;
        constants.scale = sdpaScale;
        constants.tile_n = (uint32_t)tileN;
        constants.order = (uint32_t)order;

        id<MTLComputeCommandEncoder> sdpaEnc = [cb computeCommandEncoder];
        if (!sdpaEnc) return 0;

        uint sdpa_threads = 32;
        uint sdpa_num_tgs = (uint)sdpaM;

        [sdpaEnc setComputePipelineState:psoSDPATailCausalTile];
        [sdpaEnc setBuffer:qBuf offset:0 atIndex:0];
        [sdpaEnc setBuffer:kBuf offset:0 atIndex:1];
        [sdpaEnc setBuffer:vBuf offset:0 atIndex:2];
        [sdpaEnc setBuffer:outBuf offset:0 atIndex:3];
        [sdpaEnc setBuffer:lseBuf offset:0 atIndex:4];
        [sdpaEnc setBytes:&constants length:sizeof(constants) atIndex:5];
        [sdpaEnc setThreadgroupMemoryLength:sizeof(SDPANAXThreadgroupStorage) atIndex:0];
        [sdpaEnc dispatchThreadgroups:MTLSizeMake((NSUInteger)sdpa_num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)sdpa_threads, 1, 1)];
        [sdpaEnc endEncoding];

        // Commit the single command buffer and wait once
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return 0;

        // Copy readbacks
        if (gemm_out && gemmY) {
            memcpy(gemm_out, [gemmY contents], gemmYBytes);
        }
        if (gdn_core_out) {
            memcpy(gdn_core_out, [coreOutBuf contents], (size_t)tokens * valueDim * sizeof(float));
        }
        if (sdpa_out) {
            memcpy(sdpa_out, [outBuf contents], qBytes);
        }
        if (lse_out) {
            memcpy(lse_out, [lseBuf contents], lseBytes);
        }
        return 1;
    }
}
