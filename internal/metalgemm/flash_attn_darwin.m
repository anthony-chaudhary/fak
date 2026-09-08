//go:build darwin && arm64 && cgo

#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;
int mg_init(void);

typedef struct {
    uint32_t q_tokens;
    uint32_t kv_tokens;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t block_r;
    uint32_t block_c;
    float scale;
    int32_t causal;
    int32_t sliding_window;
    int32_t has_lse;
} FlashAttnParams;

static id<MTLComputePipelineState> gFlashAttnPSO = nil;
static dispatch_once_t gFlashAttnOnce;

static NSString *kFlashAttn2MSL = @R"MSL(
#include <metal_stdlib>
using namespace metal;

struct FlashAttnParams {
    uint q_tokens;
    uint kv_tokens;
    uint num_heads;
    uint num_kv_heads;
    uint head_dim;
    uint block_r;
    uint block_c;
    float scale;
    int causal;
    int sliding_window;
    int has_lse;
};

// flash_attn_2_fwd: Metal 4 tiled FlashAttention-2 fused SDPA kernel with online softmax.
// - Threadgroup handles query block of size block_r (e.g. 8) for one query head h (and batch b).
// - Each SIMDgroup (32 threads) within the threadgroup owns exactly 1 query row t = t_start + simd_group_id.
// - Each lane owns dim_chunk indices (simd_lane + i*32), caching Q and accumulator in registers.
// - Key and Value tiles of size block_c are cooperatively loaded from DRAM into threadgroup memory.
// - Online softmax tracks running row-max m and row-sum l, rescaling accumulator acc by alpha = exp(m_old - m_new).
// - Intermediate DRAM allocation is strictly 0 (O(1) memory), eliminating O(N^2) buffer allocations.
kernel void flash_attn_2_fwd(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* Out [[buffer(3)]],
    device float* LSE [[buffer(4)]],
    constant FlashAttnParams& p [[buffer(5)]],
    threadgroup float* tg_mem [[threadgroup(0)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint q_tile = tg_pos.x;
    uint h = tg_pos.y;
    uint b = tg_pos.z;
    uint thread_idx = tid.x;
    uint total_threads = tg_size.x;

    if (simd_group_id >= p.block_r) return;

    uint t_start = q_tile * p.block_r;
    uint t = t_start + simd_group_id;

    uint grp = p.num_heads / p.num_kv_heads;
    if (grp < 1) grp = 1;
    uint kvh = h / grp;
    if (kvh >= p.num_kv_heads) kvh = p.num_kv_heads - 1;

    uint q_batch_stride = p.q_tokens * p.num_heads * p.head_dim;
    uint kv_batch_stride = p.kv_tokens * p.num_kv_heads * p.head_dim;

    device const float* Q_b = Q + (long)b * q_batch_stride;
    device const float* K_b = K + (long)b * kv_batch_stride;
    device const float* V_b = V + (long)b * kv_batch_stride;
    device float* Out_b = Out + (long)b * q_batch_stride;

    threadgroup float* tg_K = tg_mem;
    threadgroup float* tg_V = tg_mem + (p.block_c * p.head_dim);

    // Load query row t into registers (hd <= 256 => at most 8 floats per lane)
    uint q_row_stride = p.num_heads * p.head_dim;
    long q_offset = (long)t * q_row_stride + (long)h * p.head_dim;
    float q_reg[8];
    uint nd = (p.head_dim + 31) / 32;
    for (uint i = 0; i < 8; ++i) {
        uint dim = simd_lane + i * 32;
        if (i < nd && dim < p.head_dim && t < p.q_tokens) {
            q_reg[i] = Q_b[q_offset + dim];
        } else {
            q_reg[i] = 0.0f;
        }
    }

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    uint num_kv_blocks = (p.kv_tokens + p.block_c - 1) / p.block_c;
    uint global_q_max = (p.kv_tokens >= p.q_tokens) ? 
        ((p.kv_tokens - p.q_tokens) + min(t_start + p.block_r - 1, p.q_tokens - 1)) : 
        min(t_start + p.block_r - 1, p.q_tokens - 1);

    for (uint j = 0; j < num_kv_blocks; ++j) {
        uint k_start = j * p.block_c;
        if (p.causal && k_start > global_q_max) {
            break;
        }

        uint k_end = min(k_start + p.block_c, p.kv_tokens);
        uint k_len = k_end - k_start;
        uint total_load_elems = k_len * p.head_dim;
        uint kv_row_stride = p.num_kv_heads * p.head_dim;

        // Cooperative load of K and V into threadgroup memory
        for (uint idx = thread_idx; idx < total_load_elems; idx += total_threads) {
            uint row = idx / p.head_dim;
            uint col = idx % p.head_dim;
            uint global_k = k_start + row;
            long kv_offset = (long)global_k * kv_row_stride + (long)kvh * p.head_dim + col;
            tg_K[row * p.head_dim + col] = K_b[kv_offset];
            tg_V[row * p.head_dim + col] = V_b[kv_offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (t < p.q_tokens) {
            uint global_q = (p.kv_tokens >= p.q_tokens) ? ((p.kv_tokens - p.q_tokens) + t) : t;

            for (uint k = 0; k < k_len; ++k) {
                uint global_k = k_start + k;
                if (p.causal && global_k > global_q) {
                    break;
                }
                if (p.sliding_window > 0 && (int)global_k + p.sliding_window <= (int)global_q) {
                    continue;
                }

                float partial = 0.0f;
                for (uint i = 0; i < 8; ++i) {
                    uint dim = simd_lane + i * 32;
                    if (i < nd && dim < p.head_dim) {
                        partial += q_reg[i] * tg_K[k * p.head_dim + dim];
                    }
                }
                float score = simd_sum(partial) * p.scale;

                float m_new = max(m_prev, score);
                float alpha = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_new);
                float p_val = exp(score - m_new);
                l_prev = alpha * l_prev + p_val;

                for (uint i = 0; i < 8; ++i) {
                    uint dim = simd_lane + i * 32;
                    if (i < nd && dim < p.head_dim) {
                        float v_val = tg_V[k * p.head_dim + dim];
                        acc[i] = alpha * acc[i] + p_val * v_val;
                    }
                }
                m_prev = m_new;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t < p.q_tokens) {
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        for (uint i = 0; i < 8; ++i) {
            uint dim = simd_lane + i * 32;
            if (i < nd && dim < p.head_dim) {
                Out_b[q_offset + dim] = acc[i] * inv_l;
            }
        }
        if (p.has_lse && simd_lane == 0) {
            device float* LSE_b = LSE + (long)b * (p.q_tokens * p.num_heads);
            LSE_b[(long)t * p.num_heads + h] = (l_prev > 0.0f) ? (m_prev + log(l_prev)) : -INFINITY;
        }
    }
}
)MSL";

static int mg_flash_attn_init_pipeline(void) {
    dispatch_once(&gFlashAttnOnce, ^{
        if (!mg_init() || gDev == nil) return;
        NSError *error = nil;
        id<MTLLibrary> library = [gDev newLibraryWithSource:kFlashAttn2MSL options:nil error:&error];
        if (library == nil) {
            NSLog(@"flash_attn: library compilation failed: %@", error);
            return;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"flash_attn_2_fwd"];
        if (function == nil) {
            NSLog(@"flash_attn: function flash_attn_2_fwd not found in library");
            return;
        }
        gFlashAttnPSO = [gDev newComputePipelineStateWithFunction:function error:&error];
        if (gFlashAttnPSO == nil) {
            NSLog(@"flash_attn: pipeline creation failed: %@", error);
            return;
        }
    });
    return (gFlashAttnPSO != nil) ? 1 : 0;
}

int mg_flash_attn_available(void) {
    return mg_flash_attn_init_pipeline();
}

int mg_flash_attn_execute(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    float* lse,
    int batch,
    int q_tokens,
    int kv_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int causal,
    int sliding_window
) {
    if (q == NULL || k == NULL || v == NULL || out == NULL) return -1;
    if (batch <= 0 || q_tokens <= 0 || kv_tokens <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return -2;
    if (num_heads % num_kv_heads != 0) return -3;
    if (head_dim > 256) return -4;
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf((float)head_dim);
    }

    if (!mg_flash_attn_init_pipeline()) return -5;

    uint32_t block_r = 8;
    uint32_t block_c = (head_dim > 128) ? 16 : 32;

    FlashAttnParams params;
    params.q_tokens = (uint32_t)q_tokens;
    params.kv_tokens = (uint32_t)kv_tokens;
    params.num_heads = (uint32_t)num_heads;
    params.num_kv_heads = (uint32_t)num_kv_heads;
    params.head_dim = (uint32_t)head_dim;
    params.block_r = block_r;
    params.block_c = block_c;
    params.scale = scale;
    params.causal = causal ? 1 : 0;
    params.sliding_window = (sliding_window > 0) ? sliding_window : 0;
    params.has_lse = (lse != NULL) ? 1 : 0;

    size_t q_bytes = (size_t)batch * q_tokens * num_heads * head_dim * sizeof(float);
    size_t k_bytes = (size_t)batch * kv_tokens * num_kv_heads * head_dim * sizeof(float);
    size_t v_bytes = (size_t)batch * kv_tokens * num_kv_heads * head_dim * sizeof(float);
    size_t out_bytes = (size_t)batch * q_tokens * num_heads * head_dim * sizeof(float);
    size_t lse_bytes = (size_t)batch * q_tokens * num_heads * sizeof(float);

    @autoreleasepool {
        id<MTLBuffer> bufQ = [gDev newBufferWithBytes:q length:q_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [gDev newBufferWithBytes:k length:k_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [gDev newBufferWithBytes:v length:v_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [gDev newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufLSE = (lse != NULL) ?
            [gDev newBufferWithLength:lse_bytes options:MTLResourceStorageModeShared] :
            [gDev newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];

        if (!bufQ || !bufK || !bufV || !bufOut || !bufLSE) {
            return -6;
        }

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        if (cb == nil) return -7;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) return -8;

        [enc setComputePipelineState:gFlashAttnPSO];
        [enc setBuffer:bufQ offset:0 atIndex:0];
        [enc setBuffer:bufK offset:0 atIndex:1];
        [enc setBuffer:bufV offset:0 atIndex:2];
        [enc setBuffer:bufOut offset:0 atIndex:3];
        [enc setBuffer:bufLSE offset:0 atIndex:4];
        [enc setBytes:&params length:sizeof(params) atIndex:5];

        NSUInteger tg_mem_bytes = (NSUInteger)(2 * block_c * head_dim * sizeof(float));
        [enc setThreadgroupMemoryLength:tg_mem_bytes atIndex:0];

        MTLSize threadsPerTG = MTLSizeMake(block_r * 32, 1, 1);
        MTLSize threadgroups = MTLSizeMake((q_tokens + block_r - 1) / block_r, num_heads, batch);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerTG];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError) {
            NSLog(@"flash_attn: command buffer failed: %@", cb.error);
            return -9;
        }

        memcpy(out, [bufOut contents], out_bytes);
        if (lse != NULL) {
            memcpy(lse, [bufLSE contents], lse_bytes);
        }
    }
    return 0;
}
