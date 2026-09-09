//go:build darwin && arm64 && cgo

#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>
#include <stdint.h>
#include <string.h>

extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;
int mg_init(void);

typedef struct {
    uintptr_t command_buffer;
    int committed;
    int completed_wait;
    int encoders;
    int state_h2d_transfers;
    int state_d2h_transfers;
    int host_recurrence_steps;
    int owned_buffers;
    int private_state_buffers;
    int panel_h2d_transfers;
    int output_d2h_transfers;
    uint64_t state_bytes;
} mg_gdn_event;

typedef struct {
    CFTypeRef conv;
    CFTypeRef recurrent;
    uint64_t conv_handle;
    uint64_t recurrent_handle;
    int nK, nV, kHd, vHd, convKernel;
} MGGDNOwner;

// Qwen3.8 has 48 linear-attention layers, so its declared B=8 envelope owns
// 384 simultaneous lane-local states. Keep fixed storage and leave headroom for
// diagnostics or overlapping construction without reallocating an active table.
enum { MG_GDN_MAX_OWNERS = 512 };
static MGGDNOwner gGDNOwners[MG_GDN_MAX_OWNERS];
static uint64_t gGDNNextHandle = 1;
static id<MTLComputePipelineState> gGDNConvPSO;
static id<MTLComputePipelineState> gGDNQKNormPSO;
static id<MTLComputePipelineState> gGDNRecurrentPSO;
static id<MTLComputePipelineState> gGDNRecurrentPackedPSO;
static BOOL gGDNPipelineAttempted;
static int gGDNForceBaseline = 0;

void mg_gdn_set_force_baseline(int force) {
    gGDNForceBaseline = force;
}

static NSString *gGDNSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;

inline float gdn_silu(float x) { return x / (1.0f + exp(-x)); }
inline float gdn_softplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

// One lane owns one convolution channel and advances its K-1 window in token order.
kernel void gdn_conv_panel(device const float *mixed [[buffer(0)]],
                           device const float *convW [[buffer(1)]],
                           device float *convState [[buffer(2)]],
                           device float *convOut [[buffer(3)]],
                           constant int& tokens [[buffer(4)]],
                           constant int& convDim [[buffer(5)]],
                           constant int& kernelSize [[buffer(6)]],
                           uint channel [[thread_position_in_grid]]) {
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

// One threadgroup owns one (token,key-head) normalization. The reduction is the only
// order change from the scalar oracle and is held to the issue's declared tolerance.
kernel void gdn_qk_norm_panel(device const float *convOut [[buffer(0)]],
                              device float *qNorm [[buffer(1)]],
                              device float *kNorm [[buffer(2)]],
                              constant int& tokens [[buffer(3)]],
                              constant int& convDim [[buffer(4)]],
                              constant int& nK [[buffer(5)]],
                              constant int& kHd [[buffer(6)]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]],
                              uint3 groupSize [[threads_per_threadgroup]]) {
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

// One persistent threadgroup per value head owns its recurrent matrix. Each lane owns
// one value-dimension column; only the RMS reduction crosses lanes. Tokens remain serial.
kernel void gdn_recurrent_panel(device const float *convOut [[buffer(0)]],
                                device const float *qNorm [[buffer(1)]],
                                device const float *kNorm [[buffer(2)]],
                                device const float *z [[buffer(3)]],
                                device const float *b [[buffer(4)]],
                                device const float *a [[buffer(5)]],
                                device const float *aLog [[buffer(6)]],
                                device const float *dtBias [[buffer(7)]],
                                device const float *norm [[buffer(8)]],
                                device float *state [[buffer(9)]],
                                device float *core [[buffer(10)]],
                                constant int& tokens [[buffer(11)]],
                                constant int& convDim [[buffer(12)]],
                                constant int& nK [[buffer(13)]],
                                constant int& nV [[buffer(14)]],
                                constant int& kHd [[buffer(15)]],
                                constant int& vHd [[buffer(16)]],
                                constant float& eps [[buffer(17)]],
                                uint head [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]],
                                uint lanes [[threads_per_threadgroup]]) {
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

// One threadgroup per value head. Threads are partitioned into SIMDgroups of 32 lanes.
// Each SIMDgroup owns 8 rows of value dimension; 4 lanes cooperatively process 128 elements of D_k per row.
kernel void gdn_recurrent_packed_8row(device const float *convOut [[buffer(0)]],
                                      device const float *qNorm [[buffer(1)]],
                                      device const float *kNorm [[buffer(2)]],
                                      device const float *z [[buffer(3)]],
                                      device const float *b [[buffer(4)]],
                                      device const float *a [[buffer(5)]],
                                      device const float *aLog [[buffer(6)]],
                                      device const float *dtBias [[buffer(7)]],
                                      device const float *norm [[buffer(8)]],
                                      device float *state [[buffer(9)]],
                                      device float *core [[buffer(10)]],
                                      constant int& tokens [[buffer(11)]],
                                      constant int& convDim [[buffer(12)]],
                                      constant int& nK [[buffer(13)]],
                                      constant int& nV [[buffer(14)]],
                                      constant int& kHd [[buffer(15)]],
                                      constant int& vHd [[buffer(16)]],
                                      constant float& eps [[buffer(17)]],
                                      uint head [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint lanes [[threads_per_threadgroup]]) {
    if (head >= (uint)nV) return;
    int repeat = nV / nK;
    int keyHead = (int)head / repeat;
    int keyDim = nK * kHd;

    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    uint row_id = lane_id / 4;
    uint sub_lane = lane_id % 4;
    uint v_idx = simd_id * 8 + row_id;
    uint k_start = sub_lane * 32;

    float4 st[8];
    if (v_idx < (uint)vHd) {
        for (int j = 0; j < 8; ++j) {
            float4 s;
            for (int c = 0; c < 4; ++c) {
                s[c] = state[((long)head * kHd + (sub_lane * 32 + j * 4 + c)) * vHd + v_idx];
            }
            st[j] = s;
        }
    } else {
        for (int j = 0; j < 8; ++j) {
            st[j] = float4(0.0f);
        }
    }

    threadgroup float tg_sq[32];

    for (int token = 0; token < tokens; ++token) {
        device const float *qRow = qNorm + ((long)token * nK + keyHead) * kHd;
        device const float *kRow = kNorm + ((long)token * nK + keyHead) * kHd;

        float4 k_vec[8];
        float4 q_vec[8];
        device const float4 *k_ptr = (device const float4 *)(kRow + k_start);
        device const float4 *q_ptr = (device const float4 *)(qRow + k_start);
        for (int j = 0; j < 8; ++j) {
            k_vec[j] = k_ptr[j];
            q_vec[j] = q_ptr[j];
        }

        float beta = 1.0f / (1.0f + exp(-b[(long)token * nV + head]));
        float decay = exp(-exp(aLog[head]) * gdn_softplus(a[(long)token * nV + head] + dtBias[head]));

        float v_row = 0.0f;
        if (v_idx < (uint)vHd) {
            long valueIndex = (long)token * convDim + 2L * keyDim + (long)head * vHd + v_idx;
            v_row = convOut[valueIndex];
        }

        float part[8];
        for (int j = 0; j < 8; ++j) {
            st[j] *= decay;
            part[j] = dot(st[j], k_vec[j]);
        }
        float kv_mem = ((part[0] + part[1]) + (part[2] + part[3])) +
                       ((part[4] + part[5]) + (part[6] + part[7]));
        kv_mem += simd_shuffle_xor(kv_mem, 1);
        kv_mem += simd_shuffle_xor(kv_mem, 2);

        float delta = (v_row - kv_mem) * beta;

        float r_part[8];
        for (int j = 0; j < 8; ++j) {
            st[j] += delta * k_vec[j];
            r_part[j] = dot(st[j], q_vec[j]);
        }
        float readout = ((r_part[0] + r_part[1]) + (r_part[2] + r_part[3])) +
                        ((r_part[4] + r_part[5]) + (r_part[6] + r_part[7]));
        readout += simd_shuffle_xor(readout, 1);
        readout += simd_shuffle_xor(readout, 2);

        float sq = (sub_lane == 0 && v_idx < (uint)vHd) ? (readout * readout) : 0.0f;
        float simd_sq = simd_sum(sq);
        if (lane_id == 0) {
            tg_sq[simd_id] = simd_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float total_sq = 0.0f;
        uint num_simds = lanes / 32;
        for (uint s = 0; s < num_simds; ++s) {
            total_sq += tg_sq[s];
        }
        float inv = rsqrt(total_sq / (float)vHd + eps);

        if (sub_lane == 0 && v_idx < (uint)vHd) {
            long vd = (long)head * vHd + v_idx;
            core[(long)token * nV * vHd + vd] = norm[v_idx] * readout * inv * gdn_silu(z[(long)token * nV * vHd + vd]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (v_idx < (uint)vHd) {
        for (int j = 0; j < 8; ++j) {
            for (int c = 0; c < 4; ++c) {
                state[((long)head * kHd + (sub_lane * 32 + j * 4 + c)) * vHd + v_idx] = st[j][c];
            }
        }
    }
}
)MSL";

static int mg_gdn_pipelines(void) {
    @synchronized(gDev) {
        if (gGDNConvPSO != nil && gGDNQKNormPSO != nil && gGDNRecurrentPSO != nil && gGDNRecurrentPackedPSO != nil) return 1;
        if (gGDNPipelineAttempted) return 0;
        gGDNPipelineAttempted = YES;
        NSError *error = nil;
        id<MTLLibrary> library = [gDev newLibraryWithSource:gGDNSrc options:nil error:&error];
        if (library == nil) {
            NSLog(@"mg_gdn: MSL compile failed: %@", error);
            return 0;
        }
        gGDNConvPSO = [gDev newComputePipelineStateWithFunction:[library newFunctionWithName:@"gdn_conv_panel"] error:&error];
        gGDNQKNormPSO = [gDev newComputePipelineStateWithFunction:[library newFunctionWithName:@"gdn_qk_norm_panel"] error:&error];
        gGDNRecurrentPSO = [gDev newComputePipelineStateWithFunction:[library newFunctionWithName:@"gdn_recurrent_panel"] error:&error];
        gGDNRecurrentPackedPSO = [gDev newComputePipelineStateWithFunction:[library newFunctionWithName:@"gdn_recurrent_packed_8row"] error:&error];
        if (gGDNConvPSO == nil || gGDNQKNormPSO == nil || gGDNRecurrentPSO == nil || gGDNRecurrentPackedPSO == nil) {
            NSLog(@"mg_gdn: pipeline creation failed: %@", error);
            return 0;
        }
        return 1;
    }
}

static int mg_gdn_threads(int width) {
    int threads = 1;
    while (threads < width) threads <<= 1;
    return threads;
}

static id<MTLBuffer> mg_gdn_shared(const float *src, size_t count) {
    return [gDev newBufferWithBytes:src length:count * sizeof(float) options:MTLResourceStorageModeShared];
}

static int mg_gdn_zero(id<MTLBuffer> conv, id<MTLBuffer> recurrent) {
    @autoreleasepool {
        id<MTLCommandBuffer> command = [gQueue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        [encoder fillBuffer:conv range:NSMakeRange(0, conv.length) value:0];
        [encoder fillBuffer:recurrent range:NSMakeRange(0, recurrent.length) value:0];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        return command.status == MTLCommandBufferStatusCompleted;
    }
}

int mg_gdn_state_new(int nK, int nV, int kHd, int vHd, int convKernel,
                     uint64_t *convHandle, uint64_t *recurrentHandle) {
    if (!mg_init() || !mg_gdn_pipelines() || convHandle == NULL || recurrentHandle == NULL) return -1;
    *convHandle = 0;
    *recurrentHandle = 0;
    int keyDim = nK * kHd, valueDim = nV * vHd, convDim = 2 * keyDim + valueDim;
    if (nK < 1 || nV < 1 || nV % nK || kHd < 1 || kHd > 128 || vHd < 1 || vHd > 256 || convKernel < 1 || convKernel > 8) return -1;
    size_t convBytes = (size_t)(convKernel - 1) * convDim * sizeof(float);
    size_t recurrentBytes = (size_t)nV * kHd * vHd * sizeof(float);
    id<MTLBuffer> conv = [gDev newBufferWithLength:MAX((size_t)4, convBytes) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> recurrent = [gDev newBufferWithLength:recurrentBytes options:MTLResourceStorageModePrivate];
    if (conv == nil || recurrent == nil) return -1;
    if (!mg_gdn_zero(conv, recurrent)) return -1;
    @synchronized(gDev) {
        for (int owner = 0; owner < MG_GDN_MAX_OWNERS; ++owner) {
            if (gGDNOwners[owner].conv != NULL) continue;
            MGGDNOwner *slot = &gGDNOwners[owner];
            slot->conv = CFBridgingRetain(conv);
            slot->recurrent = CFBridgingRetain(recurrent);
            slot->conv_handle = gGDNNextHandle++;
            slot->recurrent_handle = gGDNNextHandle++;
            slot->nK = nK; slot->nV = nV; slot->kHd = kHd; slot->vHd = vHd; slot->convKernel = convKernel;
            *convHandle = slot->conv_handle;
            *recurrentHandle = slot->recurrent_handle;
            return owner;
        }
    }
    return -2;
}

int mg_gdn_owner_capacity(void) { return MG_GDN_MAX_OWNERS; }

static void mg_gdn_event_reset(mg_gdn_event *event) {
    if (event == NULL) return;
    memset(event, 0, sizeof(*event));
}

int mg_gdn_state_run(int owner,
                     const float *mixed, const float *z, const float *b, const float *a,
                     const float *convW, const float *aLog, const float *dtBias, const float *norm,
                     int tokens, int nK, int nV, int kHd, int vHd, int convKernel, float eps,
                     float *core, int injectPostSubmitFailure, mg_gdn_event *event) {
    mg_gdn_event_reset(event);
    @autoreleasepool {
        if (owner < 0 || owner >= MG_GDN_MAX_OWNERS || tokens < 1 || tokens > 64 ||
            mixed == NULL || z == NULL || b == NULL || a == NULL || convW == NULL || aLog == NULL ||
            dtBias == NULL || norm == NULL || core == NULL || eps <= 0) return 0;
        MGGDNOwner slot;
        @synchronized(gDev) { slot = gGDNOwners[owner]; }
        if (slot.conv == NULL || slot.recurrent == NULL || slot.nK != nK || slot.nV != nV ||
            slot.kHd != kHd || slot.vHd != vHd || slot.convKernel != convKernel) return 0;
        id<MTLBuffer> convState = (__bridge id<MTLBuffer>)slot.conv;
        id<MTLBuffer> recurrentState = (__bridge id<MTLBuffer>)slot.recurrent;
        if (event != NULL) {
            event->owned_buffers = 2;
            event->private_state_buffers = (convState.storageMode == MTLStorageModePrivate) + (recurrentState.storageMode == MTLStorageModePrivate);
            event->state_bytes = (uint64_t)convState.length + (uint64_t)recurrentState.length;
        }

        int keyDim = nK * kHd, valueDim = nV * vHd, convDim = 2 * keyDim + valueDim;
        id<MTLBuffer> mixedB = mg_gdn_shared(mixed, (size_t)tokens * convDim);
        id<MTLBuffer> zB = mg_gdn_shared(z, (size_t)tokens * valueDim);
        id<MTLBuffer> bB = mg_gdn_shared(b, (size_t)tokens * nV);
        id<MTLBuffer> aB = mg_gdn_shared(a, (size_t)tokens * nV);
        id<MTLBuffer> convWB = mg_gdn_shared(convW, (size_t)convDim * convKernel);
        id<MTLBuffer> aLogB = mg_gdn_shared(aLog, nV);
        id<MTLBuffer> dtBiasB = mg_gdn_shared(dtBias, nV);
        id<MTLBuffer> normB = mg_gdn_shared(norm, vHd);
        id<MTLBuffer> convOutB = [gDev newBufferWithLength:(size_t)tokens * convDim * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> qNormB = [gDev newBufferWithLength:(size_t)tokens * nK * kHd * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> kNormB = [gDev newBufferWithLength:(size_t)tokens * nK * kHd * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> coreB = [gDev newBufferWithLength:(size_t)tokens * valueDim * sizeof(float) options:MTLResourceStorageModeShared];
        if (mixedB == nil || zB == nil || bB == nil || aB == nil || convWB == nil || aLogB == nil ||
            dtBiasB == nil || normB == nil || convOutB == nil || qNormB == nil || kNormB == nil || coreB == nil) return 0;
        if (event != NULL) event->panel_h2d_transfers = 8;

        id<MTLCommandBuffer> command = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:gGDNConvPSO];
        [encoder setBuffer:mixedB offset:0 atIndex:0]; [encoder setBuffer:convWB offset:0 atIndex:1];
        [encoder setBuffer:convState offset:0 atIndex:2]; [encoder setBuffer:convOutB offset:0 atIndex:3];
        [encoder setBytes:&tokens length:sizeof(tokens) atIndex:4]; [encoder setBytes:&convDim length:sizeof(convDim) atIndex:5];
        [encoder setBytes:&convKernel length:sizeof(convKernel) atIndex:6];
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)convDim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        int qThreads = mg_gdn_threads(kHd);
        [encoder setComputePipelineState:gGDNQKNormPSO];
        [encoder setBuffer:convOutB offset:0 atIndex:0]; [encoder setBuffer:qNormB offset:0 atIndex:1]; [encoder setBuffer:kNormB offset:0 atIndex:2];
        [encoder setBytes:&tokens length:sizeof(tokens) atIndex:3]; [encoder setBytes:&convDim length:sizeof(convDim) atIndex:4];
        [encoder setBytes:&nK length:sizeof(nK) atIndex:5]; [encoder setBytes:&kHd length:sizeof(kHd) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)nK, (NSUInteger)tokens, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)qThreads, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        int vThreads = mg_gdn_threads(vHd);
        id<MTLComputePipelineState> recPSO = gGDNRecurrentPSO;
        int recThreads = vThreads;
        if (!gGDNForceBaseline && kHd == 128 && (vHd % 8) == 0 && gGDNRecurrentPackedPSO != nil) {
            recPSO = gGDNRecurrentPackedPSO;
            recThreads = (vHd / 8) * 32;
        }
        [encoder setComputePipelineState:recPSO];
        [encoder setBuffer:convOutB offset:0 atIndex:0]; [encoder setBuffer:qNormB offset:0 atIndex:1]; [encoder setBuffer:kNormB offset:0 atIndex:2];
        [encoder setBuffer:zB offset:0 atIndex:3]; [encoder setBuffer:bB offset:0 atIndex:4]; [encoder setBuffer:aB offset:0 atIndex:5];
        [encoder setBuffer:aLogB offset:0 atIndex:6]; [encoder setBuffer:dtBiasB offset:0 atIndex:7]; [encoder setBuffer:normB offset:0 atIndex:8];
        [encoder setBuffer:recurrentState offset:0 atIndex:9]; [encoder setBuffer:coreB offset:0 atIndex:10];
        [encoder setBytes:&tokens length:sizeof(tokens) atIndex:11]; [encoder setBytes:&convDim length:sizeof(convDim) atIndex:12];
        [encoder setBytes:&nK length:sizeof(nK) atIndex:13]; [encoder setBytes:&nV length:sizeof(nV) atIndex:14];
        [encoder setBytes:&kHd length:sizeof(kHd) atIndex:15]; [encoder setBytes:&vHd length:sizeof(vHd) atIndex:16];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:17];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)nV, 1, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)recThreads, 1, 1)];
        [encoder endEncoding];

        if (event != NULL) { event->command_buffer = (uintptr_t)(__bridge void *)command; event->encoders = 1; }
        [command commit];
        if (event != NULL) event->committed = 1;
        [command waitUntilCompleted];
        if (event != NULL) event->completed_wait = command.status == MTLCommandBufferStatusCompleted;
        if (command.status != MTLCommandBufferStatusCompleted || injectPostSubmitFailure) return -1;
        memcpy(core, coreB.contents, (size_t)tokens * valueDim * sizeof(float));
        if (event != NULL) event->output_d2h_transfers = 1;
        return 1;
    }
}

// Encode the resident GDN operation into a caller-owned projection graph. All
// projected operands are already graph-resident; only the small immutable GDN
// vectors are staged. The existing private convolution/recurrent buffers are
// mutated in place and remain the same owner handed to resident decode.
extern void *mg_graph_command_buffer(void *graph);
extern void *mg_graph_alloc_result(void *graph, int n);
extern void mg_graph_note_encoder(void *graph);

static BOOL mg_gdn_owner_compatible(MGGDNOwner a, MGGDNOwner b) {
    return a.conv != NULL && a.recurrent != NULL && b.conv != NULL && b.recurrent != NULL &&
           a.conv != b.conv && a.recurrent != b.recurrent &&
           a.nK == b.nK && a.nV == b.nV && a.kHd == b.kHd && a.vHd == b.vHd &&
           a.convKernel == b.convKernel;
}

// Save both private state buffers in the caller-owned graph command buffer.
// The Go checkpoint lease keeps both registry slots exclusively owned through
// the terminal wait and until the caller resolves the token by Restore/Close.
int mg_gdn_graph_checkpoint(void *graph, int liveOwner, int backupOwner) {
    @autoreleasepool {
        if (!graph || liveOwner < 0 || backupOwner < 0 || liveOwner >= MG_GDN_MAX_OWNERS ||
            backupOwner >= MG_GDN_MAX_OWNERS || liveOwner == backupOwner) return 0;
        MGGDNOwner live, backup;
        @synchronized(gDev) {
            live = gGDNOwners[liveOwner];
            backup = gGDNOwners[backupOwner];
        }
        if (!mg_gdn_owner_compatible(live, backup)) return 0;
        id<MTLBuffer> liveConv = (__bridge id<MTLBuffer>)live.conv;
        id<MTLBuffer> liveRecurrent = (__bridge id<MTLBuffer>)live.recurrent;
        id<MTLBuffer> backupConv = (__bridge id<MTLBuffer>)backup.conv;
        id<MTLBuffer> backupRecurrent = (__bridge id<MTLBuffer>)backup.recurrent;
        if (liveConv.length != backupConv.length || liveRecurrent.length != backupRecurrent.length) return 0;
        id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(graph);
        if (command == nil) return 0;
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        if (encoder == nil) return 0;
        [encoder copyFromBuffer:liveConv sourceOffset:0 toBuffer:backupConv destinationOffset:0 size:liveConv.length];
        [encoder copyFromBuffer:liveRecurrent sourceOffset:0 toBuffer:backupRecurrent destinationOffset:0 size:liveRecurrent.length];
        [encoder endEncoding];
        mg_graph_note_encoder(graph);
        return 1;
    }
}

// Restore is an O(1) registry transaction: logical owner/handle identities stay
// fixed while the two owning private-buffer references exchange slots.
int mg_gdn_state_swap_buffers(int liveOwner, int backupOwner) {
    if (liveOwner < 0 || backupOwner < 0 || liveOwner >= MG_GDN_MAX_OWNERS ||
        backupOwner >= MG_GDN_MAX_OWNERS || liveOwner == backupOwner) return 0;
    @synchronized(gDev) {
        MGGDNOwner *live = &gGDNOwners[liveOwner];
        MGGDNOwner *backup = &gGDNOwners[backupOwner];
        if (!mg_gdn_owner_compatible(*live, *backup)) return 0;
        CFTypeRef conv = live->conv;
        CFTypeRef recurrent = live->recurrent;
        live->conv = backup->conv;
        live->recurrent = backup->recurrent;
        backup->conv = conv;
        backup->recurrent = recurrent;
        return 1;
    }
}

void *mg_gdn_graph_encode(void *graph, int owner,
                          void *mixedPtr, void *zPtr, void *bPtr, void *aPtr,
                          const float *convW, const float *aLog, const float *dtBias, const float *norm,
                          int tokens, int nK, int nV, int kHd, int vHd, int convKernel, float eps) {
    if(!graph||owner<0||owner>=MG_GDN_MAX_OWNERS||!((tokens>=1&&tokens<=4)||tokens==32)||!mixedPtr||!zPtr||!bPtr||!aPtr||!convW||!aLog||!dtBias||!norm||eps<=0||!mg_gdn_pipelines())return NULL;
    MGGDNOwner slot;@synchronized(gDev){slot=gGDNOwners[owner];}
    if(slot.conv==NULL||slot.recurrent==NULL||slot.nK!=nK||slot.nV!=nV||slot.kHd!=kHd||slot.vHd!=vHd||slot.convKernel!=convKernel)return NULL;
    int keyDim=nK*kHd,valueDim=nV*vHd,convDim=2*keyDim+valueDim;
    id<MTLBuffer>mixed=(__bridge id<MTLBuffer>)mixedPtr,z=(__bridge id<MTLBuffer>)zPtr,b=(__bridge id<MTLBuffer>)bPtr,a=(__bridge id<MTLBuffer>)aPtr;
    id<MTLBuffer>convWB=mg_gdn_shared(convW,(size_t)convDim*convKernel),aLogB=mg_gdn_shared(aLog,nV),dtBiasB=mg_gdn_shared(dtBias,nV),normB=mg_gdn_shared(norm,vHd);
    id<MTLBuffer>convOut=[gDev newBufferWithLength:(size_t)tokens*convDim*sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer>qNorm=[gDev newBufferWithLength:(size_t)tokens*nK*kHd*sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer>kNorm=[gDev newBufferWithLength:(size_t)tokens*nK*kHd*sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer>core=(__bridge id<MTLBuffer>)mg_graph_alloc_result(graph,tokens*valueDim);
    id<MTLCommandBuffer>command=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(graph);
    if(!mixed||!z||!b||!a||!convWB||!aLogB||!dtBiasB||!normB||!convOut||!qNorm||!kNorm||!core||!command)return NULL;
    id<MTLComputeCommandEncoder>encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:gGDNConvPSO];[encoder setBuffer:mixed offset:0 atIndex:0];[encoder setBuffer:convWB offset:0 atIndex:1];[encoder setBuffer:(__bridge id<MTLBuffer>)slot.conv offset:0 atIndex:2];[encoder setBuffer:convOut offset:0 atIndex:3];[encoder setBytes:&tokens length:sizeof(tokens) atIndex:4];[encoder setBytes:&convDim length:sizeof(convDim) atIndex:5];[encoder setBytes:&convKernel length:sizeof(convKernel) atIndex:6];[encoder dispatchThreads:MTLSizeMake((NSUInteger)convDim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    int qThreads=mg_gdn_threads(kHd);[encoder setComputePipelineState:gGDNQKNormPSO];[encoder setBuffer:convOut offset:0 atIndex:0];[encoder setBuffer:qNorm offset:0 atIndex:1];[encoder setBuffer:kNorm offset:0 atIndex:2];[encoder setBytes:&tokens length:sizeof(tokens) atIndex:3];[encoder setBytes:&convDim length:sizeof(convDim) atIndex:4];[encoder setBytes:&nK length:sizeof(nK) atIndex:5];[encoder setBytes:&kHd length:sizeof(kHd) atIndex:6];[encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)nK,(NSUInteger)tokens,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)qThreads,1,1)];[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    int vThreads=mg_gdn_threads(vHd);id<MTLComputePipelineState> recPSO=gGDNRecurrentPSO;int recThreads=vThreads;if(!gGDNForceBaseline&&kHd==128&&(vHd%8)==0&&gGDNRecurrentPackedPSO!=nil){recPSO=gGDNRecurrentPackedPSO;recThreads=(vHd/8)*32;}[encoder setComputePipelineState:recPSO];[encoder setBuffer:convOut offset:0 atIndex:0];[encoder setBuffer:qNorm offset:0 atIndex:1];[encoder setBuffer:kNorm offset:0 atIndex:2];[encoder setBuffer:z offset:0 atIndex:3];[encoder setBuffer:b offset:0 atIndex:4];[encoder setBuffer:a offset:0 atIndex:5];[encoder setBuffer:aLogB offset:0 atIndex:6];[encoder setBuffer:dtBiasB offset:0 atIndex:7];[encoder setBuffer:normB offset:0 atIndex:8];[encoder setBuffer:(__bridge id<MTLBuffer>)slot.recurrent offset:0 atIndex:9];[encoder setBuffer:core offset:0 atIndex:10];[encoder setBytes:&tokens length:sizeof(tokens) atIndex:11];[encoder setBytes:&convDim length:sizeof(convDim) atIndex:12];[encoder setBytes:&nK length:sizeof(nK) atIndex:13];[encoder setBytes:&nV length:sizeof(nV) atIndex:14];[encoder setBytes:&kHd length:sizeof(kHd) atIndex:15];[encoder setBytes:&vHd length:sizeof(vHd) atIndex:16];[encoder setBytes:&eps length:sizeof(eps) atIndex:17];[encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)nV,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)recThreads,1,1)];[encoder endEncoding];return (__bridge void*)core;
}

// Encode B independent P=1 owners into one caller-owned projection graph. The
// projected operands are B-row panels, but each row receives its own private
// convolution and recurrent buffers; rows are never scanned as token order.
void *mg_gdn_graph_encode_batch(void *graph, const int *owners, int batch,
                                void *mixedPtr, void *zPtr, void *bPtr, void *aPtr,
                                const float *convW, const float *aLog, const float *dtBias, const float *norm,
                                int nK, int nV, int kHd, int vHd, int convKernel, float eps) {
    if (!graph || !owners || batch < 2 || batch > 24 || !mixedPtr || !zPtr || !bPtr || !aPtr ||
        !convW || !aLog || !dtBias || !norm || eps <= 0 || !mg_gdn_pipelines()) return NULL;

    MGGDNOwner slots[24];
    for (int row = 0; row < batch; ++row) {
        if (owners[row] < 0 || owners[row] >= MG_GDN_MAX_OWNERS) return NULL;
        for (int previous = 0; previous < row; ++previous) {
            if (owners[previous] == owners[row]) return NULL;
        }
        @synchronized(gDev) { slots[row] = gGDNOwners[owners[row]]; }
        MGGDNOwner slot = slots[row];
        if (slot.conv == NULL || slot.recurrent == NULL || slot.nK != nK || slot.nV != nV ||
            slot.kHd != kHd || slot.vHd != vHd || slot.convKernel != convKernel) return NULL;
    }

    int tokens = 1;
    int keyDim = nK * kHd, valueDim = nV * vHd, convDim = 2 * keyDim + valueDim;
    id<MTLBuffer> mixed = (__bridge id<MTLBuffer>)mixedPtr;
    id<MTLBuffer> z = (__bridge id<MTLBuffer>)zPtr;
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)bPtr;
    id<MTLBuffer> a = (__bridge id<MTLBuffer>)aPtr;
    id<MTLBuffer> convWB = mg_gdn_shared(convW, (size_t)convDim * convKernel);
    id<MTLBuffer> aLogB = mg_gdn_shared(aLog, nV);
    id<MTLBuffer> dtBiasB = mg_gdn_shared(dtBias, nV);
    id<MTLBuffer> normB = mg_gdn_shared(norm, vHd);
    id<MTLBuffer> convOut = [gDev newBufferWithLength:(size_t)batch * convDim * sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> qNorm = [gDev newBufferWithLength:(size_t)batch * keyDim * sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> kNorm = [gDev newBufferWithLength:(size_t)batch * keyDim * sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> core = (__bridge id<MTLBuffer>)mg_graph_alloc_result(graph, batch * valueDim);
    id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(graph);
    if (!mixed || !z || !b || !a || !convWB || !aLogB || !dtBiasB || !normB ||
        !convOut || !qNorm || !kNorm || !core || !command) return NULL;

    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return NULL;
    int qThreads = mg_gdn_threads(kHd);
    int vThreads = mg_gdn_threads(vHd);
    for (int row = 0; row < batch; ++row) {
        NSUInteger mixedOffset = (NSUInteger)row * convDim * sizeof(float);
        NSUInteger valueOffset = (NSUInteger)row * valueDim * sizeof(float);
        NSUInteger headOffset = (NSUInteger)row * nV * sizeof(float);
        NSUInteger keyOffset = (NSUInteger)row * keyDim * sizeof(float);

        [encoder setComputePipelineState:gGDNConvPSO];
        [encoder setBuffer:mixed offset:mixedOffset atIndex:0];
        [encoder setBuffer:convWB offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)slots[row].conv offset:0 atIndex:2];
        [encoder setBuffer:convOut offset:mixedOffset atIndex:3];
        [encoder setBytes:&tokens length:sizeof(tokens) atIndex:4];
        [encoder setBytes:&convDim length:sizeof(convDim) atIndex:5];
        [encoder setBytes:&convKernel length:sizeof(convKernel) atIndex:6];
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)convDim, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [encoder setComputePipelineState:gGDNQKNormPSO];
        [encoder setBuffer:convOut offset:mixedOffset atIndex:0];
        [encoder setBuffer:qNorm offset:keyOffset atIndex:1];
        [encoder setBuffer:kNorm offset:keyOffset atIndex:2];
        [encoder setBytes:&tokens length:sizeof(tokens) atIndex:3];
        [encoder setBytes:&convDim length:sizeof(convDim) atIndex:4];
        [encoder setBytes:&nK length:sizeof(nK) atIndex:5];
        [encoder setBytes:&kHd length:sizeof(kHd) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)nK, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake((NSUInteger)qThreads, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        id<MTLComputePipelineState> recPSO = gGDNRecurrentPSO;
        int recThreads = vThreads;
        if (!gGDNForceBaseline && kHd == 128 && (vHd % 8) == 0 && gGDNRecurrentPackedPSO != nil) {
            recPSO = gGDNRecurrentPackedPSO;
            recThreads = (vHd / 8) * 32;
        }

        [encoder setComputePipelineState:recPSO];
        [encoder setBuffer:convOut offset:mixedOffset atIndex:0];
        [encoder setBuffer:qNorm offset:keyOffset atIndex:1];
        [encoder setBuffer:kNorm offset:keyOffset atIndex:2];
        [encoder setBuffer:z offset:valueOffset atIndex:3];
        [encoder setBuffer:b offset:headOffset atIndex:4];
        [encoder setBuffer:a offset:headOffset atIndex:5];
        [encoder setBuffer:aLogB offset:0 atIndex:6];
        [encoder setBuffer:dtBiasB offset:0 atIndex:7];
        [encoder setBuffer:normB offset:0 atIndex:8];
        [encoder setBuffer:(__bridge id<MTLBuffer>)slots[row].recurrent offset:0 atIndex:9];
        [encoder setBuffer:core offset:valueOffset atIndex:10];
        [encoder setBytes:&tokens length:sizeof(tokens) atIndex:11];
        [encoder setBytes:&convDim length:sizeof(convDim) atIndex:12];
        [encoder setBytes:&nK length:sizeof(nK) atIndex:13];
        [encoder setBytes:&nV length:sizeof(nV) atIndex:14];
        [encoder setBytes:&kHd length:sizeof(kHd) atIndex:15];
        [encoder setBytes:&vHd length:sizeof(vHd) atIndex:16];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:17];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)nV, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake((NSUInteger)recThreads, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [encoder endEncoding];
    return (__bridge void *)core;
}

int mg_gdn_state_seed(int owner, const float *conv, int convElems, const float *recurrent, int recurrentElems) {
    @autoreleasepool {
        if (owner < 0 || owner >= MG_GDN_MAX_OWNERS || recurrent == NULL) return 0;
        MGGDNOwner slot;
        @synchronized(gDev) { slot = gGDNOwners[owner]; }
        if (slot.conv == NULL || slot.recurrent == NULL) return 0;
        int convDim = 2 * slot.nK * slot.kHd + slot.nV * slot.vHd;
        int wantConv = (slot.convKernel - 1) * convDim;
        int wantRecurrent = slot.nV * slot.kHd * slot.vHd;
        if (convElems != wantConv || recurrentElems != wantRecurrent ||
            (wantConv > 0 && conv == NULL)) return 0;

        id<MTLBuffer> convStage = nil;
        if (wantConv > 0) convStage = mg_gdn_shared(conv, (size_t)wantConv);
        id<MTLBuffer> recurrentStage = mg_gdn_shared(recurrent, (size_t)wantRecurrent);
        if ((wantConv > 0 && convStage == nil) || recurrentStage == nil) return 0;

        id<MTLCommandBuffer> command = [gQueue commandBuffer];
        if (command == nil) return 0;
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        if (encoder == nil) return 0;
        if (wantConv > 0) {
            [encoder copyFromBuffer:convStage sourceOffset:0
                           toBuffer:(__bridge id<MTLBuffer>)slot.conv destinationOffset:0
                               size:(size_t)wantConv * sizeof(float)];
        }
        [encoder copyFromBuffer:recurrentStage sourceOffset:0
                       toBuffer:(__bridge id<MTLBuffer>)slot.recurrent destinationOffset:0
                           size:(size_t)wantRecurrent * sizeof(float)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        return command.status == MTLCommandBufferStatusCompleted;
    }
}

int mg_gdn_state_reset(int owner) {
    if (owner < 0 || owner >= MG_GDN_MAX_OWNERS) return 0;
    MGGDNOwner slot;
    @synchronized(gDev) { slot = gGDNOwners[owner]; }
    if (slot.conv == NULL || slot.recurrent == NULL) return 0;
    return mg_gdn_zero((__bridge id<MTLBuffer>)slot.conv, (__bridge id<MTLBuffer>)slot.recurrent);
}

int mg_gdn_state_snapshot(int owner, float *conv, int convElems, float *recurrent, int recurrentElems) {
    @autoreleasepool {
        if (owner < 0 || owner >= MG_GDN_MAX_OWNERS || recurrent == NULL || recurrentElems <= 0) return 0;
        MGGDNOwner slot;
        @synchronized(gDev) { slot = gGDNOwners[owner]; }
        if (slot.conv == NULL || slot.recurrent == NULL) return 0;
        int convDim = 2 * slot.nK * slot.kHd + slot.nV * slot.vHd;
        int wantConv = (slot.convKernel - 1) * convDim;
        int wantRecurrent = slot.nV * slot.kHd * slot.vHd;
        if (convElems != wantConv || recurrentElems != wantRecurrent || (wantConv > 0 && conv == NULL)) return 0;
        id<MTLBuffer> convRead = [gDev newBufferWithLength:MAX((size_t)4, (size_t)wantConv * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> recurrentRead = [gDev newBufferWithLength:(size_t)wantRecurrent * sizeof(float) options:MTLResourceStorageModeShared];
        if (convRead == nil || recurrentRead == nil) return 0;
        id<MTLCommandBuffer> command = [gQueue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        if (wantConv > 0) [encoder copyFromBuffer:(__bridge id<MTLBuffer>)slot.conv sourceOffset:0 toBuffer:convRead destinationOffset:0 size:(size_t)wantConv * sizeof(float)];
        [encoder copyFromBuffer:(__bridge id<MTLBuffer>)slot.recurrent sourceOffset:0 toBuffer:recurrentRead destinationOffset:0 size:(size_t)wantRecurrent * sizeof(float)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) return 0;
        if (wantConv > 0) memcpy(conv, convRead.contents, (size_t)wantConv * sizeof(float));
        memcpy(recurrent, recurrentRead.contents, (size_t)wantRecurrent * sizeof(float));
        return 1;
    }
}

void mg_gdn_state_release(int owner) {
    if (owner < 0 || owner >= MG_GDN_MAX_OWNERS) return;
    @synchronized(gDev) {
        MGGDNOwner *slot = &gGDNOwners[owner];
        if (slot->conv != NULL) CFBridgingRelease(slot->conv);
        if (slot->recurrent != NULL) CFBridgingRelease(slot->recurrent);
        memset(slot, 0, sizeof(*slot));
    }
}

int mg_gdn_live_buffers(void) {
    int live = 0;
    @synchronized(gDev) {
        for (int i = 0; i < MG_GDN_MAX_OWNERS; ++i) {
            if (gGDNOwners[i].conv != NULL) ++live;
            if (gGDNOwners[i].recurrent != NULL) ++live;
        }
    }
    return live;
}

uint64_t mg_gdn_current_allocated_size(void) {
    if (!mg_init()) return 0;
    return (uint64_t)gDev.currentAllocatedSize;
}

static id<MTLComputePipelineState> gTestShufflePSO = nil;

int mg_test_run_shuffle(const float *in, float *out) {
    if (!mg_init()) return 0;
    @autoreleasepool {
        if (gTestShufflePSO == nil) {
            NSString *src = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void test_4lane_butterfly_shuffle(device const float *in [[buffer(0)]],
                                         device float *out [[buffer(1)]],
                                         uint tid [[thread_index_in_threadgroup]]) {
    float v = in[tid];
    v += simd_shuffle_xor(v, 1);
    v += simd_shuffle_xor(v, 2);
    out[tid] = v;
}
)MSL";
            NSError *error = nil;
            id<MTLLibrary> lib = [gDev newLibraryWithSource:src options:nil error:&error];
            if (lib == nil) return 0;
            id<MTLFunction> fn = [lib newFunctionWithName:@"test_4lane_butterfly_shuffle"];
            if (fn == nil) return 0;
            gTestShufflePSO = [gDev newComputePipelineStateWithFunction:fn error:&error];
            if (gTestShufflePSO == nil) return 0;
        }

        id<MTLBuffer> inBuf = [gDev newBufferWithBytes:in length:32 * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = [gDev newBufferWithLength:32 * sizeof(float) options:MTLResourceStorageModeShared];
        if (inBuf == nil || outBuf == nil) return 0;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gTestShufflePSO];
        [enc setBuffer:inBuf offset:0 atIndex:0];
        [enc setBuffer:outBuf offset:0 atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(out, outBuf.contents, 32 * sizeof(float));
        return 1;
    }
}
