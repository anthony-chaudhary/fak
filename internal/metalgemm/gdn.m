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

enum { MG_GDN_MAX_OWNERS = 256 };
static MGGDNOwner gGDNOwners[MG_GDN_MAX_OWNERS];
static uint64_t gGDNNextHandle = 1;
static id<MTLComputePipelineState> gGDNConvPSO;
static id<MTLComputePipelineState> gGDNQKNormPSO;
static id<MTLComputePipelineState> gGDNRecurrentPSO;
static BOOL gGDNPipelineAttempted;

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
)MSL";

static int mg_gdn_pipelines(void) {
    @synchronized(gDev) {
        if (gGDNConvPSO != nil && gGDNQKNormPSO != nil && gGDNRecurrentPSO != nil) return 1;
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
        if (gGDNConvPSO == nil || gGDNQKNormPSO == nil || gGDNRecurrentPSO == nil) {
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
    return -1;
}

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
        [encoder setComputePipelineState:gGDNRecurrentPSO];
        [encoder setBuffer:convOutB offset:0 atIndex:0]; [encoder setBuffer:qNormB offset:0 atIndex:1]; [encoder setBuffer:kNormB offset:0 atIndex:2];
        [encoder setBuffer:zB offset:0 atIndex:3]; [encoder setBuffer:bB offset:0 atIndex:4]; [encoder setBuffer:aB offset:0 atIndex:5];
        [encoder setBuffer:aLogB offset:0 atIndex:6]; [encoder setBuffer:dtBiasB offset:0 atIndex:7]; [encoder setBuffer:normB offset:0 atIndex:8];
        [encoder setBuffer:recurrentState offset:0 atIndex:9]; [encoder setBuffer:coreB offset:0 atIndex:10];
        [encoder setBytes:&tokens length:sizeof(tokens) atIndex:11]; [encoder setBytes:&convDim length:sizeof(convDim) atIndex:12];
        [encoder setBytes:&nK length:sizeof(nK) atIndex:13]; [encoder setBytes:&nV length:sizeof(nV) atIndex:14];
        [encoder setBytes:&kHd length:sizeof(kHd) atIndex:15]; [encoder setBytes:&vHd length:sizeof(vHd) atIndex:16];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:17];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)nV, 1, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)vThreads, 1, 1)];
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
void *mg_gdn_graph_encode(void *graph, int owner,
                          void *mixedPtr, void *zPtr, void *bPtr, void *aPtr,
                          const float *convW, const float *aLog, const float *dtBias, const float *norm,
                          int tokens, int nK, int nV, int kHd, int vHd, int convKernel, float eps) {
    if(!graph||owner<0||owner>=MG_GDN_MAX_OWNERS||(tokens!=1&&tokens!=32)||!mixedPtr||!zPtr||!bPtr||!aPtr||!convW||!aLog||!dtBias||!norm||eps<=0||!mg_gdn_pipelines())return NULL;
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
    int vThreads=mg_gdn_threads(vHd);[encoder setComputePipelineState:gGDNRecurrentPSO];[encoder setBuffer:convOut offset:0 atIndex:0];[encoder setBuffer:qNorm offset:0 atIndex:1];[encoder setBuffer:kNorm offset:0 atIndex:2];[encoder setBuffer:z offset:0 atIndex:3];[encoder setBuffer:b offset:0 atIndex:4];[encoder setBuffer:a offset:0 atIndex:5];[encoder setBuffer:aLogB offset:0 atIndex:6];[encoder setBuffer:dtBiasB offset:0 atIndex:7];[encoder setBuffer:normB offset:0 atIndex:8];[encoder setBuffer:(__bridge id<MTLBuffer>)slot.recurrent offset:0 atIndex:9];[encoder setBuffer:core offset:0 atIndex:10];[encoder setBytes:&tokens length:sizeof(tokens) atIndex:11];[encoder setBytes:&convDim length:sizeof(convDim) atIndex:12];[encoder setBytes:&nK length:sizeof(nK) atIndex:13];[encoder setBytes:&nV length:sizeof(nV) atIndex:14];[encoder setBytes:&kHd length:sizeof(kHd) atIndex:15];[encoder setBytes:&vHd length:sizeof(vHd) atIndex:16];[encoder setBytes:&eps length:sizeof(eps) atIndex:17];[encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)nV,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)vThreads,1,1)];[encoder endEncoding];return (__bridge void*)core;
}

// Encode B independent P=1 owners into one caller-owned projection graph. The
// projected operands are B-row panels, but each row receives its own private
// convolution and recurrent buffers; rows are never scanned as token order.
void *mg_gdn_graph_encode_batch(void *graph, const int *owners, int batch,
                                void *mixedPtr, void *zPtr, void *bPtr, void *aPtr,
                                const float *convW, const float *aLog, const float *dtBias, const float *norm,
                                int nK, int nV, int kHd, int vHd, int convKernel, float eps) {
    if (!graph || !owners || batch < 2 || batch > 8 || !mixedPtr || !zPtr || !bPtr || !aPtr ||
        !convW || !aLog || !dtBias || !norm || eps <= 0 || !mg_gdn_pipelines()) return NULL;

    MGGDNOwner slots[8];
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

        [encoder setComputePipelineState:gGDNRecurrentPSO];
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
                 threadsPerThreadgroup:MTLSizeMake((NSUInteger)vThreads, 1, 1)];
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


// Encode B independent P=1 state owners in one command buffer. The projection
// panels are row-major [B,width]; each row advances only its matching owner.
// This deliberately emits the three recurrent kernels per row while keeping
// all rows in the caller's single graph submission. The five weight projection
// dispatches remain B-wide and are encoded by the graph before this function.
void *mg_gdn_graph_encode_batch(void *graph, const int *owners, int batch,
                                void *mixedPtr, void *zPtr, void *bPtr, void *aPtr,
                                const float *convW, const float *aLog, const float *dtBias, const float *norm,
                                int nK, int nV, int kHd, int vHd, int convKernel, float eps) {
    if (!graph || !owners || batch < 2 || batch > 8 || !mixedPtr || !zPtr || !bPtr || !aPtr ||
        !convW || !aLog || !dtBias || !norm || !gdn_init()) return NULL;
    int keyDim=nK*kHd,valueDim=nV*vHd,convDim=2*keyDim+valueDim;
    id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(graph);
    id<MTLBuffer> mixed=(__bridge id<MTLBuffer>)mixedPtr,z=(__bridge id<MTLBuffer>)zPtr,
                  bv=(__bridge id<MTLBuffer>)bPtr,av=(__bridge id<MTLBuffer>)aPtr;
    id<MTLBuffer> core=(__bridge id<MTLBuffer>)mg_graph_alloc_result(graph,batch*valueDim);
    if(!cb||!mixed||!z||!bv||!av||!core)return NULL;
    id<MTLBuffer> cw=gdn_host_buffer(convW,(NSUInteger)convDim*convKernel*sizeof(float));
    id<MTLBuffer> al=gdn_host_buffer(aLog,(NSUInteger)nV*sizeof(float));
    id<MTLBuffer> db=gdn_host_buffer(dtBias,(NSUInteger)nV*sizeof(float));
    id<MTLBuffer> nw=gdn_host_buffer(norm,(NSUInteger)vHd*sizeof(float));
    if(!cw||!al||!db||!nw)return NULL;
    int tokens=1;
    for(int row=0;row<batch;row++){
        int owner=owners[row];
        if(owner<0||owner>=GDN_MAX_OWNERS)return NULL;
        GDNOwner*o=&gOwners[owner];
        if(!o->live||o->nK!=nK||o->nV!=nV||o->kHd!=kHd||o->vHd!=vHd||o->convKernel!=convKernel)return NULL;
        id<MTLBuffer> convOut=[gDevice newBufferWithLength:(NSUInteger)convDim*sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> qk=[gDevice newBufferWithLength:(NSUInteger)2*keyDim*sizeof(float) options:MTLResourceStorageModePrivate];
        if(!convOut||!qk)return NULL;
        NSUInteger mixedOff=(NSUInteger)row*convDim*sizeof(float), zOff=(NSUInteger)row*valueDim*sizeof(float),
                   scalarOff=(NSUInteger)row*nV*sizeof(float), coreOff=(NSUInteger)row*valueDim*sizeof(float);
        id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder];
        [e1 setComputePipelineState:gConv];[e1 setBuffer:mixed offset:mixedOff atIndex:0];[e1 setBuffer:cw offset:0 atIndex:1];
        [e1 setBuffer:o->conv offset:0 atIndex:2];[e1 setBuffer:convOut offset:0 atIndex:3];
        [e1 setBytes:&tokens length:4 atIndex:4];[e1 setBytes:&convDim length:4 atIndex:5];[e1 setBytes:&convKernel length:4 atIndex:6];
        gdn_dispatch(e1,gConv,(NSUInteger)convDim);[e1 endEncoding];
        id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder];
        [e2 setComputePipelineState:gNorm];[e2 setBuffer:convOut offset:0 atIndex:0];[e2 setBuffer:qk offset:0 atIndex:1];
        [e2 setBytes:&tokens length:4 atIndex:2];[e2 setBytes:&nK length:4 atIndex:3];[e2 setBytes:&kHd length:4 atIndex:4];[e2 setBytes:&keyDim length:4 atIndex:5];
        gdn_dispatch(e2,gNorm,(NSUInteger)(2*keyDim));[e2 endEncoding];
        id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder];
        [e3 setComputePipelineState:gRecur];[e3 setBuffer:qk offset:0 atIndex:0];[e3 setBuffer:convOut offset:(NSUInteger)2*keyDim*sizeof(float) atIndex:1];
        [e3 setBuffer:z offset:zOff atIndex:2];[e3 setBuffer:bv offset:scalarOff atIndex:3];[e3 setBuffer:av offset:scalarOff atIndex:4];
        [e3 setBuffer:al offset:0 atIndex:5];[e3 setBuffer:db offset:0 atIndex:6];[e3 setBuffer:nw offset:0 atIndex:7];
        [e3 setBuffer:o->recurrent offset:0 atIndex:8];[e3 setBuffer:core offset:coreOff atIndex:9];
        [e3 setBytes:&tokens length:4 atIndex:10];[e3 setBytes:&nK length:4 atIndex:11];[e3 setBytes:&nV length:4 atIndex:12];
        [e3 setBytes:&kHd length:4 atIndex:13];[e3 setBytes:&vHd length:4 atIndex:14];[e3 setBytes:&eps length:4 atIndex:15];
        gdn_dispatch(e3,gRecur,(NSUInteger)(nV*vHd));[e3 endEncoding];
    }
    return (__bridge void*)core;
}
