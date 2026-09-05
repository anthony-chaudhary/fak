//go:build darwin && arm64 && cgo

// q2k.m — Metal dequant-GEMV and batched GEMM for GGML Q2_K 2-bit k-quants.
//
// Q2_K super-block layout (256 weights, 84 bytes):
//   - scales: 16 bytes (low 4 bits = scale multiplier, high 4 bits = min multiplier)
//   - q:      64 bytes (256 2-bit codes, packed 4 per byte)
//   - d:      half (f16, offset 80)
//   - dmin:   half (f16, offset 82)
//
// In-shader unpack runs on GPU during the GEMV/GEMM dot product so weights stay 84-byte
// resident on Apple unified memory (~0.328 bytes/weight) without FP16 or F32 materialization.

#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>
#include <string.h>
#include <limits.h>

// Device + queue are owned by metal.m (mg_init); we reuse them.
extern id<MTLDevice>       gDev;
extern id<MTLCommandQueue> gQueue;
extern int                 mg_init(void);

static NSString *kQ2KSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// q2k_block_dot: dot one 84-byte Q2_K super-block's 256 dequanted weights against the matching
// 256-wide activation slice. Matches ggufload.dequantQ2KScalar byte-for-byte.
inline float q2k_block_dot(device const uchar* blk, device const float* xs) {
    device const uchar* scales = blk;
    device const uchar* q      = blk + 16;
    float d   = (float)(*(device const half*)(blk + 80));
    float min = (float)(*(device const half*)(blk + 82));

    float acc = 0.0f;
    int is = 0;
    int qi = 0;

    for (int n = 0; n < 256; n += 128) {
        uint shift = 0;
        for (int j = 0; j < 4; j++) {
            uchar sc0 = scales[is++];
            float dl0 = d   * (float)(sc0 & 0x0f);
            float ml0 = min * (float)(sc0 >> 4);

            uchar sc1 = scales[is++];
            float dl1 = d   * (float)(sc1 & 0x0f);
            float ml1 = min * (float)(sc1 >> 4);

            for (int l = 0; l < 16; l++) {
                float w0 = dl0 * (float)((q[qi + l] >> shift) & 3) - ml0;
                float w1 = dl1 * (float)((q[qi + 16 + l] >> shift) & 3) - ml1;
                acc += w0 * xs[n + j * 32 + l];
                acc += w1 * xs[n + j * 32 + 16 + l];
            }
            shift += 2;
        }
        qi += 32;
    }
    return acc;
}

// q2k_gemv: decode GEMV. ONE 32-lane SIMD group per output row.
// Lanes stride over nblk super-blocks and reduce via simd_sum.
kernel void q2k_gemv(device const uchar* W [[buffer(0)]],
                     device const float* X [[buffer(1)]],
                     device float*       Y [[buffer(2)]],
                     constant int&    nblk [[buffer(3)]],
                     constant int&     out [[buffer(4)]],
                     uint o   [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]]) {
    if (o >= (uint)out) return;
    device const uchar* row = W + (long)o * nblk * 84;
    float acc = 0.0f;
    for (int b = (int)lid; b < nblk; b += 32) {
        acc += q2k_block_dot(row + (long)b * 84, X + (long)b * 256);
    }
    acc = simd_sum(acc);
    if (lid == 0) {
        Y[o] = acc;
    }
}

// q2k_gemm: batched prefill GEMM for resident Q2_K rows.
// ONE 32-lane SIMD group per (output row, prompt token).
kernel void q2k_gemm(device const uchar* W [[buffer(0)]],
                     device const float* X [[buffer(1)]],
                     device float*       Y [[buffer(2)]],
                     constant int&    nblk [[buffer(3)]],
                     constant int&     out [[buffer(4)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint  lid [[thread_index_in_threadgroup]]) {
    uint o = tg.x;
    uint t = tg.y;
    if (o >= (uint)out) return;
    device const uchar* row = W + (long)o * nblk * 84;
    device const float* xs  = X + (long)t * nblk * 256;
    float acc = 0.0f;
    for (int b = (int)lid; b < nblk; b += 32) {
        acc += q2k_block_dot(row + (long)b * 84, xs + (long)b * 256);
    }
    acc = simd_sum(acc);
    if (lid == 0) {
        Y[(long)t * out + o] = acc;
    }
}
)MSL";

static id<MTLComputePipelineState> psoQ2KGemv = nil;
static id<MTLComputePipelineState> psoQ2KGemm = nil;
static int gQ2KReady = 0;

static int q2k_init(void) {
    if (gQ2KReady) return 1;
    if (!mg_init() || gDev == nil) return 0;
    NSError *err = nil;
    id<MTLLibrary> lib = [gDev newLibraryWithSource:kQ2KSrc options:nil error:&err];
    if (lib == nil) {
        NSLog(@"q2k: library compile failed: %@", err);
        return 0;
    }
    psoQ2KGemv = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q2k_gemv"] error:&err];
    psoQ2KGemm = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q2k_gemm"] error:&err];
    if (!psoQ2KGemv || !psoQ2KGemm) {
        NSLog(@"q2k: pipeline build failed: %@", err);
        return 0;
    }
    gQ2KReady = 1;
    return 1;
}

typedef struct {
    CFTypeRef buf; // retained id<MTLBuffer>
    int out, in, nblk;
    int handle; // generation * MG_MAX_Q2K + slot; -1 retires an exhausted slot
} Q2KW;

#define MG_MAX_Q2K 8192
static Q2KW gQ2K[MG_MAX_Q2K];
static int gNQ2K = 0;

// Keep generations across reset so copied/stale handles cannot alias new owners.
// Backend calls retain the existing externally serialized execution contract.
static int q2k_slot(int handle) {
    if (handle < 0) return -1;
    int slot = handle % MG_MAX_Q2K;
    if (slot >= gNQ2K || gQ2K[slot].buf == NULL || gQ2K[slot].handle != handle) return -1;
    return slot;
}

static id<MTLBuffer> gQ2KXBuf = nil; static long gQ2KXCap = 0;
static id<MTLBuffer> gQ2KYBuf = nil; static long gQ2KYCap = 0;

static void q2k_grow_scratch(long xElems, long yElems) {
    if (gQ2KXBuf == nil || gQ2KXCap < xElems) {
        gQ2KXBuf = [gDev newBufferWithLength:(NSUInteger)(xElems * 4) options:MTLResourceStorageModeShared];
        gQ2KXCap = xElems;
    }
    if (gQ2KYBuf == nil || gQ2KYCap < yElems) {
        gQ2KYBuf = [gDev newBufferWithLength:(NSUInteger)(yElems * 4) options:MTLResourceStorageModeShared];
        gQ2KYCap = yElems;
    }
}

int mg_q2k_upload(const unsigned char* raw, int out, int in) {
    if (!mg_init() || gDev == nil) return -1;
    if (!q2k_init()) return -1;
    if (in <= 0 || in % 256 != 0 || out <= 0) return -1;
    int slot = 0;
    while (slot < gNQ2K && (gQ2K[slot].buf != NULL || gQ2K[slot].handle < 0)) slot++;
    if (slot >= MG_MAX_Q2K) {
        static int capWarned = 0;
        if (!capWarned) { capWarned = 1; NSLog(@"mg_q2k_upload: q2k weight table full (%d)", MG_MAX_Q2K); }
        return -1;
    }
    int nblk = in / 256;
    long bytes = (long)out * nblk * 84;
    id<MTLBuffer> b = [gDev newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
    if (b == nil) {
        NSLog(@"mg_q2k_upload: device buffer alloc failed for %.1f MB", (double)bytes / 1e6);
        return -1;
    }
    memcpy(b.contents, raw, (size_t)bytes);
    if (slot == gNQ2K) {
        gQ2K[slot].handle = slot;
        gNQ2K++;
    }
    gQ2K[slot].buf  = CFBridgingRetain(b);
    gQ2K[slot].out  = out;
    gQ2K[slot].in   = in;
    gQ2K[slot].nblk = nblk;
    return gQ2K[slot].handle;
}

void mg_q2k_gemv(int wid, const float* x, float* y) {
    int slot = q2k_slot(wid);
    if (slot < 0) return;
    @autoreleasepool {
        Q2KW W = gQ2K[slot];
        q2k_grow_scratch((long)W.in, (long)W.out);
        memcpy(gQ2KXBuf.contents, x, (size_t)W.in * 4);

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:psoQ2KGemv];
        [e setBuffer:(__bridge id<MTLBuffer>)W.buf offset:0 atIndex:0];
        [e setBuffer:gQ2KXBuf offset:0 atIndex:1];
        [e setBuffer:gQ2KYBuf offset:0 atIndex:2];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(y, gQ2KYBuf.contents, (size_t)W.out * 4);
    }
}

void mg_q2k_gemm(int wid, const float* X, int P, float* Y) {
    int slot = q2k_slot(wid);
    if (slot < 0 || P <= 0) return;
    @autoreleasepool {
        Q2KW W = gQ2K[slot];
        q2k_grow_scratch((long)P * W.in, (long)P * W.out);
        memcpy(gQ2KXBuf.contents, X, (size_t)P * W.in * 4);

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:psoQ2KGemm];
        [e setBuffer:(__bridge id<MTLBuffer>)W.buf offset:0 atIndex:0];
        [e setBuffer:gQ2KXBuf offset:0 atIndex:1];
        [e setBuffer:gQ2KYBuf offset:0 atIndex:2];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, (NSUInteger)P, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(Y, gQ2KYBuf.contents, (size_t)P * W.out * 4);
    }
}

void mg_q2k_release(int wid) {
    int slot = q2k_slot(wid);
    if (slot < 0) return;
    CFBridgingRelease(gQ2K[slot].buf);
    gQ2K[slot].buf = NULL;
    gQ2K[slot].out = gQ2K[slot].in = gQ2K[slot].nblk = 0;
    // Never wrap a generation: retire the slot rather than revive stale handles.
    gQ2K[slot].handle = wid > INT_MAX - MG_MAX_Q2K ? -1 : wid + MG_MAX_Q2K;
}

void mg_q2k_reset(void) {
    for (int i = 0; i < gNQ2K; i++) {
        mg_q2k_release(gQ2K[i].handle);
    }
    gQ2KXBuf = nil; gQ2KXCap = 0;
    gQ2KYBuf = nil; gQ2KYCap = 0;
}
