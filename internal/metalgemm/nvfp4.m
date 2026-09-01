//go:build darwin && arm64 && cgo

// Apple M5 M=1 W4A16 NVFP4 GEMV. The packed E2M1 weights and E4M3FN
// per-16-weight scales remain resident; one SIMD group decodes each output row.
#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>
#include <math.h>
#include <string.h>

extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;

static NSString *kNVFP4Src = @R"MSL(
#include <metal_stdlib>
using namespace metal;

inline float nvfp4_e2m1(uchar nibble) {
    // Inject s|e1e0m at f16 bits 15|11:9, then exactly renormalize. This
    // preserves the +/-0.5 codes on M5, where f32/bf16 subnormals flush.
    ushort bits = (ushort(nibble & 7u) << 9) | (ushort(nibble & 8u) << 12);
    return float(as_type<half>(bits) * as_type<half>(ushort(0x7400)));
}

inline float nvfp4_e4m3fn(uchar raw) {
    uint mag = uint(raw & 0x7fu);
    uint exponent = mag >> 3;
    uint mantissa = mag & 7u;
    float value = exponent == 0u
        ? float(mantissa) * (1.0f / 512.0f)
        : ldexp(float(8u + mantissa), int(exponent) - 10);
    return (raw & 0x80u) ? -value : value;
}

kernel void nvfp4_gemv(device const uchar *packed [[buffer(0)]],
                       device const uchar *scales [[buffer(1)]],
                       device const float *x [[buffer(2)]],
                       device float *y [[buffer(3)]],
                       constant int &in [[buffer(4)]],
                       constant int &out [[buffer(5)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]]) {
    if (row >= uint(out)) return;
    uint blocks = uint(in) / 16u;
    uint rowWeight = row * uint(in);
    uint rowScale = row * blocks;
    float sum = 0.0f;
    for (uint block = lane; block < blocks; block += 32u) {
        float scale = abs(nvfp4_e4m3fn(scales[rowScale + block]));
        uint k0 = block * 16u;
        for (uint j = 0; j < 16u; ++j) {
            uint k = k0 + j;
            uchar pair = packed[(rowWeight + k) >> 1];
            uchar nibble = (k & 1u) ? (pair >> 4) : (pair & 15u);
            sum += nvfp4_e2m1(nibble) * scale * x[k];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0u) y[row] = sum;
}
)MSL";

static id<MTLComputePipelineState> psoNVFP4;
static int gNVFP4Ready;

typedef struct {
    CFTypeRef packed;
    CFTypeRef scales;
    int out, in;
} NVFP4W;

#define MG_MAX_NVFP4 8192
static NVFP4W gNVFP4[MG_MAX_NVFP4];
static int gNNVFP4;
static id<MTLBuffer> gNVFP4X;
static id<MTLBuffer> gNVFP4Y;
static long gNVFP4XCap, gNVFP4YCap;

static int nvfp4_init(void) {
    if (gNVFP4Ready) return 1;
    if (gDev == nil) return 0;
    NSError *err = nil;
    id<MTLLibrary> lib = [gDev newLibraryWithSource:kNVFP4Src options:nil error:&err];
    if (lib == nil) { NSLog(@"nvfp4: library compile failed: %@", err); return 0; }
    psoNVFP4 = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nvfp4_gemv"] error:&err];
    if (psoNVFP4 == nil) { NSLog(@"nvfp4: pipeline build failed: %@", err); return 0; }
    gNVFP4Ready = 1;
    return 1;
}

static void nvfp4_grow_scratch(long in, long out) {
    if (gNVFP4X == nil || gNVFP4XCap < in) {
        gNVFP4X = [gDev newBufferWithLength:(NSUInteger)(in * 4) options:MTLResourceStorageModeShared];
        gNVFP4XCap = in;
    }
    if (gNVFP4Y == nil || gNVFP4YCap < out) {
        gNVFP4Y = [gDev newBufferWithLength:(NSUInteger)(out * 4) options:MTLResourceStorageModeShared];
        gNVFP4YCap = out;
    }
}

int mg_nvfp4_upload(const unsigned char *packed, const unsigned char *scales, int out, int in) {
    if (!nvfp4_init() || out <= 0 || in <= 0 || in % 16 != 0 || gNNVFP4 >= MG_MAX_NVFP4) return -1;
    long packedBytes = (long)out * in / 2;
    long scaleBytes = (long)out * in / 16;
    id<MTLBuffer> pb = [gDev newBufferWithLength:(NSUInteger)packedBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> sb = [gDev newBufferWithLength:(NSUInteger)scaleBytes options:MTLResourceStorageModeShared];
    if (pb == nil || sb == nil) return -1;
    memcpy(pb.contents, packed, (size_t)packedBytes);
    memcpy(sb.contents, scales, (size_t)scaleBytes);
    int id = gNNVFP4++;
    gNVFP4[id] = (NVFP4W){CFBridgingRetain(pb), CFBridgingRetain(sb), out, in};
    return id;
}

int mg_nvfp4_gemv(int wid, const float *x, float *y) {
    if (wid < 0 || wid >= gNNVFP4 || !nvfp4_init()) return 0;
    @autoreleasepool {
        NVFP4W w = gNVFP4[wid];
        nvfp4_grow_scratch(w.in, w.out);
        if (gNVFP4X == nil || gNVFP4Y == nil) return 0;
        memcpy(gNVFP4X.contents, x, (size_t)w.in * 4);
        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:psoNVFP4];
        [enc setBuffer:(__bridge id<MTLBuffer>)w.packed offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)w.scales offset:0 atIndex:1];
        [enc setBuffer:gNVFP4X offset:0 atIndex:2];
        [enc setBuffer:gNVFP4Y offset:0 atIndex:3];
        [enc setBytes:&w.in length:sizeof(int) atIndex:4];
        [enc setBytes:&w.out length:sizeof(int) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)w.out, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(y, gNVFP4Y.contents, (size_t)w.out * 4);
        return 1;
    }
}

void mg_nvfp4_reset(void) {
    for (int i = 0; i < gNNVFP4; ++i) {
        if (gNVFP4[i].packed) CFRelease(gNVFP4[i].packed);
        if (gNVFP4[i].scales) CFRelease(gNVFP4[i].scales);
        gNVFP4[i] = (NVFP4W){0};
    }
    gNNVFP4 = 0;
    gNVFP4X = nil; gNVFP4Y = nil;
    gNVFP4XCap = 0; gNVFP4YCap = 0;
}
