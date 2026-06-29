//go:build darwin && arm64 && cgo

// metal.m — the Objective-C side of the Metal GPU GEMM backend. It owns one MTLDevice +
// command queue, a table of f16 weight matrices resident in unified memory, and runs each
// prefill projection as an MPSMatrixMultiplication (Apple's hand-tuned GEMM micro-kernel —
// the arm64 twin of the register-blocked tile fak's CPU lane never had). Weights are stored
// f16 (2 B/param, so a 7B fits in 14 GB of the 36 GB unified pool); activations are converted
// f32->f16 per call and results f16->f32, the round-trip the unified-memory shared buffers
// make a coherency flush rather than a copy.
//
// ARC note: object references stored in C structs/arrays are NOT managed by ARC, so weight
// buffers are kept alive across calls with CFBridgingRetain and released with
// CFBridgingRelease; the per-call MPS objects live inside an @autoreleasepool.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#include <CoreFoundation/CoreFoundation.h>
#include <stdlib.h> // getenv (FAK_METAL_MPS opt-in)
#include <string.h>

// f32<->f16 bulk conversion via Accelerate's vImage (SIMD): the per-call activation and
// result round-trip was a single-threaded scalar loop and measured as a large chunk of the
// "gemm+roundtrip" time. vImage treats the flat array as a 1-row image. (Non-static: shared
// with forward.m's GPU-resident path.)
void mg_f32_to_f16(const float *src, __fp16 *dst, long n) {
    vImage_Buffer s = {(void *)src, 1, (vImagePixelCount)n, (size_t)n * 4};
    vImage_Buffer d = {(void *)dst, 1, (vImagePixelCount)n, (size_t)n * 2};
    vImageConvert_PlanarFtoPlanar16F(&s, &d, 0);
}
void mg_f16_to_f32(const __fp16 *src, float *dst, long n) {
    vImage_Buffer s = {(void *)src, 1, (vImagePixelCount)n, (size_t)n * 2};
    vImage_Buffer d = {(void *)dst, 1, (vImagePixelCount)n, (size_t)n * 4};
    vImageConvert_Planar16FtoPlanarF(&s, &d, 0);
}

// Non-static: the device, queue, and weight table are shared with forward.m (the
// GPU-resident prefill), which encodes onto the same queue and reads the same f16 weights.
id<MTLDevice>       gDev = nil;
id<MTLCommandQueue> gQueue = nil;

// gMPSOK records whether MetalPerformanceShaders was validated for gDev. It is a SEPARATE,
// opt-in capability from the base device+queue: MPSSupportsMTLDevice can hard-fault the process
// in a headless / SSH context (it reaches into display-server state), so mg_init only probes it
// when FAK_METAL_MPS is set. The q4_k prefill path (q4k.m / mg_q4k_*) is pure MSL and needs NO
// MPS, so it stays usable with gMPSOK==0; only the f16 MPS GEMM entry points (mg_matmul here,
// mg_prefill in forward.m) require gMPSOK and refuse cleanly when it is off. This is what lets the
// #1085 q4_k GPU prefill run on a remote/headless Apple-Silicon box (where the device + queue +
// MSL compile + dispatch all work, but MPSSupportsMTLDevice would otherwise kill the process).
int gMPSOK = 0;

typedef struct {
    CFTypeRef buf; // retained id<MTLBuffer>, f16 [out, in] row-major
    int out;
    int in;
} MGWeight;

#define MG_MAX_W 8192
MGWeight gW[MG_MAX_W];
int gNW = 0;

// Reused scratch for the f16 activation (A) and result (C) of the current matmul. Grown on
// demand; sized in *elements* (f16 = 2 bytes each).
static id<MTLBuffer> gXBuf = nil; static int gXCap = 0;
static id<MTLBuffer> gYBuf = nil; static int gYCap = 0;

int mg_init(void) {
    if (gDev != nil) return 1;
    gDev = MTLCreateSystemDefaultDevice();
    if (gDev == nil) return 0;
    gQueue = [gDev newCommandQueue];
    if (gQueue == nil) { gDev = nil; return 0; }
    // MPS is a separate, optional capability — see gMPSOK. MPSSupportsMTLDevice can hard-fault in
    // a headless/SSH context, so only probe it when the operator opts in (FAK_METAL_MPS). Default
    // OFF keeps the pure-MSL q4_k prefill path (the #1085 GPU GEMM) fully usable headless; the f16
    // MPS GEMM paths refuse cleanly when gMPSOK is 0.
    gMPSOK = 0;
    if (getenv("FAK_METAL_MPS") != NULL) {
        gMPSOK = MPSSupportsMTLDevice(gDev) ? 1 : 0;
    }
    return 1;
}

int mg_mps_available(void) {
    if (!mg_init()) return 0;
    return gMPSOK ? 1 : 0;
}

int mg_device_name(char *name, int namelen) {
    if (!mg_init() || name == NULL || namelen <= 0) return 0;
    const char *dn = [[gDev name] UTF8String];
    if (dn) {
        strncpy(name, dn, (size_t)(namelen - 1));
        name[namelen - 1] = '\0';
    } else {
        name[0] = '\0';
    }
    return 1;
}

int mg_device_memory_total(unsigned long long *total) {
    if (!mg_init() || total == NULL) return 0;
    if (![gDev respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) return 0;
    unsigned long long v = (unsigned long long)[gDev recommendedMaxWorkingSetSize];
    if (v == 0) return 0;
    *total = v;
    return 1;
}

// mg_upload converts a row-major f32 weight [out, in] to f16 in a shared MTLBuffer and
// returns an integer handle (>=0), or -1 on failure. The f16 cast rounds each already-Q8_0
// dequantized weight to half precision (the values the CPU Q8 path multiplies as int8*scale).
int mg_upload(const float *w, int out, int in) {
    if (gDev == nil) return -1;
    if (gNW >= MG_MAX_W) {
        // Don't fail silently: the caller (metalWeights) turns a -1 into a panic, and a
        // silent cap is indistinguishable from an OOM. Warn once with the cause so the
        // fix is obvious (raise MG_MAX_W, or mg_reset between independent model loads).
        static int capWarned = 0;
        if (!capWarned) {
            capWarned = 1;
            NSLog(@"mg_upload: weight table full (MG_MAX_W=%d) — refusing upload; raise the "
                  @"cap or call mg_reset between models", MG_MAX_W);
        }
        return -1;
    }
    long n = (long)out * (long)in;
    id<MTLBuffer> b = [gDev newBufferWithLength:(NSUInteger)(n * 2)
                                        options:MTLResourceStorageModeShared];
    if (b == nil) {
        NSLog(@"mg_upload: device buffer alloc failed for %ld f16 elems (%.1f MB) — "
              @"out of unified memory?", n, (double)n * 2.0 / 1e6);
        return -1;
    }
    mg_f32_to_f16(w, (__fp16 *)b.contents, n);
    int id = gNW++;
    gW[id].buf = CFBridgingRetain(b);
    gW[id].out = out;
    gW[id].in = in;
    return id;
}

// mg_matmul computes Y[P, out] = X[P, in] * W[wid]^T (W stored [out, in], transposeRight=YES),
// f16 inputs, f32 internal accumulation (MPS), f16 result converted back to the f32 y.
void mg_matmul(int wid, const float *x, int P, float *y) {
    if (wid < 0 || wid >= gNW) return;
    if (!gMPSOK) {
        // The f16 GEMM is an MPSMatrixMultiplication; MPS was not validated for this device
        // (default headless posture — see gMPSOK / FAK_METAL_MPS). The caller (the f16 Metal
        // prefill) is expected to fall back; the q4_k MSL path does not reach here. Warn once.
        static int mpsWarned = 0;
        if (!mpsWarned) {
            mpsWarned = 1;
            NSLog(@"mg_matmul: MPS unavailable (FAK_METAL_MPS not set or unsupported); the f16 "
                  @"GEMM is disabled — use the q4_k prefill path (FAK_Q4K=1).");
        }
        return;
    }
    @autoreleasepool {
        MGWeight W = gW[wid];
        int in = W.in, out = W.out;
        id<MTLBuffer> wbuf = (__bridge id<MTLBuffer>)W.buf;
        // Grow the reused f16 scratch for the activation (A) and result (C) as needed.
        // (Assigning the global strong id directly avoids ARC write-back on &global.)
        if (gXBuf == nil || gXCap < P * in) {
            gXBuf = [gDev newBufferWithLength:(NSUInteger)((long)P * in * 2)
                                      options:MTLResourceStorageModeShared];
            gXCap = P * in;
        }
        if (gYBuf == nil || gYCap < P * out) {
            gYBuf = [gDev newBufferWithLength:(NSUInteger)((long)P * out * 2)
                                      options:MTLResourceStorageModeShared];
            gYCap = P * out;
        }
        id<MTLBuffer> xb = gXBuf;
        id<MTLBuffer> yb = gYBuf;

        mg_f32_to_f16(x, (__fp16 *)xb.contents, (long)P * in);

        MPSMatrixDescriptor *da = [MPSMatrixDescriptor matrixDescriptorWithRows:P columns:in
                                    rowBytes:(NSUInteger)in * 2 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *db = [MPSMatrixDescriptor matrixDescriptorWithRows:out columns:in
                                    rowBytes:(NSUInteger)in * 2 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *dc = [MPSMatrixDescriptor matrixDescriptorWithRows:P columns:out
                                    rowBytes:(NSUInteger)out * 2 dataType:MPSDataTypeFloat16];
        MPSMatrix *A = [[MPSMatrix alloc] initWithBuffer:xb descriptor:da];
        MPSMatrix *B = [[MPSMatrix alloc] initWithBuffer:wbuf descriptor:db];
        MPSMatrix *C = [[MPSMatrix alloc] initWithBuffer:yb descriptor:dc];

        MPSMatrixMultiplication *mm =
            [[MPSMatrixMultiplication alloc] initWithDevice:gDev
                                              transposeLeft:NO
                                             transposeRight:YES
                                                 resultRows:P
                                              resultColumns:out
                                            interiorColumns:in
                                                      alpha:1.0
                                                       beta:0.0];
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        [mm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C];
        [cb commit];
        [cb waitUntilCompleted];

        mg_f16_to_f32((__fp16 *)yb.contents, y, (long)P * out);
    }
}

void mg_free(int wid) {
    if (wid < 0 || wid >= gNW) return;
    if (gW[wid].buf != NULL) {
        CFBridgingRelease(gW[wid].buf);
        gW[wid].buf = NULL;
    }
}

// mg_fwd_reset (forward.m) clears the GPU-resident forward's per-model topology.
void mg_fwd_reset(void);

// mg_reset tears down ALL resident Metal state: it releases every uploaded weight
// buffer (the f16 projection/norm/bias set), frees the reused matmul scratch, and
// clears the GPU-resident forward's per-model wiring — returning the weight table to
// empty (gNW == 0). The device, command queue, and compiled MSL pipelines stay live
// (they are model-independent). Call ONLY when no weight handle is still in use: every
// prior id is invalidated. Its purpose is to stop the f16 weight set from accumulating
// across in-process model reloads — the per-load leak that, stacked across concurrent
// processes, helped exhaust unified memory and panic the box on 2026-06-18.
void mg_reset(void) {
    for (int i = 0; i < gNW; i++) {
        if (gW[i].buf != NULL) {
            CFBridgingRelease(gW[i].buf);
            gW[i].buf = NULL;
        }
    }
    gNW = 0;
    gXBuf = nil; gXCap = 0;   // ARC releases the prior scratch buffers
    gYBuf = nil; gYCap = 0;
    mg_fwd_reset();
}
