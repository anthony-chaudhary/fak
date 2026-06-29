//go:build darwin && arm64 && cgo

// metal_shim.m — the Metal / MetalPerformanceShaders hardware seam behind the typed
// compute.Backend, compiled DIRECTLY by cgo's clang (no offline step, unlike CUDA's nvcc
// or Vulkan's glslc — Objective-C + an embedded MSL source string build in-process). It
// mirrors cuda_kernels.cu function-for-function: every op is f32, and this is an *Approx*
// peer of the cpuref *Reference* — held to argmax-exact + logit-cosine, NOT max|Δ|=0.
// MPSMatrixMultiplication (a different reduction order than the model's fdot tree) is what
// makes that distinction real and honest.
//
// The device + command queue are owned by internal/metalgemm (gDev/gQueue): compute's
// registry backend and the model-engine GEMM/decode lane now share one Metal singleton.
// Memory management in this file is MANUAL (MRC — cgo compiles .m without -fobjc-arc):
// "new…"/"alloc" objects are +1 owned and released explicitly; each op body is wrapped
// in an @autoreleasepool so transient descriptors/command buffers don't accumulate.
// Buffers use MTLResourceStorageModeShared (Apple Silicon unified memory), so host<->device
// transfers are plain memcpy over [buffer contents]; every op commits+waits synchronously
// (the async/stream seam is a tracked follow-up), so contents are always coherent on return.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <string.h>
#include "metal_backend.h"

extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;
extern int gMPSOK;
int mg_init(void);

static id<MTLComputePipelineState> g_rmsnorm;
static id<MTLComputePipelineState> g_rope;
static id<MTLComputePipelineState> g_swiglu;
static id<MTLComputePipelineState> g_add;
static id<MTLComputePipelineState> g_addbias;
static id<MTLComputePipelineState> g_attention;
static id<MTLComputePipelineState> g_argmax;

// Embedded Metal Shading Language source for the elementwise/reduction kernels. The GEMM
// path uses MPSMatrixMultiplication, not a kernel here. Each kernel reproduces the cpuref
// arithmetic closely enough for the Approx gate; exact reduction order is NOT promised
// (and need not be — RequireReference keeps a device off the bit-identity rungs).
static const char *kShaderSrc =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "kernel void rmsnorm_f32(device const float* x [[buffer(0)]],\n"
    "                        device const float* w [[buffer(1)]],\n"
    "                        device float* y [[buffer(2)]],\n"
    "                        constant int& n [[buffer(3)]],\n"
    "                        constant float& eps [[buffer(4)]],\n"
    "                        uint r [[thread_position_in_grid]]) {\n"
    "    uint base = (uint)n * r;\n"
    "    float ss = 0.0f;\n"
    "    for (int i = 0; i < n; i++) { float v = x[base + (uint)i]; ss += v * v; }\n"
    "    float inv = rsqrt(ss / (float)n + eps);\n"
    "    for (int i = 0; i < n; i++) { y[base + (uint)i] = x[base + (uint)i] * inv * w[i]; }\n"
    "}\n"
    "\n"
    "kernel void rope_f32(device float* x [[buffer(0)]],\n"
    "                     constant int& pos [[buffer(1)]],\n"
    "                     constant int& nHeads [[buffer(2)]],\n"
    "                     constant int& headDim [[buffer(3)]],\n"
    "                     constant float& theta [[buffer(4)]],\n"
    "                     uint gid [[thread_position_in_grid]]) {\n"
    "    int hf = headDim / 2;\n"
    "    if (hf == 0) return;\n"
    "    int h = (int)gid / hf;\n"
    "    int j = (int)gid % hf;\n"
    "    if (h >= nHeads) return;\n"
    "    float freq = pow(theta, -((float)(2 * j)) / (float)headDim);\n"
    "    float ang = (float)pos * freq;\n"
    "    float c = cos(ang);\n"
    "    float s = sin(ang);\n"
    "    uint b = (uint)(h * headDim);\n"
    "    float a0 = x[b + (uint)j];\n"
    "    float a1 = x[b + (uint)(j + hf)];\n"
    "    x[b + (uint)j]        = a0 * c - a1 * s;\n"
    "    x[b + (uint)(j + hf)] = a1 * c + a0 * s;\n"
    "}\n"
    "\n"
    "kernel void swiglu_f32(device const float* g [[buffer(0)]],\n"
    "                       device const float* u [[buffer(1)]],\n"
    "                       device float* y [[buffer(2)]],\n"
    "                       uint i [[thread_position_in_grid]]) {\n"
    "    float z = g[i];\n"
    "    float sl = z / (1.0f + exp(-z));\n"
    "    y[i] = sl * u[i];\n"
    "}\n"
    "\n"
    "kernel void add_f32(device float* dst [[buffer(0)]],\n"
    "                    device const float* src [[buffer(1)]],\n"
    "                    uint i [[thread_position_in_grid]]) {\n"
    "    dst[i] += src[i];\n"
    "}\n"
    "\n"
    "kernel void add_bias_f32(device float* dst [[buffer(0)]],\n"
    "                         device const float* bias [[buffer(1)]],\n"
    "                         constant int& width [[buffer(2)]],\n"
    "                         uint i [[thread_position_in_grid]]) {\n"
    "    dst[i] += bias[(int)i % width];\n"
    "}\n"
    "\n"
    "kernel void attention_f32(device const float* q [[buffer(0)]],\n"
    "                          device const float* K [[buffer(1)]],\n"
    "                          device const float* V [[buffer(2)]],\n"
    "                          device float* outp [[buffer(3)]],\n"
    "                          constant int& nPos [[buffer(4)]],\n"
    "                          constant int& nH [[buffer(5)]],\n"
    "                          constant int& nKV [[buffer(6)]],\n"
    "                          constant int& hd [[buffer(7)]],\n"
    "                          constant float& scale [[buffer(8)]],\n"
    "                          uint h [[thread_position_in_grid]]) {\n"
    "    if ((int)h >= nH) return;\n"
    "    int grp = nH / nKV;\n"
    "    int kvh = (int)h / grp;\n"
    "    int w = nKV * hd;\n"
    "    uint qbase = h * (uint)hd;\n"
    "    float m = -3.0e38f;\n"
    "    float l = 0.0f;\n"
    "    float acc[256];\n"
    "    for (int d = 0; d < hd; d++) acc[d] = 0.0f;\n"
    "    for (int j = 0; j < nPos; j++) {\n"
    "        uint kvbase = (uint)(j * w + kvh * hd);\n"
    "        float s = 0.0f;\n"
    "        for (int d = 0; d < hd; d++) s += q[qbase + (uint)d] * K[kvbase + (uint)d];\n"
    "        s *= scale;\n"
    "        float mnew = max(m, s);\n"
    "        float corr = exp(m - mnew);\n"
    "        float p = exp(s - mnew);\n"
    "        l = l * corr + p;\n"
    "        for (int d = 0; d < hd; d++) acc[d] = acc[d] * corr + p * V[kvbase + (uint)d];\n"
    "        m = mnew;\n"
    "    }\n"
    "    float invl = (l > 0.0f) ? (1.0f / l) : 0.0f;\n"
    "    uint obase = h * (uint)hd;\n"
    "    for (int d = 0; d < hd; d++) outp[obase + (uint)d] = acc[d] * invl;\n"
    "}\n"
    "\n"
    "kernel void argmax_f32(device const float* x [[buffer(0)]],\n"
    "                       device int* result [[buffer(1)]],\n"
    "                       constant int& n [[buffer(2)]],\n"
    "                       uint tid [[thread_position_in_grid]]) {\n"
    "    if (tid != 0) return;\n"
    "    int bi = 0;\n"
    "    float bv = x[0];\n"
    "    for (int i = 1; i < n; i++) { if (x[i] > bv) { bv = x[i]; bi = i; } }\n"
    "    result[0] = bi;\n"
    "}\n";

static id<MTLComputePipelineState> make_pso(id<MTLLibrary> lib, const char *fname) {
    NSString *n = [NSString stringWithUTF8String:fname];
    id<MTLFunction> fn = [lib newFunctionWithName:n]; // +1
    if (!fn) {
        NSLog(@"fak metal: missing kernel %s", fname);
        return nil;
    }
    NSError *err = nil;
    id<MTLComputePipelineState> pso = [gDev newComputePipelineStateWithFunction:fn error:&err]; // +1
    [fn release];
    if (!pso) {
        NSLog(@"fak metal: pipeline %s failed: %@", fname, err);
        return nil;
    }
    return pso; // caller keeps (+1)
}

int fmetal_init(char *name, int namelen) {
    @autoreleasepool {
        if (mg_init() != 1 || gDev == nil || gQueue == nil || !gMPSOK) {
            return 1;
        }
        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:kShaderSrc];
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> lib = [gDev newLibraryWithSource:src options:opts error:&err]; // +1
        [opts release];
        if (!lib) {
            NSLog(@"fak metal: MSL library compile failed: %@", err);
            return 1;
        }
        g_rmsnorm = make_pso(lib, "rmsnorm_f32");
        g_rope = make_pso(lib, "rope_f32");
        g_swiglu = make_pso(lib, "swiglu_f32");
        g_add = make_pso(lib, "add_f32");
        g_addbias = make_pso(lib, "add_bias_f32");
        g_attention = make_pso(lib, "attention_f32");
        g_argmax = make_pso(lib, "argmax_f32");
        [lib release];
        if (!g_rmsnorm || !g_rope || !g_swiglu || !g_add || !g_addbias || !g_attention || !g_argmax) {
            return 1;
        }
        const char *dn = [[gDev name] UTF8String];
        if (name && namelen > 0) {
            if (dn) {
                strncpy(name, dn, (size_t)(namelen - 1));
                name[namelen - 1] = '\0';
            } else {
                name[0] = '\0';
            }
        }
        return 0;
    }
}

// ---- residency / transfers (unified shared memory) ------------------------------

void *fmetal_malloc(size_t bytes) {
    if (bytes == 0) {
        bytes = 16;
    }
    id<MTLBuffer> b = [gDev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; // +1 owned
    return (void *)b;
}

void fmetal_free(void *buf) {
    if (!buf) {
        return;
    }
    id<MTLBuffer> b = (id<MTLBuffer>)buf;
    [b release];
}

int fmetal_device_memory_total(unsigned long long *total) {
    if (!gDev || !total) {
        return 1;
    }
    if (![gDev respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
        return 1;
    }
    unsigned long long v = (unsigned long long)[gDev recommendedMaxWorkingSetSize];
    if (v == 0) {
        return 1;
    }
    *total = v;
    return 0;
}

void fmetal_h2d(void *dstBuf, const void *host, size_t bytes) {
    id<MTLBuffer> d = (id<MTLBuffer>)dstBuf;
    memcpy([d contents], host, bytes);
}

void fmetal_d2h(void *host, void *srcBuf, size_t bytes) {
    id<MTLBuffer> s = (id<MTLBuffer>)srcBuf;
    memcpy(host, [s contents], bytes);
}

void fmetal_copy_at(void *dstBuf, size_t dstOff, void *srcBuf, size_t srcOff, size_t bytes) {
    id<MTLBuffer> d = (id<MTLBuffer>)dstBuf;
    id<MTLBuffer> s = (id<MTLBuffer>)srcBuf;
    memcpy((char *)[d contents] + dstOff, (char *)[s contents] + srcOff, bytes);
}

// ---- helpers --------------------------------------------------------------------

// dispatch a 1-D grid of `gridN` threads with `pso` over `enc`, then end/commit/wait.
static void run_1d(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc,
                   id<MTLComputePipelineState> pso, NSUInteger gridN) {
    NSUInteger tw = pso.maxTotalThreadsPerThreadgroup;
    NSUInteger g = gridN;
    if (g == 0) {
        g = 1;
    }
    if (tw > g) {
        tw = g;
    }
    if (tw == 0) {
        tw = 1;
    }
    [enc dispatchThreads:MTLSizeMake(g, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// ---- primitives -----------------------------------------------------------------

void fmetal_matmul_f32(void *dW, void *dX, void *dY, int out, int in, int P) {
    @autoreleasepool {
        id<MTLBuffer> bW = (id<MTLBuffer>)dW;
        id<MTLBuffer> bX = (id<MTLBuffer>)dX;
        id<MTLBuffer> bY = (id<MTLBuffer>)dY;
        NSUInteger es = sizeof(float);
        MPSMatrixDescriptor *dA = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)P
                                                                        columns:(NSUInteger)in
                                                                       rowBytes:(NSUInteger)in * es
                                                                       dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dB = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)out
                                                                        columns:(NSUInteger)in
                                                                       rowBytes:(NSUInteger)in * es
                                                                       dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dC = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)P
                                                                        columns:(NSUInteger)out
                                                                       rowBytes:(NSUInteger)out * es
                                                                       dataType:MPSDataTypeFloat32];
        MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:bX descriptor:dA];
        MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:bW descriptor:dB];
        MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:bY descriptor:dC];
        // C[P,out] = A[P,in] * B[out,in]^T  (transposeRight=YES, interiorColumns=in).
        MPSMatrixMultiplication *mul =
            [[MPSMatrixMultiplication alloc] initWithDevice:gDev
                                              transposeLeft:NO
                                             transposeRight:YES
                                                 resultRows:(NSUInteger)P
                                              resultColumns:(NSUInteger)out
                                            interiorColumns:(NSUInteger)in
                                                      alpha:1.0
                                                       beta:0.0];
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        [mul encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        [cb commit];
        [cb waitUntilCompleted];
        [mA release];
        [mB release];
        [mC release];
        [mul release];
    }
}

void fmetal_rmsnorm_f32(void *dX, void *dW, void *dY, int rows, int n, float eps) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_rmsnorm];
        [enc setBuffer:(id<MTLBuffer>)dX offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)dW offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)dY offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(int) atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];
        run_1d(cb, enc, g_rmsnorm, (NSUInteger)rows);
    }
}

void fmetal_rope_f32(void *dX, int pos, int nHeads, int headDim, float theta) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_rope];
        [enc setBuffer:(id<MTLBuffer>)dX offset:0 atIndex:0];
        [enc setBytes:&pos length:sizeof(int) atIndex:1];
        [enc setBytes:&nHeads length:sizeof(int) atIndex:2];
        [enc setBytes:&headDim length:sizeof(int) atIndex:3];
        [enc setBytes:&theta length:sizeof(float) atIndex:4];
        run_1d(cb, enc, g_rope, (NSUInteger)(nHeads * (headDim / 2)));
    }
}

void fmetal_swiglu_f32(void *dG, void *dU, void *dY, int n) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_swiglu];
        [enc setBuffer:(id<MTLBuffer>)dG offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)dU offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)dY offset:0 atIndex:2];
        run_1d(cb, enc, g_swiglu, (NSUInteger)n);
    }
}

void fmetal_add_f32(void *dDst, void *dSrc, int n) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_add];
        [enc setBuffer:(id<MTLBuffer>)dDst offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)dSrc offset:0 atIndex:1];
        run_1d(cb, enc, g_add, (NSUInteger)n);
    }
}

void fmetal_add_bias_f32(void *dDst, void *dBias, int rows, int width) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_addbias];
        [enc setBuffer:(id<MTLBuffer>)dDst offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)dBias offset:0 atIndex:1];
        [enc setBytes:&width length:sizeof(int) atIndex:2];
        run_1d(cb, enc, g_addbias, (NSUInteger)(rows * width));
    }
}

void fmetal_attention_f32(void *dQ, void *dK, void *dV, void *dOut,
                          int nPos, int nH, int nKV, int hd, float scale) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_attention];
        [enc setBuffer:(id<MTLBuffer>)dQ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)dK offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)dV offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)dOut offset:0 atIndex:3];
        [enc setBytes:&nPos length:sizeof(int) atIndex:4];
        [enc setBytes:&nH length:sizeof(int) atIndex:5];
        [enc setBytes:&nKV length:sizeof(int) atIndex:6];
        [enc setBytes:&hd length:sizeof(int) atIndex:7];
        [enc setBytes:&scale length:sizeof(float) atIndex:8];
        run_1d(cb, enc, g_attention, (NSUInteger)nH);
    }
}

int fmetal_argmax_f32(void *dLogits, int n) {
    @autoreleasepool {
        id<MTLBuffer> rbuf = [gDev newBufferWithLength:sizeof(int)
                                                  options:MTLResourceStorageModeShared]; // +1
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_argmax];
        [enc setBuffer:(id<MTLBuffer>)dLogits offset:0 atIndex:0];
        [enc setBuffer:rbuf offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(int) atIndex:2];
        run_1d(cb, enc, g_argmax, 1);
        int r = *((int *)[rbuf contents]);
        [rbuf release];
        return r;
    }
}
