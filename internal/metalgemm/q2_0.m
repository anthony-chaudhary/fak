//go:build darwin && arm64 && cgo

// q2_0.m — the Metal ternary (Q2_0) dequant-GEMV. The 2-bit twin of q8.m/q4k.m, and the Apple-side
// half of Bonsai's on-device headline (epic #4867, issue #4873): Ternary-Bonsai-27B ships its
// weights as {-1,0,+1}·d, and this kernel is what makes that 27B usable on Apple Silicon without
// expanding it to FP16. At ~0.375 B/weight resident (8 code bytes + one f32 scale per 32-wide
// block) a 27B fits the ~4 GB on-device envelope that f16 (~54 GB) and even q4_k_m (~16 GB) do not.
//
// Format (the GGUF Q2_0 g32 form, byte-for-byte internal/model.quant_q2.go). A weight is [out, in]
// row-major with in = nblk*32. Each 32-wide block is one f32 scale d plus 8 code bytes holding 32
// signed 2-bit codes, four per byte, LOW CODE FIRST. A code c (0..3) dequantizes to d*(c-2). Because
// d = amax (the block's peak magnitude), quantizing a block only ever emits c in {1,2,3} — the -2
// slot is unreachable — so the live code set is exactly the ternary {-1, 0, +1}·d. Unpack happens
// IN-SHADER: the raw 2-bit blocks stay resident and each thread expands its own codes on the fly,
// so the bandwidth-dominant stream is the 8 code bytes per block and the GPU never sees an f16
// expansion of the weight.
//
// Correctness target. One threadgroup (a single 32-lane SIMD group) per output row; the 32 lanes
// split the row's nblk blocks and reduce via simd_sum. Each block unpacks its 32 codes, dots the
// (c-2) integers against the f32 activation, and scales the block sum by d — the same value as
// internal/model.q2MatRows (which dequants to d*(c-2) first, then dots); only the factoring of d
// and the simd_sum reduction order differ, which is why parity is within-tolerance rather than
// bit-exact. The activation is plain f32 (like q4k_gemv, unlike q8_gemv's quantized activation)
// because the CPU-ref ternary GEMV takes f32 x. Pinned by TestMetalQ2_0GemvMatchesCPU on an Apple
// host; the reference's own math obligations are pinned in every build by q2_0_witness_test.go.
//
// Shares gDev/gQueue with metal.m (one device, one queue). Its weight table (2-bit codes + f32
// block scales) is separate from the f16 (metal.m), raw-q4_k (q4k.m), and Q8 (q8.m) tables, with
// its own teardown via mg_q2_0_reset.

#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>
#include <string.h>

// Device + queue are owned by metal.m (mg_init); we reuse them.
extern id<MTLDevice>       gDev;
extern id<MTLCommandQueue> gQueue;

static NSString *kQ2Src = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// q2_0_gemv: the ternary decode GEMV. ONE threadgroup (single 32-lane SIMD group) per output row.
// The 32 lanes split the row's nblk blocks; each lane unpacks its blocks' 2-bit codes in-shader,
// accumulates sum_i (code_i - 2) * x_i over the 32-wide block, scales that block sum by the block
// scale d, and adds it to a per-lane float accumulator. simd_sum then reduces across the lanes.
// This is the in-kernel twin of internal/model.q2MatRows.
kernel void q2_0_gemm(device const uchar* codes [[buffer(0)]],
                        device const float* scales [[buffer(1)]],
                        device const float* X [[buffer(2)]],
                        device float* Y [[buffer(3)]],
                        constant int& nblk [[buffer(4)]],
                        constant int& out [[buffer(5)]],
                        constant int& p [[buffer(6)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint o = gid.x, t = gid.y;
    if (o >= (uint)out || t >= (uint)p) return;
    float acc = 0.0f;
    const uint row = o * (uint)nblk;
    device const float* x = X + t * (uint)nblk * 32u;
    for (uint b = 0; b < (uint)nblk; b++) {
        const float d = scales[row + b];
        device const uchar* q = codes + (row + b) * 8u;
        device const float* xb = x + b * 32u;
        for (uint j = 0; j < 8u; j++) {
            uchar v = q[j];
            uint k = j * 4u;
            acc += d * (float(int(v & 3u) - 2) * xb[k] + float(int((v >> 2) & 3u) - 2) * xb[k+1] +
                        float(int((v >> 4) & 3u) - 2) * xb[k+2] + float(int((v >> 6) & 3u) - 2) * xb[k+3]);
        }
    }
    Y[t * (uint)out + o] = acc;
}

kernel void q2_0_mlp_gate(device const uchar* gc [[buffer(0)]], device const float* gs [[buffer(1)]],
                           device const uchar* uc [[buffer(2)]], device const float* us [[buffer(3)]],
                           device const float* x [[buffer(4)]], device float* inter [[buffer(5)]],
                           constant int& nblk [[buffer(6)]], constant int& hidden [[buffer(7)]],
                           uint o [[thread_position_in_grid]]) {
    if (o >= (uint)hidden) return;
    float g=0.0f,u=0.0f; uint row=o*(uint)nblk;
    for(uint b=0;b<(uint)nblk;b++){ device const float* xb=x+b*32u; uchar vg,vu; float dg=gs[row+b],du=us[row+b];
      for(uint j=0;j<8u;j++){ vg=gc[(row+b)*8u+j]; vu=uc[(row+b)*8u+j]; uint k=j*4u;
        g+=dg*(float(int(vg&3u)-2)*xb[k]+float(int((vg>>2)&3u)-2)*xb[k+1]+float(int((vg>>4)&3u)-2)*xb[k+2]+float(int((vg>>6)&3u)-2)*xb[k+3]);
        u+=du*(float(int(vu&3u)-2)*xb[k]+float(int((vu>>2)&3u)-2)*xb[k+1]+float(int((vu>>4)&3u)-2)*xb[k+2]+float(int((vu>>6)&3u)-2)*xb[k+3]); }}
    inter[o]=(g/(1.0f+exp(-g)))*u;
}

kernel void q2_0_mlp_down(device const uchar* codes [[buffer(0)]], device const float* scales [[buffer(1)]],
                           device const float* x [[buffer(2)]], device float* y [[buffer(3)]],
                           constant int& nblk [[buffer(4)]], constant int& out [[buffer(5)]],
                           uint o [[thread_position_in_grid]]) {
    if(o >= (uint)out) return; float acc=0.0f; uint row=o*(uint)nblk;
    for(uint b=0;b<(uint)nblk;b++){ float d=scales[row+b]; device const uchar* q=codes+(row+b)*8u; device const float* xb=x+b*32u;
      for(uint j=0;j<8u;j++){ uchar v=q[j]; uint k=j*4u; acc+=d*(float(int(v&3u)-2)*xb[k]+float(int((v>>2)&3u)-2)*xb[k+1]+float(int((v>>4)&3u)-2)*xb[k+2]+float(int((v>>6)&3u)-2)*xb[k+3]); }} y[o]=acc;
}

kernel void q2_0_gemv(device const uchar* W    [[buffer(0)]],  // out*nblk*8 packed 2-bit codes, 4/byte low-first
                      device const float* WD   [[buffer(1)]],  // out*nblk weight block-scales
                      device const float* X    [[buffer(2)]],  // in f32 activations
                      device float*       Y    [[buffer(3)]],
                      constant int&       nblk [[buffer(4)]],
                      constant int&       out_ [[buffer(5)]],
                      uint o   [[threadgroup_position_in_grid]],
                      uint lid [[thread_index_in_threadgroup]]) {
    if (o >= (uint)out_) return;
    device const uchar* wrow = W  + (long)o * nblk * 8;
    device const float* wd   = WD + (long)o * nblk;
    float acc = 0.0f;
    for (int b = (int)lid; b < nblk; b += 32) {
        device const uchar* wb = wrow + (long)b * 8;
        device const float* xb = X    + (long)b * 32;
        float s = 0.0f;
        // 8 bytes -> 32 codes. Low code first, matching internal/model.dequantQ2Block.
        for (int q = 0; q < 8; q++) {
            uchar bits = wb[q];
            s += (float)((int)( bits       & 0x3) - 2) * xb[4*q+0];
            s += (float)((int)((bits >> 2) & 0x3) - 2) * xb[4*q+1];
            s += (float)((int)((bits >> 4) & 0x3) - 2) * xb[4*q+2];
            s += (float)((int)((bits >> 6) & 0x3) - 2) * xb[4*q+3];
        }
        acc += s * wd[b];
    }
    acc = simd_sum(acc);
    if (lid == 0) Y[o] = acc;
}

// q2_0_g128_block_dot: dot one 34-byte GGUF group-128 Q2_0 block (128 weights) against 128 activation elements.
// Layout: 2 bytes f16 scale d + 32 bytes 2-bit codes (4/byte low-first).
// Dequant: (code - 1) * d for code in {0, 1, 2}.
inline float q2_0_g128_block_dot(device const uchar* blk, device const float* xs) {
    float d = (float)(*(device const half*)blk);
    device const uchar* qs = blk + 2;
    float s = 0.0f;
    for (int i = 0; i < 32; i++) {
        uchar b = qs[i];
        int k = i * 4;
        s += (float)((int)( b       & 0x3) - 1) * xs[k + 0];
        s += (float)((int)((b >> 2) & 0x3) - 1) * xs[k + 1];
        s += (float)((int)((b >> 4) & 0x3) - 1) * xs[k + 2];
        s += (float)((int)((b >> 6) & 0x3) - 1) * xs[k + 3];
    }
    return s * d;
}

// q2_0_g128_gemv: decode GEMV for group-128 GGUF Q2_0 weights.
// ONE 32-lane SIMD group per output row. Lanes stride over nblk blocks and reduce via simd_sum.
kernel void q2_0_g128_gemv(device const uchar* W [[buffer(0)]],
                          device const float* X [[buffer(1)]],
                          device float*       Y [[buffer(2)]],
                          constant int&    nblk [[buffer(3)]],
                          constant int&     out [[buffer(4)]],
                          uint o   [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]]) {
    if (o >= (uint)out) return;
    device const uchar* row = W + (long)o * nblk * 34;
    float acc = 0.0f;
    for (int b = (int)lid; b < nblk; b += 32) {
        acc += q2_0_g128_block_dot(row + (long)b * 34, X + (long)b * 128);
    }
    acc = simd_sum(acc);
    if (lid == 0) {
        Y[o] = acc;
    }
}

// q2_0_g128_gemm: prefill GEMM for group-128 GGUF Q2_0 weights.
// ONE 32-lane SIMD group per (output row, prompt token).
kernel void q2_0_g128_gemm(device const uchar* W [[buffer(0)]],
                          device const float* X [[buffer(1)]],
                          device float*       Y [[buffer(2)]],
                          constant int&    nblk [[buffer(3)]],
                          constant int&     out [[buffer(4)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint  lid [[thread_index_in_threadgroup]]) {
    uint o = tg.x;
    uint t = tg.y;
    if (o >= (uint)out) return;
    device const uchar* row = W + (long)o * nblk * 34;
    device const float* xs  = X + (long)t * nblk * 128;
    float acc = 0.0f;
    for (int b = (int)lid; b < nblk; b += 32) {
        acc += q2_0_g128_block_dot(row + (long)b * 34, xs + (long)b * 128);
    }
    acc = simd_sum(acc);
    if (lid == 0) {
        Y[(long)t * out + o] = acc;
    }
}
)MSL";

static id<MTLComputePipelineState> psoQ2Gemv;
static id<MTLComputePipelineState> psoQ2Gemm, psoQ2MLPGate, psoQ2MLPDown;
static id<MTLComputePipelineState> psoQ2G128Gemv = nil;
static id<MTLComputePipelineState> psoQ2G128Gemm = nil;
static int gQ2Ready;

static int q2_0_init(void) {
    if (gQ2Ready) return 1;
    if (gDev == nil) return 0;
    NSError *err = nil;
    id<MTLLibrary> lib = [gDev newLibraryWithSource:kQ2Src options:nil error:&err];
    if (lib == nil) { NSLog(@"q2_0: library compile failed: %@", err); return 0; }
    psoQ2Gemv = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q2_0_gemv"] error:&err];
    psoQ2Gemm = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q2_0_gemm"] error:&err];
    psoQ2MLPGate = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q2_0_mlp_gate"] error:&err];
    psoQ2MLPDown = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q2_0_mlp_down"] error:&err];
    psoQ2G128Gemv = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q2_0_g128_gemv"] error:&err];
    psoQ2G128Gemm = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q2_0_g128_gemm"] error:&err];
    if (!psoQ2Gemv || !psoQ2Gemm || !psoQ2MLPGate || !psoQ2MLPDown || !psoQ2G128Gemv || !psoQ2G128Gemm) { NSLog(@"q2_0: pipeline build failed: %@", err); return 0; }
    gQ2Ready = 1;
    return 1;
}

typedef struct {
    CFTypeRef codes;  // retained id<MTLBuffer>, uchar [out*nblk*8]
    CFTypeRef scales; // retained id<MTLBuffer>, f32   [out*nblk]
    int out, in, nblk;
} Q2W;

#define MG_MAX_Q2 8192
static Q2W gQ2[MG_MAX_Q2];
static int gNQ2 = 0;

typedef struct {
    CFTypeRef buf; // retained id<MTLBuffer>
    int out, in, nblk;
} Q2G128W;

#define MG_MAX_Q2_G128 8192
static Q2G128W gQ2G128[MG_MAX_Q2_G128];
static int gNQ2G128 = 0;

// Reused per-call scratch: the f32 activation and the f32 result. Weights are persistent; only the
// per-call X/Y move (same discipline as q8.m's gQ8XBuf/gQ8YBuf).
static id<MTLBuffer> gQ2XBuf = nil; static long gQ2XCap = 0; // activation (f32), elems
static id<MTLBuffer> gQ2YBuf = nil; static long gQ2YCap = 0; // result (f32), elems

static void q2_0_grow_scratch(long xElems, long yElems) {
    if (gQ2XBuf == nil || gQ2XCap < xElems) {
        gQ2XBuf = [gDev newBufferWithLength:(NSUInteger)(xElems * 4) options:MTLResourceStorageModeShared];
        gQ2XCap = xElems;
    }
    if (gQ2YBuf == nil || gQ2YCap < yElems) {
        gQ2YBuf = [gDev newBufferWithLength:(NSUInteger)(yElems * 4) options:MTLResourceStorageModeShared];
        gQ2YCap = yElems;
    }
}

static id<MTLBuffer> gQ2G128XBuf = nil; static long gQ2G128XCap = 0;
static id<MTLBuffer> gQ2G128YBuf = nil; static long gQ2G128YCap = 0;

static void q2_0_g128_grow_scratch(long xElems, long yElems) {
    if (gQ2G128XBuf == nil || gQ2G128XCap < xElems) {
        gQ2G128XBuf = [gDev newBufferWithLength:(NSUInteger)(xElems * 4) options:MTLResourceStorageModeShared];
        gQ2G128XCap = xElems;
    }
    if (gQ2G128YBuf == nil || gQ2G128YCap < yElems) {
        gQ2G128YBuf = [gDev newBufferWithLength:(NSUInteger)(yElems * 4) options:MTLResourceStorageModeShared];
        gQ2G128YCap = yElems;
    }
}

// mg_q2_0_upload copies a ternary Q2_0 weight (out*nblk*8 packed code bytes + out*nblk f32 block
// scales, nblk=in/32) resident onto the GPU and returns an integer handle (>=0), or -1 on failure.
int mg_q2_0_upload(const unsigned char* codes, const float* scales, int out, int in) {
    if (gDev == nil) return -1;
    if (!q2_0_init()) return -1;
    if (in % 32 != 0 || out <= 0) return -1;
    if (gNQ2 >= MG_MAX_Q2) {
        static int capWarned = 0;
        if (!capWarned) { capWarned = 1; NSLog(@"mg_q2_0_upload: q2_0 weight table full (%d)", MG_MAX_Q2); }
        return -1;
    }
    int nblk = in / 32;
    long codeBytes  = (long)out * nblk * 8;
    long scaleBytes = (long)out * nblk * 4;
    id<MTLBuffer> cb = [gDev newBufferWithLength:(NSUInteger)codeBytes  options:MTLResourceStorageModeShared];
    id<MTLBuffer> sb = [gDev newBufferWithLength:(NSUInteger)scaleBytes options:MTLResourceStorageModeShared];
    if (cb == nil || sb == nil) {
        NSLog(@"mg_q2_0_upload: device buffer alloc failed for %.1f MB", (double)(codeBytes + scaleBytes) / 1e6);
        return -1;
    }
    memcpy(cb.contents, codes,  (size_t)codeBytes);
    memcpy(sb.contents, scales, (size_t)scaleBytes);
    int id = gNQ2++;
    gQ2[id].codes  = CFBridgingRetain(cb);
    gQ2[id].scales = CFBridgingRetain(sb);
    gQ2[id].out  = out;
    gQ2[id].in   = in;
    gQ2[id].nblk = nblk;
    return id;
}

// mg_q2_0_gemv computes y[out] = W[wid] · x for one f32 activation row x[in]. f32 result.
void mg_q2_0_gemv(int wid, const float* x, float* y) {
    if (wid < 0 || wid >= gNQ2) return;
    @autoreleasepool {
        Q2W W = gQ2[wid];
        q2_0_grow_scratch((long)W.in, (long)W.out);
        memcpy(gQ2XBuf.contents, x, (size_t)W.in * 4);

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:psoQ2Gemv];
        [e setBuffer:(__bridge id<MTLBuffer>)W.codes  offset:0 atIndex:0];
        [e setBuffer:(__bridge id<MTLBuffer>)W.scales offset:0 atIndex:1];
        [e setBuffer:gQ2XBuf offset:0 atIndex:2];
        [e setBuffer:gQ2YBuf offset:0 atIndex:3];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:4];
        [e setBytes:&W.out  length:sizeof(int) atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(y, gQ2YBuf.contents, (size_t)W.out * 4);
    }
}

// mg_q2_0_gemv_batch runs n decode GEMVs of the SAME resident ternary weight over n DIFFERENT
// activation rows (Xcat, n*in f32) into ONE command buffer (one commit/waitUntilCompleted),
// writing n result rows into Ycat (n*out f32). It is the ternary twin of mg_q4k_gemv_batch: the
// measurement primitive that isolates the per-call CPU<->GPU submit/sync cost (paid once here)
// from the kernel cost (paid n times).
void mg_q2_0_gemv_batch(int wid, const float* Xcat, int n, float* Ycat) {
    if (wid < 0 || wid >= gNQ2 || n <= 0) return;
    @autoreleasepool {
        Q2W W = gQ2[wid];
        q2_0_grow_scratch((long)n * W.in, (long)n * W.out);
        memcpy(gQ2XBuf.contents, Xcat, (size_t)n * W.in * 4);

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:psoQ2Gemv];
        [e setBuffer:(__bridge id<MTLBuffer>)W.codes  offset:0 atIndex:0];
        [e setBuffer:(__bridge id<MTLBuffer>)W.scales offset:0 atIndex:1];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:4];
        [e setBytes:&W.out  length:sizeof(int) atIndex:5];
        for (int i = 0; i < n; i++) {
            [e setBuffer:gQ2XBuf offset:(NSUInteger)((long)i * W.in  * 4) atIndex:2];
            [e setBuffer:gQ2YBuf offset:(NSUInteger)((long)i * W.out * 4) atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        [e endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(Ycat, gQ2YBuf.contents, (size_t)n * W.out * 4);
    }
}

// mg_q2_0_gemv_group runs n decode GEMVs that SHARE one f32 activation x (same in) but apply n
// DIFFERENT resident ternary weights, into ONE command buffer. This is the live decode group
// pattern (q/k/v, gate/up): it pays the per-command-buffer submit/sync once for the whole group.
// Each weight i writes Ycat[yoff[i] .. yoff[i]+out_i); yoff has n+1 entries.
void mg_q2_0_gemv_group(const int* wids, int n, const float* x, float* Ycat, const int* yoff) {
    if (n <= 0) return;
    @autoreleasepool {
        int in = gQ2[wids[0]].in;
        long ytot = (long)yoff[n];
        q2_0_grow_scratch((long)in, ytot);
        memcpy(gQ2XBuf.contents, x, (size_t)in * 4);

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:psoQ2Gemv];
        [e setBuffer:gQ2XBuf offset:0 atIndex:2]; // shared activation for every weight in the group
        for (int i = 0; i < n; i++) {
            Q2W Wi = gQ2[wids[i]];
            [e setBuffer:(__bridge id<MTLBuffer>)Wi.codes  offset:0 atIndex:0];
            [e setBuffer:(__bridge id<MTLBuffer>)Wi.scales offset:0 atIndex:1];
            [e setBuffer:gQ2YBuf offset:(NSUInteger)((long)yoff[i] * 4) atIndex:3];
            [e setBytes:&Wi.nblk length:sizeof(int) atIndex:4];
            [e setBytes:&Wi.out  length:sizeof(int) atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)Wi.out, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        [e endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(Ycat, gQ2YBuf.contents, (size_t)ytot * 4);
    }
}

void mg_q2_0_gemm(int wid, const float* X, int p, float* Y) {
    @autoreleasepool {
        if (wid < 0 || wid >= gNQ2 || p <= 0 || !q2_0_init()) return;
        Q2W W=gQ2[wid]; q2_0_grow_scratch((long)p*W.in,(long)p*W.out);
        memcpy(gQ2XBuf.contents,X,(size_t)p*W.in*4); id<MTLCommandBuffer> cmd=[gQueue commandBuffer]; id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
        [e setComputePipelineState:psoQ2Gemm]; [e setBuffer:(__bridge id<MTLBuffer>)W.codes offset:0 atIndex:0]; [e setBuffer:(__bridge id<MTLBuffer>)W.scales offset:0 atIndex:1]; [e setBuffer:gQ2XBuf offset:0 atIndex:2]; [e setBuffer:gQ2YBuf offset:0 atIndex:3]; [e setBytes:&W.nblk length:4 atIndex:4]; [e setBytes:&W.out length:4 atIndex:5]; [e setBytes:&p length:4 atIndex:6];
        [e dispatchThreads:MTLSizeMake(W.out,p,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; memcpy(Y,gQ2YBuf.contents,(size_t)p*W.out*4);
    }
}

int mg_q2_0_mlp(int gate, int up, int down, const float* x, float* y) {
    @autoreleasepool {
        if(gate<0||up<0||down<0||gate>=gNQ2||up>=gNQ2||down>=gNQ2||!q2_0_init()) return 0;
        Q2W G=gQ2[gate],U=gQ2[up],D=gQ2[down]; if(G.in!=U.in||G.out!=U.out||D.in!=G.out) return 0;
        q2_0_grow_scratch(G.in,G.out>D.out?G.out:D.out); memcpy(gQ2XBuf.contents,x,(size_t)G.in*4);
        id<MTLCommandBuffer> cmd=[gQueue commandBuffer]; id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
        [e setComputePipelineState:psoQ2MLPGate]; [e setBuffer:(__bridge id<MTLBuffer>)G.codes offset:0 atIndex:0]; [e setBuffer:(__bridge id<MTLBuffer>)G.scales offset:0 atIndex:1]; [e setBuffer:(__bridge id<MTLBuffer>)U.codes offset:0 atIndex:2]; [e setBuffer:(__bridge id<MTLBuffer>)U.scales offset:0 atIndex:3]; [e setBuffer:gQ2XBuf offset:0 atIndex:4]; [e setBuffer:gQ2YBuf offset:0 atIndex:5]; [e setBytes:&G.nblk length:4 atIndex:6]; [e setBytes:&G.out length:4 atIndex:7]; [e dispatchThreads:MTLSizeMake(G.out,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e setComputePipelineState:psoQ2MLPDown]; [e setBuffer:(__bridge id<MTLBuffer>)D.codes offset:0 atIndex:0]; [e setBuffer:(__bridge id<MTLBuffer>)D.scales offset:0 atIndex:1]; [e setBuffer:gQ2YBuf offset:0 atIndex:2]; [e setBuffer:gQ2XBuf offset:0 atIndex:3]; [e setBytes:&D.nblk length:4 atIndex:4]; [e setBytes:&D.out length:4 atIndex:5]; [e dispatchThreads:MTLSizeMake(D.out,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; memcpy(y,gQ2XBuf.contents,(size_t)D.out*4); return 1;
    }
}

// --- accessors for the GPU-resident decode forward (decode.m) ---
// The resident decode forward chains a token's matmuls into ONE command buffer, so it needs to BIND
// each projection's resident ternary buffers directly into its own encoder rather than go through
// mg_q2_0_gemv's standalone commit. Mirrors the q8 accessors. nil/zero for an out-of-range wid.
id<MTLBuffer> mg_q2_0_codes_buf(int wid)  { return (wid >= 0 && wid < gNQ2) ? (__bridge id<MTLBuffer>)gQ2[wid].codes  : nil; }
id<MTLBuffer> mg_q2_0_scales_buf(int wid) { return (wid >= 0 && wid < gNQ2) ? (__bridge id<MTLBuffer>)gQ2[wid].scales : nil; }
void mg_q2_0_dims(int wid, int* out, int* in, int* nblk) {
    if (wid < 0 || wid >= gNQ2) { *out = *in = *nblk = 0; return; }
    *out = gQ2[wid].out; *in = gQ2[wid].in; *nblk = gQ2[wid].nblk;
}

// mg_q2_0_reset releases every resident ternary weight buffer and the reused scratch, returning the
// Q2_0 table to empty. Mirrors mg_q8_reset. Call only when no Q2_0Weight handle is still in use.

void mg_q2_0_reset(void) {
    for (int i = 0; i < gNQ2; i++) {
        if (gQ2[i].codes  != NULL) { CFBridgingRelease(gQ2[i].codes);  gQ2[i].codes  = NULL; }
        if (gQ2[i].scales != NULL) { CFBridgingRelease(gQ2[i].scales); gQ2[i].scales = NULL; }
    }
    gNQ2 = 0;
    gQ2XBuf = nil; gQ2XCap = 0;
    gQ2YBuf = nil; gQ2YCap = 0;
}

// mg_q2_0_g128_upload copies a group-128 GGUF Q2_0 weight (out * nblk * 34 bytes, nblk = in / 128)
// resident onto the GPU and returns an integer handle (>= 0), or -1 on failure.
int mg_q2_0_g128_upload(const unsigned char* raw, int out, int in) {
    if (gDev == nil) return -1;
    if (!q2_0_init()) return -1;
    if (in <= 0 || in % 128 != 0 || out <= 0) return -1;
    if (gNQ2G128 >= MG_MAX_Q2_G128) {
        static int capWarned = 0;
        if (!capWarned) { capWarned = 1; NSLog(@"mg_q2_0_g128_upload: q2_0_g128 weight table full (%d)", MG_MAX_Q2_G128); }
        return -1;
    }
    int nblk = in / 128;
    long bytes = (long)out * nblk * 34;
    id<MTLBuffer> b = [gDev newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
    if (b == nil) {
        NSLog(@"mg_q2_0_g128_upload: device buffer alloc failed for %.1f MB", (double)bytes / 1e6);
        return -1;
    }
    memcpy(b.contents, raw, (size_t)bytes);
    int id = gNQ2G128++;
    gQ2G128[id].buf  = CFBridgingRetain(b);
    gQ2G128[id].out  = out;
    gQ2G128[id].in   = in;
    gQ2G128[id].nblk = nblk;
    return id;
}

// mg_q2_0_g128_gemv computes y[out] = W[wid] · x for one f32 activation row x[in]. f32 result.
void mg_q2_0_g128_gemv(int wid, const float* x, float* y) {
    if (wid < 0 || wid >= gNQ2G128) return;
    @autoreleasepool {
        Q2G128W W = gQ2G128[wid];
        q2_0_g128_grow_scratch((long)W.in, (long)W.out);
        memcpy(gQ2G128XBuf.contents, x, (size_t)W.in * 4);

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:psoQ2G128Gemv];
        [e setBuffer:(__bridge id<MTLBuffer>)W.buf offset:0 atIndex:0];
        [e setBuffer:gQ2G128XBuf offset:0 atIndex:1];
        [e setBuffer:gQ2G128YBuf offset:0 atIndex:2];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(y, gQ2G128YBuf.contents, (size_t)W.out * 4);
    }
}

// mg_q2_0_g128_gemm computes Y[p, out] = X[p, in] · W[wid]ᵀ for resident group-128 Q2_0 rows.
void mg_q2_0_g128_gemm(int wid, const float* X, int p, float* Y) {
    if (wid < 0 || wid >= gNQ2G128 || p <= 0) return;
    @autoreleasepool {
        if (!q2_0_init()) return;
        Q2G128W W = gQ2G128[wid];
        q2_0_g128_grow_scratch((long)p * W.in, (long)p * W.out);
        memcpy(gQ2G128XBuf.contents, X, (size_t)p * W.in * 4);

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:psoQ2G128Gemm];
        [e setBuffer:(__bridge id<MTLBuffer>)W.buf offset:0 atIndex:0];
        [e setBuffer:gQ2G128XBuf offset:0 atIndex:1];
        [e setBuffer:gQ2G128YBuf offset:0 atIndex:2];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, (NSUInteger)p, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(Y, gQ2G128YBuf.contents, (size_t)p * W.out * 4);
    }
}

// mg_q2_0_g128_reset releases every resident group-128 Q2_0 weight buffer and the reused scratch.
void mg_q2_0_g128_reset(void) {
    for (int i = 0; i < gNQ2G128; i++) {
        if (gQ2G128[i].buf != NULL) {
            CFBridgingRelease(gQ2G128[i].buf);
            gQ2G128[i].buf = NULL;
        }
    }
    gNQ2G128 = 0;
    gQ2G128XBuf = nil; gQ2G128XCap = 0;
    gQ2G128YBuf = nil; gQ2G128YCap = 0;
}
