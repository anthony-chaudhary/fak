//go:build darwin && arm64 && cgo

// q8.m — the Metal Q8_0 dequant-GEMV/GEMM. The Q8 twin of q4k.m, and the missing primitive for the
// GPU-resident GDN decode forward (issue #67). The q4_k GPU kernels alone CANNOT move the
// Gated-DeltaNet token mixer onto the device: every linear_attn.* projection (in_proj_qkv/z/b/a,
// out_proj) plus the full-attn q/k are reordered/unpermuted for qwen35, so ResidentQ4KEligible
// keeps them OUT of the raw-q4_k residency and they live as Q8 (internal/model.q8Tensor) — see
// fillQ4KMajority. So decode's in_proj (~16%) + out_proj (~6.4%) run as CPU Q8 GEMVs today; to
// keep the GPU busy across the whole token they must move to the device, which needs this kernel.
//
// Correctness target. One threadgroup (a single 32-lane SIMD group) per output row; the 32 lanes
// split the row's 32-wide Q8_0 blocks and reduce via simd_sum. Each block is an int32 dot of 32
// int8 code pairs, scaled by the weight-block scale × the activation-block scale, accumulated in
// float — byte-for-byte internal/model.qdot8scalar, the f32-reduced reference (only the simd_sum
// order differs from the CPU sequential accumulate). Pinned by TestMetalQ8GemvMatchesCPU.
//
// Shares gDev/gQueue with metal.m (one device, one queue). Its weight table (int8 codes + f32
// block scales) is separate from the f16 (metal.m) and raw-q4_k (q4k.m) tables, with its own
// teardown via mg_q8_reset.

#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>
#include <string.h>

// Device + queue are owned by metal.m (mg_init); we reuse them.
extern id<MTLDevice>       gDev;
extern id<MTLCommandQueue> gQueue;

static NSString *kQ8Src = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// q8_gemv: the decode GEMV. ONE threadgroup (single 32-lane SIMD group) per output row. The 32
// lanes split the row's nblk Q8_0 blocks; each block contributes int32(code·code) × wScale ×
// xScale to a per-lane float accumulator, then simd_sum reduces across the lanes. This is the
// in-kernel twin of internal/model.qdot8scalar (per-block int dot, float scale, float reduce).
kernel void q8_gemv(device const char*  W    [[buffer(0)]],  // out*in int8 codes, row-major
                    device const float* WD   [[buffer(1)]],  // out*nblk weight block-scales
                    device const char*  X    [[buffer(2)]],  // in int8 activation codes
                    device const float* XD   [[buffer(3)]],  // nblk activation block-scales
                    device float*       Y    [[buffer(4)]],
                    constant int&       nblk [[buffer(5)]],
                    constant int&       out_ [[buffer(6)]],
                    uint o   [[threadgroup_position_in_grid]],
                    uint lid [[thread_index_in_threadgroup]]) {
    if (o >= (uint)out_) return;
    device const char*  wrow = W  + (long)o * nblk * 32;
    device const float* wd   = WD + (long)o * nblk;
    float acc = 0.0f;
    for (int b = (int)lid; b < nblk; b += 32) {
        device const char* wb = wrow + (long)b * 32;
        device const char* xb = X    + (long)b * 32;
        int s = 0;
        for (int i = 0; i < 32; i++) s += (int)wb[i] * (int)xb[i];
        acc += (float)s * wd[b] * XD[b];
    }
    acc = simd_sum(acc);
    if (lid == 0) Y[o] = acc;
}

// q8_gemm: register-blocked tiled batched prefill GEMM. This is the Q8_0 sibling of q4k_gemm:
// each threadgroup owns a Q8_BM×Q8_BN output tile, walks the K axis one 32-wide Q8 block at a
// time, stages dequanted weights + activations into threadgroup memory, and accumulates a
// Q8_TM×Q8_TN register micro-tile. The whole activation panel is dispatched in one command buffer.
#define Q8_BM 64
#define Q8_BN 64
#define Q8_TM 4
#define Q8_TN 4
#define Q8_TGX 16
#define Q8_TGY 16
#define Q8_TG 256
kernel void q8_gemm(device const char*  W    [[buffer(0)]],  // out*in int8 codes, row-major
                    device const float* WD   [[buffer(1)]],  // out*nblk weight block-scales
                    device const char*  X    [[buffer(2)]],  // P*in int8 activation codes
                    device const float* XD   [[buffer(3)]],  // P*nblk activation block-scales
                    device float*       Y    [[buffer(4)]],
                    constant int&       nblk [[buffer(5)]],
                    constant int&       out_ [[buffer(6)]],
                    constant int&       P    [[buffer(7)]],
                    constant int&       t0   [[buffer(8)]],
                    constant int&       nt   [[buffer(9)]],
                    uint ob [[threadgroup_position_in_grid]],
                    uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float wbuf[Q8_BM * 32];
    threadgroup float xbuf[Q8_BN * 32];
    int in = nblk * 32;
    int o0 = (int)ob * Q8_BM;
    int tr = (int)lid / Q8_TGX;
    int tc = (int)lid % Q8_TGX;
    float acc[Q8_TM][Q8_TN];
    for (int i = 0; i < Q8_TM; i++)
        for (int j = 0; j < Q8_TN; j++) acc[i][j] = 0.0f;

    for (int b = 0; b < nblk; b++) {
        for (int idx = (int)lid; idx < Q8_BM * 32; idx += Q8_TG) {
            int row = idx >> 5, k = idx & 31;
            int orow = o0 + row;
            float val = 0.0f;
            if (orow < out_) {
                long wbase = (long)orow * in + (long)b * 32;
                val = (float)W[wbase + k] * WD[(long)orow * nblk + b];
            }
            wbuf[idx] = val;
        }
        for (int idx = (int)lid; idx < Q8_BN * 32; idx += Q8_TG) {
            int tk = idx >> 5, k = idx & 31;
            float val = 0.0f;
            if (tk < nt) {
                int t = t0 + tk;
                long xbase = (long)t * in + (long)b * 32;
                val = (float)X[xbase + k] * XD[(long)t * nblk + b];
            }
            xbuf[idx] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < 32; k++) {
            float wreg[Q8_TM], xreg[Q8_TN];
            for (int i = 0; i < Q8_TM; i++) wreg[i] = wbuf[(tr * Q8_TM + i) * 32 + k];
            for (int j = 0; j < Q8_TN; j++) xreg[j] = xbuf[(tc * Q8_TN + j) * 32 + k];
            for (int i = 0; i < Q8_TM; i++)
                for (int j = 0; j < Q8_TN; j++) acc[i][j] += wreg[i] * xreg[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < Q8_TM; i++) {
        int orow = o0 + tr * Q8_TM + i;
        if (orow >= out_) continue;
        for (int j = 0; j < Q8_TN; j++) {
            int tcol = tc * Q8_TN + j;
            if (tcol < nt) Y[(long)(t0 + tcol) * out_ + orow] = acc[i][j];
        }
    }
}
)MSL";

static id<MTLComputePipelineState> psoQ8Gemv, psoQ8Gemm;
static int gQ8Ready;

static int q8_init(void) {
    if (gQ8Ready) return 1;
    if (gDev == nil) return 0;
    NSError *err = nil;
    id<MTLLibrary> lib = [gDev newLibraryWithSource:kQ8Src options:nil error:&err];
    if (lib == nil) { NSLog(@"q8: library compile failed: %@", err); return 0; }
    psoQ8Gemv = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q8_gemv"] error:&err];
    psoQ8Gemm = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q8_gemm"] error:&err];
    if (!psoQ8Gemv || !psoQ8Gemm) { NSLog(@"q8: pipeline build failed: %@", err); return 0; }
    gQ8Ready = 1;
    return 1;
}

typedef struct {
    CFTypeRef codes;  // retained id<MTLBuffer>, int8 [out*in]
    CFTypeRef scales; // retained id<MTLBuffer>, f32  [out*nblk]
    int out, in, nblk;
} Q8W;

#define MG_MAX_Q8 8192
static Q8W gQ8[MG_MAX_Q8];
static int gNQ8 = 0;

// Reused per-call scratch: the activation codes/scales and the result. Weights are persistent;
// only the per-call X/Y move (same discipline as q4k.m's gQXBuf/gQYBuf).
static id<MTLBuffer> gQ8XBuf  = nil; static long gQ8XCap  = 0; // activation codes (int8), bytes
static id<MTLBuffer> gQ8XDBuf = nil; static long gQ8XDCap = 0; // activation scales (f32), elems
static id<MTLBuffer> gQ8YBuf  = nil; static long gQ8YCap  = 0; // result (f32), elems

static void q8_grow_scratch(long inBytes, long nblkElems, long yElems) {
    if (gQ8XBuf == nil || gQ8XCap < inBytes) {
        gQ8XBuf = [gDev newBufferWithLength:(NSUInteger)inBytes options:MTLResourceStorageModeShared];
        gQ8XCap = inBytes;
    }
    if (gQ8XDBuf == nil || gQ8XDCap < nblkElems) {
        gQ8XDBuf = [gDev newBufferWithLength:(NSUInteger)(nblkElems * 4) options:MTLResourceStorageModeShared];
        gQ8XDCap = nblkElems;
    }
    if (gQ8YBuf == nil || gQ8YCap < yElems) {
        gQ8YBuf = [gDev newBufferWithLength:(NSUInteger)(yElems * 4) options:MTLResourceStorageModeShared];
        gQ8YCap = yElems;
    }
}

// mg_q8_upload copies a Q8_0 weight (out*in int8 codes + out*nblk f32 block scales, nblk=in/32)
// resident onto the GPU and returns an integer handle (>=0), or -1 on failure.
int mg_q8_upload(const signed char* codes, const float* scales, int out, int in) {
    if (gDev == nil) return -1;
    if (!q8_init()) return -1;
    if (in % 32 != 0 || out <= 0) return -1;
    if (gNQ8 >= MG_MAX_Q8) {
        static int capWarned = 0;
        if (!capWarned) { capWarned = 1; NSLog(@"mg_q8_upload: q8 weight table full (%d)", MG_MAX_Q8); }
        return -1;
    }
    int nblk = in / 32;
    long codeBytes  = (long)out * in;
    long scaleBytes = (long)out * nblk * 4;
    id<MTLBuffer> cb = [gDev newBufferWithLength:(NSUInteger)codeBytes  options:MTLResourceStorageModeShared];
    id<MTLBuffer> sb = [gDev newBufferWithLength:(NSUInteger)scaleBytes options:MTLResourceStorageModeShared];
    if (cb == nil || sb == nil) {
        NSLog(@"mg_q8_upload: device buffer alloc failed for %.1f MB", (double)(codeBytes + scaleBytes) / 1e6);
        return -1;
    }
    memcpy(cb.contents, codes,  (size_t)codeBytes);
    memcpy(sb.contents, scales, (size_t)scaleBytes);
    int id = gNQ8++;
    gQ8[id].codes  = CFBridgingRetain(cb);
    gQ8[id].scales = CFBridgingRetain(sb);
    gQ8[id].out  = out;
    gQ8[id].in   = in;
    gQ8[id].nblk = nblk;
    return id;
}

// mg_q8_gemv computes y[out] = W[wid] · x for one Q8_0-quantized activation (xq codes [in],
// xd block scales [nblk]). f32 result.
void mg_q8_gemv(int wid, const signed char* xq, const float* xd, float* y) {
    if (wid < 0 || wid >= gNQ8) return;
    @autoreleasepool {
        Q8W W = gQ8[wid];
        q8_grow_scratch((long)W.in, (long)W.nblk, (long)W.out);
        memcpy(gQ8XBuf.contents,  xq, (size_t)W.in);
        memcpy(gQ8XDBuf.contents, xd, (size_t)W.nblk * 4);

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:psoQ8Gemv];
        [e setBuffer:(__bridge id<MTLBuffer>)W.codes  offset:0 atIndex:0];
        [e setBuffer:(__bridge id<MTLBuffer>)W.scales offset:0 atIndex:1];
        [e setBuffer:gQ8XBuf  offset:0 atIndex:2];
        [e setBuffer:gQ8XDBuf offset:0 atIndex:3];
        [e setBuffer:gQ8YBuf  offset:0 atIndex:4];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:5];
        [e setBytes:&W.out  length:sizeof(int) atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(y, gQ8YBuf.contents, (size_t)W.out * 4);
    }
}

// mg_q8_gemv_group runs n decode GEMVs that SHARE one Q8_0 activation (xq/xd, same in) but apply
// n DIFFERENT resident Q8 weights, into ONE command buffer (one commit/waitUntilCompleted). This
// is the live GDN decode access pattern — the in_proj quad (qkv,z,b,a) all read the same post-norm
// activation. Each weight i writes Ycat[yoff[i] .. yoff[i]+out_i); yoff has n+1 entries.
void mg_q8_gemv_group(const int* wids, int n, const signed char* xq, const float* xd, float* Ycat, const int* yoff) {
    if (n <= 0) return;
    @autoreleasepool {
        int in   = gQ8[wids[0]].in;
        int nblk = gQ8[wids[0]].nblk;
        long ytot = (long)yoff[n];
        q8_grow_scratch((long)in, (long)nblk, ytot);
        memcpy(gQ8XBuf.contents,  xq, (size_t)in);
        memcpy(gQ8XDBuf.contents, xd, (size_t)nblk * 4);

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:psoQ8Gemv];
        [e setBuffer:gQ8XBuf  offset:0 atIndex:2]; // shared activation for every weight in the group
        [e setBuffer:gQ8XDBuf offset:0 atIndex:3];
        for (int i = 0; i < n; i++) {
            Q8W Wi = gQ8[wids[i]];
            [e setBuffer:(__bridge id<MTLBuffer>)Wi.codes  offset:0 atIndex:0];
            [e setBuffer:(__bridge id<MTLBuffer>)Wi.scales offset:0 atIndex:1];
            [e setBuffer:gQ8YBuf offset:(NSUInteger)((long)yoff[i] * 4) atIndex:4];
            [e setBytes:&Wi.nblk length:sizeof(int) atIndex:5];
            [e setBytes:&Wi.out  length:sizeof(int) atIndex:6];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)Wi.out, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        [e endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(Ycat, gQ8YBuf.contents, (size_t)ytot * 4);
    }
}

// mg_q8_gemm computes Y[P,out] = X[P,in] * W[wid]^T for a Q8_0 activation panel in one command
// buffer. This is the Metal prefill primitive for Q8-minority projections in the resident-Q4_K
// lane (#1087): full-attn q/k and Qwen3.6 linear_attn.* no longer have to fall back to CPU qGemm8.
void mg_q8_gemm(int wid, const signed char* Xq, const float* Xd, int P, float* Y) {
    if (wid < 0 || wid >= gNQ8 || P <= 0) return;
    @autoreleasepool {
        Q8W W = gQ8[wid];
        q8_grow_scratch((long)P * W.in, (long)P * W.nblk, (long)P * W.out);
        memcpy(gQ8XBuf.contents,  Xq, (size_t)P * W.in);
        memcpy(gQ8XDBuf.contents, Xd, (size_t)P * W.nblk * 4);

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:psoQ8Gemm];
        [e setBuffer:(__bridge id<MTLBuffer>)W.codes  offset:0 atIndex:0];
        [e setBuffer:(__bridge id<MTLBuffer>)W.scales offset:0 atIndex:1];
        [e setBuffer:gQ8XBuf  offset:0 atIndex:2];
        [e setBuffer:gQ8XDBuf offset:0 atIndex:3];
        [e setBuffer:gQ8YBuf  offset:0 atIndex:4];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:5];
        [e setBytes:&W.out  length:sizeof(int) atIndex:6];
        [e setBytes:&P      length:sizeof(int) atIndex:7];
        const int BM = 64;  // output rows per threadgroup; must match Q8_BM in the MSL source
        const int BN = 64;  // token-tile width;            must match Q8_BN in the MSL source
        const int TG = 256; // threads per threadgroup;     must match Q8_TG in the MSL source
        int rowBlocks = (W.out + BM - 1) / BM;
        for (int t0 = 0; t0 < P; t0 += BN) {
            int nt = P - t0;
            if (nt > BN) nt = BN;
            [e setBytes:&t0 length:sizeof(int) atIndex:8];
            [e setBytes:&nt length:sizeof(int) atIndex:9];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)rowBlocks, 1, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)TG, 1, 1)];
        }
        [e endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(Y, gQ8YBuf.contents, (size_t)P * W.out * 4);
    }
}

// --- accessors for the GPU-resident decode forward (decode.m) ---
// The resident decode forward (issue #67) chains all of a token's matmuls into ONE command
// buffer, so it needs to BIND each projection's resident Q8 weight buffers directly into its
// own encoder rather than go through mg_q8_gemv's standalone commit. These expose the persistent
// device buffers + dims for a wid without copying. id<MTLBuffer> crosses the .m boundary fine
// (same ObjC compile unit set, one binary). nil/zero for an out-of-range wid.
id<MTLBuffer> mg_q8_codes_buf(int wid)  { return (wid >= 0 && wid < gNQ8) ? (__bridge id<MTLBuffer>)gQ8[wid].codes  : nil; }
id<MTLBuffer> mg_q8_scales_buf(int wid) { return (wid >= 0 && wid < gNQ8) ? (__bridge id<MTLBuffer>)gQ8[wid].scales : nil; }
void mg_q8_dims(int wid, int* out, int* in, int* nblk) {
    if (wid < 0 || wid >= gNQ8) { *out = *in = *nblk = 0; return; }
    *out = gQ8[wid].out; *in = gQ8[wid].in; *nblk = gQ8[wid].nblk;
}

// mg_q8_reset releases every resident Q8 weight buffer and the reused scratch, returning the Q8
// table to empty. Mirrors mg_q4k_reset. Call only when no Q8Weight handle is still in use.
void mg_q8_reset(void) {
    for (int i = 0; i < gNQ8; i++) {
        if (gQ8[i].codes  != NULL) { CFBridgingRelease(gQ8[i].codes);  gQ8[i].codes  = NULL; }
        if (gQ8[i].scales != NULL) { CFBridgingRelease(gQ8[i].scales); gQ8[i].scales = NULL; }
    }
    gNQ8 = 0;
    gQ8XBuf  = nil; gQ8XCap  = 0;
    gQ8XDBuf = nil; gQ8XDCap = 0;
    gQ8YBuf  = nil; gQ8YCap  = 0;
}
