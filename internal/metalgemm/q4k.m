//go:build darwin && arm64 && cgo

// q4k.m — the Metal q4_k dequant-GEMV/GEMM. This is the lever that the f16/MPS path
// (metal.m) cannot be: a 27B model is ~54 GB in f16, which does NOT fit the 36 GB unified
// pool, so the f16-resident route OOMs. The q4_k_m GGUF is ~16 GB and DOES fit, but MPS has
// no q4_k GEMM — so we dequant in the MSL kernel exactly the way llama.cpp's Metal backend
// does: keep the raw 144-B/256-weight super-blocks resident on the GPU, and have each thread
// reconstruct its weight row's f32 values on the fly (d*sc*nibble - dmin*m) and dot them
// against the f32 activation. The CPU int8-SDOT kernel tops out ~23 GB/s (compute-bound) and
// cannot reach the 7.29 tok/s decode / 51.55 tok/s prefill bar; the GPU has both the
// bandwidth and the parallel dequant FLOPs, which is why llama.cpp hits the bar on Metal.
//
// Correctness target. The dequant is byte-for-byte internal/model.q4kDequantSuperBlock
// (which is itself ggufload.dequantQ4K factored per super-block): super-block = d(f16,2) +
// dmin(f16,2) + scales(12, 6-bit packed via get_scale_min_k4) + q(128 nibbles); 8 sub-blocks
// of 32, weight = d*sc*code - dmin*m. So mg_q4k_gemv(W, x) ≈ q4kMatRowsRange(W, x) (the f32
// reference) up to GPU float-accumulation order — pinned by TestMetalQ4KGemvMatchesCPU.
//
// Shares gDev/gQueue with metal.m (one device, one queue). The q4_k weight table is separate
// from the f16 table (it holds raw bytes, not f16), with its own teardown via mg_q4k_reset.

#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>
#include <string.h>
#include <unistd.h>

// Device + queue are owned by metal.m (mg_init); we reuse them.
extern id<MTLDevice>       gDev;
extern id<MTLCommandQueue> gQueue;

// The MSL kernels. q4k_row_dot reconstructs one weight row's f32 values per super-block and
// dots against the matching 256-wide activation slice — the in-kernel twin of the CPU
// q4kMatRowsRange inner loop. q4k_gemv is one thread per output row (decode GEMV); q4k_gemm
// is one thread per (output row, token) over a 2-D grid (batched prefill GEMM).
static NSString *kQ4KSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// get_scale_min_k4: unpack the j-th (scale,min) 6-bit pair from the 12-byte scales field.
// Byte-for-byte internal/model.getScaleMinK4.
inline float2 q4k_scale_min(int j, device const uchar* s) {
    uchar a, b;
    if (j < 4) {
        a = s[j] & 63;
        b = s[j + 4] & 63;
    } else {
        a = (s[j + 4] & 0x0f) | ((s[j - 4] >> 6) << 4);
        b = (s[j + 4] >> 4)   | ((s[j]     >> 6) << 4);
    }
    return float2((float)a, (float)b);
}

// q4k_block_dot: dot one 144-B super-block's 256 dequanted weights against the matching
// 256-wide activation slice. Sub-block order matches the CPU reference (low nibbles 0..31 then
// high nibbles 32..63 within each 64-weight pair).
inline float q4k_block_dot(device const uchar* blk, device const float* xs) {
    float d  = (float)(*(device const half*)(blk + 0));
    float dm = (float)(*(device const half*)(blk + 2));
    device const uchar* scales = blk + 4;
    device const uchar* q = blk + 16;
    float acc = 0.0f;
    int qi = 0;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        float2 sm0 = q4k_scale_min(is,     scales);
        float2 sm1 = q4k_scale_min(is + 1, scales);
        float d1 = d * sm0.x, m1 = dm * sm0.y;
        float d2 = d * sm1.x, m2 = dm * sm1.y;
        for (int l = 0; l < 32; l++) {
            acc += (d1 * (float)(q[qi + l] & 0x0f) - m1) * xs[j + l];
        }
        for (int l = 0; l < 32; l++) {
            acc += (d2 * (float)(q[qi + l] >> 4) - m2) * xs[j + 32 + l];
        }
        qi += 32;
        is += 2;
    }
    return acc;
}

// q4k_row_dot: serial dot of a whole weight row (nblk super-blocks) — used by the batched GEMM
// where the P (token) axis already provides the GPU's parallelism.
inline float q4k_row_dot(device const uchar* row, device const float* x, int nblk) {
    float acc = 0.0f;
    for (int b = 0; b < nblk; b++) acc += q4k_block_dot(row + (long)b * 144, x + (long)b * 256);
    return acc;
}

// q4k_gemv: the decode GEMV. ONE threadgroup (a single 32-lane SIMD group) per output row, the
// 32 lanes splitting the row's super-blocks and reducing via simd_sum. A 1-thread-per-row GEMV
// underutilizes the GPU (only `out` threads → occupancy-bound at ~21 GB/s); spreading each row
// across a SIMD group raises occupancy by 32× so a single GEMV approaches the device bandwidth
// that the 7.29 tok/s decode bar needs. The simd_sum tree differs from the CPU's sequential
// accumulation only at the float-rounding level (cosine 1.0 / maxRel ~1e-6, still Approx).
kernel void q4k_gemv(device const uchar* W [[buffer(0)]],
                     device const float* X [[buffer(1)]],
                     device float*       Y [[buffer(2)]],
                     constant int&    nblk [[buffer(3)]],
                     constant int&     out [[buffer(4)]],
                     uint o   [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]]) {
    if (o >= (uint)out) return;
    device const uchar* row = W + (long)o * nblk * 144;
    float acc = 0.0f;
    for (int b = (int)lid; b < nblk; b += 32) {
        acc += q4k_block_dot(row + (long)b * 144, X + (long)b * 256);
    }
    acc = simd_sum(acc);
    if (lid == 0) Y[o] = acc;
}

// q4k_gemm: the REGISTER-BLOCKED TILED prefill GEMM (issue #1085 — the prefill kernel lever from
// MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26).
//
// Measured root cause of the old kernel's ~5% FLOP: every prior layout used ONE threadgroup per
// output row, so each threadgroup re-read the WHOLE activation panel. Fine while X fits L2 (small
// P), but at the real agentic prefill (P≥256) X spills L2 and the GEMM goes DRAM-bound on redundant
// activation reads — measured GFLOP/s fell 347→190 as P grew 22→2048, and neither a SIMD-group
// dot-reduction nor GEMV-style streaming moved it (both ~5% of FLOP). The win is a classic
// register-blocked GEMM tile:
//
//   • Each threadgroup computes a Q4K_BM×Q4K_BN output block (BM rows × BN tokens).
//   • The K (in) axis is walked one q4_k SUB-block at a time (32 weights, one scale), so the staged
//     tiles are only (BM+BN)*32 floats — small enough for high occupancy.
//   • Each thread owns a Q4K_TM×Q4K_TN register micro-tile and accumulates via the outer-product
//     inner loop, so every value staged into threadgroup memory is reused TM or TN times in
//     registers. That raises arithmetic intensity AND kills the L2-spill (each activation is read
//     out/BM× fewer times).
//
// Measured on M3 Pro at the real [17408,5120] gate/up shape: ~1375 GFLOP/s, FLAT across P=64..2048
// — vs ~211 at P=512 / 190 at P=2048 for the prior kernel (~6.5–7.2× at realistic prefill sizes;
// ~20% of the f32 FLOP ceiling). Numerically the inner sum walks the reference's own sub-block
// order, so it stays bit-close to the CPU f32 reference (TestMetalQ4KGemmMatchesCPU: cosine 1.0).
// The C side encodes one dispatch per BN-token tile into a single command buffer.
#define Q4K_BM 64         // output rows per threadgroup
#define Q4K_BN 64         // tokens per tile (must match the C-side token tile)
#define Q4K_TM 4          // output rows per thread (register micro-tile)
#define Q4K_TN 4          // tokens per thread (register micro-tile)
#define Q4K_TGX 16        // = Q4K_BN/Q4K_TN  (thread columns)
#define Q4K_TGY 16        // = Q4K_BM/Q4K_TM  (thread rows)
#define Q4K_TG 256        // = Q4K_TGX*Q4K_TGY threads
// The K (in) axis is walked one q4_k SUB-block (32 weights, one scale) at a time, so the staged
// tiles are only BM*32 + BN*32 floats (8 KB) — small enough for high occupancy — while each thread
// holds a TM*TN register accumulator and reuses every staged value TM or TN times via the
// outer-product inner loop (the standard register-blocked GEMM that lifts FLOP utilization).
kernel void q4k_gemm(device const uchar* W [[buffer(0)]],
                     device const float* X [[buffer(1)]],
                     device float*       Y [[buffer(2)]],
                     constant int&    nblk [[buffer(3)]],
                     constant int&     out [[buffer(4)]],
                     constant int&       P [[buffer(5)]],
                     constant int&      t0 [[buffer(6)]],
                     constant int&      nt [[buffer(7)]],
                     uint ob  [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float wbuf[Q4K_BM * 32]; // BM weight rows × one 32-wide sub-block
    threadgroup float xbuf[Q4K_BN * 32]; // BN token activations × one 32-wide sub-block
    int in = nblk * 256;
    int o0 = (int)ob * Q4K_BM;           // first output row this threadgroup owns
    int tr = (int)lid / Q4K_TGX;         // thread-row block 0..TGY-1
    int tc = (int)lid % Q4K_TGX;         // thread-col block 0..TGX-1
    float acc[Q4K_TM][Q4K_TN];
    for (int i = 0; i < Q4K_TM; i++)
        for (int j = 0; j < Q4K_TN; j++) acc[i][j] = 0.0f;
    for (int sblk = 0; sblk < nblk; sblk++) {
        for (int sb = 0; sb < 8; sb++) {  // 8 q4_k sub-blocks of 32 per super-block
            // Stage sub-block sb's 32 weights for the BM rows into wbuf[row*32 + k].
            for (int idx = (int)lid; idx < Q4K_BM * 32; idx += Q4K_TG) {
                int row = idx >> 5, k = idx & 31;
                int orow = o0 + row;
                float val = 0.0f;
                if (orow < out) {
                    device const uchar* blk = W + ((long)orow * nblk + sblk) * 144;
                    float d  = (float)(*(device const half*)(blk + 0));
                    float dm = (float)(*(device const half*)(blk + 2));
                    device const uchar* scales = blk + 4;
                    device const uchar* q = blk + 16;
                    uchar byte = q[(sb >> 1) * 32 + k];
                    uchar nib = (sb & 1) ? (byte >> 4) : (byte & 0x0f);
                    float2 sm = q4k_scale_min(sb, scales);
                    val = d * sm.x * (float)nib - dm * sm.y;
                }
                wbuf[idx] = val;
            }
            // Stage sub-block sb's 32 activations for the BN tokens into xbuf[tok*32 + k].
            for (int idx = (int)lid; idx < Q4K_BN * 32; idx += Q4K_TG) {
                int tk = idx >> 5, k = idx & 31;
                xbuf[idx] = (tk < nt) ? X[(long)(t0 + tk) * in + (long)sblk * 256 + sb * 32 + k] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Outer-product accumulate: each thread's TM×TN micro-tile over the 32-wide sub-block.
            for (int k = 0; k < 32; k++) {
                float wreg[Q4K_TM], xreg[Q4K_TN];
                for (int i = 0; i < Q4K_TM; i++) wreg[i] = wbuf[(tr * Q4K_TM + i) * 32 + k];
                for (int j = 0; j < Q4K_TN; j++) xreg[j] = xbuf[(tc * Q4K_TN + j) * 32 + k];
                for (int i = 0; i < Q4K_TM; i++)
                    for (int j = 0; j < Q4K_TN; j++) acc[i][j] += wreg[i] * xreg[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (int i = 0; i < Q4K_TM; i++) {
        int orow = o0 + tr * Q4K_TM + i;
        if (orow >= out) continue;
        for (int j = 0; j < Q4K_TN; j++) {
            int tcol = tc * Q4K_TN + j;
            if (tcol < nt) Y[(long)(t0 + tcol) * out + orow] = acc[i][j];
        }
    }
}

// q4k_swiglu: out[i] = silu(gate[i]) * up[i], the SwiGLU elementwise for the fused decode MLP. Run
// on the GPU between the gate/up GEMVs and the down GEMV so the I-wide intermediate never leaves
// the device. silu(z)=z/(1+exp(-z)) — matches internal/model.silu (the non-GELU activation path).
kernel void q4k_swiglu(device const float* gate [[buffer(0)]],
                       device const float* up   [[buffer(1)]],
                       device float*       out  [[buffer(2)]],
                       constant int&       n    [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= (uint)n) return;
    float g = gate[i];
    out[i] = (g / (1.0f + exp(-g))) * up[i];
}

// q6k_block_dot: dot one 210-B Q6_K super-block's 256 dequanted weights against the matching
// 256-wide activation slice. Byte-for-byte internal/model.q6kDequantSuperBlock: layout is
// ql(128) + qh(64) + scales(16, SIGNED int8) + d(f16 @ 208); the 6-bit code is
// (ql nibble | ((qh 2 bits)<<4)) with a −32 zero-point, weight = d*sc*(code−32). The scale field
// is SIGNED (device const char*), the classic MSL signedness trap — keep it `char`, not `uchar`.
inline float q6k_block_dot(device const uchar* blk, device const float* xs) {
    device const uchar* ql = blk + 0;
    device const uchar* qh = blk + 128;
    device const char*  sc = (device const char*)(blk + 192); // SIGNED int8 scales
    float d = (float)(*(device const half*)(blk + 208));
    float acc = 0.0f;
    int qlOff = 0, qhOff = 0, scOff = 0;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[qlOff + l +  0] & 0x0f) | (((qh[qhOff + l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[qlOff + l + 32] & 0x0f) | (((qh[qhOff + l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[qlOff + l +  0] >> 4)   | (((qh[qhOff + l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[qlOff + l + 32] >> 4)   | (((qh[qhOff + l] >> 6) & 3) << 4)) - 32;
            acc += d * (float)sc[scOff + is + 0] * (float)q1 * xs[n + l +  0];
            acc += d * (float)sc[scOff + is + 2] * (float)q2 * xs[n + l + 32];
            acc += d * (float)sc[scOff + is + 4] * (float)q3 * xs[n + l + 64];
            acc += d * (float)sc[scOff + is + 6] * (float)q4 * xs[n + l + 96];
        }
        qlOff += 64;
        qhOff += 32;
        scOff += 8;
    }
    return acc;
}

// q6k_gemv: the Q6_K decode GEMV, the byte-for-byte twin of q4k_gemv but over 210-B super-blocks.
// ONE 32-lane SIMD group per output row, the 32 lanes splitting the row's super-blocks and reducing
// via simd_sum. Used as stage 3 of the Q6_K-down fused MLP (mg_q4k_mlp_q6down).
kernel void q6k_gemv(device const uchar* W [[buffer(0)]],
                     device const float* X [[buffer(1)]],
                     device float*       Y [[buffer(2)]],
                     constant int&    nblk [[buffer(3)]],
                     constant int&     out [[buffer(4)]],
                     uint o   [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]]) {
    if (o >= (uint)out) return;
    device const uchar* row = W + (long)o * nblk * 210;
    float acc = 0.0f;
    for (int b = (int)lid; b < nblk; b += 32) {
        acc += q6k_block_dot(row + (long)b * 210, X + (long)b * 256);
    }
    acc = simd_sum(acc);
    if (lid == 0) Y[o] = acc;
}
)MSL";

static id<MTLComputePipelineState> psoQ4KGemv, psoQ4KGemm, psoQ4KSwiGLU, psoQ6KGemv;
static int gQ4KReady;

static int q4k_init(void) {
    if (gQ4KReady) return 1;
    if (gDev == nil) return 0;
    NSError *err = nil;
    id<MTLLibrary> lib = [gDev newLibraryWithSource:kQ4KSrc options:nil error:&err];
    if (lib == nil) { NSLog(@"q4k: library compile failed: %@", err); return 0; }
    psoQ4KGemv = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q4k_gemv"] error:&err];
    psoQ4KGemm = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q4k_gemm"] error:&err];
    psoQ4KSwiGLU = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q4k_swiglu"] error:&err];
    psoQ6KGemv = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q6k_gemv"] error:&err];
    if (!psoQ4KGemv || !psoQ4KGemm || !psoQ4KSwiGLU || !psoQ6KGemv) { NSLog(@"q4k: pipeline build failed: %@", err); return 0; }
    gQ4KReady = 1;
    return 1;
}

typedef struct {
    CFTypeRef buf; // retained id<MTLBuffer>, raw q4_k bytes [out * nblk * 144]
    int out;
    int in;
    int nblk;
} Q4KW;

#define MG_MAX_Q4 8192
static Q4KW gQ4[MG_MAX_Q4];
static int gNQ4 = 0;

static int q4k_register_buffer(id<MTLBuffer> b, int out, int in, int nblk) {
    int id = gNQ4++;
    gQ4[id].buf = CFBridgingRetain(b);
    gQ4[id].out = out;
    gQ4[id].in = in;
    gQ4[id].nblk = nblk;
    return id;
}

// Reused f32 scratch for the activation (X) and result (Y) of the current q4_k op, grown on
// demand (sized in elements). The weight buffers are persistent; only the per-call X/Y move.
static id<MTLBuffer> gQXBuf = nil; static long gQXCap = 0;
static id<MTLBuffer> gQYBuf = nil; static long gQYCap = 0;

static void q4k_grow_scratch(long xElems, long yElems) {
    if (gQXBuf == nil || gQXCap < xElems) {
        gQXBuf = [gDev newBufferWithLength:(NSUInteger)(xElems * 4) options:MTLResourceStorageModeShared];
        gQXCap = xElems;
    }
    if (gQYBuf == nil || gQYCap < yElems) {
        gQYBuf = [gDev newBufferWithLength:(NSUInteger)(yElems * 4) options:MTLResourceStorageModeShared];
        gQYCap = yElems;
    }
}

// Reused device-resident scratch for the fused MLP's I-wide gate/up/intermediate, so that buffer
// never crosses the host boundary in mg_q4k_mlp (only x[H] in and y[H] out do).
static id<MTLBuffer> gMlpGate = nil, gMlpUp = nil, gMlpInter = nil; static long gMlpCap = 0;

static void q4k_grow_mlp(long iElems) {
    if (gMlpGate != nil && gMlpCap >= iElems) return;
    gMlpGate  = [gDev newBufferWithLength:(NSUInteger)(iElems * 4) options:MTLResourceStorageModeShared];
    gMlpUp    = [gDev newBufferWithLength:(NSUInteger)(iElems * 4) options:MTLResourceStorageModeShared];
    gMlpInter = [gDev newBufferWithLength:(NSUInteger)(iElems * 4) options:MTLResourceStorageModeShared];
    gMlpCap = iElems;
}

// mg_q4k_mlp runs a whole dense SwiGLU MLP — y = down( silu(gate·x) * (up·x) ) — for ONE decode
// token in ONE command buffer, keeping the I-wide gate/up/intermediate resident on the GPU (only
// x[H] in and y[H] out cross the boundary). Three encoders order the chain via Metal's automatic
// hazard tracking on the shared scratch: (1) gate & up GEMVs (independent), (2) the SwiGLU
// elementwise, (3) the down GEMV. This collapses the MLP — ~54% of q4_k_m decode — from three
// per-matmul command buffers (each round-tripping the I-wide gate/up out + the intermediate back
// in) to one. Caller guarantees gate.out==up.out==down.in (=I) and gate.in==up.in==down.out (=H).
void mg_q4k_mlp(int gate_wid, int up_wid, int down_wid, const float* x, float* y) {
    if (gate_wid < 0 || up_wid < 0 || down_wid < 0 ||
        gate_wid >= gNQ4 || up_wid >= gNQ4 || down_wid >= gNQ4) return;
    @autoreleasepool {
        Q4KW G = gQ4[gate_wid], U = gQ4[up_wid], D = gQ4[down_wid];
        int H = G.in;
        int I = G.out;
        q4k_grow_scratch((long)H, (long)D.out);
        q4k_grow_mlp((long)I);
        id<MTLBuffer> xb = gQXBuf, yb = gQYBuf;
        memcpy(xb.contents, x, (size_t)H * 4);

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];

        // (1) gate = G·x and up = U·x (independent), one encoder
        id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
        [e1 setComputePipelineState:psoQ4KGemv];
        [e1 setBuffer:xb offset:0 atIndex:1];
        [e1 setBuffer:(__bridge id<MTLBuffer>)G.buf offset:0 atIndex:0];
        [e1 setBuffer:gMlpGate offset:0 atIndex:2];
        [e1 setBytes:&G.nblk length:sizeof(int) atIndex:3];
        [e1 setBytes:&G.out  length:sizeof(int) atIndex:4];
        [e1 dispatchThreadgroups:MTLSizeMake((NSUInteger)G.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e1 setBuffer:(__bridge id<MTLBuffer>)U.buf offset:0 atIndex:0];
        [e1 setBuffer:gMlpUp offset:0 atIndex:2];
        [e1 setBytes:&U.nblk length:sizeof(int) atIndex:3];
        [e1 setBytes:&U.out  length:sizeof(int) atIndex:4];
        [e1 dispatchThreadgroups:MTLSizeMake((NSUInteger)U.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e1 endEncoding];

        // (2) inter = silu(gate) * up
        id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
        [e2 setComputePipelineState:psoQ4KSwiGLU];
        [e2 setBuffer:gMlpGate offset:0 atIndex:0];
        [e2 setBuffer:gMlpUp offset:0 atIndex:1];
        [e2 setBuffer:gMlpInter offset:0 atIndex:2];
        [e2 setBytes:&I length:sizeof(int) atIndex:3];
        [e2 dispatchThreads:MTLSizeMake((NSUInteger)I,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 endEncoding];

        // (3) y = D·inter
        id<MTLComputeCommandEncoder> e3 = [cb computeCommandEncoder];
        [e3 setComputePipelineState:psoQ4KGemv];
        [e3 setBuffer:gMlpInter offset:0 atIndex:1];
        [e3 setBuffer:(__bridge id<MTLBuffer>)D.buf offset:0 atIndex:0];
        [e3 setBuffer:yb offset:0 atIndex:2];
        [e3 setBytes:&D.nblk length:sizeof(int) atIndex:3];
        [e3 setBytes:&D.out  length:sizeof(int) atIndex:4];
        [e3 dispatchThreadgroups:MTLSizeMake((NSUInteger)D.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e3 endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        memcpy(y, yb.contents, (size_t)D.out * 4);
    }
}

// ---- Q6_K weight table (210-B super-blocks, separate stride from the 144-B Q4_K table) ----
// The Q6_K resident store backs the fused MLP's down_proj when a q4_k_m GGUF quantizes down to
// Q6_K. Its handles share gNQ4's id space with NO overlap by living in a separate array indexed by
// (id - MG_Q6_BASE): a wid >= MG_Q6_BASE means "Q6_K table, index wid-MG_Q6_BASE". Only the fused
// MLP's stage 3 (mg_q4k_mlp_q6down) consumes a Q6_K wid, so the disjoint id range never collides.
typedef struct {
    CFTypeRef buf; // retained id<MTLBuffer>, raw Q6_K bytes [out * nblk * 210]
    int out;
    int in;
    int nblk;
} Q6KW;

#define MG_MAX_Q6 8192
#define MG_Q6_BASE 1000000 // Q6_K wids are offset by this so they never alias a Q4_K wid
static Q6KW gQ6[MG_MAX_Q6];
static int gNQ6 = 0;

// mg_q6k_upload copies a row-major Q6_K payload (out rows, in == nblk*256, 210 B/super-block)
// verbatim into a resident device buffer and returns a handle >= MG_Q6_BASE, or -1 on failure.
int mg_q6k_upload(const unsigned char* raw, int out, int in) {
    if (raw == NULL || gDev == nil) return -1;
    if (!q4k_init()) return -1;
    if (in <= 0 || in % 256 != 0 || out <= 0) return -1;
    if (gNQ6 >= MG_MAX_Q6) {
        static int q6CapWarned = 0;
        if (!q6CapWarned) { q6CapWarned = 1; NSLog(@"mg_q6k_upload: Q6_K weight table full (%d)", MG_MAX_Q6); }
        return -1;
    }
    int nblk = in / 256;
    long bytes = (long)out * nblk * 210;
    id<MTLBuffer> b = [gDev newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
    if (b == nil) {
        NSLog(@"mg_q6k_upload: device buffer alloc failed for %.1f MB", (double)bytes / 1e6);
        return -1;
    }
    memcpy(b.contents, raw, (size_t)bytes);
    int idx = gNQ6++;
    gQ6[idx].buf = CFBridgingRetain(b);
    gQ6[idx].out = out;
    gQ6[idx].in = in;
    gQ6[idx].nblk = nblk;
    return MG_Q6_BASE + idx;
}

// mg_q4k_mlp_q6down is mg_q4k_mlp with a Q6_K down_proj: stages 1 (gate/up GEMV) and 2 (SwiGLU) are
// IDENTICAL — they run over the resident gMlpGate/gMlpUp/gMlpInter scratch — only stage 3 binds the
// Q6_K down weight (gQ6[down_wid-MG_Q6_BASE]) and the Q6_K GEMV pipeline. The whole MLP still runs in
// ONE command buffer. gate_wid/up_wid are Q4_K wids; down_wid is a Q6_K wid (>= MG_Q6_BASE).
void mg_q4k_mlp_q6down(int gate_wid, int up_wid, int down_wid, const float* x, float* y) {
    if (gate_wid < 0 || up_wid < 0 || gate_wid >= gNQ4 || up_wid >= gNQ4) return;
    if (down_wid < MG_Q6_BASE || (down_wid - MG_Q6_BASE) >= gNQ6) return;
    @autoreleasepool {
        Q4KW G = gQ4[gate_wid], U = gQ4[up_wid];
        Q6KW D = gQ6[down_wid - MG_Q6_BASE];
        int H = G.in;
        int I = G.out;
        q4k_grow_scratch((long)H, (long)D.out);
        q4k_grow_mlp((long)I);
        id<MTLBuffer> xb = gQXBuf, yb = gQYBuf;
        memcpy(xb.contents, x, (size_t)H * 4);

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];

        // (1) gate = G·x and up = U·x (independent), one encoder — IDENTICAL to mg_q4k_mlp.
        id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
        [e1 setComputePipelineState:psoQ4KGemv];
        [e1 setBuffer:xb offset:0 atIndex:1];
        [e1 setBuffer:(__bridge id<MTLBuffer>)G.buf offset:0 atIndex:0];
        [e1 setBuffer:gMlpGate offset:0 atIndex:2];
        [e1 setBytes:&G.nblk length:sizeof(int) atIndex:3];
        [e1 setBytes:&G.out  length:sizeof(int) atIndex:4];
        [e1 dispatchThreadgroups:MTLSizeMake((NSUInteger)G.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e1 setBuffer:(__bridge id<MTLBuffer>)U.buf offset:0 atIndex:0];
        [e1 setBuffer:gMlpUp offset:0 atIndex:2];
        [e1 setBytes:&U.nblk length:sizeof(int) atIndex:3];
        [e1 setBytes:&U.out  length:sizeof(int) atIndex:4];
        [e1 dispatchThreadgroups:MTLSizeMake((NSUInteger)U.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e1 endEncoding];

        // (2) inter = silu(gate) * up — IDENTICAL to mg_q4k_mlp.
        id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
        [e2 setComputePipelineState:psoQ4KSwiGLU];
        [e2 setBuffer:gMlpGate offset:0 atIndex:0];
        [e2 setBuffer:gMlpUp offset:0 atIndex:1];
        [e2 setBuffer:gMlpInter offset:0 atIndex:2];
        [e2 setBytes:&I length:sizeof(int) atIndex:3];
        [e2 dispatchThreads:MTLSizeMake((NSUInteger)I,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 endEncoding];

        // (3) y = D·inter with the Q6_K GEMV pipeline (the only line that differs from mg_q4k_mlp).
        id<MTLComputeCommandEncoder> e3 = [cb computeCommandEncoder];
        [e3 setComputePipelineState:psoQ6KGemv];
        [e3 setBuffer:gMlpInter offset:0 atIndex:1];
        [e3 setBuffer:(__bridge id<MTLBuffer>)D.buf offset:0 atIndex:0];
        [e3 setBuffer:yb offset:0 atIndex:2];
        [e3 setBytes:&D.nblk length:sizeof(int) atIndex:3];
        [e3 setBytes:&D.out  length:sizeof(int) atIndex:4];
        [e3 dispatchThreadgroups:MTLSizeMake((NSUInteger)D.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e3 endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        memcpy(y, yb.contents, (size_t)D.out * 4);
    }
}

static int q4k_upload_preflight(int out, int in, int* nblk, long* bytes) {
    if (gDev == nil) return -1;
    if (!q4k_init()) return -1;
    if (in % 256 != 0) return -1;
    if (gNQ4 >= MG_MAX_Q4) {
        static int capWarned = 0;
        if (!capWarned) { capWarned = 1; NSLog(@"mg_q4k_upload: q4_k weight table full (%d)", MG_MAX_Q4); }
        return -1;
    }
    *nblk = in / 256;
    *bytes = (long)out * *nblk * 144;
    return 0;
}

static long q4k_page_round(long bytes) {
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 1) return bytes;
    long rem = bytes % page;
    if (rem == 0) return bytes;
    return bytes + (page - rem);
}

// mg_q4k_upload_nocopy wraps a row-major q4_k payload (out rows, in == nblk*256) as a shared
// Metal buffer without copying. The caller owns and pins raw until mg_q4k_reset releases the
// retained buffer. This is the Apple-unified-memory residency path: the GPU reads the same
// GGUF bytes already held by the model, so the first prefill does not pay an 8+ GB memcpy.
int mg_q4k_upload_nocopy(const unsigned char* raw, int out, int in) {
    if (raw == NULL) return -1;
    int nblk = 0;
    long bytes = 0;
    if (q4k_upload_preflight(out, in, &nblk, &bytes) != 0) return -1;
    id<MTLBuffer> b = [gDev newBufferWithBytesNoCopy:(void*)raw
                                              length:(NSUInteger)q4k_page_round(bytes)
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
    if (b == nil) {
        static int noCopyWarned = 0;
        if (!noCopyWarned) {
            noCopyWarned = 1;
            NSLog(@"mg_q4k_upload_nocopy: Metal rejected no-copy shared buffer; falling back to copy upload");
        }
        return -1;
    }
    return q4k_register_buffer(b, out, in, nblk);
}

// mg_q4k_upload copies a row-major q4_k payload (out rows, in == nblk*256) verbatim into a
// resident device buffer and returns an integer handle (>=0), or -1 on failure. The bytes ARE
// the GGUF bytes (no transform), so the kernel dequants the same super-blocks llama.cpp does.
int mg_q4k_upload(const unsigned char* raw, int out, int in) {
    int nblk = 0;
    long bytes = 0;
    if (q4k_upload_preflight(out, in, &nblk, &bytes) != 0) return -1;
    id<MTLBuffer> b = [gDev newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
    if (b == nil) {
        NSLog(@"mg_q4k_upload: device buffer alloc failed for %.1f MB", (double)bytes / 1e6);
        return -1;
    }
    memcpy(b.contents, raw, (size_t)bytes);
    return q4k_register_buffer(b, out, in, nblk);
}

// mg_q4k_gemv computes y[out] = W[wid] · x (one f32 activation row, length in). f32 in/out.
void mg_q4k_gemv(int wid, const float* x, float* y) {
    if (wid < 0 || wid >= gNQ4) return;
    @autoreleasepool {
        Q4KW W = gQ4[wid];
        q4k_grow_scratch(W.in, W.out);
        id<MTLBuffer> wbuf = (__bridge id<MTLBuffer>)W.buf;
        id<MTLBuffer> xb = gQXBuf;
        id<MTLBuffer> yb = gQYBuf;
        memcpy(xb.contents, x, (size_t)W.in * 4);

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:psoQ4KGemv];
        [e setBuffer:wbuf offset:0 atIndex:0];
        [e setBuffer:xb   offset:0 atIndex:1];
        [e setBuffer:yb   offset:0 atIndex:2];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        // One threadgroup (a single 32-lane SIMD group) per output row: the 32 lanes split the
        // row's super-blocks and reduce via simd_sum. dispatchThreadgroups (not dispatchThreads)
        // because the kernel keys off threadgroup_position_in_grid = the output row index.
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        memcpy(y, yb.contents, (size_t)W.out * 4);
    }
}

// mg_q4k_gemv_batch runs n decode GEMVs of the SAME weight wid into ONE command buffer (one
// commit + one waitUntilCompleted): Xcat is n contiguous activation rows (n*in floats), Ycat
// receives n result rows (n*out floats). It exists to MEASURE how much of mg_q4k_gemv's
// per-call cost is the CPU<->GPU submission/sync round-trip vs the kernel: if n GEMVs here cost
// ~n*kernel + one round-trip (not n round-trips), the decode wall is the per-op command buffer,
// and the fix is a one-command-buffer resident forward (issue #67). The encoder re-binds only
// the X/Y offsets between dispatches; the weight + dims are set once.
void mg_q4k_gemv_batch(int wid, const float* Xcat, int n, float* Ycat) {
    if (wid < 0 || wid >= gNQ4 || n <= 0) return;
    @autoreleasepool {
        Q4KW W = gQ4[wid];
        q4k_grow_scratch((long)n * W.in, (long)n * W.out);
        id<MTLBuffer> wbuf = (__bridge id<MTLBuffer>)W.buf;
        id<MTLBuffer> xb = gQXBuf;
        id<MTLBuffer> yb = gQYBuf;
        memcpy(xb.contents, Xcat, (size_t)n * W.in * 4);

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:psoQ4KGemv];
        [e setBuffer:wbuf offset:0 atIndex:0];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        for (int i = 0; i < n; i++) {
            [e setBuffer:xb offset:(NSUInteger)((long)i * W.in  * 4) atIndex:1];
            [e setBuffer:yb offset:(NSUInteger)((long)i * W.out * 4) atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        memcpy(Ycat, yb.contents, (size_t)n * W.out * 4);
    }
}

// mg_q4k_gemv_group runs n decode GEMVs that SHARE one activation x (length in) but apply n
// DIFFERENT resident q4_k weights, into ONE command buffer (one commit/waitUntilCompleted). This
// is the live decode access pattern: a layer's q/k/v (or gate/up, or the GDN in_proj quad) all
// read the same post-norm activation. Each weight i writes Ycat[yoff[i] .. yoff[i]+out_i); yoff
// has n+1 entries (yoff[n] = total y elems). The fixed ~submit/sync overhead is paid ONCE for the
// group and the GPU pipelines the n dispatches — the per-token win the resident forward needs.
void mg_q4k_gemv_group(const int* wids, int n, const float* x, float* Ycat, const int* yoff) {
    if (n <= 0) return;
    @autoreleasepool {
        int in = gQ4[wids[0]].in;
        long ytot = (long)yoff[n];
        q4k_grow_scratch((long)in, ytot);
        id<MTLBuffer> xb = gQXBuf;
        id<MTLBuffer> yb = gQYBuf;
        memcpy(xb.contents, x, (size_t)in * 4);

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:psoQ4KGemv];
        [e setBuffer:xb offset:0 atIndex:1]; // shared activation for every weight in the group
        for (int i = 0; i < n; i++) {
            Q4KW Wi = gQ4[wids[i]];
            [e setBuffer:(__bridge id<MTLBuffer>)Wi.buf offset:0 atIndex:0];
            [e setBuffer:yb offset:(NSUInteger)((long)yoff[i] * 4) atIndex:2];
            [e setBytes:&Wi.nblk length:sizeof(int) atIndex:3];
            [e setBytes:&Wi.out  length:sizeof(int) atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)Wi.out, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        memcpy(Ycat, yb.contents, (size_t)ytot * 4);
    }
}

// mg_q4k_gemm computes Y[P, out] = X[P, in] · W[wid]^T (batched prefill GEMM). f32 in/out,
// row-major; Y must hold P*out floats.
void mg_q4k_gemm(int wid, const float* X, int P, float* Y) {
    if (wid < 0 || wid >= gNQ4 || P <= 0) return;
    @autoreleasepool {
        Q4KW W = gQ4[wid];
        q4k_grow_scratch((long)P * W.in, (long)P * W.out);
        id<MTLBuffer> wbuf = (__bridge id<MTLBuffer>)W.buf;
        id<MTLBuffer> xb = gQXBuf;
        id<MTLBuffer> yb = gQYBuf;
        memcpy(xb.contents, X, (size_t)P * W.in * 4);

        // 2D tile: each threadgroup owns a BM×BN output block (BM rows × BN tokens), staging both
        // the weight rows and the token activations into threadgroup memory once per super-block
        // (issue #1085). Grid.x = ceil(out/BM) row-blocks; the token axis is tiled into BN-wide
        // dispatches, all in ONE command buffer so launch overhead is paid once for the whole GEMM.
        const int BM = 64;  // output rows per threadgroup; must match Q4K_BM in the MSL source
        const int BN = 64;  // token-tile width;            must match Q4K_BN in the MSL source
        const int TG = 256; // threads per threadgroup (TGX*TGY); must match Q4K_TG in the MSL source
        int rowBlocks = (W.out + BM - 1) / BM;
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:psoQ4KGemm];
        [e setBuffer:wbuf offset:0 atIndex:0];
        [e setBuffer:xb   offset:0 atIndex:1];
        [e setBuffer:yb   offset:0 atIndex:2];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        [e setBytes:&P      length:sizeof(int) atIndex:5];
        for (int t0 = 0; t0 < P; t0 += BN) {
            int nt = P - t0;
            if (nt > BN) nt = BN;
            [e setBytes:&t0 length:sizeof(int) atIndex:6];
            [e setBytes:&nt length:sizeof(int) atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)rowBlocks, 1, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)TG, 1, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        memcpy(Y, yb.contents, (size_t)P * W.out * 4);
    }
}

// mg_q4k_reset releases every resident q4_k weight buffer and the reused scratch, returning
// the q4_k table to empty. Mirrors mg_reset's role for the f16 table; the compiled pipelines
// stay live. Call only when no Q4KWeight handle is still in use.
void mg_q4k_reset(void) {
    for (int i = 0; i < gNQ4; i++) {
        if (gQ4[i].buf != NULL) {
            CFBridgingRelease(gQ4[i].buf);
            gQ4[i].buf = NULL;
        }
    }
    gNQ4 = 0;
    for (int i = 0; i < gNQ6; i++) {
        if (gQ6[i].buf != NULL) {
            CFBridgingRelease(gQ6[i].buf);
            gQ6[i].buf = NULL;
        }
    }
    gNQ6 = 0;
    gQXBuf = nil; gQXCap = 0;
    gQYBuf = nil; gQYCap = 0;
    gMlpGate = nil; gMlpUp = nil; gMlpInter = nil; gMlpCap = 0;
}
