//go:build darwin && cgo && fakmetal

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

// q4k_gemm: the TILED prefill GEMM. ONE threadgroup per output row processes a tile of up to
// Q4K_TG tokens [t0, t0+nt): for each super-block it cooperatively dequants the 256 weights
// ONCE into threadgroup memory (wbuf), then each token-thread dots that shared block against
// its activation slice. So each weight is read+dequanted once per tile and REUSED across all nt
// tokens — vs the per-(o,t) kernel that re-streamed+re-dequanted the whole row P times (which
// made the real-shape 27B prefill slower than CPU). The C side encodes one dispatch per token
// tile into a single command buffer, so launch overhead is paid once per GEMM.
#define Q4K_TG 128
kernel void q4k_gemm(device const uchar* W [[buffer(0)]],
                     device const float* X [[buffer(1)]],
                     device float*       Y [[buffer(2)]],
                     constant int&    nblk [[buffer(3)]],
                     constant int&     out [[buffer(4)]],
                     constant int&       P [[buffer(5)]],
                     constant int&      t0 [[buffer(6)]],
                     constant int&      nt [[buffer(7)]],
                     uint o   [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]]) {
    if (o >= (uint)out) return;
    threadgroup float wbuf[256];
    int in = nblk * 256;
    device const uchar* row = W + (long)o * nblk * 144;
    int token = t0 + (int)lid;
    bool active = (int)lid < nt;
    device const float* xs = active ? (X + (long)token * in) : X;
    float acc = 0.0f;
    for (int b = 0; b < nblk; b++) {
        device const uchar* blk = row + (long)b * 144;
        float d  = (float)(*(device const half*)(blk + 0));
        float dm = (float)(*(device const half*)(blk + 2));
        device const uchar* scales = blk + 4;
        device const uchar* q = blk + 16;
        // Cooperative dequant of this super-block's 256 weights into wbuf (each thread strides).
        for (int w = (int)lid; w < 256; w += Q4K_TG) {
            int p = w >> 6;          // 64-weight pair index 0..3
            int r = w & 63;
            int sub, qidx;
            uchar nib;
            if (r < 32) { sub = p * 2;     qidx = p * 32 + r;        nib = q[qidx] & 0x0f; }
            else        { sub = p * 2 + 1; qidx = p * 32 + (r - 32); nib = q[qidx] >> 4;   }
            float2 sm = q4k_scale_min(sub, scales);
            wbuf[w] = d * sm.x * (float)nib - dm * sm.y;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            device const float* xb = xs + (long)b * 256;
            float s = 0.0f;
            for (int w = 0; w < 256; w++) s += wbuf[w] * xb[w];
            acc += s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // wbuf reused next b
    }
    if (active) Y[(long)token * out + o] = acc;
}
)MSL";

static id<MTLComputePipelineState> psoQ4KGemv, psoQ4KGemm;
static int gQ4KReady;

static int q4k_init(void) {
    if (gQ4KReady) return 1;
    if (gDev == nil) return 0;
    NSError *err = nil;
    id<MTLLibrary> lib = [gDev newLibraryWithSource:kQ4KSrc options:nil error:&err];
    if (lib == nil) { NSLog(@"q4k: library compile failed: %@", err); return 0; }
    psoQ4KGemv = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q4k_gemv"] error:&err];
    psoQ4KGemm = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q4k_gemm"] error:&err];
    if (!psoQ4KGemv || !psoQ4KGemm) { NSLog(@"q4k: pipeline build failed: %@", err); return 0; }
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

// mg_q4k_upload copies a row-major q4_k payload (out rows, in == nblk*256) verbatim into a
// resident device buffer and returns an integer handle (>=0), or -1 on failure. The bytes ARE
// the GGUF bytes (no transform), so the kernel dequants the same super-blocks llama.cpp does.
int mg_q4k_upload(const unsigned char* raw, int out, int in) {
    if (gDev == nil) return -1;
    if (!q4k_init()) return -1;
    if (in % 256 != 0) return -1;
    if (gNQ4 >= MG_MAX_Q4) {
        static int capWarned = 0;
        if (!capWarned) { capWarned = 1; NSLog(@"mg_q4k_upload: q4_k weight table full (%d)", MG_MAX_Q4); }
        return -1;
    }
    int nblk = in / 256;
    long bytes = (long)out * nblk * 144;
    id<MTLBuffer> b = [gDev newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
    if (b == nil) {
        NSLog(@"mg_q4k_upload: device buffer alloc failed for %.1f MB", (double)bytes / 1e6);
        return -1;
    }
    memcpy(b.contents, raw, (size_t)bytes);
    int id = gNQ4++;
    gQ4[id].buf = CFBridgingRetain(b);
    gQ4[id].out = out;
    gQ4[id].in = in;
    gQ4[id].nblk = nblk;
    return id;
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

        // Tile the P tokens into chunks of Q4K_TG (128), encoding one dispatch per tile into a
        // SINGLE command buffer: each tile reads every weight once (the tiled kernel reuses the
        // dequanted super-block across the tile's tokens), and the launch overhead is paid once
        // for the whole GEMM instead of per tile.
        const int TG = 128; // must match Q4K_TG in the MSL source
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:psoQ4KGemm];
        [e setBuffer:wbuf offset:0 atIndex:0];
        [e setBuffer:xb   offset:0 atIndex:1];
        [e setBuffer:yb   offset:0 atIndex:2];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        [e setBytes:&P      length:sizeof(int) atIndex:5];
        for (int t0 = 0; t0 < P; t0 += TG) {
            int nt = P - t0;
            if (nt > TG) nt = TG;
            [e setBytes:&t0 length:sizeof(int) atIndex:6];
            [e setBytes:&nt length:sizeof(int) atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, 1, 1)
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
    gQXBuf = nil; gQXCap = 0;
    gQYBuf = nil; gQYCap = 0;
}
