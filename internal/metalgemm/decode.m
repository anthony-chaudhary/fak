//go:build darwin && cgo && fakmetal

// decode.m — the GPU-resident Q8 decode forward (issue #67). The DECODE twin of forward.m's
// mg_prefill: per token (P=1), the f16 activation stays on-device across every layer's seven
// projections (Q8 dequant-GEMV), RMSNorm, RoPE, GQA attention over the resident KV, SwiGLU and
// the residual adds — all encoded into ONE command buffer with a single CPU/GPU sync. Only the
// new token's embedding goes in and the pre-final-norm hidden + the new K/V row per layer come
// out (the caller applies the final norm + head and appends the KV row to its f32 cache).
//
// Why this exists. The live decode runs ~7 projection matmuls × nLayers as SEPARATE command
// buffers, each ~360 us launch/sync-bound (MAC-QWEN36-...-PERF-DIAGNOSIS): the kernel is correct
// but the per-op submit overhead dominates, pinning decode far below the llama.cpp-Metal bar. A
// one-command-buffer forward pays that overhead ONCE per token and lets the GPU pipeline the
// dispatches — the lever the BenchmarkMetalQ4KGemvBatch witness measured (11% -> 59% of device BW).
//
// Precision. Weights are Q8_0 (the same int8 codes + per-32-block f32 scale the CPU q8 path holds,
// resident via q8.m's table); the activation flows as f16 (like mg_prefill). Each projection is a
// dequant-GEMV: y = Sum (code*scale)*x with x in f16 — more accurate than the CPU int8xint8 dot
// (no activation quant), so the greedy token sequence matches the CPU Q8 path (token-parity gate,
// the same bar q4k.m's TestMetalQ4KDecodeMatchesCPU uses). Q8 streams ~half the bytes of f16, so
// this is the precision that can beat the CPU decode and approach llama.cpp-Metal Q8.
//
// Scope (v0): the dense Qwen2.5 architecture (q/k/v/o + gate/up/down, attention bias, full
// attention, standard RoPE, no QK-norm, no attn softcap, no sliding window). The per-step KV
// context is re-uploaded f32->f16 each token (cheap vs the weight stream); persistent on-GPU KV is
// a follow-on. The Gated-DeltaNet hybrid (27B) needs the gdn.m recurrence and is a separate path.

#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>
#include <math.h>
#include <stdlib.h>

// Device + queue are owned by metal.m (mg_init); we reuse them.
extern id<MTLDevice>       gDev;
extern id<MTLCommandQueue> gQueue;

// f16<->f32 helpers (metal.m) and the f16 norm/bias table (forward.m's gW). Norm/bias vectors are
// uploaded via mg_upload_vec into gW, so the decode forward reads them from the same f16 table.
void mg_f32_to_f16(const float *src, __fp16 *dst, long n);
void mg_f16_to_f32(const __fp16 *src, float *dst, long n);
typedef struct { CFTypeRef buf; int out; int in; } MGWeight;
extern MGWeight gW[];

// Q8 resident weight buffers (q8.m) — bound directly into the decode encoder.
id<MTLBuffer> mg_q8_codes_buf(int wid);
id<MTLBuffer> mg_q8_scales_buf(int wid);
void mg_q8_dims(int wid, int *out, int *in, int *nblk);

// ---- MSL kernels (f16 activations; compiled once at runtime) ----
static NSString *kDecSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// q8dq_gemv: y[out](f16) = dequant(W_q8) . x(f16). ONE threadgroup (a 32-lane SIMD group) per
// output row; the 32 lanes split the row's 32-wide Q8_0 blocks and reduce via simd_sum. x is the
// resident f16 activation (no activation quantization), so this is the f16xQ8 dequant-GEMV — the
// per-block sum of int8(code)*f16(x), scaled by the per-block weight scale wd[b].
#define Q8DQ_ROWS_PER_TG 8
kernel void q8dq_gemv(device const char*  W    [[buffer(0)]],  // out*in int8 codes, row-major
                      device const float* WD   [[buffer(1)]],  // out*nblk f32 block scales
                      device const half*  X    [[buffer(2)]],  // in f16 activation
                      device half*        Y    [[buffer(3)]],
                      constant int&       nblk [[buffer(4)]],
                      constant int&       out_ [[buffer(5)]],
                      uint tgid [[threadgroup_position_in_grid]],
                      uint litg [[thread_index_in_threadgroup]]) {
    // Q8DQ_ROWS_PER_TG simdgroups per threadgroup (256 threads): one output row per simdgroup. Packing
    // 8 rows into a threadgroup keeps far more memory requests in flight per GPU core than a lone
    // 32-thread threadgroup, hiding the int8 weight-stream latency a single GEMV can't — the decode
    // bandwidth lever. Each simdgroup folds its row with a char4×half4 dequant-dot + one simd_sum.
    uint sg   = litg / 32;
    uint lane = litg % 32;
    uint o = tgid * Q8DQ_ROWS_PER_TG + sg;
    if (o >= (uint)out_) return;
    device const char*  wrow = W  + (long)o * nblk * 32;
    device const float* wd   = WD + (long)o * nblk;
    float acc = 0.0f;
    for (int b = (int)lane; b < nblk; b += 32) {
        device const char4* wb = (device const char4*)(wrow + (long)b * 32);
        device const half4* xb = (device const half4*)(X    + (long)b * 32);
        float s = 0.0f;
        for (int i = 0; i < 8; i++) s += dot(float4(wb[i]), float4(xb[i]));
        acc += s * wd[b];
    }
    acc = simd_sum(acc);
    if (lane == 0) Y[o] = half(acc);
}

kernel void d_rmsnorm(device const half* X [[buffer(0)]],
                      device const half* W [[buffer(1)]],
                      device half* Out [[buffer(2)]],
                      constant uint& H [[buffer(3)]],
                      constant float& eps [[buffer(4)]],
                      uint t [[thread_position_in_grid]]) {
    device const half* x = X + t*H;
    float ss = 0.0f;
    for (uint i=0;i<H;i++){ float v=float(x[i]); ss += v*v; }
    float inv = rsqrt(ss/float(H) + eps);
    device half* o = Out + t*H;
    for (uint i=0;i<H;i++){ o[i] = half(float(x[i])*inv*float(W[i])); }
}

kernel void d_addbias(device half* Buf [[buffer(0)]],
                      device const half* B [[buffer(1)]],
                      constant uint& n [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    Buf[gid] = half(float(Buf[gid]) + float(B[gid % n]));
}

// d_rope: rotary embedding of a [nHeads*hd] row at absolute position `base` (one token). Matches
// forward.m's rope_k (theta^(-2j/hd)).
kernel void d_rope(device half* Buf [[buffer(0)]],
                   constant uint& nHeads [[buffer(1)]],
                   constant uint& hd [[buffer(2)]],
                   constant uint& base [[buffer(3)]],
                   constant float& theta [[buffer(4)]],
                   uint gid [[thread_position_in_grid]]) {
    uint half_ = hd/2;
    uint perTok = nHeads*half_;
    if (gid >= perTok) return;
    uint head = gid / half_;
    uint j = gid % half_;
    device half* hv = Buf + head*hd;
    float pos = float(base);
    float inv = pow(theta, -2.0f*float(j)/float(hd));
    float ang = pos*inv;
    float c = cos(ang), s = sin(ang);
    float a = float(hv[j]); float b = float(hv[j+half_]);
    hv[j] = half(a*c - b*s);
    hv[j+half_] = half(b*c + a*s);
}

// attn_decode: ONE simdgroup (32 lanes) per query head h. The single query (the new token, already
// roped) attends to keys 0..ctx-1 via online softmax over the resident KV (K,V hold ctx rows of
// w=nKV*hd). GQA: head h reads kv head h/grp. The 32 lanes split hd (hd<=128 => <=4 dims/lane) so
// q/acc stay in registers and the per-key QK dot is a simd_sum. Mirrors forward.m's attn_k.
kernel void attn_decode(device const half* Q [[buffer(0)]],   // [nH*hd] roped query
                        device const half* K [[buffer(1)]],   // [ctx*w] post-rope keys
                        device const half* V [[buffer(2)]],   // [ctx*w]
                        device half* Out [[buffer(3)]],       // [nH*hd]
                        constant uint& ctx [[buffer(4)]],
                        constant uint& nH [[buffer(5)]],
                        constant uint& hd [[buffer(6)]],
                        constant uint& w [[buffer(7)]],
                        constant uint& grp [[buffer(8)]],
                        constant float& scale [[buffer(9)]],
                        uint gid [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint h = gid / 32u;
    if (h >= nH) return;
    uint kvh = h / grp;
    device const half* q = Q + h*hd;
    float qreg[4]; uint nd = 0;
    for (uint d=lane; d<hd; d+=32u) qreg[nd++] = float(q[d]);
    float acc[4] = {0,0,0,0};
    float m = -INFINITY, l = 0.0f;
    for (uint j=0; j<ctx; j++) {
        device const half* k = K + j*w + kvh*hd;
        float partial = 0.0f; uint idx = 0;
        for (uint d=lane; d<hd; d+=32u) partial += qreg[idx++]*float(k[d]);
        float sc = simd_sum(partial) * scale;
        float mNew = max(m, sc);
        float corr = exp(m - mNew);
        float p = exp(sc - mNew);
        l = l*corr + p;
        device const half* vv = V + j*w + kvh*hd;
        idx = 0;
        for (uint d=lane; d<hd; d+=32u) { acc[idx] = acc[idx]*corr + p*float(vv[d]); idx++; }
        m = mNew;
    }
    device half* o = Out + h*hd;
    float invl = (l>0.0f) ? 1.0f/l : 0.0f;
    uint idx = 0;
    for (uint d=lane; d<hd; d+=32u) o[d] = half(acc[idx++]*invl);
}

kernel void d_silumul(device half* G [[buffer(0)]],
                      device const half* U [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    float g = float(G[i]);
    float s = g / (1.0f + exp(-g));
    G[i] = half(s * float(U[i]));
}

kernel void d_add(device half* X [[buffer(0)]],
                  device const half* Y [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    X[i] = half(float(X[i]) + float(Y[i]));
}
)MSL";

// ---- model registration ----
typedef struct {
    int q, k, v, o, gate, up, down;  // Q8 weight ids (q8.m table)
    int inNorm, postNorm;            // f16 vector ids (gW table)
    int qb, kb, vb;                  // f16 bias ids (gW) or -1
} DecLayer;

#define DEC_MAXL 128
static int gDecNL, gDecH, gDecHd, gDecNH, gDecNKV, gDecI, gDecAttnBias;
static float gDecEps, gDecTheta, gDecScale;
static DecLayer gDecL[DEC_MAXL];

static id<MTLComputePipelineState> psoDGemv, psoDNorm, psoDBias, psoDRope, psoDAttn, psoDSilu, psoDAdd;
static int gDecReady;

static int dec_init(void) {
    if (gDecReady) return 1;
    if (gDev == nil) return 0;
    NSError *err = nil;
    id<MTLLibrary> lib = [gDev newLibraryWithSource:kDecSrc options:nil error:&err];
    if (lib == nil) { NSLog(@"decode: library compile failed: %@", err); return 0; }
    #define DPSO(name) [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err]
    psoDGemv = DPSO(@"q8dq_gemv");
    psoDNorm = DPSO(@"d_rmsnorm");
    psoDBias = DPSO(@"d_addbias");
    psoDRope = DPSO(@"d_rope");
    psoDAttn = DPSO(@"attn_decode");
    psoDSilu = DPSO(@"d_silumul");
    psoDAdd  = DPSO(@"d_add");
    #undef DPSO
    if (!psoDGemv || !psoDNorm || !psoDBias || !psoDRope || !psoDAttn || !psoDSilu || !psoDAdd) {
        NSLog(@"decode: pipeline build failed: %@", err); return 0;
    }
    gDecReady = 1;
    return 1;
}

void mg_decode_config(int nLayers, int H, int hd, int nH, int nKV, int Im,
                      float eps, float theta, float scale, int attnBias) {
    gDecNL = nLayers; gDecH = H; gDecHd = hd; gDecNH = nH; gDecNKV = nKV; gDecI = Im;
    gDecEps = eps; gDecTheta = theta; gDecScale = scale; gDecAttnBias = attnBias;
}

void mg_decode_layer(int layer, int q, int k, int v, int o, int gate, int up, int down,
                     int inNorm, int postNorm, int qb, int kb, int vb) {
    if (layer < 0 || layer >= DEC_MAXL) return;
    gDecL[layer] = (DecLayer){q, k, v, o, gate, up, down, inNorm, postNorm, qb, kb, vb};
}

void mg_decode_reset(void) {
    gDecNL = gDecH = gDecHd = gDecNH = gDecNKV = gDecI = gDecAttnBias = 0;
    gDecEps = gDecTheta = gDecScale = 0.0f;
    for (int i = 0; i < DEC_MAXL; i++) gDecL[i] = (DecLayer){0};
}

// ---- encode helpers (one command buffer / encoder per decode step) ----
static id<MTLCommandBuffer> gDCB;
static id<MTLComputeCommandEncoder> gDEnc;

static id<MTLBuffer> dbuf(long elems) { // f16 device buffer
    return [gDev newBufferWithLength:(NSUInteger)(elems * 2) options:MTLResourceStorageModeShared];
}
static id<MTLBuffer> wbufOfDec(int wid) { return (__bridge id<MTLBuffer>)gW[wid].buf; }
static id<MTLComputeCommandEncoder> denc(void) {
    if (gDEnc == nil) gDEnc = [gDCB computeCommandEncoder];
    return gDEnc;
}
static void dendEnc(void) {
    if (gDEnc != nil) { [gDEnc endEncoding]; gDEnc = nil; }
}
static void d1d(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> pso, NSUInteger n) {
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > n) tg = n;
    if (tg == 0) tg = 1;
    [e setComputePipelineState:pso];
    [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
}

// q8 dequant-GEMV: Y[out](f16) = dequant(W_q8[wid]) . X(f16). One 32-lane threadgroup per row.
static void d_gemv(int wid, id<MTLBuffer> X, id<MTLBuffer> Y) {
    int out, in, nblk; mg_q8_dims(wid, &out, &in, &nblk);
    id<MTLComputeCommandEncoder> e = denc();
    [e setComputePipelineState:psoDGemv];
    [e setBuffer:mg_q8_codes_buf(wid)  offset:0 atIndex:0];
    [e setBuffer:mg_q8_scales_buf(wid) offset:0 atIndex:1];
    [e setBuffer:X offset:0 atIndex:2];
    [e setBuffer:Y offset:0 atIndex:3];
    [e setBytes:&nblk length:4 atIndex:4];
    [e setBytes:&out  length:4 atIndex:5];
    // 8 rows per threadgroup (256 threads = 8 simdgroups); ceil(out/8) threadgroups cover all rows.
    NSUInteger ntg = (NSUInteger)((out + 7) / 8);
    [e dispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}

static void d_norm(id<MTLBuffer> X, int normID, id<MTLBuffer> Out) {
    id<MTLComputeCommandEncoder> e = denc();
    [e setComputePipelineState:psoDNorm];
    [e setBuffer:X offset:0 atIndex:0];
    [e setBuffer:wbufOfDec(normID) offset:0 atIndex:1];
    [e setBuffer:Out offset:0 atIndex:2];
    uint H = gDecH; [e setBytes:&H length:4 atIndex:3];
    [e setBytes:&gDecEps length:4 atIndex:4];
    [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
}

static void d_bias(id<MTLBuffer> Buf, int biasID, int n) {
    id<MTLComputeCommandEncoder> e = denc();
    [e setComputePipelineState:psoDBias];
    [e setBuffer:Buf offset:0 atIndex:0];
    [e setBuffer:wbufOfDec(biasID) offset:0 atIndex:1];
    uint nn = n; [e setBytes:&nn length:4 atIndex:2];
    d1d(e, psoDBias, (NSUInteger)n);
}

static void d_rope_at(id<MTLBuffer> Buf, int nHeads, int base) {
    id<MTLComputeCommandEncoder> e = denc();
    [e setComputePipelineState:psoDRope];
    [e setBuffer:Buf offset:0 atIndex:0];
    uint nh = nHeads, hd = gDecHd, b = base; [e setBytes:&nh length:4 atIndex:1];
    [e setBytes:&hd length:4 atIndex:2];
    [e setBytes:&b length:4 atIndex:3];
    [e setBytes:&gDecTheta length:4 atIndex:4];
    d1d(e, psoDRope, (NSUInteger)nHeads*(gDecHd/2));
}

static void d_attn(id<MTLBuffer> Q, id<MTLBuffer> K, id<MTLBuffer> V, id<MTLBuffer> Out, int ctx) {
    id<MTLComputeCommandEncoder> e = denc();
    [e setComputePipelineState:psoDAttn];
    [e setBuffer:Q offset:0 atIndex:0];
    [e setBuffer:K offset:0 atIndex:1];
    [e setBuffer:V offset:0 atIndex:2];
    [e setBuffer:Out offset:0 atIndex:3];
    uint c = ctx, nH = gDecNH, hd = gDecHd, w = gDecNKV*gDecHd, grp = gDecNH/gDecNKV;
    [e setBytes:&c length:4 atIndex:4];
    [e setBytes:&nH length:4 atIndex:5];
    [e setBytes:&hd length:4 atIndex:6];
    [e setBytes:&w length:4 atIndex:7];
    [e setBytes:&grp length:4 atIndex:8];
    [e setBytes:&gDecScale length:4 atIndex:9];
    NSUInteger total = (NSUInteger)gDecNH*32;
    NSUInteger tg = psoDAttn.maxTotalThreadsPerThreadgroup; tg -= tg % 32; if (tg > total) tg = total; if (tg == 0) tg = 32;
    [e dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
}

static void d_silu(id<MTLBuffer> G, id<MTLBuffer> U, int n) {
    id<MTLComputeCommandEncoder> e = denc();
    [e setComputePipelineState:psoDSilu];
    [e setBuffer:G offset:0 atIndex:0];
    [e setBuffer:U offset:0 atIndex:1];
    d1d(e, psoDSilu, (NSUInteger)n);
}

static void d_add_buf(id<MTLBuffer> X, id<MTLBuffer> Y, int n) {
    id<MTLComputeCommandEncoder> e = denc();
    [e setComputePipelineState:psoDAdd];
    [e setBuffer:X offset:0 atIndex:0];
    [e setBuffer:Y offset:0 atIndex:1];
    d1d(e, psoDAdd, (NSUInteger)n);
}

// mg_decode_step runs one decode token through the whole model on the GPU in ONE command buffer.
// xEmbed: f32[H] (the new token's embedding). Kctx/Vctx: f32[nL*L*w] (the per-layer post-RoPE K and
// V already in the CPU cache, w = nKV*hd). L: number of cached positions (the new token's absolute
// position == L). Outputs: lastPre f32[H] (pre-final-norm hidden — caller applies final norm+head);
// newKraw/newKpost/newV f32[nL*w] (the new token's per-layer pre-RoPE K, post-RoPE K, V — caller
// appends to its f32 cache). Returns 1 on success, 0 if the backend declined.
int mg_decode_step(const float *xEmbed, const float *Kctx, const float *Vctx, int L,
                   float *lastPre, float *newKraw, float *newKpost, float *newV) {
    if (!dec_init()) return 0;
    int prof = getenv("FAK_DECODE_PROF") != NULL;
    CFTimeInterval t0 = prof ? CFAbsoluteTimeGetCurrent() : 0;
    @autoreleasepool {
        int H = gDecH, hd = gDecHd, nH = gDecNH, nKV = gDecNKV, Im = gDecI, w = nKV*hd, qrow = nH*hd;
        int ctx = L + 1;

        id<MTLBuffer> Xb = dbuf(H);
        mg_f32_to_f16(xEmbed, (__fp16 *)Xb.contents, (long)H);
        id<MTLBuffer> Xn = dbuf(H), Xn2 = dbuf(H);
        id<MTLBuffer> Qb = dbuf(qrow), K1 = dbuf(w), V1 = dbuf(w);
        id<MTLBuffer> attn = dbuf(qrow), tmpH = dbuf(H), Gb = dbuf(Im), Ub = dbuf(Im);
        // per-layer resident KV (context + the new row at index L) and the new pre-RoPE K stash.
        id<MTLBuffer> Kbuf[DEC_MAXL] = {0}, Vbuf[DEC_MAXL] = {0}, Kraw[DEC_MAXL] = {0};
        for (int l = 0; l < gDecNL; l++) {
            Kbuf[l] = dbuf((long)ctx*w); Vbuf[l] = dbuf((long)ctx*w); Kraw[l] = dbuf(w);
            if (L > 0) {
                mg_f32_to_f16(Kctx + (long)l*L*w, (__fp16 *)Kbuf[l].contents, (long)L*w);
                mg_f32_to_f16(Vctx + (long)l*L*w, (__fp16 *)Vbuf[l].contents, (long)L*w);
            }
        }

        CFTimeInterval tHost = prof ? CFAbsoluteTimeGetCurrent() : 0;

        gDCB = [gQueue commandBuffer];
        gDEnc = nil;
        id<MTLBlitCommandEncoder> blit;

        for (int l = 0; l < gDecNL; l++) {
            DecLayer L_ = gDecL[l];
            d_norm(Xb, L_.inNorm, Xn);                 // Xn = rmsnorm(X)
            d_gemv(L_.q, Xn, Qb);                       // Q
            d_gemv(L_.k, Xn, K1);                       // K (new row)
            d_gemv(L_.v, Xn, V1);                       // V (new row)
            if (gDecAttnBias) {
                d_bias(Qb, L_.qb, qrow);
                d_bias(K1, L_.kb, w);
                d_bias(V1, L_.vb, w);
            }
            // stash pre-RoPE K, then RoPE Q and the new K at absolute position L
            dendEnc();
            blit = [gDCB blitCommandEncoder];
            [blit copyFromBuffer:K1 sourceOffset:0 toBuffer:Kraw[l] destinationOffset:0 size:(NSUInteger)w*2];
            [blit endEncoding];
            d_rope_at(Qb, nH, L);
            d_rope_at(K1, nKV, L);
            // append the new (post-RoPE) K and V at row L of the resident KV
            dendEnc();
            blit = [gDCB blitCommandEncoder];
            [blit copyFromBuffer:K1 sourceOffset:0 toBuffer:Kbuf[l] destinationOffset:(NSUInteger)L*w*2 size:(NSUInteger)w*2];
            [blit copyFromBuffer:V1 sourceOffset:0 toBuffer:Vbuf[l] destinationOffset:(NSUInteger)L*w*2 size:(NSUInteger)w*2];
            [blit endEncoding];
            d_attn(Qb, Kbuf[l], Vbuf[l], attn, ctx);   // single-query attention over ctx keys
            d_gemv(L_.o, attn, tmpH);                   // O
            d_add_buf(Xb, tmpH, H);                     // X += O
            d_norm(Xb, L_.postNorm, Xn2);              // Xn2 = rmsnorm(X)
            d_gemv(L_.gate, Xn2, Gb);                   // gate
            d_gemv(L_.up, Xn2, Ub);                     // up
            d_silu(Gb, Ub, Im);                          // G = silu(G)*U
            d_gemv(L_.down, Gb, tmpH);                  // down
            d_add_buf(Xb, tmpH, H);                     // X += down
        }
        dendEnc();
        CFTimeInterval tEnc = prof ? CFAbsoluteTimeGetCurrent() : 0;
        [gDCB commit];
        [gDCB waitUntilCompleted];
        if (prof) {
            CFTimeInterval tGpu = CFAbsoluteTimeGetCurrent();
            fprintf(stderr, "[decode-prof L=%d] host(alloc+kvup)=%.2f encode=%.2f gpu=%.2f total=%.2f ms\n",
                L, (tHost-t0)*1000.0, (tEnc-tHost)*1000.0, (tGpu-tEnc)*1000.0, (tGpu-t0)*1000.0);
        }

        mg_f16_to_f32((const __fp16 *)Xb.contents, lastPre, (long)H);
        for (int l = 0; l < gDecNL; l++) {
            mg_f16_to_f32((const __fp16 *)Kraw[l].contents, newKraw + (long)l*w, (long)w);
            mg_f16_to_f32((const __fp16 *)Kbuf[l].contents + (long)L*w, newKpost + (long)l*w, (long)w);
            mg_f16_to_f32((const __fp16 *)Vbuf[l].contents + (long)L*w, newV + (long)l*w, (long)w);
        }
        gDCB = nil;
        return 1;
    }
}
