//go:build darwin && arm64 && cgo

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
                      device const half*  WD   [[buffer(1)]],  // out*nblk f16 block scales (GGUF Q8_0 std)
                      device const half*  X    [[buffer(2)]],  // in f16 activation
                      device half*        Y    [[buffer(3)]],
                      constant int&       nblk [[buffer(4)]],
                      constant int&       out_ [[buffer(5)]],
                      device const half*  Bias [[buffer(6)]],  // [out] bias, or a placeholder when hasBias==0
                      constant int&    hasBias [[buffer(7)]],
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
    device const char* wrow = W  + (long)o * nblk * 32;
    device const half* wd   = WD + (long)o * nblk;
    float acc = 0.0f;
    for (int b = (int)lane; b < nblk; b += 32) {
        device const char4* wb = (device const char4*)(wrow + (long)b * 32);
        device const half4* xb = (device const half4*)(X    + (long)b * 32);
        float s = 0.0f;
        for (int i = 0; i < 8; i++) s += dot(float4(wb[i]), float4(xb[i]));
        acc += s * float(wd[b]);
    }
    acc = simd_sum(acc);
    if (lane == 0) Y[o] = half(acc + (hasBias ? float(Bias[o]) : 0.0f)); // fused projection bias
}

// d_rmsnorm: ONE threadgroup over the single token's H-vector. The TG threads cooperatively sum
// x^2 (strided), reduce in threadgroup memory, then write the normed row strided. The old version
// ran the whole H-wide norm on ONE thread (a 56-dispatch serial bottleneck in the decode forward);
// this parallelizes it across the threadgroup.
kernel void d_rmsnorm(device const half* X [[buffer(0)]],
                      device const half* W [[buffer(1)]],
                      device half* Out [[buffer(2)]],
                      constant uint& H [[buffer(3)]],
                      constant float& eps [[buffer(4)]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tgsize [[threads_per_threadgroup]],
                      threadgroup float* shared [[threadgroup(0)]]) {
    float ps = 0.0f;
    for (uint i=tid;i<H;i+=tgsize){ float v=float(X[i]); ps += v*v; }
    shared[tid] = ps;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=tgsize/2; s>0; s>>=1) {
        if (tid < s) shared[tid] += shared[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(shared[0]/float(H) + eps);
    for (uint i=tid;i<H;i+=tgsize){ Out[i] = half(float(X[i])*inv*float(W[i])); }
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
// Key-parallel (flash-decode): ONE threadgroup per head, SPLITS simdgroups (tgsize/32) split the
// keys. Each simdgroup runs the online softmax over its strided key subset {sg, sg+SPLITS, …}
// (32 lanes split hd), writes its partial (m, l, acc[hd]) to threadgroup memory, and simdgroup 0
// flash-combines the SPLITS partials. A single 32-lane-per-head loop over all ctx keys (the old
// shape) underused the GPU (~7% of cores) and was O(ctx) serial; splitting the keys hides the
// growing-context latency that was the last gap to llama.cpp-Metal.
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
                        threadgroup float* shm [[threadgroup(0)]], // SPLITS*(hd+2)
                        uint h [[threadgroup_position_in_grid]],
                        uint litg [[thread_index_in_threadgroup]],
                        uint tgsize [[threads_per_threadgroup]]) {
    if (h >= nH) return;
    uint SPLITS = tgsize / 32u;
    uint sg = litg / 32u, lane = litg % 32u;
    uint kvh = h / grp;
    device const half* q = Q + h*hd;
    float qreg[4]; uint nd = 0;
    for (uint d=lane; d<hd; d+=32u) qreg[nd++] = float(q[d]);
    float acc[4] = {0,0,0,0};
    float m = -INFINITY, l = 0.0f;
    for (uint j=sg; j<ctx; j+=SPLITS) {
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
    threadgroup float* my = shm + sg*(hd+2);
    { uint idx=0; for (uint d=lane; d<hd; d+=32u) my[d] = acc[idx++]; }
    if (lane == 0) { my[hd] = m; my[hd+1] = l; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float mc = -INFINITY;
        for (uint s=0; s<SPLITS; s++) mc = max(mc, shm[s*(hd+2)+hd]);
        float lc = 0.0f;
        for (uint s=0; s<SPLITS; s++) lc += shm[s*(hd+2)+hd+1] * exp(shm[s*(hd+2)+hd] - mc);
        float invl = (lc > 0.0f) ? 1.0f/lc : 0.0f;
        device half* o = Out + h*hd;
        for (uint d=lane; d<hd; d+=32u) {
            float a = 0.0f;
            for (uint s=0; s<SPLITS; s++) a += shm[s*(hd+2)+d] * exp(shm[s*(hd+2)+hd] - mc);
            o[d] = half(a * invl);
        }
    }
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
// Optional GPU LM head: final-norm vec id (gW), head Q8 wid (q8.m), vocab. -1/0 = not registered
// (the caller applies the head on the CPU). Registering them lets the resident forward also run the
// final RMSNorm + the vocab projection on the GPU and return logits directly — no CPU head, no
// post-forward round-trip.
static int gDecFinalNorm = -1, gDecHead = -1, gDecVocab = 0;

// gDecSF16[wid] = an f16 copy of weight wid's per-block scales (q8.m stores them f32). The decode
// GEMV reads f16 scales — the GGUF Q8_0 standard — so the weight stream is ~6% fewer bytes than the
// f32-scale read, the last bandwidth lever toward the llama.cpp-Metal bar. Built once per weight.
#define DEC_MAX_W 8192
static id<MTLBuffer> gDecSF16[DEC_MAX_W];
// Persistent per-layer GPU KV: kept resident ACROSS decode steps so the common (append) path neither
// re-uploads the context nor re-allocates — the new K/V row is written in place each step. gKVLen is
// the resident row count; the host seeds (passes context) when its tracker disagrees, else appends.
static id<MTLBuffer> gKVk[DEC_MAXL], gKVv[DEC_MAXL];
static int gKVCap = 0, gKVLen = 0; // rows
static void ensureScaleF16(int wid) {
    if (wid < 0 || wid >= DEC_MAX_W || gDecSF16[wid] != nil) return;
    int out, in, nblk; mg_q8_dims(wid, &out, &in, &nblk);
    long n = (long)out * nblk;
    if (n <= 0) return;
    id<MTLBuffer> f32 = mg_q8_scales_buf(wid);
    if (f32 == nil) return;
    id<MTLBuffer> b = [gDev newBufferWithLength:(NSUInteger)(n*2) options:MTLResourceStorageModeShared];
    const float *src = (const float *)f32.contents;
    __fp16 *dst = (__fp16 *)b.contents;
    for (long i = 0; i < n; i++) dst[i] = (__fp16)src[i];
    gDecSF16[wid] = b;
}

static id<MTLComputePipelineState> psoDGemv, psoDNorm, psoDBias, psoDRope, psoDAttn, psoDSilu, psoDAdd;
static int gDecReady;

static id<MTLComputePipelineState> make_dec_pipeline(id<MTLLibrary> lib, NSString *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (fn == nil) return nil;
    MTLComputePipelineDescriptor *desc = [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = fn;
    desc.supportIndirectCommandBuffers = YES;
    id<MTLComputePipelineState> pso = [gDev newComputePipelineStateWithDescriptor:desc options:0 reflection:nil error:&err];
    if (pso == nil) {
        pso = [gDev newComputePipelineStateWithFunction:fn error:&err];
    }
    return pso;
}

static int dec_init(void) {
    if (gDecReady) return 1;
    if (gDev == nil) return 0;
    NSError *err = nil;
    id<MTLLibrary> lib = [gDev newLibraryWithSource:kDecSrc options:nil error:&err];
    if (lib == nil) { NSLog(@"decode: library compile failed: %@", err); return 0; }
    psoDGemv = make_dec_pipeline(lib, @"q8dq_gemv");
    psoDNorm = make_dec_pipeline(lib, @"d_rmsnorm");
    psoDBias = make_dec_pipeline(lib, @"d_addbias");
    psoDRope = make_dec_pipeline(lib, @"d_rope");
    psoDAttn = make_dec_pipeline(lib, @"attn_decode");
    psoDSilu = make_dec_pipeline(lib, @"d_silumul");
    psoDAdd  = make_dec_pipeline(lib, @"d_add");
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
    ensureScaleF16(q); ensureScaleF16(k); ensureScaleF16(v); ensureScaleF16(o);
    ensureScaleF16(gate); ensureScaleF16(up); ensureScaleF16(down);
}

// mg_decode_head registers the final RMSNorm vector (gW id) + the Q8 LM-head weight (q8.m wid) +
// the vocab size, so mg_decode_step (when handed a logits buffer) runs the final norm + head on the
// GPU and returns logits directly.
void mg_decode_head(int finalNormID, int headWid, int vocab) {
    gDecFinalNorm = finalNormID; gDecHead = headWid; gDecVocab = vocab;
    ensureScaleF16(headWid);
}

// ---- ICB & persistent scratch state ----
#define MG_DECODE_MODE_MULTI_CB      0
#define MG_DECODE_MODE_DIRECT_ONE_CB  1
#define MG_DECODE_MODE_ICB           2

static int gDecDispatchMode = MG_DECODE_MODE_ICB;

typedef struct {
    int command_buffers;
    int encoders;
    int icb_dispatches;
    int contiguous_blocks;
    double host_encode_ms;
    double host_wait_ms;
    double gpu_ms;
    double total_ms;
    int icb_used;
    int mode;
} mg_decode_receipt;

typedef struct {
    int nblk;
    int out_;
    int hasBias;
    int _pad;
} DecGemvConst;

typedef struct {
    uint32_t H;
    float eps;
    uint32_t _pad[2];
} DecNormConst;

typedef struct {
    uint32_t nHeads;
    uint32_t hd;
    float theta;
    uint32_t _pad;
} DecRopeConst;

typedef struct {
    uint32_t nH;
    uint32_t hd;
    uint32_t w;
    uint32_t grp;
    float scale;
    uint32_t _pad[3];
} DecAttnConst;

typedef struct {
    int k_slot;
    int v_slot;
    int rope_k_slot;
} DecLayerDynSlots;

static DecLayerDynSlots gDecDynSlots[DEC_MAXL];

static id<MTLIndirectCommandBuffer> gDecICB = nil;
static int gDecICBCmdCount = 0;
static int gDecICBBuilt = 0;

static id<MTLBuffer> gDecXb = nil;
static id<MTLBuffer> gDecXn = nil, gDecXn2 = nil;
static id<MTLBuffer> gDecQb = nil;
static id<MTLBuffer> gDecAttn = nil, gDecTmpH = nil, gDecGb = nil, gDecUb = nil;
static id<MTLBuffer> gDecLogitBuf = nil;
static int gDecAllocH = 0, gDecAllocQrow = 0, gDecAllocIm = 0, gDecAllocVocab = 0;

static id<MTLBuffer> gDecStepBuf = nil;
static id<MTLBuffer> gDecConstBuf = nil;

#define DEC_MAX_RES 2048
static id<MTLResource> gDecResList[DEC_MAX_RES];
static int gDecResCount = 0;

static id<MTLBuffer> dbuf(long elems) { // f16 device buffer
    return [gDev newBufferWithLength:(NSUInteger)(elems * 2) options:MTLResourceStorageModeShared];
}
static id<MTLBuffer> wbufOfDec(int wid) { return (__bridge id<MTLBuffer>)gW[wid].buf; }

static void add_res(id<MTLResource> res) {
    if (res == nil || gDecResCount >= DEC_MAX_RES) return;
    for (int i = 0; i < gDecResCount; i++) {
        if (gDecResList[i] == res) return;
    }
    gDecResList[gDecResCount++] = res;
}

void mg_decode_reset(void) {
    gDecNL = gDecH = gDecHd = gDecNH = gDecNKV = gDecI = gDecAttnBias = 0;
    gDecEps = gDecTheta = gDecScale = 0.0f;
    for (int i = 0; i < DEC_MAXL; i++) { gDecL[i] = (DecLayer){0}; gKVk[i] = nil; gKVv[i] = nil; }
    gDecFinalNorm = -1; gDecHead = -1; gDecVocab = 0;
    gKVCap = 0; gKVLen = 0;
    for (int i = 0; i < DEC_MAX_W; i++) gDecSF16[i] = nil; // ARC frees the f16-scale buffers
    gDecICB = nil;
    gDecICBCmdCount = 0;
    gDecICBBuilt = 0;
    gDecXb = nil; gDecXn = nil; gDecXn2 = nil; gDecQb = nil;
    gDecAttn = nil; gDecTmpH = nil; gDecGb = nil; gDecUb = nil;
    gDecLogitBuf = nil;
    gDecAllocH = 0; gDecAllocQrow = 0; gDecAllocIm = 0; gDecAllocVocab = 0;
    gDecStepBuf = nil;
    gDecConstBuf = nil;
    gDecResCount = 0;
    for (int i = 0; i < DEC_MAX_RES; i++) gDecResList[i] = nil;
}

static void dec_ensure_scratch(int wantLogits) {
    int H = gDecH, hd = gDecHd, nH = gDecNH, Im = gDecI;
    int qrow = nH * hd;
    int vocab = (wantLogits && gDecVocab > 0) ? gDecVocab : 0;
    if (gDecXb == nil || gDecAllocH != H || gDecAllocQrow != qrow || gDecAllocIm != Im || gDecAllocVocab != vocab) {
        gDecXb = dbuf(H);
        gDecXn = dbuf(H);
        gDecXn2 = dbuf(H);
        gDecQb = dbuf(qrow);
        gDecAttn = dbuf(qrow);
        gDecTmpH = dbuf(H);
        gDecGb = dbuf(Im);
        gDecUb = dbuf(Im);
        if (vocab > 0) {
            gDecLogitBuf = dbuf(vocab);
        } else {
            gDecLogitBuf = nil;
        }
        gDecAllocH = H;
        gDecAllocQrow = qrow;
        gDecAllocIm = Im;
        gDecAllocVocab = vocab;
        gDecICBBuilt = 0;
    }
}

// ---- encode helpers (one command buffer / encoder per decode step) ----
static id<MTLCommandBuffer> gDCB;
static id<MTLComputeCommandEncoder> gDEnc;

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

static void d_gemv_e(id<MTLComputeCommandEncoder> e, int wid, id<MTLBuffer> X, id<MTLBuffer> Y, long yoff, int biasID) {
    int out, in, nblk; mg_q8_dims(wid, &out, &in, &nblk);
    [e setComputePipelineState:psoDGemv];
    [e setBuffer:mg_q8_codes_buf(wid) offset:0 atIndex:0];
    [e setBuffer:gDecSF16[wid]        offset:0 atIndex:1];
    [e setBuffer:X offset:0 atIndex:2];
    [e setBuffer:Y offset:(NSUInteger)(yoff*2) atIndex:3];
    [e setBytes:&nblk length:4 atIndex:4];
    [e setBytes:&out  length:4 atIndex:5];
    int hasBias = (biasID >= 0) ? 1 : 0;
    [e setBuffer:(hasBias ? wbufOfDec(biasID) : mg_q8_scales_buf(wid)) offset:0 atIndex:6];
    [e setBytes:&hasBias length:4 atIndex:7];
    NSUInteger ntg = (NSUInteger)((out + 7) / 8);
    [e dispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
static void d_gemv(int wid, id<MTLBuffer> X, id<MTLBuffer> Y, long yoff, int biasID) {
    d_gemv_e(denc(), wid, X, Y, yoff, biasID);
}

static void d_norm_e(id<MTLComputeCommandEncoder> e, id<MTLBuffer> X, int normID, id<MTLBuffer> Out) {
    [e setComputePipelineState:psoDNorm];
    [e setBuffer:X offset:0 atIndex:0];
    [e setBuffer:wbufOfDec(normID) offset:0 atIndex:1];
    [e setBuffer:Out offset:0 atIndex:2];
    uint H = gDecH; [e setBytes:&H length:4 atIndex:3];
    [e setBytes:&gDecEps length:4 atIndex:4];
    NSUInteger TG = psoDNorm.maxTotalThreadsPerThreadgroup; if (TG > 256) TG = 256;
    NSUInteger p = 1; while (p*2 <= TG) p *= 2; TG = p;
    [e setThreadgroupMemoryLength:(NSUInteger)(TG*4) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(TG,1,1)];
}
static void d_norm(id<MTLBuffer> X, int normID, id<MTLBuffer> Out) {
    d_norm_e(denc(), X, normID, Out);
}

static void d_rope_at_e(id<MTLComputeCommandEncoder> e, id<MTLBuffer> Buf, int nHeads, int base, long off) {
    [e setComputePipelineState:psoDRope];
    [e setBuffer:Buf offset:(NSUInteger)(off*2) atIndex:0];
    uint nh = nHeads, hd = gDecHd, b = base; [e setBytes:&nh length:4 atIndex:1];
    [e setBytes:&hd length:4 atIndex:2];
    [e setBytes:&b length:4 atIndex:3];
    [e setBytes:&gDecTheta length:4 atIndex:4];
    d1d(e, psoDRope, (NSUInteger)nHeads*(gDecHd/2));
}
static void d_rope_at(id<MTLBuffer> Buf, int nHeads, int base, long off) {
    d_rope_at_e(denc(), Buf, nHeads, base, off);
}

static void d_attn_e(id<MTLComputeCommandEncoder> e, id<MTLBuffer> Q, id<MTLBuffer> K, id<MTLBuffer> V, id<MTLBuffer> Out, int ctx) {
    [e setComputePipelineState:psoDAttn];
    [e setBuffer:Q offset:0 atIndex:0];
    [e setBuffer:K offset:0 atIndex:1];
    [e setBuffer:V offset:0 atIndex:2];
    [e setBuffer:Out offset:0 atIndex:3];
    uint c = ctx, nH = gDecNH, hd = gDecHd, w = gDecNKV*gDecHd, grp = (gDecNKV > 0) ? (gDecNH/gDecNKV) : 1;
    [e setBytes:&c length:4 atIndex:4];
    [e setBytes:&nH length:4 atIndex:5];
    [e setBytes:&hd length:4 atIndex:6];
    [e setBytes:&w length:4 atIndex:7];
    [e setBytes:&grp length:4 atIndex:8];
    [e setBytes:&gDecScale length:4 atIndex:9];
    NSUInteger SPLITS = 8;
    NSUInteger maxTG = psoDAttn.maxTotalThreadsPerThreadgroup / 32;
    if (SPLITS > maxTG) SPLITS = maxTG; if (SPLITS == 0) SPLITS = 1;
    [e setThreadgroupMemoryLength:(NSUInteger)(SPLITS * (gDecHd + 2) * 4) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)gDecNH,1,1) threadsPerThreadgroup:MTLSizeMake(SPLITS*32,1,1)];
}
static void d_attn(id<MTLBuffer> Q, id<MTLBuffer> K, id<MTLBuffer> V, id<MTLBuffer> Out, int ctx) {
    d_attn_e(denc(), Q, K, V, Out, ctx);
}

static void d_silu_e(id<MTLComputeCommandEncoder> e, id<MTLBuffer> G, id<MTLBuffer> U, int n) {
    [e setComputePipelineState:psoDSilu];
    [e setBuffer:G offset:0 atIndex:0];
    [e setBuffer:U offset:0 atIndex:1];
    d1d(e, psoDSilu, (NSUInteger)n);
}
static void d_silu(id<MTLBuffer> G, id<MTLBuffer> U, int n) {
    d_silu_e(denc(), G, U, n);
}

static void d_add_buf_e(id<MTLComputeCommandEncoder> e, id<MTLBuffer> X, id<MTLBuffer> Y, int n) {
    [e setComputePipelineState:psoDAdd];
    [e setBuffer:X offset:0 atIndex:0];
    [e setBuffer:Y offset:0 atIndex:1];
    d1d(e, psoDAdd, (NSUInteger)n);
}
static void d_add_buf(id<MTLBuffer> X, id<MTLBuffer> Y, int n) {
    d_add_buf_e(denc(), X, Y, n);
}

// kv_ensure grows the resident KV to hold at least `rows`, preserving the gKVLen rows already there.
static void kv_ensure(int rows) {
    if (gKVCap >= rows) return;
    int w = gDecNKV * gDecHd;
    int newCap = rows + 512; // headroom so a decode run grows O(log) not per-token
    for (int l = 0; l < gDecNL; l++) {
        id<MTLBuffer> nk = [gDev newBufferWithLength:(NSUInteger)((long)newCap*w*2) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nv = [gDev newBufferWithLength:(NSUInteger)((long)newCap*w*2) options:MTLResourceStorageModeShared];
        if (gKVLen > 0 && gKVk[l] != nil) {
            memcpy(nk.contents, gKVk[l].contents, (size_t)((long)gKVLen*w*2));
            memcpy(nv.contents, gKVv[l].contents, (size_t)((long)gKVLen*w*2));
        }
        gKVk[l] = nk; gKVv[l] = nv; // ARC frees the old buffers
    }
    gKVCap = newCap;
    gDecICBBuilt = 0; // buffer pointers changed, rebuild ICB
}

static int dec_build_icb(int wantLogits) {
    if (gDecICBBuilt && gDecICB != nil) {
        return 1;
    }
    if (gDecNL <= 0 || gDecH <= 0) return 0;
    
    dec_ensure_scratch(wantLogits);
    
    if (gDecStepBuf == nil) {
        gDecStepBuf = [gDev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    }
    if (gDecConstBuf == nil) {
        gDecConstBuf = [gDev newBufferWithLength:65536 options:MTLResourceStorageModeShared];
    }
    
    int hasHead = (gDecFinalNorm >= 0 && gDecHead >= 0 && gDecVocab > 0);
    int totalDispatches = gDecNL * 15 + (hasHead ? 2 : 0);
    
    MTLIndirectCommandBufferDescriptor *icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
    icbDesc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
    icbDesc.inheritBuffers = NO;
    icbDesc.inheritPipelineState = NO;
    icbDesc.maxKernelBufferBindCount = 16;
    
    gDecICB = [gDev newIndirectCommandBufferWithDescriptor:icbDesc
                                           maxCommandCount:(NSUInteger)totalDispatches
                                                   options:MTLResourceStorageModeShared];
    if (gDecICB == nil) {
        return 0;
    }
    
    gDecResCount = 0;
    add_res(gDecXb);
    add_res(gDecXn);
    add_res(gDecXn2);
    add_res(gDecQb);
    add_res(gDecAttn);
    add_res(gDecTmpH);
    add_res(gDecGb);
    add_res(gDecUb);
    if (gDecLogitBuf != nil) add_res(gDecLogitBuf);
    add_res(gDecStepBuf);
    add_res(gDecConstBuf);
    
    uint8_t *constPtr = (uint8_t *)gDecConstBuf.contents;
    __block NSUInteger constOff = 0;
    
    #define ALLOC_CONST(type, val) ({ \
        NSUInteger cur = constOff; \
        *((type *)(constPtr + cur)) = (val); \
        constOff += sizeof(type); \
        if (constOff % 16 != 0) constOff += 16 - (constOff % 16); \
        cur; \
    })
    
    int H = gDecH, hd = gDecHd, nH = gDecNH, nKV = gDecNKV, Im = gDecI;
    int w = nKV * hd;
    int grp = (nKV > 0) ? (nH / nKV) : 1;
    
    NSUInteger normTG = psoDNorm.maxTotalThreadsPerThreadgroup;
    if (normTG > 256) normTG = 256;
    NSUInteger p = 1; while (p*2 <= normTG) p *= 2; normTG = p;
    
    NSUInteger spl = 8;
    NSUInteger maxAttnTG = psoDAttn.maxTotalThreadsPerThreadgroup / 32;
    if (spl > maxAttnTG) spl = maxAttnTG; if (spl == 0) spl = 1;
    
    int slot = 0;
    for (int l = 0; l < gDecNL; l++) {
        DecLayer L_ = gDecL[l];
        add_res(gKVk[l]);
        add_res(gKVv[l]);
        
        int qb = gDecAttnBias ? L_.qb : -1;
        int kb = gDecAttnBias ? L_.kb : -1;
        int vb = gDecAttnBias ? L_.vb : -1;
        
        // 0. InNorm
        {
            DecNormConst nc = {(uint32_t)H, gDecEps, {0, 0}};
            NSUInteger off = ALLOC_CONST(DecNormConst, nc);
            add_res(wbufOfDec(L_.inNorm));
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDNorm];
            [cmd setKernelBuffer:gDecXb offset:0 atIndex:0];
            [cmd setKernelBuffer:wbufOfDec(L_.inNorm) offset:0 atIndex:1];
            [cmd setKernelBuffer:gDecXn offset:0 atIndex:2];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:3];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:4];
            [cmd setThreadgroupMemoryLength:(NSUInteger)(normTG * 4) atIndex:0];
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(normTG,1,1)];
            slot++;
        }
        
        // 1. Q GEMV
        {
            int out, in, nblk; mg_q8_dims(L_.q, &out, &in, &nblk);
            int hasBias = (qb >= 0) ? 1 : 0;
            DecGemvConst gc = {nblk, out, hasBias, 0};
            NSUInteger off = ALLOC_CONST(DecGemvConst, gc);
            add_res(mg_q8_codes_buf(L_.q));
            add_res(gDecSF16[L_.q]);
            id<MTLBuffer> bBuf = hasBias ? wbufOfDec(qb) : mg_q8_scales_buf(L_.q);
            add_res(bBuf);
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDGemv];
            [cmd setKernelBuffer:mg_q8_codes_buf(L_.q) offset:0 atIndex:0];
            [cmd setKernelBuffer:gDecSF16[L_.q] offset:0 atIndex:1];
            [cmd setKernelBuffer:gDecXn offset:0 atIndex:2];
            [cmd setKernelBuffer:gDecQb offset:0 atIndex:3];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:4];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:5];
            [cmd setKernelBuffer:bBuf offset:0 atIndex:6];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 8 atIndex:7];
            NSUInteger ntg = (NSUInteger)((out + 7) / 8);
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            slot++;
        }
        
        // 2. K GEMV
        {
            gDecDynSlots[l].k_slot = slot;
            int out, in, nblk; mg_q8_dims(L_.k, &out, &in, &nblk);
            int hasBias = (kb >= 0) ? 1 : 0;
            DecGemvConst gc = {nblk, out, hasBias, 0};
            NSUInteger off = ALLOC_CONST(DecGemvConst, gc);
            add_res(mg_q8_codes_buf(L_.k));
            add_res(gDecSF16[L_.k]);
            id<MTLBuffer> bBuf = hasBias ? wbufOfDec(kb) : mg_q8_scales_buf(L_.k);
            add_res(bBuf);
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDGemv];
            [cmd setKernelBuffer:mg_q8_codes_buf(L_.k) offset:0 atIndex:0];
            [cmd setKernelBuffer:gDecSF16[L_.k] offset:0 atIndex:1];
            [cmd setKernelBuffer:gDecXn offset:0 atIndex:2];
            [cmd setKernelBuffer:gKVk[l] offset:0 atIndex:3];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:4];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:5];
            [cmd setKernelBuffer:bBuf offset:0 atIndex:6];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 8 atIndex:7];
            NSUInteger ntg = (NSUInteger)((out + 7) / 8);
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            slot++;
        }
        
        // 3. V GEMV
        {
            gDecDynSlots[l].v_slot = slot;
            int out, in, nblk; mg_q8_dims(L_.v, &out, &in, &nblk);
            int hasBias = (vb >= 0) ? 1 : 0;
            DecGemvConst gc = {nblk, out, hasBias, 0};
            NSUInteger off = ALLOC_CONST(DecGemvConst, gc);
            add_res(mg_q8_codes_buf(L_.v));
            add_res(gDecSF16[L_.v]);
            id<MTLBuffer> bBuf = hasBias ? wbufOfDec(vb) : mg_q8_scales_buf(L_.v);
            add_res(bBuf);
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDGemv];
            [cmd setKernelBuffer:mg_q8_codes_buf(L_.v) offset:0 atIndex:0];
            [cmd setKernelBuffer:gDecSF16[L_.v] offset:0 atIndex:1];
            [cmd setKernelBuffer:gDecXn offset:0 atIndex:2];
            [cmd setKernelBuffer:gKVv[l] offset:0 atIndex:3];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:4];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:5];
            [cmd setKernelBuffer:bBuf offset:0 atIndex:6];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 8 atIndex:7];
            NSUInteger ntg = (NSUInteger)((out + 7) / 8);
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            slot++;
        }
        
        // 4. RoPE Q
        {
            DecRopeConst rc = {(uint32_t)nH, (uint32_t)hd, gDecTheta, 0};
            NSUInteger off = ALLOC_CONST(DecRopeConst, rc);
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDRope];
            [cmd setKernelBuffer:gDecQb offset:0 atIndex:0];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:1];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:2];
            [cmd setKernelBuffer:gDecStepBuf offset:0 atIndex:3];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 8 atIndex:4];
            NSUInteger n = (NSUInteger)nH * (hd / 2);
            NSUInteger tg = psoDRope.maxTotalThreadsPerThreadgroup;
            if (tg > n) tg = n; if (tg == 0) tg = 1;
            [cmd concurrentDispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
            slot++;
        }
        
        // 5. RoPE K
        {
            gDecDynSlots[l].rope_k_slot = slot;
            DecRopeConst rc = {(uint32_t)nKV, (uint32_t)hd, gDecTheta, 0};
            NSUInteger off = ALLOC_CONST(DecRopeConst, rc);
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDRope];
            [cmd setKernelBuffer:gKVk[l] offset:0 atIndex:0];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:1];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:2];
            [cmd setKernelBuffer:gDecStepBuf offset:0 atIndex:3];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 8 atIndex:4];
            NSUInteger n = (NSUInteger)nKV * (hd / 2);
            NSUInteger tg = psoDRope.maxTotalThreadsPerThreadgroup;
            if (tg > n) tg = n; if (tg == 0) tg = 1;
            [cmd concurrentDispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
            slot++;
        }
        
        // 6. Attention
        {
            DecAttnConst ac = {(uint32_t)nH, (uint32_t)hd, (uint32_t)w, (uint32_t)grp, gDecScale, {0,0,0}};
            NSUInteger off = ALLOC_CONST(DecAttnConst, ac);
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDAttn];
            [cmd setKernelBuffer:gDecQb offset:0 atIndex:0];
            [cmd setKernelBuffer:gKVk[l] offset:0 atIndex:1];
            [cmd setKernelBuffer:gKVv[l] offset:0 atIndex:2];
            [cmd setKernelBuffer:gDecAttn offset:0 atIndex:3];
            [cmd setKernelBuffer:gDecStepBuf offset:4 atIndex:4];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:5];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:6];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 8 atIndex:7];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 12 atIndex:8];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 16 atIndex:9];
            [cmd setThreadgroupMemoryLength:(NSUInteger)(spl * (hd + 2) * 4) atIndex:0];
            [cmd concurrentDispatchThreadgroups:MTLSizeMake((NSUInteger)nH,1,1) threadsPerThreadgroup:MTLSizeMake(spl*32,1,1)];
            slot++;
        }
        
        // 7. O GEMV
        {
            int out, in, nblk; mg_q8_dims(L_.o, &out, &in, &nblk);
            DecGemvConst gc = {nblk, out, 0, 0};
            NSUInteger off = ALLOC_CONST(DecGemvConst, gc);
            add_res(mg_q8_codes_buf(L_.o));
            add_res(gDecSF16[L_.o]);
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDGemv];
            [cmd setKernelBuffer:mg_q8_codes_buf(L_.o) offset:0 atIndex:0];
            [cmd setKernelBuffer:gDecSF16[L_.o] offset:0 atIndex:1];
            [cmd setKernelBuffer:gDecAttn offset:0 atIndex:2];
            [cmd setKernelBuffer:gDecTmpH offset:0 atIndex:3];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:4];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:5];
            [cmd setKernelBuffer:mg_q8_scales_buf(L_.o) offset:0 atIndex:6];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 8 atIndex:7];
            NSUInteger ntg = (NSUInteger)((out + 7) / 8);
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            slot++;
        }
        
        // 8. Residual Add 1
        {
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDAdd];
            [cmd setKernelBuffer:gDecXb offset:0 atIndex:0];
            [cmd setKernelBuffer:gDecTmpH offset:0 atIndex:1];
            NSUInteger n = (NSUInteger)H;
            NSUInteger tg = psoDAdd.maxTotalThreadsPerThreadgroup;
            if (tg > n) tg = n; if (tg == 0) tg = 1;
            [cmd concurrentDispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
            slot++;
        }
        
        // 9. Post RMSNorm
        {
            DecNormConst nc = {(uint32_t)H, gDecEps, {0, 0}};
            NSUInteger off = ALLOC_CONST(DecNormConst, nc);
            add_res(wbufOfDec(L_.postNorm));
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDNorm];
            [cmd setKernelBuffer:gDecXb offset:0 atIndex:0];
            [cmd setKernelBuffer:wbufOfDec(L_.postNorm) offset:0 atIndex:1];
            [cmd setKernelBuffer:gDecXn2 offset:0 atIndex:2];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:3];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:4];
            [cmd setThreadgroupMemoryLength:(NSUInteger)(normTG * 4) atIndex:0];
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(normTG,1,1)];
            slot++;
        }
        
        // 10. Gate GEMV
        {
            int out, in, nblk; mg_q8_dims(L_.gate, &out, &in, &nblk);
            DecGemvConst gc = {nblk, out, 0, 0};
            NSUInteger off = ALLOC_CONST(DecGemvConst, gc);
            add_res(mg_q8_codes_buf(L_.gate));
            add_res(gDecSF16[L_.gate]);
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDGemv];
            [cmd setKernelBuffer:mg_q8_codes_buf(L_.gate) offset:0 atIndex:0];
            [cmd setKernelBuffer:gDecSF16[L_.gate] offset:0 atIndex:1];
            [cmd setKernelBuffer:gDecXn2 offset:0 atIndex:2];
            [cmd setKernelBuffer:gDecGb offset:0 atIndex:3];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:4];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:5];
            [cmd setKernelBuffer:mg_q8_scales_buf(L_.gate) offset:0 atIndex:6];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 8 atIndex:7];
            NSUInteger ntg = (NSUInteger)((out + 7) / 8);
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            slot++;
        }
        
        // 11. Up GEMV
        {
            int out, in, nblk; mg_q8_dims(L_.up, &out, &in, &nblk);
            DecGemvConst gc = {nblk, out, 0, 0};
            NSUInteger off = ALLOC_CONST(DecGemvConst, gc);
            add_res(mg_q8_codes_buf(L_.up));
            add_res(gDecSF16[L_.up]);
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDGemv];
            [cmd setKernelBuffer:mg_q8_codes_buf(L_.up) offset:0 atIndex:0];
            [cmd setKernelBuffer:gDecSF16[L_.up] offset:0 atIndex:1];
            [cmd setKernelBuffer:gDecXn2 offset:0 atIndex:2];
            [cmd setKernelBuffer:gDecUb offset:0 atIndex:3];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:4];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:5];
            [cmd setKernelBuffer:mg_q8_scales_buf(L_.up) offset:0 atIndex:6];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 8 atIndex:7];
            NSUInteger ntg = (NSUInteger)((out + 7) / 8);
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            slot++;
        }
        
        // 12. SwiGLU
        {
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDSilu];
            [cmd setKernelBuffer:gDecGb offset:0 atIndex:0];
            [cmd setKernelBuffer:gDecUb offset:0 atIndex:1];
            NSUInteger n = (NSUInteger)Im;
            NSUInteger tg = psoDSilu.maxTotalThreadsPerThreadgroup;
            if (tg > n) tg = n; if (tg == 0) tg = 1;
            [cmd concurrentDispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
            slot++;
        }
        
        // 13. Down GEMV
        {
            int out, in, nblk; mg_q8_dims(L_.down, &out, &in, &nblk);
            DecGemvConst gc = {nblk, out, 0, 0};
            NSUInteger off = ALLOC_CONST(DecGemvConst, gc);
            add_res(mg_q8_codes_buf(L_.down));
            add_res(gDecSF16[L_.down]);
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDGemv];
            [cmd setKernelBuffer:mg_q8_codes_buf(L_.down) offset:0 atIndex:0];
            [cmd setKernelBuffer:gDecSF16[L_.down] offset:0 atIndex:1];
            [cmd setKernelBuffer:gDecGb offset:0 atIndex:2];
            [cmd setKernelBuffer:gDecTmpH offset:0 atIndex:3];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:4];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:5];
            [cmd setKernelBuffer:mg_q8_scales_buf(L_.down) offset:0 atIndex:6];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 8 atIndex:7];
            NSUInteger ntg = (NSUInteger)((out + 7) / 8);
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            slot++;
        }
        
        // 14. Residual Add 2
        {
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDAdd];
            [cmd setKernelBuffer:gDecXb offset:0 atIndex:0];
            [cmd setKernelBuffer:gDecTmpH offset:0 atIndex:1];
            NSUInteger n = (NSUInteger)H;
            NSUInteger tg = psoDAdd.maxTotalThreadsPerThreadgroup;
            if (tg > n) tg = n; if (tg == 0) tg = 1;
            [cmd concurrentDispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
            slot++;
        }
    }
    
    // Head operations
    if (hasHead) {
        // Final Norm
        {
            DecNormConst nc = {(uint32_t)H, gDecEps, {0, 0}};
            NSUInteger off = ALLOC_CONST(DecNormConst, nc);
            add_res(wbufOfDec(gDecFinalNorm));
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDNorm];
            [cmd setKernelBuffer:gDecXb offset:0 atIndex:0];
            [cmd setKernelBuffer:wbufOfDec(gDecFinalNorm) offset:0 atIndex:1];
            [cmd setKernelBuffer:gDecXn offset:0 atIndex:2];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:3];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:4];
            [cmd setThreadgroupMemoryLength:(NSUInteger)(normTG * 4) atIndex:0];
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(normTG,1,1)];
            slot++;
        }
        
        // Head GEMV
        {
            int out, in, nblk; mg_q8_dims(gDecHead, &out, &in, &nblk);
            DecGemvConst gc = {nblk, out, 0, 0};
            NSUInteger off = ALLOC_CONST(DecGemvConst, gc);
            add_res(mg_q8_codes_buf(gDecHead));
            add_res(gDecSF16[gDecHead]);
            
            id<MTLIndirectComputeCommand> cmd = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)slot];
            [cmd setBarrier];
            [cmd setComputePipelineState:psoDGemv];
            [cmd setKernelBuffer:mg_q8_codes_buf(gDecHead) offset:0 atIndex:0];
            [cmd setKernelBuffer:gDecSF16[gDecHead] offset:0 atIndex:1];
            [cmd setKernelBuffer:gDecXn offset:0 atIndex:2];
            [cmd setKernelBuffer:gDecLogitBuf offset:0 atIndex:3];
            [cmd setKernelBuffer:gDecConstBuf offset:off atIndex:4];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 4 atIndex:5];
            [cmd setKernelBuffer:mg_q8_scales_buf(gDecHead) offset:0 atIndex:6];
            [cmd setKernelBuffer:gDecConstBuf offset:off + 8 atIndex:7];
            NSUInteger ntg = (NSUInteger)((out + 7) / 8);
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            slot++;
        }
    }
    
    #undef ALLOC_CONST
    
    gDecICBCmdCount = slot;
    gDecICBBuilt = 1;
    return 1;
}

void mg_decode_set_mode(int mode) {
    gDecDispatchMode = mode;
}

int mg_decode_get_mode(void) {
    return gDecDispatchMode;
}

int mg_decode_icb_supported(void) {
    if (!dec_init()) return 0;
    if (gDev == nil) return 0;
    MTLIndirectCommandBufferDescriptor *desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
    desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
    desc.inheritBuffers = NO;
    desc.inheritPipelineState = NO;
    desc.maxKernelBufferBindCount = 16;
    id<MTLIndirectCommandBuffer> icb = [gDev newIndirectCommandBufferWithDescriptor:desc maxCommandCount:1 options:MTLResourceStorageModeShared];
    return (icb != nil) ? 1 : 0;
}

// mg_decode_step_receipt runs one decode token and returns execution receipts for performance analysis.
int mg_decode_step_receipt(const float *xEmbed, const float *Kctx, const float *Vctx, int L,
                           float *lastPre, float *newKraw, float *newKpost, float *newV, float *logits,
                           int seedFlag, int dispatchMode, mg_decode_receipt *receipt) {
    if (!dec_init()) return 0;
    int prof = getenv("FAK_DECODE_PROF") != NULL;
    CFTimeInterval t0 = prof ? CFAbsoluteTimeGetCurrent() : 0;
    @autoreleasepool {
        int H = gDecH, hd = gDecHd, nH = gDecNH, nKV = gDecNKV, Im = gDecI, w = nKV*hd, qrow = nH*hd;
        int ctx = L + 1;
        long rowOff = (long)L * w;
        int wantLogits = (logits != NULL && gDecHead >= 0 && gDecVocab > 0);

        kv_ensure(ctx);
        if (seedFlag) {
            if (L > 0 && Kctx != NULL) {
                for (int l = 0; l < gDecNL; l++) {
                    mg_f32_to_f16(Kctx + (long)l*L*w, (__fp16 *)gKVk[l].contents, (long)L*w);
                    mg_f32_to_f16(Vctx + (long)l*L*w, (__fp16 *)gKVv[l].contents, (long)L*w);
                }
            }
            gKVLen = L;
        } else if (gKVLen != L) {
            return 0;
        }

        dec_ensure_scratch(wantLogits);
        id<MTLBuffer> Xb = gDecXb;
        id<MTLBuffer> Xn = gDecXn, Xn2 = gDecXn2;
        id<MTLBuffer> Qb = gDecQb;
        id<MTLBuffer> attn = gDecAttn, tmpH = gDecTmpH, Gb = gDecGb, Ub = gDecUb;
        id<MTLBuffer> logitBuf = wantLogits ? gDecLogitBuf : nil;

        mg_f32_to_f16(xEmbed, (__fp16 *)Xb.contents, (long)H);

        int mode = (dispatchMode >= 0) ? dispatchMode : gDecDispatchMode;
        int matOnly = getenv("FAK_DECODE_MATMUL_ONLY") != NULL;
        int noAttn  = getenv("FAK_DECODE_NO_ATTN") != NULL;

        // Fall back to direct mode if ICB is requested but unsupported or matOnly/noAttn diagnostics active
        if (mode == MG_DECODE_MODE_ICB && (matOnly || noAttn || !dec_build_icb(wantLogits))) {
            mode = MG_DECODE_MODE_DIRECT_ONE_CB;
        }

        CFTimeInterval tEncodeStart = CFAbsoluteTimeGetCurrent();

        if (mode == MG_DECODE_MODE_ICB) {
            // Update dynamic parameters in gDecStepBuf
            uint32_t *dyn = (uint32_t *)gDecStepBuf.contents;
            dyn[0] = (uint32_t)L;
            dyn[1] = (uint32_t)(L + 1);

            // Update per-layer dynamic offsets in ICB
            NSUInteger byteOff = (NSUInteger)(rowOff * 2);
            for (int l = 0; l < gDecNL; l++) {
                id<MTLIndirectComputeCommand> cmdK = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)gDecDynSlots[l].k_slot];
                [cmdK setKernelBuffer:gKVk[l] offset:byteOff atIndex:3];

                id<MTLIndirectComputeCommand> cmdV = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)gDecDynSlots[l].v_slot];
                [cmdV setKernelBuffer:gKVv[l] offset:byteOff atIndex:3];

                id<MTLIndirectComputeCommand> cmdRopeK = [gDecICB indirectComputeCommandAtIndex:(NSUInteger)gDecDynSlots[l].rope_k_slot];
                [cmdRopeK setKernelBuffer:gKVk[l] offset:byteOff atIndex:0];
            }

            CFTimeInterval tEncodeEnd = CFAbsoluteTimeGetCurrent();

            id<MTLCommandBuffer> cb = [gQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc useResources:gDecResList count:(NSUInteger)gDecResCount usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            int activeCmds = gDecNL * 15 + ((wantLogits && gDecHead >= 0 && gDecVocab > 0) ? 2 : 0);
            [enc executeCommandsInBuffer:gDecICB withRange:NSMakeRange(0, (NSUInteger)activeCmds)];
            [enc endEncoding];

            CFTimeInterval tWaitStart = CFAbsoluteTimeGetCurrent();
            [cb commit];
            [cb waitUntilCompleted];
            CFTimeInterval tWaitEnd = CFAbsoluteTimeGetCurrent();

            double gpuMs = 0;
            if (cb.status == MTLCommandBufferStatusCompleted) {
                gpuMs = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            }

            if (receipt != NULL) {
                receipt->command_buffers = 1;
                receipt->encoders = 1;
                receipt->icb_dispatches = activeCmds;
                receipt->contiguous_blocks = 1;
                receipt->host_encode_ms = (tEncodeEnd - tEncodeStart) * 1000.0;
                receipt->host_wait_ms = (tWaitEnd - tWaitStart) * 1000.0;
                receipt->gpu_ms = gpuMs;
                receipt->total_ms = (tWaitEnd - tEncodeStart) * 1000.0;
                receipt->icb_used = 1;
                receipt->mode = MG_DECODE_MODE_ICB;
            }
        } else if (mode == MG_DECODE_MODE_MULTI_CB) {
            double totalWaitMs = 0;
            double totalGpuMs = 0;
            int totalCBs = 0;

            for (int l = 0; l < gDecNL; l++) {
                DecLayer L_ = gDecL[l];
                int qb = gDecAttnBias ? L_.qb : -1, kb = gDecAttnBias ? L_.kb : -1, vb = gDecAttnBias ? L_.vb : -1;

                // CB 1: Attention block
                {
                    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                    if (!matOnly) d_norm_e(e, Xb, L_.inNorm, Xn);
                    d_gemv_e(e, L_.q, Xn, Qb, 0, qb);
                    d_gemv_e(e, L_.k, Xn, gKVk[l], rowOff, kb);
                    d_gemv_e(e, L_.v, Xn, gKVv[l], rowOff, vb);
                    if (!matOnly) {
                        d_rope_at_e(e, Qb, nH, L, 0);
                        d_rope_at_e(e, gKVk[l], nKV, L, rowOff);
                        if (!noAttn) d_attn_e(e, Qb, gKVk[l], gKVv[l], attn, ctx);
                    }
                    d_gemv_e(e, L_.o, attn, tmpH, 0, -1);
                    if (!matOnly) d_add_buf_e(e, Xb, tmpH, H);
                    [e endEncoding];
                    CFTimeInterval w0 = CFAbsoluteTimeGetCurrent();
                    [cb commit];
                    [cb waitUntilCompleted];
                    CFTimeInterval w1 = CFAbsoluteTimeGetCurrent();
                    totalWaitMs += (w1 - w0) * 1000.0;
                    if (cb.status == MTLCommandBufferStatusCompleted) totalGpuMs += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                    totalCBs++;
                }

                // CB 2: MLP block
                {
                    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                    if (!matOnly) d_norm_e(e, Xb, L_.postNorm, Xn2);
                    d_gemv_e(e, L_.gate, Xn2, Gb, 0, -1);
                    d_gemv_e(e, L_.up, Xn2, Ub, 0, -1);
                    if (!matOnly) d_silu_e(e, Gb, Ub, Im);
                    d_gemv_e(e, L_.down, Gb, tmpH, 0, -1);
                    if (!matOnly) d_add_buf_e(e, Xb, tmpH, H);
                    [e endEncoding];
                    CFTimeInterval w0 = CFAbsoluteTimeGetCurrent();
                    [cb commit];
                    [cb waitUntilCompleted];
                    CFTimeInterval w1 = CFAbsoluteTimeGetCurrent();
                    totalWaitMs += (w1 - w0) * 1000.0;
                    if (cb.status == MTLCommandBufferStatusCompleted) totalGpuMs += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                    totalCBs++;
                }
            }
            if (wantLogits) {
                id<MTLCommandBuffer> cb = [gQueue commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                d_norm_e(e, Xb, gDecFinalNorm, Xn);
                d_gemv_e(e, gDecHead, Xn, logitBuf, 0, -1);
                [e endEncoding];
                CFTimeInterval w0 = CFAbsoluteTimeGetCurrent();
                [cb commit];
                [cb waitUntilCompleted];
                CFTimeInterval w1 = CFAbsoluteTimeGetCurrent();
                totalWaitMs += (w1 - w0) * 1000.0;
                if (cb.status == MTLCommandBufferStatusCompleted) totalGpuMs += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                totalCBs++;
            }
            CFTimeInterval tEnd = CFAbsoluteTimeGetCurrent();

            if (receipt != NULL) {
                receipt->command_buffers = totalCBs;
                receipt->encoders = totalCBs;
                receipt->icb_dispatches = 0;
                receipt->contiguous_blocks = 0;
                receipt->host_encode_ms = (tEnd - tEncodeStart) * 1000.0 - totalWaitMs;
                receipt->host_wait_ms = totalWaitMs;
                receipt->gpu_ms = totalGpuMs;
                receipt->total_ms = (tEnd - tEncodeStart) * 1000.0;
                receipt->icb_used = 0;
                receipt->mode = MG_DECODE_MODE_MULTI_CB;
            }
        } else {
            // Direct one command buffer mode
            gDCB = [gQueue commandBuffer];
            gDEnc = nil;

            for (int l = 0; l < gDecNL; l++) {
                DecLayer L_ = gDecL[l];
                int qb = gDecAttnBias ? L_.qb : -1, kb = gDecAttnBias ? L_.kb : -1, vb = gDecAttnBias ? L_.vb : -1;
                if (!matOnly) d_norm(Xb, L_.inNorm, Xn);
                d_gemv(L_.q, Xn, Qb, 0, qb);
                d_gemv(L_.k, Xn, gKVk[l], rowOff, kb);
                d_gemv(L_.v, Xn, gKVv[l], rowOff, vb);
                if (!matOnly) {
                    d_rope_at(Qb, nH, L, 0);
                    d_rope_at(gKVk[l], nKV, L, rowOff);
                    if (!noAttn) d_attn(Qb, gKVk[l], gKVv[l], attn, ctx);
                }
                d_gemv(L_.o, attn, tmpH, 0, -1);
                if (!matOnly) d_add_buf(Xb, tmpH, H);
                if (!matOnly) d_norm(Xb, L_.postNorm, Xn2);
                d_gemv(L_.gate, Xn2, Gb, 0, -1);
                d_gemv(L_.up, Xn2, Ub, 0, -1);
                if (!matOnly) d_silu(Gb, Ub, Im);
                d_gemv(L_.down, Gb, tmpH, 0, -1);
                if (!matOnly) d_add_buf(Xb, tmpH, H);
            }
            if (wantLogits) {
                d_norm(Xb, gDecFinalNorm, Xn);
                d_gemv(gDecHead, Xn, logitBuf, 0, -1);
            }
            dendEnc();
            CFTimeInterval tEncodeEnd = CFAbsoluteTimeGetCurrent();

            CFTimeInterval tWaitStart = CFAbsoluteTimeGetCurrent();
            [gDCB commit];
            [gDCB waitUntilCompleted];
            CFTimeInterval tWaitEnd = CFAbsoluteTimeGetCurrent();

            double gpuMs = 0;
            if (gDCB.status == MTLCommandBufferStatusCompleted) {
                gpuMs = (gDCB.GPUEndTime - gDCB.GPUStartTime) * 1000.0;
            }

            if (receipt != NULL) {
                receipt->command_buffers = 1;
                receipt->encoders = 1;
                receipt->icb_dispatches = 0;
                receipt->contiguous_blocks = 0;
                receipt->host_encode_ms = (tEncodeEnd - tEncodeStart) * 1000.0;
                receipt->host_wait_ms = (tWaitEnd - tWaitStart) * 1000.0;
                receipt->gpu_ms = gpuMs;
                receipt->total_ms = (tWaitEnd - tEncodeStart) * 1000.0;
                receipt->icb_used = 0;
                receipt->mode = MG_DECODE_MODE_DIRECT_ONE_CB;
            }
            gDCB = nil;
        }

        if (prof) {
            CFTimeInterval tNow = CFAbsoluteTimeGetCurrent();
            fprintf(stderr, "[decode-prof L=%d mode=%d] total=%.2f ms\n", L, mode, (tNow - t0) * 1000.0);
        }

        mg_f16_to_f32((const __fp16 *)Xb.contents, lastPre, (long)H);
        if (wantLogits) mg_f16_to_f32((const __fp16 *)logitBuf.contents, logits, (long)gDecVocab);
        (void)newKraw;
        for (int l = 0; l < gDecNL; l++) {
            mg_f16_to_f32((const __fp16 *)gKVk[l].contents + (long)L*w, newKpost + (long)l*w, (long)w);
            mg_f16_to_f32((const __fp16 *)gKVv[l].contents + (long)L*w, newV + (long)l*w, (long)w);
        }
        gKVLen = L + 1;
        return 1;
    }
}

int mg_decode_step(const float *xEmbed, const float *Kctx, const float *Vctx, int L,
                   float *lastPre, float *newKraw, float *newKpost, float *newV, float *logits, int seedFlag) {
    return mg_decode_step_receipt(xEmbed, Kctx, Vctx, L, lastPre, newKraw, newKpost, newV, logits, seedFlag, -1, NULL);
}
