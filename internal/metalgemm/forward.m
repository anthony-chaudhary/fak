//go:build darwin && arm64 && cgo

// forward.m — the GPU-resident prefill forward. Where metal.m moves only the projection
// matmuls to the GPU (and pays an f16 round-trip per matmul, leaving RMSNorm/RoPE/attention/
// SwiGLU on the CPU), this runs the WHOLE prefill on the device: all activations stay f16 in
// unified memory across every layer, the seven projections are MPS matmuls, and RMSNorm, RoPE,
// the causal GQA attention, SwiGLU and the residual adds are custom MSL kernels — all encoded
// into ONE command buffer with a single CPU/GPU sync. That is what closes the gap to
// llama.cpp-Metal: the CPU attention/elementwise time and the per-matmul round-trips both
// vanish. Numerics are validated end-to-end against the CPU Q8 path (modelbench -metal -verify).
//
// Assumes a fresh prefill (base == 0): the only attention context is the P tokens computed
// here. The caller (prefillBatchedMetal) uses the hybrid metal.m path when base != 0.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#include <CoreFoundation/CoreFoundation.h>

// Shared with metal.m.
extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;
extern int gMPSOK; // f16 resident prefill uses MPS matmuls; refuse when MPS was not validated
typedef struct { CFTypeRef buf; int out; int in; } MGWeight;
extern MGWeight gW[];
extern int gNW;
int mg_upload(const float *w, int out, int in);
void mg_f32_to_f16(const float *src, __fp16 *dst, long n);
void mg_f16_to_f32(const __fp16 *src, float *dst, long n);

// ---- MSL kernels (compiled once at runtime; no offline metal compiler needed) ----
static NSString *kSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_k(device const half* X [[buffer(0)]],
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

kernel void addbias_k(device half* Buf [[buffer(0)]],
                      device const half* B [[buffer(1)]],
                      constant uint& n [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
  Buf[gid] = half(float(Buf[gid]) + float(B[gid % n]));
}

kernel void rope_k(device half* Buf [[buffer(0)]],
                   constant uint& nHeads [[buffer(1)]],
                   constant uint& hd [[buffer(2)]],
                   constant uint& base [[buffer(3)]],
                   constant float& theta [[buffer(4)]],
                   uint gid [[thread_position_in_grid]]) {
  uint half_ = hd/2;
  uint perTok = nHeads*half_;
  uint t = gid / perTok;
  uint rem = gid % perTok;
  uint head = rem / half_;
  uint j = rem % half_;
  uint rowW = nHeads*hd;
  device half* hv = Buf + t*rowW + head*hd;
  float pos = float(base + t);
  float inv = pow(theta, -2.0f*float(j)/float(hd));
  float ang = pos*inv;
  float c = cos(ang), s = sin(ang);
  float a = float(hv[j]); float b = float(hv[j+half_]);
  hv[j] = half(a*c - b*s);
  hv[j+half_] = half(b*c + a*s);
}

// One simdgroup (32 lanes) per (token t, query head h): causal online-softmax attention over
// keys 0..t, GQA via kvh = h/grp. The 32 lanes split the head dim (hd<=128 => <=4 dims/lane), so
// q/acc stay in REGISTERS (4 floats/lane) instead of a 128-float thread-private array that spills
// to device memory — and the per-key QK dot is a simd_sum reduction across the lanes. This is the
// hot kernel: with the old 1-thread-per-(t,h) spilling version it alone cost ~280ms (P=256, 1.5B);
// cooperatively it drops to tens of ms, closing the bulk of the prefill gap to llama.cpp Metal.
kernel void attn_k(device const half* Q [[buffer(0)]],
                   device const half* K [[buffer(1)]],
                   device const half* V [[buffer(2)]],
                   device half* Out [[buffer(3)]],
                   constant uint& P [[buffer(4)]],
                   constant uint& nH [[buffer(5)]],
                   constant uint& hd [[buffer(6)]],
                   constant uint& w [[buffer(7)]],
                   constant uint& grp [[buffer(8)]],
                   constant float& scale [[buffer(9)]],
                   uint gid [[thread_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]]) {
  uint sg = gid / 32u;             // one simdgroup per (t,h)
  uint h = sg % nH;
  uint t = sg / nH;
  if (t >= P) return;
  uint kvh = h / grp;
  uint qrow = nH*hd;
  device const half* q = Q + t*qrow + h*hd;
  float qreg[4], acc[4] = {0,0,0,0};   // hd<=128 => at most 4 dims per lane
  uint nd = 0;
  for (uint d=lane; d<hd; d+=32u) qreg[nd++] = float(q[d]);
  float m = -INFINITY, l = 0.0f;
  for (uint j=0; j<=t; j++) {
    device const half* k = K + j*w + kvh*hd;
    float partial = 0.0f; uint idx = 0;
    for (uint d=lane; d<hd; d+=32u) partial += qreg[idx++]*float(k[d]);
    float sc = simd_sum(partial) * scale;   // full score, broadcast to all lanes
    float mNew = max(m, sc);
    float corr = exp(m - mNew);
    float p = exp(sc - mNew);
    l = l*corr + p;
    device const half* vv = V + j*w + kvh*hd;
    idx = 0;
    for (uint d=lane; d<hd; d+=32u) { acc[idx] = acc[idx]*corr + p*float(vv[d]); idx++; }
    m = mNew;
  }
  device half* o = Out + t*qrow + h*hd;
  float invl = (l>0.0f) ? 1.0f/l : 0.0f;
  uint idx = 0;
  for (uint d=lane; d<hd; d+=32u) o[d] = half(acc[idx++]*invl);
}

kernel void silumul_k(device half* G [[buffer(0)]],
                      device const half* U [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
  float g = float(G[i]);
  float s = g / (1.0f + exp(-g));
  G[i] = half(s * float(U[i]));
}

kernel void add_k(device half* X [[buffer(0)]],
                  device const half* Y [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
  X[i] = half(float(X[i]) + float(Y[i]));
}
)MSL";

// ---- model registration state ----
typedef struct {
    int q, k, v, o, gate, up, down;   // mg_upload weight ids (f16)
    int inNorm, postNorm;             // mg_upload_vec ids (f16)
    int qb, kb, vb;                   // bias ids or -1
} MGLayer;

#define MG_MAXL 128
static int gNL, gH, gHd, gNH, gNKV, gI, gAttnBias;
static float gEps, gTheta;
static MGLayer gL[MG_MAXL];
static int gFinalNorm;

static id<MTLComputePipelineState> psoNorm, psoBias, psoRope, psoAttn, psoSilu, psoAdd;
static id<MTLComputeCommandEncoder> gEnc;
static id<MTLCommandBuffer> gCB;
static int gFwdReady;

static int mg_fwd_init(void) {
    if (gFwdReady) return 1;
    if (gDev == nil) return 0;
    NSError *err = nil;
    id<MTLLibrary> lib = [gDev newLibraryWithSource:kSrc options:nil error:&err];
    if (lib == nil) { NSLog(@"mg_fwd: library compile failed: %@", err); return 0; }
    #define PSO(name) [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err]
    psoNorm = PSO(@"rmsnorm_k");
    psoBias = PSO(@"addbias_k");
    psoRope = PSO(@"rope_k");
    psoAttn = PSO(@"attn_k");
    psoSilu = PSO(@"silumul_k");
    psoAdd  = PSO(@"add_k");
    #undef PSO
    if (!psoNorm || !psoBias || !psoRope || !psoAttn || !psoSilu || !psoAdd) {
        NSLog(@"mg_fwd: pipeline build failed: %@", err); return 0;
    }
    gFwdReady = 1;
    return 1;
}

void mg_fwd_config(int nLayers, int H, int hd, int nH, int nKV, int Im, float eps, float theta, int attnBias) {
    gNL = nLayers; gH = H; gHd = hd; gNH = nH; gNKV = nKV; gI = Im;
    gEps = eps; gTheta = theta; gAttnBias = attnBias;
}

void mg_fwd_layer(int layer, int q, int k, int v, int o, int gate, int up, int down,
                  int inNorm, int postNorm, int qb, int kb, int vb) {
    if (layer < 0 || layer >= MG_MAXL) return;
    gL[layer] = (MGLayer){q, k, v, o, gate, up, down, inNorm, postNorm, qb, kb, vb};
}

void mg_fwd_final_norm(int id) { gFinalNorm = id; }

// mg_fwd_reset clears the per-model topology (geometry, the per-layer weight-id table,
// the final-norm id) so a subsequent model re-registers from a clean slate. The
// compiled MSL pipelines (psoNorm..psoAdd) are model-independent and kept — only the
// per-model wiring is cleared. Called by mg_reset (metal.m) as part of a full teardown.
void mg_fwd_reset(void) {
    gNL = gH = gHd = gNH = gNKV = gI = gAttnBias = 0;
    gEps = gTheta = 0.0f;
    gFinalNorm = 0;
    for (int i = 0; i < MG_MAXL; i++) gL[i] = (MGLayer){0};
    gEnc = nil;   // transient per-prefill encoder/command buffer; drop any stale refs
    gCB = nil;
}

// mg_upload_vec stores a 1-D f16 vector (norm weight or bias) in the weight table.
int mg_upload_vec(const float *v, int n) { return mg_upload(v, 1, n); }

static id<MTLBuffer> mgbuf(int elems) {
    return [gDev newBufferWithLength:(NSUInteger)((long)elems * 2) options:MTLResourceStorageModeShared];
}
static id<MTLBuffer> wbufOf(int wid) { return (__bridge id<MTLBuffer>)gW[wid].buf; }

static id<MTLComputeCommandEncoder> enc(void) {
    if (gEnc == nil) gEnc = [gCB computeCommandEncoder];
    return gEnc;
}
static void endEnc(void) {
    if (gEnc != nil) { [gEnc endEncoding]; gEnc = nil; }
}
static void dispatch1d(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> pso, NSUInteger n) {
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > n) tg = n;
    if (tg == 0) tg = 1;
    [e setComputePipelineState:pso];
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// MPS matmul Y[P,out] = src[P,in] * W[wid]^T, encoded onto gCB (f16 buffers, no round-trip).
static void mm_encode(int wid, id<MTLBuffer> src, id<MTLBuffer> dst, int P) {
    endEnc();
    MGWeight W = gW[wid];
    int in = W.in, out = W.out;
    MPSMatrixDescriptor *da = [MPSMatrixDescriptor matrixDescriptorWithRows:P columns:in rowBytes:(NSUInteger)in*2 dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *db = [MPSMatrixDescriptor matrixDescriptorWithRows:out columns:in rowBytes:(NSUInteger)in*2 dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *dc = [MPSMatrixDescriptor matrixDescriptorWithRows:P columns:out rowBytes:(NSUInteger)out*2 dataType:MPSDataTypeFloat16];
    MPSMatrix *A = [[MPSMatrix alloc] initWithBuffer:src descriptor:da];
    MPSMatrix *B = [[MPSMatrix alloc] initWithBuffer:wbufOf(wid) descriptor:db];
    MPSMatrix *C = [[MPSMatrix alloc] initWithBuffer:dst descriptor:dc];
    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:gDev transposeLeft:NO transposeRight:YES
                                    resultRows:P resultColumns:out interiorColumns:in alpha:1.0 beta:0.0];
    [mm encodeToCommandBuffer:gCB leftMatrix:A rightMatrix:B resultMatrix:C];
}

static void k_rmsnorm(id<MTLBuffer> X, int normID, id<MTLBuffer> Out, int P) {
    id<MTLComputeCommandEncoder> e = enc();
    [e setComputePipelineState:psoNorm];
    [e setBuffer:X offset:0 atIndex:0];
    [e setBuffer:wbufOf(normID) offset:0 atIndex:1];
    [e setBuffer:Out offset:0 atIndex:2];
    uint H = gH; [e setBytes:&H length:4 atIndex:3];
    [e setBytes:&gEps length:4 atIndex:4];
    NSUInteger tg = psoNorm.maxTotalThreadsPerThreadgroup; if (tg > (NSUInteger)P) tg = P; if (!tg) tg = 1;
    [e dispatchThreads:MTLSizeMake(P,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
}

static void k_addbias(id<MTLBuffer> Buf, int biasID, int n, int P) {
    id<MTLComputeCommandEncoder> e = enc();
    [e setComputePipelineState:psoBias];
    [e setBuffer:Buf offset:0 atIndex:0];
    [e setBuffer:wbufOf(biasID) offset:0 atIndex:1];
    uint nn = n; [e setBytes:&nn length:4 atIndex:2];
    dispatch1d(e, psoBias, (NSUInteger)P*n);
}

static void k_rope(id<MTLBuffer> Buf, int nHeads, int P) {
    id<MTLComputeCommandEncoder> e = enc();
    [e setComputePipelineState:psoRope];
    [e setBuffer:Buf offset:0 atIndex:0];
    uint nh = nHeads, hd = gHd, base = 0; [e setBytes:&nh length:4 atIndex:1];
    [e setBytes:&hd length:4 atIndex:2];
    [e setBytes:&base length:4 atIndex:3];
    [e setBytes:&gTheta length:4 atIndex:4];
    dispatch1d(e, psoRope, (NSUInteger)P*nHeads*(gHd/2));
}

static void k_attn(id<MTLBuffer> Q, id<MTLBuffer> K, id<MTLBuffer> V, id<MTLBuffer> Out, int P, float scale) {
    id<MTLComputeCommandEncoder> e = enc();
    [e setComputePipelineState:psoAttn];
    [e setBuffer:Q offset:0 atIndex:0];
    [e setBuffer:K offset:0 atIndex:1];
    [e setBuffer:V offset:0 atIndex:2];
    [e setBuffer:Out offset:0 atIndex:3];
    uint p = P, nH = gNH, hd = gHd, w = gNKV*gHd, grp = gNH/gNKV;
    [e setBytes:&p length:4 atIndex:4];
    [e setBytes:&nH length:4 atIndex:5];
    [e setBytes:&hd length:4 atIndex:6];
    [e setBytes:&w length:4 atIndex:7];
    [e setBytes:&grp length:4 atIndex:8];
    [e setBytes:&scale length:4 atIndex:9];
    // 32 lanes (one simdgroup) per (t,h); threadgroup is a multiple of 32 so simdgroups never
    // straddle a (t,h) boundary (gid/32 selects the pair, thread_index_in_simdgroup the lane).
    NSUInteger total = (NSUInteger)P*gNH*32;
    NSUInteger tg = psoAttn.maxTotalThreadsPerThreadgroup;
    tg -= tg % 32; if (tg > total) tg = total; if (tg == 0) tg = 32;
    [e dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
}

static void k_silumul(id<MTLBuffer> G, id<MTLBuffer> U, int n) {
    id<MTLComputeCommandEncoder> e = enc();
    [e setComputePipelineState:psoSilu];
    [e setBuffer:G offset:0 atIndex:0];
    [e setBuffer:U offset:0 atIndex:1];
    dispatch1d(e, psoSilu, (NSUInteger)n);
}

static void k_add(id<MTLBuffer> X, id<MTLBuffer> Y, int n) {
    id<MTLComputeCommandEncoder> e = enc();
    [e setComputePipelineState:psoAdd];
    [e setBuffer:X offset:0 atIndex:0];
    [e setBuffer:Y offset:0 atIndex:1];
    dispatch1d(e, psoAdd, (NSUInteger)n);
}

// mg_prefill runs the whole prefill on the GPU. X is f32[P*H] (token embeddings). On return:
// lastPre = f32[H] (last token, pre-final-norm); KrawOut/KpostOut/Vout = f32[nL*P*w]
// (pre-RoPE K, post-RoPE K, V) for the CPU cache. Returns 1 on success.
int mg_prefill(const float *X, int P, float *lastPre, float *KrawOut, float *KpostOut, float *Vout) {
    // The resident f16 prefill issues MPS matmuls (mpsMatmul); without a validated MPS device
    // (default headless posture — see gMPSOK / FAK_METAL_MPS) it cannot run. Decline so the caller
    // falls back to the hybrid/CPU path. The q4_k hybrid prefill issues mg_q4k_* directly and never
    // reaches here, so it is unaffected.
    if (!gMPSOK) {
        static int mpsWarned = 0;
        if (!mpsWarned) {
            mpsWarned = 1;
            NSLog(@"mg_prefill: MPS unavailable (FAK_METAL_MPS not set or unsupported); the f16 "
                  @"prefill is disabled — use the q4_k prefill path (FAK_Q4K=1).");
        }
        return 0;
    }
    if (!mg_fwd_init()) return 0;
    @autoreleasepool {
        int H = gH, hd = gHd, nH = gNH, nKV = gNKV, Im = gI, w = nKV*hd, qrow = nH*hd;
        float scale = 1.0f / sqrtf((float)hd);

        id<MTLBuffer> Xb = mgbuf(P*H);
        mg_f32_to_f16(X, (__fp16 *)Xb.contents, (long)P*H);
        id<MTLBuffer> Xn = mgbuf(P*H), Xn2 = mgbuf(P*H);
        id<MTLBuffer> Qb = mgbuf(P*qrow), attn = mgbuf(P*qrow), tmpH = mgbuf(P*H);
        id<MTLBuffer> Gb = mgbuf(P*Im), Ub = mgbuf(P*Im);
        id<MTLBuffer> Kpost[MG_MAXL] = {0}, Kpre[MG_MAXL] = {0}, Vb[MG_MAXL] = {0};
        for (int l=0; l<gNL; l++) { Kpost[l]=mgbuf(P*w); Kpre[l]=mgbuf(P*w); Vb[l]=mgbuf(P*w); }

        int prof = getenv("FAK_QPROFILE") != NULL;
        CFTimeInterval tEncStart = prof ? CFAbsoluteTimeGetCurrent() : 0;

        gCB = [gQueue commandBuffer];
        gEnc = nil;
        id<MTLBlitCommandEncoder> blit;

        for (int l=0; l<gNL; l++) {
            MGLayer L = gL[l];
            k_rmsnorm(Xb, L.inNorm, Xn, P);                 // Xn = rmsnorm(X)
            mm_encode(L.q, Xn, Qb, P);                       // Q
            mm_encode(L.k, Xn, Kpost[l], P);                 // K
            mm_encode(L.v, Xn, Vb[l], P);                    // V
            if (gAttnBias) {
                k_addbias(Qb, L.qb, qrow, P);
                k_addbias(Kpost[l], L.kb, w, P);
                k_addbias(Vb[l], L.vb, w, P);
            }
            // stash pre-RoPE K (Kraw) before roping K
            endEnc();
            blit = [gCB blitCommandEncoder];
            [blit copyFromBuffer:Kpost[l] sourceOffset:0 toBuffer:Kpre[l] destinationOffset:0 size:(NSUInteger)P*w*2];
            [blit endEncoding];
            k_rope(Qb, nH, P);
            k_rope(Kpost[l], nKV, P);
            k_attn(Qb, Kpost[l], Vb[l], attn, P, scale);     // attnOut
            mm_encode(L.o, attn, tmpH, P);                   // O
            k_add(Xb, tmpH, P*H);                            // X += O
            k_rmsnorm(Xb, L.postNorm, Xn2, P);               // Xn2 = rmsnorm(X)
            mm_encode(L.gate, Xn2, Gb, P);                   // gate
            mm_encode(L.up, Xn2, Ub, P);                     // up
            k_silumul(Gb, Ub, P*Im);                          // G = silu(G)*U
            mm_encode(L.down, Gb, tmpH, P);                  // down
            k_add(Xb, tmpH, P*H);                            // X += down
        }
        endEnc();
        CFTimeInterval tEncEnd = prof ? CFAbsoluteTimeGetCurrent() : 0;
        [gCB commit];
        [gCB waitUntilCompleted];
        if (prof) {
            CFTimeInterval tGpu = CFAbsoluteTimeGetCurrent();
            NSLog(@"mg_prefill P=%d: cpu-encode=%.1f ms  gpu-commit->wait=%.1f ms",
                  P, (tEncEnd-tEncStart)*1000.0, (tGpu-tEncEnd)*1000.0);
        }

        // last token (pre-final-norm) -> f32
        mg_f16_to_f32((const __fp16 *)Xb.contents + (long)(P-1)*H, lastPre, H);
        // K/V back to f32 for the CPU cache
        for (int l=0; l<gNL; l++) {
            long off = (long)l*P*w;
            mg_f16_to_f32((const __fp16 *)Kpre[l].contents,  KrawOut + off,  (long)P*w);
            mg_f16_to_f32((const __fp16 *)Kpost[l].contents, KpostOut + off, (long)P*w);
            mg_f16_to_f32((const __fp16 *)Vb[l].contents,    Vout + off,     (long)P*w);
        }
        gCB = nil;
        return 1;
    }
}
