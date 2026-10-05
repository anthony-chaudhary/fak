//go:build darwin && arm64 && cgo

// prefill_attn_darwin.m — the tiled, GQA-sharing causal prefill attention kernel (fak#13695).
//
// Structure adapted from llama.cpp's Metal flash attention (kernel_flash_attn_ext in
// ggml-metal.metal, MIT; Copyright (c) 2023-2026 The ggml authors): Q tiles of 8 query rows,
// K/V tiles staged in threadgroup memory, simdgroup 8x8 matrix MMA for both QK^T and PV, and an
// online softmax that rescales the running output through a diagonal MMA. Differences that keep
// FAK's prefill contract: f32 K/V (the f32 KV cache) and f32 MMA, so the result matches the host
// attnPrefillInto oracle to float rounding; one threadgroup covers ALL heads of a GQA group (up to
// 8) for its 8 query rows, so every K/V tile load is shared by the whole group; causal masking is
// offset by the cached prefix (kv_len - P) and an optional sliding window bound W keeps keys
// (q - W, q].

#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

extern id<MTLDevice> gDev;
extern id<MTLCommandQueue> gQueue;
int mg_init(void);

typedef struct {
    int32_t P;       // query rows in this panel
    int32_t kv_len;  // cached positions the panel attends over (base + P)
    int32_t nH;      // query heads
    int32_t nKV;     // KV heads
    int32_t grp;     // nH / nKV
    int32_t hg;      // heads per threadgroup (a divisor of grp, <= 8)
    int32_t window;  // sliding window W (<= 0: full causal)
    float scale;     // softmax scale applied to q.k
} PrefillAttnParams;

// Bounded like every other FAK Metal wait (MG_Q4K_WAIT_LIMIT_MS, metal_stall.go): a stalled GPU
// must surface as a reportable timeout, never an uninterruptible waitUntilCompleted.
static const int64_t MG_PA_WAIT_LIMIT_MS = 10000;

static NSString *kPrefillAttnMSL = @R"MSL(
#include <metal_stdlib>
using namespace metal;

struct PrefillAttnParams {
    int P;
    int kv_len;
    int nH;
    int nKV;
    int grp;
    int hg;
    int window;
    float scale;
};

// Threadgroup layout: K tile [BK][HD], V tile [BK][HD], then per simdgroup an S/P tile [8][BK]
// and an 8x8 diagonal scratch. One simdgroup owns one query head and the threadgroup's 8 rows.
template <int HD, int BK, bool QREG>
inline void prefill_attn_body(device const float* Q, device const float* K, device const float* V,
                              device float* O, constant PrefillAttnParams& p, threadgroup float* tg,
                              uint2 tgp, ushort tid, ushort sg, ushort lane, ushort nthreads) {
    constexpr int ND = HD / 8;   // 8-wide head-dim tiles
    constexpr int NK = BK / 8;   // 8-wide key tiles per block
    constexpr int SPL = BK / 4;  // scores per lane in the softmax mapping (4 lanes per row)

    const int qt0 = (int)tgp.x * 8;
    const int hsub = p.grp / p.hg;
    const int kvh = (int)tgp.y / hsub;
    const int h = kvh * p.grp + ((int)tgp.y % hsub) * p.hg + (int)sg;
    const int qstride = p.nH * HD;
    const int kvstride = p.nKV * HD;
    const int base = p.kv_len - p.P;

    threadgroup float* tK = tg;
    threadgroup float* tV = tK + BK * HD;
    threadgroup float* tS = tV + BK * HD + (int)sg * (8 * BK + 64);
    threadgroup float* tD = tS + 8 * BK;

    device const float* qp = Q + (long)qt0 * qstride + (long)h * HD;
    simdgroup_float8x8 qm[QREG ? ND : 1];
    if (QREG) {
        for (int i = 0; i < ND; i++) simdgroup_load(qm[i], qp + i * 8, (ulong)qstride);
    }
    simdgroup_float8x8 om[ND];
    for (int i = 0; i < ND; i++) om[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    // Softmax mapping: lane -> row r (0..7), 4 lanes per row, each lane SPL key columns.
    const int r = (int)lane / 4;
    const int c0 = (int)lane % 4;
    const int qabs = base + qt0 + r;
    float m_run = -INFINITY;
    float l_run = 0.0f;

    const int q_last = base + min(qt0 + 8, p.P) - 1;  // newest query this tile owns
    int k_lo = 0;
    if (p.window > 0) k_lo = max(0, base + qt0 - p.window + 1);
    k_lo = (k_lo / BK) * BK;
    const int k_hi = q_last + 1;                       // causal: no key past the newest query

    for (int kb = k_lo; kb < k_hi; kb += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = (int)tid; idx < BK * HD / 4; idx += (int)nthreads) {
            const int row = idx / (HD / 4);
            const int col = (idx % (HD / 4)) * 4;
            const int key = kb + row;
            float4 kv = float4(0.0f), vv = float4(0.0f);
            if (key < p.kv_len) {
                const long off = (long)key * kvstride + (long)kvh * HD + col;
                kv = *(device const float4*)(K + off);
                vv = *(device const float4*)(V + off);
            }
            *(threadgroup float4*)(tK + row * HD + col) = kv;
            *(threadgroup float4*)(tV + row * HD + col) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S[8][BK] = Q[8][HD] . K^T[HD][BK]
        simdgroup_float8x8 sm[NK];
        for (int j = 0; j < NK; j++) sm[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (int i = 0; i < ND; i++) {
            simdgroup_float8x8 qi;
            if (QREG) {
                qi = qm[i];
            } else {
                simdgroup_load(qi, qp + i * 8, (ulong)qstride);
            }
            for (int j = 0; j < NK; j++) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, tK + j * 8 * HD + i * 8, (ulong)HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sm[j], qi, kt, sm[j]);
            }
        }
        for (int j = 0; j < NK; j++) simdgroup_store(sm[j], tS + j * 8, (ulong)BK);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax over this block for row r: mask, row max, exp, running sum.
        float sv[SPL];
        float rowmax = -INFINITY;
        for (int k = 0; k < SPL; k++) {
            const int c = c0 + 4 * k;
            const int key = kb + c;
            const bool live = key <= qabs && key < p.kv_len && (p.window <= 0 || key > qabs - p.window);
            const float x = live ? tS[r * BK + c] * p.scale : -INFINITY;
            sv[k] = x;
            rowmax = max(rowmax, x);
        }
        rowmax = max(rowmax, simd_shuffle_xor(rowmax, 1));
        rowmax = max(rowmax, simd_shuffle_xor(rowmax, 2));
        const float m_new = max(m_run, rowmax);
        const float alpha = (m_new == -INFINITY) ? 1.0f : exp(m_run - m_new);
        float psum = 0.0f;
        for (int k = 0; k < SPL; k++) {
            const float pv = (m_new == -INFINITY) ? 0.0f : exp(sv[k] - m_new);
            tS[r * BK + c0 + 4 * k] = pv;
            psum += pv;
        }
        psum += simd_shuffle_xor(psum, 1);
        psum += simd_shuffle_xor(psum, 2);
        l_run = l_run * alpha + psum;
        m_run = m_new;

        // O = diag(alpha) . O : rescale every running output row before adding this block.
        for (int e = 0; e < 2; e++) {
            const int idx = (int)lane * 2 + e;
            const int rr = idx / 8, cc = idx % 8;
            const float a = simd_shuffle(alpha, (ushort)(rr * 4));
            tD[idx] = (rr == cc) ? a : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dm;
        simdgroup_load(dm, tD, 8);
        for (int i = 0; i < ND; i++) {
            simdgroup_float8x8 t;
            simdgroup_multiply(t, dm, om[i]);
            om[i] = t;
        }

        // O += P[8][BK] . V[BK][HD]
        for (int j = 0; j < NK; j++) {
            simdgroup_float8x8 pm;
            simdgroup_load(pm, tS + j * 8, (ulong)BK);
            for (int i = 0; i < ND; i++) {
                simdgroup_float8x8 vm;
                simdgroup_load(vm, tV + j * 8 * HD + i * 8, (ulong)HD);
                simdgroup_multiply_accumulate(om[i], pm, vm, om[i]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O = diag(1/l) . O, then store the live rows.
    const float inv = (l_run > 0.0f) ? 1.0f / l_run : 0.0f;
    for (int e = 0; e < 2; e++) {
        const int idx = (int)lane * 2 + e;
        const int rr = idx / 8, cc = idx % 8;
        const float a = simd_shuffle(inv, (ushort)(rr * 4));
        tD[idx] = (rr == cc) ? a : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dm;
    simdgroup_load(dm, tD, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device float* op = O + (long)qt0 * qstride + (long)h * HD;
    const bool full = qt0 + 8 <= p.P;
    for (int i = 0; i < ND; i++) {
        simdgroup_float8x8 t;
        simdgroup_multiply(t, dm, om[i]);
        if (full) {
            simdgroup_store(t, op + i * 8, (ulong)qstride);
        } else {
            simdgroup_store(t, tD, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int e = 0; e < 2; e++) {
                const int idx = (int)lane * 2 + e;
                const int rr = idx / 8, cc = idx % 8;
                if (qt0 + rr < p.P) op[(long)rr * qstride + i * 8 + cc] = tD[idx];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#define PREFILL_ATTN_KERNEL(NAME, HD, BK, QREG) \
kernel void NAME(device const float* Q [[buffer(0)]], \
                 device const float* K [[buffer(1)]], \
                 device const float* V [[buffer(2)]], \
                 device float*       O [[buffer(3)]], \
                 constant PrefillAttnParams& p [[buffer(4)]], \
                 threadgroup float* tg [[threadgroup(0)]], \
                 uint2 tgp [[threadgroup_position_in_grid]], \
                 ushort tid [[thread_index_in_threadgroup]], \
                 ushort sg [[simdgroup_index_in_threadgroup]], \
                 ushort lane [[thread_index_in_simdgroup]]) { \
    prefill_attn_body<HD, BK, QREG>(Q, K, V, O, p, tg, tgp, tid, sg, lane, (ushort)(p.hg * 32)); \
}

PREFILL_ATTN_KERNEL(prefill_attn_hd64, 64, 16, true)
PREFILL_ATTN_KERNEL(prefill_attn_hd128, 128, 16, true)
PREFILL_ATTN_KERNEL(prefill_attn_hd256, 256, 8, false)
)MSL";

static id<MTLComputePipelineState> gPAPSO[3]; // hd 64, 128, 256
static dispatch_once_t gPAOnce;
static id<MTLBuffer> gPAQ = nil, gPAK = nil, gPAV = nil, gPAO = nil;
static size_t gPAQCap = 0, gPAKVCap = 0, gPAOCap = 0;

static int mg_pa_slot(int hd) {
    switch (hd) {
    case 64: return 0;
    case 128: return 1;
    case 256: return 2;
    }
    return -1;
}

static int mg_pa_bk(int hd) { return hd > 128 ? 8 : 16; }

static void mg_pa_init(void) {
    dispatch_once(&gPAOnce, ^{
        if (!mg_init() || gDev == nil) return;
        NSError *err = nil;
        id<MTLLibrary> lib = [gDev newLibraryWithSource:kPrefillAttnMSL options:nil error:&err];
        if (lib == nil) { NSLog(@"prefill_attn: library compile failed: %@", err); return; }
        NSString *names[3] = {@"prefill_attn_hd64", @"prefill_attn_hd128", @"prefill_attn_hd256"};
        for (int i = 0; i < 3; i++) {
            id<MTLFunction> fn = [lib newFunctionWithName:names[i]];
            if (fn) gPAPSO[i] = [gDev newComputePipelineStateWithFunction:fn error:&err];
        }
    });
}

// mg_prefill_attn_supported reports whether a head geometry has a compiled pipeline: hd in
// {64,128,256}, nH divisible by nKV.
int mg_prefill_attn_supported(int hd, int nH, int nKV) {
    mg_pa_init();
    int s = mg_pa_slot(hd);
    return s >= 0 && gPAPSO[s] != nil && nH > 0 && nKV > 0 && nH % nKV == 0;
}

static int mg_pa_heads_per_tg(int grp) {
    for (int hg = grp < 8 ? grp : 8; hg > 1; hg--) {
        if (grp % hg == 0) return hg;
    }
    return 1;
}

static id<MTLBuffer> mg_pa_grow(id<MTLBuffer> b, size_t *cap, size_t need) {
    if (b != nil && *cap >= need) return b;
    size_t n = need + need / 4;
    id<MTLBuffer> nb = [gDev newBufferWithLength:n options:MTLResourceStorageModeShared];
    if (nb != nil) *cap = n;
    return nb;
}

// mg_prefill_attn computes causal GQA attention for a P-row prefill panel into out [P, nH*hd].
// q is [P, nH*hd]; k/v are the layer's KV cache [kv_len, nKV*hd] (kv_len = cached + P, the panel's
// keys last). window <= 0 is full causal. Returns 0 on success; a negative code before any command
// buffer exists (unsupported geometry / allocation), or -10 when the bounded wait expired or the
// device reported an error (out untouched in every failure).
int mg_prefill_attn(const float* q, const float* k, const float* v, float* out,
                    int P, int kv_len, int nH, int nKV, int hd, int window, float scale,
                    double* out_gpu_ms, double* out_wait_ms) {
    if (out_gpu_ms) *out_gpu_ms = 0;
    if (out_wait_ms) *out_wait_ms = 0;
    if (!q || !k || !v || !out || P <= 0 || kv_len < P) return -1;
    if (!mg_prefill_attn_supported(hd, nH, nKV)) return -2;
    int slot = mg_pa_slot(hd);
    int grp = nH / nKV;
    int hg = mg_pa_heads_per_tg(grp);
    int bk = mg_pa_bk(hd);
    int qrows = (P + 7) / 8 * 8; // pad so every 8-row Q tile load stays in bounds
    size_t qBytes = (size_t)qrows * nH * hd * sizeof(float);
    size_t kvBytes = (size_t)kv_len * nKV * hd * sizeof(float);
    size_t oBytes = (size_t)P * nH * hd * sizeof(float);
    @autoreleasepool {
        gPAQ = mg_pa_grow(gPAQ, &gPAQCap, qBytes);
        size_t kvCap = gPAKVCap;
        gPAK = mg_pa_grow(gPAK, &kvCap, kvBytes);
        size_t vCap = gPAKVCap;
        gPAV = mg_pa_grow(gPAV, &vCap, kvBytes);
        gPAKVCap = kvCap < vCap ? kvCap : vCap;
        gPAO = mg_pa_grow(gPAO, &gPAOCap, oBytes);
        if (!gPAQ || !gPAK || !gPAV || !gPAO) return -3;
        size_t qLive = (size_t)P * nH * hd * sizeof(float);
        memcpy(gPAQ.contents, q, qLive);
        if (qBytes > qLive) memset((char *)gPAQ.contents + qLive, 0, qBytes - qLive);
        memcpy(gPAK.contents, k, kvBytes);
        memcpy(gPAV.contents, v, kvBytes);

        PrefillAttnParams prm = {P, kv_len, nH, nKV, grp, hg, window > 0 ? window : 0, scale};
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        if (cb == nil) return -4;
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        if (e == nil) return -5;
        [e setComputePipelineState:gPAPSO[slot]];
        [e setBuffer:gPAQ offset:0 atIndex:0];
        [e setBuffer:gPAK offset:0 atIndex:1];
        [e setBuffer:gPAV offset:0 atIndex:2];
        [e setBuffer:gPAO offset:0 atIndex:3];
        [e setBytes:&prm length:sizeof(prm) atIndex:4];
        NSUInteger tgBytes = (NSUInteger)(2 * bk * hd + hg * (8 * bk + 64)) * sizeof(float);
        [e setThreadgroupMemoryLength:tgBytes atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)((P + 7) / 8), (NSUInteger)(nKV * (grp / hg)), 1)
           threadsPerThreadgroup:MTLSizeMake((NSUInteger)(hg * 32), 1, 1)];
        [e endEncoding];

        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [cb addCompletedHandler:^(id<MTLCommandBuffer> b) { (void)b; dispatch_semaphore_signal(sem); }];
        CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
        [cb commit];
        long timedOut = dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(MG_PA_WAIT_LIMIT_MS * NSEC_PER_MSEC)));
        if (out_wait_ms) *out_wait_ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0;
        if (timedOut != 0 || cb.status != MTLCommandBufferStatusCompleted) {
            // The command buffer may still reference the scratch: abandon it (the buffer keeps
            // its own references) so the next call can never write under a live dispatch.
            gPAQ = nil; gPAK = nil; gPAV = nil; gPAO = nil;
            gPAQCap = 0; gPAKVCap = 0; gPAOCap = 0;
            return -10;
        }
        if (out_gpu_ms) *out_gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        memcpy(out, gPAO.contents, oBytes);
    }
    return 0;
}
