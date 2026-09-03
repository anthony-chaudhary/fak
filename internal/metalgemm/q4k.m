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
#include "q8_bridge.h"
#include <CoreFoundation/CoreFoundation.h>
#include <math.h>
#include <stdatomic.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    uintptr_t command_buffer;
    int committed;
    int completed_wait;
    int host_readback;
    int encoders;
    double gpu_milliseconds;
    double wait_milliseconds;
    int timing_available;
} mg_execution_event;

static inline void mg_execution_event_reset(mg_execution_event* event) {
    if (event == NULL) return;
    event->command_buffer = 0;
    event->committed = 0;
    event->completed_wait = 0;
    event->host_readback = 0;
    event->encoders = 0;
    event->gpu_milliseconds = 0;
    event->wait_milliseconds = 0;
    event->timing_available = 0;
}

static inline void mg_execution_event_command_buffer(mg_execution_event* event, id<MTLCommandBuffer> cb) {
    if (event == NULL) return;
    event->command_buffer = (uintptr_t)(__bridge void*)cb;
}

static inline void mg_execution_event_encoder(mg_execution_event* event, id<MTLComputeCommandEncoder> encoder) {
    if (event != NULL && encoder != nil) event->encoders++;
}

static inline void mg_execution_event_committed(mg_execution_event* event) {
    if (event == NULL) return;
    event->committed = 1;
}

static inline void mg_execution_event_waited(mg_execution_event* event, id<MTLCommandBuffer> cb, CFAbsoluteTime wait_started) {
    if (event == NULL) return;
    event->completed_wait = cb.status == MTLCommandBufferStatusCompleted;
    event->wait_milliseconds = (CFAbsoluteTimeGetCurrent() - wait_started) * 1000.0;
    double gpu_start = cb.GPUStartTime;
    double gpu_end = cb.GPUEndTime;
    if (isfinite(gpu_start) && isfinite(gpu_end) && gpu_end > gpu_start) {
        event->gpu_milliseconds = (gpu_end - gpu_start) * 1000.0;
        event->timing_available = 1;
    }
}

static inline void mg_execution_event_readback(mg_execution_event* event) {
    if (event == NULL) return;
    event->host_readback = 1;
}
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

// q4k_tile_dot_vectorized is an opt-in P=1 experiment adapted from llama.cpp's Q4_K
// float4x4 dequant tile and vector-dot topology in ggml-metal.metal at
// 17197474510622a3b4ea7d0909d70b606f542b96 (MIT; Copyright (c) 2023-2026 The ggml authors).
// Upstream selects that exact float4x4 path for small batches and a separate packed-vector
// kernel for P=1. This bounded candidate deliberately applies the tile technique at fak's
// existing P=1 seam without changing resident bytes, row geometry, or the scalar control.
inline float q4k_tile_dot_vectorized(device const uchar* blk, device const float* xs, uint tile) {
    float d  = (float)(*(device const half*)(blk + 0));
    float dm = (float)(*(device const half*)(blk + 2));
    device const uchar* scales = blk + 4;
    device const uchar* q = blk + 16 + (tile / 4) * 32 + (tile & 1) * 16;
    float2 sm = q4k_scale_min(tile / 2, scales);
    float ds = d * sm.x;
    float ms = dm * sm.y;
    float4x4 weights;
    for (int k = 0; k < 4; k++) {
        packed_uchar4 packed = *(device const packed_uchar4*)(q + 4*k);
        uchar4 codes = uchar4(packed);
        codes = (tile & 2) ? codes >> 4 : codes & uchar4(0x0f);
        weights[k] = ds * float4(codes) - ms;
    }
    device const float4x4* xv = (device const float4x4*)(xs + tile * 16);
    return dot(weights[0], (*xv)[0]) + dot(weights[1], (*xv)[1]) +
           dot(weights[2], (*xv)[2]) + dot(weights[3], (*xv)[3]);
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

// q4k_gemv_vectorized keeps q4k_gemv's proven one-SIMD-group-per-row control geometry and
// changes only the inner unpack/MAC. It is never selected unless the host explicitly opts in.
kernel void q4k_gemv_vectorized(device const uchar* W [[buffer(0)]],
                                device const float* X [[buffer(1)]],
                                device float*       Y [[buffer(2)]],
                                constant int&    nblk [[buffer(3)]],
                                constant int&     out [[buffer(4)]],
                                uint o   [[threadgroup_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]]) {
    if (o >= (uint)out) return;
    device const uchar* row = W + (long)o * nblk * 144;
    float acc = 0.0f;
    const uint tile = lid & 15;
    for (int b = (int)(lid >> 4); b < nblk; b += 2) {
        acc += q4k_tile_dot_vectorized(row + (long)b * 144, X + (long)b * 256, tile);
    }
    acc = simd_sum(acc);
    if (lid == 0) Y[o] = acc;
}

// q4k_gemv_multi is the P=4..8 decode kernel. Following llama.cpp's small-batch Metal
// topology, each SIMD group carries four output rows and each 8-lane subgroup splits one
// row's Q4_K blocks. Scalar accumulators and compile-time P specializations keep the vector
// panel in registers: a decoded weight is applied to every active row before it is discarded.
inline float q4k_sum8(float v) {
    v += simd_shuffle_down(v, 4);
    v += simd_shuffle_down(v, 2);
    v += simd_shuffle_down(v, 1);
    return v;
}

template <int N>
inline void q4k_gemv_multi_impl(device const uchar* W,
                                device const float* X,
                                device float* Y,
                                constant int& nblk,
                                constant int& out,
                                uint tg,
                                uint lane,
                                uint sg) {
    const uint tx = lane & 7;
    const uint o = tg * 8 + sg * 4 + lane / 8;
    const bool valid = o < (uint)out;
    const long xstride = (long)nblk * 256;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

    if (valid) {
        device const uchar* row = W + (long)o * nblk * 144;
        for (int b = (int)tx; b < nblk; b += 8) {
            device const uchar* blk = row + (long)b * 144;
            float d  = (float)(*(device const half*)(blk + 0));
            float dm = (float)(*(device const half*)(blk + 2));
            device const uchar* scales = blk + 4;
            device const uchar* q = blk + 16;
            const long xbase = (long)b * 256;
            int qi = 0;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                float2 sm0 = q4k_scale_min(is,     scales);
                float2 sm1 = q4k_scale_min(is + 1, scales);
                float d1 = d * sm0.x, m1 = dm * sm0.y;
                float d2 = d * sm1.x, m2 = dm * sm1.y;
                for (int l = 0; l < 32; l++) {
                    const long xi = xbase + j + l;
                    const float w = d1 * (float)(q[qi + l] & 0x0f) - m1;
                    a0 += w * X[xi];
                    a1 += w * X[xstride + xi];
                    a2 += w * X[2 * xstride + xi];
                    a3 += w * X[3 * xstride + xi];
                    if (N >= 5) a4 += w * X[4 * xstride + xi];
                    if (N >= 6) a5 += w * X[5 * xstride + xi];
                    if (N >= 7) a6 += w * X[6 * xstride + xi];
                    if (N >= 8) a7 += w * X[7 * xstride + xi];
                }
                for (int l = 0; l < 32; l++) {
                    const long xi = xbase + j + 32 + l;
                    const float w = d2 * (float)(q[qi + l] >> 4) - m2;
                    a0 += w * X[xi];
                    a1 += w * X[xstride + xi];
                    a2 += w * X[2 * xstride + xi];
                    a3 += w * X[3 * xstride + xi];
                    if (N >= 5) a4 += w * X[4 * xstride + xi];
                    if (N >= 6) a5 += w * X[5 * xstride + xi];
                    if (N >= 7) a6 += w * X[6 * xstride + xi];
                    if (N >= 8) a7 += w * X[7 * xstride + xi];
                }
                qi += 32;
                is += 2;
            }
        }
    }

    a0 = q4k_sum8(a0); a1 = q4k_sum8(a1);
    a2 = q4k_sum8(a2); a3 = q4k_sum8(a3);
    if (N >= 5) a4 = q4k_sum8(a4);
    if (N >= 6) a5 = q4k_sum8(a5);
    if (N >= 7) a6 = q4k_sum8(a6);
    if (N >= 8) a7 = q4k_sum8(a7);
    if (tx == 0 && valid) {
        Y[o] = a0; Y[(long)out + o] = a1;
        Y[2 * (long)out + o] = a2; Y[3 * (long)out + o] = a3;
        if (N >= 5) Y[4 * (long)out + o] = a4;
        if (N >= 6) Y[5 * (long)out + o] = a5;
        if (N >= 7) Y[6 * (long)out + o] = a6;
        if (N >= 8) Y[7 * (long)out + o] = a7;
    }
}

#define Q4K_MULTI_KERNEL(N) \
kernel void q4k_gemv_multi##N(device const uchar* W [[buffer(0)]], \
                               device const float* X [[buffer(1)]], \
                               device float* Y [[buffer(2)]], \
                               constant int& nblk [[buffer(3)]], \
                               constant int& out [[buffer(4)]], \
                               uint tg [[threadgroup_position_in_grid]], \
                               uint lane [[thread_index_in_simdgroup]], \
                               uint sg [[simdgroup_index_in_threadgroup]]) { \
    q4k_gemv_multi_impl<N>(W, X, Y, nblk, out, tg, lane, sg); \
}

Q4K_MULTI_KERNEL(4)
Q4K_MULTI_KERNEL(5)
Q4K_MULTI_KERNEL(6)
Q4K_MULTI_KERNEL(7)
Q4K_MULTI_KERNEL(8)

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

// q4k_gemm_mm32: the exact-P32 SIMDGROUP-MATRIX candidate. The old generic MMA tile had BN=64,
// so an exact 32-token panel spent half its matrix work and half its accumulator storage on zero
// columns. MM32 makes the output tile BM=64 x BN=32: all eight simdgroups contribute to live P32
// columns, each owning 16 rows x 16 cols = a 2x2 array of 8x8 accumulators. The K axis still walks
// one 32-wide q4_k sub-block at a time in four hardware-MMA steps. C-side selection is exact:
// FAK_Q4K_MM requests this pipeline only for P=32; P31/P33 remain on q4k_gemm.
#define Q4K_MM32_BN 32
#define Q4K_MM32_SGROW 4  // simdgroups down BM=64 -> 16 rows (2 tiles) each
#define Q4K_MM32_SGCOL 2  // simdgroups across BN=32 -> 16 cols (2 tiles) each
kernel void q4k_gemm_mm32(device const uchar* W [[buffer(0)]],
                          device const float* X [[buffer(1)]],
                          device float*       Y [[buffer(2)]],
                          constant int&    nblk [[buffer(3)]],
                          constant int&     out [[buffer(4)]],
                          constant int&       P [[buffer(5)]],
                          constant int&      t0 [[buffer(6)]],
                          constant int&      nt [[buffer(7)]],
                          uint ob   [[threadgroup_position_in_grid]],
                          uint lid  [[thread_index_in_threadgroup]],
                          uint sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float wbuf[Q4K_BM * 32]; // BM weight rows x one 32-wide sub-block, row-major [row][k] ld=32
    threadgroup float xbuf[32 * Q4K_MM32_BN]; // one sub-block x 32 tokens, K-major [k][tok]
    int in = nblk * 256;
    int o0 = (int)ob * Q4K_BM;           // first output row this threadgroup owns
    // This simdgroup's position in the 4x2 grid -> its 16-row x 16-col output region.
    int sgRow = (int)sgid / Q4K_MM32_SGCOL; // 0..3
    int sgCol = (int)sgid % Q4K_MM32_SGCOL; // 0..1
    int rowBase = sgRow * 16;            // 0,16,32,48 within the BM tile
    int colBase = sgCol * 16;            // 0 or 16 within the exact-P32 tile
    // 2 row-tiles x 2 col-tiles = 4 accumulators of 8x8, C[out_row][token].
    simdgroup_float8x8 acc[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int sblk = 0; sblk < nblk; sblk++) {
        for (int sb = 0; sb < 8; sb++) {  // 8 q4_k sub-blocks of 32 per super-block
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
                wbuf[idx] = val; // [row][k], ld=32
            }
            for (int idx = (int)lid; idx < 32 * Q4K_MM32_BN; idx += Q4K_TG) {
                int k = idx / Q4K_MM32_BN, tk = idx % Q4K_MM32_BN;
                xbuf[idx] = (tk < nt) ? X[(long)(t0 + tk) * in + (long)sblk * 256 + sb * 32 + k] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Walk the 32-wide K sub-block in four 8-wide MMA steps. xbuf has exact ld=32.
            for (int kk = 0; kk < 32; kk += 8) {
                simdgroup_float8x8 bmat[2];
                for (int j = 0; j < 2; j++) {
                    simdgroup_load(bmat[j], xbuf + kk * Q4K_MM32_BN + (colBase + j * 8), Q4K_MM32_BN);
                }
                for (int i = 0; i < 2; i++) {
                    simdgroup_float8x8 amat;
                    simdgroup_load(amat, wbuf + (rowBase + i * 8) * 32 + kk, 32);
                    for (int j = 0; j < 2; j++) {
                        simdgroup_multiply_accumulate(acc[i][j], amat, bmat[j], acc[i][j]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup float cbuf[Q4K_BM * Q4K_MM32_BN];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int r = rowBase + i * 8, c = colBase + j * 8;
            simdgroup_store(acc[i][j], cbuf + r * Q4K_MM32_BN + c, Q4K_MM32_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Cooperative write-back: each thread strides the exact 64x32 tile.
    for (int idx = (int)lid; idx < Q4K_BM * Q4K_MM32_BN; idx += Q4K_TG) {
        int r = idx / Q4K_MM32_BN, c = idx % Q4K_MM32_BN;
        int orow = o0 + r;
        int tcol = c;
        if (orow < out && tcol < nt) Y[(long)(t0 + tcol) * out + orow] = cbuf[idx];
    }
}

// q4k_gemm_m5_cooperative_smem is the clean-room Q4_K adaptation of Modular's cooperative-SMEM
// Apple M5 W4A16 mechanism (modular/modular@1c9fd2e, fp4_matmul.mojo). It preserves FAK's raw
// Q4_K/f32 contract: each 32-wide packed-weight tile is decoded once cooperatively into threadgroup
// memory, then reused by four dense simdgroup MMA K-steps. The existing q4k_gemm path remains the
// control/fallback; production routing stays disabled until a device-pinned >=1.10x crossover receipt.
//
// q4k_gemm_m5_cooperative_smem: the 64-token SIMDGROUP-MATRIX candidate. The old generic MMA tile had BN=64,
// so an exact 32-token panel spent half its matrix work and half its accumulator storage on zero
// columns. MM32 makes the output tile BM=64 x BN=32: all eight simdgroups contribute to live P32
// columns, each owning 16 rows x 16 cols = a 2x2 array of 8x8 accumulators. The K axis still walks
// one 32-wide q4_k sub-block at a time in four hardware-MMA steps. C-side selection is exact:
// FAK_Q4K_MM requests this pipeline only for P=32; P31/P33 remain on q4k_gemm.
#define Q4K_M5_BN 64
#define Q4K_M5_SGROW 2  // simdgroups down BM=64 -> 32 rows (4 tiles) each
#define Q4K_M5_SGCOL 4  // simdgroups across BN=64 -> 16 cols (2 tiles) each
kernel void q4k_gemm_m5_cooperative_smem(device const uchar* W [[buffer(0)]],
                          device const float* X [[buffer(1)]],
                          device float*       Y [[buffer(2)]],
                          constant int&    nblk [[buffer(3)]],
                          constant int&     out [[buffer(4)]],
                          constant int&       P [[buffer(5)]],
                          constant int&      t0 [[buffer(6)]],
                          constant int&      nt [[buffer(7)]],
                          uint ob   [[threadgroup_position_in_grid]],
                          uint lid  [[thread_index_in_threadgroup]],
                          uint sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float wbuf[Q4K_BM * 32]; // BM weight rows x one 32-wide sub-block, row-major [row][k] ld=32
    threadgroup float xbuf[32 * Q4K_M5_BN]; // one sub-block x 32 tokens, K-major [k][tok]
    int in = nblk * 256;
    int o0 = (int)ob * Q4K_BM;           // first output row this threadgroup owns
    // This simdgroup's position in the 4x2 grid -> its 16-row x 16-col output region.
    int sgRow = (int)sgid / Q4K_M5_SGCOL; // 0..3
    int sgCol = (int)sgid % Q4K_M5_SGCOL; // 0..1
    int rowBase = sgRow * 32;            // 0,16,32,48 within the BM tile
    int colBase = sgCol * 16;            // 0 or 16 within the exact-P32 tile
    // 2 row-tiles x 2 col-tiles = 4 accumulators of 8x8, C[out_row][token].
    simdgroup_float8x8 acc[4][2];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int sblk = 0; sblk < nblk; sblk++) {
        for (int sb = 0; sb < 8; sb++) {  // 8 q4_k sub-blocks of 32 per super-block
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
                wbuf[idx] = val; // [row][k], ld=32
            }
            for (int idx = (int)lid; idx < 32 * Q4K_M5_BN; idx += Q4K_TG) {
                int k = idx / Q4K_M5_BN, tk = idx % Q4K_M5_BN;
                xbuf[idx] = (tk < nt) ? X[(long)(t0 + tk) * in + (long)sblk * 256 + sb * 32 + k] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Walk the 32-wide K sub-block in four 8-wide MMA steps. xbuf has exact ld=32.
            for (int kk = 0; kk < 32; kk += 8) {
                simdgroup_float8x8 bmat[2];
                for (int j = 0; j < 2; j++) {
                    simdgroup_load(bmat[j], xbuf + kk * Q4K_M5_BN + (colBase + j * 8), Q4K_M5_BN);
                }
                for (int i = 0; i < 4; i++) {
                    simdgroup_float8x8 amat;
                    simdgroup_load(amat, wbuf + (rowBase + i * 8) * 32 + kk, 32);
                    for (int j = 0; j < 2; j++) {
                        simdgroup_multiply_accumulate(acc[i][j], amat, bmat[j], acc[i][j]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup float cbuf[Q4K_BM * Q4K_M5_BN];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            int r = rowBase + i * 8, c = colBase + j * 8;
            simdgroup_store(acc[i][j], cbuf + r * Q4K_M5_BN + c, Q4K_M5_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Cooperative write-back: each thread strides the exact 64x32 tile.
    for (int idx = (int)lid; idx < Q4K_BM * Q4K_M5_BN; idx += Q4K_TG) {
        int r = idx / Q4K_M5_BN, c = idx % Q4K_M5_BN;
        int orow = o0 + r;
        int tcol = c;
        if (orow < out && tcol < nt) Y[(long)(t0 + tcol) * out + orow] = cbuf[idx];
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
        for (int is = 0; is < 2; is++) {
            float ds1 = d * (float)sc[scOff + is + 0];
            float ds2 = d * (float)sc[scOff + is + 2];
            float ds3 = d * (float)sc[scOff + is + 4];
            float ds4 = d * (float)sc[scOff + is + 6];
            for (int li = 0; li < 16; li++) {
                int l = is * 16 + li;
                int q1 = (int)((ql[qlOff + l +  0] & 0x0f) | (((qh[qhOff + l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[qlOff + l + 32] & 0x0f) | (((qh[qhOff + l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[qlOff + l +  0] >> 4)   | (((qh[qhOff + l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[qlOff + l + 32] >> 4)   | (((qh[qhOff + l] >> 6) & 3) << 4)) - 32;
                acc += ds1 * (float)q1 * xs[n + l +  0];
                acc += ds2 * (float)q2 * xs[n + l + 32];
                acc += ds3 * (float)q3 * xs[n + l + 64];
                acc += ds4 * (float)q4 * xs[n + l + 96];
            }
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

// q6k_gemm: batched prefill GEMM for resident Q6_K rows. It is deliberately the simple prefill
// twin of q6k_gemv: one SIMD group per (output row, prompt token), with the 32 lanes splitting that
// row's 256-wide super-blocks. The result layout is token-major Y[t*out + o], matching the CPU
// kQuantMatRowsIntoBatch contract.
kernel void q6k_gemm(device const uchar* W [[buffer(0)]],
                     device const float* X [[buffer(1)]],
                     device float*       Y [[buffer(2)]],
                     constant int&    nblk [[buffer(3)]],
                     constant int&     out [[buffer(4)]],
                     constant int&       P [[buffer(5)]],
                     uint2 gid [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]]) {
    uint o = gid.x;
    uint t = gid.y;
    if (o >= (uint)out || t >= (uint)P) return;
    device const uchar* row = W + (long)o * nblk * 210;
    device const float* xs = X + (long)t * nblk * 256;
    float acc = 0.0f;
    for (int b = (int)lid; b < nblk; b += 32) {
        acc += q6k_block_dot(row + (long)b * 210, xs + (long)b * 256);
    }
    acc = simd_sum(acc);
    if (lid == 0) Y[(long)t * out + o] = acc;
}

// Exact Q8_0 activation quantization for a graph-resident f32 panel. One 32-lane
// threadgroup owns one block, matching model.quantizeRowQ8scalar's amax/127 and
// round-half-away-from-zero contract. The resulting codes/scales remain owned by
// the graph and may feed any number of downstream Q8 projections before its fence.
kernel void graph_quantize_q8(device const float* X [[buffer(0)]],
                              device char* Q [[buffer(1)]],
                              device float* D [[buffer(2)]],
                              constant int& blocks [[buffer(3)]],
                              uint block [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]]) {
    if (block >= (uint)blocks || lane >= 32) return;
    float v = X[(long)block * 32 + lane];
    float a = fabs(v);
    a = simd_max(a);
    float d = simd_broadcast(a / 127.0f, 0);
    if (lane == 0) D[block] = d;
    float q = d == 0.0f ? 0.0f : v / d;
    q = q >= 0.0f ? floor(q + 0.5f) : ceil(q - 0.5f);
    q = clamp(q, -127.0f, 127.0f);
    Q[(long)block * 32 + lane] = (char)q;
}
)MSL";

static id<MTLComputePipelineState> psoQ4KGemv, psoQ4KGemvVectorized, psoQ4KGemvMulti[5], psoQ4KGemm, psoQ4KGemmMM32, psoQ4KGemmM5CooperativeSMEM, psoQ4KSwiGLU, psoQ6KGemv, psoQ6KGemm, psoGraphQuantizeQ8;
static int gQ4KReady;

// q4k_gemv_pso binds selection to an executed-kernel status. A vector request never falls back:
// nil means the caller must return before allocating a command buffer or touching the output.
// vectorized_mode < 0 is the focused witness's unavailable-PSO injection; production sends 0/1.
static id<MTLComputePipelineState> q4k_gemv_pso(int vectorized_mode, int* executed) {
    *executed = 0;
    if (vectorized_mode == 0) {
        if (psoQ4KGemv == nil) return nil;
        *executed = 1;
        return psoQ4KGemv;
    }
    if (vectorized_mode < 0 || psoQ4KGemvVectorized == nil) return nil;
    *executed = 2;
    return psoQ4KGemvVectorized;
}

// q4k_gemm_pso binds exact shape selection to a typed executed identity. MM32 is eligible only for
// P=32 and explicit mode=1. P31/P33 execute scalar even when the process opt-in is enabled. A
// requested-but-unavailable MM32 candidate returns nil before scratch allocation, command-buffer
// creation, dispatch, timing publication, or caller-output mutation. mode<0 is the deterministic
// unavailable-candidate witness; production sends only 0/1.
static id<MTLComputePipelineState> q4k_gemm_pso(int P, int mm_mode, int* executed, int* token_tile) {
    *executed = 0;
    *token_tile = 64;
    if (mm_mode < 0) return nil;
    if (mm_mode == 2) {
        if (P < 64 || psoQ4KGemmM5CooperativeSMEM == nil) return nil;
        *executed = 3;
        *token_tile = 64;
        return psoQ4KGemmM5CooperativeSMEM;
    }
    if (P != 32 || mm_mode == 0) {
        if (psoQ4KGemm == nil) return nil;
        *executed = 1;
        return psoQ4KGemm;
    }
    if (mm_mode != 1 || psoQ4KGemmMM32 == nil) return nil;
    *executed = 2;
    *token_tile = 32;
    return psoQ4KGemmMM32;
}

static int q4k_init(void) {
    if (gQ4KReady) return 1;
    if (gDev == nil) return 0;
    NSError *err = nil;
    id<MTLLibrary> lib = [gDev newLibraryWithSource:kQ4KSrc options:nil error:&err];
    if (lib == nil) { NSLog(@"q4k: library compile failed: %@", err); return 0; }
    psoQ4KGemv = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q4k_gemv"] error:&err];
    psoQ4KGemvVectorized = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q4k_gemv_vectorized"] error:&err];
    for (int n = 4; n <= 8; n++) {
        NSString *name = [NSString stringWithFormat:@"q4k_gemv_multi%d", n];
        psoQ4KGemvMulti[n - 4] = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
    }
    psoQ4KGemm = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q4k_gemm"] error:&err];
    // q4k_gemm_mm32 is optional. Explicit P32 requests fail closed if this pipeline is unavailable;
    // P31/P33 and default-off P32 dispatches retain the required scalar pipeline.
    psoQ4KGemmMM32 = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q4k_gemm_mm32"] error:&err];
    // Experimental and optional: no production selector requests this PSO until an M5 crossover
    // table is backed by a sanctioned fak-native receipt.
    psoQ4KGemmM5CooperativeSMEM = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q4k_gemm_m5_cooperative_smem"] error:&err];
    psoQ4KSwiGLU = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q4k_swiglu"] error:&err];
    psoQ6KGemv = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q6k_gemv"] error:&err];
    psoQ6KGemm = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q6k_gemm"] error:&err];
    psoGraphQuantizeQ8 = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"graph_quantize_q8"] error:&err];
    if (!psoQ4KGemv || !psoQ4KGemvMulti[0] || !psoQ4KGemvMulti[1] || !psoQ4KGemvMulti[2] ||
        !psoQ4KGemvMulti[3] || !psoQ4KGemvMulti[4] || !psoQ4KGemm || !psoQ4KSwiGLU ||
        !psoQ6KGemv || !psoQ6KGemm || !psoGraphQuantizeQ8) { NSLog(@"q4k: pipeline build failed: %@", err); return 0; }
    gQ4KReady = 1;
    return 1;
}

typedef struct {
    CFTypeRef buf; // retained id<MTLBuffer>, raw q4_k bytes [out * nblk * 144]
    int out;
    int in;
    int nblk;
    NSUInteger offset;
} Q4KW;

#define MG_MAX_Q4 8192
static Q4KW gQ4[MG_MAX_Q4];
static int gNQ4 = 0;

static int q4k_register_buffer(id<MTLBuffer> b, int out, int in, int nblk) {
    if (gNQ4 >= MG_MAX_Q4) return -1;
    int id = gNQ4++;
    gQ4[id].buf = CFBridgingRetain(b);
    gQ4[id].out = out;
    gQ4[id].in = in;
    gQ4[id].nblk = nblk;
    gQ4[id].offset = 0;
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
void mg_q4k_mlp(int gate_wid, int up_wid, int down_wid, const float* x, float* y, mg_execution_event* event) {
    mg_execution_event_reset(event);
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
        mg_execution_event_command_buffer(event, cb);

        // (1) gate = G·x and up = U·x (independent), one encoder
        id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e1);
        [e1 setComputePipelineState:psoQ4KGemv];
        [e1 setBuffer:xb offset:0 atIndex:1];
        [e1 setBuffer:(__bridge id<MTLBuffer>)G.buf offset:G.offset atIndex:0];
        [e1 setBuffer:gMlpGate offset:0 atIndex:2];
        [e1 setBytes:&G.nblk length:sizeof(int) atIndex:3];
        [e1 setBytes:&G.out  length:sizeof(int) atIndex:4];
        [e1 dispatchThreadgroups:MTLSizeMake((NSUInteger)G.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e1 setBuffer:(__bridge id<MTLBuffer>)U.buf offset:U.offset atIndex:0];
        [e1 setBuffer:gMlpUp offset:0 atIndex:2];
        [e1 setBytes:&U.nblk length:sizeof(int) atIndex:3];
        [e1 setBytes:&U.out  length:sizeof(int) atIndex:4];
        [e1 dispatchThreadgroups:MTLSizeMake((NSUInteger)U.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e1 endEncoding];

        // (2) inter = silu(gate) * up
        id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e2);
        [e2 setComputePipelineState:psoQ4KSwiGLU];
        [e2 setBuffer:gMlpGate offset:0 atIndex:0];
        [e2 setBuffer:gMlpUp offset:0 atIndex:1];
        [e2 setBuffer:gMlpInter offset:0 atIndex:2];
        [e2 setBytes:&I length:sizeof(int) atIndex:3];
        [e2 dispatchThreads:MTLSizeMake((NSUInteger)I,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 endEncoding];

        // (3) y = D·inter
        id<MTLComputeCommandEncoder> e3 = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e3);
        [e3 setComputePipelineState:psoQ4KGemv];
        [e3 setBuffer:gMlpInter offset:0 atIndex:1];
        [e3 setBuffer:(__bridge id<MTLBuffer>)D.buf offset:D.offset atIndex:0];
        [e3 setBuffer:yb offset:0 atIndex:2];
        [e3 setBytes:&D.nblk length:sizeof(int) atIndex:3];
        [e3 setBytes:&D.out  length:sizeof(int) atIndex:4];
        [e3 dispatchThreadgroups:MTLSizeMake((NSUInteger)D.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e3 endEncoding];

        [cb commit];
        mg_execution_event_committed(event);
        CFAbsoluteTime wait_started = CFAbsoluteTimeGetCurrent();
        [cb waitUntilCompleted];
        mg_execution_event_waited(event, cb, wait_started);
        memcpy(y, yb.contents, (size_t)D.out * 4);
        mg_execution_event_readback(event);
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

// mg_q6k_gemv computes y[out] = W[wid] · x for a resident Q6_K weight in one command buffer.
// The fused MLP already uses q6k_gemv as stage 3; this standalone wrapper lets k-quant decode
// sites such as the Qwen3.6 Q6_K LM head stay on Metal instead of escaping to the CPU.
void mg_q6k_gemv(int wid, const float* x, float* y, mg_execution_event* event) {
    mg_execution_event_reset(event);
    if (wid < MG_Q6_BASE || (wid - MG_Q6_BASE) >= gNQ6) return;
    @autoreleasepool {
        Q6KW W = gQ6[wid - MG_Q6_BASE];
        q4k_grow_scratch((long)W.in, (long)W.out);
        id<MTLBuffer> xb = gQXBuf, yb = gQYBuf;
        memcpy(xb.contents, x, (size_t)W.in * 4);

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        mg_execution_event_command_buffer(event, cb);
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e);
        [e setComputePipelineState:psoQ6KGemv];
        [e setBuffer:(__bridge id<MTLBuffer>)W.buf offset:0 atIndex:0];
        [e setBuffer:xb offset:0 atIndex:1];
        [e setBuffer:yb offset:0 atIndex:2];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cb commit];
        mg_execution_event_committed(event);
        CFAbsoluteTime wait_started = CFAbsoluteTimeGetCurrent();
        [cb waitUntilCompleted];
        mg_execution_event_waited(event, cb, wait_started);

        memcpy(y, yb.contents, (size_t)W.out * 4);
        mg_execution_event_readback(event);
    }
}

// mg_q6k_gemm computes Y[P,out] = X[P,in] * W[wid]^T for a resident Q6_K weight in one command
// buffer. This is the prefill counterpart to q6k_gemv / mg_q4k_mlp_q6down's stage 3: dense
// q4_k_m down_proj can stay on Metal instead of using the host kQuantMatRowsIntoBatch loop.
void mg_q6k_gemm(int wid, const float* X, int P, float* Y, mg_execution_event* event) {
    mg_execution_event_reset(event);
    if (wid < MG_Q6_BASE || (wid - MG_Q6_BASE) >= gNQ6 || P <= 0) return;
    @autoreleasepool {
        Q6KW W = gQ6[wid - MG_Q6_BASE];
        q4k_grow_scratch((long)P * W.in, (long)P * W.out);
        id<MTLBuffer> xb = gQXBuf, yb = gQYBuf;
        memcpy(xb.contents, X, (size_t)P * W.in * 4);

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        mg_execution_event_command_buffer(event, cb);
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e);
        [e setComputePipelineState:psoQ6KGemm];
        [e setBuffer:(__bridge id<MTLBuffer>)W.buf offset:0 atIndex:0];
        [e setBuffer:xb offset:0 atIndex:1];
        [e setBuffer:yb offset:0 atIndex:2];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        [e setBytes:&P      length:sizeof(int) atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, (NSUInteger)P, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cb commit];
        mg_execution_event_committed(event);
        CFAbsoluteTime wait_started = CFAbsoluteTimeGetCurrent();
        [cb waitUntilCompleted];
        mg_execution_event_waited(event, cb, wait_started);

        memcpy(Y, yb.contents, (size_t)P * W.out * 4);
        mg_execution_event_readback(event);
    }
}

// ---- Batched fused expert MLP (issue #1382: the mlp_decode decode lever) ----
// A Qwen3.6-27B q4_k_m MoE layer fires top-k experts per decode token, and today each expert runs
// mg_q4k_mlp_q6down in its OWN command buffer — k separate commit/waitUntilCompleted per layer, the
// ~360us launch/sync overhead the MAC-QWEN36 decode diagnosis named paid k times. This runs ALL k
// experts' gate->silu*up->down into ONE command buffer: k independent 3-stage chains, each on its own
// scratch SLICE. One commit/waitUntilCompleted for the whole layer. All k experts consume the SAME
// token activation x[H]; each writes its own y row into Ycat[k*H]. The Go caller applies the
// gate-weighted sum (kept on the host so the reduction order matches the per-expert loop exactly).
// gate_wids/up_wids are Q4_K wids; down_wids are Q6_K wids (>= MG_Q6_BASE), matching the q4_k_m
// residency. Returns 0 on success, -1 if any wid is out of range or a shape disagrees (caller falls
// back to the per-expert path). n is the expert count (top-k).

static id<MTLBuffer> gMlpGateK = nil, gMlpUpK = nil, gMlpInterK = nil; static long gMlpKCap = 0;
static id<MTLBuffer> gQYBufK = nil; static long gQYKCap = 0;

// q4k_grow_mlp_k grows the k-wide gate/up/inter scratch (n experts * I elements each), sized in
// TOTAL elements so a larger (n, I) reallocates once and is reused across decode tokens.
static void q4k_grow_mlp_k(long totalElems) {
    if (gMlpGateK != nil && gMlpKCap >= totalElems) return;
    gMlpGateK  = [gDev newBufferWithLength:(NSUInteger)(totalElems * 4) options:MTLResourceStorageModeShared];
    gMlpUpK    = [gDev newBufferWithLength:(NSUInteger)(totalElems * 4) options:MTLResourceStorageModeShared];
    gMlpInterK = [gDev newBufferWithLength:(NSUInteger)(totalElems * 4) options:MTLResourceStorageModeShared];
    gMlpKCap = totalElems;
}

static void q4k_grow_y_k(long totalElems) {
    if (gQYBufK != nil && gQYKCap >= totalElems) return;
    gQYBufK = [gDev newBufferWithLength:(NSUInteger)(totalElems * 4) options:MTLResourceStorageModeShared];
    gQYKCap = totalElems;
}

int mg_q4k_mlp_q6down_batch(const int* gate_wids, const int* up_wids, const int* down_wids,
                            int n, const float* x, float* Ycat, mg_execution_event* event) {
    mg_execution_event_reset(event);
    if (n <= 0 || gate_wids == NULL || up_wids == NULL || down_wids == NULL) return -1;
    // Validate every expert up front and confirm a uniform (H, I, Dout) across the batch (all
    // routed experts of a layer share the FFN geometry). A mismatch declines the whole batch.
    int H = -1, I = -1, Dout = -1;
    for (int e = 0; e < n; e++) {
        int gw = gate_wids[e], uw = up_wids[e], dw = down_wids[e];
        if (gw < 0 || uw < 0 || gw >= gNQ4 || uw >= gNQ4) return -1;
        if (dw < MG_Q6_BASE || (dw - MG_Q6_BASE) >= gNQ6) return -1;
        Q4KW G = gQ4[gw], U = gQ4[uw];
        Q6KW D = gQ6[dw - MG_Q6_BASE];
        if (G.in != U.in || G.out != U.out || D.in != G.out || D.out != G.in) return -1;
        if (e == 0) { H = G.in; I = G.out; Dout = D.out; }
        else if (G.in != H || G.out != I || D.out != Dout) return -1;
    }
    @autoreleasepool {
        q4k_grow_scratch((long)H, (long)Dout); // gQXBuf holds the shared x[H]
        q4k_grow_mlp_k((long)n * I);
        q4k_grow_y_k((long)n * Dout);
        id<MTLBuffer> xb = gQXBuf;
        memcpy(xb.contents, x, (size_t)H * 4);

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        mg_execution_event_command_buffer(event, cb);

        // Stage 1: for every expert, gate = G_e*x and up = U_e*x into its own I-wide slice
        // (offset e*I). One encoder holds all 2n GEMV dispatches; distinct output slices avoid a
        // false write-after-write hazard across experts.
        id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e1);
        [e1 setComputePipelineState:psoQ4KGemv];
        [e1 setBuffer:xb offset:0 atIndex:1];
        for (int e = 0; e < n; e++) {
            Q4KW G = gQ4[gate_wids[e]], U = gQ4[up_wids[e]];
            NSUInteger off = (NSUInteger)((long)e * I * 4);
            [e1 setBuffer:(__bridge id<MTLBuffer>)G.buf offset:G.offset atIndex:0];
            [e1 setBuffer:gMlpGateK offset:off atIndex:2];
            [e1 setBytes:&G.nblk length:sizeof(int) atIndex:3];
            [e1 setBytes:&G.out  length:sizeof(int) atIndex:4];
            [e1 dispatchThreadgroups:MTLSizeMake((NSUInteger)G.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [e1 setBuffer:(__bridge id<MTLBuffer>)U.buf offset:U.offset atIndex:0];
            [e1 setBuffer:gMlpUpK offset:off atIndex:2];
            [e1 setBytes:&U.nblk length:sizeof(int) atIndex:3];
            [e1 setBytes:&U.out  length:sizeof(int) atIndex:4];
            [e1 dispatchThreadgroups:MTLSizeMake((NSUInteger)U.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        }
        [e1 endEncoding];

        // Stage 2: inter_e = silu(gate_e) * up_e over each expert's I-wide slice.
        id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e2);
        [e2 setComputePipelineState:psoQ4KSwiGLU];
        for (int e = 0; e < n; e++) {
            NSUInteger off = (NSUInteger)((long)e * I * 4);
            [e2 setBuffer:gMlpGateK offset:off atIndex:0];
            [e2 setBuffer:gMlpUpK offset:off atIndex:1];
            [e2 setBuffer:gMlpInterK offset:off atIndex:2];
            [e2 setBytes:&I length:sizeof(int) atIndex:3];
            [e2 dispatchThreads:MTLSizeMake((NSUInteger)I,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        }
        [e2 endEncoding];

        // Stage 3: y_e = D_e * inter_e (Q6_K GEMV) into Ycat row e (offset e*Dout).
        id<MTLComputeCommandEncoder> e3 = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e3);
        [e3 setComputePipelineState:psoQ6KGemv];
        for (int e = 0; e < n; e++) {
            Q6KW D = gQ6[down_wids[e] - MG_Q6_BASE];
            NSUInteger interOff = (NSUInteger)((long)e * I * 4);
            NSUInteger yOff = (NSUInteger)((long)e * Dout * 4);
            [e3 setBuffer:(__bridge id<MTLBuffer>)D.buf offset:0 atIndex:0];
            [e3 setBuffer:gMlpInterK offset:interOff atIndex:1];
            [e3 setBuffer:gQYBufK offset:yOff atIndex:2];
            [e3 setBytes:&D.nblk length:sizeof(int) atIndex:3];
            [e3 setBytes:&D.out  length:sizeof(int) atIndex:4];
            [e3 dispatchThreadgroups:MTLSizeMake((NSUInteger)D.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        }
        [e3 endEncoding];

        [cb commit];
        mg_execution_event_committed(event);
        CFAbsoluteTime wait_started = CFAbsoluteTimeGetCurrent();
        [cb waitUntilCompleted];
        mg_execution_event_waited(event, cb, wait_started);
        memcpy(Ycat, gQYBufK.contents, (size_t)n * Dout * 4);
        mg_execution_event_readback(event);
    }
    return 0;
}

// mg_q4k_mlp_q6down is mg_q4k_mlp with a Q6_K down_proj: stages 1 (gate/up GEMV) and 2 (SwiGLU) are
// IDENTICAL — they run over the resident gMlpGate/gMlpUp/gMlpInter scratch — only stage 3 binds the
// Q6_K down weight (gQ6[down_wid-MG_Q6_BASE]) and the Q6_K GEMV pipeline. The whole MLP still runs in
// ONE command buffer. gate_wid/up_wid are Q4_K wids; down_wid is a Q6_K wid (>= MG_Q6_BASE).
void mg_q4k_mlp_q6down(int gate_wid, int up_wid, int down_wid, const float* x, float* y, mg_execution_event* event) {
    mg_execution_event_reset(event);
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
        mg_execution_event_command_buffer(event, cb);

        // (1) gate = G·x and up = U·x (independent), one encoder — IDENTICAL to mg_q4k_mlp.
        id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e1);
        [e1 setComputePipelineState:psoQ4KGemv];
        [e1 setBuffer:xb offset:0 atIndex:1];
        [e1 setBuffer:(__bridge id<MTLBuffer>)G.buf offset:G.offset atIndex:0];
        [e1 setBuffer:gMlpGate offset:0 atIndex:2];
        [e1 setBytes:&G.nblk length:sizeof(int) atIndex:3];
        [e1 setBytes:&G.out  length:sizeof(int) atIndex:4];
        [e1 dispatchThreadgroups:MTLSizeMake((NSUInteger)G.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e1 setBuffer:(__bridge id<MTLBuffer>)U.buf offset:U.offset atIndex:0];
        [e1 setBuffer:gMlpUp offset:0 atIndex:2];
        [e1 setBytes:&U.nblk length:sizeof(int) atIndex:3];
        [e1 setBytes:&U.out  length:sizeof(int) atIndex:4];
        [e1 dispatchThreadgroups:MTLSizeMake((NSUInteger)U.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e1 endEncoding];

        // (2) inter = silu(gate) * up — IDENTICAL to mg_q4k_mlp.
        id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e2);
        [e2 setComputePipelineState:psoQ4KSwiGLU];
        [e2 setBuffer:gMlpGate offset:0 atIndex:0];
        [e2 setBuffer:gMlpUp offset:0 atIndex:1];
        [e2 setBuffer:gMlpInter offset:0 atIndex:2];
        [e2 setBytes:&I length:sizeof(int) atIndex:3];
        [e2 dispatchThreads:MTLSizeMake((NSUInteger)I,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 endEncoding];

        // (3) y = D·inter with the Q6_K GEMV pipeline (the only line that differs from mg_q4k_mlp).
        id<MTLComputeCommandEncoder> e3 = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e3);
        [e3 setComputePipelineState:psoQ6KGemv];
        [e3 setBuffer:gMlpInter offset:0 atIndex:1];
        [e3 setBuffer:(__bridge id<MTLBuffer>)D.buf offset:0 atIndex:0];
        [e3 setBuffer:yb offset:0 atIndex:2];
        [e3 setBytes:&D.nblk length:sizeof(int) atIndex:3];
        [e3 setBytes:&D.out  length:sizeof(int) atIndex:4];
        [e3 dispatchThreadgroups:MTLSizeMake((NSUInteger)D.out,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e3 endEncoding];

        [cb commit];
        mg_execution_event_committed(event);
        CFAbsoluteTime wait_started = CFAbsoluteTimeGetCurrent();
        [cb waitUntilCompleted];
        mg_execution_event_waited(event, cb, wait_started);
        memcpy(y, yb.contents, (size_t)D.out * 4);
        mg_execution_event_readback(event);
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
int mg_q4k_upload_span(const unsigned char* raw, size_t nbytes, size_t offset, int out, int in) {
    if (raw == NULL || nbytes == 0 || offset > nbytes) return -1;
    int nblk = 0;
    long bytes = 0;
    if (q4k_upload_preflight(out, in, &nblk, &bytes) != 0 || (size_t)bytes > nbytes - offset) return -1;
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t page_offset = offset % page;
    size_t base_offset = offset - page_offset;
    size_t buf_len = (size_t)q4k_page_round((long)(bytes + (long)page_offset));
    if (base_offset + buf_len > nbytes) {
        buf_len = nbytes - base_offset;
    }
    void* buf_ptr = (void*)(raw + base_offset);
    id<MTLBuffer> b = [gDev newBufferWithBytesNoCopy:buf_ptr
                                              length:(NSUInteger)buf_len
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
    if (b == nil) return -1;
    int id = q4k_register_buffer(b, out, in, nblk);
    if (id >= 0) gQ4[id].offset = (NSUInteger)page_offset;
    return id;
}
int mg_q4k_upload_nocopy(const unsigned char* raw, int out, int in) {
    if (raw == NULL) return -1;
    int nblk = 0;
    long bytes = 0;
    if (q4k_upload_preflight(out, in, &nblk, &bytes) != 0 || gNQ4 >= MG_MAX_Q4) return -1;
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
    if (q4k_upload_preflight(out, in, &nblk, &bytes) != 0 || gNQ4 >= MG_MAX_Q4) return -1;
    id<MTLBuffer> b = [gDev newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
    if (b == nil) {
        NSLog(@"mg_q4k_upload: device buffer alloc failed for %.1f MB", (double)bytes / 1e6);
        return -1;
    }
    memcpy(b.contents, raw, (size_t)bytes);
    return q4k_register_buffer(b, out, in, nblk);
}

// mg_q4k_gemv computes y[out] = W[wid] · x (one f32 activation row, length in). It returns
// 1 for scalar execution, 2 for vectorized execution, and 0 when no dispatch occurred.
int mg_q4k_gemv(int wid, const float* x, float* y, int vectorized_mode, mg_execution_event* event) {
    mg_execution_event_reset(event);
    if (wid < 0 || wid >= gNQ4) return 0;
    @autoreleasepool {
        int executed = 0;
        id<MTLComputePipelineState> pso = q4k_gemv_pso(vectorized_mode, &executed);
        if (pso == nil) return 0;
        Q4KW W = gQ4[wid];
        q4k_grow_scratch(W.in, W.out);
        id<MTLBuffer> wbuf = (__bridge id<MTLBuffer>)W.buf;
        id<MTLBuffer> xb = gQXBuf;
        id<MTLBuffer> yb = gQYBuf;
        memcpy(xb.contents, x, (size_t)W.in * 4);

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        mg_execution_event_command_buffer(event, cb);
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e);
        [e setComputePipelineState:pso];
        [e setBuffer:wbuf offset:W.offset atIndex:0];
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
        mg_execution_event_committed(event);
        CFAbsoluteTime wait_started = CFAbsoluteTimeGetCurrent();
        [cb waitUntilCompleted];
        mg_execution_event_waited(event, cb, wait_started);

        memcpy(y, yb.contents, (size_t)W.out * 4);
        mg_execution_event_readback(event);
    return executed;
    }
}

// mg_issue8833_q4k_encode_gemv appends the established scalar Q4_K GEMV to a caller-owned
// command buffer and caller-owned shared buffers. The mixed-QKV owner is solely responsible for
// commit, completion wait, and host readback. Keep the registered weight offset: mapped GGUF spans
// may begin after the Metal buffer base.
int mg_issue8833_q4k_encode_gemv(void* command, int wid, void* x, void* y) {
    if (command == NULL || x == NULL || y == NULL || psoQ4KGemv == nil ||
        wid < 0 || wid >= gNQ4 || gQ4[wid].buf == NULL) return 0;
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
    Q4KW W = gQ4[wid];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    if (e == nil) return 0;
    [e setComputePipelineState:psoQ4KGemv];
    [e setBuffer:(__bridge id<MTLBuffer>)W.buf offset:W.offset atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)y offset:0 atIndex:2];
    [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
    [e setBytes:&W.out length:sizeof(int) atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding];
    return 1;
}

// mg_q4k_gemv_batch runs n decode GEMVs of the SAME weight wid into ONE command buffer (one
// commit + one waitUntilCompleted): Xcat is n contiguous activation rows (n*in floats), Ycat
// receives n result rows (n*out floats). It exists to MEASURE how much of mg_q4k_gemv's
// per-call cost is the CPU<->GPU submission/sync round-trip vs the kernel: if n GEMVs here cost
// ~n*kernel + one round-trip (not n round-trips), the decode wall is the per-op command buffer,
// and the fix is a one-command-buffer resident forward (issue #67). The encoder re-binds only
// the X/Y offsets between dispatches; the weight + dims are set once.
void mg_q4k_gemv_batch(int wid, const float* Xcat, int n, float* Ycat, mg_execution_event* event) {
    mg_execution_event_reset(event);
    if (wid < 0 || wid >= gNQ4 || n <= 0) return;
    @autoreleasepool {
        Q4KW W = gQ4[wid];
        q4k_grow_scratch((long)n * W.in, (long)n * W.out);
        id<MTLBuffer> wbuf = (__bridge id<MTLBuffer>)W.buf;
        id<MTLBuffer> xb = gQXBuf;
        id<MTLBuffer> yb = gQYBuf;
        memcpy(xb.contents, Xcat, (size_t)n * W.in * 4);

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        mg_execution_event_command_buffer(event, cb);
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e);
        [e setComputePipelineState:psoQ4KGemv];
        [e setBuffer:wbuf offset:W.offset atIndex:0];
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
        mg_execution_event_committed(event);
        CFAbsoluteTime wait_started = CFAbsoluteTimeGetCurrent();
        [cb waitUntilCompleted];
        mg_execution_event_waited(event, cb, wait_started);

        memcpy(Ycat, yb.contents, (size_t)n * W.out * 4);
        mg_execution_event_readback(event);
    }
}

// mg_q4k_gemv_batch_multi applies one Q4_K weight to 4-8 activation rows in a single dispatch.
// q4k_gemv_multi owns the tile-reuse contract; the host side only copies the panel and binds it.
void mg_q4k_gemv_batch_multi(int wid, const float* Xcat, int n, float* Ycat, mg_execution_event* event) {
    mg_execution_event_reset(event);
    if (wid < 0 || wid >= gNQ4 || n < 4 || n > 8) return;
    @autoreleasepool {
        Q4KW W = gQ4[wid];
        q4k_grow_scratch((long)n * W.in, (long)n * W.out);
        id<MTLBuffer> wbuf = (__bridge id<MTLBuffer>)W.buf;
        id<MTLBuffer> xb = gQXBuf;
        id<MTLBuffer> yb = gQYBuf;
        memcpy(xb.contents, Xcat, (size_t)n * W.in * 4);

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        mg_execution_event_command_buffer(event, cb);
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e);
        [e setComputePipelineState:psoQ4KGemvMulti[n - 4]];
        [e setBuffer:wbuf offset:W.offset atIndex:0];
        [e setBuffer:xb   offset:0 atIndex:1];
        [e setBuffer:yb   offset:0 atIndex:2];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(W.out + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
        [cb commit];
        mg_execution_event_committed(event);
        CFAbsoluteTime wait_started = CFAbsoluteTimeGetCurrent();
        [cb waitUntilCompleted];
        mg_execution_event_waited(event, cb, wait_started);

        memcpy(Ycat, yb.contents, (size_t)n * W.out * 4);
        mg_execution_event_readback(event);
    }
}

// mg_q4k_gemv_group runs n decode GEMVs that SHARE one activation x (length in) but apply n
// DIFFERENT resident q4_k weights, into ONE command buffer (one commit/waitUntilCompleted). This
// is the live decode access pattern: a layer's q/k/v (or gate/up, or the GDN in_proj quad) all
// read the same post-norm activation. Each weight i writes Ycat[yoff[i] .. yoff[i]+out_i); yoff
// has n+1 entries (yoff[n] = total y elems). The fixed ~submit/sync overhead is paid ONCE for the
// group and the GPU pipelines the n dispatches — the per-token win the resident forward needs.
void mg_q4k_gemv_group(const int* wids, int n, const float* x, float* Ycat, const int* yoff, mg_execution_event* event) {
    mg_execution_event_reset(event);
    if (n <= 0) return;
    @autoreleasepool {
        int in = gQ4[wids[0]].in;
        long ytot = (long)yoff[n];
        q4k_grow_scratch((long)in, ytot);
        id<MTLBuffer> xb = gQXBuf;
        id<MTLBuffer> yb = gQYBuf;
        memcpy(xb.contents, x, (size_t)in * 4);

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        mg_execution_event_command_buffer(event, cb);
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e);
        [e setComputePipelineState:psoQ4KGemv];
        [e setBuffer:xb offset:0 atIndex:1]; // shared activation for every weight in the group
        for (int i = 0; i < n; i++) {
            Q4KW Wi = gQ4[wids[i]];
            [e setBuffer:(__bridge id<MTLBuffer>)Wi.buf offset:Wi.offset atIndex:0];
            [e setBuffer:yb offset:(NSUInteger)((long)yoff[i] * 4) atIndex:2];
            [e setBytes:&Wi.nblk length:sizeof(int) atIndex:3];
            [e setBytes:&Wi.out  length:sizeof(int) atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)Wi.out, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        [e endEncoding];
        [cb commit];
        mg_execution_event_committed(event);
        CFAbsoluteTime wait_started = CFAbsoluteTimeGetCurrent();
        [cb waitUntilCompleted];
        mg_execution_event_waited(event, cb, wait_started);

        memcpy(Ycat, yb.contents, (size_t)ytot * 4);
        mg_execution_event_readback(event);
    }
}

// mg_q4k_q8_gemv_group is the mixed full-attention projection spine: Q/K Q8 and V Q4_K
// share one caller-owned command buffer. Return 0 only before a command buffer exists, 1 after a
// completed submission, and -1 after a submitted command buffer fails.
int mg_q4k_q8_gemv_group(const int* q4_wids, int nq4, const float* x, float* q4_y, const int* q4_yoff,
                         const int* q8_wids, int nq8, const signed char* xq, const float* xd,
                         float* q8_y, const int* q8_yoff, int inject_post_submit_failure,
                         mg_execution_event* event) {
    mg_execution_event_reset(event);
    if (nq4 <= 0 || nq8 <= 0 || q4_wids == NULL || q8_wids == NULL ||
        x == NULL || xq == NULL || xd == NULL || q4_y == NULL || q8_y == NULL ||
        q4_yoff == NULL || q8_yoff == NULL) return 0;
    for (int i = 0; i < nq4; i++) {
        if (q4_wids[i] < 0 || q4_wids[i] >= gNQ4) return 0;
    }
    int in = gQ4[q4_wids[0]].in;
    for (int i = 1; i < nq4; i++) {
        if (gQ4[q4_wids[i]].in != in) return 0;
    }
    if (mg_q8_prepare_gemv_group(q8_wids, nq8, xq, xd, q8_yoff) == 0) return 0;
    long q4total = (long)q4_yoff[nq4];
    q4k_grow_scratch((long)in, q4total);
    if (gQXBuf == nil || gQYBuf == nil) return 0;
    memcpy(gQXBuf.contents, x, (size_t)in * 4);

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        if (cb == nil) return 0;
        mg_execution_event_command_buffer(event, cb);

        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        if (e == nil) return -1;
        mg_execution_event_encoder(event, e);
        [e setComputePipelineState:psoQ4KGemv];
        [e setBuffer:gQXBuf offset:0 atIndex:1];
        for (int i = 0; i < nq4; i++) {
            Q4KW W = gQ4[q4_wids[i]];
            [e setBuffer:(__bridge id<MTLBuffer>)W.buf offset:W.offset atIndex:0];
            [e setBuffer:gQYBuf offset:(NSUInteger)((long)q4_yoff[i] * 4) atIndex:2];
            [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
            [e setBytes:&W.out length:sizeof(int) atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)W.out, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        [e endEncoding];
        int q8_encoders = mg_q8_encode_gemv_group((__bridge void*)cb, q8_wids, nq8, q8_yoff);
        if (q8_encoders <= 0) {
            return -1; // candidate encoding began; fail closed rather than falling back mid-batch.
        }
		if (event != NULL) event->encoders += q8_encoders;
        [cb commit];
        mg_execution_event_committed(event);
        CFAbsoluteTime wait_started = CFAbsoluteTimeGetCurrent();
        [cb waitUntilCompleted];
        mg_execution_event_waited(event, cb, wait_started);
        if (cb.status != MTLCommandBufferStatusCompleted) return -1;
        // The caller-scoped test injection is checked only after a real native submit and wait.
        // It proves that this function's post-submit return travels through the exported Go call
        // path as MixedQ4KQ8PostSubmitError without corrupting process-global Metal state.
        if (inject_post_submit_failure != 0) return -1;

        memcpy(q4_y, gQYBuf.contents, (size_t)q4total * 4);
        mg_q8_read_gemv_group(q8_y, q8_yoff[nq8]);
        mg_execution_event_readback(event);
        return 1;
    }
}

// mg_q4k_gemm computes Y[P, out] = X[P, in] · W[wid]^T and returns the exact executed identity:
// 0=not executed, 1=scalar q4k_gemm, 2=exact-P32 q4k_gemm_mm32. Candidate selection happens before
// scratch allocation or host copies, so an unavailable explicit MM32 request cannot dispatch or
// mutate Y. out_gpu_ms is nullable and is written only after a completed dispatch.
int mg_q4k_gemm(int wid, const float* X, int P, float* Y, int mm_mode, double* out_gpu_ms, mg_execution_event* event) {
    mg_execution_event_reset(event);
    if (wid < 0 || wid >= gNQ4 || P <= 0) return 0;
    @autoreleasepool {
        Q4KW W = gQ4[wid];
        int executed = 0;
        int BN = 64;
        id<MTLComputePipelineState> pso = q4k_gemm_pso(P, mm_mode, &executed, &BN);
        if (pso == nil) return 0;
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
        const int TG = 256; // threads per threadgroup (TGX*TGY); must match Q4K_TG in the MSL source
        int rowBlocks = (W.out + BM - 1) / BM;
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        mg_execution_event_command_buffer(event, cb);
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e);
        [e setComputePipelineState:pso];
        [e setBuffer:wbuf offset:W.offset atIndex:0];
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
        mg_execution_event_committed(event);
        CFAbsoluteTime wait_started = CFAbsoluteTimeGetCurrent();
        [cb waitUntilCompleted];
        mg_execution_event_waited(event, cb, wait_started);
        // GPUStartTime/GPUEndTime are valid only after waitUntilCompleted returns (already-completed
        // cb; reading them is cheap). This is the on-GPU execution window, excluding the CPU-side
        // encode/commit/sync/H2D that dominates the q4k_metal prefill wall we are trying to split.
        if (out_gpu_ms) *out_gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;

        memcpy(Y, yb.contents, (size_t)P * W.out * 4);
        mg_execution_event_readback(event);
        return executed;
    }
}

// mg_q4k_gemm_group runs n batched prefill GEMMs that SHARE one activation panel X[P, in] but apply
// n DIFFERENT resident q4_k weights, into ONE command buffer (one commit/waitUntilCompleted). It is
// the prefill twin of mg_q4k_gemv_group: a layer's q/k/v (or gate/up, or the GDN in_proj quad) all
// read the same post-norm activation panel, so the fixed ~submit/sync overhead is paid ONCE for the
// group and the GPU pipelines the n GEMMs — the prefill-wall lever (~7 per-weight submits per layer
// collapse to ~2-3). Each weight i writes its own [P, out_i] token-major block into Ycat at element
// offset yoff[i] (= P*Σ_{j<i} out_j; yoff[n] = total y elems). Every weight must share X's `in`.
// out_gpu_ms is nullable: when non-NULL it receives the whole group's on-GPU execution window
// (cb.GPUEndTime - cb.GPUStartTime, in ms), valid after waitUntilCompleted returns. NULL is inert.
int mg_q4k_gemm_group(const int* wids, int n, const float* X, int P, float* Ycat, const int* yoff,
                      int mm_mode, double* out_gpu_ms, mg_execution_event* event) {
    mg_execution_event_reset(event);
    if (n <= 0 || P <= 0) return 0;
    @autoreleasepool {
        int executed = 0;
        int BN = 64;
        id<MTLComputePipelineState> pso = q4k_gemm_pso(P, mm_mode, &executed, &BN);
        if (pso == nil) return 0;
        int in = gQ4[wids[0]].in;
        long ytot = (long)yoff[n];
        q4k_grow_scratch((long)P * in, ytot);
        id<MTLBuffer> xb = gQXBuf;
        id<MTLBuffer> yb = gQYBuf;
        memcpy(xb.contents, X, (size_t)P * in * 4); // shared activation panel for every group member

        // 2D tile identical to mg_q4k_gemm: BM output rows × BN token-tile per threadgroup, all in
        // ONE command buffer. The BN token loop is issued per weight; the grid's row axis is the
        // weight's own out. Every dispatch reads the shared xb and writes into the weight's Y slot.
        const int BM = 64;  // must match Q4K_BM in the MSL source
        const int TG = 256; // must match Q4K_TG (TGX*TGY) in the MSL source
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        mg_execution_event_command_buffer(event, cb);
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e);
        [e setComputePipelineState:pso];
        [e setBuffer:xb offset:0 atIndex:1]; // shared X for the whole group
        [e setBytes:&P length:sizeof(int) atIndex:5];
        for (int i = 0; i < n; i++) {
            Q4KW Wi = gQ4[wids[i]];
            int rowBlocks = (Wi.out + BM - 1) / BM;
            [e setBuffer:(__bridge id<MTLBuffer>)Wi.buf offset:Wi.offset atIndex:0];
            [e setBuffer:yb offset:(NSUInteger)((long)yoff[i] * 4) atIndex:2];
            [e setBytes:&Wi.nblk length:sizeof(int) atIndex:3];
            [e setBytes:&Wi.out  length:sizeof(int) atIndex:4];
            for (int t0 = 0; t0 < P; t0 += BN) {
                int nt = P - t0;
                if (nt > BN) nt = BN;
                [e setBytes:&t0 length:sizeof(int) atIndex:6];
                [e setBytes:&nt length:sizeof(int) atIndex:7];
                [e dispatchThreadgroups:MTLSizeMake((NSUInteger)rowBlocks, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)TG, 1, 1)];
            }
        }
        [e endEncoding];
        [cb commit];
        mg_execution_event_committed(event);
        CFAbsoluteTime wait_started = CFAbsoluteTimeGetCurrent();
        [cb waitUntilCompleted];
        mg_execution_event_waited(event, cb, wait_started);
        // On-GPU execution window for the whole group (valid post-wait; excludes CPU encode/commit/
        // sync/H2D). Lets the model side split its wall-timed q4kTime into gpu_compute vs roundtrip.
        if (out_gpu_ms) *out_gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;

        memcpy(Ycat, yb.contents, (size_t)ytot * 4);
        mg_execution_event_readback(event);
        return executed;
    }
}

// mg_q4k_release releases one Q4_K buffer. Trailing tombstones collapse so transient
// upload/execute/release traffic reuses the table instead of exhausting MG_MAX_Q4.
void mg_q4k_release(int wid) {
    if (wid < 0 || wid >= gNQ4) return;
    if (gQ4[wid].buf != NULL) {
        CFBridgingRelease(gQ4[wid].buf);
        gQ4[wid].buf = NULL;
    }
    gQ4[wid].out = 0;
    gQ4[wid].in = 0;
    gQ4[wid].nblk = 0;
    while (gNQ4 > 0 && gQ4[gNQ4 - 1].buf == NULL) gNQ4--;
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

// ---- caller-owned quantized projection graph (#9267) ---------------------------
typedef struct {
    id<MTLCommandBuffer> cb;
    id<MTLBuffer> xf, xq, xd;
    NSMutableArray *results;
    int P, in, encoders, committed, readbacks, buffers;
    double gpu_ms, wait_ms;
} MGProjectionGraph;

static _Atomic int gGraphLiveOwners;
static _Atomic int gGraphLiveBuffers;

static void mg_graph_track_buffer(MGProjectionGraph *g, id<MTLBuffer> buffer) {
    if (!g || !buffer) return;
    g->buffers++;
    atomic_fetch_add_explicit(&gGraphLiveBuffers, 1, memory_order_relaxed);
}

static void mg_graph_release_tracked_buffers(MGProjectionGraph *g) {
    if (!g) return;
    int buffers=g->buffers;
    g->xf=nil;g->xq=nil;g->xd=nil;g->results=nil;g->buffers=0;
    if (buffers) atomic_fetch_sub_explicit(&gGraphLiveBuffers, buffers, memory_order_relaxed);
}

typedef struct {
    int committed, completed_wait, encoders, host_readbacks;
    double gpu_milliseconds, wait_milliseconds;
    int timing_available;
} mg_graph_receipt;

void *mg_graph_begin(const float *xf, const signed char *xq, const float *xd, int P, int in) {
    if (!q4k_init() || P <= 0 || in <= 0) return NULL;
    MGProjectionGraph *g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->P=P; g->in=in; g->cb=[gQueue commandBuffer]; g->results=[NSMutableArray array];
    if (!g->cb || !g->results) { free(g); return NULL; }
    NSUInteger nf=(NSUInteger)P*(NSUInteger)in;
    if (xf) { g->xf=[gDev newBufferWithLength:nf*sizeof(float) options:MTLResourceStorageModeShared];mg_graph_track_buffer(g,g->xf); }
    if (xq) { g->xq=[gDev newBufferWithLength:nf options:MTLResourceStorageModeShared];mg_graph_track_buffer(g,g->xq); }
    if (xd) { NSUInteger ns=(NSUInteger)P*(NSUInteger)(in/32);g->xd=[gDev newBufferWithLength:ns*sizeof(float) options:MTLResourceStorageModeShared];mg_graph_track_buffer(g,g->xd); }
    if ((xf&&!g->xf)||(xq&&!g->xq)||(xd&&!g->xd)) { g->cb=nil;mg_graph_release_tracked_buffers(g);free(g);return NULL; }
    if (xf) { memcpy([g->xf contents],xf,nf*sizeof(float));[g->results addObject:g->xf]; }
    if (xq) memcpy([g->xq contents],xq,nf);
    if (xd) { NSUInteger ns=(NSUInteger)P*(NSUInteger)(in/32);memcpy([g->xd contents],xd,ns*sizeof(float)); }
    atomic_fetch_add_explicit(&gGraphLiveOwners, 1, memory_order_relaxed);
    return g;
}

static void *mg_graph_result(MGProjectionGraph *g, NSUInteger n) {
    id<MTLBuffer> y=[gDev newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
    if (!y) return NULL; [g->results addObject:y];mg_graph_track_buffer(g,y);g->encoders++;return (__bridge void*)y;
}

void *mg_graph_quantize_q8(void *opaque, void *input, int elems, void **scales) {
    MGProjectionGraph *g=opaque; id<MTLBuffer>x=(__bridge id<MTLBuffer>)input;
    if(scales)*scales=NULL;
    if(!g||g->committed||!x||elems<=0||elems%32||![g->results containsObject:x]||!psoGraphQuantizeQ8)return NULL;
    int blocks=elems/32;
    id<MTLBuffer>q=[gDev newBufferWithLength:(NSUInteger)elems options:MTLResourceStorageModeShared];
    id<MTLBuffer>d=[gDev newBufferWithLength:(NSUInteger)blocks*sizeof(float) options:MTLResourceStorageModeShared];
    if(!q||!d)return NULL;
    [g->results addObject:q];[g->results addObject:d];mg_graph_track_buffer(g,q);mg_graph_track_buffer(g,d);
    id<MTLComputeCommandEncoder>e=[g->cb computeCommandEncoder];
    [e setComputePipelineState:psoGraphQuantizeQ8];[e setBuffer:x offset:0 atIndex:0];[e setBuffer:q offset:0 atIndex:1];[e setBuffer:d offset:0 atIndex:2];[e setBytes:&blocks length:sizeof(blocks) atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)blocks,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];
    g->encoders++;if(scales)*scales=(__bridge void*)d;return (__bridge void*)q;
}

void *mg_graph_encode_q4k(void *opaque, int wid) {
    MGProjectionGraph *g=opaque; if (!g || g->committed || !g->xf || wid<0 || wid>=gNQ4 || gQ4[wid].in!=g->in) return NULL;
    Q4KW *w=&gQ4[wid]; id<MTLBuffer> y=(__bridge id<MTLBuffer>)mg_graph_result(g,(NSUInteger)g->P*(NSUInteger)w->out); if(!y)return NULL;
    int executed=0, BN=64; id<MTLComputePipelineState> pso=q4k_gemm_pso(g->P,0,&executed,&BN); if(!pso)return NULL;
    const int BM=64,TG=256; int rowBlocks=(w->out+BM-1)/BM;
    id<MTLComputeCommandEncoder> e=[g->cb computeCommandEncoder]; [e setComputePipelineState:pso]; [e setBuffer:(__bridge id<MTLBuffer>)w->buf offset:w->offset atIndex:0]; [e setBuffer:g->xf offset:0 atIndex:1]; [e setBuffer:y offset:0 atIndex:2]; [e setBytes:&w->nblk length:sizeof(int) atIndex:3];[e setBytes:&w->out length:sizeof(int) atIndex:4];[e setBytes:&g->P length:sizeof(int) atIndex:5]; for(int t0=0;t0<g->P;t0+=BN){int nt=g->P-t0;if(nt>BN)nt=BN;[e setBytes:&t0 length:sizeof(int) atIndex:6];[e setBytes:&nt length:sizeof(int) atIndex:7];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)rowBlocks,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)TG,1,1)];}[e endEncoding]; return (__bridge void*)y;
}
void *mg_graph_encode_q4k_from(void *opaque,int wid,void*input,int elems) {
    MGProjectionGraph*g=opaque;id<MTLBuffer>x=(__bridge id<MTLBuffer>)input;if(!g||g->committed||!x||wid<0||wid>=gNQ4||gQ4[wid].in*g->P!=elems||![g->results containsObject:x])return NULL;
    Q4KW*w=&gQ4[wid];id<MTLBuffer>y=(__bridge id<MTLBuffer>)mg_graph_result(g,(NSUInteger)g->P*(NSUInteger)w->out);if(!y)return NULL;
    int executed=0,BN=64;id<MTLComputePipelineState>pso=q4k_gemm_pso(g->P,0,&executed,&BN);if(!pso)return NULL;const int BM=64,TG=256;int rowBlocks=(w->out+BM-1)/BM;
    id<MTLComputeCommandEncoder>e=[g->cb computeCommandEncoder];[e setComputePipelineState:pso];[e setBuffer:(__bridge id<MTLBuffer>)w->buf offset:w->offset atIndex:0];[e setBuffer:x offset:0 atIndex:1];[e setBuffer:y offset:0 atIndex:2];[e setBytes:&w->nblk length:sizeof(int) atIndex:3];[e setBytes:&w->out length:sizeof(int) atIndex:4];[e setBytes:&g->P length:sizeof(int) atIndex:5];for(int t0=0;t0<g->P;t0+=BN){int nt=g->P-t0;if(nt>BN)nt=BN;[e setBytes:&t0 length:sizeof(int) atIndex:6];[e setBytes:&nt length:sizeof(int) atIndex:7];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)rowBlocks,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)TG,1,1)];}[e endEncoding];return (__bridge void*)y;
}
void *mg_graph_encode_q6k(void *opaque, int wid) {
    MGProjectionGraph *g=opaque; int i=wid-MG_Q6_BASE; if(!g||g->committed||!g->xf||i<0||i>=gNQ6||gQ6[i].in!=g->in)return NULL; Q6KW *w=&gQ6[i]; id<MTLBuffer> y=(__bridge id<MTLBuffer>)mg_graph_result(g,(NSUInteger)g->P*(NSUInteger)w->out);if(!y)return NULL; id<MTLComputeCommandEncoder>e=[g->cb computeCommandEncoder];[e setComputePipelineState:psoQ6KGemm];[e setBuffer:(__bridge id<MTLBuffer>)w->buf offset:0 atIndex:0];[e setBuffer:g->xf offset:0 atIndex:1];[e setBuffer:y offset:0 atIndex:2];[e setBytes:&w->nblk length:sizeof(int) atIndex:3];[e setBytes:&w->out length:sizeof(int) atIndex:4];[e setBytes:&g->P length:sizeof(int) atIndex:5];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)w->out,(NSUInteger)g->P,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];return (__bridge void*)y;
}
void *mg_graph_encode_q6k_from(void *opaque,int wid,void*input,int elems) {
    MGProjectionGraph*g=opaque;int i=wid-MG_Q6_BASE;id<MTLBuffer>x=(__bridge id<MTLBuffer>)input;if(!g||g->committed||!x||i<0||i>=gNQ6||gQ6[i].in*g->P!=elems||![g->results containsObject:x])return NULL;Q6KW*w=&gQ6[i];id<MTLBuffer>y=(__bridge id<MTLBuffer>)mg_graph_result(g,(NSUInteger)g->P*(NSUInteger)w->out);if(!y)return NULL;id<MTLComputeCommandEncoder>e=[g->cb computeCommandEncoder];[e setComputePipelineState:psoQ6KGemm];[e setBuffer:(__bridge id<MTLBuffer>)w->buf offset:0 atIndex:0];[e setBuffer:x offset:0 atIndex:1];[e setBuffer:y offset:0 atIndex:2];[e setBytes:&w->nblk length:sizeof(int) atIndex:3];[e setBytes:&w->out length:sizeof(int) atIndex:4];[e setBytes:&g->P length:sizeof(int) atIndex:5];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)w->out,(NSUInteger)g->P,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];return (__bridge void*)y;
}
extern void *mg_q8_graph_encode(void *graph, int wid);
extern void *mg_q8_graph_encode_from(void *graph, int wid, void *q, void *d, int elems);
void *mg_graph_encode_q8(void *opaque,int wid){return mg_q8_graph_encode(opaque,wid);}
void *mg_graph_encode_q8_from(void *opaque,int wid,void*q,void*d,int elems){return mg_q8_graph_encode_from(opaque,wid,q,d,elems);}
int mg_graph_finish(void *opaque,mg_graph_receipt*r,int inject_post_submit_failure){MGProjectionGraph*g=opaque;if(r)memset(r,0,sizeof(*r));if(!g||g->committed||g->encoders==0)return 0;g->committed=1;CFAbsoluteTime t=CFAbsoluteTimeGetCurrent();[g->cb commit];[g->cb waitUntilCompleted];g->wait_ms=(CFAbsoluteTimeGetCurrent()-t)*1000.;if(r){r->committed=1;r->completed_wait=g->cb.status==MTLCommandBufferStatusCompleted;r->encoders=g->encoders;r->host_readbacks=g->readbacks;r->wait_milliseconds=g->wait_ms;if(@available(macOS 10.15,*)){double a=g->cb.GPUStartTime,b=g->cb.GPUEndTime;if(b>=a&&a>0){r->gpu_milliseconds=(b-a)*1000.;r->timing_available=1;}}}return g->cb.status==MTLCommandBufferStatusCompleted&&!inject_post_submit_failure;}
int mg_graph_read(void*opaque,void*result,float*dst,int n){MGProjectionGraph*g=opaque;id<MTLBuffer>y=(__bridge id<MTLBuffer>)result;if(!g||!g->committed||!y||!dst||n<0||![g->results containsObject:y])return 0;memcpy(dst,[y contents],(NSUInteger)n*sizeof(float));g->readbacks++;return 1;}
int mg_graph_read_pack(void*opaque,void**results,const int*sizes,int count,float*dst,int total){MGProjectionGraph*g=opaque;if(!g||!g->committed||!results||!sizes||count<=0||!dst||total<0)return 0;int off=0;for(int i=0;i<count;i++){id<MTLBuffer>y=(__bridge id<MTLBuffer>)results[i];int n=sizes[i];if(!y||n<0||off>total-n||![g->results containsObject:y])return 0;memcpy(dst+off,[y contents],(NSUInteger)n*sizeof(float));off+=n;}if(off!=total)return 0;g->readbacks++;return 1;}
void mg_graph_free(void*opaque){MGProjectionGraph*g=opaque;if(!g)return;g->cb=nil;mg_graph_release_tracked_buffers(g);atomic_fetch_sub_explicit(&gGraphLiveOwners,1,memory_order_relaxed);free(g);}

int mg_graph_live_owners(void){return atomic_load_explicit(&gGraphLiveOwners,memory_order_relaxed);}
int mg_graph_live_buffers(void){return atomic_load_explicit(&gGraphLiveBuffers,memory_order_relaxed);}

void *mg_graph_command_buffer(void *opaque){MGProjectionGraph*g=opaque;return g?(__bridge void*)g->cb:NULL;}
void *mg_graph_xf_buffer(void *opaque){MGProjectionGraph*g=opaque;return g?(__bridge void*)g->xf:NULL;}
void *mg_graph_xq_buffer(void *opaque){MGProjectionGraph*g=opaque;return g?(__bridge void*)g->xq:NULL;}
void *mg_graph_xd_buffer(void *opaque){MGProjectionGraph*g=opaque;return g?(__bridge void*)g->xd:NULL;}
int mg_graph_prompt(void *opaque){MGProjectionGraph*g=opaque;return g?g->P:0;}
int mg_graph_input(void *opaque){MGProjectionGraph*g=opaque;return g?g->in:0;}
void *mg_graph_alloc_result(void *opaque,int n){MGProjectionGraph*g=opaque;return g?mg_graph_result(g,(NSUInteger)n):NULL;}
void *mg_graph_alloc_buffer(void *opaque,int n){MGProjectionGraph*g=opaque;if(!g||g->committed||n<=0)return NULL;id<MTLBuffer>b=[gDev newBufferWithLength:(NSUInteger)n*sizeof(float) options:MTLResourceStorageModeShared];if(!b)return NULL;[g->results addObject:b];mg_graph_track_buffer(g,b);return (__bridge void*)b;}
void mg_graph_note_encoder(void *opaque){MGProjectionGraph*g=opaque;if(g&&!g->committed)g->encoders++;}
