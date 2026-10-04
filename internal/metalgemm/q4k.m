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
#include <dispatch/dispatch.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

// MG_Q4K_WAIT_LIMIT_MS bounds every Q4_K command-buffer wait. The old path called the
// uninterruptible, unbounded `[cb waitUntilCompleted]`, so a stalled GPU (submit stuck in
// IOGPUMetalCommandQueue _submitCommandBuffers) hung the served request forever. A
// dispatch_semaphore completed-handler wait turns that into a bounded, reportable timeout.
// Kept in parity with the Go-side classification budget, metal_stall.go
// DefaultCommandBufferWaitLimit (10_000ms); the two must move together or the native timeout
// and the Go classification diverge.
static const int64_t MG_Q4K_WAIT_LIMIT_MS = 10000;

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

// mg_q4k_commit_bounded is the single commit+wait used by every Q4_K path. It registers the
// completion signal BEFORE commit (Metal requires the handler be attached to the uncommitted
// buffer), commits, then waits at most MG_Q4K_WAIT_LIMIT_MS. Returns 1 when the buffer reached
// MTLCommandBufferStatusCompleted, 0 when the bounded wait timed out. On timeout it records
// completed_wait=0 in the event and does NOT call waitUntilCompleted to "bound" teardown - that
// would re-introduce the unbounded block - so the caller must skip an unwritten output readback.
static int mg_q4k_commit_bounded(id<MTLCommandBuffer> cb, mg_execution_event* event) {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b){ (void)b; dispatch_semaphore_signal(sem); }];
    [cb commit];
    mg_execution_event_committed(event);
    CFAbsoluteTime wait_started = CFAbsoluteTimeGetCurrent();
    long timed_out = dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(MG_Q4K_WAIT_LIMIT_MS * NSEC_PER_MSEC)));
    mg_execution_event_waited(event, cb, wait_started);
    if (timed_out != 0) {
        if (event != NULL) event->completed_wait = 0;
        return 0;
    }
    return 1;
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

// q4k_mul_mv is the default P=1 Q4_K decode GEMV (fak#13599). It is a port of llama.cpp's
// kernel_mul_mv_q4_K_f32_impl in ggml-metal.metal at 17197474510622a3b4ea7d0909d70b606f542b96
// (MIT; Copyright (c) 2023-2026 The ggml authors; the copyright and permission notice are
// retained by reference in q4k_hotpath.go's provenance block). Changes from upstream: the
// broadcast/batch strides are dropped (P=1 only), fak's row-major 144-B block layout is
// addressed through raw byte offsets, and the per-row loop is guarded so a tail simdgroup never
// reads or writes past `out` (upstream relies on padded rows).
//
// Geometry: Q4K_MV_NSG simdgroups per threadgroup, Q4K_MV_NR0 output rows per simdgroup. Inside
// a simdgroup, lane group ix = lane/8 walks blocks ix, ix+4, ... and each of its 8 lanes owns 32
// weights of a block: iq = (lane%8)/4 selects the 64-weight chunk pair (iq, iq+2), ir = lane%4
// the 8-byte slice inside each chunk. So all 32 lanes work even at nblk=14 (in=3584), qs loads
// are two contiguous 8-byte reads per lane, and the activation slice is cached in registers once
// per block and reused for all Q4K_MV_NR0 rows.
//
// Index contract against q4k_block_dot (byte layout d@0, dmin@2, scales@4..15, qs@16..143):
//   q1 = ushort qs[16iq+4ir .. +3] = bytes 32iq+8ir..+7 of chunk iq: low nibbles are weights
//        64iq+8ir+k (sub-block 2iq, yl[0..7]); high nibbles are 64iq+32+8ir+k (2iq+1, yl[8..15]).
//   q2 = q1+32 ushorts (+64 B) = chunk iq+2: weights 128+... (sub-blocks 2iq+4/2iq+5, yh).
//   Masks 0x000F/0x0F00/0x00F0/0xF000 pick byte 2i lo, byte 2i+1 lo, byte 2i hi, byte 2i+1 hi,
//        carrying implicit x1/x256/x16/x4096 factors that the 1/256 and 1/16 terms undo.
//   sc8[0..7] = sc(2iq), sc(2iq+1), m(2iq), m(2iq+1), sc(2iq+4), sc(2iq+5), m(2iq+4), m(2iq+5),
//        i.e. q4k_scale_min unpacked two sub-blocks at a time with kmask1/2/3.
#define Q4K_MV_NSG 2 // simdgroups per threadgroup (host: MG_Q4K_MV_THREADS = 32*NSG)
#define Q4K_MV_NR0 2 // rows per simdgroup (host: MG_Q4K_MV_ROWS = NSG*NR0)
kernel void q4k_mul_mv(device const uchar* W [[buffer(0)]],
                       device const float* X [[buffer(1)]],
                       device float*       Y [[buffer(2)]],
                       constant int&    nblk [[buffer(3)]],
                       constant int&     out [[buffer(4)]],
                       uint   tg    [[threadgroup_position_in_grid]],
                       ushort tiisg [[thread_index_in_simdgroup]],
                       ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const int first_row = ((int)tg * Q4K_MV_NSG + (int)sgitg) * Q4K_MV_NR0;
    if (first_row >= out) return;
    const int nr = min(Q4K_MV_NR0, out - first_row);
    const long rowb = (long)nblk * 144;
    device const uchar* base = W + (long)first_row * rowb;

    float yl[16], yh[16];
    float sumf[Q4K_MV_NR0];
    for (short row = 0; row < Q4K_MV_NR0; ++row) sumf[row] = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    device const float* y4 = X + (long)ix * 256 + 64 * iq + 8 * ir;
    for (int ib = ix; ib < nblk; ib += 4) {
        float4 sumy = 0.0f;
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }
        device const uchar*  blk = base + (long)ib * 144;
        device const ushort* sc = (device const ushort*)(blk + 4) + iq;
        device const ushort* q1 = (device const ushort*)(blk + 16) + 16 * iq + 4 * ir;
        device const half*   dh = (device const half*)blk;
        for (short row = 0; row < Q4K_MV_NR0; ++row) {
            if (row < nr) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
                device const ushort* q2 = q1 + 32;
                float4 acc1 = 0.0f, acc2 = 0.0f;
                for (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2 * i + 0] * (float)(q1[i] & 0x000F);
                    acc1[1] += yl[2 * i + 1] * (float)(q1[i] & 0x0F00);
                    acc1[2] += yl[2 * i + 8] * (float)(q1[i] & 0x00F0);
                    acc1[3] += yl[2 * i + 9] * (float)(q1[i] & 0xF000);
                    acc2[0] += yh[2 * i + 0] * (float)(q2[i] & 0x000F);
                    acc2[1] += yh[2 * i + 1] * (float)(q2[i] & 0x0F00);
                    acc2[2] += yh[2 * i + 8] * (float)(q2[i] & 0x00F0);
                    acc2[3] += yh[2 * i + 9] * (float)(q2[i] & 0xF000);
                }
                sumf[row] += (float)dh[0] * ((acc1[0] + 1.0f / 256.0f * acc1[1]) * sc8[0] +
                                             (acc1[2] + 1.0f / 256.0f * acc1[3]) * sc8[1] * (1.0f / 16.0f) +
                                             (acc2[0] + 1.0f / 256.0f * acc2[1]) * sc8[4] +
                                             (acc2[2] + 1.0f / 256.0f * acc2[3]) * sc8[5] * (1.0f / 16.0f)) -
                             (float)dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                             sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            }
            // Next row, same block index: every pointer advances by one row stride.
            q1 += rowb / 2;
            sc += rowb / 2;
            dh += rowb / 2;
        }
        y4 += 4 * 256;
    }
    for (short row = 0; row < Q4K_MV_NR0; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0 && row < nr) Y[first_row + row] = s;
    }
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
                    if (N >= 2) a1 += w * X[xstride + xi];
                    if (N >= 3) a2 += w * X[2 * xstride + xi];
                    if (N >= 4) a3 += w * X[3 * xstride + xi];
                    if (N >= 5) a4 += w * X[4 * xstride + xi];
                    if (N >= 6) a5 += w * X[5 * xstride + xi];
                    if (N >= 7) a6 += w * X[6 * xstride + xi];
                    if (N >= 8) a7 += w * X[7 * xstride + xi];
                }
                for (int l = 0; l < 32; l++) {
                    const long xi = xbase + j + 32 + l;
                    const float w = d2 * (float)(q[qi + l] >> 4) - m2;
                    a0 += w * X[xi];
                    if (N >= 2) a1 += w * X[xstride + xi];
                    if (N >= 3) a2 += w * X[2 * xstride + xi];
                    if (N >= 4) a3 += w * X[3 * xstride + xi];
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

    a0 = q4k_sum8(a0);
    if (N >= 2) a1 = q4k_sum8(a1);
    if (N >= 3) a2 = q4k_sum8(a2);
    if (N >= 4) a3 = q4k_sum8(a3);
    if (N >= 5) a4 = q4k_sum8(a4);
    if (N >= 6) a5 = q4k_sum8(a5);
    if (N >= 7) a6 = q4k_sum8(a6);
    if (N >= 8) a7 = q4k_sum8(a7);
    if (tx == 0 && valid) {
        Y[o] = a0;
        if (N >= 2) Y[(long)out + o] = a1;
        if (N >= 3) Y[2 * (long)out + o] = a2;
        if (N >= 4) Y[3 * (long)out + o] = a3;
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

Q4K_MULTI_KERNEL(2)
Q4K_MULTI_KERNEL(3)
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

// q6k_mul_mv is the default P=1 Q6_K decode GEMV (fak#13599), the Q6_K twin of q4k_mul_mv. It
// ports llama.cpp's kernel_mul_mv_q6_K_f32_impl in ggml-metal.metal at
// 17197474510622a3b4ea7d0909d70b606f542b96 (MIT; Copyright (c) 2023-2026 The ggml authors;
// notice retained by reference in q4k_hotpath.go). Changes from upstream: P=1 only, raw byte
// addressing of fak's 210-B blocks, and a guarded tail so no row past `out` is read or written.
//
// Geometry: Q6K_MV_NSG simdgroups per threadgroup, Q6K_MV_NR0 rows per simdgroup. Lane parity
// ix = lane%2 walks blocks ix, ix+2, ...; the other 16 lanes of each parity split one block:
// tid = lane/2, ip = tid/8 picks the 128-weight half, il = tid%8 the 4-wide l slice l0 = 4il.
//
// Index contract against q6k_block_dot (ql@0, qh@128, SIGNED scales@192, d@208): lane covers
// l = l0..l0+3 of half ip (ql += 64ip, qh += 32ip, sc += 8ip), producing weights 128ip+l
// (+0/+32/+64/+96) with scales sc[l/16 + 0/2/4/6]. Since l0 is a multiple of 4, l/16 == l0/16 for
// the lane's whole slice, so is = 8ip + l0/16 is a single per-lane scale base. The qh masks
// 0x03/0x0C/0x30/0xC0 with shifts <<4/<<2/0/>>2 equal ((qh >> {0,2,4,6}) & 3) << 4.
#define Q6K_MV_NSG 2 // simdgroups per threadgroup (host: MG_Q6K_MV_THREADS = 32*NSG)
#define Q6K_MV_NR0 2 // rows per simdgroup (host: MG_Q6K_MV_ROWS = NSG*NR0)
kernel void q6k_mul_mv(device const uchar* W [[buffer(0)]],
                       device const float* X [[buffer(1)]],
                       device float*       Y [[buffer(2)]],
                       constant int&    nblk [[buffer(3)]],
                       constant int&     out [[buffer(4)]],
                       uint   tg    [[threadgroup_position_in_grid]],
                       ushort tiisg [[thread_index_in_simdgroup]],
                       ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uchar kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
    const int first_row = ((int)tg * Q6K_MV_NSG + (int)sgitg) * Q6K_MV_NR0;
    if (first_row >= out) return;
    const int nr = min(Q6K_MV_NR0, out - first_row);
    const long rowb = (long)nblk * 210;
    device const uchar* base = W + (long)first_row * rowb;

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;
    const short il  = tid % 8;
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;
    const short y_off  = 128 * ip + l0;
    const short ql_off = 64 * ip + l0;
    const short qh_off = 32 * ip + l0;

    float sumf[Q6K_MV_NR0];
    for (short row = 0; row < Q6K_MV_NR0; ++row) sumf[row] = 0.0f;
    float yl[16];
    for (int i = ix; i < nblk; i += 2) {
        device const uchar* blk = base + (long)i * 210;
        device const uchar* q1 = blk + ql_off;
        device const uchar* q2 = q1 + 32;
        device const uchar* qh = blk + 128 + qh_off;
        device const char*  sc = (device const char*)(blk + 192) + is; // SIGNED int8 scales
        device const half*  dh = (device const half*)(blk + 208);
        device const float* y  = X + (long)i * 256 + y_off;
        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = y[l +  0];
            yl[4 * l + 1] = y[l + 32];
            yl[4 * l + 2] = y[l + 64];
            yl[4 * l + 3] = y[l + 96];
        }
        for (short row = 0; row < Q6K_MV_NR0; ++row) {
            if (row < nr) {
                float4 s = 0.0f;
                for (short l = 0; l < 4; ++l) {
                    s[0] += yl[4 * l + 0] * (float)((int)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                    s[1] += yl[4 * l + 1] * (float)((int)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                    s[2] += yl[4 * l + 2] * (float)((int)((q1[l] >> 4)  | ((qh[l] & kmask3) << 0)) - 32);
                    s[3] += yl[4 * l + 3] * (float)((int)((q2[l] >> 4)  | ((qh[l] & kmask4) >> 2)) - 32);
                }
                sumf[row] += (float)dh[0] * (s[0] * (float)sc[0] + s[1] * (float)sc[2] +
                                             s[2] * (float)sc[4] + s[3] * (float)sc[6]);
            }
            // Next row, same block index.
            q1 += rowb;
            q2 += rowb;
            qh += rowb;
            sc += rowb;
            dh += rowb / 2;
        }
    }
    for (short row = 0; row < Q6K_MV_NR0; ++row) {
        const float t = simd_sum(sumf[row]);
        if (tiisg == 0 && row < nr) Y[first_row + row] = t;
    }
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

static id<MTLComputePipelineState> psoQ4KGemv, psoQ4KGemvVectorized, psoQ4KGemvMulti[7], psoQ4KGemm, psoQ4KGemmMM32, psoQ4KGemmM5CooperativeSMEM, psoQ4KSwiGLU, psoQ6KGemv, psoQ6KGemm, psoGraphQuantizeQ8;
// Optional llama.cpp-shaped P=1 GEMVs (fak#13599). A nil PSO keeps every default P=1 dispatch on
// the legacy one-simdgroup-per-row q4k_gemv / q6k_gemv kernels.
static id<MTLComputePipelineState> psoQ4KMulMv, psoQ6KMulMv;
static int gQ4KReady;

// P=1 GEMV request modes (the `mode` argument of mg_q4k_gemv, mirrored by q4kGEMVMode in q4k.go).
// DEFAULT resolves to mul_mv unless FAK_Q4K_GEMV_KERNEL=legacy (or mg_q4k_set_p1_kernel(0)) chose
// the legacy kernels or the mul_mv PSO failed to build. Explicit requests never substitute.
#define MG_GEMV_MODE_DEFAULT    0
#define MG_GEMV_MODE_VECTORIZED 1
#define MG_GEMV_MODE_MULMV      2
#define MG_GEMV_MODE_SCALAR     3
// Executed P=1 kernel identities (q4kGEMVExecution / GEMVKernel in Go). 0 = not executed.
#define MG_GEMV_EXEC_SCALAR     1
#define MG_GEMV_EXEC_VECTORIZED 2
#define MG_GEMV_EXEC_MULMV      3
// mul_mv dispatch geometry; must equal the MSL Q4K_MV_* / Q6K_MV_* defines in kQ4KSrc.
#define MG_Q4K_MV_ROWS    4  // Q4K_MV_NSG * Q4K_MV_NR0 rows per threadgroup
#define MG_Q4K_MV_THREADS 64 // 32 * Q4K_MV_NSG
#define MG_Q6K_MV_ROWS    4  // Q6K_MV_NSG * Q6K_MV_NR0
#define MG_Q6K_MV_THREADS 64 // 32 * Q6K_MV_NSG

// gQ4KP1Kernel is the process default for DEFAULT-mode P=1 GEMVs: 1 = mul_mv, 0 = legacy. It is
// resolved from FAK_Q4K_GEMV_KERNEL exactly once (first use) unless mg_q4k_set_p1_kernel set it
// first; -1 means unresolved.
static _Atomic int gQ4KP1Kernel = -1;

// mg_q4k_p1_kernel_env_value parses a FAK_Q4K_GEMV_KERNEL value: 1 = mul_mv (unset, empty,
// "mul_mv", "mulmv", "default"), 0 = the legacy family ("legacy" or "scalar", the Go
// GEMVKernelScalar name), -1 = unrecognized. Matching is case-insensitive.
int mg_q4k_p1_kernel_env_value(const char *v) {
    if (v == NULL || *v == '\0') return 1;
    if (strcasecmp(v, "legacy") == 0 || strcasecmp(v, "scalar") == 0) return 0;
    if (strcasecmp(v, "mul_mv") == 0 || strcasecmp(v, "mulmv") == 0 || strcasecmp(v, "default") == 0) return 1;
    return -1;
}

static int q4k_p1_mulmv_default(void) {
    int k = atomic_load_explicit(&gQ4KP1Kernel, memory_order_relaxed);
    if (k < 0) {
        const char *v = getenv("FAK_Q4K_GEMV_KERNEL");
        int want = mg_q4k_p1_kernel_env_value(v);
        int unknown = want < 0;
        if (unknown) want = 1;
        int expected = -1;
        // An unrecognized value keeps the mul_mv default but is never silent: the CAS winner logs
        // it once so a misspelled rollback (e.g. "LEGACY_") is visible.
        if (atomic_compare_exchange_strong(&gQ4KP1Kernel, &expected, want) && unknown) {
            NSLog(@"q4k: FAK_Q4K_GEMV_KERNEL=%s is not recognized (use legacy|scalar|mul_mv); P=1 GEMV stays mul_mv", v);
        }
        k = atomic_load_explicit(&gQ4KP1Kernel, memory_order_relaxed);
    }
    return k;
}

// mg_q4k_set_p1_kernel overrides the DEFAULT-mode P=1 kernel family (1 mul_mv, 0 legacy) and
// returns the previous resolved value. It exists for same-binary A/B and parity witnesses; the
// production knob is FAK_Q4K_GEMV_KERNEL=legacy.
int mg_q4k_set_p1_kernel(int mulmv) {
    int prev = q4k_p1_mulmv_default();
    atomic_store_explicit(&gQ4KP1Kernel, mulmv ? 1 : 0, memory_order_relaxed);
    return prev;
}

// mg_q4k_p1_kernel reports the executed identity a DEFAULT-mode Q4_K P=1 GEMV resolves to now
// (MG_GEMV_EXEC_MULMV or MG_GEMV_EXEC_SCALAR), or 0 before the pipelines are built.
int mg_q4k_p1_kernel(void) {
    if (!gQ4KReady) return 0;
    return (q4k_p1_mulmv_default() && psoQ4KMulMv != nil) ? MG_GEMV_EXEC_MULMV : MG_GEMV_EXEC_SCALAR;
}

// q4k_gemv_pso binds selection to an executed-kernel status. An explicit request never falls
// back: nil means the caller must return before allocating a command buffer or touching the
// output. Only DEFAULT may resolve to the legacy scalar kernel when mul_mv is disabled or
// unavailable. mode < 0 is the focused witness's unavailable-PSO injection.
static id<MTLComputePipelineState> q4k_gemv_pso(int mode, int* executed) {
    *executed = 0;
    switch (mode) {
    case MG_GEMV_MODE_DEFAULT:
        if (q4k_p1_mulmv_default() && psoQ4KMulMv != nil) {
            *executed = MG_GEMV_EXEC_MULMV;
            return psoQ4KMulMv;
        }
        if (psoQ4KGemv == nil) return nil;
        *executed = MG_GEMV_EXEC_SCALAR;
        return psoQ4KGemv;
    case MG_GEMV_MODE_SCALAR:
        if (psoQ4KGemv == nil) return nil;
        *executed = MG_GEMV_EXEC_SCALAR;
        return psoQ4KGemv;
    case MG_GEMV_MODE_VECTORIZED:
        if (psoQ4KGemvVectorized == nil) return nil;
        *executed = MG_GEMV_EXEC_VECTORIZED;
        return psoQ4KGemvVectorized;
    case MG_GEMV_MODE_MULMV:
        if (psoQ4KMulMv == nil) return nil;
        *executed = MG_GEMV_EXEC_MULMV;
        return psoQ4KMulMv;
    default:
        return nil;
    }
}

// q6k_gemv_pso is the Q6_K twin. There is no vectorized Q6_K kernel: a VECTORIZED request (a
// graph whose Q4_K projections opted into q4k_gemv_vectorized) resolves like DEFAULT for Q6_K.
static id<MTLComputePipelineState> q6k_gemv_pso(int mode, int* executed) {
    *executed = 0;
    switch (mode) {
    case MG_GEMV_MODE_DEFAULT:
    case MG_GEMV_MODE_VECTORIZED:
        if (q4k_p1_mulmv_default() && psoQ6KMulMv != nil) {
            *executed = MG_GEMV_EXEC_MULMV;
            return psoQ6KMulMv;
        }
        if (psoQ6KGemv == nil) return nil;
        *executed = MG_GEMV_EXEC_SCALAR;
        return psoQ6KGemv;
    case MG_GEMV_MODE_SCALAR:
        if (psoQ6KGemv == nil) return nil;
        *executed = MG_GEMV_EXEC_SCALAR;
        return psoQ6KGemv;
    case MG_GEMV_MODE_MULMV:
        if (psoQ6KMulMv == nil) return nil;
        *executed = MG_GEMV_EXEC_MULMV;
        return psoQ6KMulMv;
    default:
        return nil;
    }
}

// q4k_p1_dispatch / q6k_p1_dispatch issue the P=1 GEMV grid for an already-bound encoder whose
// PSO came from q4k_gemv_pso / q6k_gemv_pso. The legacy kernels key off one 32-lane threadgroup
// per output row; mul_mv packs MG_Q*K_MV_ROWS rows into each 64-thread threadgroup and guards the
// tail in-kernel, so out need not be a multiple of the rows per threadgroup.
static void q4k_p1_dispatch(id<MTLComputeCommandEncoder> e, int executed, int out) {
    if (executed == MG_GEMV_EXEC_MULMV) {
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(out + MG_Q4K_MV_ROWS - 1) / MG_Q4K_MV_ROWS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MG_Q4K_MV_THREADS, 1, 1)];
        return;
    }
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)out, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void q6k_p1_dispatch(id<MTLComputeCommandEncoder> e, int executed, int out) {
    if (executed == MG_GEMV_EXEC_MULMV) {
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(out + MG_Q6K_MV_ROWS - 1) / MG_Q6K_MV_ROWS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MG_Q6K_MV_THREADS, 1, 1)];
        return;
    }
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)out, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// mg_q6k_p1_encode_default binds the DEFAULT-mode P=1 Q6_K pipeline (q6k_mul_mv unless the
// process default is legacy) on an already-open encoder from another TU (fused_swiglu.go's fused
// MLP down projection) and issues the matching grid. Buffers use the q6k_gemv/q6k_mul_mv argument
// table (W, x, y, nblk, out). Returns the executed MG_GEMV_EXEC_* identity, or 0 with nothing
// bound when no Q6_K P=1 pipeline is available.
int mg_q6k_p1_encode_default(void *encoder, void *w, void *x, void *y, int nblk, int out) {
    if (encoder == NULL || w == NULL || x == NULL || y == NULL || out <= 0) return 0;
    int executed = 0;
    id<MTLComputePipelineState> pso = q6k_gemv_pso(MG_GEMV_MODE_DEFAULT, &executed);
    if (pso == nil) return 0;
    id<MTLComputeCommandEncoder> e = (__bridge id<MTLComputeCommandEncoder>)encoder;
    [e setComputePipelineState:pso];
    [e setBuffer:(__bridge id<MTLBuffer>)w offset:0 atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)y offset:0 atIndex:2];
    [e setBytes:&nblk length:sizeof(int) atIndex:3];
    [e setBytes:&out length:sizeof(int) atIndex:4];
    q6k_p1_dispatch(e, executed, out);
    return executed;
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
    for (int n = 2; n <= 8; n++) {
        NSString *name = [NSString stringWithFormat:@"q4k_gemv_multi%d", n];
        psoQ4KGemvMulti[n - 2] = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
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
    // q4k_mul_mv / q6k_mul_mv are optional (fak#13599): a nil PSO keeps DEFAULT-mode P=1 GEMVs on
    // the required legacy kernels, and only an explicit mul_mv request fails closed. A separate
    // error slot keeps an optional failure from masking the required-pipeline diagnostic.
    NSError *mvErr = nil;
    id<MTLFunction> fnQ4KMulMv = [lib newFunctionWithName:@"q4k_mul_mv"];
    id<MTLFunction> fnQ6KMulMv = [lib newFunctionWithName:@"q6k_mul_mv"];
    psoQ4KMulMv = fnQ4KMulMv ? [gDev newComputePipelineStateWithFunction:fnQ4KMulMv error:&mvErr] : nil;
    psoQ6KMulMv = fnQ6KMulMv ? [gDev newComputePipelineStateWithFunction:fnQ6KMulMv error:&mvErr] : nil;
    if (!psoQ4KMulMv || !psoQ6KMulMv) NSLog(@"q4k: optional mul_mv pipeline unavailable, legacy P=1 GEMV stays default: %@", mvErr);
    if (!psoQ4KGemv || !psoQ4KGemvMulti[0] || !psoQ4KGemvMulti[1] || !psoQ4KGemvMulti[2] ||
        !psoQ4KGemvMulti[3] || !psoQ4KGemvMulti[4] || !psoQ4KGemvMulti[5] || !psoQ4KGemvMulti[6] || !psoQ4KGemm || !psoQ4KSwiGLU ||
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
        int p1ex = 0;
        [e1 setComputePipelineState:q4k_gemv_pso(MG_GEMV_MODE_DEFAULT, &p1ex)];
        [e1 setBuffer:xb offset:0 atIndex:1];
        [e1 setBuffer:(__bridge id<MTLBuffer>)G.buf offset:G.offset atIndex:0];
        [e1 setBuffer:gMlpGate offset:0 atIndex:2];
        [e1 setBytes:&G.nblk length:sizeof(int) atIndex:3];
        [e1 setBytes:&G.out  length:sizeof(int) atIndex:4];
        q4k_p1_dispatch(e1, p1ex, G.out);
        [e1 setBuffer:(__bridge id<MTLBuffer>)U.buf offset:U.offset atIndex:0];
        [e1 setBuffer:gMlpUp offset:0 atIndex:2];
        [e1 setBytes:&U.nblk length:sizeof(int) atIndex:3];
        [e1 setBytes:&U.out  length:sizeof(int) atIndex:4];
        q4k_p1_dispatch(e1, p1ex, U.out);
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
        [e3 setComputePipelineState:q4k_gemv_pso(MG_GEMV_MODE_DEFAULT, &p1ex)];
        [e3 setBuffer:gMlpInter offset:0 atIndex:1];
        [e3 setBuffer:(__bridge id<MTLBuffer>)D.buf offset:D.offset atIndex:0];
        [e3 setBuffer:yb offset:0 atIndex:2];
        [e3 setBytes:&D.nblk length:sizeof(int) atIndex:3];
        [e3 setBytes:&D.out  length:sizeof(int) atIndex:4];
        q4k_p1_dispatch(e3, p1ex, D.out);
        [e3 endEncoding];

        int completed = mg_q4k_commit_bounded(cb, event);
        if (completed) {
            memcpy(y, yb.contents, (size_t)D.out * 4);
            mg_execution_event_readback(event);
        }
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

static int q6k_valid(int wid) {
    int idx = wid - MG_Q6_BASE;
    return idx >= 0 && idx < gNQ6 && gQ6[idx].buf != NULL;
}

// mg_q6k_desc exposes a resident Q6_K weight's unretained buffer and shape to another TU
// (fused_swiglu.go's fused MLP down projection). Returns 0 for an invalid wid. The caller must
// keep the weight registered (Go holds q6kRegistryMu) while it uses the buffer.
int mg_q6k_desc(int wid, void **buf, int *in, int *nblk, int *out) {
    if (!q6k_valid(wid) || buf == NULL || in == NULL || nblk == NULL || out == NULL) return 0;
    Q6KW *w = &gQ6[wid - MG_Q6_BASE];
    *buf = (void *)w->buf;
    *in = w->in;
    *nblk = w->nblk;
    *out = w->out;
    return 1;
}

static int q6k_slot(void) {
    for (int i = 0; i < gNQ6; i++) if (gQ6[i].buf == NULL) return i;
    if (gNQ6 >= MG_MAX_Q6) return -1;
    return gNQ6++;
}

// mg_q6k_upload copies a row-major Q6_K payload (out rows, in == nblk*256, 210 B/super-block)
// verbatim into a resident device buffer and returns a handle >= MG_Q6_BASE, or -1 on failure.
int mg_q6k_upload(const unsigned char* raw, int out, int in) {
    if (raw == NULL || gDev == nil) return -1;
    if (!q4k_init()) return -1;
    if (in <= 0 || in % 256 != 0 || out <= 0) return -1;
    int idx = q6k_slot();
    if (idx < 0) {
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
    gQ6[idx].buf = CFBridgingRetain(b);
    gQ6[idx].out = out;
    gQ6[idx].in = in;
    gQ6[idx].nblk = nblk;
    return MG_Q6_BASE + idx;
}

// mg_q6k_upload_nocopy binds page-aligned, page-rounded caller-owned Q6_K bytes directly.
// The Go owner pins the backing until the final shared handle releases this Metal buffer.
int mg_q6k_upload_nocopy(const unsigned char* raw, int out, int in) {
    if (raw == NULL || gDev == nil) return -1;
    if (!q4k_init()) return -1;
    if (in <= 0 || in % 256 != 0 || out <= 0) return -1;
    int idx = q6k_slot();
    if (idx < 0) return -1;
    int nblk = in / 256;
    long bytes = (long)out * nblk * 210;
    long page = sysconf(_SC_PAGESIZE);
    long buffer_bytes = bytes;
    if (page > 1 && bytes % page != 0) buffer_bytes += page - bytes % page;
    id<MTLBuffer> b = [gDev newBufferWithBytesNoCopy:(void*)raw
                                              length:(NSUInteger)buffer_bytes
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
    if (b == nil) return -1;
    gQ6[idx].buf = CFBridgingRetain(b);
    gQ6[idx].out = out;
    gQ6[idx].in = in;
    gQ6[idx].nblk = nblk;
    return MG_Q6_BASE + idx;
}

// mg_q6k_release drops one native Q6_K residency slot. Interior tombstones are reusable, so
// transient MTP aliases and model teardown cannot exhaust the fixed registry over time.
void mg_q6k_release(int wid) {
    if (!q6k_valid(wid)) return;
    int idx = wid - MG_Q6_BASE;
    CFBridgingRelease(gQ6[idx].buf);
    memset(&gQ6[idx], 0, sizeof(Q6KW));
    while (gNQ6 > 0 && gQ6[gNQ6 - 1].buf == NULL) gNQ6--;
}

int mg_q6k_live_count(void) {
    int n = 0;
    for (int i = 0; i < gNQ6; i++) if (gQ6[i].buf != NULL) n++;
    return n;
}

// mg_q6k_gemv computes y[out] = W[wid] · x for a resident Q6_K weight in one command buffer.
// The fused MLP already uses q6k_gemv as stage 3; this standalone wrapper lets k-quant decode
// sites such as the Qwen3.6 Q6_K LM head stay on Metal instead of escaping to the CPU.
// mg_q6k_gemv_mode takes an MG_GEMV_MODE_* and returns the executed MG_GEMV_EXEC_* identity, or 0
// when nothing dispatched.
int mg_q6k_gemv_mode(int wid, const float* x, float* y, int mode, mg_execution_event* event) {
    mg_execution_event_reset(event);
    if (!q6k_valid(wid)) return 0;
    @autoreleasepool {
        int executed = 0;
        id<MTLComputePipelineState> pso = q6k_gemv_pso(mode, &executed);
        if (pso == nil) return 0;
        Q6KW W = gQ6[wid - MG_Q6_BASE];
        q4k_grow_scratch((long)W.in, (long)W.out);
        id<MTLBuffer> xb = gQXBuf, yb = gQYBuf;
        memcpy(xb.contents, x, (size_t)W.in * 4);

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        mg_execution_event_command_buffer(event, cb);
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        mg_execution_event_encoder(event, e);
        [e setComputePipelineState:pso];
        [e setBuffer:(__bridge id<MTLBuffer>)W.buf offset:0 atIndex:0];
        [e setBuffer:xb offset:0 atIndex:1];
        [e setBuffer:yb offset:0 atIndex:2];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        q6k_p1_dispatch(e, executed, W.out);
        [e endEncoding];
        int completed = mg_q4k_commit_bounded(cb, event);
        if (!completed) return 0; // bounded timeout: the buffer never completed, so y is unwritten

        memcpy(y, yb.contents, (size_t)W.out * 4);
        mg_execution_event_readback(event);
        return executed;
    }
}

// mg_q6k_gemv keeps the established void ABI (q4k.go still declares it) and
// runs the DEFAULT-mode P=1 Q6_K kernel.
void mg_q6k_gemv(int wid, const float* x, float* y, mg_execution_event* event) {
    (void)mg_q6k_gemv_mode(wid, x, y, MG_GEMV_MODE_DEFAULT, event);
}

// mg_q6k_gemm computes Y[P,out] = X[P,in] * W[wid]^T for a resident Q6_K weight in one command
// buffer. This is the prefill counterpart to q6k_gemv / mg_q4k_mlp_q6down's stage 3: dense
// q4_k_m down_proj can stay on Metal instead of using the host kQuantMatRowsIntoBatch loop.
void mg_q6k_gemm(int wid, const float* X, int P, float* Y, mg_execution_event* event) {
    mg_execution_event_reset(event);
    if (!q6k_valid(wid) || P <= 0) return;
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
        int completed = mg_q4k_commit_bounded(cb, event);

        if (completed) {
            memcpy(Y, yb.contents, (size_t)P * W.out * 4);
            mg_execution_event_readback(event);
        }
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
        if (!q6k_valid(dw)) return -1;
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
        int p1ex = 0;
        [e1 setComputePipelineState:q4k_gemv_pso(MG_GEMV_MODE_DEFAULT, &p1ex)];
        [e1 setBuffer:xb offset:0 atIndex:1];
        for (int e = 0; e < n; e++) {
            Q4KW G = gQ4[gate_wids[e]], U = gQ4[up_wids[e]];
            NSUInteger off = (NSUInteger)((long)e * I * 4);
            [e1 setBuffer:(__bridge id<MTLBuffer>)G.buf offset:G.offset atIndex:0];
            [e1 setBuffer:gMlpGateK offset:off atIndex:2];
            [e1 setBytes:&G.nblk length:sizeof(int) atIndex:3];
            [e1 setBytes:&G.out  length:sizeof(int) atIndex:4];
            q4k_p1_dispatch(e1, p1ex, G.out);
            [e1 setBuffer:(__bridge id<MTLBuffer>)U.buf offset:U.offset atIndex:0];
            [e1 setBuffer:gMlpUpK offset:off atIndex:2];
            [e1 setBytes:&U.nblk length:sizeof(int) atIndex:3];
            [e1 setBytes:&U.out  length:sizeof(int) atIndex:4];
            q4k_p1_dispatch(e1, p1ex, U.out);
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
        int q6ex = 0;
        [e3 setComputePipelineState:q6k_gemv_pso(MG_GEMV_MODE_DEFAULT, &q6ex)];
        for (int e = 0; e < n; e++) {
            Q6KW D = gQ6[down_wids[e] - MG_Q6_BASE];
            NSUInteger interOff = (NSUInteger)((long)e * I * 4);
            NSUInteger yOff = (NSUInteger)((long)e * Dout * 4);
            [e3 setBuffer:(__bridge id<MTLBuffer>)D.buf offset:0 atIndex:0];
            [e3 setBuffer:gMlpInterK offset:interOff atIndex:1];
            [e3 setBuffer:gQYBufK offset:yOff atIndex:2];
            [e3 setBytes:&D.nblk length:sizeof(int) atIndex:3];
            [e3 setBytes:&D.out  length:sizeof(int) atIndex:4];
            q6k_p1_dispatch(e3, q6ex, D.out);
        }
        [e3 endEncoding];

        int completed = mg_q4k_commit_bounded(cb, event);
        if (!completed) return -1; // bounded timeout: the buffer never completed, so Ycat is unwritten
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
    if (!q6k_valid(down_wid)) return;
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
        int p1ex = 0;
        [e1 setComputePipelineState:q4k_gemv_pso(MG_GEMV_MODE_DEFAULT, &p1ex)];
        [e1 setBuffer:xb offset:0 atIndex:1];
        [e1 setBuffer:(__bridge id<MTLBuffer>)G.buf offset:G.offset atIndex:0];
        [e1 setBuffer:gMlpGate offset:0 atIndex:2];
        [e1 setBytes:&G.nblk length:sizeof(int) atIndex:3];
        [e1 setBytes:&G.out  length:sizeof(int) atIndex:4];
        q4k_p1_dispatch(e1, p1ex, G.out);
        [e1 setBuffer:(__bridge id<MTLBuffer>)U.buf offset:U.offset atIndex:0];
        [e1 setBuffer:gMlpUp offset:0 atIndex:2];
        [e1 setBytes:&U.nblk length:sizeof(int) atIndex:3];
        [e1 setBytes:&U.out  length:sizeof(int) atIndex:4];
        q4k_p1_dispatch(e1, p1ex, U.out);
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
        int q6ex = 0;
        [e3 setComputePipelineState:q6k_gemv_pso(MG_GEMV_MODE_DEFAULT, &q6ex)];
        [e3 setBuffer:gMlpInter offset:0 atIndex:1];
        [e3 setBuffer:(__bridge id<MTLBuffer>)D.buf offset:0 atIndex:0];
        [e3 setBuffer:yb offset:0 atIndex:2];
        [e3 setBytes:&D.nblk length:sizeof(int) atIndex:3];
        [e3 setBytes:&D.out  length:sizeof(int) atIndex:4];
        q6k_p1_dispatch(e3, q6ex, D.out);
        [e3 endEncoding];

        int completed = mg_q4k_commit_bounded(cb, event);
        if (completed) {
            memcpy(y, yb.contents, (size_t)D.out * 4);
            mg_execution_event_readback(event);
        }
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

// mg_q4k_gemv computes y[out] = W[wid] · x (one f32 activation row, length in). mode is an
// MG_GEMV_MODE_*. It returns the executed MG_GEMV_EXEC_* identity (1 scalar q4k_gemv, 2
// q4k_gemv_vectorized, 3 q4k_mul_mv), and 0 when no dispatch occurred.
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
        // dispatchThreadgroups (not dispatchThreads): every P=1 kernel keys its rows off
        // threadgroup_position_in_grid; q4k_p1_dispatch picks the geometry for `executed`.
        q4k_p1_dispatch(e, executed, W.out);
        [e endEncoding];
        int completed = mg_q4k_commit_bounded(cb, event);
        if (!completed) return 0; // bounded timeout: the buffer never completed, so y is unwritten

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
        int p1ex = 0;
        [e setComputePipelineState:q4k_gemv_pso(MG_GEMV_MODE_DEFAULT, &p1ex)];
        [e setBuffer:wbuf offset:W.offset atIndex:0];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        for (int i = 0; i < n; i++) {
            [e setBuffer:xb offset:(NSUInteger)((long)i * W.in  * 4) atIndex:1];
            [e setBuffer:yb offset:(NSUInteger)((long)i * W.out * 4) atIndex:2];
            q4k_p1_dispatch(e, p1ex, W.out);
        }
        [e endEncoding];
        int completed = mg_q4k_commit_bounded(cb, event);

        if (completed) {
            memcpy(Ycat, yb.contents, (size_t)n * W.out * 4);
            mg_execution_event_readback(event);
        }
    }
}

// mg_q4k_gemv_batch_multi applies one Q4_K weight to 2-8 activation rows in a single dispatch.
// q4k_gemv_multi owns the tile-reuse contract; the host side only copies the panel and binds it.
void mg_q4k_gemv_batch_multi(int wid, const float* Xcat, int n, float* Ycat, mg_execution_event* event) {
    mg_execution_event_reset(event);
    if (wid < 0 || wid >= gNQ4 || n < 2 || n > 8) return;
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
        [e setComputePipelineState:psoQ4KGemvMulti[n - 2]];
        [e setBuffer:wbuf offset:W.offset atIndex:0];
        [e setBuffer:xb   offset:0 atIndex:1];
        [e setBuffer:yb   offset:0 atIndex:2];
        [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
        [e setBytes:&W.out  length:sizeof(int) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(W.out + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
        int completed = mg_q4k_commit_bounded(cb, event);

        if (completed) {
            memcpy(Ycat, yb.contents, (size_t)n * W.out * 4);
            mg_execution_event_readback(event);
        }
    }
}

int mg_q4k_gemv_wide_m(int wid, const float* Xcat, int m, float* Ycat, mg_execution_event* event) {
    if (wid < 0 || wid >= gNQ4 || m < 2 || m > 8) return 0;
    mg_q4k_gemv_batch_multi(wid, Xcat, m, Ycat, event);
    return 1;
}

int mg_q4k_gemv_wide_m_encode(void* cb_ptr, int wid, void* x_buf_ptr, void* y_buf_ptr, int m) {
    if (!cb_ptr || wid < 0 || wid >= gNQ4 || m < 2 || m > 8 || !x_buf_ptr || !y_buf_ptr) return 0;
    if (!q4k_init()) return 0;
    Q4KW W = gQ4[wid];
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cb_ptr;
    id<MTLBuffer> wbuf = (__bridge id<MTLBuffer>)W.buf;
    id<MTLBuffer> xb = (__bridge id<MTLBuffer>)x_buf_ptr;
    id<MTLBuffer> yb = (__bridge id<MTLBuffer>)y_buf_ptr;
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    if (!e) return 0;
    [e setComputePipelineState:psoQ4KGemvMulti[m - 2]];
    [e setBuffer:wbuf offset:W.offset atIndex:0];
    [e setBuffer:xb   offset:0 atIndex:1];
    [e setBuffer:yb   offset:0 atIndex:2];
    [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
    [e setBytes:&W.out  length:sizeof(int) atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(W.out + 7) / 8, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [e endEncoding];
    return 1;
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
        int p1ex = 0;
        [e setComputePipelineState:q4k_gemv_pso(MG_GEMV_MODE_DEFAULT, &p1ex)];
        [e setBuffer:xb offset:0 atIndex:1]; // shared activation for every weight in the group
        for (int i = 0; i < n; i++) {
            Q4KW Wi = gQ4[wids[i]];
            [e setBuffer:(__bridge id<MTLBuffer>)Wi.buf offset:Wi.offset atIndex:0];
            [e setBuffer:yb offset:(NSUInteger)((long)yoff[i] * 4) atIndex:2];
            [e setBytes:&Wi.nblk length:sizeof(int) atIndex:3];
            [e setBytes:&Wi.out  length:sizeof(int) atIndex:4];
            q4k_p1_dispatch(e, p1ex, Wi.out);
        }
        [e endEncoding];
        int completed = mg_q4k_commit_bounded(cb, event);

        if (completed) {
            memcpy(Ycat, yb.contents, (size_t)ytot * 4);
            mg_execution_event_readback(event);
        }
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
        int p1ex = 0;
        [e setComputePipelineState:q4k_gemv_pso(MG_GEMV_MODE_DEFAULT, &p1ex)];
        [e setBuffer:gQXBuf offset:0 atIndex:1];
        for (int i = 0; i < nq4; i++) {
            Q4KW W = gQ4[q4_wids[i]];
            [e setBuffer:(__bridge id<MTLBuffer>)W.buf offset:W.offset atIndex:0];
            [e setBuffer:gQYBuf offset:(NSUInteger)((long)q4_yoff[i] * 4) atIndex:2];
            [e setBytes:&W.nblk length:sizeof(int) atIndex:3];
            [e setBytes:&W.out length:sizeof(int) atIndex:4];
            q4k_p1_dispatch(e, p1ex, W.out);
        }
        [e endEncoding];
        int q8_encoders = mg_q8_encode_gemv_group((__bridge void*)cb, q8_wids, nq8, q8_yoff);
        if (q8_encoders <= 0) {
            return -1; // candidate encoding began; fail closed rather than falling back mid-batch.
        }
		if (event != NULL) event->encoders += q8_encoders;
        int completed = mg_q4k_commit_bounded(cb, event);
        if (!completed) return -1;
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
        int completed = mg_q4k_commit_bounded(cb, event);
        if (!completed) return 0; // bounded timeout: the buffer never completed, so Y is unwritten
        // GPUStartTime/GPUEndTime are valid only after the wait returns (already-completed
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
        int completed = mg_q4k_commit_bounded(cb, event);
        if (!completed) return 0; // bounded timeout: the buffer never completed, so Ycat is unwritten
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
    for (int i = 0; i < gNQ6; i++) mg_q6k_release(MG_Q6_BASE + i);
    gNQ6 = 0;
    gQXBuf = nil; gQXCap = 0;
    gQYBuf = nil; gQYCap = 0;
    gMlpGate = nil; gMlpUp = nil; gMlpInter = nil; gMlpCap = 0;
}

// ---- caller-owned quantized projection graph (#9267) ---------------------------
typedef struct {
    id<MTLCommandBuffer> cb;
    id<MTLSharedEvent> test_gate;
    id<MTLBuffer> xf, xq, xd;
    NSMutableArray *results;
    NSMutableDictionary *pool; // recycled [NSNumber length] -> NSMutableArray of idle buffers
    int pool_buffers;          // live buffers currently parked in pool
    int P, in, encoders, committed, readbacks, buffers, wait_limit_ms;
    uint64_t retained_buffer_bytes; // graph-tracked buffers; all remain retained until Free
    int graph_gemv_p1;         // 1 routes P=1 graph projections to the GEMV kernels
    int graph_gemv_mode;       // P=1 graph projections: MG_GEMV_MODE_* (0 default = mul_mv unless legacy)
    int graph_q4k_gemv_exec;   // bitmask of executed P=1 Q4_K kernels: bit MG_GEMV_EXEC_*
    int graph_q6k_gemv_exec;   // bitmask of executed P=1 Q6_K kernels: bit MG_GEMV_EXEC_*
    int graph_mm_mode;         // Q4_K projection candidate: 0 scalar, 2 wide-tile cooperative-SMEM
    int graph_buf_pool;        // per-shape recycle depth; 0 disables the pool
    double gpu_ms, wait_ms;
} MGProjectionGraph;

static _Atomic int gGraphLiveOwners;
static _Atomic int gGraphLiveBuffers;

static void mg_graph_track_buffer(MGProjectionGraph *g, id<MTLBuffer> buffer) {
    if (!g || !buffer) return;
    g->buffers++;
    g->retained_buffer_bytes += (uint64_t)buffer.length;
    atomic_fetch_add_explicit(&gGraphLiveBuffers, 1, memory_order_relaxed);
}

static void mg_graph_release_tracked_buffers(MGProjectionGraph *g) {
    if (!g) return;
    int buffers=g->buffers;
    g->xf=nil;g->xq=nil;g->xd=nil;g->results=nil;g->buffers=0;g->retained_buffer_bytes=0;
    if (buffers) atomic_fetch_sub_explicit(&gGraphLiveBuffers, buffers, memory_order_relaxed);
}

typedef struct {
    int committed, completed_wait, encoders, host_readbacks;
    int allocated_buffers;
    uint64_t retained_buffer_bytes;
    double gpu_milliseconds, wait_milliseconds;
    int timing_available;
    int status_code, error_code, device_ok;
    char error_text[256];
} mg_graph_receipt;

// mg_graph_set_gemv_vectorized selects the P=1 graph projection kernel variant: 1 uses
// q4k_gemv_vectorized, 0 the process default P=1 kernel (q4k_mul_mv, or the scalar q4k_gemv
// under FAK_Q4K_GEMV_KERNEL=legacy). It must be called before any encode and is inert for P!=1
// (the GEMM path). Returning 0 for an unavailable vectorized pipeline is the caller's fail-closed
// signal.
int mg_graph_set_gemv_vectorized(void *opaque, int mode) {
    MGProjectionGraph *g = opaque;
    if (!g || g->committed || g->encoders != 0) return 0;
    if (mode != 0 && psoQ4KGemvVectorized == nil) return 0;
    g->graph_gemv_mode = mode != 0 ? MG_GEMV_MODE_VECTORIZED : MG_GEMV_MODE_DEFAULT;
    return 1;
}

// mg_graph_set_gemv_kernel selects any MG_GEMV_MODE_* for this graph's P=1 projections. An
// explicit kernel whose Q4_K or Q6_K pipeline is unavailable is refused (0) before any encode, so
// the caller keeps the default identity; DEFAULT is always accepted.
int mg_graph_set_gemv_kernel(void *opaque, int mode) {
    MGProjectionGraph *g = opaque;
    if (!g || g->committed || g->encoders != 0) return 0;
    int executed = 0;
    if (q4k_gemv_pso(mode, &executed) == nil || q6k_gemv_pso(mode, &executed) == nil) return 0;
    g->graph_gemv_mode = mode;
    return 1;
}

// mg_graph_gemv_executed reports the bitmask (bit MG_GEMV_EXEC_*) of P=1 GEMV kernels this graph
// actually encoded: q6=0 for Q4_K projections, q6=1 for Q6_K.
int mg_graph_gemv_executed(void *opaque, int q6) {
    MGProjectionGraph *g = opaque;
    if (!g) return 0;
    return q6 ? g->graph_q6k_gemv_exec : g->graph_q4k_gemv_exec;
}

// mg_graph_set_mm_mode sets the Q4_K projection candidate this graph encodes: 0 is the scalar
// q4k_gemm kernel (the historical default), 2 is the wide-tile cooperative-SMEM candidate
// q4k_gemm_m5_cooperative_smem. It is the graph-side production selector for the P>=64 panel
// regime (fak#13041 / this leaf). Mode 2 is fail-closed here: it is refused before any encode
// (returning 0) when this graph's P is below the kernel's 64-token eligibility or the optional
// pipeline is unavailable, so the caller must keep the scalar identity. The Go caller only sets
// mode 2 after the device/version-pinned crossover admits it, so an unpinned margin can never reach
// this seam. Returns 1 when the mode was accepted (including the mode-0 no-op).
int mg_graph_set_mm_mode(void *opaque, int mode) {
    MGProjectionGraph *g = opaque;
    if (!g || g->committed || g->encoders != 0) return 0;
    if (mode == 0) { g->graph_mm_mode = 0; return 1; }
    if (mode == 2) {
        if (g->P < 64 || psoQ4KGemmM5CooperativeSMEM == nil) return 0;
        g->graph_mm_mode = 2;
        return 1;
    }
    return 0;
}

// mg_graph_mm_mode reports the Q4_K projection candidate the graph will request (0 scalar, 2
// wide-tile cooperative-SMEM). It is the graph-side half of the typed requested/executed identity.
int mg_graph_mm_mode(void *opaque) {
    MGProjectionGraph *g = opaque;
    return g ? g->graph_mm_mode : 0;
}

// mg_graph_set_gemv_p1 opts a graph into the P=1 GEMV projection route. It is opt-in so
// every existing caller (and every numerical parity test) keeps the prefill GEMM pipeline
// until the whole-token decode owner explicitly requests the decode kernels. mode 1 enables
// the GEMV route; the vectorized/scalar choice is made separately by
// mg_graph_set_gemv_vectorized. Returns 0 for a graph that has already encoded work.
// mg_graph_gemv_p1 reports whether this graph opted into the P=1 GEMV projection route.
int mg_graph_gemv_p1(void *opaque) {
    MGProjectionGraph *g = opaque;
    return g ? g->graph_gemv_p1 : 0;
}
// mg_graph_set_buffer_pool enables the per-shape idle-buffer recycle pool with the given
// maximum idle depth per shape (0 disables). It must be called before the first encode.
int mg_graph_set_buffer_pool(void *opaque, int depth) {
    MGProjectionGraph *g = opaque;
    if (!g || g->committed || g->encoders != 0 || depth < 0) return 0;
    g->graph_buf_pool = depth;
    return 1;
}
int mg_graph_set_gemv_p1(void *opaque, int mode) {
    MGProjectionGraph *g = opaque;
    if (!g || g->committed || g->encoders != 0) return 0;
    if (mode != 0 && psoQ4KGemv == nil) return 0;
    g->graph_gemv_p1 = mode;
    return 1;
}
void *mg_graph_begin(const float *xf, const signed char *xq, const float *xd, int P, int in) {
    if (!q4k_init() || P <= 0 || in <= 0) return NULL;
    MGProjectionGraph *g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->P=P; g->in=in; g->wait_limit_ms=MG_Q4K_WAIT_LIMIT_MS; g->cb=[gQueue commandBuffer]; g->results=[NSMutableArray array]; g->pool=[NSMutableDictionary dictionary];
    if (!g->cb || !g->results || !g->pool) { free(g); return NULL; }
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
    id<MTLBuffer> y=nil;
    if (g->graph_buf_pool > 0 && g->pool) {
        NSMutableArray *idle=[g->pool objectForKey:@(n)];
        if (idle && [idle count] > 0) { y=[idle lastObject]; [idle removeLastObject]; g->pool_buffers--; }
    }
    if (!y) { y=[gDev newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared]; if(!y)return NULL; [g->results addObject:y]; mg_graph_track_buffer(g,y); }
    g->encoders++;return (__bridge void*)y;
}
// mg_graph_recycle_result returns a buffer whose last consumer has already been encoded to
// the graph's per-shape idle list, so a later encode of the same element count reuses it
// instead of allocating. Reuse is safe because all dispatches share one command buffer and
// Metal executes encoders in order: a later write cannot overtake an earlier read. Pinned
// buffers (terminal KV / final norm) are never pooled. No-op when the pool is disabled.
void mg_graph_recycle_result(void *opaque, void *ptr) {
    MGProjectionGraph *g=opaque; id<MTLBuffer>b=(__bridge id<MTLBuffer>)ptr;
    if(!g||g->committed||!b||g->graph_buf_pool<=0||!g->pool)return;
    if(![g->results containsObject:b])return;
    NSUInteger n=b.length/sizeof(float);
    NSMutableArray *idle=[g->pool objectForKey:@(n)];
    if(!idle){idle=[NSMutableArray array];[g->pool setObject:idle forKey:@(n)];}
    // Different GraphResult handles may name the same buffer. Park each buffer
    // once, or two subsequent allocations could alias each other's live output.
    if([idle indexOfObjectIdenticalTo:b]!=NSNotFound)return;
    if([idle count]>=(NSUInteger)g->graph_buf_pool)return; // bounded ring
    [idle addObject:b];g->pool_buffers++;
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

// mg_graph_q4k_gemv encodes a SINGLE-TOKEN (P=1) Q4_K projection with a decode GEMV kernel
// instead of the prefill GEMM pipeline. At P=1 the GEMM's 64-wide token tile wastes 63/64 of its
// work. The kernel is the graph's MG_GEMV_MODE_* selection: by default q4k_mul_mv (fak#13599),
// else q4k_gemv / q4k_gemv_vectorized. `x` is the graph-owned activation or result buffer.
// Returns the output buffer, or nil BEFORE mg_graph_result (so before g->encoders or the result
// set change) when the requested pipeline is unavailable or x is missing.
static id<MTLBuffer> mg_graph_q4k_gemv(MGProjectionGraph *g, id<MTLBuffer> x, int wid) {
    Q4KW *w = &gQ4[wid];
    int executed = 0;
    id<MTLComputePipelineState> pso = q4k_gemv_pso(g->graph_gemv_mode, &executed);
    if (pso == nil || x == nil) return nil;
    id<MTLBuffer> y = (__bridge id<MTLBuffer>)mg_graph_result(g, (NSUInteger)g->P * (NSUInteger)w->out);
    if (!y) return nil;
    id<MTLComputeCommandEncoder> e = [g->cb computeCommandEncoder];
    if (e == nil) return nil;
    [e setComputePipelineState:pso];
    [e setBuffer:(__bridge id<MTLBuffer>)w->buf offset:w->offset atIndex:0];
    [e setBuffer:x offset:0 atIndex:1];
    [e setBuffer:y offset:0 atIndex:2];
    [e setBytes:&w->nblk length:sizeof(int) atIndex:3];
    [e setBytes:&w->out length:sizeof(int) atIndex:4];
    q4k_p1_dispatch(e, executed, w->out);
    [e endEncoding];
    g->graph_q4k_gemv_exec |= 1 << executed;
    return y;
}
// mg_graph_q6k_gemv is the Q6_K single-token twin of mg_graph_q4k_gemv (q6k_mul_mv by default).
static id<MTLBuffer> mg_graph_q6k_gemv(MGProjectionGraph *g, id<MTLBuffer> x, int wid) {
    int i = wid - MG_Q6_BASE;
    int executed = 0;
    id<MTLComputePipelineState> pso = q6k_gemv_pso(g->graph_gemv_mode, &executed);
    if (pso == nil || x == nil) return nil;
    id<MTLBuffer> y = (__bridge id<MTLBuffer>)mg_graph_result(g, (NSUInteger)g->P * (NSUInteger)gQ6[i].out);
    if (!y) return nil;
    id<MTLComputeCommandEncoder> e = [g->cb computeCommandEncoder];
    if (e == nil) return nil;
    [e setComputePipelineState:pso];
    [e setBuffer:(__bridge id<MTLBuffer>)gQ6[i].buf offset:0 atIndex:0];
    [e setBuffer:x offset:0 atIndex:1];
    [e setBuffer:y offset:0 atIndex:2];
    [e setBytes:&gQ6[i].nblk length:sizeof(int) atIndex:3];
    [e setBytes:&gQ6[i].out length:sizeof(int) atIndex:4];
    q6k_p1_dispatch(e, executed, gQ6[i].out);
    [e endEncoding];
    g->graph_q6k_gemv_exec |= 1 << executed;
    return y;
}
void *mg_graph_encode_q4k(void *opaque, int wid) {
    MGProjectionGraph *g=opaque; if (!g || g->committed || !g->xf || wid<0 || wid>=gNQ4 || gQ4[wid].in!=g->in) return NULL;
    if (g->graph_gemv_p1 && g->P == 1) { id<MTLBuffer> y=mg_graph_q4k_gemv(g,g->xf,wid); return y?(__bridge void*)y:NULL; }
    Q4KW *w=&gQ4[wid]; id<MTLBuffer> y=(__bridge id<MTLBuffer>)mg_graph_result(g,(NSUInteger)g->P*(NSUInteger)w->out); if(!y)return NULL;
    // Route through the graph's requested candidate. graph_mm_mode defaults to 0 (scalar); the Go
    // production selector sets 2 for the widened panel regime only when the device/version-pinned
    // crossover admits it. A mode-2 graph whose pipeline/P guard fails here returns NULL and the Go
    // caller declines fail-open before any state mutation.
    int executed=0, BN=64; id<MTLComputePipelineState> pso=q4k_gemm_pso(g->P,g->graph_mm_mode,&executed,&BN); if(!pso)return NULL;
    const int BM=64,TG=256; int rowBlocks=(w->out+BM-1)/BM;
    id<MTLComputeCommandEncoder> e=[g->cb computeCommandEncoder]; [e setComputePipelineState:pso]; [e setBuffer:(__bridge id<MTLBuffer>)w->buf offset:w->offset atIndex:0]; [e setBuffer:g->xf offset:0 atIndex:1]; [e setBuffer:y offset:0 atIndex:2]; [e setBytes:&w->nblk length:sizeof(int) atIndex:3];[e setBytes:&w->out length:sizeof(int) atIndex:4];[e setBytes:&g->P length:sizeof(int) atIndex:5]; for(int t0=0;t0<g->P;t0+=BN){int nt=g->P-t0;if(nt>BN)nt=BN;[e setBytes:&t0 length:sizeof(int) atIndex:6];[e setBytes:&nt length:sizeof(int) atIndex:7];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)rowBlocks,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)TG,1,1)];}[e endEncoding]; return (__bridge void*)y;
}
void *mg_graph_encode_q4k_from(void *opaque,int wid,void*input,int elems) {
    MGProjectionGraph*g=opaque;id<MTLBuffer>x=(__bridge id<MTLBuffer>)input;if(!g||g->committed||!x||wid<0||wid>=gNQ4||gQ4[wid].in*g->P!=elems||![g->results containsObject:x])return NULL;
    if (g->graph_gemv_p1 && g->P == 1) { id<MTLBuffer> y=mg_graph_q4k_gemv(g,x,wid); return y?(__bridge void*)y:NULL; }
    Q4KW*w=&gQ4[wid];id<MTLBuffer>y=(__bridge id<MTLBuffer>)mg_graph_result(g,(NSUInteger)g->P*(NSUInteger)w->out);if(!y)return NULL;
    int executed=0,BN=64;id<MTLComputePipelineState>pso=q4k_gemm_pso(g->P,g->graph_mm_mode,&executed,&BN);if(!pso)return NULL;const int BM=64,TG=256;int rowBlocks=(w->out+BM-1)/BM;
    id<MTLComputeCommandEncoder>e=[g->cb computeCommandEncoder];[e setComputePipelineState:pso];[e setBuffer:(__bridge id<MTLBuffer>)w->buf offset:w->offset atIndex:0];[e setBuffer:x offset:0 atIndex:1];[e setBuffer:y offset:0 atIndex:2];[e setBytes:&w->nblk length:sizeof(int) atIndex:3];[e setBytes:&w->out length:sizeof(int) atIndex:4];[e setBytes:&g->P length:sizeof(int) atIndex:5];for(int t0=0;t0<g->P;t0+=BN){int nt=g->P-t0;if(nt>BN)nt=BN;[e setBytes:&t0 length:sizeof(int) atIndex:6];[e setBytes:&nt length:sizeof(int) atIndex:7];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)rowBlocks,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)TG,1,1)];}[e endEncoding];return (__bridge void*)y;
}
void *mg_graph_encode_q6k(void *opaque, int wid) {
    MGProjectionGraph *g=opaque; int i=wid-MG_Q6_BASE; if(!g||g->committed||!g->xf||!q6k_valid(wid)||gQ6[i].in!=g->in)return NULL;
    if (g->graph_gemv_p1 && g->P == 1) { id<MTLBuffer> y=mg_graph_q6k_gemv(g,g->xf,wid); return y?(__bridge void*)y:NULL; }
    Q6KW *w=&gQ6[i]; id<MTLBuffer> y=(__bridge id<MTLBuffer>)mg_graph_result(g,(NSUInteger)g->P*(NSUInteger)w->out);if(!y)return NULL; id<MTLComputeCommandEncoder>e=[g->cb computeCommandEncoder];[e setComputePipelineState:psoQ6KGemm];[e setBuffer:(__bridge id<MTLBuffer>)w->buf offset:0 atIndex:0];[e setBuffer:g->xf offset:0 atIndex:1];[e setBuffer:y offset:0 atIndex:2];[e setBytes:&w->nblk length:sizeof(int) atIndex:3];[e setBytes:&w->out length:sizeof(int) atIndex:4];[e setBytes:&g->P length:sizeof(int) atIndex:5];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)w->out,(NSUInteger)g->P,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];return (__bridge void*)y;
}
void *mg_graph_encode_q6k_from(void *opaque,int wid,void*input,int elems) {
    MGProjectionGraph*g=opaque;int i=wid-MG_Q6_BASE;id<MTLBuffer>x=(__bridge id<MTLBuffer>)input;if(!g||g->committed||!x||!q6k_valid(wid)||gQ6[i].in*g->P!=elems||![g->results containsObject:x])return NULL;
    if (g->graph_gemv_p1 && g->P == 1) { id<MTLBuffer> y=mg_graph_q6k_gemv(g,x,wid); return y?(__bridge void*)y:NULL; }
    Q6KW*w=&gQ6[i];id<MTLBuffer>y=(__bridge id<MTLBuffer>)mg_graph_result(g,(NSUInteger)g->P*(NSUInteger)w->out);if(!y)return NULL;id<MTLComputeCommandEncoder>e=[g->cb computeCommandEncoder];[e setComputePipelineState:psoQ6KGemm];[e setBuffer:(__bridge id<MTLBuffer>)w->buf offset:0 atIndex:0];[e setBuffer:x offset:0 atIndex:1];[e setBuffer:y offset:0 atIndex:2];[e setBytes:&w->nblk length:sizeof(int) atIndex:3];[e setBytes:&w->out length:sizeof(int) atIndex:4];[e setBytes:&g->P length:sizeof(int) atIndex:5];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)w->out,(NSUInteger)g->P,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];return (__bridge void*)y;
}
extern void *mg_q8_graph_encode(void *graph, int wid);
extern void *mg_q8_graph_encode_from(void *graph, int wid, void *q, void *d, int elems);
void *mg_graph_encode_q8(void *opaque,int wid){return mg_q8_graph_encode(opaque,wid);}
void *mg_graph_encode_q8_from(void *opaque,int wid,void*q,void*d,int elems){return mg_q8_graph_encode_from(opaque,wid,q,d,elems);}
// mg_graph_finish commits and waits on the graph command buffer. The wait is bounded by
// MG_Q4K_WAIT_LIMIT_MS (a completed-handler semaphore, attached before commit), so a stalled GPU
// cannot hang the caller indefinitely: a buffer that is not Completed after the wait (timeout or
// GPU/device fault) is recorded in the receipt (status_code, error_code, device_ok, bounded
// error_text) so the caller can report a typed stall instead of a silent hang. Return stays 1 only
// for a Completed, non-injected submit and 0 otherwise; the receipt distinguishes an injected test
// failure from a real device failure.
int mg_graph_finish(void *opaque,mg_graph_receipt*r,int inject_post_submit_failure){MGProjectionGraph*g=opaque;if(r)memset(r,0,sizeof(*r));if(!g||g->committed||g->encoders==0)return 0;if(r){r->allocated_buffers=g->buffers;r->retained_buffer_bytes=g->retained_buffer_bytes;}g->committed=1;CFAbsoluteTime t=CFAbsoluteTimeGetCurrent();dispatch_semaphore_t gsem=dispatch_semaphore_create(0);[g->cb addCompletedHandler:^(id<MTLCommandBuffer> b){(void)b;dispatch_semaphore_signal(gsem);}];[g->cb commit];long gto=dispatch_semaphore_wait(gsem,dispatch_time(DISPATCH_TIME_NOW,(int64_t)(g->wait_limit_ms*NSEC_PER_MSEC)));g->wait_ms=(CFAbsoluteTimeGetCurrent()-t)*1000.;int completed=gto==0&&g->cb.status==MTLCommandBufferStatusCompleted;int inject_device_fault=inject_post_submit_failure==2;if(r){r->committed=1;r->completed_wait=inject_device_fault?0:completed;r->status_code=inject_device_fault?MTLCommandBufferStatusError:(int)g->cb.status;NSError*err=inject_device_fault?nil:g->cb.error;r->device_ok=(r->completed_wait&&err==nil)?1:0;if(inject_device_fault){snprintf(r->error_text,sizeof(r->error_text),"injected device fault");}else if(err){r->error_code=(int)err.code;NSString*desc=err.localizedDescription;if(desc){const char*u=[desc UTF8String];if(u)snprintf(r->error_text,sizeof(r->error_text),"%s",u);}}r->encoders=g->encoders;r->host_readbacks=g->readbacks;r->wait_milliseconds=g->wait_ms;if(@available(macOS 10.15,*)){double a=g->cb.GPUStartTime,b=g->cb.GPUEndTime;if(b>=a&&a>0){r->gpu_milliseconds=(b-a)*1000.;r->timing_available=1;}}}int observed=r?(inject_device_fault?0:completed):completed;return observed&&!inject_post_submit_failure;}
int mg_graph_await_terminal(void *opaque){MGProjectionGraph*g=opaque;if(!g||!g->committed||!g->cb)return 0;[g->cb waitUntilCompleted];return g->cb.status==MTLCommandBufferStatusCompleted;}
int mg_graph_read(void*opaque,void*result,float*dst,int n){MGProjectionGraph*g=opaque;id<MTLBuffer>y=(__bridge id<MTLBuffer>)result;if(!g||!g->committed||!y||!dst||n<0||![g->results containsObject:y])return 0;memcpy(dst,[y contents],(NSUInteger)n*sizeof(float));g->readbacks++;return 1;}
int mg_graph_read_pack(void*opaque,void**results,const int*sizes,int count,float*dst,int total){MGProjectionGraph*g=opaque;if(!g||!g->committed||!results||!sizes||count<=0||!dst||total<0)return 0;int off=0;for(int i=0;i<count;i++){id<MTLBuffer>y=(__bridge id<MTLBuffer>)results[i];int n=sizes[i];if(!y||n<0||off>total-n||![g->results containsObject:y])return 0;memcpy(dst+off,[y contents],(NSUInteger)n*sizeof(float));off+=n;}if(off!=total)return 0;g->readbacks++;return 1;}
void mg_graph_free(void*opaque){MGProjectionGraph*g=opaque;if(!g)return;g->cb=nil;g->test_gate=nil;g->pool=nil;g->pool_buffers=0;mg_graph_release_tracked_buffers(g);atomic_fetch_sub_explicit(&gGraphLiveOwners,1,memory_order_relaxed);free(g);}

void *mg_graph_test_hold_terminal(void *opaque,int wait_limit_ms){MGProjectionGraph*g=opaque;if(!g||g->committed||g->encoders==0||g->test_gate||wait_limit_ms<=0)return NULL;if(![gDev respondsToSelector:@selector(newSharedEvent)])return NULL;id<MTLSharedEvent>event=[gDev newSharedEvent];if(!event)return NULL;event.signaledValue=0;[g->cb encodeWaitForEvent:event value:1];g->test_gate=event;g->wait_limit_ms=wait_limit_ms;return (void *)CFBridgingRetain(event);}
void mg_graph_test_release_terminal(void *opaque){if(!opaque)return;id<MTLSharedEvent>event=(__bridge_transfer id<MTLSharedEvent>)opaque;event.signaledValue=1;}

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
// mg_graph_upload copies a second host f32 panel into a graph-owned shared buffer before
// commit, so a graph can consume two host activations (e.g. the dense prefill's residual X as
// the begin panel plus the host attention output as an uploaded panel). It is a host memcpy
// into StorageModeShared memory, not an encoder: Metal reads it when the command buffer runs.
void *mg_graph_upload(void *opaque,const float *src,int n){MGProjectionGraph*g=opaque;if(!g||g->committed||!src||n<=0)return NULL;id<MTLBuffer>b=[gDev newBufferWithLength:(NSUInteger)n*sizeof(float) options:MTLResourceStorageModeShared];if(!b)return NULL;memcpy([b contents],src,(NSUInteger)n*sizeof(float));[g->results addObject:b];mg_graph_track_buffer(g,b);return (__bridge void*)b;}
void mg_graph_note_encoder(void *opaque){MGProjectionGraph*g=opaque;if(g&&!g->committed)g->encoders++;}
