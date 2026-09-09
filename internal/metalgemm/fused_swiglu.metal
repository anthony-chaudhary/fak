#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// Fused Dequantize-GEMV-SwiGLU MSL Compute Kernels
// Study Provenance: mlc-ai/mlc-llm:compiler_pass:FuseDequantizeMatmulEwise
//
// Combines weight dequantization with dual-projection dot products (W_g and W_up)
// and evaluates SwiGLU activation in registers without round-tripping gate and up
// activation tensors to global DRAM.
// -----------------------------------------------------------------------------

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

// q4k_dual_block_dot dots one 144-B super-block from W_g and one from W_up
// simultaneously against the shared 256-wide activation slice xs.
// Reuses activation vector loads and interleaves FMA pipes.
inline float2 q4k_dual_block_dot(device const uchar* blkG, device const uchar* blkU, device const float* xs) {
    float dG  = (float)(*(device const half*)(blkG + 0));
    float dmG = (float)(*(device const half*)(blkG + 2));
    device const uchar* scalesG = blkG + 4;
    device const uchar* qG = blkG + 16;

    float dU  = (float)(*(device const half*)(blkU + 0));
    float dmU = (float)(*(device const half*)(blkU + 2));
    device const uchar* scalesU = blkU + 4;
    device const uchar* qU = blkU + 16;

    float accG = 0.0f;
    float accU = 0.0f;
    int qi = 0;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        float2 sm0_G = q4k_scale_min(is,     scalesG);
        float2 sm1_G = q4k_scale_min(is + 1, scalesG);
        float d1_G = dG * sm0_G.x, m1_G = dmG * sm0_G.y;
        float d2_G = dG * sm1_G.x, m2_G = dmG * sm1_G.y;

        float2 sm0_U = q4k_scale_min(is,     scalesU);
        float2 sm1_U = q4k_scale_min(is + 1, scalesU);
        float d1_U = dU * sm0_U.x, m1_U = dmU * sm0_U.y;
        float d2_U = dU * sm1_U.x, m2_U = dmU * sm1_U.y;

        for (int l = 0; l < 32; l++) {
            float x_val = xs[j + l];
            accG += (d1_G * (float)(qG[qi + l] & 0x0f) - m1_G) * x_val;
            accU += (d1_U * (float)(qU[qi + l] & 0x0f) - m1_U) * x_val;
        }
        for (int l = 0; l < 32; l++) {
            float x_val = xs[j + 32 + l];
            accG += (d2_G * (float)(qG[qi + l] >> 4) - m2_G) * x_val;
            accU += (d2_U * (float)(qU[qi + l] >> 4) - m2_U) * x_val;
        }
        qi += 32;
        is += 2;
    }
    return float2(accG, accU);
}

// q4k_fused_gemv_swiglu: fused Q4_K Dequant-GEMV-SwiGLU kernel.
// Geometry: 32 threads per threadgroup (1 SIMD group per output row `o`), walking `nblk` super-blocks.
// Accumulates both Gate and Up dot products in registers (`accG`, `accU`).
// Performs `simd_sum` reduction across the 32 lanes.
// Lane 0 evaluates SwiGLU non-linearity in registers:
//   silu(accG) * accU = (accG / (1.0f + exp(-accG))) * accU
// Writes directly to Inter[o], completely eliminating intermediate DRAM writes/reads for separate Gate and Up.
kernel void q4k_fused_gemv_swiglu(device const uchar* WG   [[buffer(0)]],
                                  device const uchar* WU   [[buffer(1)]],
                                  device const float* X    [[buffer(2)]],
                                  device float*       Inter[[buffer(3)]],
                                  constant int&       nblk [[buffer(4)]],
                                  constant int&       out_ [[buffer(5)]],
                                  uint o   [[threadgroup_position_in_grid]],
                                  uint lid [[thread_index_in_threadgroup]]) {
    if (o >= (uint)out_) return;
    device const uchar* rowG = WG + (long)o * nblk * 144;
    device const uchar* rowU = WU + (long)o * nblk * 144;
    float accG = 0.0f;
    float accU = 0.0f;
    for (int b = (int)lid; b < nblk; b += 32) {
        device const float* xs = X + (long)b * 256;
        float2 dual = q4k_dual_block_dot(rowG + (long)b * 144, rowU + (long)b * 144, xs);
        accG += dual.x;
        accU += dual.y;
    }
    accG = simd_sum(accG);
    accU = simd_sum(accU);
    if (lid == 0) {
        float act = (accG / (1.0f + exp(-accG))) * accU;
        Inter[o] = act;
    }
}

// q8_fused_gemv_swiglu: fused Q8_0 Dequant-GEMV-SwiGLU kernel for float32 activation.
// Evaluates Gate and Up GEMVs simultaneously and computes SwiGLU in registers without global DRAM round-trips.
kernel void q8_fused_gemv_swiglu(device const char*  WG   [[buffer(0)]],  // out*in int8 codes, row-major
                                 device const float* WDG  [[buffer(1)]],  // out*nblk weight block-scales
                                 device const char*  WU   [[buffer(2)]],  // out*in int8 codes, row-major
                                 device const float* WDU  [[buffer(3)]],  // out*nblk weight block-scales
                                 device const float* X    [[buffer(4)]],  // in float32 activation
                                 device float*       Inter[[buffer(5)]],  // out intermediate
                                 constant int&       nblk [[buffer(6)]],
                                 constant int&       out_ [[buffer(7)]],
                                 uint o   [[threadgroup_position_in_grid]],
                                 uint lid [[thread_index_in_threadgroup]]) {
    if (o >= (uint)out_) return;
    device const char*  wrowG = WG  + (long)o * nblk * 32;
    device const float* wdG   = WDG + (long)o * nblk;
    device const char*  wrowU = WU  + (long)o * nblk * 32;
    device const float* wdU   = WDU + (long)o * nblk;
    float accG = 0.0f;
    float accU = 0.0f;
    for (int b = (int)lid; b < nblk; b += 32) {
        device const char*  wbG = wrowG + (long)b * 32;
        device const char*  wbU = wrowU + (long)b * 32;
        device const float* xb  = X     + (long)b * 32;
        float sG = 0.0f;
        float sU = 0.0f;
        for (int i = 0; i < 32; i++) {
            float xi = xb[i];
            sG += (float)wbG[i] * xi;
            sU += (float)wbU[i] * xi;
        }
        accG += sG * wdG[b];
        accU += sU * wdU[b];
    }
    accG = simd_sum(accG);
    accU = simd_sum(accU);
    if (lid == 0) {
        float act = (accG / (1.0f + exp(-accG))) * accU;
        Inter[o] = act;
    }
}

// q8_fused_gemv_swiglu_quant: fused Q8_0 Dequant-GEMV-SwiGLU kernel for Q8_0 quantized activation.
kernel void q8_fused_gemv_swiglu_quant(device const char*  WG   [[buffer(0)]],
                                       device const float* WDG  [[buffer(1)]],
                                       device const char*  WU   [[buffer(2)]],
                                       device const float* WDU  [[buffer(3)]],
                                       device const char*  X    [[buffer(4)]],
                                       device const float* XD   [[buffer(5)]],
                                       device float*       Inter[[buffer(6)]],
                                       constant int&       nblk [[buffer(7)]],
                                       constant int&       out_ [[buffer(8)]],
                                       uint o   [[threadgroup_position_in_grid]],
                                       uint lid [[thread_index_in_threadgroup]]) {
    if (o >= (uint)out_) return;
    device const char*  wrowG = WG  + (long)o * nblk * 32;
    device const float* wdG   = WDG + (long)o * nblk;
    device const char*  wrowU = WU  + (long)o * nblk * 32;
    device const float* wdU   = WDU + (long)o * nblk;
    float accG = 0.0f;
    float accU = 0.0f;
    for (int b = (int)lid; b < nblk; b += 32) {
        device const char* wbG = wrowG + (long)b * 32;
        device const char* wbU = wrowU + (long)b * 32;
        device const char* xb  = X     + (long)b * 32;
        int sG = 0;
        int sU = 0;
        for (int i = 0; i < 32; i++) {
            int xi = (int)xb[i];
            sG += (int)wbG[i] * xi;
            sU += (int)wbU[i] * xi;
        }
        float xscale = XD[b];
        accG += (float)sG * wdG[b] * xscale;
        accU += (float)sU * wdU[b] * xscale;
    }
    accG = simd_sum(accG);
    accU = simd_sum(accU);
    if (lid == 0) {
        float act = (accG / (1.0f + exp(-accG))) * accU;
        Inter[o] = act;
    }
}

// -----------------------------------------------------------------------------
// Down projection kernels (for self-contained fused MLP command buffer encoding)
// -----------------------------------------------------------------------------

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
