// mm_cm1_quant.glsl - tiled fp16 cooperative-matrix GEMM over packed quantized weights.
//
// Included by one mm_cm1_<dtype>.comp wrapper per weight format; the wrapper defines exactly
// one MM_CM1_<DTYPE> selector before the include. Y[t, r] = sum_k W[r, k] * X[t, k] for every
// token t < tokens, so one weight tile is dequantized once per BN tokens instead of once per
// token (the matvec route re-streams the full matrix for every token).
//
// Adapted from ggml-vulkan (MIT, Copyright (c) 2023-2026 The ggml authors), pinned at
// llama.cpp f74e2fa6a31cc33ab4ee4ae516641cfe62df3924:
//   ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp        (COOPMAT path: shared A/B tiles,
//       coopMatLoad row-major A / column-major B, per-warp accumulator grid, staged edge store)
//   ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl  (load_a_to_shmem: dequantize into the shared A tile)
//   ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl (block field decode)
//
// MIT License. Copyright (c) 2023-2026 The ggml authors.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
// to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Changed for Fak: fixed (not specialization-constant) tiles, scalar float16_t shared tiles,
// fak's three-binding ABI (raw packed weight words, F32 X[tokens][in], F32 Y[tokens][out]) and
// push constants {outDim, inDim, tokens}; one thread dequantizes one 16-weight half of a
// 32-weight sub-block straight from the verbatim GGUF bytes. The host dispatch is exactly
// (ceil(outDim / BM), ceil(tokens / BN), 1) with a required 32-lane subgroup.

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define BLOCK 256u
#define WARP 32u
#define BM 128u
#define BN 128u
#define BK 32u
#define WM 64u
#define WN 32u
#define TM 16u
#define TN 16u
#define TK 16u
#define STRIDE (BK + 8u)
#define CMS_ROW (WM / TM)
#define CMS_COL (WN / TN)
#define NUM_WARPS (BLOCK / WARP)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer W { uint raw[]; };
layout(std430, set = 0, binding = 1) readonly buffer X { vec4 x4[]; };
layout(std430, set = 0, binding = 2) writeonly buffer Y { float y[]; };

layout(push_constant) uniform PC {
    int outDim;
    int inDim;
    int tokens;
} pc;

shared float16_t bufA[BM * STRIDE];
shared float16_t bufB[BN * STRIDE];
shared float stage[TM * TN * NUM_WARPS];

uint u16at(uint off) { return (raw[off >> 2u] >> ((off & 2u) << 3u)) & 65535u; }
uint u32at(uint off) { return u16at(off) | (u16at(off + 2u) << 16u); }

uint u8at(uint off) { return (raw[off >> 2u] >> ((off & 3u) << 3u)) & 255u; }
float f16at(uint off) { return unpackHalf2x16(u16at(off)).x; }
// ksigns_iq2xs: 7 stored sign bits plus an even-parity eighth bit.
uint ksigns(uint s7) { return s7 | ((uint(bitCount(s7)) & 1u) << 7u); }
// Eight (put8) or four (put4) grid magnitudes, one per byte, times db with sign bits sg.
void put8(inout float vals[16], uint o, uvec2 g, uint sg, float db) {
    for (uint j = 0u; j < 8u; ++j) {
        float m = float(((j < 4u ? g.x : g.y) >> (8u * (j & 3u))) & 255u) * db;
        vals[o + j] = ((sg >> j) & 1u) != 0u ? -m : m;
    }
}
void put4(inout float vals[16], uint o, uint g, uint sg, float db) {
    for (uint j = 0u; j < 4u; ++j) {
        float m = float((g >> (8u * j)) & 255u) * db;
        vals[o + j] = ((sg >> j) & 1u) != 0u ? -m : m;
    }
}

#include "mm_cm1_iq_tables.glsl"

// dequantHalf(b, c, h, vals): the 16 weights [32c + 16h, 32c + 16h + 16) of the super-block
// at byte offset b, in order. Field layouts and decode follow the iq*_matvec.comp laneDot of
// the same format (sub-block c, half h), i.e. the ggml block_iq* / block_q*_K layouts.
#if defined(MM_CM1_Q2_K)
// Q2_K (84 B): scales[16] (lo nibble scale, hi nibble min), qs[64] (2-bit codes, four shifts
// per byte), d, dmin. Sub-block c reads qs[32 * (c / 4) + 16h ..] at shift 2 * (c % 4).
const uint BLOCK_BYTES = 84u;
void setupTables() {}
void dequantHalf(uint b, uint c, uint h, out float vals[16]) {
    uint wb = b >> 2u;
    vec2 dm = unpackHalf2x16(raw[wb + 20u]);
    uint sc = u8at(b + 2u * c + h);
    float dl = dm.x * float(sc & 15u);
    float ml = dm.y * float(sc >> 4u);
    uint shift = 2u * (c & 3u);
    uint qw = wb + 4u + (c >> 2u) * 8u + h * 4u;
    for (uint w = 0u; w < 4u; ++w) {
        uint q = raw[qw + w] >> shift;
        for (uint j = 0u; j < 4u; ++j) vals[4u * w + j] = dl * float((q >> (8u * j)) & 3u) - ml;
    }
}
#elif defined(MM_CM1_Q4_K)
// Q4_K (144 B): d, dmin, scales[12] (6-bit scale/min pairs), qs[128]. Sub-block c is the low
// (c even) or high (c odd) nibbles of qs[32 * (c / 2) ..].
const uint BLOCK_BYTES = 144u;
void setupTables() {}
void dequantHalf(uint b, uint c, uint h, out float vals[16]) {
    uint wb = b >> 2u;
    vec2 dm = unpackHalf2x16(raw[wb]);
    uint sc, mn;
    if (c < 4u) {
        sc = u8at(b + 4u + c) & 63u;
        mn = u8at(b + 8u + c) & 63u;
    } else {
        uint hi = u8at(b + 8u + c);
        sc = (hi & 15u) | ((u8at(b + c) >> 6u) << 4u);
        mn = (hi >> 4u) | ((u8at(b + 4u + c) >> 6u) << 4u);
    }
    float dl = dm.x * float(sc);
    float ml = dm.y * float(mn);
    uint shift = 4u * (c & 1u);
    uint qw = wb + 4u + (c >> 1u) * 8u + h * 4u;
    for (uint w = 0u; w < 4u; ++w) {
        uint q = raw[qw + w] >> shift;
        for (uint j = 0u; j < 4u; ++j) vals[4u * w + j] = dl * float((q >> (8u * j)) & 15u) - ml;
    }
}
#elif defined(MM_CM1_IQ2_XXS)
// IQ2_XXS (66 B): d, then per sub-block aux0 = four grid indices, aux1 = four 7-bit sign
// codes | scale << 28; db = d * (0.5 + scale) * 0.25.
const uint BLOCK_BYTES = 66u;
shared uvec2 iq2xxs_grid[256];
void setupTables() { iq2xxs_grid[gl_LocalInvocationID.x] = iq2xxs_grid_const[gl_LocalInvocationID.x]; }
void dequantHalf(uint b, uint c, uint h, out float vals[16]) {
    float d = f16at(b);
    uint aux0 = u32at(b + 2u + 8u * c);
    uint aux1 = u32at(b + 6u + 8u * c);
    float db = d * (0.5 + float(aux1 >> 28u)) * 0.25;
    for (uint k = 0u; k < 2u; ++k) {
        uint l = 2u * h + k;
        put8(vals, 8u * k, iq2xxs_grid[(aux0 >> (8u * l)) & 255u], ksigns((aux1 >> (7u * l)) & 127u), db);
    }
}
#elif defined(MM_CM1_IQ2_XS)
// IQ2_XS (74 B): d, qs u16[32] (9-bit grid index | 7-bit sign code << 9), scales[8] nibbles.
const uint BLOCK_BYTES = 74u;
shared uvec2 iq2xs_grid[512];
void setupTables() {
    for (uint i = gl_LocalInvocationID.x; i < 512u; i += BLOCK) iq2xs_grid[i] = iq2xs_grid_const[i];
}
void dequantHalf(uint b, uint c, uint h, out float vals[16]) {
    float d = f16at(b);
    uint sc = u8at(b + 66u + c);
    float db = d * (0.5 + float((sc >> (4u * h)) & 15u)) * 0.25;
    for (uint k = 0u; k < 2u; ++k) {
        uint q = u16at(b + 2u + 2u * (4u * c + 2u * h + k));
        put8(vals, 8u * k, iq2xs_grid[q & 511u], ksigns(q >> 9u), db);
    }
}
#elif defined(MM_CM1_IQ2_S)
// IQ2_S (82 B): d, qs[32] grid low bytes, signs[32], qh[8] (2 high index bits per group),
// scales[8] nibbles.
const uint BLOCK_BYTES = 82u;
shared uvec2 iq2s_grid[1024];
void setupTables() {
    for (uint i = gl_LocalInvocationID.x; i < 1024u; i += BLOCK) iq2s_grid[i] = iq2s_grid_const[i];
}
void dequantHalf(uint b, uint c, uint h, out float vals[16]) {
    float d = f16at(b);
    uint sc = u8at(b + 74u + c);
    uint qh = u8at(b + 66u + c);
    float db = d * (0.5 + float((sc >> (4u * h)) & 15u)) * 0.25;
    uint q2 = u16at(b + 2u + 4u * c + 2u * h);
    uint s2 = u16at(b + 34u + 4u * c + 2u * h);
    for (uint k = 0u; k < 2u; ++k) {
        uint l = 2u * h + k;
        uint idx = ((q2 >> (8u * k)) & 255u) | ((qh << (8u - 2u * l)) & 768u);
        put8(vals, 8u * k, iq2s_grid[idx], (s2 >> (8u * k)) & 255u, db);
    }
}
#elif defined(MM_CM1_IQ3_XXS)
// IQ3_XXS (98 B): d, qs[64] grid indices, then per sub-block u32 = four 7-bit sign codes |
// scale << 28; db = d * (0.5 + scale) * 0.5.
const uint BLOCK_BYTES = 98u;
shared uint iq3xxs_grid[256];
void setupTables() { iq3xxs_grid[gl_LocalInvocationID.x] = iq3xxs_grid_const[gl_LocalInvocationID.x]; }
void dequantHalf(uint b, uint c, uint h, out float vals[16]) {
    float d = f16at(b);
    uint aux = u32at(b + 66u + 4u * c);
    float db = d * (0.5 + float(aux >> 28u)) * 0.5;
    for (uint k = 0u; k < 2u; ++k) {
        uint l = 2u * h + k;
        uint sg = ksigns((aux >> (7u * l)) & 127u);
        uint q2 = u16at(b + 2u + 8u * c + 2u * l);
        put4(vals, 8u * k, iq3xxs_grid[q2 & 255u], sg, db);
        put4(vals, 8u * k + 4u, iq3xxs_grid[q2 >> 8u], sg >> 4u, db);
    }
}
#elif defined(MM_CM1_IQ3_S)
// IQ3_S (110 B): d, qs[64], qh[8] (9th index bit), signs[32], scales[4] nibbles;
// db = d * (1 + 2 * scale).
const uint BLOCK_BYTES = 110u;
shared uint iq3s_grid[512];
void setupTables() {
    for (uint i = gl_LocalInvocationID.x; i < 512u; i += BLOCK) iq3s_grid[i] = iq3s_grid_const[i];
}
void dequantHalf(uint b, uint c, uint h, out float vals[16]) {
    float d = f16at(b);
    uint sc = u8at(b + 106u + (c >> 1u));
    float db = d * float(1u + 2u * ((sc >> (4u * (c & 1u))) & 15u));
    uint qh = u8at(b + 66u + c);
    uint s2 = u16at(b + 74u + 4u * c + 2u * h);
    for (uint k = 0u; k < 2u; ++k) {
        uint l = 2u * h + k;
        uint q2 = u16at(b + 2u + 8u * c + 2u * l);
        uint i1 = (q2 & 255u) | ((qh << (8u - 2u * l)) & 256u);
        uint i2 = (q2 >> 8u) | ((qh << (7u - 2u * l)) & 256u);
        uint sg = (s2 >> (8u * k)) & 255u;
        put4(vals, 8u * k, iq3s_grid[i1], sg, db);
        put4(vals, 8u * k + 4u, iq3s_grid[i2], sg >> 4u, db);
    }
}
#elif defined(MM_CM1_IQ1_S)
// IQ1_S (50 B): d, qs[32], qh u16[8]; each group is eight signed 2-bit grid values plus
// delta = qh & 0x8000 ? -1/8 : 1/8, times dl = d * (2 * ((qh >> 12) & 7) + 1). The grid
// packs two 16-bit entries per word.
const uint BLOCK_BYTES = 50u;
shared uint iq1s_grid[1024];
void setupTables() {
    for (uint i = gl_LocalInvocationID.x; i < 1024u; i += BLOCK) iq1s_grid[i] = iq1s_grid_const[i];
}
void dequantHalf(uint b, uint c, uint h, out float vals[16]) {
    float d = f16at(b);
    uint qh = u16at(b + 34u + 2u * c);
    float dl = d * float(2u * ((qh >> 12u) & 7u) + 1u);
    float delta = (qh & 32768u) != 0u ? -0.125 : 0.125;
    uint q2 = u16at(b + 2u + 4u * c + 2u * h);
    for (uint k = 0u; k < 2u; ++k) {
        uint l = 2u * h + k;
        uint idx = ((q2 >> (8u * k)) & 255u) | (((qh >> (3u * l)) & 7u) << 8u);
        int g = int((iq1s_grid[idx >> 1u] >> (16u * (idx & 1u))) & 65535u);
        for (uint j = 0u; j < 8u; ++j) vals[8u * k + j] = dl * (float(bitfieldExtract(g, int(2u * j), 2)) + delta);
    }
}
#elif defined(MM_CM1_IQ4_XS)
// IQ4_XS (136 B, word aligned): d, scales_h u16 (2 bits per sub-block), scales_l[4] nibbles,
// qs[128]; sub-block c is the low (h = 0) then high (h = 1) nibbles of qs[16c ..], each a
// kvalues_iq4nl index, times dl = d * (scale - 32).
const uint BLOCK_BYTES = 136u;
shared float kv[16];
const float KVALUES[16] = float[16](-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
                                    1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0);
void setupTables() {
    if (gl_LocalInvocationID.x < 16u) kv[gl_LocalInvocationID.x] = KVALUES[gl_LocalInvocationID.x];
}
void dequantHalf(uint b, uint c, uint h, out float vals[16]) {
    uint wb = b >> 2u;
    uint w0 = raw[wb];
    float d = unpackHalf2x16(w0).x;
    uint sl = (raw[wb + 1u] >> (8u * (c >> 1u) + 4u * (c & 1u))) & 15u;
    float dl = d * float(int(sl | (((w0 >> (16u + 2u * c)) & 3u) << 4u)) - 32);
    for (uint k = 0u; k < 4u; ++k) {
        uint w = raw[wb + 2u + c * 4u + k] >> (4u * h);
        for (uint j = 0u; j < 4u; ++j) vals[4u * k + j] = dl * kv[(w >> (8u * j)) & 15u];
    }
}
#else
#error "mm_cm1_quant.glsl requires one MM_CM1_<DTYPE> selector"
#endif

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint warp = gl_SubgroupID;
    uint lane = gl_SubgroupInvocationID;
    uint warpR = warp % (BM / WM);
    uint warpC = warp / (BM / WM);

    uint row0 = gl_WorkGroupID.x * BM;
    uint tok0 = gl_WorkGroupID.y * BN;
    uint outDim = uint(pc.outDim);
    uint tokens = uint(pc.tokens);
    uint inDim = uint(pc.inDim);

    setupTables();
    barrier();

    coopmat<float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cacheA;
    coopmat<float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> cacheB[CMS_COL];
    coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> sums[CMS_ROW * CMS_COL];
    for (uint i = 0u; i < CMS_ROW * CMS_COL; ++i) {
        sums[i] = coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0);
    }

    // Each thread owns one (row or token, 16-wide K half) of the BM x BK and BN x BK tiles.
    uint lr = tid >> 1u;
    uint lh = tid & 1u;
    uint aRow = row0 + lr;
    uint bTok = tok0 + lr;
    uint xRow4 = bTok * (inDim >> 2u);

    for (uint kb = 0u; kb < inDim; kb += BK) {
        float vals[16];
        if (aRow < outDim) {
            dequantHalf((aRow * (inDim >> 8u) + (kb >> 8u)) * BLOCK_BYTES, (kb >> 5u) & 7u, lh, vals);
        } else {
            for (uint j = 0u; j < 16u; ++j) vals[j] = 0.0;
        }
        for (uint j = 0u; j < 16u; ++j) bufA[lr * STRIDE + lh * 16u + j] = float16_t(vals[j]);

        if (bTok < tokens) {
            uint xi = xRow4 + ((kb + lh * 16u) >> 2u);
            for (uint j = 0u; j < 4u; ++j) {
                vec4 v = x4[xi + j];
                uint o = lr * STRIDE + lh * 16u + 4u * j;
                bufB[o + 0u] = float16_t(v.x);
                bufB[o + 1u] = float16_t(v.y);
                bufB[o + 2u] = float16_t(v.z);
                bufB[o + 3u] = float16_t(v.w);
            }
        } else {
            for (uint j = 0u; j < 16u; ++j) bufB[lr * STRIDE + lh * 16u + j] = float16_t(0.0);
        }
        barrier();

        for (uint i = 0u; i < BK; i += TK) {
            for (uint cc = 0u; cc < CMS_COL; ++cc) {
                coopMatLoad(cacheB[cc], bufB, (warpC * WN + cc * TN) * STRIDE + i, STRIDE,
                            gl_CooperativeMatrixLayoutColumnMajor);
            }
            for (uint cr = 0u; cr < CMS_ROW; ++cr) {
                coopMatLoad(cacheA, bufA, (warpR * WM + cr * TM) * STRIDE + i, STRIDE,
                            gl_CooperativeMatrixLayoutRowMajor);
                for (uint cc = 0u; cc < CMS_COL; ++cc) {
                    sums[cc * CMS_ROW + cr] = coopMatMulAdd(cacheA, cacheB[cc], sums[cc * CMS_ROW + cr]);
                }
            }
        }
        barrier();
    }

    // Accumulator element (i, j) is output row dr + i of token dc + j, so a column-major store
    // with stride outDim lands at Y[(dc + j) * outDim + dr + i].
    uint dr = row0 + warpR * WM;
    uint dc = tok0 + warpC * WN;
    bool aligned = (outDim & 3u) == 0u;
    uint storeR = lane % TM;
    uint storeC = lane / TM;
    for (uint cr = 0u; cr < CMS_ROW; ++cr) {
        for (uint cc = 0u; cc < CMS_COL; ++cc) {
            uint r = dr + cr * TM;
            uint t = dc + cc * TN;
            bool full = r + TM <= outDim && t + TN <= tokens;
            if (full && aligned) {
                coopMatStore(sums[cc * CMS_ROW + cr], y, t * outDim + r, outDim,
                             gl_CooperativeMatrixLayoutColumnMajor);
            } else if (r < outDim && t < tokens) {
                coopMatStore(sums[cc * CMS_ROW + cr], stage, warp * TM * TN, TM,
                             gl_CooperativeMatrixLayoutColumnMajor);
                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared,
                               gl_SemanticsAcquireRelease);
                for (uint col = 0u; col < TN; col += WARP / TM) {
                    uint rr = r + storeR;
                    uint tt = t + col + storeC;
                    if (rr < outDim && tt < tokens) {
                        y[tt * outDim + rr] = stage[warp * TM * TN + (col + storeC) * TM + storeR];
                    }
                }
                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared,
                               gl_SemanticsAcquireRelease);
            }
        }
    }
}
