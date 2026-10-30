// Q3_K decode arithmetic direct-ported from ggml-org/llama.cpp
// commit 3c809f44383bcebc318b8cddd8c6d11aaa861e9b,
// ggml/src/ggml-cuda/dequantize.cuh::dequantize_q3_K.
// FAK adaptation: byte-addressed GGUF blocks; fuse decode into F32 GEMV/token
// panels rather than materializing dequantized weights. No SM100-only ISA.
/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once

// Exact GGUF Q3_K ABI: hmask[32], qs[64], scales[12], little-endian f16 d.
// Byte-addressing avoids assuming 4-byte alignment for 110-byte blocks.
static __device__ __forceinline__ float fak_q3k_value(const uint8_t *b, int i) {
    const int n = i / 128;
    const int j = (i % 128) / 32;
    const int l = i % 32;
    const int is = 8*n + 2*j + l/16;
    const uint8_t *scales = b + 96;
    const int us = is < 4 ? (scales[is] & 0xF) | (((scales[is+8] >> 0) & 3) << 4) :
                   is < 8 ? (scales[is] & 0xF) | (((scales[is+4] >> 2) & 3) << 4) :
                   is < 12 ? (scales[is-8] >> 4) | (((scales[is] >> 4) & 3) << 4) :
                             (scales[is-8] >> 4) | (((scales[is-4] >> 6) & 3) << 4);
    const unsigned short dh = (unsigned short)b[108] | ((unsigned short)b[109] << 8);
    const float dl = __half2float(__ushort_as_half(dh)) * (us - 32);
    const int code = ((b[32 + 32*n + l] >> (2*j)) & 3) - ((b[l] & (1 << (4*n+j))) ? 0 : 4);
    return dl * code;
}

// Eight independent output warps, four token rows sharing each weight decode.
// Each block exits only at a whole-warp boundary; shuffle uses all 32 lanes.
template<int token_tile>
__global__ void k_q3k_panel(const uint8_t *W, const float *X, float *Y,
                            int out, int in, int P) {
    const int lane = threadIdx.x & 31;
    const int o = (int)(blockIdx.x * 8u) + (threadIdx.x >> 5);
    const int token0 = (int)(blockIdx.y * token_tile);
    if (o >= out || token0 >= P) return;
    float sums[token_tile] = {};
    const int nsb = in / 256;
    const uint8_t *wr = W + (size_t)o * nsb * 110;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t *b = wr + (size_t)sb * 110;
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            const float w = fak_q3k_value(b, 32*g + lane);
            #pragma unroll
            for (int t = 0; t < token_tile; ++t) {
                if (token0 + t < P) sums[t] += w * X[(size_t)(token0+t)*in + (size_t)sb*256 + 32*g + lane];
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < token_tile; ++t) {
        for (int offset = 16; offset; offset >>= 1) sums[t] += __shfl_down_sync(0xffffffffu, sums[t], offset);
        if (lane == 0 && token0+t < P) Y[(size_t)(token0+t)*out + o] = sums[t];
    }
}

// The f32 arm of fcuda_q3k_matmul_f32 (cuda/q3k_mmvq.cuh dispatches): decode, panels below 128.
static int fcuda_q3k_panel_f32(const uint8_t *W, const float *X, float *Y,
                               int out, int in, int P) {
    if (!W || !X || !Y || out <= 0 || in <= 0 || P <= 0 || in % 256 != 0)
        return (int)cudaErrorInvalidValue;
    const unsigned int tile = P >= 4 ? 4u : 1u;
    const unsigned int gy = ((unsigned int)P + tile - 1) / tile;
    if (gy > 65535u) return (int)cudaErrorInvalidConfiguration;
    const dim3 grid(((unsigned int)out + 7) / 8, gy);
    // No scratch allocation and no host fallback. Caller records native failures.
    if (tile == 4) {
        FCUDA_TS("k_q3k_panel", grid.x*grid.y, k_q3k_panel<4><<<grid,256,0,g_stream>>>(W,X,Y,out,in,P));
    } else {
        FCUDA_TS("k_q3k_panel", grid.x*grid.y, k_q3k_panel<1><<<grid,256,0,g_stream>>>(W,X,Y,out,in,P));
    }
    return (int)cudaGetLastError();
}
