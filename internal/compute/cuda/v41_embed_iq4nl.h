// IQ4_NL math adapted from ggml-org/llama.cpp at
// e0f142d5ddc1461ce721644d92af33db23545105:
// ggml/src/ggml-cuda/dequantize.cuh:408-421 (dequantize_iq4_nl),
// ggml/src/ggml-common.h:447-452,1120-1122 (block layout/codebook).
// Changes: raw byte loads, exact half widening and separate rounded F32
// multiplication reuse Fak's shared helpers. The caller supplies the existing
// fcuda_kvalues_iq4nl codebook; no duplicate device table/runtime is imported.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#ifndef FAK_V41_EMBED_IQ4NL_H
#define FAK_V41_EMBED_IQ4NL_H

#include "v41_embed_gather_math.h"
#include <limits.h>

#ifdef __CUDACC__
#define V41_IQ4NL_HD __host__ __device__ __forceinline__
#else
#define V41_IQ4NL_HD inline
#endif

// 32 lanes decode eight 18-byte/32-value blocks (one 256-value panel).
// GGUF IQ4_NL is split-half, NOT adjacent: byte j holds values j and j+16.
// The native launch must use exactly 32 threads; tid is in [0,32).
V41_IQ4NL_HD void v41_eg_iq4nl_thread(const uint8_t *b,float *yy,int tid,const signed char *values) {
  const int il=tid/8,ib=tid%8;
  const uint8_t *x=b+18*ib,*q4=x+2+4*il;
  float *y=yy+32*ib+4*il;
  const float d=v41_eg_half((uint16_t)(x[0]|((uint16_t)x[1]<<8)));
  for(int j=0;j<4;++j) {
    y[j]=v41_eg_mul(d,(float)values[q4[j]&15]);
    y[j+16]=v41_eg_mul(d,(float)values[q4[j]>>4]);
  }
}

// Fak-owned glue (Apache-2.0, repository LICENSE); the MIT notice above is
// retained for the upstream decoder. Direct adaptation of Fak's
// shaders/v41_engram_row_decode.comp finiteBits,
// first-fault reduction and poison word. Keep the existing Go row contract:
// UINT32_MAX is the private reduction identity; published status is 0 or
// 1+2*j+kind, with kind=1 for every IQ4_NL non-finite decoded value. Unlike
// packed E4M3, an IQ4_NL scale NaN is not a separate kind=0 byte fault.
// Inspect bits rather than isfinite, independently of fast-math assumptions.
static_assert(sizeof(float)==4&&sizeof(uint32_t)==4,"Engram row status requires binary32 words");
V41_IQ4NL_HD uint32_t v41_eg_row_fault_key(float value,uint32_t j) {
  union {float f;uint32_t u;} v;v.f=value;
  return (v.u&0x7f800000u)==0x7f800000u?2*j+1:UINT32_MAX;
}
V41_IQ4NL_HD uint32_t v41_eg_row_fault_word(uint32_t key) {
  return key==UINT32_MAX?0:key+1;
}
V41_IQ4NL_HD void v41_eg_row_poison(float *target,uint32_t seen) {
  if(seen) {
    union {uint32_t u;float f;} v;v.u=0x7fc00000u;
    *target=v.f;
  }
}

#undef V41_IQ4NL_HD

// Separate ABI geometry: do not extend or reinterpret ABI2's mode 0/1/2.
// Signed-int32 encoded table/output extents match the existing byte allocator.
// Every ID is refused before native state lookup, allocation or launch.
static inline bool v41_eg_iq4nl_geometry(const intptr_t *ids,int rows,int hidden,int vocab,uint64_t *rb,uint64_t *ob) {
  if(!ids||!rb||!ob||rows<=0||hidden<=0||hidden%256||vocab<=0)return false;
  const uint64_t rowbytes=(uint64_t)(hidden/256)*144;
  const uint64_t outputbytes=(uint64_t)rows*(uint64_t)hidden*4;
  if(outputbytes>INT_MAX||rowbytes*(uint64_t)vocab>INT_MAX)return false;
  for(int r=0;r<rows;++r)if(ids[r]<0||ids[r]>=vocab)return false;
  *rb=rowbytes;*ob=outputbytes;return true;
}

// Checked Engram-shaped rows are deliberately narrower than the original
// embedding gather: fixed 144-byte/256-value rows and a [1,256] output prefix.
// This matches V41EngramRowDecodeMaxDim and keeps its status words <=512.
// Pure admission: no output arguments change on refusal, and all IDs are
// checked before the wrapper can inspect native state or launch any work.
static inline bool v41_eg_iq4nl_rows_geometry(const intptr_t *ids,int rows,int dim,int vocab,uint64_t *tb,uint64_t *ob,uint64_t *sb) {
  if(!ids||!tb||!ob||!sb||rows<=0||dim<=0||dim>256||vocab<=0)return false;
  const uint64_t tablebytes=(uint64_t)vocab*144;
  const uint64_t outputbytes=(uint64_t)rows*(uint64_t)dim*4;
  const uint64_t statusbytes=(uint64_t)rows*4;
  if(tablebytes>INT_MAX||outputbytes>INT_MAX||statusbytes>INT_MAX)return false;
  for(int r=0;r<rows;++r)if(ids[r]<0||ids[r]>=vocab)return false;
  *tb=tablebytes;*ob=outputbytes;*sb=statusbytes;return true;
}
static inline bool v41_eg_row_poison_geometry(int rows,int count,int index) {
  return rows>0&&count>0&&index>=0&&index<count&&
    (uint64_t)rows*4<=INT_MAX&&(uint64_t)count*4<=INT_MAX;
}

#endif
