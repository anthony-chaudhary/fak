// SPDX-License-Identifier: Apache-2.0
// FAK's stored Engram row contract, shared by the checked CUDA family.
// Packed integer arithmetic is adapted without numerical changes from
// shaders/v41_engram_row_decode.comp at 04257226173c7871d96d3d77dc5975b8a61fb93d.
// Q2_K and IQ4_NL use the existing MIT-attributed llama.cpp helpers instead
// of a second decoder/codebook. CUDA launch, stream and ownership stay in FAK.
#ifndef FAK_V41_ENGRAM_ROWS_H
#define FAK_V41_ENGRAM_ROWS_H
#include "v41_embed_iq4nl.h"

#ifdef __CUDACC__
#define V41_ROWS_HD __host__ __device__ __forceinline__
#else
#define V41_ROWS_HD inline
#endif

// This is the Go Engram encoding namespace, not embedding ABI2's modes or
// Vulkan's native poison mode. The original embedding ABIs are unchanged.
V41_ROWS_HD int v41_eg_rows_stride(int mode) {
  return mode==0?264:mode==1?84:mode==2?1024:mode==3?144:0;
}
V41_ROWS_HD uint32_t v41_eg_rows_word(const uint8_t *p) {
  return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
V41_ROWS_HD uint32_t v41_eg_rows_bf16(uint32_t bits) {
  if((bits&0x7f800000u)!=0x7f800000u)bits+=0x7fffu+((bits>>16)&1u);
  return bits&0xffff0000u;
}
// The FAK packed row is 256 E4M3FN bytes followed by eight E8M0 bytes.
// The product is constructed in integer arithmetic, including subnormals,
// then BF16 RNE is applied. Do not substitute the donor's unrounded F32
// output or allow a compiler's FTZ/fast-math setting to alter these bits.
// kind: 0 NaN byte, 1 nonfinite decoded value, 2 clean.
V41_ROWS_HD uint32_t v41_eg_rows_packed(const uint8_t *row,uint32_t j,uint32_t *kind) {
  const uint32_t code=row[j],scale=row[256+(j>>5)];
  *kind=2;
  if((code&0x7fu)==0x7fu||scale==0xffu){*kind=0;return 0;}
  const uint32_t sign=(code&0x80u)<<24,e=(code>>3)&15u;
  const uint32_t mant=e==0?(code&7u):(8u|(code&7u));
  const int q=e==0?-9:(int)e-10;
  uint32_t bits=sign;
  if(mant!=0) {
    const int k=q+(int)scale-127;
    const int msb=mant>=8?3:mant>=4?2:mant>=2?1:0;
    const int x=k+msb;
    if(x>127)bits=sign|0x7f800000u;
    else if(x>=-126)bits=sign|((uint32_t)(x+127)<<23)|((mant<<(23-msb))&0x7fffffu);
    else bits=sign|(mant<<(k+149)); // minimum k is -136, hence shift >=13
  }
  bits=v41_eg_rows_bf16(bits);
  if((bits&0x7f800000u)==0x7f800000u)*kind=1;
  return bits;
}
#undef V41_ROWS_HD

#endif
