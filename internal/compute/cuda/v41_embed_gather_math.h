// Portions adapted from llama.cpp, MIT (c) 2023-2026 The ggml authors.
// See v41_embed_gather.LICENSE and docs/cuda-v41-embedding-port.md.
#ifndef FAK_V41_EMBED_GATHER_MATH_H
#define FAK_V41_EMBED_GATHER_MATH_H
#include <stdint.h>
#ifdef __CUDACC__
#define V41_EG_HD __host__ __device__ __forceinline__
#else
#define V41_EG_HD inline
#endif
V41_EG_HD float v41_eg_half(uint16_t h) {
  uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(h>>10)&31, mant=h&1023, bits;
  if(exp==0) {
    if(mant==0)bits=sign;
    else {int e=-14;while(!(mant&1024)){mant<<=1;--e;}bits=sign|((uint32_t)(e+127)<<23)|((mant&1023)<<13);}
  } else if(exp==31)bits=sign|0x7f800000u|(mant<<13);
  else bits=sign|((exp+112)<<23)|(mant<<13);
  union {uint32_t u;float f;} v;v.u=bits;return v.f;
}
V41_EG_HD float v41_eg_mul(float a,float b) {
#ifdef __CUDA_ARCH__
  return __fmul_rn(a,b);
#else
  volatile float r=a*b;return r;
#endif
}
V41_EG_HD float v41_eg_sub(float a,float b) {
#ifdef __CUDA_ARCH__
  return __fsub_rn(a,b);
#else
  volatile float r=a-b;return r;
#endif
}
// Direct adaptation of dequantize_q2_K's 64-thread, four-values/thread map.
// Byte loads preserve raw GGUF layout without C struct packing assumptions.
// Unlike upstream's fused expression, Fak's host lookup explicitly uses 0-ml
// for q==0 and separate rounded products; preserve signed-zero/Inf behavior.
V41_EG_HD void v41_eg_q2_thread(const uint8_t *b,float *yy,int tid) {
  const int n=tid/32,l=tid-32*n,is=8*n+l/16;
  const uint8_t q=b[16+32*n+l];
  const float d=v41_eg_half((uint16_t)(b[80]|((uint16_t)b[81]<<8)));
  const float m=v41_eg_half((uint16_t)(b[82]|((uint16_t)b[83]<<8)));
  for(int j=0;j<4;++j) {
    const uint8_t sc=b[is+2*j], quant=(q>>(2*j))&3;
    const float dl=v41_eg_mul(d,(float)(sc&15)),ml=v41_eg_mul(m,(float)(sc>>4));
    const float prod=quant==0?0.0f:quant==1?dl:v41_eg_mul(dl,(float)quant);
    yy[128*n+l+32*j]=v41_eg_sub(prod,ml);
  }
}
// llama.cpp e0f142d5 dequantize_q5_K: 64 lanes, four values per lane.
// GGUF Q5_K is exactly 176 bytes: d/dmin at 0/2, packed scales at 4,
// high-bit planes at 16, and low nibbles at 48 (256 values per block).
// Preserve model.q5kDequantSuperBlock's separate F32 products/subtraction;
// unlike Q2_K, Q5_K does not special-case a zero quantized code.
V41_EG_HD void v41_eg_q5_scale_min(const uint8_t *s,int j,uint8_t &scale,uint8_t &minimum) {
  if(j<4) {scale=s[j]&63;minimum=s[j+4]&63;}
  else {scale=(s[j+4]&15)|((s[j-4]>>6)<<4);minimum=(s[j+4]>>4)|((s[j]>>6)<<4);}
}
V41_EG_HD void v41_eg_q5_thread(const uint8_t *b,float *yy,int tid) {
  const int il=tid/16,ir=tid%16,is=2*il;
  const float d=v41_eg_half((uint16_t)(b[0]|((uint16_t)b[1]<<8)));
  const float m=v41_eg_half((uint16_t)(b[2]|((uint16_t)b[3]<<8)));
  const uint8_t *ql=b+48+32*il+2*ir,*qh=b+16+2*ir;
  uint8_t sc,mn;
  v41_eg_q5_scale_min(b+4,is,sc,mn);
  const float d1=v41_eg_mul(d,(float)sc),m1=v41_eg_mul(m,(float)mn);
  v41_eg_q5_scale_min(b+4,is+1,sc,mn);
  const float d2=v41_eg_mul(d,(float)sc),m2=v41_eg_mul(m,(float)mn);
  const uint8_t hm=(uint8_t)(1<<(2*il));
  float *y=yy+64*il+2*ir;
  for(int l=0;l<2;++l) {
    const uint8_t lo=(ql[l]&15)+((qh[l]&hm)?16:0);
    const uint8_t hi=(ql[l]>>4)+((qh[l]&(hm<<1))?16:0);
    y[l]=v41_eg_sub(v41_eg_mul(d1,(float)lo),m1);
    y[l+32]=v41_eg_sub(v41_eg_mul(d2,(float)hi),m2);
  }
}
#undef V41_EG_HD
#endif
