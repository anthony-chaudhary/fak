// SPDX-License-Identifier: Apache-2.0
// Direct FAK shaders/v41_engram_gate.comp port; see PROVENANCE.md.
#ifdef FAK_CUDA_V41_ENGRAM_GATE
__device__ static float engram_exp(float x) {
  if(x>88.72283935546875f) return __uint_as_float(0x7f800000u);
  if(x<-103.97207641601562f) return 0.f;
  float n=nearbyintf(__fmul_rn(x,1.4426950408889634f));
  float r0=__fmaf_rn(n,-0.693359375f,x);
  float r=__fmaf_rn(n,2.12194440e-4f,r0);
  float p=1.9875691500e-4f;
  p=__fmaf_rn(p,r,1.3981999507e-3f);
  p=__fmaf_rn(p,r,8.3334519073e-3f);
  p=__fmaf_rn(p,r,4.1665795894e-2f);
  p=__fmaf_rn(p,r,1.6666665459e-1f);
  p=__fmaf_rn(p,r,5.0000001201e-1f);
  float e=__fadd_rn(__fmaf_rn(p,__fmul_rn(r,r),r),1.f);
  int k=(int)n;
  if(k>127) { e=__fmul_rn(e,2.f); --k; }
  if(k<-126) {
    float tiny=__fmul_rn(e,__uint_as_float(0x00800000u));
    // k+126 is in [-24,-1]; exact power of two, no approximate exp2.
    return __fmul_rn(tiny,__uint_as_float((unsigned)(k+253)<<23));
  }
  return __fmul_rn(e,__uint_as_float((unsigned)(k+127)<<23));
}
__device__ static float engram_bf16(float v){
 unsigned u=__float_as_uint(v);
 if((u&0x7f800000u)!=0x7f800000u) u+=0x7fffu+((u>>16)&1u);
 return __uint_as_float(u&0xffff0000u);
}
__device__ static float engram_rsqrt_host(float v){
 unsigned bits=__float_as_uint(v);
 if(bits==0x80000000u)return __uint_as_float(0xff800000u);
 if(bits>0x7f800000u)return __uint_as_float(0x7fc00000u);
 return __double2float_rn(__ddiv_rn(1.0,__dsqrt_rn((double)v)));
}
__global__ static void k_v41_engram(const float *s,const float *p,const float *n,float *out,float *status,int rows,int h,float eps){
 // Scalar owner per stream preserves every serial oracle reduction. A later
 // independently proved parallel publication can optimize this baseline.
 for(size_t g=(size_t)blockIdx.x*blockDim.x+threadIdx.x;g<(size_t)rows*4;g+=(size_t)gridDim.x*blockDim.x){
  size_t stream=g*h,row=g/4,st=g%4,keybase=(row*5+st)*h,valuebase=(row*5+4)*h;
  float h2=0.f,k2=0.f,dot=0.f;bool bad=false;
  for(int i=0;i<h;++i){
   float hi=s[stream+i],key=engram_bf16(p[keybase+i]);
   bad=bad||!isfinite(key)||!isfinite(engram_bf16(p[valuebase+i]));
   h2=__fadd_rn(h2,__fmul_rn(hi,hi));k2=__fadd_rn(k2,__fmul_rn(key,key));
   float qk=__fmul_rn(n[st*h+i],n[(4+st)*h+i]);
   dot=__fadd_rn(dot,__fmul_rn(__fmul_rn(hi,qk),key));
  }
  float yh=__fadd_rn(__fdiv_rn(h2,(float)h),eps),yk=__fadd_rn(__fdiv_rn(k2,(float)h),eps);
  dot=__fmul_rn(__fmul_rn(__fmul_rn(dot,engram_rsqrt_host(yh)),engram_rsqrt_host(yk)),engram_rsqrt_host((float)h));
  float a=fabsf(dot),floorv=__uint_as_float(0x358637bdu);
  float mag=__double2float_rn(__dsqrt_rn((double)(a>floorv?a:floorv)));
  float z=__uint_as_float((__float_as_uint(mag)&0x7fffffffu)|(__float_as_uint(dot)&0x80000000u));
  float den=__fadd_rn(1.f,engram_exp(-z));
  float gate=isinf(den)?0.f:__fdiv_rn(1.f,den);
  for(int i=0;i<h;++i)out[stream+i]=engram_bf16(__fadd_rn(s[stream+i],__fmul_rn(gate,engram_bf16(p[valuebase+i]))));
  status[g]=bad?1.f:0.f;
 }
}
extern "C" int fak_v41_engram_state(void) {
  if(!g_blas||!g_stream||g_capture_open||v41_candidates_stream_uncertain()) return -3;
  cudaError_t e=cudaPeekAtLastError(); if(e!=cudaSuccess) return 10000+(int)e;
  int device=-1; e=cudaGetDevice(&device); if(e!=cudaSuccess) return 10000+(int)e;
  if(device!=0) return -3;
  cudaStreamCaptureStatus capture; e=cudaStreamIsCapturing(g_stream,&capture);
  if(e!=cudaSuccess) return 10000+(int)e;
  if(capture!=cudaStreamCaptureStatusNone) return -3;
  e=cudaStreamQuery(g_stream); if(e==cudaErrorNotReady) return -3;
  return e==cudaSuccess?0:10000+(int)e;
}
extern "C" int fak_v41_engram_input(const float *p,uint64_t count) {
  int state=fak_v41_engram_state(); if(state) return state;
  if(!p||!count||count>UINT32_MAX||count>SIZE_MAX/sizeof(float)) return -2;
  auto it=g_live.find((void*)p); if(it==g_live.end()||it->second<count*sizeof(float)) return -2;
  cudaPointerAttributes attr={}; cudaError_t e=cudaPointerGetAttributes(&attr,p);
  if(e!=cudaSuccess) return 11000+(int)e;
  return attr.type==cudaMemoryTypeDevice&&attr.device==0?0:-2;
}
extern "C" int fak_v41_engram_run(const float *s,const float *p,const float *n,float *out,float *status,int rows,int h,float eps){
 if(rows<=0||h<=0||h>65536||!isfinite(eps)||(uint64_t)rows*5*h>INT32_MAX)return -1;
 const float *ptrs[5]={s,p,n,out,status};uint64_t counts[5]={(uint64_t)rows*4*h,(uint64_t)rows*5*h,(uint64_t)8*h,(uint64_t)rows*4*h,(uint64_t)rows*4};
 for(int i=3;i<5;++i)for(int j=0;j<i;++j)if(ptrs[i]==ptrs[j])return -2;
 for(int i=0;i<5;++i){int x=fak_v41_engram_input(ptrs[i],counts[i]);if(x)return x;}
 unsigned blocks=(unsigned)(((uint64_t)rows*4+127)/128);if(blocks>65535)blocks=65535;
 k_v41_engram<<<blocks,128,0,g_stream>>>(s,p,n,out,status,rows,h,eps);
 cudaError_t e=cudaGetLastError();int x=e==cudaSuccess?0:30000+(int)e;
 if(!x){e=cudaStreamSynchronize(g_stream);if(e!=cudaSuccess)x=40000+(int)e;}
 if(x){e=cudaStreamSynchronize(g_stream);if(e!=cudaSuccess){for(int i=0;i<5;++i)g_v41_engram_quarantine[i]=ptrs[i];g_v41_engram_uncertain=true;return 50000+(int)e;}return x;}
 return 0;
}
// Checked witness readback; a failed drain pins the source, never certifies bytes.
extern "C" int fak_v41_engram_readback(float *host,const float *device,uint64_t count) {
  if(!host) return -2;
  int status=fak_v41_engram_input(device,count); if(status) return status;
  int copied=fcuda_d2h_checked(host,device,(size_t)count*sizeof(float));
  if(copied) {
    cudaError_t e=cudaStreamSynchronize(g_stream);
    if(e!=cudaSuccess) { g_v41_engram_quarantine[0]=device; g_v41_engram_uncertain=true; return 50000+(int)e; }
    return 59000+copied;
  }
  return 0;
}
#endif
