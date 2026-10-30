// SPDX-License-Identifier: Apache-2.0
// Ported from FAK shaders/v41_compressor_pool.comp. See PROVENANCE.md.
// Included by cuda_kernels.cu; no fast math, FTZ, or contraction permitted.
#ifdef FAK_CUDA_V41_COMPRESSOR_POOL
__device__ static float v41_pool_exp(float x) {
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
__global__ static void k_v41_pool(const float *kv,const float *sc,float *out,int ratio,int width) {
 for(size_t d=(size_t)blockIdx.x*blockDim.x+threadIdx.x;d<(size_t)width;d+=(size_t)gridDim.x*blockDim.x) {
  float mx=sc[d];
  for(int t=0;t<ratio;++t) {
   size_t j=(size_t)t*width+d;
   if(!isfinite(sc[j])||!isfinite(kv[j])) { atomicExch((int*)(out+width),1); return; }
   if(sc[j]>mx) mx=sc[j];
  }
  float denom=0.f;
  for(int t=0;t<ratio;++t) denom=__fadd_rn(denom,v41_pool_exp(__fsub_rn(sc[(size_t)t*width+d],mx)));
  float acc=0.f;
  for(int t=0;t<ratio;++t) {
   size_t j=(size_t)t*width+d;
   float w=v41_pool_exp(__fsub_rn(sc[j],mx));
   acc=__fadd_rn(acc,__fmul_rn(kv[j],__fdiv_rn(w,denom)));
  }
  if(!isfinite(acc)) atomicExch((int*)(out+width),1);
  out[d]=acc;
 }
}
extern "C" int fak_v41_pool_state(void) {
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
extern "C" int fak_v41_pool_input(const float *p,uint64_t count) {
  int state=fak_v41_pool_state(); if(state) return state;
  if(!p||!count||count>UINT32_MAX||count>SIZE_MAX/sizeof(float)) return -2;
  auto it=g_live.find((void*)p); if(it==g_live.end()||it->second<count*sizeof(float)) return -2;
  cudaPointerAttributes attr={}; cudaError_t e=cudaPointerGetAttributes(&attr,p);
  if(e!=cudaSuccess) return 11000+(int)e;
  return attr.type==cudaMemoryTypeDevice&&attr.device==0?0:-2;
}
extern "C" int fak_v41_pool_run(const float *kv,const float *sc,float *out,int ratio,int width) {
 if(ratio<2||ratio>4096||width<=0||(uint64_t)ratio*width>INT32_MAX) return -1;
 if(out==kv||out==sc) return -2;
 const float *ptrs[3]={kv,sc,out};
 uint64_t counts[3]={(uint64_t)ratio*width,(uint64_t)ratio*width,(uint64_t)width+1};
 for(int i=0;i<3;++i) { int s=fak_v41_pool_input(ptrs[i],counts[i]); if(s) return s; }
 cudaError_t e;
 FCUDA_TS("fill", dim3(0, 0), e = cudaMemsetAsync(out + width, 0, sizeof(int), g_stream));
 int status=e==cudaSuccess?0:20000+(int)e;
 if(!status) {
  unsigned blocks=((unsigned)width+127)/128; if(blocks>65535) blocks=65535;
  FCUDA_TS("k_v41_pool", blocks, k_v41_pool<<<blocks,128,0,g_stream>>>(kv,sc,out,ratio,width));
  e=cudaGetLastError(); if(e!=cudaSuccess) status=30000+(int)e;
 }
 if(!status) { e=cudaStreamSynchronize(g_stream); if(e!=cudaSuccess) status=40000+(int)e; }
 if(status) {
  e=cudaStreamSynchronize(g_stream);
  if(e!=cudaSuccess) { for(int i=0;i<3;++i) g_v41_pool_quarantine[i]=ptrs[i]; g_v41_pool_uncertain=true; return 50000+(int)e; }
  return status;
 }
 int invalid=0;
 fcuda_ts_untimed_transfer();
 e=cudaMemcpy(&invalid,out+width,sizeof(int),cudaMemcpyDeviceToHost);
 if(e!=cudaSuccess) return 60000+(int)e;
 return invalid?-4:0;
}
// Qualification-only checked readback; no ordinary Read/logging fallback.
extern "C" int fak_v41_pool_readback(float *host, const float *device, uint64_t count) {
 if(!host) return -2;
 int status=fak_v41_pool_input(device,count); if(status) return status;
 int copied=fcuda_d2h_checked(host,device,(size_t)count*sizeof(float));
 if(copied) {
  cudaError_t drained=cudaStreamSynchronize(g_stream);
  if(drained!=cudaSuccess) {
   g_v41_pool_quarantine[0]=device;
   g_v41_pool_uncertain=true;
   return 50000+(int)drained;
  }
  return 59000+copied;
 }
 return 0;
}
#endif
