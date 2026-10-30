// SPDX-License-Identifier: Apache-2.0
// Direct port of shaders/v41_routed_swiglu.comp, blob
// 782669d69e5ca477a3fc88f06527f4f451d6b16e. See PROVENANCE.md.
#ifdef FAK_CUDA_V41_ROUTED_SWIGLU
__device__ static unsigned routed_swiglu_bf16(unsigned u) {
 return (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
}
__device__ static bool routed_swiglu_nonfinite(unsigned u) {
 return (u & 0x7f800000u) == 0x7f800000u;
}
__global__ static void k_v41_routed_swiglu(const float *gate,const float *up,float *out,unsigned *status,int count,float limit,float weight) {
 unsigned first=blockIdx.x*blockDim.x+threadIdx.x;
 if(first==0){status[0]=0x52535731u;status[1]=(unsigned)count;}
 for(unsigned i=first;i<(unsigned)count;i+=gridDim.x*blockDim.x){
  unsigned gb=__float_as_uint(gate[i]),ub=__float_as_uint(up[i]);
  unsigned gr=routed_swiglu_bf16(gb),ur=routed_swiglu_bf16(ub);
  if(routed_swiglu_nonfinite(gb)||routed_swiglu_nonfinite(ub)||routed_swiglu_nonfinite(gr)||routed_swiglu_nonfinite(ur)){
   atomicMin(status+2,i);out[i]=__uint_as_float(0x7fc00000u);continue;
  }
  float g=__uint_as_float(gr),u=__uint_as_float(ur);
  if(limit>0.f){g=fminf(g,limit);u=fmaxf(-limit,fminf(u,limit));}
  // Precise donor order: do not contract or substitute approximate division.
  // expf is device precision, as permitted by V41RoutedSwiGLUBackend;
  // qualification must bound it against the host oracle.
  float t=__fadd_rn(1.f,expf(-g));
  float f=__fmul_rn(__fdiv_rn(g,t),u);
  if(routed_swiglu_nonfinite(__float_as_uint(f))){
   atomicMin(status+2,i);out[i]=__uint_as_float(0x7fc00000u);continue;
  }
  float w=__fmul_rn(f,weight);unsigned wb=__float_as_uint(w),wr=routed_swiglu_bf16(wb);
  if(routed_swiglu_nonfinite(wb)||routed_swiglu_nonfinite(wr)){
   atomicMin(status+3,i);wr=0x7fc00000u;
  }
  out[i]=__uint_as_float(wr);
 }
}
extern "C" int fak_v41_routed_swiglu_state(void) {
  if(!g_blas||!g_stream||g_capture_open||v41_candidates_stream_uncertain()) return -3;
  cudaError_t e=cudaPeekAtLastError(); if(e!=cudaSuccess) return 10000+(int)e;
  int device=-1; e=cudaGetDevice(&device); if(e!=cudaSuccess) return 10000+(int)e;
  if(device!=0) return -3;
  cudaStreamCaptureStatus capture; e=cudaStreamIsCapturing(g_stream,&capture);
  if(e!=cudaSuccess) return 10000+(int)e;
  if(capture!=cudaStreamCaptureStatusNone) return -3;
  return 0;
}
extern "C" int fak_v41_routed_swiglu_input(const float *p,uint64_t count) {
  int state=fak_v41_routed_swiglu_state(); if(state) return state;
  if(!p||!count||count>UINT32_MAX||count>SIZE_MAX/sizeof(float)) return -2;
  auto it=g_live.find((void*)p); if(it==g_live.end()||it->second<count*sizeof(float)) return -2;
  cudaPointerAttributes attr={}; cudaError_t e=cudaPointerGetAttributes(&attr,p);
  if(e!=cudaSuccess) return 11000+(int)e;
  return attr.type==cudaMemoryTypeDevice&&attr.device==0?0:-2;
}

extern "C" int fak_v41_routed_swiglu_run(const float *g,const float *u,float *out,unsigned *status,unsigned *host,int count,float limit,float weight){
 if(count<=0||!isfinite(limit)||!isfinite(weight)||!host)return -1;
 const float *ptrs[4]={g,u,out,(const float*)status};
 for(int i=2;i<4;++i)for(int j=0;j<i;++j)if(ptrs[i]==ptrs[j])return -2;
 for(int i=0;i<4;++i){int x=fak_v41_routed_swiglu_input(ptrs[i],i==3?4:(uint64_t)count);if(x)return x;}
 // The freshly initialized record is ordered after both GEMMs on g_stream.
 cudaError_t e=cudaMemsetAsync(status,0xff,16,g_stream);int x=e==cudaSuccess?0:20000+(int)e;
 if(!x){
  unsigned blocks=((unsigned)count+127)/128;if(blocks>65535)blocks=65535;
  k_v41_routed_swiglu<<<blocks,128,0,g_stream>>>(g,u,out,status,count,limit,weight);
  e=cudaGetLastError();if(e!=cudaSuccess)x=30000+(int)e;
 }
 // A checked completion is required before examining status or publishing out.
 if(!x){e=cudaStreamSynchronize(g_stream);if(e!=cudaSuccess)x=40000+(int)e;}
 if(!x){e=cudaMemcpy(host,status,16,cudaMemcpyDeviceToHost);if(e!=cudaSuccess)x=45000+(int)e;}
 if(x){e=cudaStreamSynchronize(g_stream);if(e!=cudaSuccess){
  for(int i=0;i<4;++i)g_v41_routed_swiglu_quarantine[i]=ptrs[i];
  g_v41_routed_swiglu_uncertain=true;return 50000+(int)e;
 }return x;}
 return 0;
}
#endif
