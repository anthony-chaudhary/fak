// Adapted from ggml-org/llama.cpp dsv4-hc.cu at
// 3c809f44383bcebc318b8cddd8c6d11aaa861e9b; see cuda_v41_mhc.LICENSE.
// Included by cuda_kernels.cu. Contiguous FAK ABI, not ggml's tensor ABI.
#ifdef FAK_CUDA_V41_MHC
__device__ static float mhc_bf16(float v) {
  unsigned u = __float_as_uint(v); u += 0x7fffu + ((u >> 16) & 1u);
  return __uint_as_float(u & 0xffff0000u);
}
__device__ static void mhc_norm(float *c, float eps, bool columns) {
  for (int i=0;i<4;++i) {
    float sum=0.f;
    for(int j=0;j<4;++j) sum=__fadd_rn(sum,c[columns?j*4+i:i*4+j]);
    const float den=__fadd_rn(sum,eps);
    for(int j=0;j<4;++j) { int k=columns?j*4+i:i*4+j; c[k]=__fdiv_rn(c[k],den); }
  }
}
__device__ static void mhc_split(const float *m,const float *s,const float *b,float *out,int iters,float eps,float rms) {
  for(int i=0;i<8;++i) {
    float v=__fadd_rn(__fmul_rn(__fmul_rn(m[i],rms),s[i/4]),b[i]);
    float gate=__fdiv_rn(1.f,__fadd_rn(1.f,expf(-v)));
    out[i]=i<4?__fadd_rn(gate,eps):__fmul_rn(2.f,gate);
  }
  float c[16];
  for(int i=0;i<16;++i) c[i]=__fadd_rn(__fmul_rn(__fmul_rn(m[8+i],rms),s[2]),b[8+i]);
  for(int r=0;r<4;++r) {
    float mx=c[r*4]; for(int j=1;j<4;++j) if(c[r*4+j]>mx) mx=c[r*4+j];
    float sum=0.f;
    for(int j=0;j<4;++j) { c[r*4+j]=expf(__fsub_rn(c[r*4+j],mx)); sum=__fadd_rn(sum,c[r*4+j]); }
    for(int j=0;j<4;++j) c[r*4+j]=__fadd_rn(__fdiv_rn(c[r*4+j],sum),eps);
  }
  mhc_norm(c,eps,true);
  for(int i=1;i<iters;++i) { mhc_norm(c,eps,false); mhc_norm(c,eps,true); }
  for(int i=0;i<16;++i) out[8+i]=c[i];
}
__global__ static void k_mhc(int op,const float *a,const float *b,const float *c,float *out,int rows,int hidden,int iters,float eps,float rms_eps) {
  size_t n=(op==1||op==4)?(size_t)rows:(size_t)rows*hidden*(op==2?1:4);
  for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)blockDim.x*gridDim.x) {
    if(op==1) { mhc_split(a+i*24,b,c,out+i*24,iters,eps,1.f); continue; }
    if(op==4) {
      float ss=0.f; const size_t width=(size_t)hidden*4;
      for(size_t j=0;j<width;++j) { float v=b[i*width+j]; ss=__fadd_rn(ss,__fmul_rn(v,v)); }
      float mean=__fadd_rn(__fdiv_rn(ss,(float)width),rms_eps);
      float rms=(float)(1.0/sqrt((double)mean));
      mhc_split(a+i*24,c,c+3,out+i*24,iters,eps,rms); continue;
    }
    size_t d=i%hidden, row=i/(size_t)hidden/(op==2?1:4);
    if(op==0) { out[i]=mhc_bf16(a[row*hidden+d]); continue; }
    int dst=(i/hidden)%4; float sum=0.f;
    const float *streams=op==2?a:b, *coeff=op==2?b:c;
    for(int src=0;src<4;++src) {
      float w=coeff[row*24+(op==2?src:8+src*4+dst)];
      sum=__fadd_rn(sum,__fmul_rn(w,streams[(row*4+src)*hidden+d]));
    }
    if(op==3) sum=__fadd_rn(sum,__fmul_rn(coeff[row*24+4+dst],mhc_bf16(a[row*hidden+d])));
    out[i]=mhc_bf16(sum);
  }
}
extern "C" int fak_v41_mhc_state(void) {
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
extern "C" int fak_v41_mhc_input(const float *p,uint64_t count) {
  int state=fak_v41_mhc_state(); if(state) return state;
  if(!p||!count||count>UINT32_MAX||count>SIZE_MAX/sizeof(float)) return -2;
  auto it=g_live.find((void*)p); if(it==g_live.end()||it->second<count*sizeof(float)) return -2;
  cudaPointerAttributes attr={}; cudaError_t e=cudaPointerGetAttributes(&attr,p);
  if(e!=cudaSuccess) return 11000+(int)e;
  return attr.type==cudaMemoryTypeDevice&&attr.device==0?0:-2;
}
extern "C" int fak_v41_mhc_run(int op,const float *a,const float *b,const float *c,float *out,int rows,int hidden,int iters,float eps,float rms_eps) {
  if(op<0||op>4||rows<=0||((op!=1)&&hidden<=0)) return -1;
  if((op==1||op==4)&&(iters<=0||!isfinite(eps)||eps<=0)) return -1;
  if(op==4&&(!isfinite(rms_eps)||rms_eps<=0)) return -1;
  uint64_t x=(uint64_t)rows*(uint64_t)hidden, streams=4*x, coeff=(uint64_t)rows*24;
  uint64_t counts[4]={0,0,0,0};
  switch(op) {
    case 0: counts[0]=x; counts[3]=streams; break;
    case 1: counts[0]=coeff; counts[1]=3; counts[2]=24; counts[3]=coeff; break;
    case 2: counts[0]=streams; counts[1]=coeff; counts[3]=x; break;
    case 3: counts[0]=x; counts[1]=streams; counts[2]=coeff; counts[3]=streams; break;
    case 4: counts[0]=coeff; counts[1]=streams; counts[2]=27; counts[3]=coeff; break;
  }
  const float *p[4]={a,b,c,out};
  if(out==a||out==b||out==c) return -2;
  for(int i=0;i<4;++i) if(counts[i]) { int s=fak_v41_mhc_input(p[i],counts[i]); if(s) return s; }
  uint64_t n=(op==1||op==4)?(uint64_t)rows:counts[3];
  unsigned blocks=(unsigned)((n+127)/128); if(blocks>65535) blocks=65535;
  FCUDA_TS("k_mhc", blocks, k_mhc<<<blocks,128,0,g_stream>>>(op,a,b,c,out,rows,hidden,iters,eps,rms_eps));
  cudaError_t e=cudaGetLastError(); int status=e==cudaSuccess?0:30000+(int)e;
  if(!status) { e=cudaStreamSynchronize(g_stream); if(e!=cudaSuccess) status=40000+(int)e; }
  if(status) {
    e=cudaStreamSynchronize(g_stream);
    if(e!=cudaSuccess) { for(int i=0;i<4;++i) g_v41_mhc_quarantine[i]=p[i]; g_v41_mhc_uncertain=true; return 50000+(int)e; }
    return status;
  }
  return 0;
}
// Checked witness readback; a failed drain pins the source, never certifies bytes.
extern "C" int fak_v41_mhc_readback(float *host,const float *device,uint64_t count) {
  if(!host) return -2;
  int status=fak_v41_mhc_input(device,count); if(status) return status;
  int copied=fcuda_d2h_checked(host,device,(size_t)count*sizeof(float));
  if(copied) {
    cudaError_t e=cudaStreamSynchronize(g_stream);
    if(e!=cudaSuccess) { g_v41_mhc_quarantine[0]=device; g_v41_mhc_uncertain=true; return 50000+(int)e; }
    return 59000+copied;
  }
  return 0;
}
#endif
