#pragma once
#ifdef FAK_CUDA_DEEPGEMM_WKV
#include "../cuda_deepgemm_wkv.h"
#include "../cuda_mxfp8_cast.h"
extern "C" int fcuda_deepgemm_wkv_state(void) {
 if(g_deepgemm_wkv_uncertain||v41_candidates_stream_uncertain())return FAK_DG_WKV_UNCERTAIN;
 if(g_capture_open)return FAK_DG_WKV_CAPTURE;
 cudaStreamCaptureStatus capture;cudaError_t e=cudaStreamIsCapturing(g_stream,&capture);if(e)return (int)e;
 if(capture!=cudaStreamCaptureStatusNone)return FAK_DG_WKV_CAPTURE;
 fak_deepgemm_wkv_facts f{};return fcuda_deepgemm_wkv_facts(&f);
}
static bool dg_wkv_extent(const void *p,size_t n,size_t alignment) {
 if(!p||((uintptr_t)p%alignment))return false;
 auto it=g_live.find((void*)p);if(it==g_live.end()||it->second<n)return false;
 cudaPointerAttributes a{};return cudaPointerGetAttributes(&a,p)==cudaSuccess&&a.type==cudaMemoryTypeDevice&&a.device==0;
}
extern "C" int fcuda_deepgemm_wkv_upload(void *dst,const void *src,size_t n) {
 int rc=fcuda_deepgemm_wkv_state();if(rc)return rc;
 if(!src||!dg_wkv_extent(dst,n,16))return FAK_DG_WKV_BUFFER;
 cudaError_t e=cudaMemcpy(dst,src,n,cudaMemcpyHostToDevice);
 if(e){cudaError_t drain=cudaStreamSynchronize(g_stream);if(drain){g_deepgemm_wkv_uncertain=true;g_deepgemm_wkv_quarantine[0]=dst;}return (int)e;}
 g_h2d_bytes+=n;return 0;
}
extern "C" int fcuda_deepgemm_wkv_release(void *ptr) {
 if(!ptr)return 0;int rc=fcuda_deepgemm_wkv_state();if(rc)return rc;
 auto it=g_live.find(ptr);if(it==g_live.end())return FAK_DG_WKV_BUFFER;
 cudaError_t e=cudaStreamSynchronize(g_stream);if(e){g_deepgemm_wkv_uncertain=true;g_deepgemm_wkv_quarantine[0]=ptr;return (int)e;}
 e=cudaFree(ptr);if(e){g_deepgemm_wkv_uncertain=true;g_deepgemm_wkv_quarantine[0]=ptr;return (int)e;}g_live.erase(it);return 0;
}
extern "C" int fcuda_deepgemm_wkv_run(const float *x,const uint8_t *b,const uint32_t *sfb,uint8_t *a,uint32_t *sfa,uint32_t *status,void *d,int rows) {
 int rc=fcuda_deepgemm_wkv_state();if(rc)return rc;
 fak_mxfp8_cast_layout plan{};if(rows<=0||rows%128||fak_mxfp8_cast_plan(rows,5120,&plan))return FAK_DG_WKV_SHAPE;
 const void *p[7]={x,b,sfb,a,sfa,status,d};size_t n[7]={plan.input_bytes,512*5120,512*160,plan.code_bytes,plan.scale_bytes,4,(size_t)rows*512*2};
 for(int i=0;i<7;i++){
  if(!dg_wkv_extent(p[i],n[i],i==5?4:16))return FAK_DG_WKV_BUFFER;
  uintptr_t begin=(uintptr_t)p[i];if(begin>UINTPTR_MAX-n[i])return FAK_DG_WKV_BUFFER;
  for(int j=0;j<i;j++)if(begin<(uintptr_t)p[j]+n[j]&&(uintptr_t)p[j]<begin+n[i])return FAK_DG_WKV_BUFFER;
 }
 auto finish=[&](int code){cudaError_t e=cudaStreamSynchronize(g_stream);if(e){g_deepgemm_wkv_uncertain=true;for(int i=0;i<7;i++)g_deepgemm_wkv_quarantine[i]=(void*)p[i];return code?code:(int)e;}return code;};
 rc=fcuda_mxfp8_cast_f32(x,plan.input_bytes,a,plan.code_bytes,sfa,plan.scale_bytes,status,rows,5120);
 if(rc)return finish(rc==FAK_MXFP8_CAST_NONFINITE?FAK_DG_WKV_NONFINITE:rc);
 rc=fak_deepgemm_wkv_launch(a,b,sfa,sfb,d,rows,(void*)g_stream);
 return finish(rc);
}
#endif
