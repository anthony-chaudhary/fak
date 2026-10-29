#pragma once
#ifdef FAK_CUDA_FLASHMLA_PREFILL
#include "cuda_flashmla_sparse_prefill.h"
// Included after allocator/service definitions. State is declared before them
// so every exact-pointer retirement route can recognize terminal quarantine.
extern "C" int fak_flashmla_prefill_observe(uint64_t *live,int *uncertain) {
 if(!live||!uncertain)return -1;*live=(uint64_t)g_live.size();*uncertain=v41_candidates_stream_uncertain()?1:0;
 if(*uncertain||g_capture_open)return -2;
 cudaStreamCaptureStatus capture;cudaError_t e=cudaStreamIsCapturing(g_stream,&capture);if(e)return (int)e;
 if(capture!=cudaStreamCaptureStatusNone)return -2;return (int)cudaStreamSynchronize(g_stream);
}
extern "C" int fak_flashmla_prefill_run(const fak_flashmla_input *xp,const fak_flashmla_output *op) {
 if(!xp||!op)return -1;const auto& x=*xp;const auto& o=*op;
 if(!x.q||!x.kv||!x.indices||!x.lengths||!x.positions||!x.sink||!x.table||!o.values||!o.max_logits||!o.lse)return -1;
 if(x.queries<=0||x.queries>INT_MAX/32768||x.keys<=0||x.keys>INT_MAX/512||x.topk<=0||x.topk%8||x.topk>INT_MAX/x.queries||x.table_rows<=0||x.table_rows>INT_MAX/64)return -1;
 if(g_capture_open||v41_candidates_stream_uncertain())return -2;
 cudaStreamCaptureStatus capture;cudaError_t ce=cudaStreamIsCapturing(g_stream,&capture);if(ce)return (int)ce;if(capture!=cudaStreamCaptureStatusNone)return -2;
 if((ce=cudaPeekAtLastError())!=cudaSuccess)return (int)ce;
 fak_flashmla_identity identity{};int result=fak_flashmla_prefill_kernel_info(&identity);if(result)return result;
 size_t q=(size_t)x.queries,k=(size_t)x.keys,a=(q+3)&~size_t(3);
 const size_t bytes[14]={q*32768*2,k*512*2,q*x.topk*4,q*4,q*4,64*4,(size_t)x.table_rows*64*4,q*32768*2,q*32768,8*32*a*4,q*64*4,q*64*4,q*32768*4,4};
 const void *host[7]={x.q,x.kv,x.indices,x.lengths,x.positions,x.sink,x.table};
 void *b[14]={};bool submitted=false;
 auto retain=[&](){g_flashmla_uncertain=true;for(int i=0;i<14;i++)g_flashmla_quarantine[i]=b[i];};
 auto finish=[&](int status){
  if(submitted){cudaError_t drained=cudaStreamSynchronize(g_stream);if(drained!=cudaSuccess){retain();return status?status:(int)drained;}}
  bool failed=false;
  for(int i=0;i<14;i++)if(b[i]){cudaError_t freed=cudaFree(b[i]);if(freed==cudaSuccess){g_live.erase(b[i]);b[i]=nullptr;}else{failed=true;if(!status)status=(int)freed;}}
  if(failed)retain();return status;
 };
 try {
  // Reserve every operand before copying or launching. No managed fallback,
  // pool reuse, or caller-owned device allocations enter this operation.
  for(int i=0;i<14;i++){ce=cudaMalloc(&b[i],bytes[i]);if(ce)return finish((int)ce);g_live.emplace(b[i],bytes[i]);}
  for(int i=0;i<14;i++){
   auto it=g_live.find(b[i]);if(it==g_live.end()||it->second!=bytes[i])return finish(-4);
   for(int j=0;j<i;j++)if(b[i]==b[j])return finish(-4);
  }
  // Synchronous host copies are intentional: no in-flight operation may retain
  // a Go pointer after a failed drain returns. The shared stream is blocking.
  submitted=true;
  for(int i=0;i<7;i++){ce=cudaMemcpy(b[i],host[i],bytes[i],cudaMemcpyHostToDevice);if(ce)return finish((int)ce);g_h2d_bytes+=bytes[i];}
  submitted=true;ce=cudaMemsetAsync(b[13],0,4,g_stream);if(ce)return finish((int)ce);
  result=fak_flashmla_prefill_launch(x,b,(void*)g_stream);if(result)return finish(result);
  ce=cudaStreamSynchronize(g_stream);if(ce){retain();return (int)ce;}
  int numerical=0;ce=cudaMemcpy(&numerical,b[13],4,cudaMemcpyDeviceToHost);if(ce)return finish((int)ce);g_host_bytes+=4;
  if(numerical)return finish(-5);
  void *dest[3]={o.values,o.max_logits,o.lse};int ids[3]={12,10,11};
  for(int j=0;j<3;j++){int i=ids[j];ce=cudaMemcpy(dest[j],b[i],bytes[i],cudaMemcpyDeviceToHost);if(ce)return finish((int)ce);g_host_bytes+=bytes[i];}
  return finish(0);
 }catch(...){return finish(-6);}
}
extern "C" int fak_flashmla_prefill_host(const uint16_t *q,const uint16_t *kv,const int32_t *ids,
 const int32_t *lengths,const uint32_t *positions,const float *sink,const float *table,
 int nq,int nk,int topk,int rows,float scale,float *out,float *maxlog,float *lse) {
 fak_flashmla_input x{q,kv,ids,lengths,positions,sink,table,nq,nk,topk,rows,scale};
 fak_flashmla_output o{out,maxlog,lse};return fak_flashmla_prefill_run(&x,&o);
}
#endif
