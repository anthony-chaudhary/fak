#pragma once
#ifdef FAK_CUDA_FLASHMLA_DECODE
#include "cuda_flashmla_sparse_decode_owner.h"
// Private qualification bridge only. cudaMu owns g_stream; no imported stream.
static void *g_flashmla_decode_device_quarantine[14]={};
static void *g_flashmla_decode_host_quarantine[10]={};
extern "C" int fak_flashmla_decode_host(const uint16_t *q,const uint8_t *kv,const int32_t *ids,const int32_t *length,const float *sink,
 const uint8_t *extra,const int32_t *extra_ids,const int32_t *extra_length,uint16_t *out,float *lse,
 int blocks,int page,int topk,int extra_blocks,int extra_page,int extra_topk,float scale,int invariant) {
 const fak_flashmla_decode_input input{q,kv,ids,length,sink,extra,extra_ids,extra_length,out,lse,blocks,page,topk,extra_blocks,extra_page,extra_topk,scale};
 const auto *xp=&input;
 int status=v41_checked_stream_state();if(status)return status;
 const auto x=*xp;
 // The Go boundary validates host extents and contents before this private ABI.
 if(!x.query_bf16||!x.kv||!x.indices||!x.length||!x.output_bf16||!x.lse)return -1;
 if(x.blocks<=0||x.page_size<=0||x.page_size>INT_MAX/528||x.blocks>(INT_MAX/528)/x.page_size)return -1;
 if(x.extra_topk&&(x.extra_blocks<=0||x.extra_page_size<=0||x.extra_page_size>INT_MAX/528||x.extra_blocks>(INT_MAX/528)/x.extra_page_size||!x.extra_kv||!x.extra_indices||!x.extra_length))return -1;
 if(!isfinite(x.scale)||x.scale<=0||!isfinite(x.scale*1.4426950408889634f)||*x.length<0||*x.length>x.topk)return -1;
 if(x.extra_topk&&(*x.extra_length<0||*x.extra_length>x.extra_topk))return -1;
 if(!x.extra_topk&&(x.extra_kv||x.extra_indices||x.extra_length||x.extra_blocks||x.extra_page_size))return -1;
 fak_flashmla_decode_workspace_sizes sizes{};
 status=fak_flashmla_decode_workspace_query(x.topk,x.extra_topk,invariant,&sizes);if(status)return status;
 const size_t bytes[14]={65536,(size_t)x.blocks*x.page_size*528,(size_t)x.topk*4,4,x.sink?256u:0u,
  x.extra_topk?(size_t)x.extra_blocks*x.extra_page_size*528:0u,(size_t)x.extra_topk*4,x.extra_topk?4u:0u,
  65536,256,sizes.metadata,sizes.num_splits,sizes.partial_lse,sizes.partial_output};
 const void *source[8]={x.query_bf16,x.kv,x.indices,x.length,x.sink,x.extra_kv,x.extra_indices,x.extra_length};
 size_t total=0;for(int i=0;i<8;i++){if(bytes[i]>(64u<<20)-total)return -1;total+=bytes[i];}
 void *host[10]={},*device[14]={};bool completed=false;
 auto retain=[&](){g_v41_checked_stream_uncertain.store(true);for(int i=0;i<14;i++)g_flashmla_decode_device_quarantine[i]=device[i];for(int i=0;i<10;i++)g_flashmla_decode_host_quarantine[i]=host[i];};
 auto finish=[&](int code){
  // Even an enqueue failure can follow scheduler/copy work. Never infer completion.
  int drained=completed?0:v41_checked_stream_finish();if(drained){retain();return code?code:drained;}
  bool failed=false;
  for(int i=0;i<14;i++)if(device[i]){cudaError_t e=cudaFree(device[i]);if(e==cudaSuccess){g_live.erase(device[i]);device[i]=nullptr;}else{failed=true;if(!code)code=(int)e;}}
  if(failed){retain();return code;}
  for(int i=0;i<10;i++)free(host[i]);
  return code;
 };
 try {
  // Snapshot C-owned bytes before enqueue: failed drains never retain Go pointers.
  for(int i=0;i<10;i++)if(bytes[i]){host[i]=malloc(bytes[i]);if(!host[i])return finish(-4);if(i<8)memcpy(host[i],source[i],bytes[i]);}
  for(int i=0;i<14;i++)if(bytes[i]){cudaError_t e=cudaMalloc(&device[i],bytes[i]);if(e)return finish((int)e);g_live.emplace(device[i],bytes[i]);}
  for(int i=0;i<14;i++)if(bytes[i]){status=v41_checked_stream_owned_bytes(device[i],bytes[i]);if(status)return finish(status);}
  fcuda_ts_untimed_transfer();
  for(int i=0;i<8;i++)if(bytes[i]){cudaError_t e=cudaMemcpyAsync(device[i],host[i],bytes[i],cudaMemcpyHostToDevice,g_stream);if(e)return finish((int)e);g_h2d_bytes+=bytes[i];}
  fak_flashmla_decode_input d=x;
  d.query_bf16=(uint16_t*)device[0];d.kv=(uint8_t*)device[1];d.indices=(int32_t*)device[2];d.length=(int32_t*)device[3];d.sink=(float*)device[4];
  d.extra_kv=(uint8_t*)device[5];d.extra_indices=(int32_t*)device[6];d.extra_length=(int32_t*)device[7];d.output_bf16=(uint16_t*)device[8];d.lse=(float*)device[9];
  fak_flashmla_decode_workspace w{device[10],bytes[10],(int32_t*)device[11],bytes[11],(float*)device[12],bytes[12],(float*)device[13],bytes[13]};
  status=fak_flashmla_decode_launch(&d,&w,invariant,(void*)g_stream);if(status)return finish(status);
  // Device results target C-owned staging on the same stream; no Go pointer
  // participates in an operation whose completion can become unknown.
  fcuda_ts_untimed_transfer();
  cudaError_t e=cudaMemcpyAsync(host[8],device[8],bytes[8],cudaMemcpyDeviceToHost,g_stream);if(e)return finish((int)e);g_host_bytes+=bytes[8];
  e=cudaMemcpyAsync(host[9],device[9],bytes[9],cudaMemcpyDeviceToHost,g_stream);if(e)return finish((int)e);g_host_bytes+=bytes[9];
  status=v41_checked_stream_finish();if(status){retain();return status;}completed=true;
  memcpy(x.output_bf16,host[8],bytes[8]);memcpy(x.lse,host[9],bytes[9]);
  return finish(0);
 } catch(...) {return finish(-5);}
}
#endif
