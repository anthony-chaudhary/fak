// MIT-derived selected-row kernels. See v41_embed_gather.LICENSE.
// Included after v41_checked_stream.cuh and iq_gemv.cuh; never stand-alone.
#ifdef FAK_CUDA_V41_EMBED_GATHER
#include "v41_embed_gather_math.h"
#include "v41_embed_gather_geometry.h"
static __global__ void v41_eg_f32(const uint32_t *src,uint32_t *dst,int hidden) {
  // getrows.cu k_get_rows_float indexing, specialized to a selected row.
  // Integer copies preserve every bit (NaN payloads and signed zeros included).
  for(uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;i<(uint64_t)hidden;i+=(uint64_t)gridDim.x*blockDim.x)dst[i]=src[i];
}
static __global__ void v41_eg_q2(const uint8_t *src,float *dst,int blocks) {
  // getrows.cu k_get_rows_kq: 64 lanes decode four values per lane.
  for(int64_t ib=blockIdx.x;ib<blocks;ib+=gridDim.x)
    v41_eg_q2_thread(src+ib*84,dst+ib*256,threadIdx.x);
}
static __global__ void v41_eg_q5(const uint8_t *src,float *dst,int blocks) {
  for(int64_t ib=blockIdx.x;ib<blocks;ib+=gridDim.x)
    v41_eg_q5_thread(src+ib*176,dst+ib*256,threadIdx.x);
}
extern "C" unsigned fak_v41_embed_abi(void){return 2;}
extern "C" int fak_v41_embed_state(void){return v41_checked_stream_state();}
extern "C" int fak_v41_embed_finish(void){return v41_checked_stream_finish();}
extern "C" int fak_v41_embed_input(const void *p,uint64_t bytes){return v41_checked_stream_owned_bytes(p,bytes);}
extern "C" int fak_v41_embed_gather(const void *table,float *out,const intptr_t *ids,int rows,int hidden,int vocab,int mode) {
  // Go borrows IDs for this synchronous call only. No host pointer is retained.
  uint64_t rowbytes=0,outputbytes=0;
  if(!v41_eg_geometry(ids,rows,hidden,vocab,mode,&rowbytes,&outputbytes))return -1;
  int st=v41_checked_stream_state();if(st)return st;
  st=v41_checked_stream_owned_bytes(table,rowbytes*(uint64_t)vocab);if(st)return st;
  st=v41_checked_stream_owned_bytes(out,outputbytes);if(st)return st;
  if(table==out)return -2;
  for(int r=0;r<rows;++r) {
    const uint8_t *src=(const uint8_t *)table+(uint64_t)ids[r]*rowbytes;
    float *dst=out+(uint64_t)r*hidden;
    if(mode==2) {
      const int blocks=hidden/256;
      v41_eg_q5<<<blocks<65535?blocks:65535,64,0,g_stream>>>(src,dst,blocks);
    } else if(mode==1) {
      const int blocks=hidden/256;
      v41_eg_q2<<<blocks<65535?blocks:65535,64,0,g_stream>>>(src,dst,blocks);
    } else {
      const int blocks=(hidden-1)/256+1;
      v41_eg_f32<<<blocks<65535?blocks:65535,256,0,g_stream>>>((const uint32_t *)src,(uint32_t *)dst,hidden);
    }
    st=v41_checked_stream_error(20000,cudaPeekAtLastError());if(st)return st;
  }
  return v41_checked_stream_finish();
}

// Optional IQ4_NL spine, not a Go/model capability. Both native feature defines
// are required; ordinary embedding builds and ABI2's accepted modes are intact.
// getrows.cu:44-70,395-398 at llama.cpp e0f142d5ddc1461ce721644d92af33db23545105
// supplies the 32-lane selected-row mapping. The MIT notice is retained in
// v41_embed_iq4nl.h. No upstream context, stream, cache or scheduler is imported.
#ifdef FAK_CUDA_V41_EMBED_IQ4NL
#include "v41_embed_iq4nl.h"
// iq_gemv.cuh is included first by the canonical CUDA TU. Use its single
// device codebook definition directly, without a host-pass const declaration.
static __global__ void v41_eg_iq4nl(const uint8_t *src,float *dst,int blocks) {
  for(int64_t ib=blockIdx.x;ib<blocks;ib+=gridDim.x)
    v41_eg_iq4nl_thread(src+ib*144,dst+ib*256,threadIdx.x,fcuda_kvalues_iq4nl);
}
extern "C" unsigned fak_v41_embed_iq4nl_abi(void){return 1;}
extern "C" int fak_v41_embed_iq4nl_gather(const void *table,float *out,const intptr_t *ids,int rows,int hidden,int vocab) {
  // Like ABI2, this requires the registered cudaDev/cudaMu caller, exact tracked
  // device allocation bases and borrowed IDs that remain live until return.
  // The caller must propagate a selected failure; CPU replay is never valid.
  uint64_t rowbytes=0,outputbytes=0;
  if(!v41_eg_iq4nl_geometry(ids,rows,hidden,vocab,&rowbytes,&outputbytes))return -1;
  int st=v41_checked_stream_state();if(st)return st;
  st=v41_checked_stream_owned_bytes(table,rowbytes*(uint64_t)vocab);if(st)return st;
  st=v41_checked_stream_owned_bytes(out,outputbytes);if(st)return st;
  if(table==out)return -2;
  const int blocks=hidden/256;
  for(int r=0;r<rows;++r) {
    const uint8_t *src=(const uint8_t *)table+(uint64_t)ids[r]*rowbytes;
    float *dst=out+(uint64_t)r*hidden;
    v41_eg_iq4nl<<<blocks<65535?blocks:65535,32,0,g_stream>>>(src,dst,blocks);
    st=v41_checked_stream_error(20000,cudaPeekAtLastError());if(st)return st;
  }
  return v41_checked_stream_finish();
}

// Separate checked-row ABI; the original ABI2 and IQ4_NL ABI1 stay unchanged.
// This directly reuses the existing Engram Q2 row status/poison contract, not
// the unrelated BF16 four-word status envelope. No Go capability is implied.
static_assert(sizeof(unsigned int)==4,"Engram row atomic status requires uint32");
static __global__ void v41_eg_iq4nl_row(const uint8_t *src,float *dst,uint32_t *status,int dim) {
  __shared__ float decoded[256];
  __shared__ unsigned int fault;
  const unsigned int tid=threadIdx.x;
  if(tid==0)fault=UINT32_MAX;
  v41_eg_iq4nl_thread(src,decoded,tid,fcuda_kvalues_iq4nl);
  __syncthreads();
  for(unsigned int j=tid;j<(unsigned int)dim;j+=32) {
    const float value=decoded[j];
    dst[j]=value;
    const uint32_t key=v41_eg_row_fault_key(value,j);
    if(key!=UINT32_MAX)atomicMin(&fault,(unsigned int)key);
  }
  __syncthreads();
  // A fresh result, including clean zero, is written once on every decode.
  if(tid==0)*status=v41_eg_row_fault_word(fault);
}
static __global__ void v41_eg_iq4nl_poison(float *target,const uint32_t *status,int rows,int index) {
  __shared__ unsigned int fault;
  if(threadIdx.x==0)fault=0;
  __syncthreads();
  uint32_t seen=0;
  for(uint64_t r=threadIdx.x;r<(uint64_t)rows;r+=blockDim.x)seen|=status[r];
  if(seen)atomicOr(&fault,1u);
  __syncthreads();
  if(threadIdx.x==0)v41_eg_row_poison(target+index,fault);
}
extern "C" unsigned fak_v41_embed_iq4nl_rows_abi(void){return 1;}
// Synchronous host borrowing follows runtime_memory.cuh's H2D copy/accounting,
// but returns a checked error rather than swallowing it. No host pointer is
// retained. Fence this call's default stream explicitly before the private
// stream: pageable H2D may return after staging, and per-thread default-stream
// builds do not order that DMA against g_stream implicitly.
extern "C" int fak_v41_embed_iq4nl_upload(void *device,const void *host,uint64_t bytes) {
  if(!host||bytes==0||bytes>INT_MAX||bytes%144)return -1;
  int st=v41_checked_stream_state();if(st)return st;
  st=v41_checked_stream_owned_bytes(device,bytes);if(st)return st;
  fcuda_ts_untimed_transfer();
  const cudaError_t copied=cudaMemcpy(device,host,(size_t)bytes,cudaMemcpyHostToDevice);
  st=v41_checked_stream_error(30000,copied);if(st)return st;
  g_h2d_bytes+=(size_t)bytes;
  st=v41_checked_stream_error(40000,cudaStreamSynchronize(0));if(st)return st;
  return v41_checked_stream_finish();
}
extern "C" int fak_v41_embed_iq4nl_gather_rows(const void *table,float *out,uint32_t *status,const intptr_t *ids,int rows,int dim,int vocab) {
  // The future cudaDev/cudaMu caller owns all three allocations and borrows
  // IDs until return. Return 0 certifies execution, NOT numeric cleanliness:
  // inspect/propagate row status before consuming the panel. A nonzero return
  // invalidates both outputs; never inspect stale status or replay on CPU.
  uint64_t tablebytes=0,outputbytes=0,statusbytes=0;
  if(!v41_eg_iq4nl_rows_geometry(ids,rows,dim,vocab,&tablebytes,&outputbytes,&statusbytes))return -1;
  if(table==(const void *)out||table==(const void *)status||(const void *)out==(const void *)status)return -2;
  int st=v41_checked_stream_state();if(st)return st;
  st=v41_checked_stream_owned_bytes(table,tablebytes);if(st)return st;
  st=v41_checked_stream_owned_bytes(out,outputbytes);if(st)return st;
  st=v41_checked_stream_owned_bytes(status,statusbytes);if(st)return st;
  for(int r=0;r<rows;++r) {
    const uint8_t *src=(const uint8_t *)table+(uint64_t)ids[r]*144;
    v41_eg_iq4nl_row<<<1,32,0,g_stream>>>(src,out+(uint64_t)r*dim,status+r,dim);
    st=v41_checked_stream_error(20000,cudaPeekAtLastError());if(st)return st;
  }
  // Arithmetic faults only set row status; uncertain CUDA completion uses
  // the shared terminal/quarantine path and never masquerades as row data.
  return v41_checked_stream_finish();
}
extern "C" int fak_v41_embed_iq4nl_row_poison(float *target,const uint32_t *status,int rows,int targetCount,int index) {
  if(!v41_eg_row_poison_geometry(rows,targetCount,index))return -1;
  if((const void *)target==(const void *)status)return -2;
  int st=v41_checked_stream_state();if(st)return st;
  st=v41_checked_stream_owned_bytes(target,(uint64_t)targetCount*4);if(st)return st;
  st=v41_checked_stream_owned_bytes(status,(uint64_t)rows*4);if(st)return st;
  v41_eg_iq4nl_poison<<<1,256,0,g_stream>>>(target,status,rows,index);
  st=v41_checked_stream_error(20000,cudaPeekAtLastError());if(st)return st;
  return v41_checked_stream_finish();
}
#endif
#endif
