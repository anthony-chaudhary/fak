// Dedicated SM100a/103a translation unit; never compiled by the generic matrix.
#include "cuda_flashmla_sparse_prefill.h"
#include <cuda_runtime.h>
#include <cstring>
#include "cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.cuh"
namespace core = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn;
static constexpr core::Config flash_config{SparseAttnFwdMode::Prefill,ModelType::V41,ModelType::V41,64,false};

__global__ static void flash_permute(const uint16_t *in,uint16_t *out,int rows) {
 uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=(uint64_t)rows*32768)return;
 uint64_t row=i/32768; unsigned h=(i%32768)/512,d=i%512;
 out[row*32768+d%16+h*16+(d/16)*1024]=in[i];
}

// Integer IEEE decoding avoids FTZ in the donor's --use_fast_math build. Every
// finite E4M3*UE8M0 value needs at most four significand bits; even the smallest
// nonzero product (2^-136) is exactly representable as a float32 subnormal.
__device__ static float flash_decode(unsigned v,unsigned sf,int *status) {
 unsigned sign=(v&128u)<<24,e=(v>>3)&15u,f=v&7u;
 if((e==15&&f==7)||sf==255){atomicExch(status,1);return 0;}
 unsigned m=e?8+f:f;if(!m)return __uint_as_float(sign);
 int power=(e?(int)e-10:-9)+(int)sf-127;
 int top=31-__clz(m), unbiased=power+top;
 if(unbiased>127){atomicExch(status,1);return 0;}
 unsigned bits;
 if(unbiased>=-126)bits=((unsigned)(unbiased+127)<<23)|((m<<(23-top))&0x7fffff);
 else {int shift=power+149;if(shift<0||shift>23){atomicExch(status,1);return 0;}bits=m<<shift;}
 return __uint_as_float(sign|bits);
}
__global__ static void flash_unpack(const uint8_t *fp8,const uint32_t *sf,float *out,
 float *maxlog,float *lse,const int *ids,const int *lengths,int topk,int aligned,int *status) {
 unsigned q=blockIdx.x/64,h=blockIdx.x%64,d=threadIdx.x;
 __shared__ int active;
 if(d==0){active=0;for(int k=0;k<lengths[q];k++)if(ids[(uint64_t)q*topk+k]>=0){active=1;break;}}
 __syncthreads();
 if(!active){out[(uint64_t)q*32768+h*512+d]=0;if(d==0){maxlog[q*64+h]=-CUDART_INF_F;lse[q*64+h]=CUDART_INF_F;}return;}
 unsigned group=h/8,within=h%8,sb=within+(d/32)*8;
 unsigned scale=(sf[(uint64_t)group*32*aligned+(sb/4)*aligned+q]>>((sb%4)*8))&255;
 unsigned value=fp8[(uint64_t)q*32768+group*4096+within*32+(d/32)*256+d%32];
 out[(uint64_t)q*32768+h*512+d]=flash_decode(value,scale,status);
}
extern "C" int fak_flashmla_prefill_kernel_info(fak_flashmla_identity *i) {
 if(!i)return -1;*i={};cudaError_t e=cudaGetDevice(&i->device);if(e)return (int)e;
 cudaDeviceProp p{};e=cudaGetDeviceProperties(&p,i->device);if(e)return (int)e;
 i->major=p.major;i->minor=p.minor;i->sms=p.multiProcessorCount;
 std::memcpy(i->name,p.name,sizeof(i->name));std::memcpy(i->uuid,p.uuid.bytes,16);
 e=cudaDriverGetVersion(&i->driver);if(e)return (int)e;
 e=cudaRuntimeGetVersion(&i->runtime);if(e)return (int)e;
 cudaFuncAttributes a{};e=cudaFuncGetAttributes(&a,core::fwd_kernel<core::Kernel<flash_config>>);if(e)return (int)e;
 i->binary_version=a.binaryVersion;i->ptx_version=a.ptxVersion;
 return i->device==0&&i->major==10&&(i->minor==0||i->minor==3)&&i->binary_version>=100?0:-2;
}
int fak_flashmla_prefill_launch(const fak_flashmla_input& x,void **b,void *opaque) {
 cudaStream_t stream=(cudaStream_t)opaque;
 fak_flashmla_identity id{};int e=fak_flashmla_prefill_kernel_info(&id);if(e)return e;
 int aligned=(x.queries+3)&~3;
 flash_permute<<<((uint64_t)x.queries*32768+255)/256,256,0,stream>>>((uint16_t*)b[0],(uint16_t*)b[7],x.queries);
 if((e=(int)cudaPeekAtLastError()))return e;
 core::ParamT<SparseAttnFwdMode::Prefill> p{};
 p.s_q=x.queries;p.s_kv=x.keys;p.h_q=64;p.h_kv=1;p.d_qk=512;p.d_v=512;p.topk=x.topk;
 p.sm_scale=x.scale;p.sm_scale_div_log2=x.scale*1.4426950408889634074f;
 p.q=(bf16*)b[7];p.kv=(bf16*)b[1];p.indices=(int*)b[2];p.topk_length=(int*)b[3];p.token_positions=(uint32_t*)b[4];p.attn_sink=(float*)b[5];p.cos_sin_cache=(float*)b[6];
 p.stride_q_s_q=32768;p.stride_q_h_q=512;p.stride_kv_s_kv=512;p.stride_kv_h_kv=512;p.stride_indices_s_q=x.topk;p.stride_indices_h_kv=x.topk;
 p.out=nullptr;p.max_logits=(float*)b[10];p.lse=(float*)b[11];p.mode=SparseAttnFwdMode::Prefill;p.num_sm=id.sms;p.stream=stream;
 p.enable_q_norm=false;p.rms_norm_eps=0;p.is_rope_neox_style=false;p.rope_dim=64;
 p.n_wv_group=8;p.wv_group_size=8;p.num_per_channels=32;p.use_tma_aligned_col_major_sf=true;p.round_sf=true;p.use_packed_ue8m0=true;
 p.out_fp8=(fp8_e4m3*)b[8];p.out_sf=(uint32_t*)b[9];p.stride_out_sf_wv_group=32*aligned;p.stride_out_sf_head_dim=aligned;
 try {core::run_fused_norm_rope_attn_rope_cast_fwd_kernel<flash_config>(p);}catch(...){return -3;}
 if((e=(int)cudaPeekAtLastError()))return e;
 flash_unpack<<<x.queries*64,512,0,stream>>>((uint8_t*)b[8],(uint32_t*)b[9],(float*)b[12],(float*)b[10],(float*)b[11],(int*)b[2],(int*)b[3],x.topk,aligned,(int*)b[13]);
 return (int)cudaPeekAtLastError();
}
