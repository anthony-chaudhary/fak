// Dedicated family-SASS translation unit. No Torch, runtime JIT or fallback.
#include "cuda_deepgemm_wkv.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <climits>
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>
#ifndef FAK_DEEPGEMM_WKV_SMS
#error Explicit physical SM count is required
#endif
static_assert(FAK_DEEPGEMM_WKV_SMS>0);
using DGShared=deep_gemm::layout::SM100FP8FP4GemmSharedStorage<5,2,2,128,128,128,128,64,128,128,1,cutlass::float_e4m3_t,cutlass::float_e4m3_t,cutlass::bfloat16_t>;
static_assert(sizeof(DGShared)<=232448);
#define DG_WKV_KERNEL deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl< \
 cute::UMMA::Major::K,cute::UMMA::Major::K,32,32,32,0,512,5120, \
 128,128,128,1,128,128,128,5,2,128,128,1,false,FAK_DEEPGEMM_WKV_SMS, \
 false,true,deep_gemm::GemmType::Normal,false,cutlass::float_e4m3_t, \
 cutlass::float_e4m3_t,cutlass::bfloat16_t,deep_gemm::epilogue::operators::Identity>
extern "C" int fcuda_deepgemm_wkv_facts(fak_deepgemm_wkv_facts *out) {
 if(!out)return FAK_DG_WKV_SHAPE;*out={};cudaError_t e=cudaGetDevice(&out->device);if(e)return (int)e;
 cudaDeviceProp p{};e=cudaGetDeviceProperties(&p,out->device);if(e)return (int)e;
 out->major=p.major;out->minor=p.minor;out->sms=p.multiProcessorCount;out->compiled_sms=FAK_DEEPGEMM_WKV_SMS;
 std::memcpy(out->name,p.name,sizeof(out->name));std::memcpy(out->uuid,p.uuid.bytes,16);
 e=cudaRuntimeGetVersion(&out->runtime_version);if(e)return (int)e;
 cudaFuncAttributes a{};e=cudaFuncGetAttributes(&a,DG_WKV_KERNEL);if(e)return (int)e;out->binary_version=a.binaryVersion;
 if(out->device!=0||out->major!=10||(out->minor!=0&&out->minor!=3)||out->sms!=FAK_DEEPGEMM_WKV_SMS||out->runtime_version<12090||out->binary_version<100||p.sharedMemPerBlockOptin<sizeof(DGShared))return FAK_DG_WKV_UNAVAILABLE;
 return 0;
}
static int dg_map(CUtensorMap *map,void *ptr,CUtensorMapDataType dt,uint64_t inner,uint64_t outer,uint64_t rowbytes,uint32_t boxinner,uint32_t boxouter,CUtensorMapSwizzle swizzle) {
 uint64_t dims[2]={inner,outer},strides[1]={rowbytes};uint32_t box[2]={boxinner,boxouter},elements[2]={1,1};
 CUresult r=cuTensorMapEncodeTiled(map,dt,2,ptr,dims,strides,box,elements,CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 return r==CUDA_SUCCESS?0:200000+(int)r;
}
int fak_deepgemm_wkv_launch(const uint8_t *a,const uint8_t *b,const uint32_t *sfa,const uint32_t *sfb,void *d,int rows,void *opaque) {
 if(rows<=0||rows%128||rows>INT_MAX/5120)return FAK_DG_WKV_SHAPE;
 fak_deepgemm_wkv_facts facts{};int rc=fcuda_deepgemm_wkv_facts(&facts);if(rc)return rc;
 cudaStream_t stream=(cudaStream_t)opaque;cudaStreamCaptureStatus capture;
 cudaError_t e=cudaStreamIsCapturing(stream,&capture);if(e)return (int)e;if(capture!=cudaStreamCaptureStatusNone)return FAK_DG_WKV_CAPTURE;
 // Match pinned Major-K A/B and MN-major scale descriptors. M is divisible128,
 // so the four-row packed-scale padding equals M exactly.
 CUtensorMap ma{},mb{},msa{},msb{},md{};
 if((rc=dg_map(&ma,(void*)a,CU_TENSOR_MAP_DATA_TYPE_UINT8,5120,rows,5120,128,128,CU_TENSOR_MAP_SWIZZLE_128B)))return rc;
 if((rc=dg_map(&mb,(void*)b,CU_TENSOR_MAP_DATA_TYPE_UINT8,5120,512,5120,128,128,CU_TENSOR_MAP_SWIZZLE_128B)))return rc;
 if((rc=dg_map(&msa,(void*)sfa,CU_TENSOR_MAP_DATA_TYPE_INT32,rows,40,(uint64_t)rows*4,128,1,CU_TENSOR_MAP_SWIZZLE_NONE)))return rc;
 if((rc=dg_map(&msb,(void*)sfb,CU_TENSOR_MAP_DATA_TYPE_INT32,512,40,512*4,128,1,CU_TENSOR_MAP_SWIZZLE_NONE)))return rc;
 if((rc=dg_map(&md,d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,512,rows,512*2,64,128,CU_TENSOR_MAP_SWIZZLE_128B)))return rc;
 e=cudaFuncSetAttribute(DG_WKV_KERNEL,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(DGShared));if(e)return (int)e;
 static_assert(sizeof(cute::TmaDescriptor)==sizeof(CUtensorMap));
 cute::TmaDescriptor ta,tb,tsa,tsb,td;std::memcpy(&ta,&ma,sizeof(ma));std::memcpy(&tb,&mb,sizeof(mb));std::memcpy(&tsa,&msa,sizeof(msa));std::memcpy(&tsb,&msb,sizeof(msb));std::memcpy(&td,&md,sizeof(md));
 int *groups=nullptr;uint32_t m=rows,n=512,k=5120;deep_gemm::EpilogueOperatorArgs ep{};
 void *args[]={&groups,&m,&n,&k,&ep,&ta,&tb,&tsa,&tsb,&td};
 e=cudaLaunchKernel((const void*)DG_WKV_KERNEL,dim3(FAK_DEEPGEMM_WKV_SMS),dim3(256),args,sizeof(DGShared),stream);if(e)return (int)e;
 return (int)cudaPeekAtLastError();
}
