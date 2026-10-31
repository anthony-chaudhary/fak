// Qualification-only direct upstream port, FlashMLA 2e5429fc (MIT).
// Dedicated CUDA >=13.1 C++20 sm_100a/sm_103a TU; never generic/sm120.
// Reuses the existing pinned FlashMLA/CUTLASS roots; no ATen/PyTorch runtime.
#include "cuda_flashmla_sparse_decode.h"
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#ifndef FLASH_MLA_IS_BUILD_ON_CUDA
#define FLASH_MLA_IS_BUILD_ON_CUDA
#endif
#include "cuda_kernels/sm100/decode/sparse/head64/kernel.cuh"
// Include official implementation sources into this one arch-specific TU.
#include "cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu"
#include "cuda_kernels/smxx/decode/combine/combine.cu"

namespace decode = sm100::decode::sparse::head64;
static constexpr decode::Config config{ModelType::V41, ModelType::V41, false};
static bool aligned(const void *p, uintptr_t n) {
    return p && reinterpret_cast<uintptr_t>(p) % n == 0;
}

extern "C" int fak_flashmla_decode_launch(const fak_flashmla_decode_input *xp,
    const fak_flashmla_decode_workspace *workspace, int batch_invariant, void *opaque) {
    if (!xp) return -1;
    const auto &x = *xp;
    if (!aligned(x.query_bf16,16) || !aligned(x.kv,16) || !aligned(x.indices,8) ||
        !aligned(x.output_bf16,16) || !aligned(x.lse,4) ||
        (x.length && !aligned(x.length,4)) || (x.sink && !aligned(x.sink,4))) return -1;
    if (x.blocks <= 0 || x.page_size <= 0 || x.page_size > INT_MAX/528 ||
        x.blocks > (INT_MAX/528)/x.page_size || x.topk <= 0 || x.topk % 64 ||
        x.extra_topk < 0 || x.extra_topk % 64 || x.topk > (1<<20) ||
        x.extra_topk > (1<<20) || (batch_invariant != 0 && batch_invariant != 1) || !std::isfinite(x.scale) || x.scale <= 0 ||
        !std::isfinite(x.scale*1.4426950408889634074f)) return -1;
    if (x.extra_topk) {
        if (!aligned(x.extra_kv,16) || !aligned(x.extra_indices,8) ||
            (x.extra_length && !aligned(x.extra_length,4)) ||
            x.extra_blocks <= 0 || x.extra_page_size <= 0 ||
            x.extra_page_size > INT_MAX/528 || x.extra_blocks > (INT_MAX/528)/x.extra_page_size) return -1;
    } else if (x.extra_kv || x.extra_indices || x.extra_length || x.extra_blocks || x.extra_page_size) return -1;
    int device = -1;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return static_cast<int>(status);
    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop,device);
    if (status != cudaSuccess) return static_cast<int>(status);
    if (prop.major != 10 || (prop.minor != 0 && prop.minor != 3)) return -2;
    const bool split = !batch_invariant && x.topk+x.extra_topk>640;
    const int parts = prop.multiProcessorCount;
    if (split && (parts <= 0 || parts > 160)) return -2;
    if (split) {
        if (!workspace || !aligned(workspace->metadata,32) || !aligned(workspace->num_splits,4) ||
            !aligned(workspace->partial_lse,16) || !aligned(workspace->partial_output,16) ||
            workspace->metadata_bytes != (size_t)parts*sizeof(DecodingSchedMeta) ||
            workspace->num_splits_bytes != 2*sizeof(int) ||
            workspace->partial_lse_bytes != (size_t)(parts+1)*64*sizeof(float) ||
            workspace->partial_output_bytes != (size_t)(parts+1)*32768*sizeof(float)) return -1;
    }
    auto stream = static_cast<cudaStream_t>(opaque);
    cudaStreamCaptureStatus capture;
    status = cudaStreamIsCapturing(stream,&capture);
    if (status != cudaSuccess) return static_cast<int>(status);
    if (capture != cudaStreamCaptureStatusNone) return -1;
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
    SparseAttnDecodeParams p{};
    p.b=1; p.s_q=1; p.h_q=64; p.h_kv=1; p.d_qk=512; p.d_v=512;
    p.sm_scale=x.scale; p.sm_scale_div_log2=x.scale*1.4426950408889634074f;
    p.num_blocks=x.blocks; p.page_block_size=x.page_size; p.topk=x.topk;
    p.model_type=ModelType::V41; p.extra_model_type=ModelType::V41;
    p.q=(bf16*)x.query_bf16; p.kv=(bf16*)x.kv; p.indices=(int*)x.indices;
    p.topk_length=(int*)x.length; p.attn_sink=(float*)x.sink;
    p.out=(bf16*)x.output_bf16; p.lse=x.lse;
    p.extra_num_blocks=x.extra_blocks; p.extra_page_block_size=x.extra_page_size; p.extra_topk=x.extra_topk;
    p.extra_kv=(bf16*)x.extra_kv; p.extra_indices=(int*)x.extra_indices; p.extra_topk_length=(int*)x.extra_length;
    p.stride_q_b=32768; p.stride_q_s_q=32768; p.stride_q_h_q=512;
    // Upstream API receives uint8 FP8 storage: these KV strides are BYTES,
    // despite bf16 pointer declarations in SparseAttnDecodeParams.
    p.stride_kv_block=x.page_size*528; p.stride_kv_row=528;
    p.stride_indices_b=x.topk; p.stride_indices_s_q=x.topk;
    p.stride_lse_b=64; p.stride_lse_s_q=64;
    p.stride_o_b=32768; p.stride_o_s_q=32768; p.stride_o_h_q=512;
    p.stride_extra_kv_block=x.extra_page_size*528; p.stride_extra_kv_row=x.extra_topk?528:0;
    p.stride_extra_indices_b=x.extra_topk; p.stride_extra_indices_s_q=x.extra_topk;
    p.num_sm=prop.multiProcessorCount; p.stream=stream; p.enable_split_kv=false;
    try {
        if (split) {
            p.enable_split_kv=true; p.num_sm_parts=parts;
            p.tile_scheduler_metadata_ptr=(DecodingSchedMeta*)workspace->metadata;
            p.num_splits_ptr=(int*)workspace->num_splits;
            p.lse_accum=workspace->partial_lse; p.o_accum=workspace->partial_output;
            p.stride_lse_accum_split=64; p.stride_lse_accum_s_q=64;
            p.stride_o_accum_split=32768; p.stride_o_accum_s_q=32768; p.stride_o_accum_h_q=512;
            GetDecodeSchedMetaParams scheduler{1,1,64,5,parts,x.topk,x.extra_topk,
                (int*)x.length,(int*)x.extra_length,p.tile_scheduler_metadata_ptr,p.num_splits_ptr,stream};
            smxx::decode::run_get_decoding_sched_meta_kernel(scheduler);
            status=cudaPeekAtLastError(); if(status!=cudaSuccess)return (int)status;
            static constexpr decode::Config split_config{ModelType::V41,ModelType::V41,true};
            decode::run_flash_splitkv_mla_fp8_sparse_kernel<split_config>(p);
            status=cudaPeekAtLastError(); if(status!=cudaSuccess)return (int)status;
            smxx::decode::run_flash_mla_combine_kernel(p);
        } else {
            decode::run_flash_splitkv_mla_fp8_sparse_kernel<config>(p);
        }
    }
    catch (...) { return -3; }
    return static_cast<int>(cudaPeekAtLastError());
}

extern "C" int fak_flashmla_decode_nosplit_launch(const fak_flashmla_decode_input *x,void *stream) {
    if(!x || x->topk<0 || x->extra_topk<0 || x->topk>640 || x->extra_topk>640-x->topk)return -1;
    return fak_flashmla_decode_launch(x,nullptr,1,stream);
}
extern "C" int fak_flashmla_decode_workspace_query(int topk,int extra_topk,int invariant,
    fak_flashmla_decode_workspace_sizes *sizes) {
    if(!sizes)return -1; *sizes={};
    if(topk<=0 || topk>(1<<20) || topk%64 || extra_topk<0 || extra_topk>(1<<20) || extra_topk%64 ||
       (invariant!=0 && invariant!=1))return -1;
    int device;cudaError_t e=cudaGetDevice(&device);if(e)return (int)e;
    cudaDeviceProp prop{};e=cudaGetDeviceProperties(&prop,device);if(e)return (int)e;
    if(prop.major!=10 || (prop.minor!=0 && prop.minor!=3))return -2;
    if(invariant || topk+extra_topk<=640)return 0;
    int parts=prop.multiProcessorCount;if(parts<=0 || parts>160)return -2;
    *sizes={(size_t)parts*sizeof(DecodingSchedMeta),2*sizeof(int),
            (size_t)(parts+1)*64*sizeof(float),(size_t)(parts+1)*32768*sizeof(float),parts};
    return 0;
}
