// Standalone required-device component witness; no Go/model/default wiring.
// Run only with an authorized SM100a/SM103a target and exclusive resource grant.
// CUDA >=13.1, from repository root; adapt verified CUDA_HOME/CCCL include root:
// nvcc -std=c++20 -O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
//   -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
//   -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
//   -DFLASH_MLA_IS_BUILD_ON_CUDA -gencode=arch=compute_100a,code=sm_100a \
//   -Iinternal/compute -Ithird_party/flashmla/csrc \
//   -Ithird_party/flashmla/csrc/3rdparty/kerutils/include \
//   -Ithird_party/cutlass/include -I${CUDA_HOME}/include/cccl \
//   internal/compute/cuda_flashmla_sparse_decode.cu \
//   internal/compute/testdata/flashmla_decode_device_qualification.cu \
//   -lcuda -lcudart -o /owner-reserved-path/flashmla-decode-device
// Do not separately compile scheduler/combine .cu files included by adapter.
// Change BOTH architecture names to 103a only for an admitted matching target.
// SM100a adapter object was actually compiled; harness linking/execution and
// SM103a remain UNRUN. Never treat build success as device/parity/performance.
// Upstream FlashMLA MIT 2e5429fc5653bab6e081f09477126f731882a6a9;
// CUTLASS BSD-3 4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b, licenses retained.
#include "cuda_flashmla_sparse_decode.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

static void check(cudaError_t e,const char *where) {
    if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s: %d %s\n",where,(int)e,cudaGetErrorString(e));std::_Exit(2);}
}
static float from_bf16(uint16_t b){uint32_t bits=(uint32_t)b<<16;float f;std::memcpy(&f,&bits,4);return f;}

static std::vector<uint16_t> run(int extra_rows,bool invalid,int invariant) {
    constexpr int primary_rows=128,heads=64,dim=512,guard=64;
    int total=primary_rows+extra_rows;
    std::vector<uint8_t> kv(primary_rows*528),extra(extra_rows*528);
    for(int row=0;row<primary_rows;row++){
        std::memset(kv.data()+row*528,0x38,512); // exact E4M3 value1
        std::memset(kv.data()+row*528+512,127,16); // exact E8M0 scale1
    }
    for(int row=0;row<extra_rows;row++){
        std::memset(extra.data()+row*528,0x40,512); // exact E4M3 value2
        std::memset(extra.data()+row*528+512,127,16);
    }
    std::vector<uint16_t> q(heads*dim,0),out(heads*dim+guard);
    std::vector<int32_t> ids(primary_rows),extra_ids(extra_rows);
    for(int i=0;i<primary_rows;i++)ids[i]=invalid?-1:i;
    for(int i=0;i<extra_rows;i++)extra_ids[i]=invalid?-1:i;
    float sink_value=std::log((float)total);
    std::vector<float> sink(heads,sink_value),lse(heads+guard);
    std::vector<void*> owned;
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream create");
    auto alloc=[&](size_t bytes,const void *host=nullptr){
        void *p=nullptr;check(cudaMalloc(&p,bytes),"allocate");owned.push_back(p);
        if(host)check(cudaMemcpyAsync(p,host,bytes,cudaMemcpyHostToDevice,stream),"upload");
        return p;
    };
    fak_flashmla_decode_input x{};
    x.query_bf16=(uint16_t*)alloc(q.size()*2,q.data());
    x.kv=(uint8_t*)alloc(kv.size(),kv.data());x.indices=(int32_t*)alloc(ids.size()*4,ids.data());
    x.extra_kv=(uint8_t*)alloc(extra.size(),extra.data());x.extra_indices=(int32_t*)alloc(extra_ids.size()*4,extra_ids.data());
    x.sink=(float*)alloc(sink.size()*4,sink.data());
    x.output_bf16=(uint16_t*)alloc(out.size()*2);x.lse=(float*)alloc(lse.size()*4);
    check(cudaMemsetAsync(x.output_bf16,0x5a,out.size()*2,stream),"output guard");
    check(cudaMemsetAsync(x.lse,0x5a,lse.size()*4,stream),"lse guard");
    x.blocks=1;x.page_size=primary_rows;x.topk=primary_rows;
    x.extra_blocks=extra_rows/64;x.extra_page_size=64;x.extra_topk=extra_rows;
    x.scale=1/std::sqrt(512.f);
    fak_flashmla_decode_workspace_sizes sizes{};
    int rc=fak_flashmla_decode_workspace_query(x.topk,x.extra_topk,invariant,&sizes);
    if(rc){std::fprintf(stderr,"workspace query refused %d\n",rc);std::_Exit(3);}
    fak_flashmla_decode_workspace w{};
    if(sizes.partitions){
        w.metadata=alloc(sizes.metadata);w.metadata_bytes=sizes.metadata;
        w.num_splits=(int32_t*)alloc(sizes.num_splits);w.num_splits_bytes=sizes.num_splits;
        w.partial_lse=(float*)alloc(sizes.partial_lse);w.partial_lse_bytes=sizes.partial_lse;
        w.partial_output=(float*)alloc(sizes.partial_output);w.partial_output_bytes=sizes.partial_output;
    }
    rc=fak_flashmla_decode_launch(&x,sizes.partitions?&w:nullptr,invariant,(void*)stream);
    // On any failure this must conclusively drain before freeing any operand.
    check(cudaStreamSynchronize(stream),"operation drain");
    if(rc){std::fprintf(stderr,"launch refused/failed %d\n",rc);std::_Exit(4);}
    check(cudaMemcpy(out.data(),x.output_bf16,out.size()*2,cudaMemcpyDeviceToHost),"output read");
    check(cudaMemcpy(lse.data(),x.lse,lse.size()*4,cudaMemcpyDeviceToHost),"lse read");
    float want=invalid?0.f:(primary_rows+2.f*extra_rows)/(total+std::exp(sink_value));
    float max_error=0;
    for(int i=0;i<heads*dim;i++){
        float got=from_bf16(out[i]);float error=std::abs(got-want);
        if(!std::isfinite(got)||error>0.008f||(invalid&&got!=0.f)){
            std::fprintf(stderr,"FAIL rowcount=%d split=%d invalid=%d i=%d got=%g want=%g\n",total,sizes.partitions,invalid,i,got,want);std::_Exit(5);
        }
        if(error>max_error)max_error=error;
    }
    for(int i=heads*dim;i<(int)out.size();i++)if(out[i]!=0x5a5a)std::_Exit(6);
    // Upstream publishes LSE before adding sink to the output denominator:
    // head64/kernel.cuh296-299 and combine.cu127-129 at the pinned revision.
    // Empty attention uses +Inf as a sentinel, not the sink's log weight.
    for(int h=0;h<heads;h++){
        if(invalid){if(!std::isinf(lse[h])||lse[h]<0)std::_Exit(9);}
        else if(!std::isfinite(lse[h])||std::abs(lse[h]-std::log((float)total))>2e-4f)std::_Exit(10);
    }
    for(int i=heads;i<(int)lse.size();i++){uint32_t b;std::memcpy(&b,&lse[i],4);if(b!=0x5a5a5a5a)std::_Exit(7);}
    for(void *p:owned)check(cudaFree(p),"free after drain");
    check(cudaStreamDestroy(stream),"stream destroy");
    std::printf("PASS real FlashMLA rows=%d partitions=%d invalid=%d invariant=%d max_error=%g\n",total,sizes.partitions,invalid,invariant,max_error);
    out.resize(heads*dim);return out;
}
int main(){
    cudaDeviceProp prop;int device;check(cudaGetDevice(&device),"device");check(cudaGetDeviceProperties(&prop,device),"properties");
    std::printf("device=%d name=%s sm=%d%d sms=%d\n",device,prop.name,prop.major,prop.minor,prop.multiProcessorCount);
    if(prop.major!=10||(prop.minor!=0&&prop.minor!=3)){std::fprintf(stderr,"unsupported target; no test run\n");return 77;}
    run(512,false,0);run(512,true,0);
    auto split=run(4096,false,0);auto invariant=run(4096,false,1);run(4096,true,0);
    for(size_t i=0;i<split.size();i++)if(std::abs(from_bf16(split[i])-from_bf16(invariant[i]))>0.008f)return 8;
    puts("PASS device known-value/sink/all-invalid/split-vs-invariant qualification only; no general model parity or performance claim");
}
