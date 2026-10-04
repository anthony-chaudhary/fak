//go:build darwin && arm64 && cgo

#import <Metal/Metal.h>
#include <limits.h>
#include <math.h>

extern id<MTLDevice> gDev;
extern void *mg_graph_command_buffer(void *graph);
extern void *mg_graph_alloc_result(void *graph, int n);
extern void *mg_graph_alloc_buffer(void *graph, int n);
extern void mg_graph_note_encoder(void *graph);
extern int mg_graph_prompt(void *graph);
extern int mg_graph_input(void *graph);

static id<MTLComputePipelineState> qgNorm, qgAdd, qgSwiGLU, qgSplit, qgQK, qgAttn, qgAttnOnline, qgLaneSplit, qgLaneQK, qgLaneAttn, qgAttnSplit, qgAttnCombine, qgBias;
static id<MTLComputePipelineState> qgAttnMLX[8];
static id<MTLComputePipelineState> qgAttnMLXPart[8];
static unsigned long long qgMLXDispatchCount;
static BOOL qgAttempted, qgReady;

static NSString *qgSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;
#include <metal_simdgroup>

kernel void qg_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                    device float *y [[buffer(2)]], constant int& width [[buffer(3)]],
                    constant float& eps [[buffer(4)]], constant int& gain1p [[buffer(5)]],
                    constant int& sourceRow [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_threadgroup]]) {
    int r = sourceRow >= 0 ? sourceRow : (int)row;
    threadgroup float sums[256]; float ss=0.0f;
    for(int i=(int)lane;i<width;i+=256){float v=x[(long)r*width+i];ss+=v*v;}
    sums[lane]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint o=128;o;o>>=1){if(lane<o)sums[lane]+=sums[lane+o];threadgroup_barrier(mem_flags::mem_threadgroup);}
    float inv=rsqrt(sums[0]/(float)width+eps);
    int dr=sourceRow>=0?0:(int)row;
    for(int i=(int)lane;i<width;i+=256){float gain=gain1p?1.0f+w[i]:w[i];y[(long)dr*width+i]=x[(long)r*width+i]*inv*gain;}
}

kernel void qg_add(device float*x [[buffer(0)]],device const float*y [[buffer(1)]],constant int&n [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<(uint)n)x[i]+=y[i];}
kernel void qg_bias(device float*x [[buffer(0)]],device const float*b [[buffer(1)]],constant int&width [[buffer(2)]],constant int&n [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i<(uint)n)x[i]+=b[i%(uint)width];}
kernel void qg_swiglu(device float*g [[buffer(0)]],device const float*u [[buffer(1)]],constant int&n [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<(uint)n){float v=g[i];g[i]=(v/(1.0f+exp(-v)))*u[i];}}
kernel void qg_split(device const float*src [[buffer(0)]],device float*q [[buffer(1)]],device float*gate [[buffer(2)]],constant int&qwidth [[buffer(3)]],constant int&hd [[buffer(4)]],constant int&rows [[buffer(5)]],uint i [[thread_position_in_grid]]){if(i<(uint)(rows*qwidth)){int t=(int)i/qwidth,j=(int)i-t*qwidth,h=j/hd,d=j-h*hd;long base=(long)t*2*qwidth+(long)h*2*hd;q[i]=src[base+d];gate[i]=src[base+hd+d];}}

kernel void qg_qk(device const float*qIn [[buffer(0)]],device const float*kIn [[buffer(1)]],
                  device const float*qw [[buffer(2)]],device const float*kw [[buffer(3)]],
                  device float*qOut [[buffer(4)]],device float*kRaw [[buffer(5)]],device float*kPost [[buffer(6)]],
                  constant int&nH [[buffer(7)]],constant int&nKV [[buffer(8)]],constant int&hd [[buffer(9)]],constant int&rotary [[buffer(10)]],
                  constant int&base [[buffer(11)]],device const float*cosv [[buffer(12)]],device const float*sinv [[buffer(13)]],constant float&qkEps [[buffer(14)]],constant int&gain1p [[buffer(15)]],constant int&qknorm [[buffer(16)]],constant int&qnw [[buffer(17)]],constant int&knw [[buffer(18)]],
                  uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    int h=(int)group.x,t=(int)group.y;threadgroup float qs[256],ks[256];
    float q=0,k=0;if(h<nH&&lane<(uint)hd)q=qIn[((long)t*nH+h)*hd+lane];if(h<nKV&&lane<(uint)hd)k=kIn[((long)t*nKV+h)*hd+lane];
    float qss=0,kss=0;
    if(qnw==hd)qss=q*q;else for(int i=(int)lane;i<nH*hd;i+=256){float z=qIn[(long)t*nH*hd+i];qss+=z*z;}
    if(knw==hd)kss=k*k;else for(int i=(int)lane;i<nKV*hd;i+=256){float z=kIn[(long)t*nKV*hd+i];kss+=z*z;}
    qs[lane]=qss;ks[lane]=kss;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint o=128;o;o>>=1){if(lane<o){qs[lane]+=qs[lane+o];ks[lane]+=ks[lane+o];}threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(qknorm&&h<nH&&lane<(uint)hd){float w=qw[(qnw==hd?0:h*hd)+(int)lane];q*=rsqrt(qs[0]/(float)qnw+qkEps)*(gain1p?1.0f+w:w);}
    if(qknorm&&h<nKV&&lane<(uint)hd){float w=kw[(knw==hd?0:h*hd)+(int)lane];k*=rsqrt(ks[0]/(float)knw+qkEps)*(gain1p?1.0f+w:w);}
    if(h<nH&&lane<(uint)hd)qOut[((long)t*nH+h)*hd+lane]=q;
    if(h<nKV&&lane<(uint)hd){long ix=((long)t*nKV+h)*hd+lane;kRaw[ix]=k;kPost[ix]=k;}
    // The rotate-half below reads qOut/kPost elements another SIMD-group wrote to DEVICE
    // memory, so the barrier must order device writes, not only threadgroup memory.
    threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
    int halfn=rotary/2;if(lane<(uint)halfn){int j=(int)lane;float c=cosv[(long)t*halfn+j],s=sinv[(long)t*halfn+j];
        if(h<nH){long i=((long)t*nH+h)*hd+j;float av=qOut[i],bv=qOut[i+halfn];qOut[i]=av*c-bv*s;qOut[i+halfn]=av*s+bv*c;}
        if(h<nKV){long i=((long)t*nKV+h)*hd+j;float av=kPost[i],bv=kPost[i+halfn];kPost[i]=av*c-bv*s;kPost[i+halfn]=av*s+bv*c;}}
}

// MLX sdpa_vector single-pass core, copied and adapted from
// mlx/backend/metal/kernels/sdpa_vector.h@073d2252c96754e9e57ed6633be3f8ecb7058548.
// Copyright © 2024 Apple Inc.
// Changes: fp32 token-major strides, causal row bound, optional gated epilogue, wrappers.
/*
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
template <int D>
void qg_mlx_attention(const device float* queries, const device float* keys,
    const device float* values, const device float* gate, device float* out,
    int N, int gqa_factor, int nH, int nKV, float scale, int gated,
    uint3 tid, uint simd_gid, uint simd_lid,
    threadgroup float* outputs, threadgroup float* max_scores,
    threadgroup float* sum_exp_scores) {
  constexpr int V = D;
  const int k_head_stride = D;
  const int v_head_stride = D;
  const int k_seq_stride = nKV * D;
  const int v_seq_stride = nKV * D;
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];


  // Fak rows are token-major fp32, with interleaved KV heads.
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_seq_idx * nH + q_batch_head_idx;
  queries += o_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  gate += o_offset * V + simd_gid * v_per_thread;
  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -3.402823466e+38f; // MLX Limits<float>::finite_min.
  U sum_exp_score = 0;
  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;

  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = gated ? o[i] / (1.0f + exp(-gate[i])) : o[i];
    }
  }
}

// qg_mlx_attention_part is the first pass of the MLX sdpa_vector two-pass split
// (sdpa_vector_2pass_1 shape): ONE 1024-thread threadgroup per (head, KV split), with the
// split's keys strided over 32 SIMDgroups exactly as qg_mlx_attention strides the whole
// axis. It writes the split's UNNORMALIZED partial (acc[D], running max m, denominator)
// in the qg_attn_split layout, so qg_attn_combine merges it unchanged. The historical
// qg_attn_split gives each (head, split) one 32-lane SIMDgroup walking ~2048 keys
// serially: at a 2k-token dense decode that is 2*nH SIMDgroups for the whole GPU and
// measured ~5 ms per layer (fak#13599), against ~0.1 ms for this shape.
template <int D>
void qg_mlx_attention_part(const device float* queries, const device float* keys,
    const device float* values, device float* part, int lo, int hi, int splits,
    int gqa_factor, int nKV, float scale, uint3 tid, uint simd_gid, uint simd_lid,
    threadgroup float* outputs, threadgroup float* max_scores,
    threadgroup float* sum_exp_scores) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int per_thread = D / BD;
  const int seq_stride = nKV * D;
  const int head = tid.x, split = tid.y, kv_head = head / gqa_factor;
  queries += head * D + simd_lid * per_thread;
  keys += kv_head * D + (lo + (int)simd_gid) * seq_stride + simd_lid * per_thread;
  values += kv_head * D + (lo + (int)simd_gid) * seq_stride + simd_lid * per_thread;
  float q[per_thread], o[per_thread];
  for (int i = 0; i < per_thread; i++) { q[i] = scale * queries[i]; o[i] = 0; }
  float max_score = -3.402823466e+38f, sum_exp_score = 0;
  for (int i = lo + (int)simd_gid; i < hi; i += BN) {
    float score = 0;
    for (int j = 0; j < per_thread; j++) score += q[j] * keys[j];
    score = simd_sum(score);
    float new_max = max(max_score, score);
    float factor = fast::exp(max_score - new_max), exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < per_thread; j++) o[j] = o[j] * factor + exp_score * values[j];
    keys += BN * seq_stride;
    values += BN * seq_stride;
  }
  if (simd_lid == 0) { max_scores[simd_gid] = max_score; sum_exp_scores[simd_gid] = sum_exp_score; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  float new_max = simd_max(max_score);
  float factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  long po = ((long)head * splits + split) * (D + 2);
  for (int i = 0; i < per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    for (int i = 0; i < per_thread; i++) part[po + simd_gid * per_thread + i] = o[i];
    if (simd_gid == 0) { part[po + D] = new_max; part[po + D + 1] = sum_exp_score; }
  }
}

#define QG_MLX_PART(D) \
kernel void qg_attn_mlx_part_##D(device const float*q [[buffer(0)]], device const float*k [[buffer(1)]], \
    device const float*v [[buffer(2)]], device float*part [[buffer(3)]], constant int&total [[buffer(4)]], \
    constant int&base [[buffer(5)]], constant int&nH [[buffer(6)]], constant int&nKV [[buffer(7)]], \
    constant int&hd [[buffer(8)]], constant float&scale [[buffer(9)]], constant int&splits [[buffer(10)]], \
    constant int&chunk [[buffer(11)]], uint3 tid [[threadgroup_position_in_grid]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  threadgroup float outputs[32*32], max_scores[32], sum_exp_scores[32]; \
  int upto = min(total, base + 1), lo = (int)tid.y * chunk, hi = min(lo + chunk, upto); \
  qg_mlx_attention_part<D>(q,k,v,part,lo,hi,splits,nH/nKV,nKV,scale,tid,sg,lane,outputs,max_scores,sum_exp_scores); \
}
QG_MLX_PART(32)
QG_MLX_PART(64)
QG_MLX_PART(96)
QG_MLX_PART(128)
QG_MLX_PART(160)
QG_MLX_PART(192)
QG_MLX_PART(224)
QG_MLX_PART(256)
#undef QG_MLX_PART

#define QG_MLX_ATTN(D) \
kernel void qg_attn_mlx_##D(device const float*q [[buffer(0)]], device const float*k [[buffer(1)]], \
    device const float*v [[buffer(2)]], device const float*gate [[buffer(3)]], device float*out [[buffer(4)]], \
    constant int&total [[buffer(5)]], constant int&base [[buffer(6)]], constant int&nH [[buffer(7)]], \
    constant int&nKV [[buffer(8)]], constant int&hd [[buffer(9)]], constant float&scale [[buffer(10)]], \
    constant int&gated [[buffer(11)]], uint3 tid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  threadgroup float outputs[32*32], max_scores[32], sum_exp_scores[32]; \
  qg_mlx_attention<D>(q,k,v,gate,out,min(total,base+int(tid.y)+1),nH/nKV,nH,nKV,scale,gated,tid,sg,lane,outputs,max_scores,sum_exp_scores); \
}
QG_MLX_ATTN(32)
QG_MLX_ATTN(64)
QG_MLX_ATTN(96)
QG_MLX_ATTN(128)
QG_MLX_ATTN(160)
QG_MLX_ATTN(192)
QG_MLX_ATTN(224)
QG_MLX_ATTN(256)
#undef QG_MLX_ATTN

kernel void qg_attn(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*gate [[buffer(3)]],device float*out [[buffer(4)]],
                    constant int&total [[buffer(5)]],constant int&base [[buffer(6)]],constant int&nH [[buffer(7)]],constant int&nKV [[buffer(8)]],constant int&hd [[buffer(9)]],constant float&scale [[buffer(10)]],constant int&gated [[buffer(11)]],
                    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    int h=(int)group.x,t=(int)group.y,kh=h/(nH/nKV),upto=base+t+1;threadgroup float score[4096],mx,den;
    if(lane==0){float m=-INFINITY;for(int j=0;j<upto;j++){float z=0;for(int d=0;d<hd;d++)z+=q[((long)t*nH+h)*hd+d]*k[((long)j*nKV+kh)*hd+d];z*=scale;score[j]=z;m=max(m,z);}float sum=0;for(int j=0;j<upto;j++){score[j]=exp(score[j]-m);sum+=score[j];}mx=m;den=sum;}threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane<(uint)hd){float z=0;for(int j=0;j<upto;j++)z+=score[j]*v[((long)j*nKV+kh)*hd+lane];long i=((long)t*nH+h)*hd+lane;out[i]=gated?(z/den)/(1.0f+exp(-gate[i])):z/den;}
}

// Long-context ordered attention uses the same register-resident online
// softmax recurrence as split-KV FlashDecoding. One SIMDgroup owns a query
// head, so attention state is O(head_dim) rather than O(context).
kernel void qg_attn_online(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*gate [[buffer(3)]],device float*out [[buffer(4)]],
                           constant int&total [[buffer(5)]],constant int&base [[buffer(6)]],constant int&nH [[buffer(7)]],constant int&nKV [[buffer(8)]],constant int&hd [[buffer(9)]],constant float&scale [[buffer(10)]],constant int&gated [[buffer(11)]],
                           uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    int h=(int)group.x,t=(int)group.y,kh=h/(nH/nKV),upto=min(total,base+t+1);float qr[8],acc[8];
    for(int i=0;i<8;i++){int d=(int)lane+i*32;qr[i]=d<hd?q[((long)t*nH+h)*hd+d]:0.0f;acc[i]=0.0f;}
    float m=-INFINITY,den=0.0f;
    for(int j=0;j<upto;j++){
        float partial=0.0f;for(int i=0;i<8;i++){int d=(int)lane+i*32;if(d<hd)partial+=qr[i]*k[((long)j*nKV+kh)*hd+d];}
        float score=simd_sum(partial)*scale,newm=max(m,score),alpha=isfinite(m)?exp(m-newm):0.0f,p=exp(score-newm);
        den=den*alpha+p;m=newm;
        for(int i=0;i<8;i++){int d=(int)lane+i*32;if(d<hd)acc[i]=acc[i]*alpha+p*v[((long)j*nKV+kh)*hd+d];}
    }
    for(int i=0;i<8;i++){int d=(int)lane+i*32;if(d<hd){long ix=((long)t*nH+h)*hd+d;out[ix]=gated?(acc[i]/den)/(1.0f+exp(-gate[ix])):acc[i]/den;}}
}


// qg_attn_split + qg_attn_combine: split-KV flash decoding for the P=1 decode
// attention. The historical qg_attn_online gives one SIMDgroup per query head and
// walks the whole KV axis serially, so at long context a decode token is limited to
// nH-wide parallelism and is GPU-bound (measured ~20 ms for ONE full-attention layer
// at base=20480). Split-KV partitions the KV axis into `splits` ranges; one
// SIMDgroup per (head, split) computes a partial online-softmax state (m, den,
// acc[hd]) in parallel, then qg_attn_combine merges the partials with a global max.
// This exposes nH*splits-wide parallelism, the same split-KV/FlashDecoding idea the
// batch path uses. Partial layout: part[(h*splits+s)*(hd+2) + {0:dim..hd-1:dim, hd:m, hd+1:den}].
kernel void qg_attn_split(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*part [[buffer(3)]],
                          constant int&total [[buffer(4)]],constant int&base [[buffer(5)]],constant int&nH [[buffer(6)]],constant int&nKV [[buffer(7)]],constant int&hd [[buffer(8)]],constant float&scale [[buffer(9)]],
                          constant int&splits [[buffer(10)]],constant int&chunk [[buffer(11)]],
                          uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    int h=(int)group.x,s=(int)group.y,kh=h/(nH/nKV),upto=min(total,base+1);
    int lo=s*chunk,hi=min(lo+chunk,upto);
    float qr[8],acc[8];for(int i=0;i<8;i++){int d=(int)lane+i*32;qr[i]=d<hd?q[(long)h*hd+d]:0.0f;acc[i]=0.0f;}
    float m=-INFINITY,den=0.0f;
    for(int j=lo;j<hi;j++){
        float partial=0.0f;for(int i=0;i<8;i++){int d=(int)lane+i*32;if(d<hd)partial+=qr[i]*k[((long)j*nKV+kh)*hd+d];}
        float score=simd_sum(partial)*scale,newm=max(m,score),alpha=isfinite(m)?exp(m-newm):0.0f,p=exp(score-newm);
        den=den*alpha+p;m=newm;
        for(int i=0;i<8;i++){int d=(int)lane+i*32;if(d<hd)acc[i]=acc[i]*alpha+p*v[((long)j*nKV+kh)*hd+d];}
    }
    long po=((long)h*splits+s)*(hd+2);
    for(int i=0;i<8;i++){int d=(int)lane+i*32;if(d<hd)part[po+d]=acc[i];}
    if(lane==0){part[po+hd]=m;part[po+hd+1]=den;}
}
kernel void qg_attn_combine(device const float*part [[buffer(0)]],device const float*gate [[buffer(1)]],device float*out [[buffer(2)]],
                            constant int&nH [[buffer(3)]],constant int&hd [[buffer(4)]],constant int&splits [[buffer(5)]],constant int&gated [[buffer(6)]],
                            uint h [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    if(h>=(uint)nH)return;
    float m=-INFINITY;
    for(int s=0;s<splits;s++){float ps=part[(long)(h*splits+s)*(hd+2)+hd];m=max(m,ps);}
    float den=0.0f,acc[8];for(int i=0;i<8;i++)acc[i]=0.0f;
    for(int s=0;s<splits;s++){
        long po=(long)(h*splits+s)*(hd+2);float ps=part[po+hd];float w=isfinite(ps)?exp(ps-m):0.0f;
        den+=w*part[po+hd+1];
        for(int i=0;i<8;i++){int d=(int)lane+i*32;if(d<hd)acc[i]+=w*part[po+d];}
    }
    for(int i=0;i<8;i++){int d=(int)lane+i*32;if(d<hd){long ix=(long)h*hd+d;out[ix]=gated?(acc[i]/den)/(1.0f+exp(-gate[ix])):acc[i]/den;}}
}

kernel void qg_lane_split(device const float*src [[buffer(0)]],device float*q [[buffer(1)]],device float*gate [[buffer(2)]],constant int&batch [[buffer(3)]],constant int&qwidth [[buffer(4)]],constant int&hd [[buffer(5)]],uint i [[thread_position_in_grid]]){if(i<(uint)(batch*qwidth)){int r=(int)i/qwidth,j=(int)i-r*qwidth,h=j/hd,d=j-h*hd;long base=(long)r*2*qwidth+(long)h*2*hd;q[i]=src[base+d];gate[i]=src[base+hd+d];}}

kernel void qg_lane_qk(device const float*qIn [[buffer(0)]],device const float*kIn [[buffer(1)]],device const float*qw [[buffer(2)]],device const float*kw [[buffer(3)]],device float*qOut [[buffer(4)]],device float*kRaw [[buffer(5)]],device float*kPost [[buffer(6)]],constant int&batch [[buffer(7)]],constant int&nH [[buffer(8)]],constant int&nKV [[buffer(9)]],constant int&hd [[buffer(10)]],constant int&rotary [[buffer(11)]],device const int*pos [[buffer(12)]],device const float*cosv [[buffer(13)]],device const float*sinv [[buffer(14)]],constant float&qkEps [[buffer(15)]],constant int&gain1p [[buffer(16)]],constant int&qknorm [[buffer(17)]],constant int&qnw [[buffer(18)]],constant int&knw [[buffer(19)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    int h=(int)group.x,r=(int)group.y;threadgroup float qs[256],ks[256];if(r>=batch)return;
    float q=0,k=0;if(h<nH&&lane<(uint)hd)q=qIn[((long)r*nH+h)*hd+lane];if(h<nKV&&lane<(uint)hd)k=kIn[((long)r*nKV+h)*hd+lane];float qss=0,kss=0;
    if(qnw==hd)qss=q*q;else for(int i=(int)lane;i<nH*hd;i+=256){float z=qIn[(long)r*nH*hd+i];qss+=z*z;}if(knw==hd)kss=k*k;else for(int i=(int)lane;i<nKV*hd;i+=256){float z=kIn[(long)r*nKV*hd+i];kss+=z*z;}
    qs[lane]=qss;ks[lane]=kss;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint o=128;o;o>>=1){if(lane<o){qs[lane]+=qs[lane+o];ks[lane]+=ks[lane+o];}threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(qknorm&&h<nH&&lane<(uint)hd){float w=qw[(qnw==hd?0:h*hd)+(int)lane];q*=rsqrt(qs[0]/(float)qnw+qkEps)*(gain1p?1.0f+w:w);}if(qknorm&&h<nKV&&lane<(uint)hd){float w=kw[(knw==hd?0:h*hd)+(int)lane];k*=rsqrt(ks[0]/(float)knw+qkEps)*(gain1p?1.0f+w:w);}
    if(h<nH&&lane<(uint)hd)qOut[((long)r*nH+h)*hd+lane]=q;if(h<nKV&&lane<(uint)hd){long ix=((long)r*nKV+h)*hd+lane;kRaw[ix]=k;kPost[ix]=k;}threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
    int halfn=rotary/2;if(lane<(uint)halfn){int j=(int)lane,p=pos[r];float c=cosv[(long)p*halfn+j],sn=sinv[(long)p*halfn+j];if(h<nH){long i=((long)r*nH+h)*hd+j;float x=qOut[i],y=qOut[i+halfn];qOut[i]=x*c-y*sn;qOut[i+halfn]=x*sn+y*c;}if(h<nKV){long i=((long)r*nKV+h)*hd+j;float x=kPost[i],y=kPost[i+halfn];kPost[i]=x*c-y*sn;kPost[i+halfn]=x*sn+y*c;}}
}

// qg_lane_attn: independent-lane (batched decode) full attention over each
// lane's own KV prefix. Historically this materialized a threadgroup score[4096],
// which hard-capped every lane at a 4096-token context and forced the caller to
// decline longer sessions (positions[r] >= 4096). It now uses the SAME
// one-SIMDgroup-per-(head,lane) ordered online-softmax recurrence as
// qg_attn_online / qg_attn_split: attention state is O(head_dim), not
// O(context), so there is no fixed register/threadgroup array and no 4096 cap.
// Each lane r attends to its own contiguous KV range [offsets[r], offsets[r]+n).
kernel void qg_lane_attn(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*gate [[buffer(3)]],device float*out [[buffer(4)]],device const int*offsets [[buffer(5)]],device const int*lengths [[buffer(6)]],constant int&batch [[buffer(7)]],constant int&nH [[buffer(8)]],constant int&nKV [[buffer(9)]],constant int&hd [[buffer(10)]],constant float&scale [[buffer(11)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    int h=(int)group.x,r=(int)group.y,kh=h/(nH/nKV);if(r>=batch)return;int off=offsets[r],n=lengths[r];float qr[8],acc[8];
    for(int i=0;i<8;i++){int d=(int)lane+i*32;qr[i]=d<hd?q[((long)r*nH+h)*hd+d]:0.0f;acc[i]=0.0f;}
    float m=-INFINITY,den=0.0f;
    for(int j=0;j<n;j++){
        float partial=0.0f;for(int i=0;i<8;i++){int d=(int)lane+i*32;if(d<hd)partial+=qr[i]*k[((long)(off+j)*nKV+kh)*hd+d];}
        float score=simd_sum(partial)*scale,newm=max(m,score),alpha=isfinite(m)?exp(m-newm):0.0f,p=exp(score-newm);
        den=den*alpha+p;m=newm;
        for(int i=0;i<8;i++){int d=(int)lane+i*32;if(d<hd)acc[i]=acc[i]*alpha+p*v[((long)(off+j)*nKV+kh)*hd+d];}
    }
    for(int i=0;i<8;i++){int d=(int)lane+i*32;if(d<hd){long ix=((long)r*nH+h)*hd+d;out[ix]=(acc[i]/den)/(1.0f+exp(-gate[ix]));}}
}
)MSL";

static id<MTLComputePipelineState> qg_pipeline(id<MTLLibrary> library, NSString *name, NSError **error) {
    id<MTLFunction> function=[library newFunctionWithName:name];
    if(!function)return nil;
    return[gDev newComputePipelineStateWithFunction:function error:error];
}

static int qg_init(void) {
    @synchronized(gDev) {
        if(qgReady)return 1;
        if(qgAttempted)return 0;
        qgAttempted=YES;
        NSError *error=nil;
        id<MTLLibrary> library=[gDev newLibraryWithSource:qgSource options:nil error:&error];
        if(!library){NSLog(@"qwen35 graph compile: %@",error);return 0;}

        // Keep initialization transactional. Publishing even one pipeline before
        // all lane and P32 functions exist lets a partial set masquerade as a
        // ready graph and turns a clean admission decline into a nil-PSO signal.
        id<MTLComputePipelineState> norm=qg_pipeline(library,@"qg_norm",&error);
        id<MTLComputePipelineState> add=qg_pipeline(library,@"qg_add",&error);
        id<MTLComputePipelineState> swiglu=qg_pipeline(library,@"qg_swiglu",&error);
        id<MTLComputePipelineState> split=qg_pipeline(library,@"qg_split",&error);
        id<MTLComputePipelineState> qk=qg_pipeline(library,@"qg_qk",&error);
        id<MTLComputePipelineState> attn=qg_pipeline(library,@"qg_attn",&error);
        id<MTLComputePipelineState> attnOnline=qg_pipeline(library,@"qg_attn_online",&error);
        id<MTLComputePipelineState> laneSplit=qg_pipeline(library,@"qg_lane_split",&error);
        id<MTLComputePipelineState> laneQK=qg_pipeline(library,@"qg_lane_qk",&error);
        id<MTLComputePipelineState> laneAttn=qg_pipeline(library,@"qg_lane_attn",&error);
        id<MTLComputePipelineState> attnSplit=qg_pipeline(library,@"qg_attn_split",&error);
        id<MTLComputePipelineState> attnCombine=qg_pipeline(library,@"qg_attn_combine",&error);
        id<MTLComputePipelineState> bias=qg_pipeline(library,@"qg_bias",&error);
        if(!bias||!norm||!add||!swiglu||!split||!qk||!attn||!attnOnline||!laneSplit||!laneQK||!laneAttn||!attnSplit||!attnCombine){
            NSLog(@"qwen35 graph pipeline initialization failed: %@",error);
            return 0;
        }
        qgNorm=norm;qgAdd=add;qgSwiGLU=swiglu;qgSplit=split;qgQK=qk;qgAttn=attn;qgAttnOnline=attnOnline;
        qgLaneSplit=laneSplit;qgLaneQK=laneQK;qgLaneAttn=laneAttn;qgAttnSplit=attnSplit;qgAttnCombine=attnCombine;qgBias=bias;
        // Candidate pipelines are optional: a device lacking 1024-thread groups
        // retains the proven graph, rather than declining all attention.
        for(int i=0;i<8;i++) {
            NSString *name=[NSString stringWithFormat:@"qg_attn_mlx_%d",(i+1)*32];
            id<MTLComputePipelineState> candidate=qg_pipeline(library,name,&error);
            if(candidate.maxTotalThreadsPerThreadgroup>=1024)qgAttnMLX[i]=candidate;
            name=[NSString stringWithFormat:@"qg_attn_mlx_part_%d",(i+1)*32];
            candidate=qg_pipeline(library,name,&error);
            if(candidate.maxTotalThreadsPerThreadgroup>=1024)qgAttnMLXPart[i]=candidate;
        }
        qgReady=YES;
        return 1;
    }
}

int mg_qwen35_graph_ready(void){return qg_init();}
static id<MTLBuffer> qg_host(const float*p,int n){return[gDev newBufferWithBytes:p length:(NSUInteger)n*sizeof(float) options:MTLResourceStorageModeShared];}
static void qg_dispatch(id<MTLComputeCommandEncoder>e,id<MTLComputePipelineState>p,int n){int t=(int)p.maxTotalThreadsPerThreadgroup;if(t>n)t=n;if(t<1)t=1;[e dispatchThreads:MTLSizeMake((NSUInteger)n,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)t,1,1)];}
// QG_MAX_ROWS mirrors metalgemm.PromptPanelMaxTokens (graph.go). The ordered-panel
// kernels loop generically over the graph's row count; the historical {1,2,3,4,32}
// enumeration was conservative, not a hardware bound. Keep the two in lockstep: a
// mismatch turns a wider Go-side admission into a native nil-PSO decline.
#define QG_MAX_ROWS 128
static int qg_ordered_rows(void*g){int rows=mg_graph_prompt(g);return rows>=1&&rows<=QG_MAX_ROWS?rows:0;}
// qg_split_policy picks split-KV flash decoding for the single-token decode at long
// context: one SIMDgroup per (head, KV split) exposes nH*splits-wide parallelism
// instead of one SIMDgroup per head walking the whole KV axis. Only at rows==1
// (decode); the prefill panels keep the proven all-rows paths. splits grows with
// total so each range stays ~1024-2048 tokens. Past the 32-split cap the chunk
// widens to cover every row: a fixed 2048 chunk with 32 splits only reached 65536
// tokens and silently dropped the tail of a longer context.
// Physical A/B keeps the historical path available; default dimensions have
// local hardware parity and attention-only GPU-time evidence.
unsigned long long mg_qwen35_attention_mlx_dispatch_count(void){
    return __atomic_load_n(&qgMLXDispatchCount,__ATOMIC_RELAXED);
}
static void qg_note_attention_dispatch(id<MTLComputePipelineState>p){
    for(int i=0;i<8;i++)if(p&&p==qgAttnMLX[i]){
        __atomic_add_fetch(&qgMLXDispatchCount,1,__ATOMIC_RELAXED);return;
    }
}
static id<MTLComputePipelineState> qg_attention_pipeline(int total,int hd,int*threads){
    const char*raw=getenv("FAK_QWEN35_ATTN_MLX");
    int useMLX=(hd==64||hd==128||hd==256);
    if(raw&&raw[0]=='0')useMLX=0;
    else if(raw&&raw[0]=='1')useMLX=1;
    if(useMLX&&total<=4096&&hd>=32&&hd<=256&&hd%32==0&&qgAttnMLX[hd/32-1]){
        *threads=1024;return qgAttnMLX[hd/32-1];
    }
    *threads=total<=4096?256:32;
    return total<=4096?qgAttn:qgAttnOnline;
}
// qg_split_pipeline picks the split-KV first pass: the 1024-thread MLX two-pass
// partial (qg_attn_mlx_part_D) whenever the head dim has one, else the historical
// one-SIMDgroup qg_attn_split. FAK_QWEN35_ATTN_SPLIT_MLX=0 restores the historical
// kernel (and its 2048-token chunk) for A/B.
static id<MTLComputePipelineState> qg_split_pipeline(int hd,int*threads){
    const char*raw=getenv("FAK_QWEN35_ATTN_SPLIT_MLX");
    if(!(raw&&raw[0]=='0')&&hd>=32&&hd<=256&&hd%32==0&&qgAttnMLXPart[hd/32-1]){*threads=1024;return qgAttnMLXPart[hd/32-1];}
    *threads=32;return qgAttnSplit;
}
// mlxPart sizes the chunk for the MLX partial: 1024 keys spread over 32 SIMDgroups
// (32 keys each) keeps every (head, split) threadgroup short while exposing
// nH*ceil(total/1024) threadgroups; the historical kernel keeps its 2048 chunk.
static void qg_split_policy_for(int rows,int total,int mlxPart,int*splits,int*chunk,int*useSplit){
    *splits=1;*chunk=mlxPart?1024:2048;*useSplit=0;
    const char*rawSplit=getenv("FAK_QWEN35_ATTN_SPLIT");
    if(rawSplit&&rawSplit[0]=='0')return;
    if(rows!=1||total<=2048)return;
    *splits=(total+*chunk-1)/(*chunk);
    if(*splits>32){*splits=32;*chunk=(total+*splits-1)/(*splits);}
    *useSplit=*splits>1;
}
static void qg_split_policy(int rows,int total,int*splits,int*chunk,int*useSplit){qg_split_policy_for(rows,total,0,splits,chunk,useSplit);}

void *mg_qwen35_graph_norm(void*g,void*input,const float*w,int rows,int width,float eps,int gain1p,int lastOnly){if(!qg_init()||!g||!input||!w||rows<=0||width<=0||eps<=0||(lastOnly&&!qg_ordered_rows(g))||mg_graph_prompt(g)!=rows)return NULL;id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);id<MTLBuffer>x=(__bridge id<MTLBuffer>)input,wb=qg_host(w,width),y=(__bridge id<MTLBuffer>)mg_graph_alloc_result(g,lastOnly?width:rows*width);if(!cb||!x||!wb||!y)return NULL;id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgNorm];[e setBuffer:x offset:0 atIndex:0];[e setBuffer:wb offset:0 atIndex:1];[e setBuffer:y offset:0 atIndex:2];[e setBytes:&width length:4 atIndex:3];[e setBytes:&eps length:4 atIndex:4];[e setBytes:&gain1p length:4 atIndex:5];int row=lastOnly?rows-1:-1;[e setBytes:&row length:4 atIndex:6];[e dispatchThreadgroups:MTLSizeMake(lastOnly?1:rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];return(__bridge void*)y;}
int mg_qwen35_graph_add(void*g,void*xp,void*yp,int n){if(!qg_init()||!g||!xp||!yp||n<=0)return 0;id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgAdd];[e setBuffer:(__bridge id<MTLBuffer>)xp offset:0 atIndex:0];[e setBuffer:(__bridge id<MTLBuffer>)yp offset:0 atIndex:1];[e setBytes:&n length:4 atIndex:2];qg_dispatch(e,qgAdd,n);[e endEncoding];mg_graph_note_encoder(g);return 1;}
// mg_qwen35_graph_bias broadcasts a host bias vector of `width` floats over `rows` rows of a
// graph-owned panel in place (x[i] += b[i % width]). The bias is staged through qg_host like
// every other per-op constant, so the caller accounts width*4 upload bytes per call.
int mg_qwen35_graph_bias(void*g,void*xp,const float*b,int rows,int width){if(!qg_init()||!g||!xp||!b||rows<=0||width<=0||rows>INT_MAX/width)return 0;id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);id<MTLBuffer>x=(__bridge id<MTLBuffer>)xp,bb=qg_host(b,width);int n=rows*width;if(!cb||!bb||(NSUInteger)n*sizeof(float)>x.length)return 0;id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];if(!e)return 0;[e setComputePipelineState:qgBias];[e setBuffer:x offset:0 atIndex:0];[e setBuffer:bb offset:0 atIndex:1];[e setBytes:&width length:4 atIndex:2];[e setBytes:&n length:4 atIndex:3];qg_dispatch(e,qgBias,n);[e endEncoding];mg_graph_note_encoder(g);return 1;}
int mg_qwen35_graph_swiglu(void*g,void*gp,void*up,int n){if(!qg_init()||!g||!gp||!up||n<=0)return 0;id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgSwiGLU];[e setBuffer:(__bridge id<MTLBuffer>)gp offset:0 atIndex:0];[e setBuffer:(__bridge id<MTLBuffer>)up offset:0 atIndex:1];[e setBytes:&n length:4 atIndex:2];qg_dispatch(e,qgSwiGLU,n);[e endEncoding];mg_graph_note_encoder(g);return 1;}
int mg_qwen35_graph_split(void*g,void*srcp,int qwidth,int hd,void**qout,void**gateout){int rows=qg_ordered_rows(g);if(!qg_init()||!g||!srcp||!rows||qwidth<=0||hd<=0||!qout||!gateout)return 0;id<MTLBuffer>q=(__bridge id<MTLBuffer>)mg_graph_alloc_result(g,rows*qwidth),gate=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,rows*qwidth);if(!q||!gate)return 0;id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgSplit];[e setBuffer:(__bridge id<MTLBuffer>)srcp offset:0 atIndex:0];[e setBuffer:q offset:0 atIndex:1];[e setBuffer:gate offset:0 atIndex:2];[e setBytes:&qwidth length:4 atIndex:3];[e setBytes:&hd length:4 atIndex:4];[e setBytes:&rows length:4 atIndex:5];qg_dispatch(e,qgSplit,rows*qwidth);[e endEncoding];*qout=(__bridge void*)q;*gateout=(__bridge void*)gate;return 1;}

int mg_qwen35_graph_attention(void*g,void*qp,void*kp,void*vp,void*gatep,const float*qw,const float*kw,const float*cosv,const float*sinv,const float*prefixK,const float*prefixV,int base,int nH,int nKV,int hd,int rotary,float scale,float qkEps,int gain1p,int qknorm,int qnw,int knw,void**outp,void**krawp,void**kpostp,void**vcurp){
    int rows=qg_ordered_rows(g);if(!qg_init()||!g||!rows||!qp||!kp||!vp||!qw||!kw||!cosv||!sinv||base<0||base>INT_MAX-rows||nH<1||nKV<1||nH%nKV||hd<2||hd>256||nH>INT_MAX/hd||nKV>INT_MAX/hd||rotary<2||rotary>hd||rotary%2||!isfinite(scale)||scale<=0||!isfinite(qkEps)||qkEps<=0||!outp||!krawp||!kpostp||!vcurp)return 0;int qwidth=nH*hd,kvwidth=nKV*hd,total=base+rows;if((qnw!=hd&&qnw!=qwidth)||(knw!=hd&&knw!=kvwidth)||rows>INT_MAX/qwidth||rows>INT_MAX/kvwidth||total>INT_MAX/kvwidth||(NSUInteger)total*(NSUInteger)kvwidth>NSUIntegerMax/sizeof(float))return 0;int qn=rows*qwidth,kn=rows*kvwidth;
    id<MTLBuffer>qo=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,qn),kr=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,kn),kpo=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,kn),out=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,qn);id<MTLBuffer>qa=qg_host(qw,qnw),ka=qg_host(kw,knw);if(!qo||!kr||!kpo||!out||!qa||!ka)return 0;
    // A nil gate is the ungated (dense Llama/Qwen2) epilogue: the kernels skip the sigmoid
    // and `out` is bound at the gate index only so every declared buffer slot is valid.
    int gated=gatep!=NULL;id<MTLBuffer>gateb=gated?(__bridge id<MTLBuffer>)gatep:out;
    id<MTLBuffer>cbv=qg_host(cosv,rows*(rotary/2)),sbv=qg_host(sinv,rows*(rotary/2));if(!cbv||!sbv)return 0;id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgQK];[e setBuffer:(__bridge id<MTLBuffer>)qp offset:0 atIndex:0];[e setBuffer:(__bridge id<MTLBuffer>)kp offset:0 atIndex:1];[e setBuffer:qa offset:0 atIndex:2];[e setBuffer:ka offset:0 atIndex:3];[e setBuffer:qo offset:0 atIndex:4];[e setBuffer:kr offset:0 atIndex:5];[e setBuffer:kpo offset:0 atIndex:6];[e setBytes:&nH length:4 atIndex:7];[e setBytes:&nKV length:4 atIndex:8];[e setBytes:&hd length:4 atIndex:9];[e setBytes:&rotary length:4 atIndex:10];[e setBytes:&base length:4 atIndex:11];[e setBuffer:cbv offset:0 atIndex:12];[e setBuffer:sbv offset:0 atIndex:13];[e setBytes:&qkEps length:4 atIndex:14];[e setBytes:&gain1p length:4 atIndex:15];[e setBytes:&qknorm length:4 atIndex:16];[e setBytes:&qnw length:4 atIndex:17];[e setBytes:&knw length:4 atIndex:18];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)MAX(nH,nKV),rows,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];mg_graph_note_encoder(g);
    int prefixElems=base*nKV*hd;id<MTLBuffer>kall=[gDev newBufferWithLength:(NSUInteger)total*nKV*hd*sizeof(float) options:MTLResourceStorageModeShared],vall=[gDev newBufferWithLength:(NSUInteger)total*nKV*hd*sizeof(float) options:MTLResourceStorageModeShared];if(!kall||!vall)return 0;if(prefixElems){if(!prefixK||!prefixV)return 0;memcpy(kall.contents,prefixK,(size_t)prefixElems*sizeof(float));memcpy(vall.contents,prefixV,(size_t)prefixElems*sizeof(float));}
    id<MTLBlitCommandEncoder>b=[cb blitCommandEncoder];[b copyFromBuffer:kpo sourceOffset:0 toBuffer:kall destinationOffset:(NSUInteger)prefixElems*sizeof(float) size:(NSUInteger)kn*sizeof(float)];[b copyFromBuffer:(__bridge id<MTLBuffer>)vp sourceOffset:0 toBuffer:vall destinationOffset:(NSUInteger)prefixElems*sizeof(float) size:(NSUInteger)kn*sizeof(float)];[b endEncoding];mg_graph_note_encoder(g);
    int useSplit=0,splits=1,chunk=2048;
    int splitThreads=32;id<MTLComputePipelineState>splitPSO=qg_split_pipeline(hd,&splitThreads);qg_split_policy_for(rows,total,splitPSO!=qgAttnSplit,&splits,&chunk,&useSplit);
    if(useSplit&&splitPSO&&qgAttnCombine){
        id<MTLBuffer>part=[gDev newBufferWithLength:(NSUInteger)nH*(NSUInteger)splits*(NSUInteger)(hd+2)*sizeof(float) options:MTLResourceStorageModePrivate];
        if(!part)return 0;
        e=[cb computeCommandEncoder];if(!e)return 0;[e setComputePipelineState:splitPSO];[e setBuffer:qo offset:0 atIndex:0];[e setBuffer:kall offset:0 atIndex:1];[e setBuffer:vall offset:0 atIndex:2];[e setBuffer:part offset:0 atIndex:3];[e setBytes:&total length:4 atIndex:4];[e setBytes:&base length:4 atIndex:5];[e setBytes:&nH length:4 atIndex:6];[e setBytes:&nKV length:4 atIndex:7];[e setBytes:&hd length:4 atIndex:8];[e setBytes:&scale length:4 atIndex:9];[e setBytes:&splits length:4 atIndex:10];[e setBytes:&chunk length:4 atIndex:11];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)nH,(NSUInteger)splits,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)splitThreads,1,1)];[e endEncoding];mg_graph_note_encoder(g);
        e=[cb computeCommandEncoder];if(!e)return 0;[e setComputePipelineState:qgAttnCombine];[e setBuffer:part offset:0 atIndex:0];[e setBuffer:gateb offset:0 atIndex:1];[e setBuffer:out offset:0 atIndex:2];[e setBytes:&nH length:4 atIndex:3];[e setBytes:&hd length:4 atIndex:4];[e setBytes:&splits length:4 atIndex:5];[e setBytes:&gated length:4 atIndex:6];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)nH,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];mg_graph_note_encoder(g);
    } else {
    int attnThreads=0;id<MTLComputePipelineState>attn=qg_attention_pipeline(total,hd,&attnThreads);e=[cb computeCommandEncoder];[e setComputePipelineState:attn];[e setBuffer:qo offset:0 atIndex:0];[e setBuffer:kall offset:0 atIndex:1];[e setBuffer:vall offset:0 atIndex:2];[e setBuffer:gateb offset:0 atIndex:3];[e setBuffer:out offset:0 atIndex:4];[e setBytes:&total length:4 atIndex:5];[e setBytes:&base length:4 atIndex:6];[e setBytes:&nH length:4 atIndex:7];[e setBytes:&nKV length:4 atIndex:8];[e setBytes:&hd length:4 atIndex:9];[e setBytes:&scale length:4 atIndex:10];[e setBytes:&gated length:4 atIndex:11];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)nH,rows,1) threadsPerThreadgroup:MTLSizeMake(attnThreads,1,1)];qg_note_attention_dispatch(attn);[e endEncoding];mg_graph_note_encoder(g);
    }
    *outp=(__bridge void*)out;*krawp=(__bridge void*)kr;*kpostp=(__bridge void*)kpo;*vcurp=vp;return 1;
}

// Device-resident KV prefix (#13087). The default attention entry above allocates a
// fresh `kall`/`vall` pair per panel, memcpy's the host prefix into it, and the caller
// then readbacks every layer's K/Kpost/V to append to the host cache — so panel p+1
// cannot encode until panel p's host readback completes and the host prefix is
// re-uploaded. The sequence-local device KV pair persists ACROSS panels instead: it is
// a plain `gDev` allocation owned by the caller (NOT tracked in `g->results`), so
// `mg_graph_free` never reclaims it. `mg_qwen35_graph_kv_alloc`/`_free` own that
// lifetime and `mg_qwen35_graph_attention_dkv` reads the SAME device K/V pair as the
// attention prefix and blits the panel's new K/V rows into it at `kvOff + prefixElems`
// — a device-side append that removes both the host prefix memcpy and the per-panel
// host KV readback.
//
// The pair is three buffers (KRaw, KPost and V) shared by every full-attention
// layer: the caller reserves `layers * tokens * kvWidth` floats per side and passes
// the layer's row offset `kvOff`. The layout matches the historical per-panel
// `kall`/`vall` exactly — `kall` held KPost rows and `vall` held raw V rows, both
// `total * nKV * hd` wide, while KRaw is the extra pre-norm key the host cache also
// keeps for state identity — so swapping the per-panel pair for one persistent pair
// is numerically transparent.
// mg_qwen35_graph_kv_alloc returns a caller-owned device buffer that survives
// `mg_graph_free`: it is NOT tracked in `g->results` and is retained (+1) with
// CFBridgingRetain so the ARC local does not free it when this function returns.
// mg_qwen35_graph_kv_free releases exactly that +1. This is the ownership that
// lets one KV triple live across every panel of a walk.
void *mg_qwen35_graph_kv_alloc(int elems){if(elems<=0||gDev==nil)return NULL;id<MTLBuffer>b=[gDev newBufferWithLength:(NSUInteger)elems*sizeof(float) options:MTLResourceStorageModeShared];return b?(__bridge_retained void*)b:NULL;}
void mg_qwen35_graph_kv_free(void*kv){if(kv){id<MTLBuffer>b=(__bridge_transfer id<MTLBuffer>)kv;b=nil;}}
int mg_qwen35_graph_kv_upload(void*kv,const float*src,int elems){
    if(!kv||!src||elems<=0)return 0;id<MTLBuffer>b=(__bridge id<MTLBuffer>)kv;
    if((NSUInteger)elems*sizeof(float)>b.length)return 0;memcpy([b contents],src,(NSUInteger)elems*sizeof(float));return 1;
}
int mg_qwen35_graph_kv_download(void*kv,float*dst,int elems){
    if(!kv||!dst||elems<=0)return 0;id<MTLBuffer>b=(__bridge id<MTLBuffer>)kv;
    if((NSUInteger)elems*sizeof(float)>b.length)return 0;memcpy(dst,[b contents],(NSUInteger)elems*sizeof(float));return 1;
}

// Offset-addressed staging for the persistent KV triple. The offset-0 entries above stage
// [0, off+n) through the host to reach row `off`, which makes seeding a layer slice
// O(offset); these copy exactly `elems` floats at float offset `off` (bounds-checked
// against the buffer length), so a per-layer seed or a one-row catch-up costs O(rows).
int mg_qwen35_graph_kv_upload_at(void*kv,long off,const float*src,long elems){
    if(!kv||!src||off<0||elems<=0)return 0;id<MTLBuffer>b=(__bridge id<MTLBuffer>)kv;
    if((NSUInteger)(off+elems)*sizeof(float)>b.length)return 0;memcpy((char*)[b contents]+(NSUInteger)off*sizeof(float),src,(NSUInteger)elems*sizeof(float));return 1;
}
int mg_qwen35_graph_kv_download_at(void*kv,long off,float*dst,long elems){
    if(!kv||!dst||off<0||elems<=0)return 0;id<MTLBuffer>b=(__bridge id<MTLBuffer>)kv;
    if((NSUInteger)(off+elems)*sizeof(float)>b.length)return 0;memcpy(dst,(const char*)[b contents]+(NSUInteger)off*sizeof(float),(NSUInteger)elems*sizeof(float));return 1;
}

// mg_qwen35_graph_attention_dkv is mg_qwen35_graph_attention with the KV prefix held
// on the device. `kvK`/`kvV` are the persistent K/V buffers and `kvOff` is this
// full-attention layer's row offset inside them (`layer * tokens * kvWidth`); the pair
// must already hold `base` prefix rows at that offset (the caller keeps it current
// across panels, so panel p reads exactly the rows panel p-1 appended). The function
// blits the panel's new K/V rows into the pair at `kvOff + prefixElems` in place; no
// host memcpy and no fresh per-panel allocation. `kraw`/`kpost`/`vcur` remain
// graph-temporary results the caller may read back once at the END of the walk.
int mg_qwen35_graph_attention_dkv(void*g,void*qp,void*kp,void*vp,void*gatep,const float*qw,const float*kw,const float*cosv,const float*sinv,void*kvKraw,void*kvKpost,void*kvV,int kvOff,int base,int nH,int nKV,int hd,int rotary,float scale,float qkEps,int gain1p,int qknorm,int qnw,int knw,void**outp,void**krawp,void**kpostp,void**vcurp){
    int rows=qg_ordered_rows(g);if(!qg_init()||!g||!rows||!qp||!kp||!vp||!qw||!kw||!cosv||!sinv||!kvKpost||!kvV||kvOff<0||base<0||base>INT_MAX-rows||nH<1||nKV<1||nH%nKV||hd<2||hd>256||nH>INT_MAX/hd||nKV>INT_MAX/hd||rotary<2||rotary>hd||rotary%2||!isfinite(scale)||scale<=0||!isfinite(qkEps)||qkEps<=0||!outp||!krawp||!kpostp||!vcurp)return 0;int qwidth=nH*hd,kvwidth=nKV*hd,total=base+rows;if((qnw!=hd&&qnw!=qwidth)||(knw!=hd&&knw!=kvwidth)||rows>INT_MAX/qwidth||rows>INT_MAX/kvwidth||total>INT_MAX/kvwidth||(NSUInteger)total*(NSUInteger)kvwidth>NSUIntegerMax/sizeof(float))return 0;int qn=rows*qwidth,kn=rows*kvwidth,prefixElems=base*kvwidth;
    // A NULL kvKraw is an attend-only pair (NewDeviceKVAttendOnly): the attention reads only
    // KPost/V, so the KRaw append is skipped and `kraw` stays a graph result for readback.
    id<MTLBuffer>kall=(__bridge id<MTLBuffer>)kvKpost,vall=(__bridge id<MTLBuffer>)kvV,krawall=kvKraw?(__bridge id<MTLBuffer>)kvKraw:nil;
    // The layer's device slices must hold through `total` rows (prefix + this panel).
    if((NSUInteger)(kvOff+prefixElems+kn)*sizeof(float)>kall.length||(NSUInteger)(kvOff+prefixElems+kn)*sizeof(float)>vall.length||(krawall&&(NSUInteger)(kvOff+prefixElems+kn)*sizeof(float)>krawall.length))return 0;
    id<MTLBuffer>qo=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,qn),kr=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,kn),kpo=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,kn),out=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,qn);id<MTLBuffer>qa=qg_host(qw,qnw),ka=qg_host(kw,knw);if(!qo||!kr||!kpo||!out||!qa||!ka)return 0;
    // A nil gate is the ungated (dense Llama/Qwen2) epilogue: the kernels skip the sigmoid
    // and `out` is bound at the gate index only so every declared buffer slot is valid.
    int gated=gatep!=NULL;id<MTLBuffer>gateb=gated?(__bridge id<MTLBuffer>)gatep:out;
    id<MTLBuffer>cbv=qg_host(cosv,rows*(rotary/2)),sbv=qg_host(sinv,rows*(rotary/2));if(!cbv||!sbv)return 0;id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgQK];[e setBuffer:(__bridge id<MTLBuffer>)qp offset:0 atIndex:0];[e setBuffer:(__bridge id<MTLBuffer>)kp offset:0 atIndex:1];[e setBuffer:qa offset:0 atIndex:2];[e setBuffer:ka offset:0 atIndex:3];[e setBuffer:qo offset:0 atIndex:4];[e setBuffer:kr offset:0 atIndex:5];[e setBuffer:kpo offset:0 atIndex:6];[e setBytes:&nH length:4 atIndex:7];[e setBytes:&nKV length:4 atIndex:8];[e setBytes:&hd length:4 atIndex:9];[e setBytes:&rotary length:4 atIndex:10];[e setBytes:&base length:4 atIndex:11];[e setBuffer:cbv offset:0 atIndex:12];[e setBuffer:sbv offset:0 atIndex:13];[e setBytes:&qkEps length:4 atIndex:14];[e setBytes:&gain1p length:4 atIndex:15];[e setBytes:&qknorm length:4 atIndex:16];[e setBytes:&qnw length:4 atIndex:17];[e setBytes:&knw length:4 atIndex:18];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)MAX(nH,nKV),rows,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];mg_graph_note_encoder(g);
    // Append this panel's three host-visible rows into the persistent device pair:
    // pre-norm KRaw (`kr`), post-norm KPost (`kpo`) and raw V (`vp`) — exactly the
    // three rows the host path appended at the per-panel readback, so KRaw/K/Kpost/V
    // parity holds byte-for-byte. Device blits into kvOff+prefixElems replace the old
    // host memcpy of the prefix, so panel p+1 reads the device rows directly.
    id<MTLBlitCommandEncoder>b=[cb blitCommandEncoder];if(krawall)[b copyFromBuffer:kr sourceOffset:0 toBuffer:krawall destinationOffset:(NSUInteger)(kvOff+prefixElems)*sizeof(float) size:(NSUInteger)kn*sizeof(float)];[b copyFromBuffer:kpo sourceOffset:0 toBuffer:kall destinationOffset:(NSUInteger)(kvOff+prefixElems)*sizeof(float) size:(NSUInteger)kn*sizeof(float)];[b copyFromBuffer:(__bridge id<MTLBuffer>)vp sourceOffset:0 toBuffer:vall destinationOffset:(NSUInteger)(kvOff+prefixElems)*sizeof(float) size:(NSUInteger)kn*sizeof(float)];[b endEncoding];mg_graph_note_encoder(g);
    int useSplit=0,splits=1,chunk=2048;
    int splitThreads=32;id<MTLComputePipelineState>splitPSO=qg_split_pipeline(hd,&splitThreads);qg_split_policy_for(rows,total,splitPSO!=qgAttnSplit,&splits,&chunk,&useSplit);
    if(useSplit&&splitPSO&&qgAttnCombine){
        id<MTLBuffer>part=[gDev newBufferWithLength:(NSUInteger)nH*(NSUInteger)splits*(NSUInteger)(hd+2)*sizeof(float) options:MTLResourceStorageModePrivate];
        if(!part)return 0;
        e=[cb computeCommandEncoder];if(!e)return 0;[e setComputePipelineState:splitPSO];[e setBuffer:qo offset:0 atIndex:0];[e setBuffer:kall offset:(NSUInteger)kvOff*sizeof(float) atIndex:1];[e setBuffer:vall offset:(NSUInteger)kvOff*sizeof(float) atIndex:2];[e setBuffer:part offset:0 atIndex:3];[e setBytes:&total length:4 atIndex:4];[e setBytes:&base length:4 atIndex:5];[e setBytes:&nH length:4 atIndex:6];[e setBytes:&nKV length:4 atIndex:7];[e setBytes:&hd length:4 atIndex:8];[e setBytes:&scale length:4 atIndex:9];[e setBytes:&splits length:4 atIndex:10];[e setBytes:&chunk length:4 atIndex:11];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)nH,(NSUInteger)splits,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)splitThreads,1,1)];[e endEncoding];mg_graph_note_encoder(g);
        e=[cb computeCommandEncoder];if(!e)return 0;[e setComputePipelineState:qgAttnCombine];[e setBuffer:part offset:0 atIndex:0];[e setBuffer:gateb offset:0 atIndex:1];[e setBuffer:out offset:0 atIndex:2];[e setBytes:&nH length:4 atIndex:3];[e setBytes:&hd length:4 atIndex:4];[e setBytes:&splits length:4 atIndex:5];[e setBytes:&gated length:4 atIndex:6];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)nH,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];mg_graph_note_encoder(g);
    } else {
    int attnThreads=0;id<MTLComputePipelineState>attn=qg_attention_pipeline(total,hd,&attnThreads);e=[cb computeCommandEncoder];[e setComputePipelineState:attn];[e setBuffer:qo offset:0 atIndex:0];[e setBuffer:kall offset:(NSUInteger)kvOff*sizeof(float) atIndex:1];[e setBuffer:vall offset:(NSUInteger)kvOff*sizeof(float) atIndex:2];[e setBuffer:gateb offset:0 atIndex:3];[e setBuffer:out offset:0 atIndex:4];[e setBytes:&total length:4 atIndex:5];[e setBytes:&base length:4 atIndex:6];[e setBytes:&nH length:4 atIndex:7];[e setBytes:&nKV length:4 atIndex:8];[e setBytes:&hd length:4 atIndex:9];[e setBytes:&scale length:4 atIndex:10];[e setBytes:&gated length:4 atIndex:11];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)nH,rows,1) threadsPerThreadgroup:MTLSizeMake(attnThreads,1,1)];qg_note_attention_dispatch(attn);[e endEncoding];mg_graph_note_encoder(g);
    }
    *outp=(__bridge void*)out;*krawp=(__bridge void*)kr;*kpostp=(__bridge void*)kpo;*vcurp=vp;return 1;
}



// Packed Q8_0 KV consumer (#12981). The F32 entries above hold every attended K/V
// row at 4 bytes/element; at a 20480-token context the 16 full-attention layers of
// the 27B Qwen3.8 carry 3.75 GiB of token KV that way. These entries keep the
// attended rows (post-RoPE K and raw V) in the host cache's realized Q8_0 layout
// instead (internal/model/kvcache_q8.go kvPackedRow): one int8 code per element and
// one f32 scale per 32-element block, 1.125 bytes/element. The attention kernels
// dequantize on read, so no F32 K/V mirror is ever materialized on the device, and
// the panel's own rows are quantized on the device BEFORE it attends, exactly as the
// host q8 path appends a row and then attends over it (KVCache.appendKV +
// decodeRowInto). KRaw stays f32 and host-owned: it is returned per panel, never
// packed, so Evict's single-rotation re-positioning stays exact.
//
// The pipelines live in their own library, compiled lazily on the first Q8 use, so
// a Q8 compile or pipeline failure declines only the Q8 entries and can never turn
// the proven F32 graph into a nil-PSO decline (qg_init stays transactional).
//
// qg8_quant reproduces model.QuantizeKVQ8_0: scale = maxAbs/127 as a correctly
// rounded f32 division, code = round-half-away-from-zero(x/scale) clamped to
// [-128,127]. The quotient is estimated with the fast-math divide and then corrected
// by two fma residual tests whose signs are exact, so the code is the nearest
// integer to the EXACT x/scale. The host codec's float64 product agrees with that
// everywhere except an exact .5 tie (x == (k+0.5)*scale), a measure-zero input at
// which the host's rounded float64 reciprocal may break the tie either way. The fma
// residual tests assume normal floats: a block whose maxAbs is below ~2.5e-29 can
// see its residual flushed to zero (Metal FTZ) and differ from the host by one code.
static id<MTLComputePipelineState> qg8Quant,qg8Attn,qg8AttnSplit;
static BOOL qg8Attempted,qg8Ready;

static NSString *qg8Source=@R"MSL(
#include <metal_stdlib>
using namespace metal;

// One SIMD-group per 32-element block: simd_max is the block's maxAbs.
kernel void qg8_quant(device const float*src [[buffer(0)]],device char*codes [[buffer(1)]],device float*scales [[buffer(2)]],constant int&blocks [[buffer(3)]],
                      uint blk [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    if((int)blk>=blocks)return;
    long i=(long)blk*32+lane;float x=src[i],amax=simd_max(fabs(x)),s=0.0f;int c=0;
    if(amax>0.0f){
        s=precise::divide(amax,127.0f);
        c=(int)rint(x/s);
        float hi=fma(-((float)c+0.5f),s,x);
        if(hi>0.0f||(hi==0.0f&&(float)c+0.5f>0.0f))c++;
        else{float lo=fma(-((float)c-0.5f),s,x);if(lo<0.0f||(lo==0.0f&&(float)c-0.5f<0.0f))c--;}
        c=clamp(c,-128,127);
    }
    codes[i]=(char)c;if(lane==0)scales[blk]=s;
}

// qg8_attn is qg_attn_online over packed rows. Row j of KV head kh keeps its codes
// at (j*nKV+kh)*hd and its scales at (j*nKV+kh)*(hd/32); lane element d=lane+32*i
// always lies in block i of that head, so each block scale is one uniform load.
kernel void qg8_attn(device const float*q [[buffer(0)]],device const char*kc [[buffer(1)]],device const float*ks [[buffer(2)]],
                     device const char*vc [[buffer(3)]],device const float*vs [[buffer(4)]],device const float*gate [[buffer(5)]],device float*out [[buffer(6)]],
                     constant int&total [[buffer(7)]],constant int&base [[buffer(8)]],constant int&nH [[buffer(9)]],constant int&nKV [[buffer(10)]],constant int&hd [[buffer(11)]],constant float&scale [[buffer(12)]],
                     uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    int h=(int)group.x,t=(int)group.y,kh=h/(nH/nKV),upto=min(total,base+t+1),nb=hd/32;float qr[8],acc[8];
    for(int i=0;i<8;i++){qr[i]=i<nb?q[((long)t*nH+h)*hd+(int)lane+i*32]:0.0f;acc[i]=0.0f;}
    float m=-INFINITY,den=0.0f;
    for(int j=0;j<upto;j++){
        long row=(long)j*nKV+kh,cb=row*hd,sb=row*nb;
        float partial=0.0f;for(int i=0;i<8;i++){if(i<nb)partial+=qr[i]*((float)kc[cb+(int)lane+i*32]*ks[sb+i]);}
        float score=simd_sum(partial)*scale,newm=max(m,score),alpha=isfinite(m)?exp(m-newm):0.0f,p=exp(score-newm);
        den=den*alpha+p;m=newm;
        for(int i=0;i<8;i++){if(i<nb)acc[i]=acc[i]*alpha+p*((float)vc[cb+(int)lane+i*32]*vs[sb+i]);}
    }
    for(int i=0;i<8;i++){if(i<nb){long ix=((long)t*nH+h)*hd+(int)lane+i*32;out[ix]=(acc[i]/den)/(1.0f+exp(-gate[ix]));}}
}

// qg8_attn_split is qg_attn_split over packed rows; its partials use the same
// part[(h*splits+s)*(hd+2)] layout, so qg_attn_combine merges them unchanged.
kernel void qg8_attn_split(device const float*q [[buffer(0)]],device const char*kc [[buffer(1)]],device const float*ks [[buffer(2)]],
                           device const char*vc [[buffer(3)]],device const float*vs [[buffer(4)]],device float*part [[buffer(5)]],
                           constant int&total [[buffer(6)]],constant int&base [[buffer(7)]],constant int&nH [[buffer(8)]],constant int&nKV [[buffer(9)]],constant int&hd [[buffer(10)]],constant float&scale [[buffer(11)]],
                           constant int&splits [[buffer(12)]],constant int&chunk [[buffer(13)]],
                           uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    int h=(int)group.x,s=(int)group.y,kh=h/(nH/nKV),upto=min(total,base+1),nb=hd/32,lo=s*chunk,hi=min(lo+chunk,upto);float qr[8],acc[8];
    for(int i=0;i<8;i++){qr[i]=i<nb?q[(long)h*hd+(int)lane+i*32]:0.0f;acc[i]=0.0f;}
    float m=-INFINITY,den=0.0f;
    for(int j=lo;j<hi;j++){
        long row=(long)j*nKV+kh,cb=row*hd,sb=row*nb;
        float partial=0.0f;for(int i=0;i<8;i++){if(i<nb)partial+=qr[i]*((float)kc[cb+(int)lane+i*32]*ks[sb+i]);}
        float score=simd_sum(partial)*scale,newm=max(m,score),alpha=isfinite(m)?exp(m-newm):0.0f,p=exp(score-newm);
        den=den*alpha+p;m=newm;
        for(int i=0;i<8;i++){if(i<nb)acc[i]=acc[i]*alpha+p*((float)vc[cb+(int)lane+i*32]*vs[sb+i]);}
    }
    long po=((long)h*splits+s)*(hd+2);
    for(int i=0;i<8;i++){if(i<nb)part[po+(int)lane+i*32]=acc[i];}
    if(lane==0){part[po+hd]=m;part[po+hd+1]=den;}
}
)MSL";

static int qg8_init(void){
    if(!qg_init())return 0;
    @synchronized(gDev){
        if(qg8Ready)return 1;
        if(qg8Attempted)return 0;
        qg8Attempted=YES;
        NSError *error=nil;
        id<MTLLibrary> library=[gDev newLibraryWithSource:qg8Source options:nil error:&error];
        if(!library){NSLog(@"qwen35 graph q8 kv compile: %@",error);return 0;}
        id<MTLComputePipelineState> quant=qg_pipeline(library,@"qg8_quant",&error);
        id<MTLComputePipelineState> attn=qg_pipeline(library,@"qg8_attn",&error);
        id<MTLComputePipelineState> split=qg_pipeline(library,@"qg8_attn_split",&error);
        if(!quant||!attn||!split){NSLog(@"qwen35 graph q8 kv pipeline initialization failed: %@",error);return 0;}
        qg8Quant=quant;qg8Attn=attn;qg8AttnSplit=split;
        qg8Ready=YES;
        return 1;
    }
}
int mg_qwen35_graph_kvq8_ready(void){return qg8_init();}

// Caller-owned packed store sides (#12981): byte-addressed twins of
// mg_qwen35_graph_kv_alloc/_upload/_download. Released by mg_qwen35_graph_kv_free.
void *mg_qwen35_graph_kvq8_alloc(long bytes){if(bytes<=0||gDev==nil)return NULL;id<MTLBuffer>b=[gDev newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];return b?(__bridge_retained void*)b:NULL;}
int mg_qwen35_graph_kvq8_write(void*buf,long off,const void*src,long bytes){
    if(!buf||!src||off<0||bytes<=0)return 0;id<MTLBuffer>b=(__bridge id<MTLBuffer>)buf;
    if((NSUInteger)off>b.length||(NSUInteger)bytes>b.length-(NSUInteger)off)return 0;memcpy((char*)[b contents]+off,src,(size_t)bytes);return 1;
}
int mg_qwen35_graph_kvq8_read(void*buf,long off,void*dst,long bytes){
    if(!buf||!dst||off<0||bytes<=0)return 0;id<MTLBuffer>b=(__bridge id<MTLBuffer>)buf;
    if((NSUInteger)off>b.length||(NSUInteger)bytes>b.length-(NSUInteger)off)return 0;memcpy(dst,(const char*)[b contents]+off,(size_t)bytes);return 1;
}

// qg8_rows validates the shared Q8 attention geometry and returns the panel row
// count, or 0 to decline. hd must be a multiple of the 32-element block so a block
// never straddles two heads (the host layout's own invariant).
static int qg8_rows(void*g,int base,int nH,int nKV,int hd,int rotary,float scale,float qkEps,int qnw,int knw){
    if(!g||!qg8_init())return 0;
    int rows=qg_ordered_rows(g);
    if(!rows||base<0||base>INT_MAX-rows||nH<1||nKV<1||nH%nKV||hd<32||hd>256||hd%32||nH>INT_MAX/hd||nKV>INT_MAX/hd||rotary<2||rotary>hd||rotary%2||!isfinite(scale)||scale<=0||!isfinite(qkEps)||qkEps<=0)return 0;
    int qwidth=nH*hd,kvwidth=nKV*hd,total=base+rows;
    if((qnw!=hd&&qnw!=qwidth)||(knw!=hd&&knw!=kvwidth)||rows>INT_MAX/qwidth||rows>INT_MAX/kvwidth||total>INT_MAX/kvwidth)return 0;
    return rows;
}

// qg8_encode_attention encodes Q/K norm+RoPE (the F32 qg_qk kernel, unchanged),
// quantizes the panel's KPost/V rows into graph-owned packed results, appends them
// at row `base` of the layer slice that starts `elemOff` elements into the packed
// store, then attends over the packed prefix+panel.
static int qg8_encode_attention(void*g,int rows,void*qp,void*kp,void*vp,void*gatep,const float*qw,const float*kw,const float*cosv,const float*sinv,
    id<MTLBuffer>kc,id<MTLBuffer>ks,id<MTLBuffer>vc,id<MTLBuffer>vs,NSUInteger elemOff,
    int base,int nH,int nKV,int hd,int rotary,float scale,float qkEps,int gain1p,int qknorm,int qnw,int knw,
    void**outp,void**krawp,void**kpostp,void**vcurp,void**kcp,void**ksp,void**vcp,void**vsp){
    int qwidth=nH*hd,kvwidth=nKV*hd,total=base+rows,qn=rows*qwidth,kn=rows*kvwidth,blocks=kn/32;
    id<MTLBuffer>qo=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,qn),kr=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,kn),kpo=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,kn),out=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,qn);
    id<MTLBuffer>curKc=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,kn/4),curKs=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,blocks),curVc=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,kn/4),curVs=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,blocks);
    id<MTLBuffer>qa=qg_host(qw,qnw),ka=qg_host(kw,knw),cbv=qg_host(cosv,rows*(rotary/2)),sbv=qg_host(sinv,rows*(rotary/2));
    id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);
    if(!qo||!kr||!kpo||!out||!curKc||!curKs||!curVc||!curVs||!qa||!ka||!cbv||!sbv||!cb)return 0;
    id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];if(!e)return 0;[e setComputePipelineState:qgQK];[e setBuffer:(__bridge id<MTLBuffer>)qp offset:0 atIndex:0];[e setBuffer:(__bridge id<MTLBuffer>)kp offset:0 atIndex:1];[e setBuffer:qa offset:0 atIndex:2];[e setBuffer:ka offset:0 atIndex:3];[e setBuffer:qo offset:0 atIndex:4];[e setBuffer:kr offset:0 atIndex:5];[e setBuffer:kpo offset:0 atIndex:6];[e setBytes:&nH length:4 atIndex:7];[e setBytes:&nKV length:4 atIndex:8];[e setBytes:&hd length:4 atIndex:9];[e setBytes:&rotary length:4 atIndex:10];[e setBytes:&base length:4 atIndex:11];[e setBuffer:cbv offset:0 atIndex:12];[e setBuffer:sbv offset:0 atIndex:13];[e setBytes:&qkEps length:4 atIndex:14];[e setBytes:&gain1p length:4 atIndex:15];[e setBytes:&qknorm length:4 atIndex:16];[e setBytes:&qnw length:4 atIndex:17];[e setBytes:&knw length:4 atIndex:18];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)MAX(nH,nKV),rows,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];mg_graph_note_encoder(g);
    // Quantize this panel's post-RoPE K and raw V rows (the host appendKV pair).
    e=[cb computeCommandEncoder];if(!e)return 0;[e setComputePipelineState:qg8Quant];[e setBytes:&blocks length:4 atIndex:3];
    [e setBuffer:kpo offset:0 atIndex:0];[e setBuffer:curKc offset:0 atIndex:1];[e setBuffer:curKs offset:0 atIndex:2];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)blocks,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e setBuffer:(__bridge id<MTLBuffer>)vp offset:0 atIndex:0];[e setBuffer:curVc offset:0 atIndex:1];[e setBuffer:curVs offset:0 atIndex:2];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)blocks,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e endEncoding];mg_graph_note_encoder(g);
    // Device append at row `base` of the layer slice: codes are byte-addressed,
    // scales f32-addressed at one per 32 elements.
    NSUInteger codeDst=elemOff+(NSUInteger)base*(NSUInteger)kvwidth,scaleDst=codeDst/32*sizeof(float),scaleBytes=(NSUInteger)blocks*sizeof(float);
    id<MTLBlitCommandEncoder>b=[cb blitCommandEncoder];if(!b)return 0;
    [b copyFromBuffer:curKc sourceOffset:0 toBuffer:kc destinationOffset:codeDst size:(NSUInteger)kn];[b copyFromBuffer:curKs sourceOffset:0 toBuffer:ks destinationOffset:scaleDst size:scaleBytes];
    [b copyFromBuffer:curVc sourceOffset:0 toBuffer:vc destinationOffset:codeDst size:(NSUInteger)kn];[b copyFromBuffer:curVs sourceOffset:0 toBuffer:vs destinationOffset:scaleDst size:scaleBytes];
    [b endEncoding];mg_graph_note_encoder(g);
    NSUInteger codeOff=elemOff,scaleOff=elemOff/32*sizeof(float);
    int useSplit=0,splits=1,chunk=2048;
    qg_split_policy(rows,total,&splits,&chunk,&useSplit);
    if(useSplit){
        id<MTLBuffer>part=[gDev newBufferWithLength:(NSUInteger)nH*(NSUInteger)splits*(NSUInteger)(hd+2)*sizeof(float) options:MTLResourceStorageModePrivate];
        if(!part)return 0;
        e=[cb computeCommandEncoder];if(!e)return 0;[e setComputePipelineState:qg8AttnSplit];[e setBuffer:qo offset:0 atIndex:0];[e setBuffer:kc offset:codeOff atIndex:1];[e setBuffer:ks offset:scaleOff atIndex:2];[e setBuffer:vc offset:codeOff atIndex:3];[e setBuffer:vs offset:scaleOff atIndex:4];[e setBuffer:part offset:0 atIndex:5];[e setBytes:&total length:4 atIndex:6];[e setBytes:&base length:4 atIndex:7];[e setBytes:&nH length:4 atIndex:8];[e setBytes:&nKV length:4 atIndex:9];[e setBytes:&hd length:4 atIndex:10];[e setBytes:&scale length:4 atIndex:11];[e setBytes:&splits length:4 atIndex:12];[e setBytes:&chunk length:4 atIndex:13];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)nH,(NSUInteger)splits,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];mg_graph_note_encoder(g);
        e=[cb computeCommandEncoder];if(!e)return 0;[e setComputePipelineState:qgAttnCombine];[e setBuffer:part offset:0 atIndex:0];[e setBuffer:(__bridge id<MTLBuffer>)gatep offset:0 atIndex:1];[e setBuffer:out offset:0 atIndex:2];[e setBytes:&nH length:4 atIndex:3];[e setBytes:&hd length:4 atIndex:4];[e setBytes:&splits length:4 atIndex:5];{int gated=1;[e setBytes:&gated length:4 atIndex:6];}[e dispatchThreadgroups:MTLSizeMake((NSUInteger)nH,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];mg_graph_note_encoder(g);
    } else {
        e=[cb computeCommandEncoder];if(!e)return 0;[e setComputePipelineState:qg8Attn];[e setBuffer:qo offset:0 atIndex:0];[e setBuffer:kc offset:codeOff atIndex:1];[e setBuffer:ks offset:scaleOff atIndex:2];[e setBuffer:vc offset:codeOff atIndex:3];[e setBuffer:vs offset:scaleOff atIndex:4];[e setBuffer:(__bridge id<MTLBuffer>)gatep offset:0 atIndex:5];[e setBuffer:out offset:0 atIndex:6];[e setBytes:&total length:4 atIndex:7];[e setBytes:&base length:4 atIndex:8];[e setBytes:&nH length:4 atIndex:9];[e setBytes:&nKV length:4 atIndex:10];[e setBytes:&hd length:4 atIndex:11];[e setBytes:&scale length:4 atIndex:12];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)nH,(NSUInteger)rows,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];mg_graph_note_encoder(g);
    }
    *outp=(__bridge void*)out;*krawp=(__bridge void*)kr;*kpostp=(__bridge void*)kpo;*vcurp=vp;
    *kcp=(__bridge void*)curKc;*ksp=(__bridge void*)curKs;*vcp=(__bridge void*)curVc;*vsp=(__bridge void*)curVs;
    return 1;
}

// mg_qwen35_graph_attention_q8 is mg_qwen35_graph_attention with a host-owned PACKED
// prefix: `base` rows of int8 codes plus their block scales for K and V, copied
// into a per-panel packed pair (1.125 B/element instead of the F32 entry's 4).
int mg_qwen35_graph_attention_q8(void*g,void*qp,void*kp,void*vp,void*gatep,const float*qw,const float*kw,const float*cosv,const float*sinv,
    const signed char*prefixKc,const float*prefixKs,const signed char*prefixVc,const float*prefixVs,
    int base,int nH,int nKV,int hd,int rotary,float scale,float qkEps,int gain1p,int qknorm,int qnw,int knw,
    void**outp,void**krawp,void**kpostp,void**vcurp,void**kcp,void**ksp,void**vcp,void**vsp){
    int rows=qg8_rows(g,base,nH,nKV,hd,rotary,scale,qkEps,qnw,knw);
    if(!rows||!qp||!kp||!vp||!gatep||!qw||!kw||!cosv||!sinv||!outp||!krawp||!kpostp||!vcurp||!kcp||!ksp||!vcp||!vsp)return 0;
    NSUInteger kvwidth=(NSUInteger)nKV*(NSUInteger)hd,codeBytes=(NSUInteger)(base+rows)*kvwidth,prefixCodes=(NSUInteger)base*kvwidth;
    if(prefixCodes&&(!prefixKc||!prefixKs||!prefixVc||!prefixVs))return 0;
    id<MTLBuffer>kc=[gDev newBufferWithLength:codeBytes options:MTLResourceStorageModeShared],ks=[gDev newBufferWithLength:codeBytes/32*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer>vc=[gDev newBufferWithLength:codeBytes options:MTLResourceStorageModeShared],vs=[gDev newBufferWithLength:codeBytes/32*sizeof(float) options:MTLResourceStorageModeShared];
    if(!kc||!ks||!vc||!vs)return 0;
    if(prefixCodes){memcpy(kc.contents,prefixKc,prefixCodes);memcpy(ks.contents,prefixKs,prefixCodes/32*sizeof(float));memcpy(vc.contents,prefixVc,prefixCodes);memcpy(vs.contents,prefixVs,prefixCodes/32*sizeof(float));}
    return qg8_encode_attention(g,rows,qp,kp,vp,gatep,qw,kw,cosv,sinv,kc,ks,vc,vs,0,base,nH,nKV,hd,rotary,scale,qkEps,gain1p,qknorm,qnw,knw,outp,krawp,kpostp,vcurp,kcp,ksp,vcp,vsp);
}

// mg_qwen35_graph_attention_dkv_q8 is mg_qwen35_graph_attention_dkv over a
// caller-owned PACKED store: the layer slice starts `elemOff` elements in (codes are
// bytes, scales one f32 per 32 elements) and must already hold `base` rows. The
// panel's packed rows are appended there on the device, so the store persists across
// panels and decode steps with no host prefix copy and no F32 mirror.
int mg_qwen35_graph_attention_dkv_q8(void*g,void*qp,void*kp,void*vp,void*gatep,const float*qw,const float*kw,const float*cosv,const float*sinv,
    void*kvKc,void*kvKs,void*kvVc,void*kvVs,long elemOff,
    int base,int nH,int nKV,int hd,int rotary,float scale,float qkEps,int gain1p,int qknorm,int qnw,int knw,
    void**outp,void**krawp,void**kpostp,void**vcurp,void**kcp,void**ksp,void**vcp,void**vsp){
    int rows=qg8_rows(g,base,nH,nKV,hd,rotary,scale,qkEps,qnw,knw);
    if(!rows||!qp||!kp||!vp||!gatep||!qw||!kw||!cosv||!sinv||!kvKc||!kvKs||!kvVc||!kvVs||elemOff<0||elemOff%32||!outp||!krawp||!kpostp||!vcurp||!kcp||!ksp||!vcp||!vsp)return 0;
    id<MTLBuffer>kc=(__bridge id<MTLBuffer>)kvKc,ks=(__bridge id<MTLBuffer>)kvKs,vc=(__bridge id<MTLBuffer>)kvVc,vs=(__bridge id<MTLBuffer>)kvVs;
    NSUInteger end=(NSUInteger)elemOff+(NSUInteger)(base+rows)*(NSUInteger)nKV*(NSUInteger)hd;
    if(end>kc.length||end>vc.length||end/32*sizeof(float)>ks.length||end/32*sizeof(float)>vs.length)return 0;
    return qg8_encode_attention(g,rows,qp,kp,vp,gatep,qw,kw,cosv,sinv,kc,ks,vc,vs,(NSUInteger)elemOff,base,nH,nKV,hd,rotary,scale,qkEps,gain1p,qknorm,qnw,knw,outp,krawp,kpostp,vcurp,kcp,ksp,vcp,vsp);
}

int mg_qwen35_graph_attention_batch(void*g,void*qgatep,void*kp,void*vp,const float*qw,const float*kw,const float*cosv,const float*sinv,const int*positions,const int*offsets,const int*lengths,const float*prefixK,const float*prefixV,int totalKV,int batch,int modelWidth,int attentionWidth,int kvWidth,int nH,int nKV,int hd,int rotary,float scale,float qkEps,int gain1p,int qknorm,int qnw,int knw,void**outp,void**krawp,void**kpostp,void**vcurp){
    if(!qg_init()||!g||!qgatep||!kp||!vp||!qw||!kw||!cosv||!sinv||!positions||!offsets||!lengths||!prefixK||!prefixV||!outp||!krawp||!kpostp||!vcurp||batch<2||batch>24||modelWidth<=0||attentionWidth<=0||kvWidth<=0||nH<1||nKV<1||nH%nKV||hd<2||hd>256||nH>INT_MAX/hd||nKV>INT_MAX/hd||attentionWidth!=nH*hd||kvWidth!=nKV*hd||modelWidth>INT_MAX/batch||attentionWidth>INT_MAX/(2*batch)||kvWidth>INT_MAX/batch||totalKV<batch||totalKV>INT_MAX/kvWidth||rotary<2||rotary>hd||rotary%2||!isfinite(scale)||scale<=0||!isfinite(qkEps)||qkEps<=0||(qnw!=hd&&qnw!=attentionWidth)||(knw!=hd&&knw!=kvWidth)||mg_graph_prompt(g)!=batch||mg_graph_input(g)!=modelWidth)return 0;
    int maxPos=0,expectedOffset=0;for(int r=0;r<batch;r++){if(positions[r]<0||lengths[r]!=positions[r]+1||offsets[r]!=expectedOffset)return 0;expectedOffset+=lengths[r];maxPos=MAX(maxPos,positions[r]);}if(expectedOffset!=totalKV)return 0;
    id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);
    id<MTLBuffer>q=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,batch*attentionWidth),gate=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,batch*attentionWidth),qout=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,batch*attentionWidth),kr=(__bridge id<MTLBuffer>)mg_graph_alloc_result(g,batch*kvWidth),kpst=(__bridge id<MTLBuffer>)mg_graph_alloc_result(g,batch*kvWidth),vc=(__bridge id<MTLBuffer>)mg_graph_alloc_result(g,batch*kvWidth),out=(__bridge id<MTLBuffer>)mg_graph_alloc_result(g,batch*attentionWidth);
    id<MTLBuffer>qwb=qg_host(qw,qnw==hd?hd:attentionWidth),kwb=qg_host(kw,knw==hd?hd:kvWidth),cbf=qg_host(cosv,(maxPos+1)*(rotary/2)),sbf=qg_host(sinv,(maxPos+1)*(rotary/2)),posb=[gDev newBufferWithBytes:positions length:(NSUInteger)batch*sizeof(int) options:MTLResourceStorageModeShared],offb=[gDev newBufferWithBytes:offsets length:(NSUInteger)batch*sizeof(int) options:MTLResourceStorageModeShared],lenb=[gDev newBufferWithBytes:lengths length:(NSUInteger)batch*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer>allK=qg_host(prefixK,totalKV*kvWidth),allV=qg_host(prefixV,totalKV*kvWidth);if(!cb||!q||!gate||!qout||!kr||!kpst||!vc||!out||!qwb||!kwb||!cbf||!sbf||!posb||!offb||!lenb||!allK||!allV)return 0;
    id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgLaneSplit];[e setBuffer:(__bridge id<MTLBuffer>)qgatep offset:0 atIndex:0];[e setBuffer:q offset:0 atIndex:1];[e setBuffer:gate offset:0 atIndex:2];[e setBytes:&batch length:4 atIndex:3];[e setBytes:&attentionWidth length:4 atIndex:4];[e setBytes:&hd length:4 atIndex:5];qg_dispatch(e,qgLaneSplit,batch*attentionWidth);[e endEncoding];
    e=[cb computeCommandEncoder];[e setComputePipelineState:qgLaneQK];[e setBuffer:q offset:0 atIndex:0];[e setBuffer:(__bridge id<MTLBuffer>)kp offset:0 atIndex:1];[e setBuffer:qwb offset:0 atIndex:2];[e setBuffer:kwb offset:0 atIndex:3];[e setBuffer:qout offset:0 atIndex:4];[e setBuffer:kr offset:0 atIndex:5];[e setBuffer:kpst offset:0 atIndex:6];[e setBytes:&batch length:4 atIndex:7];[e setBytes:&nH length:4 atIndex:8];[e setBytes:&nKV length:4 atIndex:9];[e setBytes:&hd length:4 atIndex:10];[e setBytes:&rotary length:4 atIndex:11];[e setBuffer:posb offset:0 atIndex:12];[e setBuffer:cbf offset:0 atIndex:13];[e setBuffer:sbf offset:0 atIndex:14];[e setBytes:&qkEps length:4 atIndex:15];[e setBytes:&gain1p length:4 atIndex:16];[e setBytes:&qknorm length:4 atIndex:17];[e setBytes:&qnw length:4 atIndex:18];[e setBytes:&knw length:4 atIndex:19];[e dispatchThreadgroups:MTLSizeMake(MAX(nH,nKV),batch,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
    id<MTLBlitCommandEncoder>bl=[cb blitCommandEncoder];for(int r=0;r<batch;r++){NSUInteger dst=((NSUInteger)offsets[r]+lengths[r]-1)*kvWidth*sizeof(float),src=(NSUInteger)r*kvWidth*sizeof(float);[bl copyFromBuffer:kpst sourceOffset:src toBuffer:allK destinationOffset:dst size:(NSUInteger)kvWidth*sizeof(float)];[bl copyFromBuffer:(__bridge id<MTLBuffer>)vp sourceOffset:src toBuffer:allV destinationOffset:dst size:(NSUInteger)kvWidth*sizeof(float)];[bl copyFromBuffer:(__bridge id<MTLBuffer>)vp sourceOffset:src toBuffer:vc destinationOffset:src size:(NSUInteger)kvWidth*sizeof(float)];}[bl endEncoding];
    e=[cb computeCommandEncoder];[e setComputePipelineState:qgLaneAttn];[e setBuffer:qout offset:0 atIndex:0];[e setBuffer:allK offset:0 atIndex:1];[e setBuffer:allV offset:0 atIndex:2];[e setBuffer:gate offset:0 atIndex:3];[e setBuffer:out offset:0 atIndex:4];[e setBuffer:offb offset:0 atIndex:5];[e setBuffer:lenb offset:0 atIndex:6];[e setBytes:&batch length:4 atIndex:7];[e setBytes:&nH length:4 atIndex:8];[e setBytes:&nKV length:4 atIndex:9];[e setBytes:&hd length:4 atIndex:10];[e setBytes:&scale length:4 atIndex:11];[e dispatchThreadgroups:MTLSizeMake(nH,batch,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];mg_graph_note_encoder(g);mg_graph_note_encoder(g);mg_graph_note_encoder(g);mg_graph_note_encoder(g);*outp=(__bridge void*)out;*krawp=(__bridge void*)kr;*kpostp=(__bridge void*)kpst;*vcurp=(__bridge void*)vc;return 1;
}
