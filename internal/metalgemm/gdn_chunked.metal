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

Direct copy of MLX gated_delta_update.h helper macros and PROCESS_CHUNK_SG
from ml-explore/mlx commit 073d2252c96754e9e57ed6633be3f8ecb7058548.
Adaptations: native panel strides, fused log-decay/beta, K-by-V state layout,
and fused gated RMS normalization. C=8; partial chunks retain masked loads.
*/
#include <metal_stdlib>
using namespace metal;
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define AT(TILE, IDX) TILE.thread_elements()[IDX]
#define SUB(TILE0, TILE1, TILE2)                \
  {                                             \
    AT(TILE0, 0) = AT(TILE1, 0) - AT(TILE2, 0); \
    AT(TILE0, 1) = AT(TILE1, 1) - AT(TILE2, 1); \
  }
#define ADD(TILE0, TILE1, TILE2)                \
  {                                             \
    AT(TILE0, 0) = AT(TILE1, 0) + AT(TILE2, 0); \
    AT(TILE0, 1) = AT(TILE1, 1) + AT(TILE2, 1); \
  }
#define FMA(TILE0, S, TILE1, TILE2)                 \
  {                                                 \
    AT(TILE0, 0) = S * AT(TILE1, 0) + AT(TILE2, 0); \
    AT(TILE0, 1) = S * AT(TILE1, 1) + AT(TILE2, 1); \
  }

#define SCALE(TILE0, S) \
  {                     \
    AT(TILE0, 0) *= S;  \
    AT(TILE0, 1) *= S;  \
  }
#define SCALE2(TILE0, S0, S1) \
  {                           \
    AT(TILE0, 0) *= S0;       \
    AT(TILE0, 1) *= S1;       \
  }
#define SCALE_TRI(TILE0, S0, S1)            \
  {                                         \
    AT(TILE0, 0) *= fn > fm ? 0.f : S0;     \
    AT(TILE0, 1) *= fn + 1 > fm ? 0.f : S1; \
  }
#define SCALE_TRIEQ(TILE0, S0, S1)           \
  {                                          \
    AT(TILE0, 0) *= fn >= fm ? 0.f : S0;     \
    AT(TILE0, 1) *= fn + 1 >= fm ? 0.f : S1; \
  }

// lambdas are not supported in metal 14 so porting to macros.

// non transposed
#define LOAD_M(M, SRC, LD, B)                                                  \
  if constexpr (B) {                                                           \
    AT(M, 0) =                                                                 \
        static_cast<float>((fm < valid_rows) ? ((SRC)[fm * (LD) + fn]) : 0.f); \
    AT(M, 1) = static_cast<float>(                                             \
        (fm < valid_rows) ? ((SRC)[fm * (LD) + fn + 1]) : 0.f);                \
  } else {                                                                     \
    AT(M, 0) = static_cast<float>((SRC)[fm * (LD) + fn]);                      \
    AT(M, 1) = static_cast<float>((SRC)[fm * (LD) + fn + 1]);                  \
  }

// transposed load: sequence is the column -> mask fn / fn+1
#define LOAD_MT(M, SRC, LD, B)                                                 \
  if constexpr (B) {                                                           \
    AT(M, 0) =                                                                 \
        static_cast<float>((fn < valid_rows) ? ((SRC)[fn * (LD) + fm]) : 0.f); \
    AT(M, 1) = static_cast<float>(                                             \
        (fn + 1 < valid_rows) ? ((SRC)[(fn + 1) * (LD) + fm]) : 0.f);          \
  } else {                                                                     \
    AT(M, 0) = static_cast<float>((SRC)[fn * (LD) + fm]);                      \
    AT(M, 1) = static_cast<float>((SRC)[(fn + 1) * (LD) + fm]);                \
  }

#define PROCESS_CHUNK_SG(B, S_tile, VALID)                                     \
  {                                                                            \
    const short valid_rows = (VALID);                                          \
                                                                               \
    float g_val = (thread_index_in_simdgroup < (uint)valid_rows) \
        ? log_decay[thread_index_in_simdgroup] : 0.0f; \
                                                                               \
    float gamma_val = simd_prefix_inclusive_sum(g_val);                        \
                                                                               \
    if (thread_index_in_simdgroup < C) {                                       \
      gamma[thread_index_in_simdgroup] = gamma_val;                            \
    }                                                                          \
    simdgroup_barrier(mem_flags::mem_threadgroup);                             \
                                                                               \
    float gamma_fm = metal::fast::exp(gamma[fm]);                              \
    float gamma_fmdfn = metal::fast::exp(gamma[fm] - gamma[fn]);               \
    float gamma_fmdfn1 = metal::fast::exp(gamma[fm] - gamma[fn + 1]);          \
    float gamma_Cdfn = metal::fast::exp(gamma[C - 1] - gamma[fn]);             \
    float gamma_Cdfn1 = metal::fast::exp(gamma[C - 1] - gamma[fn + 1]);        \
    float gamma_C = metal::fast::exp(gamma[C - 1]);                            \
                                                                               \
    float beta_fm = (fm < valid_rows) ? beta_[fm] : 0.0f;        \
                                                                               \
    KKt_tile = make_filled_simdgroup_matrix<float, 8>(0.f);                    \
    MLX_MTL_PRAGMA_UNROLL                                                      \
    for (int kk = 0; kk < Dk; kk += 8) {                                       \
      LOAD_M(K_tile, k_ + kk, Dk * Hk, B)                                      \
      LOAD_MT(KT_tile, k_ + kk, Dk * Hk, B)                                    \
      simdgroup_multiply_accumulate(KKt_tile, K_tile, KT_tile, KKt_tile);      \
    }                                                                          \
                                                                               \
    KKtK_tile = KKt_tile;                                                      \
    SCALE_TRIEQ(KKtK_tile, beta_fm, beta_fm)                                   \
                                                                               \
    simdgroup_float8x8 Tinv, P;                                                \
    AT(P, 0) = AT(KKtK_tile, 0);                                               \
    AT(P, 1) = AT(KKtK_tile, 1);                                               \
    SUB(Tinv, I_tile, KKtK_tile)                                               \
                                                                               \
    MLX_MTL_PRAGMA_UNROLL                                                      \
    for (int step = 1; (1 << step) < C; step++) {                              \
      simdgroup_multiply(P, P, P);                                             \
      simdgroup_multiply_accumulate(Tinv, Tinv, P, Tinv);                      \
    }                                                                          \
                                                                               \
    WS_tile = make_filled_simdgroup_matrix<float, 8>(0.f);                     \
    MLX_MTL_PRAGMA_UNROLL                                                      \
    for (int kk = 0; kk < Dk; kk += 8) {                                       \
      LOAD_M(K_tile, k_ + kk, Dk * Hk, B)                                      \
      SCALE(K_tile, beta_fm)                                                   \
      simdgroup_multiply(W_tile, Tinv, K_tile);                                \
      SCALE(W_tile, gamma_fm)                                                  \
      simdgroup_multiply_accumulate(WS_tile, W_tile, S_tile[kk / 8], WS_tile); \
    }                                                                          \
                                                                               \
    SCALE_TRI(Tinv, gamma_fmdfn, gamma_fmdfn1)                                 \
                                                                               \
    LOAD_M(V_tile, v_ + dv_idx, convDim, B)                                    \
    SCALE(V_tile, beta_fm)                                                     \
    simdgroup_multiply(U_tile, Tinv, V_tile);                                  \
    SUB(delta_tile, U_tile, WS_tile)                                           \
                                                                               \
    tmp_tile = make_filled_simdgroup_matrix<float, 8>(0.f);                    \
    QKt_tile = make_filled_simdgroup_matrix<float, 8>(0.f);                    \
    MLX_MTL_PRAGMA_UNROLL                                                      \
    for (int kk = 0; kk < Dk; kk += 8) {                                       \
      LOAD_M(Q_tile, q_ + kk, Hk * Dk, B)                                      \
      LOAD_MT(K_tile, k_ + kk, Hk * Dk, B)                                     \
      simdgroup_multiply_accumulate(QKt_tile, Q_tile, K_tile, QKt_tile);       \
      SCALE(Q_tile, gamma_fm)                                                  \
      simdgroup_multiply_accumulate(                                           \
          tmp_tile, Q_tile, S_tile[kk / 8], tmp_tile);                         \
    }                                                                          \
                                                                               \
    SCALE_TRI(QKt_tile, gamma_fmdfn, gamma_fmdfn1)                             \
                                                                               \
    simdgroup_multiply_accumulate(out_tile, QKt_tile, delta_tile, tmp_tile);   \
                                                                               \
    if (fm < valid_rows) {                                                     \
      y[fm * Hv * Dv + dv_idx + fn] = static_cast<InT>(AT(out_tile, 0));       \
      y[fm * Hv * Dv + dv_idx + fn + 1] = static_cast<InT>(AT(out_tile, 1));   \
    }                                                                          \
                                                                               \
    MLX_MTL_PRAGMA_UNROLL                                                      \
    for (int kk = 0; kk < Dk; kk += 8) {                                       \
      LOAD_MT(K_tile, k_ + kk, Hk * Dk, B)                                     \
      SCALE2(K_tile, gamma_Cdfn, gamma_Cdfn1)                                  \
      simdgroup_multiply(KD_tile, K_tile, delta_tile);                         \
      FMA(S_tile[kk / 8], gamma_C, S_tile[kk / 8], KD_tile)                    \
    }                                                                          \
  }


kernel void gdn_recurrent_chunk8(device const float *convOut [[buffer(0)]],
 device const float *qNorm [[buffer(1)]], device const float *kNorm [[buffer(2)]],
 device const float *z [[buffer(3)]], device const float *b [[buffer(4)]],
 device const float *a [[buffer(5)]], device const float *aLog [[buffer(6)]],
 device const float *dtBias [[buffer(7)]], device const float *norm [[buffer(8)]],
 device float *state [[buffer(9)]], device float *core [[buffer(10)]],
 constant int& tokens [[buffer(11)]], constant int& convDim [[buffer(12)]],
 constant int& nK [[buffer(13)]], constant int& nV [[buffer(14)]],
 constant int& kHd [[buffer(15)]], constant int& vHd [[buffer(16)]],
 constant float& eps [[buffer(17)]], uint head [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]], uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
 const int Dk=128, C=8, Dv=vHd, Hk=nK, Hv=nV;
 typedef float InT;
 const uint hv_idx=head, hk_idx=head/(nV/nK), sg_id=tid/32;
 const int dv_idx=sg_id*8;
 const short qid=thread_index_in_simdgroup/4;
 const short fm=(qid&4)+((thread_index_in_simdgroup/2)%4);
 const short fn=(qid&2)*2+(thread_index_in_simdgroup%2)*2;
 device const float *q_=qNorm+hk_idx*Dk, *k_=kNorm+hk_idx*Dk;
 device const float *v_=convOut+2*nK*Dk+head*Dv;
 device float *y=core+head*Dv;
 threadgroup float gamma_all[8*32], log_decay[8], beta_values[8];
 threadgroup float squares[8*32];
 threadgroup float *gamma=gamma_all+sg_id*C;
 threadgroup float *beta_=beta_values;
 simdgroup_float8x8 S_tile[Dk/8];
 simdgroup_float8x8 V_tile,K_tile,KT_tile,Q_tile,W_tile,U_tile,WS_tile;
 simdgroup_float8x8 delta_tile,tmp_tile,QKt_tile,out_tile,KD_tile,KKtK_tile,KKt_tile;
 simdgroup_float8x8 I_tile=make_filled_simdgroup_matrix<float,8>(0.f);
 AT(I_tile,0)=(fm==fn)?1.f:0.f;
 AT(I_tile,1)=(fm==fn+1)?1.f:0.f;
 for(int kk=0;kk<Dk;kk+=8)
   simdgroup_load(S_tile[kk/8],state+(long)head*Dk*Dv+kk*Dv+dv_idx,Dv);
 for(int t=0;t<tokens;t+=C) {
   if(tid<C) {
     int token=t+tid;
     float av=token<tokens?a[(long)token*Hv+head]+dtBias[head]:0.f;
     log_decay[tid]=token<tokens?-exp(aLog[head])*gdn_softplus(av):0.f;
     beta_values[tid]=token<tokens?1.f/(1.f+exp(-b[(long)token*Hv+head])):0.f;
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
   if(t+C<=tokens) { PROCESS_CHUNK_SG(false,S_tile,C); }
   else { PROCESS_CHUNK_SG(true,S_tile,short(tokens-t)); }
   // Each SIMDgroup owns eight columns; normalize all value columns together.
   threadgroup_barrier(mem_flags::mem_device);
   for(int row=0;row<min(C,tokens-t);++row) {
     uint vi=sg_id*8+thread_index_in_simdgroup;
     float sq=thread_index_in_simdgroup<8?y[row*Hv*Dv+vi]*y[row*Hv*Dv+vi]:0.f;
     float sum=simd_sum(sq);
     if(thread_index_in_simdgroup==0) squares[row*32+sg_id]=sum;
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
   for(int row=0;row<min(C,tokens-t);++row) {
     float sum=0.f;
     for(int sg=0;sg<Dv/8;++sg) sum+=squares[row*32+sg];
     uint vi=sg_id*8+thread_index_in_simdgroup;
     if(thread_index_in_simdgroup<8) {
       long idx=(long)(t+row)*Hv*Dv+head*Dv+vi;
       core[idx]=norm[vi]*core[idx]*rsqrt(sum/Dv+eps)*gdn_silu(z[idx]);
     }
   }
   threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
   q_+=C*Hk*Dk; k_+=C*Hk*Dk; v_+=C*convDim; y+=C*Hv*Dv;
 }
 for(int kk=0;kk<Dk;kk+=8)
   simdgroup_store(S_tile[kk/8],state+(long)head*Dk*Dv+kk*Dv+dv_idx,Dv);
}
