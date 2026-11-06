// SPDX-License-Identifier: Apache-2.0
// Original Fak exact-ownership wrapper. Numerical attention remains the existing
// F32 online-softmax kernel in kv_fp8.cuh, not a tensor-core or graph port.
// Include after kv_fp8_append.cuh; intentionally unlinked from production.
#ifndef FAK_CUDA_KV_FP8_ATTENTION_OWNER_CUH
#define FAK_CUDA_KV_FP8_ATTENTION_OWNER_CUH
#include "../cuda_kv_fp8_attention_owner.h"
#include "kv_fp8_append.cuh"

extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_attention_owned(
    const fcuda_kv_fp8_append_destination *k,
    const float *q,size_t q_bytes,float *out,size_t out_bytes,
    int tokens,int prefix,int heads,float scale,int *status) {
  using namespace fak_kv_fp8_lifecycle;
  result r=admit(FCUDA_KV_FP8_PHASE_ATTENTION);
  if (r.status) return r;
  const int kvheads=4,hd=256;
  int rows=0,elements=0,old_rows=0;
  if (!k || !fak_kv_fp8_append::geometry(prefix,tokens,kvheads,hd,&rows,&elements,&old_rows) ||
      heads<=0 || heads%kvheads || heads>INT_MAX/hd || tokens>INT_MAX/(heads*hd) ||
      (size_t)(tokens*heads*hd)>SIZE_MAX/sizeof(float) ||
      !(scale>0.0f) || !isfinite(scale))
    return refusal((int)cudaErrorInvalidValue,r.phase);
  const size_t expected=(size_t)tokens*heads*hd*sizeof(float);
  if (q_bytes!=expected || out_bytes!=expected || !fak_kv_fp8_append::input_live(q,expected) ||
      !live(out,expected,FCUDA_KV_FP8_OWNER_WORKSPACE) || (uintptr_t)out%alignof(float) ||
      !live(status,sizeof(int),FCUDA_KV_FP8_OWNER_WORKSPACE) || (uintptr_t)status%alignof(int) ||
      !fak_kv_fp8_append::disjoint(q,expected,out,expected) ||
      !fak_kv_fp8_append::disjoint(q,expected,status,sizeof(int)) ||
      !fak_kv_fp8_append::disjoint(out,expected,status,sizeof(int)))
    return refusal((int)cudaErrorInvalidValue,r.phase);
  const size_t total_rows=(size_t)old_rows+rows,total=total_rows*hd;
  const size_t sizes[5]={total*4,total,total,total_rows*4,total_rows*4};
  for (int i=0;i<5;++i) {
    if (k->bytes[i]!=sizes[i] || !live(k->plane[i],sizes[i],FCUDA_KV_FP8_OWNER_PLANE) ||
        ((i==0 || i>=3) && (uintptr_t)k->plane[i]%alignof(float)) ||
        !fak_kv_fp8_append::disjoint(k->plane[i],sizes[i],q,expected) ||
        !fak_kv_fp8_append::disjoint(k->plane[i],sizes[i],out,expected) ||
        !fak_kv_fp8_append::disjoint(k->plane[i],sizes[i],status,sizeof(int)))
      return refusal((int)cudaErrorInvalidValue,r.phase);
    for (int j=0;j<i;++j)
      if (!fak_kv_fp8_append::disjoint(k->plane[i],sizes[i],k->plane[j],sizes[j]))
        return refusal((int)cudaErrorInvalidValue,r.phase);
  }
  int code=fcuda_kv_fp8_attention_async(q,(const uint8_t *)k->plane[1],(const float *)k->plane[3],
      (const uint8_t *)k->plane[2],(const float *)k->plane[4],out,
      tokens,prefix,heads,kvheads,hd,scale,status);
  return code ? failed(code,r.phase) : r;
}
#endif
