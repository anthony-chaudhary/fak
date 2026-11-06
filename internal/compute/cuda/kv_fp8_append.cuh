// SPDX-License-Identifier: Apache-2.0
// Original Fak ownership/ingress wrapper. Arithmetic is the already-pinned
// Apache-2.0 vLLM-derived kv_fp8.cuh conversion, unchanged by this leaf.
// Include AFTER kv_fp8_lifecycle.cuh. Not linked into the production TU.
#ifndef FAK_CUDA_KV_FP8_APPEND_CUH
#define FAK_CUDA_KV_FP8_APPEND_CUH
#include "../cuda_kv_fp8_append.h"
#include "kv_fp8_lifecycle.cuh"

namespace fak_kv_fp8_append {
using namespace fak_kv_fp8_lifecycle;
static bool input_live(const void *p, size_t bytes) {
  uintptr_t start = (uintptr_t)p;
  if (!p || !bytes || start % alignof(float) || bytes > UINTPTR_MAX-start) return false;
  auto it = g_live.find((void *)p);
  if (it == g_live.end() || it->second != bytes ||
      g_managed_live.find((void *)p) != g_managed_live.end() ||
      g_panel_live.find((void *)p) != g_panel_live.end() ||
      kv_snapshot_release_refused((void *)p)) return false;
  fak_alloc_tags_guard guard;
  auto tag = g_alloc_tags.find(p);
  if (tag == g_alloc_tags.end()) return false;
  const fak_alloc_tag &t = tag->second;
  // No weight, KV/recurrent owner, idle pool, graph pool or managed fallback.
  return t.bytes == bytes && t.kind == FAK_ALLOC_DEVICE_LOCAL &&
      (t.cls == FAK_ALLOC_ACTIVATIONS_SCRATCH || t.cls == FAK_ALLOC_PREFILL_SCRATCH ||
       t.cls == FAK_ALLOC_ATTENTION_SCRATCH);
}
static bool inputs(const fcuda_kv_fp8_append_input *in, size_t bytes) {
  return in && in->bytes == bytes && input_live(in->raw,bytes) &&
      input_live(in->key,bytes) && input_live(in->value,bytes);
}
static bool disjoint(const void *a, size_t na, const void *b, size_t nb) {
  uintptr_t x=(uintptr_t)a, y=(uintptr_t)b;
  if (!a || !b || na>UINTPTR_MAX-x || nb>UINTPTR_MAX-y) return false;
  return x+na<=y || y+nb<=x;
}
static bool geometry(int prefix,int tokens,int heads,int hd,int *rows,int *elements,int *old_rows) {
  if (prefix<0 || tokens<=0 || heads<=0 || hd<=0 || tokens>INT_MAX-prefix) return false;
  const int all=prefix+tokens;
  if (all>INT_MAX/heads || all*heads>INT_MAX/hd) return false;
  *rows=tokens*heads; *elements=*rows*hd; *old_rows=prefix*heads;
  // Required F32 byte extents must fit size_t even on a 32-bit host.
  return (size_t)(all*heads*hd)<=SIZE_MAX/sizeof(float);
}
static bool destination(const fcuda_kv_fp8_append_destination *d,
                        const fcuda_kv_fp8_append_input *in,int rows,int hd,int *status) {
  if (!d) return false;
  size_t elements=(size_t)rows*(size_t)hd;
  const size_t sizes[5]={elements*4,elements,elements,(size_t)rows*4,(size_t)rows*4};
  for (int i=0;i<5;++i) {
    if (d->bytes[i]!=sizes[i] || !live(d->plane[i],sizes[i],FCUDA_KV_FP8_OWNER_PLANE) ||
        ((i==0 || i>=3) && (uintptr_t)d->plane[i]%alignof(float))) return false;
    for (int j=0;j<i;++j) if (!disjoint(d->plane[i],sizes[i],d->plane[j],sizes[j])) return false;
    if (!disjoint(d->plane[i],sizes[i],in->raw,in->bytes) ||
        !disjoint(d->plane[i],sizes[i],in->key,in->bytes) ||
        !disjoint(d->plane[i],sizes[i],in->value,in->bytes) ||
        !disjoint(d->plane[i],sizes[i],status,sizeof(int))) return false;
  }
  return true;
}
} // namespace fak_kv_fp8_append

extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_append_validate_owned(
    const fcuda_kv_fp8_append_input *input,int elements,int *status) {
  using namespace fak_kv_fp8_append;
  result r=admit(FCUDA_KV_FP8_PHASE_APPEND_VALIDATE);
  if (r.status) return r;
  if (elements<=0 || (size_t)elements>SIZE_MAX/sizeof(float) ||
      !inputs(input,(size_t)elements*sizeof(float)) ||
      !live(status,sizeof(int),FCUDA_KV_FP8_OWNER_WORKSPACE) ||
      (uintptr_t)status%alignof(int) ||
      !disjoint(status,sizeof(int),input->raw,input->bytes) ||
      !disjoint(status,sizeof(int),input->key,input->bytes) ||
      !disjoint(status,sizeof(int),input->value,input->bytes))
    return refusal((int)cudaErrorInvalidValue,r.phase);
  const int code=fcuda_kv_fp8_validate_async(input->raw,input->key,input->value,elements,status);
  return code ? failed(code,r.phase) : r;
}

extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_append_ingest_owned(
    const fcuda_kv_fp8_append_destination *dst,const fcuda_kv_fp8_append_input *input,
    int prefix,int tokens,int heads,int hd,int *status) {
  using namespace fak_kv_fp8_append;
  result r=admit(FCUDA_KV_FP8_PHASE_APPEND_INGEST);
  if (r.status) return r;
  int rows=0,elements=0,old_rows=0;
  if (!geometry(prefix,tokens,heads,hd,&rows,&elements,&old_rows) ||
      !inputs(input,(size_t)elements*sizeof(float)) ||
      !live(status,sizeof(int),FCUDA_KV_FP8_OWNER_WORKSPACE) ||
      (uintptr_t)status%alignof(int) ||
      !destination(dst,input,old_rows+rows,hd,status) ||
      !disjoint(status,sizeof(int),input->raw,input->bytes) ||
      !disjoint(status,sizeof(int),input->key,input->bytes) ||
      !disjoint(status,sizeof(int),input->value,input->bytes))
    return refusal((int)cudaErrorInvalidValue,r.phase);
  const size_t offset=(size_t)old_rows*(size_t)hd;
  cudaError_t e=cudaMemcpyAsync((float *)dst->plane[0]+offset,input->raw,input->bytes,
                               cudaMemcpyDeviceToDevice,g_stream);
  if (e!=cudaSuccess) return failed((int)e,r.phase);
  int code=fcuda_kv_fp8_quantize_async((uint8_t *)dst->plane[1]+offset,
      (float *)dst->plane[3]+old_rows,input->key,rows,hd,status);
  if (code) return failed(code,r.phase);
  code=fcuda_kv_fp8_quantize_async((uint8_t *)dst->plane[2]+offset,
      (float *)dst->plane[4]+old_rows,input->value,rows,hd,status);
  return code ? failed(code,r.phase) : r;
}
#endif
