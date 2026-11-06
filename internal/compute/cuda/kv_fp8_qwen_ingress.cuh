// SPDX-License-Identifier: Apache-2.0
// Original Fak ownership wrapper. Exact arithmetic is reused from existing
// transformer_ops.cuh, not the distinct fused vLLM QK implementation.
// Include after transformer_ops.cuh, kv_fp8_append.cuh and kv_fp8_rotary.cuh.
// Unlinked qualification leaf: no production TU or model activation.
#ifndef FAK_CUDA_KV_FP8_QWEN_INGRESS_CUH
#define FAK_CUDA_KV_FP8_QWEN_INGRESS_CUH
#include "../cuda_kv_fp8_qwen_ingress.h"

namespace fak_kv_fp8_qwen {
using namespace fak_kv_fp8_append;
// g_ts_on is immutable after process initialization and read under cudaMu.
// Original FCUDA_TS executes its body even after a timing begin failure;
// original panel RoPE has two launches with only CUDA last-error checks.
// Refuse timing before this leaf's allocations, not only after a hidden fault.
static result admit_qwen(int phase) {
 result r=admit(phase);
 if (r.status) return r; // preserve pre-existing terminal/runtime cause first
 if (g_ts_on) return refusal(FCUDA_KV_FP8_QWEN_TIMING_UNSUPPORTED,phase);
 return r;
}
static result producer_result(int code, int phase) {
 // Prefer a durable first observed terminal cause to a later return code.
 if (v41_candidates_stream_uncertain())
  return failed(g_cuda_cleanup_status ? g_cuda_cleanup_status :
      (code ? code : (int)cudaErrorNotReady),phase);
 if (code) return failed(code,phase);
 return admit_qwen(phase); // includes non-clearing runtime observation
}
static bool weight_live(const void *p, size_t bytes) {
 uintptr_t start=(uintptr_t)p;
 if (!p || bytes!=256*sizeof(float) || start%alignof(float) || bytes>UINTPTR_MAX-start ||
     g_managed_live.find((void *)p)!=g_managed_live.end() ||
     g_panel_live.find((void *)p)!=g_panel_live.end() || kv_snapshot_release_refused((void *)p)) return false;
 auto it=g_live.find((void *)p);
 if (it==g_live.end() || it->second!=bytes) return false;
 fak_alloc_tags_guard guard;
 auto tag=g_alloc_tags.find(p);
 return tag!=g_alloc_tags.end() && tag->second.bytes==bytes &&
     tag->second.kind==FAK_ALLOC_DEVICE_LOCAL && tag->second.cls==FAK_ALLOC_WEIGHTS;
}
static bool valid(const fcuda_kv_fp8_qwen_stage *s) {
 int rows=0,elements=0,oldrows=0;
 if (!s || s->heads!=24 || s->rotary!=64 || s->producer!=FCUDA_KV_FP8_QWEN_PARTIAL_PANEL_V1 ||
     !(s->epsilon>0) || !isfinite(s->epsilon) || !(s->theta>0) || !isfinite(s->theta) ||
     !geometry(s->first,s->tokens,4,256,&rows,&elements,&oldrows) ||
     s->tokens>INT_MAX/(24*256) || (size_t)(s->tokens*24*256)>SIZE_MAX/sizeof(float)) return false;
 const size_t q=(size_t)s->tokens*24*256*sizeof(float), k=(size_t)s->tokens*4*256*sizeof(float);
 const size_t ins[5]={q,k,k,256*sizeof(float),256*sizeof(float)}, outs[4]={q,k,q,k};
 if (!live(s->status,sizeof(int),FCUDA_KV_FP8_OWNER_WORKSPACE) || (uintptr_t)s->status%alignof(int)) return false;
 for (int i=0;i<5;++i) {
  if (s->input_bytes[i]!=ins[i] ||
      !(i<3 ? input_live(s->input[i],ins[i]) : weight_live(s->input[i],ins[i])) ||
      !disjoint(s->input[i],ins[i],s->status,sizeof(int))) return false;
 }
 for (int i=0;i<4;++i) {
  if (s->work_bytes[i]!=outs[i] || (uintptr_t)s->work[i]%alignof(float) ||
      !live(s->work[i],outs[i],FCUDA_KV_FP8_OWNER_WORKSPACE) ||
      !disjoint(s->work[i],outs[i],s->status,sizeof(int))) return false;
  for (int j=0;j<5;++j)
   if (!disjoint(s->work[i],outs[i],s->input[j],ins[j])) return false;
  for (int j=0;j<i;++j)
   if (!disjoint(s->work[i],outs[i],s->work[j],outs[j])) return false;
 }
 return true; // read-only inputs may share storage; writable spans never may
}
static int scan(const float *p, size_t bytes, int *status) {
 // Geometry bound proves count<=INT_MAX. Repetition is intentional: use the
 // existing numerical primitive without adding a second finite-policy kernel.
 return fcuda_kv_fp8_validate_async(p,p,p,(int)(bytes/sizeof(float)),status);
}
} // namespace fak_kv_fp8_qwen

extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_qwen_admit_owned(void) {
 return fak_kv_fp8_qwen::admit_qwen(FCUDA_KV_FP8_PHASE_QWEN_ADMIT);
}
extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_qwen_validate_owned(const fcuda_kv_fp8_qwen_stage *s) {
 using namespace fak_kv_fp8_qwen;
 result r=admit_qwen(FCUDA_KV_FP8_PHASE_QWEN_VALIDATE);
 if (r.status) return r;
 if (!valid(s)) return refusal((int)cudaErrorInvalidValue,r.phase);
 for (int i=0;i<5;++i) {
  r=producer_result(scan(s->input[i],s->input_bytes[i],s->status),r.phase);
  if (r.status) return r;
 }
 return r; // caller must check completion/numerical status before norm
}
extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_qwen_norm_owned(const fcuda_kv_fp8_qwen_stage *s) {
 using namespace fak_kv_fp8_qwen;
 result r=admit_qwen(FCUDA_KV_FP8_PHASE_QWEN_NORM);
 if (r.status) return r;
 if (!valid(s)) return refusal((int)cudaErrorInvalidValue,r.phase);
 // EXACT original qwen35SequenceRMSNormLocked path: block tree, not dispatched warp reduction.
 fcuda_rmsnorm_f32(s->input[0],s->input[3],s->work[0],s->tokens*24,256,s->epsilon);
 r=producer_result((int)cudaGetLastError(),r.phase); if (r.status) return r;
 fcuda_rmsnorm_f32(s->input[1],s->input[4],s->work[1],s->tokens*4,256,s->epsilon);
 r=producer_result((int)cudaGetLastError(),r.phase); if (r.status) return r;
 for (int i=0;i<2;++i) {
  r=producer_result(scan(s->work[i],s->work_bytes[i],s->status),r.phase);
  if (r.status) return r;
 }
 return r; // caller must check before rotary reads normalized workspace
}
extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_qwen_rope_owned(const fcuda_kv_fp8_qwen_stage *s) {
 using namespace fak_kv_fp8_qwen;
 result r=admit_qwen(FCUDA_KV_FP8_PHASE_QWEN_ROPE);
 if (r.status) return r;
 if (!valid(s)) return refusal((int)cudaErrorInvalidValue,r.phase);
 const int code=fak_qwen35_partial_rope_panel_f32(s->work[0],s->work[1],s->work[2],s->work[3],
     s->tokens,s->first,24,4,256,64,s->theta);
 r=producer_result(code,r.phase); if (r.status) return r;
 for (int i=2;i<4;++i) {
  r=producer_result(scan(s->work[i],s->work_bytes[i],s->status),r.phase);
  if (r.status) return r;
 }
 return r; // finite inputs may overflow rotation: checked finish precedes append
}
#endif
