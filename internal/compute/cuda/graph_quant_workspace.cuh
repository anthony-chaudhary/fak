// SPDX-License-Identifier: Apache-2.0
// Qualification-only fixed P=1 Q4_K/Q8_0 graph scratch. Arithmetic is reused
// unchanged from q4k_mmvq.cuh (llama.cpp MIT pin in that file) and quant_matmul.cuh.
// Included after fak_graph_once and graph_once_capture_admit; never a standalone TU.
// No HAL/GDN capability or serving activation is supplied by this leaf.
#ifndef FAK_GRAPH_QUANT_WORKSPACE_CUH
#define FAK_GRAPH_QUANT_WORKSPACE_CUH
#include <new>

enum { FAK_GRAPH_QUANT_Q4_K = 1, FAK_GRAPH_QUANT_Q8_0 = 2 };
struct fak_graph_quant_op {
  int kind, out, in;
  const void *weights;
  const float *scales; // Q8 block-32 scales; must be null for packed Q4_K
  const float *input;
  float *output;
};
struct graph_quant_span { uintptr_t lo, hi; };
static bool graph_quant_overlap(graph_quant_span a, graph_quant_span b) {
  return a.lo < b.hi && b.lo < a.hi;
}
static bool graph_quant_live(const void *p, size_t bytes, graph_quant_span *span) {
  uintptr_t lo = (uintptr_t)p;
  if (!p || !bytes || lo % alignof(float) || bytes > UINTPTR_MAX-lo) return false;
  auto it = g_live.find((void *)p);
  if (it == g_live.end() || it->second < bytes) return false;
  *span = {lo, lo+bytes};
  return true;
}
// Span order: input, output, weights, optional Q8 scales. Exact allocation bases
// deliberately exclude arbitrary slices. Pinning/content generations are an
// additional caller contract, not established by a live-address lookup.
static bool graph_quant_op_spans(const fak_graph_quant_op &op, graph_quant_span span[4]) {
  if (op.in <= 0 || op.out <= 0 || op.in % 32 ||
      (size_t)op.in > SIZE_MAX/sizeof(float) ||
      (size_t)op.out > SIZE_MAX/sizeof(float)) return false;
  size_t weight_bytes, scale_bytes = 0;
  if (op.kind == FAK_GRAPH_QUANT_Q4_K) {
    if (op.in % 256 || op.scales || op.out > INT_MAX/(op.in/256)) return false;
    // The reused MMVQ kernel's row*blocks_per_row is signed int.
    size_t blocks = (size_t)op.out*(op.in/256);
    if (blocks > SIZE_MAX/sizeof(fak_q4k_mmvq::block_q4_K)) return false;
    weight_bytes = blocks*sizeof(fak_q4k_mmvq::block_q4_K);
  } else if (op.kind == FAK_GRAPH_QUANT_Q8_0) {
    if (!op.scales || (size_t)op.out > SIZE_MAX/(size_t)op.in) return false;
    weight_bytes = (size_t)op.out*op.in;
    size_t blocks = weight_bytes/32;
    if (blocks > SIZE_MAX/sizeof(float)) return false;
    scale_bytes = blocks*sizeof(float);
  } else return false;
  span[3] = {0,0};
  return graph_quant_live(op.input,(size_t)op.in*sizeof(float),&span[0]) &&
    graph_quant_live(op.output,(size_t)op.out*sizeof(float),&span[1]) &&
    graph_quant_live(op.weights,weight_bytes,&span[2]) &&
    (!scale_bytes || graph_quant_live(op.scales,scale_bytes,&span[3])) &&
    !graph_quant_overlap(span[0],span[1]);
}
static bool graph_quant_plan_valid(fak_graph_once *s, const fak_graph_quant_op *ops, int count,
                                    bool check_cross_alias) {
  if (!s || !s->position || !ops || count <= 0 || count > 4096) return false;
  uintptr_t p = (uintptr_t)s->position;
  if (p % alignof(int) || p > UINTPTR_MAX-2*sizeof(int)) return false;
  graph_quant_span position{p,p+2*sizeof(int)};
  for (int i=0; i<count; ++i) {
    graph_quant_span a[4];
    if (!graph_quant_op_spans(ops[i],a) || graph_quant_overlap(a[1],position)) return false;
    // Mutable outputs may be reused by sequential operations and feed later
    // inputs, but no operation may overwrite ANY frozen weight or scale plane.
    if (check_cross_alias) {
      for (int j=0; j<count; ++j) {
        graph_quant_span b[4];
        if (!graph_quant_op_spans(ops[j],b) || graph_quant_overlap(a[1],b[2]) ||
            graph_quant_overlap(a[1],b[3])) return false;
      }
    }
  }
  return true;
}
static int graph_quant_error(fak_graph_once *s, cudaError_t e, int rc) {
  s->terminal = true;
  if (e != cudaSuccess && !s->quant_error) s->quant_error = (int)e;
  if (e != cudaSuccess && e != cudaErrorMemoryAllocation && !g_cuda_cleanup_status)
    g_cuda_cleanup_status = (int)e;
  return rc;
}
static bool graph_once_quant_ready(fak_graph_once *s) {
  if (!s->quant_prepared) return !s->quant_ops && s->quant_count == 0;
  if (!s->quant_ops || s->quant_count <= 0) return false;
  for (int i=0; i<3; ++i)
    if ((s->quant_bytes[i] != 0) != (s->quant_scratch[i] != nullptr)) return false;
  return graph_quant_plan_valid(s,(const fak_graph_quant_op *)s->quant_ops,s->quant_count,false);
}
static bool graph_once_quant_idle(fak_graph_once *s) {
  if (!s->quant_prepared) return true; // preserve the preexisting dense envelope
  cudaStreamCaptureStatus capture;
  cudaError_t e = cudaStreamIsCapturing(g_stream,&capture);
  if (e != cudaSuccess) { graph_quant_error(s,e,3); return false; }
  return capture == cudaStreamCaptureStatusNone && graph_once_quant_ready(s);
}
// budget bounds ADDITIONAL private quant device bytes only. The caller separately
// admits global device capacity and retains the position/model/KV allocations.
// No existing plan can resize or be replaced, even after a preparation failure.
extern "C" int fak_graph_once_quant_prepare(void *owner, const fak_graph_quant_op *ops,
                                            int count, size_t budget) {
  fak_graph_once *s = (fak_graph_once *)owner;
  if (!s || s->terminal || !s->position || s->graph || s->exec || s->quant_ops ||
      s->quant_prepared || g_capture_open || g_graph_once_capture ||
      v41_candidates_stream_uncertain() || !graph_quant_plan_valid(s,ops,count,true)) return 1;
  cudaStreamCaptureStatus capture;
  cudaError_t e = cudaStreamIsCapturing(g_stream,&capture);
  if (e != cudaSuccess) return graph_quant_error(s,e,3);
  if (capture != cudaStreamCaptureStatusNone) return 1;
  e = cudaPeekAtLastError();
  if (e != cudaSuccess) return graph_quant_error(s,e,3);
  size_t need[3] = {0,0,0};
  for (int i=0; i<count; ++i) {
    size_t n = (size_t)ops[i].in;
    if (ops[i].kind == FAK_GRAPH_QUANT_Q4_K) {
      if (n/32 > SIZE_MAX/sizeof(fak_q4k_mmvq::block_q8_1)) return 1;
      size_t bytes = (n/32)*sizeof(fak_q4k_mmvq::block_q8_1);
      if (need[0] < bytes) need[0] = bytes;
    } else {
      if (need[1] < n) need[1] = n;
      if (need[2] < (n/32)*sizeof(float)) need[2] = (n/32)*sizeof(float);
    }
  }
  size_t total = 0;
  for (int i=0; i<3; ++i) {
    if (need[i] > budget-total) return 2;
    total += need[i];
  }
  // Retain the immutable host plan before device allocation. The bound count
  // makes this product safe on both 32-bit and 64-bit hosts.
  s->quant_ops = calloc((size_t)count,sizeof(fak_graph_quant_op));
  if (!s->quant_ops) return 2;
  memcpy(s->quant_ops,ops,(size_t)count*sizeof(fak_graph_quant_op));
  s->quant_count = count;
  for (int i=0; i<3; ++i) s->quant_bytes[i] = need[i];
  for (int i=0; i<3; ++i) {
    if (!need[i]) continue;
    // Store the pointer directly in its retained owner slot before accounting.
    e = fcuda_alloc_status(cudaMalloc(&s->quant_scratch[i],need[i]));
    if (e != cudaSuccess || !s->quant_scratch[i])
      return graph_quant_error(s,e == cudaSuccess ? cudaErrorMemoryAllocation : e,2);
    try {
      fak_alloc_track(s->quant_scratch[i],FAK_ALLOC_GRAPH_POOL,FAK_ALLOC_DEVICE_LOCAL,
                      FAK_ALLOC_SITE("graph-once-fixed-quant"),need[i]);
      s->quant_tracked[i] = true;
    } catch (const std::bad_alloc &) {
      // No submitted GPU work. Keep the pointer, and release it without an
      // untrack if tag insertion failed. Do not throw through the C ABI.
      return graph_quant_error(s,cudaErrorMemoryAllocation,2);
    }
  }
  s->quant_prepared = true;
  return 0;
}
// Every recorded operation uses frozen dimensions/addresses and this owner's
// private scratch. The cursor counts records, NEVER completed tokens or kernels.
extern "C" int fak_graph_once_quant_matmul(void *owner, int operation) {
  fak_graph_once *s = (fak_graph_once *)owner;
  if (!graph_once_capture_admit(s)) return 1;
  if (v41_candidates_stream_uncertain() || !s->quant_prepared ||
      operation != s->quant_cursor || operation < 0 || operation >= s->quant_count) {
    s->terminal = true; return 1;
  }
  const fak_graph_quant_op &op = ((const fak_graph_quant_op *)s->quant_ops)[operation];
  graph_quant_span span[4];
  if (!graph_quant_op_spans(op,span)) { s->terminal = true; return 1; }
  cudaStreamCaptureStatus capture;
  cudaError_t e = cudaStreamIsCapturing(g_stream,&capture);
  if (e != cudaSuccess) return graph_quant_error(s,e,3);
  if (capture != cudaStreamCaptureStatusActive) { s->terminal = true; return 1; }
  e = cudaPeekAtLastError();
  if (e != cudaSuccess) return graph_quant_error(s,e,3);
  if (op.kind == FAK_GRAPH_QUANT_Q4_K) {
    using namespace fak_q4k_mmvq;
    auto *scratch = (block_q8_1 *)s->quant_scratch[0];
    dim3 qgrid((unsigned)op.in/256,1), grid((unsigned)op.out), block(32,4);
    FCUDA_TS("quantize_q8_1", qgrid, quantize_q8_1<<<qgrid,256,0,g_stream>>>(op.input,scratch,op.in));
    e = cudaGetLastError();
    if (e != cudaSuccess) return graph_quant_error(s,e,3);
    FCUDA_TS("k_q4k_mmvq_gemv", grid, mul_mat_vec_q<1><<<grid,block,0,g_stream>>>(op.weights,scratch,op.output,op.in,op.out));
    e = cudaGetLastError();
    if (e != cudaSuccess) return graph_quant_error(s,e,3);
  } else {
    auto *codes = (signed char *)s->quant_scratch[1];
    auto *scales = (float *)s->quant_scratch[2];
    dim3 qgrid((unsigned)op.in/32,1), grid((unsigned)op.out/8+(op.out%8 != 0),1);
    FCUDA_TS("k_q8_quant_act", qgrid, k_q8_quant_act<<<qgrid,32,0,g_stream>>>(op.input,codes,scales,1,op.in,32));
    e = cudaGetLastError();
    if (e != cudaSuccess) return graph_quant_error(s,e,3);
    FCUDA_TS("k_q8_gemm", grid, k_q8_gemm<<<grid,256,0,g_stream>>>((const signed char *)op.weights,op.scales,codes,scales,op.output,op.out,op.in,1,32));
    e = cudaGetLastError();
    if (e != cudaSuccess) return graph_quant_error(s,e,3);
  }
  ++s->quant_cursor;
  return 0;
}
// Only call after checked drain AND graph/exec destruction. Retain all remaining
// pointers on a failed free; never remove accounting until successful retirement.
static int graph_once_quant_retire(fak_graph_once *s) {
  for (int i=0; i<3; ++i) {
    if (!s->quant_scratch[i]) continue;
    cudaError_t e = cudaFree(s->quant_scratch[i]);
    if (e != cudaSuccess) {
      // Even an OOM-labelled retirement error has no successful disposition.
      // Close shared admission so release cannot retry this pointer.
      if (!g_cuda_cleanup_status) g_cuda_cleanup_status = (int)e;
      return graph_quant_error(s,e,2);
    }
    if (s->quant_tracked[i]) fak_alloc_untrack(s->quant_scratch[i]);
    s->quant_scratch[i] = nullptr;
    s->quant_bytes[i] = 0;
    s->quant_tracked[i] = false;
  }
  free(s->quant_ops);
  s->quant_ops = nullptr;
  s->quant_count = s->quant_cursor = 0;
  s->quant_prepared = false;
  return 0;
}
#endif
