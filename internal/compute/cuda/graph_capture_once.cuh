// Qualification-only #10945: one capture, stable device position scalars, replay many.
// Not connected to the HAL. The caller holds cudaMu and retains EVERY captured
// input/output/KV/scratch allocation unchanged until release returns zero.
// Existing RoPE/flash arithmetic is shared; no scalar-dependent launch grid.
struct fak_graph_once {
  int *position;
  int capacity;
  int next;
  bool terminal;
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  // POD fields keep calloc construction and existing create extraction valid.
  void *quant_ops;
  void *quant_scratch[3];
  size_t quant_bytes[3];
  bool quant_tracked[3];
  int quant_count, quant_cursor, quant_error;
  bool quant_prepared;
};
static fak_graph_once *g_graph_once_capture = nullptr;
static bool graph_once_quant_ready(fak_graph_once *s);
static bool graph_once_quant_idle(fak_graph_once *s);
static int graph_once_quant_retire(fak_graph_once *s);

__global__ void k_graph_once_position(int *state, int pos) {
  state[0] = pos;
  state[1] = pos + 1;
}
__global__ void k_graph_once_copyrow(float *dst, const float *src,
                                    const int *position, int width) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)width) dst[(size_t)position[0] * width + i] = src[i];
}

// A failed submission may have enqueued work. Drain before deciding anything
// about lifetime. Poison remains sticky even when a later release can drain.
static int graph_once_drain(fak_graph_once *s) {
  cudaError_t drain = cudaStreamSynchronize(g_stream);
  if (drain != cudaSuccess) {
    s->terminal = true;
    // Latch the FIRST failed completion immediately. Never retry the fence
    // before recording uncertainty: a later successful drain cannot repair it.
    if (!g_cuda_cleanup_status) g_cuda_cleanup_status = (int)drain;
    return 5;
  }
  return 0;
}
static int graph_once_fail(fak_graph_once *s, int code) {
  s->terminal = true;
  int drain = graph_once_drain(s);
  return drain ? drain : code;
}
static int graph_once_set_position(fak_graph_once *s, int pos) {
  FCUDA_TS("k_graph_once_position", 1, k_graph_once_position<<<1, 1, 0, g_stream>>>(s->position, pos));
  return cudaGetLastError() == cudaSuccess ? 0 : graph_once_fail(s, 3);
}

extern "C" int fak_graph_once_create(int capacity, void **owner) {
  if (!owner || *owner || capacity <= 0 || capacity == INT_MAX || g_capture_open ||
      v41_candidates_stream_uncertain()) return 1;
  fak_graph_once *s = (fak_graph_once *)calloc(1, sizeof(fak_graph_once));
  if (!s) return 2;
  s->capacity = capacity;
  // Publish the owner before native allocation so every partial failure remains
  // explicitly releasable, never silently leaked or mistaken for an empty owner.
  *owner = s;
  if (fcuda_alloc_status(cudaMalloc((void **)&s->position, 2 * sizeof(int))) != cudaSuccess) {
    s->terminal = true;
    return 2;
  }
  return 0;
}

extern "C" int fak_graph_once_begin(void *owner, int pos) {
  fak_graph_once *s = (fak_graph_once *)owner;
  if (!s || s->terminal || !s->position || s->graph || s->exec ||
      g_capture_open || g_graph_once_capture || pos < 0 || pos >= s->capacity ||
      v41_candidates_stream_uncertain()) return 1;
  if (!graph_once_quant_idle(s)) { s->terminal = true; return 1; }
  s->quant_cursor = 0;
  int rc = graph_once_set_position(s, pos);
  if (rc) return rc;
  if (graph_once_drain(s)) return 5;
  if (cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeGlobal) != cudaSuccess)
    return graph_once_fail(s, 3);
  s->next = pos + 1;
  g_graph_once_capture = s;
  g_capture_open = true;
  return 0;
}

// These wrappers are capture-only, and consume the SAME owner scalar pair.
// Invalid geometry poisons capture; end must abort before release, never launch a partial graph.
static int graph_once_capture_admit(fak_graph_once *s) {
  return s && s == g_graph_once_capture && g_capture_open && !s->terminal;
}
#include "graph_quant_workspace.cuh"
#include "graph_gdn_capture.cuh"
extern "C" int fak_graph_once_rope(void *owner, float *x, int heads, int hd, double theta) {
  fak_graph_once *s = (fak_graph_once *)owner;
  if (!graph_once_capture_admit(s)) return 1;
  if (!x || heads <= 0 || hd <= 0 || (hd & 1) || heads > INT_MAX / hd ||
      !(theta > 0.0) || !isfinite(theta)) { s->terminal = true; return 1; }
  unsigned total = (unsigned)heads * (hd / 2);
  FCUDA_TS("k_rope", total / 128 + (total % 128 != 0), k_rope<true><<<total / 128 + (total % 128 != 0), 128, 0, g_stream>>>(x, 0, heads, hd, theta, s->position));
  if (cudaGetLastError() != cudaSuccess) { s->terminal = true; return 3; }
  return 0;
}
// Native prerequisite only: no Go/HAL caller or hybrid admission is added here.
// Caller retains all four buffers and the graph owner through checked retirement.
// These span checks establish live device allocation extents, not dependency
// generations or the model's public execution scope; those remain caller gates.
static bool graph_once_partial_rope_span(const void *ptr, size_t bytes,
                                         uintptr_t *lo, uintptr_t *hi) {
  uintptr_t begin = (uintptr_t)ptr;
  if (!ptr || !bytes || begin % alignof(float) || bytes > UINTPTR_MAX - begin)
    return false;
  *lo = begin;
  *hi = begin + bytes;
  return true;
}
static bool graph_once_partial_rope_inputs(fak_graph_once *s,
    const float *q, const float *k, float *qout, float *kout,
    int qheads, int kheads, int hd, int rotary, double theta) {
  if (!s || !s->position || qheads <= 0 || kheads <= 0 || hd <= 0 ||
      rotary <= 0 || rotary > hd || (rotary & 1) || qheads % kheads ||
      !(theta > 0.0) || !isfinite(theta) ||
      qheads > INT_MAX / hd || kheads > INT_MAX / hd ||
      (size_t)qheads > SIZE_MAX / sizeof(float) / (size_t)hd ||
      (size_t)kheads > SIZE_MAX / sizeof(float) / (size_t)hd) return false;
  size_t qbytes = (size_t)qheads * hd * sizeof(float);
  size_t kbytes = (size_t)kheads * hd * sizeof(float);
  const void *ptrs[4] = {q, k, qout, kout};
  size_t bytes[4] = {qbytes, kbytes, qbytes, kbytes};
  uintptr_t lo[4], hi[4], position_lo, position_hi;
  if (!graph_once_partial_rope_span(s->position, 2 * sizeof(int),
                                    &position_lo, &position_hi)) return false;
  for (int i = 0; i < 4; ++i) {
    if (!graph_once_partial_rope_span(ptrs[i], bytes[i], &lo[i], &hi[i])) return false;
    auto found = g_live.find((void *)ptrs[i]);
    if (found == g_live.end() || found->second < bytes[i]) return false;
  }
  // Q and K may share read-only storage. Neither output may overlap either
  // input, the other output, or the owner's retained position/length scalars.
  for (int i = 2; i < 4; ++i) {
    if (lo[i] < position_hi && position_lo < hi[i]) return false;
    for (int j = 0; j < 4; ++j)
      if (i != j && lo[i] < hi[j] && lo[j] < hi[i]) return false;
  }
  return true;
}
extern "C" int fak_graph_once_partial_rope(void *owner,
    const float *q, const float *k, float *qout, float *kout,
    int qheads, int kheads, int hd, int rotary, double theta) {
  fak_graph_once *s = (fak_graph_once *)owner;
  if (!graph_once_capture_admit(s)) return 1;
  if (v41_candidates_stream_uncertain() ||
      !graph_once_partial_rope_inputs(s, q, k, qout, kout, qheads, kheads,
                                     hd, rotary, theta)) {
    s->terminal = true; return 1;
  }
  // Do not erase or attribute a preceding CUDA error to this operation.
  if (cudaPeekAtLastError() != cudaSuccess) { s->terminal = true; return 3; }
  unsigned qn = (unsigned)qheads * (unsigned)hd;
  unsigned kn = (unsigned)kheads * (unsigned)hd;
  unsigned qblocks = qn / 256 + (qn % 256 != 0);
  unsigned kblocks = kn / 256 + (kn % 256 != 0);
  FCUDA_TS("k_partial_rope_copy", qblocks, k_partial_rope_copy<true><<<qblocks,256,0,g_stream>>>(qout,q,0,qheads,hd,rotary,theta,s->position));
  if (cudaGetLastError() != cudaSuccess) { s->terminal = true; return 3; }
  FCUDA_TS("k_partial_rope_copy", kblocks, k_partial_rope_copy<true><<<kblocks,256,0,g_stream>>>(kout,k,0,kheads,hd,rotary,theta,s->position));
  if (cudaGetLastError() != cudaSuccess) { s->terminal = true; return 3; }
  return 0; // Recorded only. No completed-token counter or host state advances.
}

extern "C" int fak_graph_once_kv_write(void *owner, float *dst, const float *src, int width) {
  fak_graph_once *s = (fak_graph_once *)owner;
  if (!graph_once_capture_admit(s)) return 1;
  if (!dst || !src || width <= 0 || (size_t)s->capacity > SIZE_MAX / sizeof(float) / (size_t)width) {
    s->terminal = true; return 1;
  }
  FCUDA_TS("k_graph_once_copyrow", (unsigned)width / 256 + (width % 256 != 0), k_graph_once_copyrow<<<(unsigned)width / 256 + (width % 256 != 0), 256, 0, g_stream>>>(dst, src, s->position, width));
  if (cudaGetLastError() != cudaSuccess) { s->terminal = true; return 3; }
  return 0;
}
extern "C" int fak_graph_once_attention(void *owner, const float *q, const float *k,
                                        const float *v, float *out, int heads, int kvHeads,
                                        int hd, float scale) {
  fak_graph_once *s = (fak_graph_once *)owner;
  if (!graph_once_capture_admit(s)) return 1;
  if (!q || !k || !v || !out || heads <= 0 || kvHeads <= 0 || heads % kvHeads ||
      hd <= 0 || hd > FLASH_THREADS * FLASH_ACC_MAX || heads > INT_MAX / hd || !isfinite(scale)) {
    s->terminal = true; return 1;
  }
  size_t shmem = ((size_t)hd + FLASH_THREADS) * sizeof(float);
  FCUDA_TS("k_flash_attention", heads, k_flash_attention<true><<<heads, FLASH_THREADS, shmem, g_stream>>>(q, k, v, out, 0, heads, kvHeads, hd, scale, s->position));
  if (cudaGetLastError() != cudaSuccess) { s->terminal = true; return 3; }
  return 0;
}

static int graph_once_launch(fak_graph_once *s) {
  if (cudaGraphLaunch(s->exec, g_stream) != cudaSuccess) return graph_once_fail(s, 4);
  if (graph_once_drain(s)) return 5;
  return 0;
}
extern "C" int fak_graph_once_end(void *owner) {
  fak_graph_once *s = (fak_graph_once *)owner;
  if (!s || s != g_graph_once_capture || !g_capture_open) return 1;
  // Incomplete/invalid quant recording must close capture but never launch it.
  if (!graph_once_quant_ready(s) || s->quant_cursor != s->quant_count) s->terminal = true;
  g_graph_once_capture = nullptr;
  g_capture_open = false;
  cudaError_t e = cudaStreamEndCapture(g_stream, &s->graph);
  if (e != cudaSuccess || !s->graph || s->terminal || v41_candidates_stream_uncertain()) return graph_once_fail(s, 3);
  if (cudaGraphInstantiate(&s->exec, s->graph, 0) != cudaSuccess) return graph_once_fail(s, 3);
  return graph_once_launch(s);
}
extern "C" int fak_graph_once_replay(void *owner, int pos) {
  fak_graph_once *s = (fak_graph_once *)owner;
  if (!s || s->terminal || !s->exec || g_capture_open || g_graph_once_capture ||
      pos != s->next || pos < 0 || pos >= s->capacity || v41_candidates_stream_uncertain()) return 1;
  if (!graph_once_quant_idle(s)) { s->terminal = true; return 1; }
  int rc = graph_once_set_position(s, pos);
  if (rc) return rc;
  rc = graph_once_launch(s);
  if (!rc) s->next++;
  return rc;
}
extern "C" int fak_graph_once_release(void **owner) {
  if (!owner || !*owner) return 0;
  fak_graph_once *s = (fak_graph_once *)*owner;
  if (g_capture_open || g_graph_once_capture || v41_candidates_stream_uncertain()) return 1;
  // A failed fence retains graph, exec, scalar and ALL caller-owned buffers.
  if (graph_once_drain(s)) return 5;
  if (s->exec) {
    if (cudaGraphExecDestroy(s->exec) != cudaSuccess) { s->terminal = true; return 3; }
    s->exec = nullptr;
  }
  if (s->graph) {
    if (cudaGraphDestroy(s->graph) != cudaSuccess) { s->terminal = true; return 3; }
    s->graph = nullptr;
  }
  int quant_rc = graph_once_quant_retire(s);
  if (quant_rc) return quant_rc;
  if (s->position) {
    if (cudaFree(s->position) != cudaSuccess) { s->terminal = true; return 2; }
    s->position = nullptr;
  }
  free(s);
  *owner = nullptr;
  return 0;
}

// HAL adapter: copy a fresh embedding into the graph-owned stable input before
// replay. No pageable host-copy lifetime and no silent void-copy failure.
extern "C" int fak_graph_once_copy_input(void *owner, void *dst, const void *src, size_t bytes) {
  fak_graph_once *s = (fak_graph_once *)owner;
  if (!s || s->terminal || !s->exec || !dst || !src || !bytes || g_capture_open ||
      v41_candidates_stream_uncertain()) return 1;
  cudaError_t e;
  FCUDA_TS("d2d_copy", dim3(0, 0), e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, g_stream));
  return e == cudaSuccess ? 0 : graph_once_fail(s, 3);
}
extern "C" int fak_graph_once_abort(void *owner) {
  fak_graph_once *s = (fak_graph_once *)owner;
  if (!s || s != g_graph_once_capture || !g_capture_open) return 1;
  s->terminal = true;
  g_graph_once_capture = nullptr;
  g_capture_open = false;
  cudaError_t e = cudaStreamEndCapture(g_stream, &s->graph);
  int drain = graph_once_drain(s);
  return drain ? drain : (e == cudaSuccess ? 0 : 3);
}
