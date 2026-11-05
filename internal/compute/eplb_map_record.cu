// On-device EPLB logical-to-physical expert remap + per-physical hit-count
// recording. Included by cuda_kernels.cu (one-line #include, build wiring owned
// by the successor policy-hook packet) so the kernel shares g_stream, the pooled
// allocator and the transfer witnesses of the single libfakcuda object.
//
// ADAPT of vllm-project/vllm
// vllm/model_executor/layers/fused_moe/router/base_router.py at pin
// ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0): the fused Triton
// _eplb_map_and_record_i32_kernel, re-expressed as a CUDA grid-stride kernel for
// fak's flat C ABI. Divergences from upstream, all deliberate:
//   - record_enabled arrives as a plain int32 argument instead of a pointer load
//     (single scalar, identical semantics).
//   - HAS_NUM_UNPADDED becomes runtime: num_unpadded_tokens == -1 disables the
//     unpadded gate (values below -1 refuse with status -1), matching the Go
//     oracle's NumUnpaddedTokens sentinel.
//   - Out-of-range logical ids keep upstream's graceful -1 fallback on device
//     (a bad id mid-batch must not abort the forward pass); the strict typed
//     refusal lives in the host Go API (ErrEplbLogicalIDOutOfRange).
//
// Contract (mirrored by the Go oracle EplbMapRecord in eplb_map_record.go):
//   - Input: flat int32 topk logical-expert panel [num_tokens][num_active_experts].
//   - Entry i maps to logical_to_physical[logical*map_slots + replica_idx] where
//     replica_idx = uint32(uint64(i / num_active_experts) * 2654435769) %
//     max(logical_replica_count[logical], 1); invalid logical ids store -1.
//   - Hit counts atomicAdd one per entry, gated by record_enabled, the unpadded
//     window (i < num_unpadded_tokens * num_active_experts) and
//     0 <= physical_id < out_size; out-of-range map values store raw but never
//     record.
// Statuses: 0 ok, -1 null pointer or refused geometry, -2 graph capture open,
// -3 device allocation failure, positive cudaError_t for copy/launch/execution.

#include <limits.h>

#define EMR_THREADS 256
#define EMR_MAX_BLOCKS 1024

// One thread per panel entry, grid-strided: remap the logical id to its physical
// replica via the Knuth hash of the token index, store it, and record the hit.
__global__ void k_emr_map_record_i32(
    const int32_t *__restrict__ topk_ids,
    const int32_t *__restrict__ logical_replica_count,
    const int32_t *__restrict__ logical_to_physical,
    int32_t *__restrict__ out_ids,
    int32_t *__restrict__ out_hits,
    int32_t numel,
    int32_t num_active_experts,
    int32_t num_logical_experts,
    int32_t map_slots,
    int32_t out_size,
    int32_t record_enabled,
    int32_t num_unpadded_tokens) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t off = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; off < numel; off += stride) {
    const int32_t expert_id = __ldg(topk_ids + off);
    const bool valid_expert = (expert_id >= 0) && (expert_id < num_logical_experts);
    const int32_t safe_expert_id = valid_expert ? expert_id : 0;

    int32_t replica_count = 1;
    if (valid_expert) replica_count = __ldg(logical_replica_count + safe_expert_id);
    if (replica_count < 1) replica_count = 1;
    const int64_t token_idx = off / num_active_experts;
    const uint32_t hashed = (uint32_t)((uint64_t)token_idx * 2654435769ULL);
    const int32_t replica_idx = (int32_t)(hashed % (uint32_t)replica_count);
    const int64_t map_index = (int64_t)safe_expert_id * map_slots + replica_idx;
    const int32_t physical_id = valid_expert ? __ldg(logical_to_physical + map_index) : -1;
    out_ids[off] = physical_id;

    if (!record_enabled) continue;
    if (num_unpadded_tokens >= 0 && off >= (int64_t)num_unpadded_tokens * num_active_experts) continue;
    if (physical_id >= 0 && physical_id < out_size) atomicAdd(out_hits + physical_id, 1);
  }
}

// emr_geom is the validated byte geometry of one map-and-record call.
struct emr_geom {
  size_t ids, replicas, map, out_ids, out_hits;
};

// emr_geometry validates shape and sizes every operand; false means status -1.
static bool emr_geometry(int64_t numel, int32_t num_active_experts, int32_t num_logical_experts,
                         int32_t map_slots, int32_t out_size, int32_t num_unpadded_tokens,
                         emr_geom *g) {
  if (numel < 0 || numel > INT32_MAX) return false;
  if (num_active_experts <= 0 || num_logical_experts <= 0 || map_slots <= 0 || out_size <= 0) return false;
  if (num_unpadded_tokens < -1) return false;
  const int64_t map_len = (int64_t)num_logical_experts * map_slots;
  if (map_len < 0 || map_len > INT32_MAX) return false;
  g->ids = (size_t)numel * sizeof(int32_t);
  g->replicas = (size_t)num_logical_experts * sizeof(int32_t);
  g->map = (size_t)map_len * sizeof(int32_t);
  g->out_ids = g->ids;
  g->out_hits = (size_t)out_size * sizeof(int32_t);
  return true;
}

static size_t g_emr_launches = 0;

// One synchronous map-and-record: upload the three read operands, launch, fence,
// read both outputs back. Host buffers must carry exactly the documented sizes
// (validated against the recomputed geometry). Statuses as the file header.
extern "C" int fcuda_eplb_map_record(
    const int32_t *topk_ids, size_t topk_ids_bytes,
    const int32_t *logical_replica_count, size_t logical_replica_count_bytes,
    const int32_t *logical_to_physical, size_t logical_to_physical_bytes,
    int32_t num_active_experts, int32_t num_logical_experts,
    int32_t map_slots, int32_t out_size,
    int32_t record_enabled, int32_t num_unpadded_tokens,
    int32_t *out_ids, size_t out_ids_bytes,
    int32_t *out_hits, size_t out_hits_bytes) {
  if (!topk_ids || !logical_replica_count || !logical_to_physical || !out_ids || !out_hits) return -1;
  if (topk_ids_bytes % sizeof(int32_t) != 0) return -1;
  emr_geom g;
  if (!emr_geometry((int64_t)(topk_ids_bytes / sizeof(int32_t)), num_active_experts,
                    num_logical_experts, map_slots, out_size, num_unpadded_tokens, &g)) return -1;
  if (logical_replica_count_bytes != g.replicas || logical_to_physical_bytes != g.map) return -1;
  if (out_ids_bytes != g.out_ids || out_hits_bytes != g.out_hits) return -1;
  if (g_capture_open) return -2;
  const int32_t numel = (int32_t)(topk_ids_bytes / sizeof(int32_t));
  void *dIds = fcuda_malloc(g.ids);
  void *dReplicas = fcuda_malloc(g.replicas);
  void *dMap = fcuda_malloc(g.map);
  void *dOutIDs = fcuda_malloc(g.out_ids);
  void *dOutHits = fcuda_malloc(g.out_hits);
  int status = 0;
  cudaError_t e = cudaSuccess;
  if (!dIds || !dReplicas || !dMap || !dOutIDs || !dOutHits) {
    fcuda_free(dIds); fcuda_free(dReplicas); fcuda_free(dMap);
    fcuda_free(dOutIDs); fcuda_free(dOutHits);
    return -3;
  }
  (void)cudaGetLastError(); // discard stale status before attributing this call
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dIds, topk_ids, g.ids, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += g.ids;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dReplicas, logical_replica_count, g.replicas, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += g.replicas;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dMap, logical_to_physical, g.map, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += g.map;
  if (numel > 0) {
    unsigned int grid = (unsigned int)((numel + EMR_THREADS - 1) / EMR_THREADS);
    if (grid > EMR_MAX_BLOCKS) grid = EMR_MAX_BLOCKS;
    FCUDA_TS("k_emr_map_record_i32", grid,
             k_emr_map_record_i32<<<grid, EMR_THREADS, 0, g_stream>>>(
                 (const int32_t *)dIds, (const int32_t *)dReplicas, (const int32_t *)dMap,
                 (int32_t *)dOutIDs, (int32_t *)dOutHits, numel,
                 num_active_experts, num_logical_experts, map_slots, out_size,
                 record_enabled, num_unpadded_tokens));
    if ((e = cudaGetLastError()) != cudaSuccess) { status = (int)e; goto done; }
  }
  if ((e = cudaStreamSynchronize(g_stream)) != cudaSuccess) { status = (int)e; goto done; }
  ++g_emr_launches;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(out_ids, dOutIDs, g.out_ids, cudaMemcpyDeviceToHost)) != cudaSuccess) { status = (int)e; goto done; }
  g_host_bytes += g.out_ids;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(out_hits, dOutHits, g.out_hits, cudaMemcpyDeviceToHost)) != cudaSuccess) { status = (int)e; goto done; }
  g_host_bytes += g.out_hits;
done:
  fcuda_free(dIds);
  fcuda_free(dReplicas);
  fcuda_free(dMap);
  fcuda_free(dOutIDs);
  fcuda_free(dOutHits);
  return status;
}

// Witness: cumulative successful operation completions.
extern "C" size_t fcuda_eplb_map_record_launches(void) {
  return g_emr_launches;
}
