#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

using cudaError_t = int;
using cudaStream_t = void *;
static constexpr int cudaSuccess = 0;
static constexpr int cudaErrorMemoryAllocation = 2;
static constexpr int cudaMemcpyHostToDevice = 1;
static constexpr int cudaMemcpyDeviceToHost = 2;
static constexpr int cudaMemcpyDeviceToDevice = 3;
static cudaStream_t cudaStreamLegacy = reinterpret_cast<void *>(1);
static cudaStream_t g_stream = reinterpret_cast<void *>(99);
static int g_cuda_cleanup_status = 0;
static bool g_v41_router_uncertain = false, g_v41_mmid_uncertain = false, g_v41_bf16_uncertain = false;
static int selected = 0, view = 0, get_error = 0, select_error = 0, op_error = 0;
static int fence_error = 0;
static int last_error = 0; // the runtime's per-thread last error, as the fake records it
static int operations = 0, fences = 0, evictions = 0, untimed = 0;
static size_t g_h2d_bytes = 0, g_host_bytes = 0;
static std::unordered_map<void *, int> owners;
static const void *g_v41_bf16_quarantine[3] = {};
static const void *g_v41_rope_quarantine[5] = {};
static int g_v41_rope_test_free_attempts = 0;
#define FAK_CUDA_V41_BF16_PUBLISH
#define FAK_CUDA_V41_TAIL_ROPE
#define FAK_CUDA_V41_TAIL_ROPE_TEST

[[maybe_unused]] static cudaError_t cudaGetDevice(int *out) {
  if (get_error) return get_error;
  *out = selected;
  return 0;
}
static cudaError_t cudaSetDevice(int device) {
  selected = view = device;
  if (device == 1 && select_error) return select_error;
  return 0;
}
static const char *cudaGetErrorString(cudaError_t) { return "injected runtime error"; }
static cudaError_t cudaGetLastError() {
  int e = last_error;
  last_error = 0;
  return e;
}
static cudaError_t cudaMalloc(void **out, size_t bytes) {
  ++operations;
  if (op_error) return last_error = op_error;
  *out = malloc(bytes);
  owners[*out] = selected;
  return 0;
}
static cudaError_t cudaFree(void *ptr) {
  ++operations;
  assert(owners.at(ptr) == selected);
  if (op_error) return op_error;
  owners.erase(ptr);
  free(ptr);
  return 0;
}
static cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, int kind) {
  ++operations;
  if (kind != cudaMemcpyDeviceToHost) assert(owners.at(dst) == selected);
  if (kind != cudaMemcpyHostToDevice) assert(owners.at(const_cast<void *>(src)) == selected);
  if (op_error) return op_error;
  memcpy(dst, src, n);
  return 0;
}
static cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n, int kind, cudaStream_t stream) {
  // A process-owned foreign stream must never cross this production call.
  assert(stream == cudaStreamLegacy && stream != g_stream);
  return cudaMemcpy(dst, src, n, kind);
}
[[maybe_unused]] static cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  assert(selected == 1 && view == 1 && stream == cudaStreamLegacy);
  ++fences;
  return fence_error;
}
static void q4_coeff_cache_evict(const void *ptr) {
  assert(owners.at(const_cast<void *>(ptr)) == selected);
  ++evictions;
}
static void fcuda_ts_untimed_transfer() { ++untimed; }
#define CK(call) do { (void)(call); } while (0)
#define FAK_CUDA_DEVICE_SCOPE_HOST_TEST
#include "alloc_ledger.h"
#include "runtime_services.cuh"

static void restored() { assert(selected == 0 && view == 0); }

int main() {
  int input = 42, output = 0;
  void *a = fcuda_malloc_on(1, sizeof(input));
  restored();
  assert(a && owners.at(a) == 1);
  void *b = fcuda_malloc_on(1, sizeof(input));
  restored();
  fcuda_h2d_on(1, a, &input, sizeof(input));
  restored();
  assert(g_h2d_bytes == sizeof(input));
  fcuda_d2d_on(1, b, a, sizeof(input));
  restored();
  assert(fences == 1);
  fcuda_d2h_on(1, &output, b, sizeof(output));
  restored();
  assert(output == input && g_host_bytes == sizeof(output) && untimed == 3);

  fence_error = 11;
  fcuda_d2d_on(1, b, a, sizeof(input));
  restored();
  assert(fences == 2);
  fence_error = 0;

  op_error = 7;
  const auto h2d_before = g_h2d_bytes, d2h_before = g_host_bytes;
  assert(fcuda_malloc_on(1, 1) == nullptr);
  restored();
  fcuda_h2d_on(1, a, &input, sizeof(input));
  restored();
  fcuda_d2h_on(1, &output, a, sizeof(output));
  restored();
  fcuda_d2d_on(1, b, a, sizeof(input));
  restored();
  fcuda_free_on(1, b);
  restored();
  assert(owners.count(b) == 1 && fences == 2 && g_h2d_bytes == h2d_before && g_host_bytes == d2h_before);
  op_error = 0;
  last_error = 0;

  // A capacity miss is consumed where it is raised, so the next status reader does not see a
  // stale "out of memory"; any other allocator status stays pending for that reader.
  op_error = cudaErrorMemoryAllocation;
  assert(fcuda_malloc_on(1, 1) == nullptr);
  restored();
  assert(last_error == 0);
  op_error = 7;
  assert(fcuda_malloc_on(1, 1) == nullptr);
  restored();
  assert(last_error == 7);
  last_error = 0;
  op_error = 0;

  const int before_selection = operations;
  select_error = 9; // Even a selector that changes state before failing must unwind.
  assert(fcuda_malloc_on(1, 1) == nullptr);
  restored();
  fcuda_h2d_on(1, a, &input, sizeof(input));
  restored();
  fcuda_d2h_on(1, &output, a, sizeof(output));
  restored();
  fcuda_d2d_on(1, b, a, sizeof(input));
  restored();
  fcuda_free_on(1, b);
  restored();
  assert(operations == before_selection);
  select_error = 0;

  get_error = 5;
  assert(fcuda_malloc_on(1, 1) == nullptr);
  restored();
  assert(operations == before_selection);
  get_error = 0;

  const int before_quarantine = operations, before_evictions = evictions;
  g_v41_bf16_quarantine[0] = a;
  fcuda_free_on(1, a);
  restored();
  g_v41_bf16_quarantine[0] = nullptr;
  g_v41_rope_quarantine[0] = a;
  fcuda_free_on(1, a);
  restored();
  assert(operations == before_quarantine && evictions == before_evictions);
  g_v41_rope_quarantine[0] = nullptr;
  fcuda_free_on(1, a);
  fcuda_free_on(1, b);
  restored();
  assert(owners.empty() && g_v41_rope_test_free_attempts == 3);
  // Every successful device allocation was ledgered and every free decremented it.
  int64_t cells[FAK_ALLOC_CELLS * 5] = {};
  int64_t untracked = -1;
  assert(fcuda_alloc_ledger_read(cells, FAK_ALLOC_CELLS, &untracked) == FAK_ALLOC_CELLS);
  const int other = FAK_ALLOC_OTHER * FAK_ALLOC_KIND_COUNT + FAK_ALLOC_DEVICE_LOCAL;
  assert(cells[other * 5 + 0] == 0 && cells[other * 5 + 2] == 0);
  assert(cells[other * 5 + 3] > 0 && cells[other * 5 + 3] == cells[other * 5 + 4]);
  assert(untracked == 0);
  return 0;
}
