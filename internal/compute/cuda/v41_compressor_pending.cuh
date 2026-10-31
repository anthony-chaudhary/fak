// Existing single CUDA stream, known compressor producers only. Published
// before enqueue. Distinct exact allocation bases stay live until checked Read
// or explicit teardown completion; this is not an enlarged fault quarantine.
#ifdef FAK_CUDA_V41_COMPRESSOR
static const size_t V41C_PENDING_MAX = 4096;
static const size_t V41C_PENDING_BYTES = (size_t)1 << 30;
struct v41c_pending_base { void *pointer; size_t bytes; };
static v41c_pending_base g_v41c_pending[V41C_PENDING_MAX] = {};
static size_t g_v41c_pending_count = 0, g_v41c_pending_bytes = 0;
static uint64_t g_v41c_pending_emits = 0;
static bool g_v41c_pending_completed = false;

static bool v41c_pending_contains(const void *pointer) {
  for (size_t i = 0; i < g_v41c_pending_count; ++i)
    if (g_v41c_pending[i].pointer == pointer) return true;
  return false;
}

extern "C" int fak_v41_compressor_pending_retain(void *pointer, size_t bytes) {
  if (!g_stream || g_capture_open || v41_candidates_stream_uncertain()) return -3;
  auto found = g_live.find(pointer);
  if (!pointer || !bytes || found == g_live.end() || found->second < bytes) return -2;
  // Go validates device ownership too; retain the complete native allocation,
  // not a smaller forged view that would understate the byte bound.
  if (v41c_pending_contains(pointer)) { g_v41c_pending_completed = false; return 0; }
  if (g_v41c_pending_count == V41C_PENDING_MAX || found->second > V41C_PENDING_BYTES-g_v41c_pending_bytes) return -2;
  g_v41c_pending[g_v41c_pending_count++] = {pointer, found->second};
  g_v41c_pending_bytes += found->second;
  g_v41c_pending_completed = false;
  return 0;
}

extern "C" void fak_v41_compressor_pending_fail(void) {
  g_v41_compressor_uncertain = true;
  g_v41c_pending_completed = false;
}

// Success proves completion but does not release ownership: a subsequent
// checked readback failure must still retain all producer/ring/output bases.
extern "C" int fak_v41_compressor_pending_finish(void) {
  if (v41_candidates_stream_uncertain() || !g_stream || g_capture_open) return -3;
  if (!g_v41c_pending_count) return 0;
  const cudaError_t status = cudaStreamSynchronize(g_stream);
  if (status != cudaSuccess) {
    fak_v41_compressor_pending_fail();
    return 50000 + (int)status;
  }
  g_v41c_pending_completed = true;
  return 0;
}

extern "C" int fak_v41_compressor_pending_commit(void) {
  if (v41_candidates_stream_uncertain()) return -3;
  if (g_v41c_pending_count && !g_v41c_pending_completed) return -3;
  for (size_t i=0; i<g_v41c_pending_count; ++i) g_v41c_pending[i] = {};
  g_v41c_pending_count = g_v41c_pending_bytes = 0;
  g_v41_compressor_completed += g_v41c_pending_emits;
  g_v41c_pending_emits = 0;
  g_v41c_pending_completed = false;
  return 0;
}
#else
static bool v41c_pending_contains(const void *) { return false; }
#endif
