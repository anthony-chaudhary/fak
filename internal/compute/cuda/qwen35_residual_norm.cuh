// Checked ordinary-sequence adapter for the EXISTING in-tree Apache-2.0
// k_rmsnorm_fused_residual_add / fcuda_rmsnorm_fused_residual_add_f32 at
// No arithmetic kernel is added or re-credited here. Included after that file.
#include <stdint.h>

static unsigned long long g_qwen35_residual_norm_launches = 0;

// Device 0 launch limits are fixed for the process: resolved once, so admission
// stays off the driver. 0 unresolved, -1 unsupported, else the row grid limit.
// Unsynchronized: every caller holds the backend lock.
static int g_qwen35_residual_norm_max_rows = 0;

// Admission refuses an open capture: every caller allocates Y from the pool and
// fences the stream, and neither is capture-stable.
extern "C" int fak_qwen35_residual_norm_capability(int rows, int n) {
  if (rows <= 0 || n <= 0 || rows > INT_MAX / n || g_capture_open) return 0;
  int device = -1;
  cudaError_t e = cudaGetDevice(&device);
  if (e != cudaSuccess) return -(int)e;
  if (device != 0) return 0;
  cudaStreamCaptureStatus capture;
  e = cudaStreamIsCapturing(g_stream, &capture);
  if (e != cudaSuccess) return -(int)e;
  if (capture != cudaStreamCaptureStatusNone) return 0;
  if (g_qwen35_residual_norm_max_rows == 0) {
    cudaDeviceProp prop;
    e = cudaGetDeviceProperties(&prop, 0);
    if (e != cudaSuccess) return -(int)e;
    const bool fits = prop.maxThreadsPerBlock >= 256 && prop.maxThreadsDim[0] >= 256 &&
        prop.sharedMemPerBlock >= 256 * sizeof(float) && prop.maxGridSize[0] > 0;
    g_qwen35_residual_norm_max_rows = fits ? prop.maxGridSize[0] : -1;
  }
  return rows <= g_qwen35_residual_norm_max_rows;
}

// X is intentionally the existing kernel's ResidualOut. Its addend and norm
// weights must remain independent; fresh Y may alias none of the inputs.
static bool qwen35_residual_norm_aliased(uintptr_t x, uintptr_t delta, uintptr_t w, uintptr_t y,
                                         size_t panelBytes, size_t weightBytes) {
  const uintptr_t ptrs[4] = {x, delta, w, y};
  const size_t bytes[4] = {panelBytes, panelBytes, weightBytes, panelBytes};
  for (int a = 0; a < 4; ++a) {
    for (int b = a + 1; b < 4; ++b) {
      if (a == 1 && b == 2) continue; // Delta and W are both read-only.
      if (ptrs[a] < ptrs[b] + bytes[b] && ptrs[b] < ptrs[a] + bytes[a]) return true;
    }
  }
  return false;
}

static bool qwen35_residual_norm_extent(const void *p, size_t bytes) {
  if (!p || bytes == 0 || (uintptr_t)p > UINTPTR_MAX - bytes) return false;
  auto it = g_live.find((void *)p);
  if (it != g_live.end()) return it->second >= bytes;
  auto panel = g_panel_live.find((void *)p);
  return panel != g_panel_live.end() && panel->second >= bytes;
}

extern "C" int fak_qwen35_residual_norm_f32(
    float *X, const float *Delta, const float *W, float *Y,
    int rows, int n, float eps) {
  if (rows <= 0 || n <= 0 || rows > INT_MAX / n || !(eps > 0.f) || !isfinite(eps)) return -1;
  const int supported = fak_qwen35_residual_norm_capability(rows, n);
  if (supported != 1) return supported < 0 ? 88000 - supported : -2;
  const size_t panelBytes = (size_t)rows * n * sizeof(float);
  const void *ptrs[4] = {X, Delta, W, Y};
  const size_t bytes[4] = {panelBytes, panelBytes, (size_t)n * sizeof(float), panelBytes};
  for (int i = 0; i < 4; ++i) {
    if (!qwen35_residual_norm_extent(ptrs[i], bytes[i])) return -3;
    cudaPointerAttributes attrs;
    const cudaError_t e = cudaPointerGetAttributes(&attrs, ptrs[i]);
    if (e != cudaSuccess) return 89000 + (int)e;
    if (attrs.type != cudaMemoryTypeDevice || attrs.device != 0) return -5;
  }
  if (qwen35_residual_norm_aliased((uintptr_t)X, (uintptr_t)Delta, (uintptr_t)W, (uintptr_t)Y,
                                   panelBytes, bytes[2])) return -4;
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return 90000 + (int)e;
  // Existing kernel materializes float val=x+delta in X, accumulates that same
  // F32 value with the original local sum/tree, then rereads X for normalized Y.
  // Its per-thread indices are disjoint, so this intentional in-place residual
  // does not race; the reduction barrier separates the two passes.
  fcuda_rmsnorm_fused_residual_add_f32(X, Delta, W, X, Y, rows, n, eps);
  e = cudaGetLastError();
  if (e != cudaSuccess) return 91000 + (int)e;
  ++g_qwen35_residual_norm_launches; // Enqueue only; sequence fence completes it.
  return 0;
}

extern "C" unsigned long long fak_qwen35_residual_norm_launch_count(void) {
  return g_qwen35_residual_norm_launches;
}
