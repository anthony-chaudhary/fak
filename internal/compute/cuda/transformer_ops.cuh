// ---- RMSNorm: one block per row -------------------------------------------------
__global__ void k_rmsnorm(const float *X, const float *W, float *Y, int rows, int n, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *x = X + (size_t)r * n;
  float *y = Y + (size_t)r * n;
  __shared__ float red[256];
  float local = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) local += x[i] * x[i];
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * W[i];
}
extern "C" void fcuda_rmsnorm_f32(const float *dX, const float *dW, float *dY, int rows, int n, float eps) {
  FCUDA_TS("k_rmsnorm", rows, k_rmsnorm<<<rows, 256, 0, g_stream>>>(dX, dW, dY, rows, n, eps));
}

// ---- RMSNorm: warp-per-row (fitting widths <= 1024) -----------------------------
__global__ void k_rmsnorm_warp_per_row(const float *X, const float *W, float *Y, int rows, int n, float eps) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x % 32;
  if (warp_id >= rows) return;
  const float *x = X + (size_t)warp_id * n;
  float *y = Y + (size_t)warp_id * n;
  float local = 0.f;
  for (int i = lane; i < n; i += 32) local += x[i] * x[i];
  for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(0xffffffff, local, offset);
  }
  float mean = __shfl_sync(0xffffffff, local, 0) / (float)n;
  float inv = rsqrtf(mean + eps);
  for (int i = lane; i < n; i += 32) y[i] = x[i] * inv * W[i];
}

extern "C" void fcuda_rmsnorm_dispatched_f32(const float *dX, const float *dW, float *dY, int rows, int n, float eps) {
  if (n <= 1024) {
    int warps = 8;
    int blocks = (rows + warps - 1) / warps;
    FCUDA_TS("k_rmsnorm_warp_per_row", blocks, k_rmsnorm_warp_per_row<<<blocks, warps * 32, 0, g_stream>>>(dX, dW, dY, rows, n, eps));
  } else {
    FCUDA_TS("k_rmsnorm", rows, k_rmsnorm<<<rows, 256, 0, g_stream>>>(dX, dW, dY, rows, n, eps));
  }
}

// ---- Fused RMSNorm + Residual Add ----------------------------------------------
__global__ void k_rmsnorm_fused_residual_add(
    const float *X, const float *ResidualIn, const float *W,
    float *ResidualOut, float *Y, int rows, int n, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *x = X + (size_t)r * n;
  const float *rin = ResidualIn + (size_t)r * n;
  float *rout = ResidualOut ? (ResidualOut + (size_t)r * n) : nullptr;
  float *y = Y + (size_t)r * n;
  __shared__ float red[256];
  float local = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float val = x[i] + rin[i];
    if (rout) rout[i] = val;
    local += val * val;
  }
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float val = rout ? rout[i] : (x[i] + rin[i]);
    y[i] = val * inv * W[i];
  }
}
extern "C" void fcuda_rmsnorm_fused_residual_add_f32(
    const float *dX, const float *dResidualIn, const float *dW,
    float *dResidualOut, float *dY, int rows, int n, float eps) {
  FCUDA_TS("k_rmsnorm_fused_residual_add", rows, k_rmsnorm_fused_residual_add<<<rows, 256, 0, g_stream>>>(
      dX, dResidualIn, dW, dResidualOut, dY, rows, n, eps));
}

// ---- RoPE (HF non-interleaved rotate_half) at one absolute position -------------
template <bool device_position = false>
__global__ void k_rope(float *X, int pos, int nHeads, int headDim, double theta, const int *position = nullptr) {
  if (device_position) pos = position[0];
  int half = headDim / 2;
  int t = blockIdx.x * blockDim.x + threadIdx.x; // one (head, j) pair
  if (t >= nHeads * half) return;
  int h = t / half, j = t % half;
  double inv = 1.0 / pow(theta, (double)(2 * j) / (double)headDim);
  double a = (double)pos * inv;
  float c = (float)cos(a), s = (float)sin(a);
  float *hv = X + (size_t)h * headDim;
  float x0 = hv[j], x1 = hv[j + half];
  hv[j]        = x0 * c - x1 * s;
  hv[j + half] = x1 * c + x0 * s;
}
extern "C" void fcuda_rope_f32(float *dX, int pos, int nHeads, int headDim, double theta) {
  int total = nHeads * (headDim / 2);
  FCUDA_TS("k_rope", (total + 127) / 128, k_rope<false><<<(total + 127) / 128, 128, 0, g_stream>>>(dX, pos, nHeads, headDim, theta));
}

// Prompt-panel RoPE for the dense HAL prefill: row t of X [tokens, nHeads*headDim] is
// rotated at start+t into Y with k_rope's exact per-pair arithmetic.
__global__ void k_rope_panel(const float *X, float *Y, int tokens, int start,
                             int nHeads, int headDim, double theta) {
  int half = headDim / 2;
  size_t perRow = (size_t)nHeads * half;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; // one (row, head, j) pair
  if (t >= (size_t)tokens * perRow) return;
  int row = (int)(t / perRow);
  int within = (int)(t - (size_t)row * perRow);
  int h = within / half, j = within % half;
  double inv = 1.0 / pow(theta, (double)(2 * j) / (double)headDim);
  double a = (double)(start + row) * inv;
  float c = (float)cos(a), s = (float)sin(a);
  size_t base = (size_t)row * nHeads * headDim + (size_t)h * headDim;
  float x0 = X[base + j], x1 = X[base + j + half];
  Y[base + j]        = x0 * c - x1 * s;
  Y[base + j + half] = x1 * c + x0 * s;
}
extern "C" int fcuda_rope_panel_f32(const float *dX, float *dY, int tokens, int start,
                                    int nHeads, int headDim, double theta) {
  if (!dX || !dY || tokens <= 0 || start < 0 || nHeads <= 0 || headDim <= 0 ||
      (headDim & 1) != 0 || !(theta > 0.0) || start > INT_MAX - tokens) return -1;
  size_t total = (size_t)tokens * nHeads * (headDim / 2);
  cudaGetLastError();
  FCUDA_TS("k_rope_panel", (unsigned)((total + 127) / 128), k_rope_panel<<<(unsigned)((total + 127) / 128), 128, 0, g_stream>>>(
      dX, dY, tokens, start, nHeads, headDim, theta));
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 70000 + (int)launch;
}


__global__ void k_partial_rope_copy(float *out, const float *in, int pos,
                                    int nHeads, int headDim, int rotaryDim,
                                    double theta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = nHeads * headDim;
  if (i >= n) return;
  int d = i % headDim;
  int h = i / headDim;
  if (d >= rotaryDim) { out[i] = in[i]; return; }
  int half = rotaryDim / 2;
  int j = d < half ? d : d - half;
  double freq = pow(theta, -(2.0 * j) / rotaryDim);
  float cs = cos((double)pos * freq), sn = sin((double)pos * freq);
  float a = in[h * headDim + j], b = in[h * headDim + j + half];
  out[i] = d < half ? a * cs - b * sn : b * cs + a * sn;
}
extern "C" void fcuda_partial_rope_qk_f32(const float *dQ, const float *dK,
                                             float *dQOut, float *dKOut, int pos,
                                             int nQHeads, int nKHeads, int headDim,
                                             int rotaryDim, double theta) {
  int qn = nQHeads * headDim, kn = nKHeads * headDim;
  FCUDA_TS("k_partial_rope_copy", (qn+255)/256, k_partial_rope_copy<<<(qn+255)/256,256,0,g_stream>>>(dQOut,dQ,pos,nQHeads,headDim,rotaryDim,theta));
  CK(cudaGetLastError());
  FCUDA_TS("k_partial_rope_copy", (kn+255)/256, k_partial_rope_copy<<<(kn+255)/256,256,0,g_stream>>>(dKOut,dK,pos,nKHeads,headDim,rotaryDim,theta));
  CK(cudaGetLastError());
}
__global__ void k_sigmoid_mul(float *x, const float *gate, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}
extern "C" void fcuda_sigmoid_mul_f32(float *dX, const float *dGate, int n) {
  FCUDA_TS("k_sigmoid_mul", (n+255)/256, k_sigmoid_mul<<<(n+255)/256,256,0,g_stream>>>(dX,dGate,n));
  CK(cudaGetLastError());
}
__global__ void k_split_qwen35_qg(const float *qg, float *q, float *gate,
                                    int nHeads, int headDim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = nHeads * headDim;
  if (i >= n) return;
  int h = i / headDim, d = i % headDim;
  int src = h * 2 * headDim + d;
  q[i] = qg[src];
  gate[i] = qg[src + headDim];
}
extern "C" void fcuda_split_qwen35_qg_f32(const float *dQG, float *dQ, float *dGate,
                                             int nHeads, int headDim) {
  int n = nHeads * headDim;
  FCUDA_TS("k_split_qwen35_qg", (n+255)/256, k_split_qwen35_qg<<<(n+255)/256,256,0,g_stream>>>(dQG,dQ,dGate,nHeads,headDim));
  CK(cudaGetLastError());
}

__global__ void k_qwen35_split_qg_panel(const float *qg, float *q, float *gate,
                                         int tokens, int nHeads, int headDim) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t width = (size_t)nHeads * headDim;
  size_t n = (size_t)tokens * width;
  if (i >= n) return;
  int token = (int)(i / width);
  int within = (int)(i - (size_t)token * width);
  int head = within / headDim;
  int dim = within - head * headDim;
  size_t src = (size_t)token * 2 * width + (size_t)head * 2 * headDim + dim;
  q[i] = qg[src];
  gate[i] = qg[src + headDim];
}

extern "C" int fak_qwen35_split_qg_panel_f32(
    const float *dQG, float *dQ, float *dGate,
    int tokens, int nHeads, int headDim) {
  if (!dQG || !dQ || !dGate || tokens <= 0 || nHeads <= 0 || headDim <= 0) return -1;
  cudaGetLastError();
  size_t n = (size_t)tokens * nHeads * headDim;
  FCUDA_TS("k_qwen35_split_qg_panel", (n + 255) / 256, k_qwen35_split_qg_panel<<<(n + 255) / 256, 256, 0, g_stream>>>(
      dQG, dQ, dGate, tokens, nHeads, headDim));
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 30000 + (int)launch;
}

__global__ void k_qwen35_partial_rope_panel(
    float *out, const float *in, int tokens, int startPos,
    int nHeads, int headDim, int rotaryDim, double theta) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t width = (size_t)nHeads * headDim;
  size_t n = (size_t)tokens * width;
  if (i >= n) return;
  int token = (int)(i / width);
  int within = (int)(i - (size_t)token * width);
  int head = within / headDim;
  int dim = within - head * headDim;
  if (dim >= rotaryDim) { out[i] = in[i]; return; }
  int half = rotaryDim / 2;
  int j = dim < half ? dim : dim - half;
  double freq = pow(theta, -(2.0 * j) / rotaryDim);
  double angle = (double)(startPos + token) * freq;
  float cs = (float)cos(angle), sn = (float)sin(angle);
  size_t base = (size_t)token * width + (size_t)head * headDim;
  float a = in[base + j], b = in[base + j + half];
  out[i] = dim < half ? a * cs - b * sn : b * cs + a * sn;
}

extern "C" int fak_qwen35_partial_rope_panel_f32(
    const float *dQ, const float *dK, float *dQOut, float *dKOut,
    int tokens, int startPos, int nQHeads, int nKHeads, int headDim,
    int rotaryDim, double theta) {
  if (!dQ || !dK || !dQOut || !dKOut || tokens <= 0 || startPos < 0 ||
      nQHeads <= 0 || nKHeads <= 0 || headDim <= 0 || rotaryDim <= 0 ||
      rotaryDim > headDim || (rotaryDim & 1) != 0 || !(theta > 0.0)) return -1;
  cudaGetLastError();
  size_t qn = (size_t)tokens * nQHeads * headDim;
  size_t kn = (size_t)tokens * nKHeads * headDim;
  FCUDA_TS("k_qwen35_partial_rope_panel", (qn + 255) / 256, k_qwen35_partial_rope_panel<<<(qn + 255) / 256, 256, 0, g_stream>>>(
      dQOut, dQ, tokens, startPos, nQHeads, headDim, rotaryDim, theta));
  cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) return 40000 + (int)launch;
  FCUDA_TS("k_qwen35_partial_rope_panel", (kn + 255) / 256, k_qwen35_partial_rope_panel<<<(kn + 255) / 256, 256, 0, g_stream>>>(
      dKOut, dK, tokens, startPos, nKHeads, headDim, rotaryDim, theta));
  launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 41000 + (int)launch;
}

// ---- SwiGLU / residual add / bias add -------------------------------------------
__global__ void k_swiglu(const float *G, const float *U, float *Y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = G[i];
  Y[i] = (g / (1.f + expf(-g))) * U[i];
}
extern "C" void fcuda_swiglu_f32(const float *dG, const float *dU, float *dY, int n) {
  FCUDA_TS("k_swiglu", (n + 255) / 256, k_swiglu<<<(n + 255) / 256, 256, 0, g_stream>>>(dG, dU, dY, n));
}
__global__ void k_add(float *D, const float *S, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) D[i] += S[i];
}
extern "C" void fcuda_add_f32(float *dDst, const float *dSrc, int n) {
  FCUDA_TS("k_add", (n + 255) / 256, k_add<<<(n + 255) / 256, 256, 0, g_stream>>>(dDst, dSrc, n));
}
__global__ void k_add_bias(float *D, const float *B, int rows, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width) return;
  D[i] += B[i % width];
}
extern "C" void fcuda_add_bias_f32(float *dDst, const float *dBias, int rows, int width) {
  FCUDA_TS("k_add_bias", (rows * width + 255) / 256, k_add_bias<<<(rows * width + 255) / 256, 256, 0, g_stream>>>(dDst, dBias, rows, width));
}

// k_copyrow appends a KV row into a fixed-base buffer at a SCALAR float offset. It replaces
// the AppendKV cudaMemcpyAsync, whose destination pointer grew every token — a moving
// pointer that cudaGraphExecUpdate cannot patch, so a captured decode graph could not be
// reused (it re-instantiated each token). A kernel with the offset as a scalar arg IS
// ExecUpdate-patchable, so one captured graph now serves the whole growing cache.
__global__ void k_copyrow(float *dstBase, const float *src, int offset, int n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) dstBase[(size_t)offset + i] = src[i];
}
extern "C" void fcuda_kv_write(float *dstBase, const float *src, int offset, int n) {
  if (n <= 0) return;
  unsigned blocks = ((unsigned)n + 255u) / 256u;
  FCUDA_TS("k_copyrow", blocks, k_copyrow<<<blocks, 256, 0, g_stream>>>(dstBase, src, offset, n));
}
extern "C" int fak_qwen35_kv_write_f32(
    float *dstBase, const float *src, int offset, int n) {
  if (!dstBase || !src || offset < 0 || n <= 0 || offset > INT_MAX - n) return -1;
  cudaError_t prior = cudaGetLastError();
  if (prior != cudaSuccess) return 51000 + (int)prior;
  unsigned blocks = ((unsigned)n + 255u) / 256u;
  FCUDA_TS("k_copyrow", blocks, k_copyrow<<<blocks, 256, 0, g_stream>>>(dstBase, src, offset, n));
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 52000 + (int)launch;
}
