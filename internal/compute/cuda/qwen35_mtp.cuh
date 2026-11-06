// CUDA F32 MTP pre-layer: adaptation of the existing in-tree Vulkan composition.
// Original k_rmsnorm arithmetic and SGEMM mapping remain unchanged. No donor
// source is added. Included after transformer_ops.cuh in the shared CUDA TU.
#include <stdint.h>
#include <limits.h>

// BEGIN checked MTP native core.
static int mtp_prelayer_fail(int status) {
  if (status != 0 && g_cuda_cleanup_status == 0) g_cuda_cleanup_status = status;
  return g_cuda_cleanup_status != 0 ? g_cuda_cleanup_status : status;
}
extern "C" void fcuda_qwen35_mtp_stop(void) {
  (void)mtp_prelayer_fail(95001);
}
static int mtp_prelayer_cuda(cudaError_t error) {
  return error == cudaSuccess ? 0 : mtp_prelayer_fail(92000 + (int)error);
}
static int mtp_prelayer_status(void) {
  if (v41_candidates_stream_uncertain())
    return mtp_prelayer_fail(g_cuda_cleanup_status != 0 ? g_cuda_cleanup_status : 95002);
  return mtp_prelayer_cuda(cudaPeekAtLastError());
}
static bool mtp_prelayer_geometry(int h, float eps) {
  // Bound the original signed RMS loop and FC products, on 32-bit hosts too.
  return h > 0 && h <= 5120 && eps > 0.f && isfinite(eps) &&
      (size_t)h <= SIZE_MAX / ((size_t)2 * h * sizeof(float));
}
static bool mtp_prelayer_extent(const void *p, size_t bytes) {
  if (!p || bytes == 0 || (uintptr_t)p > UINTPTR_MAX - bytes) return false;
  auto found = g_live.find(const_cast<void *>(p));
  return found != g_live.end() && found->second >= bytes;
}
static bool mtp_prelayer_overlap(const void *a, size_t na, const void *b, size_t nb) {
  return (uintptr_t)a < (uintptr_t)b + nb && (uintptr_t)b < (uintptr_t)a + na;
}
static int mtp_prelayer_state(void) {
  if (v41_candidates_stream_uncertain())
    return mtp_prelayer_fail(g_cuda_cleanup_status != 0 ? g_cuda_cleanup_status : 95002);
  // FCUDA_TS can start work after a profiler failure. Refuse immutable profile-on
  // before any allocation or producer instead of pretending a postcheck prevents it.
  if (!g_stream || !g_blas || g_capture_open || g_pool_graph_deep || g_ts_on) return -1;
  int status = mtp_prelayer_status();
  if (status != 0) return status;
  int device = -1;
  status = mtp_prelayer_cuda(cudaGetDevice(&device));
  if (status != 0) return status;
  if (device != 0) return -1;
  cudaStreamCaptureStatus capture;
  status = mtp_prelayer_cuda(cudaStreamIsCapturing(g_stream, &capture));
  if (status != 0) return status;
  if (capture != cudaStreamCaptureStatusNone) return -1;
  cudaDeviceProp prop;
  status = mtp_prelayer_cuda(cudaGetDeviceProperties(&prop, device));
  if (status != 0) return status;
  if (prop.maxThreadsPerBlock < 256 || prop.maxThreadsDim[0] < 256 ||
      prop.sharedMemPerBlock < 256 * sizeof(float) || prop.maxGridSize[0] < 1) return -1;
  cudaStream_t stream;
  cublasStatus_t blas = cublasGetStream(g_blas, &stream);
  if (blas != CUBLAS_STATUS_SUCCESS) return mtp_prelayer_fail(93000 + (int)blas);
  cublasPointerMode_t mode;
  blas = cublasGetPointerMode(g_blas, &mode);
  if (blas != CUBLAS_STATUS_SUCCESS) return mtp_prelayer_fail(93000 + (int)blas);
  if (stream != g_stream || mode != CUBLAS_POINTER_MODE_HOST) return -1;
  return 0;
}
static int mtp_prelayer_device(const void *p, size_t bytes) {
  if (!mtp_prelayer_extent(p, bytes)) return -2;
  cudaPointerAttributes attr;
  int status = mtp_prelayer_cuda(cudaPointerGetAttributes(&attr, p));
  if (status != 0) return status;
  return attr.type == cudaMemoryTypeDevice && attr.device == 0 ? 0 : -2;
}
extern "C" int fcuda_qwen35_mtp_admit(
    const float *prior, const float *embedding, const float *hidden_norm,
    const float *embedding_norm, const float *fc, int h, float eps) {
  if (!mtp_prelayer_geometry(h, eps)) return -2;
  const void *inputs[5] = {prior, embedding, hidden_norm, embedding_norm, fc};
  const size_t vector_bytes = (size_t)h * sizeof(float);
  // Host-only extent refusal precedes driver queries.
  for (int i = 0; i < 5; ++i)
    if (!mtp_prelayer_extent(inputs[i], i == 4 ? 2 * h * vector_bytes : vector_bytes)) return -2;
  int status = mtp_prelayer_state();
  if (status != 0) return status;
  for (int i = 0; i < 5; ++i) {
    status = mtp_prelayer_device(inputs[i], i == 4 ? 2 * h * vector_bytes : vector_bytes);
    if (status != 0) return status;
  }
  return 0;
}
extern "C" int fcuda_qwen35_mtp_f32(
    const float *prior, const float *embedding, const float *hidden_norm,
    const float *embedding_norm, const float *fc, float *joined, float *out,
    int h, float eps) {
  int status = fcuda_qwen35_mtp_admit(prior, embedding, hidden_norm, embedding_norm, fc, h, eps);
  if (status != 0) return status;
  const size_t vb = (size_t)h * sizeof(float);
  status = mtp_prelayer_device(joined, 2 * vb);
  if (status != 0) return status;
  status = mtp_prelayer_device(out, vb);
  if (status != 0) return status;
  const void *inputs[5] = {prior, embedding, hidden_norm, embedding_norm, fc};
  if (mtp_prelayer_overlap(joined, 2 * vb, out, vb)) return -2;
  for (int i = 0; i < 5; ++i) {
    const size_t bytes = i == 4 ? 2 * h * vb : vb;
    if (mtp_prelayer_overlap(joined, 2 * vb, inputs[i], bytes) ||
        mtp_prelayer_overlap(out, vb, inputs[i], bytes)) return -2;
  }
  // Exactly the existing arithmetic, writing into the final concatenation.
  // Read-only inputs may overlap each other; fresh outputs may overlap none.
  fcuda_rmsnorm_f32(embedding, embedding_norm, joined, 1, h, eps);
  status = mtp_prelayer_status();
  if (status != 0) return status;
  fcuda_rmsnorm_f32(prior, hidden_norm, joined + h, 1, h, eps);
  status = mtp_prelayer_status();
  if (status != 0) return status;
  const float alpha = 1.f, beta = 0.f;
  cublasStatus_t blas = CUBLAS_STATUS_SUCCESS;
  // Same mapping as fcuda_matmul_f32, but do not erase failure with a zero fill.
  FCUDA_TS("cublas_sgemm", dim3(h, 1),
      blas = cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, h, 1, 2 * h,
          &alpha, fc, 2 * h, joined, 2 * h, &beta, out, h));
  if (blas != CUBLAS_STATUS_SUCCESS) return mtp_prelayer_fail(93000 + (int)blas);
  status = mtp_prelayer_status();
  if (status != 0) return status;
  status = mtp_prelayer_cuda(cudaStreamSynchronize(g_stream));
  if (status != 0) return status;
  return mtp_prelayer_status();
}
// END checked MTP native core.
