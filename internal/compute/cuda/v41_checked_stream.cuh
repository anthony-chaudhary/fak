// Shared MHC/router/embedding safety seam, included after model_candidates.cuh. The
// registered cudaDev and cudaMu own the private g_stream; never import a stream.
#ifdef FAK_CUDA_V41_CHECKED_STREAM
static int v41_checked_stream_error(int base,cudaError_t error) {
  if(error==cudaSuccess)return 0;
  g_v41_checked_stream_uncertain=true;
  return base+(int)error;
}
static int v41_checked_stream_state(void) {
  if(!g_blas||!g_stream||g_capture_open||v41_candidates_stream_uncertain())return -3;
  cudaError_t e=cudaPeekAtLastError();if(e!=cudaSuccess)return v41_checked_stream_error(10000,e);
  int device=-1;e=cudaGetDevice(&device);if(e!=cudaSuccess)return v41_checked_stream_error(10000,e);
  if(device!=0)return -3;
  cudaStreamCaptureStatus capture;e=cudaStreamIsCapturing(g_stream,&capture);
  if(e!=cudaSuccess)return v41_checked_stream_error(10000,e);
  if(capture!=cudaStreamCaptureStatusNone)return -3;
  // cudaDev/cudaMu provenance is checked in Go before this call. FIFO permits
  // pending same-runtime dependencies; final checked completion owns the fence.
  e=cudaStreamQuery(g_stream);
  return e==cudaSuccess||e==cudaErrorNotReady?0:v41_checked_stream_error(10000,e);
}
static int v41_checked_stream_owned_bytes(const void *p,uint64_t bytes) {
  if(!p||bytes==0||bytes>SIZE_MAX)return -2;
  auto found=g_live.find(const_cast<void *>(p));
  if(found==g_live.end()||found->second<(size_t)bytes)return -2;
  cudaPointerAttributes attr;cudaError_t e=cudaPointerGetAttributes(&attr,p);
  if(e!=cudaSuccess)return v41_checked_stream_error(11000,e);
  return attr.type==cudaMemoryTypeDevice&&attr.device==0?0:-2;
}
static int v41_checked_stream_owned(const void *p,uint64_t count) {
  if(count==0||count>INT_MAX||count>SIZE_MAX/sizeof(float))return -2;
  return v41_checked_stream_owned_bytes(p,count*sizeof(float));
}
static int v41_checked_stream_finish(void) {
  if(g_v41_checked_stream_uncertain||!g_stream||g_capture_open)return -3;
  return v41_checked_stream_error(50000,cudaStreamSynchronize(g_stream));
}
#endif
