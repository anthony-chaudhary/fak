#ifndef FAK_CUDA_DEVICE_SCOPE_H
#define FAK_CUDA_DEVICE_SCOPE_H

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA device selection belongs to the calling host thread. Keep temporary
// selections inside one C ABI call; separate Go calls can use different threads.
class FcudaDeviceScope {
 public:
  FcudaDeviceScope()
      : previous_(0), status_(cudaGetDevice(&previous_)) {
    if (status_ != cudaSuccess) {
      fprintf(stderr, "fak-cuda: capture current device failed: %s\n",
              cudaGetErrorString(status_));
    }
  }

  ~FcudaDeviceScope() {
    if (status_ != cudaSuccess) return;
    // Restoration is best effort if CUDA itself rejects the saved device.
    // Do not discard a successful allocation or replace the primary error.
    const cudaError_t restored = cudaSetDevice(previous_);
    if (restored != cudaSuccess) {
      fprintf(stderr, "fak-cuda: restore device %d failed: %s\n",
              previous_, cudaGetErrorString(restored));
    }
  }

  cudaError_t status() const { return status_; }

  FcudaDeviceScope(const FcudaDeviceScope &) = delete;
  FcudaDeviceScope &operator=(const FcudaDeviceScope &) = delete;

 private:
  int previous_;
  cudaError_t status_;
};

#endif
