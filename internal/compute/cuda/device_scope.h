#pragma once

class fcuda_device_scope {
  int previous = 0;
  int result = 0;
  bool saved = false;

 public:
  explicit fcuda_device_scope(int device) {
    cudaError_t e = cudaGetDevice(&previous);
    if (e != cudaSuccess) {
      fprintf(stderr, "fak-cuda: cudaGetDevice failed: %s\n", cudaGetErrorString(e));
      result = (int)e + 1000;
      return;
    }
    // A failed selector can report an earlier asynchronous error. Once the
    // caller device is known, every exit must attempt to restore it.
    saved = true;
    result = fcuda_set_device(device);
  }
  fcuda_device_scope(const fcuda_device_scope &) = delete;
  fcuda_device_scope &operator=(const fcuda_device_scope &) = delete;
  ~fcuda_device_scope() {
    if (saved && fcuda_set_device(previous) != 0)
      fprintf(stderr, "fak-cuda: failed to restore caller device %d\n", previous);
  }
  int status() const { return result; }
};

