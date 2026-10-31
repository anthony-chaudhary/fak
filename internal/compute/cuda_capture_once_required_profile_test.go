//go:build fak_cuda_capture_once_device_witness && (!cuda || !cgo || !fak_cuda_capture_once_qualification)

package compute

import "testing"

// Required witness selection must fail, not disappear, when the native profile is absent.
func TestCUDAGraphOnceRequiredBorrowedLifetime(t *testing.T) {
	t.Fatal("capture-once required-device witness requires CGO_ENABLED=1 and cuda,fak_cuda_capture_once_qualification build tags; no skip or CPU fallback")
}
