//go:build fak_cuda_capture_once_capacity_witness && (!cuda || !cgo || !fak_cuda_capture_once_qualification)

package compute

import "testing"

func TestCUDAGraphOnceRequiredLastCapacity(t *testing.T) {
	t.Fatal("last-capacity required-device witness requires CGO_ENABLED=1 and cuda,fak_cuda_capture_once_qualification; no skip/fallback")
}
