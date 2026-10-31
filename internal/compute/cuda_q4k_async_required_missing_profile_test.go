//go:build fak_cuda_q4k_async_witness && (!cuda || !cgo || !fak_cuda_q4k_qkv_qualification || !fak_cuda_v41_index_select || !fak_cuda_q4k_qkv_async)

package compute

import "testing"

// fak-test:runtime fast est=1ms lane=optin
func TestCUDAQ4KAsyncOwnedPhysical(t *testing.T) {
	t.Fatal("required Q4 async witness lacks cuda+cgo+Q4 qualification+index-select+async profile; no fallback")
}
