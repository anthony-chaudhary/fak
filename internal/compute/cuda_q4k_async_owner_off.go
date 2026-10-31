//go:build cuda && cgo && (!fak_cuda_q4k_qkv_qualification || !fak_cuda_v41_index_select || !fak_cuda_q4k_qkv_async)

package compute

func cudaQ4KAsyncRequireDrainLocked(c *cudaBackend) {}

// Terminal cleanup must preserve the primary failure and all storage.
func cudaQ4KAsyncReleaseBlockedLocked() bool { return false }
