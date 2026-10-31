//go:build cuda && cgo && ((fak_cuda_v41_compressor && !fak_cuda_v41_compressor_pool && fak_cuda_v41_compressor_pending) || (fak_cuda_q4k_qkv_qualification && fak_cuda_v41_index_select && fak_cuda_q4k_qkv_async))

package compute

// Async owners have not qualified capture-once coexistence.
func cudaAsyncGraphOnceAllowed() bool { return false }
