//go:build cuda && cgo && (!fak_cuda_v41_compressor || fak_cuda_v41_compressor_pool || !fak_cuda_v41_compressor_pending)

package compute

func cudaV41CompressorProducedLocked(c *cudaBackend, b *cudaBuf)           {}
func cudaV41CompressorSubmitLocked(b *cudaBuf)                             {}
func cudaV41CompressorDeferFreeLocked(c *cudaBackend, b *cudaBuf) bool     { return false }
func cudaV41CompressorPendingIndex(b *cudaBuf) int                         { return -1 }
func cudaV41CompressorReadBeginLocked(c *cudaBackend) bool                 { return false }
func cudaV41CompressorReadEndLocked(c *cudaBackend, pending, success bool) {}
func cudaV41CompressorDrainLocked(c *cudaBackend)                          {}
