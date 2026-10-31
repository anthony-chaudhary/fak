//go:build cuda && cgo && !fak_cuda_capture_once_qualification

package compute

import "unsafe"

func (c *cudaBackend) GraphOnceInvalidate()                                     {}
func (c *cudaBackend) graphOnceInvalidateKVLocked(*cudaKV)                      {}
func (c *cudaBackend) graphOnceRoPELocked(Tensor, int, int, float64) bool       { return false }
func (c *cudaBackend) graphOnceKVWriteLocked(*dslice, unsafe.Pointer, int) bool { return false }
func (c *cudaBackend) graphOnceAttentionLocked(Tensor, Tensor, *cudaKV, int, int, int, int, float32) bool {
	return false
}

func (c *cudaBackend) graphOnceFreeDependencyLocked(*cudaBuf) {}
