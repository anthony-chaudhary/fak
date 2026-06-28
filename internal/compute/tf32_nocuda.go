//go:build !cuda

// tf32_nocuda.go — the non-cuda twin of EnableCUDATF32 (tf32_cuda.go). The default
// `go build ./cmd/fak` excludes the CUDA backend entirely, so there is no f32 SGEMM on a device
// to switch to tensor-core TF32 math; this empty function lets a host call the enable seam
// unconditionally without a build-tag branch at the call site. The flag is simply inert on a
// CPU-only binary — exactly as a tensor-core prefill lever for a device GEMM should be.
package compute

// EnableCUDATF32 is a no-op in the non-cuda build (no device, no cuBLAS handle to retune).
func EnableCUDATF32() {}
