//go:build cuda

// tf32_cuda.go — the runtime enable seam for the TF32 tensor-core math mode on the f32 SGEMM
// path (Lever 4 of the H100-KERNEL-5X-ROADMAP). The gate `tf32Enabled` is read once at init()
// from FAK_CUDA_TF32, but a host (e.g. a future `fak serve --cuda-tf32`) needs to flip it from
// a parsed flag AFTER init() has run. EnableCUDATF32 flips the gate and re-applies it to the
// live cuBLAS handle, so a post-init enable cleanly routes the f32 prefill GEMMs through the
// tensor cores.
//
// This lives in a separate cuda-tagged file (not cuda.go) so the enable seam is a one-line,
// low-blast-radius edit, exactly like graph_cuda.go's EnableCUDAGraph. It shares the `compute`
// package with cuda.go and so reads its private tf32Enabled var and the applyCUDATF32 helper
// (which owns the single `import "C"` cuBLAS-handle mutation) directly.
package compute

// EnableCUDATF32 turns on the TF32 tensor-core math mode at runtime and applies it to the
// cuBLAS handle. No-op-safe to call when no CUDA device was found at init: applyCUDATF32 routes
// through fcuda_set_tf32, which guards a nil handle. The non-cuda build provides an empty twin
// (tf32_nocuda.go) so a host can call this unconditionally without a build-tag branch.
func EnableCUDATF32() {
	tf32Enabled = true
	applyCUDATF32()
}
