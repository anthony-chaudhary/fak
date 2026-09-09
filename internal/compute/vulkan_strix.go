//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"github.com/anthony-chaudhary/fak/internal/compute/strix"
)

// StrixSFence executes an SFENCE instruction via internal/compute/strix to drain
// CPU write-combining store buffers (WCBs) prior to GPU queue submissions.
func (v *vulkanBackend) StrixSFence() {
	strix.SFence()
}

// StrixCoherencyFence guarantees store buffer coherency by issuing an SFENCE barrier
// between CPU host writes and GPU queue submissions when running on Strix Halo APU silicon.
func (v *vulkanBackend) StrixCoherencyFence() {
	if v != nil && isStrixHaloArch(v.tier) {
		strix.SFence()
	}
}

// VulkanStrixPreQueueSubmit issues an SFENCE barrier prior to GPU queue submission
// if running on AMD Strix Halo APU silicon.
func (v *vulkanBackend) VulkanStrixPreQueueSubmit() {
	if v != nil && isStrixHaloArch(v.tier) {
		strix.SFence()
	}
}

// VulkanStrixPostHostWrite issues an SFENCE barrier immediately following host writes
// to uncached speculative write-combining (USWC) buffers.
func (v *vulkanBackend) VulkanStrixPostHostWrite() {
	if v != nil && isStrixHaloArch(v.tier) {
		strix.SFence()
	}
}

// StrixWave32WMMAMatMul executes native Wave32 WMMA matrix multiplication (C = A * B)
// on AMD Strix Halo RDNA 3.5 silicon with store buffer coherency barriers.
func (v *vulkanBackend) StrixWave32WMMAMatMul(M, N, K int, A, B []float32) ([]float32, strix.WMMATelemetry, error) {
	strix.SFence()
	tiler := strix.NewWave32RetiledMatMul(strix.DefaultWave32WMMAConfig())
	res, telem, err := tiler.MatMul(M, N, K, A, B)
	strix.SFence()
	return res, telem, err
}

// VulkanStrixSFence issues an SFENCE memory barrier to flush CPU write-combining store buffers.
func VulkanStrixSFence() {
	strix.SFence()
}

// VulkanStrixCoherencyFence issues an SFENCE barrier between CPU host writes and GPU submissions.
func VulkanStrixCoherencyFence() {
	strix.SFence()
}

// VulkanStrixQueueSubmitFence issues an SFENCE memory barrier if isStrix is true.
func VulkanStrixQueueSubmitFence(isStrix bool) {
	if isStrix {
		strix.SFence()
	}
}

// VulkanStrixWave32WMMA executes a Wave32 WMMA matmul with store buffer coherency fencing.
func VulkanStrixWave32WMMA(M, N, K int, A, B []float32) ([]float32, strix.WMMATelemetry, error) {
	strix.SFence()
	tiler := strix.NewWave32RetiledMatMul(strix.DefaultWave32WMMAConfig())
	res, telem, err := tiler.MatMul(M, N, K, A, B)
	strix.SFence()
	return res, telem, err
}

// VulkanStrixWave32WMMASupported checks if backend tier matches Strix Halo GFX1151 and supports cooperative matrix.
func VulkanStrixWave32WMMASupported(b Backend) bool {
	if b == nil {
		return false
	}
	vb, ok := b.(*vulkanBackend)
	if !ok {
		return false
	}
	return isStrixHaloArch(vb.tier) && vb.haveCoopmat
}

// VulkanStrixIsGFX1151 reports whether the architecture string indicates Strix Halo silicon.
func VulkanStrixIsGFX1151(arch string) bool {
	return isStrixHaloArch(arch)
}
